"""
agent_orchestrator.py
=====================
Enterprise Multi-Agent Customer Support System
Built with Strands Agents SDK + Amazon Bedrock AgentCore

Architecture implemented:

  Customer Request
        │
  OrchestratorAgent  (Claude Haiku 4.5 - fast routing, manages WorkflowState)
        │
   ┌────┼────────────────────┬────────────────────────┐
   │    │                    │                        │
InventoryAgent   PolicyAgent   RefundAgent  CommunicationAgent
(DynamoDB)    (Multi-Agent RAG)  (DynamoDB)   (composes response)
                    │
         ┌──────────┼──────────┐
    ReturnsPolicyRetriever  ShippingPolicyRetriever  WarrantyPolicyRetriever
        (KB: returns)           (KB: shipping)           (KB: warranty)
         └──────────── all run in PARALLEL ────────────┘

Shared state flows through DynamoDB WorkflowStateTable.
OrchestratorAgent creates state at start, each routing tool reads and
updates it after the worker responds.

Commands:
  python src/agent_orchestrator.py test            # 3 scenarios, local run, traced to X-Ray
  python src/agent_orchestrator.py chat            # interactive terminal chat
  python src/agent_orchestrator.py deploy          # Tasks 3-6 deployment pipeline (uses the AgentCore CLI)
  python src/agent_orchestrator.py invoke "<msg>"  # call the deployed AgentCore Runtime
  python src/agent_orchestrator.py serve           # HTTP server (what AgentCore Runtime runs)

Deployment uses the AgentCore CLI (`agentcore`, npm package @aws/agentcore,
https://github.com/aws/agentcore-cli) through the pre-written helper
src/agentcore_cli.py - see the README for the prerequisites (Node.js 20+, uv).
"""

import boto3
import json
import time
import os
import sys
import uuid
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from decimal import Decimal
from typing import Optional

# Ensure the parent directory is on sys.path so config.py and
# bedrock_kb_retrieval.py are importable regardless of where this
# script is invoked from (e.g. python src/agent_orchestrator.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Strands Agents SDK - see: https://github.com/strands-agents/sdk-python
from strands import Agent
from strands.models import BedrockModel
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

import config
from bedrock_kb_retrieval import retrieve_from_knowledge_base, format_kb_results

# Configure logging for debugging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────
# OUTPUT UTILITIES
# ─────────────────────────────────────────────────────
# Terminal trace UI, ANSI colour constants, and agent metadata
# are defined in agent_utils.py - keeping this file focused on
# agent architecture.
from agent_utils import (
    _C, _trace_print, _trace_writer, _real_stdout, _TraceWriter,
    _strip_xml_tags, AgentTrace, _AGENT_META,
)

# ─────────────────────────────────────────────────────
# OBSERVABILITY
# ─────────────────────────────────────────────────────
# `tool` is the Strands @tool decorator wrapped so that every tool call is
# recorded as an X-Ray subsegment (the orchestrator's route_to_* tools become
# the worker-agent nodes on the X-Ray Service Map) and logged at INFO level.
# Use it exactly like `strands.tool`:  @tool  above each tool function.
from agent_observability import (
    tool, tracer, setup_logging, flush_logs, print_trace_hint,
    apply_observability_config, wait_for_runtime_ready,
)


# ─────────────────────────────────────────────────────
# AWS CLIENTS
# ─────────────────────────────────────────────────────
bedrock_agent_client = boto3.client('bedrock-agent', region_name=config.AWS_REGION)
bedrock_runtime      = boto3.client('bedrock-runtime', region_name=config.AWS_REGION)
agentcore_client     = boto3.client('bedrock-agentcore', region_name=config.AWS_REGION)
agentcore_control    = boto3.client('bedrock-agentcore-control', region_name=config.AWS_REGION)
dynamodb             = boto3.resource('dynamodb', region_name=config.AWS_REGION)
logs_client          = boto3.client('logs', region_name=config.AWS_REGION)


# ═══════════════════════════════════════════════════════
#  WORKFLOW STATE - SHARED DynamoDB STATE OBJECT
#
#  WorkflowState stores the accumulated context for one customer session:
#    - What the InventoryAgent found (order status, eligibility, customer tier)
#    - What the PolicyAgent found (relevant policy text)
#    - What the RefundAgent decided (approval/denial, reference number)
#    - The CommunicationAgent's final draft
#
#  The `version` field enables optimistic locking: every write is a
#  conditional DynamoDB update that fails if someone else updated first.
#  If the condition fails, the update is retried after a fresh read.
# ═══════════════════════════════════════════════════════

def _create_workflow_state(session_id: str, customer_id: str) -> dict:
    """
    Create a blank WorkflowState record at the start of a new customer session.

    Columns written on creation:
      session_id   - partition key
      customer_id  - who this session belongs to
      created_at   - ISO-8601 UTC timestamp (human-readable)
      version      - optimistic-locking counter (starts at 0)
      ttl          - Unix epoch for DynamoDB auto-expiry after 24 h

    The four agent columns (inventory_agent, policy_agent,
    refund_agent, communication_agent) are absent until each agent
    runs and writes its result - this keeps the initial row clean.
    """
    state = {
        'session_id':  session_id,
        'customer_id': customer_id,
        'created_at':  time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'version':     0,
        'ttl':         int(time.time()) + (24 * 3600),
    }
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    table.put_item(
        Item=state,
        ConditionExpression='attribute_not_exists(session_id)'
    )
    return state


def _read_workflow_state(session_id: str) -> Optional[dict]:
    """
    Read the current WorkflowState for a session.
    """
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    response = table.get_item(Key={'session_id': session_id})
    return response.get('Item')


# Trace singleton - created after _read_workflow_state so AgentTrace.summary()
# can read DynamoDB WorkflowState. The read_state_fn avoids a circular import.
trace = AgentTrace(read_state_fn=_read_workflow_state)


def _update_workflow_state(session_id: str, updates: dict,
                           expected_version: int, max_retries: int = 3) -> dict:
    """
    Update WorkflowState with optimistic locking.
    """
    from boto3.dynamodb.conditions import Attr

    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)

    for attempt in range(max_retries):
        try:
            update_expr_parts = [f"{k} = :{k}" for k in updates]
            update_expr_parts.append("version = :new_version")
            update_expr = "SET " + ", ".join(update_expr_parts)

            expr_values = {f":{k}": v for k, v in updates.items()}
            expr_values[':new_version']      = expected_version + 1
            expr_values[':expected_version'] = expected_version

            table.update_item(
                Key={'session_id': session_id},
                UpdateExpression=update_expr,
                ConditionExpression='version = :expected_version',
                ExpressionAttributeValues=expr_values
            )
            return _read_workflow_state(session_id)

        except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"WorkflowState update failed after {max_retries} retries "
                    f"(session: {session_id}). Too many concurrent writes."
                )
            logger.warning(
                f"WorkflowState version conflict on attempt {attempt+1}, retrying..."
            )
            current = _read_workflow_state(session_id)
            if current:
                expected_version = int(current['version'])
            time.sleep(0.1 * (attempt + 1))

    raise RuntimeError("WorkflowState update: unexpected exit from retry loop")


def _to_native(value):
    """
    Recursively convert DynamoDB `Decimal` values to int/float.

    boto3 returns every DynamoDB number as `Decimal`, which cannot be
    serialised to JSON - so tool results are passed through this first.
    """
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, dict):
        return {k: _to_native(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_native(v) for v in value]
    return value


def _run_worker_agent(agent: Agent, prompt: str) -> str:
    """
    Invoke a worker agent for ONE routing step and return its answer as text.

    Workers are built once and reused. Their conversation history is cleared
    before every call so one customer's facts can never leak into another
    request and the context does not grow without bound.
    """
    agent.messages.clear()
    return str(agent(prompt))


# ═══════════════════════════════════════════════════════
#  TASK 2 - MULTI-AGENT ORCHESTRATION
# ═══════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────
#  2.A - INVENTORY AGENT
# ───────────────────────────────────────────────────────

def build_inventory_agent() -> Agent:
    """
    Build the Inventory Agent.

    Gathers order and customer facts from DynamoDB. Does NOT make decisions -
    only retrieves data for the OrchestratorAgent to share with downstream agents.
    """

    # Worker model (Claude Sonnet 4.5), low temperature: factual, repeatable lookups.
    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        temperature=0.1,
        region_name=config.AWS_REGION,
    )

    system_prompt = """You are the InventoryAgent for NovaMart's customer support system.

ROLE
You are a DATA GATHERER. You look up order and customer facts in DynamoDB and report
them accurately. You NEVER decide whether an order is eligible for a return or refund,
you never interpret company policy, and you never guess or invent data.

TOOLS
- check_order_status(customer_id, order_id): one order's status, product, dates, price.
- get_customer_tier(customer_id): the customer's tier (Standard or Premium) and profile.
- list_customer_orders(customer_id): every order belonging to the customer.

PROCEDURE
1. Read the customer ID (and the order ID, if the request mentions one) from the request.
2. If an order ID is given, call check_order_status. If the customer refers to an order
   without an ID (e.g. "my headphones"), call list_customer_orders to find it.
3. If the request involves a return, a refund, or any question about the account or
   tier, ALSO call get_customer_tier.
4. Report ONLY what the tools returned. If a record is not found, say so plainly.

OUTPUT FORMAT (plain facts, no opinions)
CUSTOMER: <customer_id>, <name>, tier <Standard|Premium>
ORDER: <order_id>, <product_name>, status <status>, ordered <order_date>,
       <days_since_order> days ago, price <price>, quantity <quantity>
Include every fact the tools returned that could matter downstream.

Do NOT state or hint whether a return or refund is allowed. That decision belongs to
the RefundAgent."""

    @tool
    def check_order_status(customer_id: str, order_id: str) -> dict:
        """
        Look up one order in DynamoDB and report its status, product, dates
        and amount. Reports facts only - it does NOT decide return eligibility.

        Args:
            customer_id: The customer's unique identifier (e.g. CUST-001)
            order_id: The order identifier (e.g. ORD-27176)

        Returns:
            Order record (order_id, status, product_name, order_date, price, ...)
            plus today's date and days_since_order, or a not-found message
        """
        try:
            orders_table = dynamodb.Table(config.ORDERS_TABLE)
            # Orders table has a composite key: customer_id (HASH) + order_id (RANGE)
            response = orders_table.get_item(
                Key={'customer_id': customer_id, 'order_id': order_id}
            )
        except ClientError as exc:
            return {'found': False, 'error': f"Order lookup failed: {exc}"}

        item = response.get('Item')
        if not item:
            return {
                'found': False,
                'message': f"No order {order_id} found for customer {customer_id}.",
            }

        order = _to_native(item)
        # Plain date arithmetic is a FACT (not a decision) - it spares the
        # downstream RefundAgent from guessing what "today" is.
        order['today'] = date.today().isoformat()
        try:
            ordered_on = datetime.strptime(order['order_date'], '%Y-%m-%d').date()
            order['days_since_order'] = (date.today() - ordered_on).days
        except (KeyError, ValueError):
            order['days_since_order'] = None
        return {'found': True, 'order': order}

    @tool
    def get_customer_tier(customer_id: str) -> dict:
        """
        Retrieve a customer's tier (Standard or Premium) from DynamoDB.
        Standard customers have a 30-day return window; Premium customers have 60 days.

        Args:
            customer_id: The customer's unique identifier

        Returns:
            Customer profile including tier and account details
        """
        try:
            customers_table = dynamodb.Table(config.CUSTOMERS_TABLE)
            response = customers_table.get_item(Key={'customer_id': customer_id})
        except ClientError as exc:
            return {'found': False, 'error': f"Customer lookup failed: {exc}"}

        item = response.get('Item')
        if not item:
            return {'found': False,
                    'message': f"No customer {customer_id} found."}

        customer = _to_native(item)
        return {
            'found': True,
            'customer_id':       customer.get('customer_id'),
            'name':              customer.get('name'),
            'tier':              customer.get('tier'),
            'account_created':   customer.get('account_created'),
            'total_orders':      customer.get('total_orders'),
            'preferred_contact': customer.get('preferred_contact'),
            # Email is deliberately left out: agents don't need it, and
            # keeping PII out of model context is good practice.
        }

    @tool
    def list_customer_orders(customer_id: str) -> dict:
        """
        Retrieve all orders for a customer from DynamoDB.

        Args:
            customer_id: The customer's unique identifier

        Returns:
            List of all orders with order_id, status, order_date, and amount
        """
        try:
            orders_table = dynamodb.Table(config.ORDERS_TABLE)
            query_args = {'KeyConditionExpression': Key('customer_id').eq(customer_id)}
            items = []
            while True:
                response = orders_table.query(**query_args)
                items.extend(response.get('Items', []))
                if 'LastEvaluatedKey' not in response:
                    break
                query_args['ExclusiveStartKey'] = response['LastEvaluatedKey']
        except ClientError as exc:
            return {'found': False, 'error': f"Order listing failed: {exc}"}

        orders = []
        for raw in items:
            item = _to_native(raw)
            try:
                amount = round(float(item.get('price', 0)) * int(item.get('quantity', 1)), 2)
            except (TypeError, ValueError):
                amount = None
            orders.append({
                'order_id':     item.get('order_id'),
                'product_name': item.get('product_name'),
                'status':       item.get('status'),
                'order_date':   item.get('order_date'),
                'quantity':     item.get('quantity'),
                'amount':       amount,
            })
        orders.sort(key=lambda o: o.get('order_date') or '', reverse=True)
        return {'customer_id': customer_id, 'order_count': len(orders), 'orders': orders}

    return Agent(
        name='InventoryAgent',
        model=model,
        system_prompt=system_prompt,
        tools=[check_order_status, get_customer_tier, list_customer_orders],
    )


# ───────────────────────────────────────────────────────
#  2.B - REFUND AGENT
# ───────────────────────────────────────────────────────

def build_refund_agent() -> Agent:
    """
    Build the Refund Agent.

    Makes return/refund eligibility decisions based on order facts from
    WorkflowState and applies the correct policy window per customer tier.
    """

    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        temperature=0.1,
        region_name=config.AWS_REGION,
    )

    system_prompt = """You are the RefundAgent for NovaMart's customer support system.

ROLE
You decide whether an order can be returned and, when it can, start the return.

RETURN WINDOWS (by customer tier)
- Standard customers: 30 days from the order date
- Premium customers:  60 days from the order date
An order is within the window when days_since_order <= the window for that tier.

DECISION PROCESS - follow it in this exact order
1. ALWAYS call get_inventory_context(session_id) FIRST. The session ID is in the request.
   It returns the facts the InventoryAgent gathered: customer tier, order status,
   order date and days_since_order. Never decide from memory or from the customer's claims.
2. If the context is empty or the order was not found, do NOT approve anything. Say that
   the order could not be verified and ask the customer to confirm the order number.
3. If the order status is "return_initiated", a return already exists: do not start a
   second one. Report that it is already in progress (and its reference, if shown).
4. Otherwise check BOTH conditions:
   a. The order status is "delivered". Orders that are processing, shipped or cancelled
      cannot be returned (explain what the status means for the customer).
   b. days_since_order is within the tier's window (30 Standard / 60 Premium).
5. If BOTH conditions pass: call initiate_refund(customer_id, order_id, reason) once, using
   the customer's stated reason (or "Customer requested return" if none was given).
6. If either condition fails: do NOT call initiate_refund. Explain exactly why, with the
   numbers (tier, window, days since order, order status).

OUTPUT FORMAT
DECISION: APPROVED or DENIED
TIER: <tier> (<window>-day window)
ORDER: <order_id>, <product_name>, status <status>, <days_since_order> days since order
REASON: <one or two factual sentences>
RETURN_REFERENCE: <reference from initiate_refund, or "none">
NEXT_STEPS: <the instructions returned by initiate_refund, or "none">

Be accurate and consistent: the same facts must always give the same decision."""

    @tool
    def get_inventory_context(session_id: str) -> dict:
        """
        Read the WorkflowState to access facts gathered by the InventoryAgent.

        Args:
            session_id: The current session identifier

        Returns:
            The inventory_agent field from WorkflowState, or empty dict if not yet set
        """
        state = _read_workflow_state(session_id)
        if not state or not state.get('inventory_agent'):
            return {}
        return {
            'session_id':      session_id,
            'customer_id':     state.get('customer_id'),
            'inventory_agent': state['inventory_agent'],
        }

    @tool
    def initiate_refund(customer_id: str, order_id: str, reason: str) -> dict:
        """
        Initiate a return by updating the order record in DynamoDB.

        Args:
            customer_id: The customer's unique identifier
            order_id: The order to return
            reason: Customer-provided reason for the return

        Returns:
            Confirmation dict with return_reference number and instructions
        """
        orders_table = dynamodb.Table(config.ORDERS_TABLE)
        key = {'customer_id': customer_id, 'order_id': order_id}
        instructions = ("Pack the item in its original packaging with all accessories, "
                        "attach the prepaid return label sent to your email, and drop it "
                        "off with the carrier. Your refund is issued within 5-7 business "
                        "days after the item is received.")
        try:
            existing = orders_table.get_item(Key=key).get('Item')
            if not existing:
                return {'success': False,
                        'message': f"Order {order_id} not found for customer {customer_id}."}

            # Idempotent: never create a second return for the same order.
            if existing.get('status') == 'return_initiated' and existing.get('return_reference'):
                return {'success': True, 'already_initiated': True,
                        'order_id': order_id,
                        'return_reference': existing['return_reference'],
                        'instructions': instructions}

            return_reference = f"RET-{uuid.uuid4().hex[:8].upper()}"
            orders_table.update_item(
                Key=key,
                UpdateExpression=("SET #status = :status, return_reference = :ref, "
                                  "return_reason = :reason, return_initiated_at = :ts"),
                ConditionExpression='attribute_exists(order_id)',
                ExpressionAttributeNames={'#status': 'status'},   # 'status' is a reserved word
                ExpressionAttributeValues={
                    ':status': 'return_initiated',
                    ':ref':    return_reference,
                    ':reason': reason,
                    ':ts':     time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                },
            )
        except ClientError as exc:
            return {'success': False, 'error': f"Could not initiate return: {exc}"}

        return {'success': True, 'already_initiated': False, 'order_id': order_id,
                'return_reference': return_reference, 'instructions': instructions}

    return Agent(
        name='RefundAgent',
        model=model,
        system_prompt=system_prompt,
        tools=[get_inventory_context, initiate_refund],
    )


# ───────────────────────────────────────────────────────
#  2.C - POLICY AGENT - MULTI-AGENT RAG
# ───────────────────────────────────────────────────────

def build_policy_agent() -> Agent:
    """
    Build the Policy Agent - a multi-agent RAG system.

    Internally creates three specialized retriever sub-agents that run in
    PARALLEL, each querying its own Knowledge Base. The coordinator synthesizes
    the combined results into a complete, grounded policy answer.
    """

    def _retriever_model() -> BedrockModel:
        """Deterministic (temperature 0.0) worker model shared by the retrievers."""
        return BedrockModel(
            model_id=config.WORKER_MODEL_ID,
            temperature=0.0,
            region_name=config.AWS_REGION,
        )

    def _retriever_prompt(domain: str, tool_name: str) -> str:
        return (f"You are the {domain}PolicyRetrieverAgent. Your only job is to fetch "
                f"{domain.lower()} policy passages from the NovaMart {domain} Policy "
                f"knowledge base.\n"
                f"1. Call {tool_name} exactly once with the user's question as the query.\n"
                f"2. Return the retrieved passages exactly as given, keeping their sources.\n"
                f"3. Do not answer the question yourself and do not add outside knowledge.\n"
                f"4. If nothing relevant is returned, reply: \"No relevant {domain.lower()} "
                f"policy found.\"")

    # ── Retriever 1: Returns ──────────────────────────────────────────
    @tool
    def retrieve_returns_policy(query: str) -> str:
        """
        Retrieve relevant passages from the Returns Policy knowledge base.

        Args:
            query: The customer's question, in natural language

        Returns:
            Formatted top passages (text, relevance score, source) or a no-results message
        """
        return format_kb_results(retrieve_from_knowledge_base(config.RETURNS_KB_ID, query))

    returns_retriever = Agent(
        name='ReturnsPolicyRetrieverAgent',
        model=_retriever_model(),
        system_prompt=_retriever_prompt('Returns', 'retrieve_returns_policy'),
        tools=[retrieve_returns_policy],
    )

    # ── Retriever 2: Shipping ─────────────────────────────────────────
    @tool
    def retrieve_shipping_policy(query: str) -> str:
        """
        Retrieve relevant passages from the Shipping Policy knowledge base.

        Args:
            query: The customer's question, in natural language

        Returns:
            Formatted top passages (text, relevance score, source) or a no-results message
        """
        return format_kb_results(retrieve_from_knowledge_base(config.SHIPPING_KB_ID, query))

    shipping_retriever = Agent(
        name='ShippingPolicyRetrieverAgent',
        model=_retriever_model(),
        system_prompt=_retriever_prompt('Shipping', 'retrieve_shipping_policy'),
        tools=[retrieve_shipping_policy],
    )

    # ── Retriever 3: Warranty ─────────────────────────────────────────
    @tool
    def retrieve_warranty_policy(query: str) -> str:
        """
        Retrieve relevant passages from the Warranty Policy knowledge base.

        Args:
            query: The customer's question, in natural language

        Returns:
            Formatted top passages (text, relevance score, source) or a no-results message
        """
        return format_kb_results(retrieve_from_knowledge_base(config.WARRANTY_KB_ID, query))

    warranty_retriever = Agent(
        name='WarrantyPolicyRetrieverAgent',
        model=_retriever_model(),
        system_prompt=_retriever_prompt('Warranty', 'retrieve_warranty_policy'),
        tools=[retrieve_warranty_policy],
    )

    @tool
    def search_all_policies(query: str) -> str:
        """
        Query all three policy knowledge bases IN PARALLEL and return combined results.

        Runs ReturnsPolicyRetrieverAgent, ShippingPolicyRetrieverAgent, and
        WarrantyPolicyRetrieverAgent simultaneously, then combines their findings.

        Args:
            query: The customer's policy question

        Returns:
            Combined policy passages from all three knowledge bases
        """
        # Domain name -> retriever sub-agent (order = display order)
        retrievers = {
            'Returns':  returns_retriever,
            'Shipping': shipping_retriever,
            'Warranty': warranty_retriever,
        }

        # ── Trace: show parallel KB dispatch to learners ──────────────────
        trace.kb_start({
            'Returns':  config.RETURNS_KB_ID,
            'Shipping': config.SHIPPING_KB_ID,
            'Warranty': config.WARRANTY_KB_ID,
        })

        def _run_retriever(domain: str, agent, query: str) -> tuple:
            """
            Run one retriever sub-agent and return (domain, result_text).

            stdout is suppressed globally for all threads by the
            _TraceWriter._suppress_parallel flag set in kb_start().
            This covers both the direct worker thread and any internal
            streaming child threads that Strands SDK spawns internally -
            which do NOT inherit thread-local variables and therefore cannot
            be suppressed with a thread-local capture approach.
            Results are returned as values and printed cleanly and
            sequentially by trace.kb_result() after all futures join.
            """
            try:
                agent.messages.clear()          # each query starts with a clean slate
                return domain, str(agent(query))
            except Exception as exc:            # one failing KB must not sink the others
                logger.warning(f"{domain} retriever failed: {exc}")
                return domain, f"[{domain} knowledge base unavailable: {exc}]"

        # Fan out: one thread per retriever, collect each as it completes
        results = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(_run_retriever, domain, agent, query): domain
                for domain, agent in retrievers.items()
            }
            for future in as_completed(futures):
                domain, text = future.result()
                results[domain] = text

        # ── Trace: all KBs responded - print each result sequentially ─────
        trace.kb_done(len(retrievers))
        for domain in ['Returns', 'Shipping', 'Warranty']:
            trace.kb_result(domain, results.get(domain, '[No results]'))

        # Combine in a fixed order so the output is deterministic
        return "\n\n".join(
            f"=== {domain} Policy Knowledge Base ===\n{results.get(domain, '[No results]')}"
            for domain in retrievers
        )

    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        temperature=0.2,
        region_name=config.AWS_REGION,
    )

    system_prompt = """You are the PolicyAgent for NovaMart's customer support system.

ROLE
You answer questions about NovaMart's return, shipping and warranty policies using ONLY
text retrieved from the company's policy knowledge bases.

PROCEDURE
1. ALWAYS call search_all_policies FIRST, passing the customer's question as the query.
   It searches the Returns, Shipping and Warranty knowledge bases in parallel.
2. Read all three result sections. Use only passages that are relevant to the question.
3. Synthesize ONE clear, accurate answer grounded in those passages. Where useful, state
   which policy area it comes from (Returns, Shipping or Warranty) and give specifics
   such as time windows, fees and conditions, including differences between Standard
   and Premium tiers.
4. If the passages do not cover the question, say so plainly. NEVER invent policy details
   and never use outside knowledge to fill gaps.
5. You know policy text only. You have no access to any customer's account, tier or
   orders, so never claim to know them.

Keep the answer factual and well organized; the CommunicationAgent will turn it into the
customer-facing reply."""

    return Agent(
        name='PolicyAgent',
        model=model,
        system_prompt=system_prompt,
        tools=[search_all_policies],
    )


# ───────────────────────────────────────────────────────
#  2.D - COMMUNICATION AGENT
# ───────────────────────────────────────────────────────

def build_communication_agent() -> Agent:
    """
    Build the Communication Agent.

    Drafts the final customer-facing message by reading the full WorkflowState
    and composing a coherent, empathetic response.
    """

    # Slightly higher temperature than the other workers for a warm, natural tone.
    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        temperature=0.3,
        region_name=config.AWS_REGION,
    )

    system_prompt = """You are the CommunicationAgent for NovaMart's customer support team.
You write the FINAL message the customer will read.

PROCEDURE
1. ALWAYS call get_full_workflow_context(session_id) first. It returns what the other
   specialists found: inventory_agent (order and customer facts), policy_agent (policy
   information) and refund_agent (the return decision and reference number).
2. Write one reply to the customer's CURRENT request. Include every relevant finding:
   - Return/refund: the decision, the reason (tier, return window, days since order),
     the return reference number and the next steps, exactly as the specialists reported.
   - Order status or account questions: the actual order details or tier.
   - Policy questions: the specific policy details (windows, fees, conditions).
   Ignore findings that are left over from an earlier, unrelated question.
3. Math or calculation requests: no specialist is involved, so compute the answer yourself
   from the customer's request. Show the steps briefly, keep full precision during the
   calculation, and round currency to 2 decimals only at the very end.

STYLE
- Warm, professional and empathetic. Greet the customer by first name if it is known.
- If the news is disappointing (a denied return), acknowledge the frustration, explain
  the reason clearly and offer a helpful alternative or next step.
- Plain language, short paragraphs. Never mention agents, databases, tools, versions,
  sessions or any internal system.
- Use only facts present in the workflow context. Never invent order details, dates,
  amounts, policies or reference numbers. If something is missing, say what you can
  and offer to help further.
- End with an offer of further help."""

    @tool
    def get_full_workflow_context(session_id: str) -> dict:
        """
        Read the complete WorkflowState to access all findings from previous agents.

        Args:
            session_id: The current session identifier

        Returns:
            Full WorkflowState dict (inventory_agent, policy_agent, refund_agent)
        """
        state = _read_workflow_state(session_id)
        if not state:
            return {}
        # Internal bookkeeping (version, ttl) is of no use to the reply.
        return _to_native({k: v for k, v in state.items() if k not in ('version', 'ttl')})

    return Agent(
        name='CommunicationAgent',
        model=model,
        system_prompt=system_prompt,
        tools=[get_full_workflow_context],
    )


# ───────────────────────────────────────────────────────
#  2.E - ORCHESTRATOR AGENT
# ───────────────────────────────────────────────────────

def build_orchestrator_agent(
    inventory_agent:      Agent,
    refund_agent:         Agent,
    policy_agent:         Agent,
    communication_agent:  Agent,
) -> Agent:
    """
    Build the Orchestrator Agent that routes requests and manages WorkflowState.
    """

    # Fast, cost-efficient model; temperature 0.0 for deterministic routing.
    model = BedrockModel(
        model_id=config.ORCHESTRATOR_MODEL_ID,
        temperature=0.0,
        region_name=config.AWS_REGION,
    )

    system_prompt = """You are the OrchestratorAgent for NovaMart's customer support system.
You route each request to specialist agents and manage the shared WorkflowState.
You do NOT answer questions yourself and you NEVER write the customer-facing reply.

Every message starts with "[Session ID: <id>] [Customer ID: <id>]" followed by the
customer's request. Use those exact IDs in every tool call.

ROUTING RULES
1. EVERY request, always: call initialize_session FIRST.
2. Order status, return or refund requests: call route_to_inventory_agent first. For a
   return or refund request, then call route_to_refund_agent. (A pure order-status question
   needs the inventory agent only.)
3. Policy meaning questions (return windows, shipping rates, warranty terms): call
   route_to_policy_agent.
4. Account questions ("what is my tier?", "am I premium?"): call route_to_inventory_agent.
   NEVER use the policy agent for these: it knows policy text, not customer data.
5. Math or calculation questions: skip Inventory, Policy and Refund entirely. After
   initialize_session, go straight to route_to_communication_agent.
6. EVERY request, always, as the very LAST step: call route_to_communication_agent so
   that the CommunicationAgent composes the final reply.

A request can combine rules (e.g. a return plus a policy question): run each needed
specialist in the order above, then finish with the communication agent.

HARD REQUIREMENTS
- Call tools ONE AT A TIME and wait for each result before the next call. Never call
  route_to_communication_agent in the same step as another tool.
- route_to_communication_agent is ALWAYS your final tool call, with no exceptions.
- Pass the customer's request to the tools verbatim.
- After route_to_communication_agent returns, reply with the text it returned, word for
  word, and add nothing. Never compose, edit or summarize the customer reply yourself.
- If a tool reports an error, fix the cause if you can (for example call initialize_session)
  and continue. Always still finish with route_to_communication_agent."""

    # Each routing tool follows the same pattern:
    #   1. read the current WorkflowState  (_read_workflow_state)
    #   2. invoke the worker agent
    #   3. write its result back with optimistic locking
    #      (_update_workflow_state(session_id, {'<column>': text}, expected_version))
    # The terminal trace UI can show each step: call trace.step_start('inventory_agent')
    # before the worker runs and trace.step_done('inventory_agent', old_version) after.

    _NOT_INITIALIZED = ("Error: no WorkflowState exists for this session yet. "
                        "Call initialize_session first, then retry.")

    @tool
    def route_to_inventory_agent(session_id: str, customer_id: str, request: str) -> str:
        """
        Route an order-related request to the Inventory Agent to gather order facts.
        Call this FIRST for any request involving order status, history, or returns.

        Args:
            session_id:  The current session identifier (from the customer request)
            customer_id: The customer's unique identifier
            request:     The customer's original request

        Returns:
            Inventory facts retrieved by the InventoryAgent
        """
        state = _read_workflow_state(session_id)                      # 1. read
        if state is None:
            return _NOT_INITIALIZED
        expected_version = int(state['version'])

        trace.step_start('inventory_agent')
        trace.agent_section('INVENTORY AGENT')
        text = _run_worker_agent(                                     # 2. invoke worker
            inventory_agent,
            f"Customer ID: {customer_id}\n"
            f"Customer request: {request}\n\n"
            f"Gather the relevant order facts and the customer's tier, and report them.",
        )
        _update_workflow_state(session_id, {'inventory_agent': text},  # 3. write back
                               expected_version)
        trace.step_done('inventory_agent', expected_version)
        return text

    @tool
    def route_to_policy_agent(session_id: str, request: str) -> str:
        """
        Route a policy question to the Policy Agent (multi-agent RAG).
        Call this for questions about return policies, shipping, or warranties.

        Args:
            session_id: The current session identifier
            request:    The customer's policy question

        Returns:
            Policy information retrieved and synthesized by PolicyAgent
        """
        state = _read_workflow_state(session_id)
        if state is None:
            return _NOT_INITIALIZED
        expected_version = int(state['version'])

        trace.step_start('policy_agent')
        trace.agent_section('POLICY AGENT')
        text = _run_worker_agent(policy_agent, request)
        _update_workflow_state(session_id, {'policy_agent': text}, expected_version)
        trace.step_done('policy_agent', expected_version)
        return text

    @tool
    def route_to_refund_agent(session_id: str, customer_id: str, request: str) -> str:
        """
        Route a return/refund request to the Refund Agent.
        Call this AFTER route_to_inventory_agent has gathered order facts.

        Args:
            session_id:  The current session identifier
            customer_id: The customer's unique identifier
            request:     The return/refund request

        Returns:
            Refund decision from the RefundAgent
        """
        state = _read_workflow_state(session_id)
        if state is None:
            return _NOT_INITIALIZED
        expected_version = int(state['version'])

        trace.step_start('refund_agent')
        trace.agent_section('REFUND AGENT')
        text = _run_worker_agent(
            refund_agent,
            f"Session ID: {session_id}\n"
            f"Customer ID: {customer_id}\n"
            f"Customer request: {request}\n\n"
            f"Decide whether this return is allowed and, if so, start it.",
        )
        _update_workflow_state(session_id, {'refund_agent': text}, expected_version)
        trace.step_done('refund_agent', expected_version)
        return text

    @tool
    def route_to_communication_agent(session_id: str, customer_id: str,
                                     original_request: str) -> str:
        """
        Route to the Communication Agent to compose the final customer response.
        Call this LAST - after all relevant worker agents have run.

        Args:
            session_id:       The current session identifier
            customer_id:      The customer's unique identifier
            original_request: The customer's original message

        Returns:
            Final customer-facing response drafted by CommunicationAgent
        """
        state = _read_workflow_state(session_id)
        if state is None:
            return _NOT_INITIALIZED
        expected_version = int(state['version'])

        trace.step_start('communication_agent')
        trace.agent_section('COMMUNICATION AGENT')
        text = _run_worker_agent(
            communication_agent,
            f"Session ID: {session_id}\n"
            f"Customer ID: {customer_id}\n"
            f"Customer's original request: {original_request}\n\n"
            f"Read the workflow context and write the final reply to the customer.",
        )
        _update_workflow_state(session_id, {'communication_agent': text}, expected_version)
        trace.step_done('communication_agent', expected_version)
        return text

    @tool
    def initialize_session(session_id: str, customer_id: str) -> str:
        """
        Create a blank WorkflowState record at the start of each new session.
        Call this at the VERY BEGINNING of processing every customer request.

        Args:
            session_id:  A unique identifier for this session
            customer_id: The customer's identifier

        Returns:
            Confirmation that the session was initialized
        """
        try:
            _create_workflow_state(session_id, customer_id)
        except ClientError as exc:
            if exc.response['Error']['Code'] != 'ConditionalCheckFailedException':
                raise
            # Multi-turn chat reuses one session id: keep the existing record.
            existing = _read_workflow_state(session_id) or {}
            return (f"Session {session_id} already initialized "
                    f"(version {int(existing.get('version', 0))}). Continue routing.")
        return f"Session {session_id} initialized for customer {customer_id} (version 0)."

    return Agent(
        name='OrchestratorAgent',
        model=model,
        system_prompt=system_prompt,
        tools=[initialize_session, route_to_inventory_agent, route_to_policy_agent,
               route_to_refund_agent, route_to_communication_agent],
    )


# ═══════════════════════════════════════════════════════
#  AGENT GRAPH HELPERS
# ═══════════════════════════════════════════════════════

def _apply_guardrail(agents: list) -> None:
    """
    Attach the Bedrock Guardrail (Task 3) to every agent's BedrockModel.
    Guardrails are enforced per model invocation, so once GUARDRAIL_ID /
    GUARDRAIL_VERSION are known (in .env locally, as runtime environment
    variables when deployed) every agent in the graph runs behind the
    guardrail - no change to the agents themselves is needed.
    """
    guardrail_id      = config.GUARDRAIL_ID
    guardrail_version = config.GUARDRAIL_VERSION
    if not guardrail_id or not guardrail_version:
        return
    for agent in agents:
        model = getattr(agent, 'model', None)
        if model is not None and hasattr(model, 'update_config'):
            model.update_config(guardrail_id=guardrail_id,
                                guardrail_version=guardrail_version)


def build_agent_graph(verbose: bool = False) -> Agent:
    """Build all five agents, apply the guardrail, return the orchestrator."""
    def _ok(label):
        if verbose:
            print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  {label}{_C.RESET}", flush=True)

    inventory_agent     = build_inventory_agent();     _ok('InventoryAgent')
    refund_agent        = build_refund_agent();        _ok('RefundAgent')
    policy_agent        = build_policy_agent();        _ok('PolicyAgent')
    communication_agent = build_communication_agent(); _ok('CommunicationAgent')
    orchestrator = build_orchestrator_agent(
        inventory_agent, refund_agent, policy_agent, communication_agent
    )
    _ok('Orchestrator')
    _apply_guardrail([inventory_agent, refund_agent, policy_agent,
                      communication_agent, orchestrator])
    if verbose and config.GUARDRAIL_ID:
        print(f"  {_C.GRY}          Guardrail {config.GUARDRAIL_ID} "
              f"(v{config.GUARDRAIL_VERSION}) attached to all agents{_C.RESET}")
    return orchestrator


# ═══════════════════════════════════════════════════════
#  DEPLOYMENT TOOLING - AgentCore CLI
#
#  The runtime is deployed with the AgentCore CLI (`agentcore`, npm package
#  @aws/agentcore - https://github.com/aws/agentcore-cli) through the
#  pre-written helper src/agentcore_cli.py:
#
#    agentcore_cli.stage_runtime_code()   copies this file, its helper modules
#                                         and config.py to build/runtime/ with a
#                                         pyproject.toml of the runtime deps
#    agentcore_cli.configure_runtime()    writes the runtime settings (network
#                                         mode, protocol, execution role, env
#                                         vars) to agentcore/agentcore.json
#    agentcore_cli.deploy()               runs `agentcore deploy -y`: the CLI
#                                         downloads arm64 / Python 3.12 wheels
#                                         with uv, zips them with the code
#                                         (direct code deployment) and creates
#                                         or updates the runtime via CDK
#    agentcore_cli.deployed_runtime_arn() reads the ARN the CLI recorded
#
#  Inside the runtime this same file is the entry point: it is started with
#  no command-line argument and serves HTTP (see run_serve). The marker file
#  written next to it by stage_runtime_code() tells __main__ to do so.
# ═══════════════════════════════════════════════════════

_RUNTIME_MARKER = '.agentcore-runtime'         # written by agentcore_cli.stage_runtime_code()
_SRC_DIR        = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════
#  TASK 3 - AGENTCORE DEPLOYMENT + GUARDRAILS
# ═══════════════════════════════════════════════════════

def create_guardrail() -> tuple[str, str]:
    """
    Create a Bedrock Guardrail for enterprise safety enforcement.

    Blocks harmful content, PII exposure, off-topic subjects, and profanity.
    Returns (guardrail_id, guardrail_version).
    """
    bedrock_client = boto3.client('bedrock', region_name=config.AWS_REGION)

    # Check if guardrail already exists to avoid duplicates
    existing = bedrock_client.list_guardrails()
    for g in existing.get('guardrails', []):
        if g['name'] == config.GUARDRAIL_NAME:
            guardrail_id = g['id']
            versions = bedrock_client.list_guardrails(guardrailIdentifier=guardrail_id)
            guardrail_version = 'DRAFT'
            for v in versions.get('guardrails', []):
                if v.get('version', 'DRAFT') != 'DRAFT':
                    guardrail_version = v['version']
            print(f"Guardrail already exists: {guardrail_id} (version: {guardrail_version})")
            return guardrail_id, guardrail_version

    # Denied topics - one DENY topic per entry in config.GUARDRAIL_BLOCKED_TOPICS.
    # Definitions are kept under the 200-character limit. The pricing topic is
    # defined as haggling / changing an advertised price and explicitly EXCLUDES
    # arithmetic on a price already stated (the project's math scenario).
    topic_details = {
        "competitor products": {
            "definition": ("Requests to compare NovaMart with, recommend, or promote the "
                           "products, prices or stores of competing retailers or "
                           "e-commerce companies."),
            "examples": [
                "Is Amazon cheaper than NovaMart for this laptop?",
                "Which competitor store has better headphones than you?",
                "Should I buy this at Best Buy instead?",
            ],
        },
        "pricing negotiations": {
            "definition": ("Haggling, bargaining or asking to change or lower an advertised "
                           "price or get a special deal. Not simple arithmetic on a price "
                           "and discount the user already gave."),
            "examples": [
                "Can you knock 30 percent off the listed price if I buy today?",
                "I will pay half of the listed price, do we have a deal?",
                "Give me a special discount or I will shop elsewhere.",
            ],
        },
        "legal threats": {
            "definition": ("Threats of lawsuits or legal action, mentions of suing, lawyers "
                           "or regulators against NovaMart, and requests for legal advice "
                           "or liability rulings."),
            "examples": [
                "I will sue NovaMart if you do not refund me right now.",
                "My lawyer will contact you about this order.",
                "Is NovaMart legally liable for my damaged package?",
            ],
        },
    }
    topics_config = [
        {
            'name':       topic.title(),
            'definition': topic_details[topic]['definition'],
            'examples':   topic_details[topic]['examples'],
            'type':       'DENY',
        }
        for topic in config.GUARDRAIL_BLOCKED_TOPICS
    ]

    response = bedrock_client.create_guardrail(
        name=config.GUARDRAIL_NAME,
        description=("NovaMart enterprise guardrail: blocks harmful content, protects PII "
                     "and denies competitor, price-negotiation and legal-threat topics."),
        # Content filters (applied to both user input and model output)
        contentPolicyConfig={
            'filtersConfig': [
                {'type': 'SEXUAL',     'inputStrength': 'HIGH',   'outputStrength': 'HIGH'},
                {'type': 'VIOLENCE',   'inputStrength': 'HIGH',   'outputStrength': 'HIGH'},
                {'type': 'HATE',       'inputStrength': 'HIGH',   'outputStrength': 'HIGH'},
                {'type': 'INSULTS',    'inputStrength': 'MEDIUM', 'outputStrength': 'MEDIUM'},
                {'type': 'MISCONDUCT', 'inputStrength': 'MEDIUM', 'outputStrength': 'MEDIUM'},
            ]
        },
        # PII: block card numbers and SSNs outright; mask emails and phone numbers
        sensitiveInformationPolicyConfig={
            'piiEntitiesConfig': [
                {'type': 'CREDIT_DEBIT_CARD_NUMBER',  'action': 'BLOCK'},
                {'type': 'US_SOCIAL_SECURITY_NUMBER', 'action': 'BLOCK'},
                {'type': 'EMAIL',                     'action': 'ANONYMIZE'},
                {'type': 'PHONE',                     'action': 'ANONYMIZE'},
            ]
        },
        # Denied topics use the STANDARD tier, which requires a cross-region profile
        topicPolicyConfig={
            'topicsConfig': topics_config,
            'tierConfig':   {'tierName': 'STANDARD'},
        },
        crossRegionConfig={'guardrailProfileIdentifier': 'us.guardrail.v1:0'},
        # Managed profanity list
        wordPolicyConfig={'managedWordListsConfig': [{'type': 'PROFANITY'}]},
        blockedInputMessaging=("I'm sorry, but I can't help with that request. "
                               "I'm happy to help with orders, returns, shipping or "
                               "warranty questions instead."),
        blockedOutputsMessaging=("I'm sorry, but I can't share that response. "
                                 "Please rephrase your question, or ask me about your "
                                 "orders, returns, shipping or warranty."),
        clientRequestToken=str(uuid.uuid4()),
    )
    guardrail_id = response['guardrailId']
    print(f"Guardrail created: {guardrail_id} (DRAFT) - waiting for it to be READY",
          end='', flush=True)

    # Creation is asynchronous: a version can only be cut once the DRAFT is READY.
    def _wait_until_ready(version: str, timeout: int = 300) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = bedrock_client.get_guardrail(
                guardrailIdentifier=guardrail_id, guardrailVersion=version
            ).get('status')
            if status == 'READY':
                print(' ready.')
                return
            if status in ('FAILED', 'DELETING'):
                raise RuntimeError(f"Guardrail {guardrail_id} v{version} entered status {status}")
            print('.', end='', flush=True)
            time.sleep(3)
        raise TimeoutError(f"Guardrail {guardrail_id} v{version} not READY after {timeout}s")

    _wait_until_ready('DRAFT')

    # Promote DRAFT -> immutable numbered version (agents must not run on DRAFT)
    version_response = bedrock_client.create_guardrail_version(
        guardrailIdentifier=guardrail_id,
        description='Initial published version',
        clientRequestToken=str(uuid.uuid4()),
    )
    guardrail_version = str(version_response['version'])
    print(f"Guardrail version {guardrail_version} created - waiting for it to be READY",
          end='', flush=True)
    _wait_until_ready(guardrail_version)

    return guardrail_id, guardrail_version


def deploy_to_agentcore_runtime(
    orchestrator_agent: Agent,
    guardrail_id: str,
    guardrail_version: str
) -> str:
    """
    Deploy the multi-agent system to Amazon Bedrock AgentCore Runtime with
    the AgentCore CLI (src/agentcore_cli.py wraps it).

    AgentCore does not serialize Python objects, so `orchestrator_agent` is
    not uploaded directly. Instead the pre-written staging step copies this
    file, which doubles as the HTTP entry point (see run_serve), together
    with its helper modules and config.py to build/runtime/. `agentcore
    deploy` then packages that directory with arm64 dependencies and creates
    or updates the runtime ("direct code deployment"). Re-running is safe:
    an unchanged runtime is left alone, a changed one is updated in place.

    The guardrail is attached by environment variables: inside the runtime
    build_agent_graph() reads GUARDRAIL_ID / GUARDRAIL_VERSION and applies
    them to every agent's model (see _apply_guardrail), exactly as `test`
    and `chat` do locally.

    Returns:
        The AgentCore Runtime ARN
    """
    import agentcore_cli

    runtime_name = config.AGENTCORE_RUNTIME_NAME
    print(f"  AWS Account: {config.ACCOUNT_ID}  |  Region: {config.AWS_REGION}")
    print(f"  Runtime: {runtime_name}  |  CLI project: agentcore/agentcore.json "
          f"(stack {agentcore_cli.stack_name()})")
    previous_arn = agentcore_cli.deployed_runtime_arn()
    if previous_arn:
        print(f"  Runtime already deployed - updating it: {previous_arn}")

    # Stage the code the CLI packages (src modules + config.py + pyproject.toml).
    agentcore_cli.stage_runtime_code()

    # 1. Runtime environment variables (read by the deployed agent via config.py,
    #    and by _apply_guardrail() to attach the guardrail to every agent).
    runtime_env = {
        'AWS_REGION':        config.AWS_REGION,
        'PROJECT_NAME':      config.PROJECT_NAME,
        'RETURNS_KB_ID':     config.RETURNS_KB_ID,
        'SHIPPING_KB_ID':    config.SHIPPING_KB_ID,
        'WARRANTY_KB_ID':    config.WARRANTY_KB_ID,
        'AGENT_LOG_GROUP':   config.AGENT_LOG_GROUP,
        'GUARDRAIL_ID':      guardrail_id,
        'GUARDRAIL_VERSION': str(guardrail_version),     # numbered version, never DRAFT
    }
    missing_kbs = [k for k in ('RETURNS_KB_ID', 'SHIPPING_KB_ID', 'WARRANTY_KB_ID')
                   if not runtime_env[k]]
    if missing_kbs:
        print(f"  [Warning] {', '.join(missing_kbs)} not set in .env - the runtime "
              f"would deploy without them. Complete Task 5 first.")

    # 2. Write the runtime settings to agentcore/agentcore.json
    agentcore_cli.configure_runtime(
        env_vars=runtime_env,
        network_mode='PUBLIC',
        protocol='HTTP',
        execution_role_arn=config.AGENTCORE_ROLE_ARN,
    )

    # 3. Package and deploy (runs `agentcore deploy -y`; CDK/CloudFormation underneath)
    agentcore_cli.deploy()

    # 4. Read back the ARN the CLI recorded
    runtime_arn = agentcore_cli.deployed_runtime_arn()

    if not runtime_arn:
        raise RuntimeError("AgentCore CLI deployment finished but no runtime ARN was recorded "
                           "(check `agentcore status` and the CLI logs).")

    # Wait for the runtime to become READY and return its ARN.
    print(f"  Runtime deployed: {runtime_arn}")
    print("  Waiting for runtime status READY", end='', flush=True)
    wait_for_runtime_ready(agentcore_control, runtime_arn.split('/')[-1])
    print(' ready.')
    return runtime_arn


# ═══════════════════════════════════════════════════════
#  TASK 4 - MEMORY
# ═══════════════════════════════════════════════════════

def configure_memory(runtime_arn: str) -> str:
    """
    Create an AgentCore Memory resource for session-scoped conversational
    context. Uses the SESSION_SUMMARY (summaryMemoryStrategy) strategy with
    7-day event retention.

    Returns:
        The memory resource ARN
    """
    memory_name = config.MEMORY_NAME
    existing = agentcore_control.list_memories()
    for m in existing.get('memories', []):
        if m['id'].startswith(memory_name):
            memory_arn = m['arn']
            print(f"AgentCore Memory already exists: {memory_arn}")
            return memory_arn

    response = agentcore_control.create_memory(
        name=memory_name,
        description=("NovaMart customer support memory: stores a summary of each "
                     "conversation session so the agents keep multi-turn context."),
        eventExpiryDuration=7,                                   # days of event retention
        memoryStrategies=[{
            'summaryMemoryStrategy': {                           # SESSION_SUMMARY strategy
                'name': 'SessionSummary',
                'namespaces': ['/summaries/{actorId}/{sessionId}'],
            }
        }],
        clientToken=str(uuid.uuid4()),                           # idempotency
    )

    # Wait until the memory resource is ACTIVE and return its ARN.
    memory = response['memory']
    print(f"  Memory created: {memory['arn']}  (status: {memory['status']})")
    print("  Waiting for memory status ACTIVE", end='', flush=True)
    deadline = time.time() + 300
    while memory['status'] != 'ACTIVE' and time.time() < deadline:
        time.sleep(10)
        print('.', end='', flush=True)
        memory = agentcore_control.get_memory(memoryId=memory['id'])['memory']
        if memory['status'] == 'FAILED':
            raise RuntimeError(f"Memory creation failed: {memory.get('failureReason')}")
    print(' ready.' if memory['status'] == 'ACTIVE' else f" status {memory['status']}")
    return memory['arn']


# ═══════════════════════════════════════════════════════
#  TASK 6 - OBSERVABILITY
# ═══════════════════════════════════════════════════════

def configure_observability(runtime_arn: str) -> None:
    """
    Configure observability for the deployed agent:
    - Agent logs → CloudWatch Logs at INFO level (config.AGENT_LOG_GROUP)
    - Execution traces → AWS X-Ray at 100% sampling

    The loggingConfiguration built here is applied by
    apply_observability_config() (agent_observability.py):
      cloudWatchConfig -> log group created; runtime env AGENT_LOG_GROUP /
                          AGENT_LOG_LEVEL so the deployed agent ships its logs there
      xRayConfig       -> CloudWatch Transaction Search enabled with the given
                          sampling percentage; runtime env AGENT_TRACING_ENABLED /
                          AGENT_TRACE_SAMPLING_RATE
    """
    logging_configuration = {
        'cloudWatchConfig': {
            'logGroupName': config.AGENT_LOG_GROUP,
            'logLevel':     'INFO',
            'enabled':      True,
        },
        'xRayConfig': {
            'enabled':      True,
            'samplingRate': 1.0,          # trace 100% of requests
        },
    }

    try:
        summary = apply_observability_config(runtime_arn, logging_configuration)
        print(f"  CloudWatch log group : {summary.get('log_group', config.AGENT_LOG_GROUP)}")
        print(f"  X-Ray sampling rate  : "
              f"{logging_configuration['xRayConfig']['samplingRate']:.0%}")
    except Exception as e:
        print(f"[Note] Observability configuration failed: {e}")


# ═══════════════════════════════════════════════════════
#  AGENTCORE GATEWAY DEPLOYMENT
#
#  Production equivalent of in-process @tool functions.
#  Registers Lambda-backed tools on a managed MCP endpoint so tools
#  can be independently deployed, versioned, and discovered at runtime.
#
#  Deployment pattern:
#    Local dev  → LambdaGateway + gateway.register_target(...)
#    Production → deploy_agentcore_gateway() using real AWS API
#
#  Requires Lambda tool functions to be deployed separately.
#  Set ORDERS_FUNCTION, POLICY_FUNCTION, CUSTOMERS_FUNCTION in .env
#  to the deployed Lambda function names.
# ═══════════════════════════════════════════════════════

# Lambda function names for gateway tool backends (set in .env after deploying)
_ORDERS_FUNCTION = os.environ.get('ORDERS_FUNCTION', '')
_POLICY_FUNCTION = os.environ.get('POLICY_FUNCTION', '')
_CUSTOMERS_FUNCTION = os.environ.get('CUSTOMERS_FUNCTION', '')


def _gw_get_function_arn(function_name: str) -> str:
    """Resolve a Lambda function name to its full ARN."""
    lambda_client = boto3.client('lambda', region_name=config.AWS_REGION)
    resp = lambda_client.get_function(FunctionName=function_name)
    return resp['Configuration']['FunctionArn']


def _gw_stack_uuid() -> str:
    """Return the short UUID from the project CloudFormation stack ID.
    Gives the gateway a stable name so re-runs never hit ConflictException."""
    cf = boto3.client('cloudformation', region_name=config.AWS_REGION)
    stacks = cf.describe_stacks(StackName=config.PROJECT_NAME)
    stack_id = stacks['Stacks'][0]['StackId']
    full_uuid = stack_id.split('/')[-1]
    return full_uuid.split('-')[0]


def _gw_wait_for_ready(agentcore_ctrl, gateway_id: str, timeout: int = 120) -> str:
    """Poll until the gateway reaches READY status. Returns the gateway URL."""
    deadline = time.time() + timeout
    first    = True
    while time.time() < deadline:
        gw     = agentcore_ctrl.get_gateway(gatewayIdentifier=gateway_id)
        status = gw['status']
        if status == 'READY':
            if not first:
                print(' ready.')
            return gw.get('gatewayUrl', '')
        if 'FAILED' in status:
            print(f' failed: {status}')
            raise RuntimeError(f"Gateway {gateway_id} entered status {status}")
        if first:
            print('    Gateway provisioning (async — normal AWS behaviour)',
                  end='', flush=True)
            first = False
        print('.', end='', flush=True)
        time.sleep(5)
    raise TimeoutError(f"Gateway {gateway_id} not READY after {timeout}s")


def _gw_get_or_create(agentcore_ctrl, name: str, role_arn: str,
                       instructions: str) -> tuple[str, str]:
    """Create an AgentCore Gateway, or reuse it if it already exists."""
    try:
        gw = agentcore_ctrl.create_gateway(
            name=name,
            roleArn=role_arn,
            protocolType='MCP',
            authorizerType='NONE',
            protocolConfiguration={'mcp': {'instructions': instructions,
                                            'searchType': 'SEMANTIC'}},
        )
        gw_id  = gw['gatewayId']
        print(f'    Gateway ID  : {gw_id}')
        print(f'    Status      : {gw["status"]}')
        gw_url = _gw_wait_for_ready(agentcore_ctrl, gw_id)
        print(f'    Gateway URL : {gw_url}')
        return gw_id, gw_url
    except agentcore_ctrl.exceptions.ConflictException:
        print(f"    Gateway '{name}' already exists — reusing it.")
        gateways = agentcore_ctrl.list_gateways().get('items', [])
        existing = next((g for g in gateways if g['name'] == name), None)
        if not existing:
            raise RuntimeError(f"Gateway '{name}' not found after ConflictException")
        gw_id  = existing['gatewayId']
        print(f'    Gateway ID  : {gw_id}')
        gw_url = _gw_wait_for_ready(agentcore_ctrl, gw_id)
        print(f'    Gateway URL : {gw_url}')
        return gw_id, gw_url


def _gw_create_target(agentcore_ctrl, gateway_id: str, t: dict,
                       lambda_arn: str) -> None:
    """Register one Lambda target on the gateway. Skips if it already exists."""
    payload = dict(
        gatewayIdentifier=gateway_id,
        name=t['name'],
        description=t['description'],
        targetConfiguration={
            'mcp': {
                'lambda': {
                    'lambdaArn': lambda_arn,
                    'toolSchema': {
                        'inlinePayload': [{
                            'name':        t['tool_name'],
                            'description': t['tool_description'],
                            'inputSchema': {
                                'type': 'object',
                                'properties': {
                                    t['param_name']: {
                                        'type':        'string',
                                        'description': t['param_desc'],
                                    }
                                },
                                'required': [t['param_name']],
                            },
                        }]
                    },
                }
            }
        },
        credentialProviderConfigurations=[
            {'credentialProviderType': 'GATEWAY_IAM_ROLE'}
        ],
    )
    try:
        resp = agentcore_ctrl.create_gateway_target(**payload)
        print(f"    [{resp['status']:12s}] {t['name']} → target {resp['targetId']}")
    except agentcore_ctrl.exceptions.ConflictException:
        print(f"    [already exists] {t['name']} — skipped")


def deploy_agentcore_gateway() -> dict:
    """
    Create an AgentCore Gateway and register the NovaMart tool Lambda targets.

    Optional extension to the in-process @tool functions. Resolve configured
    Lambda functions first; if none exist, skip gateway creation. Otherwise
    create/reuse the gateway and submit its targets. Connecting agents to this
    MCP endpoint requires separate integration; this starter uses in-process tools.

    Requires Lambda tool functions to be deployed via a separate stack.
    Set ORDERS_FUNCTION, POLICY_FUNCTION, CUSTOMERS_FUNCTION in .env.

    Returns:
        A SKIPPED result with a reason, or gateway details and target count.
    """
    targets = [
        {
            'name':             'orders-api',
            'description':      'Look up order details, status, and return eligibility for a customer',
            'function':         _ORDERS_FUNCTION,
            'tool_name':        'check_order_status',
            'tool_description': 'Check order status and return eligibility for a specific order',
            'param_name':       'order_id',
            'param_desc':       'Order ID (e.g. ORD-27176)',
        },
        {
            'name':             'policy-api',
            'description':      'Retrieve return, shipping, and warranty policy text from knowledge bases',
            'function':         _POLICY_FUNCTION,
            'tool_name':        'search_policies',
            'tool_description': 'Search all policy knowledge bases for a customer query',
            'param_name':       'query',
            'param_desc':       'Customer question about returns, shipping, or warranty',
        },
        {
            'name':             'customers-api',
            'description':      'Look up customer tier (Standard or Premium) and account details',
            'function':         _CUSTOMERS_FUNCTION,
            'tool_name':        'get_customer_tier',
            'tool_description': 'Get customer tier and account information by customer ID',
            'param_name':       'customer_id',
            'param_desc':       'Customer ID (e.g. CUST-001)',
        },
    ]

    # Resolve optional Lambda targets before creating any gateway resources.
    available = []
    for target in targets:
        if not target['function']:
            continue
        try:
            available.append((target, _gw_get_function_arn(target['function'])))
        except ClientError as exc:
            if exc.response['Error']['Code'] != 'ResourceNotFoundException':
                raise
            print(f"    [Skipped] {target['name']}: Lambda function not found")

    if not available:
        return {'status': 'SKIPPED', 'reason': 'No configured Lambda tool functions are available.'}

    agentcore_ctrl = boto3.client('bedrock-agentcore-control',
                                   region_name=config.AWS_REGION)

    try:
        gw_uuid = _gw_stack_uuid()
    except Exception:
        gw_uuid = config.PROJECT_NAME

    gw_name = f"novamart-support-{gw_uuid}"
    print(f"  Calling create_gateway (name: {gw_name})...")
    gateway_id, gateway_url = _gw_get_or_create(
        agentcore_ctrl, gw_name, config.AGENTCORE_ROLE_ARN,
        "NovaMart customer support gateway. Provides order lookup, "
        "policy search, and customer tier tools.",
    )

    print(f"\n  Registering {len(available)} Gateway targets...")
    for target, lambda_arn in available:
        _gw_create_target(agentcore_ctrl, gateway_id, target, lambda_arn)

    return {'gateway_id': gateway_id, 'gateway_url': gateway_url,
            'status': 'TARGETS_SUBMITTED', 'target_count': len(available)}



# ═══════════════════════════════════════════════════════
#  RUNTIME INVOCATION
# ═══════════════════════════════════════════════════════

def invoke_agent(session_id: str, customer_id: str, user_message: str) -> dict:
    """
    Invoke the deployed agent via AgentCore Runtime (see run_serve).

    AgentCore requires runtimeSessionId to be at least 33 characters, so the
    short project session id is embedded in a longer, unique runtime session id.
    """
    if not config.AGENTCORE_RUNTIME_ARN:
        raise RuntimeError("AGENTCORE_RUNTIME_ARN is not set - run the deploy command first")

    runtime_session_id = f"{session_id}-{uuid.uuid4().hex}"     # >= 33 chars
    payload = json.dumps({
        'prompt':      user_message,
        'session_id':  session_id,
        'customer_id': customer_id,
    })
    response = agentcore_client.invoke_agent_runtime(
        agentRuntimeArn=config.AGENTCORE_RUNTIME_ARN,
        runtimeSessionId=runtime_session_id,
        contentType='application/json',
        accept='application/json',
        payload=payload,
    )
    body = response['response'].read()
    try:
        return json.loads(body)
    except (TypeError, ValueError):
        return {'result': body.decode('utf-8', errors='replace') if isinstance(body, bytes) else str(body)}


# ═══════════════════════════════════════════════════════
#  DEPLOYMENT ENTRY POINT
# ═══════════════════════════════════════════════════════

def deploy_all():
    """Full deployment pipeline. Run after completing all tasks."""
    print("\n" + "="*60)
    print("  Deploying Enterprise Multi-Agent System")
    print("="*60 + "\n")

    # Fail fast if the AgentCore CLI (used by Steps 3 and 5) is missing.
    import agentcore_cli
    print(f"AgentCore CLI: {agentcore_cli.cli_version()} ({agentcore_cli.cli_path()})\n")

    print("Step 1/6: Building agent graph...")
    inventory_agent     = build_inventory_agent()
    refund_agent        = build_refund_agent()
    policy_agent        = build_policy_agent()
    communication_agent = build_communication_agent()
    orchestrator = build_orchestrator_agent(
        inventory_agent, refund_agent, policy_agent, communication_agent
    )
    print("  All 5 agents initialized\n")

    print("Step 2/6: Creating Bedrock Guardrail...")
    guardrail_id, guardrail_version = create_guardrail()
    print()

    print("Step 3/6: Deploying to AgentCore Runtime...")
    runtime_arn = deploy_to_agentcore_runtime(orchestrator, guardrail_id, guardrail_version)
    print()

    print("Step 4/6: Configuring Memory...")
    memory_arn = configure_memory(runtime_arn)
    print()

    print("Step 5/6: Configuring Observability...")
    configure_observability(runtime_arn)
    print()

    print("Step 6/6: Deploying AgentCore Gateway...")
    try:
        gw = deploy_agentcore_gateway()
        if gw['status'] == 'SKIPPED':
            print(f"  [Skipped] Gateway: {gw['reason']}")
        else:
            print(f"  Gateway URL : {gw['gateway_url']}")
            print("  Lambda targets submitted; connect an MCP client separately to use them.")
    except Exception as e:
        print(f"  [Note] Optional Gateway deployment failed: {e}")
        print(f"  (Deploy Lambda tool functions and set ORDERS_FUNCTION etc. in .env to enable)")
    print()

    print("="*60)
    print("  Deployment Complete!")
    print("="*60)
    print(f"\n  Add these to your .env file:")
    print(f"  AGENTCORE_RUNTIME_ARN={runtime_arn}")
    print(f"  GUARDRAIL_ID={guardrail_id}")
    print(f"  GUARDRAIL_VERSION={guardrail_version}\n")
    print(f"  Then try the deployed runtime:")
    print(f"  python src/agent_orchestrator.py invoke \"What is the return policy for premium customers?\"")
    print(f"  or with the CLI:  agentcore invoke \"What is the return policy for premium customers?\"")
    print(f"  (agentcore status / agentcore logs show the deployed runtime and its logs)\n")
    return runtime_arn, guardrail_id


# ═══════════════════════════════════════════════════════
#  LOCAL TEST SCENARIOS
# ═══════════════════════════════════════════════════════

# Order IDs match infrastructure/seed_data.py.
TEST_CASES = [
    ("CUST-001", "I want to return my wireless headphones from order ORD-27176"),
    ("CUST-002", "What is the return policy for premium customers?"),
    ("CUST-003", "How much would 5 items at $29.99 be with a 10% discount?"),
]

# Test customers shown by the chat command. Data matches seed_data.py.
TEST_CUSTOMERS = [
    ("CUST-001", "Alice Johnson", "Premium",  "ORD-27176", "Wireless Headphones Pro"),
    ("CUST-002", "Bob Smith",     "Standard", "ORD-28001", "Mechanical Keyboard K2"),
    ("CUST-003", "Carol Davis",   "Premium",  "ORD-29001", "Laptop UltraBook 14"),
    ("CUST-004", "David Lee",     "Standard", "ORD-30001", "Phone Case Slim"),
]


def run_test_scenarios() -> None:
    """Run the three scenarios locally; every request is traced to X-Ray."""
    print("Running local agent test...")
    setup_logging(to_cloudwatch=True)
    orchestrator = build_agent_graph()

    for customer_id, query in TEST_CASES:
        session_id = str(uuid.uuid4())[:8]
        print(f"\n{'─'*60}")
        print(f"Session: {session_id} | Customer: {customer_id}")
        print(f"Query: {query}")
        prompt = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {query}"
        with tracer.trace_request(session_id, customer_id, query):
            response = orchestrator(prompt)
        print(f"Response: {response}")
        print_trace_hint()
    flush_logs()


def run_chat() -> None:
    """Interactive terminal chat - educational mode."""
    W = _C.W

    # ── Welcome banner ────────────────────────────────────────────────
    print()
    print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
    print(f"  {_C.ORCH}{_C.BOLD}{'NovaMart -- Multi-Agent Customer Support':^{W}}{_C.RESET}")
    print(f"  {_C.GRY}{'Strands Agents SDK  +  Amazon Bedrock AgentCore':^{W}}{_C.RESET}")
    print(f"  {_C.GRY}{'=' * W}{_C.RESET}")

    # ── Test customers ────────────────────────────────────────────────
    print()
    print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
    print(f"  {_C.BOLD}Test Customers{_C.RESET}")
    print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
    print(f"  {_C.GRY}{'ID':<10}  {'Name':<18}  {'Tier':<10}  {'Order':<12}  Product{_C.RESET}")
    print(f"  {_C.GRY}{'─'*8}  {'─'*16}  {'─'*8}  {'─'*10}  {'─'*20}{_C.RESET}")
    for cid, name, tier, order, product in TEST_CUSTOMERS:
        tier_col = _C.INV if tier == 'Premium' else _C.GRY
        print(f"  {_C.BOLD}{cid}{_C.RESET}  {name:<18}  "
              f"{tier_col}{tier:<10}{_C.RESET}  {order}  {product}")
    print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
    print()

    customer_id = (
        input(f"  Enter Customer ID (default: CUST-001): ").strip()
        or "CUST-001"
    )
    session_id  = str(uuid.uuid4())[:8]
    print()
    print(f"  {_C.GRY}Session  : {_C.RESET}{_C.BOLD}{session_id}{_C.RESET}")
    print(f"  {_C.GRY}Customer : {_C.RESET}{_C.BOLD}{customer_id}{_C.RESET}")
    print(f"  {_C.GRY}Type a question and press Enter.  Type 'quit' to exit.{_C.RESET}")
    print()

    # ── Build agents and show initialization order.
    print(f"  {_C.GRY}[SYSTEM]  Initializing agent graph...{_C.RESET}")
    setup_logging(to_cloudwatch=True)
    orchestrator = build_agent_graph(verbose=True)
    print(f"  {_C.GRY}[SYSTEM]  All 5 agents ready.{_C.RESET}")
    print()

    # ── Conversation loop ─────────────────────────────────────────────
    while True:
        try:
            user_input = input(
                f"  {_C.BOLD}You >{_C.RESET} "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {_C.GRY}Session ended.{_C.RESET}")
            break

        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit', 'q'):
            print(f"  {_C.GRY}Session ended.{_C.RESET}")
            break

        prompt  = (f"[Session ID: {session_id}] "
                   f"[Customer ID: {customer_id}] {user_input}")
        t0_turn = time.time()

        # ── Install proxy, run orchestrator (traced), restore stdout ───
        trace.new_turn()
        sys.stdout = _trace_writer
        try:
            with tracer.trace_request(session_id, customer_id, user_input):
                response = orchestrator(prompt)
        finally:
            sys.stdout = _real_stdout   # always restore, even on exception

        elapsed = time.time() - t0_turn

        # ── Resolve the final customer-facing text ────────────────────
        final_state = _read_workflow_state(session_id) or {}
        comm_result = final_state.get('communication_agent', '')
        text = _strip_xml_tags(comm_result or str(response))

        # ── DynamoDB workflow state summary ───────────────────────────
        trace.summary(session_id, elapsed)

        # ── Final customer-facing response ────────────────────────────
        print()
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
        print(f"  {_C.COM}{_C.BOLD}AGENT RESPONSE{_C.RESET}")
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
        for line in text.splitlines():
            print(f"  {line}")
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
        if tracer.last_trace_id:
            print(f"  {_C.GRY}X-Ray trace : {tracer.last_trace_id}"
                  f"{'' if tracer.last_published else '  (not published)'}{_C.RESET}")
        print()
    flush_logs()


def run_invoke(message: str, customer_id: str = "CUST-001") -> None:
    """Send one message to the deployed AgentCore Runtime and print the reply."""
    session_id = str(uuid.uuid4())[:8]
    print(f"Invoking {config.AGENTCORE_RUNTIME_ARN}")
    print(f"Session: {session_id} | Customer: {customer_id}")
    print(f"Query: {message}\n")
    result = invoke_agent(session_id, customer_id, message)
    print(f"Response: {result.get('result', result)}")
    if result.get('trace_id'):
        print(f"X-Ray trace: {result['trace_id']}")


def run_serve() -> None:
    """
    HTTP entry point executed inside Amazon Bedrock AgentCore Runtime.

    BedrockAgentCoreApp (bedrock-agentcore SDK) exposes the contract the
    runtime expects - POST /invocations and GET /ping on port 8080 - and hands
    each request payload to the function decorated with @app.entrypoint.

    Request payload (see invoke_agent):
        {"prompt": "<customer message>", "customer_id": "CUST-001", "session_id": "abc12345"}
    Response:
        {"result": "<final customer-facing text>", "session_id": ..., "trace_id": ...}

    The five-agent graph is built once (first request) and reused. Guardrail,
    tracing and logging are applied exactly as in the local test/chat modes,
    from the runtime's environment variables.
    """
    from bedrock_agentcore import BedrockAgentCoreApp

    os.environ.setdefault('AGENT_RUNTIME_MODE', 'agentcore-runtime')
    if os.environ.get('AGENT_LOG_GROUP') and 'AGENT_LOG_TO_CLOUDWATCH' not in os.environ:
        os.environ['AGENT_LOG_TO_CLOUDWATCH'] = 'true'

    app   = BedrockAgentCoreApp()
    lock  = threading.Lock()
    graph = {}

    def _orchestrator():
        with lock:
            if 'agent' not in graph:
                setup_logging()
                graph['agent'] = build_agent_graph()
        return graph['agent']

    @app.entrypoint
    def invoke(payload, context=None):
        payload     = payload or {}
        prompt      = payload.get('prompt') or payload.get('message') or ''
        customer_id = payload.get('customer_id') or 'CUST-001'
        session_id  = payload.get('session_id') or (
            getattr(context, 'session_id', None) or uuid.uuid4().hex)[:8]
        if not prompt:
            return {'error': "payload must include 'prompt'"}

        enriched = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {prompt}"
        with tracer.trace_request(session_id, customer_id, prompt):
            response = _orchestrator()(enriched)

        state = _read_workflow_state(session_id) or {}
        text  = _strip_xml_tags(state.get('communication_agent', '') or str(response))
        flush_logs()
        return {'result': text, 'session_id': session_id, 'customer_id': customer_id,
                'trace_id': tracer.last_trace_id}

    app.run()


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) > 1 else ''

    # Inside the AgentCore Runtime package (marker file next to this script)
    # the entry point is started without arguments -> serve HTTP.
    if not command and os.path.exists(os.path.join(_SRC_DIR, _RUNTIME_MARKER)):
        command = 'serve'

    if command == 'deploy':
        deploy_all()

    elif command == 'serve':
        run_serve()

    elif command == 'test':
        run_test_scenarios()

    elif command == 'chat':
        run_chat()

    elif command == 'invoke':
        if len(sys.argv) < 3:
            print('Usage: python src/agent_orchestrator.py invoke "<message>" [CUSTOMER_ID]')
            sys.exit(1)
        run_invoke(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "CUST-001")

    else:
        print("Usage:")
        print("  python src/agent_orchestrator.py deploy           # Deploy to AgentCore (Tasks 3-6)")
        print("  python src/agent_orchestrator.py test             # Run the 3 test scenarios locally")
        print("  python src/agent_orchestrator.py chat             # Interactive terminal chat")
        print("  python src/agent_orchestrator.py invoke \"<msg>\"   # Call the deployed runtime")
        print("  python src/agent_orchestrator.py serve            # HTTP server (used inside AgentCore Runtime)")