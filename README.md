# Enterprise Multi-Agent Architecture with Amazon Bedrock AgentCore

**Submission by Devanshi Malhotra**
Udacity **Future AWS Agent Engineer** Nanodegree
Course: *Multi-Agent Systems and Protocol Integration on AWS*
Project: **Multi-Agent E-commerce RAG (NovaMart)**

A production-grade multi-agent customer support system built with the [Strands Agents SDK](https://github.com/strands-agents/sdk-python) and Amazon Bedrock AgentCore. The system automatically routes customer support requests to specialist agents, retrieves grounded knowledge via **parallel multi-agent RAG**, enforces enterprise safety guardrails, and maintains shared workflow state across agents. It is observable through CloudWatch and X-Ray.

<img width="1140" height="824" alt="NovaMart architecture overview" src="diagrams/architecture-overview.png" />

---

## Architecture

```
Customer Request
      │
OrchestratorAgent (Claude Haiku 4.5 - routes requests, manages WorkflowState)
      │
   ┌──┼──────────────────────────┬──────────────────┐
   │  │                          │                  │
InventoryAgent    PolicyAgent    RefundAgent    CommunicationAgent
(DynamoDB)     (Multi-Agent RAG)  (DynamoDB)     (synthesizes response)
                     │
         ┌───────────┼───────────┐
    ReturnsPolicyRetriever  ShippingPolicyRetriever  WarrantyPolicyRetriever
    (Bedrock KB)          (Bedrock KB)              (Bedrock KB)
         └───────── run in PARALLEL ───────────┘

Shared State: DynamoDB WorkflowStateTable (with optimistic locking)
```

### Agent Roles

| Agent | Model | Responsibility | Tools |
|-------|-------|-----------------|-------|
| **OrchestratorAgent** | Claude Haiku 4.5 | Routes requests, creates/updates WorkflowState | `initialize_session`, `route_to_inventory_agent`, `route_to_policy_agent`, `route_to_refund_agent`, `route_to_communication_agent` |
| **InventoryAgent** | Claude Sonnet 4.5 | Gathers order and customer facts from DynamoDB | `check_order_status`, `get_customer_tier`, `list_customer_orders` |
| **PolicyAgent** | Claude Sonnet 4.5 | Coordinates parallel RAG retrieval from 3 KBs, synthesizes results | `search_all_policies` (internally fans out to 3 retriever sub-agents) |
| **RefundAgent** | Claude Sonnet 4.5 | Makes return/refund eligibility decisions (30-day Standard, 60-day Premium) | `get_inventory_context`, `initiate_refund` |
| **CommunicationAgent** | Claude Sonnet 4.5 | Drafts final, empathetic customer-facing response | `get_full_workflow_context` |

<img width="1165" height="842" alt="Agents and tools reference" src="diagrams/agents-and-tools-reference.png" />

### Key Features

- **Multi-Agent RAG**: PolicyAgent runs 3 specialized retriever sub-agents in parallel using ThreadPoolExecutor and combines their results
- **Shared Workflow State**: DynamoDB `WorkflowStateTable` with optimistic locking (version-based conditional writes) ensures consistency across concurrent agent updates
- **Enterprise Guardrails**: Bedrock Guardrails block harmful content, PII, and off-topic conversations
- **Observability**: Every request is traced to AWS X-Ray (Orchestrator → Worker → Knowledge Base call chain) and logged to CloudWatch Logs - locally and from the deployed runtime
- **Bedrock Knowledge Bases**: Uses an S3 Vectors backing store (vector bucket + one index per KB, created by the stack) for policy retrieval

---

## Submission Deliverables

| Deliverable | Location |
|---|---|
| Completed orchestrator (Tasks 2, 3, 4, 6) | `src/agent_orchestrator.py` |
| Populated environment file (KB IDs, Runtime ARN, Guardrail ID/version) | `.env` |
| Test suite result: **120/120** | `screenshots/test-all-120.png` |
| X-Ray Service Map (`NovaMart-Orchestrator` → workers → Knowledge Bases) | `screenshots/xray-service-map.png` |
| X-Ray node list (full node names, backup) | `screenshots/xray-service-map-list.png` |

**Implementation summary**

- 5 agents: Orchestrator (Claude Haiku 4.5) plus Inventory, Policy, Refund and Communication workers (Claude Sonnet 4.5)
- Policy agent fans out to 3 retrievers (returns, shipping, warranty Knowledge Bases on S3 Vectors) in parallel with `ThreadPoolExecutor`
- Bedrock Guardrail (numbered version 1) with content, PII, denied-topic and profanity policies
- Deployed to AgentCore Runtime with the AgentCore CLI, with AgentCore Memory (session summary, 7-day retention)
- CloudWatch logging at INFO level and X-Ray tracing at 100% sampling

### Screenshots

**Test suite: 120/120**

![Test suite 120/120](screenshots/test-all-120.png)

**X-Ray Service Map**

![X-Ray Service Map](screenshots/xray-service-map.png)

**X-Ray node list**

![X-Ray node list](screenshots/xray-service-map-list.png)

---

<img width="1164" height="803" alt="Request flow scenarios" src="diagrams/request-flow-scenarios.png" />

## Prerequisites

- Python 3.12 (matches the deployed runtime)
- AWS account with Bedrock access enabled (`us-east-1` region)
- IAM permissions for: Bedrock, Bedrock AgentCore, DynamoDB, S3, S3 Vectors, CloudFormation, CloudWatch, X-Ray (the AgentCore CLI deploys through AWS CDK, so `cloudformation:*`, `iam:PassRole` and the CDK bootstrap roles are needed too)
- Bedrock Knowledge Bases created manually in AWS Console (Task 5)
- **AgentCore CLI** (`agentcore`, npm package [`@aws/agentcore`](https://github.com/aws/agentcore-cli)) - the deploy command uses it to package and deploy the runtime (Task 3). It needs **Node.js 20+** and [**uv**](https://docs.astral.sh/uv/getting-started/installation/):

  ```bash
  npm install -g @aws/agentcore@0.30.0
  agentcore --version
  ```

  > The CLI replaces the older Python `bedrock-agentcore-starter-toolkit`. Both install an `agentcore` command, so uninstall the old one if you have it: `pip uninstall bedrock-agentcore-starter-toolkit`.
- The AgentCore Runtime runs on arm64 and Python 3.12; `agentcore deploy` downloads matching wheels for you with `uv` (needs internet access)

---

## Setup

All TODOs in `src/agent_orchestrator.py` are implemented. The CLI configuration and support modules are provided infrastructure. Runtime deployment and runtime configuration updates use the AgentCore CLI; application service calls still use the AWS SDK.

Use the tested CLI version **0.30.0** shown above. The supplied CDK compatibility fix keeps HTTP explicit in the deployed runtime metadata.

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy .env template
cp .env.example .env

# Verify AWS resources
python config.py
```

Expected output: Configuration table showing all resource names. Fields showing `(not yet created)` are expected - they are populated as each task is completed.

### 3. Verify the AgentCore CLI

```bash
agentcore --version      # 0.30.0 or newer
agentcore validate       # checks agentcore/agentcore.json (the runtime definition)
```

### 4. Seed Initial Data

Run once after the CloudFormation stack reaches `CREATE_COMPLETE`:

```bash
python infrastructure/seed_data.py
```

This populates:
- DynamoDB tables with four mock customers (`CUST-001` … `CUST-004`) and their orders. The data is deterministic - the order IDs used by `demo.py`, `test` mode and the `chat` welcome table (e.g. `ORD-27176`) always exist
- S3 `policy-docs` bucket with sample policy documents under `policies/returns/`, `policies/shipping/`, `policies/warranty/`

---

## Project Structure

```
novamart-agentcore/
│
├── config.py                              # Central configuration (reads CloudFormation exports + env vars)
├── requirements.txt                       # Python dependencies
├── .env                                   # Populated environment values (KB IDs, Runtime ARN, Guardrail)
├── .env.example                           # Environment variable template
├── README.md                              # This file
│
├── agentcore/                             # AgentCore CLI project (used by `agentcore deploy`)
│   ├── agentcore.json                     # Runtime definition: entry point, network mode, protocol, env vars
│   ├── aws-targets.json                   # Deployment target (filled in by the CLI on first deploy)
│   └── cdk/                               # CDK app the CLI deploys with (managed by the CLI)
│
├── src/                                   # Implementation files
│   ├── agent_orchestrator.py             # Multi-agent orchestration (Tasks 2, 3, 4, 6) ⭐
│   ├── agent_utils.py                    # Pre-written: terminal trace UI utilities
│   ├── agent_observability.py            # Pre-written: X-Ray tracing + CloudWatch logging layer
│   ├── agentcore_cli.py                  # Pre-written: AgentCore CLI wrapper (stage code, agentcore deploy)
│   ├── bedrock_kb_retrieval.py           # Pre-written: KB retrieval helper
│   └── demo.py                           # Pre-written: demo script
│
├── build/runtime/                         # Created by deploy: the code the CLI packages (gitignored)
│
├── infrastructure/
│   ├── starter_stack.yaml                 # CloudFormation: foundation infra (DynamoDB, S3, S3 Vectors, IAM, CloudWatch)
│   ├── seed_data.py                       # Data seeding script
│   └── cleanup.py                         # Deletes everything the project created (run when done)
│
├── screenshots/                           # Test result and X-Ray Service Map screenshots
│   ├── test-all-120.png
│   ├── xray-service-map.png
│   └── xray-service-map-list.png
│
└── tests/
    └── test_agent.py                      # Automated test suite
```

### Provided files (not modified)
- `config.py` - Central configuration
- `tests/test_agent.py` - Automated tests
- `infrastructure/` - Pre-deployed resources
- `src/agent_utils.py` - Terminal trace UI utilities
- `src/agent_observability.py` - X-Ray tracing and CloudWatch logging
- `src/agentcore_cli.py` - AgentCore CLI wrapper
- `agentcore/` - AgentCore CLI project (the deploy command writes the runtime settings into `agentcore.json` for you)
- `src/bedrock_kb_retrieval.py` - KB retrieval helper
- `src/demo.py` - Demo script

---

## Tasks & Implementation

### Task 2 - Multi-Agent Graph
**File:** `src/agent_orchestrator.py`

5 Strands Agents in an Orchestrator → Workers hierarchy:

#### **2.A - InventoryAgent**
Gathers order and customer facts from DynamoDB (no decisions).
- **Tools:** `check_order_status`, `get_customer_tier`, `list_customer_orders`
- **Model:** Claude Sonnet 4.5 (`config.WORKER_MODEL_ID`) | **Temperature:** 0.1
- The Orders table has a composite key (`customer_id` + `order_id`), so `check_order_status` takes both

#### **2.B - RefundAgent**
Makes return/refund eligibility decisions based on customer tier and dates.
- **Tools:** `get_inventory_context`, `initiate_refund`
- **Model:** Claude Sonnet 4.5 (`config.WORKER_MODEL_ID`) | **Temperature:** 0.1

#### **2.C - PolicyAgent** ⭐
**Multi-Agent RAG Coordinator:** Runs 3 retriever sub-agents in parallel.
```
PolicyAgent
    ├─ ReturnsPolicyRetriever  → config.RETURNS_KB_ID
    ├─ ShippingPolicyRetriever → config.SHIPPING_KB_ID
    └─ WarrantyPolicyRetriever → config.WARRANTY_KB_ID
```
- **Tool:** `search_all_policies` - fans out to all 3 retrievers via ThreadPoolExecutor
- **Output:** Combined results from all three Knowledge Bases
- **Model:** Claude Sonnet 4.5 (`config.WORKER_MODEL_ID`) | **Temperature:** 0.2 (retrievers: 0.0)

#### **2.D - CommunicationAgent**
Composes final customer-facing response from accumulated WorkflowState.
- **Tools:** `get_full_workflow_context`
- **Model:** Claude Sonnet 4.5 (`config.WORKER_MODEL_ID`) | **Temperature:** 0.3

#### **2.E - OrchestratorAgent**
Routes requests and manages shared WorkflowState.

- **Tools:** `initialize_session`, `route_to_inventory_agent`, `route_to_policy_agent`, `route_to_refund_agent`, `route_to_communication_agent`
- **Model:** Claude Haiku 4.5 (`config.ORCHESTRATOR_MODEL_ID` - fast, cost-efficient) | **Temperature:** 0.0 (deterministic)

The orchestrator system prompt enforces these routing rules:

| Rule | Trigger | Action |
|------|---------|--------|
| 1 | Every request, always | Call `initialize_session` first |
| 2 | Order status / return / refund requests | Call `route_to_inventory_agent` → then `route_to_refund_agent` |
| 3 | Policy meaning questions (windows, rates, terms) | Call `route_to_policy_agent` |
| 4 | Account questions ("what is my tier?", "am I premium?") | Call `route_to_inventory_agent` - **never** `route_to_policy_agent` (it only knows policy text, not customer data) |
| 5 | Math / calculation questions | Calculate without Inventory, Policy or Refund; call CommunicationAgent last |
| 6 | Every request, always (last step) | Call `route_to_communication_agent` to compose the final reply |

> **CRITICAL:** The Orchestrator never writes the final customer-facing response itself. It always delegates to `route_to_communication_agent` as its very last action.

**Test:**
```bash
python tests/test_agent.py task2
```

---

### Task 3 - AgentCore Runtime & Guardrails
**File:** `src/agent_orchestrator.py`

- **`create_guardrail()`** - Creates a Bedrock Guardrail with:
  - Content filtering: SEXUAL, VIOLENCE, HATE (HIGH strength) + INSULTS, MISCONDUCT (MEDIUM strength)
  - PII handling (emails, phones anonymized; credit cards and SSNs blocked)
  - Topic blocking (competitor products, pricing negotiations, legal threats), using `topicPolicyConfig.tierConfig = {"tierName": "STANDARD"}` and top-level `crossRegionConfig = {"guardrailProfileIdentifier": "us.guardrail.v1:0"}`. Pricing negotiations are defined narrowly as haggling or changing an advertised price, so arithmetic with an already-specified price and discount is permitted. See [AWS safeguard tiers](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-tiers.html).
  - Profanity filtering (managed word list)
  - `create_guardrail_version()` so the returned version is a number, not `DRAFT`

- **`deploy_to_agentcore_runtime()`** - Deploys the system to AgentCore Runtime with the guardrail attached, using the **AgentCore CLI** (wrapped by `src/agentcore_cli.py`):
  1. Builds the runtime environment variables dict (region, project name, KB IDs, log group, `GUARDRAIL_ID`, `GUARDRAIL_VERSION`)
  2. `agentcore_cli.configure_runtime(env_vars=..., network_mode='PUBLIC', protocol='HTTP', execution_role_arn=config.AGENTCORE_ROLE_ARN)` writes those settings into `agentcore/agentcore.json`
  3. `agentcore_cli.deploy()` runs `agentcore deploy -y`
  4. `agentcore_cli.deployed_runtime_arn()` returns the ARN the CLI recorded
  - **How the guardrail is attached:** guardrails are enforced per model call, so `_apply_guardrail()` sets `guardrail_id` / `guardrail_version` on every agent's `BedrockModel` whenever `GUARDRAIL_ID` and `GUARDRAIL_VERSION` are known - from `.env` locally, and from the runtime environment variables when deployed.

**Deploy:**
```bash
python src/agent_orchestrator.py deploy
```

The first deployment also bootstraps AWS CDK in the account (a `CDKToolkit` stack) - allow 5-10 minutes. The command prints the Runtime ARN and Guardrail ID/Version, which are copied into `.env`:

```
AGENTCORE_RUNTIME_ARN=<printed arn>
GUARDRAIL_ID=<printed id>
GUARDRAIL_VERSION=<printed version>
```

**Test:**
```bash
python tests/test_agent.py task3
```

---

### Task 4 - Memory
**File:** `src/agent_orchestrator.py`

- **`configure_memory(runtime_arn)`** - Creates an AgentCore Memory resource with `agentcore_control.create_memory()`:
  - Strategy: SESSION_SUMMARY (`summaryMemoryStrategy`)
  - Storage: `eventExpiryDuration=7` days
  - Descriptive name and description, returns the memory ARN

**Test:**
```bash
python tests/test_agent.py task4
```

---

### Task 5 - Bedrock Knowledge Bases (done in AWS Console)

Three self-managed Knowledge Bases (not Managed KBs) that share the same S3 data bucket and S3 Vectors bucket. Only the prefix and vector index differ:

| Setting | Returns KB | Shipping KB | Warranty KB |
|---|---|---|---|
| Name | `novamart-returns-policy-kb` | `novamart-shipping-policy-kb` | `novamart-warranty-policy-kb` |
| S3 prefix | `policies/returns/` | `policies/shipping/` | `policies/warranty/` |
| Embedding model | `amazon.titan-embed-text-v2:0` | same | same |
| Vector store | S3 Vectors → existing vector bucket | same | same |
| Vector index | `returns-policy-index` | `shipping-policy-index` | `warranty-policy-index` |

Each data source was synced after creation, and the IDs are stored in `.env`:
```
RETURNS_KB_ID=
SHIPPING_KB_ID=
WARRANTY_KB_ID=
```

**Test:**
```bash
python tests/test_agent.py task5
```

---

### Task 6 - Observability
**File:** `src/agent_orchestrator.py`

- **`configure_observability(runtime_arn)`** - Builds a `loggingConfiguration` and passes it to `apply_observability_config()` inside `try/except`:
  - **CloudWatch:** `cloudWatchConfig` - INFO-level logs to `config.AGENT_LOG_GROUP` (`/aws/bedrock/agentcore/udacity-agentcore`), `enabled=True`
  - **X-Ray:** `xRayConfig` - `enabled=True`, `samplingRate=1.0`

**How traces reach X-Ray:** `agent_observability.py` records every request as one X-Ray trace. Each `route_to_*` tool becomes a worker-agent node and each Knowledge Base retrieval a `KnowledgeBase:*` node.

**Verify end-to-end tracing:**
```bash
python tests/test_agent.py task6

# Live test - each scenario prints its X-Ray trace id
python src/agent_orchestrator.py test

# View the map in AWS Console → CloudWatch → X-Ray traces → Service map
```

---

### Deployment Pipeline

`python src/agent_orchestrator.py deploy` executes a 6-step pipeline:

| Step | What it does |
|------|-------------|
| 1/6 | Build the 5-agent graph locally |
| 2/6 | Create the Bedrock Guardrail |
| 3/6 | Stage the code in `build/runtime/`, write the runtime settings to `agentcore/agentcore.json` and deploy with `agentcore deploy -y` |
| 4/6 | Create AgentCore Memory |
| 5/6 | Configure Observability (CloudWatch + X-Ray Transaction Search) |
| 6/6 | Deploy AgentCore Gateway *(optional, skipped if Lambda tool functions are not deployed)* |

---

### Task 7 - End-to-End Test Scenarios

| Scenario | Expected Routing |
|----------|-----------------|
| `"I want to return my order ORD-27176"` (CUST-001) | Orchestrator → Inventory → Refund → Communication |
| `"What is the return policy for premium customers?"` | Orchestrator → Policy (3 parallel KB retrievers) → Communication |
| `"How much are 5 items at $29.99 with 10% off?"` | Orchestrator → Communication (no Inventory, Policy or Refund) |

All three scenarios ran successfully. The results are shown in the screenshots above: **120/120** on `python tests/test_agent.py all`, and the X-Ray Service Map showing `NovaMart-Orchestrator` connected to InventoryAgent, RefundAgent, PolicyAgent, CommunicationAgent and the three `KnowledgeBase:*` nodes.

---

## Testing

### Automated Tests

```bash
python tests/test_agent.py task2   # Multi-agent orchestration
python tests/test_agent.py task3   # AgentCore deployment
python tests/test_agent.py task4   # Memory
python tests/test_agent.py task5   # Knowledge base retrieval
python tests/test_agent.py task6   # Observability
python tests/test_agent.py all     # Full suite (120/120)
```

### Live Testing

```bash
python src/agent_orchestrator.py test     # 3 end-to-end scenarios, traced to X-Ray
python src/agent_orchestrator.py chat     # interactive terminal chat with step-by-step trace
python src/agent_orchestrator.py invoke "What is the return policy for premium customers?" CUST-002   # deployed runtime

# or with the AgentCore CLI
agentcore invoke "What is the return policy for premium customers?"
agentcore status
agentcore logs
```

---

## Infrastructure

The foundation infrastructure is defined in `infrastructure/starter_stack.yaml` and provisions DynamoDB tables, the S3 policy-documents bucket, an S3 Vectors bucket with three vector indexes, the AgentCore execution role, and the CloudWatch log group.

Deploy it once (stack name `udacity-agentcore`, region `us-east-1`, with `CAPABILITY_NAMED_IAM`), then run `python infrastructure/seed_data.py`.

---

## Environment Variables

The `.env` file holds values populated as each task is completed:

```bash
# AWS Settings
AWS_REGION=us-east-1
PROJECT_NAME=udacity-agentcore

# Bedrock Knowledge Base IDs (Task 5)
RETURNS_KB_ID=
SHIPPING_KB_ID=
WARRANTY_KB_ID=

# AgentCore Runtime (Task 3)
AGENTCORE_RUNTIME_ARN=

# Guardrail (Task 3) - numbered version, not DRAFT
GUARDRAIL_ID=
GUARDRAIL_VERSION=
```

No AWS access keys or session tokens are stored in this file.

---

## Troubleshooting

**Config loading errors?** Run `python config.py` to see what is and isn't loaded.

| Variable | Where it comes from | What to do if missing |
|---|---|---|
| DynamoDB tables, S3 + S3 Vectors buckets, IAM role, log group | CloudFormation exports only | Verify stack `udacity-agentcore` is `CREATE_COMPLETE` in `us-east-1` |
| `RETURNS_KB_ID`, `SHIPPING_KB_ID`, `WARRANTY_KB_ID` | `.env` | Create and sync the KBs, then paste the IDs |
| `AGENTCORE_RUNTIME_ARN` | `.env` | Run `python src/agent_orchestrator.py deploy` and paste the ARN |
| `GUARDRAIL_ID`, `GUARDRAIL_VERSION` | `.env` | The deploy command prints these values |

**X-Ray traces not appearing?** Select a short time range ("Last 5 minutes") and allow up to 60 seconds. If a run says the trace was *not published*, check `xray:PutTraceSegments` permission and that `AGENT_TRACING_ENABLED` is not `false`.

**`agentcore: command not found`?** Install Node.js 20+ and run `npm install -g @aws/agentcore@0.30.0`. If `agentcore` prints toolkit-style help, remove the old toolkit: `pip uninstall bedrock-agentcore-starter-toolkit`.

**`agentcore deploy` fails while packaging?** The CLI needs [`uv`](https://docs.astral.sh/uv/getting-started/installation/) on your PATH and internet access. Check `agentcore/.cli/logs/`.

**`agentcore deploy` fails with a CDK / CloudFormation error?** Check the stack `AgentCore-udacity-default` in the CloudFormation console for the failing resource, fix it, and re-run the deploy.

---

## Clean Up

After the project has been submitted:

```bash
python infrastructure/cleanup.py          # dry run - lists what would be deleted
python infrastructure/cleanup.py --yes    # deletes it
```

---

## Documentation & Resources

- [Strands Agents SDK](https://github.com/strands-agents/sdk-python)
- [Amazon Bedrock AgentCore](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/)
- [AgentCore CLI](https://github.com/aws/agentcore-cli)
- [Bedrock Knowledge Bases](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html)
- [Bedrock Guardrails](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-create.html)
- [DynamoDB Optimistic Locking](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/transaction-apis.html)

---

*Submitted by Devanshi Malhotra for the Udacity Future AWS Agent Engineer Nanodegree.*