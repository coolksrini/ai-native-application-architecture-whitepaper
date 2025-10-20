# Chapter 13: Framework Evolution

## Introduction

Modern web frameworks (FastAPI, Spring Boot, Express, Django) were built for REST APIs and traditional web apps. To support AI-native development, they need to evolve. This chapter outlines what frameworks must provide to make MCP-first development as easy as REST development today.

**Goal:** Make AI-native as natural as traditional web development.

---

## What Frameworks Need to Add

### The Vision

```python
# Today: REST is first-class
@app.get("/api/orders/{id}")
async def get_order(id: str):
    return get_order_from_db(id)

# Tomorrow: MCP is equally first-class
@app.mcp_tool()
async def get_order(order_id: str) -> Order:
    """Retrieve order by ID"""
    return get_order_from_db(order_id)

# Framework handles:
# - Tool schema generation
# - Type validation
# - Security
# - Observability
# - Training data export
# - Evaluation helpers
```

---

## 1. Native MCP Protocol Support

### Current State (Manual)

```python
# Today: Developers implement MCP manually
from fastapi import FastAPI
import json

app = FastAPI()

@app.post("/mcp/execute")
async def execute_tool(request: dict):
    tool_name = request["tool"]
    parameters = request["parameters"]
    
    # Manual routing
    if tool_name == "get_order":
        return get_order(**parameters)
    elif tool_name == "create_payment":
        return create_payment(**parameters)
    # ... manual handling for every tool

@app.get("/mcp/schema")
async def get_schema():
    # Manually maintain schema
    return {
        "tools": [
            {
                "name": "get_order",
                "description": "...",
                "parameters": {...}
            }
        ]
    }
```

### Desired State (Framework Native)

```python
# Tomorrow: Framework handles MCP protocol
from fastapi import FastAPI
from fastapi.mcp import MCPRouter

app = FastAPI()
mcp = MCPRouter()

@mcp.tool()
async def get_order(order_id: str) -> Order:
    """Retrieve order details by ID"""
    return db.get_order(order_id)

@mcp.tool()
async def create_payment(amount: int, customer_id: str) -> Payment:
    """Process a payment for a customer"""
    return payment_processor.charge(amount, customer_id)

# Framework automatically:
# - Generates tool schemas from type hints
# - Handles MCP protocol
# - Provides /mcp/schema endpoint
# - Provides /mcp/execute endpoint
# - Validates parameters
# - Handles errors in MCP format

app.include_router(mcp, prefix="/mcp")
```

---

## 2. Automatic Schema Generation

### From Type Hints to Tool Schema

```python
from typing import Optional, List
from pydantic import BaseModel
from enum import Enum

class PaymentStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"

class Payment(BaseModel):
    id: str
    amount: int
    status: PaymentStatus
    created_at: datetime

@mcp.tool(
    examples=[
        {
            "query": "Get payment ABC123",
            "params": {"payment_id": "ABC123"}
        }
    ]
)
async def get_payment(
    payment_id: str,
    include_history: bool = False
) -> Payment:
    """
    Retrieve payment details.
    
    Args:
        payment_id: The unique payment identifier
        include_history: Whether to include transaction history
    
    Returns:
        Payment object with all details
    """
    pass

# Framework generates:
{
  "name": "get_payment",
  "description": "Retrieve payment details.",
  "input_schema": {
    "type": "object",
    "properties": {
      "payment_id": {
        "type": "string",
        "description": "The unique payment identifier"
      },
      "include_history": {
        "type": "boolean",
        "description": "Whether to include transaction history",
        "default": false
      }
    },
    "required": ["payment_id"]
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "id": {"type": "string"},
      "amount": {"type": "integer"},
      "status": {
        "type": "string",
        "enum": ["pending", "completed", "failed"]
      },
      "created_at": {"type": "string", "format": "date-time"}
    }
  }
}
```

---

## 3. Built-In Security Layers

### Multi-Layer Authorization

```python
from fastapi.mcp import MCPSecurity, UserContext, AgentContext

@mcp.tool(
    security=MCPSecurity(
        # Layer 1: User authorization
        require_user_scope=["payments:read"],
        
        # Layer 2: Agent authorization
        require_agent_scope=["access:financial_data"],
        
        # Layer 3: Risk-based controls
        risk_level="medium",
        require_human_approval=lambda ctx, params: params["amount"] > 10000,
        
        # Rate limiting
        rate_limit="100/hour",
        
        # Audit logging
        audit_log=True
    )
)
async def create_payment(
    amount: int,
    customer_id: str,
    user_context: UserContext = Depends(),
    agent_context: AgentContext = Depends()
) -> Payment:
    """Create a payment (amount in cents)"""
    
    # Framework has already validated:
    # - User has payments:write scope
    # - Agent is authorized for financial data
    # - Rate limits not exceeded
    # - Audit log entry created
    
    # Just implement business logic
    return payment_processor.charge(amount, customer_id)
```

### Automatic Audit Logging

```python
# Framework automatically logs:
audit_log.record({
    "timestamp": "2025-01-15T10:30:00Z",
    "user_id": "user_123",
    "agent_id": "claude_sonnet_4.5",
    "tool": "create_payment",
    "parameters": {"amount": 5000, "customer_id": "cust_456"},
    "user_authorized": True,
    "agent_authorized": True,
    "intent_validated": True,
    "risk_level": "medium",
    "result": "success"
})
```

---

## 4. Observability Integration

### Built-In Metrics

```python
@mcp.tool(
    observability={
        "track_latency": True,
        "track_errors": True,
        "track_intent_accuracy": True,
        "custom_metrics": ["payment_amount", "payment_status"]
    }
)
async def create_payment(amount: int, customer_id: str) -> Payment:
    """Create payment"""
    
    # Framework automatically tracks:
    # - Tool call count
    # - Latency (p50, p95, p99)
    # - Error rate
    # - Success rate
    # - Custom metrics
    
    result = payment_processor.charge(amount, customer_id)
    
    # Emit custom metric
    metrics.track("payment_amount", amount)
    metrics.track("payment_status", result.status)
    
    return result
```

### Dashboard Generation

```python
# Framework provides automatic dashboard
app.mount_dashboard("/metrics/mcp")

# Accessible at: http://localhost:8000/metrics/mcp
# Shows:
# - Tool call rates
# - Success rates per tool
# - Latency distributions
# - Error breakdowns
# - Intent recognition accuracy
# - Cost per tool call
```

---

## 5. Training Data Export

### Automatic Training Data Generation

**Framework should implement the decentralized pattern from Chapter 11:**

```python
@mcp.tool(
    training={
        "export_examples": True,
        "examples": [
            {
                "query": "Charge customer $50",
                "explanation": "Amount always in cents"
            }
        ],
        "edge_cases": [
            {
                "scenario": "Ambiguous amount",
                "handling": "Clarify with user"
            }
        ],
        "collect_production_data": True,  # Auto-collect successful interactions
        "quality_filter": lambda ex: ex.user_satisfaction >= 4.0
    }
)
async def create_payment(amount: int, currency: str = "USD"):
    """Create a payment (amount in cents)"""
    pass

# Framework provides standardized endpoints (Chapter 11 pattern):
GET /mcp/training-dataset
GET /mcp/test-dataset

# Returns training data in format compatible with:
# - Prompt engineering (few-shot examples)
# - Cloud fine-tuning (OpenAI format)
# - Local LoRA (HuggingFace format)
```

**Framework-generated `/mcp/training-dataset` endpoint:**

```python
# Automatically generated by framework
@app.get("/mcp/training-dataset")
async def get_training_dataset(
    auth: ServiceAuth = Depends(verify_ml_orchestrator),
    format: str = "anthropic_messages"  # or "openai", "huggingface"
):
    """
    Framework auto-generates this endpoint for every service.
    Aggregates training data from all @mcp.tool decorators.
    """

    if not auth.has_role("ml_orchestrator"):
        raise HTTPException(403, "Unauthorized")

    # Collect from all decorated tools
    training_examples = []

    for tool in app.mcp_tools:
        # Developer-provided examples
        training_examples.extend(tool.training_config.examples)

        # Auto-collected production examples (if enabled)
        if tool.training_config.collect_production_data:
            production_examples = await collect_production_examples(
                tool_name=tool.name,
                quality_filter=tool.training_config.quality_filter
            )
            training_examples.extend(production_examples)

    # Format for requested provider
    formatted = format_training_data(training_examples, format)

    return {
        "service": app.title,
        "version": app.version,
        "format": format,
        "training_examples": formatted,
        "metadata": {
            "total_examples": len(formatted),
            "tools_covered": len(app.mcp_tools),
            "last_updated": datetime.now().isoformat()
        }
    }

@app.get("/mcp/test-dataset")
async def get_test_dataset(
    auth: ServiceAuth = Depends(verify_ml_orchestrator)
):
    """Framework-generated test scenarios for evaluation"""

    if not auth.has_role("ml_orchestrator"):
        raise HTTPException(403, "Unauthorized")

    # Collect critical test scenarios from all tools
    test_scenarios = []

    for tool in app.mcp_tools:
        if tool.testing_config:
            test_scenarios.extend(tool.testing_config.scenarios)

    return {
        "service": app.title,
        "test_scenarios": test_scenarios,
        "total_scenarios": len(test_scenarios),
        "min_accuracy_required": 0.95
    }
```

**Benefits of framework-generated endpoints:**
- ✅ **Zero configuration** - Automatically available for every service
- ✅ **Standardized format** - Works with centralized ML orchestrator
- ✅ **Multiple export formats** - Prompt engineering, OpenAI, HuggingFace
- ✅ **Production data collection** - Opt-in automatic collection
- ✅ **Decentralized ownership** - Each service owns its training data
- ✅ **Weekly retraining** - ML orchestrator can auto-aggregate

---

## 6. Context Management Primitives

### Built-In Conversation Context (Chapter 8 Pattern)

**Frameworks should provide context management as a first-class feature:**

```python
from fastapi.mcp import ConversationContext, ContextCompression

@mcp.tool(
    context={
        "max_tokens": 8000,  # Auto-compress if exceeded
        "relevance_scoring": True,  # Keep only relevant context
        "compression_strategy": "smart_summarize"
    }
)
async def get_order_status(
    order_id: str,
    context: ConversationContext = Depends()
) -> OrderStatus:
    """
    Get order status with full conversation context.
    Framework handles context size management automatically.
    """

    # Access conversation history
    if context.mentioned_recently("shipping address"):
        # User already discussed shipping
        include_shipping_details = True
    else:
        include_shipping_details = False

    # Check user's previous queries
    if context.previous_query_failed():
        # Add extra context for recovery
        add_troubleshooting_info = True

    order = get_order(order_id)

    return format_order_status(
        order,
        include_shipping=include_shipping_details,
        include_troubleshooting=add_troubleshooting_info
    )
```

**Framework-provided ConversationContext:**

```python
class ConversationContext:
    """Framework-provided context manager"""

    def __init__(self):
        # Automatic context tracking
        self.conversation_id: str
        self.user_id: str
        self.session_id: str

        # Message history
        self.messages: List[Message]  # Full history
        self.compressed_messages: List[Message]  # Auto-compressed

        # Metadata
        self.total_tokens: int  # Tracked automatically
        self.tools_called: List[str]  # History of tool calls
        self.failed_attempts: int  # For error recovery

    def mentioned_recently(
        self,
        topic: str,
        lookback_messages: int = 5
    ) -> bool:
        """Check if topic was mentioned in recent messages"""
        recent = self.messages[-lookback_messages:]
        return any(topic.lower() in msg.content.lower() for msg in recent)

    def previous_query_failed(self) -> bool:
        """Check if previous query had errors"""
        if len(self.messages) < 2:
            return False
        return self.messages[-2].error_occurred

    def get_relevant_context(
        self,
        current_query: str,
        max_tokens: int = 4000
    ) -> List[Message]:
        """
        Framework automatically extracts relevant context.
        Uses semantic similarity to current query.
        """
        # Framework handles relevance scoring
        scored = self.score_relevance(self.messages, current_query)

        # Return highest-scoring messages within token budget
        return self.select_within_budget(scored, max_tokens)

    def auto_compress(self, max_tokens: int = 8000):
        """
        Automatic compression when context grows large.
        Framework uses smart summarization.
        """
        if self.total_tokens > max_tokens:
            # Compress older messages
            self.compressed_messages = compress_messages(
                self.messages,
                target_tokens=max_tokens * 0.7,
                preserve_recent_count=3  # Never compress last 3 messages
            )
```

**Automatic context compression:**

```python
@mcp.tool(
    context={
        "max_tokens": 8000,
        "auto_compress": True,  # Framework handles automatically
        "compression_strategy": "smart_summarize",
        "preserve_tool_calls": True  # Never compress tool call history
    }
)
async def complex_workflow(
    context: ConversationContext = Depends()
):
    """
    Framework automatically manages context:
    - Compresses when >8000 tokens
    - Preserves recent messages
    - Keeps all tool calls
    - Maintains semantic relevance
    """

    # Developer doesn't worry about context size
    # Framework handles it automatically

    return process_with_context(context)
```

**Context storage backends:**

```python
# Framework supports multiple backends
app = FastAPI(
    mcp_config={
        "context_storage": "redis",  # or "dynamodb", "postgres"
        "context_ttl": 3600,  # 1 hour
        "max_context_tokens": 8000
    }
)

# Automatic context persistence
@mcp.tool()
async def my_tool(context: ConversationContext = Depends()):
    # Framework auto-saves context after each tool call
    # Loads context at start of next tool call
    # Developer doesn't manage storage
    pass
```

**Benefits of framework-provided context:**
- ✅ **Automatic compression** - No manual token counting
- ✅ **Relevance scoring** - Keep important context, drop irrelevant
- ✅ **Storage abstraction** - Pluggable backends (Redis, DynamoDB, etc.)
- ✅ **Error recovery** - Track failed attempts for better handling
- ✅ **Tool history** - Automatic tracking of what tools were called
- ✅ **Session management** - Handle multi-turn conversations

---

## 7. Evaluation Helpers

### Test Generation

```python
@mcp.tool(
    testing={
        "generate_tests": True,
        "accuracy_threshold": 0.95
    }
)
async def get_order(order_id: str) -> Order:
    """Get order details"""
    pass

# Framework generates:
@pytest.mark.llm_test
async def test_get_order_intent():
    test_cases = [
        "Show me order 123",
        "Get order details for 123",
        "What's the status of order 123"
    ]
    
    correct = 0
    for query in test_cases:
        result = await ai_agent.process(query)
        if result.tool == "get_order" and \
           result.params["order_id"] == "123":
            correct += 1
    
    accuracy = correct / len(test_cases)
    assert accuracy >= 0.95
```

---

## Framework Comparison

### FastAPI (Python)

**Current Capabilities:**
- ✓ Excellent type system
- ✓ Automatic OpenAPI generation
- ✓ Dependency injection
- ✓ Good performance

**Needs to Add:**
- ❌ Native MCP support
- ❌ MCP schema generation
- ❌ Triple-layer auth for AI
- ❌ LLM observability
- ❌ Training data export

**Proposed API:**

```python
from fastapi import FastAPI
from fastapi.mcp import MCP, MCPSecurity, MCPTraining

app = FastAPI()
mcp = MCP()

@mcp.tool(
    security=MCPSecurity(...),
    training=MCPTraining(...),
    observability={...}
)
async def my_tool():
    pass

app.include_router(mcp)
```

### Spring Boot (Java)

**Current Capabilities:**
- ✓ Enterprise-grade features
- ✓ Comprehensive ecosystem
- ✓ Strong security
- ✓ Production-proven

**Needs to Add:**
- ❌ MCP protocol support
- ❌ AI-specific annotations
- ❌ LLM observability

**Proposed API:**

```java
@RestController
@MCPController
public class PaymentController {
    
    @MCPTool(
        description = "Create a payment",
        security = @MCPSecurity(
            userScopes = {"payments:write"},
            agentScopes = {"financial:access"}
        ),
        training = @MCPTraining(
            exportExamples = true
        )
    )
    public Payment createPayment(
        @MCPParam("amount") Integer amount,
        @MCPParam("customer_id") String customerId
    ) {
        return paymentService.process(amount, customerId);
    }
}
```

### Express (Node.js)

**Current Capabilities:**
- ✓ Lightweight
- ✓ Flexible middleware
- ✓ Large ecosystem

**Needs to Add:**
- ❌ Type-safe MCP support
- ❌ Schema generation
- ❌ Built-in security layers

**Proposed API:**

```typescript
import express from 'express';
import { MCPRouter, MCPSecurity } from 'express-mcp';

const app = express();
const mcp = new MCPRouter();

mcp.tool({
  name: 'create_payment',
  description: 'Create a payment',
  security: MCPSecurity({
    userScopes: ['payments:write'],
    agentScopes: ['financial:access']
  }),
  handler: async (params: {
    amount: number;
    customerId: string;
  }) => {
    return paymentService.process(params.amount, params.customerId);
  },
  schema: {
    amount: { type: 'number', description: 'Amount in cents' },
    customerId: { type: 'string' }
  }
});

app.use('/mcp', mcp.router());
```

---

## Developer Experience Goals

### Make MCP as Easy as REST

**Today (REST):**
```python
# 1. Define endpoint
@app.get("/api/orders/{id}")
async def get_order(id: str):
    return db.get_order(id)

# 2. Test
response = client.get("/api/orders/123")
assert response.status_code == 200

# 3. Document (automatic via OpenAPI)
# 4. Deploy
```

**Tomorrow (MCP):**
```python
# 1. Define tool
@mcp.tool()
async def get_order(order_id: str) -> Order:
    """Get order details"""
    return db.get_order(order_id)

# 2. Test (framework-provided)
await test_tool_intent("get_order", [
    "Show me order 123",
    "Get order details for 123"
], accuracy_threshold=0.95)

# 3. Document (automatic via MCP schema)
# 4. Train (automatic training data export)
# 5. Evaluate (automatic test generation)
# 6. Deploy
```

**Key: Same simplicity, AI-native benefits.**

---

## Migration Path for Frameworks

### Phase 1: Experimental Support (Month 1-3)

```python
# Add as separate module
pip install fastapi-mcp  # Community package

from fastapi import FastAPI
from fastapi_mcp import MCPRouter

app = FastAPI()
mcp = MCPRouter()

@mcp.tool()
async def my_tool():
    pass

app.include_router(mcp, prefix="/mcp")
```

### Phase 2: Official Integration (Month 4-9)

```python
# Included in framework
from fastapi import FastAPI
from fastapi.mcp import MCP  # Now official

app = FastAPI()

@app.mcp_tool()  # First-class decorator
async def my_tool():
    pass
```

### Phase 3: Full Feature Parity (Month 10-18)

```python
# MCP equals REST in capabilities
from fastapi import FastAPI

app = FastAPI()

# Both equally supported
@app.get("/api/orders")  # REST
async def get_orders_rest():
    pass

@app.mcp_tool()  # MCP
async def get_orders():
    pass

# Same observability, security, testing
```

---

## Community Ecosystem

### What's Needed

**Core Framework Support:**
- FastAPI, Spring Boot, Express, Django, Rails
- Native MCP protocol handling
- Schema generation
- Security primitives

**Developer Tools:**
- MCP CLI for testing tools locally
- Schema validators
- Training data formatters
- Evaluation frameworks

**Observability:**
- Prometheus exporters for MCP metrics
- Grafana dashboards
- Datadog/New Relic integrations
- OpenTelemetry support

**Testing:**
- Pytest plugins for LLM tests
- Jest plugins for Node.js
- JUnit extensions for Java
- Test data generators

---

## Open Source Opportunities

### High-Impact Projects

**1. fastapi-mcp**
```
Native MCP support for FastAPI
- Tool decorator
- Schema generation
- Type validation
- Training data export

Status: Needed
Impact: High
Difficulty: Medium
```

**2. mcp-cli**
```
CLI tool for MCP development
- Test tools locally
- Validate schemas
- Generate training data
- Run evaluations

Status: Needed
Impact: High
Difficulty: Low
```

**3. mcp-observability**
```
Observability stack for MCP
- Prometheus metrics
- Grafana dashboards
- Intent tracking
- Cost monitoring

Status: Needed
Impact: Medium
Difficulty: Medium
```

**4. pytest-mcp**
```
Pytest plugin for LLM testing
- @pytest.mark.llm_test
- Statistical assertions
- Evaluation datasets
- Test data generation

Status: Needed
Impact: High
Difficulty: Low
```

---

## Call to Action for Framework Authors

### Recommendations

**For FastAPI Team:**
1. Add `fastapi.mcp` module
2. Leverage existing Pydantic for schemas
3. Integrate with existing security system
4. Provide migration guide from REST

**For Spring Boot Team:**
1. Create `@MCPController` annotation
2. Generate schemas from Java types
3. Integrate with Spring Security
4. Support both REST and MCP coexistence

**For Express Team:**
1. Create `express-mcp` middleware
2. TypeScript-first approach
3. Leverage existing middleware ecosystem
4. Provide TypeScript type definitions

**For Django Team:**
1. Add `django-mcp` package
2. Integrate with Django ORM models
3. Use existing authentication system
4. Generate schemas from Django models

---

## Summary: Framework Checklist

### Essential Features

```
✅ Native MCP Protocol Support
  └─ Handle MCP requests/responses

✅ Automatic Schema Generation
  └─ From type hints/annotations

✅ Security Primitives
  ├─ User authorization
  ├─ Agent authorization
  └─ Intent validation

✅ Observability Hooks
  ├─ Tool call metrics
  ├─ Intent tracking
  └─ Cost monitoring

✅ Training Data Export (Chapter 11 Pattern)
  ├─ Auto-generated /mcp/training-dataset endpoint
  ├─ Auto-generated /mcp/test-dataset endpoint
  ├─ Multiple export formats (prompt engineering, OpenAI, LoRA)
  ├─ Production data auto-collection
  └─ Decentralized ownership per service

✅ Context Management (Chapter 8 Pattern)
  ├─ ConversationContext dependency injection
  ├─ Automatic compression when >max tokens
  ├─ Relevance scoring & filtering
  ├─ Storage backend abstraction
  └─ Tool call history tracking

✅ Evaluation Helpers
  ├─ Test generation
  ├─ Accuracy measurement
  └─ Regression detection

✅ Development Tools
  ├─ Local testing
  ├─ Schema validation
  └─ Documentation generation
```

---

## Key Takeaways

✓ **Frameworks must evolve** - MCP support needs to be first-class, as natural as REST

✓ **Training data endpoints auto-generated** - Framework should create `/mcp/training-dataset` and `/mcp/test-dataset` endpoints automatically (Chapter 11 pattern); supports prompt engineering, cloud fine-tuning, and local LoRA

✓ **Context management as primitive** - Framework-provided ConversationContext with automatic compression, relevance scoring, and storage abstraction (Chapter 8 pattern)

✓ **Leverage existing patterns** - Build on REST foundations (routing, dependency injection, type validation)

✓ **Developer experience matters** - Make MCP as easy as REST; same simplicity, AI-native benefits

✓ **Complete security layers** - User auth + Agent auth + Intent validation + Risk-based controls

✓ **Built-in observability** - Auto-track tool calls, latency, costs, intent accuracy

✓ **Community opportunity** - Open source ecosystem needed (fastapi-mcp, mcp-cli, pytest-mcp)

✓ **Gradual adoption** - Start with experimental packages, move to official framework integration

✓ **Multiple languages** - Python, Java, Node.js, Ruby, Go all need support

✓ **Complete stack** - Framework, tools, observability, testing, training data, context management

---

**[← Previous: Chapter 12 - Migration](chapter-12-migration.md) | [Next: Chapter 14 - Case Studies →](chapter-14-case-studies.md)**