# Chapter 5: MCP-Enabled Microservices

## Introduction

This chapter resolves the central question: **Are MCP services different from microservices?**

**Answer: No.** An MCP service that handles a specific business domain, scales independently, has its own database, and provides failure isolation IS a microservice. The only difference is it speaks MCP instead of REST.

The distinction is artificial—it's like asking if "HTTP microservices" are different from "gRPC microservices." The protocol changes, not the architectural pattern.

**What is MCP?** The Model Context Protocol (MCP) is an open protocol that standardizes how applications provide context to LLMs.[24] Just as USB-C provides a standardized way to connect devices to peripherals, MCP provides a standardized way to connect AI models to different data sources and tools.[24] Anthropic released MCP with official SDKs in Python, TypeScript, C#, and Java.[25]

---

## The Artificial Distinction

### What People Think

```
"MCP Services" (New Thing)
vs.
"Microservices" (Old Thing)

Implication: These are different architectural approaches
```

### The Reality

```
Microservice = Small, autonomous service with single business responsibility

Communication Protocol = How services talk (REST, gRPC, MCP, etc.)

An MCP-enabled microservice is still a microservice.
Protocol ≠ Architecture
```

### Side-by-Side Comparison

#### Traditional Microservice (REST)

```python
# Payment microservice using REST
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.post("/api/payments")
async def create_payment(request: PaymentRequest):
    """REST endpoint for creating payments"""
    try:
        result = payment_processor.process(
            amount=request.amount,
            user_id=request.user_id
        )
        return result
    except InsufficientFunds:
        raise HTTPException(status_code=402)

# Characteristics:
# ✓ Single responsibility: Payment processing
# ✓ Own database: payment_db
# ✓ Independent scaling: Based on payment volume
# ✓ Failure isolation: Crashes don't affect other services
# ✓ Interface: REST API
```

#### MCP-Enabled Microservice

```python
# Payment microservice using MCP
from fastapi import FastAPI
from fastapi_mcp import MCPRouter

app = FastAPI()
mcp = MCPRouter()

@mcp.tool()
async def create_payment(
    amount: int,
    user_context: UserContext
) -> PaymentResult:
    """
    Process a payment for the current user.
    Amount must be in cents (e.g., $10 = 1000).
    """
    try:
        result = payment_processor.process(
            amount=amount,
            user_id=user_context.user_id
        )
        return result
    except InsufficientFunds:
        raise InsufficientFundsError("User has insufficient funds")

# Characteristics:
# ✓ Single responsibility: Payment processing (SAME)
# ✓ Own database: payment_db (SAME)
# ✓ Independent scaling: Based on payment volume (SAME)
# ✓ Failure isolation: Crashes don't affect other services (SAME)
# ✓ Interface: MCP tools (DIFFERENT)
```

**Everything except the interface is identical.**

---

## When to Split Services (AI-Native or Not)

The decision criteria for microservices remain unchanged:

### Decision Framework

```
Should I split Service X from the monolith?

YES if:
├─ Different scaling characteristics
│  Example: ML inference needs GPU, web API doesn't
│
├─ Different technology requirements
│  Example: Python for ML, Go for high-performance API
│
├─ Team boundaries
│  Example: Team A owns payments, Team B owns inventory
│
├─ Compliance/security isolation
│  Example: PCI-DSS for payment data
│
└─ Different deployment cadence
   Example: Frequently updated features vs. stable core

NO if:
├─ Same scaling needs
├─ Shared data model (would require distributed transactions)
├─ Small team (< 10 people)
└─ Early stage (requirements changing rapidly)
```

### Real-World Example: E-Commerce Platform

**Scenario:** You're building an e-commerce platform with AI-native interface.

#### Modular Monolith (Starting Point)

```
┌─────────────────────────────────────────────┐
│    E-Commerce Monolith (MCP-Enabled)        │
│                                             │
│  ┌──────────────┐  ┌──────────────┐        │
│  │   Products   │  │   Orders     │        │
│  │   Module     │  │   Module     │        │
│  └──────────────┘  └──────────────┘        │
│                                             │
│  ┌──────────────┐  ┌──────────────┐        │
│  │   Payments   │  │   Inventory  │        │
│  │   Module     │  │   Module     │        │
│  └──────────────┘  └──────────────┘        │
│                                             │
│  All exposed via MCP tools                  │
└─────────────────────────────────────────────┘
         │
    ┌────▼────┐
    │   DB    │
    └─────────┘
```

**Start here:**[26][27]
- One codebase
- Modular structure (internal boundaries)
- Single database (or schema per module)
- All MCP tools exposed from one service
- Easy to develop and deploy

**Why start with a modular monolith?** Martin Fowler's research shows "almost all successful microservice stories have started with monolith architectures that got too big and were broken up."[26] The key is designing the monolith carefully with attention to modularity, making it relatively simple to shift to microservices when needed.[26]

#### When to Extract: Payment Service

```
Problem observed:
- Payment processing is slow during checkout
- Needs more resources (CPU/memory)
- Rest of application is fast
- Scaling entire monolith for payments is wasteful

Solution: Extract payment service

┌────────────────────┐     ┌──────────────────┐
│  E-Commerce App    │     │ Payment Service  │
│  (Orders, Products,│─────│  (MCP)           │
│   Inventory)       │ MCP │                  │
└────────────────────┘     └──────────────────┘
         │                          │
    ┌────▼────┐              ┌─────▼─────┐
    │   DB    │              │ Payment DB│
    └─────────┘              └───────────┘

Benefits:
✓ Scale payment service independently (3x replicas)
✓ Main app stays at 1x replica (saves cost)
✓ Payment failures don't crash entire app
✓ Can upgrade payment processing without redeploying everything
```

#### When to Extract: ML Recommendation Service

```
Problem:
- Recommendation generation is CPU-intensive
- Needs GPU instances
- Requires Python ML stack
- Rest of app is Go/Node.js

Solution: Extract as separate service

┌──────────────┐   ┌─────────────────────┐
│ E-Commerce   │   │ Recommendation      │
│ App          │───│ Service (MCP)       │
└──────────────┘   │ - Python            │
                   │ - TensorFlow        │
                   │ - GPU instances     │
                   └─────────────────────┘

Benefits:
✓ Different tech stack (Python vs. Go)
✓ Different infrastructure (GPU vs. CPU)
✓ Can scale independently
✓ ML team can deploy without affecting main app
```

---

## The Architecture: AI-Native Microservices

### Full System Diagram

```
┌─────────────────────────────────────────────┐
│          User (Voice/Chat/Web)              │
└──────────────────┬──────────────────────────┘
                   │ Natural Language
┌──────────────────▼──────────────────────────┐
│          AI Agent (LLM)                     │
│  - Interprets intent                        │
│  - Discovers available tools                │
│  - Orchestrates tool calls                  │
│  - Synthesizes responses                    │
└────┬──────────┬──────────┬─────────────┬───┘
     │ MCP      │ MCP      │ MCP         │ MCP
     │          │          │             │
┌────▼─────┐ ┌─▼────────┐ ┌▼───────────┐ ┌▼──────────┐
│ Payment  │ │  Auth    │ │ Inventory  │ │Recommend  │
│ Service  │ │  Service │ │ Service    │ │Service    │
│          │ │          │ │            │ │           │
│ - Python │ │ - Go     │ │ - Node.js  │ │- Python   │
│ - 3x     │ │ - 2x     │ │ - 2x       │ │- GPU      │
└────┬─────┘ └─┬────────┘ └┬───────────┘ └┬──────────┘
     │         │            │              │
┌────▼────┐ ┌─▼──────┐ ┌──▼────────┐ ┌───▼─────────┐
│Payment  │ │User    │ │Product    │ │Model        │
│Database │ │Database│ │Database   │ │Cache        │
└─────────┘ └────────┘ └───────────┘ └─────────────┘
```

### Service Boundaries

Each service owns:
1. **Business Logic**: Its domain responsibilities
2. **Data**: Its own database/storage
3. **Scaling**: Independent of other services
4. **Deployment**: Can be deployed separately
5. **Interface**: MCP tools for its domain

### Inter-Service Communication

**Option 1: AI Orchestrates All**
```
User: "Buy laptop and use my saved card"

AI orchestrates:
1. search_products("laptop")     → Inventory Service
2. add_to_cart(product_id)       → Cart Service
3. get_saved_payment_methods()   → Payment Service
4. create_order()                → Order Service
5. charge_payment()              → Payment Service
```

AI makes all the calls. Services don't call each other.

**Option 2: Service-to-Service (When Needed)**
```python
# Payment service might need to verify order
@mcp.tool()
async def charge_payment(order_id: str, payment_method_id: str):
    # Call order service to verify
    order = await order_service_client.get_order(order_id)
    
    if order.status != "pending":
        raise InvalidOrderStateError()
    
    # Process payment
    result = payment_gateway.charge(...)
    return result
```

Use traditional service-to-service (gRPC, REST) for internal calls where MCP overhead isn't needed.

---

## Implementing an MCP Microservice

### Complete Example: Product Search Service

```python
# product_service.py
from fastapi import FastAPI, Depends
from fastapi_mcp import MCPRouter
from typing import List, Optional
import asyncpg

app = FastAPI(title="Product Search Service")
mcp = MCPRouter()

# Database connection
async def get_db():
    conn = await asyncpg.connect(os.getenv("DATABASE_URL"))
    try:
        yield conn
    finally:
        await conn.close()

# Business logic (unchanged from traditional service)
class ProductRepository:
    def __init__(self, db):
        self.db = db
    
    async def search(
        self,
        query: str,
        filters: Optional[dict] = None,
        limit: int = 10
    ) -> List[dict]:
        sql = """
            SELECT id, name, description, price, in_stock
            FROM products
            WHERE 
                (name ILIKE $1 OR description ILIKE $1)
                AND ($2::jsonb IS NULL OR attributes @> $2)
                AND in_stock = true
            LIMIT $3
        """
        rows = await self.db.fetch(
            sql,
            f"%{query}%",
            filters,
            limit
        )
        return [dict(row) for row in rows]

# MCP tools (the new interface)
@mcp.tool(
    examples=[
        {
            "query": "Find laptops under $1000",
            "params": {
                "query": "laptop",
                "filters": {"max_price": 1000}
            }
        }
    ]
)
async def search_products(
    query: str,
    filters: Optional[dict] = None,
    limit: int = 10,
    db = Depends(get_db)
) -> List[dict]:
    """
    Search for products by name or description.
    
    Use this when the user wants to find products. Supports
    filtering by price, category, and other attributes.
    
    Args:
        query: Search term (e.g., "laptop", "wireless mouse")
        filters: Optional filters like {"max_price": 1000, "category": "electronics"}
        limit: Maximum number of results (default 10)
    
    Returns:
        List of products matching the search criteria
    """
    repo = ProductRepository(db)
    return await repo.search(query, filters, limit)

@mcp.tool()
async def get_product_details(
    product_id: str,
    db = Depends(get_db)
) -> dict:
    """
    Get detailed information about a specific product.
    
    Use this after search_products to get full details about
    a product the user is interested in.
    
    Args:
        product_id: The unique product identifier
    
    Returns:
        Complete product information including specs, reviews, etc.
    """
    row = await db.fetchrow(
        "SELECT * FROM products WHERE id = $1",
        product_id
    )
    if not row:
        raise NotFoundError(f"Product {product_id} not found")
    return dict(row)

# Mount MCP router
app.include_router(mcp, prefix="/mcp")

# Traditional REST endpoints can coexist
@app.get("/api/products/search")
async def search_products_rest(
    q: str,
    limit: int = 10,
    db = Depends(get_db)
):
    """Traditional REST endpoint - can exist alongside MCP"""
    repo = ProductRepository(db)
    return await repo.search(q, None, limit)

# Health check
@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Deployment

```yaml
# kubernetes/product-service.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: product-service
spec:
  replicas: 2  # Scale independently
  selector:
    matchLabels:
      app: product-service
  template:
    metadata:
      labels:
        app: product-service
    spec:
      containers:
      - name: product-service
        image: myregistry/product-service:1.2.3
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: product-db-secret
              key: url
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: product-service
spec:
  selector:
    app: product-service
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
```

---

## Discovery & Orchestration

### How AI Discovers Services

```python
# AI agent startup
class AIAgent:
    def __init__(self):
        self.available_tools = {}
        self.discover_tools()
    
    def discover_tools(self):
        """Discover all MCP services and their tools"""
        services = [
            "http://product-service/mcp",
            "http://payment-service/mcp",
            "http://order-service/mcp",
            "http://auth-service/mcp"
        ]
        
        for service_url in services:
            # Fetch MCP schema
            schema = requests.get(f"{service_url}/schema").json()
            
            # Register tools
            for tool in schema["tools"]:
                self.available_tools[tool["name"]] = {
                    "url": f"{service_url}/execute",
                    "schema": tool,
                    "service": service_url
                }
        
        logger.info(f"Discovered {len(self.available_tools)} tools")
```

### Tool Execution

```python
async def execute_tool(self, tool_name: str, parameters: dict):
    """Execute a tool on its service"""
    if tool_name not in self.available_tools:
        raise ToolNotFoundError(f"Unknown tool: {tool_name}")
    
    tool_info = self.available_tools[tool_name]
    
    # Call the service
    response = await httpx.post(
        tool_info["url"],
        json={
            "tool": tool_name,
            "parameters": parameters
        },
        headers={"Authorization": f"Bearer {self.agent_token}"}
    )
    
    if response.status_code != 200:
        raise ToolExecutionError(f"Tool failed: {response.text}")
    
    return response.json()
```

---

## Enterprise MCP Registry: Managing Multiple Services

### The Challenge at Scale

As organizations grow their AI-native architecture, they face a new operational challenge: **How do AI agents discover and access dozens or hundreds of MCP services?**

**Small deployment (5-10 services):**
```python
# Hardcoded service list works fine
services = [
    "http://product-service/mcp",
    "http://payment-service/mcp",
    "http://order-service/mcp",
    "http://auth-service/mcp"
]
```

**Enterprise deployment (50-500 services):**
```
Problem:
- 50+ microservices each exposing MCP tools
- Services deployed across regions/clusters
- New services added weekly
- Services version independently
- Need governance and access control
- Require discovery without hardcoding

Traditional solution: Service mesh, Consul, Kubernetes DNS
AI-native addition: MCP Registry
```

### The MCP Registry Pattern

**Purpose:** Centralized catalog of available MCP services, their tools, and metadata for dynamic discovery.[33]

```
┌─────────────────────────────────────────────┐
│          AI Agent Orchestrator              │
└──────────────────┬──────────────────────────┘
                   │ 1. "What tools are available?"
                   ↓
┌──────────────────────────────────────────────┐
│          Enterprise MCP Registry             │
│  - Service catalog                           │
│  - Tool schemas                              │
│  - Health status                             │
│  - Access policies                           │
│  - Version metadata                          │
└─────┬──────────┬──────────┬─────────────────┘
      │          │          │
      │ Registered services
      ↓          ↓          ↓
┌─────────┐ ┌─────────┐ ┌──────────┐
│Payment  │ │Inventory│ │Analytics │
│Service  │ │Service  │ │Service   │
└─────────┘ └─────────┘ └──────────┘
```

### Registry Implementation Options

#### Option 1: Simple Registry Service

```python
# registry_service.py
from fastapi import FastAPI
from typing import List, Dict
import httpx

app = FastAPI(title="MCP Registry")

# In-memory registry (use database in production)
service_registry: Dict[str, dict] = {}

@app.post("/registry/register")
async def register_service(service_info: dict):
    """Services register themselves on startup"""
    service_id = service_info["service_id"]

    # Fetch and validate MCP schema
    schema_url = f"{service_info['base_url']}/mcp/schema"
    async with httpx.AsyncClient() as client:
        response = await client.get(schema_url)
        schema = response.json()

    # Store in registry
    service_registry[service_id] = {
        "service_id": service_id,
        "base_url": service_info["base_url"],
        "tools": schema["tools"],
        "version": service_info.get("version", "unknown"),
        "health_endpoint": f"{service_info['base_url']}/health",
        "registered_at": datetime.utcnow().isoformat(),
        "tags": service_info.get("tags", [])
    }

    return {"status": "registered", "service_id": service_id}

@app.get("/registry/services")
async def list_services() -> List[dict]:
    """AI agent queries available services"""
    # Filter out unhealthy services
    healthy_services = []
    for service in service_registry.values():
        if await check_health(service["health_endpoint"]):
            healthy_services.append(service)

    return healthy_services

@app.get("/registry/search")
async def search_tools(query: str) -> List[dict]:
    """Search for tools by capability"""
    results = []
    for service in service_registry.values():
        for tool in service["tools"]:
            if query.lower() in tool["name"].lower() or \
               query.lower() in tool.get("description", "").lower():
                results.append({
                    "service_id": service["service_id"],
                    "tool": tool,
                    "base_url": service["base_url"]
                })
    return results

async def check_health(health_url: str) -> bool:
    """Check if service is healthy"""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(health_url)
            return response.status_code == 200
    except:
        return False
```

#### Option 2: Federated Registry (Multi-Region)

**For global enterprises with regional deployments:**

```
┌──────────────────────────────────────────────┐
│      Global MCP Registry (Aggregator)        │
│      - Federates regional registries         │
│      - Cross-region tool discovery           │
└────┬──────────────┬─────────────────┬────────┘
     │              │                 │
     ↓              ↓                 ↓
┌─────────────┐ ┌────────────┐ ┌───────────┐
│US-East      │ │EU-West     │ │APAC       │
│Registry     │ │Registry    │ │Registry   │
│             │ │            │ │           │
│50 services  │ │40 services │ │30 services│
└─────────────┘ └────────────┘ └───────────┘
```

```python
# Federated registry query
@app.get("/registry/services/federated")
async def get_all_services(region: Optional[str] = None):
    """Query across all regional registries"""
    registries = {
        "us-east": "https://registry-us.company.com",
        "eu-west": "https://registry-eu.company.com",
        "apac": "https://registry-apac.company.com"
    }

    if region:
        registries = {region: registries[region]}

    all_services = []
    async with httpx.AsyncClient() as client:
        tasks = [
            client.get(f"{url}/registry/services")
            for url in registries.values()
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        for response in responses:
            if isinstance(response, httpx.Response):
                all_services.extend(response.json())

    return all_services
```

#### Option 3: Azure API Center Integration

**For enterprises using Azure:**[38]

Azure API Center can serve as an MCP registry by treating each MCP service as an API.

```python
# Register MCP service in Azure API Center
from azure.mgmt.apicenter import ApiCenterMgmt

def register_mcp_in_azure(service_info: dict):
    """Register MCP service in Azure API Center"""
    client = ApiCenterMgmt(credential, subscription_id)

    # Create API entry for MCP service
    api = client.apis.create_or_update(
        resource_group_name="mcp-services",
        service_name="enterprise-api-center",
        api_name=service_info["service_id"],
        payload={
            "title": service_info["title"],
            "kind": "mcp",  # Custom kind for MCP
            "description": service_info["description"],
            "contacts": service_info.get("contacts", []),
            "customProperties": {
                "mcp_endpoint": service_info["base_url"] + "/mcp",
                "mcp_version": "1.0",
                "tool_count": len(service_info["tools"])
            }
        }
    )

    # Register each tool as an operation
    for tool in service_info["tools"]:
        client.api_operations.create_or_update(
            resource_group_name="mcp-services",
            service_name="enterprise-api-center",
            api_name=service_info["service_id"],
            operation_name=tool["name"],
            payload={
                "title": tool["name"],
                "description": tool.get("description", ""),
                "summary": tool.get("summary", "")
            }
        )
```

**Benefits of Azure API Center approach:**
- Unified governance for REST APIs and MCP tools
- Built-in compliance and security policies
- Enterprise authentication integration
- Existing monitoring and analytics

### Service Self-Registration

Services register themselves on startup:

```python
# payment_service.py - Self-registration on startup
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Register with MCP Registry on startup"""
    # Startup
    registry_url = os.getenv("MCP_REGISTRY_URL")
    service_info = {
        "service_id": "payment-service",
        "base_url": os.getenv("SERVICE_URL"),
        "version": "1.2.3",
        "tags": ["payments", "transactions", "billing"]
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{registry_url}/registry/register",
            json=service_info
        )
        if response.status_code == 200:
            logger.info("Successfully registered with MCP Registry")
        else:
            logger.error(f"Failed to register: {response.text}")

    yield

    # Shutdown: Deregister
    async with httpx.AsyncClient() as client:
        await client.delete(
            f"{registry_url}/registry/services/{service_info['service_id']}"
        )

app = FastAPI(lifespan=lifespan)
```

### AI Agent Discovery Flow

```python
# AI agent uses registry for dynamic discovery
class EnterpriseAIAgent:
    def __init__(self, registry_url: str):
        self.registry_url = registry_url
        self.tools_cache = {}
        self.last_refresh = None

    async def refresh_tools(self):
        """Periodically refresh available tools from registry"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.registry_url}/registry/services"
            )
            services = response.json()

        # Build tool catalog
        self.tools_cache = {}
        for service in services:
            for tool in service["tools"]:
                tool_id = f"{service['service_id']}.{tool['name']}"
                self.tools_cache[tool_id] = {
                    "service_url": service["base_url"],
                    "tool_schema": tool,
                    "service_version": service["version"]
                }

        self.last_refresh = datetime.utcnow()
        logger.info(f"Refreshed {len(self.tools_cache)} tools from registry")

    async def find_tools_for_intent(self, user_query: str) -> List[str]:
        """Search registry for relevant tools"""
        # Refresh if stale (>5 minutes)
        if not self.last_refresh or \
           (datetime.utcnow() - self.last_refresh).seconds > 300:
            await self.refresh_tools()

        # Use LLM to determine which tools to use
        available_tools = [
            {
                "name": tool_id,
                "description": tool_info["tool_schema"].get("description")
            }
            for tool_id, tool_info in self.tools_cache.items()
        ]

        # LLM selects relevant tools
        selected_tools = await self.llm.select_tools(
            query=user_query,
            available_tools=available_tools
        )

        return selected_tools
```

### Governance and Access Control

```python
# Add policy enforcement to registry
@app.get("/registry/services")
async def list_services(
    user_context: UserContext = Depends(get_user_context)
) -> List[dict]:
    """Return only services user has access to"""
    all_services = service_registry.values()

    # Filter by user permissions
    allowed_services = []
    for service in all_services:
        if has_access(user_context, service):
            allowed_services.append(service)

    return allowed_services

def has_access(user_context: UserContext, service: dict) -> bool:
    """Check if user can access this service"""
    # Check user's role against service requirements
    required_roles = service.get("required_roles", [])
    user_roles = user_context.roles

    if not required_roles:
        return True  # Public service

    return any(role in required_roles for role in user_roles)
```

### Monitoring Registry Health

```python
# Background task to monitor service health
@app.on_event("startup")
async def start_health_monitor():
    asyncio.create_task(health_check_loop())

async def health_check_loop():
    """Continuously monitor registered services"""
    while True:
        for service_id, service in service_registry.items():
            is_healthy = await check_health(service["health_endpoint"])

            # Update service status
            service["healthy"] = is_healthy
            service["last_health_check"] = datetime.utcnow().isoformat()

            if not is_healthy:
                logger.warning(f"Service {service_id} is unhealthy")
                # Could trigger alerts, remove from rotation, etc.

        # Check every 30 seconds
        await asyncio.sleep(30)
```

### This Solves Real Problems

**Without MCP Registry:**
- AI agents hardcode service URLs
- Adding new services requires agent redeployment
- No visibility into available tools
- Manual service discovery process
- Difficult to enforce governance

**With MCP Registry:**
- AI agents discover services dynamically
- New services auto-register and become available
- Centralized catalog of capabilities
- Search and filter tools by intent
- Enforce access control and compliance

### Industry Adoption

**MCP Registry standardization:**[33][37] The official MCP Registry launched in September 2025 as "an open catalog and API for publicly available MCP servers." The goal is to standardize how servers are distributed and discovered, with open-source architecture allowing compatible sub-registries.[33]

**Enterprise implementations:**[39] IBM's ContextForge provides "a feature-rich gateway, proxy and MCP Registry that federates MCP and REST services," demonstrating enterprise-scale registry patterns.[39]

---

## The Hybrid Reality

Most enterprises will run **both** REST and MCP:

```
┌────────────────────┐          ┌────────────────────┐
│  Traditional UI    │          │  AI Assistant      │
│  (Admin Dashboard) │          │  (Customer Facing) │
└─────────┬──────────┘          └─────────┬──────────┘
          │ REST                          │ MCP
          │                               │
          └───────────┬───────────────────┘
                      │
              ┌───────▼────────┐
              │   Service      │
              │                │
              │  REST + MCP    │
              │  (Both)        │
              └────────────────┘

Same business logic, two interfaces.
```

### Supporting Both

```python
# Core business logic
class PaymentService:
    async def process_payment(self, amount, user_id):
        # Business logic here
        pass

# REST interface
@app.post("/api/payments")
async def create_payment_rest(
    request: PaymentRequest,
    user: User = Depends(get_current_user)
):
    return await payment_service.process_payment(
        request.amount,
        user.id
    )

# MCP interface
@mcp.tool()
async def create_payment(
    amount: int,
    user_context: UserContext = Depends()
):
    """Process a payment for the current user"""
    return await payment_service.process_payment(
        amount,
        user_context.user_id
    )
```

---

## Summary

**Key Points:**

✓ **MCP services ARE microservices** - just with a different protocol

✓ **Decomposition criteria unchanged** - split by scaling, tech, teams, or compliance

✓ **Start with modular monolith** - extract services as needed

✓ **Each service owns its domain** - data, logic, scaling, deployment

✓ **AI orchestrates across services** - no hardcoded service mesh from client

✓ **Hybrid approach is common** - REST for internal/admin, MCP for AI

---

## References

[24] Anthropic. "Introducing the Model Context Protocol." Anthropic News, November 2024. Available at: https://www.anthropic.com/news/model-context-protocol
   - Official MCP announcement: "MCP is an open protocol that standardizes how applications provide context to LLMs"
   - "Just as USB-C provides a standardized way to connect your devices to various peripherals and accessories, MCP provides a standardized way to connect AI models to different data sources and tools"

[25] InfoQ. "Anthropic Publishes Model Context Protocol Specification for LLM App Integration." December 2024. Available at: https://www.infoq.com/news/2024/12/anthropic-model-context-protocol/
   - "Anthropic open-sourced the Model Context Protocol (MCP), a new standard for connecting AI assistants to the systems where data lives"
   - "The protocol was released with software development kits (SDKs) in programming languages including Python, TypeScript, C# and Java"
   - Official documentation: https://docs.claude.com/en/docs/mcp

[26] Fowler, M. "Monolith First." MartinFowler.com, June 2015. Available at: https://martinfowler.com/bliki/MonolithFirst.html
   - "Almost all the successful microservice stories have started with a monolith that got too big and was broken up"
   - "Design a monolith carefully, paying attention to modularity within the software, both at the API boundaries and how the data is stored. This makes it relatively simple to shift to microservices"

[27] Tech World with Milan. "Why should you build a (modular) monolith first?" Newsletter, 2024. Available at: https://newsletter.techworld-with-milan.com/p/why-you-should-build-a-modular-monolith
   - "Following a modular approach brings many similar benefits to microservices like re-usability, easy refactoring, better organized dependencies and teams"
   - "An adequately produced modular monolith can be transformed into a microservice solution if needed"
   - Recommended path: "Monolith > apps > services > microservices"

[38] Microsoft Learn. "Azure API Center - Overview." Available at: https://learn.microsoft.com/en-us/azure/api-center/overview
   - "Azure API Center enables tracking all of your APIs in a centralized location for discovery, reuse, and governance"
   - "Inventory all APIs in your organization in a centralized place. APIs can be of any type (e.g., REST, GraphQL, gRPC)"
   - Design-time API governance and centralized API discovery solution

[39] IBM. "MCP Context Forge - Model Context Protocol Gateway." Available at: https://github.com/IBM/mcp-context-forge
   - "A Model Context Protocol (MCP) Gateway & Registry. Serves as a central management point for tools, resources, and prompts"
   - "Converts REST API endpoints to MCP, composes virtual MCP servers with added security and observability"
   - "ContextForge MCP Gateway is a feature-rich gateway, proxy and MCP Registry that federates MCP and REST services"

---

## Key Takeaways

✓ **MCP services ARE microservices** - An MCP-enabled service that handles a specific business domain, scales independently, has its own database, and provides failure isolation IS a microservice; only the protocol changes (MCP instead of REST), not the architectural pattern; the distinction is artificial like "HTTP microservices" vs "gRPC microservices"

✓ **Decomposition criteria remain unchanged** - Split services based on traditional factors: different scaling characteristics, different technology requirements, team boundaries, compliance/security isolation, or different deployment cadence; AI-native doesn't change when to use microservices

✓ **Start with modular monolith, extract as needed** - Martin Fowler's guidance holds: almost all successful microservice stories started with a monolith; design monolith carefully with modular boundaries, making it simple to extract services when scaling, tech stack, or team factors require it

✓ **Enterprise MCP Registry essential at scale** - Small deployments (5-10 services) can hardcode service URLs; enterprise deployments (50-500 services) need centralized MCP Registry for dynamic discovery, health monitoring, access control, and governance; implementations include custom registries, federated multi-region registries, or Azure API Center integration

✓ **Hybrid REST+MCP approach is common** - Most enterprises run both: REST for traditional UI/admin dashboards, MCP for AI-native interfaces; both use same business logic; services self-register with registry on startup; AI agents discover and refresh tools dynamically every 5 minutes

✓ **AI orchestrates across services** - AI agent discovers available tools from registry, selects relevant tools for user intent, and executes cross-service workflows; services don't need to know about each other—AI handles orchestration; traditional service-to-service calls (gRPC/REST) still used for internal operations where MCP overhead unnecessary

---

**[← Previous: Chapter 4 - What's New](chapter-4-whats-new) | [Next: Chapter 6 - UI Layer →](chapter-6-ui-layer)**