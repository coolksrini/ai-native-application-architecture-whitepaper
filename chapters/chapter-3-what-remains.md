# Chapter 3: What Remains

## Introduction

Amid the excitement about AI-native architecture, it's crucial to recognize what doesn't change. Many hard-won lessons from decades of software engineering remain valid. This chapter identifies the fundamentals that persist across the paradigm shift.

---

## The Microservices Pattern Remains Valid

### The Core Principle

**Microservices decompose systems by business domain, not by technology.**[15]

Martin Fowler defines the microservice architectural style as "an approach to developing a single application as a suite of small services, each running in its own process and communicating with lightweight mechanisms."[16] This principle is orthogonal to whether services speak REST or MCP. The reasons for splitting (or not splitting) into separate services remain the same.

**Key insight:** When breaking a software system into components, look for points where you can imagine rewriting a component without affecting its collaborators.[16] This applies equally to REST-based and MCP-based services.

### When to Use Microservices (AI-Native or Not)

#### ✓ Split into Separate Services When:

**1. Different Scaling Needs**
```
Payment Service:
- High volume during checkout hours
- Needs horizontal scaling
- Latency-sensitive

Analytics Service:
- Batch processing overnight
- Needs vertical scaling (memory)
- Latency-tolerant

→ Separate services make sense
```

**2. Technology Heterogeneity**
```
Auth Service:
- Written in Go for performance
- Uses Redis for session storage

ML Recommendation Service:
- Written in Python for ML libraries
- Uses GPU instances
- Different deployment cadence

→ Different tech stacks → Separate services
```

**3. Team Boundaries**
```
Payments Team (6-10 people):
- Owns payment processing
- Deploys independently
- On-call for payment issues

Inventory Team (6-10 people):
- Owns stock management
- Deploys independently
- On-call for inventory issues

→ Team autonomy → Separate services
```

Best practice: Each service should be small enough to be owned by an autonomous team of 6-10 people.[17] This ensures the service remains cohesive and manageable.

**4. Compliance/Security Isolation**
```
PCI-DSS Compliant Service:
- Payment data
- Strict audit requirements
- Limited access

Regular Application Services:
- No sensitive data
- Standard security

→ Regulatory boundary → Separate service
```

#### ✗ Keep as Monolith When:

```
Small team (<10 people):
- Everyone knows the codebase
- Deployment coordination is easy
- Communication overhead of microservices > benefits

Shared data model:
- Orders and order items are tightly coupled
- Frequent joins between tables
- Splitting would require complex distributed transactions

Early stage / MVP:
- Requirements changing rapidly
- Refactoring across service boundaries is painful
- Premature optimization

No clear scaling differences:
- All features have similar load
- All scale together
- Microservices add complexity without benefits
```

### The Key Insight

```
Traditional Microservice (REST):
Payment Service
├── Business logic: Process payments
├── Data: Payment database
├── Scaling: Independent
└── Interface: REST API + OpenAPI schema

AI-Native Microservice (MCP):
Payment Service
├── Business logic: Process payments (SAME)
├── Data: Payment database (SAME)
├── Scaling: Independent (SAME)
└── Interface: MCP protocol + tool schemas (DIFFERENT)

The architectural pattern is identical.
Only the protocol changed.
```

### MCP: Not Just for AI

**Important clarification:** MCP is not "AI-only." The protocol can be consumed by both traditional clients and AI agents:

```python
# MCP service exposes tools with rich schemas
@app.mcp_tool()
async def create_payment(
    amount: int,
    payment_method_id: str,
    user_context: UserContext
) -> PaymentResult:
    """
    Process a payment for the current user.

    Args:
        amount: Payment amount in cents (e.g., $10 = 1000)
        payment_method_id: ID of the payment method to charge

    Returns:
        Payment result with transaction ID and status
    """
    return await payment_service.process(amount, payment_method_id, user_context.user_id)
```

**How different clients use the same MCP service:**

**Traditional Client (e.g., React app):**
```typescript
// Traditional client uses MCP like it would use REST+OpenAPI
// - Reads schema for validation and type safety
// - Makes direct, hardcoded calls
// - Uses schema to guide traditional flows

const paymentClient = new MCPClient('http://payment-service/mcp');

// Client directly calls the tool (knows what it wants)
const result = await paymentClient.call('create_payment', {
  amount: 1000,  // $10.00
  payment_method_id: 'pm_123'
});

// Uses schema for:
// - TypeScript type generation
// - Input validation
// - Error handling
// - Documentation
```

**AI Agent:**
```python
# AI agent uses MCP dynamically
# - Discovers tools at runtime
# - Interprets tool descriptions semantically
# - Orchestrates multiple tools based on user intent

# User says: "Charge John's card $10 for the consultation"

# AI reasoning:
# 1. Discovers create_payment tool
# 2. Understands it needs amount in cents (from description)
# 3. Converts $10 → 1000 cents
# 4. Finds John's payment method
# 5. Calls create_payment(amount=1000, payment_method_id="pm_john")

# Same tool, different usage pattern
```

**The key difference:**
- **Traditional clients**: Use MCP schemas like OpenAPI (static, hardcoded flows)
- **AI agents**: Use MCP schemas for dynamic discovery and orchestration
- **Both**: Read the same service, same schema, same tools

This makes MCP a superset of REST+OpenAPI, not a replacement. You get:
- Rich schemas that traditional clients can use
- Semantic descriptions that AI agents can understand
- Single service supporting both interaction modes

---

## Business Logic Remains Unchanged

### Core Operations Are the Same

```python
# This function doesn't care about REST vs. MCP
def process_payment(
    amount: int,
    user_id: str,
    payment_method_id: str
) -> PaymentResult:
    # 1. Validate
    if amount <= 0:
        raise ValueError("Amount must be positive")
    
    # 2. Check balance/limits
    user = get_user(user_id)
    if amount > user.payment_limit:
        raise InsufficientLimitError()
    
    # 3. Process payment
    payment_gateway = get_payment_gateway()
    result = payment_gateway.charge(
        amount=amount,
        payment_method=payment_method_id
    )
    
    # 4. Record transaction
    transaction = create_transaction_record(
        user_id=user_id,
        amount=amount,
        status=result.status
    )
    
    # 5. Update user balance
    update_user_balance(user_id, -amount)
    
    # 6. Send notification
    send_payment_confirmation(user_id, transaction.id)
    
    return PaymentResult(
        transaction_id=transaction.id,
        status=result.status
    )
```

**The only thing that changes is the wrapper:**

```python
# Traditional REST wrapper
@app.post("/api/payments")
async def create_payment_endpoint(
    request: PaymentRequest,
    user: User = Depends(get_current_user)
):
    result = process_payment(
        amount=request.amount,
        user_id=user.id,
        payment_method_id=request.payment_method_id
    )
    return result

# AI-Native MCP wrapper
@app.mcp_tool()
async def create_payment(
    amount: int,
    payment_method_id: str,
    user_context: UserContext = Depends(get_user_context)
) -> PaymentResult:
    """
    Process a payment for the current user.
    Amount should be in cents (e.g., $10.00 = 1000).
    """
    result = process_payment(
        amount=amount,
        user_id=user_context.user_id,
        payment_method_id=payment_method_id
    )
    return result
```

### Domain Logic Stays Domain Logic

```python
# Inventory management - unchanged
class InventoryManager:
    def reserve_items(self, order_items: List[OrderItem]) -> Reservation:
        for item in order_items:
            if not self.check_availability(item.product_id, item.quantity):
                raise OutOfStockError(item.product_id)
        
        reservation = self.create_reservation(order_items)
        return reservation
    
    def commit_reservation(self, reservation_id: str):
        reservation = self.get_reservation(reservation_id)
        for item in reservation.items:
            self.decrement_stock(item.product_id, item.quantity)
        reservation.status = "committed"

# This logic works the same whether called from REST or MCP
```

### Validation Rules Don't Change

```python
# Business rules stay the same
def validate_order(order: Order) -> ValidationResult:
    errors = []
    
    # Rule 1: Minimum order amount
    if order.total < 10.00:
        errors.append("Minimum order is $10")
    
    # Rule 2: Valid shipping address
    if not is_valid_address(order.shipping_address):
        errors.append("Invalid shipping address")
    
    # Rule 3: Items in stock
    for item in order.items:
        if not is_in_stock(item.product_id, item.quantity):
            errors.append(f"{item.name} is out of stock")
    
    return ValidationResult(valid=len(errors) == 0, errors=errors)
```

---

## Data Storage & Modeling Remain Critical

### Database Design Doesn't Change

```sql
-- This schema is the same for REST or MCP services
CREATE TABLE users (
    id UUID PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE orders (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    total DECIMAL(10,2),
    status VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE order_items (
    id UUID PRIMARY KEY,
    order_id UUID REFERENCES orders(id),
    product_id UUID,
    quantity INTEGER,
    price DECIMAL(10,2)
);

-- Indexes for performance
CREATE INDEX idx_orders_user_id ON orders(user_id);
CREATE INDEX idx_orders_created_at ON orders(created_at);
```

### ACID Transactions Still Matter

```python
# Transaction boundaries remain critical
async def create_order(user_id: str, items: List[CartItem]):
    async with database.transaction():
        # 1. Create order record
        order = await db.orders.create(user_id=user_id)
        
        # 2. Create order items
        for item in items:
            await db.order_items.create(
                order_id=order.id,
                product_id=item.product_id,
                quantity=item.quantity
            )
        
        # 3. Update inventory
        for item in items:
            await db.inventory.decrement(
                product_id=item.product_id,
                quantity=item.quantity
            )
        
        # Either all succeed or all rollback
        return order
```

### Caching Strategies Don't Change

```python
# Cache patterns remain the same
@cache(ttl=3600)  # Cache for 1 hour
def get_product(product_id: str) -> Product:
    return db.query(Product).filter(Product.id == product_id).first()

# Redis for session data
def get_user_session(session_id: str) -> Session:
    cached = redis.get(f"session:{session_id}")
    if cached:
        return Session.parse(cached)
    
    session = db.query(Session).filter(Session.id == session_id).first()
    redis.setex(f"session:{session_id}", 3600, session.json())
    return session
```

### Data Modeling Best Practices Persist

```python
# Normalization vs. Denormalization trade-offs remain
# Event sourcing, CQRS, and other patterns still apply

# Example: Event sourcing for audit trail
class PaymentEvent:
    timestamp: datetime
    event_type: str  # 'created', 'authorized', 'captured', 'refunded'
    payment_id: str
    amount: int
    metadata: dict

# The pattern works regardless of REST or MCP
```

---

## Infrastructure & DevOps Fundamentals Persist

### Containerization & Orchestration

```yaml
# Kubernetes deployment - works for REST or MCP services
apiVersion: apps/v1
kind: Deployment
metadata:
  name: payment-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: payment-service
  template:
    metadata:
      labels:
        app: payment-service
    spec:
      containers:
      - name: payment-service
        image: payment-service:v1.2.3
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-secret
              key: url
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

### CI/CD Pipelines

```yaml
# GitHub Actions workflow - adds MCP evaluation, but CI/CD basics remain
name: Deploy Payment Service

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      # Traditional tests still needed
      - name: Run unit tests
        run: pytest tests/unit
      
      - name: Run integration tests
        run: pytest tests/integration
      
      # NEW: MCP-specific tests
      - name: Run MCP evaluation
        run: python evaluate_mcp.py
      
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Build Docker image
        run: docker build -t payment-service:${{ github.sha }} .
      
      - name: Push to registry
        run: docker push payment-service:${{ github.sha }}
  
  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Kubernetes
        run: kubectl apply -f k8s/
```

### Monitoring & Alerting

```python
# Observability basics remain critical
import logging
from prometheus_client import Counter, Histogram

# Metrics
payment_requests = Counter('payment_requests_total', 'Total payment requests')
payment_duration = Histogram('payment_duration_seconds', 'Payment processing time')

# Logging
logger = logging.getLogger(__name__)

async def process_payment(amount: int, user_id: str):
    payment_requests.inc()
    
    with payment_duration.time():
        try:
            logger.info(f"Processing payment: amount={amount}, user={user_id}")
            result = await payment_gateway.charge(amount, user_id)
            logger.info(f"Payment successful: {result.transaction_id}")
            return result
        except PaymentError as e:
            logger.error(f"Payment failed: {e}", exc_info=True)
            raise

# These fundamentals don't change with MCP
```

### Security Basics

```python
# Authentication & authorization principles remain
# Encryption at rest and in transit
# Secrets management
# Rate limiting
# Input validation

# Example: Rate limiting still needed
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

@app.mcp_tool()
@limiter.limit("100/minute")  # Still need rate limits
async def search_products(query: str) -> List[Product]:
    """Search products by name or description"""
    return await db.search_products(query)
```

---

## Software Engineering Principles Endure

### SOLID Principles Still Apply

```python
# Single Responsibility Principle
class PaymentProcessor:
    """Only handles payment processing"""
    def process(self, payment: Payment) -> Result:
        pass

class NotificationService:
    """Only handles notifications"""
    def send(self, user_id: str, message: str):
        pass

# Dependency Inversion
class OrderService:
    def __init__(
        self,
        payment_processor: PaymentProcessor,  # Depend on abstractions
        notification_service: NotificationService
    ):
        self.payment = payment_processor
        self.notifications = notification_service
```

### Error Handling Patterns Persist

```python
# Try-except, retries, circuit breakers all still relevant
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def call_payment_gateway(amount: int):
    """Retry with exponential backoff"""
    try:
        return await payment_gateway.charge(amount)
    except PaymentGatewayTimeout:
        logger.warning("Payment gateway timeout, retrying...")
        raise  # Will retry
    except PaymentGatewayError as e:
        logger.error(f"Payment gateway error: {e}")
        raise  # Won't retry, permanent error
```

### Code Quality Matters

```python
# Clean code, tests, documentation - all still important

def calculate_order_total(items: List[OrderItem]) -> Decimal:
    """
    Calculate total order amount including tax.
    
    Args:
        items: List of order items with price and quantity
        
    Returns:
        Total amount in cents
        
    Example:
        >>> items = [OrderItem(price=1000, quantity=2)]
        >>> calculate_order_total(items)
        Decimal('2000')
    """
    subtotal = sum(item.price * item.quantity for item in items)
    tax = subtotal * Decimal('0.0825')  # 8.25% tax rate
    return subtotal + tax

# Tests
def test_calculate_order_total():
    items = [
        OrderItem(price=Decimal('10.00'), quantity=2),
        OrderItem(price=Decimal('5.00'), quantity=1)
    ]
    total = calculate_order_total(items)
    assert total == Decimal('27.06')  # $25 + 8.25% tax
```

---

## The Architecture Diagram Is Similar

### Traditional Microservices

```
┌─────────────────────────────────────────┐
│          Frontend (React)               │
└────────────┬────────────────────────────┘
             │ HTTP/REST
┌────────────▼────────────────────────────┐
│          API Gateway                    │
└────┬──────────┬──────────┬─────────────┘
     │          │          │
┌────▼─────┐ ┌─▼────────┐ ┌▼───────────┐
│ Payment  │ │  Auth    │ │ Inventory  │
│ Service  │ │  Service │ │ Service    │
└──────────┘ └──────────┘ └────────────┘
     │            │            │
┌────▼────┐  ┌───▼────┐  ┌───▼────────┐
│Payment  │  │User    │  │Product     │
│Database │  │Database│  │Database    │
└─────────┘  └────────┘  └────────────┘
```

### AI-Native Microservices

```
┌─────────────────────────────────────────┐
│     User (Voice/Chat/Hybrid)            │
└────────────┬────────────────────────────┘
             │ Natural Language
┌────────────▼────────────────────────────┐
│          AI Agent (LLM)                 │
└────┬──────────┬──────────┬─────────────┘
     │ MCP      │ MCP      │ MCP
┌────▼─────┐ ┌─▼────────┐ ┌▼───────────┐
│ Payment  │ │  Auth    │ │ Inventory  │
│ Service  │ │  Service │ │ Service    │
│ (MCP)    │ │  (MCP)   │ │ (MCP)      │
└──────────┘ └──────────┘ └────────────┘
     │            │            │
┌────▼────┐  ┌───▼────┐  ┌───▼────────┐
│Payment  │  │User    │  │Product     │
│Database │  │Database│  │Database    │
└─────────┘  └────────┘  └────────────┘
```

**The structure is identical. Only the protocol changed.**

---

## What About the Hybrid Approach?

Most real applications will run both REST and MCP:

```
┌──────────────────┐     ┌──────────────────┐
│  Traditional UI  │     │  AI Assistant    │
│  (Admin Panel)   │     │  (User Facing)   │
└────────┬─────────┘     └────────┬─────────┘
         │ REST                   │ MCP
         └───────────┬────────────┘
                     │
              ┌──────▼──────┐
              │   Service   │
              │             │
              │ REST + MCP  │
              └─────────────┘
```

**The service can support both:**

```python
# Same business logic, two interfaces
class PaymentService:
    async def process_payment(self, amount, user_id):
        # Core business logic
        pass

# REST interface
@app.post("/api/payments")
async def create_payment_rest(request: PaymentRequest):
    return await payment_service.process_payment(
        request.amount,
        request.user_id
    )

# MCP interface
@app.mcp_tool()
async def create_payment(amount: int, user_context: UserContext):
    return await payment_service.process_payment(
        amount,
        user_context.user_id
    )
```

---

## Summary: What Doesn't Change

| Aspect | Status | Details |
|--------|--------|---------|
| **Microservices Pattern** | ✓ SAME | Decompose by domain, scale independently |
| **Business Logic** | ✓ SAME | Payment processing, validation, etc. |
| **Data Storage** | ✓ SAME | SQL, NoSQL, caching strategies |
| **Infrastructure** | ✓ SAME | Kubernetes, Docker, CI/CD |
| **Monitoring** | ✓ SAME | Logs, metrics, traces (plus new AI metrics) |
| **Security Basics** | ✓ SAME | Encryption, secrets, rate limiting |
| **Code Quality** | ✓ SAME | SOLID, testing, documentation |

---

## Key Takeaways

✓ **Microservices remain valid** - the pattern doesn't change, only the protocol

✓ **Business logic is unchanged** - payment processing works the same regardless of interface

✓ **Infrastructure fundamentals persist** - containers, orchestration, CI/CD all still needed

✓ **Data storage patterns endure** - databases, transactions, caching haven't changed

✓ **Software engineering principles apply** - SOLID, clean code, testing still matter

✓ **Don't throw out decades of knowledge** - AI-native builds on, not replaces, existing best practices

---

## References

[15] Fowler, M. "Microservices Principles: Decomposition." Tamerlan.dev, 2024. Available at: https://tamerlan.dev/microservices-principles-decomposition/
   - "Decompose systems by business capability and not technology"
   - Reduces complications of different technology teams interacting

[16] Fowler, M. and Lewis, J. "Microservices: a definition of this new architectural term." MartinFowler.com, March 2014. Available at: https://martinfowler.com/articles/microservices.html
   - Official definition: "An approach to developing a single application as a suite of small services, each running in its own process and communicating with lightweight mechanisms"
   - Key insight: "Look for points where you can imagine rewriting a component without affecting its collaborators"

[17] Graph AI. "Martin Fowler's Insights on Microservices: A Comprehensive Guide." Graph AI Blog, 2024. Available at: https://www.graphapp.ai/blog/martin-fowler-s-insights-on-microservices-a-comprehensive-guide
   - "Each service should be small enough to be owned by an autonomous team of 6-10 people"
   - Services must be cohesive (implementing strongly related functionality) and loosely coupled

---

**[← Previous: Chapter 2 - What Changes](chapter-2-what-changes.md) | [Next: Chapter 4 - What's Entirely New →](chapter-4-whats-new.md)**