# Chapter 12: The Migration Path

## Introduction

Moving from a traditional web application to an AI-native architecture is not an all-or-nothing proposition. This chapter provides a practical, phase-based approach for migrating existing systems while maintaining business continuity.

**Key principle:** Incremental adoption with continuous value delivery.

---

## The Three-Phase Migration Strategy

### Overview

```
PHASE 1: Hybrid Foundation (Months 1-3)
├─ Add MCP alongside REST
├─ Deploy AI assistant for simple queries
├─ Traditional UI remains primary
└─ 10% AI-native, 90% traditional

PHASE 2: AI-First Features (Months 4-8)
├─ New features built AI-native first
├─ Migrate high-traffic read operations
├─ Begin fine-tuning on enterprise tools
└─ 40% AI-native, 60% traditional

PHASE 3: AI-Native Core (Months 9-18)
├─ Most interactions via AI
├─ Traditional UI for admin/complex tasks
├─ Complete scenario evaluation suite
└─ 80% AI-native, 20% traditional
```

---

## Phase 1: Hybrid Foundation (Months 1-3)

### Goals

- Prove AI-native concept with minimal risk
- Build technical foundation
- Learn what works and what doesn't
- Deliver immediate value to users

### Step 1: Add MCP Endpoints Alongside REST

**Don't replace REST—augment it.**

```python
# Existing REST endpoint - keep it
@app.get("/api/orders")
async def get_orders_rest(
    user: User = Depends(get_current_user),
    limit: int = 10
):
    """Existing REST endpoint - unchanged"""
    orders = db.query_orders(user.id, limit=limit)
    return {"orders": orders}

# New MCP tool - add alongside
@app.mcp_tool()
async def get_user_orders(
    limit: int = 10,
    user_context: UserContext = Depends(get_user_context)
) -> List[Order]:
    """
    Retrieve user's order history.
    Use this when user asks about their orders.
    """
    orders = db.query_orders(user_context.user_id, limit=limit)
    return orders

# Both use the same business logic
# Frontend uses REST, AI uses MCP
```

### Step 1.5: Add Training Data Endpoints

**Enable decentralized training data collection (Chapter 11 pattern):**

```python
# payment-service/mcp_training.py
from fastapi import Depends
from fastapi_mcp import MCPRouter

mcp = MCPRouter()

@mcp.get("/training-dataset")
async def get_training_dataset(
    auth: ServiceAuth = Depends(verify_ml_orchestrator)
):
    """
    Expose training data for enterprise fine-tuning.
    Only accessible by central ML orchestrator.

    Phase 1: Start collecting manually curated examples
    Phase 2: Add production-validated examples
    Phase 3: Continuous automated collection
    """

    if not auth.has_role("ml_orchestrator"):
        raise HTTPException(403, "Unauthorized")

    # Return training examples for this service
    return {
        "service": "payment-service",
        "training_examples": [
            {
                "user_query": "Charge customer $50",
                "correct_tool": "create_payment",
                "correct_parameters": {
                    "amount": 5000,  # cents
                    "currency": "USD"
                },
                "quality": "human_curated",
                "source": "developer"
            },
            # More examples...
        ],
        "metadata": {
            "total_examples": 47,
            "last_updated": "2025-01-20T10:30:00Z"
        }
    }

@mcp.get("/test-dataset")
async def get_test_dataset(
    auth: ServiceAuth = Depends(verify_ml_orchestrator)
):
    """Critical test scenarios for this service"""

    if not auth.has_role("ml_orchestrator"):
        raise HTTPException(403, "Unauthorized")

    return {
        "service": "payment-service",
        "test_scenarios": [
            {
                "id": "PAY-001",
                "priority": "critical",
                "query": "Charge customer $100 for invoice INV-456",
                "expected_tool": "create_payment",
                "expected_params": {
                    "amount": 10000,
                    "invoice_id": "INV-456"
                },
                "min_accuracy_required": 0.98
            },
            # More scenarios...
        ]
    }
```

**Benefits:**
- ✅ Each service owns its training data
- ✅ ML orchestrator auto-aggregates for fine-tuning
- ✅ No manual export/import process
- ✅ Scales with number of services
- ✅ Enables weekly retraining without coordination

**Timeline:**
```
Phase 1 (Month 1-2): Add endpoints to first 3 services
Phase 2 (Month 4-6): Rollout to all services
Phase 3 (Month 9+): Automated collection from production
```

### Step 2: Choose Low-Risk Starting Point

**Select features that are:**
- ✅ Read-only (no data modification)
- ✅ High-volume (immediate impact)
- ✅ Simple (single-service queries)
- ✅ Low-risk (not critical business functions)

**Good starting points:**
```
✓ Order status lookup
✓ Product search
✓ FAQ / Help queries
✓ Account information display
✓ Transaction history

✗ Avoid initially:
✗ Payment processing
✗ Data deletion
✗ Permission changes
✗ Multi-step transactions
```

### Step 3: Deploy AI Assistant Widget

**Embed conversational interface in existing UI:**

```html
<!-- existing-app.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Your App</title>
</head>
<body>
    <!-- Existing UI unchanged -->
    <nav>...</nav>
    <main>
        <!-- Your existing application -->
    </main>
    
    <!-- NEW: AI Assistant widget -->
    <div id="ai-assistant">
        <button id="ai-chat-toggle">
            💬 Ask AI Assistant
        </button>
        <div id="ai-chat-window" style="display: none;">
            <!-- Chat interface -->
        </div>
    </div>
    
    <script src="/static/ai-assistant.js"></script>
</body>
</html>
```

**Assistant handles simple queries:**
```javascript
// ai-assistant.js
class AIAssistant {
    async handleUserMessage(message) {
        // Send to AI agent
        const response = await fetch('/api/ai/chat', {
            method: 'POST',
            body: JSON.stringify({
                message,
                conversation_id: this.conversationId
            })
        });
        
        const result = await response.json();
        this.displayResponse(result.response);
    }
}

// User can still use traditional UI or ask AI
```

### Step 4: Measure and Learn

**Track both modalities:**

```python
# Analytics
analytics.track("interaction", {
    "type": "traditional_ui",  # or "ai_assistant"
    "feature": "order_lookup",
    "success": True,
    "time_to_complete": 12.5  # seconds
})

# Compare metrics
traditional_metrics = {
    "avg_clicks_to_goal": 4.2,
    "avg_time_to_goal": 45.3,  # seconds
    "completion_rate": 0.78
}

ai_metrics = {
    "avg_turns_to_goal": 2.1,
    "avg_time_to_goal": 18.7,  # seconds
    "completion_rate": 0.84
}

# AI is 2.4x faster with higher completion!
```

### Phase 1 Success Criteria

```
✅ MCP infrastructure deployed
✅ AI assistant handling 10%+ of queries
✅ User satisfaction ≥4.0/5 for AI interactions
✅ No degradation of traditional UI performance
✅ Initial metrics showing AI value
```

---

## Phase 2: AI-First Features (Months 4-8)

### Goals

- Make AI the default for new features
- Migrate high-value existing features
- Begin model fine-tuning
- Scale AI coverage to 40%

### Step 1: New Features AI-Native First

**Policy: All new features start with conversational interface.**

```python
# Old approach: Build UI first
# 1. Design mockups
# 2. Build React components
# 3. Create REST endpoints
# 4. Wire everything together

# New approach: Build MCP tools first
@app.mcp_tool()
async def get_personalized_recommendations(
    category: Optional[str] = None,
    user_context: UserContext = Depends()
) -> List[Product]:
    """
    Get AI-powered product recommendations for user.
    
    This is a NEW feature - built AI-native first.
    """
    user_preferences = await get_user_preferences(user_context.user_id)
    purchase_history = await get_purchase_history(user_context.user_id)
    
    recommendations = recommendation_engine.generate(
        user_preferences,
        purchase_history,
        category_filter=category
    )
    
    return recommendations

# Traditional UI can be added later if needed
```

### Step 2: Migrate High-Value Endpoints

**Identify candidates:**

```python
# Analyze existing endpoints
endpoint_metrics = {
    "/api/orders": {
        "requests_per_day": 50000,
        "avg_response_time": 250,
        "user_satisfaction": 3.2,  # Low!
        "complexity": "medium"
    },
    "/api/products/search": {
        "requests_per_day": 120000,
        "avg_response_time": 180,
        "user_satisfaction": 3.8,
        "complexity": "high"  # Complex filters
    }
}

# Prioritize by: volume × (5 - satisfaction) × complexity
# High volume + low satisfaction + high complexity = best candidates
```

**Migration pattern:**

```python
# 1. Create MCP tool
@app.mcp_tool()
async def search_products_smart(
    query: str,
    user_context: UserContext = Depends()
) -> SearchResults:
    """
    Intelligent product search that understands natural language.
    
    Examples:
    - "laptops under $1000 with good battery"
    - "gifts for a 10-year-old who likes science"
    - "replacement for my broken iPhone charger"
    """
    # Enhanced with AI understanding
    intent = parse_search_intent(query)
    filters = extract_filters(query)
    
    # Use existing search logic
    results = product_search_engine.search(
        query=intent.refined_query,
        filters=filters,
        user_id=user_context.user_id
    )
    
    return results

# 2. Keep REST for backwards compatibility
@app.get("/api/products/search")
async def search_products_rest(q: str, filters: dict = None):
    """Legacy REST endpoint - maintained for compatibility"""
    return product_search_engine.search(q, filters)

# 3. Gradually shift traffic: 10% → 25% → 50% → 75% → 90%
```

### Step 3: Begin Enterprise Fine-Tuning

**Collect training data from Phase 1:**

```python
# Analyze Phase 1 conversations
production_logs = load_conversation_logs("phase1")

successful_interactions = [
    log for log in production_logs
    if log.user_satisfied and log.goal_completed
]

failed_interactions = [
    log for log in production_logs
    if not log.goal_completed or log.error_occurred
]

# Create training examples from successful patterns
training_examples = []
for interaction in successful_interactions:
    training_examples.append({
        "user_query": interaction.initial_query,
        "correct_tools": interaction.tools_called,
        "correct_parameters": interaction.parameters_used
    })

# Create corrections for failures
for interaction in failed_interactions:
    # Human reviews and creates correct version
    correction = human_review(interaction)
    training_examples.append({
        "user_query": interaction.initial_query,
        "correct_tools": correction.tools,
        "correct_parameters": correction.parameters,
        "mistake_made": interaction.error_type
    })
```

### Step 3.5: Choose Your Training Strategy

**Decision point: How will you improve model accuracy?**

Based on Chapter 11's three training paths, choose your strategy:

**Path 1: Prompt Engineering (Months 4-6)**
```python
# Use if:
# - Using Claude (no fine-tuning API)
# - Phase 1 had <10K users
# - Can achieve 90-95% accuracy with good prompts

class Phase2PromptEngineering:
    def optimize_system_prompt(self, training_examples):
        """Use training data to build better prompts"""

        # Select best examples for few-shot prompting
        few_shot_examples = self.select_representative_examples(
            training_examples,
            count=15  # Increased from Phase 1's 10
        )

        # Build enhanced system prompt
        system_prompt = self.build_system_prompt_with_examples(
            few_shot_examples,
            include_edge_cases=True
        )

        # Measure improvement
        accuracy = self.evaluate_on_test_set()
        print(f"Prompt engineering accuracy: {accuracy:.2%}")

        return system_prompt

# Cost: $3K-6K/mo (higher context, base model rates)
# Best for: Early stage, validating demand
```

**Path 2: Cloud Fine-Tuning (Months 4-8)**
```python
# Use if:
# - Using OpenAI
# - Phase 1 had 10K-100K users
# - Need 95-98% accuracy
# - Not privacy-sensitive

class Phase2CloudFineTuning:
    def fine_tune_on_openai(self, training_examples):
        """Upload training data to OpenAI for fine-tuning"""

        # Format for OpenAI
        formatted = self.format_for_openai(training_examples)

        # Upload and fine-tune
        job = openai.FineTuningJob.create(
            training_file=formatted.file_id,
            model="gpt-4-0125-preview",
            suffix="acme-phase2-v1"
        )

        # Wait for completion
        fine_tuned_model = self.wait_for_completion(job.id)

        # Evaluate
        accuracy = self.evaluate(fine_tuned_model)
        print(f"Fine-tuned accuracy: {accuracy:.2%}")

        return fine_tuned_model

# Cost: $80 (one-time) + $1.5K-3K/mo
# Best for: Growing user base, need better accuracy
```

**Path 3: Local LoRA Fine-Tuning (Months 6-12)**
```python
# Use if:
# - Phase 1 had >100K users
# - Privacy-sensitive data
# - Want cost optimization at scale
# - Have ML team

class Phase2LocalLoRA:
    def fine_tune_locally(self, training_examples):
        """Fine-tune Llama/Mistral with LoRA on your infrastructure"""

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-70B"
        )

        # Configure LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"]
        )

        # Train LoRA adapter
        adapter = self.train_lora_adapter(
            base_model,
            lora_config,
            training_examples
        )

        # Evaluate
        accuracy = self.evaluate_with_adapter(adapter)
        print(f"LoRA fine-tuned accuracy: {accuracy:.2%}")

        return adapter

# Cost: $30K-50K (one-time infrastructure) + $1K-2K/mo
# Best for: High volume, privacy needs, long-term investment
```

**Recommendation: Graduate through paths**

```
Phase 2 (Months 4-8):
├─ Start: Prompt engineering
├─ If accuracy < 92%: Move to cloud fine-tuning
└─ If volume > 100K users: Plan for local LoRA in Phase 3

Phase 3 (Months 9-18):
├─ High volume → Migrate to local LoRA
├─ Medium volume → Stay on cloud fine-tuning
└─ Low volume → Optimize prompts further
```

### Step 4: Expand Coverage

**Target distribution:**

```
Information Retrieval: → 70% AI-native
├─ Order status
├─ Product search
├─ Account info
└─ Transaction history

Simple Transactions: → 40% AI-native
├─ Add to cart
├─ Update preferences
└─ Save favorites

Complex Transactions: → 10% AI-native
├─ Checkout (with traditional fallback)
└─ Returns (with human oversight)

Admin Functions: → 0% AI-native
├─ Keep traditional UI
└─ Too complex, not high volume
```

### Step 5: Market and Drive Adoption

**Don't build it and assume they'll come—actively drive adoption:**

**Phase 2A: Internal Communication (Month 4)**

```
Email to all users:
---
Subject: Try our new AI assistant - get answers 2x faster

We've added an AI assistant to help you find orders, track shipments,
and answer questions instantly.

[Try it now button]

Early users are seeing:
• 60% faster task completion
• Fewer clicks to find what you need
• 4.3/5 satisfaction rating

Available now in the bottom-right corner of every page.
```

**Phase 2B: In-App Prompts (Month 4-5)**

```javascript
// Contextual hints
class AdoptionDriver {
    showContextualHint(user_action) {
        if (user_action == "clicked_orders_5_times") {
            showTooltip({
                message: "💡 Tip: Try asking 'Show my recent orders' in the AI assistant",
                position: "ai-assistant-button",
                dismissable: true
            });
        }

        if (user_action == "searched_products_with_complex_filters") {
            showTooltip({
                message: "💡 Try: 'Show me laptops under $1000 with good battery life'",
                position: "ai-assistant-button"
            });
        }
    }
}
```

**Phase 2C: Power User Program (Month 5-6)**

```python
# Identify power users
power_users = identify_users_where(
    monthly_sessions > 20,
    tenure_months > 6,
    satisfied_with_product=True
)

# Invite to beta program
for user in power_users:
    send_email(
        user=user,
        template="ai_beta_invite",
        benefits=[
            "Early access to new AI features",
            "Direct feedback channel to product team",
            "Influence future development"
        ]
    )

# Collect feedback
beta_feedback = collect_feedback_from(power_users)

# Iterate based on feedback
prioritize_features(beta_feedback.top_requests)
```

**Phase 2D: Success Metrics Dashboard (Month 6-8)**

```python
# Track adoption metrics
adoption_metrics = {
    "ai_assistant_usage": {
        "weekly_active_users": 12500,  # 25% of user base
        "growth_rate": "+15% week-over-week",
        "avg_interactions_per_user": 3.2,
        "completion_rate": 0.84
    },

    "user_satisfaction": {
        "ai_interactions": 4.4,  # out of 5
        "traditional_ui": 3.8,
        "net_promoter_score": +42  # up from +28
    },

    "business_impact": {
        "support_ticket_reduction": -35,  # percent
        "time_to_resolution": -60,  # percent faster
        "task_completion_rate": +22  # percent
    }
}

# Share wins publicly
announce_to_company({
    "title": "AI Assistant Update: 25% Adoption, 35% Fewer Support Tickets",
    "metrics": adoption_metrics,
    "next_steps": "Expanding to inventory and shipping features in Q3"
})
```

**Phase 2E: Gradual Rollout Strategy**

```
Week 1-2:   Internal team only (dogfooding)
Week 3-4:   Beta program (power users)
Week 5-6:   10% of users (A/B test)
Week 7-8:   25% of users
Week 9-10:  50% of users
Week 11-12: 75% of users
Week 13+:   100% availability

At each stage:
✓ Monitor metrics
✓ Collect feedback
✓ Fix issues before next expansion
✓ Communicate wins to stakeholders
```

**Marketing Best Practices:**

```yaml
Do:
  - Highlight time savings (quantified)
  - Show before/after comparisons
  - Use real user testimonials
  - Make it easy to try (one click)
  - Provide contextual hints
  - Celebrate wins publicly

Don't:
  - Force users to AI assistant
  - Remove traditional UI too soon
  - Overpromise capabilities
  - Ignore negative feedback
  - Launch without metrics
  - Assume adoption will be organic
```

### Phase 2 Success Criteria

```
✅ 40% of user interactions via AI
✅ Fine-tuned model deployed
✅ Scenario evaluation suite established
✅ AI satisfaction scores ≥4.2/5
✅ Cost-per-interaction decreasing
✅ Support ticket volume down 20%
```

---

## Phase 3: AI-Native Core (Months 9-18)

### Goals

- AI becomes primary interface
- Traditional UI for edge cases
- Full scenario coverage
- Continuous improvement loop

### Step 1: Invert the Default

**Change user entry point:**

```
Before (Phase 1-2):
User lands on traditional UI → Can optionally use AI assistant

After (Phase 3):
User lands in conversational interface → Can drop to traditional UI for complex tasks
```

**Implementation:**

```html
<!-- app-v3.html -->
<body>
    <!-- Primary interface: Conversational -->
    <div id="conversation-interface">
        <h1>What can I help you with?</h1>
        <input 
            type="text" 
            placeholder="Try: 'Show my orders' or 'Find a gift under $50'"
            id="ai-input"
            autofocus
        />
    </div>
    
    <!-- Secondary: Traditional UI -->
    <button id="use-traditional-ui">
        Switch to classic view
    </button>
    
    <div id="traditional-ui" style="display: none;">
        <!-- Full traditional app available when needed -->
    </div>
</body>
```

### Step 2: Handle Edge Cases

**Identify when traditional UI is better:**

```python
class InterfaceRouter:
    def recommend_interface(self, user_query: str, context: dict) -> str:
        """Decide which interface is best for this task"""
        
        # Complex multi-step workflows → traditional
        if self.is_complex_workflow(user_query):
            return "traditional"
        
        # Requires precise visual manipulation → traditional
        if self.requires_visual_precision(user_query):
            return "traditional"
        
        # User explicitly requested traditional UI
        if "show me the form" in user_query.lower():
            return "traditional"
        
        # Power user features → traditional
        if context.get("power_user") and self.is_advanced_feature(user_query):
            return "traditional"
        
        # Everything else → conversational
        return "conversational"

# Example usage
if interface_router.recommend_interface(user_query, context) == "traditional":
    response = {
        "message": "This task is easier with our classic interface. I'll open it for you.",
        "action": "open_traditional_ui",
        "prefill": extracted_data  # Pre-populate form
    }
```

### Step 3: Complete Scenario Coverage

**Comprehensive test suite:**

```
Scenario Categories:
├─ Core User Journeys (50 scenarios)
│  ├─ Account creation & onboarding
│  ├─ Product discovery & purchase
│  ├─ Order tracking & management
│  └─ Returns & refunds
│
├─ Edge Cases (30 scenarios)
│  ├─ Out of stock handling
│  ├─ Payment failures
│  ├─ Address validation errors
│  └─ Inventory synchronization issues
│
├─ Security Scenarios (20 scenarios)
│  ├─ Permission boundary tests
│  ├─ Data access controls
│  ├─ High-risk action approvals
│  └─ Fraud detection
│
└─ Error Recovery (20 scenarios)
   ├─ Service timeouts
   ├─ Invalid user input
   ├─ Conflicting state
   └─ Third-party API failures

Total: 120 scenarios
Target: >95% pass rate
```

### Step 4: Continuous Improvement Loop

**Production feedback → Training → Deployment:**

```python
# Daily pipeline
class ContinuousImprovement:
    async def daily_cycle(self):
        # 1. Collect production failures
        failures = await self.collect_failures_last_24h()
        
        # 2. Human review creates corrections
        corrections = await self.human_review_queue(failures)
        
        # 3. Add to training dataset
        await self.update_training_data(corrections)
        
        # 4. Weekly: Re-fine-tune model
        if datetime.now().weekday() == 0:  # Monday
            await self.retrain_model()
        
        # 5. Evaluate new model
        eval_results = await self.run_evaluation_suite()
        
        # 6. Deploy if improved
        if eval_results.better_than_current():
            await self.deploy_model(eval_results.model_id)
            
        # 7. Monitor for regression
        await self.monitor_production_metrics()

# Runs automatically
scheduler.add_job(
    continuous_improvement.daily_cycle,
    trigger='cron',
    hour=2  # 2 AM daily
)
```

### Phase 3 Success Criteria

```
✅ 80% of interactions AI-native
✅ Traditional UI < 20% usage
✅ Scenario pass rate >95%
✅ User satisfaction >4.5/5
✅ Support cost reduced 50%
✅ Continuous improvement pipeline operational
✅ Model retraining weekly
✅ Zero-downtime model deployment
```

---

## Decision Trees for Migration

### Should I Migrate This Feature?

```
Is feature high-volume (>1000 requests/day)?
├─ YES → Is it read-only?
│  ├─ YES → Migrate in Phase 1 ✓
│  └─ NO → Is it simple transaction?
│     ├─ YES → Migrate in Phase 2 ✓
│     └─ NO → Is it critical?
│        ├─ YES → Phase 3 with extra testing ⚠️
│        └─ NO → Migrate in Phase 2-3 ✓
│
└─ NO → Is user satisfaction low (<3.5)?
   ├─ YES → Could AI help?
   │  ├─ YES → Migrate in Phase 2 ✓
   │  └─ NO → Keep traditional UI ✗
   │
   └─ NO → Keep traditional UI ✗
```

### Build New Feature: Where to Start?

```
New feature request received
│
├─ Is it conversational by nature?
│  └─ YES → Build MCP-native first ✓
│
├─ Is it primarily information retrieval?
│  └─ YES → Build MCP-native first ✓
│
├─ Does it require complex visual interaction?
│  └─ YES → Build traditional UI first
│     (Consider: Can AI assist/augment?)
│
└─ Is it an admin/power-user feature?
   └─ YES → Build traditional UI
      (AI optional)
```

---

## Real-World Migration Example

### Case: E-Commerce Order Management

**Starting point:**
- 50,000 daily "Where's my order?" inquiries
- Traditional: Navigate to Orders → Find order → Click tracking
- Average time: 45 seconds
- User satisfaction: 3.2/5
- Support tickets: 500/day for order tracking

**Phase 1: AI Assistant for Order Lookup**

```python
# Month 1: Deploy AI widget
@app.mcp_tool()
async def track_my_order(
    order_identifier: Optional[str] = None,
    user_context: UserContext = Depends()
) -> OrderTracking:
    """Find and track user's order"""
    
    if order_identifier:
        # User specified order
        order = find_order(order_identifier, user_context.user_id)
    else:
        # Get most recent order
        order = get_latest_order(user_context.user_id)
    
    tracking = get_tracking_info(order.tracking_number)
    return tracking

# Results after Month 1:
# - 8% of users tried AI assistant
# - Those who tried: 4.1/5 satisfaction
# - Avg time: 18 seconds (2.5x faster)
# - Support tickets: 475/day (-5%)
```

**Phase 2: Expand and Fine-Tune**

```python
# Month 4: Add more order capabilities
@app.mcp_tool()
async def cancel_my_order(order_id: str):
    """Cancel an order if still eligible"""
    pass

@app.mcp_tool()
async def return_item(order_id: str, item_id: str, reason: str):
    """Initiate return for an item"""
    pass

# Fine-tune on 3 months of conversation data
# Add edge case handling

# Results after Month 6:
# - 35% of order interactions via AI
# - 4.4/5 satisfaction
# - Support tickets: 300/day (-40%)
# - Cost per interaction: -60%
```

**Phase 3: AI-First Order Management**

```python
# Month 12: Conversational becomes default
# User lands on: "What would you like to do with your orders?"

# Results after Month 15:
# - 78% of order interactions via AI
# - 4.6/5 satisfaction
# - Support tickets: 150/day (-70%)
# - Time to resolution: 15 seconds average
# - NPS +25 points
```

---

## Common Pitfalls and Solutions

### ❌ Pitfall 1: "Big Bang" Migration

**Mistake:** Try to migrate entire application at once

**Solution:** Incremental, feature-by-feature migration with metrics at each step

### ❌ Pitfall 2: Removing Traditional UI Too Soon

**Mistake:** Force all users to AI before ready

**Solution:** Keep traditional UI available. Let users choose. Monitor adoption naturally.

### ❌ Pitfall 3: Skipping Fine-Tuning

**Mistake:** Deploy with generic model, accept 70% accuracy

**Solution:** Fine-tune on enterprise data before Phase 2. Accuracy matters.

### ❌ Pitfall 4: No Scenario Testing

**Mistake:** Test individual tools, ignore end-to-end journeys

**Solution:** Scenario evaluation from Phase 2 onward. Gate deployments on pass rates.

### ❌ Pitfall 5: Ignoring Cost

**Mistake:** Scale AI interactions without monitoring LLM costs

**Solution:** Track cost-per-interaction. Optimize prompt engineering. Cache common queries.

---

## Summary

**Migration Success Formula:**

```
Start Small (Phase 1)
  → Prove value with low-risk features
  
Scale Gradually (Phase 2)
  → Expand coverage, begin fine-tuning
  
Transform Fully (Phase 3)
  → AI-native core with traditional fallback

Throughout:
  ✓ Measure everything
  ✓ Keep traditional UI available
  ✓ Fine-tune continuously
  ✓ Test scenarios rigorously
  ✓ Listen to users
```

---

## Key Takeaways

✓ **Incremental migration over 3 phases** - Phase 1 (Months 1-3): Hybrid foundation with MCP alongside REST, AI assistant for 10% of queries; Phase 2 (Months 4-8): AI-first for new features, 40% AI-native; Phase 3 (Months 9-18): AI-native core with 80% coverage, traditional UI for edge cases

✓ **Start with low-risk, high-volume features** - Prioritize read-only operations (order status, product search, account info) for Phase 1; avoid payment processing, data deletion, and complex transactions initially; success builds confidence for later phases

✓ **Never remove traditional UI entirely** - Keep traditional interface available throughout all phases; let users choose their preferred interaction mode; some tasks (complex admin, visual design) genuinely benefit from traditional UI

✓ **Training strategy evolves with scale** - Graduate through training paths: prompt engineering (Phase 2, <10K users, $3K-6K/mo) → cloud fine-tuning (10K-100K users, $1.5K-3K/mo) → local LoRA (>100K users, privacy needs, $1K-2K/mo); accuracy requirements increase from 90% to 98%

✓ **Scenario testing gates deployments** - Build comprehensive scenario suite (120+ end-to-end tests) by Phase 3; target >95% pass rate before releases; continuous improvement loop: production failures → human review → training data → weekly retraining → evaluation → deployment

✓ **Active adoption marketing required** - Don't assume users will discover AI features; use in-app prompts, power user programs, contextual hints, and success metrics dashboards; track adoption, iterate based on feedback, communicate wins publicly

---

**[← Previous: Chapter 11 - Training](chapter-11-training) | [Next: Chapter 13 - Frameworks →](chapter-13-frameworks)**