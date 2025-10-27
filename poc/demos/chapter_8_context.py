"""
Chapter 8 Demo: Context Management for Multi-Turn Conversations
===============================================================

Demonstrates how context management enables:
1. Multi-turn conversations with memory
2. Smart context window management
3. Automatic context cleanup
4. Context-aware decision making
5. State persistence across turns

This demo shows how AI systems remember and use conversation context.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List


class Chapter8Demo:
    """Chapter 8: Context Management - Interactive Demo"""

    def __init__(self):
        self.demo_title = "Chapter 8: Context Management for Multi-Turn Conversations"
        self.demo_description = (
            "Demonstrates how context management enables intelligent, "
            "stateful AI conversations without requiring full conversation history."
        )

    async def demo_stateless_vs_stateful(self):
        """Demo 1: Stateless vs Stateful systems"""
        print("\n" + "="*70)
        print("DEMO 1: Stateless vs Stateful AI Systems")
        print("="*70)
        print(f"\nComparison: Two different approaches to multi-turn conversation")
        print("-" * 70)

        print(f"""
Stateless System (No Context):

Turn 1: User: "What laptops do you have?"
        AI: Search products... "We have Dell, HP, Lenovo..."

Turn 2: User: "Which is cheapest?"
        AI: ERROR - No context about what "which"?
           → System asks: "Which brand?" (frustrating!)

Turn 3: User: "The laptops you mentioned earlier"
        AI: ERROR - Earlier conversation not stored

Problem:
├─ Each request is independent
├─ User must repeat context
├─ Conversation breaks down quickly
├─ Not suitable for multi-turn interactions


Stateful System with Context (This Demo):

Turn 1: User: "What laptops do you have?"
        AI: Search products... "We have Dell, HP, Lenovo..."
        ✓ Context stored: {products: [product_list]}

Turn 2: User: "Which is cheapest?"
        AI: Looks at context → Knows about laptops
           "The Lenovo at $599 is cheapest"
        ✓ Context updated: {selected_product: Lenovo}

Turn 3: User: "Tell me more about it"
        AI: Looks at context → Knows which product
           "Lenovo specs: Intel i5, 8GB RAM..."
        ✓ Conversation flows naturally

Benefits:
├─ Context flows through conversation
├─ Users don't repeat themselves
├─ System understands references ("it", "that")
├─ Natural, coherent conversations
└─ Better user experience
        """)

    async def demo_context_lifecycle(self):
        """Demo 2: Context lifecycle management"""
        print("\n" + "="*70)
        print("DEMO 2: Context Lifecycle")
        print("="*70)
        print(f"\nScenario: How context moves through the system")
        print("-" * 70)

        print(f"""
Context Lifecycle:

1. INITIALIZATION
   ├─ Conversation starts
   ├─ Context created: empty
   ├─ Timestamp: 2025-02-10T14:30:00Z
   └─ Storage: In-memory cache

2. TURN 1: User asks about products
   User: "Show me gaming laptops under $2000"
   
   ├─ Intent: Browse products
   ├─ Service: ProductService.SearchProducts
   ├─ Result: {products: [...], total: 15}
   ├─ Context stored:
   │  {
   │    "last_intent": "browse_products",
   │    "last_service": "product_service",
   │    "products": [...],
   │    "search_query": "gaming laptops",
   │    "price_range": {min: 0, max: 2000},
   │    "result_count": 15,
   │    "timestamp": 14:30:00
   │  }
   └─ Size: ~5KB

3. TURN 2: User refines search
   User: "Sort by price, cheapest first"
   
   ├─ Intent: Refine search (sorting)
   ├─ Context used: Know previous search was laptops
   ├─ New service call: Sort {products} by price
   ├─ Result: {products: [...sorted...]}
   ├─ Context updated:
   │  {
   │    "last_intent": "sort_results",
   │    "last_service": "product_service",
   │    "products": [...sorted...],
   │    "sort_order": "price_asc",
   │    "previous_intents": ["browse_products", "sort_results"],
   │    "timestamp": 14:30:45
   │  }
   └─ Size: ~5KB

4. TURN 3: User picks one product
   User: "Tell me about the cheapest one"
   
   ├─ Intent: Get details
   ├─ Context used: Know cheapest product from sorted list
   ├─ Service call: ProductService.GetProductDetails
   ├─ Result: {product_details: {...}}
   ├─ Context updated:
   │  {
   │    "selected_product": {product info},
   │    "product_id": "prod_12345",
   │    "previous_searches": [...],
   │    "timestamp": 14:31:00
   │  }
   └─ Size: ~8KB

5. TURN 4: User adds to cart
   User: "Add it to my cart"
   
   ├─ Intent: Add to cart
   ├─ Context used: Know product_id from selection
   ├─ Service call: CartService.AddToCart
   ├─ Result: {cart_id: "cart_123", items: 1}
   ├─ Context updated:
   │  {
   │    "cart_id": "cart_123",
   │    "selected_product": {...},
   │    "user_journey": [
   │      "browsed products",
   │      "sorted by price", 
   │      "viewed details",
   │      "added to cart"
   │    ]
   │  }
   └─ Size: ~10KB

6. CLEANUP (Automatic)
   
   Trigger: Conversation idle for 30 minutes
   OR Manual: User logs out
   
   ├─ Context persisted to database (for history)
   ├─ In-memory cache cleared
   ├─ Tokens released
   ├─ Audit trail recorded
   └─ Ready for next session

Context Growth Considerations:

├─ Size grows with each turn
├─ Risk: Context becomes too large
├─ Solution: Smart context pruning
│  ├─ Remove old/irrelevant context
│  ├─ Keep recent and important info
│  ├─ Compress large arrays
│  └─ Automatic every 10 turns
└─ Result: Bounded context size
        """)

    async def demo_context_window_management(self):
        """Demo 3: Context window management"""
        print("\n" + "="*70)
        print("DEMO 3: Context Window Management")
        print("="*70)
        print(f"\nScenario: Keeping context within manageable size")
        print("-" * 70)

        print(f"""
The Challenge:

LLMs have limited context windows:
├─ GPT-3.5: ~4,000 tokens
├─ GPT-4: ~8,000 tokens
├─ Claude: ~100,000 tokens (more generous)
└─ Each token costs money

Traditional approach (Bad):
  Include full conversation history
  → Uses lots of tokens
  → Costs increase with conversation length
  → Model performance degrades with long context

AI-Native Approach (Good):
  Store full history, but send only relevant context to LLM

Smart Context Selection Algorithm:

1. RECENT (Always include)
   ├─ Last 5 turns (complete)
   ├─ Ensures continuity
   └─ Size: ~2KB

2. RELEVANT (Based on current intent)
   ├─ Previous searches for same product? Include.
   ├─ Previous price filters? Include.
   ├─ Old conversation about weather? Skip.
   ├─ Size: ~1-3KB

3. IMPORTANT (Based on markers)
   ├─ User explicitly said "remember this"
   ├─ Decision points (cart added, purchase made)
   ├─ Errors or special events
   ├─ Size: ~0-2KB

4. SUMMARY (If history is long)
   ├─ Original: 50 turns = 25KB
   ├─ Summarize: "User browsed laptops, picked one, added to cart"
   ├─ Compression: 25KB → 0.5KB
   └─ Include in context

Total Context Sent to LLM:
├─ Base: System prompt = 0.5KB
├─ Recent turns: ~2KB
├─ Relevant context: ~2KB
├─ Important events: ~1KB
├─ Summary: ~0.5KB
└─ Total: ~6KB (manageable)

Example: 100-Turn Conversation

Turn 100 Request:
  Context stored: 100 turns × 0.2KB = 20KB
  
  What to send to LLM:
  ├─ Turns 95-100: 6 turns (recent) = 1.2KB
  ├─ Similar searches: If searching again = 0.5KB
  ├─ Cart items: Previous additions = 0.3KB
  ├─ User preferences: Remembered filters = 0.2KB
  └─ Summary: Full journey = 0.3KB
  
  Total sent: ~2.5KB (vs 20KB!)
  Token savings: 87.5%
  Cost savings: Same order!

Benefits:
├─ Low latency (less to process)
├─ Lower costs (fewer tokens)
├─ Better performance (focused context)
├─ Long conversations become feasible
└─ All history preserved for audit trail
        """)

    async def demo_context_aware_decisions(self):
        """Demo 4: Context-aware decision making"""
        print("\n" + "="*70)
        print("DEMO 4: Context-Aware Decision Making")
        print("="*70)
        print(f"\nScenario: Decisions change based on accumulated context")
        print("-" * 70)

        print(f"""
Scenario A: New User, First Time Buying

Turn 1: "Show me laptops"
  Context: {total_previous_purchases: 0}
  Decision: Show introductory/helpful information
  Response: "Here are our top-rated laptops. Confused about specs?
             Let me help! What will you use it for?"

Turn 2: "Gaming"
  Context: {interest: gaming, purchase_history: empty}
  Decision: Suggest gaming-specific features
  Response: "Gaming laptops need good GPU and refresh rate.
             These 3 have excellent gaming specs under $2000."


Scenario B: Loyal Customer, Many Purchases

Turn 1: "Show me laptops"
  Context: {total_purchases: 47, lifetime_value: $15000}
  Decision: Show premium options first
  Response: "Welcome back! Based on your history,
             here are premium gaming laptops on sale today."

Turn 2: "Gaming"
  Context: {interest: gaming, last_gaming_purchase: 3mo_ago, 
             satisfaction: very_high}
  Decision: Recommend upgrades to existing model
  Response: "Your current laptop is an XPS 15.
             The new model has 40% better GPU. Want to trade in?"


Scenario C: Someone with Low Budget History

Turn 1: "Show me laptops"
  Context: {budget_preference: $300-600, 
             previous_complaints: expensive}
  Decision: Show budget-friendly options
  Response: "Here are quality laptops in your usual price range."

Turn 2: "Cheapest one"
  Context: {budget_constraint: clear}
  Decision: Filter for absolute cheapest
  Response: "This Lenovo at $399 is our absolute cheapest.
             Good specs for the price."


Scenario D: Indecisive Visitor (Multiple Changes)

Turn 1-3: Changes mind about product multiple times
  Context: {decision_changes: 3, time_spent: 15_minutes}
  Decision: Maybe user needs help deciding
  Response: "You're checking out a few options.
             Want me to compare these 2 side-by-side?"

Result:
  Decision tree changes based on user HISTORY and BEHAVIOR
  Not just current request, but ACCUMULATED CONTEXT
        """)

    async def demo_multi_step_orchestration_with_context(self):
        """Demo 5: Multi-step orchestration using context"""
        print("\n" + "="*70)
        print("DEMO 5: Multi-Step Orchestration with Context")
        print("="*70)
        print(f"\nScenario: Complex multi-service workflow coordinated via context")
        print("-" * 70)

        print(f"""
E-Commerce Checkout Flow Using Context:

Step 1: Browse & Select
  User: "I want a laptop with 16GB RAM under $1500"
  
  Context:
  ├─ search_params: {ram: 16GB, price_max: 1500}
  ├─ results: [laptop_1, laptop_2, laptop_3]
  └─ status: browsing

Step 2: Add to Cart
  User: "Add the Dell to my cart"
  
  Context updated:
  ├─ selected_product: laptop_1 (Dell)
  ├─ cart_items: [laptop_1]
  └─ status: cart_updated

Step 3: Review Cart
  User: "Show me my cart"
  
  Context query:
  ├─ Retrieve: cart_items, prices, discounts
  ├─ Calculate: Subtotal = $1,299, Tax = $103.92
  └─ Total = $1,402.92

Step 4: Apply Discount
  User: "I have a coupon code"
  
  Context action:
  ├─ Store: coupon_code = "SAVE20"
  ├─ Validation: Orchestrator validates coupon
  ├─ Calculation: 20% off = -$259.80
  └─ New total: $1,143.12

Step 5: Checkout
  User: "Let's checkout"
  
  Context orchestration:
  ├─ Service 1: OrderService
  │  └─ Create order with context data
  ├─ Service 2: PaymentService
  │  └─ Process payment using stored card
  ├─ Service 3: ShippingService
  │  └─ Schedule delivery
  └─ Result: Order confirmation

Step 6: Follow-up
  User (next day): "Where's my order?"
  
  Context retrieval:
  ├─ Session context: Expired
  ├─ Database query: Retrieve last_order_context
  ├─ Result: Order #12345, Tracking info
  └─ Response: "Your order arrives tomorrow"

Context Persistence:

In-Memory (Current session):
  ├─ Duration: User logged in
  ├─ Size: ~50KB
  ├─ Access: Fast (< 1ms)
  └─ Expires: On logout

Database (Historical):
  ├─ Duration: Permanent
  ├─ Size: Unlimited
  ├─ Access: Slower (10-100ms)
  └─ Use case: Future reference, audit

Result:
  Seamless multi-step workflow
  Context flows across 3+ services
  User context preserved even across sessions
        """)

    async def demo_conversation_example(self):
        """Demo 6: Complete conversation example"""
        print("\n" + "="*70)
        print("DEMO 6: Complete Conversation Example")
        print("="*70)
        print(f"\nScenario: Full multi-turn conversation with context")
        print("-" * 70)

        conversation = [
            {
                "turn": 1,
                "user": "I'm looking for a gaming laptop",
                "context_before": "{}",
                "ai_action": "SearchProducts(category=gaming_laptop)",
                "result": "Found 23 gaming laptops",
                "context_after": "{search_category: gaming_laptop, results: 23}"
            },
            {
                "turn": 2,
                "user": "Under $1500 please",
                "context_before": "{search_category: gaming_laptop, results: 23}",
                "ai_action": "Filter(category=gaming, price_max=1500)",
                "result": "12 matches within budget",
                "context_after": "{search_category: gaming_laptop, price_max: 1500, results: 12}"
            },
            {
                "turn": 3,
                "user": "What about the most expensive one?",
                "context_before": "{search_category: gaming_laptop, price_max: 1500, results: 12}",
                "ai_action": "GetProduct(id=results[0].id) // Knows to use filtered results",
                "result": "ASUS ROG $1,499",
                "context_after": "{selected_product: ASUS ROG, price: 1499}"
            },
            {
                "turn": 4,
                "user": "How much RAM does it have?",
                "context_before": "{selected_product: ASUS ROG, price: 1499}",
                "ai_action": "Lookup context - already have details",
                "result": "32GB DDR5",
                "context_after": "{selected_product: ASUS ROG, specs_reviewed: true}"
            },
            {
                "turn": 5,
                "user": "Add it to my cart",
                "context_before": "{selected_product: ASUS ROG, specs_reviewed: true}",
                "ai_action": "CartService.Add(product_id=ASUS_ROG_ID)",
                "result": "Added to cart (1 item)",
                "context_after": "{cart_items: [ASUS ROG], ready_to_checkout: true}"
            },
            {
                "turn": 6,
                "user": "Checkout",
                "context_before": "{cart_items: [ASUS ROG], ready_to_checkout: true}",
                "ai_action": "OrderService.Create(cart_context)",
                "result": "Order #ORD-456789",
                "context_after": "{order_id: ORD-456789, status: completed}"
            }
        ]

        print(f"\n{'Turn':<6} {'User':<30} {'AI Action':<40} {'Result':<20}")
        print(f"{'-'*96}")
        
        for turn_data in conversation:
            print(f"\nTurn {turn_data['turn']}:")
            print(f"  User: {turn_data['user']}")
            print(f"  AI Action: {turn_data['ai_action']}")
            print(f"  Result: {turn_data['result']}")
            print(f"  Context: {turn_data['context_after']}")

        print(f"\n{'─'*70}")
        print(f"Key Points:")
        print(f"  • Each turn uses previous context")
        print(f"  • User doesn't repeat information")
        print(f"  • System understands references")
        print(f"  • Natural conversation flow")
        print(f"  • Orchestrator coordinates multiple services")

    async def run(self):
        """Run all Chapter 8 demos"""
        print(f"\n{'█'*70}")
        print(f"{'█'*70}")
        print(f"█ {self.demo_title.center(68)} █")
        print(f"█ {self.demo_description.center(68)} █")
        print(f"{'█'*70}")
        print(f"{'█'*70}")

        try:
            await self.demo_stateless_vs_stateful()
            await self.demo_context_lifecycle()
            await self.demo_context_window_management()
            await self.demo_context_aware_decisions()
            await self.demo_multi_step_orchestration_with_context()
            await self.demo_conversation_example()

            print(f"\n{'='*70}")
            print(f"KEY TAKEAWAYS - Chapter 8: Context Management")
            print(f"{'='*70}")
            print(f"""
1. Stateful Systems Enable Real Conversations
   → Context flows from turn to turn
   → Users don't repeat themselves
   → System understands references
   → Natural, coherent interactions

2. Context Lifecycle is Managed Automatically
   → Initialized at conversation start
   → Updated after each action
   → Cleaned up on timeout/logout
   → Available for future reference

3. Context Window Management is Critical
   → LLMs have limited context (tokens = money)
   → Smart pruning keeps context focused
   → Full history preserved in database
   → Only relevant context sent to LLM

4. Context Enables Intelligent Decisions
   → Decision-making adapts to history
   → New users get different treatment than loyal customers
   → Recommendations based on context
   → Personalization emerges naturally

5. Multi-Step Orchestration Uses Context
   → Context coordinates across services
   → Each service contributes to context
   → Context persists across service calls
   → Enables complex workflows

6. Context is State, Not History
   → State = Current decision-relevant info
   → History = Full audit trail (separate)
   → State is compact and focused
   → History is detailed and permanent

7. Persistence Strategies Matter
   → In-memory: Fast, current session
   → Database: Durable, available later
   → Both needed for complete system

The core insight: Context management separates AI-native systems 
from stateless APIs. It enables the intelligence and conversational
feel users expect from AI applications.

Next: Chapter 10 explores testing and validation of these systems!
            """)

        except Exception as e:
            print(f"\n✗ Error during demo: {e}")
            import traceback
            traceback.print_exc()


async def main():
    demo = Chapter8Demo()
    await demo.run()


if __name__ == "__main__":
    asyncio.run(main())
