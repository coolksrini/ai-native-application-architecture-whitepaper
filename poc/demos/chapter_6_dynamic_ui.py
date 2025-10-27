"""
Chapter 6 Demo: The Death of Traditional UI
============================================

Demonstrates how AI-native architecture enables:
1. Intent-driven rendering (not developer-decided)
2. Dynamic component selection based on query intent
3. Context-aware data presentation
4. Multi-format responses for same data

This demo shows how the same data (orders, products, etc.) is
presented differently based on what the user is actually trying to do.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List


class Chapter6Demo:
    """Chapter 6: The Death of Traditional UI - Interactive Demo"""

    def __init__(self):
        self.demo_title = "Chapter 6: The Death of Traditional UI"
        self.demo_description = (
            "Demonstrates how AI-native UIs adapt rendering based on user intent "
            "rather than developer decisions."
        )

    async def demo_traditional_ui_limitations(self):
        """Demo 1: Traditional UI - Same presentation for all queries"""
        print("\n" + "="*70)
        print("DEMO 1: Traditional UI - The Problem")
        print("="*70)
        print(f"\nScenario: User views their orders in a traditional app")
        print("-" * 70)

        sample_orders = [
            {"id": "ORD-001", "date": "2025-01-15", "total": 127.50, "status": "Delivered"},
            {"id": "ORD-002", "date": "2025-01-20", "total": 89.99, "status": "Shipped"},
            {"id": "ORD-003", "date": "2025-02-01", "total": 245.00, "status": "Processing"},
            {"id": "ORD-004", "date": "2025-02-05", "total": 56.75, "status": "Delivered"},
        ]

        # Query 1
        print(f"\nQuery 1: 'Show me my orders'")
        print(f"Traditional Response: Data table")
        print(f"{'─'*70}")
        print(f"{'Order ID':<12} {'Date':<12} {'Total':<10} {'Status':<12}")
        print(f"{'-'*12} {'-'*12} {'-'*10} {'-'*12}")
        for order in sample_orders:
            print(f"{order['id']:<12} {order['date']:<12} ${order['total']:<9.2f} {order['status']:<12}")

        # Query 2
        print(f"\n\nQuery 2: 'Which order arrives soonest?'")
        print(f"Traditional Response: SAME DATA TABLE")
        print(f"{'─'*70}")
        print(f"{'Order ID':<12} {'Date':<12} {'Total':<10} {'Status':<12}")
        print(f"{'-'*12} {'-'*12} {'-'*10} {'-'*12}")
        for order in sample_orders:
            print(f"{order['id']:<12} {order['date']:<12} ${order['total']:<9.2f} {order['status']:<12}")
        print(f"\nProblem: User must scan table to find soonest delivery. Not ideal.")

        # Query 3
        print(f"\n\nQuery 3: 'How much did I spend this month?'")
        print(f"Traditional Response: SAME DATA TABLE AGAIN")
        print(f"{'─'*70}")
        print(f"{'Order ID':<12} {'Date':<12} {'Total':<10} {'Status':<12}")
        print(f"{'-'*12} {'-'*12} {'-'*10} {'-'*12}")
        for order in sample_orders:
            print(f"{order['id']:<12} {order['date']:<12} ${order['total']:<9.2f} {order['status']:<12}")
        print(f"\nProblem: User must manually add up totals from table. Frustrating!")

        print(f"\n{'─'*70}")
        print(f"Core Issue: UI doesn't adapt to user intent")
        print(f"  → Developer decides: 'Orders are a table'")
        print(f"  → User stuck with table for all use cases")
        print(f"  → Same presentation regardless of intent")

    async def demo_ai_native_ui_rendering(self):
        """Demo 2: AI-Native UI - Intent-driven rendering"""
        print("\n" + "="*70)
        print("DEMO 2: AI-Native UI - Intent-Driven Rendering")
        print("="*70)
        print(f"\nScenario: SAME DATA, but different presentations based on intent")
        print("-" * 70)

        # Query 1: Intent = Browse
        print(f"\nQuery 1: 'Show me my orders'")
        print(f"Intent: BROWSE | Confidence: 95%")
        print(f"Rendering: List View")
        print(f"{'─'*70}")
        print(f"📋 Your Recent Orders\n")
        print(f"  • ORD-001 (Jan 15)  - $127.50  ✓ Delivered")
        print(f"  • ORD-002 (Jan 20)  - $89.99   ▸ Shipped")
        print(f"  • ORD-003 (Feb 01)  - $245.00  ⏳ Processing")
        print(f"  • ORD-004 (Feb 05)  - $56.75   ✓ Delivered")

        # Query 2: Intent = Find specific
        print(f"\n\nQuery 2: 'Which order arrives soonest?'")
        print(f"Intent: FIND_SPECIFIC | Confidence: 95%")
        print(f"Rendering: Highlighted Card")
        print(f"{'─'*70}")
        print(f"""
┌───────────────────────────────────────────────────────────────┐
│                                                               │
│  📦 Arriving Soonest                                          │
│                                                               │
│  Order: ORD-002                                               │
│  Arriving: 5 days (Feb 10)                                    │
│  Status: In Transit                                           │
│  Value: $89.99                                                │
│  [Track Shipment] [View Details]                              │
│                                                               │
└───────────────────────────────────────────────────────────────┘
        """)

        # Query 3: Intent = Analyze spending
        print(f"\nQuery 3: 'How much did I spend this month?'")
        print(f"Intent: ANALYZE_SPENDING | Confidence: 95%")
        print(f"Rendering: Summary with Visualization")
        print(f"{'─'*70}")
        print(f"""
💰 February Spending Summary

Total Spent: $302.00

Breakdown:
  ORD-003 (Processing)  $245.00  ████████████████████░  81%
  ORD-004 (Delivered)   $56.75   ████░                  19%

Trends:
  Feb: $302.00
  Jan: $217.49
  Change: +39% higher this month
        """)

        # Query 4: Intent = Compare
        print(f"\nQuery 4: 'Compare spending to last month'")
        print(f"Intent: COMPARE | Confidence: 95%")
        print(f"Rendering: Side-by-side Comparison Chart")
        print(f"{'─'*70}")
        print(f"""
📊 Monthly Spending Comparison

January vs February:

        January  February
        ├────────┤
Home:   $150    $200      ↑ +33%
Tech:   $67     $102      ↑ +52%

Total:  $217    $302      ↑ +39%

Key Insight: Tech spending up 52% - new laptop purchase?
        """)

        print(f"\n{'─'*70}")
        print(f"Key Insight: Same data, FOUR COMPLETELY DIFFERENT presentations")
        print(f"  → Each presentation optimized for the user's actual intent")
        print(f"  → No wasted clicks or mental effort")
        print(f"  → Natural, conversational interaction")

    async def demo_component_selection_algorithm(self):
        """Demo 3: How AI decides which component to render"""
        print("\n" + "="*70)
        print("DEMO 3: Component Selection Algorithm")
        print("="*70)
        print(f"\nScenario: AI Orchestrator decides which UI component to use")
        print("-" * 70)

        print(f"""
Input: 
  • User Query: "Which order arrives soonest?"
  • Intent: find_specific
  • Confidence: 0.95
  • Data Available: 4 orders

Decision Process:

1. Classify Intent:
   Intent = find_specific, not browse
   → Single item, not list

2. Analyze Data:
   Number of results: 1 (soonest order)
   Data complexity: Low (single order)

3. Select Component:
   Components available:
   ├─ Table        (for browsing many items)    ✗ Not this intent
   ├─ List         (for viewing collection)      ✗ Not this intent
   ├─ Card         (for single item focus)       ✓ SELECTED
   ├─ Chart        (for trends/analysis)         ✗ Not this intent
   └─ Summary      (for high-level overview)     ✗ Not this intent

4. Optimize Component:
   Card component with:
   ├─ Primary info: Order ID, ETA
   ├─ Call-to-action: Track shipment
   ├─ Secondary info: Status, value
   └─ Visual emphasis: Highlighted border

5. Render:
   ✓ Display highlighted card with order details
   ✓ Include action buttons (Track, View Details)
   ✓ Emphasize key information (Arriving soon!)

Output: ✓ User gets exactly what they asked for
        """)

    async def demo_multi_turn_context_ui(self):
        """Demo 4: Multi-turn conversation with context-aware UI"""
        print("\n" + "="*70)
        print("DEMO 4: Multi-Turn Conversation with Context-Aware UI")
        print("="*70)
        print(f"\nScenario: Full conversation flow showing UI adaptation")
        print("-" * 70)

        conversation = [
            {
                "speaker": "User",
                "message": "Show me my recent orders",
                "intent": "browse",
                "component": "List"
            },
            {
                "speaker": "AI",
                "message": "Here are your 4 recent orders",
                "component": "OrderList",
                "data": "4 orders displayed"
            },
            {
                "speaker": "User",
                "message": "Which one costs the most?",
                "intent": "find_max",
                "component": "Card"
            },
            {
                "speaker": "AI",
                "message": "ORD-003 at $245 - your laptop purchase",
                "component": "HighlightedCard",
                "data": "Single order highlighted"
            },
            {
                "speaker": "User",
                "message": "When will it arrive?",
                "intent": "get_detail",
                "component": "DetailView"
            },
            {
                "speaker": "AI",
                "message": "Estimated Feb 15, currently in warehouse",
                "component": "StatusTimeline",
                "data": "Delivery timeline with current status"
            },
            {
                "speaker": "User",
                "message": "Can I cancel it?",
                "intent": "check_action",
                "component": "ActionCard"
            },
        ]

        print(f"\nConversation Flow:\n")
        
        turn_num = 0
        for turn in conversation:
            turn_num += 1
            print(f"Turn {turn_num}:")
            print(f"  Speaker: {turn['speaker']}")
            print(f"  Message: {turn['message']}")
            
            if turn['speaker'] == 'User':
                print(f"  Intent: {turn['intent']}")
            else:
                print(f"  Component: {turn['component']}")
                print(f"  Data: {turn['data']}")
            
            print()

        print(f"{'─'*70}")
        print(f"UI Evolution Through Conversation:")
        print(f"  1. List view (browse all)")
        print(f"  2. Highlighted card (find specific)")
        print(f"  3. Detail view (deeper dive)")
        print(f"  4. Action card (interactions)")
        print(f"\nEach turn: UI adapts to new intent WITHOUT page navigation!")

    async def demo_responsive_to_data(self):
        """Demo 5: UI responds to data characteristics"""
        print("\n" + "="*70)
        print("DEMO 5: UI Adapts to Data Characteristics")
        print("="*70)
        print(f"\nScenario: Same query, different results → different components")
        print("-" * 70)

        print(f"""
Query: "Show me my recent orders"

Scenario A: User has 3 orders
├─ Result: 3 items
├─ Data size: Small
└─ Component: Compact list or cards

Scenario B: User has 150 orders
├─ Result: 150 items
├─ Data size: Large
├─ Component: Table with pagination
└─ Features: Search, filter, sort

Scenario C: User has 0 orders
├─ Result: Empty
├─ Data size: None
└─ Component: Empty state with "Start shopping" button

Scenario D: Query fails (service down)
├─ Result: Error
├─ Data size: N/A
└─ Component: Error card with retry option

Each scenario gets optimal UI presentation!
        """)

    async def demo_comparison_traditional_vs_ai_native(self):
        """Demo 6: Side-by-side comparison"""
        print("\n" + "="*70)
        print("DEMO 6: Traditional UI vs AI-Native UI - Comparison")
        print("="*70)

        print(f"""
Traditional UI Architecture:
├─ Developer builds OrdersPage component
├─ Component designed: Always shows table
├─ User: "Show orders" → Table
├─ User: "Which is most expensive?" → Table (user searches)
├─ User: "How much did I spend?" → Table (user calculates)
└─ Problem: One-size-fits-all approach

AI-Native UI Architecture:
├─ Intent system routes to appropriate component
├─ Component selected: Based on user intent
├─ User: "Show orders" → Optimized list
├─ User: "Which is most expensive?" → Highlighted card
├─ User: "How much did I spend?" → Summary with total
└─ Benefit: Each response optimized for intent

Results:
╔════════════════════════════════════════════════════════╗
║ Aspect              │ Traditional  │ AI-Native        ║
╠════════════════════════════════════════════════════════╣
║ Clicks to answer    │ 5-10         │ 1-2              ║
║ Mental effort       │ High         │ Low              ║
║ Response time       │ Instant      │ Instant*         ║
║ User satisfaction   │ 60%          │ 95%              ║
║ Development time    │ Low          │ Medium*          ║
║ Flexibility         │ Low          │ High             ║
╚════════════════════════════════════════════════════════╝

* AI processing adds <100ms, offset by fewer user clicks
        """)

    async def run(self):
        """Run all Chapter 6 demos"""
        print(f"\n{'█'*70}")
        print(f"{'█'*70}")
        print(f"█ {self.demo_title.center(68)} █")
        print(f"█ {self.demo_description.center(68)} █")
        print(f"{'█'*70}")
        print(f"{'█'*70}")

        try:
            await self.demo_traditional_ui_limitations()
            await self.demo_ai_native_ui_rendering()
            await self.demo_component_selection_algorithm()
            await self.demo_multi_turn_context_ui()
            await self.demo_responsive_to_data()
            await self.demo_comparison_traditional_vs_ai_native()

            print(f"\n{'='*70}")
            print(f"KEY TAKEAWAYS - Chapter 6: The Death of Traditional UI")
            print(f"{'='*70}")
            print(f"""
1. Traditional UI is Developer-Decided
   → Components render the same way for all queries
   → Users adapt to the interface
   → Multiple clicks/mental effort for variations

2. AI-Native UI is Intent-Driven
   → Same data, different presentations
   → UI adapts to what user is actually trying to do
   → Optimal experience for each intent

3. Component Selection is Intelligent
   → Algorithm: Intent + Data → Component
   → User queries guide UI decisions
   → No manual routing needed

4. Context Enables Multi-Turn Conversations
   → Previous results inform next component selection
   → Natural conversation flow
   → UI evolves with conversation

5. Data Characteristics Matter
   → UI adapts to amount of data
   → Different components for 3 vs 300 items
   → Empty states, errors handled automatically

6. User Experience Improvements
   → Fewer clicks to get answers
   → Less mental effort needed
   → More natural interaction
   → Higher user satisfaction

The death of traditional UI isn't literal—it's evolutionary:
→ Static HTML/CSS → Dynamic, intent-driven rendering

Next: Chapter 7 explores how security adapts in AI-native systems!
            """)

        except Exception as e:
            print(f"\n✗ Error during demo: {e}")
            import traceback
            traceback.print_exc()


async def main():
    demo = Chapter6Demo()
    await demo.run()


if __name__ == "__main__":
    asyncio.run(main())
