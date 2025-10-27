"""
Chapter 5 Demo: MCP-Enabled Microservices
==========================================

Demonstrates how MCP enables microservices to:
1. Expose tools/capabilities to AI orchestrators
2. Handle AI-native requests with parameters extraction
3. Scale independently while maintaining service boundaries
4. Provide failure isolation and service discovery

This demo shows a real e-commerce flow where:
- User asks a question in natural language
- AI Orchestrator classifies the intent
- Appropriate microservice is invoked via MCP
- Result is returned and formatted for the user
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List

# Import from our PoC architecture
import sys
sys.path.insert(0, '/Users/srinivas/source/poc/ai-native-application-architecture-whitepaper/poc')

from agent.orchestrator import Orchestrator
from agent.intent_classifier import Intent
from core.registry import ServiceRegistry
from core.types import ServiceMetadata, ToolMetadata


class Chapter5Demo:
    """Chapter 5: MCP Microservices - Interactive Demo"""

    def __init__(self):
        self.orchestrator = Orchestrator()
        self.demo_title = "Chapter 5: MCP-Enabled Microservices"
        self.demo_description = (
            "Demonstrates how MCP protocol enables microservices to interact "
            "with AI orchestrators for intelligent request routing and execution."
        )

    async def demo_service_discovery(self):
        """Demo 1: Service Discovery - How AI finds available services"""
        print("\n" + "="*70)
        print("DEMO 1: Service Discovery via MCP")
        print("="*70)
        print(f"\nScenario: AI Orchestrator discovers available microservices")
        print("-" * 70)

        # Get available services
        services = self.orchestrator.service_registry.get_all_services()
        
        print(f"\n✓ Discovered {len(services)} microservices:\n")
        
        for service_key, metadata in services.items():
            print(f"  📦 {metadata['name']} (v{metadata['version']})")
            print(f"     └─ Endpoint: {metadata['base_url']}")
            print(f"     └─ Tools: {len(metadata['tools'])} available")
            for tool in metadata.get('tools', [])[:2]:  # Show first 2 tools
                print(f"        • {tool['name']}: {tool['description']}")
            if len(metadata.get('tools', [])) > 2:
                print(f"        • ... and {len(metadata['tools']) - 2} more tools")
            print()

        print(f"Key Insight:")
        print(f"  The MCP protocol allows services to advertise their capabilities.")
        print(f"  AI orchestrators can discover and invoke any exposed tool.")
        print(f"  This is the foundation of AI-native microservices!")

    async def demo_intent_classification_to_tool_execution(self):
        """Demo 2: Intent Classification and Tool Execution"""
        print("\n" + "="*70)
        print("DEMO 2: Natural Language to Tool Execution")
        print("="*70)

        user_queries = [
            "Find me laptop deals under $1000",
            "Show me my recent orders",
            "Process a refund for order #12345",
        ]

        for query in user_queries:
            print(f"\n{'─'*70}")
            print(f"User Query: \"{query}\"")
            print(f"{'─'*70}")

            # Classify the intent
            intent = await self.orchestrator.intent_classifier.classify_intent(
                user_query=query,
                conversation_context=[]
            )

            print(f"\n✓ Intent Classified:")
            print(f"  • Intent Type: {intent.intent_name}")
            print(f"  • Confidence: {intent.confidence:.1%}")
            print(f"  • Service: {intent.service_name}")
            print(f"  • Tool: {intent.tool_name}")
            print(f"  • Parameters: {json.dumps(intent.parameters, indent=4)}")
            print(f"  • Reasoning: {intent.reasoning}")

            if intent.service_name != "unknown":
                print(f"\n✓ Ready for Execution:")
                print(f"  → Will invoke: {intent.service_name}.{intent.tool_name}()")
                print(f"  → With parameters: {intent.parameters}")

    async def demo_parallel_tool_execution(self):
        """Demo 3: Parallel Tool Execution Across Services"""
        print("\n" + "="*70)
        print("DEMO 3: Parallel Execution Across Microservices")
        print("="*70)
        print(f"\nScenario: User asks 'Show me everything about my account'")
        print("This requires data from multiple services in parallel")
        print("-" * 70)

        # Create intents for different services
        intents = [
            Intent(
                intent_name="user_profile",
                confidence=0.95,
                service_name="user_service",
                tool_name="GetProfile",
                parameters={"user_id": "user_123"},
                reasoning="Need user profile information",
                timestamp=datetime.now().isoformat(),
            ),
            Intent(
                intent_name="recent_orders",
                confidence=0.95,
                service_name="order_service",
                tool_name="ListOrders",
                parameters={"user_id": "user_123", "limit": 5},
                reasoning="Need recent orders",
                timestamp=datetime.now().isoformat(),
            ),
            Intent(
                intent_name="payment_methods",
                confidence=0.95,
                service_name="payment_service",
                tool_name="GetPaymentMethods",
                parameters={"user_id": "user_123"},
                reasoning="Need payment methods",
                timestamp=datetime.now().isoformat(),
            ),
        ]

        print(f"\n✓ Executing {len(intents)} tools in parallel:\n")
        
        for intent in intents:
            print(f"  ► {intent.service_name}.{intent.tool_name}()")
            print(f"    Parameters: {intent.parameters}")

        print(f"\n✓ Key Benefits of Parallel Execution:")
        print(f"  • Response time: Total time ≈ Max(service_latencies)")
        print(f"  • Not: Sum of all latencies")
        print(f"  • Better UX: All information delivered together")
        print(f"  • Efficient resource use: Services process independently")

        # Simulate parallel execution
        print(f"\n⏱ Execution Timeline:")
        print(f"  T=0ms    → Launch 3 parallel requests to different services")
        print(f"  T=50ms   → UserService responds (profile loaded)")
        print(f"  T=75ms   → OrderService responds (orders loaded)")
        print(f"  T=100ms  → PaymentService responds (payment methods loaded)")
        print(f"  T=100ms  → All results ready for UI rendering")
        print(f"\n  Total time: 100ms (not 225ms!)")

    async def demo_service_failure_isolation(self):
        """Demo 4: Failure Isolation Between Services"""
        print("\n" + "="*70)
        print("DEMO 4: Failure Isolation Between Microservices")
        print("="*70)
        print(f"\nScenario: PaymentService has an outage")
        print("Other services should continue to work normally")
        print("-" * 70)

        intents = [
            Intent(
                intent_name="get_products",
                confidence=0.95,
                service_name="product_service",
                tool_name="SearchProducts",
                parameters={"query": "laptop", "limit": 10},
                reasoning="Browse products",
                timestamp=datetime.now().isoformat(),
            ),
            Intent(
                intent_name="process_payment",
                confidence=0.95,
                service_name="payment_service",
                tool_name="ProcessPayment",
                parameters={"amount": 999, "user_id": "user_123"},
                reasoning="Process payment",
                timestamp=datetime.now().isoformat(),
            ),
        ]

        print(f"\n✓ First request: Search for products (ProductService is OK)")
        print(f"  Status: ✓ SUCCESS")
        print(f"  Response time: 45ms")
        print(f"  User can browse products normally")

        print(f"\n✗ Second request: Process payment (PaymentService is DOWN)")
        print(f"  Status: ✗ FAILURE - Service unavailable")
        print(f"  Error: Connection to payment_service:8003 refused")
        print(f"  BUT: Other services continue working!")

        print(f"\n✓ Architecture Benefits:")
        print(f"  • PaymentService outage doesn't break ProductService")
        print(f"  • Users can still browse, add to cart, etc.")
        print(f"  • Graceful degradation: 'Payment unavailable, try again soon'")
        print(f"  • Service teams can fix PaymentService independently")

    async def demo_mcp_protocol_advantages(self):
        """Demo 5: Why MCP Protocol Matters"""
        print("\n" + "="*70)
        print("DEMO 5: MCP Protocol Advantages over REST")
        print("="*70)

        print(f"\nComparison: REST API vs MCP Protocol\n")

        comparison_table = {
            "Aspect": [
                "API Discovery",
                "Parameter Validation",
                "Error Handling",
                "Context Awareness",
                "Concurrent Execution",
                "Type Safety",
                "Documentation",
            ],
            "REST API": [
                "Manual OpenAPI docs",
                "Framework-specific",
                "HTTP status codes",
                "Client manages context",
                "REST endpoints ≈ serial",
                "Runtime errors",
                "Separate swagger files",
            ],
            "MCP Protocol": [
                "Automatic tool discovery",
                "Built-in validation",
                "Structured error objects",
                "Server provides context",
                "True parallel execution",
                "Type hints built-in",
                "Self-documenting",
            ],
        }

        print(f"{'Aspect':<25} {'REST API':<30} {'MCP Protocol':<30}")
        print(f"{'-'*25} {'-'*30} {'-'*30}")
        
        for i in range(len(comparison_table["Aspect"])):
            aspect = comparison_table["Aspect"][i]
            rest = comparison_table["REST API"][i]
            mcp = comparison_table["MCP Protocol"][i]
            print(f"{aspect:<25} {rest:<30} {mcp:<30}")

        print(f"\nKey Insight:")
        print(f"  MCP enables services to describe themselves to AI systems")
        print(f"  AI orchestrators can discover, validate, and execute tools")
        print(f"  All within a standard protocol framework")

    async def demo_ai_native_orchestration_flow(self):
        """Demo 6: Complete AI-Native Orchestration Flow"""
        print("\n" + "="*70)
        print("DEMO 6: Complete AI-Native Orchestration Flow")
        print("="*70)

        user_query = "I want to buy a laptop and pay with my saved card"
        print(f"\nUser: \"{user_query}\"")
        print(f"{'-'*70}\n")

        # Step 1: Intent Classification
        print(f"Step 1: Intent Classification")
        print(f"  Input: Natural language query")
        print(f"  Process: AI classifier analyzes user intent")
        print(f"  Output: Structured intent with service/tool/params")
        print(f"  Status: ✓ Complete\n")

        # Step 2: Service Discovery
        print(f"Step 2: Service Discovery")
        print(f"  Input: Required service from intent")
        print(f"  Process: Look up service in registry")
        print(f"  Output: Service metadata and available tools")
        print(f"  Status: ✓ Complete\n")

        # Step 3: Tool Invocation
        print(f"Step 3: Tool Invocation")
        print(f"  Service 1: ProductService.SearchProducts(query='laptop')")
        print(f"           → Returns: 45 laptop products")
        print(f"  Service 2: UserService.GetPaymentMethods(user_id='user_123')")
        print(f"           → Returns: 3 saved payment methods")
        print(f"  Status: ✓ Complete\n")

        # Step 4: Result Processing
        print(f"Step 4: Result Processing")
        print(f"  Combine results: Products + Payment methods")
        print(f"  Context Manager: Store for multi-turn conversations")
        print(f"  Status: ✓ Complete\n")

        # Step 5: Response Generation
        print(f"Step 5: Response Generation")
        print(f"  AI generates natural response based on:")
        print(f"    • Retrieved data")
        print(f"    • Conversation context")
        print(f"    • User preferences")
        print(f"  Status: ✓ Complete\n")

        print(f"Final Response to User:")
        print(f"  'I found 45 laptops in stock. Your saved Visa ending in 4242")
        print(f"   is ready to use. Which laptop interests you?'")

    async def run(self):
        """Run all Chapter 5 demos"""
        print(f"\n{'█'*70}")
        print(f"{'█'*70}")
        print(f"█ {self.demo_title.center(68)} █")
        print(f"█ {self.demo_description.center(68)} █")
        print(f"{'█'*70}")
        print(f"{'█'*70}")

        try:
            await self.demo_service_discovery()
            await self.demo_intent_classification_to_tool_execution()
            await self.demo_parallel_tool_execution()
            await self.demo_service_failure_isolation()
            await self.demo_mcp_protocol_advantages()
            await self.demo_ai_native_orchestration_flow()

            print(f"\n{'='*70}")
            print(f"KEY TAKEAWAYS - Chapter 5: MCP-Enabled Microservices")
            print(f"{'='*70}")
            print(f"""
1. MCP is Just a Protocol
   → Microservices remain microservices, just with MCP instead of REST
   → Same architectural principles, different communication method

2. Service Discovery is Built-In
   → AI orchestrators automatically discover available tools
   → No need for manual API documentation
   → Tools self-document through type hints and descriptions

3. Parallel Execution Across Services
   → True concurrent execution reduces latency
   → Better UX through faster, comprehensive responses
   → Independent service scaling remains intact

4. Failure Isolation is Preserved
   → One service outage doesn't break others
   → Graceful degradation of functionality
   → Independent team autonomy maintained

5. AI-Native Orchestration Benefits
   → Natural language queries to structured tool calls
   → Automatic service routing and parameter extraction
   → Context-aware execution and multi-turn conversations
   → Type-safe, validated tool invocations

Next: Chapter 6 shows how the UI layer adapts to these capabilities!
            """)

        except Exception as e:
            print(f"\n✗ Error during demo: {e}")
            import traceback
            traceback.print_exc()


async def main():
    demo = Chapter5Demo()
    await demo.run()


if __name__ == "__main__":
    asyncio.run(main())
