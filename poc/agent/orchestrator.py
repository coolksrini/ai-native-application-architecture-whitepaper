"""
AI Orchestrator Module - Phase 3
Main orchestrator that coordinates all Phase 3 components.

This module handles:
- Discovering all Phase 2 services
- Processing user queries end-to-end
- Intent classification and tool execution
- Multi-turn conversation management
- Response generation and formatting
- Metrics collection and reporting
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import json

from agent.discovery import ServiceDiscovery, PHASE_2_SERVICES_CONFIG
from agent.intent_classifier import IntentClassifier, Intent, ConversationTurn
from agent.tool_executor import ToolExecutor, ExecutionResult
from agent.context_manager import ContextManager, ContextTurn

logger = logging.getLogger(__name__)


@dataclass
class OrchestrationRequest:
    """A request for the orchestrator to process."""

    user_query: str
    user_id: str = "anonymous"
    session_id: str = "default"
    request_id: str = ""
    timestamp: str = ""

    def __post_init__(self):
        """Initialize derived fields."""
        if not self.request_id:
            self.request_id = f"req-{int(time.time() * 1000)}"
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class OrchestrationResponse:
    """Response from the orchestrator."""

    request_id: str
    user_query: str
    response_text: str
    intent: Intent
    execution_results: List[ExecutionResult]
    context_stats: Dict[str, Any]
    execution_time_ms: float
    timestamp: str = ""

    def __post_init__(self):
        """Initialize derived fields."""
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "user_query": self.user_query,
            "response_text": self.response_text,
            "intent": self.intent.to_dict() if self.intent else None,
            "execution_results": [r.to_dict() for r in self.execution_results],
            "context_stats": self.context_stats,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp,
        }


class AIOrchestrator:
    """Main orchestrator for coordinating all Phase 3 components."""

    def __init__(
        self,
        service_discovery_timeout: int = 10,
        max_retries: int = 2,
        max_context_tokens: int = 4096,
    ):
        """Initialize the orchestrator.

        Args:
            service_discovery_timeout: Timeout for service discovery in seconds
            max_retries: Maximum retries for failed operations
            max_context_tokens: Maximum tokens to keep in context
        """
        self.discovery = ServiceDiscovery()
        self.intent_classifier = IntentClassifier()
        self.tool_executor = None  # Initialized after discovery
        self.context_manager = ContextManager(max_context_tokens=max_context_tokens)
        self.service_discovery_timeout = service_discovery_timeout
        self.max_retries = max_retries
        self.max_context_tokens = max_context_tokens
        self.is_initialized = False
        self.initialization_error = None
        self.orchestration_history: List[OrchestrationResponse] = []
        self.service_metadata: Dict[str, Any] = {}

    async def initialize(self) -> bool:
        """Initialize the orchestrator by discovering services.

        Returns:
            True if initialization successful, False otherwise
        """
        logger.info("Initializing AI Orchestrator...")

        try:
            # Discover services
            logger.info("Discovering Phase 2 services...")
            discovered_services = await self.discovery.discover_all_services(
                PHASE_2_SERVICES_CONFIG
            )

            if not discovered_services:
                error_msg = "No services discovered"
                logger.error(error_msg)
                self.initialization_error = error_msg
                return False

            # Convert metadata to flat dictionary for tool executor
            self.service_metadata = {}
            available_tools = {}

            for service_id, metadata in discovered_services.items():
                service_info = {
                    "name": metadata.service_name,
                    "version": metadata.service_version,
                    "base_url": metadata.base_url,
                    "port": metadata.port,
                    "tools": metadata.get_tools(),
                }
                self.service_metadata[service_id] = service_info
                available_tools[metadata.service_name] = metadata.get_tool_names()

            # Initialize tool executor with discovered services
            self.tool_executor = ToolExecutor(self.service_metadata)

            # Initialize intent classifier with available tools
            self.intent_classifier = IntentClassifier(available_tools)

            logger.info(f"✅ Orchestrator initialized with {len(discovered_services)} services")
            logger.info(f"   Available tools: {sum(len(t) for t in available_tools.values())}")

            self.is_initialized = True
            return True

        except Exception as e:
            error_msg = f"Initialization failed: {str(e)}"
            logger.error(error_msg)
            self.initialization_error = error_msg
            return False

    async def process_query(
        self, request: OrchestrationRequest
    ) -> OrchestrationResponse:
        """Process a user query end-to-end.

        Args:
            request: The orchestration request

        Returns:
            OrchestrationResponse with results
        """
        if not self.is_initialized:
            error_msg = self.initialization_error or "Orchestrator not initialized"
            logger.error(error_msg)

            return OrchestrationResponse(
                request_id=request.request_id,
                user_query=request.user_query,
                response_text=f"ERROR: {error_msg}",
                intent=None,
                execution_results=[],
                context_stats={},
                execution_time_ms=0.0,
            )

        start_time = time.time()

        try:
            logger.info(f"Processing query: {request.user_query}")

            # Step 1: Classify intent
            logger.debug("Step 1: Classifying intent...")
            intent = self.intent_classifier.classify_intent(
                request.user_query, use_context=True
            )

            # Step 2: Execute tool if intent is known
            execution_results = []
            if intent.intent_name != "unknown":
                logger.debug("Step 2: Executing tool...")

                user_context = {
                    "user_id": request.user_id,
                    "session_id": request.session_id,
                }

                result = await self.tool_executor.execute_tool(
                    intent, user_context, agent_id="orchestrator"
                )
                execution_results.append(result)
                logger.debug(f"   Tool execution: {result.success}")

            # Step 3: Add to context history
            logger.debug("Step 3: Updating context...")
            response_text = self._generate_response(intent, execution_results)
            self.context_manager.add_turn(request.user_query, response_text)

            # Step 4: Add to conversation history for intent classifier
            turn = ConversationTurn(
                user_query=request.user_query,
                intent=intent,
                response=response_text,
                timestamp=datetime.now().isoformat(),
            )
            self.intent_classifier.add_conversation_turn(turn)

            execution_time_ms = (time.time() - start_time) * 1000

            logger.info(
                f"✅ Query processed successfully ({execution_time_ms:.0f}ms)"
            )

            response = OrchestrationResponse(
                request_id=request.request_id,
                user_query=request.user_query,
                response_text=response_text,
                intent=intent,
                execution_results=execution_results,
                context_stats=self.context_manager.get_context_stats(),
                execution_time_ms=execution_time_ms,
            )

            self.orchestration_history.append(response)
            return response

        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            logger.error(error_msg)

            execution_time_ms = (time.time() - start_time) * 1000

            return OrchestrationResponse(
                request_id=request.request_id,
                user_query=request.user_query,
                response_text=f"ERROR: {error_msg}",
                intent=None,
                execution_results=[],
                context_stats=self.context_manager.get_context_stats(),
                execution_time_ms=execution_time_ms,
            )

    def _generate_response(
        self, intent: Intent, execution_results: List[ExecutionResult]
    ) -> str:
        """Generate a response based on intent and execution results.

        Args:
            intent: The classified intent
            execution_results: Results from tool execution

        Returns:
            Response text
        """
        if intent.intent_name == "unknown":
            return (
                f"I didn't understand your query: '{intent.reasoning}'. "
                f"Could you please rephrase or try asking about products, orders, payments, or inventory?"
            )

        if not execution_results:
            return f"Ready to {intent.intent_name.replace('_', ' ')}, but no results available."

        result = execution_results[0]

        if not result.success:
            return f"Failed to {intent.intent_name.replace('_', ' ')}: {result.error}"

        # Format successful result
        result_text = f"✅ Successfully executed {intent.tool_name}\n"
        result_text += f"Service: {intent.service_name}\n"
        result_text += f"Time: {result.execution_time_ms:.0f}ms\n"

        if result.result:
            if isinstance(result.result, dict):
                result_text += f"Result: {json.dumps(result.result, indent=2)}"
            elif isinstance(result.result, list):
                result_text += f"Result: {len(result.result)} items found"
            else:
                result_text += f"Result: {str(result.result)}"

        return result_text

    def get_orchestration_stats(self) -> Dict[str, Any]:
        """Get overall orchestration statistics.

        Returns:
            Dictionary with orchestration stats
        """
        total_requests = len(self.orchestration_history)
        successful = sum(
            1
            for r in self.orchestration_history
            if r.execution_results
            and all(res.success for res in r.execution_results)
        )

        if not self.orchestration_history:
            return {
                "total_requests": 0,
                "successful": 0,
                "failed": 0,
                "success_rate": 0.0,
                "average_execution_time_ms": 0.0,
            }

        total_time = sum(r.execution_time_ms for r in self.orchestration_history)
        avg_time = total_time / total_requests if total_requests > 0 else 0

        return {
            "total_requests": total_requests,
            "successful": successful,
            "failed": total_requests - successful,
            "success_rate": successful / total_requests if total_requests > 0 else 0,
            "average_execution_time_ms": avg_time,
            "total_execution_time_ms": total_time,
            "services_discovered": len(self.service_metadata),
            "context_stats": self.context_manager.get_context_stats(),
        }

    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all discovered services.

        Returns:
            Dictionary with service status
        """
        return {
            "discovery": self.discovery.get_service_status(),
            "tool_performance": (
                self.tool_executor.get_tool_performance()
                if self.tool_executor
                else {}
            ),
            "orchestration_stats": self.get_orchestration_stats(),
        }

    async def run_interactive_mode(self):
        """Run the orchestrator in interactive mode."""
        print("\n" + "=" * 60)
        print("🤖 AI Orchestrator - Interactive Mode")
        print("=" * 60)
        print("\nInitializing... Please wait.\n")

        # Initialize
        if not await self.initialize():
            print(f"❌ Initialization failed: {self.initialization_error}")
            return

        print("✅ Orchestrator ready!\n")
        print("Enter queries (type 'quit' to exit, 'stats' for statistics):\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() == "quit":
                    print("\n👋 Goodbye!")
                    break

                if user_input.lower() == "stats":
                    stats = self.get_orchestration_stats()
                    print("\n📊 Statistics:")
                    for key, value in stats.items():
                        if key != "context_stats":
                            print(f"   {key}: {value}")
                    print()
                    continue

                # Process query
                request = OrchestrationRequest(
                    user_query=user_input,
                    user_id="interactive_user",
                    session_id="interactive_session",
                )

                response = await self.process_query(request)

                print(f"\nAssistant: {response.response_text}")
                print(f"Time: {response.execution_time_ms:.0f}ms\n")

            except KeyboardInterrupt:
                print("\n\n👋 Interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")


async def main():
    """Example usage of the orchestrator."""
    orchestrator = AIOrchestrator()

    # Initialize
    print("🚀 Starting AI Orchestrator...\n")
    success = await orchestrator.initialize()

    if not success:
        print(f"❌ Initialization failed: {orchestrator.initialization_error}")
        return

    print("✅ Orchestrator initialized successfully!\n")

    # Example queries
    test_queries = [
        "Search for laptops",
        "What's the status of my order",
        "Process a payment of $99.99",
        "Check our inventory levels",
    ]

    print("Processing example queries:\n")

    for query in test_queries:
        request = OrchestrationRequest(
            user_query=query,
            user_id="demo_user",
            session_id="demo_session",
        )

        response = await orchestrator.process_query(request)

        print(f"📝 Query: {query}")
        print(f"   Intent: {response.intent.intent_name if response.intent else 'unknown'}")
        if response.intent:
            print(f"   Confidence: {response.intent.confidence:.2%}")
        print(f"   Time: {response.execution_time_ms:.0f}ms")
        print()

    # Show statistics
    print("\n📊 Orchestration Statistics:")
    stats = orchestrator.get_orchestration_stats()
    for key, value in stats.items():
        if key != "context_stats":
            if isinstance(value, float):
                print(f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
