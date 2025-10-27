"""
Tool Executor Module - Phase 3
Executes tools from discovered services with authorization and error handling.

This module handles:
- Selecting best tool based on intent
- Executing tool calls via A2A endpoints
- Authorization context management
- Error handling and retries
- Result collection and formatting
- Performance tracking
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import httpx

from agent.intent_classifier import Intent

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of a tool execution."""

    success: bool
    tool_name: str
    service_name: str
    result_data: Any
    parameters: Dict[str, Any] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    timestamp: str = ""

    def __post_init__(self):
        """Initialize derived fields."""
        if self.parameters is None:
            self.parameters = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ToolExecutor:
    """Executes tools from discovered services."""

    def __init__(
        self,
        service_metadata: Dict[str, Any] = None,
        timeout_seconds: int = 10,
        max_retries: int = 2,
        base_url: str = "http://localhost:8000",
    ):
        """Initialize the tool executor.

        Args:
            service_metadata: Dictionary mapping service_id to ServiceMetadata
            timeout_seconds: HTTP request timeout in seconds
            max_retries: Maximum number of retries for failed executions
            base_url: Base URL for service communication
        """
        self.service_metadata = service_metadata or {}
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.base_url = base_url
        self.execution_history: List[ExecutionResult] = []
        self.performance_stats: Dict[str, Dict[str, float]] = {}

    async def execute_tool(
        self,
        intent: Intent,
        user_context: Optional[Dict[str, Any]] = None,
        agent_id: str = "orchestrator",
    ) -> ExecutionResult:
        """Execute a tool based on the intent.

        Args:
            intent: The Intent object containing tool and parameters
            user_context: Additional context (user_id, session_id, etc.)
            agent_id: ID of the agent making the request

        Returns:
            ExecutionResult with the tool execution output
        """
        logger.info(f"Executing tool: {intent.tool_name} from {intent.service_name}")

        if intent.service_name == "unknown" or intent.tool_name == "unknown":
            return ExecutionResult(
                success=False,
                tool_name=intent.tool_name,
                service_name=intent.service_name,
                parameters=intent.parameters,
                result_data=None,
                error="Unknown intent - cannot execute",
                execution_time_ms=0.0,
                timestamp=datetime.now().isoformat(),
            )

        # Find the service
        service_metadata = self._find_service(intent.service_name)
        if not service_metadata:
            return ExecutionResult(
                success=False,
                tool_name=intent.tool_name,
                service_name=intent.service_name,
                parameters=intent.parameters,
                result_data=None,
                error=f"Service {intent.service_name} not found in metadata",
                execution_time_ms=0.0,
                timestamp=datetime.now().isoformat(),
            )

        # Verify tool is available
        available_tools = [t["name"] for t in service_metadata.get("tools", [])]
        if intent.tool_name not in available_tools:
            return ExecutionResult(
                success=False,
                tool_name=intent.tool_name,
                service_name=intent.service_name,
                parameters=intent.parameters,
                result_data=None,
                error=f"Tool {intent.tool_name} not available in {intent.service_name}",
                execution_time_ms=0.0,
                timestamp=datetime.now().isoformat(),
            )

        # Execute with retries
        for attempt in range(self.max_retries + 1):
            result = await self._execute_with_authorization(
                intent, service_metadata, user_context, agent_id
            )

            if result.success:
                self.execution_history.append(result)
                self._update_performance_stats(result)
                logger.info(
                    f"Tool execution successful: {intent.tool_name} "
                    f"({result.execution_time_ms:.0f}ms)"
                )
                return result

            # Retry on failure
            if attempt < self.max_retries:
                logger.warning(
                    f"Tool execution failed, retrying ({attempt + 1}/{self.max_retries}): {result.error}"
                )
                await asyncio.sleep(1)  # Wait before retry
            else:
                logger.error(f"Tool execution failed after {self.max_retries + 1} attempts")

        self.execution_history.append(result)
        return result

    async def _execute_with_authorization(
        self,
        intent: Intent,
        service_metadata: Dict[str, Any],
        user_context: Optional[Dict[str, Any]],
        agent_id: str,
    ) -> ExecutionResult:
        """Execute tool with authorization context.

        Args:
            intent: The intent to execute
            service_metadata: Service metadata
            user_context: User context information
            agent_id: Agent ID making the request

        Returns:
            ExecutionResult
        """
        start_time = time.time()

        try:
            # Prepare authorization context
            auth_context = {
                "user_id": user_context.get("user_id", "anonymous")
                if user_context
                else "anonymous",
                "agent_id": agent_id,
                "session_id": user_context.get("session_id", "default")
                if user_context
                else "default",
            }

            # Prepare tool execution request
            request_body = {
                "tool_name": intent.tool_name,
                "parameters": intent.parameters,
                "auth_context": auth_context,
            }

            # Build service URL
            service_url = service_metadata.get("base_url")
            port = service_metadata.get("port")
            if not service_url or not port:
                raise ValueError("Service metadata missing base_url or port")

            execute_url = f"{service_url}:{port}/a2a/execute-tool"

            logger.debug(f"Sending request to {execute_url}")
            logger.debug(f"Request body: {request_body}")

            # Execute the tool via A2A endpoint
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(execute_url, json=request_body)
                response.raise_for_status()

                execution_time_ms = (time.time() - start_time) * 1000
                response_data = response.json()

                logger.debug(f"Response: {response_data}")

                # Extract result
                result_data = response_data.get("result", response_data)

                return ExecutionResult(
                    success=True,
                    tool_name=intent.tool_name,
                    service_name=intent.service_name,
                    parameters=intent.parameters,
                    result_data=result_data,
                    error=None,
                    execution_time_ms=execution_time_ms,
                    timestamp=datetime.now().isoformat(),
                )

        except httpx.HTTPStatusError as e:
            execution_time_ms = (time.time() - start_time) * 1000
            error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
            logger.error(f"HTTP error executing tool: {error_msg}")

            return ExecutionResult(
                success=False,
                tool_name=intent.tool_name,
                service_name=intent.service_name,
                parameters=intent.parameters,
                result_data=None,
                error=error_msg,
                execution_time_ms=execution_time_ms,
                timestamp=datetime.now().isoformat(),
            )

        except httpx.RequestError as e:
            execution_time_ms = (time.time() - start_time) * 1000
            error_msg = f"Request error: {str(e)}"
            logger.error(f"Request error executing tool: {error_msg}")

            return ExecutionResult(
                success=False,
                tool_name=intent.tool_name,
                service_name=intent.service_name,
                parameters=intent.parameters,
                result_data=None,
                error=error_msg,
                execution_time_ms=execution_time_ms,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            error_msg = f"Execution error: {str(e)}"
            logger.error(f"Error executing tool: {error_msg}")

            return ExecutionResult(
                success=False,
                tool_name=intent.tool_name,
                service_name=intent.service_name,
                parameters=intent.parameters,
                result_data=None,
                error=error_msg,
                execution_time_ms=execution_time_ms,
                timestamp=datetime.now().isoformat(),
            )

    def _find_service(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Find service metadata by name.

        Args:
            service_name: Name of the service to find

        Returns:
            Service metadata dict or None
        """
        for service_id, metadata in self.service_metadata.items():
            if metadata.get("name") == service_name or service_name in service_id:
                return metadata

        return None

    def _update_performance_stats(self, result: ExecutionResult) -> None:
        """Update performance statistics for a tool.

        Args:
            result: The execution result to update stats from
        """
        key = f"{result.service_name}:{result.tool_name}"

        if key not in self.performance_stats:
            self.performance_stats[key] = {
                "count": 0,
                "total_time_ms": 0.0,
                "min_time_ms": float("inf"),
                "max_time_ms": 0.0,
                "success_count": 0,
                "error_count": 0,
            }

        stats = self.performance_stats[key]
        stats["count"] += 1
        stats["total_time_ms"] += result.execution_time_ms
        stats["min_time_ms"] = min(stats["min_time_ms"], result.execution_time_ms)
        stats["max_time_ms"] = max(stats["max_time_ms"], result.execution_time_ms)

        if result.success:
            stats["success_count"] += 1
        else:
            stats["error_count"] += 1

    async def execute_parallel_tools(
        self,
        intents: List[Intent],
        user_context: Optional[Dict[str, Any]] = None,
        agent_id: str = "orchestrator",
    ) -> List[ExecutionResult]:
        """Execute multiple tools in parallel.

        Args:
            intents: List of Intent objects to execute
            user_context: Additional context
            agent_id: Agent ID making the request

        Returns:
            List of ExecutionResult objects
        """
        logger.info(f"Executing {len(intents)} tools in parallel")

        tasks = [
            self.execute_tool(intent, user_context, agent_id) for intent in intents
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        clean_results = []
        for result in results:
            if isinstance(result, ExecutionResult):
                clean_results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Exception during parallel execution: {result}")
                clean_results.append(
                    ExecutionResult(
                        success=False,
                        tool_name="unknown",
                        service_name="unknown",
                        parameters={},
                        result_data=None,
                        error=str(result),
                        execution_time_ms=0.0,
                        timestamp=datetime.now().isoformat(),
                    )
                )

        return clean_results

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics.

        Returns:
            Dictionary with execution stats
        """
        total_executions = len(self.execution_history)
        successful = sum(1 for r in self.execution_history if r.success)
        failed = total_executions - successful

        if not self.execution_history:
            return {
                "total_executions": 0,
                "successful": 0,
                "failed": 0,
                "success_rate": 0.0,
                "average_execution_time_ms": 0.0,
            }

        total_time = sum(r.execution_time_ms for r in self.execution_history)
        avg_time = total_time / total_executions if total_executions > 0 else 0

        return {
            "total_executions": total_executions,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total_executions if total_executions > 0 else 0,
            "average_execution_time_ms": avg_time,
            "min_execution_time_ms": min(
                r.execution_time_ms for r in self.execution_history
            ),
            "max_execution_time_ms": max(
                r.execution_time_ms for r in self.execution_history
            ),
        }

    def get_tool_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics by tool.

        Returns:
            Dictionary mapping tool keys to performance stats
        """
        result = {}

        for key, stats in self.performance_stats.items():
            result[key] = {
                "executions": stats["count"],
                "successful": stats["success_count"],
                "failed": stats["error_count"],
                "success_rate": (
                    stats["success_count"] / stats["count"] if stats["count"] > 0 else 0
                ),
                "avg_time_ms": stats["total_time_ms"] / stats["count"]
                if stats["count"] > 0
                else 0,
                "min_time_ms": stats["min_time_ms"],
                "max_time_ms": stats["max_time_ms"],
            }

        return result

    def get_recent_executions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent execution results.

        Args:
            limit: Maximum number of recent executions to return

        Returns:
            List of execution results as dictionaries
        """
        recent = self.execution_history[-limit:]
        return [r.to_dict() for r in recent]


async def main():
    """Example usage of the tool executor."""
    # Mock service metadata
    service_metadata = {
        "product_service:8001": {
            "name": "product_service",
            "base_url": "http://localhost",
            "port": 8001,
            "tools": [
                {"name": "SearchProducts"},
                {"name": "GetProduct"},
                {"name": "ListCategories"},
            ],
        },
        "order_service:8002": {
            "name": "order_service",
            "base_url": "http://localhost",
            "port": 8002,
            "tools": [
                {"name": "CreateOrder"},
                {"name": "GetOrder"},
                {"name": "ListOrders"},
            ],
        },
    }

    executor = ToolExecutor(service_metadata)

    # Create sample intents
    intent1 = Intent(
        intent_name="search_products",
        confidence=0.9,
        service_name="product_service",
        tool_name="SearchProducts",
        parameters={"query": "laptop"},
        reasoning="Test search",
        timestamp=datetime.now().isoformat(),
    )

    intent2 = Intent(
        intent_name="get_order",
        confidence=0.85,
        service_name="order_service",
        tool_name="GetOrder",
        parameters={"order_id": "ORD-001"},
        reasoning="Test get order",
        timestamp=datetime.now().isoformat(),
    )

    print("🔧 Tool Executor Example\n")

    # Execute single tool
    print("Executing SearchProducts...")
    result1 = await executor.execute_tool(intent1)
    print(f"Result: {result1.success}")
    print(f"Execution time: {result1.execution_time_ms:.0f}ms")
    if result1.error:
        print(f"Error: {result1.error}")
    print()

    # Execute parallel tools
    print("Executing tools in parallel...")
    results = await executor.execute_parallel_tools([intent1, intent2])
    print(f"Executed {len(results)} tools")
    for r in results:
        print(f"  - {r.tool_name}: {r.success}")
    print()

    # Show statistics
    print("📊 Execution Statistics:")
    stats = executor.get_execution_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
