"""
AI Orchestration Agent Package - Phase 3

This package contains the core orchestration components:
- discovery: Service discovery and metadata fetching
- intent_classifier: Intent classification and parameter extraction
- tool_executor: Tool execution via A2A endpoints
- context_manager: Multi-turn conversation context management
- orchestrator: Main orchestrator coordinating all components
"""

from agent.discovery import ServiceDiscovery, ServiceMetadata, PHASE_2_SERVICES_CONFIG
from agent.intent_classifier import (
    IntentClassifier,
    Intent,
    ConversationTurn,
)
from agent.tool_executor import ToolExecutor, ExecutionResult
from agent.context_manager import ContextManager, ContextTurn
from agent.orchestrator import AIOrchestrator, OrchestrationRequest, OrchestrationResponse

__version__ = "1.0.0"

__all__ = [
    "ServiceDiscovery",
    "ServiceMetadata",
    "PHASE_2_SERVICES_CONFIG",
    "IntentClassifier",
    "Intent",
    "ConversationTurn",
    "ToolExecutor",
    "ExecutionResult",
    "ContextManager",
    "ContextTurn",
    "AIOrchestrator",
    "OrchestrationRequest",
    "OrchestrationResponse",
]
