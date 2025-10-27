"""POC core framework initialization."""

from poc.core.agent_card import A2AAgentCardBuilder, create_agent_card
from poc.core.audit_logger import AuditLogger, get_audit_logger
from poc.core.auth import TripleLayerAuthorizationChecker, get_authorization_checker
from poc.core.exceptions import (
    AIServiceError,
    AuthenticationError,
    AuthorizationError,
    ContextWindowExceeded,
    IntentValidationError,
    PromptInjectionDetected,
    ServiceNotFound,
    ServiceRegistryError,
    SuspiciousActivityDetected,
    ToolExecutionError,
)
from poc.core.service_registry import A2AServiceRegistry, get_service_registry
from poc.core.types import (
    AgentContext,
    AuditLogEntry,
    ComponentSpecification,
    IntentContext,
    RiskLevel,
    ServiceMetadata,
    TestResult,
    ToolCall,
    UserContext,
)

__all__ = [
    # Exceptions
    "AIServiceError",
    "AuthenticationError",
    "AuthorizationError",
    "ContextWindowExceeded",
    "IntentValidationError",
    "PromptInjectionDetected",
    "ServiceNotFound",
    "ServiceRegistryError",
    "SuspiciousActivityDetected",
    "ToolExecutionError",
    # Types
    "AgentContext",
    "AuditLogEntry",
    "ComponentSpecification",
    "IntentContext",
    "RiskLevel",
    "ServiceMetadata",
    "TestResult",
    "ToolCall",
    "UserContext",
    # Authorization
    "TripleLayerAuthorizationChecker",
    "get_authorization_checker",
    # Audit
    "AuditLogger",
    "get_audit_logger",
    # Service Registry
    "A2AServiceRegistry",
    "get_service_registry",
    # Agent Card
    "A2AAgentCardBuilder",
    "create_agent_card",
]
