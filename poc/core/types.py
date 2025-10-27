"""Shared data types for AI-Native POC."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class RiskLevel(str, Enum):
    """Risk classification for actions."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class UserContext:
    """User authentication and authorization context."""

    user_id: str
    email: str
    scopes: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    session_token: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions

    def has_scope(self, scope: str) -> bool:
        """Check if user has specific scope."""
        return scope in self.scopes


@dataclass
class AgentContext:
    """AI agent authentication and authorization context."""

    agent_id: str
    agent_version: str
    authorized_domains: List[str] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.MEDIUM
    agent_token: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_authorized_for(self, domain: str) -> bool:
        """Check if agent is authorized for specific domain."""
        return domain in self.authorized_domains or "*" in self.authorized_domains


@dataclass
class IntentContext:
    """Intent recognition and validation context."""

    original_query: str
    recognized_intent: str
    confidence: float
    parameters: Dict[str, Any] = field(default_factory=dict)
    conversation_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def validate_intent(
        self,
        expected_action: str,
        actual_params: Dict[str, Any],
    ) -> bool:
        """Validate if intent matches action and parameters."""
        return self.recognized_intent == expected_action


@dataclass
class ToolCall:
    """Information about a tool call."""

    tool_name: str
    parameters: Dict[str, Any]
    service_id: str = ""
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0


@dataclass
class AuditLogEntry:
    """Single audit log entry."""

    timestamp: datetime
    user_id: str
    agent_id: str
    action: str
    resource: str
    result: str  # "success" or "failure"
    risk_level: RiskLevel = RiskLevel.LOW
    details: Dict[str, Any] = field(default_factory=dict)
    conversation_id: str = ""


@dataclass
class ServiceMetadata:
    """Metadata about an MCP/A2A service."""

    service_id: str
    service_name: str
    version: str
    description: str
    base_url: str
    tools: List[Dict[str, Any]] = field(default_factory=list)
    health_endpoint: str = "/health"
    registered_at: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)
    required_roles: List[str] = field(default_factory=list)


@dataclass
class ComponentSpecification:
    """UI component specification returned by renderer."""

    component: str
    props: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    """Result of a probabilistic test."""

    test_name: str
    total_runs: int
    successful_runs: int
    failed_runs: int
    accuracy: float
    threshold: float
    passed: bool
    error_details: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        return (self.successful_runs / self.total_runs * 100) if self.total_runs > 0 else 0.0
