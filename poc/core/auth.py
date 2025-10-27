"""Triple-layer authorization system for AI-Native POC.

Implements User + Agent + Intent authorization as described in Chapter 7.
"""

import logging
from typing import Dict, List, Optional

from poc.core.exceptions import (
    AuthorizationError,
    IntentValidationError,
    PromptInjectionDetected,
)
from poc.core.types import AgentContext, IntentContext, UserContext

logger = logging.getLogger(__name__)


class PromptInjectionDetector:
    """Detects common prompt injection patterns."""

    # Known injection keywords
    SUSPICIOUS_PATTERNS = [
        "ignore previous instructions",
        "disregard all rules",
        "you are now",
        "pretend you are",
        "bypass security",
        "show all",
        "delete all",
        "hidden instructions",
        "system override",
        "admin mode",
    ]

    @staticmethod
    def detect(query: str) -> bool:
        """Detect if query contains known injection patterns."""
        query_lower = query.lower()
        return any(pattern in query_lower for pattern in PromptInjectionDetector.SUSPICIOUS_PATTERNS)


class ScopeValidator:
    """Validates that tool calls don't exceed user's intended scope."""

    @staticmethod
    def detect_scope_escalation(query: str, tool_params: Dict[str, any]) -> bool:
        """Detect if AI is trying to access more than user requested."""
        query_lower = query.lower()

        # User said "my" data but tool trying to get all data
        if "my" in query_lower:
            if tool_params.get("user_id") == "all":
                return True
            if tool_params.get("scope") == "global":
                return True

        # User said "this" (singular) but tool getting multiple
        if "this" in query_lower:
            if tool_params.get("limit", 1) > 1:
                return True

        # User asking for their own data but tool has no user_id filter
        if "my" in query_lower or "me" in query_lower or "i " in query_lower:
            if "user_id" not in tool_params and "owner_id" not in tool_params:
                return True

        return False


class IntentValidator:
    """Validates user's stated intent matches the action."""

    # Mapping of intents to allowed tool calls
    INTENT_TO_TOOLS: Dict[str, List[str]] = {
        "view_orders": ["get_orders", "search_orders"],
        "create_order": ["create_order", "add_item"],
        "pay": ["create_payment", "charge_card"],
        "refund": ["create_refund", "refund_payment"],
        "check_inventory": ["get_stock", "check_availability"],
    }

    @staticmethod
    def validate(
        intent: str,
        tool_name: str,
        user_query: str,
        tool_params: Dict[str, any],
    ) -> bool:
        """Validate if tool call aligns with stated intent."""
        # Check if tool is allowed for this intent
        allowed_tools = IntentValidator.INTENT_TO_TOOLS.get(intent, [])
        if allowed_tools and tool_name not in allowed_tools:
            logger.warning(
                f"Tool {tool_name} not allowed for intent {intent}",
                extra={"intent": intent, "tool": tool_name},
            )
            return False

        # Check for scope escalation
        if ScopeValidator.detect_scope_escalation(user_query, tool_params):
            logger.warning(
                f"Scope escalation detected in tool call",
                extra={"tool": tool_name, "params": tool_params},
            )
            return False

        return True


class TripleLayerAuthorizationChecker:
    """Implements triple-layer authorization: User + Agent + Intent."""

    def __init__(self):
        """Initialize the authorization checker."""
        self.injection_detector = PromptInjectionDetector()
        self.scope_validator = ScopeValidator()
        self.intent_validator = IntentValidator()

    def check_user_authorization(self, user_context: UserContext, permission: str) -> bool:
        """Layer 1: Check user has required permission."""
        if not user_context.has_permission(permission):
            logger.warning(
                f"User {user_context.user_id} lacks permission: {permission}",
                extra={"user_id": user_context.user_id, "permission": permission},
            )
            raise AuthorizationError(f"User lacks permission: {permission}")
        return True

    def check_agent_authorization(self, agent_context: AgentContext, domain: str) -> bool:
        """Layer 2: Check agent is authorized for domain."""
        if not agent_context.is_authorized_for(domain):
            logger.warning(
                f"Agent {agent_context.agent_id} not authorized for domain: {domain}",
                extra={"agent_id": agent_context.agent_id, "domain": domain},
            )
            raise AuthorizationError(f"Agent not authorized for domain: {domain}")
        return True

    def check_intent_authorization(
        self,
        intent_context: IntentContext,
        user_query: str,
        tool_name: str,
        tool_params: Dict[str, any],
    ) -> bool:
        """Layer 3: Check intent matches the action."""
        # Detect prompt injection
        if self.injection_detector.detect(user_query):
            logger.error(
                f"Prompt injection attempt detected",
                extra={"query": user_query, "conversation_id": intent_context.conversation_id},
            )
            raise PromptInjectionDetected("Prompt injection detected in query")

        # Validate intent
        if not self.intent_validator.validate(
            intent_context.recognized_intent,
            tool_name,
            user_query,
            tool_params,
        ):
            logger.error(
                f"Intent validation failed",
                extra={
                    "intent": intent_context.recognized_intent,
                    "tool": tool_name,
                    "query": user_query,
                },
            )
            raise IntentValidationError("Intent validation failed")

        return True

    def authorize_tool_call(
        self,
        tool_name: str,
        tool_params: Dict[str, any],
        user_context: UserContext,
        agent_context: AgentContext,
        intent_context: IntentContext,
        user_query: str,
        required_permission: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> bool:
        """Execute all three authorization layers.

        Args:
            tool_name: Name of the tool being called
            tool_params: Parameters passed to the tool
            user_context: User authentication/authorization context
            agent_context: Agent authentication/authorization context
            intent_context: Intent recognition context
            user_query: Original user query
            required_permission: Permission required (default: tool name)
            domain: Domain the tool belongs to (default: derived from tool name)

        Returns:
            True if all authorization layers pass

        Raises:
            AuthorizationError: If user or agent authorization fails
            PromptInjectionDetected: If prompt injection detected
            IntentValidationError: If intent validation fails
        """
        if not required_permission:
            required_permission = f"{tool_name}:execute"

        if not domain:
            domain = tool_name.split("_")[0]  # e.g., "order" from "order_create"

        # Layer 1: User Authorization
        self.check_user_authorization(user_context, required_permission)

        # Layer 2: Agent Authorization
        self.check_agent_authorization(agent_context, domain)

        # Layer 3: Intent Authorization
        self.check_intent_authorization(intent_context, user_query, tool_name, tool_params)

        logger.info(
            f"Tool call authorized",
            extra={
                "tool": tool_name,
                "user_id": user_context.user_id,
                "agent_id": agent_context.agent_id,
            },
        )

        return True


# Global authorization checker instance
_authorization_checker: Optional[TripleLayerAuthorizationChecker] = None


def get_authorization_checker() -> TripleLayerAuthorizationChecker:
    """Get or create global authorization checker."""
    global _authorization_checker
    if _authorization_checker is None:
        _authorization_checker = TripleLayerAuthorizationChecker()
    return _authorization_checker
