"""Audit logging system for AI-Native POC.

Implements immutable audit trail as described in Chapter 7.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from poc.core.types import AuditLogEntry, RiskLevel

logger = logging.getLogger(__name__)


class AuditLogger:
    """Comprehensive audit logging for AI-Native systems."""

    def __init__(self, log_file: Optional[Path] = None):
        """Initialize audit logger.

        Args:
            log_file: Path to audit log file (default: audit.jsonl)
        """
        self.log_file = log_file or Path("./audit.jsonl")
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.log_buffer: List[Dict[str, Any]] = []
        self.buffer_size = 10
        self._flush_lock = asyncio.Lock()

    def record_interaction(
        self,
        user_id: str,
        agent_id: str,
        conversation_id: str,
        action: str,
        resource: str,
        result: str,
        risk_level: RiskLevel = RiskLevel.LOW,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an interaction in audit log.

        Args:
            user_id: ID of the user
            agent_id: ID of the agent
            conversation_id: ID of the conversation
            action: Action performed
            resource: Resource affected
            result: Result of action (success/failure)
            risk_level: Risk level of the action
            details: Additional details
        """
        entry = AuditLogEntry(
            timestamp=datetime.utcnow(),
            user_id=user_id,
            agent_id=agent_id,
            action=action,
            resource=resource,
            result=result,
            risk_level=risk_level,
            details=details or {},
            conversation_id=conversation_id,
        )

        self.log_buffer.append(self._entry_to_dict(entry))

        if len(self.log_buffer) >= self.buffer_size:
            asyncio.create_task(self._flush_async())

    def record_authorization_check(
        self,
        user_id: str,
        agent_id: str,
        conversation_id: str,
        layer: str,  # "user", "agent", or "intent"
        resource: str,
        authorized: bool,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record authorization check result.

        Args:
            user_id: ID of the user
            agent_id: ID of the agent
            conversation_id: ID of the conversation
            layer: Authorization layer (user/agent/intent)
            resource: Resource being accessed
            authorized: Whether authorized
            details: Additional details
        """
        self.record_interaction(
            user_id=user_id,
            agent_id=agent_id,
            conversation_id=conversation_id,
            action=f"authorization_check_{layer}",
            resource=resource,
            result="success" if authorized else "failure",
            risk_level=RiskLevel.HIGH if not authorized else RiskLevel.LOW,
            details={
                "layer": layer,
                "authorized": authorized,
                **(details or {}),
            },
        )

    def record_tool_call(
        self,
        user_id: str,
        agent_id: str,
        conversation_id: str,
        tool_name: str,
        tool_params: Dict[str, Any],
        result: str,
        execution_time_ms: float,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record tool execution.

        Args:
            user_id: ID of the user
            agent_id: ID of the agent
            conversation_id: ID of the conversation
            tool_name: Name of the tool
            tool_params: Tool parameters
            result: Result of tool call (success/failure)
            execution_time_ms: Execution time in milliseconds
            details: Additional details
        """
        self.record_interaction(
            user_id=user_id,
            agent_id=agent_id,
            conversation_id=conversation_id,
            action="tool_execution",
            resource=tool_name,
            result=result,
            risk_level=RiskLevel.MEDIUM,
            details={
                "tool_name": tool_name,
                "tool_params": tool_params,
                "execution_time_ms": execution_time_ms,
                **(details or {}),
            },
        )

    def record_security_incident(
        self,
        user_id: str,
        agent_id: str,
        conversation_id: str,
        incident_type: str,
        description: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record security incident.

        Args:
            user_id: ID of the user
            agent_id: ID of the agent
            conversation_id: ID of the conversation
            incident_type: Type of security incident
            description: Description of incident
            details: Additional details
        """
        self.record_interaction(
            user_id=user_id,
            agent_id=agent_id,
            conversation_id=conversation_id,
            action=f"security_incident_{incident_type}",
            resource=incident_type,
            result="failure",
            risk_level=RiskLevel.CRITICAL,
            details={
                "incident_type": incident_type,
                "description": description,
                **(details or {}),
            },
        )

    def query_logs(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        action: Optional[str] = None,
        risk_level: Optional[RiskLevel] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Query audit logs.

        Args:
            user_id: Filter by user ID
            agent_id: Filter by agent ID
            action: Filter by action
            risk_level: Filter by risk level
            start_time: Filter by start time
            end_time: Filter by end time

        Returns:
            List of matching audit log entries
        """
        results = []

        try:
            with open(self.log_file, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if self._matches_filters(
                            entry,
                            user_id,
                            agent_id,
                            action,
                            risk_level,
                            start_time,
                            end_time,
                        ):
                            results.append(entry)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in audit log: {line}")
                        continue
        except FileNotFoundError:
            logger.warning(f"Audit log file not found: {self.log_file}")

        return results

    async def _flush_async(self) -> None:
        """Asynchronously flush log buffer to disk."""
        async with self._flush_lock:
            if not self.log_buffer:
                return

            try:
                with open(self.log_file, "a") as f:
                    for entry in self.log_buffer:
                        f.write(json.dumps(entry) + "\n")
                self.log_buffer.clear()
                logger.debug(f"Flushed audit logs to {self.log_file}")
            except IOError as e:
                logger.error(f"Failed to flush audit logs: {e}")

    def flush(self) -> None:
        """Synchronously flush log buffer to disk."""
        if not self.log_buffer:
            return

        try:
            with open(self.log_file, "a") as f:
                for entry in self.log_buffer:
                    f.write(json.dumps(entry) + "\n")
            self.log_buffer.clear()
            logger.debug(f"Flushed audit logs to {self.log_file}")
        except IOError as e:
            logger.error(f"Failed to flush audit logs: {e}")

    @staticmethod
    def _entry_to_dict(entry: AuditLogEntry) -> Dict[str, Any]:
        """Convert audit log entry to dictionary."""
        return {
            "timestamp": entry.timestamp.isoformat(),
            "user_id": entry.user_id,
            "agent_id": entry.agent_id,
            "action": entry.action,
            "resource": entry.resource,
            "result": entry.result,
            "risk_level": entry.risk_level.value,
            "conversation_id": entry.conversation_id,
            "details": entry.details,
        }

    @staticmethod
    def _matches_filters(
        entry: Dict[str, Any],
        user_id: Optional[str],
        agent_id: Optional[str],
        action: Optional[str],
        risk_level: Optional[RiskLevel],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
    ) -> bool:
        """Check if audit log entry matches filters."""
        if user_id and entry["user_id"] != user_id:
            return False
        if agent_id and entry["agent_id"] != agent_id:
            return False
        if action and entry["action"] != action:
            return False
        if risk_level and entry["risk_level"] != risk_level.value:
            return False

        if start_time:
            entry_time = datetime.fromisoformat(entry["timestamp"])
            if entry_time < start_time:
                return False

        if end_time:
            entry_time = datetime.fromisoformat(entry["timestamp"])
            if entry_time > end_time:
                return False

        return True


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get or create global audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
