"""Core exceptions for AI-Native POC."""


class AIServiceError(Exception):
    """Base exception for AI service errors."""

    pass


class AuthorizationError(AIServiceError):
    """Raised when authorization check fails."""

    pass


class AuthenticationError(AIServiceError):
    """Raised when authentication fails."""

    pass


class IntentValidationError(AIServiceError):
    """Raised when intent validation fails."""

    pass


class PromptInjectionDetected(AIServiceError):
    """Raised when prompt injection is detected."""

    pass


class SuspiciousActivityDetected(AIServiceError):
    """Raised when suspicious activity is detected."""

    pass


class ServiceNotFound(AIServiceError):
    """Raised when a service cannot be found."""

    pass


class ToolExecutionError(AIServiceError):
    """Raised when tool execution fails."""

    pass


class ContextWindowExceeded(AIServiceError):
    """Raised when context window is exceeded."""

    pass


class ServiceRegistryError(AIServiceError):
    """Raised when service registry operations fail."""

    pass
