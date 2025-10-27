"""
Base service class for A2A-enabled FastAPI microservices.

Provides common functionality for all domain services:
- A2A Agent Card generation and serving
- Service health checks
- Tool execution framework
- Authorization integration
- Audit logging integration
"""

import json
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

# Import core framework components
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.types import UserContext, AgentContext, IntentContext, ToolCall, AuditLogEntry
from core.exceptions import (
    AuthorizationError,
    PromptInjectionDetected,
    IntentValidationError,
    ToolExecutionError,
)
from core.auth import TripleLayerAuthorizationChecker
from core.audit_logger import AuditLogger
from core.agent_card import A2AAgentCardBuilder


logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """Definition of a tool that this service provides."""
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable
    required_permissions: Optional[List[str]] = None


class BaseService:
    """
    Base class for all A2A-enabled microservices.
    
    Handles:
    - A2A protocol compliance (Agent Card serving, tool execution)
    - Authorization (triple-layer checks)
    - Audit logging
    - Health monitoring
    - Common error handling
    """
    
    def __init__(
        self,
        service_name: str,
        service_version: str,
        port: int,
        db_path: str,
    ):
        """
        Initialize base service.
        
        Args:
            service_name: Name of this service (e.g., "ProductService")
            service_version: Version (e.g., "1.0.0")
            port: Port to run on
            db_path: Path to SQLite database for this service
        """
        self.service_name = service_name
        self.service_version = service_version
        self.port = port
        self.db_path = db_path
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title=service_name,
            version=service_version,
            description=f"A2A-enabled {service_name}",
        )
        
        # Initialize security & audit systems
        self.auth_checker = TripleLayerAuthorizationChecker()
        self.audit_logger = AuditLogger(f"{service_name.lower()}_audit.jsonl")
        
        # Tools registry
        self.tools: Dict[str, ToolDefinition] = {}
        
        # Agent Card builder
        self.agent_card_builder = A2AAgentCardBuilder(
            agent_id=f"{service_name}:v{service_version}",
            name=service_name,
            version=service_version,
        )
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self) -> None:
        """Setup FastAPI routes for A2A protocol."""
        
        @self.app.get("/health")
        async def health() -> Dict[str, Any]:
            """Health check endpoint."""
            return {
                "status": "healthy",
                "service": self.service_name,
                "version": self.service_version,
                "timestamp": datetime.utcnow().isoformat(),
            }
        
        @self.app.get("/a2a/agent-card")
        async def get_agent_card() -> Dict[str, Any]:
            """
            Serve A2A Agent Card describing this service's capabilities.
            
            Returns:
                Agent Card with service metadata and tools
            """
            try:
                # Add all registered tools to the card
                for tool_name, tool_def in self.tools.items():
                    self.agent_card_builder.add_tool(
                        name=tool_def.name,
                        description=tool_def.description,
                        parameters=tool_def.parameters,
                    )
                
                card = self.agent_card_builder.build()
                
                # Log agent card request
                await self.audit_logger.record_interaction(
                    user_id="system",
                    agent_id="discovery",
                    action="agent_card_requested",
                    details={"service": self.service_name},
                    risk_level="LOW",
                )
                
                return card
                
            except Exception as e:
                logger.error(f"Error generating agent card: {e}")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to generate agent card"
                )
        
        @self.app.post("/a2a/execute-tool")
        async def execute_tool(request: Request) -> Dict[str, Any]:
            """
            Execute a tool with full authorization and audit logging.
            
            Request body:
                {
                    "tool_name": "SearchProducts",
                    "parameters": {...},
                    "user_context": {...},
                    "agent_context": {...},
                    "intent_context": {...}
                }
            
            Returns:
                {
                    "success": true,
                    "result": {...},
                    "execution_time_ms": 125
                }
            """
            start_time = datetime.utcnow()
            
            try:
                # Parse request
                body = await request.json()
                tool_name = body.get("tool_name")
                parameters = body.get("parameters", {})
                user_dict = body.get("user_context", {})
                agent_dict = body.get("agent_context", {})
                intent_dict = body.get("intent_context", {})
                
                # Reconstruct context objects
                user_context = UserContext(**user_dict) if user_dict else None
                agent_context = AgentContext(**agent_dict) if agent_dict else None
                intent_context = IntentContext(**intent_dict) if intent_dict else None
                
                # Validate tool exists
                if tool_name not in self.tools:
                    raise ToolExecutionError(f"Tool not found: {tool_name}")
                
                tool_def = self.tools[tool_name]
                
                # Authorization: Triple-layer check
                auth_result = await self.auth_checker.check_authorization(
                    user_context=user_context,
                    agent_context=agent_context,
                    intent_context=intent_context,
                    required_permissions=tool_def.required_permissions or [],
                )
                
                if not auth_result.get("authorized"):
                    reason = auth_result.get("reason", "Unknown")
                    
                    # Log security incident
                    await self.audit_logger.record_security_incident(
                        user_id=user_context.user_id if user_context else "unknown",
                        agent_id=agent_context.agent_id if agent_context else "unknown",
                        incident_type="authorization_failed",
                        details={
                            "tool": tool_name,
                            "reason": reason,
                        },
                    )
                    
                    raise AuthorizationError(reason)
                
                # Execute tool
                result = await tool_def.handler(**parameters)
                
                # Calculate execution time
                execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                # Log successful execution
                await self.audit_logger.record_tool_call(
                    user_id=user_context.user_id if user_context else "unknown",
                    agent_id=agent_context.agent_id if agent_context else "unknown",
                    tool_name=tool_name,
                    parameters=parameters,
                    result=result,
                    execution_time_ms=execution_time,
                )
                
                return {
                    "success": True,
                    "result": result,
                    "execution_time_ms": round(execution_time, 2),
                }
                
            except AuthorizationError as e:
                logger.warning(f"Authorization failed: {e}")
                raise HTTPException(status_code=403, detail=str(e))
                
            except PromptInjectionDetected as e:
                logger.warning(f"Prompt injection detected: {e}")
                raise HTTPException(status_code=400, detail="Invalid input detected")
                
            except ToolExecutionError as e:
                logger.error(f"Tool execution error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
            except Exception as e:
                logger.error(f"Unexpected error in tool execution: {e}")
                raise HTTPException(status_code=500, detail="Tool execution failed")
    
    def register_tool(
        self,
        name: str,
        description: str,
        handler: Callable,
        parameters: Dict[str, Any],
        required_permissions: Optional[List[str]] = None,
    ) -> None:
        """
        Register a tool that this service provides.
        
        Args:
            name: Tool name (e.g., "SearchProducts")
            description: Human-readable description
            handler: Async callable that executes the tool
            parameters: JSON schema for parameters
            required_permissions: List of required user permissions
        """
        self.tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
            required_permissions=required_permissions,
        )
        logger.info(f"Registered tool: {name}")
    
    async def run(self) -> None:
        """
        Start the service.
        
        Usage:
            service = ProductService()
            await service.run()
        """
        import uvicorn
        
        logger.info(f"Starting {self.service_name} on port {self.port}")
        
        config = uvicorn.Config(
            self.app,
            host="127.0.0.1",
            port=self.port,
            log_level="info",
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI application for testing or other use."""
        return self.app
