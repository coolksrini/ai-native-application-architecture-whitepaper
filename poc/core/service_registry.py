"""A2A (Agent2Agent) Service Registry and Discovery.

Implements the service discovery pattern using A2A Agent Cards
as described in Chapter 5.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import httpx

from poc.core.exceptions import ServiceNotFound, ServiceRegistryError
from poc.core.types import ServiceMetadata

logger = logging.getLogger(__name__)


class A2AServiceRegistry:
    """Registry for discovering A2A services and their capabilities."""

    def __init__(self, ttl_seconds: int = 300):
        """Initialize service registry.

        Args:
            ttl_seconds: Time-to-live for cached service metadata
        """
        self.services: Dict[str, ServiceMetadata] = {}
        self.cache_times: Dict[str, datetime] = {}
        self.ttl = timedelta(seconds=ttl_seconds)
        self.health_status: Dict[str, bool] = {}

    def register_service(
        self,
        service_id: str,
        service_name: str,
        version: str,
        base_url: str,
        description: str = "",
        tools: Optional[List[Dict]] = None,
        tags: Optional[List[str]] = None,
    ) -> ServiceMetadata:
        """Register a service in the registry.

        Args:
            service_id: Unique service identifier
            service_name: Human-readable service name
            version: Service version
            base_url: Base URL of the service
            description: Service description
            tools: List of available tools/operations
            tags: Tags for categorizing the service

        Returns:
            ServiceMetadata of registered service
        """
        metadata = ServiceMetadata(
            service_id=service_id,
            service_name=service_name,
            version=version,
            base_url=base_url,
            description=description,
            tools=tools or [],
            tags=tags or [],
        )

        self.services[service_id] = metadata
        self.cache_times[service_id] = datetime.utcnow()
        logger.info(f"Registered service: {service_id} ({service_name})")

        return metadata

    def get_service(self, service_id: str) -> ServiceMetadata:
        """Get service metadata by ID.

        Args:
            service_id: ID of the service

        Returns:
            ServiceMetadata

        Raises:
            ServiceNotFound: If service not found
        """
        if service_id not in self.services:
            raise ServiceNotFound(f"Service not found: {service_id}")

        # Check if cache is stale
        if self._is_cache_stale(service_id):
            self._refresh_service(service_id)

        return self.services[service_id]

    def list_services(self, tags: Optional[List[str]] = None) -> List[ServiceMetadata]:
        """List all registered services, optionally filtered by tags.

        Args:
            tags: Filter services by tags

        Returns:
            List of ServiceMetadata
        """
        services = list(self.services.values())

        # Filter out unhealthy services
        services = [s for s in services if self.health_status.get(s.service_id, True)]

        # Filter by tags if provided
        if tags:
            services = [s for s in services if any(tag in s.tags for tag in tags)]

        return services

    def search_tools(self, query: str) -> List[Dict]:
        """Search for tools by name or description.

        Args:
            query: Search query

        Returns:
            List of tools matching the query
        """
        results = []
        query_lower = query.lower()

        for service in self.services.values():
            for tool in service.tools:
                tool_name = tool.get("name", "").lower()
                tool_description = tool.get("description", "").lower()

                if query_lower in tool_name or query_lower in tool_description:
                    results.append(
                        {
                            "service_id": service.service_id,
                            "service_name": service.service_name,
                            "tool": tool,
                            "base_url": service.base_url,
                        }
                    )

        return results

    def check_service_health(self, service_id: str) -> bool:
        """Check if service is healthy.

        Args:
            service_id: ID of the service

        Returns:
            True if service is healthy, False otherwise
        """
        if service_id not in self.services:
            return False

        service = self.services[service_id]
        health_url = f"{service.base_url}{service.health_endpoint}"

        try:
            response = httpx.get(health_url, timeout=5.0)
            is_healthy = response.status_code == 200
            self.health_status[service_id] = is_healthy
            return is_healthy
        except httpx.RequestError as e:
            logger.warning(f"Health check failed for {service_id}: {e}")
            self.health_status[service_id] = False
            return False

    async def check_service_health_async(self, service_id: str) -> bool:
        """Asynchronously check if service is healthy.

        Args:
            service_id: ID of the service

        Returns:
            True if service is healthy, False otherwise
        """
        if service_id not in self.services:
            return False

        service = self.services[service_id]
        health_url = f"{service.base_url}{service.health_endpoint}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(health_url, timeout=5.0)
                is_healthy = response.status_code == 200
                self.health_status[service_id] = is_healthy
                return is_healthy
        except httpx.RequestError as e:
            logger.warning(f"Health check failed for {service_id}: {e}")
            self.health_status[service_id] = False
            return False

    def get_agent_card(self, service_id: str) -> Dict:
        """Get A2A Agent Card for a service.

        Args:
            service_id: ID of the service

        Returns:
            A2A Agent Card dictionary

        Raises:
            ServiceNotFound: If service not found
        """
        service = self.get_service(service_id)

        return {
            "id": service.service_id,
            "name": service.service_name,
            "version": service.version,
            "description": service.description,
            "url": service.base_url,
            "tools": service.tools,
            "tags": service.tags,
        }

    def deregister_service(self, service_id: str) -> None:
        """Deregister a service from the registry.

        Args:
            service_id: ID of the service to deregister
        """
        if service_id in self.services:
            del self.services[service_id]
            del self.cache_times[service_id]
            if service_id in self.health_status:
                del self.health_status[service_id]
            logger.info(f"Deregistered service: {service_id}")

    def _is_cache_stale(self, service_id: str) -> bool:
        """Check if service cache is stale."""
        if service_id not in self.cache_times:
            return True

        cache_time = self.cache_times[service_id]
        return datetime.utcnow() - cache_time > self.ttl

    def _refresh_service(self, service_id: str) -> None:
        """Refresh service metadata from the service."""
        try:
            service = self.services[service_id]
            agent_card_url = f"{service.base_url}/a2a/agent-card"

            response = httpx.get(agent_card_url, timeout=5.0)
            response.raise_for_status()

            card = response.json()
            service.tools = card.get("tools", [])
            self.cache_times[service_id] = datetime.utcnow()
            logger.debug(f"Refreshed service metadata for: {service_id}")
        except httpx.RequestError as e:
            logger.warning(f"Failed to refresh service {service_id}: {e}")


# Global service registry instance
_service_registry: Optional[A2AServiceRegistry] = None


def get_service_registry() -> A2AServiceRegistry:
    """Get or create global service registry."""
    global _service_registry
    if _service_registry is None:
        _service_registry = A2AServiceRegistry()
    return _service_registry
