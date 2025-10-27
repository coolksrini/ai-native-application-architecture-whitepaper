"""
Service Discovery Module - Phase 3
Discovers Phase 2 A2A microservices and fetches their capabilities.

This module handles:
- Querying the service registry for all registered services
- Fetching A2A Agent Cards from each service
- Caching metadata with TTL-based expiration
- Graceful handling of service unavailability
- Performance monitoring and logging
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import httpx
import logging

logger = logging.getLogger(__name__)


class ServiceMetadata:
    """Metadata for a discovered service."""

    def __init__(
        self,
        service_id: str,
        service_name: str,
        service_version: str,
        base_url: str,
        port: int,
        agent_card: Dict[str, Any],
        discovered_at: datetime,
        ttl_seconds: int = 300,
    ):
        """Initialize service metadata.
        
        Args:
            service_id: Unique service identifier
            service_name: Human-readable service name
            service_version: Service version string
            base_url: Base URL for the service
            port: Port the service is running on
            agent_card: A2A Agent Card containing tools and metadata
            discovered_at: When the service was discovered
            ttl_seconds: Time-to-live for this metadata (default 5 minutes)
        """
        self.service_id = service_id
        self.service_name = service_name
        self.service_version = service_version
        self.base_url = base_url
        self.port = port
        self.agent_card = agent_card
        self.discovered_at = discovered_at
        self.ttl_seconds = ttl_seconds

    def is_expired(self) -> bool:
        """Check if metadata has expired based on TTL."""
        expiration = self.discovered_at + timedelta(seconds=self.ttl_seconds)
        return datetime.now() > expiration

    def get_tools(self) -> List[Dict[str, Any]]:
        """Extract tools from agent card."""
        return self.agent_card.get("tools", [])

    def get_tool_names(self) -> List[str]:
        """Get list of all available tool names for this service."""
        return [tool["name"] for tool in self.get_tools()]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "service_id": self.service_id,
            "service_name": self.service_name,
            "service_version": self.service_version,
            "base_url": self.base_url,
            "port": self.port,
            "tools": self.get_tool_names(),
            "discovered_at": self.discovered_at.isoformat(),
            "ttl_seconds": self.ttl_seconds,
            "is_expired": self.is_expired(),
        }


class ServiceDiscovery:
    """Discovers and manages A2A microservices."""

    def __init__(self, timeout_seconds: int = 5):
        """Initialize the discovery service.
        
        Args:
            timeout_seconds: HTTP request timeout (default 5 seconds)
        """
        self.timeout_seconds = timeout_seconds
        self.metadata_cache: Dict[str, ServiceMetadata] = {}
        self.discovery_failures: Dict[str, int] = {}
        self.last_discovery: Optional[datetime] = None

    async def discover_service(
        self, service_name: str, base_url: str, port: int
    ) -> Optional[ServiceMetadata]:
        """Discover a single service and fetch its agent card.
        
        Args:
            service_name: Name of the service to discover
            base_url: Base URL of the service (e.g., "http://localhost")
            port: Port the service is running on
            
        Returns:
            ServiceMetadata if successful, None if discovery fails
        """
        service_id = f"{service_name}:{port}"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                # First, check health endpoint
                health_url = f"{base_url}:{port}/health"
                try:
                    health_response = await client.get(health_url)
                    health_response.raise_for_status()
                except Exception as e:
                    logger.warning(
                        f"Health check failed for {service_name} at {health_url}: {e}"
                    )
                    self.discovery_failures[service_id] = (
                        self.discovery_failures.get(service_id, 0) + 1
                    )
                    return None

                # Fetch agent card
                card_url = f"{base_url}:{port}/a2a/agent-card"
                card_response = await client.get(card_url)
                card_response.raise_for_status()
                agent_card = card_response.json()

                # Extract metadata from agent card
                service_version = agent_card.get("version", "1.0.0")

                metadata = ServiceMetadata(
                    service_id=service_id,
                    service_name=service_name,
                    service_version=service_version,
                    base_url=base_url,
                    port=port,
                    agent_card=agent_card,
                    discovered_at=datetime.now(),
                    ttl_seconds=300,  # 5-minute TTL
                )

                # Cache the metadata
                self.metadata_cache[service_id] = metadata
                self.discovery_failures[service_id] = 0  # Reset failure counter

                logger.info(
                    f"Successfully discovered {service_name} with {len(metadata.get_tools())} tools"
                )
                return metadata

        except Exception as e:
            logger.error(f"Error discovering service {service_name}: {e}")
            self.discovery_failures[service_id] = (
                self.discovery_failures.get(service_id, 0) + 1
            )
            return None

    async def discover_all_services(
        self, services_config: List[Dict[str, Any]]
    ) -> Dict[str, ServiceMetadata]:
        """Discover all services from configuration.
        
        Args:
            services_config: List of service configs with keys:
                - service_name: Name of the service
                - base_url: Base URL (e.g., "http://localhost")
                - port: Port number
                
        Returns:
            Dictionary mapping service_id to ServiceMetadata
        """
        logger.info(f"Starting discovery of {len(services_config)} services...")
        discovery_start = time.time()

        # Run all discoveries concurrently
        tasks = [
            self.discover_service(
                service["service_name"],
                service["base_url"],
                service["port"],
            )
            for service in services_config
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and build cache
        discovered = {}
        for result in results:
            if isinstance(result, ServiceMetadata):
                discovered[result.service_id] = result

        elapsed = time.time() - discovery_start
        success_count = len(discovered)
        logger.info(
            f"Discovery complete: {success_count}/{len(services_config)} services "
            f"discovered in {elapsed:.2f}s"
        )

        self.last_discovery = datetime.now()
        return discovered

    def get_service_metadata(self, service_id: str) -> Optional[ServiceMetadata]:
        """Get cached metadata for a service.
        
        Args:
            service_id: The service identifier (e.g., "product_service:8001")
            
        Returns:
            ServiceMetadata if available and not expired, None otherwise
        """
        metadata = self.metadata_cache.get(service_id)
        if metadata and not metadata.is_expired():
            return metadata
        
        # Remove expired metadata from cache
        if metadata and metadata.is_expired():
            del self.metadata_cache[service_id]
        
        return None

    def get_all_available_services(self) -> Dict[str, ServiceMetadata]:
        """Get all available (non-expired) services from cache.
        
        Returns:
            Dictionary mapping service_id to ServiceMetadata for all non-expired services
        """
        available = {}
        expired_ids = []

        for service_id, metadata in self.metadata_cache.items():
            if not metadata.is_expired():
                available[service_id] = metadata
            else:
                expired_ids.append(service_id)

        # Clean up expired entries
        for service_id in expired_ids:
            del self.metadata_cache[service_id]

        return available

    def find_services_by_tool(self, tool_name: str) -> List[ServiceMetadata]:
        """Find services that provide a specific tool.
        
        Args:
            tool_name: Name of the tool to search for
            
        Returns:
            List of ServiceMetadata for services providing this tool
        """
        matching_services = []
        
        for service_id, metadata in self.get_all_available_services().items():
            if tool_name in metadata.get_tool_names():
                matching_services.append(metadata)

        return matching_services

    def get_all_available_tools(self) -> Dict[str, List[str]]:
        """Get all available tools indexed by service.
        
        Returns:
            Dictionary mapping service_name to list of tool names
        """
        tools_by_service = {}
        
        for service_id, metadata in self.get_all_available_services().items():
            tools_by_service[metadata.service_name] = metadata.get_tool_names()

        return tools_by_service

    def get_discovery_stats(self) -> Dict[str, Any]:
        """Get discovery statistics and health metrics.
        
        Returns:
            Dictionary with discovery stats
        """
        total_services = len(self.metadata_cache)
        available_services = len(self.get_all_available_services())
        expired_services = total_services - available_services
        total_tools = sum(
            len(metadata.get_tools())
            for metadata in self.get_all_available_services().values()
        )

        # Calculate failure rate
        total_discovery_attempts = sum(self.discovery_failures.values())

        return {
            "total_discovered": total_services,
            "available_now": available_services,
            "expired": expired_services,
            "total_tools_available": total_tools,
            "last_discovery": self.last_discovery.isoformat()
            if self.last_discovery
            else None,
            "failures_by_service": self.discovery_failures,
            "cache_entries": len(self.metadata_cache),
        }

    def get_service_status(self) -> Dict[str, Any]:
        """Get detailed status of all services.
        
        Returns:
            Dictionary with service status information
        """
        status = {}
        
        for service_id, metadata in self.metadata_cache.items():
            status[service_id] = {
                "name": metadata.service_name,
                "version": metadata.service_version,
                "url": f"{metadata.base_url}:{metadata.port}",
                "tools": metadata.get_tool_names(),
                "available": not metadata.is_expired(),
                "discovered_at": metadata.discovered_at.isoformat(),
                "expires_at": (
                    metadata.discovered_at + timedelta(seconds=metadata.ttl_seconds)
                ).isoformat(),
            }

        return status

    def refresh_service(self, service_id: str) -> bool:
        """Force refresh a service's metadata (remove from cache).
        
        Args:
            service_id: The service to refresh
            
        Returns:
            True if the service was in cache and removed, False otherwise
        """
        if service_id in self.metadata_cache:
            del self.metadata_cache[service_id]
            return True
        return False

    def refresh_all_services(self) -> int:
        """Force refresh all services (clear cache).
        
        Returns:
            Number of services that were refreshed
        """
        count = len(self.metadata_cache)
        self.metadata_cache.clear()
        self.discovery_failures.clear()
        return count


# Default Phase 2 services configuration
PHASE_2_SERVICES_CONFIG = [
    {"service_name": "product_service", "base_url": "http://localhost", "port": 8001},
    {"service_name": "order_service", "base_url": "http://localhost", "port": 8002},
    {"service_name": "payment_service", "base_url": "http://localhost", "port": 8003},
    {
        "service_name": "inventory_service",
        "base_url": "http://localhost",
        "port": 8004,
    },
]


async def main():
    """Example usage of the discovery module."""
    discovery = ServiceDiscovery()

    # Discover all Phase 2 services
    discovered_services = await discovery.discover_all_services(
        PHASE_2_SERVICES_CONFIG
    )

    print(f"\n✅ Discovered {len(discovered_services)} services\n")

    # Show service details
    for service_id, metadata in discovered_services.items():
        print(f"📦 {metadata.service_name} (port {metadata.port})")
        print(f"   Version: {metadata.service_version}")
        print(f"   Tools: {', '.join(metadata.get_tool_names())}")
        print()

    # Show available tools
    print("🔧 Available Tools by Service:")
    for service_name, tools in discovery.get_all_available_tools().items():
        print(f"   {service_name}: {', '.join(tools)}")

    # Show statistics
    print("\n📊 Discovery Statistics:")
    stats = discovery.get_discovery_stats()
    for key, value in stats.items():
        if key != "failures_by_service":
            print(f"   {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
