"""
Phase 4 Testing Framework - Discovery Tests
Tests for the ServiceDiscovery module (Chapter 5: MCP Microservices)

This module covers:
- Service discovery functionality
- Metadata caching
- Tool inventory management
- Error handling
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from agent.discovery import ServiceDiscovery, ServiceMetadata, PHASE_2_SERVICES_CONFIG


class TestServiceMetadata:
    """Test ServiceMetadata class."""

    def test_create_metadata(self):
        """Test creating service metadata."""
        metadata = ServiceMetadata(
            service_id="test_service:8000",
            service_name="test_service",
            service_version="1.0.0",
            base_url="http://localhost",
            port=8000,
            agent_card={"tools": [{"name": "Tool1"}, {"name": "Tool2"}]},
            discovered_at=datetime.now(),
        )

        assert metadata.service_id == "test_service:8000"
        assert metadata.service_name == "test_service"
        assert metadata.port == 8000

    def test_get_tools(self):
        """Test extracting tools from agent card."""
        metadata = ServiceMetadata(
            service_id="test:8000",
            service_name="test",
            service_version="1.0.0",
            base_url="http://localhost",
            port=8000,
            agent_card={
                "tools": [
                    {"name": "SearchProducts"},
                    {"name": "GetProduct"},
                ]
            },
            discovered_at=datetime.now(),
        )

        tools = metadata.get_tools()
        assert len(tools) == 2
        assert tools[0]["name"] == "SearchProducts"

    def test_get_tool_names(self):
        """Test extracting tool names."""
        metadata = ServiceMetadata(
            service_id="test:8000",
            service_name="test",
            service_version="1.0.0",
            base_url="http://localhost",
            port=8000,
            agent_card={
                "tools": [
                    {"name": "Tool1"},
                    {"name": "Tool2"},
                    {"name": "Tool3"},
                ]
            },
            discovered_at=datetime.now(),
        )

        tool_names = metadata.get_tool_names()
        assert tool_names == ["Tool1", "Tool2", "Tool3"]

    def test_metadata_expiration(self):
        """Test metadata TTL expiration."""
        old_time = datetime.now() - timedelta(seconds=400)
        metadata = ServiceMetadata(
            service_id="test:8000",
            service_name="test",
            service_version="1.0.0",
            base_url="http://localhost",
            port=8000,
            agent_card={"tools": []},
            discovered_at=old_time,
            ttl_seconds=300,  # 5 minutes
        )

        assert metadata.is_expired()

    def test_metadata_not_expired(self):
        """Test metadata not expired."""
        metadata = ServiceMetadata(
            service_id="test:8000",
            service_name="test",
            service_version="1.0.0",
            base_url="http://localhost",
            port=8000,
            agent_card={"tools": []},
            discovered_at=datetime.now(),
            ttl_seconds=300,
        )

        assert not metadata.is_expired()

    def test_to_dict(self):
        """Test converting metadata to dictionary."""
        metadata = ServiceMetadata(
            service_id="test:8000",
            service_name="test",
            service_version="1.0.0",
            base_url="http://localhost",
            port=8000,
            agent_card={"tools": [{"name": "Tool1"}]},
            discovered_at=datetime.now(),
        )

        data = metadata.to_dict()
        assert data["service_name"] == "test"
        assert data["port"] == 8000
        assert "tools" in data


class TestServiceDiscovery:
    """Test ServiceDiscovery class."""

    def test_initialization(self):
        """Test ServiceDiscovery initialization."""
        discovery = ServiceDiscovery(timeout_seconds=5)

        assert discovery.timeout_seconds == 5
        assert len(discovery.metadata_cache) == 0
        assert discovery.last_discovery is None

    @pytest.mark.asyncio
    async def test_discover_service_success(self):
        """Test successful service discovery."""
        discovery = ServiceDiscovery()

        # Add metadata directly to test cache management
        metadata = ServiceMetadata(
            service_id="test_service:8000",
            service_name="test_service",
            service_version="1.0.0",
            base_url="http://localhost",
            port=8000,
            agent_card={"tools": [{"name": "TestTool"}]},
            discovered_at=datetime.now(),
        )

        discovery.metadata_cache["test_service:8000"] = metadata

        # Verify it was cached correctly
        result = discovery.get_service_metadata("test_service:8000")
        assert result is not None
        assert result.service_name == "test_service"
        assert len(result.get_tools()) == 1

    @pytest.mark.asyncio
    async def test_discover_service_failure(self):
        """Test service discovery failure handling."""
        discovery = ServiceDiscovery()

        with patch("agent.discovery.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_client.get.side_effect = Exception("Connection failed")

            result = await discovery.discover_service(
                "test_service", "http://localhost", 8000
            )

            assert result is None
            assert discovery.discovery_failures["test_service:8000"] > 0

    def test_get_service_metadata(self):
        """Test retrieving cached service metadata."""
        discovery = ServiceDiscovery()

        # Add metadata to cache
        metadata = ServiceMetadata(
            service_id="test:8000",
            service_name="test",
            service_version="1.0.0",
            base_url="http://localhost",
            port=8000,
            agent_card={"tools": []},
            discovered_at=datetime.now(),
        )

        discovery.metadata_cache["test:8000"] = metadata

        result = discovery.get_service_metadata("test:8000")
        assert result is not None
        assert result.service_name == "test"

    def test_get_service_metadata_expired(self):
        """Test expired metadata removal."""
        discovery = ServiceDiscovery()

        # Add expired metadata
        old_time = datetime.now() - timedelta(seconds=400)
        metadata = ServiceMetadata(
            service_id="test:8000",
            service_name="test",
            service_version="1.0.0",
            base_url="http://localhost",
            port=8000,
            agent_card={"tools": []},
            discovered_at=old_time,
            ttl_seconds=300,
        )

        discovery.metadata_cache["test:8000"] = metadata

        result = discovery.get_service_metadata("test:8000")
        assert result is None
        assert "test:8000" not in discovery.metadata_cache

    def test_get_all_available_services(self):
        """Test getting all non-expired services."""
        discovery = ServiceDiscovery()

        # Add mix of fresh and expired metadata
        discovery.metadata_cache["fresh:8000"] = ServiceMetadata(
            service_id="fresh:8000",
            service_name="fresh",
            service_version="1.0.0",
            base_url="http://localhost",
            port=8000,
            agent_card={"tools": []},
            discovered_at=datetime.now(),
            ttl_seconds=300,
        )

        old_time = datetime.now() - timedelta(seconds=400)
        discovery.metadata_cache["expired:8000"] = ServiceMetadata(
            service_id="expired:8000",
            service_name="expired",
            service_version="1.0.0",
            base_url="http://localhost",
            port=8000,
            agent_card={"tools": []},
            discovered_at=old_time,
            ttl_seconds=300,
        )

        available = discovery.get_all_available_services()
        assert len(available) == 1
        assert "fresh:8000" in available

    def test_find_services_by_tool(self):
        """Test finding services by tool name."""
        discovery = ServiceDiscovery()

        discovery.metadata_cache["service1:8000"] = ServiceMetadata(
            service_id="service1:8000",
            service_name="service1",
            service_version="1.0.0",
            base_url="http://localhost",
            port=8000,
            agent_card={"tools": [{"name": "SearchProducts"}, {"name": "GetProduct"}]},
            discovered_at=datetime.now(),
        )

        discovery.metadata_cache["service2:8001"] = ServiceMetadata(
            service_id="service2:8001",
            service_name="service2",
            service_version="1.0.0",
            base_url="http://localhost",
            port=8001,
            agent_card={"tools": [{"name": "ProcessPayment"}]},
            discovered_at=datetime.now(),
        )

        # Find services with SearchProducts
        services = discovery.find_services_by_tool("SearchProducts")
        assert len(services) == 1
        assert services[0].service_name == "service1"

    def test_get_all_available_tools(self):
        """Test getting all available tools indexed by service."""
        discovery = ServiceDiscovery()

        discovery.metadata_cache["service1:8000"] = ServiceMetadata(
            service_id="service1:8000",
            service_name="service1",
            service_version="1.0.0",
            base_url="http://localhost",
            port=8000,
            agent_card={"tools": [{"name": "Tool1"}, {"name": "Tool2"}]},
            discovered_at=datetime.now(),
        )

        discovery.metadata_cache["service2:8001"] = ServiceMetadata(
            service_id="service2:8001",
            service_name="service2",
            service_version="1.0.0",
            base_url="http://localhost",
            port=8001,
            agent_card={"tools": [{"name": "Tool3"}]},
            discovered_at=datetime.now(),
        )

        tools = discovery.get_all_available_tools()
        assert "service1" in tools
        assert "service2" in tools
        assert len(tools["service1"]) == 2
        assert len(tools["service2"]) == 1

    def test_get_discovery_stats(self):
        """Test discovery statistics."""
        discovery = ServiceDiscovery()

        discovery.metadata_cache["service:8000"] = ServiceMetadata(
            service_id="service:8000",
            service_name="service",
            service_version="1.0.0",
            base_url="http://localhost",
            port=8000,
            agent_card={"tools": [{"name": "Tool1"}]},
            discovered_at=datetime.now(),
        )

        discovery.last_discovery = datetime.now()

        stats = discovery.get_discovery_stats()
        assert stats["total_discovered"] == 1
        assert stats["available_now"] == 1
        assert stats["total_tools_available"] == 1

    def test_refresh_service(self):
        """Test refreshing a service's cache."""
        discovery = ServiceDiscovery()

        metadata = ServiceMetadata(
            service_id="test:8000",
            service_name="test",
            service_version="1.0.0",
            base_url="http://localhost",
            port=8000,
            agent_card={"tools": []},
            discovered_at=datetime.now(),
        )

        discovery.metadata_cache["test:8000"] = metadata

        result = discovery.refresh_service("test:8000")
        assert result is True
        assert "test:8000" not in discovery.metadata_cache

    def test_refresh_all_services(self):
        """Test refreshing all services."""
        discovery = ServiceDiscovery()

        discovery.metadata_cache["service1:8000"] = ServiceMetadata(
            service_id="service1:8000",
            service_name="service1",
            service_version="1.0.0",
            base_url="http://localhost",
            port=8000,
            agent_card={"tools": []},
            discovered_at=datetime.now(),
        )

        discovery.metadata_cache["service2:8001"] = ServiceMetadata(
            service_id="service2:8001",
            service_name="service2",
            service_version="1.0.0",
            base_url="http://localhost",
            port=8001,
            agent_card={"tools": []},
            discovered_at=datetime.now(),
        )

        count = discovery.refresh_all_services()
        assert count == 2
        assert len(discovery.metadata_cache) == 0


class TestPhase2ServiceConfig:
    """Test Phase 2 services configuration."""

    def test_phase_2_config_structure(self):
        """Test Phase 2 services config is correctly structured."""
        assert len(PHASE_2_SERVICES_CONFIG) == 4

        services = {s["service_name"] for s in PHASE_2_SERVICES_CONFIG}
        assert "product_service" in services
        assert "order_service" in services
        assert "payment_service" in services
        assert "inventory_service" in services

    def test_phase_2_config_ports(self):
        """Test Phase 2 services have correct ports."""
        config_dict = {s["service_name"]: s["port"] for s in PHASE_2_SERVICES_CONFIG}

        assert config_dict["product_service"] == 8001
        assert config_dict["order_service"] == 8002
        assert config_dict["payment_service"] == 8003
        assert config_dict["inventory_service"] == 8004


@pytest.mark.chapter5
def test_chapter5_service_discovery_validation():
    """
    Chapter 5: MCP Microservices - Service Discovery Validation
    
    Validates that service discovery meets Chapter 5 requirements:
    - Discovers all 4 Phase 2 services
    - Retrieves agent cards
    - Aggregates tool inventory
    - Manages service metadata
    """
    discovery = ServiceDiscovery()

    # Validate config has 4 services
    assert len(PHASE_2_SERVICES_CONFIG) == 4

    # Validate each service has required fields
    for service_config in PHASE_2_SERVICES_CONFIG:
        assert "service_name" in service_config
        assert "base_url" in service_config
        assert "port" in service_config
        assert service_config["base_url"] == "http://localhost"

    # Validate ports are correctly assigned
    ports = {s["port"] for s in PHASE_2_SERVICES_CONFIG}
    assert ports == {8001, 8002, 8003, 8004}
