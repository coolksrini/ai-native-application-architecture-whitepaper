"""
ServiceRunner - Orchestrates starting and managing all microservices.

Responsibilities:
1. Create instances of all 4 services
2. Register them with the service registry
3. Start them on their designated ports
4. Monitor health
5. Provide management endpoints

Services:
- ProductService (port 8001)
- OrderService (port 8002)
- PaymentService (port 8003)
- InventoryService (port 8004)
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.service_registry import get_service_registry
from services.product_service import create_product_service
from services.order_service import create_order_service
from services.payment_service import create_payment_service
from services.inventory_service import create_inventory_service

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ServiceRunner:
    """Manages all microservices."""
    
    def __init__(self):
        """Initialize ServiceRunner."""
        self.services = []
        self.registry = get_service_registry()
        self.tasks = []
    
    def create_services(self) -> List[Any]:
        """
        Create all service instances.
        
        Returns:
            List of service instances
        """
        logger.info("Creating microservices...")
        
        services = [
            create_product_service(),
            create_order_service(),
            create_payment_service(),
            create_inventory_service(),
        ]
        
        self.services = services
        logger.info(f"Created {len(services)} services")
        
        return services
    
    def register_services(self) -> None:
        """Register all services with the service registry."""
        logger.info("Registering services with registry...")
        
        for service in self.services:
            try:
                # Create service metadata
                tools = [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    }
                    for tool in service.tools.values()
                ]
                
                # Register service
                self.registry.register_service(
                    name=service.service_name,
                    version=service.service_version,
                    url=f"http://127.0.0.1:{service.port}",
                    tools=tools,
                    tags=["microservice", "a2a"],
                    metadata={
                        "description": f"A2A-enabled {service.service_name}",
                        "port": service.port,
                    }
                )
                
                logger.info(f"Registered {service.service_name}")
                
            except Exception as e:
                logger.error(f"Failed to register {service.service_name}: {e}")
    
    async def check_service_health(self, service: Any) -> bool:
        """
        Check if a service is healthy.
        
        Args:
            service: Service instance
            
        Returns:
            True if healthy, False otherwise
        """
        try:
            import httpx
            
            url = f"http://127.0.0.1:{service.port}/health"
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=5.0)
                return response.status_code == 200
                
        except Exception as e:
            logger.warning(f"Health check failed for {service.service_name}: {e}")
            return False
    
    async def monitor_services(self) -> None:
        """
        Continuously monitor service health.
        
        Logs warnings if services become unhealthy.
        """
        logger.info("Starting service health monitor...")
        
        while True:
            try:
                for service in self.services:
                    healthy = await self.check_service_health(service)
                    status = "healthy" if healthy else "unhealthy"
                    if not healthy:
                        logger.warning(f"{service.service_name} is {status}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(30)
    
    async def start_services(self) -> None:
        """Start all services concurrently."""
        logger.info("Starting all services...")
        
        # Create tasks for each service
        for service in self.services:
            logger.info(f"Starting {service.service_name} on port {service.port}...")
            task = asyncio.create_task(service.run())
            self.tasks.append(task)
        
        # Start health monitor
        monitor_task = asyncio.create_task(self.monitor_services())
        self.tasks.append(monitor_task)
        
        logger.info("All services started")
        
        # Wait for all tasks
        try:
            await asyncio.gather(*self.tasks)
        except KeyboardInterrupt:
            logger.info("Received interrupt, shutting down...")
            for task in self.tasks:
                task.cancel()
    
    async def run(self) -> None:
        """
        Run the service orchestrator.
        
        Sequence:
        1. Create services
        2. Register with registry
        3. Start services
        4. Monitor health
        """
        try:
            # Create services
            self.create_services()
            
            # Register with registry
            self.register_services()
            
            # Start services
            await self.start_services()
            
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        except Exception as e:
            logger.error(f"ServiceRunner error: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of all services."""
        return {
            "services": [
                {
                    "name": service.service_name,
                    "version": service.service_version,
                    "port": service.port,
                    "tools": list(service.tools.keys()),
                }
                for service in self.services
            ],
            "registry_services": self.registry.list_services(),
        }


async def main():
    """Main entry point."""
    runner = ServiceRunner()
    
    # Show startup banner
    print("\n" + "="*60)
    print("AI-NATIVE POC - SERVICE RUNNER")
    print("="*60)
    print("\nStarting microservices on ports:")
    print("  - ProductService:  8001")
    print("  - OrderService:    8002")
    print("  - PaymentService:  8003")
    print("  - InventoryService: 8004")
    print("\nRegistry available at: http://127.0.0.1/registry")
    print("="*60 + "\n")
    
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
