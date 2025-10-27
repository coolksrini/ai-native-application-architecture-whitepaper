"""
Microservices package for AI-Native POC.

Services:
- ProductService: Product catalog operations
- OrderService: Order management
- PaymentService: Payment processing
- InventoryService: Inventory management

All services use A2A protocol for agent-to-agent communication.
"""

from services.base_service import BaseService
from services.product_service import ProductService, create_product_service
from services.order_service import OrderService, create_order_service
from services.payment_service import PaymentService, create_payment_service
from services.inventory_service import InventoryService, create_inventory_service
from services.service_runner import ServiceRunner

__all__ = [
    "BaseService",
    "ProductService",
    "OrderService",
    "PaymentService",
    "InventoryService",
    "ServiceRunner",
    "create_product_service",
    "create_order_service",
    "create_payment_service",
    "create_inventory_service",
]
