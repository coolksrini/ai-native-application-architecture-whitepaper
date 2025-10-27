"""
OrderService - Handles order management operations.

Domain Services:
- CreateOrder: Create a new order from items
- GetOrder: Retrieve order details and status
- ListOrders: List orders for a user
- CancelOrder: Cancel an existing order
- UpdateOrderStatus: Update order fulfillment status

Database: SQLite (orders.db)
Port: 8002
"""

import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import logging

from services.base_service import BaseService

logger = logging.getLogger(__name__)


class OrderService(BaseService):
    """Order management service."""
    
    # Mock orders data - in production would be in database
    ORDERS = {
        "ORD001": {
            "order_id": "ORD001",
            "user_id": "USER123",
            "status": "shipped",
            "created_at": (datetime.utcnow() - timedelta(days=5)).isoformat(),
            "updated_at": (datetime.utcnow() - timedelta(days=2)).isoformat(),
            "items": [
                {"product_id": "PROD001", "quantity": 1, "price": 79.99},
                {"product_id": "PROD002", "quantity": 2, "price": 12.99},
            ],
            "total": 105.97,
            "shipping_address": "123 Main St, City, ST 12345",
        },
        "ORD002": {
            "order_id": "ORD002",
            "user_id": "USER123",
            "status": "pending",
            "created_at": (datetime.utcnow() - timedelta(days=1)).isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "items": [
                {"product_id": "PROD006", "quantity": 1, "price": 129.99},
            ],
            "total": 129.99,
            "shipping_address": "123 Main St, City, ST 12345",
        },
        "ORD003": {
            "order_id": "ORD003",
            "user_id": "USER456",
            "status": "delivered",
            "created_at": (datetime.utcnow() - timedelta(days=10)).isoformat(),
            "updated_at": (datetime.utcnow() - timedelta(days=7)).isoformat(),
            "items": [
                {"product_id": "PROD003", "quantity": 1, "price": 24.99},
                {"product_id": "PROD004", "quantity": 1, "price": 8.99},
            ],
            "total": 33.98,
            "shipping_address": "456 Oak Ave, Town, ST 54321",
        },
    }
    
    ORDER_STATUSES = ["pending", "processing", "shipped", "delivered", "cancelled"]
    
    def __init__(self):
        """Initialize OrderService."""
        super().__init__(
            service_name="OrderService",
            service_version="1.0.0",
            port=8002,
            db_path="orders.db",
        )
        
        # Register tools
        self.register_tool(
            name="CreateOrder",
            description="Create a new order with items",
            handler=self.create_order,
            parameters={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User ID",
                    },
                    "items": {
                        "type": "array",
                        "description": "List of items to order",
                        "items": {
                            "type": "object",
                            "properties": {
                                "product_id": {"type": "string"},
                                "quantity": {"type": "integer"},
                                "price": {"type": "number"},
                            },
                        },
                    },
                    "shipping_address": {
                        "type": "string",
                        "description": "Delivery address",
                    },
                },
                "required": ["user_id", "items", "shipping_address"],
            },
            required_permissions=["order:create"],
        )
        
        self.register_tool(
            name="GetOrder",
            description="Get order details and current status",
            handler=self.get_order,
            parameters={
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Order ID",
                    },
                },
                "required": ["order_id"],
            },
            required_permissions=["order:read"],
        )
        
        self.register_tool(
            name="ListOrders",
            description="List orders for a user",
            handler=self.list_orders,
            parameters={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User ID",
                    },
                    "status": {
                        "type": "string",
                        "description": "Filter by status (optional)",
                    },
                },
                "required": ["user_id"],
            },
            required_permissions=["order:read"],
        )
        
        self.register_tool(
            name="CancelOrder",
            description="Cancel an existing order",
            handler=self.cancel_order,
            parameters={
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Order ID to cancel",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Cancellation reason",
                    },
                },
                "required": ["order_id"],
            },
            required_permissions=["order:cancel"],
        )
        
        self.register_tool(
            name="UpdateOrderStatus",
            description="Update order fulfillment status",
            handler=self.update_order_status,
            parameters={
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Order ID",
                    },
                    "status": {
                        "type": "string",
                        "description": f"New status ({', '.join(self.ORDER_STATUSES)})",
                    },
                },
                "required": ["order_id", "status"],
            },
            required_permissions=["order:admin"],
        )
    
    async def create_order(
        self,
        user_id: str,
        items: List[Dict[str, Any]],
        shipping_address: str,
    ) -> Dict[str, Any]:
        """
        Create a new order.
        
        Args:
            user_id: User creating order
            items: List of items
            shipping_address: Delivery address
            
        Returns:
            New order details
        """
        order_id = f"ORD{len(self.ORDERS) + 1:03d}"
        
        # Calculate total
        total = sum(item.get("price", 0) * item.get("quantity", 0) for item in items)
        
        order = {
            "order_id": order_id,
            "user_id": user_id,
            "status": "pending",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "items": items,
            "total": round(total, 2),
            "shipping_address": shipping_address,
        }
        
        self.ORDERS[order_id] = order
        logger.info(f"Created order {order_id} for user {user_id}")
        
        return {
            "success": True,
            "order": order,
        }
    
    async def get_order(self, order_id: str) -> Dict[str, Any]:
        """
        Get order details.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order details or error
        """
        if order_id in self.ORDERS:
            logger.info(f"Retrieved order {order_id}")
            return {
                "found": True,
                "order": self.ORDERS[order_id],
            }
        
        logger.warning(f"Order not found: {order_id}")
        return {
            "found": False,
            "error": f"Order {order_id} not found",
        }
    
    async def list_orders(
        self,
        user_id: str,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        List orders for a user.
        
        Args:
            user_id: User ID
            status: Optional status filter
            
        Returns:
            List of user's orders
        """
        orders = [
            order for order in self.ORDERS.values()
            if order["user_id"] == user_id
        ]
        
        # Filter by status if provided
        if status:
            orders = [o for o in orders if o["status"] == status]
        
        logger.info(f"Listed {len(orders)} orders for user {user_id}")
        
        return {
            "user_id": user_id,
            "count": len(orders),
            "orders": orders,
        }
    
    async def cancel_order(
        self,
        order_id: str,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            reason: Cancellation reason
            
        Returns:
            Cancellation result
        """
        if order_id not in self.ORDERS:
            logger.warning(f"Cannot cancel non-existent order: {order_id}")
            return {
                "success": False,
                "error": f"Order {order_id} not found",
            }
        
        order = self.ORDERS[order_id]
        
        # Can only cancel pending or processing orders
        if order["status"] not in ["pending", "processing"]:
            logger.warning(f"Cannot cancel order in {order['status']} status")
            return {
                "success": False,
                "error": f"Cannot cancel order in {order['status']} status",
            }
        
        order["status"] = "cancelled"
        order["updated_at"] = datetime.utcnow().isoformat()
        
        logger.info(f"Cancelled order {order_id}. Reason: {reason or 'No reason provided'}")
        
        return {
            "success": True,
            "order": order,
        }
    
    async def update_order_status(
        self,
        order_id: str,
        status: str,
    ) -> Dict[str, Any]:
        """
        Update order status.
        
        Args:
            order_id: Order ID
            status: New status
            
        Returns:
            Updated order
        """
        if order_id not in self.ORDERS:
            logger.warning(f"Cannot update non-existent order: {order_id}")
            return {
                "success": False,
                "error": f"Order {order_id} not found",
            }
        
        if status not in self.ORDER_STATUSES:
            logger.warning(f"Invalid status: {status}")
            return {
                "success": False,
                "error": f"Invalid status. Must be one of: {', '.join(self.ORDER_STATUSES)}",
            }
        
        order = self.ORDERS[order_id]
        order["status"] = status
        order["updated_at"] = datetime.utcnow().isoformat()
        
        logger.info(f"Updated order {order_id} status to {status}")
        
        return {
            "success": True,
            "order": order,
        }


def create_order_service() -> OrderService:
    """Factory function to create OrderService instance."""
    return OrderService()


if __name__ == "__main__":
    import asyncio
    
    service = create_order_service()
    
    # Run the service
    asyncio.run(service.run())
