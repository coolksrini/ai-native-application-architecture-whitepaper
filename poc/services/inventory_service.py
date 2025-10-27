"""
InventoryService - Handles inventory management operations.

Domain Services:
- CheckStock: Check product availability
- ReserveInventory: Reserve items for an order
- ReleaseReservation: Release reserved items
- UpdateStock: Update product stock levels
- GetInventoryStats: Get inventory statistics

Database: SQLite (inventory.db)
Port: 8004
"""

import asyncio
import json
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import logging

from services.base_service import BaseService

logger = logging.getLogger(__name__)


class InventoryService(BaseService):
    """Inventory management service."""
    
    # Mock inventory data - in production would be in database
    INVENTORY = {
        "PROD001": {
            "product_id": "PROD001",
            "name": "Wireless Headphones",
            "available": 150,
            "reserved": 10,
            "total": 160,
            "reorder_level": 50,
            "reorder_quantity": 100,
            "warehouse_location": "A-12-3",
            "last_updated": datetime.utcnow().isoformat(),
        },
        "PROD002": {
            "product_id": "PROD002",
            "name": "USB-C Cable",
            "available": 500,
            "reserved": 25,
            "total": 525,
            "reorder_level": 200,
            "reorder_quantity": 500,
            "warehouse_location": "B-05-1",
            "last_updated": datetime.utcnow().isoformat(),
        },
        "PROD003": {
            "product_id": "PROD003",
            "name": "Phone Case",
            "available": 320,
            "reserved": 15,
            "total": 335,
            "reorder_level": 100,
            "reorder_quantity": 200,
            "warehouse_location": "C-08-2",
            "last_updated": datetime.utcnow().isoformat(),
        },
        "PROD004": {
            "product_id": "PROD004",
            "name": "Screen Protector",
            "available": 200,
            "reserved": 5,
            "total": 205,
            "reorder_level": 75,
            "reorder_quantity": 150,
            "warehouse_location": "C-09-1",
            "last_updated": datetime.utcnow().isoformat(),
        },
        "PROD005": {
            "product_id": "PROD005",
            "name": "Laptop Stand",
            "available": 85,
            "reserved": 2,
            "total": 87,
            "reorder_level": 40,
            "reorder_quantity": 75,
            "warehouse_location": "A-15-4",
            "last_updated": datetime.utcnow().isoformat(),
        },
        "PROD006": {
            "product_id": "PROD006",
            "name": "Mechanical Keyboard",
            "available": 60,
            "reserved": 8,
            "total": 68,
            "reorder_level": 30,
            "reorder_quantity": 75,
            "warehouse_location": "B-20-2",
            "last_updated": datetime.utcnow().isoformat(),
        },
        "PROD007": {
            "product_id": "PROD007",
            "name": "Mouse Pad",
            "available": 180,
            "reserved": 10,
            "total": 190,
            "reorder_level": 60,
            "reorder_quantity": 150,
            "warehouse_location": "C-10-3",
            "last_updated": datetime.utcnow().isoformat(),
        },
        "PROD008": {
            "product_id": "PROD008",
            "name": "USB Hub",
            "available": 110,
            "reserved": 5,
            "total": 115,
            "reorder_level": 50,
            "reorder_quantity": 100,
            "warehouse_location": "B-12-1",
            "last_updated": datetime.utcnow().isoformat(),
        },
    }
    
    # Reservations tracking
    RESERVATIONS = {
        "RES001": {
            "reservation_id": "RES001",
            "order_id": "ORD001",
            "items": [
                {"product_id": "PROD001", "quantity": 1},
                {"product_id": "PROD002", "quantity": 2},
            ],
            "status": "active",
            "created_at": (datetime.utcnow() - timedelta(days=5)).isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(days=2)).isoformat(),
        },
        "RES002": {
            "reservation_id": "RES002",
            "order_id": "ORD002",
            "items": [
                {"product_id": "PROD006", "quantity": 1},
            ],
            "status": "active",
            "created_at": (datetime.utcnow() - timedelta(days=1)).isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(days=3)).isoformat(),
        },
    }
    
    def __init__(self):
        """Initialize InventoryService."""
        super().__init__(
            service_name="InventoryService",
            service_version="1.0.0",
            port=8004,
            db_path="inventory.db",
        )
        
        # Register tools
        self.register_tool(
            name="CheckStock",
            description="Check product availability and stock levels",
            handler=self.check_stock,
            parameters={
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "Product ID",
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Quantity needed (optional)",
                    },
                },
                "required": ["product_id"],
            },
            required_permissions=["inventory:read"],
        )
        
        self.register_tool(
            name="ReserveInventory",
            description="Reserve items for an order",
            handler=self.reserve_inventory,
            parameters={
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Order ID",
                    },
                    "items": {
                        "type": "array",
                        "description": "Items to reserve",
                        "items": {
                            "type": "object",
                            "properties": {
                                "product_id": {"type": "string"},
                                "quantity": {"type": "integer"},
                            },
                        },
                    },
                },
                "required": ["order_id", "items"],
            },
            required_permissions=["inventory:reserve"],
        )
        
        self.register_tool(
            name="ReleaseReservation",
            description="Release a reservation",
            handler=self.release_reservation,
            parameters={
                "type": "object",
                "properties": {
                    "reservation_id": {
                        "type": "string",
                        "description": "Reservation ID",
                    },
                },
                "required": ["reservation_id"],
            },
            required_permissions=["inventory:reserve"],
        )
        
        self.register_tool(
            name="UpdateStock",
            description="Update product stock levels",
            handler=self.update_stock,
            parameters={
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "Product ID",
                    },
                    "quantity_change": {
                        "type": "integer",
                        "description": "Change in quantity (positive or negative)",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for update",
                    },
                },
                "required": ["product_id", "quantity_change"],
            },
            required_permissions=["inventory:write"],
        )
        
        self.register_tool(
            name="GetInventoryStats",
            description="Get inventory statistics",
            handler=self.get_inventory_stats,
            parameters={
                "type": "object",
                "properties": {},
            },
            required_permissions=["inventory:read"],
        )
    
    async def check_stock(
        self,
        product_id: str,
        quantity: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Check product stock.
        
        Args:
            product_id: Product ID
            quantity: Optional quantity to check
            
        Returns:
            Stock information
        """
        if product_id not in self.INVENTORY:
            logger.warning(f"Product not found: {product_id}")
            return {
                "found": False,
                "error": f"Product {product_id} not found",
            }
        
        inv = self.INVENTORY[product_id]
        
        result = {
            "found": True,
            "product_id": product_id,
            "available": inv["available"],
            "reserved": inv["reserved"],
            "total": inv["total"],
        }
        
        # If quantity requested, check availability
        if quantity is not None:
            result["quantity_requested"] = quantity
            result["available_for_order"] = inv["available"] >= quantity
        
        logger.info(f"Checked stock for {product_id}")
        
        return result
    
    async def reserve_inventory(
        self,
        order_id: str,
        items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Reserve items for an order.
        
        Args:
            order_id: Order ID
            items: List of items to reserve
            
        Returns:
            Reservation result
        """
        # Check availability of all items first
        for item in items:
            product_id = item.get("product_id")
            quantity = item.get("quantity", 0)
            
            if product_id not in self.INVENTORY:
                logger.warning(f"Cannot reserve: product not found {product_id}")
                return {
                    "success": False,
                    "error": f"Product {product_id} not found",
                }
            
            inv = self.INVENTORY[product_id]
            if inv["available"] < quantity:
                logger.warning(f"Insufficient stock for {product_id}")
                return {
                    "success": False,
                    "error": f"Insufficient stock for {product_id}. Available: {inv['available']}, Requested: {quantity}",
                }
        
        # Reserve all items
        for item in items:
            product_id = item.get("product_id")
            quantity = item.get("quantity", 0)
            
            self.INVENTORY[product_id]["available"] -= quantity
            self.INVENTORY[product_id]["reserved"] += quantity
        
        # Create reservation
        reservation_id = f"RES{len(self.RESERVATIONS) + 1:03d}"
        reservation = {
            "reservation_id": reservation_id,
            "order_id": order_id,
            "items": items,
            "status": "active",
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(days=3)).isoformat(),
        }
        
        self.RESERVATIONS[reservation_id] = reservation
        logger.info(f"Reserved inventory for order {order_id}: {reservation_id}")
        
        return {
            "success": True,
            "reservation": reservation,
        }
    
    async def release_reservation(
        self,
        reservation_id: str,
    ) -> Dict[str, Any]:
        """
        Release a reservation.
        
        Args:
            reservation_id: Reservation ID
            
        Returns:
            Release result
        """
        if reservation_id not in self.RESERVATIONS:
            logger.warning(f"Reservation not found: {reservation_id}")
            return {
                "success": False,
                "error": f"Reservation {reservation_id} not found",
            }
        
        reservation = self.RESERVATIONS[reservation_id]
        
        if reservation["status"] != "active":
            logger.warning(f"Cannot release inactive reservation: {reservation_id}")
            return {
                "success": False,
                "error": f"Cannot release {reservation['status']} reservation",
            }
        
        # Release items
        for item in reservation["items"]:
            product_id = item.get("product_id")
            quantity = item.get("quantity", 0)
            
            self.INVENTORY[product_id]["available"] += quantity
            self.INVENTORY[product_id]["reserved"] -= quantity
        
        reservation["status"] = "released"
        logger.info(f"Released reservation: {reservation_id}")
        
        return {
            "success": True,
            "reservation": reservation,
        }
    
    async def update_stock(
        self,
        product_id: str,
        quantity_change: int,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Update product stock.
        
        Args:
            product_id: Product ID
            quantity_change: Change in quantity
            reason: Reason for update
            
        Returns:
            Updated stock
        """
        if product_id not in self.INVENTORY:
            logger.warning(f"Product not found: {product_id}")
            return {
                "success": False,
                "error": f"Product {product_id} not found",
            }
        
        inv = self.INVENTORY[product_id]
        old_total = inv["total"]
        
        inv["total"] += quantity_change
        if quantity_change > 0:
            inv["available"] += quantity_change
        else:
            # Deduct from available first
            inv["available"] = max(0, inv["available"] + quantity_change)
        
        inv["last_updated"] = datetime.utcnow().isoformat()
        
        logger.info(f"Updated stock for {product_id}: {old_total} → {inv['total']} ({reason or 'no reason'})")
        
        return {
            "success": True,
            "product_id": product_id,
            "old_total": old_total,
            "new_total": inv["total"],
            "available": inv["available"],
            "inventory": inv,
        }
    
    async def get_inventory_stats(self) -> Dict[str, Any]:
        """
        Get inventory statistics.
        
        Returns:
            Inventory statistics
        """
        total_items = sum(inv["total"] for inv in self.INVENTORY.values())
        total_available = sum(inv["available"] for inv in self.INVENTORY.values())
        total_reserved = sum(inv["reserved"] for inv in self.INVENTORY.values())
        
        # Products needing reorder
        low_stock = [
            inv for inv in self.INVENTORY.values()
            if inv["available"] <= inv["reorder_level"]
        ]
        
        logger.info(f"Generated inventory statistics")
        
        return {
            "total_skus": len(self.INVENTORY),
            "total_items": total_items,
            "total_available": total_available,
            "total_reserved": total_reserved,
            "low_stock_count": len(low_stock),
            "low_stock_products": low_stock,
            "timestamp": datetime.utcnow().isoformat(),
        }


def create_inventory_service() -> InventoryService:
    """Factory function to create InventoryService instance."""
    return InventoryService()


if __name__ == "__main__":
    import asyncio
    
    service = create_inventory_service()
    
    # Run the service
    asyncio.run(service.run())
