"""
ProductService - Handles product catalog operations.

Domain Services:
- SearchProducts: Full-text search across product catalog
- GetProduct: Retrieve detailed product information
- ListCategories: List all product categories
- GetProductsByCategory: Filter products by category

Database: SQLite (products.db)
Port: 8001
"""

import asyncio
import json
from typing import Any, Dict, List, Optional
from datetime import datetime
import logging

from services.base_service import BaseService

logger = logging.getLogger(__name__)


class ProductService(BaseService):
    """Product catalog management service."""
    
    # Mock product data - in production would be in database
    PRODUCTS = [
        {
            "id": "PROD001",
            "name": "Wireless Headphones",
            "category": "Electronics",
            "price": 79.99,
            "description": "High-quality wireless headphones with noise cancellation",
            "stock": 150,
            "rating": 4.5,
            "reviews": 245,
        },
        {
            "id": "PROD002",
            "name": "USB-C Cable",
            "category": "Electronics",
            "price": 12.99,
            "description": "Durable USB-C charging and data cable, 2 meters",
            "stock": 500,
            "rating": 4.7,
            "reviews": 1203,
        },
        {
            "id": "PROD003",
            "name": "Phone Case",
            "category": "Accessories",
            "price": 24.99,
            "description": "Protective phone case with shockproof design",
            "stock": 320,
            "rating": 4.2,
            "reviews": 567,
        },
        {
            "id": "PROD004",
            "name": "Screen Protector",
            "category": "Accessories",
            "price": 8.99,
            "description": "Tempered glass screen protector for phone displays",
            "stock": 200,
            "rating": 4.6,
            "reviews": 890,
        },
        {
            "id": "PROD005",
            "name": "Laptop Stand",
            "category": "Office",
            "price": 39.99,
            "description": "Ergonomic adjustable laptop stand for desks",
            "stock": 85,
            "rating": 4.4,
            "reviews": 234,
        },
        {
            "id": "PROD006",
            "name": "Mechanical Keyboard",
            "category": "Electronics",
            "price": 129.99,
            "description": "RGB mechanical keyboard with Cherry MX switches",
            "stock": 60,
            "rating": 4.8,
            "reviews": 450,
        },
        {
            "id": "PROD007",
            "name": "Mouse Pad",
            "category": "Accessories",
            "price": 19.99,
            "description": "Large extended mouse pad with non-slip base",
            "stock": 180,
            "rating": 4.3,
            "reviews": 312,
        },
        {
            "id": "PROD008",
            "name": "USB Hub",
            "category": "Electronics",
            "price": 34.99,
            "description": "7-port USB 3.0 hub with power adapter",
            "stock": 110,
            "rating": 4.5,
            "reviews": 445,
        },
    ]
    
    CATEGORIES = [
        {"name": "Electronics", "description": "Electronic devices and gadgets"},
        {"name": "Accessories", "description": "Protective and utility accessories"},
        {"name": "Office", "description": "Office productivity equipment"},
    ]
    
    def __init__(self):
        """Initialize ProductService."""
        super().__init__(
            service_name="ProductService",
            service_version="1.0.0",
            port=8001,
            db_path="products.db",
        )
        
        # Register tools
        self.register_tool(
            name="SearchProducts",
            description="Search products by name or description",
            handler=self.search_products,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (product name or keyword)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
            required_permissions=["product:read"],
        )
        
        self.register_tool(
            name="GetProduct",
            description="Get detailed information about a specific product",
            handler=self.get_product,
            parameters={
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "Product ID (e.g., PROD001)",
                    },
                },
                "required": ["product_id"],
            },
            required_permissions=["product:read"],
        )
        
        self.register_tool(
            name="ListCategories",
            description="List all available product categories",
            handler=self.list_categories,
            parameters={
                "type": "object",
                "properties": {},
            },
            required_permissions=["product:read"],
        )
        
        self.register_tool(
            name="GetProductsByCategory",
            description="Get all products in a specific category",
            handler=self.get_products_by_category,
            parameters={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Product category name",
                    },
                },
                "required": ["category"],
            },
            required_permissions=["product:read"],
        )
    
    async def search_products(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        Search products by name or description.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of matching products
        """
        query_lower = query.lower()
        results = []
        
        for product in self.PRODUCTS:
            # Search in name and description
            if (query_lower in product["name"].lower() or 
                query_lower in product["description"].lower()):
                results.append(product)
        
        # Sort by rating and limit results
        results = sorted(
            results,
            key=lambda x: x["rating"],
            reverse=True
        )[:limit]
        
        logger.info(f"Search query '{query}' returned {len(results)} results")
        
        return {
            "query": query,
            "count": len(results),
            "products": results,
        }
    
    async def get_product(self, product_id: str) -> Dict[str, Any]:
        """
        Get detailed product information.
        
        Args:
            product_id: Product ID
            
        Returns:
            Product details or error
        """
        for product in self.PRODUCTS:
            if product["id"] == product_id:
                logger.info(f"Retrieved product {product_id}")
                return {
                    "found": True,
                    "product": product,
                }
        
        logger.warning(f"Product not found: {product_id}")
        return {
            "found": False,
            "error": f"Product {product_id} not found",
        }
    
    async def list_categories(self) -> Dict[str, Any]:
        """
        List all product categories.
        
        Returns:
            List of categories
        """
        logger.info("Listed all categories")
        return {
            "count": len(self.CATEGORIES),
            "categories": self.CATEGORIES,
        }
    
    async def get_products_by_category(self, category: str) -> Dict[str, Any]:
        """
        Get all products in a category.
        
        Args:
            category: Category name
            
        Returns:
            Products in category
        """
        products = [p for p in self.PRODUCTS if p["category"] == category]
        
        logger.info(f"Retrieved {len(products)} products from category '{category}'")
        
        return {
            "category": category,
            "count": len(products),
            "products": products,
        }


def create_product_service() -> ProductService:
    """Factory function to create ProductService instance."""
    return ProductService()


if __name__ == "__main__":
    import asyncio
    
    service = create_product_service()
    
    # Run the service
    asyncio.run(service.run())
