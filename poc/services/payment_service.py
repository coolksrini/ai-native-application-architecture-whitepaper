"""
PaymentService - Handles payment processing operations.

Domain Services:
- ProcessPayment: Process payment for an order
- RefundPayment: Refund a previous payment
- GetPaymentHistory: Get payment transaction history
- ValidatePaymentMethod: Validate payment method

Database: SQLite (payments.db)
Port: 8003
"""

import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import logging

from services.base_service import BaseService

logger = logging.getLogger(__name__)


class PaymentService(BaseService):
    """Payment processing service."""
    
    # Mock payments data - in production would be in database
    PAYMENTS = {
        "PAY001": {
            "payment_id": "PAY001",
            "order_id": "ORD001",
            "user_id": "USER123",
            "amount": 105.97,
            "currency": "USD",
            "status": "completed",
            "payment_method": "credit_card",
            "last_4_digits": "4242",
            "created_at": (datetime.utcnow() - timedelta(days=5)).isoformat(),
            "completed_at": (datetime.utcnow() - timedelta(days=5)).isoformat(),
        },
        "PAY002": {
            "payment_id": "PAY002",
            "order_id": "ORD002",
            "user_id": "USER123",
            "amount": 129.99,
            "currency": "USD",
            "status": "pending",
            "payment_method": "credit_card",
            "last_4_digits": "5555",
            "created_at": (datetime.utcnow() - timedelta(days=1)).isoformat(),
            "completed_at": None,
        },
        "PAY003": {
            "payment_id": "PAY003",
            "order_id": "ORD003",
            "user_id": "USER456",
            "amount": 33.98,
            "currency": "USD",
            "status": "completed",
            "payment_method": "paypal",
            "last_4_digits": None,
            "created_at": (datetime.utcnow() - timedelta(days=10)).isoformat(),
            "completed_at": (datetime.utcnow() - timedelta(days=10)).isoformat(),
        },
    }
    
    PAYMENT_METHODS = ["credit_card", "debit_card", "paypal", "bank_transfer"]
    PAYMENT_STATUSES = ["pending", "processing", "completed", "failed", "refunded"]
    
    def __init__(self):
        """Initialize PaymentService."""
        super().__init__(
            service_name="PaymentService",
            service_version="1.0.0",
            port=8003,
            db_path="payments.db",
        )
        
        # Register tools
        self.register_tool(
            name="ProcessPayment",
            description="Process payment for an order",
            handler=self.process_payment,
            parameters={
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Order ID",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "User ID",
                    },
                    "amount": {
                        "type": "number",
                        "description": "Amount to charge",
                    },
                    "currency": {
                        "type": "string",
                        "description": "Currency code (e.g., USD)",
                        "default": "USD",
                    },
                    "payment_method": {
                        "type": "string",
                        "description": "Payment method (credit_card, debit_card, paypal, bank_transfer)",
                    },
                    "payment_details": {
                        "type": "object",
                        "description": "Payment method details (card number, token, etc)",
                    },
                },
                "required": ["order_id", "user_id", "amount", "payment_method", "payment_details"],
            },
            required_permissions=["payment:process"],
        )
        
        self.register_tool(
            name="RefundPayment",
            description="Refund a previous payment",
            handler=self.refund_payment,
            parameters={
                "type": "object",
                "properties": {
                    "payment_id": {
                        "type": "string",
                        "description": "Payment ID to refund",
                    },
                    "amount": {
                        "type": "number",
                        "description": "Amount to refund (partial refund if less than original)",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Refund reason",
                    },
                },
                "required": ["payment_id", "amount"],
            },
            required_permissions=["payment:refund"],
        )
        
        self.register_tool(
            name="GetPaymentHistory",
            description="Get payment transaction history",
            handler=self.get_payment_history,
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
            required_permissions=["payment:read"],
        )
        
        self.register_tool(
            name="ValidatePaymentMethod",
            description="Validate a payment method",
            handler=self.validate_payment_method,
            parameters={
                "type": "object",
                "properties": {
                    "payment_method": {
                        "type": "string",
                        "description": "Payment method (credit_card, debit_card, paypal, bank_transfer)",
                    },
                    "details": {
                        "type": "object",
                        "description": "Payment method details",
                    },
                },
                "required": ["payment_method"],
            },
            required_permissions=["payment:validate"],
        )
    
    async def process_payment(
        self,
        order_id: str,
        user_id: str,
        amount: float,
        currency: str = "USD",
        payment_method: str = None,
        payment_details: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Process payment for an order.
        
        Args:
            order_id: Order ID
            user_id: User ID
            amount: Amount to charge
            currency: Currency code
            payment_method: Payment method
            payment_details: Payment method details
            
        Returns:
            Payment result
        """
        payment_id = f"PAY{len(self.PAYMENTS) + 1:03d}"
        
        # Validate payment method
        if payment_method not in self.PAYMENT_METHODS:
            logger.warning(f"Invalid payment method: {payment_method}")
            return {
                "success": False,
                "error": f"Invalid payment method. Must be one of: {', '.join(self.PAYMENT_METHODS)}",
            }
        
        # Simulate payment processing
        # In production: call payment gateway (Stripe, etc)
        import random
        success = random.random() > 0.05  # 95% success rate
        
        if not success:
            logger.warning(f"Payment processing failed for order {order_id}")
            return {
                "success": False,
                "error": "Payment processing failed. Please try again.",
                "payment_id": payment_id,
            }
        
        # Extract last 4 digits for security
        last_4_digits = None
        if payment_method == "credit_card" and payment_details:
            card_number = payment_details.get("card_number", "")
            last_4_digits = card_number[-4:] if len(card_number) >= 4 else "****"
        
        payment = {
            "payment_id": payment_id,
            "order_id": order_id,
            "user_id": user_id,
            "amount": amount,
            "currency": currency,
            "status": "completed",
            "payment_method": payment_method,
            "last_4_digits": last_4_digits,
            "created_at": datetime.utcnow().isoformat(),
            "completed_at": datetime.utcnow().isoformat(),
        }
        
        self.PAYMENTS[payment_id] = payment
        logger.info(f"Payment {payment_id} processed successfully for order {order_id}")
        
        return {
            "success": True,
            "payment": payment,
        }
    
    async def refund_payment(
        self,
        payment_id: str,
        amount: float,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Refund a payment.
        
        Args:
            payment_id: Payment ID to refund
            amount: Amount to refund
            reason: Refund reason
            
        Returns:
            Refund result
        """
        if payment_id not in self.PAYMENTS:
            logger.warning(f"Cannot refund non-existent payment: {payment_id}")
            return {
                "success": False,
                "error": f"Payment {payment_id} not found",
            }
        
        payment = self.PAYMENTS[payment_id]
        
        # Can only refund completed payments
        if payment["status"] != "completed":
            logger.warning(f"Cannot refund payment with status {payment['status']}")
            return {
                "success": False,
                "error": f"Cannot refund payment with status {payment['status']}",
            }
        
        # Check amount
        if amount > payment["amount"]:
            logger.warning(f"Refund amount exceeds payment amount")
            return {
                "success": False,
                "error": "Refund amount cannot exceed payment amount",
            }
        
        # Create refund record (in production, would create separate refund entry)
        if amount == payment["amount"]:
            payment["status"] = "refunded"
        else:
            payment["status"] = "partial_refund"
        
        logger.info(f"Refunded ${amount} for payment {payment_id}. Reason: {reason or 'No reason'}")
        
        return {
            "success": True,
            "payment": payment,
            "refund_amount": amount,
        }
    
    async def get_payment_history(
        self,
        user_id: str,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get payment history for a user.
        
        Args:
            user_id: User ID
            status: Optional status filter
            
        Returns:
            Payment history
        """
        payments = [
            p for p in self.PAYMENTS.values()
            if p["user_id"] == user_id
        ]
        
        # Filter by status if provided
        if status:
            payments = [p for p in payments if p["status"] == status]
        
        # Sort by date descending
        payments = sorted(
            payments,
            key=lambda p: p["created_at"],
            reverse=True
        )
        
        # Calculate totals
        total_amount = sum(p["amount"] for p in payments)
        
        logger.info(f"Retrieved payment history for user {user_id} ({len(payments)} payments)")
        
        return {
            "user_id": user_id,
            "count": len(payments),
            "total_amount": total_amount,
            "currency": "USD",
            "payments": payments,
        }
    
    async def validate_payment_method(
        self,
        payment_method: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Validate a payment method.
        
        Args:
            payment_method: Payment method type
            details: Payment details
            
        Returns:
            Validation result
        """
        if payment_method not in self.PAYMENT_METHODS:
            logger.warning(f"Invalid payment method: {payment_method}")
            return {
                "valid": False,
                "error": f"Unknown payment method: {payment_method}",
            }
        
        # Validate based on method
        if payment_method == "credit_card" and details:
            card_number = details.get("card_number", "")
            if not card_number or len(card_number) < 13:
                return {
                    "valid": False,
                    "error": "Invalid card number",
                }
        
        logger.info(f"Validated payment method: {payment_method}")
        
        return {
            "valid": True,
            "payment_method": payment_method,
        }


def create_payment_service() -> PaymentService:
    """Factory function to create PaymentService instance."""
    return PaymentService()


if __name__ == "__main__":
    import asyncio
    
    service = create_payment_service()
    
    # Run the service
    asyncio.run(service.run())
