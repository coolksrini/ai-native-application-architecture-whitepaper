"""
Intent Classifier Module - Phase 3
Classifies user queries into intents and extracts parameters using Claude.

This module handles:
- Parsing user natural language queries
- Classifying queries into specific intents (search, create, process, etc.)
- Extracting parameters needed for tool execution
- Assigning confidence scores to predictions
- Supporting multi-turn conversation context
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import re

logger = logging.getLogger(__name__)


@dataclass
class Intent:
    """Represents an identified intent from user input."""

    intent_name: str
    confidence: float
    service_name: str
    tool_name: str
    parameters: Dict[str, Any]
    reasoning: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ConversationTurn:
    """Represents a single turn in a multi-turn conversation."""

    user_query: str
    intent: Intent
    response: Optional[str] = None
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "user_query": self.user_query,
            "intent": self.intent.to_dict() if self.intent else None,
            "response": self.response,
            "timestamp": self.timestamp,
        }
        return data


class IntentClassifier:
    """Classifies user queries into intents and extracts parameters."""

    def __init__(self, available_tools: Optional[Dict[str, List[str]]] = None):
        """Initialize the intent classifier.

        Args:
            available_tools: Dictionary mapping service names to tool names.
                           Example: {"product_service": ["SearchProducts", "GetProduct"]}
        """
        self.available_tools = available_tools or {}
        self.conversation_history: List[ConversationTurn] = []
        self.max_history = 10  # Keep last N turns for context

    def add_conversation_turn(self, turn: ConversationTurn) -> None:
        """Add a turn to conversation history.

        Args:
            turn: ConversationTurn to add
        """
        self.conversation_history.append(turn)

        # Keep only recent history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history :]

    def get_conversation_context(self) -> str:
        """Get recent conversation context as a string.

        Returns:
            Formatted conversation history for context
        """
        if not self.conversation_history:
            return "No previous conversation context."

        context = "Recent conversation:\n"
        for i, turn in enumerate(self.conversation_history[-5:], 1):  # Last 5 turns
            context += f"\n{i}. User: {turn.user_query}\n"
            if turn.response:
                context += f"   Assistant: {turn.response}\n"

        return context

    def classify_intent(
        self, user_query: str, use_context: bool = True
    ) -> Intent:
        """Classify a user query into an intent.

        This is a rule-based implementation that can be replaced with
        Claude API calls for more sophisticated intent detection.

        Args:
            user_query: The user's natural language query
            use_context: Whether to use conversation history for context

        Returns:
            Intent object with classification results
        """
        logger.info(f"Classifying intent for query: {user_query}")

        # Rule-based intent classification
        intent = self._rule_based_classify(user_query)

        # If no high-confidence rule matched, try pattern-based classification
        if intent.confidence < 0.7:
            intent = self._pattern_based_classify(user_query)

        # Apply context if available and enabled
        if use_context and self.conversation_history:
            intent = self._apply_context(intent, user_query)

        intent.timestamp = datetime.now().isoformat()
        logger.info(
            f"Classified intent: {intent.intent_name} (confidence: {intent.confidence:.2f})"
        )

        return intent

    def _rule_based_classify(self, user_query: str) -> Intent:
        """Apply rule-based intent classification.

        Args:
            user_query: The user query to classify

        Returns:
            Intent object
        """
        query_lower = user_query.lower()

        # Product-related intents
        # Check for search patterns: "search", "find", "look for", "what" + product-related words
        if any(
            keyword in query_lower
            for keyword in ["search", "find", "look", "what"]
        ) and any(
            keyword in query_lower
            for keyword in ["product", "laptop", "item", "thing", "stuff"]
        ):
            return Intent(
                intent_name="search_products",
                confidence=0.85,
                service_name="product_service",
                tool_name="SearchProducts",
                parameters=self._extract_search_terms(user_query),
                reasoning="Query contains product search keywords",
                timestamp="",
            )

        if any(
            keyword in query_lower
            for keyword in ["product", "get product", "details", "information"]
        ):
            product_id = self._extract_product_id(user_query)
            if product_id:
                return Intent(
                    intent_name="get_product",
                    confidence=0.9,
                    service_name="product_service",
                    tool_name="GetProduct",
                    parameters={"product_id": product_id},
                    reasoning=f"Query contains product ID {product_id}",
                    timestamp="",
                )

        if any(
            keyword in query_lower for keyword in ["category", "categories", "browse"]
        ):
            if "list" in query_lower or "show" in query_lower or "all" in query_lower:
                return Intent(
                    intent_name="list_categories",
                    confidence=0.9,
                    service_name="product_service",
                    tool_name="ListCategories",
                    parameters={},
                    reasoning="Query requests list of all categories",
                    timestamp="",
                )
            else:
                category = self._extract_category(user_query)
                if category:
                    return Intent(
                        intent_name="get_products_by_category",
                        confidence=0.85,
                        service_name="product_service",
                        tool_name="GetProductsByCategory",
                        parameters={"category": category},
                        reasoning=f"Query filters by category: {category}",
                        timestamp="",
                    )

        # Order-related intents
        if any(
            keyword in query_lower
            for keyword in ["create order", "place order", "new order", "buy", "purchase"]
        ):
            return Intent(
                intent_name="create_order",
                confidence=0.9,
                service_name="order_service",
                tool_name="CreateOrder",
                parameters=self._extract_order_parameters(user_query),
                reasoning="Query contains order creation keywords",
                timestamp="",
            )

        if any(
            keyword in query_lower
            for keyword in [
                "get order",
                "order status",
                "show order",
                "retrieve order",
            ]
        ):
            order_id = self._extract_order_id(user_query)
            if order_id:
                return Intent(
                    intent_name="get_order",
                    confidence=0.9,
                    service_name="order_service",
                    tool_name="GetOrder",
                    parameters={"order_id": order_id},
                    reasoning=f"Query contains order ID {order_id}",
                    timestamp="",
                )

        if any(
            keyword in query_lower for keyword in ["list orders", "show orders", "my orders"]
        ):
            user_id = self._extract_user_id(user_query)
            return Intent(
                intent_name="list_orders",
                confidence=0.85,
                service_name="order_service",
                tool_name="ListOrders",
                parameters={"user_id": user_id or "current_user"},
                reasoning="Query requests order list",
                timestamp="",
            )

        if any(
            keyword in query_lower for keyword in ["cancel order", "delete order", "remove order"]
        ):
            order_id = self._extract_order_id(user_query)
            if order_id:
                return Intent(
                    intent_name="cancel_order",
                    confidence=0.9,
                    service_name="order_service",
                    tool_name="CancelOrder",
                    parameters={"order_id": order_id},
                    reasoning=f"Query cancels order {order_id}",
                    timestamp="",
                )

        if any(
            keyword in query_lower
            for keyword in ["update order", "change order status", "order status"]
        ):
            order_id = self._extract_order_id(user_query)
            status = self._extract_status(user_query)
            if order_id:
                return Intent(
                    intent_name="update_order_status",
                    confidence=0.85,
                    service_name="order_service",
                    tool_name="UpdateOrderStatus",
                    parameters={"order_id": order_id, "new_status": status or "pending"},
                    reasoning="Query updates order status",
                    timestamp="",
                )

        # Payment-related intents
        if any(
            keyword in query_lower
            for keyword in ["process payment", "pay", "charge", "payment"]
        ):
            if "history" in query_lower or "past" in query_lower:
                user_id = self._extract_user_id(user_query)
                return Intent(
                    intent_name="get_payment_history",
                    confidence=0.85,
                    service_name="payment_service",
                    tool_name="GetPaymentHistory",
                    parameters={"user_id": user_id or "current_user"},
                    reasoning="Query requests payment history",
                    timestamp="",
                )
            else:
                return Intent(
                    intent_name="process_payment",
                    confidence=0.9,
                    service_name="payment_service",
                    tool_name="ProcessPayment",
                    parameters=self._extract_payment_parameters(user_query),
                    reasoning="Query contains payment processing keywords",
                    timestamp="",
                )

        if any(
            keyword in query_lower for keyword in ["refund", "return", "money back"]
        ):
            return Intent(
                intent_name="refund_payment",
                confidence=0.9,
                service_name="payment_service",
                tool_name="RefundPayment",
                parameters=self._extract_refund_parameters(user_query),
                reasoning="Query contains refund keywords",
                timestamp="",
            )

        if any(
            keyword in query_lower
            for keyword in ["validate payment", "check payment method", "payment method"]
        ):
            return Intent(
                intent_name="validate_payment_method",
                confidence=0.85,
                service_name="payment_service",
                tool_name="ValidatePaymentMethod",
                parameters=self._extract_payment_method(user_query),
                reasoning="Query validates payment method",
                timestamp="",
            )

        # Inventory-related intents
        if any(
            keyword in query_lower
            for keyword in ["check stock", "inventory", "stock", "available"]
        ):
            if "reserve" in query_lower or "hold" in query_lower:
                return Intent(
                    intent_name="reserve_inventory",
                    confidence=0.85,
                    service_name="inventory_service",
                    tool_name="ReserveInventory",
                    parameters=self._extract_inventory_parameters(user_query),
                    reasoning="Query reserves inventory",
                    timestamp="",
                )
            elif "release" in query_lower or "unreserve" in query_lower:
                reservation_id = self._extract_reservation_id(user_query)
                return Intent(
                    intent_name="release_reservation",
                    confidence=0.85,
                    service_name="inventory_service",
                    tool_name="ReleaseReservation",
                    parameters={"reservation_id": reservation_id or "unknown"},
                    reasoning="Query releases inventory reservation",
                    timestamp="",
                )
            else:
                return Intent(
                    intent_name="check_stock",
                    confidence=0.9,
                    service_name="inventory_service",
                    tool_name="CheckStock",
                    parameters=self._extract_stock_parameters(user_query),
                    reasoning="Query checks inventory/stock levels",
                    timestamp="",
                )

        if any(
            keyword in query_lower
            for keyword in ["update stock", "adjust inventory", "inventory stats"]
        ):
            if "stats" in query_lower:
                return Intent(
                    intent_name="get_inventory_stats",
                    confidence=0.9,
                    service_name="inventory_service",
                    tool_name="GetInventoryStats",
                    parameters={},
                    reasoning="Query requests inventory statistics",
                    timestamp="",
                )
            else:
                return Intent(
                    intent_name="update_stock",
                    confidence=0.85,
                    service_name="inventory_service",
                    tool_name="UpdateStock",
                    parameters=self._extract_stock_update_parameters(user_query),
                    reasoning="Query updates stock levels",
                    timestamp="",
                )

        # Default: low confidence unknown intent
        return Intent(
            intent_name="unknown",
            confidence=0.3,
            service_name="unknown",
            tool_name="unknown",
            parameters={},
            reasoning="Query did not match any known intent pattern",
            timestamp="",
        )

    def _pattern_based_classify(self, user_query: str) -> Intent:
        """Apply pattern-based classification for edge cases.

        Args:
            user_query: The user query to classify

        Returns:
            Intent object
        """
        query_lower = user_query.lower()

        # Try to match keywords to services
        if "product" in query_lower:
            return Intent(
                intent_name="product_query",
                confidence=0.6,
                service_name="product_service",
                tool_name="SearchProducts",
                parameters={"query": user_query},
                reasoning="Query contains 'product' keyword",
                timestamp="",
            )

        if "order" in query_lower:
            return Intent(
                intent_name="order_query",
                confidence=0.6,
                service_name="order_service",
                tool_name="ListOrders",
                parameters={"user_id": "current_user"},
                reasoning="Query contains 'order' keyword",
                timestamp="",
            )

        if "payment" in query_lower or "pay" in query_lower:
            return Intent(
                intent_name="payment_query",
                confidence=0.6,
                service_name="payment_service",
                tool_name="GetPaymentHistory",
                parameters={"user_id": "current_user"},
                reasoning="Query contains payment-related keyword",
                timestamp="",
            )

        if "inventory" in query_lower or "stock" in query_lower:
            return Intent(
                intent_name="inventory_query",
                confidence=0.6,
                service_name="inventory_service",
                tool_name="GetInventoryStats",
                parameters={},
                reasoning="Query contains inventory-related keyword",
                timestamp="",
            )

        # Complete fallback
        return Intent(
            intent_name="unknown",
            confidence=0.2,
            service_name="unknown",
            tool_name="unknown",
            parameters={},
            reasoning="No patterns matched the query",
            timestamp="",
        )

    def _apply_context(self, intent: Intent, user_query: str) -> Intent:
        """Apply conversation context to refine intent classification.

        Args:
            intent: The initially classified intent
            user_query: The user query

        Returns:
            Refined Intent object
        """
        # Check for pronouns that reference previous intent
        if any(
            pronoun in user_query.lower()
            for pronoun in ["it", "that", "this", "same", "more"]
        ):
            if self.conversation_history:
                last_turn = self.conversation_history[-1]
                if last_turn.intent:
                    # Use context for refinement
                    intent.reasoning += " [Context: Referencing previous query]"
                    intent.confidence = min(intent.confidence * 1.1, 0.99)

        return intent

    def _extract_search_terms(self, query: str) -> Dict[str, Any]:
        """Extract search terms from query."""
        # Simple extraction - can be enhanced
        keywords = query.split()
        terms = [
            w for w in keywords if len(w) > 2 and w.lower() not in ["search", "product", "find"]
        ]
        return {"query": " ".join(terms) if terms else query}

    def _extract_product_id(self, query: str) -> Optional[str]:
        """Extract product ID from query."""
        match = re.search(r"product[_ ]?id[:\s]+(\d+)", query, re.IGNORECASE)
        if match:
            return match.group(1)
        match = re.search(r"#(\d+)", query)
        if match:
            return match.group(1)
        return None

    def _extract_order_id(self, query: str) -> Optional[str]:
        """Extract order ID from query."""
        match = re.search(r"order[_ ]?id[:\s]+([A-Z0-9]+)", query, re.IGNORECASE)
        if match:
            return match.group(1)
        match = re.search(r"ORD[_-]?(\d+)", query, re.IGNORECASE)
        if match:
            return f"ORD-{match.group(1)}"
        return None

    def _extract_category(self, query: str) -> Optional[str]:
        """Extract product category from query."""
        categories = ["electronics", "accessories", "office"]
        for cat in categories:
            if cat.lower() in query.lower():
                return cat
        return None

    def _extract_user_id(self, query: str) -> Optional[str]:
        """Extract user ID from query."""
        match = re.search(r"user[_ ]?id[:\s]+([A-Za-z0-9]+)", query, re.IGNORECASE)
        if match:
            return match.group(1)
        match = re.search(r"for[_ ]?user[_ ]([A-Za-z0-9]+)", query, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _extract_status(self, query: str) -> Optional[str]:
        """Extract status from query."""
        statuses = ["pending", "processing", "shipped", "delivered", "cancelled"]
        for status in statuses:
            if status.lower() in query.lower():
                return status
        return None

    def _extract_order_parameters(self, query: str) -> Dict[str, Any]:
        """Extract order creation parameters from query."""
        return {
            "user_id": self._extract_user_id(query) or "current_user",
            "items": [],
            "total_price": 0.0,
        }

    def _extract_payment_parameters(self, query: str) -> Dict[str, Any]:
        """Extract payment parameters from query."""
        return {
            "user_id": self._extract_user_id(query) or "current_user",
            "amount": self._extract_amount(query) or 100.0,
            "payment_method": self._extract_payment_method(query).get(
                "payment_method", "credit_card"
            ),
        }

    def _extract_refund_parameters(self, query: str) -> Dict[str, Any]:
        """Extract refund parameters from query."""
        return {
            "payment_id": self._extract_payment_id(query) or "unknown",
            "amount": self._extract_amount(query),
            "reason": "Customer requested refund",
        }

    def _extract_payment_method(self, query: str) -> Dict[str, Any]:
        """Extract payment method from query."""
        methods = ["credit_card", "debit_card", "paypal", "bank_transfer"]
        for method in methods:
            if method.replace("_", " ").lower() in query.lower():
                return {"payment_method": method}
        return {"payment_method": "credit_card"}

    def _extract_inventory_parameters(self, query: str) -> Dict[str, Any]:
        """Extract inventory parameters from query."""
        return {
            "product_id": self._extract_product_id(query) or "unknown",
            "quantity": self._extract_quantity(query) or 1,
        }

    def _extract_stock_parameters(self, query: str) -> Dict[str, Any]:
        """Extract stock check parameters from query."""
        return {
            "product_id": self._extract_product_id(query) or "unknown",
        }

    def _extract_stock_update_parameters(self, query: str) -> Dict[str, Any]:
        """Extract stock update parameters from query."""
        return {
            "product_id": self._extract_product_id(query) or "unknown",
            "change": self._extract_quantity(query) or 1,
        }

    def _extract_reservation_id(self, query: str) -> Optional[str]:
        """Extract reservation ID from query."""
        match = re.search(r"reservation[_ ]?id[:\s]+([A-Z0-9]+)", query, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _extract_amount(self, query: str) -> Optional[float]:
        """Extract amount/price from query."""
        match = re.search(r"\$?([\d.]+)", query)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None

    def _extract_quantity(self, query: str) -> Optional[int]:
        """Extract quantity from query."""
        match = re.search(r"(?:quantity|amount|qty)[:\s]+(\d+)", query, re.IGNORECASE)
        if match:
            return int(match.group(1))
        match = re.search(r"(\d+)\s+(?:units|items|pieces)", query, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    def _extract_payment_id(self, query: str) -> Optional[str]:
        """Extract payment ID from query."""
        match = re.search(r"payment[_ ]?id[:\s]+([A-Z0-9]+)", query, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def get_classifier_stats(self) -> Dict[str, Any]:
        """Get classifier statistics."""
        if not self.conversation_history:
            return {
                "total_classifications": 0,
                "conversation_turns": 0,
                "average_confidence": 0.0,
            }

        confidences = [
            turn.intent.confidence
            for turn in self.conversation_history
            if turn.intent and turn.intent.confidence > 0
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        intent_counts = {}
        for turn in self.conversation_history:
            if turn.intent:
                intent_name = turn.intent.intent_name
                intent_counts[intent_name] = intent_counts.get(intent_name, 0) + 1

        return {
            "total_classifications": len(self.conversation_history),
            "conversation_turns": len(self.conversation_history),
            "average_confidence": avg_confidence,
            "intent_distribution": intent_counts,
            "unique_intents": len(intent_counts),
        }


async def main():
    """Example usage of the intent classifier."""
    # Initialize with available tools
    available_tools = {
        "product_service": [
            "SearchProducts",
            "GetProduct",
            "ListCategories",
            "GetProductsByCategory",
        ],
        "order_service": ["CreateOrder", "GetOrder", "ListOrders", "CancelOrder", "UpdateOrderStatus"],
        "payment_service": [
            "ProcessPayment",
            "RefundPayment",
            "GetPaymentHistory",
            "ValidatePaymentMethod",
        ],
        "inventory_service": [
            "CheckStock",
            "ReserveInventory",
            "ReleaseReservation",
            "UpdateStock",
            "GetInventoryStats",
        ],
    }

    classifier = IntentClassifier(available_tools)

    # Example queries
    test_queries = [
        "Search for laptop in electronics",
        "What's the status of order ORD-001?",
        "Process a payment of $99.99 using credit card",
        "Check inventory for product #123",
    ]

    print("🤖 Intent Classification Examples\n")

    for query in test_queries:
        intent = classifier.classify_intent(query)
        turn = ConversationTurn(
            user_query=query,
            intent=intent,
            response="Processing...",
            timestamp=datetime.now().isoformat(),
        )
        classifier.add_conversation_turn(turn)

        print(f"📝 Query: {query}")
        print(f"   Intent: {intent.intent_name}")
        print(f"   Confidence: {intent.confidence:.2%}")
        print(f"   Service: {intent.service_name}")
        print(f"   Tool: {intent.tool_name}")
        print(f"   Parameters: {intent.parameters}")
        print()

    # Show statistics
    print("📊 Classification Statistics:")
    stats = classifier.get_classifier_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
