"""
Phase 4 Testing Framework - Intent Classifier Tests
Tests for the IntentClassifier module (Chapter 2: Intent Recognition)

This module covers:
- Intent classification accuracy
- Parameter extraction
- Confidence scoring
- Multi-turn context awareness
"""

import pytest
from datetime import datetime

from agent.intent_classifier import (
    IntentClassifier,
    Intent,
    ConversationTurn,
)


class TestIntent:
    """Test Intent class."""

    def test_create_intent(self):
        """Test creating an intent."""
        intent = Intent(
            intent_name="search_products",
            confidence=0.95,
            service_name="product_service",
            tool_name="SearchProducts",
            parameters={"query": "laptop"},
            reasoning="Found product search keywords",
            timestamp=datetime.now().isoformat(),
        )

        assert intent.intent_name == "search_products"
        assert intent.confidence == 0.95
        assert intent.service_name == "product_service"

    def test_intent_to_dict(self):
        """Test converting intent to dictionary."""
        intent = Intent(
            intent_name="search_products",
            confidence=0.95,
            service_name="product_service",
            tool_name="SearchProducts",
            parameters={"query": "laptop"},
            reasoning="Found product search keywords",
            timestamp=datetime.now().isoformat(),
        )

        data = intent.to_dict()
        assert data["intent_name"] == "search_products"
        assert data["confidence"] == 0.95


class TestConversationTurn:
    """Test ConversationTurn class."""

    def test_create_turn(self):
        """Test creating a conversation turn."""
        intent = Intent(
            intent_name="search_products",
            confidence=0.95,
            service_name="product_service",
            tool_name="SearchProducts",
            parameters={"query": "laptop"},
            reasoning="Found product search keywords",
            timestamp=datetime.now().isoformat(),
        )

        turn = ConversationTurn(
            user_query="Search for laptops",
            intent=intent,
            response="Found 15 results",
        )

        assert turn.user_query == "Search for laptops"
        assert turn.intent == intent
        assert turn.response == "Found 15 results"


class TestIntentClassifier:
    """Test IntentClassifier class."""

    def test_initialization(self):
        """Test IntentClassifier initialization."""
        available_tools = {
            "product_service": ["SearchProducts", "GetProduct"],
            "order_service": ["CreateOrder", "GetOrder"],
        }

        classifier = IntentClassifier(available_tools)

        assert classifier.available_tools == available_tools
        assert len(classifier.conversation_history) == 0

    def test_classify_search_products(self):
        """Test classifying product search intent."""
        classifier = IntentClassifier()

        intent = classifier.classify_intent("Search for laptops")

        assert intent.intent_name == "search_products"
        assert intent.service_name == "product_service"
        assert intent.tool_name == "SearchProducts"
        assert intent.confidence >= 0.85

    def test_classify_get_product(self):
        """Test classifying product get intent."""
        classifier = IntentClassifier()

        intent = classifier.classify_intent("Get product_id: 123")

        assert intent.intent_name == "get_product"
        assert intent.service_name == "product_service"
        assert intent.tool_name == "GetProduct"

    def test_classify_list_categories(self):
        """Test classifying list categories intent."""
        classifier = IntentClassifier()

        intent = classifier.classify_intent("Show all categories")

        assert intent.intent_name == "list_categories"
        assert intent.service_name == "product_service"

    def test_classify_create_order(self):
        """Test classifying create order intent."""
        classifier = IntentClassifier()

        intent = classifier.classify_intent("Create order for user")

        assert intent.intent_name == "create_order"
        assert intent.service_name == "order_service"
        assert intent.tool_name == "CreateOrder"

    def test_classify_get_order(self):
        """Test classifying get order intent."""
        classifier = IntentClassifier()

        intent = classifier.classify_intent("Show order ORD-001")

        assert intent.intent_name == "get_order"
        assert intent.service_name == "order_service"

    def test_classify_process_payment(self):
        """Test classifying process payment intent."""
        classifier = IntentClassifier()

        intent = classifier.classify_intent("Process a payment of $99.99")

        assert intent.intent_name == "process_payment"
        assert intent.service_name == "payment_service"

    def test_classify_check_stock(self):
        """Test classifying check stock intent."""
        classifier = IntentClassifier()

        intent = classifier.classify_intent("Check inventory levels")

        assert intent.intent_name == "check_stock"
        assert intent.service_name == "inventory_service"

    def test_classify_unknown_intent(self):
        """Test classifying unknown intent."""
        classifier = IntentClassifier()

        intent = classifier.classify_intent("xyz random query with no keywords")

        assert intent.intent_name == "unknown"
        assert intent.confidence < 0.5

    def test_parameter_extraction_search(self):
        """Test extracting search parameters."""
        classifier = IntentClassifier()

        intent = classifier.classify_intent("Find gaming laptops")

        assert "query" in intent.parameters

    def test_parameter_extraction_product_id(self):
        """Test extracting product ID."""
        classifier = IntentClassifier()

        intent = classifier.classify_intent("Get product #123")

        assert "product_id" in intent.parameters
        assert intent.parameters["product_id"] == "123"

    def test_parameter_extraction_category(self):
        """Test extracting category parameter."""
        classifier = IntentClassifier()

        intent = classifier.classify_intent("Filter products by category electronics")

        assert "category" in intent.parameters
        assert intent.parameters["category"] == "electronics"

    def test_add_conversation_turn(self):
        """Test adding turn to conversation history."""
        classifier = IntentClassifier()

        intent = Intent(
            intent_name="search_products",
            confidence=0.95,
            service_name="product_service",
            tool_name="SearchProducts",
            parameters={"query": "laptop"},
            reasoning="Test",
            timestamp=datetime.now().isoformat(),
        )

        turn = ConversationTurn(
            user_query="Search for laptops",
            intent=intent,
            response="Found results",
        )

        classifier.add_conversation_turn(turn)

        assert len(classifier.conversation_history) == 1
        assert classifier.conversation_history[0].user_query == "Search for laptops"

    def test_conversation_history_limit(self):
        """Test conversation history respects max limit."""
        classifier = IntentClassifier()
        classifier.max_history = 5

        intent = Intent(
            intent_name="search_products",
            confidence=0.95,
            service_name="product_service",
            tool_name="SearchProducts",
            parameters={"query": "test"},
            reasoning="Test",
            timestamp=datetime.now().isoformat(),
        )

        # Add 10 turns
        for i in range(10):
            turn = ConversationTurn(
                user_query=f"Query {i}",
                intent=intent,
                response="Response",
            )
            classifier.add_conversation_turn(turn)

        # Should only keep last 5
        assert len(classifier.conversation_history) == 5

    def test_get_conversation_context(self):
        """Test getting conversation context."""
        classifier = IntentClassifier()

        intent = Intent(
            intent_name="search_products",
            confidence=0.95,
            service_name="product_service",
            tool_name="SearchProducts",
            parameters={"query": "test"},
            reasoning="Test",
            timestamp=datetime.now().isoformat(),
        )

        turn = ConversationTurn(
            user_query="Search for laptops",
            intent=intent,
            response="Found 15 results",
        )

        classifier.add_conversation_turn(turn)

        context = classifier.get_conversation_context()
        assert "Search for laptops" in context
        assert "Found 15 results" in context

    def test_get_classifier_stats(self):
        """Test getting classifier statistics."""
        classifier = IntentClassifier()

        intent = Intent(
            intent_name="search_products",
            confidence=0.95,
            service_name="product_service",
            tool_name="SearchProducts",
            parameters={"query": "test"},
            reasoning="Test",
            timestamp=datetime.now().isoformat(),
        )

        turn = ConversationTurn(
            user_query="Search for laptops",
            intent=intent,
            response="Found 15 results",
        )

        classifier.add_conversation_turn(turn)

        stats = classifier.get_classifier_stats()
        assert stats["total_classifications"] == 1
        assert "search_products" in stats["intent_distribution"]

    def test_confidence_scoring(self):
        """Test that confidence scores are reasonable."""
        classifier = IntentClassifier()

        # High confidence for clear intent
        intent1 = classifier.classify_intent("Search for laptops in electronics")
        assert intent1.confidence >= 0.85

        # Lower confidence for ambiguous
        intent2 = classifier.classify_intent("something random")
        assert intent2.confidence < 0.5

    def test_multiple_intents_in_sequence(self):
        """Test classifying multiple intents in sequence."""
        classifier = IntentClassifier()

        queries = [
            "Search for laptops",
            "Show product 123",
            "Create an order",
            "Process payment",
            "Check stock",
        ]

        for query in queries:
            intent = classifier.classify_intent(query)
            assert intent.intent_name != "unknown"

        assert len(classifier.conversation_history) == 0  # Not added yet


@pytest.mark.chapter2
def test_chapter2_intent_recognition_validation():
    """
    Chapter 2: Intent Recognition - Validation
    
    Validates that intent classification meets Chapter 2 requirements:
    - Recognizes user intent from natural language
    - Extracts parameters accurately
    - Provides confidence scores
    - Maintains conversation context
    """
    classifier = IntentClassifier()

    # Test intent recognition
    intent = classifier.classify_intent("Search for laptops")
    assert intent.intent_name == "search_products"
    assert intent.confidence > 0.8

    # Test parameter extraction
    assert "query" in intent.parameters

    # Test multiple intents
    test_cases = [
        ("Search for product", "search_products"),
        ("Create order", "create_order"),
        ("Process payment", "process_payment"),
        ("Check inventory", "check_stock"),
    ]

    for query, expected_intent in test_cases:
        result = classifier.classify_intent(query)
        assert result.intent_name == expected_intent
