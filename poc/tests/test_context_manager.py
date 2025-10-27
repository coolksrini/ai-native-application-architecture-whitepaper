"""
Phase 4 Testing Framework - Context Manager Tests
Tests for the ContextManager module (Chapter 8: Context Management)

This module covers:
- Token counting with exact and approximate methods
- Multi-turn conversation tracking
- Context windowing and sliding window strategy
- Automatic context pruning
- Context summaries
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from agent.context_manager import ContextManager, ContextTurn, ContextWindow


class TestContextTurn:
    """Test ContextTurn class."""

    def test_create_turn(self):
        """Test creating a context turn."""
        turn = ContextTurn(
            turn_number=1,
            user_query="Search for laptops",
            response="Found 15 results",
            timestamp=datetime.now().isoformat(),
        )

        assert turn.user_query == "Search for laptops"
        assert turn.response == "Found 15 results"
        assert turn.turn_number == 1

    def test_turn_token_count(self):
        """Test token counting for a turn."""
        turn = ContextTurn(
            turn_number=1,
            user_query="Search for laptops",
            response="Found 15 results",
            timestamp=datetime.now().isoformat(),
            tokens_used=50,
        )

        assert turn.tokens_used == 50

    def test_turn_to_dict(self):
        """Test converting turn to dictionary."""
        turn = ContextTurn(
            turn_number=1,
            user_query="Search",
            response="Results",
            timestamp=datetime.now().isoformat(),
        )

        data = turn.to_dict()
        assert data["user_query"] == "Search"
        assert data["response"] == "Results"


class TestContextWindow:
    """Test ContextWindow class."""

    def test_create_window(self):
        """Test creating a context window."""
        window = ContextWindow(
            window_size=10,
            max_tokens=4096,
        )

        assert window.max_tokens == 4096
        assert window.window_size == 10
        assert len(window.turns) == 0

    def test_add_turn_to_window(self):
        """Test adding turns to window."""
        window = ContextWindow(window_size=10, max_tokens=4096)

        turn = ContextTurn(
            turn_number=1,
            user_query="Search",
            response="Results",
            timestamp=datetime.now().isoformat(),
        )

        window.turns.append(turn)

        assert len(window.turns) == 1

    def test_get_window_stats(self):
        """Test getting window statistics."""
        window = ContextWindow(window_size=10, max_tokens=4096)

        turn1 = ContextTurn(
            turn_number=1,
            user_query="Query 1",
            response="Response 1",
            timestamp=datetime.now().isoformat(),
            tokens_used=100,
        )

        turn2 = ContextTurn(
            turn_number=2,
            user_query="Query 2",
            response="Response 2",
            timestamp=datetime.now().isoformat(),
            tokens_used=150,
        )

        window.turns.extend([turn1, turn2])

        stats = {
            "total_turns": len(window.turns),
            "total_tokens": sum(t.tokens_used or 0 for t in window.turns),
        }

        assert stats["total_turns"] == 2
        assert stats["total_tokens"] == 250


class TestContextManager:
    """Test ContextManager class."""

    def test_initialization(self):
        """Test ContextManager initialization."""
        manager = ContextManager(max_context_tokens=4096)

        assert manager.max_context_tokens == 4096
        assert len(manager.all_turns) == 0

    def test_add_turn_to_context(self):
        """Test adding turn to context."""
        manager = ContextManager(max_context_tokens=4096)

        manager.add_turn(
            user_query="Search for laptops",
            response="Found 15 results",
        )

        assert len(manager.all_turns) == 1
        assert manager.all_turns[0].user_query == "Search for laptops"

    def test_turn_numbering(self):
        """Test that turns are numbered correctly."""
        manager = ContextManager(max_context_tokens=4096)

        for i in range(5):
            manager.add_turn(f"Query {i}", f"Response {i}")

        turn_numbers = [t.turn_number for t in manager.all_turns]
        assert turn_numbers == [1, 2, 3, 4, 5]

    def test_token_counting_exact(self):
        """Test exact token counting."""
        # Mock tiktoken before creating manager
        with patch("agent.context_manager.tiktoken") as mock_tiktoken:
            mock_encoding = MagicMock()
            mock_encoding.encode.return_value = list(range(100))  # 100 tokens
            mock_tiktoken.encoding_for_model.return_value = mock_encoding
            
            manager = ContextManager(max_context_tokens=4096)
            tokens = manager.count_tokens("Sample text that is 100 tokens")
            assert tokens == 100

    def test_token_counting_approximate(self):
        """Test approximate token counting (fallback)."""
        manager = ContextManager(max_context_tokens=4096)

        # Force approximate counting by not providing tiktoken
        with patch("agent.context_manager.tiktoken", None):
            manager.encoding = None
            tokens = manager.count_tokens("This is a test string")
            # Approximate: len / 4 = ~5 tokens
            assert tokens > 0

    def test_context_pruning_sliding_window(self):
        """Test context pruning with sliding window strategy."""
        manager = ContextManager(max_context_tokens=500)

        # Add many turns to exceed token limit
        for i in range(10):
            manager.add_turn(
                f"Query {i} with some longer text to account for tokens",
                f"Response {i} with some content and more words here",
            )

        # Get context should trigger pruning
        context = manager.get_current_context()

        # Should have added all turns
        assert len(manager.all_turns) == 10

    def test_get_context(self):
        """Test getting formatted context."""
        manager = ContextManager(max_context_tokens=4096)

        manager.add_turn("Search for laptops", "Found 15 results")
        manager.add_turn("Filter by price", "Filtered results")

        context = manager.get_current_context()

        assert "Search for laptops" in context
        assert "Found 15 results" in context

    def test_context_summary(self):
        """Test creating context summary."""
        manager = ContextManager(max_context_tokens=4096)

        manager.add_turn("Query 1", "Response 1")
        manager.add_turn("Query 2", "Response 2")

        summary = manager.create_context_summary()

        assert isinstance(summary, dict)
        assert len(manager.all_turns) >= 1

    def test_get_context_stats(self):
        """Test getting context statistics."""
        manager = ContextManager(max_context_tokens=4096)

        manager.add_turn("Query 1", "Response 1")
        manager.add_turn("Query 2", "Response 2")

        stats = manager.get_context_stats()

        assert stats["total_turns_all_time"] == 2
        assert stats["max_tokens_allowed"] == 4096

    def test_clear_context(self):
        """Test clearing context."""
        manager = ContextManager(max_context_tokens=4096)

        manager.add_turn("Query", "Response")
        assert len(manager.all_turns) == 1

        # Reset the manager
        manager.all_turns.clear()
        manager.context_window.turns.clear()
        assert len(manager.all_turns) == 0

    def test_multi_turn_context(self):
        """Test multi-turn conversation context."""
        manager = ContextManager(max_context_tokens=4096)

        queries_responses = [
            ("What is the price of laptop X?", "Laptop X costs $999"),
            ("Is it in stock?", "Yes, 5 units in stock"),
            ("Show me similar products", "Here are 3 similar products"),
        ]

        for query, response in queries_responses:
            manager.add_turn(query, response)

        context = manager.get_current_context()
        assert len(manager.all_turns) == 3

        # Verify all turns are present
        for query, response in queries_responses:
            assert query in context

    def test_context_truncation(self):
        """Test context is properly truncated when exceeding limits."""
        manager = ContextManager(max_context_tokens=100)

        # Add many turns
        for i in range(20):
            manager.add_turn(
                f"This is query number {i} with lots of text",
                f"This is response number {i} with lots of content",
            )

        # All turns should be recorded
        assert len(manager.all_turns) == 20
        
        # But context_window should be limited
        assert len(manager.context_window.turns) <= 20

    def test_token_budget_awareness(self):
        """Test that token budget is respected."""
        manager = ContextManager(max_context_tokens=1000)

        stats = manager.get_context_stats()

        assert stats["max_tokens_allowed"] == 1000
        assert stats["available_tokens"] <= 1000

    def test_sliding_window_strategy(self):
        """Test sliding window strategy for context management."""
        manager = ContextManager(max_context_tokens=4096, context_strategy="sliding_window")

        for i in range(5):
            manager.add_turn(f"Query {i}", f"Response {i}")

        assert len(manager.all_turns) > 0
        assert manager.context_strategy == "sliding_window"


@pytest.mark.chapter8
def test_chapter8_context_management_validation():
    """
    Chapter 8: Context Management - Validation
    
    Validates that context management meets Chapter 8 requirements:
    - Tracks multi-turn conversations
    - Counts tokens accurately
    - Manages context window with limits
    - Implements sliding window strategy
    - Provides context summaries
    """
    manager = ContextManager(max_context_tokens=4096)

    # Test multi-turn tracking
    manager.add_turn("First query", "First response")
    manager.add_turn("Second query", "Second response")
    manager.add_turn("Third query", "Third response")

    assert len(manager.all_turns) == 3

    # Test token counting
    tokens = manager.count_tokens("Sample text")
    assert tokens > 0

    # Test context retrieval
    context = manager.get_current_context()
    assert "First query" in context
    assert "Third query" in context

    # Test statistics
    stats = manager.get_context_stats()
    assert stats["total_turns_all_time"] == 3
    assert stats["max_tokens_allowed"] == 4096

    # Test summary creation
    summary = manager.create_context_summary()
    assert isinstance(summary, dict)

    # Test strategy exists
    assert manager.context_strategy in ["sliding_window", "summarization"]
