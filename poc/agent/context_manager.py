"""
Context Manager Module - Phase 3
Manages conversation context and token counting for multi-turn interactions.

This module handles:
- Token counting using tiktoken
- Multi-turn conversation tracking
- Context windowing strategies (sliding window, summarization)
- Memory optimization
- Context pruning and summarization
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
import json

try:
    import tiktoken
except ImportError:
    tiktoken = None

logger = logging.getLogger(__name__)

# Default model for token counting
DEFAULT_MODEL = "gpt-3.5-turbo"


@dataclass
class ContextTurn:
    """A single turn in the conversation context."""

    turn_number: int
    user_query: str
    response: Optional[str] = None
    tokens_used: int = 0
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def get_total_tokens(self) -> int:
        """Get total tokens for this turn (query + response)."""
        return self.tokens_used


@dataclass
class ContextWindow:
    """Represents a window of context for the conversation."""

    window_size: int
    max_tokens: int
    turns: List[ContextTurn] = field(default_factory=list)
    total_tokens: int = 0

    def add_turn(self, turn: ContextTurn) -> bool:
        """Add a turn to the window.

        Args:
            turn: ContextTurn to add

        Returns:
            True if turn was added, False if window is full
        """
        potential_tokens = self.total_tokens + turn.get_total_tokens()

        if potential_tokens > self.max_tokens and len(self.turns) > 0:
            return False

        self.turns.append(turn)
        self.total_tokens += turn.get_total_tokens()
        return True

    def remove_oldest(self) -> Optional[ContextTurn]:
        """Remove the oldest turn from the window.

        Returns:
            The removed turn or None if window is empty
        """
        if self.turns:
            removed = self.turns.pop(0)
            self.total_tokens -= removed.get_total_tokens()
            return removed
        return None

    def get_context_string(self) -> str:
        """Get all turns as a single context string.

        Returns:
            Formatted context string
        """
        lines = []
        for turn in self.turns:
            lines.append(f"User: {turn.user_query}")
            if turn.response:
                lines.append(f"Assistant: {turn.response}")

        return "\n".join(lines)


class ContextManager:
    """Manages conversation context and token counting."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_context_tokens: int = 4096,
        context_strategy: str = "sliding_window",
    ):
        """Initialize the context manager.

        Args:
            model: Model name for token counting (gpt-3.5-turbo, gpt-4, etc.)
            max_context_tokens: Maximum tokens to keep in context
            context_strategy: Strategy for managing context ("sliding_window", "summarization")
        """
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.context_strategy = context_strategy
        self.encoding = self._get_encoding()
        self.all_turns: List[ContextTurn] = []
        self.context_window = ContextWindow(
            window_size=10, max_tokens=max_context_tokens
        )
        self.summaries: List[Dict[str, Any]] = []
        self.current_turn_number = 0

    def _get_encoding(self):
        """Get tiktoken encoding for the model.

        Returns:
            Tiktoken encoding object or None if tiktoken unavailable
        """
        if tiktoken is None:
            logger.warning(
                "tiktoken not available - token counting will be approximated"
            )
            return None

        try:
            return tiktoken.encoding_for_model(self.model)
        except KeyError:
            logger.warning(f"Model {self.model} not found, using cl100k_base encoding")
            return tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        if self.encoding is None:
            # Approximate: ~4 characters per token
            return len(text) // 4

        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            return len(text) // 4  # Fallback approximation

    def add_turn(
        self, user_query: str, response: Optional[str] = None
    ) -> ContextTurn:
        """Add a conversation turn to the context.

        Args:
            user_query: User's query
            response: Assistant's response (optional)

        Returns:
            The ContextTurn that was added
        """
        self.current_turn_number += 1

        # Count tokens
        query_tokens = self.count_tokens(user_query)
        response_tokens = self.count_tokens(response) if response else 0
        total_tokens = query_tokens + response_tokens

        turn = ContextTurn(
            turn_number=self.current_turn_number,
            user_query=user_query,
            response=response,
            tokens_used=total_tokens,
            timestamp=datetime.now().isoformat(),
            metadata={
                "query_tokens": query_tokens,
                "response_tokens": response_tokens,
            },
        )

        self.all_turns.append(turn)

        # Add to context window with overflow handling
        while not self.context_window.add_turn(turn) and len(self.context_window.turns) > 0:
            removed_turn = self.context_window.remove_oldest()
            if removed_turn:
                logger.debug(
                    f"Removed turn {removed_turn.turn_number} from context "
                    f"({removed_turn.get_total_tokens()} tokens)"
                )

        logger.info(
            f"Added turn {self.current_turn_number}: {query_tokens} query tokens + "
            f"{response_tokens} response tokens = {total_tokens} total"
        )

        return turn

    def get_current_context(self) -> str:
        """Get the current context string.

        Returns:
            Context as a string
        """
        return self.context_window.get_context_string()

    def get_context_summary(self) -> str:
        """Get a summary of the context (for very long conversations).

        Returns:
            Summary of recent context
        """
        if not self.context_window.turns:
            return "No context available."

        summary_lines = []

        # Add recent turns
        for turn in self.context_window.turns[-3:]:  # Last 3 turns
            summary_lines.append(f"- Turn {turn.turn_number}: {turn.user_query[:100]}")

        return "Recent context:\n" + "\n".join(summary_lines)

    def get_context_stats(self) -> Dict[str, Any]:
        """Get statistics about the current context.

        Returns:
            Dictionary with context statistics
        """
        context_tokens = self.context_window.total_tokens
        available_tokens = self.max_context_tokens - context_tokens

        return {
            "total_turns_all_time": len(self.all_turns),
            "turns_in_context": len(self.context_window.turns),
            "tokens_in_context": context_tokens,
            "max_tokens_allowed": self.max_context_tokens,
            "available_tokens": available_tokens,
            "token_utilization": context_tokens / self.max_context_tokens,
            "context_strategy": self.context_strategy,
        }

    def should_prune_context(self) -> bool:
        """Check if context should be pruned.

        Returns:
            True if pruning is recommended
        """
        available = self.max_context_tokens - self.context_window.total_tokens
        return available < (self.max_context_tokens * 0.1)  # Less than 10% available

    def prune_context_sliding_window(self, keep_recent: int = 5) -> int:
        """Prune context using sliding window strategy.

        Args:
            keep_recent: Number of recent turns to keep

        Returns:
            Number of turns pruned
        """
        pruned = 0

        # Keep only the most recent turns
        while len(self.context_window.turns) > keep_recent:
            self.context_window.remove_oldest()
            pruned += 1

        logger.info(f"Pruned {pruned} turns from context (sliding window)")
        return pruned

    def prune_context_by_importance(self, keep_count: int = 5) -> int:
        """Prune context by removing least important turns.

        Args:
            keep_count: Minimum number of turns to keep

        Returns:
            Number of turns pruned
        """
        if len(self.context_window.turns) <= keep_count:
            return 0

        pruned = 0

        # Calculate importance score for each turn
        # (simplified: older turns are less important)
        while len(self.context_window.turns) > keep_count:
            self.context_window.remove_oldest()
            pruned += 1

        logger.info(f"Pruned {pruned} turns from context (importance-based)")
        return pruned

    def create_context_summary(self) -> Dict[str, Any]:
        """Create a summary of the pruned context.

        Returns:
            Summary dictionary containing key information
        """
        summary = {
            "summary_timestamp": datetime.now().isoformat(),
            "turns_summarized": len(self.context_window.turns),
            "key_topics": self._extract_topics(),
            "total_tokens_summarized": self.context_window.total_tokens,
            "context_preview": self._get_summary_preview(),
        }

        self.summaries.append(summary)
        return summary

    def _extract_topics(self) -> List[str]:
        """Extract key topics from context.

        Returns:
            List of identified topics
        """
        topics = []

        for turn in self.context_window.turns:
            query_lower = turn.user_query.lower()

            if "product" in query_lower or "search" in query_lower:
                if "product" not in topics:
                    topics.append("product_queries")

            if "order" in query_lower:
                if "order" not in topics:
                    topics.append("order_management")

            if "payment" in query_lower or "pay" in query_lower:
                if "payment" not in topics:
                    topics.append("payment_processing")

            if "inventory" in query_lower or "stock" in query_lower:
                if "inventory" not in topics:
                    topics.append("inventory_management")

        return topics

    def _get_summary_preview(self) -> str:
        """Get a preview of the context summary.

        Returns:
            Preview string
        """
        if not self.context_window.turns:
            return "No context"

        first = self.context_window.turns[0].user_query[:50]
        last = self.context_window.turns[-1].user_query[:50]

        return f"From '{first}...' to '{last}...'"

    def get_remaining_token_budget(self) -> int:
        """Get remaining token budget for new input.

        Returns:
            Number of tokens available for new input
        """
        return self.max_context_tokens - self.context_window.total_tokens

    def can_fit_query(self, query: str) -> bool:
        """Check if a query can fit in the current context.

        Args:
            query: Query to check

        Returns:
            True if query can fit without pruning
        """
        query_tokens = self.count_tokens(query)
        return query_tokens <= self.get_remaining_token_budget()

    def get_all_turns(self) -> List[Dict[str, Any]]:
        """Get all turns (including pruned).

        Returns:
            List of all turns as dictionaries
        """
        return [turn.to_dict() for turn in self.all_turns]

    def get_context_turns(self) -> List[Dict[str, Any]]:
        """Get only turns in current context window.

        Returns:
            List of context window turns as dictionaries
        """
        return [turn.to_dict() for turn in self.context_window.turns]

    def export_context(self) -> Dict[str, Any]:
        """Export complete context state.

        Returns:
            Dictionary with full context data
        """
        return {
            "model": self.model,
            "max_context_tokens": self.max_context_tokens,
            "context_strategy": self.context_strategy,
            "current_turn_number": self.current_turn_number,
            "stats": self.get_context_stats(),
            "all_turns": self.get_all_turns(),
            "context_turns": self.get_context_turns(),
            "summaries": self.summaries,
        }


async def main():
    """Example usage of the context manager."""
    print("🧠 Context Manager Example\n")

    manager = ContextManager(
        model=DEFAULT_MODEL,
        max_context_tokens=2000,
        context_strategy="sliding_window",
    )

    # Simulate multi-turn conversation
    conversations = [
        ("Search for laptop", "Found 15 laptops in our database"),
        ("Show me the details", "Laptop details retrieved successfully"),
        ("Add to cart", "Item added to cart"),
        ("Check inventory", "500 units in stock"),
        ("Process payment", "Payment processed successfully"),
    ]

    for query, response in conversations:
        turn = manager.add_turn(query, response)
        print(f"Turn {turn.turn_number}:")
        print(f"  Query tokens: {turn.metadata['query_tokens']}")
        print(f"  Response tokens: {turn.metadata['response_tokens']}")
        print()

    # Show statistics
    print("📊 Context Statistics:")
    stats = manager.get_context_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")

    print()
    print("📝 Current Context:")
    print(manager.get_current_context())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
