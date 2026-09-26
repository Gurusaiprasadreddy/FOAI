"""
coordination/message_bus.py
Simple pub/sub blackboard for Blue agent coordination.

Protocol per tick:
  1. Environment emits alerts → Red posts to 'alerts'.
  2. Monitor reads 'alerts', posts updated belief to 'belief'.
  3. Response reads 'belief', posts actions to 'actions', executes them.
  4. Every N ticks, Patch Scheduler reads 'pending_jobs', posts to 'patch_plan'.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional


class MessageBus:
    """Lightweight publish/subscribe blackboard shared by all Blue agents.

    Topics:
        - ``alerts``     : List[AlertEvent] — noisy IDS alerts from the environment
        - ``belief``     : dict{node -> probability} — Monitor's belief state
        - ``actions``    : List[ActionEvent] — Response agent's chosen actions
        - ``patch_plan`` : List[str] — Patch Scheduler's ordered job list
    """

    VALID_TOPICS = {"alerts", "belief", "actions", "patch_plan"}

    def __init__(self) -> None:
        self._topics: Dict[str, List[Any]] = defaultdict(list)
        self._tick_index: Dict[str, int] = {}  # tick of last publish per topic

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish(self, topic: str, message: Any, tick: int = 0) -> None:
        """Append a message to a topic queue.

        Args:
            topic: Topic name (one of VALID_TOPICS or custom).
            message: Any serialisable Python object.
            tick: Current simulation tick for debugging / ordering.
        """
        self._topics[topic].append(message)
        self._tick_index[topic] = tick

    # ------------------------------------------------------------------
    # Consuming
    # ------------------------------------------------------------------

    def latest(self, topic: str) -> Optional[Any]:
        """Return the most-recently published message on a topic."""
        msgs = self._topics.get(topic, [])
        return msgs[-1] if msgs else None

    def all_messages(self, topic: str) -> List[Any]:
        """Return all accumulated messages on a topic (full history)."""
        return list(self._topics.get(topic, []))

    def since_tick(self, topic: str, tick: int) -> List[Any]:
        """Return messages posted at or after *tick*.

        .. note::
            This requires messages to be tuples of (tick, payload).
            If messages are plain objects, use :meth:`all_messages` instead.
        """
        result = []
        for msg in self._topics.get(topic, []):
            if isinstance(msg, tuple) and len(msg) >= 2 and msg[0] >= tick:
                result.append(msg)
        return result

    def clear(self, topic: str) -> None:
        """Clear all messages on a topic (call between ticks to avoid unbounded growth)."""
        self._topics[topic] = []

    def reset(self) -> None:
        """Clear the entire bus — call at the start of each scenario run."""
        self._topics = defaultdict(list)
        self._tick_index = {}

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def topic_length(self, topic: str) -> int:
        return len(self._topics.get(topic, []))

    def last_tick(self, topic: str) -> Optional[int]:
        return self._tick_index.get(topic)

    def __repr__(self) -> str:
        summary = {t: len(v) for t, v in self._topics.items()}
        return f"MessageBus({summary})"
