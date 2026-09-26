"""
agents/base_agent.py
Abstract base class for all SentinelNet agents.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from sentinelnet.environment.network_graph import NetworkGraph
    from sentinelnet.coordination.message_bus import MessageBus


class BaseAgent(ABC):
    """All agents (Red and Blue) inherit from this.

    Subclasses must implement :meth:`step`, which is called once per
    simulation tick by the :class:`~sentinelnet.environment.simulator.Simulator`.
    """

    def __init__(self, name: str, bus: Optional["MessageBus"] = None) -> None:
        self.name = name
        self.bus = bus
        self.tick: int = 0

    @abstractmethod
    def step(self, graph: "NetworkGraph") -> Any:
        """Advance the agent by one simulation tick.

        Args:
            graph: The current network graph (shared environment).

        Returns:
            Agent-specific result (e.g., an event, an action, or None).
        """

    def on_reset(self) -> None:
        """Called when the simulation resets — override to clear state."""
        self.tick = 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
