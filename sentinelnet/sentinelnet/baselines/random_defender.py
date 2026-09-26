"""
baselines/random_defender.py
Baseline defender that takes a uniformly random action each tick.
Used as the weakest baseline to show improvement from algorithmic agents.
"""
from __future__ import annotations

import random
from typing import Optional

from sentinelnet.agents.base_agent import BaseAgent
from sentinelnet.environment.events import ActionEvent
from sentinelnet.environment.network_graph import NetworkGraph, NodeState
from sentinelnet.coordination.message_bus import MessageBus


class RandomDefender(BaseAgent):
    """Picks a random defensive action on a random non-compromised node each tick.

    Actions: isolate, patch, restore, noop (25% each).
    """

    ACTIONS = ["isolate", "patch", "restore", "noop"]

    def __init__(self, seed: int = 0, bus: Optional[MessageBus] = None) -> None:
        super().__init__(name="RandomDefender", bus=bus)
        self.rng = random.Random(seed)
        self._total_cost: int = 0

    def step(self, graph: NetworkGraph) -> Optional[ActionEvent]:
        self.tick += 1
        nodes = graph.all_nodes()
        if not nodes:
            return None

        action_type = self.rng.choice(self.ACTIONS)
        node = self.rng.choice(nodes)

        if action_type == "noop":
            return None

        success = graph.apply_action(action_type, node)
        if not success:
            return None

        self._total_cost += 1
        event = ActionEvent(
            tick=self.tick,
            action_type=action_type,
            node=node,
            agent=self.name,
            cost=1,
        )
        if self.bus:
            self.bus.publish("actions", event, tick=self.tick)
        return event

    def on_reset(self) -> None:
        super().on_reset()
        self._total_cost = 0
