"""
baselines/greedy_defender.py
Greedy baseline defender that isolates the highest-vulnerability neighbour
of the most-recently alerted node. Represents a simple reactive strategy
without lookahead or belief tracking.
"""
from __future__ import annotations

from typing import Optional

from sentinelnet.agents.base_agent import BaseAgent
from sentinelnet.environment.events import ActionEvent, AlertEvent
from sentinelnet.environment.network_graph import NetworkGraph, NodeState
from sentinelnet.coordination.message_bus import MessageBus


class GreedyDefender(BaseAgent):
    """Greedy heuristic defender.

    Strategy:
      1. Read the latest alert from the bus.
      2. Find the highest-vulnerability unpatched neighbour of the alert node.
      3. Isolate that neighbour immediately.

    If there is no alert, restore the oldest isolated node (if any) to
    maintain uptime.
    """

    def __init__(self, bus: Optional[MessageBus] = None) -> None:
        super().__init__(name="GreedyDefender", bus=bus)
        self._isolated_history: list = []
        self._total_cost: int = 0

    def step(self, graph: NetworkGraph) -> Optional[ActionEvent]:
        self.tick += 1
        alert: Optional[AlertEvent] = self.bus.latest("alerts") if self.bus else None

        if alert and alert.node:
            # Find highest-vuln neighbour to isolate
            alert_node = alert.node
            if alert_node not in graph.all_nodes():
                return None

            neighbors = [
                (nb, graph.node_data(nb))
                for nb in graph.g.successors(alert_node)
                if graph.node_data(nb).state not in (NodeState.ISOLATED, NodeState.COMPROMISED)
            ]

            if not neighbors:
                return None

            # Pick the neighbour with the highest effective vulnerability
            target_nb, target_data = max(neighbors, key=lambda x: x[1].effective_vuln)

            graph.isolate(target_nb)
            self._isolated_history.append(target_nb)
            self._total_cost += 1

            event = ActionEvent(
                tick=self.tick,
                action_type="isolate",
                node=target_nb,
                agent=self.name,
                cost=1,
            )
            if self.bus:
                self.bus.publish("actions", event, tick=self.tick)
            return event

        else:
            # No alert — restore oldest isolated node to maintain uptime
            if self._isolated_history:
                restore_node = self._isolated_history.pop(0)
                if restore_node in graph.all_nodes():
                    graph.restore(restore_node)
                    self._total_cost += 1
                    event = ActionEvent(
                        tick=self.tick,
                        action_type="restore",
                        node=restore_node,
                        agent=self.name,
                        cost=1,
                    )
                    if self.bus:
                        self.bus.publish("actions", event, tick=self.tick)
                    return event
            return None

    def on_reset(self) -> None:
        super().on_reset()
        self._isolated_history = []
        self._total_cost = 0
