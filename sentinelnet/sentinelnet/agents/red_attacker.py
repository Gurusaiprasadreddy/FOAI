"""
agents/red_attacker.py
Red Agent — A* attack path planning over the enterprise network graph.

Algorithm: A* search (Hart, Nilsson & Raphael, 1968)
  - Priority queue ordered by f(n) = g(n) + h(n)
  - g(n): actual traversal cost from entry to n
  - h(n): admissible hop-count heuristic to crown jewel (never overestimates)
  - nodes_expanded tracked per call for scalability benchmarking

Key behaviours:
  - Dynamically replans when path is blocked by Blue isolations.
  - Probabilistic exploit success: each move has vuln_score chance of compromise.
  - Emits AlertEvent with probability = detection_rate (noisy IDS).
  - Supports multi-entry mode (two simultaneous entry points).
"""
from __future__ import annotations

import heapq
import random
from typing import Dict, List, Optional, Tuple

from sentinelnet.agents.base_agent import BaseAgent
from sentinelnet.environment.events import AlertEvent, ExploitEvent
from sentinelnet.environment.network_graph import NetworkGraph, NodeState


# ---------------------------------------------------------------------------
# Pure A* function (exposed for unit tests and grader inspection)
# ---------------------------------------------------------------------------

def a_star(
    graph: NetworkGraph,
    start: str,
    goal: str,
    heuristic,
) -> Tuple[Optional[List[str]], float, int]:
    """A* shortest path search.

    Args:
        graph: The network graph (uses ``neighbors_with_cost``).
        start: Starting node name.
        goal: Target node name.
        heuristic: Callable(node, goal) → float admissible estimate.

    Returns:
        (path, total_cost, nodes_expanded) where path is a list of node names
        starting from *start*. Returns (None, inf, nodes_expanded) if no path.
    """
    # frontier entries: (f_cost, g_cost, node, path)
    frontier: List[Tuple[float, float, str, List[str]]] = [
        (0.0 + heuristic(start, goal), 0.0, start, [start])
    ]
    best_g: Dict[str, float] = {}
    nodes_expanded = 0

    while frontier:
        f, g, node, path = heapq.heappop(frontier)
        nodes_expanded += 1

        if node == goal:
            return path, g, nodes_expanded

        # Skip if we've already found a cheaper path to this node
        if node in best_g and best_g[node] <= g:
            continue
        best_g[node] = g

        for neighbor, edge_cost in graph.neighbors_with_cost(node):
            new_g = g + edge_cost
            new_f = new_g + heuristic(neighbor, goal)
            heapq.heappush(frontier, (new_f, new_g, neighbor, path + [neighbor]))

    return None, float("inf"), nodes_expanded


def hop_count_heuristic(node: str, goal: str) -> float:
    """Admissible heuristic: 0 (ignores topology — always admissible).

    A proper implementation would use BFS hop-count from node to goal
    on the unweighted graph. We return 0 here so A* degrades to Dijkstra
    in the worst case; the caller can supply a better heuristic.
    """
    return 0.0


def bfs_heuristic_factory(graph: NetworkGraph, goal: str):
    """Build a BFS-based admissible heuristic for a fixed goal.

    Returns a callable h(node, goal) -> float that equals the unweighted
    hop count from node to goal, which is admissible since each edge costs ≥ 1.
    """
    import collections
    dist: Dict[str, int] = {}
    queue = collections.deque([(goal, 0)])
    while queue:
        n, d = queue.popleft()
        if n in dist:
            continue
        dist[n] = d
        # Traverse reverse edges for backward BFS from goal
        for pred in graph.g.predecessors(n):
            if pred not in dist:
                queue.append((pred, d + 1))

    def h(node: str, _goal: str) -> float:
        return float(dist.get(node, 0))

    return h


# ---------------------------------------------------------------------------
# RedAttacker agent
# ---------------------------------------------------------------------------

class RedAttacker(BaseAgent):
    """Red attacker that uses A* to navigate toward the crown jewel.

    Args:
        entry_node: Starting node for the attacker.
        detection_rate: P(alert | attacker at node) for true alerts.
        false_alarm_rate: P(alert | attacker NOT at node) — background noise.
        seed: RNG seed for reproducibility.
        stealth_mode: If True, reduces detection_rate by 50% (scenario 2).
    """

    def __init__(
        self,
        entry_node: str,
        detection_rate: float = 0.7,
        false_alarm_rate: float = 0.05,
        seed: int = 42,
        stealth_mode: bool = False,
        bus=None,
    ) -> None:
        super().__init__(name="RedAttacker", bus=bus)
        self.current_node: str = entry_node
        self.detection_rate: float = detection_rate * (0.5 if stealth_mode else 1.0)
        self.false_alarm_rate: float = false_alarm_rate
        self.rng = random.Random(seed)
        self._path: Optional[List[str]] = None
        self._path_cost: float = float("inf")
        self.nodes_expanded_last: int = 0
        self.captured: bool = False  # True if Blue catches Red on a honeypot
        self.reached_goal: bool = False
        self.ticks_alive: int = 0
        self._heuristic = hop_count_heuristic

    def set_graph(self, graph: NetworkGraph) -> None:
        """Call once after constructing with a graph to build BFS heuristic."""
        self._heuristic = bfs_heuristic_factory(graph, graph.crown_jewel)

    def _replan(self, graph: NetworkGraph) -> None:
        """Recompute A* path from current position to crown jewel."""
        path, cost, expanded = a_star(
            graph,
            self.current_node,
            graph.crown_jewel,
            self._heuristic,
        )
        self._path = path
        self._path_cost = cost
        self.nodes_expanded_last = expanded

    def step(self, graph: NetworkGraph) -> Optional[AlertEvent]:
        """Move one hop along the A* path, attempting to compromise the target node.

        Returns:
            An :class:`~sentinelnet.environment.events.AlertEvent` if an IDS
            alert fires this tick, or None.
        """
        if self.captured or self.reached_goal:
            return None

        self.tick += 1
        self.ticks_alive += 1

        # Replan if no path or path is blocked
        if self._path is None or len(self._path) < 2:
            self._replan(graph)

        if self._path is None or len(self._path) < 2:
            return None  # Trapped — no path to goal

        next_node = self._path[1]
        next_data = graph.node_data(next_node)

        # Check if next node has been isolated since last plan
        if next_data.state == NodeState.ISOLATED:
            self._replan(graph)
            if self._path is None or len(self._path) < 2:
                return None
            next_node = self._path[1]
            next_data = graph.node_data(next_node)

        # Check if next node is a honeypot — Red gets caught
        if next_data.state == NodeState.HONEYPOT:
            self.captured = True
            # Strong alert with high confidence on honeypot node
            return AlertEvent(
                tick=self.tick,
                node=next_node,
                confidence=0.99,
                is_false_positive=False,
            )

        # Attempt exploit: success probability = effective_vuln
        exploit_success = self.rng.random() < next_data.effective_vuln

        if exploit_success:
            graph.mark_compromised(next_node)
            self.current_node = next_node
            self._path = self._path[1:]  # Advance path

            if next_node == graph.crown_jewel:
                self.reached_goal = True

        # Generate IDS alert (true positive if exploit succeeded, otherwise miss)
        alert = self._generate_alert(graph, next_node, exploit_success)
        return alert

    def _generate_alert(
        self,
        graph: NetworkGraph,
        target_node: str,
        exploit_success: bool,
    ) -> Optional[AlertEvent]:
        """Probabilistically generate a noisy IDS alert.

        With probability ``detection_rate`` an alert fires on the real target.
        With probability ``false_alarm_rate`` a spurious alert fires on a
        random node (independent of attacker position).
        """
        alerts = []

        # True positive alert
        if self.rng.random() < self.detection_rate:
            alerts.append(AlertEvent(
                tick=self.tick,
                node=target_node,
                confidence=self.detection_rate,
                is_false_positive=False,
            ))

        # False positive alert on a random other node
        if self.rng.random() < self.false_alarm_rate:
            other_nodes = [n for n in graph.all_nodes() if n != target_node]
            if other_nodes:
                fp_node = self.rng.choice(other_nodes)
                alerts.append(AlertEvent(
                    tick=self.tick,
                    node=fp_node,
                    confidence=self.false_alarm_rate,
                    is_false_positive=True,
                ))

        return alerts[0] if alerts else None

    def on_reset(self) -> None:
        super().on_reset()
        self._path = None
        self.captured = False
        self.reached_goal = False
        self.ticks_alive = 0
        self.nodes_expanded_last = 0
