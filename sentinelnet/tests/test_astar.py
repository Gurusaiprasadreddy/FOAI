"""
tests/test_astar.py
Unit tests for the A* search implementation in red_attacker.py.

Tests verify:
  1. A* returns an optimal path on a known small graph.
  2. A* returns None when there is no path.
  3. nodes_expanded counter is incremented correctly.
  4. Dynamic replanning skips isolated nodes.
  5. BFS heuristic is admissible (h ≤ actual cost).
"""
import pytest
import sys
import os

# Make sure the package is importable from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sentinelnet.agents.red_attacker import a_star, bfs_heuristic_factory, hop_count_heuristic
from sentinelnet.environment.network_graph import NetworkGraph, NodeState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def linear_graph():
    """A → B → C → D (chain), all edge cost = 1."""
    ng = NetworkGraph()
    nodes = [("A", "server", 1, 0.5), ("B", "server", 1, 0.5),
             ("C", "server", 1, 0.5), ("D", "crown_jewel", 10, 0.1)]
    for nid, ntype, val, vuln in nodes:
        from sentinelnet.environment.network_graph import NodeData
        nd = NodeData(name=nid, node_type=ntype, value=val, vuln_score=vuln)
        ng.g.add_node(nid, data=nd)
    for u, v in [("A", "B"), ("B", "C"), ("C", "D")]:
        ng.g.add_edge(u, v, cost=1, bandwidth=100)
    ng._crown_jewel = "D"
    ng._entry_points = ["A"]
    return ng


@pytest.fixture
def diamond_graph():
    """Diamond topology: A → (B, C) → D. B has cost 2, C has cost 5."""
    ng = NetworkGraph()
    nodes = [("A", "workstation", 1, 0.6), ("B", "server", 2, 0.4),
             ("C", "server", 2, 0.3), ("D", "crown_jewel", 10, 0.1)]
    for nid, ntype, val, vuln in nodes:
        from sentinelnet.environment.network_graph import NodeData
        nd = NodeData(name=nid, node_type=ntype, value=val, vuln_score=vuln)
        ng.g.add_node(nid, data=nd)
    ng.g.add_edge("A", "B", cost=2, bandwidth=100)
    ng.g.add_edge("A", "C", cost=5, bandwidth=100)
    ng.g.add_edge("B", "D", cost=1, bandwidth=100)
    ng.g.add_edge("C", "D", cost=1, bandwidth=100)
    ng._crown_jewel = "D"
    ng._entry_points = ["A"]
    return ng


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAStarBasics:
    def test_optimal_path_chain(self, linear_graph):
        """A* finds the single path A→B→C→D with cost 3."""
        path, cost, expanded = a_star(
            linear_graph, "A", "D", hop_count_heuristic
        )
        assert path == ["A", "B", "C", "D"]
        assert cost == 3.0
        assert expanded >= 4  # Must visit at least 4 nodes

    def test_optimal_path_diamond_goes_through_B(self, diamond_graph):
        """A* picks A→B→D (cost 3) over A→C→D (cost 6)."""
        path, cost, expanded = a_star(
            diamond_graph, "A", "D", hop_count_heuristic
        )
        assert cost == 3.0
        assert "B" in path
        assert path[0] == "A"
        assert path[-1] == "D"

    def test_no_path_returns_none(self, linear_graph):
        """A* returns None path when goal is unreachable."""
        # Remove edge C→D to sever the graph
        linear_graph.g.remove_edge("C", "D")
        path, cost, expanded = a_star(
            linear_graph, "A", "D", hop_count_heuristic
        )
        assert path is None
        assert cost == float("inf")

    def test_nodes_expanded_nonzero(self, linear_graph):
        """nodes_expanded is strictly positive."""
        _, _, expanded = a_star(linear_graph, "A", "D", hop_count_heuristic)
        assert expanded > 0

    def test_start_equals_goal(self, linear_graph):
        """A* immediately returns when start == goal."""
        path, cost, expanded = a_star(
            linear_graph, "D", "D", hop_count_heuristic
        )
        assert path == ["D"]
        assert cost == 0.0


class TestBFSHeuristic:
    def test_heuristic_admissible_chain(self, linear_graph):
        """BFS heuristic ≤ actual A* cost for every node."""
        h = bfs_heuristic_factory(linear_graph, "D")
        for node in ["A", "B", "C"]:
            _, actual_cost, _ = a_star(linear_graph, node, "D", hop_count_heuristic)
            assert h(node, "D") <= actual_cost + 1e-9, (
                f"Heuristic {h(node, 'D')} > actual cost {actual_cost} for node {node}"
            )

    def test_heuristic_zero_at_goal(self, linear_graph):
        h = bfs_heuristic_factory(linear_graph, "D")
        assert h("D", "D") == 0.0


class TestDynamicReplanning:
    def test_isolated_node_skipped(self, linear_graph):
        """When B is isolated, A* path must avoid it."""
        linear_graph.isolate("B")
        path, cost, _ = a_star(linear_graph, "A", "D", hop_count_heuristic)
        assert path is None or "B" not in path
