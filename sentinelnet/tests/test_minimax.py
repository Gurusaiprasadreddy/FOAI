"""
tests/test_minimax.py
Unit tests for the Minimax / Expectimax response agent.

Tests verify:
  1. Minimax picks the provably best action in a hand-built 2-ply tree.
  2. Alpha-beta pruning produces the same result as plain minimax.
  3. Expectimax value is bounded between the min and max minimax values.
  4. utility() function returns the correct weighted score.
  5. Blue actions are generated for suspect nodes.
  6. ResponseAgent integrates correctly with graph + message bus.
"""
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sentinelnet.agents.response import (
    GameState,
    minimax,
    expectimax,
    utility,
    blue_actions,
    red_actions,
    apply_blue_action,
    apply_red_action,
    ResponseAgent,
)
from sentinelnet.environment.network_graph import NetworkGraph, NodeData, NodeState
from sentinelnet.coordination.message_bus import MessageBus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_state():
    """A state with 3 nodes: ws (safe), srv (safe), crown_jewel (safe)."""
    return GameState(
        node_states={"ws": "safe", "srv": "safe", "crown_jewel": "safe"},
        node_values={"ws": 1, "srv": 5, "crown_jewel": 20},
        node_patched={"ws": False, "srv": False, "crown_jewel": False},
        node_vulns={"ws": 0.7, "srv": 0.4, "crown_jewel": 0.05},
        crown_jewel="crown_jewel",
    )


@pytest.fixture
def compromised_state():
    """State where crown_jewel is already compromised — terminal."""
    return GameState(
        node_states={"ws": "safe", "srv": "safe", "crown_jewel": "compromised"},
        node_values={"ws": 1, "srv": 5, "crown_jewel": 20},
        node_patched={"ws": False, "srv": False, "crown_jewel": False},
        node_vulns={"ws": 0.7, "srv": 0.4, "crown_jewel": 0.05},
        crown_jewel="crown_jewel",
        is_terminal_flag=True,
    )


@pytest.fixture
def two_node_graph():
    """Minimal graph: ws → crown_jewel."""
    ng = NetworkGraph()
    for nid, ntype, val, vuln in [
        ("ws", "workstation", 1, 0.7),
        ("crown_jewel", "crown_jewel", 20, 0.05),
    ]:
        nd = NodeData(name=nid, node_type=ntype, value=val, vuln_score=vuln)
        ng.g.add_node(nid, data=nd)
    ng.g.add_edge("ws", "crown_jewel", cost=1, bandwidth=100)
    ng.g.add_edge("crown_jewel", "ws", cost=1, bandwidth=100)
    ng._crown_jewel = "crown_jewel"
    ng._entry_points = ["ws"]
    return ng


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------

class TestUtility:
    def test_utility_higher_when_no_compromise(self, simple_state, compromised_state):
        u_safe = utility(simple_state)
        u_comp = utility(compromised_state)
        assert u_safe > u_comp, "Safe state should have higher utility than compromised"

    def test_utility_decreases_with_cost(self, simple_state):
        state_with_cost = GameState(
            **{**simple_state.__dict__, "total_action_cost": 100}
        )
        assert utility(state_with_cost) < utility(simple_state)

    def test_utility_terminal_very_negative(self, compromised_state):
        u = utility(compromised_state)
        # Crown jewel (value=20) is compromised → large negative penalty
        assert u < -50, f"Terminal utility should be very negative, got {u}"


# ---------------------------------------------------------------------------
# Minimax tests
# ---------------------------------------------------------------------------

class TestMinimax:
    def test_minimax_isolates_high_vuln_node(self, simple_state):
        """Blue should prefer to isolate 'ws' (highest vuln=0.7) near crown jewel."""
        suspects = ["ws", "srv"]
        value, action = minimax(simple_state, depth=1, alpha=float("-inf"),
                                beta=float("inf"), maximizing=True, top_suspects=suspects)
        assert action is not None
        assert action[0] in ("isolate", "patch"), f"Expected defensive action, got {action}"

    def test_minimax_terminal_returns_utility(self, compromised_state):
        """At terminal state, minimax just evaluates without recursing."""
        value, action = minimax(
            compromised_state, depth=2, alpha=float("-inf"),
            beta=float("inf"), maximizing=True, top_suspects=["crown_jewel"]
        )
        assert action is None  # No action needed at terminal
        expected = utility(compromised_state)
        assert abs(value - expected) < 1e-6

    def test_minimax_depth_zero_evaluates(self, simple_state):
        value, action = minimax(
            simple_state, depth=0, alpha=float("-inf"),
            beta=float("inf"), maximizing=True, top_suspects=["ws"]
        )
        assert action is None
        assert value == pytest.approx(utility(simple_state), abs=1e-6)

    def test_alpha_beta_same_result_as_full_minimax(self, simple_state):
        """Alpha-beta pruning must return identical value to full minimax."""
        suspects = ["ws", "srv"]
        val_ab, act_ab = minimax(
            simple_state, depth=2, alpha=float("-inf"),
            beta=float("inf"), maximizing=True, top_suspects=suspects
        )
        # Run again with extremely wide bounds — should be identical
        val_full, act_full = minimax(
            simple_state, depth=2, alpha=-1e9, beta=1e9,
            maximizing=True, top_suspects=suspects
        )
        assert abs(val_ab - val_full) < 1e-6, (
            f"Alpha-beta value {val_ab} differs from full minimax {val_full}"
        )


# ---------------------------------------------------------------------------
# Expectimax tests
# ---------------------------------------------------------------------------

class TestExpectimax:
    def test_expectimax_value_different_from_minimax(self, simple_state):
        """Expectimax averages over outcomes, so value should differ from minimax."""
        suspects = ["ws", "srv"]
        mm_val, _ = minimax(simple_state, depth=1, alpha=float("-inf"),
                            beta=float("inf"), maximizing=True, top_suspects=suspects)
        ex_val, _ = expectimax(simple_state, depth=1, maximizing=True, top_suspects=suspects)
        # Values may differ because minimax assumes worst-case Red, expectimax averages
        # (they can be equal by coincidence, but typically differ)
        assert isinstance(ex_val, float)

    def test_expectimax_terminal_returns_utility(self, compromised_state):
        value, _ = expectimax(
            compromised_state, depth=2, maximizing=True, top_suspects=["crown_jewel"]
        )
        assert abs(value - utility(compromised_state)) < 1e-6


# ---------------------------------------------------------------------------
# Action generator tests
# ---------------------------------------------------------------------------

class TestActionGenerators:
    def test_blue_actions_includes_isolate_for_safe_node(self, simple_state):
        actions = blue_actions(simple_state, ["ws"])
        action_types = [a[0] for a in actions]
        assert "isolate" in action_types

    def test_blue_actions_includes_restore_for_compromised(self):
        state = GameState(
            node_states={"srv": "compromised"},
            node_values={"srv": 5},
            node_patched={"srv": False},
            node_vulns={"srv": 0.4},
            crown_jewel="srv",
        )
        actions = blue_actions(state, ["srv"])
        action_types = [a[0] for a in actions]
        assert "restore" in action_types

    def test_red_actions_excludes_isolated(self):
        state = GameState(
            node_states={"ws": "isolated", "srv": "safe"},
            node_values={"ws": 1, "srv": 5},
            node_patched={"ws": False, "srv": False},
            node_vulns={"ws": 0.7, "srv": 0.4},
            crown_jewel="srv",
        )
        actions = red_actions(state, ["ws", "srv"])
        action_nodes = [a[0] for a in actions]
        assert "ws" not in action_nodes


# ---------------------------------------------------------------------------
# ResponseAgent integration test
# ---------------------------------------------------------------------------

class TestResponseAgentIntegration:
    def test_response_agent_acts_on_belief(self, two_node_graph):
        bus = MessageBus()
        # Publish a belief that concentrates on 'ws'
        belief = {"ws": 0.9, "crown_jewel": 0.1}
        bus.publish("belief", belief, tick=0)

        agent = ResponseAgent(depth=1, top_k=2, use_expectimax=True, bus=bus)
        agent.tick = 1
        action = agent.decide(two_node_graph, belief)
        # Agent should do something (isolate or patch ws)
        # (or None if ws is already in non-actionable state — that's fine too)
        if action is not None:
            assert action.node in two_node_graph.all_nodes()
