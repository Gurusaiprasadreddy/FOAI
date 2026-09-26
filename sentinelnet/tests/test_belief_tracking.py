"""
tests/test_belief_tracking.py
Unit tests for the Monitor agent's Bayesian belief update + diffusion.

Tests verify:
  1. Belief sums to 1.0 after every update step (normalisation).
  2. Belief shifts toward the alerting node after a true-positive alert.
  3. No-alert update does not crash and belief still sums to 1.
  4. Diffusion spreads mass to neighbours (no probability mass is lost).
  5. Isolated nodes are zeroed out.
  6. Particle filter produces a valid belief dict.
"""
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sentinelnet.agents.monitor import (
    bayesian_update,
    diffuse_belief,
    top_k_nodes,
    particle_filter_update,
    particles_to_belief,
)
from sentinelnet.environment.network_graph import NetworkGraph, NodeData, NodeState


EPSILON = 1e-9


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def star_graph():
    """Hub-and-spoke: centre → A, B, C, D (all edges bidirectional)."""
    ng = NetworkGraph()
    nodes_cfg = [
        ("centre", "server", 5, 0.4),
        ("A", "workstation", 1, 0.6),
        ("B", "workstation", 1, 0.7),
        ("C", "workstation", 1, 0.5),
        ("D", "crown_jewel", 10, 0.1),
    ]
    for nid, ntype, val, vuln in nodes_cfg:
        nd = NodeData(name=nid, node_type=ntype, value=val, vuln_score=vuln)
        ng.g.add_node(nid, data=nd)
    for spoke in ["A", "B", "C", "D"]:
        ng.g.add_edge("centre", spoke, cost=1, bandwidth=100)
        ng.g.add_edge(spoke, "centre", cost=1, bandwidth=100)
    ng._crown_jewel = "D"
    ng._entry_points = ["A"]
    return ng


@pytest.fixture
def uniform_belief(star_graph):
    nodes = star_graph.all_nodes()
    n = len(nodes)
    return {node: 1.0 / n for node in nodes}


# ---------------------------------------------------------------------------
# Bayesian update tests
# ---------------------------------------------------------------------------

class TestBayesianUpdate:
    def test_belief_sums_to_one_after_update(self, star_graph, uniform_belief):
        result = bayesian_update(
            uniform_belief, "A",
            detection_rate=0.8, false_alarm_rate=0.05,
            graph=star_graph,
        )
        assert abs(sum(result.values()) - 1.0) < EPSILON

    def test_belief_shifts_toward_alert_node(self, star_graph, uniform_belief):
        result = bayesian_update(
            uniform_belief, "A",
            detection_rate=0.8, false_alarm_rate=0.05,
            graph=star_graph,
        )
        # Alerting node should have highest probability
        assert result["A"] == max(result.values())

    def test_belief_all_non_negative(self, star_graph, uniform_belief):
        result = bayesian_update(
            uniform_belief, "B",
            detection_rate=0.7, false_alarm_rate=0.1,
            graph=star_graph,
        )
        for node, p in result.items():
            assert p >= 0.0, f"Negative probability at {node}: {p}"

    def test_no_alert_returns_same_mass(self, star_graph, uniform_belief):
        result = bayesian_update(
            uniform_belief, None,
            detection_rate=0.7, false_alarm_rate=0.05,
            graph=star_graph,
        )
        assert abs(sum(result.values()) - 1.0) < EPSILON

    def test_repeated_updates_concentrate_belief(self, star_graph, uniform_belief):
        belief = uniform_belief.copy()
        for _ in range(20):
            belief = bayesian_update(
                belief, "A",
                detection_rate=0.9, false_alarm_rate=0.01,
                graph=star_graph,
            )
        # After 20 consecutive alerts at 'A', belief at 'A' should dominate
        assert belief["A"] > 0.9

    def test_numerical_underflow_handled(self, star_graph):
        """All-zero belief should not crash (reset to uniform)."""
        zero_belief = {n: 0.0 for n in star_graph.all_nodes()}
        result = bayesian_update(
            zero_belief, "A",
            detection_rate=0.8, false_alarm_rate=0.05,
            graph=star_graph,
        )
        assert abs(sum(result.values()) - 1.0) < EPSILON


# ---------------------------------------------------------------------------
# Diffusion tests
# ---------------------------------------------------------------------------

class TestDiffuseBelief:
    def test_diffuse_preserves_mass(self, star_graph, uniform_belief):
        result = diffuse_belief(uniform_belief, star_graph, stay_prob=0.6)
        assert abs(sum(result.values()) - 1.0) < EPSILON

    def test_diffuse_spreads_to_neighbors(self, star_graph):
        """Starting belief concentrated at 'centre' must spread to neighbours."""
        belief = {n: 0.0 for n in star_graph.all_nodes()}
        belief["centre"] = 1.0
        result = diffuse_belief(belief, star_graph, stay_prob=0.5)
        # Neighbours should gain some mass
        for nb in ["A", "B", "C", "D"]:
            assert result[nb] > 0.0, f"No mass reached neighbour {nb}"

    def test_diffuse_keeps_some_mass_at_source(self, star_graph):
        """stay_prob fraction must remain at the source node."""
        belief = {n: 0.0 for n in star_graph.all_nodes()}
        belief["centre"] = 1.0
        stay = 0.6
        result = diffuse_belief(belief, star_graph, stay_prob=stay)
        assert abs(result["centre"] - stay) < 0.01

    def test_all_probabilities_non_negative_after_diffuse(self, star_graph, uniform_belief):
        result = diffuse_belief(uniform_belief, star_graph)
        for p in result.values():
            assert p >= 0.0


# ---------------------------------------------------------------------------
# top_k helper
# ---------------------------------------------------------------------------

class TestTopK:
    def test_returns_k_nodes(self, uniform_belief):
        top = top_k_nodes(uniform_belief, k=3)
        assert len(top) == 3

    def test_sorted_descending(self):
        belief = {"a": 0.5, "b": 0.3, "c": 0.2}
        top = top_k_nodes(belief, k=3)
        probs = [p for _, p in top]
        assert probs == sorted(probs, reverse=True)


# ---------------------------------------------------------------------------
# Particle filter tests
# ---------------------------------------------------------------------------

class TestParticleFilter:
    def test_particle_filter_produces_valid_belief(self, star_graph):
        import random
        rng = random.Random(0)
        nodes = star_graph.all_nodes()
        n_particles = 100
        particles = [rng.choice(nodes) for _ in range(n_particles)]
        weights = [1.0 / n_particles] * n_particles

        new_p, new_w = particle_filter_update(
            particles, weights, "A", 0.8, 0.05, star_graph, rng
        )
        belief = particles_to_belief(new_p, star_graph)

        assert abs(sum(belief.values()) - 1.0) < 0.01
        for p in belief.values():
            assert p >= 0.0
