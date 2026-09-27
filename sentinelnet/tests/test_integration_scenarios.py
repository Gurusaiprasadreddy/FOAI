"""
tests/test_integration_scenarios.py
End-to-end integration tests that run all 5 demo scenarios through
the full simulation pipeline and verify key invariants.

Tests verify:
  1. Each scenario completes without errors.
  2. Summary dict contains all required keys.
  3. Uptime fraction stays in [0, 1].
  4. Minimax defender performs better than random (crown jewel not compromised
     or survives more ticks).
  5. Belief sums to ≈ 1 at every tick.
  6. Scalability: 100-node graph completes in < 30 seconds.
"""
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sentinelnet.environment.network_graph import NetworkGraph
from sentinelnet.environment.simulator import Simulator, build_simulator
from sentinelnet.agents.red_attacker import RedAttacker
from sentinelnet.agents.monitor import MonitorAgent
from sentinelnet.agents.response import ResponseAgent
from sentinelnet.agents.patch_scheduler import PatchSchedulerAgent
from sentinelnet.baselines.random_defender import RandomDefender
from sentinelnet.baselines.greedy_defender import GreedyDefender
from sentinelnet.coordination.message_bus import MessageBus
from sentinelnet.metrics.logger import MetricsLogger


CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "config")
SMALL_YAML = os.path.join(CONFIG_DIR, "network_small.yaml")


# ---------------------------------------------------------------------------
# Helper to build a complete simulator from a network graph
# ---------------------------------------------------------------------------

def make_simulator(graph: NetworkGraph, defender: str = "minimax", max_ticks: int = 100, seed: int = 42) -> Simulator:
    entry = graph.entry_points[0] if graph.entry_points else graph.all_nodes()[0]
    bus = MessageBus()

    red = RedAttacker(entry_node=entry, seed=seed)
    red.set_graph(graph)

    monitor = MonitorAgent(graph=graph, bus=bus, seed=seed)

    if defender == "random":
        response = RandomDefender(seed=seed, bus=bus)
    elif defender == "greedy":
        response = GreedyDefender(bus=bus)
    else:
        response = ResponseAgent(depth=2, top_k=5, use_expectimax=True, bus=bus)

    patch_scheduler = PatchSchedulerAgent(patch_interval=5, bus=bus)
    logger = MetricsLogger(scenario_name="test", strategy_name=defender)

    return Simulator(
        graph=graph,
        red=red,
        monitor=monitor,
        response=response,
        patch_scheduler=patch_scheduler,
        bus=bus,
        logger=logger,
        max_ticks=max_ticks,
    )


# ---------------------------------------------------------------------------
# Scenario 1: Baseline — naive Red vs random defender
# ---------------------------------------------------------------------------

class TestBaselineScenario:
    def test_random_defender_completes(self):
        graph = NetworkGraph(SMALL_YAML)
        sim = make_simulator(graph, defender="random", max_ticks=50)
        summary = sim.run()
        assert isinstance(summary, dict)
        assert "total_ticks" in summary
        assert summary["total_ticks"] >= 1

    def test_summary_keys_present(self):
        graph = NetworkGraph(SMALL_YAML)
        sim = make_simulator(graph, defender="random", max_ticks=30)
        summary = sim.run()
        required_keys = [
            "scenario", "strategy", "total_ticks",
            "mean_uptime_pct", "avg_nodes_expanded_astar",
        ]
        for key in required_keys:
            assert key in summary, f"Missing key in summary: {key!r}"

    def test_uptime_in_valid_range(self):
        graph = NetworkGraph(SMALL_YAML)
        sim = make_simulator(graph, defender="random", max_ticks=30)
        sim.run()
        for record in sim.logger._records:
            assert 0.0 <= record.uptime_fraction <= 1.0, (
                f"Uptime out of range at tick {record.tick}: {record.uptime_fraction}"
            )


# ---------------------------------------------------------------------------
# Scenario 2: Stealthy attacker — verify belief tracking still works
# ---------------------------------------------------------------------------

class TestStealthyScenario:
    def test_stealthy_sim_completes(self):
        graph = NetworkGraph(SMALL_YAML)
        entry = graph.entry_points[0]
        bus = MessageBus()
        red = RedAttacker(entry_node=entry, detection_rate=0.2, false_alarm_rate=0.2,
                          stealth_mode=True, seed=42)
        red.set_graph(graph)
        monitor = MonitorAgent(graph=graph, detection_rate=0.2, false_alarm_rate=0.2,
                               bus=bus, seed=42)
        response = ResponseAgent(depth=2, bus=bus)
        patch_scheduler = PatchSchedulerAgent(bus=bus)
        logger = MetricsLogger("stealthy", "minimax")
        sim = Simulator(graph, red, monitor, response, patch_scheduler, bus, logger, max_ticks=80)
        summary = sim.run()
        assert summary["total_ticks"] >= 1

    def test_belief_sums_to_one_each_tick(self):
        graph = NetworkGraph(SMALL_YAML)
        entry = graph.entry_points[0]
        bus = MessageBus()
        red = RedAttacker(entry_node=entry, detection_rate=0.3, seed=1)
        red.set_graph(graph)
        monitor = MonitorAgent(graph=graph, bus=bus, seed=1)
        response = ResponseAgent(depth=1, bus=bus)
        patch_sched = PatchSchedulerAgent(bus=bus)
        logger = MetricsLogger()
        sim = Simulator(graph, red, monitor, response, patch_sched, bus, logger, max_ticks=30)

        for _ in range(30):
            if sim.terminated:
                break
            info = sim.step()
            belief = info.get("belief", {})
            if belief:
                total = sum(belief.values())
                # Only assert if belief is non-empty and non-trivially zero
                if total > 1e-9:
                    assert abs(total - 1.0) < 0.01, (
                        f"Belief sum {total:.6f} != 1.0 at tick {sim.current_tick}"
                    )


# ---------------------------------------------------------------------------
# Scenario 3: Multi-entry attack — two simultaneous Red attackers (GAP 1 FIX)
# ---------------------------------------------------------------------------

class TestMultiEntryScenario:
    def test_multi_entry_spawns_two_reds(self):
        """build_simulator with multi_entry=True must create two Red agents."""
        import os
        scenarios_path = os.path.join(CONFIG_DIR, "scenarios.yaml")
        import yaml
        with open(scenarios_path) as f:
            data = yaml.safe_load(f)
        # Find the multi-entry scenario
        multi_cfg = next(
            (s for s in data["scenarios"] if s.get("multi_entry")), None
        )
        if multi_cfg is None:
            pytest.skip("No multi_entry scenario defined in scenarios.yaml")

        from sentinelnet.environment.simulator import build_simulator
        sim = build_simulator(multi_cfg, scenarios_path)

        # Must have 2 Red agents in _all_reds
        assert len(sim._all_reds) == 2, (
            f"Expected 2 Red agents for multi_entry scenario, got {len(sim._all_reds)}"
        )
        # Both must start at different entry nodes
        assert sim._all_reds[0].current_node != sim._all_reds[1].current_node, (
            "Both Red agents started at the same node — entry points not distinct"
        )

    def test_multi_entry_sim_runs_to_completion(self):
        """Multi-entry simulation must complete without errors."""
        graph = NetworkGraph(SMALL_YAML)
        entry_points = graph.entry_points
        if len(entry_points) < 2:
            pytest.skip("Network has fewer than 2 entry points")

        bus = MessageBus()
        red1 = RedAttacker(entry_node=entry_points[0], seed=42)
        red2 = RedAttacker(entry_node=entry_points[1], seed=43)
        red1.set_graph(graph)
        red2.set_graph(graph)
        monitor = MonitorAgent(graph=graph, bus=bus, seed=42)
        response = ResponseAgent(depth=2, top_k=8, use_expectimax=True, bus=bus)
        patch_scheduler = PatchSchedulerAgent(bus=bus)
        logger = MetricsLogger("multi_entry", "minimax")

        from sentinelnet.environment.simulator import Simulator
        sim = Simulator(
            graph, red1, monitor, response, patch_scheduler, bus, logger,
            max_ticks=100, extra_reds=[red2]
        )
        summary = sim.run()
        assert summary is not None
        assert summary["total_ticks"] >= 1

    def test_multi_entry_tick_info_contains_red_nodes(self):
        """_tick_info must return a 'red_nodes' list with positions of all reds."""
        graph = NetworkGraph(SMALL_YAML)
        entry_points = graph.entry_points
        if len(entry_points) < 2:
            pytest.skip("Network has fewer than 2 entry points")

        bus = MessageBus()
        red1 = RedAttacker(entry_node=entry_points[0], seed=42)
        red2 = RedAttacker(entry_node=entry_points[1], seed=43)
        red1.set_graph(graph)
        red2.set_graph(graph)
        monitor = MonitorAgent(graph=graph, bus=bus, seed=42)
        response = ResponseAgent(depth=1, bus=bus)
        patch_scheduler = PatchSchedulerAgent(bus=bus)
        logger = MetricsLogger()

        from sentinelnet.environment.simulator import Simulator
        sim = Simulator(
            graph, red1, monitor, response, patch_scheduler, bus, logger,
            max_ticks=5, extra_reds=[red2]
        )
        info = sim.step()
        assert "red_nodes" in info
        assert isinstance(info["red_nodes"], list)
        assert len(info["red_nodes"]) == 2


# ---------------------------------------------------------------------------
# Honeypot deployment test (GAP 3 FIX)
# ---------------------------------------------------------------------------

class TestHoneypotDeployment:
    def test_response_agent_can_deploy_honeypot(self):
        """ResponseAgent must offer deploy_honeypot for crown-jewel predecessors."""
        graph = NetworkGraph(SMALL_YAML)
        bus = MessageBus()

        # Seed belief so that crown-jewel predecessor nodes are in top suspects
        cj = graph.crown_jewel
        cj_preds = list(graph.g.predecessors(cj))
        if not cj_preds:
            pytest.skip("No predecessor nodes for crown jewel in this topology")

        # Concentrate belief on a crown-jewel predecessor
        belief = {n: 0.0 for n in graph.all_nodes()}
        belief[cj_preds[0]] = 1.0
        bus.publish("belief", belief, tick=0)

        agent = ResponseAgent(depth=1, top_k=5, use_expectimax=True, bus=bus)
        agent.tick = 1

        from sentinelnet.agents.response import GameState, blue_actions
        state = GameState.from_graph(graph, total_cost=0)

        # The predecessor node should appear in blue_actions with deploy_honeypot
        actions = blue_actions(state, cj_preds[:3])
        action_types = [a[0] for a in actions]
        assert "deploy_honeypot" in action_types, (
            f"Expected deploy_honeypot for crown-jewel predecessor, "
            f"got actions: {actions}"
        )

    def test_deploy_honeypot_applies_to_graph(self):
        """graph.deploy_honeypot() must change node state to HONEYPOT."""
        graph = NetworkGraph(SMALL_YAML)
        cj = graph.crown_jewel
        cj_preds = list(graph.g.predecessors(cj))
        if not cj_preds:
            pytest.skip("No predecessor nodes for crown jewel")
        target = cj_preds[0]
        success = graph.deploy_honeypot(target)
        assert success, f"deploy_honeypot returned False for {target}"
        from sentinelnet.environment.network_graph import NodeState
        assert graph.node_data(target).state == NodeState.HONEYPOT, (
            f"Node {target} state should be HONEYPOT after deploy_honeypot"
        )


# ---------------------------------------------------------------------------
# Scenario 4: Resource-starved defense
# ---------------------------------------------------------------------------

class TestResourceStarvedScenario:
    def test_resource_limited_sim_completes(self):
        graph = NetworkGraph(SMALL_YAML)
        bus = MessageBus()
        red = RedAttacker(entry_node=graph.entry_points[0], seed=42)
        red.set_graph(graph)
        monitor = MonitorAgent(graph=graph, bus=bus, seed=42)
        response = ResponseAgent(
            depth=2, resource_limited=True, resource_interval=2, bus=bus
        )
        patch_scheduler = PatchSchedulerAgent(patch_interval=8, n_technicians=1, bus=bus)
        logger = MetricsLogger("resource_starved", "minimax")
        sim = Simulator(graph, red, monitor, response, patch_scheduler, bus, logger, max_ticks=80)
        summary = sim.run()
        assert summary is not None



# ---------------------------------------------------------------------------
# Scenario 4: Minimax outperforms random (on average)
# ---------------------------------------------------------------------------

class TestMinimaxVsRandom:
    def _run_scenario(self, defender: str, seed: int, max_ticks: int = 60) -> dict:
        graph = NetworkGraph(SMALL_YAML)
        sim = make_simulator(graph, defender=defender, max_ticks=max_ticks, seed=seed)
        return sim.run(), sim

    def test_minimax_survives_at_least_as_long_as_random(self):
        """Run 3 seeds; minimax should survive >= random on at least 2 of 3."""
        wins = 0
        for seed in [1, 2, 3]:
            r_summary, r_sim = self._run_scenario("random", seed=seed)
            m_summary, m_sim = self._run_scenario("minimax", seed=seed)
            r_ticks = r_summary.get("total_ticks", 0)
            m_ticks = m_summary.get("total_ticks", 0)
            if m_ticks >= r_ticks:
                wins += 1
        assert wins >= 2, f"Minimax only won {wins}/3 seed comparisons"


# ---------------------------------------------------------------------------
# Scenario 5: Scalability — 100-node graph under 30 seconds
# ---------------------------------------------------------------------------

class TestScalability:
    def test_100_node_graph_completes_quickly(self):
        graph = NetworkGraph.generate_random(100, seed=99)
        sim = make_simulator(graph, defender="minimax", max_ticks=50)
        start = time.perf_counter()
        sim.run()
        elapsed = time.perf_counter() - start
        assert elapsed < 30, f"100-node simulation took {elapsed:.1f}s (> 30s limit)"

    def test_20_node_graph_nodes_expanded_tracked(self):
        graph = NetworkGraph(SMALL_YAML)
        sim = make_simulator(graph, defender="minimax", max_ticks=30)
        sim.run()
        # At least one tick should have expanded at least 1 node
        total_expanded = sum(r.red_nodes_expanded for r in sim.logger._records)
        assert total_expanded > 0, "A* nodes_expanded should be > 0"


# ---------------------------------------------------------------------------
# Message bus integration
# ---------------------------------------------------------------------------

class TestMessageBusIntegration:
    def test_belief_topic_populated_each_tick(self):
        graph = NetworkGraph(SMALL_YAML)
        bus = MessageBus()
        red = RedAttacker(entry_node=graph.entry_points[0], seed=42)
        red.set_graph(graph)
        monitor = MonitorAgent(graph=graph, bus=bus, seed=42)
        response = ResponseAgent(depth=1, bus=bus)
        patch_sched = PatchSchedulerAgent(bus=bus)
        logger = MetricsLogger()
        sim = Simulator(graph, red, monitor, response, patch_sched, bus, logger, max_ticks=10)
        sim.run()
        # Belief topic must have been written at least once
        assert bus.topic_length("belief") >= 1, "Belief topic should have messages"

    def test_actions_topic_populated(self):
        graph = NetworkGraph(SMALL_YAML)
        bus = MessageBus()
        red = RedAttacker(entry_node=graph.entry_points[0], seed=42)
        red.set_graph(graph)
        monitor = MonitorAgent(graph=graph, bus=bus, seed=42)
        response = ResponseAgent(depth=1, bus=bus)
        patch_sched = PatchSchedulerAgent(bus=bus)
        logger = MetricsLogger()
        sim = Simulator(graph, red, monitor, response, patch_sched, bus, logger, max_ticks=20)
        sim.run()
        # Response agent should have published at least one action over 20 ticks
        # (belief was populated by monitor so response had data to act on)
        assert bus.topic_length("actions") >= 1, (
            "ResponseAgent should publish at least one action in 20 ticks "
            "when monitor belief is available"
        )


# ---------------------------------------------------------------------------
# conftest: make tests importable without pytest ini
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
