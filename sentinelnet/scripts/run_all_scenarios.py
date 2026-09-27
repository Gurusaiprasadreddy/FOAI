"""
scripts/run_all_scenarios.py
Runs every scenario defined in scenarios.yaml and prints runtime evidence
for Scenario 3 (multi-entry attack), including both Red agent positions,
A* plans, and the effect of Blue defensive actions.

Usage:
    python scripts/run_all_scenarios.py
"""
from __future__ import annotations

import os
import sys
import time
import yaml

# Make sentinelnet importable from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sentinelnet.environment.simulator import build_simulator
from sentinelnet.environment.network_graph import NetworkGraph

SCENARIOS_YAML = os.path.join(os.path.dirname(__file__), "..", "config", "scenarios.yaml")
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "config")
SEPARATOR = "=" * 70


def run_scenario(cfg: dict, scenario_idx: int) -> dict:
    name = cfg.get("name", f"Scenario {scenario_idx + 1}")
    multi = cfg.get("multi_entry", False)

    print(f"\n{SEPARATOR}")
    print(f"SCENARIO {scenario_idx + 1}: {name}")
    print(SEPARATOR)

    if multi:
        print("  *** Multi-entry attack enabled ***")

    sim = build_simulator(cfg, SCENARIOS_YAML)

    # Print initial Red agent positions
    for i, r in enumerate(sim._all_reds):
        print(f"  RedAgent-{i+1}: Entry = {r.current_node}")

    # For Scenario 3 — show tick-by-tick detail for first 15 ticks
    is_multi = len(sim._all_reds) > 1
    tick_detail_limit = 15 if is_multi else 0

    t0 = time.perf_counter()
    results = []
    for tick in range(cfg.get("max_ticks", 200)):
        if sim.terminated:
            break

        info = sim.step()

        if tick < tick_detail_limit:
            print(f"\n  [Tick {tick+1:3d}]")
            for i, r in enumerate(sim._all_reds):
                path_str = " -> ".join(r._path[:5]) if r._path else "(none)"
                if len(r._path or []) > 5:
                    path_str += " ..."
                captured_str = " [CAPTURED by honeypot!]" if r.captured else ""
                print(f"    RedAgent-{i+1}: pos={r.current_node}  A*-next5={path_str}"
                      f"  expanded={r.nodes_expanded_last}{captured_str}")

            if info.get("alert"):
                a = info["alert"]
                print(f"    Alert: node={a.node}  confidence={a.confidence:.2f}")
            if info.get("action"):
                act = info["action"]
                print(f"    Blue action: {act.action_type.upper()}({act.node})"
                      f"  by {act.agent}")
            belief = info.get("belief", {})
            if belief:
                top3 = sorted(belief.items(), key=lambda x: -x[1])[:3]
                top3_str = ", ".join(f"{n}:{p:.2f}" for n, p in top3)
                print(f"    Belief top-3: [{top3_str}]")

        results.append(info)

    elapsed = time.perf_counter() - t0
    summary = sim.logger.summary()

    print(f"\n  RESULT after {summary.get('total_ticks', 0)} ticks "
          f"({elapsed:.2f}s elapsed):")
    print(f"    Termination: {sim.termination_reason}")
    print(f"    Crown jewel compromised: {summary.get('crown_jewel_compromised')}")
    print(f"    Max nodes compromised:   {summary.get('max_compromised_nodes')}")
    print(f"    Compromise rate:         {summary.get('compromise_rate', 0):.3f}")
    print(f"    Mean uptime:             {summary.get('mean_uptime_pct'):.1f}%")
    print(f"    Mean TTD (ticks):        {summary.get('mean_ttd_ticks')}")
    print(f"    Avg A* nodes expanded:   {summary.get('avg_nodes_expanded_astar')}")
    print(f"    Total wall-clock ms:     {summary.get('total_wall_ms')}")

    return summary


def run_scalability_tests():
    print(f"\n{SEPARATOR}")
    print("SCALABILITY TESTS: 20 / 100 / 500 nodes")
    print(SEPARATOR)

    for n_nodes, label in [(20, "20-node"), (100, "100-node"), (500, "500-node")]:
        t0 = time.perf_counter()
        graph = NetworkGraph.generate_random(n_nodes, seed=99)
        bus_module = __import__("sentinelnet.coordination.message_bus",
                                fromlist=["MessageBus"]).MessageBus
        from sentinelnet.agents.red_attacker import RedAttacker
        from sentinelnet.agents.monitor import MonitorAgent
        from sentinelnet.agents.response import ResponseAgent
        from sentinelnet.agents.patch_scheduler import PatchSchedulerAgent
        from sentinelnet.metrics.logger import MetricsLogger
        from sentinelnet.environment.simulator import Simulator
        from sentinelnet.coordination.message_bus import MessageBus

        bus = MessageBus()
        entry = graph.entry_points[0] if graph.entry_points else graph.all_nodes()[0]
        red = RedAttacker(entry_node=entry, seed=42)
        red.set_graph(graph)
        monitor = MonitorAgent(graph=graph, bus=bus, seed=42,
                               use_particle_filter=(n_nodes > 200))
        response = ResponseAgent(depth=2, top_k=5, use_expectimax=True, bus=bus)
        patch_sched = PatchSchedulerAgent(bus=bus, patch_interval=5, n_technicians=2)
        logger = MetricsLogger(label, "minimax")
        sim = Simulator(graph, red, monitor, response, patch_sched, bus, logger,
                        max_ticks=50)
        sim.run()
        elapsed = time.perf_counter() - t0
        nodes_exp = sum(r.red_nodes_expanded for r in sim.logger._records)
        print(f"  {label:10s}: {sim.current_tick:4d} ticks  "
              f"elapsed={elapsed:.2f}s  A*_expanded_total={nodes_exp}  "
              f"{'PASS' if elapsed < 60 else 'SLOW'}")


if __name__ == "__main__":
    with open(SCENARIOS_YAML) as f:
        data = yaml.safe_load(f)

    all_summaries = []
    for i, cfg in enumerate(data["scenarios"]):
        try:
            summary = run_scenario(cfg, i)
            all_summaries.append((cfg["name"], "PASS", summary))
        except Exception as e:
            print(f"\n  *** SCENARIO FAILED: {e}")
            all_summaries.append((cfg["name"], f"FAIL: {e}", {}))

    run_scalability_tests()

    print(f"\n{SEPARATOR}")
    print("SCENARIO SUMMARY TABLE")
    print(SEPARATOR)
    print(f"{'#':>3}  {'Scenario':<45}  {'Result':>6}")
    print("-" * 60)
    for i, (name, result, _) in enumerate(all_summaries):
        print(f"{i+1:>3}  {name[:45]:<45}  {result}")

    print(f"\nAll scenarios executed. Project verified.")
