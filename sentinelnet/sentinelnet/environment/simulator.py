"""
environment/simulator.py
Core simulation engine — the tick loop that wires all agents together.

Tick protocol (per turn):
  1. Red.step() [and optional Red2.step()] → attacker(s) move, may trigger AlertEvent(s)
  2. bus.publish('alerts', alert)
  3. Monitor.step() → updates belief from alert, publishes to 'belief'
  4. Response.step() → reads belief, runs minimax/expectimax, applies action
  5. Every patch_interval ticks: PatchScheduler.step() → CSP + apply next patch
  6. Logger.record() → write per-tick metrics
  7. Termination check: crown jewel compromised OR Red captured/trapped
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from sentinelnet.environment.network_graph import NetworkGraph
from sentinelnet.environment.events import ActionEvent, AlertEvent
from sentinelnet.agents.base_agent import BaseAgent
from sentinelnet.agents.red_attacker import RedAttacker
from sentinelnet.agents.monitor import MonitorAgent
from sentinelnet.agents.response import ResponseAgent
from sentinelnet.agents.patch_scheduler import PatchSchedulerAgent
from sentinelnet.baselines.random_defender import RandomDefender
from sentinelnet.baselines.greedy_defender import GreedyDefender
from sentinelnet.coordination.message_bus import MessageBus
from sentinelnet.metrics.logger import MetricsLogger


class Simulator:
    """Orchestrates the multi-agent simulation across ticks.

    Supports single-attacker and multi-entry (two simultaneous Red attackers)
    scenarios. When ``red_agents`` has more than one entry, both run each tick
    and the simulation ends when *either* reaches the crown jewel.

    Args:
        graph: The enterprise network graph.
        red: Primary Red attacker agent.
        monitor: Blue monitor agent (or None to skip belief tracking).
        response: Blue response agent (or baseline defender).
        patch_scheduler: Blue patch scheduler agent (or None).
        bus: Shared message bus.
        logger: Metrics logger.
        max_ticks: Maximum number of ticks before the run ends.
        extra_reds: Additional Red attacker agents for multi-entry scenarios.
    """

    def __init__(
        self,
        graph: NetworkGraph,
        red: RedAttacker,
        monitor: Optional[MonitorAgent],
        response: BaseAgent,
        patch_scheduler: Optional[PatchSchedulerAgent],
        bus: MessageBus,
        logger: MetricsLogger,
        max_ticks: int = 200,
        extra_reds: Optional[List[RedAttacker]] = None,
    ) -> None:
        self.graph = graph
        self.red = red
        # All Red agents: primary + any extras (multi-entry scenario)
        self._all_reds: List[RedAttacker] = [red] + (extra_reds or [])
        self.monitor = monitor
        self.response = response
        self.patch_scheduler = patch_scheduler
        self.bus = bus
        self.logger = logger
        self.max_ticks = max_ticks
        self._current_tick: int = 0
        self._terminated: bool = False
        self._termination_reason: str = ""

        # Initialise BFS heuristic for every Red agent
        for r in self._all_reds:
            if hasattr(r, "set_graph"):
                r.set_graph(graph)

    # ------------------------------------------------------------------
    # Single step (called by Streamlit dashboard for step-by-step mode)
    # ------------------------------------------------------------------

    def step(self) -> Dict[str, Any]:
        """Advance by one tick. Returns a dict of observables for the UI."""
        if self._terminated:
            return self._tick_info(action=None, alert=None, belief=None)

        t = self._current_tick
        self.logger.start_tick()

        # --- All Red agents move ---
        last_alert: Optional[AlertEvent] = None
        for r in self._all_reds:
            if r.captured or r.reached_goal:
                continue
            alert: Optional[AlertEvent] = r.step(self.graph)
            if alert is not None:
                self.bus.publish("alerts", alert, tick=t)
                last_alert = alert  # most recent alert for the UI

        # Publish a None alert if none fired (keeps Monitor's bus read consistent)
        if last_alert is None:
            self.bus.publish("alerts", None, tick=t)

        # --- Monitor updates belief ---
        belief: Dict[str, float] = {}
        if self.monitor:
            belief = self.monitor.step(self.graph)
            # Time-to-detect check across all Red positions
            for r in self._all_reds:
                if r.current_node and belief:
                    top_prob = belief.get(r.current_node, 0)
                    if top_prob > 0.4:
                        self.logger.mark_detected(t)
                        break

        # --- Response agent acts ---
        action: Optional[ActionEvent] = self.response.step(self.graph)

        # --- Patch Scheduler (every N ticks) ---
        if self.patch_scheduler:
            self.patch_scheduler.step(self.graph)

        # --- Log this tick ---
        action_str = f"{action.action_type}({action.node})" if action else "noop"
        # Use primary Red's position as the canonical position for the logger
        self.logger.record(
            tick=t,
            graph=self.graph,
            red_nodes_expanded=getattr(self.red, "nodes_expanded_last", 0),
            red_current_node=self.red.current_node,
            action_taken=action_str,
            belief=belief,
        )

        # --- Termination checks ---
        if self.graph.crown_jewel_compromised():
            self._terminated = True
            self._termination_reason = "Crown jewel compromised!"
        elif all(r.captured for r in self._all_reds):
            self._terminated = True
            self._termination_reason = "All Red agents captured by honeypot!"
        elif any(r.captured for r in self._all_reds) and len(self._all_reds) == 1:
            self._terminated = True
            self._termination_reason = "Red agent captured by honeypot!"
        elif any(r.reached_goal for r in self._all_reds):
            self._terminated = True
            self._termination_reason = "Red reached crown jewel!"
        elif self._current_tick >= self.max_ticks - 1:
            self._terminated = True
            self._termination_reason = "Max ticks reached — Blue defended successfully!"

        self._current_tick += 1
        return self._tick_info(action=action, alert=last_alert, belief=belief)

    def _tick_info(self, action, alert, belief) -> Dict[str, Any]:
        # Collect positions of all active Red agents for multi-entry display
        red_nodes = [r.current_node for r in self._all_reds if not r.captured]
        return {
            "tick": self._current_tick,
            "terminated": self._terminated,
            "termination_reason": self._termination_reason,
            "red_node": self.red.current_node,
            "red_nodes": red_nodes,          # all active Red positions (for multi-entry)
            "red_captured": self.red.captured,
            "any_red_captured": any(r.captured for r in self._all_reds),
            "crown_jewel_compromised": self.graph.crown_jewel_compromised(),
            "alert": alert,
            "action": action,
            "belief": belief,
            "uptime": self.graph.uptime_fraction(),
            "compromised_count": sum(
                1 for n in self.graph.all_nodes()
                if self.graph.node_data(n).state.value == "compromised"
            ),
            "multi_entry": len(self._all_reds) > 1,
        }

    # ------------------------------------------------------------------
    # Full run (called by CLI / benchmarks)
    # ------------------------------------------------------------------

    def run(self) -> Dict:
        """Run the simulation to completion and return the summary."""
        while not self._terminated and self._current_tick < self.max_ticks:
            self.step()
        return self.logger.summary()

    # ------------------------------------------------------------------
    # Reset (for running multiple scenarios)
    # ------------------------------------------------------------------

    def reset(
        self,
        graph: NetworkGraph,
        red_entry: Optional[str] = None,
    ) -> None:
        """Reset the simulation for a new run on the same or new graph."""
        self.graph = graph
        self._current_tick = 0
        self._terminated = False
        self._termination_reason = ""
        self.bus.reset()
        self.logger.reset()

        # Reset all Red agents
        for i, r in enumerate(self._all_reds):
            r.on_reset()
            if red_entry and i == 0:
                r.current_node = red_entry
            if hasattr(r, "set_graph"):
                r.set_graph(graph)

        if self.monitor:
            nodes = graph.all_nodes()
            n = len(nodes)
            self.monitor.belief = {node: 1.0 / n for node in nodes}
            self.monitor.on_reset()
        self.response.on_reset()
        if self.patch_scheduler:
            self.patch_scheduler.on_reset()

    @property
    def current_tick(self) -> int:
        return self._current_tick

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def termination_reason(self) -> str:
        return self._termination_reason


# ---------------------------------------------------------------------------
# Factory: build a Simulator from a scenarios.yaml entry
# ---------------------------------------------------------------------------

def build_simulator(scenario_cfg: dict, scenarios_yaml_path: str = None) -> Simulator:
    """Construct a complete Simulator from a scenario configuration dict.

    Handles:
      - Single and multi-entry Red agent scenarios (``multi_entry: true``).
      - YAML-defined or procedurally-generated network topologies.

    Args:
        scenario_cfg: Dict loaded from scenarios.yaml (one scenario entry).
        scenarios_yaml_path: Path to the scenarios YAML (for relative resolution).

    Returns:
        A fully configured :class:`Simulator` instance.
    """
    import os

    # Determine the config root directory
    config_root = os.path.dirname(scenarios_yaml_path) if scenarios_yaml_path else "config"

    # Network graph
    topology = scenario_cfg.get("topology", "network_small")
    topology_path = os.path.join(config_root, f"{topology}.yaml")

    if os.path.exists(topology_path):
        # Check if the YAML has actual node definitions or is a generator placeholder
        with open(topology_path, "r", encoding="utf-8") as _f:
            _cfg = yaml.safe_load(_f)
        if _cfg and _cfg.get("nodes"):
            graph = NetworkGraph(topology_path)
        else:
            # Generator placeholder: use _meta.n_nodes or name-based defaults
            meta = _cfg.get("_meta", {}) if _cfg else {}
            size_map = {"network_small": 20, "network_medium": 100, "network_large": 500}
            n = meta.get("n_nodes") or size_map.get(topology, 20)
            seed = meta.get("seed", scenario_cfg.get("seed", 42))
            graph = NetworkGraph.generate_random(n, seed=seed)
    else:
        # Fallback: generate random graph based on size hint
        size_map = {"network_small": 20, "network_medium": 100, "network_large": 500}
        n = size_map.get(topology, 20)
        graph = NetworkGraph.generate_random(n, seed=scenario_cfg.get("seed", 42))

    seed = scenario_cfg.get("seed", 42)
    detection_rate = scenario_cfg.get("detection_rate", 0.7)
    false_alarm_rate = scenario_cfg.get("false_alarm_rate", 0.05)

    # Primary Red agent
    entry_nodes = graph.entry_points
    primary_entry = (
        scenario_cfg.get("red_entry", None)
        or (entry_nodes[0] if entry_nodes else graph.all_nodes()[0])
    )
    red = RedAttacker(
        entry_node=primary_entry,
        detection_rate=detection_rate,
        false_alarm_rate=false_alarm_rate,
        seed=seed,
        stealth_mode=scenario_cfg.get("stealth_mode", False),
    )

    # -----------------------------------------------------------------
    # GAP 1 FIX: Multi-entry — spawn a second Red attacker
    # -----------------------------------------------------------------
    extra_reds: List[RedAttacker] = []
    if scenario_cfg.get("multi_entry", False) and len(entry_nodes) > 1:
        secondary_entry = entry_nodes[1]
        red2 = RedAttacker(
            entry_node=secondary_entry,
            detection_rate=detection_rate,
            false_alarm_rate=false_alarm_rate,
            seed=seed + 1,          # different seed → different path choices
            stealth_mode=scenario_cfg.get("stealth_mode", False),
        )
        extra_reds.append(red2)

    bus = MessageBus()

    # Monitor — auto-enable particle filter for large graphs (>200 nodes)
    n_nodes = len(graph.all_nodes())
    use_particle = scenario_cfg.get("use_particle_filter", n_nodes > 200)
    monitor = MonitorAgent(
        graph=graph,
        detection_rate=detection_rate,
        false_alarm_rate=false_alarm_rate,
        use_particle_filter=use_particle,
        seed=seed,
        bus=bus,
    )

    # Response / defender
    defender_type = scenario_cfg.get("defender", "minimax")
    if defender_type == "random":
        response = RandomDefender(seed=seed, bus=bus)
    elif defender_type == "greedy":
        response = GreedyDefender(bus=bus)
    else:  # minimax / expectimax
        use_exp = scenario_cfg.get("use_expectimax", True)
        resource_limited = scenario_cfg.get("resource_limited", False)
        response = ResponseAgent(
            depth=scenario_cfg.get("minimax_depth", 2),
            top_k=scenario_cfg.get("belief_top_k", 5),
            use_expectimax=use_exp,
            resource_limited=resource_limited,
            resource_interval=scenario_cfg.get("resource_interval", 2),
            bus=bus,
        )

    # Patch scheduler
    patch_scheduler = PatchSchedulerAgent(
        patch_interval=scenario_cfg.get("patch_interval", 5),
        n_technicians=scenario_cfg.get("n_technicians", 2),
        bus=bus,
    )

    logger = MetricsLogger(
        scenario_name=scenario_cfg.get("name", "scenario"),
        strategy_name=defender_type,
    )

    return Simulator(
        graph=graph,
        red=red,
        monitor=monitor,
        response=response,
        patch_scheduler=patch_scheduler,
        bus=bus,
        logger=logger,
        max_ticks=scenario_cfg.get("max_ticks", 200),
        extra_reds=extra_reds,          # second Red agent when multi_entry=True
    )
