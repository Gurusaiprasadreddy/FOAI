"""
metrics/logger.py
Per-tick metrics recorder for SentinelNet simulations.

Records:
  - Compromise count (number of compromised nodes)
  - Time-to-detect (tick at which Monitor first suspected true attacker location)
  - Service uptime fraction
  - A* nodes expanded by Red (for scalability benchmarking)
  - Wall-clock time per tick

Outputs:
  - summary() dict for the results table
  - to_dataframe() for pandas analysis
  - to_csv(path) for persistence
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from sentinelnet.environment.network_graph import NetworkGraph, NodeState


@dataclass
class TickRecord:
    """One row of per-tick data."""
    tick: int
    compromised_count: int
    compromised_value: int
    uptime_fraction: float
    red_nodes_expanded: int
    wall_clock_ms: float
    red_current_node: str
    action_taken: str
    belief_entropy: float


class MetricsLogger:
    """Records and summarises simulation metrics across ticks.

    Args:
        scenario_name: Label for this run (appears in the results table).
        strategy_name: Defender strategy name (e.g., 'minimax', 'random').
    """

    def __init__(self, scenario_name: str = "default", strategy_name: str = "minimax") -> None:
        self.scenario_name = scenario_name
        self.strategy_name = strategy_name
        self._records: List[TickRecord] = []
        self._time_to_detect: Optional[int] = None  # Tick when belief concentrated
        self._start_time: float = time.perf_counter()
        self._tick_start: float = time.perf_counter()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def start_tick(self) -> None:
        """Call at the very start of each tick."""
        self._tick_start = time.perf_counter()

    def record(
        self,
        tick: int,
        graph: NetworkGraph,
        red_nodes_expanded: int = 0,
        red_current_node: str = "",
        action_taken: str = "",
        belief: Optional[Dict[str, float]] = None,
    ) -> None:
        """Record metrics for one tick.

        Args:
            tick: Current simulation tick.
            graph: The current network graph.
            red_nodes_expanded: Nodes expanded by A* this tick.
            red_current_node: Red's current position (ground truth).
            action_taken: String description of Blue's action.
            belief: Monitor's belief dict (used to compute entropy).
        """
        elapsed_ms = (time.perf_counter() - self._tick_start) * 1000.0

        compromised = [
            n for n in graph.all_nodes()
            if graph.node_data(n).state == NodeState.COMPROMISED
        ]

        entropy = self._entropy(belief) if belief else 0.0

        self._records.append(TickRecord(
            tick=tick,
            compromised_count=len(compromised),
            compromised_value=graph.compromised_value(),
            uptime_fraction=graph.uptime_fraction(),
            red_nodes_expanded=red_nodes_expanded,
            wall_clock_ms=elapsed_ms,
            red_current_node=red_current_node,
            action_taken=action_taken,
            belief_entropy=entropy,
        ))

    @staticmethod
    def _entropy(belief: Dict[str, float]) -> float:
        """Shannon entropy of the belief distribution (lower = more focused)."""
        import math
        h = 0.0
        for p in belief.values():
            if p > 1e-12:
                h -= p * math.log2(p)
        return h

    def mark_detected(self, tick: int) -> None:
        """Call when Monitor first narrows down to true attacker location."""
        if self._time_to_detect is None:
            self._time_to_detect = tick

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        """Return the summary statistics dict for the results table."""
        if not self._records:
            return {}

        total_ticks = len(self._records)
        final = self._records[-1]

        # Compromise rate = fraction of ticks where crown jewel is compromised
        # (simplified: was it ever compromised?)
        ever_compromised = any(r.compromised_count > 0 for r in self._records)
        max_compromised = max(r.compromised_count for r in self._records)

        mean_uptime = sum(r.uptime_fraction for r in self._records) / total_ticks
        avg_nodes_expanded = (
            sum(r.red_nodes_expanded for r in self._records) / total_ticks
        )
        total_wall_ms = sum(r.wall_clock_ms for r in self._records)
        max_wall_ms = max(r.wall_clock_ms for r in self._records)

        return {
            "scenario": self.scenario_name,
            "strategy": self.strategy_name,
            "total_ticks": total_ticks,
            "compromise_rate": max_compromised / max(1, len(self._records[0].__dataclass_fields__)) if self._records else 0,
            "crown_jewel_compromised": ever_compromised,
            "max_compromised_nodes": max_compromised,
            "mean_ttd_ticks": self._time_to_detect if self._time_to_detect else total_ticks,
            "mean_uptime_pct": round(mean_uptime * 100, 2),
            "avg_nodes_expanded_astar": round(avg_nodes_expanded, 1),
            "total_wall_ms": round(total_wall_ms, 1),
            "max_tick_wall_ms": round(max_wall_ms, 2),
        }

    def to_dataframe(self):
        """Convert records to a pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame([
            {
                "tick": r.tick,
                "compromised_count": r.compromised_count,
                "compromised_value": r.compromised_value,
                "uptime_fraction": r.uptime_fraction,
                "red_nodes_expanded": r.red_nodes_expanded,
                "wall_clock_ms": r.wall_clock_ms,
                "red_current_node": r.red_current_node,
                "action_taken": r.action_taken,
                "belief_entropy": r.belief_entropy,
            }
            for r in self._records
        ])

    def to_csv(self, path: str) -> None:
        """Save records to CSV."""
        self.to_dataframe().to_csv(path, index=False)

    def reset(self) -> None:
        """Clear all records — call between scenario runs."""
        self._records = []
        self._time_to_detect = None
        self._start_time = time.perf_counter()
