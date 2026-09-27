"""
tests/test_gap2_metrics_logger.py
Deterministic regression tests for Gap 2 — MetricsLogger.summary() compromise_rate bug.

The correct formula is:
    compromise_rate = (number of ticks where compromised_count > 0) / total_ticks

Test cases (all deterministic):
  - 4 records, 1 compromised  → 1/4 = 0.25
  - 4 records, 2 compromised  → 2/4 = 0.50
  - 4 records, 0 compromised  → 0/4 = 0.00
  - 4 records, 4 compromised  → 4/4 = 1.00
  - 0 records                 → 0 (safe empty result, {} from summary())
  - 1 record, compromised     → 1/1 = 1.00
  - 1 record, not compromised → 0/1 = 0.00
"""
import os
import sys
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sentinelnet.metrics.logger import MetricsLogger, TickRecord
from sentinelnet.environment.network_graph import NetworkGraph, NodeData, NodeState

SMALL_YAML = os.path.join(os.path.dirname(__file__), "..", "config", "network_small.yaml")


# ---------------------------------------------------------------------------
# Helper: inject synthetic TickRecord objects without running a simulation
# ---------------------------------------------------------------------------

def _make_record(tick: int, compromised_count: int) -> TickRecord:
    """Create a synthetic TickRecord with the given compromise count."""
    return TickRecord(
        tick=tick,
        compromised_count=compromised_count,
        compromised_value=compromised_count * 5,
        uptime_fraction=max(0.0, 1.0 - compromised_count * 0.1),
        red_nodes_expanded=1,
        wall_clock_ms=1.0,
        red_current_node="ws0",
        action_taken="noop",
        belief_entropy=0.0,
    )


def _logger_with_records(compromised_pattern: list) -> MetricsLogger:
    """Build a MetricsLogger whose records match the given pattern.

    Args:
        compromised_pattern: List of ints representing compromised_count per tick.
    Returns:
        MetricsLogger with those records pre-injected.
    """
    logger = MetricsLogger(scenario_name="test", strategy_name="test")
    for tick, count in enumerate(compromised_pattern):
        logger._records.append(_make_record(tick, count))
    return logger


# ---------------------------------------------------------------------------
# Core compromise_rate tests
# ---------------------------------------------------------------------------

class TestCompromiseRateCalculation:

    def test_one_of_four_compromised(self):
        """4 records, 1 tick compromised → compromise_rate == 0.25."""
        logger = _logger_with_records([0, 1, 0, 0])  # tick 1 has 1 compromise
        s = logger.summary()
        assert abs(s["compromise_rate"] - 0.25) < 1e-9, (
            f"Expected 0.25, got {s['compromise_rate']}"
        )

    def test_two_of_four_compromised(self):
        """4 records, 2 ticks compromised → compromise_rate == 0.50."""
        logger = _logger_with_records([1, 1, 0, 0])
        s = logger.summary()
        assert abs(s["compromise_rate"] - 0.50) < 1e-9, (
            f"Expected 0.50, got {s['compromise_rate']}"
        )

    def test_none_of_four_compromised(self):
        """4 records, 0 ticks compromised → compromise_rate == 0.00."""
        logger = _logger_with_records([0, 0, 0, 0])
        s = logger.summary()
        assert s["compromise_rate"] == 0.0, (
            f"Expected 0.0, got {s['compromise_rate']}"
        )

    def test_all_of_four_compromised(self):
        """4 records, all 4 ticks compromised → compromise_rate == 1.00."""
        logger = _logger_with_records([2, 1, 3, 1])  # all non-zero
        s = logger.summary()
        assert abs(s["compromise_rate"] - 1.0) < 1e-9, (
            f"Expected 1.0, got {s['compromise_rate']}"
        )

    def test_single_record_compromised(self):
        """1 record, compromised → compromise_rate == 1.0."""
        logger = _logger_with_records([1])
        s = logger.summary()
        assert abs(s["compromise_rate"] - 1.0) < 1e-9, (
            f"Expected 1.0, got {s['compromise_rate']}"
        )

    def test_single_record_not_compromised(self):
        """1 record, not compromised → compromise_rate == 0.0."""
        logger = _logger_with_records([0])
        s = logger.summary()
        assert s["compromise_rate"] == 0.0, (
            f"Expected 0.0, got {s['compromise_rate']}"
        )

    def test_empty_records_returns_empty_dict(self):
        """0 records → summary() returns {} (safe fallback — no division by zero)."""
        logger = MetricsLogger()
        s = logger.summary()
        assert s == {}, (
            f"Empty logger should return empty dict, got {s}"
        )

    def test_compromise_rate_not_dependent_on_field_count(self):
        """REGRESSION: old code divided by len(__dataclass_fields__) == 8.
        Ensure the new denominator is NOT 8 when we have 8 records."""
        # 8 records total, 2 compromised → correct = 2/8 = 0.25
        # Old bug: 2/8 = 0.25 (accidentally correct for 8 records)
        # But with 4 records, 1 compromised → old = 1/8 = 0.125, correct = 1/4 = 0.25
        logger = _logger_with_records([1, 0, 0, 0])  # 4 records, 1 compromised
        s = logger.summary()
        # Old buggy value: 1 / 8 = 0.125
        # Correct value:   1 / 4 = 0.25
        assert abs(s["compromise_rate"] - 0.25) < 1e-9, (
            f"compromise_rate={s['compromise_rate']} — old bug would give 0.125"
        )


# ---------------------------------------------------------------------------
# Additional summary metric sanity checks
# ---------------------------------------------------------------------------

class TestSummaryMetricSanity:

    def test_summary_contains_all_required_keys(self):
        """summary() must contain all expected metric keys."""
        logger = _logger_with_records([0, 1, 0])
        s = logger.summary()
        required = [
            "scenario", "strategy", "total_ticks",
            "compromise_rate", "crown_jewel_compromised",
            "max_compromised_nodes", "mean_ttd_ticks",
            "mean_uptime_pct", "avg_nodes_expanded_astar",
            "total_wall_ms", "max_tick_wall_ms",
        ]
        for k in required:
            assert k in s, f"Missing key '{k}' in summary()"

    def test_total_ticks_matches_record_count(self):
        """total_ticks in summary must equal the number of records."""
        logger = _logger_with_records([0, 1, 2, 0, 1])
        s = logger.summary()
        assert s["total_ticks"] == 5

    def test_max_compromised_nodes_correct(self):
        """max_compromised_nodes must equal the largest compromised_count."""
        logger = _logger_with_records([1, 0, 3, 2])
        s = logger.summary()
        assert s["max_compromised_nodes"] == 3

    def test_mean_uptime_in_valid_range(self):
        """mean_uptime_pct must be in [0, 100]."""
        logger = _logger_with_records([0, 1, 2])
        s = logger.summary()
        assert 0.0 <= s["mean_uptime_pct"] <= 100.0

    def test_compromise_rate_in_valid_range(self):
        """compromise_rate must always be in [0, 1]."""
        for pattern in [[0], [1], [0, 0, 0], [1, 1, 1], [0, 1, 0, 1]]:
            logger = _logger_with_records(pattern)
            s = logger.summary()
            assert 0.0 <= s["compromise_rate"] <= 1.0, (
                f"compromise_rate out of [0,1] for pattern {pattern}: {s['compromise_rate']}"
            )

    def test_mark_detected_sets_time_to_detect(self):
        """mark_detected() must set _time_to_detect and appear in summary."""
        logger = _logger_with_records([0, 0, 0, 1, 0])
        logger.mark_detected(tick=3)
        s = logger.summary()
        assert s["mean_ttd_ticks"] == 3


# ---------------------------------------------------------------------------
# End-to-end: run a real simulation and check compromise_rate is sane
# ---------------------------------------------------------------------------

class TestCompromiseRateEndToEnd:
    def test_real_simulation_compromise_rate_in_range(self):
        """Running a real simulation must produce compromise_rate in [0,1]."""
        from sentinelnet.environment.simulator import build_simulator
        import yaml
        scenarios_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "scenarios.yaml"
        )
        with open(scenarios_path) as f:
            data = yaml.safe_load(f)
        cfg = dict(data["scenarios"][0])
        cfg["max_ticks"] = 40
        sim = build_simulator(cfg, scenarios_path)
        summary = sim.run()
        cr = summary.get("compromise_rate", -1)
        assert 0.0 <= cr <= 1.0, (
            f"compromise_rate={cr} is outside [0,1] in a real simulation"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
