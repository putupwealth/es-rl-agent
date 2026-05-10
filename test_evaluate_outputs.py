"""Milestone 1 — verify that evaluation artifacts are written with required structure.

Tests that artifact-writing helpers produce the correct files and columns,
including edge cases such as zero-trade and zero-step runs.

Run with:
    python test_evaluate_outputs.py
"""

import json
import tempfile
from pathlib import Path

import pandas as pd

from evaluate import (
    ELIGIBILITY_DIAGNOSTICS_FIELDS,
    STEPS_REQUIRED_COLUMNS,
    TRADES_COLUMNS,
    build_steps_dataframe,
    build_trade_breakdown,
    build_trades_dataframe,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_step_log(overrides=None):
    base = {
        "timestamp": "2024-01-01 09:00:00",
        "action": 0,
        "reward": 0.0,
        "cumulative_reward": 0.0,
        "position": 0,
        "trade_count": 0,
        "realized_equity": 0.0,
        "unrealized_pnl": 0.0,
        "total_equity": 0.0,
        "exit_reason": None,
        "valid_long_zone": 0,
        "valid_short_zone": 0,
        "allowed_long_entry": 0,
        "allowed_short_entry": 0,
        "entry_rule_trigger": None,
        "blocked_reason": None,
        "attempted_entry_action": None,
    }
    if overrides:
        base.update(overrides)
    return base


def _make_summary(eligibility_overrides=None):
    eligibility = {
        "total_steps": 100,
        "valid_long_zone_steps": 10,
        "valid_short_zone_steps": 5,
        "valid_long_zone_pct": 0.10,
        "valid_short_zone_pct": 0.05,
        "long_entry_attempts_on_valid": 0,
        "short_entry_attempts_on_valid": 0,
        "long_entry_attempts_on_invalid": 0,
        "short_entry_attempts_on_invalid": 0,
    }
    if eligibility_overrides:
        eligibility.update(eligibility_overrides)
    return {
        "experiment": {"run_id": "test_run"},
        "performance": {
            "final_realized_equity": 0.0,
            "final_total_equity": 0.0,
            "cumulative_reward": 0.0,
        },
        "trades": {"total_trades": 0},
        "actions": {"0": 100},
        "eligibility_diagnostics": eligibility,
    }


# ---------------------------------------------------------------------------
# Tests: build_steps_dataframe
# ---------------------------------------------------------------------------

def test_steps_df_zero_steps():
    """steps.csv must have all required columns even when no steps were taken."""
    df = build_steps_dataframe([])
    for col in STEPS_REQUIRED_COLUMNS:
        assert col in df.columns, f"Missing column {col!r} in zero-step DataFrame"
    assert len(df) == 0
    print("  PASS  test_steps_df_zero_steps")


def test_steps_df_normal():
    """steps.csv must contain all required columns for a normal run."""
    logs = [_make_step_log({"action": 0}), _make_step_log({"action": 1})]
    df = build_steps_dataframe(logs)
    for col in STEPS_REQUIRED_COLUMNS:
        assert col in df.columns, f"Missing column {col!r} in normal steps DataFrame"
    assert len(df) == 2
    print("  PASS  test_steps_df_normal")


def test_steps_df_zero_trade_run():
    """All required columns must be present in a run where no trades are executed."""
    logs = [
        _make_step_log({
            "action": 0,
            "valid_long_zone": 1,
            "valid_short_zone": 0,
            "attempted_entry_action": None,
            "blocked_reason": None,
        })
        for _ in range(10)
    ]
    df = build_steps_dataframe(logs)
    for col in STEPS_REQUIRED_COLUMNS:
        assert col in df.columns, f"Missing column {col!r} in zero-trade steps DataFrame"
    assert len(df) == 10
    print("  PASS  test_steps_df_zero_trade_run")


def test_steps_csv_round_trip():
    """Columns written to disk and read back must include all required columns."""
    logs = [_make_step_log() for _ in range(3)]
    df = build_steps_dataframe(logs)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "steps.csv"
        df.to_csv(path, index=False)
        loaded = pd.read_csv(path)
    for col in STEPS_REQUIRED_COLUMNS:
        assert col in loaded.columns, f"Missing column {col!r} after CSV round-trip"
    print("  PASS  test_steps_csv_round_trip")


def test_steps_csv_round_trip_empty():
    """steps.csv written from an empty run must still contain required column headers."""
    df = build_steps_dataframe([])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "steps.csv"
        df.to_csv(path, index=False)
        loaded = pd.read_csv(path)
    for col in STEPS_REQUIRED_COLUMNS:
        assert col in loaded.columns, f"Missing column {col!r} after empty CSV round-trip"
    assert len(loaded) == 0
    print("  PASS  test_steps_csv_round_trip_empty")


# ---------------------------------------------------------------------------
# Tests: build_trades_dataframe
# ---------------------------------------------------------------------------

def test_trades_df_empty():
    """trades.csv must have defined column headers even when no trades executed."""
    df = build_trades_dataframe([])
    assert len(df) == 0
    for col in TRADES_COLUMNS:
        assert col in df.columns, f"Missing column {col!r} in empty trades DataFrame"
    print("  PASS  test_trades_df_empty")


def test_trades_csv_round_trip_empty():
    """trades.csv written from an empty run must have column headers, not just an index."""
    df = build_trades_dataframe([])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trades.csv"
        df.to_csv(path, index=False)
        loaded = pd.read_csv(path)
    for col in TRADES_COLUMNS:
        assert col in loaded.columns, f"Missing column {col!r} after empty trades CSV round-trip"
    assert len(loaded) == 0
    print("  PASS  test_trades_csv_round_trip_empty")


# ---------------------------------------------------------------------------
# Tests: build_trade_breakdown
# ---------------------------------------------------------------------------

def test_trade_breakdown_empty():
    """trade_breakdown.csv must have defined columns even when no trades exist."""
    empty_trades = build_trades_dataframe([])
    breakdown = build_trade_breakdown(empty_trades)
    assert len(breakdown) == 0
    for col in ["direction", "setup_type", "bias_alignment", "trades", "avg_pnl", "total_pnl", "win_rate"]:
        assert col in breakdown.columns, f"Missing column {col!r} in empty breakdown DataFrame"
    print("  PASS  test_trade_breakdown_empty")


def test_trade_breakdown_csv_round_trip_empty():
    """trade_breakdown.csv must be readable with correct columns when no trades exist."""
    empty_trades = build_trades_dataframe([])
    breakdown = build_trade_breakdown(empty_trades)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trade_breakdown.csv"
        breakdown.to_csv(path, index=False)
        loaded = pd.read_csv(path)
    assert len(loaded) == 0
    for col in ["direction", "setup_type", "bias_alignment", "trades", "avg_pnl", "total_pnl", "win_rate"]:
        assert col in loaded.columns, f"Missing column {col!r} after empty breakdown CSV round-trip"
    print("  PASS  test_trade_breakdown_csv_round_trip_empty")


# ---------------------------------------------------------------------------
# Tests: eval_summary.json eligibility diagnostics
# ---------------------------------------------------------------------------

def test_summary_eligibility_diagnostics_fields():
    """eval_summary.json must contain all required eligibility_diagnostics fields."""
    summary = _make_summary()
    assert "eligibility_diagnostics" in summary, "Missing 'eligibility_diagnostics' in summary"
    diag = summary["eligibility_diagnostics"]
    for field in ELIGIBILITY_DIAGNOSTICS_FIELDS:
        assert field in diag, f"Missing field {field!r} in eligibility_diagnostics"
    print("  PASS  test_summary_eligibility_diagnostics_fields")


def test_summary_written_and_readable():
    """eval_summary.json must be written and re-read with all required eligibility fields."""
    summary = _make_summary()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "eval_summary.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f)
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
    diag = loaded.get("eligibility_diagnostics", {})
    for field in ELIGIBILITY_DIAGNOSTICS_FIELDS:
        assert field in diag, f"Missing field {field!r} after round-trip"
    print("  PASS  test_summary_written_and_readable")


# ---------------------------------------------------------------------------
# Tests: all four artifacts written to disk
# ---------------------------------------------------------------------------

def test_all_four_artifacts_written():
    """All four required artifact files must be created by the writing helpers."""
    with tempfile.TemporaryDirectory() as tmp:
        report_dir = Path(tmp)

        # steps.csv
        steps_path = report_dir / "steps.csv"
        build_steps_dataframe([]).to_csv(steps_path, index=False)
        assert steps_path.exists(), "steps.csv was not written"

        # trades.csv
        trades_df = build_trades_dataframe([])
        trades_path = report_dir / "trades.csv"
        trades_df.to_csv(trades_path, index=False)
        assert trades_path.exists(), "trades.csv was not written"

        # trade_breakdown.csv
        breakdown_path = report_dir / "trade_breakdown.csv"
        build_trade_breakdown(trades_df).to_csv(breakdown_path, index=False)
        assert breakdown_path.exists(), "trade_breakdown.csv was not written"

        # eval_summary.json
        summary_path = report_dir / "eval_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(_make_summary(), f)
        assert summary_path.exists(), "eval_summary.json was not written"

        # Verify readable with correct structure
        loaded_steps = pd.read_csv(steps_path)
        for col in STEPS_REQUIRED_COLUMNS:
            assert col in loaded_steps.columns, f"Missing column {col!r} in written steps.csv"

        loaded_trades = pd.read_csv(trades_path)
        for col in TRADES_COLUMNS:
            assert col in loaded_trades.columns, f"Missing column {col!r} in written trades.csv"

        loaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert "eligibility_diagnostics" in loaded_summary
        for field in ELIGIBILITY_DIAGNOSTICS_FIELDS:
            assert field in loaded_summary["eligibility_diagnostics"], \
                f"Missing field {field!r} in written eval_summary.json"

    print("  PASS  test_all_four_artifacts_written")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    test_steps_df_zero_steps,
    test_steps_df_normal,
    test_steps_df_zero_trade_run,
    test_steps_csv_round_trip,
    test_steps_csv_round_trip_empty,
    test_trades_df_empty,
    test_trades_csv_round_trip_empty,
    test_trade_breakdown_empty,
    test_trade_breakdown_csv_round_trip_empty,
    test_summary_eligibility_diagnostics_fields,
    test_summary_written_and_readable,
    test_all_four_artifacts_written,
]

if __name__ == "__main__":
    failures = []
    for test_fn in TESTS:
        try:
            test_fn()
        except Exception as exc:
            failures.append((test_fn.__name__, exc))
            print(f"  FAIL  {test_fn.__name__}: {exc}")

    print()
    if failures:
        print(f"{len(failures)} test(s) FAILED out of {len(TESTS)}.")
        raise SystemExit(1)
    else:
        print(f"All {len(TESTS)} tests passed.")
