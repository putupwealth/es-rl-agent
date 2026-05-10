"""Milestone 2 — tests for the deterministic evaluation verifier.

Tests cover:
- missing required files
- missing required steps.csv columns
- no setup opportunity (no valid setup bars)
- inactive policy (valid setups, but agent never acted)
- invalid action heavy (most attempts on invalid bars, few/no trades)
- active but blocked (valid attempts, all blocked, no trades)
- active but unprofitable (trades exist, non-flat, economics poor)
- behaviorally alive (valid setups, valid attempts, trades, non-flat)

Run with:
    python test_verify_eval_output.py
"""

import json
import tempfile
from pathlib import Path

import pandas as pd

from scripts.verify_eval_output import (
    STEPS_REQUIRED_COLUMNS,
    VERIFIER_VERSION,
    classify,
    compute_checks_and_metrics,
    load_inputs,
    validate_steps_columns,
    verify,
    write_verification,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_summary(
    *,
    total_steps=200,
    valid_long_zone_steps=50,
    valid_short_zone_steps=30,
    long_entry_attempts_on_valid=0,
    short_entry_attempts_on_valid=0,
    long_entry_attempts_on_invalid=0,
    short_entry_attempts_on_invalid=0,
    total_trades=0,
    total_pnl=0.0,
    final_realized_equity=0.0,
    run_id="test_run",
):
    total_valid = valid_long_zone_steps + valid_short_zone_steps
    return {
        "experiment": {"run_id": run_id},
        "performance": {
            "final_realized_equity": final_realized_equity,
            "final_total_equity": final_realized_equity,
            "cumulative_reward": 0.0,
        },
        "trades": {
            "total_trades": total_trades,
            "total_pnl": total_pnl,
            "win_rate": 0.0,
            "avg_pnl": 0.0,
        },
        "actions": {"0": total_steps},
        "eligibility_diagnostics": {
            "total_steps": total_steps,
            "valid_long_zone_steps": valid_long_zone_steps,
            "valid_short_zone_steps": valid_short_zone_steps,
            "valid_long_zone_pct": round(valid_long_zone_steps / max(1, total_steps), 4),
            "valid_short_zone_pct": round(valid_short_zone_steps / max(1, total_steps), 4),
            "long_entry_attempts_on_valid": long_entry_attempts_on_valid,
            "short_entry_attempts_on_valid": short_entry_attempts_on_valid,
            "long_entry_attempts_on_invalid": long_entry_attempts_on_invalid,
            "short_entry_attempts_on_invalid": short_entry_attempts_on_invalid,
        },
    }


def _make_steps_df(
    rows=10,
    *,
    position=0,
    action=0,
    attempted_entry_action=None,
    blocked_reason=None,
    valid_long_zone=1,
    valid_short_zone=0,
    extra_cols=None,
):
    """Return a minimal steps DataFrame with all required columns."""
    data = {
        "action": [action] * rows,
        "position": [position] * rows,
        "attempted_entry_action": [attempted_entry_action] * rows,
        "blocked_reason": [blocked_reason] * rows,
        "valid_long_zone": [valid_long_zone] * rows,
        "valid_short_zone": [valid_short_zone] * rows,
        "trade_count": [0] * rows,
        "reward": [0.0] * rows,
    }
    if extra_cols:
        data.update(extra_cols)
    return pd.DataFrame(data)


def _make_trades_df(rows=0):
    """Return a minimal trades DataFrame (empty by default)."""
    if rows == 0:
        return pd.DataFrame(columns=["direction", "pnl"])
    return pd.DataFrame(
        {"direction": ["LONG"] * rows, "pnl": [100.0] * rows}
    )


def _write_report(
    tmp_dir,
    *,
    summary=None,
    steps_df=None,
    trades_df=None,
    omit=None,
):
    """Write a report directory and return its Path."""
    report_dir = Path(tmp_dir) / "test_run"
    report_dir.mkdir(parents=True, exist_ok=True)
    omit = omit or set()

    if "eval_summary.json" not in omit:
        s = summary if summary is not None else _make_summary()
        with open(report_dir / "eval_summary.json", "w", encoding="utf-8") as f:
            json.dump(s, f)

    if "steps.csv" not in omit:
        df = steps_df if steps_df is not None else _make_steps_df()
        df.to_csv(report_dir / "steps.csv", index=False)

    if "trades.csv" not in omit:
        tdf = trades_df if trades_df is not None else _make_trades_df()
        tdf.to_csv(report_dir / "trades.csv", index=False)

    return report_dir


# ---------------------------------------------------------------------------
# Tests: load_inputs
# ---------------------------------------------------------------------------

def test_load_inputs_missing_all_files():
    """load_inputs must report errors when all required files are absent."""
    with tempfile.TemporaryDirectory() as tmp:
        empty_dir = Path(tmp) / "empty_run"
        empty_dir.mkdir()
        _, _, _, errors = load_inputs(empty_dir)
    assert errors, "Expected errors for missing files"
    assert any("Missing required files" in e for e in errors)
    print("  PASS  test_load_inputs_missing_all_files")


def test_load_inputs_missing_one_file():
    """load_inputs must report an error when only steps.csv is missing."""
    with tempfile.TemporaryDirectory() as tmp:
        report_dir = _write_report(tmp, omit={"steps.csv"})
        _, steps_df, _, errors = load_inputs(report_dir)
    assert errors, "Expected errors for missing steps.csv"
    assert steps_df is None, "steps_df should be None when file is missing"
    print("  PASS  test_load_inputs_missing_one_file")


def test_load_inputs_all_valid():
    """load_inputs must return no errors when all files are present and valid."""
    with tempfile.TemporaryDirectory() as tmp:
        report_dir = _write_report(tmp)
        summary, steps_df, trades_df, errors = load_inputs(report_dir)
    assert not errors, f"Unexpected errors: {errors}"
    assert summary is not None
    assert steps_df is not None
    assert trades_df is not None
    print("  PASS  test_load_inputs_all_valid")


# ---------------------------------------------------------------------------
# Tests: validate_steps_columns
# ---------------------------------------------------------------------------

def test_validate_steps_columns_all_present():
    """validate_steps_columns must return no errors when all columns exist."""
    df = _make_steps_df()
    errors = validate_steps_columns(df)
    assert not errors, f"Unexpected column errors: {errors}"
    print("  PASS  test_validate_steps_columns_all_present")


def test_validate_steps_columns_missing_one():
    """validate_steps_columns must report an error for each missing column."""
    df = _make_steps_df()
    df = df.drop(columns=["blocked_reason"])
    errors = validate_steps_columns(df)
    assert errors, "Expected column validation error"
    assert "blocked_reason" in errors[0]
    print("  PASS  test_validate_steps_columns_missing_one")


def test_validate_steps_columns_missing_several():
    """validate_steps_columns must flag all missing columns in one message."""
    df = _make_steps_df()
    df = df.drop(columns=["blocked_reason", "valid_long_zone", "reward"])
    errors = validate_steps_columns(df)
    assert errors, "Expected column validation error"
    for col in ["blocked_reason", "valid_long_zone", "reward"]:
        assert col in errors[0], f"Expected {col!r} mentioned in error"
    print("  PASS  test_validate_steps_columns_missing_several")


# ---------------------------------------------------------------------------
# Tests: verify — missing or invalid outputs
# ---------------------------------------------------------------------------

def test_verify_missing_files():
    """verify() must return missing_or_invalid_outputs / FAIL when files absent."""
    with tempfile.TemporaryDirectory() as tmp:
        report_dir = _write_report(tmp, omit={"eval_summary.json", "steps.csv"})
        result = verify(str(report_dir))

    assert result["verdict"] == "FAIL"
    assert result["diagnosis"] == "missing_or_invalid_outputs"
    assert result["version"] == VERIFIER_VERSION
    assert result["checks"] == {}
    assert result["metrics"] == {}
    print("  PASS  test_verify_missing_files")


def test_verify_missing_required_columns():
    """verify() must return missing_or_invalid_outputs / FAIL for missing columns."""
    steps_df = _make_steps_df()
    steps_df = steps_df.drop(columns=["position", "reward"])
    with tempfile.TemporaryDirectory() as tmp:
        report_dir = _write_report(tmp, steps_df=steps_df)
        result = verify(str(report_dir))

    assert result["verdict"] == "FAIL"
    assert result["diagnosis"] == "missing_or_invalid_outputs"
    assert "position" in result["reason"] or "reward" in result["reason"]
    print("  PASS  test_verify_missing_required_columns")


# ---------------------------------------------------------------------------
# Tests: verify — no_setup_opportunity
# ---------------------------------------------------------------------------

def test_verify_no_setup_opportunity():
    """verify() must return no_setup_opportunity / FAIL when no valid bars exist."""
    summary = _make_summary(
        valid_long_zone_steps=0,
        valid_short_zone_steps=0,
    )
    steps_df = _make_steps_df(valid_long_zone=0, valid_short_zone=0)
    with tempfile.TemporaryDirectory() as tmp:
        report_dir = _write_report(tmp, summary=summary, steps_df=steps_df)
        result = verify(str(report_dir))

    assert result["verdict"] == "FAIL"
    assert result["diagnosis"] == "no_setup_opportunity"
    print("  PASS  test_verify_no_setup_opportunity")


# ---------------------------------------------------------------------------
# Tests: verify — inactive_policy
# ---------------------------------------------------------------------------

def test_verify_inactive_policy():
    """verify() must return inactive_policy / FAIL for a dead zero-trade run."""
    summary = _make_summary(
        valid_long_zone_steps=50,
        valid_short_zone_steps=0,
        long_entry_attempts_on_valid=0,
        short_entry_attempts_on_valid=0,
        long_entry_attempts_on_invalid=0,
        short_entry_attempts_on_invalid=0,
        total_trades=0,
    )
    # steps_df: agent always holds, position stays 0
    steps_df = _make_steps_df(
        rows=100,
        action=0,
        position=0,
        attempted_entry_action=None,
        blocked_reason=None,
        valid_long_zone=1,
        valid_short_zone=0,
    )
    with tempfile.TemporaryDirectory() as tmp:
        report_dir = _write_report(tmp, summary=summary, steps_df=steps_df)
        result = verify(str(report_dir))

    assert result["verdict"] == "FAIL"
    assert result["diagnosis"] == "inactive_policy"
    assert result["checks"]["valid_setup_exists"] is True
    assert result["checks"]["entry_actions_present"] is False
    assert result["checks"]["fully_flat"] is True
    print("  PASS  test_verify_inactive_policy")


# ---------------------------------------------------------------------------
# Tests: verify — invalid_action_heavy
# ---------------------------------------------------------------------------

def test_verify_invalid_action_heavy():
    """verify() must return invalid_action_heavy / WARN when most attempts are on invalid bars."""
    summary = _make_summary(
        valid_long_zone_steps=20,
        valid_short_zone_steps=0,
        long_entry_attempts_on_valid=2,
        short_entry_attempts_on_valid=0,
        long_entry_attempts_on_invalid=15,
        short_entry_attempts_on_invalid=0,
        total_trades=0,
    )
    # Build steps where many attempted entries are on invalid bars
    rows = 100
    data = {
        "action": [1] * 17 + [0] * (rows - 17),
        "position": [0] * rows,
        "attempted_entry_action": [1] * 17 + [None] * (rows - 17),
        "blocked_reason": ["invalid_zone"] * 17 + [None] * (rows - 17),
        "valid_long_zone": [1] * 2 + [0] * 15 + [1] * (rows - 17),
        "valid_short_zone": [0] * rows,
        "trade_count": [0] * rows,
        "reward": [0.0] * rows,
    }
    steps_df = pd.DataFrame(data)
    with tempfile.TemporaryDirectory() as tmp:
        report_dir = _write_report(tmp, summary=summary, steps_df=steps_df)
        result = verify(str(report_dir))

    assert result["verdict"] == "WARN"
    assert result["diagnosis"] == "invalid_action_heavy"
    print("  PASS  test_verify_invalid_action_heavy")


# ---------------------------------------------------------------------------
# Tests: verify — active_but_blocked
# ---------------------------------------------------------------------------

def test_verify_active_but_blocked():
    """verify() must return active_but_blocked / WARN when valid attempts are all blocked."""
    summary = _make_summary(
        valid_long_zone_steps=50,
        valid_short_zone_steps=0,
        long_entry_attempts_on_valid=10,
        short_entry_attempts_on_valid=0,
        long_entry_attempts_on_invalid=0,
        short_entry_attempts_on_invalid=0,
        total_trades=0,
    )
    rows = 50
    data = {
        "action": [1] * 10 + [0] * 40,
        "position": [0] * rows,
        "attempted_entry_action": [1] * 10 + [None] * 40,
        "blocked_reason": ["max_trades_reached"] * 10 + [None] * 40,
        "valid_long_zone": [1] * rows,
        "valid_short_zone": [0] * rows,
        "trade_count": [0] * rows,
        "reward": [0.0] * rows,
    }
    steps_df = pd.DataFrame(data)
    with tempfile.TemporaryDirectory() as tmp:
        report_dir = _write_report(tmp, summary=summary, steps_df=steps_df)
        result = verify(str(report_dir))

    assert result["verdict"] == "WARN"
    assert result["diagnosis"] == "active_but_blocked"
    assert result["checks"]["valid_attempts_exist"] is True
    assert result["checks"]["trades_exist"] is False
    print("  PASS  test_verify_active_but_blocked")


# ---------------------------------------------------------------------------
# Tests: verify — active_but_unprofitable
# ---------------------------------------------------------------------------

def test_verify_active_but_unprofitable():
    """verify() must return active_but_unprofitable / WARN when trades exist but PnL <= 0."""
    summary = _make_summary(
        valid_long_zone_steps=50,
        valid_short_zone_steps=0,
        long_entry_attempts_on_valid=5,
        short_entry_attempts_on_valid=0,
        long_entry_attempts_on_invalid=0,
        short_entry_attempts_on_invalid=0,
        total_trades=3,
        total_pnl=-150.0,
        final_realized_equity=-150.0,
    )
    rows = 50
    data = {
        "action": [1] * 5 + [2] * 3 + [0] * 42,
        "position": [1] * 5 + [0] * 3 + [0] * 42,
        "attempted_entry_action": [1] * 5 + [None] * 45,
        "blocked_reason": [None] * rows,
        "valid_long_zone": [1] * rows,
        "valid_short_zone": [0] * rows,
        "trade_count": [1] * 5 + [2] * 3 + [3] * 42,
        "reward": [-10.0] * 5 + [0.0] * 45,
    }
    steps_df = pd.DataFrame(data)
    trades_df = pd.DataFrame({"direction": ["LONG"] * 3, "pnl": [-50.0] * 3})
    with tempfile.TemporaryDirectory() as tmp:
        report_dir = _write_report(tmp, summary=summary, steps_df=steps_df, trades_df=trades_df)
        result = verify(str(report_dir))

    assert result["verdict"] == "WARN"
    assert result["diagnosis"] == "active_but_unprofitable"
    assert result["checks"]["trades_exist"] is True
    print("  PASS  test_verify_active_but_unprofitable")


# ---------------------------------------------------------------------------
# Tests: verify — behaviorally_alive
# ---------------------------------------------------------------------------

def test_verify_behaviorally_alive():
    """verify() must return behaviorally_alive / PASS for a healthy active run."""
    summary = _make_summary(
        valid_long_zone_steps=60,
        valid_short_zone_steps=20,
        long_entry_attempts_on_valid=8,
        short_entry_attempts_on_valid=2,
        long_entry_attempts_on_invalid=0,
        short_entry_attempts_on_invalid=0,
        total_trades=5,
        total_pnl=500.0,
        final_realized_equity=500.0,
    )
    rows = 80
    data = {
        "action": [1] * 8 + [2] * 2 + [0] * 70,
        "position": [1] * 8 + [-1] * 2 + [0] * 70,
        "attempted_entry_action": [1] * 8 + [2] * 2 + [None] * 70,
        "blocked_reason": [None] * rows,
        "valid_long_zone": [1] * 60 + [0] * 20,
        "valid_short_zone": [0] * 60 + [1] * 20,
        "trade_count": list(range(1, rows + 1)),
        "reward": [50.0] * 5 + [0.0] * 75,
    }
    steps_df = pd.DataFrame(data)
    trades_df = pd.DataFrame(
        {"direction": ["LONG"] * 5, "pnl": [100.0] * 5}
    )
    with tempfile.TemporaryDirectory() as tmp:
        report_dir = _write_report(tmp, summary=summary, steps_df=steps_df, trades_df=trades_df)
        result = verify(str(report_dir))

    assert result["verdict"] == "PASS"
    assert result["diagnosis"] == "behaviorally_alive"
    assert result["checks"]["valid_setup_exists"] is True
    assert result["checks"]["valid_attempts_exist"] is True
    assert result["checks"]["trades_exist"] is True
    assert result["checks"]["fully_flat"] is False
    print("  PASS  test_verify_behaviorally_alive")


# ---------------------------------------------------------------------------
# Tests: verification.json output structure
# ---------------------------------------------------------------------------

def test_verify_output_schema():
    """verify() result must contain all required top-level keys."""
    with tempfile.TemporaryDirectory() as tmp:
        report_dir = _write_report(tmp)
        result = verify(str(report_dir))

    required_keys = [
        "version",
        "run_id",
        "report_dir",
        "verdict",
        "diagnosis",
        "reason",
        "inputs",
        "checks",
        "metrics",
        "generated_at",
    ]
    for key in required_keys:
        assert key in result, f"Missing required key {key!r} in verification result"
    print("  PASS  test_verify_output_schema")


def test_write_verification_creates_file():
    """write_verification must create verification.json in the report directory."""
    with tempfile.TemporaryDirectory() as tmp:
        report_dir = _write_report(tmp)
        result = verify(str(report_dir))
        out_path = write_verification(result, report_dir)

        assert out_path.exists(), "verification.json was not created"
        loaded = json.loads(out_path.read_text(encoding="utf-8"))
        assert loaded["version"] == VERIFIER_VERSION
        assert loaded["verdict"] in {"FAIL", "WARN", "PASS"}
        assert loaded["diagnosis"] in {
            "missing_or_invalid_outputs",
            "no_setup_opportunity",
            "inactive_policy",
            "invalid_action_heavy",
            "active_but_blocked",
            "active_but_unprofitable",
            "behaviorally_alive",
        }
    print("  PASS  test_write_verification_creates_file")


def test_verify_run_id_from_summary():
    """verify() must use run_id from eval_summary.json when available."""
    summary = _make_summary(run_id="my_custom_run_42")
    with tempfile.TemporaryDirectory() as tmp:
        report_dir = _write_report(tmp, summary=summary)
        result = verify(str(report_dir))

    assert result["run_id"] == "my_custom_run_42"
    print("  PASS  test_verify_run_id_from_summary")


# ---------------------------------------------------------------------------
# Tests: classify unit tests
# ---------------------------------------------------------------------------

def test_classify_no_setup():
    checks = dict(
        valid_setup_exists=False,
        entry_actions_present=False,
        valid_attempts_exist=False,
        trades_exist=False,
        fully_flat=True,
    )
    metrics = dict(
        total_valid_attempts=0,
        total_invalid_attempts=0,
        total_trades=0,
        total_pnl=0.0,
        blocked_step_count=0,
    )
    diagnosis, verdict, _ = classify(checks, metrics)
    assert diagnosis == "no_setup_opportunity"
    assert verdict == "FAIL"
    print("  PASS  test_classify_no_setup")


def test_classify_inactive():
    checks = dict(
        valid_setup_exists=True,
        entry_actions_present=False,
        valid_attempts_exist=False,
        trades_exist=False,
        fully_flat=True,
    )
    metrics = dict(
        total_valid_attempts=0,
        total_invalid_attempts=0,
        total_trades=0,
        total_pnl=0.0,
        blocked_step_count=0,
    )
    diagnosis, verdict, _ = classify(checks, metrics)
    assert diagnosis == "inactive_policy"
    assert verdict == "FAIL"
    print("  PASS  test_classify_inactive")


def test_classify_behaviorally_alive():
    checks = dict(
        valid_setup_exists=True,
        entry_actions_present=True,
        valid_attempts_exist=True,
        trades_exist=True,
        fully_flat=False,
    )
    metrics = dict(
        total_valid_attempts=10,
        total_invalid_attempts=1,
        total_trades=5,
        total_pnl=250.0,
        blocked_step_count=0,
    )
    diagnosis, verdict, _ = classify(checks, metrics)
    assert diagnosis == "behaviorally_alive"
    assert verdict == "PASS"
    print("  PASS  test_classify_behaviorally_alive")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    test_load_inputs_missing_all_files,
    test_load_inputs_missing_one_file,
    test_load_inputs_all_valid,
    test_validate_steps_columns_all_present,
    test_validate_steps_columns_missing_one,
    test_validate_steps_columns_missing_several,
    test_verify_missing_files,
    test_verify_missing_required_columns,
    test_verify_no_setup_opportunity,
    test_verify_inactive_policy,
    test_verify_invalid_action_heavy,
    test_verify_active_but_blocked,
    test_verify_active_but_unprofitable,
    test_verify_behaviorally_alive,
    test_verify_output_schema,
    test_write_verification_creates_file,
    test_verify_run_id_from_summary,
    test_classify_no_setup,
    test_classify_inactive,
    test_classify_behaviorally_alive,
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
