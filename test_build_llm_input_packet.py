"""Milestone 3 — tests for the LLM input packet generator.

Run with:
    python test_build_llm_input_packet.py
"""

import json
import tempfile
from pathlib import Path

import pandas as pd

from scripts.build_llm_input_packet import (
    MAX_SAMPLE_ROWS_PER_CATEGORY,
    PACKET_VERSION,
    aggregate_blocked_reason_counts,
    build_packet,
    extract_step_samples,
    write_packet,
)


def _write_json(path: Path, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _make_verification(*, verdict="PASS", diagnosis="behaviorally_alive", reason="ok", metrics=None):
    return {
        "version": "1.0.0",
        "run_id": "run-123",
        "report_dir": "reports/run-123",
        "verdict": verdict,
        "diagnosis": diagnosis,
        "reason": reason,
        "metrics": metrics or {"total_steps": 100, "total_trades": 2},
    }


def _make_summary(*, run_id="run-123"):
    return {
        "experiment": {
            "run_id": run_id,
            "model_name": "ppo_model.zip",
            "model_path": "models/ppo_model.zip",
            "evaluation_seed": 42,
            "test_rows": 500,
        }
    }


def _make_steps(rows=5):
    data = []
    for i in range(rows):
        data.append(
            {
                "step": i,
                "action": 0,
                "position": 0,
                "attempted_entry_action": 1 if i % 2 == 0 else pd.NA,
                "blocked_reason": "entry_not_allowed" if i % 3 == 0 else pd.NA,
                "valid_long_zone": 1 if i % 2 == 0 else 0,
                "valid_short_zone": 1 if i % 4 == 0 else 0,
                "trade_count": 0,
                "reward": 0.0,
            }
        )
    return pd.DataFrame(data)


def _make_trades():
    return pd.DataFrame(
        [
            {"direction": "LONG", "pnl": 10.0},
            {"direction": "SHORT", "pnl": -5.0},
            {"direction": "LONG", "pnl": 0.0},
        ]
    )


def _write_report(tmp: str, *, with_verification=True, malformed_verification=False, steps_rows=5, empty_trades=False):
    report_dir = Path(tmp) / "run-123"
    report_dir.mkdir(parents=True, exist_ok=True)

    if with_verification:
        if malformed_verification:
            (report_dir / "verification.json").write_text("{bad json", encoding="utf-8")
        else:
            _write_json(report_dir / "verification.json", _make_verification())

    _write_json(report_dir / "eval_summary.json", _make_summary())
    _make_steps(steps_rows).to_csv(report_dir / "steps.csv", index=False)

    trades_df = pd.DataFrame(columns=["direction", "pnl"]) if empty_trades else _make_trades()
    trades_df.to_csv(report_dir / "trades.csv", index=False)

    return report_dir


def test_packet_generation_from_valid_run():
    with tempfile.TemporaryDirectory() as tmp:
        report_dir = _write_report(tmp)
        packet = build_packet(str(report_dir))
        out_path = write_packet(packet, report_dir)

        assert out_path.exists()

    assert packet["version"] == PACKET_VERSION
    assert packet["run_id"] == "run-123"
    assert packet["verdict"] == "PASS"
    assert packet["diagnosis"] == "behaviorally_alive"
    assert packet["metrics"]["total_steps"] == 100
    assert packet["trade_summary"]["total_trades"] == 3
    assert "input_errors" not in packet
    print("  PASS  test_packet_generation_from_valid_run")


def test_compact_sample_extraction_bounds():
    with tempfile.TemporaryDirectory() as tmp:
        report_dir = _write_report(tmp, steps_rows=75)
        packet = build_packet(str(report_dir))

    samples = packet["step_samples"]
    assert len(samples["valid_zone_rows"]) <= MAX_SAMPLE_ROWS_PER_CATEGORY
    assert len(samples["attempted_entry_rows"]) <= MAX_SAMPLE_ROWS_PER_CATEGORY
    assert len(samples["blocked_reason_rows"]) <= MAX_SAMPLE_ROWS_PER_CATEGORY
    assert "steps" not in packet
    print("  PASS  test_compact_sample_extraction_bounds")


def test_zero_trade_empty_trades_case():
    with tempfile.TemporaryDirectory() as tmp:
        report_dir = _write_report(tmp, empty_trades=True)
        packet = build_packet(str(report_dir))

    trade_summary = packet["trade_summary"]
    assert trade_summary["total_trades"] == 0
    assert trade_summary["wins"] == 0
    assert trade_summary["losses"] == 0
    assert trade_summary["total_pnl"] == 0.0
    print("  PASS  test_zero_trade_empty_trades_case")


def test_blocked_reason_aggregation():
    steps_df = pd.DataFrame(
        [
            {"blocked_reason": "entry_not_allowed"},
            {"blocked_reason": "entry_not_allowed"},
            {"blocked_reason": "risk_limit"},
            {"blocked_reason": pd.NA},
            {"blocked_reason": ""},
        ]
    )
    counts = aggregate_blocked_reason_counts(steps_df)

    assert counts["entry_not_allowed"] == 2
    assert counts["risk_limit"] == 1
    assert "" not in counts
    print("  PASS  test_blocked_reason_aggregation")


def test_missing_or_malformed_verification_handled_cleanly():
    with tempfile.TemporaryDirectory() as tmp_missing:
        report_missing = _write_report(tmp_missing, with_verification=False)
        packet_missing = build_packet(str(report_missing))

    assert packet_missing["verdict"] == "FAIL"
    assert packet_missing["diagnosis"] == "missing_or_invalid_outputs"
    assert any("verification.json" in err for err in packet_missing["input_errors"])

    with tempfile.TemporaryDirectory() as tmp_bad:
        report_bad = _write_report(tmp_bad, malformed_verification=True)
        packet_bad = build_packet(str(report_bad))

    assert packet_bad["verdict"] == "FAIL"
    assert packet_bad["diagnosis"] == "missing_or_invalid_outputs"
    assert any("verification.json is not parseable" in err for err in packet_bad["input_errors"])
    print("  PASS  test_missing_or_malformed_verification_handled_cleanly")


def test_extract_step_samples_categories_present():
    samples = extract_step_samples(_make_steps(rows=12), max_rows=4)
    assert set(samples.keys()) == {"valid_zone_rows", "attempted_entry_rows", "blocked_reason_rows"}
    assert all(len(v) <= 4 for v in samples.values())
    print("  PASS  test_extract_step_samples_categories_present")


TESTS = [
    test_packet_generation_from_valid_run,
    test_compact_sample_extraction_bounds,
    test_zero_trade_empty_trades_case,
    test_blocked_reason_aggregation,
    test_missing_or_malformed_verification_handled_cleanly,
    test_extract_step_samples_categories_present,
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
