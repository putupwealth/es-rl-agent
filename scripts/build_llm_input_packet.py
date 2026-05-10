"""Milestone 3 — LLM input packet generator.

Reads one evaluation report directory, loads deterministic verifier output and
core report artifacts, then writes a compact ``llm_input_packet.json`` for
later LLM review.

Usage::

    python scripts/build_llm_input_packet.py <report_dir>

Examples::

    python scripts/build_llm_input_packet.py reports/run_123
    python scripts/build_llm_input_packet.py reports/latest_run.txt
"""

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


PACKET_VERSION = "1.0.0"
MAX_SAMPLE_ROWS_PER_CATEGORY = 10
STEP_SAMPLE_COLUMNS = [
    "step",
    "timestamp",
    "action",
    "position",
    "attempted_entry_action",
    "blocked_reason",
    "valid_long_zone",
    "valid_short_zone",
    "trade_count",
    "reward",
]
MALFORMED_METRICS_ERROR_MSG = "verification.json contains malformed metrics"


def resolve_report_dir(report_dir_arg: str) -> Path:
    """Resolve a report directory argument.

    Accepts either:
    - a direct report directory path, or
    - a .txt pointer file whose contents are the real report directory path
      (for example reports/latest_run.txt).
    """
    path = Path(report_dir_arg)

    if path.is_file() and path.suffix.lower() == ".txt":
        resolved = path.read_text(encoding="utf-8").strip()
        if not resolved:
            raise ValueError(f"Latest run pointer is empty: {path}")
        return Path(resolved)

    return path


def _to_json_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _non_empty_mask(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip() != ""


def _attempt_mask(series: pd.Series) -> pd.Series:
    """Return a boolean mask for rows that represent entry attempts.

    Attempted entries are treated as present when:
    - numeric values are 1 or 2, or
    - the value is non-numeric but non-empty text.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_attempt = numeric.isin([1, 2])
    textual_attempt = numeric.isna() & _non_empty_mask(series)
    return numeric_attempt | textual_attempt


def load_inputs(report_dir: Path):
    verification = None
    summary = None
    steps_df = None
    trades_df = None
    errors: List[str] = []

    verification_path = report_dir / "verification.json"
    summary_path = report_dir / "eval_summary.json"
    steps_path = report_dir / "steps.csv"
    trades_path = report_dir / "trades.csv"

    if not verification_path.exists():
        errors.append("Missing required file: verification.json")
    else:
        try:
            with open(verification_path, encoding="utf-8") as f:
                verification = json.load(f)
        except Exception as exc:
            errors.append(f"verification.json is not parseable: {exc}")

    if not summary_path.exists():
        errors.append("Missing required file: eval_summary.json")
    else:
        try:
            with open(summary_path, encoding="utf-8") as f:
                summary = json.load(f)
        except Exception as exc:
            errors.append(f"eval_summary.json is not parseable: {exc}")

    if not steps_path.exists():
        errors.append("Missing required file: steps.csv")
    else:
        try:
            steps_df = pd.read_csv(steps_path)
        except Exception as exc:
            errors.append(f"steps.csv is not parseable: {exc}")

    if not trades_path.exists():
        errors.append("Missing required file: trades.csv")
    else:
        try:
            trades_df = pd.read_csv(trades_path)
        except Exception as exc:
            errors.append(f"trades.csv is not parseable: {exc}")

    return verification, summary, steps_df, trades_df, errors


def aggregate_blocked_reason_counts(steps_df: pd.DataFrame) -> dict:
    if steps_df is None or "blocked_reason" not in steps_df.columns or steps_df.empty:
        return {}

    blocked = steps_df.loc[_non_empty_mask(steps_df["blocked_reason"]), "blocked_reason"]
    if blocked.empty:
        return {}

    counts = blocked.astype(str).str.strip().value_counts()
    return {reason: int(count) for reason, count in counts.items()}


def _rows_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    columns = [col for col in STEP_SAMPLE_COLUMNS if col in df.columns]
    records = []

    for _, row in df.loc[:, columns].iterrows():
        record = {}
        for col in columns:
            value = _to_json_value(row[col])
            if value is None:
                continue
            record[col] = value
        records.append(record)

    return records


def extract_step_samples(steps_df: pd.DataFrame, max_rows: int = MAX_SAMPLE_ROWS_PER_CATEGORY) -> dict:
    """Extract bounded, compact step samples used as LLM evidence.

    Categories:
    - valid_zone_rows: rows where either long or short valid zone is true
    - attempted_entry_rows: rows with attempted entry actions
    - blocked_reason_rows: rows with non-empty blocked reasons
    """
    samples = {
        "valid_zone_rows": [],
        "attempted_entry_rows": [],
        "blocked_reason_rows": [],
    }

    if steps_df is None or steps_df.empty:
        return samples

    valid_long = (
        pd.to_numeric(steps_df["valid_long_zone"], errors="coerce") == 1
        if "valid_long_zone" in steps_df.columns
        else pd.Series(False, index=steps_df.index)
    )
    valid_short = (
        pd.to_numeric(steps_df["valid_short_zone"], errors="coerce") == 1
        if "valid_short_zone" in steps_df.columns
        else pd.Series(False, index=steps_df.index)
    )
    valid_zone_mask = valid_long | valid_short

    attempt_mask = (
        _attempt_mask(steps_df["attempted_entry_action"])
        if "attempted_entry_action" in steps_df.columns
        else pd.Series(False, index=steps_df.index)
    )

    blocked_mask = (
        _non_empty_mask(steps_df["blocked_reason"])
        if "blocked_reason" in steps_df.columns
        else pd.Series(False, index=steps_df.index)
    )

    samples["valid_zone_rows"] = _rows_to_records(steps_df.loc[valid_zone_mask].head(max_rows))
    samples["attempted_entry_rows"] = _rows_to_records(steps_df.loc[attempt_mask].head(max_rows))
    samples["blocked_reason_rows"] = _rows_to_records(steps_df.loc[blocked_mask].head(max_rows))

    return samples


def summarize_trades(trades_df: pd.DataFrame) -> dict:
    """Build a compact trade summary from trades.csv data."""
    if trades_df is None or trades_df.empty:
        return {
            "total_trades": 0,
            "long_trades": 0,
            "short_trades": 0,
            "wins": 0,
            "losses": 0,
            "breakeven": 0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
        }

    pnl = (
        pd.to_numeric(trades_df["pnl"], errors="coerce").fillna(0.0)
        if "pnl" in trades_df.columns
        else pd.Series([0.0] * len(trades_df))
    )

    if "direction" in trades_df.columns:
        direction = trades_df["direction"].astype(str).str.upper()
    else:
        direction = pd.Series([], dtype="object")

    total_trades = int(len(trades_df))
    total_pnl = float(pnl.sum())

    return {
        "total_trades": total_trades,
        "long_trades": int((direction == "LONG").sum()) if len(direction) else 0,
        "short_trades": int((direction == "SHORT").sum()) if len(direction) else 0,
        "wins": int((pnl > 0).sum()),
        "losses": int((pnl < 0).sum()),
        "breakeven": int((pnl == 0).sum()),
        "total_pnl": total_pnl,
        "avg_pnl": float(total_pnl / total_trades) if total_trades else 0.0,
    }


def build_packet(report_dir_str: str) -> dict:
    """Build the compact `llm_input_packet.json` payload for one report directory."""
    report_dir = Path(report_dir_str)
    run_id = report_dir.name

    verification, summary, steps_df, trades_df, errors = load_inputs(report_dir)

    if summary:
        run_id = summary.get("experiment", {}).get("run_id", run_id)

    run_metadata = {
        "model_name": summary.get("experiment", {}).get("model_name") if summary else None,
        "model_path": summary.get("experiment", {}).get("model_path") if summary else None,
        "evaluation_seed": summary.get("experiment", {}).get("evaluation_seed") if summary else None,
        "test_rows": summary.get("experiment", {}).get("test_rows") if summary else None,
    }

    verdict = verification.get("verdict") if isinstance(verification, dict) else None
    diagnosis = verification.get("diagnosis") if isinstance(verification, dict) else None
    reason = verification.get("reason") if isinstance(verification, dict) else None
    metrics = verification.get("metrics") if isinstance(verification, dict) else {}

    if verdict is None:
        verdict = "FAIL" if errors else "UNKNOWN"
    if diagnosis is None:
        diagnosis = "missing_or_invalid_outputs" if errors else "unknown"
    if reason is None:
        reason = "; ".join(errors) if errors else ""

    if not isinstance(metrics, dict):
        errors.append(MALFORMED_METRICS_ERROR_MSG)
        metrics = {}

    packet = {
        "version": PACKET_VERSION,
        "run_id": run_id,
        "report_dir": str(report_dir),
        "run_metadata": run_metadata,
        "verdict": verdict,
        "diagnosis": diagnosis,
        "reason": reason,
        "metrics": metrics,
        "blocked_reason_counts": aggregate_blocked_reason_counts(steps_df),
        "step_samples": extract_step_samples(steps_df),
        "trade_summary": summarize_trades(trades_df),
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    if errors:
        packet["input_errors"] = errors

    return packet


def write_packet(packet: dict, report_dir: Path) -> Path:
    out_path = report_dir / "llm_input_packet.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(packet, f, indent=2)
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an LLM input packet for a single report directory."
    )
    parser.add_argument("report_dir", help="Path to the evaluation report directory or latest_run.txt pointer.")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        report_dir = resolve_report_dir(args.report_dir)
    except Exception as exc:
        print(f"ERROR: Could not resolve report directory: {exc}", file=sys.stderr)
        sys.exit(1)

    if not report_dir.exists():
        print(f"ERROR: Report directory does not exist: {report_dir}", file=sys.stderr)
        sys.exit(1)

    if not report_dir.is_dir():
        print(f"ERROR: Path is not a directory: {report_dir}", file=sys.stderr)
        sys.exit(1)

    packet = build_packet(str(report_dir))
    out_path = write_packet(packet, report_dir)

    print(f"Verdict:   {packet['verdict']}")
    print(f"Diagnosis: {packet['diagnosis']}")
    print(f"Written:   {out_path}")

    if packet.get("input_errors"):
        for error in packet["input_errors"]:
            print(f"Input warning: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()