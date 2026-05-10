"""Milestone 2 — Deterministic evaluation verifier.

Reads one evaluation report directory, validates the required artifacts
produced by Milestone 1, classifies the run behaviour, and writes
``verification.json`` into that same directory for later LLM review.

Usage::

    python scripts/verify_eval_output.py <report_dir>

Examples::

    python scripts/verify_eval_output.py reports/run_123
    python scripts/verify_eval_output.py reports/latest_run.txt

Exit codes:
    0  PASS or WARN
    1  FAIL or unrecoverable error
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


VERIFIER_VERSION = "1.0.0"

# Fraction of entry attempts that must be on invalid bars before the run is
# classified as invalid_action_heavy (exclusive lower bound).
INVALID_ACTION_RATIO_THRESHOLD = 0.5

# Maximum number of completed trades that still qualifies a run as
# "very few trades" for the invalid_action_heavy check (inclusive upper bound).
INVALID_ACTION_MAX_TRADES = 1

# Required columns for steps.csv (mirrors STEPS_REQUIRED_COLUMNS in evaluate.py).
STEPS_REQUIRED_COLUMNS = [
    "action",
    "position",
    "attempted_entry_action",
    "blocked_reason",
    "valid_long_zone",
    "valid_short_zone",
    "trade_count",
    "reward",
]


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Input loading and validation
# ---------------------------------------------------------------------------

def load_inputs(report_dir: Path):
    """Load and validate the three required input files.

    Returns
    -------
    summary : dict or None
    steps_df : pd.DataFrame or None
    trades_df : pd.DataFrame or None
    errors : list[str]
        Non-empty when one or more inputs are missing or unparseable.
    """
    summary = None
    steps_df = None
    trades_df = None
    errors: list = []

    summary_path = report_dir / "eval_summary.json"
    steps_path = report_dir / "steps.csv"
    trades_path = report_dir / "trades.csv"

    missing = [
        name
        for name, path in [
            ("eval_summary.json", summary_path),
            ("steps.csv", steps_path),
            ("trades.csv", trades_path),
        ]
        if not path.exists()
    ]
    if missing:
        errors.append(f"Missing required files: {', '.join(missing)}")
        return summary, steps_df, trades_df, errors

    try:
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
    except Exception as exc:
        errors.append(f"eval_summary.json is not parseable: {exc}")

    try:
        steps_df = pd.read_csv(steps_path, low_memory=False)
    except Exception as exc:
        errors.append(f"steps.csv is not parseable: {exc}")

    try:
        trades_df = pd.read_csv(trades_path, low_memory=False)
    except Exception as exc:
        errors.append(f"trades.csv is not parseable: {exc}")

    return summary, steps_df, trades_df, errors


def validate_steps_columns(steps_df: pd.DataFrame) -> list:
    """Return a list of error strings for any missing required columns."""
    missing = [col for col in STEPS_REQUIRED_COLUMNS if col not in steps_df.columns]
    if missing:
        return [f"Missing required steps.csv columns: {', '.join(missing)}"]
    return []


# ---------------------------------------------------------------------------
# Checks and metrics
# ---------------------------------------------------------------------------

def compute_checks_and_metrics(summary, steps_df, trades_df):
    """Compute all helper conditions and raw metrics.

    Returns
    -------
    checks : dict[str, bool]
    metrics : dict[str, int | float]
    """
    diag = summary.get("eligibility_diagnostics", {}) if summary else {}
    perf = summary.get("performance", {}) if summary else {}
    trades_info = summary.get("trades", {}) if summary else {}

    total_steps = int(diag.get("total_steps", 0))
    valid_long_steps = int(diag.get("valid_long_zone_steps", 0))
    valid_short_steps = int(diag.get("valid_short_zone_steps", 0))
    valid_long_zone_pct = float(diag.get("valid_long_zone_pct", 0.0))
    valid_short_zone_pct = float(diag.get("valid_short_zone_pct", 0.0))
    long_attempts_on_valid = int(diag.get("long_entry_attempts_on_valid", 0))
    short_attempts_on_valid = int(diag.get("short_entry_attempts_on_valid", 0))
    long_attempts_on_invalid = int(diag.get("long_entry_attempts_on_invalid", 0))
    short_attempts_on_invalid = int(diag.get("short_entry_attempts_on_invalid", 0))
    total_trades = int(trades_info.get("total_trades", 0))
    total_pnl = float(trades_info.get("total_pnl", 0.0))
    final_realized_equity = float(perf.get("final_realized_equity", 0.0))

    total_valid_attempts = long_attempts_on_valid + short_attempts_on_valid
    total_invalid_attempts = long_attempts_on_invalid + short_attempts_on_invalid

    # Count blocked steps from the step log for finer resolution.
    blocked_step_count = 0
    if steps_df is not None and "blocked_reason" in steps_df.columns:
        blocked_step_count = int(steps_df["blocked_reason"].notna().sum())

    # --- Helper conditions ---

    # At least one bar had a valid long or short setup.
    valid_setup_exists = (valid_long_steps + valid_short_steps) > 0

    # The agent issued at least one entry action (attempted to enter the market).
    # Use attempted_entry_action from the step log when available; fall back to
    # the attempt counts recorded in the summary.
    if steps_df is not None and "attempted_entry_action" in steps_df.columns:
        entry_actions_present = bool(steps_df["attempted_entry_action"].notna().any())
    else:
        entry_actions_present = (total_valid_attempts + total_invalid_attempts) > 0

    # At least one entry was attempted on a valid setup bar.
    valid_attempts_exist = total_valid_attempts > 0

    # At least one completed trade was logged.
    trades_exist = total_trades > 0

    # The agent never left a flat state: no trades and position stayed at 0.
    if steps_df is not None and "position" in steps_df.columns and len(steps_df) > 0:
        max_abs_position = float(steps_df["position"].abs().max())
        fully_flat = (not trades_exist) and (max_abs_position == 0.0)
    else:
        fully_flat = not trades_exist

    checks = {
        "valid_setup_exists": valid_setup_exists,
        "entry_actions_present": entry_actions_present,
        "valid_attempts_exist": valid_attempts_exist,
        "trades_exist": trades_exist,
        "fully_flat": fully_flat,
    }

    metrics = {
        "total_steps": total_steps,
        "valid_long_zone_steps": valid_long_steps,
        "valid_short_zone_steps": valid_short_steps,
        "valid_long_zone_pct": valid_long_zone_pct,
        "valid_short_zone_pct": valid_short_zone_pct,
        "long_entry_attempts_on_valid": long_attempts_on_valid,
        "short_entry_attempts_on_valid": short_attempts_on_valid,
        "long_entry_attempts_on_invalid": long_attempts_on_invalid,
        "short_entry_attempts_on_invalid": short_attempts_on_invalid,
        "total_valid_attempts": total_valid_attempts,
        "total_invalid_attempts": total_invalid_attempts,
        "total_trades": total_trades,
        "total_pnl": total_pnl,
        "final_realized_equity": final_realized_equity,
        "blocked_step_count": blocked_step_count,
    }

    return checks, metrics


# ---------------------------------------------------------------------------
# Diagnosis / verdict classification
# ---------------------------------------------------------------------------

def classify(checks: dict, metrics: dict) -> tuple:
    """Return ``(diagnosis, verdict, reason)`` for the given checks/metrics.

    Diagnosis values
    ----------------
    missing_or_invalid_outputs, no_setup_opportunity, inactive_policy,
    invalid_action_heavy, active_but_blocked, active_but_unprofitable,
    behaviorally_alive

    Verdict values
    --------------
    FAIL, WARN, PASS
    """
    valid_setup_exists = checks["valid_setup_exists"]
    entry_actions_present = checks["entry_actions_present"]
    valid_attempts_exist = checks["valid_attempts_exist"]
    trades_exist = checks["trades_exist"]
    fully_flat = checks["fully_flat"]

    total_valid_attempts = metrics["total_valid_attempts"]
    total_invalid_attempts = metrics["total_invalid_attempts"]
    total_trades = metrics["total_trades"]
    total_pnl = metrics["total_pnl"]
    blocked_step_count = metrics["blocked_step_count"]

    # 1. No valid setup bars at all.
    if not valid_setup_exists:
        return (
            "no_setup_opportunity",
            "FAIL",
            "No valid setup bars were observed during the evaluation period.",
        )

    # 2. Valid setups existed but the policy did nothing.
    if (
        valid_setup_exists
        and not entry_actions_present
        and not valid_attempts_exist
        and not trades_exist
        and fully_flat
    ):
        return (
            "inactive_policy",
            "FAIL",
            "Valid setup bars existed but the policy made no entry attempts and executed no trades.",
        )

    # 3. Many entry attempts were on invalid bars and trades are absent/very low.
    total_attempts = total_valid_attempts + total_invalid_attempts
    if total_attempts > 0:
        invalid_ratio = total_invalid_attempts / total_attempts
    else:
        invalid_ratio = 0.0

    if invalid_ratio > INVALID_ACTION_RATIO_THRESHOLD and total_trades <= INVALID_ACTION_MAX_TRADES:
        return (
            "invalid_action_heavy",
            "WARN",
            (
                f"More than half of all entry attempts were on invalid bars "
                f"({invalid_ratio:.1%} invalid) with very few trades ({total_trades})."
            ),
        )

    # 4. Valid attempts were made but every attempt was blocked; no trades executed.
    if valid_attempts_exist and not trades_exist and blocked_step_count > 0:
        return (
            "active_but_blocked",
            "WARN",
            (
                f"Valid entry attempts were made but all were blocked "
                f"({blocked_step_count} blocked steps); no trades executed."
            ),
        )

    # 5. Trades occurred and the run is non-flat, but economics are poor.
    if trades_exist and not fully_flat and total_pnl <= 0.0:
        return (
            "active_but_unprofitable",
            "WARN",
            f"Trades were executed but economics are poor (total PnL: {total_pnl:.2f}).",
        )

    # 6. Full behavioural signal: setups, valid attempts, trades, non-flat.
    if valid_setup_exists and valid_attempts_exist and trades_exist and not fully_flat:
        return (
            "behaviorally_alive",
            "PASS",
            (
                "Agent is behaviorally active: valid setups observed, valid entry "
                "attempts made, trades executed, and the result is non-flat."
            ),
        )

    # Fallback — catches edge cases such as entry attempts with no valid attempts
    # and no clear dominant pattern.
    return (
        "inactive_policy",
        "FAIL",
        "Policy appears inactive or indeterminate based on available evidence.",
    )


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def verify(report_dir_str: str) -> dict:
    """Validate one report directory and return a verification result dict.

    Parameters
    ----------
    report_dir_str : str
        Path to the evaluation report directory.

    Returns
    -------
    dict
        Complete verification result, ready to be serialised as
        ``verification.json``.
    """
    report_dir = Path(report_dir_str)
    run_id = report_dir.name  # default; may be overridden from summary

    inputs = {
        "eval_summary_json": str(report_dir / "eval_summary.json"),
        "steps_csv": str(report_dir / "steps.csv"),
        "trades_csv": str(report_dir / "trades.csv"),
    }

    summary, steps_df, trades_df, load_errors = load_inputs(report_dir)

    col_errors: list = []
    if steps_df is not None:
        col_errors = validate_steps_columns(steps_df)

    all_errors = load_errors + col_errors

    if all_errors:
        return {
            "version": VERIFIER_VERSION,
            "run_id": run_id,
            "report_dir": str(report_dir),
            "verdict": "FAIL",
            "diagnosis": "missing_or_invalid_outputs",
            "reason": "; ".join(all_errors),
            "inputs": inputs,
            "checks": {},
            "metrics": {},
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # Prefer the run_id embedded in the summary.
    if summary:
        run_id = summary.get("experiment", {}).get("run_id", run_id)

    checks, metrics = compute_checks_and_metrics(summary, steps_df, trades_df)
    diagnosis, verdict, reason = classify(checks, metrics)

    return {
        "version": VERIFIER_VERSION,
        "run_id": run_id,
        "report_dir": str(report_dir),
        "verdict": verdict,
        "diagnosis": diagnosis,
        "reason": reason,
        "inputs": inputs,
        "checks": checks,
        "metrics": metrics,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def write_verification(result: dict, report_dir: Path) -> Path:
    """Serialise *result* as ``verification.json`` inside *report_dir*."""
    out_path = report_dir / "verification.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify evaluation outputs for a single report directory."
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

    result = verify(str(report_dir))
    out_path = write_verification(result, report_dir)

    print(f"Verdict:   {result['verdict']}")
    print(f"Diagnosis: {result['diagnosis']}")
    print(f"Reason:    {result['reason']}")
    print(f"Written:   {out_path}")

    if result["verdict"] == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()