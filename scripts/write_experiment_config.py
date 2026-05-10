"""
Write experiment configuration metadata for a run.

This script creates or updates:
    reports/<run_id>/experiment_config.json

Usage:
    python scripts/write_experiment_config.py reports/latest_run.txt
    python scripts/write_experiment_config.py reports/latest_run.txt --reward-version v2
    python scripts/write_experiment_config.py reports/run_123 --feature-version breakout_v3 --seed 42
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_RUN_TARGET = "reports/latest_run.txt"
DEFAULT_OUTPUT_FILE = "experiment_config.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Write experiment configuration metadata for a report directory."
    )
    parser.add_argument(
        "run_target",
        nargs="?",
        default=DEFAULT_RUN_TARGET,
        help="Report directory or pointer file. Defaults to reports/latest_run.txt",
    )
    parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help="Output filename inside the report directory.",
    )

    # Core experiment identity
    parser.add_argument("--experiment-name", default=None, help="Optional experiment name.")
    parser.add_argument("--reward-version", default=None, help="Reward system version label.")
    parser.add_argument("--feature-version", default=None, help="Feature set version label.")
    parser.add_argument("--environment-version", default=None, help="Environment version label.")
    parser.add_argument("--policy-version", default=None, help="Policy version label.")

    # Training / evaluation metadata
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Training timesteps.")
    parser.add_argument("--data-file", default=None, help="Training/evaluation data file path.")
    parser.add_argument("--model-file", default=None, help="Model file path used for evaluation.")
    parser.add_argument("--model-dir", default=None, help="Model directory.")
    parser.add_argument("--run-id", default=None, help="Explicit run id if you want to store it.")
    parser.add_argument("--notes", default=None, help="Optional freeform notes.")

    # Reward / rules / features
    parser.add_argument("--entry-window", default=None, help="Entry time window description.")
    parser.add_argument("--max-trades-per-day", type=int, default=None, help="Max trades per day.")
    parser.add_argument("--max-hold-bars", type=int, default=None, help="Maximum hold bars.")
    parser.add_argument("--stop-loss", type=float, default=None, help="Stop loss value.")
    parser.add_argument("--take-profit", type=float, default=None, help="Take profit value.")
    parser.add_argument("--commission", type=float, default=None, help="Commission setting.")
    parser.add_argument("--invalid-action-penalty", type=float, default=None, help="Penalty for invalid actions.")
    parser.add_argument("--hold-penalty", type=float, default=None, help="Penalty for holding.")
    parser.add_argument("--overtrade-penalty", type=float, default=None, help="Penalty for overtrading.")
    parser.add_argument("--drawdown-penalty", type=float, default=None, help="Penalty for drawdown.")
    parser.add_argument("--uses-rth-filter", action="store_true", help="Whether RTH filter is enabled.")
    parser.add_argument("--uses-zone-gating", action="store_true", help="Whether zone gating is enabled.")
    parser.add_argument("--uses-time-features", action="store_true", help="Whether time/session features are enabled.")

    return parser.parse_args()


def resolve_report_dir(run_target: str) -> Path:
    path = Path(run_target)

    if path.is_file() and path.suffix.lower() == ".txt":
        resolved = path.read_text(encoding="utf-8").strip()
        if not resolved:
            raise ValueError(f"Run pointer is empty: {path}")
        return Path(resolved)

    return path


def compact_dict(d: dict) -> dict:
    """
    Remove keys whose values are None.
    Keep booleans, zeroes, and empty strings only if explicitly provided.
    """
    return {k: v for k, v in d.items() if v is not None}


def main():
    args = parse_args()

    try:
        report_dir = resolve_report_dir(args.run_target)
        if not report_dir.exists() or not report_dir.is_dir():
            raise FileNotFoundError(f"Report directory not found: {report_dir}")

        output_path = report_dir / args.output_file

        payload = {
            "written_at_utc": datetime.now(timezone.utc).isoformat(),
            "report_dir": str(report_dir),
            "run_id": args.run_id or report_dir.name,
            "experiment_name": args.experiment_name,
            "training": compact_dict(
                {
                    "seed": args.seed,
                    "total_timesteps": args.total_timesteps,
                    "data_file": args.data_file,
                    "model_dir": args.model_dir,
                    "model_file": args.model_file,
                    "policy_version": args.policy_version,
                }
            ),
            "versions": compact_dict(
                {
                    "reward_version": args.reward_version,
                    "feature_version": args.feature_version,
                    "environment_version": args.environment_version,
                }
            ),
            "rules": compact_dict(
                {
                    "entry_window": args.entry_window,
                    "max_trades_per_day": args.max_trades_per_day,
                    "max_hold_bars": args.max_hold_bars,
                    "uses_rth_filter": args.uses_rth_filter if args.uses_rth_filter else None,
                    "uses_zone_gating": args.uses_zone_gating if args.uses_zone_gating else None,
                    "uses_time_features": args.uses_time_features if args.uses_time_features else None,
                }
            ),
            "reward": compact_dict(
                {
                    "commission": args.commission,
                    "invalid_action_penalty": args.invalid_action_penalty,
                    "hold_penalty": args.hold_penalty,
                    "overtrade_penalty": args.overtrade_penalty,
                    "drawdown_penalty": args.drawdown_penalty,
                    "stop_loss": args.stop_loss,
                    "take_profit": args.take_profit,
                }
            ),
            "notes": args.notes,
        }

        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Written: {output_path}")

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()