"""
Unified pipeline runner for training, evaluation, post-eval processing, and run comparison.

Command behavior
----------------

1) Default:
       python scripts/run_pipeline.py

   Runs:
       - evaluate.py
       - scripts/run_post_eval.py
       - scripts/write_experiment_config.py

   Does NOT run:
       - train.py
       - scripts/compare_runs.py

2) Train + evaluate + post-eval:
       python scripts/run_pipeline.py --train

   Runs:
       - train.py
       - evaluate.py
       - scripts/run_post_eval.py
       - scripts/write_experiment_config.py

   Does NOT run:
       - scripts/compare_runs.py

3) Compare added:
       python scripts/run_pipeline.py --compare

   Runs:
       - evaluate.py
       - scripts/run_post_eval.py
       - scripts/write_experiment_config.py
       - scripts/compare_runs.py

   Also writes comparison CSV by default to:
       reports/comparisons/run_comparison_YYYYMMDD_HHMMSS.csv

4) Train + evaluate + post-eval + compare:
       python scripts/run_pipeline.py --train --compare

   Runs:
       - train.py
       - evaluate.py
       - scripts/run_post_eval.py
       - scripts/write_experiment_config.py
       - scripts/compare_runs.py

   Also writes comparison CSV by default to:
       reports/comparisons/run_comparison_YYYYMMDD_HHMMSS.csv

5) All stages:
       python scripts/run_pipeline.py --all

   Runs:
       - train.py
       - evaluate.py
       - scripts/run_post_eval.py
       - scripts/write_experiment_config.py
       - scripts/compare_runs.py

   Also writes comparison CSV by default to:
       reports/comparisons/run_comparison_YYYYMMDD_HHMMSS.csv

6) Evaluate only:
       python scripts/run_pipeline.py --evaluate-only

   Runs:
       - evaluate.py

7) Post-eval only:
       python scripts/run_pipeline.py --post-eval-only

   Runs:
       - scripts/run_post_eval.py
       - scripts/write_experiment_config.py

8) Compare only:
       python scripts/run_pipeline.py --compare-only

   Runs:
       - scripts/compare_runs.py

   Also writes comparison CSV by default to:
       reports/comparisons/run_comparison_YYYYMMDD_HHMMSS.csv

Common examples
---------------

    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --train
    python scripts/run_pipeline.py --compare
    python scripts/run_pipeline.py --train --compare
    python scripts/run_pipeline.py --all
    python scripts/run_pipeline.py --evaluate-only
    python scripts/run_pipeline.py --post-eval-only
    python scripts/run_pipeline.py --compare-only
    python scripts/run_pipeline.py --compare-only --latest 10
    python scripts/run_pipeline.py --all --latest 10 --csv-dir reports/comparisons

Notes
-----

- --train by itself does NOT run compare.
- --compare adds the comparison stage after evaluation/post-eval.
- --all runs every stage.
- compare output CSV defaults to reports/comparisons/.
"""

import argparse
import subprocess
import sys


DEFAULT_LATEST_RUN_POINTER = "reports/latest_run.txt"
DEFAULT_REPORTS_DIR = "reports"
DEFAULT_COMPARISON_CSV_DIR = "reports/comparisons"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run train/evaluate/post-eval/compare pipeline with flexible options."
    )

    # ------------------------------------------------------------------
    # PIPELINE STAGE FLAGS
    # ------------------------------------------------------------------

    parser.add_argument("--train", action="store_true", help="Run train.py before evaluation.")
    parser.add_argument("--compare", action="store_true", help="Run compare_runs.py after post-eval.")
    parser.add_argument("--all", action="store_true", help="Run all stages: train, evaluate, post-eval, compare.")
    parser.add_argument("--evaluate-only", action="store_true", help="Run only evaluate.py.")
    parser.add_argument("--post-eval-only", action="store_true", help="Run only scripts/run_post_eval.py.")
    parser.add_argument("--compare-only", action="store_true", help="Run only scripts/compare_runs.py.")

    # ------------------------------------------------------------------
    # SHARED PATHS / BASIC OPTIONS
    # ------------------------------------------------------------------
    parser.add_argument("--latest-run-pointer", default=DEFAULT_LATEST_RUN_POINTER, help="Path to latest run pointer.")
    parser.add_argument("--reports-dir", default=DEFAULT_REPORTS_DIR, help="Reports directory for evaluate.py / compare_runs.py.")
    parser.add_argument("--latest", type=int, default=None, help="Latest N runs for compare_runs.py.")
    parser.add_argument(
        "--csv-dir",
        default=DEFAULT_COMPARISON_CSV_DIR,
        help="Directory for timestamped comparison CSV output.",
    )

    # ------------------------------------------------------------------
    # PASS-THROUGH OPTIONS FOR train.py / evaluate.py
    # ------------------------------------------------------------------
    parser.add_argument("--data-file", default=None, help="Optional data file path for train.py/evaluate.py.")
    parser.add_argument("--model-dir", default=None, help="Optional model directory for train.py.")
    parser.add_argument("--latest-model-pointer", default=None, help="Optional latest model pointer for train.py/evaluate.py.")
    parser.add_argument("--run-id", default=None, help="Optional explicit run ID for train.py or evaluate.py.")
    parser.add_argument("--version", type=int, default=None, help="Optional version for train.py.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for train.py/evaluate.py.")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Optional total timesteps for train.py.")
    parser.add_argument("--model-file", default=None, help="Optional explicit model file for evaluate.py.")

    # ------------------------------------------------------------------
    # EXPERIMENT CONFIG METADATA
    # ------------------------------------------------------------------
    parser.add_argument("--experiment-name", default=None, help="Optional experiment name for experiment_config.json.")
    parser.add_argument("--reward-version", default=None, help="Reward system version label.")
    parser.add_argument("--feature-version", default=None, help="Feature set version label.")
    parser.add_argument("--environment-version", default=None, help="Environment version label.")
    parser.add_argument("--policy-version", default=None, help="Policy version label.")
    parser.add_argument("--notes", default=None, help="Optional notes for experiment_config.json.")
    parser.add_argument("--entry-window", default=None, help="Entry window description.")
    parser.add_argument("--max-trades-per-day", type=int, default=None, help="Max trades per day.")
    parser.add_argument("--max-hold-bars", type=int, default=None, help="Max hold bars.")
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

    # ------------------------------------------------------------------
    # EXTRA OPTIONS FOR compare_runs.py
    # ------------------------------------------------------------------
    parser.add_argument("--rank-by", action="store_true", help="Sort compare output by composite_score.")
    parser.add_argument("--only-pass", action="store_true", help="Only include PASS runs in comparison.")
    parser.add_argument("--min-trades", type=int, default=None, help="Only include runs with at least this many trades.")
    parser.add_argument(
        "--max-invalid-attempts",
        type=int,
        default=None,
        help="Only include runs with at most this many invalid attempts.",
    )
    parser.add_argument("--markdown", action="store_true", help="Use markdown output for compare_runs.py.")
    parser.add_argument("--full", action="store_true", help="Use full output for compare_runs.py.")
    parser.add_argument("--color", action="store_true", help="Use colorized output for compare_runs.py.")
    parser.add_argument("--top-n", type=int, default=None, help="Top-N summary size for compare_runs.py.")

    return parser.parse_args()


def run_command(cmd):
    """Run a subprocess command and stop the pipeline if it fails."""
    print(f"\n>>> Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"ERROR: Command failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def build_train_command(args):
    """Build command for train.py."""
    cmd = [sys.executable, "train.py"]

    if args.data_file:
        cmd += ["--data-file", args.data_file]
    if args.model_dir:
        cmd += ["--model-dir", args.model_dir]
    if args.latest_model_pointer:
        cmd += ["--latest-pointer", args.latest_model_pointer]
    if args.run_id:
        cmd += ["--run-id", args.run_id]
    if args.version is not None:
        cmd += ["--version", str(args.version)]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    if args.total_timesteps is not None:
        cmd += ["--total-timesteps", str(args.total_timesteps)]

    return cmd


def build_evaluate_command(args):
    """Build command for evaluate.py."""
    cmd = [sys.executable, "evaluate.py"]

    if args.data_file:
        cmd += ["--data-file", args.data_file]
    if args.model_file:
        cmd += ["--model-file", args.model_file]
    if args.latest_model_pointer:
        cmd += ["--latest-model-pointer", args.latest_model_pointer]
    if args.reports_dir:
        cmd += ["--reports-dir", args.reports_dir]
    if args.latest_run_pointer:
        cmd += ["--latest-run-pointer", args.latest_run_pointer]
    if args.run_id:
        cmd += ["--run-id", args.run_id]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]

    return cmd


def build_post_eval_command(args):
    """Build command for scripts/run_post_eval.py."""
    return [
        sys.executable,
        "scripts/run_post_eval.py",
        args.latest_run_pointer,
    ]


def build_experiment_config_command(args):
    """Build command for scripts/write_experiment_config.py."""
    cmd = [
        sys.executable,
        "scripts/write_experiment_config.py",
        args.latest_run_pointer,
    ]

    if args.experiment_name:
        cmd += ["--experiment-name", args.experiment_name]
    if args.reward_version:
        cmd += ["--reward-version", args.reward_version]
    if args.feature_version:
        cmd += ["--feature-version", args.feature_version]
    if args.environment_version:
        cmd += ["--environment-version", args.environment_version]
    if args.policy_version:
        cmd += ["--policy-version", args.policy_version]
    if args.notes:
        cmd += ["--notes", args.notes]

    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    if args.total_timesteps is not None:
        cmd += ["--total-timesteps", str(args.total_timesteps)]
    if args.data_file:
        cmd += ["--data-file", args.data_file]
    if args.model_file:
        cmd += ["--model-file", args.model_file]
    if args.model_dir:
        cmd += ["--model-dir", args.model_dir]
    if args.run_id:
        cmd += ["--run-id", args.run_id]

    if args.entry_window:
        cmd += ["--entry-window", args.entry_window]
    if args.max_trades_per_day is not None:
        cmd += ["--max-trades-per-day", str(args.max_trades_per_day)]
    if args.max_hold_bars is not None:
        cmd += ["--max-hold-bars", str(args.max_hold_bars)]
    if args.stop_loss is not None:
        cmd += ["--stop-loss", str(args.stop_loss)]
    if args.take_profit is not None:
        cmd += ["--take-profit", str(args.take_profit)]
    if args.commission is not None:
        cmd += ["--commission", str(args.commission)]
    if args.invalid_action_penalty is not None:
        cmd += ["--invalid-action-penalty", str(args.invalid_action_penalty)]
    if args.hold_penalty is not None:
        cmd += ["--hold-penalty", str(args.hold_penalty)]
    if args.overtrade_penalty is not None:
        cmd += ["--overtrade-penalty", str(args.overtrade_penalty)]
    if args.drawdown_penalty is not None:
        cmd += ["--drawdown-penalty", str(args.drawdown_penalty)]

    if args.uses_rth_filter:
        cmd += ["--uses-rth-filter"]
    if args.uses_zone_gating:
        cmd += ["--uses-zone-gating"]
    if args.uses_time_features:
        cmd += ["--uses-time-features"]

    return cmd


def build_compare_command(args):
    """Build command for scripts/compare_runs.py."""
    cmd = [sys.executable, "scripts/compare_runs.py"]

    if args.reports_dir:
        cmd += ["--reports-dir", args.reports_dir]
    if args.latest is not None:
        cmd += ["--latest", str(args.latest)]

    # Compare should default CSV output to reports/comparisons/.
    if args.csv_dir:
        cmd += ["--csv-dir", args.csv_dir]

    if args.rank_by:
        cmd += ["--rank-by"]
    if args.only_pass:
        cmd += ["--only-pass"]
    if args.min_trades is not None:
        cmd += ["--min-trades", str(args.min_trades)]
    if args.max_invalid_attempts is not None:
        cmd += ["--max-invalid-attempts", str(args.max_invalid_attempts)]
    if args.markdown:
        cmd += ["--markdown"]
    if args.full:
        cmd += ["--full"]
    if args.color:
        cmd += ["--color"]
    if args.top_n is not None:
        cmd += ["--top-n", str(args.top_n)]

    return cmd


def main():
    args = parse_args()

    if args.all:
        args.train = True
        args.compare = True

    selected_only_modes = sum(
        [
            1 if args.evaluate_only else 0,
            1 if args.post_eval_only else 0,
            1 if args.compare_only else 0,
        ]
    )
    if selected_only_modes > 1:
        print(
            "ERROR: Use only one of --evaluate-only, --post-eval-only, or --compare-only.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.all and (args.evaluate_only or args.post_eval_only or args.compare_only):
        print(
            "ERROR: --all cannot be combined with --evaluate-only, --post-eval-only, or --compare-only.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.compare_only:
        run_command(build_compare_command(args))
        return

    if args.post_eval_only:
        run_command(build_post_eval_command(args))
        run_command(build_experiment_config_command(args))
        return

    if args.evaluate_only:
        run_command(build_evaluate_command(args))
        return

    if args.train:
        run_command(build_train_command(args))

    run_command(build_evaluate_command(args))
    run_command(build_post_eval_command(args))
    run_command(build_experiment_config_command(args))

    if args.compare:
        run_command(build_compare_command(args))

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()