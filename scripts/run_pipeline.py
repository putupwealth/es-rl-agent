"""
Unified pipeline runner for training, evaluation, post-eval processing, and run comparison.

Command behavior
----------------

1) Default:
       python scripts/run_pipeline.py

   Runs:
       - evaluate.py
       - scripts/run_post_eval.py

   Does NOT run:
       - train.py
       - scripts/compare_runs.py

2) Train + evaluate + post-eval:
       python scripts/run_pipeline.py --train

   Runs:
       - train.py
       - evaluate.py
       - scripts/run_post_eval.py

   Does NOT run:
       - scripts/compare_runs.py

3) Compare added:
       python scripts/run_pipeline.py --compare

   Runs:
       - evaluate.py
       - scripts/run_post_eval.py
       - scripts/compare_runs.py

   Also writes comparison CSV by default to:
       reports/comparisons/run_comparison_YYYYMMDD_HHMMSS.csv

4) Train + evaluate + post-eval + compare:
       python scripts/run_pipeline.py --train --compare

   Runs:
       - train.py
       - evaluate.py
       - scripts/run_post_eval.py
       - scripts/compare_runs.py

   Also writes comparison CSV by default to:
       reports/comparisons/run_comparison_YYYYMMDD_HHMMSS.csv

5) All stages:
       python scripts/run_pipeline.py --all

   Runs:
       - train.py
       - evaluate.py
       - scripts/run_post_eval.py
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

    # --train
    # Adds train.py before the normal evaluate + post-eval flow.
    #
    # Example:
    #     python scripts/run_pipeline.py --train
    #
    # Runs:
    #     1. train.py
    #     2. evaluate.py
    #     3. scripts/run_post_eval.py
    #
    # Does NOT run compare unless --compare is also passed.
    parser.add_argument("--train", action="store_true", help="Run train.py before evaluation.")

    # --compare
    # Adds compare_runs.py after evaluation + post-eval.
    #
    # Example:
    #     python scripts/run_pipeline.py --compare
    #
    # Runs:
    #     1. evaluate.py
    #     2. scripts/run_post_eval.py
    #     3. scripts/compare_runs.py
    #
    # This also writes a comparison CSV by default to reports/comparisons/.
    parser.add_argument("--compare", action="store_true", help="Run compare_runs.py after post-eval.")

    # --all
    # Runs every major stage in the pipeline.
    #
    # Example:
    #     python scripts/run_pipeline.py --all
    #
    # Runs:
    #     1. train.py
    #     2. evaluate.py
    #     3. scripts/run_post_eval.py
    #     4. scripts/compare_runs.py
    #
    # This is the easiest "run everything" option.
    parser.add_argument("--all", action="store_true", help="Run all stages: train, evaluate, post-eval, compare.")

    # --evaluate-only
    # Runs only evaluate.py and exits.
    parser.add_argument("--evaluate-only", action="store_true", help="Run only evaluate.py.")

    # --post-eval-only
    # Runs only scripts/run_post_eval.py and exits.
    parser.add_argument("--post-eval-only", action="store_true", help="Run only scripts/run_post_eval.py.")

    # --compare-only
    # Runs only scripts/compare_runs.py and exits.
    # Also writes CSV by default.
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
        "--latest-run-pointer",
        args.latest_run_pointer,
    ]


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

    # --------------------------------------------------------------
    # Expand --all into full pipeline behavior.
    #
    # python scripts/run_pipeline.py --all
    #
    # means:
    #   - train
    #   - evaluate
    #   - post-eval
    #   - compare
    # --------------------------------------------------------------
    if args.all:
        args.train = True
        args.compare = True

    # Prevent conflicting "only" modes.
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

    # Prevent mixing --all with "*-only" modes.
    if args.all and (args.evaluate_only or args.post_eval_only or args.compare_only):
        print(
            "ERROR: --all cannot be combined with --evaluate-only, --post-eval-only, or --compare-only.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --------------------------------------------------------------
    # MODE A: compare only
    #
    # Command:
    #   python scripts/run_pipeline.py --compare-only
    #
    # Runs:
    #   1. scripts/compare_runs.py
    # --------------------------------------------------------------
    if args.compare_only:
        run_command(build_compare_command(args))
        return

    # --------------------------------------------------------------
    # MODE B: post-eval only
    #
    # Command:
    #   python scripts/run_pipeline.py --post-eval-only
    #
    # Runs:
    #   1. scripts/run_post_eval.py
    # --------------------------------------------------------------
    if args.post_eval_only:
        run_command(build_post_eval_command(args))
        return

    # --------------------------------------------------------------
    # MODE C: evaluate only
    #
    # Command:
    #   python scripts/run_pipeline.py --evaluate-only
    #
    # Runs:
    #   1. evaluate.py
    # --------------------------------------------------------------
    if args.evaluate_only:
        run_command(build_evaluate_command(args))
        return

    # --------------------------------------------------------------
    # MODE D: normal / full pipeline
    #
    # Default command:
    #   python scripts/run_pipeline.py
    #
    # Runs:
    #   1. evaluate.py
    #   2. scripts/run_post_eval.py
    #
    # Train command:
    #   python scripts/run_pipeline.py --train
    #
    # Runs:
    #   1. train.py
    #   2. evaluate.py
    #   3. scripts/run_post_eval.py
    #
    # Compare command:
    #   python scripts/run_pipeline.py --compare
    #
    # Runs:
    #   1. evaluate.py
    #   2. scripts/run_post_eval.py
    #   3. scripts/compare_runs.py
    #
    # All command:
    #   python scripts/run_pipeline.py --all
    #
    # Runs:
    #   1. train.py
    #   2. evaluate.py
    #   3. scripts/run_post_eval.py
    #   4. scripts/compare_runs.py
    # --------------------------------------------------------------
    if args.train:
        run_command(build_train_command(args))

    run_command(build_evaluate_command(args))
    run_command(build_post_eval_command(args))

    if args.compare:
        run_command(build_compare_command(args))

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()