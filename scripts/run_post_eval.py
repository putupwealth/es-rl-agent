"""
Run post-evaluation steps for one report directory.

This script runs:
1. Deterministic verifier
2. LLM input packet builder
3. LLM review generator

Usage:
    python scripts/run_post_eval.py
    python scripts/run_post_eval.py reports/latest_run.txt
    python scripts/run_post_eval.py reports/run_123
"""

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_RUN_TARGET = "reports/latest_run.txt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run verifier, packet builder, and LLM review for one report."
    )
    parser.add_argument(
        "run_target",
        nargs="?",
        default=DEFAULT_RUN_TARGET,
        help="Report directory or pointer file. Defaults to reports/latest_run.txt",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for child scripts.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model name to pass to review_with_llm.py",
    )
    parser.add_argument(
        "--stdout-review",
        action="store_true",
        help="Print LLM review to stdout instead of writing llm_review.md",
    )
    return parser.parse_args()


def run_step(cmd, step_name: str):
    print(f"\n=== {step_name} ===")
    print(" ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"{step_name} failed with exit code {result.returncode}")


def resolve_report_dir(run_target: str) -> Path:
    path = Path(run_target)

    if path.is_file() and path.suffix.lower() == ".txt":
        resolved = path.read_text(encoding="utf-8").strip()
        if not resolved:
            raise ValueError(f"Run pointer is empty: {path}")
        return Path(resolved)

    return path


def main():
    args = parse_args()

    try:
        run_step(
            [args.python, "scripts/verify_eval_output.py", args.run_target],
            "Verify evaluation output",
        )

        run_step(
            [args.python, "scripts/build_llm_input_packet.py", args.run_target],
            "Build LLM input packet",
        )

        review_cmd = [
            args.python,
            "scripts/review_with_llm.py",
            args.run_target,
            "--model",
            args.model,
        ]
        if args.stdout_review:
            review_cmd.append("--stdout-only")

        run_step(
            review_cmd,
            "Generate LLM review",
        )

        report_dir = resolve_report_dir(args.run_target)

        print("\n=== Post-eval complete ===")
        print(f"Report directory:      {report_dir}")
        print(f"Verification file:     {report_dir / 'verification.json'}")
        print(f"LLM input packet:      {report_dir / 'llm_input_packet.json'}")
        if args.stdout_review:
            print("LLM review:            printed to stdout")
        else:
            print(f"LLM review:            {report_dir / 'llm_review.md'}")

    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()