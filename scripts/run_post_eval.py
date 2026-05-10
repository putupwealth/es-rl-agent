"""
Run post-evaluation processing for the latest evaluation report.

Commands:
    python scripts/run_post_eval.py
    python scripts/run_post_eval.py --latest-run-pointer reports/latest_run.txt
"""

import subprocess
import sys
from pathlib import Path


LATEST_RUN_POINTER = Path("reports/latest_run.txt")


def main():
    if not LATEST_RUN_POINTER.exists():
        print(f"ERROR: Latest run pointer not found: {LATEST_RUN_POINTER}", file=sys.stderr)
        sys.exit(1)

    print(f"Using latest run pointer: {LATEST_RUN_POINTER}")
    try:
        resolved_run_dir = LATEST_RUN_POINTER.read_text(encoding="utf-8").strip()
    except Exception as exc:
        print(f"ERROR: Could not read latest run pointer: {exc}", file=sys.stderr)
        sys.exit(1)

    if not resolved_run_dir:
        print(f"ERROR: Latest run pointer is empty: {LATEST_RUN_POINTER}", file=sys.stderr)
        sys.exit(1)

    print(f"Resolved report directory: {resolved_run_dir}")

    print("\nRunning verifier...")
    verify_result = subprocess.run(
        [sys.executable, "scripts/verify_eval_output.py", str(LATEST_RUN_POINTER)]
    )
    if verify_result.returncode != 0:
        print("ERROR: Verifier failed.", file=sys.stderr)
        sys.exit(1)

    print("\nRunning LLM packet builder...")
    packet_result = subprocess.run(
        [sys.executable, "scripts/build_llm_input_packet.py", str(LATEST_RUN_POINTER)]
    )
    if packet_result.returncode != 0:
        print("ERROR: LLM packet builder failed.", file=sys.stderr)
        sys.exit(1)

    print("\nPost-eval pipeline completed successfully.")
    print(f"Artifacts available in: {resolved_run_dir}")
    print(f"Verification file: {resolved_run_dir}\\verification.json")
    print(f"LLM packet file:   {resolved_run_dir}\\llm_input_packet.json")


if __name__ == "__main__":
    main()