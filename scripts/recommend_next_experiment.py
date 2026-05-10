"""
Recommend the next experiment based on recent evaluation runs.

This Phase 5 V1 implementation is deterministic and rule-based.
It scans recent report folders, loads evaluation/verification/config metadata,
identifies the latest run and best benchmark run, detects repeated failure
patterns, and prints a recommendation. It can also save the recommendation
to a markdown file.

Examples
--------
python scripts/recommend_next_experiment.py
python scripts/recommend_next_experiment.py --reports-dir reports --latest 10
python scripts/recommend_next_experiment.py --reports-dir reports --latest 15 --save
python scripts/recommend_next_experiment.py --out reports/recommendations/my_recommendation.md
"""

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


DEFAULT_REPORTS_DIR = "reports"
DEFAULT_RECOMMENDATIONS_DIR = "reports/recommendations"


def safe_load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def format_pct(value) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return str(value)


def format_num(value, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return str(value)


def format_label(value) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    return str(value)


def is_run_directory(path: Path) -> bool:
    if not path.is_dir():
        return False

    name_lower = path.name.lower()

    if name_lower in {"latest", "comparisons", "recommendations"}:
        return False

    if "_eval_" in name_lower:
        return False

    return (
        (path / "eval_summary.json").exists()
        or (path / "verification.json").exists()
        or (path / "experiment_config.json").exists()
        or (path / "llm_input_packet.json").exists()
        or (path / "steps.csv").exists()
        or (path / "trades.csv").exists()
    )


def collect_run_dirs(reports_dir: str, latest: int) -> List[Path]:
    base = Path(reports_dir)
    if not base.exists():
        raise FileNotFoundError(f"Reports directory does not exist: {base}")
    if not base.is_dir():
        raise NotADirectoryError(f"Reports directory is not a directory: {base}")

    run_dirs = [p for p in base.iterdir() if is_run_directory(p)]
    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    if latest is not None:
        run_dirs = run_dirs[:latest]

    return run_dirs


def extract_run_row(report_dir: Path) -> Dict:
    summary = safe_load_json(report_dir / "eval_summary.json") or {}
    verification = safe_load_json(report_dir / "verification.json") or {}
    experiment_config = safe_load_json(report_dir / "experiment_config.json") or {}

    experiment = summary.get("experiment", {})
    performance = summary.get("performance", {})
    trades = summary.get("trades", {})
    eligibility = summary.get("eligibility_diagnostics", {})

    metrics = verification.get("metrics", {})

    config_training = experiment_config.get("training", {})
    config_versions = experiment_config.get("versions", {})
    config_rules = experiment_config.get("rules", {})
    config_reward = experiment_config.get("reward", {})

    long_invalid = eligibility.get("long_entry_attempts_on_invalid", metrics.get("long_entry_attempts_on_invalid", 0)) or 0
    short_invalid = eligibility.get("short_entry_attempts_on_invalid", metrics.get("short_entry_attempts_on_invalid", 0)) or 0
    long_valid = eligibility.get("long_entry_attempts_on_valid", metrics.get("long_entry_attempts_on_valid", 0)) or 0
    short_valid = eligibility.get("short_entry_attempts_on_valid", metrics.get("short_entry_attempts_on_valid", 0)) or 0

    total_invalid_attempts = long_invalid + short_invalid
    total_valid_attempts = long_valid + short_valid
    total_attempts = total_valid_attempts + total_invalid_attempts
    invalid_attempt_ratio = (total_invalid_attempts / total_attempts) if total_attempts else None

    verdict = verification.get("verdict")
    diagnosis = verification.get("diagnosis")
    verdict_score = {"PASS": 2, "WARN": 1, "FAIL": 0}.get(verdict, -1)

    total_pnl = trades.get("total_pnl")
    total_trades = trades.get("total_trades")
    win_rate = trades.get("win_rate")

    benchmark_score = None
    try:
        benchmark_score = (
            verdict_score * 100000.0
            + (float(total_pnl) if total_pnl is not None else 0.0) * 10.0
            + (float(total_trades) if total_trades is not None else 0.0) * 5.0
            - float(total_invalid_attempts) * 0.5
        )
    except Exception:
        benchmark_score = None

    return {
        "report_dir": str(report_dir),
        "run_id": experiment.get("run_id", report_dir.name),
        "verdict": verdict,
        "diagnosis": diagnosis,
        "reason": verification.get("reason"),
        "reward_version": config_versions.get("reward_version"),
        "feature_version": config_versions.get("feature_version"),
        "environment_version": config_versions.get("environment_version"),
        "policy_version": config_training.get("policy_version"),
        "seed": config_training.get("seed", experiment.get("evaluation_seed")),
        "experiment_name": experiment_config.get("experiment_name"),
        "notes": experiment_config.get("notes"),
        "uses_rth_filter": config_rules.get("uses_rth_filter"),
        "uses_zone_gating": config_rules.get("uses_zone_gating"),
        "uses_time_features": config_rules.get("uses_time_features"),
        "entry_window": config_rules.get("entry_window"),
        "max_trades_per_day": config_rules.get("max_trades_per_day"),
        "max_hold_bars": config_rules.get("max_hold_bars"),
        "invalid_action_penalty": config_reward.get("invalid_action_penalty"),
        "hold_penalty": config_reward.get("hold_penalty"),
        "overtrade_penalty": config_reward.get("overtrade_penalty"),
        "drawdown_penalty": config_reward.get("drawdown_penalty"),
        "commission": config_reward.get("commission"),
        "stop_loss": config_reward.get("stop_loss"),
        "take_profit": config_reward.get("take_profit"),
        "final_realized_equity": performance.get("final_realized_equity"),
        "final_total_equity": performance.get("final_total_equity"),
        "cumulative_reward": performance.get("cumulative_reward"),
        "total_trades": total_trades,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "total_steps": eligibility.get("total_steps", metrics.get("total_steps")),
        "valid_long_zone_pct": eligibility.get("valid_long_zone_pct", metrics.get("valid_long_zone_pct")),
        "valid_short_zone_pct": eligibility.get("valid_short_zone_pct", metrics.get("valid_short_zone_pct")),
        "long_entry_attempts_on_valid": long_valid,
        "short_entry_attempts_on_valid": short_valid,
        "long_entry_attempts_on_invalid": long_invalid,
        "short_entry_attempts_on_invalid": short_invalid,
        "total_valid_attempts": total_valid_attempts,
        "total_invalid_attempts": total_invalid_attempts,
        "invalid_attempt_ratio": invalid_attempt_ratio,
        "blocked_step_count": metrics.get("blocked_step_count"),
        "benchmark_score": benchmark_score,
        "folder_modified_time": report_dir.stat().st_mtime,
        "experiment_config_written_at": experiment_config.get("written_at_utc"),
        "verification_generated_at": verification.get("generated_at"),
    }


def build_dataframe(run_dirs: List[Path]) -> pd.DataFrame:
    rows = [extract_run_row(report_dir) for report_dir in run_dirs]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def pick_latest_run(df: pd.DataFrame) -> Optional[pd.Series]:
    if df.empty:
        return None
    temp = df.sort_values(by="folder_modified_time", ascending=False, na_position="last")
    if temp.empty:
        return None
    return temp.iloc[0]


def pick_best_benchmark_run(df: pd.DataFrame) -> Optional[pd.Series]:
    if df.empty:
        return None

    temp = df.copy()

    temp["verdict_rank"] = temp["verdict"].map({"PASS": 3, "WARN": 2, "FAIL": 1}).fillna(0)
    temp["total_pnl_num"] = pd.to_numeric(temp["total_pnl"], errors="coerce").fillna(float("-inf"))
    temp["total_trades_num"] = pd.to_numeric(temp["total_trades"], errors="coerce").fillna(0)
    temp["invalid_attempts_num"] = pd.to_numeric(temp["total_invalid_attempts"], errors="coerce").fillna(float("inf"))
    temp["benchmark_score_num"] = pd.to_numeric(temp["benchmark_score"], errors="coerce").fillna(float("-inf"))

    temp = temp.sort_values(
        by=["verdict_rank", "benchmark_score_num", "total_pnl_num", "total_trades_num", "invalid_attempts_num"],
        ascending=[False, False, False, False, True],
        na_position="last",
    )

    if temp.empty:
        return None
    return temp.iloc[0]


def detect_patterns(df: pd.DataFrame, latest_row: pd.Series) -> List[str]:
    patterns: List[str] = []

    if df.empty:
        return patterns

    diagnosis_counts = Counter([str(v) for v in df["diagnosis"].dropna().tolist()])
    verdict_counts = Counter([str(v) for v in df["verdict"].dropna().tolist()])

    latest_diagnosis = latest_row.get("diagnosis")
    latest_verdict = latest_row.get("verdict")
    latest_total_trades = pd.to_numeric(pd.Series([latest_row.get("total_trades")]), errors="coerce").iloc[0]
    latest_invalid_ratio = latest_row.get("invalid_attempt_ratio")
    latest_invalid_attempts = pd.to_numeric(pd.Series([latest_row.get("total_invalid_attempts")]), errors="coerce").iloc[0]

    if latest_diagnosis and diagnosis_counts.get(str(latest_diagnosis), 0) >= 2:
        patterns.append(
            f"Recent runs repeatedly show diagnosis '{latest_diagnosis}' "
            f"({diagnosis_counts[str(latest_diagnosis)]} of {len(df)} scanned runs)."
        )

    if latest_verdict and verdict_counts.get(str(latest_verdict), 0) >= 2:
        patterns.append(
            f"Recent runs repeatedly end with verdict '{latest_verdict}' "
            f"({verdict_counts[str(latest_verdict)]} of {len(df)} scanned runs)."
        )

    if pd.notna(latest_total_trades) and float(latest_total_trades) == 0:
        patterns.append("Latest run executed zero trades.")

    if latest_invalid_ratio is not None and not pd.isna(latest_invalid_ratio) and float(latest_invalid_ratio) >= 0.5:
        patterns.append(
            f"Latest run has a high invalid attempt ratio ({format_pct(latest_invalid_ratio)})."
        )

    if pd.notna(latest_invalid_attempts) and float(latest_invalid_attempts) > 1000:
        patterns.append(
            f"Latest run made a large number of invalid attempts ({int(latest_invalid_attempts)})."
        )

    recent_with_config = df[df["experiment_config_written_at"].notna()] if "experiment_config_written_at" in df.columns else pd.DataFrame()
    if not recent_with_config.empty:
        same_reward = recent_with_config[
            recent_with_config["reward_version"].astype(str) == str(latest_row.get("reward_version"))
        ]
        if len(same_reward) >= 2 and latest_diagnosis:
            same_reward_same_diag = same_reward[
                same_reward["diagnosis"].astype(str) == str(latest_diagnosis)
            ]
            if len(same_reward_same_diag) >= 2:
                patterns.append(
                    f"Runs with reward_version={format_label(latest_row.get('reward_version'))} "
                    f"are repeatedly ending in diagnosis '{format_label(latest_diagnosis)}'."
                )

    return patterns


def build_recommendation(
    latest_row: Optional[pd.Series],
    benchmark_row: Optional[pd.Series],
    patterns: List[str],
) -> Tuple[List[str], List[str], Dict[str, str]]:
    recommendation_lines: List[str] = []
    verify_lines: List[str] = []
    suggested_metadata: Dict[str, str] = {}

    if latest_row is None:
        recommendation_lines.append("No latest run available to analyze.")
        verify_lines.append("- [ ] Generate at least one evaluation run.")
        return recommendation_lines, verify_lines, suggested_metadata

    latest_verdict = format_label(latest_row.get("verdict"))
    latest_diagnosis = format_label(latest_row.get("diagnosis"))
    latest_trades = pd.to_numeric(pd.Series([latest_row.get("total_trades")]), errors="coerce").iloc[0]
    latest_invalid_ratio = latest_row.get("invalid_attempt_ratio")
    latest_total_pnl = latest_row.get("total_pnl")

    if benchmark_row is not None:
        recommendation_lines.append(
            f"Keep {benchmark_row.get('run_id')} as the current benchmark until a newer run clearly beats it."
        )

    if latest_diagnosis == "invalid_action_heavy":
        recommendation_lines.append(
            "Prioritize reducing invalid entry attempts before introducing more reward complexity."
        )
        recommendation_lines.append(
            "Inspect action masking, entry gating, and observation-to-action alignment for long/short entry decisions."
        )
        recommendation_lines.append(
            "Avoid stacking multiple experimental changes at once; change one control at a time."
        )

        suggested_metadata["reward_version"] = (
            f"{format_label(latest_row.get('reward_version'))}_gating_fix"
            if latest_row.get("reward_version") is not None and not pd.isna(latest_row.get("reward_version"))
            else "gating_fix_v1"
        )
        suggested_metadata["feature_version"] = format_label(latest_row.get("feature_version"))
        suggested_metadata["environment_version"] = format_label(latest_row.get("environment_version"))
        suggested_metadata["notes"] = "Phase 5 recommendation: reduce invalid entries before further reward tuning."

        verify_lines.extend(
            [
                "- [ ] Confirm invalid attempt ratio drops materially from the latest run.",
                "- [ ] Confirm at least some entry attempts occur on valid bars.",
                "- [ ] Confirm the next run executes non-zero trades.",
            ]
        )

    elif pd.notna(latest_trades) and float(latest_trades) == 0:
        recommendation_lines.append(
            "The latest run is not behaviorally active, so focus on restoring trade participation first."
        )
        recommendation_lines.append(
            "Review gating strictness, blocked-action reasons, and whether valid setups are reachable by the policy."
        )

        suggested_metadata["reward_version"] = format_label(latest_row.get("reward_version"))
        suggested_metadata["feature_version"] = format_label(latest_row.get("feature_version"))
        suggested_metadata["environment_version"] = format_label(latest_row.get("environment_version"))
        suggested_metadata["notes"] = "Phase 5 recommendation: restore non-zero trade activity before tuning profitability."

        verify_lines.extend(
            [
                "- [ ] Confirm the next run produces at least one completed trade.",
                "- [ ] Confirm blocked reasons shift away from entry rejection patterns.",
                "- [ ] Confirm valid entry attempts are non-zero.",
            ]
        )

    elif latest_verdict == "PASS" and latest_total_pnl is not None and not pd.isna(latest_total_pnl):
        recommendation_lines.append(
            "The latest run is a viable candidate configuration; next step should be robustness validation."
        )
        recommendation_lines.append(
            "Run a seed sweep or repeat evaluation to confirm the behavior is stable and not a one-off result."
        )

        suggested_metadata["reward_version"] = format_label(latest_row.get("reward_version"))
        suggested_metadata["feature_version"] = format_label(latest_row.get("feature_version"))
        suggested_metadata["environment_version"] = format_label(latest_row.get("environment_version"))
        suggested_metadata["notes"] = "Phase 5 recommendation: validate robustness across seeds."

        verify_lines.extend(
            [
                "- [ ] Confirm performance remains positive across multiple seeds.",
                "- [ ] Confirm verdict stays PASS across repeated runs.",
                "- [ ] Confirm total trades and invalid attempt ratio remain stable.",
            ]
        )

    else:
        recommendation_lines.append(
            "Use the benchmark run as the reference point and make one targeted change in the next experiment."
        )
        recommendation_lines.append(
            "Prefer a small, testable change over a broader config rewrite."
        )

        suggested_metadata["reward_version"] = format_label(latest_row.get("reward_version"))
        suggested_metadata["feature_version"] = format_label(latest_row.get("feature_version"))
        suggested_metadata["environment_version"] = format_label(latest_row.get("environment_version"))
        suggested_metadata["notes"] = "Phase 5 recommendation: make one targeted change and compare against benchmark."

        verify_lines.extend(
            [
                "- [ ] Confirm whether verdict improves versus the latest run.",
                "- [ ] Confirm total_pnl and total_trades do not regress.",
                "- [ ] Confirm invalid attempt ratio does not worsen.",
            ]
        )

    if latest_invalid_ratio is not None and not pd.isna(latest_invalid_ratio) and float(latest_invalid_ratio) >= 0.9:
        recommendation_lines.append(
            "Because invalid attempt ratio is extremely high, investigate policy-action feasibility before reward retuning."
        )

    if patterns:
        recommendation_lines.append(
            "Repeated recent patterns suggest the current failure mode is systematic rather than random."
        )

    return recommendation_lines, verify_lines, suggested_metadata


def render_markdown(
    df: pd.DataFrame,
    latest_row: Optional[pd.Series],
    benchmark_row: Optional[pd.Series],
    patterns: List[str],
    recommendation_lines: List[str],
    verify_lines: List[str],
    suggested_metadata: Dict[str, str],
) -> str:
    lines: List[str] = []

    lines.append("# Next Experiment Recommendation")
    lines.append("")
    lines.append(f"Generated at: {datetime.now().isoformat()}")
    lines.append(f"Scanned runs: {len(df)}")
    lines.append("")

    lines.append("## Best benchmark run")
    if benchmark_row is None:
        lines.append("- No benchmark run found.")
    else:
        lines.extend(
            [
                f"- run_id: {format_label(benchmark_row.get('run_id'))}",
                f"- verdict: {format_label(benchmark_row.get('verdict'))}",
                f"- diagnosis: {format_label(benchmark_row.get('diagnosis'))}",
                f"- total_pnl: {format_num(benchmark_row.get('total_pnl'))}",
                f"- total_trades: {format_label(benchmark_row.get('total_trades'))}",
                f"- invalid_attempt_ratio: {format_pct(benchmark_row.get('invalid_attempt_ratio'))}",
                f"- reward_version: {format_label(benchmark_row.get('reward_version'))}",
                f"- feature_version: {format_label(benchmark_row.get('feature_version'))}",
                f"- environment_version: {format_label(benchmark_row.get('environment_version'))}",
            ]
        )
    lines.append("")

    lines.append("## Latest run")
    if latest_row is None:
        lines.append("- No latest run found.")
    else:
        lines.extend(
            [
                f"- run_id: {format_label(latest_row.get('run_id'))}",
                f"- verdict: {format_label(latest_row.get('verdict'))}",
                f"- diagnosis: {format_label(latest_row.get('diagnosis'))}",
                f"- total_pnl: {format_num(latest_row.get('total_pnl'))}",
                f"- total_trades: {format_label(latest_row.get('total_trades'))}",
                f"- invalid_attempt_ratio: {format_pct(latest_row.get('invalid_attempt_ratio'))}",
                f"- total_invalid_attempts: {format_label(latest_row.get('total_invalid_attempts'))}",
                f"- reward_version: {format_label(latest_row.get('reward_version'))}",
                f"- feature_version: {format_label(latest_row.get('feature_version'))}",
                f"- environment_version: {format_label(latest_row.get('environment_version'))}",
                f"- reason: {format_label(latest_row.get('reason'))}",
            ]
        )
    lines.append("")

    lines.append("## Observed patterns")
    if patterns:
        for item in patterns:
            lines.append(f"- {item}")
    else:
        lines.append("- No strong repeated pattern detected from the scanned runs.")
    lines.append("")

    lines.append("## Recommendation")
    for item in recommendation_lines:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## Suggested metadata for next run")
    if suggested_metadata:
        for key, value in suggested_metadata.items():
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- No suggested metadata available.")
    lines.append("")

    lines.append("## What to verify next run")
    for item in verify_lines:
        lines.append(item)
    lines.append("")

    return "\n".join(lines)


def print_console_summary(
    df: pd.DataFrame,
    latest_row: Optional[pd.Series],
    benchmark_row: Optional[pd.Series],
    patterns: List[str],
    recommendation_lines: List[str],
    verify_lines: List[str],
    suggested_metadata: Dict[str, str],
):
    print("\n=== NEXT EXPERIMENT RECOMMENDATION ===")
    print(f"Scanned runs: {len(df)}")

    print("\nBest benchmark run:")
    if benchmark_row is None:
        print("- No benchmark run found.")
    else:
        print(f"- run_id: {format_label(benchmark_row.get('run_id'))}")
        print(f"- verdict: {format_label(benchmark_row.get('verdict'))}")
        print(f"- diagnosis: {format_label(benchmark_row.get('diagnosis'))}")
        print(f"- total_pnl: {format_num(benchmark_row.get('total_pnl'))}")
        print(f"- total_trades: {format_label(benchmark_row.get('total_trades'))}")
        print(f"- invalid_attempt_ratio: {format_pct(benchmark_row.get('invalid_attempt_ratio'))}")

    print("\nLatest run:")
    if latest_row is None:
        print("- No latest run found.")
    else:
        print(f"- run_id: {format_label(latest_row.get('run_id'))}")
        print(f"- verdict: {format_label(latest_row.get('verdict'))}")
        print(f"- diagnosis: {format_label(latest_row.get('diagnosis'))}")
        print(f"- total_pnl: {format_num(latest_row.get('total_pnl'))}")
        print(f"- total_trades: {format_label(latest_row.get('total_trades'))}")
        print(f"- invalid_attempt_ratio: {format_pct(latest_row.get('invalid_attempt_ratio'))}")
        print(f"- total_invalid_attempts: {format_label(latest_row.get('total_invalid_attempts'))}")

    print("\nObserved pattern:")
    if patterns:
        for item in patterns:
            print(f"- {item}")
    else:
        print("- No strong repeated pattern detected.")

    print("\nRecommendation:")
    for item in recommendation_lines:
        print(f"- {item}")

    print("\nSuggested metadata for next run:")
    if suggested_metadata:
        for key, value in suggested_metadata.items():
            print(f"- {key}: {value}")
    else:
        print("- No suggested metadata available.")

    print("\nWhat to verify next run:")
    for item in verify_lines:
        print(item)


def parse_args():
    parser = argparse.ArgumentParser(description="Recommend the next experiment based on recent evaluation runs.")
    parser.add_argument("--reports-dir", default=DEFAULT_REPORTS_DIR, help=f"Reports directory to scan. Default: {DEFAULT_REPORTS_DIR}")
    parser.add_argument("--latest", type=int, default=10, help="Number of most recent runs to scan.")
    parser.add_argument("--save", action="store_true", help="Save recommendation markdown to the recommendations directory.")
    parser.add_argument("--out", default=None, help="Explicit markdown output path.")
    return parser.parse_args()


def main():
    args = parse_args()

    run_dirs = collect_run_dirs(args.reports_dir, args.latest)
    if not run_dirs:
        print("No run directories found.")
        return

    df = build_dataframe(run_dirs)
    if df.empty:
        print("No usable run data found.")
        return

    latest_row = pick_latest_run(df)
    benchmark_row = pick_best_benchmark_run(df)
    patterns = detect_patterns(df, latest_row) if latest_row is not None else []
    recommendation_lines, verify_lines, suggested_metadata = build_recommendation(
        latest_row=latest_row,
        benchmark_row=benchmark_row,
        patterns=patterns,
    )

    print_console_summary(
        df=df,
        latest_row=latest_row,
        benchmark_row=benchmark_row,
        patterns=patterns,
        recommendation_lines=recommendation_lines,
        verify_lines=verify_lines,
        suggested_metadata=suggested_metadata,
    )

    markdown = render_markdown(
        df=df,
        latest_row=latest_row,
        benchmark_row=benchmark_row,
        patterns=patterns,
        recommendation_lines=recommendation_lines,
        verify_lines=verify_lines,
        suggested_metadata=suggested_metadata,
    )

    output_path: Optional[Path] = None
    if args.out:
        output_path = Path(args.out)
    elif args.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(DEFAULT_RECOMMENDATIONS_DIR) / f"recommendation_{timestamp}.md"

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
        print(f"\nSaved recommendation: {output_path}")


if __name__ == "__main__":
    main()