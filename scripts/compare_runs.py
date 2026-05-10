"""Compare multiple evaluation runs side by side.

Supported input modes
---------------------
1. Explicit report directories:
       python scripts/compare_runs.py reports/run_a reports/run_b

2. Pointer files such as reports/latest_run.txt:
       python scripts/compare_runs.py reports/latest_run.txt

3. Scan all run directories under a reports directory:
       python scripts/compare_runs.py --reports-dir reports

4. Scan latest N run directories under a reports directory:
       python scripts/compare_runs.py --reports-dir reports --latest 5

5. Mixed explicit inputs plus reports-dir scan:
       python scripts/compare_runs.py reports/run_a reports/latest_run.txt --reports-dir reports --latest 10

Optional output:
       python scripts/compare_runs.py --reports-dir reports --csv-out reports/run_comparison.csv
       python scripts/compare_runs.py --reports-dir reports --markdown
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


DEFAULT_REPORTS_DIR = "reports"


class Ansi:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    CYAN = "\033[36m"


def color_text(text: str, color: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{color}{text}{Ansi.RESET}"


def format_pct(value) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return str(value)


def format_num(value, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return str(value)


def resolve_report_dir(path_str: str) -> Path:
    """Resolve either a report directory path or a .txt pointer file."""
    path = Path(path_str)

    if path.is_file() and path.suffix.lower() == ".txt":
        resolved = path.read_text(encoding="utf-8").strip()
        if not resolved:
            raise ValueError(f"Pointer file is empty: {path}")
        return Path(resolved)

    return path


def safe_load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def is_run_directory(path: Path) -> bool:
    """Heuristic for whether a directory looks like an evaluation run folder."""
    if not path.is_dir():
        return False

    if path.name.lower() in {"latest"}:
        return False

    return (
        (path / "eval_summary.json").exists()
        or (path / "verification.json").exists()
        or (path / "llm_input_packet.json").exists()
        or (path / "steps.csv").exists()
        or (path / "trades.csv").exists()
    )


def collect_run_dirs(explicit_paths: List[str], reports_dir: Optional[str], latest: Optional[int]) -> List[Path]:
    run_dirs: List[Path] = []
    seen = set()

    for raw in explicit_paths:
        resolved = resolve_report_dir(raw)
        if not resolved.exists():
            print(f"WARNING: Skipping missing path: {resolved}", file=sys.stderr)
            continue
        if not resolved.is_dir():
            print(f"WARNING: Skipping non-directory path: {resolved}", file=sys.stderr)
            continue

        resolved_key = str(resolved.resolve())
        if resolved_key not in seen:
            run_dirs.append(resolved)
            seen.add(resolved_key)

    if reports_dir:
        base = Path(reports_dir)
        if not base.exists():
            print(f"WARNING: reports dir does not exist: {base}", file=sys.stderr)
        elif not base.is_dir():
            print(f"WARNING: reports dir is not a directory: {base}", file=sys.stderr)
        else:
            scanned = [p for p in base.iterdir() if is_run_directory(p)]
            scanned.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            if latest is not None:
                scanned = scanned[:latest]

            for resolved in scanned:
                resolved_key = str(resolved.resolve())
                if resolved_key not in seen:
                    run_dirs.append(resolved)
                    seen.add(resolved_key)

    return run_dirs


def extract_run_row(report_dir: Path) -> Dict:
    summary = safe_load_json(report_dir / "eval_summary.json") or {}
    verification = safe_load_json(report_dir / "verification.json") or {}
    packet = safe_load_json(report_dir / "llm_input_packet.json") or {}

    experiment = summary.get("experiment", {})
    performance = summary.get("performance", {})
    trades = summary.get("trades", {})
    eligibility = summary.get("eligibility_diagnostics", {})

    metrics = verification.get("metrics", {})
    packet_trade_summary = packet.get("trade_summary", {})

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

    verdict_score = {
        "PASS": 2,
        "WARN": 1,
        "FAIL": 0,
    }.get(verdict, -1)

    total_pnl = trades.get("total_pnl", packet_trade_summary.get("total_pnl"))
    total_trades = trades.get("total_trades", packet_trade_summary.get("total_trades"))
    win_rate = trades.get("win_rate")

    composite_score = None
    try:
        pnl_component = float(total_pnl) if total_pnl is not None else 0.0
        valid_component = float(total_valid_attempts) * 0.5
        invalid_penalty = float(total_invalid_attempts) * 0.1
        verdict_component = float(verdict_score) * 1000.0
        composite_score = verdict_component + pnl_component + valid_component - invalid_penalty
    except Exception:
        composite_score = None

    return {
        "report_dir": str(report_dir),
        "run_id": experiment.get("run_id", report_dir.name),
        "model_name": experiment.get("model_name"),
        "model_path": experiment.get("model_path"),
        "evaluation_seed": experiment.get("evaluation_seed"),
        "test_rows": experiment.get("test_rows"),

        "verdict": verdict,
        "diagnosis": diagnosis,
        "reason": verification.get("reason"),

        "final_realized_equity": performance.get("final_realized_equity"),
        "final_total_equity": performance.get("final_total_equity"),
        "cumulative_reward": performance.get("cumulative_reward"),

        "total_trades": total_trades,
        "long_trades": trades.get("long_trades", packet_trade_summary.get("long_trades")),
        "short_trades": trades.get("short_trades", packet_trade_summary.get("short_trades")),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_pnl": trades.get("avg_pnl", packet_trade_summary.get("avg_pnl")),

        "total_steps": eligibility.get("total_steps", metrics.get("total_steps")),
        "valid_long_zone_steps": eligibility.get("valid_long_zone_steps", metrics.get("valid_long_zone_steps")),
        "valid_short_zone_steps": eligibility.get("valid_short_zone_steps", metrics.get("valid_short_zone_steps")),
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

        "verdict_score": verdict_score,
        "composite_score": composite_score,

        "verification_generated_at": verification.get("generated_at"),
        "packet_generated_at": packet.get("generated_at"),
        "folder_modified_time": report_dir.stat().st_mtime,
    }


def build_dataframe(run_dirs: List[Path]) -> pd.DataFrame:
    rows = [extract_run_row(report_dir) for report_dir in run_dirs]
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    preferred_columns = [
        "run_id",
        "verdict",
        "diagnosis",
        "composite_score",
        "final_realized_equity",
        "total_pnl",
        "avg_pnl",
        "win_rate",
        "total_trades",
        "long_trades",
        "short_trades",
        "total_valid_attempts",
        "total_invalid_attempts",
        "invalid_attempt_ratio",
        "long_entry_attempts_on_valid",
        "short_entry_attempts_on_valid",
        "long_entry_attempts_on_invalid",
        "short_entry_attempts_on_invalid",
        "valid_long_zone_pct",
        "valid_short_zone_pct",
        "blocked_step_count",
        "cumulative_reward",
        "evaluation_seed",
        "model_name",
        "report_dir",
    ]

    existing = [c for c in preferred_columns if c in df.columns]
    remaining = [c for c in df.columns if c not in existing]
    return df[existing + remaining]


def apply_filters(
    df: pd.DataFrame,
    filter_verdict: Optional[List[str]],
    filter_diagnosis: Optional[List[str]],
    min_trades: Optional[int],
    max_invalid_attempts: Optional[int],
    only_pass: bool,
) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    if only_pass:
        out = out[out["verdict"] == "PASS"]

    if filter_verdict:
        allowed = {v.strip().upper() for v in filter_verdict}
        out = out[out["verdict"].astype(str).str.upper().isin(allowed)]

    if filter_diagnosis:
        allowed = {v.strip() for v in filter_diagnosis}
        out = out[out["diagnosis"].astype(str).isin(allowed)]

    if min_trades is not None and "total_trades" in out.columns:
        out = out[pd.to_numeric(out["total_trades"], errors="coerce").fillna(0) >= min_trades]

    if max_invalid_attempts is not None and "total_invalid_attempts" in out.columns:
        out = out[pd.to_numeric(out["total_invalid_attempts"], errors="coerce").fillna(float("inf")) <= max_invalid_attempts]

    return out


def colorize_verdict(verdict: str, enabled: bool) -> str:
    verdict_str = "" if verdict is None else str(verdict)
    if verdict_str == "PASS":
        return color_text(verdict_str, Ansi.GREEN, enabled)
    if verdict_str == "WARN":
        return color_text(verdict_str, Ansi.YELLOW, enabled)
    if verdict_str == "FAIL":
        return color_text(verdict_str, Ansi.RED, enabled)
    return verdict_str


def make_pretty_display_df(df: pd.DataFrame, color: bool) -> pd.DataFrame:
    out = df.copy()

    if "verdict" in out.columns:
        out["verdict"] = out["verdict"].apply(lambda v: colorize_verdict(v, color))

    for col in ["win_rate", "valid_long_zone_pct", "valid_short_zone_pct", "invalid_attempt_ratio"]:
        if col in out.columns:
            out[col] = out[col].apply(format_pct)

    for col in ["composite_score", "final_realized_equity", "total_pnl", "avg_pnl", "cumulative_reward"]:
        if col in out.columns:
            out[col] = out[col].apply(format_num)

    return out


def print_summary_table(df: pd.DataFrame, color: bool):
    if df.empty:
        print("No runs found.")
        return

    display_cols = [
        "run_id",
        "verdict",
        "diagnosis",
        "composite_score",
        "total_pnl",
        "total_trades",
        "win_rate",
        "total_valid_attempts",
        "total_invalid_attempts",
        "invalid_attempt_ratio",
        "final_realized_equity",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    display_df = make_pretty_display_df(df[display_cols].copy(), color=color)

    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 240,
        "display.max_colwidth", 60,
    ):
        print("\n=== RUN COMPARISON ===")
        print(display_df.to_string(index=False))


def print_markdown_table(df: pd.DataFrame):
    if df.empty:
        print("No runs found.")
        return

    display_cols = [
        "run_id",
        "verdict",
        "diagnosis",
        "composite_score",
        "total_pnl",
        "total_trades",
        "win_rate",
        "total_invalid_attempts",
        "invalid_attempt_ratio",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    table_df = df[display_cols].copy()

    for col in ["win_rate", "invalid_attempt_ratio"]:
        if col in table_df.columns:
            table_df[col] = table_df[col].apply(format_pct)

    for col in ["composite_score", "total_pnl"]:
        if col in table_df.columns:
            table_df[col] = table_df[col].apply(format_num)

    def fmt(v):
        if pd.isna(v):
            return ""
        return str(v)

    headers = display_cols
    print("\n| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for _, row in table_df.iterrows():
        print("| " + " | ".join(fmt(row[col]) for col in headers) + " |")


def print_top_n(df: pd.DataFrame, metric: str, n: int, ascending: bool = False):
    if df.empty or metric not in df.columns:
        return

    temp = df.copy()
    temp_metric = pd.to_numeric(temp[metric], errors="coerce")
    temp = temp.loc[temp_metric.notna()].copy()
    temp[metric] = temp_metric[temp_metric.notna()]

    if temp.empty:
        return

    temp = temp.sort_values(by=metric, ascending=ascending).head(n)

    print(f"\n=== TOP {n} by {metric} ({'lowest' if ascending else 'highest'}) ===")
    cols = [c for c in ["run_id", "verdict", "diagnosis", metric] if c in temp.columns]
    pretty = temp[cols].copy()

    if metric in {"win_rate", "invalid_attempt_ratio", "valid_long_zone_pct", "valid_short_zone_pct"}:
        pretty[metric] = pretty[metric].apply(format_pct)
    else:
        pretty[metric] = pretty[metric].apply(format_num)

    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(pretty.to_string(index=False))


def print_best_run_hints(df: pd.DataFrame, color: bool):
    if df.empty:
        return

    print("\n=== QUICK HINTS ===")

    pnl_df = df.dropna(subset=["total_pnl"]) if "total_pnl" in df.columns else pd.DataFrame()
    if not pnl_df.empty:
        best_pnl_row = pnl_df.loc[pnl_df["total_pnl"].astype(float).idxmax()]
        print(f"Best total_pnl: {best_pnl_row['run_id']} ({format_num(best_pnl_row['total_pnl'])})")

    score_df = df.dropna(subset=["composite_score"]) if "composite_score" in df.columns else pd.DataFrame()
    if not score_df.empty:
        best_score_row = score_df.loc[score_df["composite_score"].astype(float).idxmax()]
        print(f"Best composite score: {best_score_row['run_id']} ({format_num(best_score_row['composite_score'])})")

    trades_df = df.dropna(subset=["total_trades"]) if "total_trades" in df.columns else pd.DataFrame()
    if not trades_df.empty:
        most_trades_row = trades_df.loc[trades_df["total_trades"].astype(float).idxmax()]
        print(f"Most trades: {most_trades_row['run_id']} ({most_trades_row['total_trades']})")

    invalid_df = df.dropna(subset=["total_invalid_attempts"]) if "total_invalid_attempts" in df.columns else pd.DataFrame()
    if not invalid_df.empty:
        least_invalid_row = invalid_df.loc[invalid_df["total_invalid_attempts"].astype(float).idxmin()]
        print(f"Least invalid attempts: {least_invalid_row['run_id']} ({least_invalid_row['total_invalid_attempts']})")

    pass_df = df[df["verdict"] == "PASS"] if "verdict" in df.columns else pd.DataFrame()
    warn_df = df[df["verdict"] == "WARN"] if "verdict" in df.columns else pd.DataFrame()
    fail_df = df[df["verdict"] == "FAIL"] if "verdict" in df.columns else pd.DataFrame()

    pass_text = color_text(f"PASS runs: {len(pass_df)}/{len(df)}", Ansi.GREEN, color)
    warn_text = color_text(f"WARN runs: {len(warn_df)}/{len(df)}", Ansi.YELLOW, color)
    fail_text = color_text(f"FAIL runs: {len(fail_df)}/{len(df)}", Ansi.RED, color)

    print(pass_text)
    print(warn_text)
    print(fail_text)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare multiple evaluation run folders side by side.")
    parser.add_argument("paths", nargs="*", help="Optional explicit report directories or .txt pointer files.")
    parser.add_argument(
        "--reports-dir",
        default=None,
        help=f"Scan run directories under this reports folder. Default if nothing else provided: {DEFAULT_REPORTS_DIR}",
    )
    parser.add_argument("--latest", type=int, default=None, help="Only include the latest N run directories.")
    parser.add_argument(
        "--sort-by",
        default="folder_modified_time",
        help="Column to sort by. Examples: total_pnl, composite_score, total_trades, folder_modified_time",
    )
    parser.add_argument("--ascending", action="store_true", help="Sort ascending instead of descending.")
    parser.add_argument("--csv-out", default=None, help="Optional path to write the full comparison table as CSV.")
    parser.add_argument("--full", action="store_true", help="Print the full dataframe.")
    parser.add_argument("--markdown", action="store_true", help="Print a markdown table.")
    parser.add_argument("--filter-verdict", nargs="*", default=None, help="Filter to one or more verdicts.")
    parser.add_argument("--filter-diagnosis", nargs="*", default=None, help="Filter to one or more diagnoses.")
    parser.add_argument("--min-trades", type=int, default=None, help="Only include runs with at least this many trades.")
    parser.add_argument(
        "--max-invalid-attempts",
        type=int,
        default=None,
        help="Only include runs with at most this many invalid attempts.",
    )
    parser.add_argument("--only-pass", action="store_true", help="Only include PASS runs.")
    parser.add_argument("--rank-by", action="store_true", help="Shortcut to sort by composite_score descending.")
    parser.add_argument("--color", action="store_true", help="Enable ANSI colorized output.")
    parser.add_argument("--top-n", type=int, default=3, help="How many top runs to show in top-N summaries.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.paths and not args.reports_dir:
        args.reports_dir = DEFAULT_REPORTS_DIR

    try:
        run_dirs = collect_run_dirs(
            explicit_paths=args.paths,
            reports_dir=args.reports_dir,
            latest=args.latest,
        )
    except Exception as exc:
        print(f"ERROR: Failed while collecting run directories: {exc}", file=sys.stderr)
        sys.exit(1)

    if not run_dirs:
        print("ERROR: No valid run directories found.", file=sys.stderr)
        sys.exit(1)

    df = build_dataframe(run_dirs)

    if df.empty:
        print("ERROR: No comparable run data found.", file=sys.stderr)
        sys.exit(1)

    df = apply_filters(
        df=df,
        filter_verdict=args.filter_verdict,
        filter_diagnosis=args.filter_diagnosis,
        min_trades=args.min_trades,
        max_invalid_attempts=args.max_invalid_attempts,
        only_pass=args.only_pass,
    )

    if df.empty:
        print("No runs remaining after filters.")
        sys.exit(0)

    sort_by = "composite_score" if args.rank_by else args.sort_by
    ascending = args.ascending if not args.rank_by else False

    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending, na_position="last")
    else:
        print(f"WARNING: sort column not found, skipping sort: {sort_by}", file=sys.stderr)

    if args.csv_out:
        csv_path = Path(args.csv_out)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Wrote CSV comparison: {csv_path}")

    if args.full:
        pretty_df = make_pretty_display_df(df.copy(), color=args.color)
        with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None,
            "display.width", 300,
            "display.max_colwidth", 100,
        ):
            print(pretty_df.to_string(index=False))
    elif args.markdown:
        print_markdown_table(df)
    else:
        print_summary_table(df, color=args.color)

    print_best_run_hints(df, color=args.color)
    print_top_n(df, metric="total_pnl", n=args.top_n, ascending=False)
    print_top_n(df, metric="composite_score", n=args.top_n, ascending=False)
    print_top_n(df, metric="total_invalid_attempts", n=args.top_n, ascending=True)
    print_top_n(df, metric="win_rate", n=args.top_n, ascending=False)


if __name__ == "__main__":
    main()