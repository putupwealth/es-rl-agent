"""
Commands:
    python evaluate.py
    python evaluate.py --model-file models/<model_name>.zip
    python evaluate.py --run-id custom_run_name
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO

from src.levels import add_pdh_pdl, add_breakout_features
from src.features import add_htf_bias
from src.env import ESBreakoutEnv


DATA_FILE = "data/ES_1min_all_sessions.csv"
DEFAULT_LATEST_MODEL_POINTER = "models/latest_model.txt"
DEFAULT_REPORTS_DIR = "reports"
DEFAULT_EVAL_SEED = 42
DEFAULT_LATEST_RUN_POINTER = "reports/latest_run.txt"

# Required columns that must be present in steps.csv for later verification.
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

# Full ordered column set for trades.csv.
TRADES_COLUMNS = [
    "entry_time",
    "direction",
    "entry_price",
    "entry_action",
    "PDH",
    "PDL",
    "first_break_above_PDH",
    "first_break_below_PDL",
    "break_above_PDH",
    "break_below_PDL",
    "near_PDH",
    "near_PDL",
    "bias_long",
    "bias_short",
    "trend_1h_up",
    "trend_4h_up",
    "is_rth",
    "is_eth",
    "setup_type",
    "bias_alignment",
    "valid_long_zone",
    "valid_short_zone",
    "allowed_long_entry",
    "allowed_short_entry",
    "entry_rule_trigger",
    "blocked_reason",
    "exit_time",
    "exit_price",
    "pnl",
    "exit_action",
    "exit_reason",
]

# Fields that must be present inside the "eligibility_diagnostics" section of
# eval_summary.json for the deterministic verifier and LLM review pipeline.
ELIGIBILITY_DIAGNOSTICS_FIELDS = [
    "total_steps",
    "valid_long_zone_steps",
    "valid_short_zone_steps",
    "valid_long_zone_pct",
    "valid_short_zone_pct",
    "long_entry_attempts_on_valid",
    "short_entry_attempts_on_valid",
    "long_entry_attempts_on_invalid",
    "short_entry_attempts_on_invalid",
]


def classify_setup(row):
    if row["first_break_above_PDH"] == 1:
        return "first_breakout_above_PDH"
    if row["break_above_PDH"] == 1:
        return "breakout_above_PDH"
    if row["first_break_below_PDL"] == 1:
        return "first_breakdown_below_PDL"
    if row["break_below_PDL"] == 1:
        return "breakdown_below_PDL"
    if row["near_PDH"] == 1 and row["near_PDL"] == 1:
        return "near_PDH_PDL_no_breakout"
    if row["near_PDH"] == 1:
        return "near_PDH_no_breakout"
    if row["near_PDL"] == 1:
        return "near_PDL_no_breakout"
    return "no_level_context"


def classify_bias_alignment(position, row):
    if (position == 1 and row["bias_long"] == 1) or (position == -1 and row["bias_short"] == 1):
        return "aligned"
    if (position == 1 and row["bias_short"] == 1) or (position == -1 and row["bias_long"] == 1):
        return "counter"
    return "neutral"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO model for ES PDH/PDL strategy.")
    parser.add_argument("--data-file", default=DATA_FILE, help="Path to input CSV data.")
    parser.add_argument("--model-file", default=None, help="Path to model file or base model path for PPO.load.")
    parser.add_argument(
        "--latest-model-pointer",
        default=DEFAULT_LATEST_MODEL_POINTER,
        help="Path to latest model pointer file used when --model-file is omitted.",
    )
    parser.add_argument("--reports-dir", default=DEFAULT_REPORTS_DIR, help="Base directory for evaluation reports.")
    parser.add_argument("--run-id", default=None, help="Optional explicit run ID for report output folder.")
    parser.add_argument("--seed", type=int, default=DEFAULT_EVAL_SEED, help="Evaluation seed.")
    return parser.parse_args()


def resolve_model_file(model_file_arg: Optional[str], latest_model_pointer: str) -> Path:
    if model_file_arg:
        return Path(model_file_arg)

    pointer = Path(latest_model_pointer)
    if not pointer.exists():
        raise FileNotFoundError(
            f"No --model-file provided and latest pointer not found: {pointer}. "
            "Train first or pass --model-file explicitly."
        )

    model_path = pointer.read_text(encoding="utf-8").strip()
    if not model_path:
        raise ValueError(f"Latest model pointer is empty: {pointer}")

    return Path(model_path)


def resolve_run_id(model_file: Path, reports_dir: Path, explicit_run_id: Optional[str]) -> str:
    if explicit_run_id:
        return explicit_run_id

    base_run_id = model_file.stem
    run_id = base_run_id
    report_dir = reports_dir / run_id

    if report_dir.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{base_run_id}_eval_{timestamp}"

    return run_id


def write_latest_run_pointer(report_dir: Path, latest_run_pointer: Path):
    """Write a lightweight pointer to the most recent evaluation report directory."""
    latest_run_pointer.parent.mkdir(parents=True, exist_ok=True)
    latest_run_pointer.write_text(str(report_dir), encoding="utf-8")
    print(f"Updated latest run pointer: {latest_run_pointer}")
    print(f"Latest run points to: {report_dir}")


def build_steps_dataframe(step_logs: list) -> pd.DataFrame:
    """Return a DataFrame for step_logs with all required columns guaranteed.

    When step_logs is empty (e.g. an episode that terminates immediately) a
    zero-row DataFrame with the full column set is returned so that steps.csv
    always has a proper header for downstream consumers.
    """
    if step_logs:
        df = pd.DataFrame(step_logs)
    else:
        df = pd.DataFrame(columns=STEPS_REQUIRED_COLUMNS)

    for col in STEPS_REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    return df


def build_trades_dataframe(trades: list) -> pd.DataFrame:
    """Return a DataFrame for the trades list with a stable column set.

    When trades is empty a zero-row DataFrame with the full TRADES_COLUMNS
    header is returned so that trades.csv always has meaningful column names.
    """
    if trades:
        return pd.DataFrame(trades)
    return pd.DataFrame(columns=TRADES_COLUMNS)


def build_trade_breakdown(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Return a grouped trade-breakdown DataFrame.

    When trades_df is empty the returned DataFrame still has the expected
    column headers so that trade_breakdown.csv is never completely bare.
    """
    if trades_df.empty:
        return pd.DataFrame(
            columns=["direction", "setup_type", "bias_alignment", "trades", "avg_pnl", "total_pnl", "win_rate"]
        )

    return (
        trades_df.groupby(["direction", "setup_type", "bias_alignment"], dropna=False)
        .agg(
            trades=("pnl", "count"),
            avg_pnl=("pnl", "mean"),
            total_pnl=("pnl", "sum"),
            win_rate=("pnl", lambda x: float((x > 0).mean())),
        )
        .reset_index()
    )


def main():
    args = parse_args()

    model_file = resolve_model_file(args.model_file, args.latest_model_pointer)
    reports_dir = Path(args.reports_dir)
    latest_run_pointer = Path(DEFAULT_LATEST_RUN_POINTER)
    run_id = resolve_run_id(model_file=model_file, reports_dir=reports_dir, explicit_run_id=args.run_id)
    report_dir = reports_dir / run_id

    report_file = report_dir / "equity_curve.png"
    trades_file = report_dir / "trades.csv"
    step_log_file = report_dir / "steps.csv"
    breakdown_file = report_dir / "trade_breakdown.csv"
    summary_file = report_dir / "eval_summary.json"

    print(f"Model file: {model_file}")
    print(f"Run ID: {run_id}")
    print(f"Report directory: {report_dir}")

    print("Loading data...")
    df = pd.read_csv(args.data_file)

    print("Building features...")
    df = add_pdh_pdl(df)
    df = add_breakout_features(df)
    df = add_htf_bias(df)

    df = df[df["is_roll_period"] == 0].reset_index(drop=True)

    split = int(len(df) * 0.8)
    test_df = df.iloc[split:].reset_index(drop=True)

    print(f"Test rows: {len(test_df):,}")

    env = ESBreakoutEnv(
        df=test_df,
        max_steps=len(test_df) - 2,
        point_value=50,
        commission=5.0,
        max_trades=10,
    )

    model = PPO.load(str(model_file))

    obs, _ = env.reset(seed=args.seed, options={"start_idx": 0})

    realized_equity_curve = []
    unrealized_pnl_curve = []
    total_equity_curve = []
    actions = []
    trades = []
    step_logs = []
    cumulative_reward = 0.0

    current_trade = None
    done = False

    while not done:
        row = env.df.iloc[env.current_idx].copy()
        prev_position = env.position
        prev_entry_price = env.entry_price

        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward

        new_position = info["position"]

        # Entry detected
        if prev_position == 0 and new_position != 0:
            current_trade = {
                "entry_time": row["timestamp"],
                "direction": "LONG" if new_position == 1 else "SHORT",
                "entry_price": row["close"],
                "entry_action": action,
                "PDH": row["PDH"],
                "PDL": row["PDL"],
                "first_break_above_PDH": row["first_break_above_PDH"],
                "first_break_below_PDL": row["first_break_below_PDL"],
                "break_above_PDH": row["break_above_PDH"],
                "break_below_PDL": row["break_below_PDL"],
                "near_PDH": row["near_PDH"],
                "near_PDL": row["near_PDL"],
                "bias_long": row["bias_long"],
                "bias_short": row["bias_short"],
                "trend_1h_up": row["trend_1h_up"],
                "trend_4h_up": row["trend_4h_up"],
                "is_rth": row["is_rth"],
                "is_eth": row["is_eth"],
                "setup_type": classify_setup(row),
                "bias_alignment": classify_bias_alignment(new_position, row),
                "valid_long_zone": info.get("valid_long_zone"),
                "valid_short_zone": info.get("valid_short_zone"),
                "allowed_long_entry": info.get("allowed_long_entry"),
                "allowed_short_entry": info.get("allowed_short_entry"),
                "entry_rule_trigger": info.get("entry_rule_trigger"),
                "blocked_reason": info.get("blocked_reason"),
            }

        # Exit detected
        if prev_position != 0 and new_position == 0 and current_trade is not None:
            exit_price = row["close"]
            pnl = (exit_price - prev_entry_price) * prev_position * 50 - 5.0

            current_trade.update(
                {
                    "exit_time": row["timestamp"],
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "exit_action": action,
                    "exit_reason": info.get("exit_reason"),
                }
            )

            trades.append(current_trade)
            current_trade = None

        actions.append(action)
        realized_equity_curve.append(info["realized_equity"])
        unrealized_pnl_curve.append(info["unrealized_pnl"])
        total_equity_curve.append(info["total_equity"])
        step_logs.append(
            {
                "timestamp": row["timestamp"],
                "action": action,
                "reward": reward,
                "cumulative_reward": cumulative_reward,
                "position": info["position"],
                "trade_count": info["trade_count"],
                "realized_equity": info["realized_equity"],
                "unrealized_pnl": info["unrealized_pnl"],
                "total_equity": info["total_equity"],
                "exit_reason": info["exit_reason"],
                "valid_long_zone": info.get("valid_long_zone"),
                "valid_short_zone": info.get("valid_short_zone"),
                "allowed_long_entry": info.get("allowed_long_entry"),
                "allowed_short_entry": info.get("allowed_short_entry"),
                "entry_rule_trigger": info.get("entry_rule_trigger"),
                "blocked_reason": info.get("blocked_reason"),
                "attempted_entry_action": info.get("attempted_entry_action"),
            }
        )

        done = terminated or truncated

    print("\n========== EVALUATION ==========")
    print(f"Final realized equity: ${realized_equity_curve[-1]:.2f}")
    print(f"Final total equity: ${total_equity_curve[-1]:.2f}")
    print(f"Cumulative reward: {cumulative_reward:.2f}")
    action_counts = {a: actions.count(a) for a in set(actions)}
    print("Action counts:", action_counts)
    print(f"Logged trades: {len(trades)}")

    # Eligibility diagnostics
    total_steps = len(step_logs)
    valid_long_steps = sum(1 for s in step_logs if s.get("valid_long_zone") == 1)
    valid_short_steps = sum(1 for s in step_logs if s.get("valid_short_zone") == 1)
    entry_attempts_on_valid_long = sum(
        1 for s in step_logs
        if s.get("valid_long_zone") == 1 and s.get("attempted_entry_action") == 1
    )
    entry_attempts_on_valid_short = sum(
        1 for s in step_logs
        if s.get("valid_short_zone") == 1 and s.get("attempted_entry_action") == 2
    )
    invalid_long_attempts = sum(
        1 for s in step_logs
        if s.get("valid_long_zone") == 0 and s.get("attempted_entry_action") == 1
    )
    invalid_short_attempts = sum(
        1 for s in step_logs
        if s.get("valid_short_zone") == 0 and s.get("attempted_entry_action") == 2
    )
    print("\n========== ELIGIBILITY DIAGNOSTICS ==========")
    print(f"Total steps: {total_steps}")
    print(f"Valid long zone steps: {valid_long_steps} ({valid_long_steps / max(1, total_steps):.1%})")
    print(f"Valid short zone steps: {valid_short_steps} ({valid_short_steps / max(1, total_steps):.1%})")
    print(f"Long entry attempts on valid bars: {entry_attempts_on_valid_long}")
    print(f"Short entry attempts on valid bars: {entry_attempts_on_valid_short}")
    print(f"Long entry attempts on INVALID bars: {invalid_long_attempts}")
    print(f"Short entry attempts on INVALID bars: {invalid_short_attempts}")

    report_dir.mkdir(parents=True, exist_ok=True)
    build_steps_dataframe(step_logs).to_csv(step_log_file, index=False)
    print(f"Saved step log: {step_log_file}")

    trades_df = build_trades_dataframe(trades)
    trades_df.to_csv(trades_file, index=False)
    print(f"Saved trades: {trades_file}")

    if trades_df.empty:
        print("No completed trades logged.")
    else:
        print("\n========== TRADES ==========")
        print(trades_df)

    breakdown_df = build_trade_breakdown(trades_df)
    if not breakdown_df.empty:
        print("\n========== TRADE BREAKDOWN ==========")
        print(breakdown_df)

    breakdown_df.to_csv(breakdown_file, index=False)
    print(f"Saved breakdown: {breakdown_file}")

    summary = {
        "experiment": {
            "run_id": run_id,
            "model_path": str(model_file),
            "model_name": model_file.name,
            "evaluation_seed": args.seed,
            "test_rows": int(len(test_df)),
        },
        "performance": {
            "final_realized_equity": float(realized_equity_curve[-1]) if realized_equity_curve else 0.0,
            "final_total_equity": float(total_equity_curve[-1]) if total_equity_curve else 0.0,
            "cumulative_reward": float(cumulative_reward),
        },
        "trades": {
            "total_trades": int(len(trades)),
            "long_trades": int(sum(1 for t in trades if t["direction"] == "LONG")),
            "short_trades": int(sum(1 for t in trades if t["direction"] == "SHORT")),
        },
        "actions": {str(k): int(v) for k, v in action_counts.items()},
        "eligibility_diagnostics": {
            "total_steps": total_steps,
            "valid_long_zone_steps": valid_long_steps,
            "valid_short_zone_steps": valid_short_steps,
            "valid_long_zone_pct": round(valid_long_steps / max(1, total_steps), 4),
            "valid_short_zone_pct": round(valid_short_steps / max(1, total_steps), 4),
            "long_entry_attempts_on_valid": entry_attempts_on_valid_long,
            "short_entry_attempts_on_valid": entry_attempts_on_valid_short,
            "long_entry_attempts_on_invalid": invalid_long_attempts,
            "short_entry_attempts_on_invalid": invalid_short_attempts,
        },
    }

    if trades_df.empty:
        summary["trades"].update(
            {
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "setup_breakdown": {},
                "bias_alignment_breakdown": {},
                "direction_breakdown": {},
            }
        )
    else:
        summary["trades"].update(
            {
                "win_rate": float((trades_df["pnl"] > 0).mean()),
                "total_pnl": float(trades_df["pnl"].sum()),
                "avg_pnl": float(trades_df["pnl"].mean()),
                "setup_breakdown": trades_df["setup_type"].value_counts(dropna=False).to_dict(),
                "bias_alignment_breakdown": trades_df["bias_alignment"].value_counts(dropna=False).to_dict(),
                "direction_breakdown": trades_df["direction"].value_counts(dropna=False).to_dict(),
            }
        )

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_file}")

    plt.figure(figsize=(12, 5))
    plt.plot(realized_equity_curve, label="Realized equity")
    plt.plot(unrealized_pnl_curve, label="Unrealized PnL")
    plt.plot(total_equity_curve, label="Total equity (MTM)")
    plt.title(f"ES PDH/PDL - Equity Curves ({run_id})")
    plt.xlabel("Step")
    plt.ylabel("Equity ($)")
    plt.grid(True)
    plt.legend()
    plt.savefig(report_file)
    print(f"Saved report: {report_file}")
    plt.show()

    write_latest_run_pointer(report_dir=report_dir, latest_run_pointer=latest_run_pointer)


if __name__ == "__main__":
    main()