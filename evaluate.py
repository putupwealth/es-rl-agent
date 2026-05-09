import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from pathlib import Path

from src.levels import add_pdh_pdl, add_breakout_features
from src.features import add_htf_bias
from src.env import ESBreakoutEnv


DATA_FILE = "data/ES_1min_all_sessions.csv"
MODEL_FILE = "models/es_pdh_pdl_ppo_v1"
REPORT_FILE = "reports/es_pdh_pdl_equity_curve_v1.png"
TRADES_FILE = "reports/es_pdh_pdl_trades_v1.csv"
STEP_LOG_FILE = "reports/es_pdh_pdl_steps_v1.csv"
BREAKDOWN_FILE = "reports/es_pdh_pdl_trade_breakdown_v1.csv"


def classify_setup(row):
    if row["first_break_above_PDH"] == 1:
        return "PDH_breakout"
    if row["first_break_below_PDL"] == 1:
        return "PDL_breakdown"
    return "other"


def classify_bias_alignment(position, row):
    if (position == 1 and row["bias_long"] == 1) or (position == -1 and row["bias_short"] == 1):
        return "aligned"
    if (position == 1 and row["bias_short"] == 1) or (position == -1 and row["bias_long"] == 1):
        return "counter"
    return "neutral"


print("Loading data...")
df = pd.read_csv(DATA_FILE)

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

model = PPO.load(MODEL_FILE)

obs, _ = env.reset(seed=42, options={"start_idx": 0})

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
        }

    # Exit detected
    if prev_position != 0 and new_position == 0 and current_trade is not None:
        exit_price = row["close"]
        pnl = (exit_price - prev_entry_price) * prev_position * 50 - 5.0

        current_trade.update({
            "exit_time": row["timestamp"],
            "exit_price": exit_price,
            "pnl": pnl,
            "exit_action": action,
            "exit_reason": info.get("exit_reason"),
        })

        trades.append(current_trade)
        current_trade = None

    actions.append(action)
    realized_equity_curve.append(info["realized_equity"])
    unrealized_pnl_curve.append(info["unrealized_pnl"])
    total_equity_curve.append(info["total_equity"])
    step_logs.append({
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
    })

    done = terminated or truncated

print("\n========== EVALUATION ==========")
print(f"Final realized equity: ${realized_equity_curve[-1]:.2f}")
print(f"Final total equity: ${total_equity_curve[-1]:.2f}")
print(f"Cumulative reward: {cumulative_reward:.2f}")
action_counts = {a: actions.count(a) for a in set(actions)}
print("Action counts:", action_counts)
print(f"Logged trades: {len(trades)}")

Path("reports").mkdir(parents=True, exist_ok=True)
pd.DataFrame(step_logs).to_csv(STEP_LOG_FILE, index=False)
print(f"Saved step log: {STEP_LOG_FILE}")

if trades:
    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(TRADES_FILE, index=False)

    print("\n========== TRADES ==========")
    print(trades_df)
    print(f"\nSaved trades: {TRADES_FILE}")

    breakdown_df = (
        trades_df.groupby(["direction", "setup_type", "bias_alignment"], dropna=False)
        .agg(
            trades=("pnl", "count"),
            avg_pnl=("pnl", "mean"),
            total_pnl=("pnl", "sum"),
            win_rate=("pnl", lambda x: float((x > 0).mean())),
        )
        .reset_index()
    )
    breakdown_df.to_csv(BREAKDOWN_FILE, index=False)
    print("\n========== TRADE BREAKDOWN ==========")
    print(breakdown_df)
    print(f"Saved breakdown: {BREAKDOWN_FILE}")
else:
    print("No completed trades logged.")

plt.figure(figsize=(12, 5))
plt.plot(realized_equity_curve, label="Realized equity")
plt.plot(unrealized_pnl_curve, label="Unrealized PnL")
plt.plot(total_equity_curve, label="Total equity (MTM)")
plt.title("ES PDH/PDL PPO V1 - Equity Curves")
plt.xlabel("Step")
plt.ylabel("Equity ($)")
plt.grid(True)
plt.legend()
plt.savefig(REPORT_FILE)
print(f"Saved report: {REPORT_FILE}")
plt.show()
