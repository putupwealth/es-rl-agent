import pandas as pd
from stable_baselines3 import PPO
from datetime import datetime
from pathlib import Path
import argparse

from src.levels import add_pdh_pdl, add_breakout_features
from src.features import add_htf_bias
from src.env import ESBreakoutEnv


DATA_FILE = "data/ES_1min_all_sessions.csv"

def default_run_id(version: int, seed: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"es_pdh_pdl_v{version}_s{seed}_{timestamp}"


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO model for ES PDH/PDL strategy.")
    parser.add_argument("--data-file", default=DATA_FILE, help="Path to input CSV data.")
    parser.add_argument("--model-dir", default="models", help="Directory to save trained model artifacts.")
    parser.add_argument("--latest-pointer", default="models/latest_model.txt", help="Path to latest model pointer file.")
    parser.add_argument("--run-id", default=None, help="Optional explicit run ID.")
    parser.add_argument("--version", type=int, default=2, help="Run version used when generating run ID.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used in generated run ID.")
    return parser.parse_args()


def main():
    args = parse_args()
    run_id = args.run_id or default_run_id(version=args.version, seed=args.seed)
    model_path = Path(args.model_dir) / run_id
    saved_model_file = model_path.with_suffix(".zip")

    print("Loading data...")
    df = pd.read_csv(args.data_file)

    print("Building features...")
    df = add_pdh_pdl(df)
    df = add_breakout_features(df)
    df = add_htf_bias(df)

    # Remove roll-period data before training.
    df = df[df["is_roll_period"] == 0].reset_index(drop=True)

    # Restrict training to RTH bars only: entries are RTH-gated so ETH bars
    # just teach the policy that doing nothing is always correct.
    df = df[df["is_rth"] == 1].reset_index(drop=True)

    split = int(len(df) * 0.8)
    train_df = df.iloc[:split].reset_index(drop=True)
    test_df = df.iloc[split:].reset_index(drop=True)

    if len(train_df) < 100:
        raise ValueError("Not enough rows in training split for RL training.")

    print(f"Training rows: {len(train_df):,}")
    print(f"Holdout rows: {len(test_df):,}")
    print(f"Run ID: {run_id}")

    env = ESBreakoutEnv(
        df=train_df,
        # ~390 bars = one RTH session (09:30–16:00). Shorter episodes improve
        # credit assignment when setups are sparse.
        max_steps=min(390, len(train_df) - 2),
        point_value=50,
        commission=5.0,
        max_trades=10,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        # Explicit entropy bonus to maintain exploration in a sparse-action
        # trading environment where inactivity has zero variance reward.
        ent_coef=0.01,
    )

    print("Training started...")
    model.learn(total_timesteps=1_000_00)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    print(f"Model saved: {saved_model_file}")

    latest_pointer = Path(args.latest_pointer)
    latest_pointer.parent.mkdir(parents=True, exist_ok=True)
    latest_pointer.write_text(str(saved_model_file), encoding="utf-8")
    print(f"Updated latest pointer: {latest_pointer} -> {saved_model_file}")


if __name__ == "__main__":
    main()
