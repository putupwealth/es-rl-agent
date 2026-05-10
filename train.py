"""
Train PPO model for ES PDH/PDL strategy.

Commands:
    python train.py
    python train.py --version 2 --seed 42
    python train.py --run-id es_pdh_pdl_v2_s42_20260510_154210
    python train.py --data-file data/ES_1min_all_sessions.csv
    python train.py --model-dir models --latest-pointer models/latest_model.txt
"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO

from src.env import ESBreakoutEnv
from src.features import add_htf_bias
from src.levels import add_breakout_features, add_pdh_pdl


DATA_FILE = "data/ES_1min_all_sessions.csv"
DEFAULT_MODEL_DIR = "models"
DEFAULT_LATEST_POINTER = "models/latest_model.txt"
DEFAULT_TOTAL_TIMESTEPS = 100_000


def default_run_id(version: int, seed: int) -> str:
    """Build a standard timestamped model/run identifier."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"es_pdh_pdl_v{version}_s{seed}_{timestamp}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PPO model for ES PDH/PDL strategy."
    )
    parser.add_argument(
        "--data-file",
        default=DATA_FILE,
        help="Path to input CSV data.",
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Directory to save trained model artifacts.",
    )
    parser.add_argument(
        "--latest-pointer",
        default=DEFAULT_LATEST_POINTER,
        help="Path to latest model pointer file.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional explicit run ID. If omitted, one is generated automatically.",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=2,
        help="Version used when generating run ID.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used when generating run ID.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=DEFAULT_TOTAL_TIMESTEPS,
        help="Total PPO training timesteps.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    run_id = args.run_id or default_run_id(version=args.version, seed=args.seed)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / run_id
    saved_model_file = model_path.with_suffix(".zip")

    print("Loading data...")
    df = pd.read_csv(args.data_file, low_memory=False)

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
    print(f"Holdout rows:  {len(test_df):,}")
    print(f"Run ID:        {run_id}")
    print(f"Model file:    {saved_model_file}")

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
        seed=args.seed,
    )

    print("Training started...")
    model.learn(total_timesteps=args.total_timesteps)

    model.save(str(model_path))
    print(f"Model saved: {saved_model_file}")

    # Store a repo-relative path in the pointer for consistency with other scripts.
    latest_pointer = Path(args.latest_pointer)
    latest_pointer.parent.mkdir(parents=True, exist_ok=True)
    latest_pointer.write_text(str(saved_model_file), encoding="utf-8")
    print(f"Updated latest pointer: {latest_pointer} -> {saved_model_file}")


if __name__ == "__main__":
    main()