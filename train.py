import pandas as pd
from stable_baselines3 import PPO

from src.levels import add_pdh_pdl, add_breakout_features
from src.features import add_htf_bias
from src.env import ESBreakoutEnv


DATA_FILE = "data/ES_1min_all_sessions.csv"
MODEL_FILE = "models/es_pdh_pdl_ppo_v1"

print("Loading data...")
df = pd.read_csv(DATA_FILE)

print("Building features...")
df = add_pdh_pdl(df)
df = add_breakout_features(df)
df = add_htf_bias(df)

# Remove roll-period data for V1
df = df[df["is_roll_period"] == 0].reset_index(drop=True)

print(f"Training rows: {len(df):,}")

env = ESBreakoutEnv(
    df=df,
    max_steps=5000,
    point_value=50,
    commission=5.0,
    max_trades=3,
)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
)

print("Training started...")
model.learn(total_timesteps=300_000)

model.save(MODEL_FILE)

print(f"Model saved: {MODEL_FILE}")