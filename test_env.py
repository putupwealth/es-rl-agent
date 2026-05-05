import pandas as pd
from stable_baselines3.common.env_checker import check_env

from src.levels import add_pdh_pdl, add_breakout_features
from src.features import add_htf_bias
from src.env import ESBreakoutEnv

df = pd.read_csv("data/ES_1min_all_sessions.csv")

df = add_pdh_pdl(df)
df = add_breakout_features(df)
df = add_htf_bias(df)

env = ESBreakoutEnv(df)

check_env(env, warn=True)

obs, _ = env.reset()
print("Environment OK")
print("Observation shape:", obs.shape)
print("First observation:", obs)