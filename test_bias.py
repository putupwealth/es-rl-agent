import pandas as pd
from src.levels import add_pdh_pdl, add_breakout_features
from src.features import add_htf_bias

df = pd.read_csv("data/ES_1min_all_sessions.csv")

df = add_pdh_pdl(df)
df = add_breakout_features(df)
df = add_htf_bias(df)

print(df[[
    "timestamp",
    "close",
    "PDH",
    "PDL",
    "first_break_above_PDH",
    "first_break_below_PDL",
    "trend_1h_up",
    "trend_4h_up",
    "bias_long",
    "bias_short",
]].head(30))

print("\nBias counts:")
print(df[["bias_long", "bias_short"]].sum())