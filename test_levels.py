import pandas as pd
from src.levels import add_pdh_pdl, add_breakout_features

df = pd.read_csv("data/ES_1min_all_sessions.csv")

df = add_pdh_pdl(df)
df = add_breakout_features(df)

print("first_break_above_PDH count:", df["first_break_above_PDH"].sum())
print("first_break_below_PDL count:", df["first_break_below_PDL"].sum())

print(df[df["first_break_above_PDH"] == 1][[
    "timestamp", "close", "PDH", "PDL", "first_break_above_PDH"
]].head(20))

print(df[[
    "timestamp",
    "close",
    "PDH",
    "PDL",
    "near_PDH",
    "break_above_PDH",
    "first_break_above_PDH",
    "retest_PDH"
]].head(50))