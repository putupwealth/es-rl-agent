import os
from pathlib import Path
from datetime import time

import pandas as pd
import databento as db
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.getenv("DATABENTO_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing DATABENTO_API_KEY in .env file")

client = db.Historical(API_KEY)

# ================== CONFIG ==================
DATASET = "GLBX.MDP3"

PRODUCT = "ES"
SYMBOL = "ES.v.0"       # Continuous front-month ES
SCHEMA = "ohlcv-1m"

START = "2022-01-01"
END = "2026-01-01"      # End is exclusive

OUTPUT_FILE = f"data/{PRODUCT}_1min_all_sessions.csv"
# ============================================

Path("data").mkdir(exist_ok=True)

print(f"Downloading {SYMBOL} 1-min data from {START} to {END}...")

data = client.timeseries.get_range(
    dataset=DATASET,
    symbols=[SYMBOL],
    schema=SCHEMA,
    start=START,
    end=END,
    stype_in="continuous",
)

df = data.to_df()

# Normalize timestamp index
if df.index.tz is None:
    df.index = pd.to_datetime(df.index).tz_localize("UTC")
else:
    df.index = pd.to_datetime(df.index).tz_convert("UTC")

# Convert to US/Eastern for CME/RTH logic
df.index = df.index.tz_convert("US/Eastern")

# ================== SESSION FLAGS ==================
df["session"] = "ETH"

rth_mask = (
    (df.index.time >= time(9, 30)) &
    (df.index.time <= time(16, 0))
)

df.loc[rth_mask, "session"] = "RTH"
df["is_rth"] = (df["session"] == "RTH").astype(int)
df["is_eth"] = (df["session"] == "ETH").astype(int)

print("Session counts:")
print(df["session"].value_counts())

# ================== ROLL PERIOD FLAG ==================
df["is_roll_period"] = 0

roll_dates = pd.to_datetime([
    "2022-03-18", "2022-06-17", "2022-09-16", "2022-12-16",
    "2023-03-17", "2023-06-16", "2023-09-15", "2023-12-15",
    "2024-03-15", "2024-06-21", "2024-09-20", "2024-12-20",
    "2025-03-21", "2025-06-20", "2025-09-19", "2025-12-19",
]).tz_localize("US/Eastern")

for roll_date in roll_dates:
    mask = (
        (df.index >= roll_date - pd.Timedelta(days=5)) &
        (df.index <= roll_date + pd.Timedelta(days=3))
    )
    df.loc[mask, "is_roll_period"] = 1

print(f"Marked {df['is_roll_period'].sum():,} bars as roll period")

# Save timestamp as normal column
df = df.reset_index()

if "ts_event" in df.columns:
    df = df.rename(columns={"ts_event": "timestamp"})
elif "index" in df.columns:
    df = df.rename(columns={"index": "timestamp"})

wanted_cols = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "symbol",
    "session",
    "is_rth",
    "is_eth",
    "is_roll_period",
]

existing_cols = [col for col in wanted_cols if col in df.columns]
df = df[existing_cols]

df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved: {OUTPUT_FILE}")
print(f"Rows: {len(df):,}")
print(f"Columns: {df.columns.tolist()}")
print(df.head())