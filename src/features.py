import pandas as pd


def add_htf_bias(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert("US/Eastern")

    df = df.sort_values("timestamp").set_index("timestamp")

    # ---------- 1H candles ----------
    h1 = df.resample("1h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    h1["ema9_1h"] = h1["close"].ewm(span=9, adjust=False).mean()
    h1["ema21_1h"] = h1["close"].ewm(span=21, adjust=False).mean()
    h1["trend_1h_up"] = (h1["ema9_1h"] > h1["ema21_1h"]).astype(int)
    h1["trend_1h_down"] = (h1["ema9_1h"] < h1["ema21_1h"]).astype(int)

    # ---------- 4H candles ----------
    h4 = df.resample("4h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    h4["ema9_4h"] = h4["close"].ewm(span=9, adjust=False).mean()
    h4["ema21_4h"] = h4["close"].ewm(span=21, adjust=False).mean()
    h4["trend_4h_up"] = (h4["ema9_4h"] > h4["ema21_4h"]).astype(int)
    h4["trend_4h_down"] = (h4["ema9_4h"] < h4["ema21_4h"]).astype(int)

    # ---------- merge back to 1m ----------
    df = pd.merge_asof(
        df.sort_index(),
        h1[["trend_1h_up", "trend_1h_down", "ema9_1h", "ema21_1h"]].sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )

    df = pd.merge_asof(
        df.sort_index(),
        h4[["trend_4h_up", "trend_4h_down", "ema9_4h", "ema21_4h"]].sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )

    df["bias_long"] = (
        (df["trend_1h_up"] == 1) &
        (df["trend_4h_up"] == 1)
    ).astype(int)

    df["bias_short"] = (
        (df["trend_1h_down"] == 1) &
        (df["trend_4h_down"] == 1)
    ).astype(int)

    df = df.reset_index()

    return df