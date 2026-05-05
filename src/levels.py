import pandas as pd


def add_pdh_pdl(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert("US/Eastern")

    # Extract date
    df["date"] = df["timestamp"].dt.date

    # ================== RTH ONLY ==================
    rth_df = df[df["is_rth"] == 1]

    daily = rth_df.groupby("date").agg(
        day_high=("high", "max"),
        day_low=("low", "min"),
        day_close=("close", "last")
    ).reset_index()

    # Shift to get previous day levels
    daily["PDH"] = daily["day_high"].shift(1)
    daily["PDL"] = daily["day_low"].shift(1)
    daily["PDC"] = daily["day_close"].shift(1)

    # Merge back to full dataset
    df = df.merge(
        daily[["date", "PDH", "PDL", "PDC"]],
        on="date",
        how="left"
    )

    # Drop first day (no previous levels)
    df = df.dropna(subset=["PDH", "PDL"]).reset_index(drop=True)

    return df


def add_breakout_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Distance to levels
    df["dist_to_PDH"] = df["close"] - df["PDH"]
    df["dist_to_PDL"] = df["close"] - df["PDL"]

    # Near level (within 5 ES points)
    threshold = 5

    df["near_PDH"] = (df["dist_to_PDH"].abs() <= threshold).astype(int)
    df["near_PDL"] = (df["dist_to_PDL"].abs() <= threshold).astype(int)

    # Breakout logic
    df["break_above_PDH"] = (df["close"] > df["PDH"]).astype(int)
    df["break_below_PDL"] = (df["close"] < df["PDL"]).astype(int)

    # First breakout event only
    prev_close = df.groupby("date")["close"].shift(1)

    df["first_break_above_PDH"] = (
    (df["close"] > df["PDH"]) &
    (prev_close <= df["PDH"])
    ).astype(int)

    df["first_break_below_PDL"] = (
    (df["close"] < df["PDL"]) &
    (prev_close >= df["PDL"])
    ).astype(int)

# Retest logic after breakout
    df["retest_PDH"] = (
    (df["break_above_PDH"] == 1) &
    (df["near_PDH"] == 1)
    ).astype(int)

    df["retest_PDL"] = (
    (df["break_below_PDL"] == 1) &
    (df["near_PDL"] == 1)
    ).astype(int)

    return df