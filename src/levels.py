import numpy as np
import pandas as pd


def _bars_since_first_event(series: pd.Series) -> pd.Series:
    """Return bars since the first 1-valued event within a group.

    Returns 0 before the event occurs, 1 on the event bar, 2 on the next bar,
    and so on.  If the event never occurs in the group every value is 0.
    """
    out = np.zeros(len(series), dtype=np.int32)
    count = 0
    for i, v in enumerate(series.values):
        if v == 1:
            count = 1
        elif count > 0:
            count += 1
        out[i] = count
    return pd.Series(out, index=series.index)


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

    # Bars since first breakout event for each day (0 = no event yet, 1+ = count)
    df["bars_since_long_break"] = (
        df.groupby("date", group_keys=False)["first_break_above_PDH"]
        .apply(_bars_since_first_event)
    )
    df["bars_since_short_break"] = (
        df.groupby("date", group_keys=False)["first_break_below_PDL"]
        .apply(_bars_since_first_event)
    )

    return df