from src.env import ESBreakoutEnv

import pandas as pd
from stable_baselines3.common.env_checker import check_env


def make_base_rows():
    return [
        {
            "close": 5000.0,
            "first_break_above_PDH": 0,
            "first_break_below_PDL": 0,
            "break_above_PDH": 0,
            "break_below_PDL": 0,
            "near_PDH": 0,
            "near_PDL": 1,
            "trend_1h_up": 0,
            "trend_1h_down": 0,
            "trend_4h_up": 0,
            "trend_4h_down": 0,
            "bias_long": 0,
            "bias_short": 0,
            "is_rth": 1,
            "is_eth": 0,
            "is_roll_period": 0,
        },
        {
            "close": 5001.0,
            "first_break_above_PDH": 1,
            "first_break_below_PDL": 0,
            "break_above_PDH": 1,
            "break_below_PDL": 0,
            "near_PDH": 1,
            "near_PDL": 0,
            "trend_1h_up": 1,
            "trend_1h_down": 0,
            "trend_4h_up": 1,
            "trend_4h_down": 0,
            "bias_long": 1,
            "bias_short": 0,
            "is_rth": 1,
            "is_eth": 0,
            "is_roll_period": 0,
        },
        {
            "close": 4999.0,
            "first_break_above_PDH": 0,
            "first_break_below_PDL": 1,
            "break_above_PDH": 0,
            "break_below_PDL": 1,
            "near_PDH": 0,
            "near_PDL": 1,
            "trend_1h_up": 0,
            "trend_1h_down": 1,
            "trend_4h_up": 0,
            "trend_4h_down": 1,
            "bias_long": 0,
            "bias_short": 1,
            "is_rth": 1,
            "is_eth": 0,
            "is_roll_period": 0,
        },
        {
            "close": 5000.0,
            "first_break_above_PDH": 0,
            "first_break_below_PDL": 0,
            "break_above_PDH": 0,
            "break_below_PDL": 0,
            "near_PDH": 0,
            "near_PDL": 0,
            "trend_1h_up": 0,
            "trend_1h_down": 0,
            "trend_4h_up": 0,
            "trend_4h_down": 0,
            "bias_long": 0,
            "bias_short": 0,
            "is_rth": 1,
            "is_eth": 0,
            "is_roll_period": 0,
        },
        {
            "close": 5000.5,
            "first_break_above_PDH": 0,
            "first_break_below_PDL": 0,
            "break_above_PDH": 0,
            "break_below_PDL": 0,
            "near_PDH": 0,
            "near_PDL": 0,
            "trend_1h_up": 0,
            "trend_1h_down": 0,
            "trend_4h_up": 0,
            "trend_4h_down": 0,
            "bias_long": 0,
            "bias_short": 0,
            "is_rth": 1,
            "is_eth": 0,
            "is_roll_period": 0,
        },
        {
            "close": 5000.25,
            "first_break_above_PDH": 0,
            "first_break_below_PDL": 0,
            "break_above_PDH": 0,
            "break_below_PDL": 0,
            "near_PDH": 0,
            "near_PDL": 0,
            "trend_1h_up": 0,
            "trend_1h_down": 0,
            "trend_4h_up": 0,
            "trend_4h_down": 0,
            "bias_long": 0,
            "bias_short": 0,
            "is_rth": 1,
            "is_eth": 0,
            "is_roll_period": 0,
        },
    ]


def make_env(rows):
    return ESBreakoutEnv(
        df=pd.DataFrame(rows),
        max_steps=2,
        commission=5.0,
        max_trades=10,
        rth_only_entries=True,
    )


env = make_env(make_base_rows())
check_env(env, warn=True)

obs, _ = env.reset()
print("Environment OK")
print("Observation shape:", obs.shape)
print("First observation:", obs)

# Invalid LONG attempt: near PDL but no bullish breakout context.
env = make_env(make_base_rows())
env.reset(seed=42, options={"start_idx": 0})
_, reward, _, _, info = env.step(1)
assert info["position"] == 0, "LONG should be blocked without PDH breakout context"
assert info["valid_long_zone"] == 0
assert info["blocked_reason"] == "invalid_long_zone"
assert reward < 0

# Valid LONG attempt: breakout above PDH.
env = make_env(make_base_rows())
env.reset(seed=42, options={"start_idx": 1})
_, _, _, _, info = env.step(1)
assert info["position"] == 1, "LONG should be allowed on PDH breakout context"
assert info["valid_long_zone"] == 1
assert info["allowed_long_entry"] == 1
assert info["entry_rule_trigger"] == "long_breakout_context"

# Invalid SHORT attempt: no bearish breakdown context.
env = make_env(make_base_rows())
env.reset(seed=42, options={"start_idx": 1})
_, reward, _, _, info = env.step(2)
assert info["position"] == 0, "SHORT should be blocked without PDL breakdown context"
assert info["valid_short_zone"] == 0
assert info["blocked_reason"] == "invalid_short_zone"
assert reward < 0

# Valid SHORT attempt: breakdown below PDL.
env = make_env(make_base_rows())
env.reset(seed=42, options={"start_idx": 2})
_, _, _, _, info = env.step(2)
assert info["position"] == -1, "SHORT should be allowed on PDL breakdown context"
assert info["valid_short_zone"] == 1
assert info["allowed_short_entry"] == 1
assert info["entry_rule_trigger"] == "short_breakdown_context"

print("Entry gating checks passed")
