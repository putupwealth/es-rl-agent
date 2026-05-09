from src.env import ESBreakoutEnv

import pandas as pd
from stable_baselines3.common.env_checker import check_env


def make_base_rows():
    return [
        {
            # Row 0: near PDL, no breakout context
            "close": 5000.0,
            "first_break_above_PDH": 0,
            "first_break_below_PDL": 0,
            "break_above_PDH": 0,
            "break_below_PDL": 0,
            "near_PDH": 0,
            "near_PDL": 1,
            "retest_PDH": 0,
            "retest_PDL": 0,
            "trend_1h_up": 0,
            "trend_1h_down": 0,
            "trend_4h_up": 0,
            "trend_4h_down": 0,
            "bias_long": 0,
            "bias_short": 0,
            "is_rth": 1,
            "is_eth": 0,
            "is_roll_period": 0,
            "dist_to_PDH": -20.0,
            "dist_to_PDL": 0.0,
            "bars_since_long_break": 0,
            "bars_since_short_break": 0,
        },
        {
            # Row 1: first break above PDH + retest, bias_long
            "close": 5001.0,
            "first_break_above_PDH": 1,
            "first_break_below_PDL": 0,
            "break_above_PDH": 1,
            "break_below_PDL": 0,
            "near_PDH": 1,
            "near_PDL": 0,
            "retest_PDH": 1,
            "retest_PDL": 0,
            "trend_1h_up": 1,
            "trend_1h_down": 0,
            "trend_4h_up": 1,
            "trend_4h_down": 0,
            "bias_long": 1,
            "bias_short": 0,
            "is_rth": 1,
            "is_eth": 0,
            "is_roll_period": 0,
            "dist_to_PDH": 1.0,
            "dist_to_PDL": -20.0,
            "bars_since_long_break": 1,
            "bars_since_short_break": 0,
        },
        {
            # Row 2: first break below PDL + retest, bias_short
            "close": 4999.0,
            "first_break_above_PDH": 0,
            "first_break_below_PDL": 1,
            "break_above_PDH": 0,
            "break_below_PDL": 1,
            "near_PDH": 0,
            "near_PDL": 1,
            "retest_PDH": 0,
            "retest_PDL": 1,
            "trend_1h_up": 0,
            "trend_1h_down": 1,
            "trend_4h_up": 0,
            "trend_4h_down": 1,
            "bias_long": 0,
            "bias_short": 1,
            "is_rth": 1,
            "is_eth": 0,
            "is_roll_period": 0,
            "dist_to_PDH": -20.0,
            "dist_to_PDL": -1.0,
            "bars_since_long_break": 0,
            "bars_since_short_break": 1,
        },
        {
            # Row 3: no context
            "close": 5000.0,
            "first_break_above_PDH": 0,
            "first_break_below_PDL": 0,
            "break_above_PDH": 0,
            "break_below_PDL": 0,
            "near_PDH": 0,
            "near_PDL": 0,
            "retest_PDH": 0,
            "retest_PDL": 0,
            "trend_1h_up": 0,
            "trend_1h_down": 0,
            "trend_4h_up": 0,
            "trend_4h_down": 0,
            "bias_long": 0,
            "bias_short": 0,
            "is_rth": 1,
            "is_eth": 0,
            "is_roll_period": 0,
            "dist_to_PDH": -10.0,
            "dist_to_PDL": 10.0,
            "bars_since_long_break": 0,
            "bars_since_short_break": 0,
        },
        {
            # Row 4: no context
            "close": 5000.5,
            "first_break_above_PDH": 0,
            "first_break_below_PDL": 0,
            "break_above_PDH": 0,
            "break_below_PDL": 0,
            "near_PDH": 0,
            "near_PDL": 0,
            "retest_PDH": 0,
            "retest_PDL": 0,
            "trend_1h_up": 0,
            "trend_1h_down": 0,
            "trend_4h_up": 0,
            "trend_4h_down": 0,
            "bias_long": 0,
            "bias_short": 0,
            "is_rth": 1,
            "is_eth": 0,
            "is_roll_period": 0,
            "dist_to_PDH": -10.0,
            "dist_to_PDL": 10.0,
            "bars_since_long_break": 0,
            "bars_since_short_break": 0,
        },
        {
            # Row 5: no context
            "close": 5000.25,
            "first_break_above_PDH": 0,
            "first_break_below_PDL": 0,
            "break_above_PDH": 0,
            "break_below_PDL": 0,
            "near_PDH": 0,
            "near_PDL": 0,
            "retest_PDH": 0,
            "retest_PDL": 0,
            "trend_1h_up": 0,
            "trend_1h_down": 0,
            "trend_4h_up": 0,
            "trend_4h_down": 0,
            "bias_long": 0,
            "bias_short": 0,
            "is_rth": 1,
            "is_eth": 0,
            "is_roll_period": 0,
            "dist_to_PDH": -10.0,
            "dist_to_PDL": 10.0,
            "bars_since_long_break": 0,
            "bars_since_short_break": 0,
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

# Observation size: 17 binary + 4 continuous + 3 state = 24
# Compute dynamically so this stays in sync with src/env.py.
_check_env = make_env(make_base_rows())
_expected_obs_size = (
    len(_check_env.feature_cols)
    + len(_check_env.continuous_feature_cols)
    + 3  # position, unrealized_pnl/1000, bars_held/max_hold_bars
)
assert obs.shape == (_expected_obs_size,), f"Expected obs shape ({_expected_obs_size},), got {obs.shape}"
print("Observation shape check passed")

# Invalid LONG attempt: near PDL but no bullish breakout context.
env = make_env(make_base_rows())
env.reset(seed=42, options={"start_idx": 0})
_, reward, _, _, info = env.step(1)
assert info["position"] == 0, "LONG should be blocked without PDH breakout context"
assert info["valid_long_zone"] == 0
assert info["blocked_reason"] == "invalid_long_zone"
assert reward < 0, f"Expected negative reward for invalid LONG, got {reward}"

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
assert reward < 0, f"Expected negative reward for invalid SHORT, got {reward}"

# Valid SHORT attempt: breakdown below PDL.
env = make_env(make_base_rows())
env.reset(seed=42, options={"start_idx": 2})
_, _, _, _, info = env.step(2)
assert info["position"] == -1, "SHORT should be allowed on PDL breakdown context"
assert info["valid_short_zone"] == 1
assert info["allowed_short_entry"] == 1
assert info["entry_rule_trigger"] == "short_breakdown_context"

print("Entry gating checks passed")

# Action 3 while flat must not be a free no-op: it should produce a penalty.
env = make_env(make_base_rows())
env.reset(seed=42, options={"start_idx": 0})
_, reward, _, _, info = env.step(3)
assert info["position"] == 0, "Action 3 while flat should not open a position"
assert info["blocked_reason"] == "exit_while_flat"
assert reward < 0, f"Expected penalty for action 3 while flat, got {reward}"
print("Flat action 3 penalty check passed")

# Penalty magnitudes: invalid-zone block should be softer than -3 (old value).
env = make_env(make_base_rows())
env.reset(seed=42, options={"start_idx": 0})
_, reward, _, _, info = env.step(1)
# New penalty is -1.0; old was -3. Check it is negative but not as severe.
assert reward == -1.0, f"Expected zone-gate penalty of -1.0, got {reward}"
print("Softened zone-gate penalty check passed")
