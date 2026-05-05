import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ESBreakoutEnv(gym.Env):
    def __init__(
        self,
        df,
        max_steps=5000,
        point_value=50,
        commission=5.0,
        max_trades=3,
        max_hold_bars=96,
        stop_loss_points=10,
        take_profit_points=25,
        rth_only_entries=True,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.max_steps = max_steps
        self.point_value = point_value
        self.commission = commission
        self.max_trades = max_trades
        self.max_hold_bars = max_hold_bars
        self.stop_loss_points = stop_loss_points
        self.take_profit_points = take_profit_points
        self.rth_only_entries = rth_only_entries

        self.action_space = spaces.Discrete(4)

        self.feature_cols = [
            "first_break_above_PDH",
            "first_break_below_PDL",
            "break_above_PDH",
            "break_below_PDL",
            "near_PDH",
            "near_PDL",
            "trend_1h_up",
            "trend_4h_up",
            "bias_long",
            "bias_short",
            "is_rth",
            "is_eth",
            "is_roll_period",
        ]

        obs_size = len(self.feature_cols) + 3

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        max_start = len(self.df) - self.max_steps - 2
        self.start_idx = int(self.np_random.integers(0, max_start))
        self.current_idx = self.start_idx
        self.end_idx = self.start_idx + self.max_steps

        self.position = 0
        self.entry_price = 0.0
        self.entry_idx = None

        self.trade_count = 0
        self.equity = 0.0
        self.peak_equity = 0.0

        return self._get_obs(), {}

    def _get_unrealized_pnl(self):
        if self.position == 0:
            return 0.0

        price = self.df.iloc[self.current_idx]["close"]
        return (price - self.entry_price) * self.position * self.point_value

    def _bars_held(self):
        if self.entry_idx is None:
            return 0
        return self.current_idx - self.entry_idx

    def _get_obs(self):
        row = self.df.iloc[self.current_idx]
        features = [row[col] for col in self.feature_cols]

        unrealized_pnl = self._get_unrealized_pnl()

        obs = np.array(
            features + [
                self.position,
                unrealized_pnl / 1000.0,
                self._bars_held() / max(1, self.max_hold_bars),
            ],
            dtype=np.float32,
        )

        return obs

    def _close_position(self, price):
        realized = (price - self.entry_price) * self.position * self.point_value
        realized_after_cost = realized - self.commission

        self.equity += realized_after_cost

        self.position = 0
        self.entry_price = 0.0
        self.entry_idx = None

        return realized_after_cost

    def step(self, action):
        row = self.df.iloc[self.current_idx]
        next_row = self.df.iloc[self.current_idx + 1]

        price = row["close"]
        next_price = next_row["close"]

        reward = 0.0
        terminated = False
        truncated = False
        exit_reason = None

        # Reward/penalty while holding position
        if self.position != 0:
            pnl_change = (next_price - price) * self.position * self.point_value
            reward += pnl_change

        # Hard exits
        if self.position != 0:
            unrealized_points = (price - self.entry_price) * self.position
            bars_held = self._bars_held()

            if unrealized_points <= -self.stop_loss_points:
                realized = self._close_position(price)
                reward += realized - 50
                exit_reason = "stop_loss"

            elif unrealized_points >= self.take_profit_points:
                realized = self._close_position(price)
                reward += realized + 50
                exit_reason = "take_profit"

            elif bars_held >= self.max_hold_bars:
                realized = self._close_position(price)
                reward += realized - 10
                exit_reason = "max_hold"

            elif row["is_rth"] == 0 and self.rth_only_entries:
                realized = self._close_position(price)
                reward += realized - 10
                exit_reason = "outside_rth_exit"

        near_or_event = (
            row["near_PDH"] == 1 or
            row["near_PDL"] == 1 or
            row["first_break_above_PDH"] == 1 or
            row["first_break_below_PDL"] == 1
        )

        # RTH-only entries
        if self.rth_only_entries and row["is_rth"] != 1 and action in [1, 2]:
            reward -= 5
            action = 0

        # Penalize entries away from PDH/PDL
        if action in [1, 2] and not near_or_event:
            reward -= 5
            action = 0

        # LONG entry
        if action == 1 and self.position == 0:
            if self.trade_count < self.max_trades:
                self.position = 1
                self.entry_price = price
                self.entry_idx = self.current_idx
                self.trade_count += 1
                reward -= self.commission

                # Strong bonus for valid long breakout with bias
                if row["bias_long"] == 1 and row["first_break_above_PDH"] == 1:
                    reward += 50

                # Mild penalty for going long against short bias
                if row["bias_short"] == 1:
                    reward -= 15

        # SHORT entry
        elif action == 2 and self.position == 0:
            if self.trade_count < self.max_trades:
                self.position = -1
                self.entry_price = price
                self.entry_idx = self.current_idx
                self.trade_count += 1
                reward -= self.commission

                # Strong bonus for valid short breakdown with bias
                if row["bias_short"] == 1 and row["first_break_below_PDL"] == 1:
                    reward += 50

                # Mild penalty for going short against long bias
                if row["bias_long"] == 1:
                    reward -= 15

        # Manual EXIT
        elif action == 3 and self.position != 0:
            realized = self._close_position(price)
            reward += realized

            if realized > 500:
                reward += 50
            elif realized > 250:
                reward += 20
            elif -50 < realized < 50:
                reward -= 10

            exit_reason = "agent_exit"

        # No reward for waiting now
        if action == 0 and not near_or_event and self.position == 0:
            reward += 0.0

        # Drawdown penalty
        self.peak_equity = max(self.peak_equity, self.equity)
        drawdown = self.peak_equity - self.equity

        if drawdown > 1000:
            reward -= 50

        self.current_idx += 1

        if self.current_idx >= self.end_idx:
            truncated = True

            if self.position != 0:
                final_price = self.df.iloc[self.current_idx]["close"]
                realized = self._close_position(final_price)
                reward += realized
                exit_reason = "episode_end"

        info = {
            "equity": self.equity,
            "position": self.position,
            "trade_count": self.trade_count,
            "exit_reason": exit_reason,
        }

        return self._get_obs(), reward, terminated, truncated, info