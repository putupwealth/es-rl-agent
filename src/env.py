import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ESBreakoutEnv(gym.Env):
    def __init__(
        self,
        df,
        max_steps=5000,
        point_value=50,      # ES = $50 per point
        commission=5.0,
        max_trades=3,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.max_steps = max_steps
        self.point_value = point_value
        self.commission = commission
        self.max_trades = max_trades

        # 0 = HOLD, 1 = LONG, 2 = SHORT, 3 = EXIT
        self.action_space = spaces.Discrete(4)

        # Observation features count
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

        # + position + unrealized pnl
        obs_size = len(self.feature_cols) + 2

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

        self.position = 0  # 0 flat, 1 long, -1 short
        self.entry_price = 0.0
        self.trade_count = 0
        self.equity = 0.0
        self.peak_equity = 0.0

        return self._get_obs(), {}

    def _get_unrealized_pnl(self):
        if self.position == 0:
            return 0.0

        price = self.df.iloc[self.current_idx]["close"]
        return (price - self.entry_price) * self.position * self.point_value

    def _get_obs(self):
        row = self.df.iloc[self.current_idx]

        features = [row[col] for col in self.feature_cols]
        unrealized_pnl = self._get_unrealized_pnl()

        obs = np.array(
            features + [self.position, unrealized_pnl / 1000.0],
            dtype=np.float32,
        )

        return obs

    def step(self, action):
        row = self.df.iloc[self.current_idx]
        next_row = self.df.iloc[self.current_idx + 1]

        price = row["close"]
        next_price = next_row["close"]

        reward = 0.0
        terminated = False
        truncated = False

        # Reward/penalty for price movement while in position
        if self.position != 0:
            pnl_change = (next_price - price) * self.position * self.point_value
            reward += pnl_change

        # Force model to focus only around PDH/PDL events/areas
        near_or_event = (
            row["near_PDH"] == 1 or
            row["near_PDL"] == 1 or
            row["first_break_above_PDH"] == 1 or
            row["first_break_below_PDL"] == 1
        )

        # Penalize new entries away from key levels
        if action in [1, 2] and not near_or_event:
            reward -= 20
            action = 0

        # LONG
        if action == 1:
            if self.position == 0 and self.trade_count < self.max_trades:
                self.position = 1
                self.entry_price = price
                self.trade_count += 1
                reward -= self.commission

                # Bonus for long aligned with bias
                if row["bias_long"] == 1 and row["first_break_above_PDH"] == 1:
                    reward += 10

                # Penalty for long against short bias
                if row["bias_short"] == 1:
                    reward -= 10

        # SHORT
        elif action == 2:
            if self.position == 0 and self.trade_count < self.max_trades:
                self.position = -1
                self.entry_price = price
                self.trade_count += 1
                reward -= self.commission

                # Bonus for short aligned with bias
                if row["bias_short"] == 1 and row["first_break_below_PDL"] == 1:
                    reward += 10

                # Penalty for short against long bias
                if row["bias_long"] == 1:
                    reward -= 10

        # EXIT
        elif action == 3:
            if self.position != 0:
                realized = (price - self.entry_price) * self.position * self.point_value
                self.equity += realized - self.commission
                reward += realized - self.commission

                # High-RR style reward
                if realized > 300:
                    reward += 25
                elif realized > 150:
                    reward += 10

                # Penalty for tiny random exits
                if -50 < realized < 50:
                    reward -= 10

                self.position = 0
                self.entry_price = 0.0

        # Overtrade penalty
        if self.trade_count > self.max_trades:
            reward -= 50

        # Drawdown penalty
        self.peak_equity = max(self.peak_equity, self.equity)
        drawdown = self.peak_equity - self.equity

        if drawdown > 1000:
            reward -= 50

        # Small reward for waiting when not near setup
        if action == 0 and not near_or_event and self.position == 0:
            reward += 0.05

        self.current_idx += 1

        if self.current_idx >= self.end_idx:
            truncated = True

            # Force close open position
            if self.position != 0:
                final_price = self.df.iloc[self.current_idx]["close"]
                realized = (final_price - self.entry_price) * self.position * self.point_value
                self.equity += realized - self.commission
                reward += realized - self.commission
                self.position = 0

        info = {
            "equity": self.equity,
            "position": self.position,
            "trade_count": self.trade_count,
        }

        return self._get_obs(), reward, terminated, truncated, info