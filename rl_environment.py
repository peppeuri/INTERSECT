"""
INTERSECT — Autonomous AI Trading System
rl_environment.py: Gymnasium-compatible trading environment
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any


class TradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 1000.0,
        max_leverage: int = 3,
        max_position_pct: float = 0.02,
        commission: float = 0.0005,
        slippage: float = 0.0003,
        max_drawdown_pct: float = 0.20,
        episode_length: int = 500,
        window_size: int = 20,
        regime_probs: Optional[np.ndarray] = None,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.max_leverage = max_leverage
        self.max_position_pct = max_position_pct
        self.commission = commission
        self.slippage = slippage
        self.max_drawdown_pct = max_drawdown_pct
        self.episode_length = episode_length
        self.window_size = window_size
        self.regime_probs = regime_probs

        self.n_regimes = 6
        self.n_actions = 7

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._get_state_dim(),),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_actions)

        self._precompute_indicators()
        self.reset()

    def _get_state_dim(self) -> int:
        return 20 + 1 + 1 + 1 + 1 + 1 + self.n_regimes + 3 + 1 + 1 + 1

    def _precompute_indicators(self):
        closes = self.df['close'].values
        highs = self.df['high'].values
        lows = self.df['low'].values
        volumes = self.df['volume'].values

        self.price_changes = np.zeros(len(closes))
        for i in range(1, len(closes)):
            self.price_changes[i] = (closes[i] - closes[i - 1]) / closes[i - 1]

        self.vol_sma = pd.Series(volumes).rolling(20).mean().values
        self.vol_ratio = np.where(self.vol_sma > 0, volumes / self.vol_sma, 1.0)

        self.bb_upper, self.bb_mid, self.bb_lower = self._compute_bb(closes, 20, 2.0)
        self.bb_position = np.where(
            (self.bb_upper - self.bb_lower) > 0,
            (closes - self.bb_lower) / (self.bb_upper - self.bb_lower),
            0.5
        )

        self.rsi = self._compute_rsi(closes, 14)
        self.adx = self._compute_adx(highs, lows, closes, 14)
        self.atr = self._compute_atr(highs, lows, closes, 14)
        self.atr_pct = np.where(closes > 0, self.atr / closes, 0)

        if self.regime_probs is None:
            self.regime_probs = np.ones((len(closes), self.n_regimes)) / self.n_regimes

    def _compute_bb(self, closes, period, std_mult):
        mid = pd.Series(closes).rolling(period).mean().values
        std = pd.Series(closes).rolling(period).std(ddof=0).values
        upper = mid + std_mult * std
        lower = mid - std_mult * std
        return upper, mid, lower

    def _compute_rsi(self, closes, period):
        delta = np.diff(closes, prepend=closes[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(period).mean().values
        avg_loss = pd.Series(loss).rolling(period).mean().values
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _compute_adx(self, highs, lows, closes, period):
        plus_dm = np.diff(highs, prepend=highs[0])
        minus_dm = np.diff(lows, prepend=lows[0])
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)

        tr = np.maximum(
            highs - lows,
            np.maximum(np.abs(highs - np.roll(closes, 1)),
                       np.abs(lows - np.roll(closes, 1)))
        )
        tr[0] = highs[0] - lows[0]

        atr = pd.Series(tr).rolling(period).mean().values
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean().values / np.where(atr > 0, atr, 1)
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean().values / np.where(atr > 0, atr, 1)
        dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, plus_di + minus_di, 1)
        adx = pd.Series(dx).rolling(period).mean().values
        return adx

    def _compute_atr(self, highs, lows, closes, period):
        tr = np.maximum(
            highs - lows,
            np.maximum(np.abs(highs - np.roll(closes, 1)),
                       np.abs(lows - np.roll(closes, 1)))
        )
        tr[0] = highs[0] - lows[0]
        return pd.Series(tr).rolling(period).mean().values

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.equity = self.initial_capital
        self.peak_equity = self.initial_capital
        self.position = 0
        self.position_side = 0
        self.entry_price = 0.0
        self.leverage = 1
        self.unrealized_pnl = 0.0
        self.time_since_trade = 0
        self.trades_count = 0
        self.total_pnl = 0.0
        self.done = False
        self.truncated = False
        return self._get_obs(), {}

    def _get_obs(self):
        i = self.current_step
        if i >= len(self.df):
            i = len(self.df) - 1

        price_changes = self.price_changes[max(0, i - 20):i + 1]
        if len(price_changes) < 21:
            price_changes = np.pad(price_changes, (21 - len(price_changes), 0), 'constant')

        obs = np.concatenate([
            price_changes[-20:],
            [self.vol_ratio[i]],
            [self.bb_position[i]],
            [self.rsi[i] / 100.0],
            [self.adx[i] / 100.0],
            [self.atr_pct[i]],
            self.regime_probs[i] if i < len(self.regime_probs) else np.ones(self.n_regimes) / self.n_regimes,
            [
                1.0 if self.position_side == 1 else 0.0,
                1.0 if self.position_side == -1 else 0.0,
                1.0 if self.position_side == 0 else 0.0
            ],
            [self.unrealized_pnl / self.equity if self.equity > 0 else 0.0],
            [min(self.time_since_trade / 500.0, 1.0)],
            [self.equity / self.initial_capital]
        ]).astype(np.float32)

        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def step(self, action):
        if self.done or self.truncated:
            return self._get_obs(), 0.0, True, True, {}

        price = self.df['close'].iloc[self.current_step]
        reward = 0.0

        if action == 0:
            pass
        elif action in [1, 2, 3] and self.position_side <= 0:
            if self.position_side == -1:
                reward += self._close_position(price)
            self._enter_long(price, action)
        elif action in [4, 5] and self.position_side >= 0:
            if self.position_side == 1:
                reward += self._close_position(price)
            self._enter_short(price, action - 3)
        elif action == 6 and self.position != 0:
            reward += self._close_position(price)

        if self.position != 0:
            self.unrealized_pnl = self._calculate_unrealized_pnl(price)
            reward += self.unrealized_pnl * 0.01

        self.current_step += 1
        self.time_since_trade += 1

        drawdown = (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0
        reward -= 0.01 * drawdown
        reward -= 0.001 * (self.leverage ** 2)

        self.peak_equity = max(self.peak_equity, self.equity)

        if drawdown >= self.max_drawdown_pct:
            self.done = True
            reward -= 1.0

        if self.current_step >= self.window_size + self.episode_length:
            self.truncated = True

        if self.equity <= self.initial_capital * 0.1:
            self.done = True
            reward -= 10.0

        return self._get_obs(), float(reward), self.done, self.truncated, self._get_info()

    def _enter_long(self, price, action):
        self.leverage = action
        max_size = (self.equity * self.max_position_pct * self.leverage) / price
        self.position = max_size
        self.position_side = 1
        self.entry_price = price * (1 + self.slippage)
        cost = self.position * self.entry_price * self.commission
        self.equity -= cost
        self.time_since_trade = 0
        self.trades_count += 1

    def _enter_short(self, price, action):
        self.leverage = action
        max_size = (self.equity * self.max_position_pct * self.leverage) / price
        self.position = max_size
        self.position_side = -1
        self.entry_price = price * (1 - self.slippage)
        cost = self.position * self.entry_price * self.commission
        self.equity -= cost
        self.time_since_trade = 0
        self.trades_count += 1

    def _close_position(self, price):
        if self.position == 0:
            return 0.0

        exit_price = price * (1 - self.slippage) if self.position_side == 1 else price * (1 + self.slippage)
        pnl = self.position * (exit_price - self.entry_price) * self.position_side
        cost = self.position * exit_price * self.commission
        net_pnl = pnl - cost

        self.equity += net_pnl
        self.total_pnl += net_pnl

        r = net_pnl / (self.position * self.entry_price) if self.position * self.entry_price > 0 else 0
        r += 0.1 if net_pnl > 0 else -0.3

        self.position = 0
        self.position_side = 0
        self.entry_price = 0.0
        self.leverage = 1
        self.unrealized_pnl = 0.0
        self.time_since_trade = 0

        return r

    def _calculate_unrealized_pnl(self, price):
        if self.position == 0:
            return 0.0
        return self.position * (price - self.entry_price) * self.position_side

    def _get_info(self):
        return {
            'equity': self.equity,
            'drawdown': (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0,
            'position': self.position,
            'position_side': self.position_side,
            'leverage': self.leverage,
            'total_pnl': self.total_pnl,
            'trades': self.trades_count,
            'step': self.current_step
        }

    def render(self):
        print(f"Step: {self.current_step}, Equity: {self.equity:.2f}, "
              f"Position: {self.position_side * self.position:.4f}, "
              f"DD: {(self.peak_equity - self.equity) / self.peak_equity * 100:.2f}%")

    def set_regime_probs(self, probs: np.ndarray):
        self.regime_probs = probs


def make_env(df: pd.DataFrame, **kwargs) -> TradingEnv:
    return TradingEnv(df, **kwargs)
