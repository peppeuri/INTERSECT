"""
INTERSECT — Autonomous AI Trading System
market_regime_classifier.py: Unsupervised HMM regime classifier with BIC state selection
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
import joblib
import os
import time
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler


class MarketRegimeClassifier:
    def __init__(
        self,
        n_states: int = 6,
        max_states: int = 8,
        lookback: int = 20,
        retrain_interval: int = 7 * 96,
        min_train_samples: int = 1000,
        bic_improvement_threshold: float = 0.05,
        covariance_type: str = 'full',
        n_iter: int = 100,
        random_state: int = 42,
    ):
        self.n_states = n_states
        self.max_states = max_states
        self.lookback = lookback
        self.retrain_interval = retrain_interval
        self.min_train_samples = min_train_samples
        self.bic_improvement_threshold = bic_improvement_threshold
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state

        self.model = None
        self.scaler = StandardScaler()
        self.step_count = 0
        self.last_retrain = 0
        self.best_bic = np.inf
        self.best_n_states = n_states

        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)

    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        volumes = df['volume'].values

        log_returns = np.diff(np.log(closes), prepend=closes[0])
        vol_changes = np.diff(np.log(volumes + 1), prepend=np.log(volumes[0] + 1))
        atr = self._compute_atr(highs, lows, closes, 14)
        atr_pct = np.where(closes > 0, atr / closes, 0)
        bb_upper, bb_mid, bb_lower = self._compute_bb(closes, 20, 2.0)
        bb_width = np.where(bb_mid > 0, (bb_upper - bb_lower) / bb_mid, 0)
        btc_corr = np.ones(len(closes)) * 0.7

        features = np.column_stack([
            log_returns[-self.lookback:],
            vol_changes[-self.lookback:],
            atr_pct[-self.lookback:],
            bb_width[-self.lookback:],
            btc_corr[-self.lookback:],
        ])
        return features

    def _compute_atr(self, highs, lows, closes, period):
        tr = np.maximum(highs - lows,
                        np.maximum(np.abs(highs - np.roll(closes, 1)),
                                   np.abs(lows - np.roll(closes, 1))))
        tr[0] = highs[0] - lows[0]
        return pd.Series(tr).rolling(period).mean().values

    def _compute_bb(self, closes, period, std_mult):
        mid = pd.Series(closes).rolling(period).mean().values
        std = pd.Series(closes).rolling(period).std(ddof=0).values
        upper = mid + std_mult * std
        lower = mid - std_mult * std
        return upper, mid, lower

    def fit(self, df: pd.DataFrame) -> bool:
        features = self.extract_features(df)
        features = features[~np.isnan(features).any(axis=1)]
        if len(features) < self.min_train_samples:
            return False

        features_scaled = self.scaler.fit_transform(features)

        best_model = None
        best_bic = np.inf
        best_n = self.n_states

        for n in range(max(2, self.n_states - 1), min(self.max_states + 1, 9)):
            try:
                model = hmm.GaussianHMM(
                    n_components=n, covariance_type=self.covariance_type,
                    n_iter=self.n_iter, random_state=self.random_state, verbose=False
                )
                model.fit(features_scaled)
                bic = model.bic(features_scaled)
                if bic < best_bic:
                    best_bic = bic
                    best_model = model
                    best_n = n
            except Exception:
                continue

        if best_model is not None:
            improvement = (self.best_bic - best_bic) / abs(self.best_bic) if self.best_bic != np.inf else 1.0
            if improvement > self.bic_improvement_threshold or self.model is None:
                self.model = best_model
                self.best_bic = best_bic
                self.best_n_states = best_n
                self.n_states = best_n
                self._log(f'HMM retrained: {self.n_states} states, BIC={best_bic:.2f}, improvement={improvement:.3f}')
            else:
                self._log(f'HMM no improvement: current={self.best_n_states} states, BIC={self.best_bic:.2f}')

        self.last_retrain = self.step_count
        self.save()
        return True

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.ones((len(df), self.n_states)) / self.n_states

        features = self.extract_features(df)
        features = features[~np.isnan(features).any(axis=1)]
        if len(features) == 0:
            return np.ones((len(df), self.n_states)) / self.n_states

        features_scaled = self.scaler.transform(features)
        try:
            log_probs = self.model.predict_proba(features_scaled)
            result = np.ones((len(df), self.n_states)) / self.n_states
            valid_idx = ~np.isnan(self.extract_features(df)).any(axis=1)
            result[valid_idx] = log_probs
            return result
        except Exception:
            return np.ones((len(df), self.n_states)) / self.n_states

    def get_current_regime(self, df: pd.DataFrame) -> Tuple[int, np.ndarray]:
        probs = self.predict_proba(df)
        current_probs = probs[-1] if len(probs) > 0 else np.ones(self.n_states) / self.n_states
        regime = np.argmax(current_probs)
        return regime, current_probs

    def should_retrain(self) -> bool:
        return self.step_count - self.last_retrain >= self.retrain_interval

    def increment_step(self):
        self.step_count += 1

    def save(self, path: str = 'models/regime_classifier.joblib'):
        joblib.dump({
            'model': self.model, 'scaler': self.scaler,
            'n_states': self.n_states, 'best_n_states': self.best_n_states,
            'best_bic': self.best_bic, 'step_count': self.step_count,
            'last_retrain': self.last_retrain,
        }, path)

    def load(self, path: str = 'models/regime_classifier.joblib'):
        if os.path.exists(path):
            data = joblib.load(path)
            self.model = data['model']
            self.scaler = data['scaler']
            self.n_states = data['n_states']
            self.best_n_states = data['best_n_states']
            self.best_bic = data['best_bic']
            self.step_count = data['step_count']
            self.last_retrain = data['last_retrain']
            self._log(f'Regime classifier loaded from {path}')

    def _log(self, msg: str):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f'{timestamp} | REGIME | {msg}\n'
        print(log_entry.strip())
        with open('logs/regime_classifier.log', 'a') as f:
            f.write(log_entry)


def create_regime_classifier(**kwargs) -> MarketRegimeClassifier:
    return MarketRegimeClassifier(**kwargs)
