"""
INTERSECT — Autonomous AI Trading System
trade_database.py: SQLite trades + memory-mapped numpy experience replay
"""

import numpy as np
import sqlite3
import json
import os
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import deque


class TradeDatabase:
    def __init__(
        self,
        db_path: str = 'metrics.db',
        experiences_path: str = 'experiences.npy',
        max_experiences: int = 100000,
        state_dim: int = 35,
    ):
        self.db_path = db_path
        self.experiences_path = experiences_path
        self.max_experiences = max_experiences
        self.state_dim = state_dim
        exp_dim = state_dim + 1 + 1 + state_dim + 1
        self.experience_dim = exp_dim

        self._init_sqlite()
        self._init_experiences_array()
        self._lock = threading.Lock()
        self.experience_count = 0

        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('logs', exist_ok=True)

    def _init_sqlite(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, action INTEGER, side TEXT,
                leverage INTEGER, size REAL, price REAL,
                sl REAL, tp REAL, regime INTEGER,
                confidence REAL, ensemble_weights TEXT,
                pnl REAL DEFAULT 0, exit_reason TEXT,
                exit_price REAL, exit_timestamp TEXT
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS daily_metrics (
                date TEXT PRIMARY KEY, equity REAL, drawdown REAL,
                sharpe REAL, win_rate REAL, profit_factor REAL,
                n_trades INTEGER, pnl REAL
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS agent_metrics (
                timestamp TEXT, agent_type TEXT,
                rolling_sharpe REAL, entropy_coef REAL, lr REAL,
                exploration_mode INTEGER, buffer_size INTEGER,
                PRIMARY KEY (timestamp, agent_type)
            )
        ''')
        conn.commit()
        conn.close()

    def _init_experiences_array(self):
        shape = (self.max_experiences, self.experience_dim)
        if os.path.exists(self.experiences_path):
            self.experiences = np.lib.format.open_memmap(
                self.experiences_path, mode='r+', dtype=np.float32, shape=shape
            )
            self.experience_count = int(np.sum(~np.isnan(self.experiences[:, 0])))
        else:
            self.experiences = np.lib.format.open_memmap(
                self.experiences_path, mode='w+', dtype=np.float32, shape=shape
            )
            self.experiences[:] = np.nan
            self.experience_count = 0

    def record_trade(self, trade: Dict) -> int:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''
                INSERT INTO trades (timestamp, action, side, leverage, size, price, sl, tp, regime, confidence, ensemble_weights)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (trade['timestamp'], trade['action'], trade['side'], trade['leverage'],
                  trade['size'], trade['price'], trade['sl'], trade['tp'],
                  trade['regime'], trade.get('confidence', 0),
                  json.dumps(trade.get('ensemble_weights', {}))))
            conn.commit()
            tid = c.lastrowid
            conn.close()
            return tid

    def update_trade_exit(self, trade_id: int, pnl: float, exit_reason: str, exit_price: float):
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''
                UPDATE trades SET pnl=?, exit_reason=?, exit_price=?, exit_timestamp=?
                WHERE id=?
            ''', (pnl, exit_reason, exit_price, datetime.now().isoformat(), trade_id))
            conn.commit()
            conn.close()

    def get_recent_trades(self, n: int = 20) -> List[Dict]:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute('SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?', (n,))
            rows = [dict(r) for r in c.fetchall()]
            conn.close()
            return rows

    def record_experience(self, state, action, reward, next_state, done):
        with self._lock:
            idx = self.experience_count % self.max_experiences
            exp = np.concatenate([
                state.flatten(), [action], [reward],
                next_state.flatten(), [float(done)]
            ]).astype(np.float32)
            if len(exp) != self.experience_dim:
                return
            self.experiences[idx] = exp
            self.experience_count = min(self.experience_count + 1, self.max_experiences)
            if self.experience_count % 1000 == 0:
                self.experiences.flush()

    def get_experiences(self, n: int = None) -> np.ndarray:
        with self._lock:
            valid = min(self.experience_count, self.max_experiences)
            if valid == 0:
                return np.empty((0, self.experience_dim), dtype=np.float32)
            if n is None or n >= valid:
                data = self.experiences[:valid].copy()
            else:
                idx = np.random.choice(valid, n, replace=False)
                data = self.experiences[idx].copy()
            return data[~np.isnan(data).any(axis=1)]

    def record_daily_metrics(self, metrics: Dict):
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            date = datetime.now().date().isoformat()
            c.execute('''
                INSERT OR REPLACE INTO daily_metrics
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (date, metrics.get('equity', 0), metrics.get('drawdown', 0),
                  metrics.get('sharpe', 0), metrics.get('win_rate', 0),
                  metrics.get('profit_factor', 0), metrics.get('n_trades', 0),
                  metrics.get('pnl', 0)))
            conn.commit()
            conn.close()

    def get_rolling_metrics(self, window: int = 50) -> Dict:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute('SELECT * FROM trades WHERE pnl != 0 ORDER BY timestamp DESC LIMIT ?', (window,))
            rows = c.fetchall()
            conn.close()

            if not rows:
                return {}

            pnls = np.array([r['pnl'] for r in rows])
            wins = pnls[pnls > 0]
            losses = pnls[pnls < 0]
            wr = len(wins) / len(pnls) * 100 if len(pnls) > 0 else 0
            pf = abs(wins.sum() / losses.sum()) if len(losses) > 0 else float('inf')
            sharpe = (pnls.mean() / (pnls.std() + 1e-8)) * np.sqrt(252 * 96) if len(pnls) > 1 else 0

            peak = 0
            running = 0
            max_dd = 0
            for p in pnls:
                running += p
                peak = max(peak, running)
                max_dd = max(max_dd, (peak - running) / peak if peak > 0 else 0)

            return {
                'total_return': float(pnls.sum()), 'win_rate': float(wr),
                'profit_factor': float(pf), 'sharpe': float(sharpe),
                'max_drawdown': max_dd * 100, 'n_trades': len(pnls),
                'avg_win': float(wins.mean()) if len(wins) > 0 else 0,
                'avg_loss': float(losses.mean()) if len(losses) > 0 else 0,
            }

    def get_daily_pnl(self) -> float:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('SELECT pnl FROM daily_metrics WHERE date = ?', (datetime.now().date().isoformat(),))
            row = c.fetchone()
            conn.close()
            return row[0] if row else 0.0

    def get_monthly_pnl(self) -> float:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            start = datetime.now().replace(day=1).date().isoformat()
            c.execute('SELECT SUM(pnl) FROM daily_metrics WHERE date >= ?', (start,))
            row = c.fetchone()
            conn.close()
            return row[0] if row and row[0] else 0.0

    def backup(self, path: str = None):
        if path is None:
            path = f'backups/metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        import shutil
        shutil.copy2(self.db_path, path)
        shutil.copy2(self.experiences_path, path.replace('.db', '_experiences.npy'))

    def flush(self):
        with self._lock:
            self.experiences.flush()


def create_trade_database(**kwargs) -> TradeDatabase:
    return TradeDatabase(**kwargs)
