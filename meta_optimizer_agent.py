"""
INTERSECT — Autonomous AI Trading System
meta_optimizer_agent.py: CMA-ES meta-optimizer with paper trading validation + evolution DB
"""

import numpy as np
import json
import os
import time
import sqlite3
from cma import CMAEvolutionStrategy
from typing import Dict, List, Tuple, Optional, Any
import threading
from datetime import datetime


class MetaOptimizerAgent:
    def __init__(
        self,
        ppo_agent,
        sac_agent,
        env_factory,
        db_path: str = 'agent_evolution.db',
        eval_interval_days: int = 7,
        paper_trading_days: int = 7,
        degradation_threshold: float = 0.10,
        cma_population: int = 8,
        cma_sigma: float = 0.3,
        param_bounds: Dict = None,
    ):
        self.ppo_agent = ppo_agent
        self.sac_agent = sac_agent
        self.env_factory = env_factory
        self.db_path = db_path
        self.eval_interval_days = eval_interval_days
        self.paper_trading_days = paper_trading_days
        self.degradation_threshold = degradation_threshold
        self.cma_population = cma_population
        self.cma_sigma = cma_sigma

        self.param_bounds = param_bounds or {
            'lr': (1e-5, 1e-3),
            'entropy_coef': (0.005, 0.05),
            'gae_lambda': (0.9, 0.99),
            'lstm_hidden': (64, 256),
            'dropout': (0.1, 0.3),
        }

        self.best_sharpe = -np.inf
        self.best_agent_type = 'ppo'
        self.best_params = {}
        self.current_paper_agent = None
        self.paper_agent_type = None
        self.paper_start_time = None
        self.paper_expected_sharpe = None
        self.last_eval = 0
        self.eval_count = 0

        self._init_db()
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        self._load_best_params()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS agent_evolution (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, agent_type TEXT, params TEXT,
                sharpe REAL, max_drawdown REAL, win_rate REAL,
                profit_factor REAL, n_trades INTEGER,
                status TEXT, promoted INTEGER DEFAULT 0
            )
        ''')
        conn.commit()
        conn.close()

    def _load_best_params(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            SELECT agent_type, params, sharpe FROM agent_evolution
            WHERE promoted = 1 ORDER BY sharpe DESC LIMIT 1
        ''')
        row = c.fetchone()
        conn.close()
        if row:
            self.best_agent_type = row[0]
            self.best_params = json.loads(row[1])
            self.best_sharpe = row[2]
            self._log(f'Loaded best params: {self.best_agent_type} with Sharpe={self.best_sharpe:.4f}')

    def evaluate_and_optimize(self, force: bool = False):
        current_time = time.time()
        if not force and current_time - self.last_eval < self.eval_interval_days * 86400:
            return

        ppo_sharpe = self.ppo_agent.get_metrics().get('rolling_sharpe', 0)
        sac_sharpe = self.sac_agent.get_metrics().get('rolling_sharpe', 0)
        current_best = max(ppo_sharpe, sac_sharpe)
        current_type = 'ppo' if ppo_sharpe >= sac_sharpe else 'sac'

        self._log(f'Evaluation: PPO Sharpe={ppo_sharpe:.4f}, SAC Sharpe={sac_sharpe:.4f}')

        if current_best > self.best_sharpe:
            self.best_sharpe = current_best
            self.best_agent_type = current_type
            self._save_best_params(current_type, current_best)

        degradation = (self.best_sharpe - current_best) / abs(self.best_sharpe) if self.best_sharpe != 0 else 1.0
        if degradation > self.degradation_threshold or force:
            self._log(f'Degradation: {degradation:.2%}. Starting CMA-ES.')
            self._run_cma_es()

        self.last_eval = current_time
        self.eval_count += 1

    def _run_cma_es(self):
        def objective(params):
            lr, ec, gae, hidden, drop = params
            agent = self._create_test_agent(lr, ec, gae, int(hidden), drop)
            return -self._evaluate_agent(agent)

        x0 = [
            self.ppo_agent.lr, self.ppo_agent.entropy_coef,
            self.ppo_agent.gae_lambda,
            self.ppo_agent.policy_net.lstm.hidden_size,
            getattr(self.ppo_agent.policy_net.lstm, 'dropout', 0.2)
        ]
        bounds = [
            self.param_bounds['lr'], self.param_bounds['entropy_coef'],
            self.param_bounds['gae_lambda'], self.param_bounds['lstm_hidden'],
            self.param_bounds['dropout']
        ]

        es = CMAEvolutionStrategy(x0, self.cma_sigma, {
            'popsize': self.cma_population,
            'bounds': [[b[0] for b in bounds], [b[1] for b in bounds]],
            'maxiter': 10, 'seed': int(time.time()) % 10000,
        })

        best_sharpe = -np.inf
        best_params = None
        while not es.stop():
            solutions = es.ask()
            fitness = []
            for sol in solutions:
                f = objective(sol)
                fitness.append(f)
                if -f > best_sharpe:
                    best_sharpe = -f
                    best_params = sol
            es.tell(solutions, fitness)

        if best_params is not None and best_sharpe > self.best_sharpe * (1 - self.degradation_threshold):
            self._launch_paper_trading(best_params, best_sharpe)

    def _create_test_agent(self, lr, ec, gae, lstm_hidden, dropout):
        from ppo_agent import PPOAgent
        agent = PPOAgent(
            state_dim=self.ppo_agent.state_dim, action_dim=self.ppo_agent.action_dim,
            lstm_hidden=lstm_hidden, lr=lr, entropy_coef=ec,
            gae_lambda=gae, dropout=dropout, device='cpu',
        )
        agent.policy_net.load_state_dict(self.ppo_agent.policy_net.state_dict())
        return agent

    def _evaluate_agent(self, agent) -> float:
        env = self.env_factory()
        total = 0
        for _ in range(3):
            obs, _ = env.reset()
            agent.reset_lstm()
            terminated = truncated = False
            ep_r = 0
            while not terminated and not truncated:
                action, _, _, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_r += reward
            total += ep_r
        env.close()
        return total / 3

    def _launch_paper_trading(self, params: np.ndarray, expected_sharpe: float):
        lr, ec, gae, hidden, drop = params
        self._log(f'Paper trading: lr={lr:.2e} entropy={ec:.4f} gae={gae:.3f} hidden={int(hidden)} dropout={drop:.3f}')
        agent = self._create_test_agent(lr, ec, gae, int(hidden), drop)
        self.current_paper_agent = agent
        self.paper_agent_type = 'ppo'
        self.paper_start_time = time.time()
        self.paper_expected_sharpe = expected_sharpe
        threading.Thread(target=self._paper_trading_loop, daemon=True).start()

    def _paper_trading_loop(self):
        env = self.env_factory()
        for day in range(self.paper_trading_days):
            obs, _ = env.reset()
            self.current_paper_agent.reset_lstm()
            done = truncated = False
            while not done and not truncated:
                action, _, _, _ = self.current_paper_agent.predict(obs, deterministic=True)
                next_obs, reward, done, truncated, info = env.step(action)
                self.current_paper_agent.store_transition(obs, action, reward, next_obs, done or truncated, 0, 0)
                obs = next_obs
            self._log(f'Paper day {day+1}/{self.paper_trading_days} complete. Equity: {info.get("equity", 0):.2f}')
        env.close()

        paper_sharpe = self.current_paper_agent.get_metrics()['rolling_sharpe']
        self._log(f'Paper complete. Sharpe: {paper_sharpe:.4f} vs Expected: {self.paper_expected_sharpe:.4f}')
        if paper_sharpe > self.best_sharpe * 0.95:
            self._promote_paper_agent()
        else:
            self._archive_paper_agent(paper_sharpe)

    def _promote_paper_agent(self):
        self._log('Promoting paper agent to LIVE')
        self.ppo_agent.policy_net.load_state_dict(self.current_paper_agent.policy_net.state_dict())
        self.ppo_agent.optimizer.load_state_dict(self.current_paper_agent.optimizer.state_dict())
        self.ppo_agent.lr = self.current_paper_agent.lr
        self.ppo_agent.entropy_coef = self.current_paper_agent.entropy_coef
        self.best_sharpe = self.current_paper_agent.get_metrics()['rolling_sharpe']
        self._save_best_params('ppo', self.best_sharpe)
        self.current_paper_agent = None

    def _archive_paper_agent(self, sharpe: float):
        self._log(f'Archiving paper agent (Sharpe={sharpe:.4f})')
        self._save_agent_to_db('ppo', {
            'lr': self.current_paper_agent.lr,
            'entropy_coef': self.current_paper_agent.entropy_coef,
            'gae_lambda': self.current_paper_agent.gae_lambda,
            'lstm_hidden': self.current_paper_agent.policy_net.lstm.hidden_size,
            'dropout': getattr(self.current_paper_agent.policy_net.lstm, 'dropout', 0.2)
        }, sharpe, status='archived_paper')
        self.current_paper_agent = None

    def _save_best_params(self, agent_type: str, sharpe: float):
        params = {'lr': self.ppo_agent.lr, 'entropy_coef': self.ppo_agent.entropy_coef,
                  'gae_lambda': self.ppo_agent.gae_lambda,
                  'lstm_hidden': self.ppo_agent.policy_net.lstm.hidden_size,
                  'dropout': getattr(self.ppo_agent.policy_net.lstm, 'dropout', 0.2)}
        self._save_agent_to_db(agent_type, params, sharpe, status='promoted')

    def _save_agent_to_db(self, agent_type: str, params: Dict, sharpe: float, status: str = 'evaluated'):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT INTO agent_evolution (timestamp, agent_type, params, sharpe, status)
            VALUES (?, ?, ?, ?, ?)
        ''', (datetime.now().isoformat(), agent_type, json.dumps(params), sharpe, status))
        conn.commit()
        conn.close()

    def get_best_agent(self):
        return self.best_agent_type, self.best_sharpe

    def _log(self, msg: str):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f'{timestamp} | META | {msg}\n'
        print(log_entry.strip())
        with open('logs/meta_optimizer.log', 'a') as f:
            f.write(log_entry)
