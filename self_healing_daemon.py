"""
INTERSECT — Autonomous AI Trading System
self_healing_daemon.py: Monitors agents, activates exploration, detects overfit & drift
"""

import threading
import time
import os
import numpy as np
from collections import deque


class SelfHealingDaemon:
    def __init__(
        self,
        ppo_agent,
        sac_agent,
        regime_classifier,
        env_factory,
        check_interval: int = 300,
        exploration_sharpe_threshold: float = -0.5,
        stuck_equity_hours: int = 72,
        overfit_window_days: int = 14,
        overfit_threshold: float = 2.0,
        kl_threshold: float = 0.5,
        cache_warm_interval: int = 6 * 3600,
    ):
        self.ppo_agent = ppo_agent
        self.sac_agent = sac_agent
        self.regime_classifier = regime_classifier
        self.env_factory = env_factory
        self.check_interval = check_interval
        self.exploration_sharpe_threshold = exploration_sharpe_threshold
        self.stuck_equity_hours = stuck_equity_hours
        self.overfit_window_days = overfit_window_days
        self.overfit_threshold = overfit_threshold
        self.kl_threshold = kl_threshold
        self.cache_warm_interval = cache_warm_interval

        self.running = False
        self.thread = None
        self.last_equity = 1000.0
        self.last_equity_time = time.time()
        self.last_cache_warm = 0
        self.startup_time = time.time()
        self.feature_history = deque(maxlen=1000)
        self.backtest_performance = deque(maxlen=100)
        self.live_performance = deque(maxlen=100)

        os.makedirs('logs', exist_ok=True)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        self._log('Self-healing daemon started')

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=10)
        self._log('Self-healing daemon stopped')

    def record_equity(self, equity: float):
        self.last_equity = equity
        self.last_equity_time = time.time()

    def record_features(self, features: np.ndarray):
        self.feature_history.append(features)

    def record_backtest_perf(self, sharpe: float):
        self.backtest_performance.append(sharpe)

    def record_live_perf(self, sharpe: float):
        self.live_performance.append(sharpe)

    def _monitor_loop(self):
        while self.running:
            try:
                self._check_agent_health()
                self._check_performance_degradation()
                self._check_equity_stuck()
                self._check_overfit()
                self._check_feature_drift()
                self._cache_warm()
            except Exception as e:
                self._log(f'Loop error: {e}')
            time.sleep(self.check_interval)

    def _check_agent_health(self):
        try:
            grace = max(self.check_interval * 3, 1800)
            if time.time() - self.startup_time < grace:
                return
            ppo = self.ppo_agent.get_metrics()
            if ppo.get('total_steps', 0) == 0:
                self._log('PPO appears dead, recovering')
                self._recover_agent('ppo')
        except Exception as e:
            self._log(f'Health error: {e}')
            self._recover_agent('ppo')

    def _recover_agent(self, name: str):
        import glob
        ckpts = sorted(glob.glob(f'checkpoints/agent_{name}_*.pt'), key=os.path.getmtime)
        if ckpts:
            path = ckpts[-1]
            if name == 'ppo':
                self.ppo_agent.load_checkpoint(path, strict=False)
            else:
                self.sac_agent.load_checkpoint(path, strict=False)
            self._log(f'Recovered {name} from {os.path.basename(path)} (partial load)')
            self.ppo_agent._activate_exploration_mode()
        else:
            self._log(f'No checkpoints found for {name} — starting fresh with exploration')
            self.ppo_agent._activate_exploration_mode()

    def _check_performance_degradation(self):
        if len(self.ppo_agent.rolling_sharpe) >= 7:
            recent = np.mean(list(self.ppo_agent.rolling_sharpe)[-7:])
            if recent < self.exploration_sharpe_threshold and not self.ppo_agent.exploration_mode:
                self._log(f'7d Sharpe={recent:.4f} < {self.exploration_sharpe_threshold}. Activating exploration.')
                self.ppo_agent._activate_exploration_mode()

    def _check_equity_stuck(self):
        hours = (time.time() - self.last_equity_time) / 3600
        if hours > self.stuck_equity_hours:
            self._log(f'Equity stuck {hours:.1f}h. Forcing exploration.')
            self.ppo_agent._activate_exploration_mode()
            self.last_equity_time = time.time()

    def _check_overfit(self):
        if len(self.backtest_performance) >= 10 and len(self.live_performance) >= 10:
            bt = np.mean(list(self.backtest_performance)[-self.overfit_window_days:])
            lv = np.mean(list(self.live_performance)[-self.overfit_window_days:])
            if bt > 0 and lv / bt < (1 / self.overfit_threshold):
                self._log(f'Overfit: bt={bt:.4f} live={lv:.4f}. Increasing dropout.')
                current = getattr(self.ppo_agent.policy_net.lstm, 'dropout', 0.2)
                new = min(current + 0.1, 0.5)
                self.ppo_agent.policy_net.lstm.dropout = new
                self._log(f'Dropout {current:.2f} -> {new:.2f}')

    def _check_feature_drift(self):
        if len(self.feature_history) < 100:
            return
        recent = np.array(list(self.feature_history)[-100:])
        older = np.array(list(self.feature_history)[:100]) if len(self.feature_history) >= 200 else recent
        rm, om = np.mean(recent, axis=0), np.mean(older, axis=0)
        rs, os_ = np.std(recent, axis=0) + 1e-8, np.std(older, axis=0) + 1e-8
        kl = 0.5 * np.sum(np.log(rs / os_) + (os_**2 + (om - rm)**2) / rs**2 - 1)
        if kl > self.kl_threshold:
            self._log(f'Feature drift KL={kl:.4f}. Triggering retrain.')
            self.ppo_agent.replay_buffer.clear()
            self.ppo_agent._activate_exploration_mode()
            self.regime_classifier.last_retrain = 0

    def _cache_warm(self):
        if time.time() - self.last_cache_warm > self.cache_warm_interval:
            self._log('Cache warming...')
            env = self.env_factory()
            obs, _ = env.reset()
            for _ in range(2000):
                action, _, _, _ = self.ppo_agent.predict(obs, deterministic=True)
                obs, _, done, truncated, _ = env.step(action)
                if done or truncated:
                    obs, _ = env.reset()
            self.ppo_agent.reset_lstm()
            self.last_cache_warm = time.time()
            env.close()
            self._log('Cache warm complete')

    def _log(self, msg: str):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f'{timestamp} | HEALING | {msg}\n'
        print(log_entry.strip())
        with open('logs/self_healing.log', 'a') as f:
            f.write(log_entry)


def create_self_healing_daemon(**kwargs):
    return SelfHealingDaemon(**kwargs)
