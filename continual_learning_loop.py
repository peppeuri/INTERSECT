"""
INTERSECT — Autonomous AI Trading System
continual_learning_loop.py: Orchestrates trading, inference, updates, retraining schedules
"""

import time
import threading
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional
import os
import glob


class ContinualLearningLoop:
    def __init__(
        self,
        ppo_agent,
        sac_agent,
        regime_classifier,
        ensemble,
        env_factory,
        bitget_client,
        self_healing_daemon,
        meta_optimizer,
        trade_db,
        loop_interval: int = 900,
        online_update_interval: int = 1,
        batch_retrain_interval: int = 86400,
        regime_retrain_interval: int = 7 * 86400,
        eval_interval: int = 7 * 86400,
        full_retrain_interval: int = 30 * 86400,
        archive_interval: int = 90 * 86400,
    ):
        self.ppo = ppo_agent
        self.sac = sac_agent
        self.regime = regime_classifier
        self.ensemble = ensemble
        self.env_factory = env_factory
        self.client = bitget_client
        self.healing = self_healing_daemon
        self.meta = meta_optimizer
        self.db = trade_db

        self.loop_interval = loop_interval
        self.online_update_interval = online_update_interval
        self.batch_retrain_interval = batch_retrain_interval
        self.regime_retrain_interval = regime_retrain_interval
        self.eval_interval = eval_interval
        self.full_retrain_interval = full_retrain_interval
        self.archive_interval = archive_interval

        self.last_online = 0
        self.last_batch = 0
        self.last_regime = 0
        self.last_eval = 0
        self.last_full = 0
        self.last_archive = 0

        self.running = False
        self.thread = None
        self.equity = 1000.0
        self.trades_today = 0
        self.last_trade_date = datetime.now().date()

        os.makedirs('logs', exist_ok=True)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._main_loop, daemon=True)
        self.thread.start()
        self._log('Continual learning loop started')

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=30)
        self._log('Continual learning loop stopped')

    def _main_loop(self):
        while self.running:
            try:
                start = time.time()
                self._process_bar()

                now = time.time()
                if now - self.last_online >= self.online_update_interval * 60:
                    if len(self.ppo.recent_buffer) >= 16:
                        self.ppo.train_online()
                        self._log('PPO online update')
                    self.last_online = now

                if now - self.last_batch >= self.batch_retrain_interval:
                    if len(self.ppo.replay_buffer) >= self.ppo.batch_size:
                        self.ppo.train_batch()
                        self.sac.train()
                        self.ppo.save_checkpoint(f'checkpoints/agent_ppo_{int(now)}.pt')
                        self.sac.save_checkpoint(f'checkpoints/agent_sac_{int(now)}.pt')
                        self._log('Batch retrain complete')
                    self.last_batch = now

                if now - self.last_regime >= self.regime_retrain_interval:
                    candles = self.client.get_candles(limit=5000)
                    if candles:
                        df = self._candles_to_df(candles)
                        self.regime.fit(df)
                        self._log('Regime retrained')
                    self.last_regime = now

                if now - self.last_eval >= self.eval_interval:
                    self.meta.evaluate_and_optimize()
                    self.last_eval = now

                if now - self.last_full >= self.full_retrain_interval:
                    self._full_retrain()
                    self.last_full = now

                if now - self.last_archive >= self.archive_interval:
                    self._archive()
                    self.last_archive = now

                self._check_daily_reset()

                elapsed = time.time() - start
                time.sleep(max(0, self.loop_interval - elapsed))

            except Exception as e:
                self._log(f'Loop error: {e}')
                time.sleep(60)

    def _process_bar(self):
        candles = self.client.get_candles(limit=100)
        if not candles:
            return

        df = self._candles_to_df(candles)
        price = candles[-1]['c']
        regime, rprobs = self.regime.get_current_regime(df)

        env = self.env_factory(df=df.tail(500))
        env.set_regime_probs(rprobs[-min(len(rprobs), 500):])
        obs = env._get_obs()

        pa, pp, pv, _ = self.ppo.predict(obs)
        sa, sp, sv = self.sac.predict(obs)

        action, probs, conf, info = self.ensemble.vote({'ppo': (pa, pp, pv), 'sac': (sa, sp, sv)})

        if info.get('action') == 'no_trade':
            self._log(f'No trade: {info.get("reason")}')
            return

        sf = info.get('size_factor', 1.0)
        if action not in (0, None):
            self._execute(action, price, sf, df, regime, info)

        self.ppo.update_rolling_metrics(env.equity)
        self.sac.update_rolling_metrics(env.equity)
        self.equity = env.equity

    def _execute(self, action, price, sf, df, regime, info):
        side_map = {
            0: ('close', 'close', 0),
            1: ('buy', 'open', 1), 2: ('buy', 'open', 2), 3: ('buy', 'open', 3),
            4: ('sell', 'open', 1), 5: ('sell', 'open', 2), 6: ('sell', 'open', 3),
        }
        if action not in side_map:
            return

        side, ts, lev = side_map[action]

        if action == 0:
            self._log('Close position')
            pos = self.client.get_position()
            if pos:
                self.client.close_position(pos)
                self.db.record_trade({
                    'timestamp': datetime.now().isoformat(), 'action': action, 'side': 'close',
                    'leverage': 0, 'size': 0, 'price': price, 'sl': 0, 'tp': 0,
                    'regime': int(regime), 'confidence': info.get('confidence', 0),
                    'ensemble_weights': info.get('weights', {}),
                })
                self.trades_today += 1
            return

        size = max(round(((20.0 * sf * lev) / price) / 0.01) * 0.01, 0.01)
        sl = round(price * (0.98 if side == 'buy' else 1.02), 2)
        tp = round(price * (1.03 if side == 'buy' else 0.97), 2)

        self.client.set_leverage(lev, 'long' if side == 'buy' else 'short')
        resp = self.client.place_order(side, ts, size, price, lev)

        if resp.get('code') == '00000':
            oid = resp['data'].get('orderId')
            self._log(f'Trade: {side} {size} ETH @ {price:.2f} (lev={lev}x) order={oid}')
            ok = self.client.place_sl_tp('long' if side == 'buy' else 'short', sl, tp)
            if not ok:
                self._log('SL/TP failed, closing')
                pos = self.client.get_position()
                if pos:
                    self.client.close_position(pos)
            self.db.record_trade({
                'timestamp': datetime.now().isoformat(), 'action': action, 'side': side,
                'leverage': lev, 'size': size, 'price': price, 'sl': sl, 'tp': tp,
                'regime': int(regime), 'confidence': info.get('confidence', 0),
                'ensemble_weights': info.get('weights', {}),
            })
            self.trades_today += 1
        else:
            self._log(f'Trade failed: {resp}')

    def _full_retrain(self):
        self._log('Full retrain on historical data...')
        candles = self.client.get_candles(limit=20000)
        if not candles:
            return
        df = self._candles_to_df(candles)
        env = self.env_factory(df=df)
        for _ in range(50):
            obs, _ = env.reset()
            self.ppo.reset_lstm()
            done = False
            while not done:
                action, _, _, _ = self.ppo.predict(obs)
                next_obs, reward, done, truncated, _ = env.step(action)
                self.ppo.store_transition(obs, action, reward, next_obs, done, 0, 0)
                obs = next_obs
                if truncated:
                    break
        self.ppo.train_batch()
        self.ppo.save_checkpoint(f'checkpoints/agent_ppo_full_{int(time.time())}.pt')
        self._log('Full retrain complete')

    def _archive(self):
        for pattern in ('checkpoints/agent_ppo_*.pt', 'checkpoints/agent_sac_*.pt'):
            ckpts = sorted(glob.glob(pattern), key=os.path.getmtime)
            if len(ckpts) > 20:
                for old in ckpts[:-20]:
                    os.remove(old)
                    self._log(f'Archived: {os.path.basename(old)}')

    def _check_daily_reset(self):
        today = datetime.now().date()
        if today != self.last_trade_date:
            self.trades_today = 0
            self.last_trade_date = today
            self._log(f'New day: {today}')

    def _candles_to_df(self, candles):
        df = pd.DataFrame(candles)
        df = df.rename(columns={'ts': 'datetime', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
        return df.set_index('datetime').sort_index()

    def get_status(self) -> Dict:
        ppo_m = self.ppo.get_metrics()
        sac_m = self.sac.get_metrics()

        def safe(v, default=0.0):
            import math
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                return default
            return v

        return {
            'ppo_sharpe': safe(ppo_m.get('rolling_sharpe', 0)),
            'sac_sharpe': safe(sac_m.get('rolling_sharpe', 0)),
            'exploration': ppo_m.get('exploration_mode', False),
            'total_steps': safe(ppo_m.get('total_steps', 0)),
            'buffer_size': safe(ppo_m.get('buffer_size', 0)),
            'equity': safe(self.equity, 0),
            'trades_today': self.trades_today,
            'n_regimes': self.regime.n_states,
            'best_sharpe': safe(self.meta.best_sharpe),
        }

    def _log(self, msg: str):
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        entry = f'{ts} | LOOP | {msg}\n'
        print(entry.strip())
        with open('logs/continual_learning.log', 'a') as f:
            f.write(entry)


def create_continual_learning_loop(**kwargs) -> ContinualLearningLoop:
    return ContinualLearningLoop(**kwargs)
