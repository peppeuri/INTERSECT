#!/usr/bin/env python3
"""
INTERSECT — Autonomous AI Trading System
main.py: Entry point — starts trading bot threads + FastAPI web server
"""

import os
import sys
import time
import signal
import logging
import threading
import uvicorn
import pandas as pd

from config import config
from rl_environment import make_env
from ppo_agent import PPOAgent
from sac_agent import SACAgent
from market_regime_classifier import create_regime_classifier
from meta_optimizer_agent import MetaOptimizerAgent
from ensemble_voting import EnsembleVoting
from self_healing_daemon import create_self_healing_daemon
from continual_learning_loop import create_continual_learning_loop
from trade_database import create_trade_database
from infrastructure import create_bitget_client

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-5s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('logs/main.log', encoding='utf-8')],
)
log = logging.getLogger('main')

for d in ['logs', 'checkpoints', 'models', 'backups']:
    os.makedirs(d, exist_ok=True)


def create_env_factory(bitget_client):
    def factory(df=None):
        if df is None:
            candles = bitget_client.get_candles(limit=500)
            if not candles:
                import numpy as np
                df = pd.DataFrame({'open': [3000], 'high': [3010], 'low': [2990],
                                   'close': [3005], 'volume': [100]})
            else:
                df = pd.DataFrame(candles)
                df = df.rename(columns={'ts': 'datetime', 'o': 'open', 'h': 'high',
                                        'l': 'low', 'c': 'close', 'v': 'volume'})
                df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
                df = df.set_index('datetime').sort_index()
        return make_env(df, initial_capital=config.INITIAL_CAPITAL)
    return factory


def main():
    log.info('═' * 60)
    log.info('  INTERSECT — Autonomous AI Trading System')
    log.info(f'  ETH-USDT | Bitget USDT-Futures | {config.TIMEFRAME}')
    log.info(f'  Demo: {config.DEMO} | Initial Capital: ${config.INITIAL_CAPITAL}')
    log.info('═' * 60)

    bitget = create_bitget_client(
        api_key=config.API_KEY, api_secret=config.API_SECRET,
        passphrase=config.PASSPHRASE, base_url=config.BASE_URL,
        demo=config.DEMO, symbol=config.SYMBOL,
        product_type=config.PRODUCT_TYPE, margin_coin=config.MARGIN_COIN,
        margin_mode=config.MARGIN_MODE,
    )

    connected = bitget.test_connection()

    if connected:
        candles = bitget.get_candles(limit=1000)
    else:
        log.warning('Bitget not connected — using synthetic data. Set API_KEY/API_SECRET env vars.')
        candles = None

    if not candles:
        log.info('Generating synthetic market data for dashboard...')
        import numpy as np
        dates = pd.date_range(end=pd.Timestamp.now(), periods=1000, freq='15min')
        base = 3000
        r = np.random.RandomState(42)
        synthetic = []
        for i, ts in enumerate(dates):
            o = base + r.randn() * 10
            h = o + abs(r.randn() * 5)
            l = o - abs(r.randn() * 5)
            c = (h + l) / 2 + r.randn()
            synthetic.append({'datetime': ts, 'o': o, 'h': h, 'l': l, 'c': c, 'v': r.rand() * 1000})
        df = pd.DataFrame(synthetic).set_index('datetime')
    else:
        df = pd.DataFrame(candles)
        df = df.rename(columns={'ts': 'datetime', 'o': 'open', 'h': 'high',
                                'l': 'low', 'c': 'close', 'v': 'volume'})
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
        df = df.set_index('datetime').sort_index()

    sample_env = make_env(df[-500:], initial_capital=config.INITIAL_CAPITAL)
    state_dim = sample_env.observation_space.shape[0]
    action_dim = sample_env.action_space.n

    log.info(f'State dim: {state_dim}, Action dim: {action_dim}')

    ppo = PPOAgent(state_dim=state_dim, action_dim=action_dim, lstm_hidden=128,
                   lr=3e-4, entropy_coef=0.01, dropout=0.2, device='cpu')
    sac = SACAgent(state_dim=state_dim, action_dim=action_dim, hidden_dim=256, device='cpu')

    regime_clf = create_regime_classifier(n_states=6, max_states=8)

    ensemble = EnsembleVoting(min_sharpe_threshold=0.2, boost_sharpe_threshold=1.0)

    trade_db = create_trade_database(state_dim=state_dim)

    if connected:
        env_factory = create_env_factory(bitget)
    else:
        def env_factory(df=None):
            return make_env(df if df is not None else df_local, initial_capital=config.INITIAL_CAPITAL)
        df_local = df

    healing = create_self_healing_daemon(
        ppo_agent=ppo, sac_agent=sac, regime_classifier=regime_clf,
        env_factory=env_factory,
    )

    meta = MetaOptimizerAgent(
        ppo_agent=ppo, sac_agent=sac, env_factory=env_factory,
    )

    loop = create_continual_learning_loop(
        ppo_agent=ppo, sac_agent=sac, regime_classifier=regime_clf,
        ensemble=ensemble, env_factory=env_factory, bitget_client=bitget,
        self_healing_daemon=healing, meta_optimizer=meta, trade_db=trade_db,
    )

    if connected:
        loop.start()
        healing.start()
        log.info('Bot agents started.')
    else:
        log.warning('Bot agents not started (no Bitget connection). Dashboard only.')

    log.info('Launching web dashboard on port 8080...')

    from app import create_app
    app = create_app(loop, trade_db, meta, ppo, sac, regime_clf, config)

    def shutdown(sig, frame):
        log.info('Shutdown signal received')
        loop.stop()
        healing.stop()
        trade_db.flush()
        ppo.save_checkpoint(f'checkpoints/agent_ppo_final_{int(time.time())}.pt')
        sac.save_checkpoint(f'checkpoints/agent_sac_final_{int(time.time())}.pt')
        regime_clf.save()
        log.info('Shutdown complete')
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    uvicorn.run(app, host='0.0.0.0', port=8080, log_level='info')


if __name__ == '__main__':
    main()
