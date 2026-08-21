#!/usr/bin/env python3
"""
INTERSECT — Autonomous AI Trading System
main.py: Entry point — starts per-symbol trading bot threads + FastAPI web server
"""

import os
import sys
import time
import glob
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


def _candles_to_df(candles):
    df = pd.DataFrame(candles)
    df = df.rename(columns={'ts': 'datetime', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
    return df.set_index('datetime').sort_index()


def create_env_factory(bitget_client):
    def factory(df=None):
        if df is None:
            candles = bitget_client.get_candles(limit=500)
            if not candles:
                import numpy as np
                df = pd.DataFrame({'open': [3000], 'high': [3010], 'low': [2990],
                                   'close': [3005], 'volume': [100]})
            else:
                df = _candles_to_df(candles)
        return make_env(df, initial_capital=config.INITIAL_CAPITAL)
    return factory


def _clean_incompatible_checkpoints(current_state_dim: int, symbol_lower: str):
    import torch
    count = 0
    for pattern in (f'checkpoints/{symbol_lower}/agent_ppo_*.pt', f'checkpoints/agent_ppo_{symbol_lower}_*.pt'):
        for ckpt in glob.glob(pattern):
            try:
                data = torch.load(ckpt, map_location='cpu', weights_only=False)
                arch = data.get('architecture', {})
                ckpt_state_dim = arch.get('state_dim')
                if ckpt_state_dim is not None and ckpt_state_dim != current_state_dim:
                    dst = ckpt.replace('checkpoints/', 'backups/incompat_')
                    os.rename(ckpt, dst)
                    log.warning(f'Incompatible checkpoint moved: {ckpt}')
                    count += 1
            except Exception:
                pass
    for pattern in (f'checkpoints/{symbol_lower}/agent_sac_*.pt', f'checkpoints/agent_sac_{symbol_lower}_*.pt'):
        for ckpt in glob.glob(pattern):
            try:
                data = torch.load(ckpt, map_location='cpu', weights_only=False)
                arch = data.get('architecture', {})
                ckpt_state_dim = arch.get('state_dim')
                if ckpt_state_dim is not None and ckpt_state_dim != current_state_dim:
                    dst = ckpt.replace('checkpoints/', 'backups/incompat_sac_')
                    os.rename(ckpt, dst)
                    log.warning(f'Incompatible SAC checkpoint moved: {ckpt}')
                    count += 1
            except Exception:
                pass
    if count:
        log.info(f'Cleaned {count} incompatible checkpoints')


def _create_symbol_stack(symbol, connected, shared_df=None):
    sym_lower = symbol.lower()
    log.info(f'Creating stack for {symbol}...')

    bitget = create_bitget_client(
        api_key=config.API_KEY, api_secret=config.API_SECRET,
        passphrase=config.PASSPHRASE, base_url=config.BASE_URL,
        demo=config.DEMO, symbol=symbol,
        product_type=config.PRODUCT_TYPE, margin_coin=config.MARGIN_COIN,
        margin_mode=config.MARGIN_MODE,
    )

    if connected:
        candles = bitget.get_candles(limit=1000)
        df = _candles_to_df(candles) if candles else shared_df
    else:
        df = shared_df

    if df is None:
        log.warning(f'{symbol}: no data available, using synthetic')
        import numpy as np
        dates = pd.date_range(end=pd.Timestamp.now(), periods=1000, freq='15min')
        r = np.random.RandomState(42 if sym_lower == 'ethusdt' else 43)
        base = 3000 if 'ETH' in symbol else 150
        synthetic = []
        for ts in dates:
            o = base + r.randn() * (base * 0.005)
            h = o + abs(r.randn() * (base * 0.003))
            l = o - abs(r.randn() * (base * 0.003))
            c = (h + l) / 2 + r.randn()
            synthetic.append({'datetime': ts, 'open': o, 'high': h, 'low': l, 'close': c, 'volume': r.rand() * 1000})
        df = pd.DataFrame(synthetic).set_index('datetime')

    sample_env = make_env(df[-500:], initial_capital=config.INITIAL_CAPITAL)
    state_dim = sample_env.observation_space.shape[0]
    action_dim = sample_env.action_space.n

    log.info(f'{symbol}: state_dim={state_dim}, action_dim={action_dim}')

    _clean_incompatible_checkpoints(state_dim, sym_lower)

    ppo = PPOAgent(state_dim=state_dim, action_dim=action_dim, lstm_hidden=128,
                   lr=3e-4, entropy_coef=0.01, dropout=0.2, device='cpu')
    sac = SACAgent(state_dim=state_dim, action_dim=action_dim, hidden_dim=256, device='cpu')

    regime_clf = create_regime_classifier(n_states=6, max_states=8)

    for name, agent in [('ppo', ppo), ('sac', sac)]:
        ckpts = sorted(glob.glob(f'checkpoints/{sym_lower}/agent_{name}_*.pt'), key=os.path.getmtime)
        if not ckpts:
            ckpts = sorted(glob.glob(f'checkpoints/agent_{name}_{sym_lower}_*.pt'), key=os.path.getmtime)
        if ckpts:
            agent.load_checkpoint(ckpts[-1], strict=False)
            log.info(f'{symbol}: loaded {name}: {os.path.basename(ckpts[-1])}')
        else:
            log.warning(f'{symbol}: no {name} checkpoint — starting fresh')

    regime_clf.load(path=f'checkpoints/{sym_lower}/regime')

    ensemble = EnsembleVoting(min_sharpe_threshold=0.2, boost_sharpe_threshold=1.0)

    os.makedirs(f'checkpoints/{sym_lower}', exist_ok=True)
    trade_db = create_trade_database(
        db_path=f'checkpoints/{sym_lower}/metrics.db',
        experiences_path=f'checkpoints/{sym_lower}/experiences.npy',
        state_dim=state_dim,
    )

    if connected:
        env_factory = create_env_factory(bitget)
    else:
        def env_factory(df_local=df, _df=df):
            return make_env(df_local, initial_capital=config.INITIAL_CAPITAL)

    healing = create_self_healing_daemon(
        ppo_agent=ppo, sac_agent=sac, regime_classifier=regime_clf,
        env_factory=env_factory,
    )

    meta = MetaOptimizerAgent(
        ppo_agent=ppo, sac_agent=sac, env_factory=env_factory,
        db_path=f'checkpoints/{sym_lower}/agent_evolution.db',
    )

    loop = create_continual_learning_loop(
        ppo_agent=ppo, sac_agent=sac, regime_classifier=regime_clf,
        ensemble=ensemble, env_factory=env_factory, bitget_client=bitget,
        self_healing_daemon=healing, meta_optimizer=meta, trade_db=trade_db,
        symbol=symbol,
    )

    return {
        'symbol': symbol, 'bitget': bitget, 'ppo': ppo, 'sac': sac,
        'regime_clf': regime_clf, 'ensemble': ensemble, 'trade_db': trade_db,
        'env_factory': env_factory, 'healing': healing, 'meta': meta, 'loop': loop,
    }


def main():
    symbols = config.SYMBOLS
    log.info('═' * 60)
    log.info('  INTERSECT — Autonomous AI Trading System')
    log.info(f'  Symbols: {", ".join(symbols)} | Bitget USDT-Futures | {config.TIMEFRAME}')
    log.info(f'  Demo: {config.DEMO} | Initial Capital: ${config.INITIAL_CAPITAL}')
    log.info('═' * 60)

    test_client = create_bitget_client(
        api_key=config.API_KEY, api_secret=config.API_SECRET,
        passphrase=config.PASSPHRASE, base_url=config.BASE_URL,
        demo=config.DEMO, symbol=symbols[0],
        product_type=config.PRODUCT_TYPE, margin_coin=config.MARGIN_COIN,
        margin_mode=config.MARGIN_MODE,
    )
    connected = test_client.test_connection()

    shared_df = None
    if connected:
        candles = test_client.get_candles(limit=1000)
        if candles:
            shared_df = _candles_to_df(candles)
    else:
        log.warning('Bitget not connected — using synthetic data.')

    stacks = {}
    for sym in symbols:
        stacks[sym] = _create_symbol_stack(sym, connected, shared_df)

    for sym, s in stacks.items():
        if connected:
            s['loop'].start()
            s['healing'].start()
            log.info(f'{sym}: bot started')
        else:
            log.warning(f'{sym}: bot not started (no connection)')

    log.info('Launching web dashboard on port 8080...')

    from app import create_app
    app = create_app(stacks, config)

    def shutdown(sig, frame):
        log.info('Shutdown signal received')
        for sym, s in stacks.items():
            s['loop'].stop()
            s['healing'].stop()
            s['trade_db'].flush()
            s['ppo'].save_checkpoint(f'checkpoints/{sym.lower()}/agent_ppo_final_{int(time.time())}.pt')
            s['sac'].save_checkpoint(f'checkpoints/{sym.lower()}/agent_sac_final_{int(time.time())}.pt')
            s['regime_clf'].save(path=f'checkpoints/{sym.lower()}/regime')
            log.info(f'{sym}: saved')
        log.info('Shutdown complete')
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    uvicorn.run(app, host='0.0.0.0', port=8080, log_level='info')


if __name__ == '__main__':
    main()
