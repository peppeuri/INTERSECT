"""
INTERSECT — Autonomous AI Trading System
config.py: Environment variables and strategy configuration loader
"""

import os
import json
import logging
from typing import Dict, Any

log = logging.getLogger('config')

_CONFIG_DEFAULTS = {
    'BB_PERIOD': 24,
    'BB_STD': 2.1,
    'ATR_PERIOD': 14,
    'RSI_PERIOD': 14,
    'EMA_FAST': 50,
    'EMA_SLOW': 200,
    'SL_PCT_BASE': 0.018,
    'TP_SL_RATIO': 1.5,
    'MAX_LEVERAGE': 3,
    'MAX_POSITION_PCT': 0.02,
    'MAX_DAILY_LOSS_PCT': 0.05,
    'MAX_CONSECUTIVE_LOSSES': 3,
    'COOLDOWN_HOURS': 2,
    'DRAWDOWN_SIZE_REDUCTION': 0.5,
    'DRAWDOWN_THRESHOLD': 0.15,
    'LOOP_SECONDS': 60,
    'OPTIMIZE_EVERY_N_TRADES': 100,
    'OPTIMIZE_EVERY_HOURS': 24,
    'MIN_WINRATE_LONG': 0.45,
    'MIN_WINRATE_SHORT_V3': 0.40,
    'MIN_WINRATE_SHORT_V4': 0.35,
    '_meta': {
        'sharpe': None,
        'profit_factor': None,
        'updated_at': None,
        'regime': None,
    },
}

CONFIG_PATH = 'strategy_config.json'


class Config:
    def __init__(self):
        self.API_KEY = os.environ.get('API_KEY', '')
        self.API_SECRET = os.environ.get('API_SECRET', '')
        self.PASSPHRASE = os.environ.get('PASSPHRASE', '')
        self.BASE_URL = os.environ.get('BASE_URL', 'https://api.bitget.com')
        self.DEMO = os.environ.get('DEMO', 'true').lower() == 'true'
        self.SYMBOL = os.environ.get('SYMBOL', 'ETHUSDT')
        self.PRODUCT_TYPE = os.environ.get('PRODUCT_TYPE', 'USDT-FUTURES')
        self.MARGIN_COIN = os.environ.get('MARGIN_COIN', 'USDT')
        self.MARGIN_MODE = os.environ.get('MARGIN_MODE', 'crossed')
        self.TIMEFRAME = os.environ.get('TIMEFRAME', '15m')
        self.INITIAL_CAPITAL = float(os.environ.get('INITIAL_CAPITAL', '1000.0'))
        self.CSV_PATH = os.environ.get('CSV_PATH', '')
        self.strategy = self._load_strategy_config()

    def _load_strategy_config(self) -> Dict[str, Any]:
        try:
            if not os.path.exists(CONFIG_PATH):
                with open(CONFIG_PATH, 'w') as f:
                    json.dump(_CONFIG_DEFAULTS, f, indent=4)
                return dict(_CONFIG_DEFAULTS)

            with open(CONFIG_PATH, 'r') as f:
                cfg = json.load(f)

            for k, default in _CONFIG_DEFAULTS.items():
                if k not in cfg or not isinstance(cfg[k], (int, float)) and k != '_meta':
                    cfg[k] = default

            return cfg

        except Exception as e:
            log.error(f'Config load error ({e}), using defaults.')
            return dict(_CONFIG_DEFAULTS)

    def save_strategy_config(self, cfg: Dict[str, Any]):
        with open(CONFIG_PATH, 'w') as f:
            json.dump(cfg, f, indent=4)
        self.strategy = cfg
        log.info('Strategy config saved.')

    def get(self, key: str, default=None):
        return self.strategy.get(key, _CONFIG_DEFAULTS.get(key, default))


config = Config()
