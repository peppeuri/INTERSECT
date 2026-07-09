"""
INTERSECT — Autonomous AI Trading System
infrastructure.py: Bitget API client with retry logic + scheduler
"""

import requests
import hashlib
import hmac
import json
import time
import uuid
import base64
import logging
from typing import Dict, List, Optional
from urllib.parse import urlencode

log = logging.getLogger('infrastructure')


class BitgetClient:
    RECV_WINDOW = '5000'

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        passphrase: str,
        base_url: str = 'https://api.bitget.com',
        demo: bool = True,
        symbol: str = 'ETHUSDT',
        product_type: str = 'USDT-FUTURES',
        margin_coin: str = 'USDT',
        margin_mode: str = 'crossed',
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.base_url = base_url
        self.demo = demo
        self.symbol = symbol
        self.product_type = product_type
        self.margin_coin = margin_coin
        self.margin_mode = margin_mode
        self._session = requests.Session()
        self._current_leverage: Dict[str, int] = {}

    def _timestamp(self) -> str:
        return str(int(time.time() * 1000))

    def _sign(self, timestamp: str, method: str, path: str, query: str = '', body: str = '') -> str:
        qs = f'?{query}' if query else ''
        prehash = timestamp + method.upper() + path + qs + body
        mac = hmac.new(self.api_secret.encode('utf-8'), prehash.encode('utf-8'), hashlib.sha256).digest()
        return base64.b64encode(mac).decode()

    def _headers(self, timestamp: str, sign: str) -> Dict:
        h = {
            'ACCESS-KEY': self.api_key,
            'ACCESS-SIGN': sign,
            'ACCESS-TIMESTAMP': timestamp,
            'ACCESS-PASSPHRASE': self.passphrase,
            'ACCESS-RECV-WINDOW': self.RECV_WINDOW,
            'Content-Type': 'application/json',
            'locale': 'en-US',
        }
        if self.demo:
            h['paptrading'] = '1'
        return h

    def get(self, path: str, params: Dict = None) -> Dict:
        for attempt in range(3):
            try:
                qs = urlencode(sorted((params or {}).items()))
                ts = self._timestamp()
                sig = self._sign(ts, 'GET', path, qs)
                url = self.base_url + path + (f'?{qs}' if qs else '')
                r = self._session.get(url, headers=self._headers(ts, sig), timeout=15)
                if not r.text.strip():
                    return {'code': '-1', 'msg': f'Empty (HTTP {r.status_code})'}
                return r.json()
            except Exception as e:
                log.warning(f'GET {path} attempt {attempt+1}/3: {e}')
                time.sleep(2 ** attempt)
        return {'code': '-1', 'msg': 'Max retries'}

    def post(self, path: str, body: Dict) -> Dict:
        for attempt in range(3):
            try:
                payload = json.dumps(body)
                ts = self._timestamp()
                sig = self._sign(ts, 'POST', path, '', payload)
                r = self._session.post(
                    self.base_url + path, headers=self._headers(ts, sig),
                    data=payload, timeout=15,
                )
                if not r.text.strip():
                    return {'code': '-1', 'msg': f'Empty (HTTP {r.status_code})'}
                return r.json()
            except Exception as e:
                log.warning(f'POST {path} attempt {attempt+1}/3: {e}')
                time.sleep(2 ** attempt)
        return {'code': '-1', 'msg': 'Max retries'}

    def get_candles(self, limit: int = 100, granularity: str = '15m') -> List[Dict]:
        resp = self.get('/api/v2/mix/market/candles', {
            'symbol': self.symbol, 'productType': self.product_type,
            'granularity': granularity, 'limit': str(limit),
        })
        if resp.get('code') != '00000':
            log.error(f'get_candles error: {resp}')
            return []
        candles = []
        for row in resp.get('data', []):
            candles.append({'ts': int(row[0]), 'o': float(row[1]), 'h': float(row[2]),
                           'l': float(row[3]), 'c': float(row[4]), 'v': float(row[5])})
        candles.sort(key=lambda x: x['ts'])
        return candles

    def get_ticker(self) -> Optional[float]:
        resp = self.get('/api/v2/mix/market/ticker', {
            'symbol': self.symbol, 'productType': self.product_type,
        })
        if resp.get('code') != '00000':
            return None
        data = resp.get('data', [])
        return float(data[0]['lastPr']) if data else None

    def get_position(self) -> Optional[Dict]:
        resp = self.get('/api/v2/mix/position/single-position', {
            'symbol': self.symbol, 'productType': self.product_type, 'marginCoin': self.margin_coin,
        })
        if resp.get('code') != '00000':
            return None
        data = resp.get('data')
        if isinstance(data, dict):
            return data if float(data.get('total', 0)) > 0 else None
        for pos in (data or []):
            if float(pos.get('total', 0)) > 0:
                return pos
        return None

    def get_balance(self) -> float:
        resp = self.get('/api/v2/mix/account/account', {
            'symbol': self.symbol, 'productType': self.product_type, 'marginCoin': self.margin_coin,
        })
        if resp.get('code') != '00000':
            return 0.0
        return float(resp.get('data', {}).get('available', 0))

    def set_leverage(self, lever: int, hold_side: str):
        if self._current_leverage.get(hold_side) == lever:
            return
        resp = self.post('/api/v2/mix/account/set-leverage', {
            'symbol': self.symbol, 'productType': self.product_type,
            'marginCoin': self.margin_coin, 'leverage': str(lever), 'holdSide': hold_side,
        })
        if resp.get('code') != '00000':
            log.warning(f'set_leverage error: {resp}')
        else:
            self._current_leverage[hold_side] = lever

    def place_order(self, side: str, trade_side: str, size: float, price: float, lever: int) -> Dict:
        hold = 'long' if side == 'buy' else 'short'
        self.set_leverage(lever, hold)
        body = {
            'symbol': self.symbol, 'productType': self.product_type,
            'marginMode': self.margin_mode, 'marginCoin': self.margin_coin,
            'size': str(size), 'side': side, 'tradeSide': trade_side,
            'orderType': 'market', 'force': 'gtc', 'clientOid': uuid.uuid4().hex[:32],
        }
        return self.post('/api/v2/mix/order/place-order', body)

    def place_sl_tp(self, hold_side: str, sl_price: float, tp_price: float) -> bool:
        pos = None
        for attempt in range(10):
            pos = self.get_position()
            if pos:
                break
            time.sleep(1)

        if not pos:
            log.error('Position not found after 10s. SL/TP NOT placed.')
            return False

        size = str(pos.get('total', '0'))
        all_ok = True

        for plan_type, trigger in [('loss_plan', sl_price), ('profit_plan', tp_price)]:
            body = {
                'symbol': self.symbol, 'productType': self.product_type,
                'marginCoin': self.margin_coin, 'planType': plan_type,
                'triggerPrice': str(round(trigger, 2)), 'triggerType': 'mark_price',
                'executePrice': '0', 'holdSide': hold_side, 'size': size,
                'clientOid': uuid.uuid4().hex[:32],
            }
            placed = False
            for attempt in range(5):
                resp = self.post('/api/v2/mix/order/place-tpsl-order', body)
                if resp.get('code') == '00000':
                    log.info(f'SL/TP placed: {plan_type} @ {trigger:.2f}')
                    placed = True
                    break
                elif resp.get('code') == '43023' and attempt < 4:
                    time.sleep(2)
                else:
                    log.error(f'SL/TP ({plan_type}) FAILED: {resp}')
                    break
            if not placed:
                all_ok = False
        return all_ok

    def close_position(self, pos: Dict):
        hold = pos.get('holdSide', 'long')
        size = float(pos.get('total', 0))
        side = 'sell' if hold == 'long' else 'buy'
        body = {
            'symbol': self.symbol, 'productType': self.product_type,
            'marginMode': self.margin_mode, 'marginCoin': self.margin_coin,
            'size': str(size), 'side': side, 'tradeSide': 'close',
            'orderType': 'market', 'force': 'gtc', 'clientOid': uuid.uuid4().hex[:32],
        }
        resp = self.post('/api/v2/mix/order/place-order', body)
        if resp.get('code') != '00000':
            log.error(f'close_position error: {resp}')
        else:
            log.info(f'Position {hold} closed, size={size}')

    def test_connection(self) -> bool:
        resp = self.get('/api/v2/mix/account/account', {
            'symbol': self.symbol, 'productType': self.product_type, 'marginCoin': self.margin_coin,
        })
        if resp.get('code') == '00000':
            bal = float(resp.get('data', {}).get('available', 0))
            log.info(f'Connection OK | Balance: {bal:.4f} USDT')
            return True
        log.error(f'Connection FAILED: {resp}')
        return False


def create_bitget_client(**kwargs) -> BitgetClient:
    return BitgetClient(**kwargs)
