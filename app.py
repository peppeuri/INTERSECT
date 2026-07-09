"""
INTERSECT — Autonomous AI Trading System
app.py: FastAPI web dashboard + REST API + healthcheck
"""

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import HTMLResponse, JSONResponse
import json
import time
from datetime import datetime


def create_app(loop, trade_db, meta_optimizer, ppo_agent, sac_agent, regime_clf, config):
    app = FastAPI(title='INTERSECT', version='1.0.0')

    @app.get('/health')
    def health():
        return {
            'status': 'ok',
            'timestamp': datetime.now().isoformat(),
            'demo': config.DEMO,
            'symbol': config.SYMBOL,
        }

    @app.get('/')
    def dashboard():
        status = loop.get_status()
        metrics = trade_db.get_rolling_metrics(50)
        df = _get_recent_df(loop.client, 500)
        regime, _ = regime_clf.get_current_regime(df) if df is not None else (0, None)

        html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>INTERSECT — Dashboard</title>
            <meta http-equiv="refresh" content="60">
            <style>
                body {{ font-family: 'Courier New', monospace; background: #0a0a0a; color: #00ffaa; margin: 0; padding: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                h1 {{ color: #00ffaa; border-bottom: 1px solid #00ffaa; padding-bottom: 10px; }}
                .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0; }}
                .card {{ background: #111; border: 1px solid #00ffaa33; border-radius: 8px; padding: 15px; }}
                .card h3 {{ margin: 0 0 10px 0; color: #888; font-size: 12px; text-transform: uppercase; }}
                .card .value {{ font-size: 24px; font-weight: bold; }}
                .card .sub {{ font-size: 12px; color: #666; margin-top: 5px; }}
                .green {{ color: #00ffaa; }} .red {{ color: #ff4444; }} .yellow {{ color: #ffaa00; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #00ffaa33; padding: 8px 12px; text-align: left; }}
                th {{ background: #111; color: #888; text-transform: uppercase; font-size: 11px; }}
                .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; }}
                .badge.live {{ background: #00ffaa22; color: #00ffaa; }}
                .badge.demo {{ background: #ffaa0022; color: #ffaa00; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔷 INTERSECT <span class="badge {"live" if not config.DEMO else "demo"}">{"LIVE" if not config.DEMO else "DEMO"}</span></h1>
                <p>ETH-USDT | {config.TIMEFRAME} | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

                <div class="grid">
                    <div class="card">
                        <h3>PPO Sharpe</h3>
                        <div class="value { "green" if status.get("ppo_sharpe", 0) > 0.5 else ("yellow" if status.get("ppo_sharpe", 0) > 0 else "red") }">
                            {status.get("ppo_sharpe", 0):.4f}
                        </div>
                        <div class="sub">SAC: {status.get("sac_sharpe", 0):.4f}</div>
                    </div>
                    <div class="card">
                        <h3>Best Sharpe (Meta)</h3>
                        <div class="value green">{status.get("best_sharpe", 0):.4f}</div>
                        <div class="sub">All-time best agent</div>
                    </div>
                    <div class="card">
                        <h3>Total Steps</h3>
                        <div class="value">{status.get("total_steps", 0):,}</div>
                        <div class="sub">Buffer: {status.get("buffer_size", 0):,}</div>
                    </div>
                    <div class="card">
                        <h3>Regime States</h3>
                        <div class="value">{status.get("n_regimes", 0)}</div>
                        <div class="sub">HMM states</div>
                    </div>
                    <div class="card">
                        <h3>Exploration Mode</h3>
                        <div class="value {"yellow" if status.get("exploration", False) else "green"}">
                            {"ACTIVE" if status.get("exploration", False) else "OFF"}
                        </div>
                    </div>
                    <div class="card">
                        <h3>Trades Today</h3>
                        <div class="value">{status.get("trades_today", 0)}</div>
                        <div class="sub">Equity: ${status.get("equity", 0):.2f}</div>
                    </div>
                </div>

                <h2>Rolling Metrics (last 50 trades)</h2>
                <div class="grid">
                    <div class="card">
                        <h3>Win Rate</h3>
                        <div class="value">{metrics.get("win_rate", 0):.1f}%</div>
                    </div>
                    <div class="card">
                        <h3>Profit Factor</h3>
                        <div class="value">{metrics.get("profit_factor", 0):.2f}</div>
                    </div>
                    <div class="card">
                        <h3>Max Drawdown</h3>
                        <div class="value red">{metrics.get("max_drawdown", 0):.2f}%</div>
                    </div>
                    <div class="card">
                        <h3>Total Return</h3>
                        <div class="value {"green" if metrics.get("total_return", 0) > 0 else "red"}">
                            ${metrics.get("total_return", 0):.2f}
                        </div>
                    </div>
                </div>

                <h2>Recent Trades</h2>
                <table>
                    <tr>
                        <th>Time</th><th>Side</th><th>Size</th><th>Price</th><th>Lev</th><th>PnL</th>
                    </tr>
        '''
        trades = trade_db.get_recent_trades(20)
        for t in trades:
            pnl = t.get('pnl', 0) or 0
            color = 'green' if pnl > 0 else ('red' if pnl < 0 else '')
            html += f'''
                    <tr>
                        <td>{t.get("timestamp", "")[:19]}</td>
                        <td>{t.get("side", "").upper()}</td>
                        <td>{t.get("size", 0):.4f}</td>
                        <td>${t.get("price", 0):.2f}</td>
                        <td>{t.get("leverage", 1)}x</td>
                        <td class="{color}">${pnl:.2f}</td>
                    </tr>'''

        html += '''
                </table>
                <p style="color:#444; font-size:11px;">INTERSECT v1.0 — Autonomous AI Trading System</p>
            </div>
        </body>
        </html>'''
        return HTMLResponse(html)

    @app.get('/api/metrics')
    def api_metrics():
        status = loop.get_status()
        metrics = trade_db.get_rolling_metrics(50)
        return {**status, **{'rolling': metrics}}

    @app.get('/api/trades')
    def api_trades(n: int = 50):
        return trade_db.get_recent_trades(n)

    @app.get('/api/config')
    def api_config():
        return config.strategy

    @app.post('/api/config')
    def update_config(data: dict = Body(...)):
        current = config.strategy
        current.update(data)
        config.save_strategy_config(current)
        return {'status': 'ok', 'config': current}

    @app.get('/api/agent-evolution')
    def api_evolution():
        import sqlite3
        import os
        path = getattr(meta_optimizer, 'db_path', 'agent_evolution.db')
        if not os.path.exists(path):
            return []
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT * FROM agent_evolution ORDER BY timestamp DESC LIMIT 50')
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        return rows

    @app.get('/api/shutdown')
    def shutdown():
        import os
        import signal
        os.kill(os.getpid(), signal.SIGTERM)
        return {'status': 'shutting_down'}

    return app


def _get_recent_df(client, limit=500):
    import pandas as pd
    candles = client.get_candles(limit=limit)
    if not candles:
        return None
    df = pd.DataFrame(candles)
    df = df.rename(columns={'ts': 'datetime', 'o': 'open', 'h': 'high',
                            'l': 'low', 'c': 'close', 'v': 'volume'})
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
    return df.set_index('datetime').sort_index()
