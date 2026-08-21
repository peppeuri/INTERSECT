"""
INTERSECT — Autonomous AI Trading System
app.py: FastAPI web dashboard + REST API + healthcheck (multi-symbol)
"""

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import HTMLResponse, JSONResponse
import json
import time
from datetime import datetime


def create_app(stacks, config):
    app = FastAPI(title='INTERSECT', version='2.0.0')

    @app.get('/health')
    def health():
        return {
            'status': 'ok',
            'timestamp': datetime.now().isoformat(),
            'demo': config.DEMO,
            'symbols': list(stacks.keys()),
        }

    @app.get('/')
    def dashboard():
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        badge = "live" if not config.DEMO else "demo"
        badge_text = "LIVE" if not config.DEMO else "DEMO"

        symbols_html = ''
        for sym, s in stacks.items():
            status = s['loop'].get_status()
            metrics = s['trade_db'].get_rolling_metrics(50)
            pos = s['bitget'].get_position()

            sym_label = f"{sym[:3]}-{sym[3:]}" if len(sym) > 3 else sym
            pos_html = ''
            if pos:
                side = pos.get('holdSide', '?').upper()
                size = pos.get('total', '?')
                entry = pos.get('openPriceAvg', '?')
                upl = float(pos.get('unrealizedPL', 0))
                upl_cls = 'green' if upl > 0 else 'red'
                pos_html = f'''
                    <div class="card">
                        <h3>Position</h3>
                        <div class="value">{side} {size} @ ${float(entry):,.2f}</div>
                        <div class="sub {upl_cls}">uPnL: ${upl:+.2f}</div>
                    </div>'''
            else:
                pos_html = '<div class="card"><h3>Position</h3><div class="value" style="color:#666">None</div></div>'

            trades_html = ''
            trades = s['trade_db'].get_recent_trades(10)
            for t in trades:
                pnl = t.get('pnl', 0) or 0
                pnl_cls = 'green' if pnl > 0 else ('red' if pnl < 0 else '')
                trades_html += f'''
                    <tr>
                        <td>{t.get("timestamp", "")[:19]}</td>
                        <td>{t.get("side", "").upper()}</td>
                        <td>{t.get("size", 0):.4f}</td>
                        <td>${t.get("price", 0):,.2f}</td>
                        <td>{t.get("leverage", 1)}x</td>
                        <td class="{pnl_cls}">${pnl:+.2f}</td>
                    </tr>'''

            ppo_sharpe = status.get('ppo_sharpe', 0)
            ppo_cls = 'green' if ppo_sharpe > 0.5 else ('yellow' if ppo_sharpe > 0 else 'red')

            symbols_html += f'''
                <h2 style="color:#00ffaa; margin-top:30px;">{sym_label}</h2>
                <div class="grid">
                    <div class="card">
                        <h3>PPO Sharpe</h3>
                        <div class="value {ppo_cls}">{ppo_sharpe:.4f}</div>
                        <div class="sub">SAC: {status.get("sac_sharpe", 0):.4f}</div>
                    </div>
                    <div class="card">
                        <h3>Best Sharpe (Meta)</h3>
                        <div class="value green">{status.get("best_sharpe", 0):.4f}</div>
                    </div>
                    <div class="card">
                        <h3>Total Steps</h3>
                        <div class="value">{status.get("total_steps", 0):,}</div>
                        <div class="sub">Buffer: {status.get("buffer_size", 0):,}</div>
                    </div>
                    <div class="card">
                        <h3>Exploration</h3>
                        <div class="value {"yellow" if status.get("exploration") else "green"}">
                            {"ACTIVE" if status.get("exploration") else "OFF"}
                        </div>
                    </div>
                    <div class="card">
                        <h3>Trades Today</h3>
                        <div class="value">{status.get("trades_today", 0)}</div>
                    </div>
                    {pos_html}
                </div>

                <h3>Rolling Metrics (last 50 trades)</h3>
                <div class="grid">
                    <div class="card"><h3>Win Rate</h3><div class="value">{metrics.get("win_rate", 0):.1f}%</div></div>
                    <div class="card"><h3>Profit Factor</h3><div class="value">{metrics.get("profit_factor", 0):.2f}</div></div>
                    <div class="card"><h3>Max Drawdown</h3><div class="value red">{metrics.get("max_drawdown", 0):.2f}%</div></div>
                    <div class="card"><h3>Total Return</h3>
                        <div class="value {"green" if metrics.get("total_return", 0) > 0 else "red"}">${metrics.get("total_return", 0):+.2f}</div>
                    </div>
                </div>

                <h3>Recent Trades</h3>
                <table>
                    <tr><th>Time</th><th>Side</th><th>Size</th><th>Price</th><th>Lev</th><th>PnL</th></tr>
                    {trades_html}
                </table>'''

        html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>INTERSECT — Dashboard</title>
            <meta http-equiv="refresh" content="60">
            <style>
                body {{ font-family: 'Courier New', monospace; background: #0a0a0a; color: #00ffaa; margin: 0; padding: 20px; }}
                .container {{ max-width: 1400px; margin: 0 auto; }}
                h1 {{ color: #00ffaa; border-bottom: 1px solid #00ffaa; padding-bottom: 10px; }}
                .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 15px; margin: 15px 0; }}
                .card {{ background: #111; border: 1px solid #00ffaa33; border-radius: 8px; padding: 15px; }}
                .card h3 {{ margin: 0 0 10px 0; color: #888; font-size: 12px; text-transform: uppercase; }}
                .card .value {{ font-size: 22px; font-weight: bold; }}
                .card .sub {{ font-size: 12px; color: #666; margin-top: 5px; }}
                .green {{ color: #00ffaa; }} .red {{ color: #ff4444; }} .yellow {{ color: #ffaa00; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                th, td {{ border: 1px solid #00ffaa33; padding: 8px 12px; text-align: left; }}
                th {{ background: #111; color: #888; text-transform: uppercase; font-size: 11px; }}
                .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; }}
                .badge.live {{ background: #00ffaa22; color: #00ffaa; }}
                .badge.demo {{ background: #ffaa0022; color: #ffaa00; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>INTERSECT <span class="badge {badge}">{badge_text}</span></h1>
                <p>{", ".join(stacks.keys())} | {config.TIMEFRAME} | {now_str}</p>
                {symbols_html}
                <p style="color:#444; font-size:11px; margin-top:40px;">INTERSECT v2.0 — Multi-Asset AI Trading System</p>
            </div>
        </body>
        </html>'''
        return HTMLResponse(html)

    @app.get('/api/metrics')
    def api_metrics():
        result = {}
        for sym, s in stacks.items():
            status = s['loop'].get_status()
            metrics = s['trade_db'].get_rolling_metrics(50)
            result[sym] = {**status, 'rolling': metrics}
        return result

    @app.get('/api/trades/{symbol}')
    def api_trades(symbol: str, n: int = 50):
        s = stacks.get(symbol.upper())
        if not s:
            raise HTTPException(404, f'Unknown symbol: {symbol}')
        return s['trade_db'].get_recent_trades(n)

    @app.get('/api/config')
    def api_config():
        return config.strategy

    @app.post('/api/config')
    def update_config(data: dict = Body(...)):
        current = config.strategy
        current.update(data)
        config.save_strategy_config(current)
        return {'status': 'ok', 'config': current}

    @app.get('/api/agent-evolution/{symbol}')
    def api_evolution(symbol: str):
        import sqlite3
        import os
        s = stacks.get(symbol.upper())
        if not s:
            raise HTTPException(404, f'Unknown symbol: {symbol}')
        path = s['meta'].db_path
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
