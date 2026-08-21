# Multi-Asset Trading — ETH-USDT + SOL-USDT

**Date:** 2026-08-21
**Status:** Approved

## Goal

Add SOL-USDT as a second trading asset alongside ETH-USDT, with independent models, same rules (leverage 10x, SL 2%/TP 3%, anti-churn, exploration), shared USDT balance, no capital cap.

## Architecture

Per-symbol independent stack:

```
ETHUSDT                          SOLUSDT
├── BitgetClient(symbol=ETHUSDT) ├── BitgetClient(symbol=SOLUSDT)
├── PPOAgent                     ├── PPOAgent
├── SACAgent                     ├── SACAgent
├── RegimeClassifier             ├── RegimeClassifier
├── ContinualLearningLoop        ├── ContinualLearningLoop
├── SelfHealingDaemon            ├── SelfHealingDaemon
└── MetaOptimizerAgent           └── MetaOptimizerAgent
```

Shared:
- Same API credentials (Bitget demo account)
- Same USDT balance (cross-margin)
- Same FastAPI dashboard (multi-symbol)
- Same config file

## Config Changes

`config.py`:
```python
SYMBOLS = [s.strip() for s in os.environ.get('SYMBOLS', 'ETHUSDT,SOLUSDT').split(',')]
```
Backward compat: if `SYMBOLS` not set, use `SYMBOL` env var (default `ETHUSDT`).

## main.py Changes

1. Parse `SYMBOLS` list from config
2. For each symbol:
   - Create `BitgetClient(symbol=symbol)`
   - Create PPO + SAC agents
   - Create regime classifier
   - Create continual learning loop
   - Create self-healing daemon
   - Create meta optimizer
   - Store in dicts keyed by symbol
3. Start all loops and healing daemons
4. Pass all to dashboard

## Dashboard Changes

- Per-symbol sections (ETH / SOL)
- Show: PPO Sharpe, SAC Sharpe, position, recent trades
- Trades table includes symbol column

## Checkpoints

Prefixed by symbol: `checkpoints/agent_ppo_{symbol}_{timestamp}.pt`
Symbol dirs: `checkpoints/eth/`, `checkpoints/sol/` (optional, prefix is sufficient)

## Files Changed

- `config.py` — SYMBOLS list
- `main.py` — per-symbol stack creation, dict of loops/agents
- `app.py` — multi-symbol dashboard
- `continual_learning_loop.py` — log prefix per symbol (minor)

## Files NOT Changed

- `infrastructure.py` — already parametrized via `symbol`
- `rl_environment.py` — data-driven, symbol-agnostic
- `ppo_agent.py`, `sac_agent.py` — model-agnostic
- `ensemble_voting.py` — per-loop instance
- `market_regime_classifier.py` — per-loop instance

## Trade-offs

- ~2x RAM (2 PPO + 2 SAC models in memory) — acceptable for VPS
- Both bots share same USDT balance — user chose no capital cap
- Adding BTC later = add to SYMBOLS list, duplicate is already parameterized
