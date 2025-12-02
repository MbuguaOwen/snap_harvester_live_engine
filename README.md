# Snap Harvester Live (BTCUSDT)

Minimal live-only stack for running the ShockFlip Snap Harvester v2 on BTCUSDT with the frozen 2024 model. Research tooling has been removed or archived; this tree focuses on live execution plus a lightweight replay mode for parity checks.

## Repo map
- `configs/snap_harvester_live_btc.yaml` - single live config (risk/meta unchanged).
- `snap_harvester/live/runner.py` - replay from historical bars/events.
- `snap_harvester/live/app.py` - live orchestrator wiring Binance feed/execution + Telegram HUD.
- `snap_harvester/live/{binance_feed.py,execution.py,telegram_hud.py}` - live-only adapters.
- `results/models/hgb_snap_2024_BTC.joblib` - frozen classifier used for routing.
- Research/backtest code is parked in `snap_harvester/archive/` and configs in `configs/archive/`.

## Config + environment
Use `configs/snap_harvester_live_btc.yaml`. Risk/meta blocks stay identical to the 2024 profile. New blocks:
- `binance`: `symbol`, `base_url`, `ws_url`, `api_key_env`, `api_secret_env`, `quote_risk_per_trade`, `use_oco_brackets`, `max_retries`, `order_timeout_sec`, `heartbeat_sec`, `stale_timeout_sec`, `max_open_trades`.
- `telegram`: `enabled`, `bot_token_env`, `chat_id_env`, flags for noise control.

Required env vars for live:
- `BINANCE_API_KEY`, `BINANCE_API_SECRET`
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` (only if Telegram HUD enabled)

## Replay mode (parity / paper)
```
python -m snap_harvester.live.runner --config configs/snap_harvester_live_btc.yaml --mode replay
```
Uses the configured bars/events paths, applies the frozen model, and writes a routed tape to `paths.replay_trades_out`.

Parity checks:
```
python scripts/compare_replay_parity.py --research results/meta/snap_routed_tape_2025_BTC.csv --epsilon 1e-6
python scripts/router_parity_check.py --config configs/snap_harvester_live_btc.yaml
```

## Live mode (canonical)
Single supported entrypoint (ShockFlip detector + Binance aggTrade WS wired in):
```
python -m snap_harvester.live.app --config configs/snap_harvester_live_btc.yaml
```
Components:
- `ShockFlipDetector` uses the research feature stack (`core.features` / `core.shockflip_detector`) and builds 1m bars via `core.data_loader.resample_ticks_to_bars`.
- `BinanceBarFeed`: 1m futures klines via websocket; REST backfill on reconnect; stale detection (no new entries when stale).
- `BinanceExecutionClient`: market entry + exchange-hosted SL/TP brackets (STOP/TAKE_PROFIT market), deterministic `clientOrderId`, restart flatten helper.
- `TelegramHUD`: concise HUD messages (health, opens/closes; event decisions optional).
- `TradeEngine`: barrier logic; live path uses `open_trade_from_live_fill` so SL/TP use actual fills.

## No-drift parity mandate
- The model is frozen reality. Any change to ShockFlipConfig (z_band, jump_band, location_filter, etc.), feature windows (ATR, z_window, div windows), or barrier geometry (0.5R/3R/H=30) voids the OOS parity. You’ll still have a system, but it won’t be the 80% / 3R system.
- Config drift is now the real enemy. If a future commit touches `shockflip_live.py`, `app.py`, `meta_builder.py`, `router.py`, or the YAMLs and doesn’t re-run:
  - `python scripts/router_parity_check.py --config configs/snap_harvester_live_btc.yaml`
  - `python scripts/compare_replay_parity.py --research results/meta/snap_routed_tape_2025_BTC.csv --epsilon 1e-6`
  then assume parity is broken until proven otherwise.
- Data/feed differences are the only drift left: live tick stream ≠ historical CSVs (endpoint behavior, microstructure quirks, WS drops/reconnects, partial fills, throttling). That’s market/infra, not code parity.
- Honest guarantee (scope-limited): “For BTCUSDT, Snap Harvester v2, under `configs/snap_harvester_live_btc.yaml`, using this repo and these artifacts, the live engine is a mechanical clone of the research engine that produced the 2025 OOS tape.” It does not automatically cover other symbols, configs, or future refactors.

## Safety rails
- Idempotent client order IDs and lookup before re-sending.
- Startup flatten + open-order cancel to avoid ghost state.
- Feed staleness and Binance health checks -> SAFE_PAUSE (no new entries) with Telegram warning.
- Exchange-side brackets only; no synthetic fills from bars.

## Deployment artifact
- Shipping tree excludes research/backtest outputs; keep `configs/snap_harvester_live_btc.yaml`, `snap_harvester/live/*`, `snap_harvester/{config.py,logging_utils.py,modeling.py,metrics.py,utils/ticks.py}`, and `results/models/hgb_snap_2024_BTC.joblib` (+ optional `.gitkeep`).


python -m snap_harvester.live.app --config configs/snap_harvester_live_btc.yaml
