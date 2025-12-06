from __future__ import annotations

import argparse
import csv
import hashlib
import queue
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests

from snap_harvester.config import load_config
from snap_harvester.logging_utils import get_logger
from .binance_feed import BinanceBarFeed
from .execution import BinanceExecutionClient
from .shockflip_live import ShockFlipDetector, Tick
from .meta_builder import LiveMetaBuilder
from .router import LiveRouter
from .telegram_hud import TelegramHUD
from .trade_engine import TradeEngine

# ShockFlipConfig comes from the research stack (made importable via shockflip_live.py)
from core.shockflip_detector import ShockFlipConfig  # type: ignore


class ShockFlipEventFeed:
    """
    ShockFlip event feed. Upstream ShockFlip detector pushes events into this queue.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[Dict[str, Any]] = queue.Queue()

    def put(self, event: Dict[str, Any]) -> None:
        self._queue.put(event)

    def get_nowait(self) -> Optional[Dict[str, Any]]:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None


def _append_csv(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(record.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def _make_client_order_id(trade_id: str) -> str:
    digest = hashlib.sha1(trade_id.encode("utf-8")).hexdigest()[:20]
    return f"SNAP-{digest}"


def run_live_engine(config_path: str | Path, event_feed: ShockFlipEventFeed | None = None) -> ShockFlipEventFeed:
    cfg = load_config(config_path)
    logger = get_logger("live_app")

    meta_builder = LiveMetaBuilder(cfg)
    router = LiveRouter(cfg, model_path=cfg["live"]["model_path"])
    trade_engine = TradeEngine(cfg)
    bar_feed = BinanceBarFeed(cfg)
    exec_client = BinanceExecutionClient(cfg)
    hud = TelegramHUD(cfg)

    data_cfg = cfg["data"]
    risk_cfg = cfg["risk"]
    events_log_path = Path("results/live/events.csv")
    trades_log_path = Path("results/live/trades.csv")

    # Start services
    exec_client.ensure_connection()
    exec_client.flatten_all_positions()
    bar_feed.start()

    event_feed = event_feed or _start_shockflip_pipeline(cfg, logger)
    safe_pause = False
    last_health_ping = time.time()
    binance_ok = True

    logger.info("Live engine running with config=%s", config_path)

    try:
        while True:
            # Handle ShockFlip events if any
            ev = event_feed.get_nowait()
            if ev is not None:
                _handle_event(
                    event=ev,
                    cfg=cfg,
                    risk_cfg=risk_cfg,
                    meta_builder=meta_builder,
                    router=router,
                    trade_engine=trade_engine,
                    exec_client=exec_client,
                    hud=hud,
                    events_log_path=events_log_path,
                    safe_pause=safe_pause,
                    max_open_trades=int(cfg.get("binance", {}).get("max_open_trades", 1)),
                )

            # Consume Binance bars
            bar = bar_feed.get_bar(timeout=1.0)
            if bar is not None:
                closed = trade_engine.on_new_bar(
                    {
                        data_cfg.get("bars_time_col", "timestamp"): bar.timestamp,
                        data_cfg.get("bars_high_col", "high"): bar.high,
                        data_cfg.get("bars_low_col", "low"): bar.low,
                        data_cfg.get("bars_close_col", "close"): bar.close,
                    }
                )
                for tr in closed:
                    hud.notify_trade_close(tr)
                    rec = trade_engine.to_record(tr)
                    rec["binance_symbol"] = exec_client.config.symbol
                    _append_csv(trades_log_path, rec)

            # Health checks
            feed_stale = bar_feed.is_stale()
            if feed_stale and not safe_pause:
                safe_pause = True
                hud.notify_health(
                    status="FEED_STALE_PAUSE",
                    details={
                        "feed_stale": True,
                        "binance_ok": binance_ok,
                        "open_trades": len(trade_engine.open_trades()),
                    },
                )
                logger.warning("Feed stale -> SAFE_PAUSE enabled")
            elif safe_pause and not feed_stale:
                safe_pause = False
                hud.notify_health(
                    status="RESUMED",
                    details={
                        "feed_stale": False,
                        "binance_ok": binance_ok,
                        "open_trades": len(trade_engine.open_trades()),
                    },
                )

            now = time.time()
            if now - last_health_ping > 60:
                try:
                    exec_client.ensure_connection()
                    binance_ok = True
                except Exception as exc:  # pylint: disable=broad-except
                    logger.error("Binance health check failed: %s", exc)
                    binance_ok = False
                    safe_pause = True
                hud.notify_health(
                    status="HEARTBEAT",
                    details={
                        "feed_stale": feed_stale,
                        "binance_ok": binance_ok,
                        "open_trades": len(trade_engine.open_trades()),
                    },
                )
                last_health_ping = now
    except KeyboardInterrupt:
        logger.info("Shutting down live engine")
    finally:
        bar_feed.stop()
    return event_feed


def _start_shockflip_pipeline(cfg: dict, logger) -> ShockFlipEventFeed:
    """
    Start ShockFlipDetector + Binance aggTrade websocket and return the shared event feed.
    """
    from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient  # type: ignore

    symbol = cfg.get("binance", {}).get("symbol", "BTCUSDT")
    base_url = cfg.get("binance", {}).get("base_url", "https://fapi.binance.com")

    # Live ShockFlip geometry aligned with proven Nov 2025 OOS walkforward
    sf_cfg_raw = cfg.get("shockflip", {})
    location_require_extreme = bool(sf_cfg_raw.get("location_require_extreme", True))
    dynamic_thresholds = sf_cfg_raw.get("dynamic_thresholds", {"enabled": False}) or {"enabled": False}
    sf_cfg = ShockFlipConfig(
        source="imbalance",
        z_window=int(sf_cfg_raw.get("z_window", 90)),
        z_band=float(sf_cfg_raw.get("z_band", 1.2)),
        jump_band=float(sf_cfg_raw.get("jump_band", 1.5)),
        persistence_bars=int(sf_cfg_raw.get("persistence_bars", 1)),
        persistence_ratio=0.5,
        dynamic_thresholds=dynamic_thresholds,
        location_filter={
            "donchian_window": int(sf_cfg_raw.get("donchian_window", 40)),
            "require_extreme": location_require_extreme,
        },
    )
    min_bars = int(sf_cfg_raw.get("min_bars", 60))

    detector = ShockFlipDetector(symbol=symbol, cfg=sf_cfg, min_bars=min_bars)
    _preseed_shockflip(
        detector=detector,
        symbol=symbol,
        base_url=base_url,
        minutes=max(sf_cfg.z_window, min_bars) + 30,
        logger=logger,
    )
    event_feed = ShockFlipEventFeed()
    tick_count = 0

    def handle_aggtrade(_, msg: Dict[str, Any]) -> None:
        nonlocal tick_count
        try:
            tick = Tick(
                timestamp=int(msg.get("T") or msg.get("E")),
                price=float(msg["p"]),
                qty=float(msg["q"]),
                is_buyer_maker=bool(msg["m"]),
            )
        except Exception:
            return

        tick_count += 1
        if tick_count % 100 == 0:
            logger.info("[WS] received %d aggTrades", tick_count)

        ev = detector.update(tick)
        if ev is not None:
            logger.info("ShockFlip LIVE event: %s", ev)
            event_feed.put(ev)

    ws = UMFuturesWebsocketClient(on_message=handle_aggtrade)
    ws.agg_trade(symbol=symbol.lower())
    logger.info("Started ShockFlip detector websocket for %s", symbol)
    return event_feed


def _preseed_shockflip(detector: ShockFlipDetector, symbol: str, base_url: str, minutes: int, logger) -> None:
    """
    Fetch recent aggTrades and seed the ShockFlip detector so z-window features are hot on launch.
    """
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - minutes * 60 * 1000
    params = {"symbol": symbol, "limit": 1000, "startTime": start_ms}
    url = f"{base_url.rstrip('/')}/fapi/v1/aggTrades"

    ticks = []
    attempts = 0
    try:
        while True:
            attempts += 1
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break

            last_ts = None
            for item in data:
                ts = int(item.get("T") or item.get("E"))
                if ts is None or ts < start_ms:
                    continue
                ticks.append(
                    {
                        "ts": pd.to_datetime(ts, unit="ms", utc=True),
                        "price": float(item["p"]),
                        "qty": float(item["q"]),
                        "is_buyer_maker": bool(item["m"]),
                    }
                )
                last_ts = ts if last_ts is None else max(last_ts, ts)

            if last_ts is None or len(data) < params["limit"]:
                break
            params["startTime"] = last_ts + 1
            if attempts > 50:
                break
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("ShockFlip preseed failed (warmup will be live-only): %s", exc)
        return

    if not ticks:
        logger.info("ShockFlip preseed: no ticks fetched; warmup will occur live.")
        return

    tick_df = pd.DataFrame(ticks)
    detector.preload_ticks(tick_df)
    logger.info("ShockFlip preseeded with %d aggTrades (~%d min window)", len(ticks), minutes)


def _handle_event(
    event: Dict[str, Any],
    cfg: dict,
    risk_cfg: dict,
    meta_builder: LiveMetaBuilder,
    router: LiveRouter,
    trade_engine: TradeEngine,
    exec_client: BinanceExecutionClient,
    hud: TelegramHUD,
    events_log_path: Path,
    safe_pause: bool,
    max_open_trades: int,
) -> None:
    logger = get_logger("live_app")
    decision_ts = time.time()
    try:
        meta_row = meta_builder.build_meta_row(event)
        should_route, p_hat = router.score_and_route(meta_row)
        hud.notify_event_decision(event, p_hat=p_hat, routed=should_route)

        rec = dict(event)
        rec.update({"p_hat": p_hat, "routed": should_route})
        rec["event_bar_time"] = pd.to_datetime(
            event.get("event_bar_time", event.get("timestamp")), utc=True
        ).isoformat()
        rec["decision_ts"] = datetime.fromtimestamp(decision_ts, tz=timezone.utc).isoformat()
        _append_csv(events_log_path, rec)

        if not should_route:
            return
        if safe_pause:
            logger.warning("SAFE_PAUSE active; skipping new trade")
            return
        if len(trade_engine.open_trades()) >= max_open_trades:
            logger.warning("Max open trades reached (%d); skipping", max_open_trades)
            return

        raw_side = event.get("side", 0)
        if isinstance(raw_side, str):
            s = raw_side.strip().upper()
            if s in ("BUY", "LONG", "+1", "1"):
                side = 1
            elif s in ("SELL", "SHORT", "-1"):
                side = -1
            else:
                logger.warning("Invalid side on event: %s", raw_side)
                return
        else:
            try:
                side = int(raw_side)
            except (TypeError, ValueError):
                logger.warning("Invalid side on event: %s", raw_side)
                return

        if side not in (1, -1):
            logger.warning("Invalid side on event: %s", side)
            return
        atr = float(event["atr"])
        if atr <= 0:
            logger.warning("Invalid ATR on event: %s", atr)
            return

        risk_unit = atr * float(risk_cfg["risk_k_atr"])
        sl_dist = risk_unit * float(risk_cfg["sl_r_multiple"])
        tp_dist = risk_unit * float(risk_cfg["tp_r_multiple"])
        be_dist = risk_unit * float(risk_cfg["be_r_multiple"])

        trade_id = trade_engine.make_trade_id(event)
        client_order_id = _make_client_order_id(trade_id)

        qty = exec_client.calculate_qty(
            entry_price=float(event.get("close", event.get("price", 0.0))),
            r_unit=risk_unit,
            quote_risk=exec_client.config.quote_risk_per_trade,
        )

        exec_result = exec_client.submit_entry_and_brackets(
            side=side,
            quantity=qty,
            sl_dist=sl_dist,
            tp_dist=tp_dist,
            be_dist=be_dist,
            client_order_id=client_order_id,
        )
        fill_ts = time.time()
        decision_price = float(event.get("close", event.get("price", 0.0)))
        slippage_raw = exec_result["fill_price"] - decision_price
        slippage_bps = (slippage_raw / decision_price) * 10_000 if decision_price else 0.0
        dir_sign = 1 if side > 0 else -1
        slippage_r = (slippage_raw * dir_sign) / max(risk_unit, 1e-8)
        event_to_fill_ms = (fill_ts - decision_ts) * 1000.0

        trade = trade_engine.open_trade_from_live_fill(
            event=event,
            p_hat=p_hat,
            fill_price=exec_result["fill_price"],
            r_unit=risk_unit,
            sl_price=exec_result["sl_price"],
            be_price=exec_result["be_price"],
            tp_price=exec_result["tp_price"],
            meta_row=meta_row,
        )
        hud.notify_trade_open(trade)
        open_rec = trade_engine.to_record(trade)
        open_rec.update(
            {
                "decision_price": decision_price,
                "fill_price": exec_result["fill_price"],
                "slippage_bps": slippage_bps,
                "slippage_r": slippage_r,
                "event_to_fill_ms": event_to_fill_ms,
                "decision_ts": datetime.fromtimestamp(decision_ts, tz=timezone.utc).isoformat(),
                "fill_ts": datetime.fromtimestamp(fill_ts, tz=timezone.utc).isoformat(),
                "client_order_id": client_order_id,
            }
        )
        _append_csv(events_log_path.parent / "trades.csv", open_rec)
        logger.info(
            "Opened trade %s @ %.2f (qty %.6f)",
            trade.id,
            trade.entry_price,
            qty,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Failed to process event: %s", exc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/snap_harvester_live_btc.yaml")
    args = parser.parse_args()
    run_live_engine(args.config)
