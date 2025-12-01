from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

from snap_harvester.config import load_config
from snap_harvester.logging_utils import get_logger
from .feed import ReplayBarFeed, ReplayEventFeed
from .meta_builder import LiveMetaBuilder
from .router import LiveRouter
from .trade_engine import TradeEngine


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def run_replay(cfg_path: str | Path) -> None:
    """
    Replay mode:

    - Loads minutes + ShockFlip events.
    - Applies frozen 2024 BTC model to each event (p_hat).
    - Routes trades (currently: all events, or ML-threshold if enabled).
    - Streams bars through TradeEngine to produce r_final.
    - Writes a routed tape CSV for parity checks.
    """
    cfg = load_config(cfg_path)
    logger = get_logger("live_replay")

    data_cfg = cfg["data"]
    paths_cfg: Dict[str, Any] = cfg.get("paths", {})
    live_cfg: Dict[str, Any] = cfg.get("live", {})

    symbol = data_cfg["symbols"][0]
    bars_path_template = data_cfg["bars_path_template"]
    bars_path = bars_path_template.format(symbol=symbol)
    events_path = data_cfg["events_path"]
    model_path = live_cfg.get("model_path", "results/models/hgb_snap_2024_BTC.joblib")

    out_path = Path(paths_cfg.get("replay_trades_out", f"results/live/replay_routed_tape_2025_{symbol}.csv"))
    _ensure_dir(out_path)

    logger.info("Replay mode: bars=%s events=%s model=%s", bars_path, events_path, model_path)

    bar_feed = ReplayBarFeed(
        path=bars_path,
        time_col=data_cfg.get("bars_time_col", "timestamp"),
        open_col=data_cfg.get("bars_open_col", "open"),
        high_col=data_cfg.get("bars_high_col", "high"),
        low_col=data_cfg.get("bars_low_col", "low"),
        close_col=data_cfg.get("bars_close_col", "close"),
        volume_col=data_cfg.get("bars_volume_col", "volume"),
    )
    event_feed = ReplayEventFeed(path=events_path, time_col=data_cfg.get("events_time_col", "timestamp"))

    meta_builder = LiveMetaBuilder(cfg)
    router = LiveRouter(cfg, model_path=model_path)
    trade_engine = TradeEngine(cfg)

    # Group events by bar timestamp for efficient lookup
    events_by_ts: Dict[pd.Timestamp, List[Dict[str, Any]]] = {}
    for ev in event_feed:
        ts = pd.to_datetime(ev.get("event_bar_time", ev.get("timestamp")), utc=True)
        events_by_ts.setdefault(ts, []).append(ev)

    records: List[Dict[str, Any]] = []

    for bar in bar_feed:
        bar_ts = bar.timestamp

        # Handle events that fire on this bar
        for ev in events_by_ts.get(bar_ts, []):
            meta_row = meta_builder.build_meta_row(ev)
            should_route, p_hat = router.score_and_route(meta_row)
            if not should_route:
                continue
            trade = trade_engine.on_new_event(ev, p_hat=p_hat, meta_row=meta_row)
            rec = trade_engine.to_record(trade)
            rec["timestamp"] = trade.entry_ts.isoformat()
            records.append(rec)

        # Update open trades with this bar
        closed = trade_engine.on_new_bar(
            {
                data_cfg.get("bars_time_col", "timestamp"): bar_ts,
                data_cfg.get("bars_high_col", "high"): bar.high,
                data_cfg.get("bars_low_col", "low"): bar.low,
                data_cfg.get("bars_close_col", "close"): bar.close,
            }
        )
        for tr in closed:
            # Update existing record for this trade id (if any) with exit info
            for rec in records:
                if rec["id"] == tr.id:
                    rec.update(trade_engine.to_record(tr))
                    break

    df = pd.DataFrame.from_records(records)
    df.to_csv(out_path, index=False)
    logger.info("Saved replay routed tape -> %s (n=%d)", out_path, len(df))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config (e.g., configs/snap_harvester_live_btc.yaml)",
    )
    parser.add_argument(
        "--mode",
        choices=["replay"],
        default="replay",
        help="Engine mode. Currently only 'replay' is wired in this module.",
    )
    args = parser.parse_args()

    if args.mode == "replay":
        run_replay(args.config)
    else:
        raise SystemExit(f"Unsupported mode: {args.mode!r}")


if __name__ == "__main__":
    main()
