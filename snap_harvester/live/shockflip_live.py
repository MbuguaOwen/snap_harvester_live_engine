from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from .shockflip_core import (
    ShockFlipConfig,
    add_core_features,
    add_hypothesis_features,
    detect_shockflip_signals,
    resample_ticks_to_bars,
)


@dataclass
class Tick:
    """Minimal tick representation from Binance aggTrade stream."""

    timestamp: int  # milliseconds
    price: float
    qty: float
    is_buyer_maker: bool


class ShockFlipDetector:
    """
    Live ShockFlip detector that mirrors the research stack.

    - Aggregates aggTrade ticks into 1m bars exactly like core.data_loader.resample_ticks_to_bars.
    - Applies research feature builders (add_core_features, add_hypothesis_features).
    - Runs core.shockflip_detector.detect_shockflip_signals with the canonical ShockFlipConfig.
    """

    def __init__(
        self,
        symbol: str,
        cfg: ShockFlipConfig | None = None,
        min_bars: int = 240,
    ) -> None:
        self.symbol = symbol
        self.cfg = cfg or ShockFlipConfig()
        self.min_bars = int(min_bars)

        self._ticks: List[Dict[str, Any]] = []
        self._current_minute: Optional[int] = None
        self._last_signal_ts: Optional[int] = None
        self._bars: pd.DataFrame = pd.DataFrame()
        self.last_trade_ts: Optional[float] = None  # unix seconds of latest aggTrade
        self.trades_in_last_sec: int = 0
        self._last_trade_log_sec: int = 0

    def preload_ticks(self, ticks: pd.DataFrame) -> None:
        """
        Seed the detector with historical aggTrades so z-score windows are ready on launch.
        """
        if ticks.empty:
            return
        bars = resample_ticks_to_bars(ticks, timeframe="1min", symbol=self.symbol)
        if bars.empty:
            return
        self._bars = bars.tail(1000).reset_index(drop=True)
        # Track the latest minute so live ticks continue cleanly
        last_ts = pd.to_datetime(self._bars["timestamp"].iloc[-1], utc=True)
        self._current_minute = int(last_ts.value // 1_000_000 // 60000)

    def update(self, tick: Tick) -> Optional[Dict[str, Any]]:
        """
        Ingest one trade tick and emit an event dict when a ShockFlip triggers.
        """
        minute = tick.timestamp // 60000

        if self._current_minute is None:
            self._current_minute = minute

        # Buffer tick
        self._ticks.append(
            {
                "ts": pd.to_datetime(tick.timestamp, unit="ms", utc=True),
                "price": float(tick.price),
                "qty": float(tick.qty),
                "is_buyer_maker": bool(tick.is_buyer_maker),
            }
        )

        # Only process on minute change to avoid partial bars
        if minute == self._current_minute:
            return None

        # Minute rolled: resample buffered ticks to bars and compute features
        self._current_minute = minute
        ticks_df = pd.DataFrame(self._ticks)
        self._ticks = []  # reset buffer for next minute
        if ticks_df.empty:
            return None

        new_bars = resample_ticks_to_bars(ticks_df, timeframe="1min", symbol=self.symbol)
        if new_bars.empty:
            return None

        # Append new bars to existing history (including preseed) and keep a rolling window
        if self._bars is None or self._bars.empty:
            bars = new_bars
        else:
            bars = pd.concat([self._bars, new_bars], ignore_index=True)
        bars = bars.drop_duplicates(subset=["timestamp"]).tail(1000).reset_index(drop=True)
        self._bars = bars

        # Require a minimum history window before attempting detection
        if len(self._bars) < max(self.min_bars, 1):
            return None

        # Apply research feature stack with live ShockFlipConfig geometry
        donchian_window = int(self.cfg.location_filter.get("donchian_window", 120))
        feat = add_core_features(
            self._bars,
            z_window=self.cfg.z_window,
            atr_window=60,
            donchian_window=donchian_window,
        )
        feat = add_hypothesis_features(feat, prior_flow_window=60, div_window=60, atr_pct_window=5000)
        feat = detect_shockflip_signals(feat, self.cfg)
        if feat.empty or len(feat) < max(10, self.cfg.z_window // 4):
            return None

        latest = feat.iloc[-1]
        signal = int(latest.get("shockflip_signal", 0))
        if signal == 0:
            return None

        ts = pd.to_datetime(latest["timestamp"], utc=True)
        ts_ms = int(ts.value // 1_000_000)

        # Avoid double firing on same bar
        if self._last_signal_ts is not None and ts_ms == self._last_signal_ts:
            return None
        self._last_signal_ts = ts_ms

        side = 1 if signal > 0 else -1
        bar_time = ts.floor("T")

        event: Dict[str, Any] = latest.to_dict()
        event.update(
            {
                "timestamp": ts.isoformat(),
                "event_bar_time": bar_time.isoformat(),
                "event_bar_offset_min": 0,
                "event_id": f"{self.symbol}-{bar_time.isoformat()}-{len(feat)-1}",
                "symbol": self.symbol,
                "side": side,
                # Convenience aliases used elsewhere
                "open": float(latest["open"]),
                "high": float(latest["high"]),
                "low": float(latest["low"]),
                "close": float(latest["close"]),
                "volume": float(latest.get("volume", 0.0)),
                "buy_qty": float(latest.get("buy_qty", 0.0)),
                "sell_qty": float(latest.get("sell_qty", 0.0)),
                "imbalance": float(latest.get("imbalance", 0.0)),
                "imbalance_z": float(latest.get("imbalance_z", 0.0)),
                "shockflip_z": float(latest.get("imbalance_z", 0.0)),
                "range_atr": float((latest.get("high", 0.0) - latest.get("low", 0.0)) / (latest.get("atr", 1e-9))),
            }
        )

        # Fill decile placeholders if absent
        for col in ("shock_intensity_z_decile", "div_score_decile", "rel_vol_decile"):
            if col not in event:
                event[col] = None
        return event
