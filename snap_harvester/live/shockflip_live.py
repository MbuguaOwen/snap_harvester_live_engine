from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class ShockFlipConfig:
    source: str = "imbalance"
    z_window: int = 240
    z_band: float = 2.5
    jump_band: float = 3.0
    persistence_bars: int = 6
    persistence_ratio: float = 0.60
    location_filter: Dict = field(default_factory=lambda: {"donchian_window": 120, "require_extreme": True})
    dynamic_thresholds: Dict = field(default_factory=lambda: {"enabled": False, "percentile": 0.99})


@dataclass
class Tick:
    """Minimal tick representation from Binance aggTrade stream."""
    timestamp: int  # milliseconds
    price: float
    qty: float
    is_buyer_maker: bool


class ShockFlipDetector:
    """
    Live ShockFlip detector that builds 1m bars from aggTrade ticks and applies
    a ShockFlip signal similar to the research stack.
    """

    def __init__(self, symbol: str, cfg: ShockFlipConfig | None = None, atr_window: int = 14) -> None:
        self.symbol = symbol
        self.cfg = cfg or ShockFlipConfig()
        self.atr_window = atr_window

        self.window: Deque[Tick] = deque(maxlen=5000)
        self.current_minute: Optional[int] = None
        self.current_bar: Dict[str, Any] = {}
        self.bars: list[Dict[str, Any]] = []
        self.bars_df: pd.DataFrame = pd.DataFrame()
        self.last_signal_ts: Optional[int] = None

    def update(self, tick: Tick) -> Optional[Dict[str, Any]]:
        """
        Ingest one trade tick and, when a ShockFlip triggers, return an event dict compatible with the live engine.
        """
        self.window.append(tick)
        minute = tick.timestamp // 60000
        if self.current_minute is None:
            self._start_new_minute(minute, tick)
            return None

        if minute != self.current_minute:
            self._finalize_current_bar()
            self._start_new_minute(minute, tick)
        else:
            self._update_current_bar(tick)

        if self.bars_df.empty or len(self.bars_df) < max(10, self.cfg.z_window // 4):
            return None

        signals_df = detect_shockflip_signals(self.bars_df, self.cfg)
        if signals_df.empty:
            return None
        latest = signals_df.iloc[-1]
        signal = int(latest.get("shockflip_signal", 0))
        ts_ms = int(latest["timestamp"])

        # Avoid double firing on same bar
        if signal == 0 or (self.last_signal_ts is not None and ts_ms == self.last_signal_ts):
            return None

        self.last_signal_ts = ts_ms
        side = 1 if signal > 0 else -1

        iso_ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()
        bar_time_dt = datetime.fromtimestamp((ts_ms // 60000) * 60, tz=timezone.utc).replace(second=0, microsecond=0)
        bar_time = bar_time_dt.isoformat()

        event: Dict[str, Any] = {
            "timestamp": iso_ts,
            "event_bar_time": bar_time,
            "event_id": f"{self.symbol}-{bar_time}-{ts_ms}",
            "symbol": self.symbol,
            "side": side,
            "open": float(latest["open"]),
            "high": float(latest["high"]),
            "low": float(latest["low"]),
            "close": float(latest["close"]),
            "volume": float(latest["volume"]),
            "buy_qty": float(latest.get("buy_qty", 0.0)),
            "sell_qty": float(latest.get("sell_qty", 0.0)),
            "atr": float(latest.get("atr", 0.0)),
            "atr_pct": float(latest.get("atr_pct", 0.0)),
            "rel_vol": float(latest.get("rel_vol", 0.0)),
            "imbalance": float(latest.get("imbalance", 0.0)),
            "imbalance_z": float(latest.get("imbalance_z", 0.0)),
            "shockflip_z": float(latest.get("shockflip_z", 0.0)),
            "shock_intensity_z": float(latest.get("shock_intensity_z", 0.0)),
            "range_atr": float(latest.get("range_atr", 0.0)),
            "donchian_loc": float(latest.get("donchian_loc", 0.0)),
            "price_flow_div": float(latest.get("price_flow_div", 0.0)),
            "prior_flow_sign": float(latest.get("prior_flow_sign", 0.0)),
            "div_score": float(latest.get("div_score", 0.0)),
            "trend_dir": int(latest.get("trend_dir", 0)),
            "trend_aligned": int(latest.get("trend_aligned", 0)),
            "trend_slope": float(latest.get("trend_slope", 0.0)),
            "stall_flag": int(latest.get("stall_flag", 0)),
            "event_bar_offset_min": 0,
        }
        return event

    # --- Internal helpers -------------------------------------------------

    def _start_new_minute(self, minute: int, tick: Tick) -> None:
        self.current_minute = minute
        self.current_bar = {
            "timestamp": minute * 60000,  # ms at minute start
            "open": tick.price,
            "high": tick.price,
            "low": tick.price,
            "close": tick.price,
            "volume": 0.0,
            "buy_qty": 0.0,
            "sell_qty": 0.0,
        }
        self._update_current_bar(tick)

    def _update_current_bar(self, tick: Tick) -> None:
        self.current_bar["high"] = max(self.current_bar["high"], tick.price)
        self.current_bar["low"] = min(self.current_bar["low"], tick.price)
        self.current_bar["close"] = tick.price
        self.current_bar["volume"] += tick.qty
        if tick.is_buyer_maker:
            self.current_bar["sell_qty"] += tick.qty
        else:
            self.current_bar["buy_qty"] += tick.qty

    def _finalize_current_bar(self) -> None:
        if not self.current_bar:
            return
        self._append_bar_with_features(self.current_bar)
        self.current_bar = {}

    def _append_bar_with_features(self, bar: Dict[str, Any]) -> None:
        self.bars.append(bar)
        self.bars_df = pd.DataFrame(self.bars)

        # ATR
        if len(self.bars_df) >= 2:
            highs = self.bars_df["high"].to_numpy()
            lows = self.bars_df["low"].to_numpy()
            closes = self.bars_df["close"].to_numpy()
            trs = []
            for i in range(1, len(self.bars_df)):
                tr = max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]),
                )
                trs.append(tr)
            tr_series = pd.Series(trs)
            atr_series = tr_series.rolling(self.atr_window).mean()
            atr_full = pd.Series([np.nan] + list(atr_series))
            self.bars_df["atr"] = atr_full
        else:
            self.bars_df["atr"] = np.nan

        # Volume stats
        vol_ma = self.bars_df["volume"].rolling(self.cfg.z_window, min_periods=max(10, self.cfg.z_window // 4)).mean()
        self.bars_df["rel_vol"] = self.bars_df["volume"] / (vol_ma + 1e-9)
        self.bars_df["atr_pct"] = self.bars_df["atr"] / (self.bars_df["close"].shift(1).abs() + 1e-9)

        # Imbalance raw
        self.bars_df["imbalance"] = self.bars_df["buy_qty"] - self.bars_df["sell_qty"]
        self.bars_df["imbalance_z"] = 0.0  # filled in detect_shockflip_signals
        # Trend placeholders
        self.bars_df["trend_dir"] = 0
        self.bars_df["trend_aligned"] = 0
        self.bars_df["trend_slope"] = 0.0
        self.bars_df["stall_flag"] = 0


def detect_shockflip_signals(df: pd.DataFrame, cfg: ShockFlipConfig) -> pd.DataFrame:
    """
    Apply ShockFlip-like logic on 1m bars.
    Expected columns: timestamp (ms), open/high/low/close, volume, buy_qty, sell_qty, atr.
    """
    if df.empty:
        return df

    df = df.copy()

    # 1) Imbalance z-score
    vol_ma = df["volume"].rolling(cfg.z_window, min_periods=max(10, cfg.z_window // 4)).mean()
    imb_raw = (df["buy_qty"] - df["sell_qty"]) / (vol_ma + 1e-9)
    mean = imb_raw.rolling(cfg.z_window, min_periods=max(10, cfg.z_window // 4)).mean()
    std = imb_raw.rolling(cfg.z_window, min_periods=max(10, cfg.z_window // 4)).std()
    df["imbalance_z"] = (imb_raw - mean) / (std + 1e-9)
    df["shockflip_z"] = df["imbalance_z"]

    # Dynamic thresholds are optional; default is the static research band.
    current_z_band = cfg.z_band
    if cfg.dynamic_thresholds.get("enabled", False):
        roll_z_high = (
            df["imbalance_z"]
            .abs()
            .rolling(cfg.z_window, min_periods=max(10, cfg.z_window // 4))
            .quantile(cfg.dynamic_thresholds.get("percentile", 0.99))
        )
        if not roll_z_high.empty:
            current_z_band = max(current_z_band, float(roll_z_high.iloc[-1]))

    long_condition = df["imbalance_z"] <= -current_z_band
    short_condition = df["imbalance_z"] >= current_z_band

    # 2) Shock filter using ATR
    if "atr" in df.columns:
        df["range_atr"] = (df["high"] - df["low"]) / (df["atr"].abs() + 1e-9)
        shock_cond = df["range_atr"] > cfg.jump_band
        long_condition = long_condition & shock_cond
        short_condition = short_condition & shock_cond
    else:
        df["range_atr"] = np.nan

    # 3) Location filter (Donchian extremes)
    if cfg.location_filter.get("require_extreme", False):
        d_win = cfg.location_filter.get("donchian_window", 120)
        d_low = df["low"].rolling(d_win, min_periods=max(10, d_win // 4)).min()
        d_high = df["high"].rolling(d_win, min_periods=max(10, d_win // 4)).max()
        is_low = df["low"] <= d_low * 1.001
        is_high = df["high"] >= d_high * 0.999
        long_condition = long_condition & is_low
        short_condition = short_condition & is_high
        df["donchian_loc"] = np.where(is_high, 1.0, np.where(is_low, -1.0, 0.0))
    else:
        df["donchian_loc"] = 0.0

    # 4) Simple div/flow placeholders
    df["prior_flow_sign"] = np.sign(df["imbalance_z"].shift(1).fillna(0.0))
    df["price_flow_div"] = df["prior_flow_sign"] * (df["close"] - df["close"].shift(1))
    df["div_score"] = df["imbalance_z"] * df["price_flow_div"]
    df["shock_intensity_z"] = df["range_atr"]

    # Trend proxies
    df["trend_slope"] = df["close"].diff().rolling(20, min_periods=5).mean().fillna(0.0)
    df["trend_dir"] = np.sign(df["trend_slope"]).astype(int)
    df["trend_aligned"] = (df["trend_dir"] == np.sign(df["imbalance_z"])).astype(int)
    df["stall_flag"] = (df["trend_slope"].abs() < 1e-6).astype(int)

    df["shockflip_signal"] = 0
    df.loc[long_condition, "shockflip_signal"] = 1
    df.loc[short_condition, "shockflip_signal"] = -1

    return df
