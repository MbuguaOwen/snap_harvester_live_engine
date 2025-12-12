"""
Lightweight ShockFlip feature stack vendored for live use.

This mirrors the research `core` modules so the live engine can run without
mounting the full research repo. It includes:
- Tick -> 1m bar aggregation with buy/sell volume
- Core + hypothesis feature builders
- ShockFlipConfig and detection logic
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from snap_harvester.utils.ticks import get_tick_size


# --------------------------------------------------------------------------- #
# Tick -> bar aggregation
# --------------------------------------------------------------------------- #

def resample_ticks_to_bars(
    ticks: pd.DataFrame,
    timeframe: str = "1min",
    symbol: Optional[str] = None,
    tick_size: Optional[float] = None,
) -> pd.DataFrame:
    """Aggregate ticks into OHLCV bars with buy/sell volume."""
    if ticks.empty:
        return pd.DataFrame()

    resolved_tick = tick_size
    if resolved_tick is None and symbol:
        resolved_tick = get_tick_size(symbol)

    df = ticks.copy()
    if resolved_tick is not None:
        # Snap prices to the exchange tick grid
        price_arr = df["price"].to_numpy()
        tick_arr = np.rint(price_arr / resolved_tick).astype("int64")
        df["tick"] = tick_arr
        df["price"] = tick_arr.astype("float64") * resolved_tick

    df = df.set_index("ts")

    ohlc = df["price"].resample(timeframe).ohlc()
    vol = df["qty"].resample(timeframe).sum().rename("volume")

    # buy_qty = aggressor is buyer (is_buyer_maker=False)
    buy_qty = df.loc[~df["is_buyer_maker"], "qty"].resample(timeframe).sum().rename("buy_qty")
    # sell_qty = aggressor is seller (is_buyer_maker=True)
    sell_qty = df.loc[df["is_buyer_maker"], "qty"].resample(timeframe).sum().rename("sell_qty")

    bars = pd.concat([ohlc, vol, buy_qty, sell_qty], axis=1)
    bars = bars.dropna(subset=["open", "high", "low", "close"])
    bars[["buy_qty", "sell_qty"]] = bars[["buy_qty", "sell_qty"]].fillna(0.0)
    bars = bars.sort_index()

    return bars.reset_index().rename(columns={"ts": "timestamp"})


# --------------------------------------------------------------------------- #
# Feature builders (copied from research)
# --------------------------------------------------------------------------- #

def rolling_zscore(series: pd.Series, window: int, eps: float = 1e-9) -> pd.Series:
    """Causal rolling z-score (includes current bar in window)."""
    roll_mean = series.rolling(window, min_periods=window).mean()
    roll_std = series.rolling(window, min_periods=window).std(ddof=0)
    z = (series - roll_mean) / (roll_std + eps)
    return z


def compute_orderflow_features(bars: pd.DataFrame, eps: float = 1e-9) -> pd.DataFrame:
    """Add Q+, Q-, delta, imbalance, and z-scores to bars."""
    df = bars.copy()
    df["q_plus"] = df["buy_qty"].astype(float)
    df["q_minus"] = df["sell_qty"].astype(float)
    df["delta"] = df["q_plus"] - df["q_minus"]
    df["imbalance"] = (df["q_plus"] - df["q_minus"]) / (df["q_plus"] + df["q_minus"] + eps)
    return df


def compute_atr(bars: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Compute ATR(window) using classic True Range definition."""
    df = bars.copy()
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["tr"] = tr
    df["atr"] = tr.rolling(window, min_periods=window).mean()
    return df


def compute_donchian(bars: pd.DataFrame, window: int = 120, eps: float = 1e-9) -> pd.DataFrame:
    """Compute Donchian channel and location."""
    df = bars.copy()
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    donchian_high = high.rolling(window, min_periods=window).max()
    donchian_low = low.rolling(window, min_periods=window).min()

    df["donchian_high"] = donchian_high
    df["donchian_low"] = donchian_low

    rng = donchian_high - donchian_low
    df["donchian_loc"] = (close - donchian_low) / (rng + eps)
    df["at_upper_extreme"] = high >= donchian_high
    df["at_lower_extreme"] = low <= donchian_low
    return df


def add_core_features(
    bars: pd.DataFrame,
    z_window: int = 240,
    atr_window: int = 60,
    donchian_window: int = 120,
) -> pd.DataFrame:
    """Convenience: compute orderflow, imbalance z, ATR, Donchian."""
    df = compute_orderflow_features(bars)
    df["imbalance_z"] = rolling_zscore(df["imbalance"], window=z_window)
    df = compute_atr(df, window=atr_window)
    df = compute_donchian(df, window=donchian_window)
    return df


def compute_prior_flow_sign(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    H1: Prior order-flow sign before the ShockFlip.
    """
    out = df.copy()
    if "delta" in out.columns:
        flow = out["delta"].astype(float)
    elif "q_plus" in out.columns and "q_minus" in out.columns:
        flow = out["q_plus"].astype(float) - out["q_minus"].astype(float)
    else:
        raise KeyError("compute_prior_flow_sign expected 'delta' or ('q_plus','q_minus') in the DataFrame.")
    roll = flow.rolling(window=window, min_periods=1).sum()
    out["prior_flow_sum"] = roll
    out["prior_flow_sign"] = np.sign(roll).astype(int)
    return out


def compute_price_flow_divergence(
    df: pd.DataFrame,
    window: int = 60,
    price_col: str = "close",
) -> pd.DataFrame:
    """H2: Price/flow divergence over a window."""
    out = df.copy()
    if price_col not in out.columns:
        raise KeyError(f"compute_price_flow_divergence expected '{price_col}' column.")

    if "delta" in out.columns:
        flow = out["delta"].astype(float)
    elif "q_plus" in out.columns and "q_minus" in out.columns:
        flow = out["q_plus"].astype(float) - out["q_minus"].astype(float)
    else:
        raise KeyError("compute_price_flow_divergence expected 'delta' or ('q_plus','q_minus').")

    price = np.log(out[price_col].astype(float).clip(lower=1e-12))
    price_chg = price.diff(window)
    flow_chg = flow.rolling(window=window, min_periods=1).sum()

    def zscore(s: pd.Series) -> pd.Series:
        m = s.mean()
        v = s.std(ddof=0)
        if not np.isfinite(v) or v == 0.0:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - m) / v

    price_z = zscore(price_chg)
    flow_z = zscore(flow_chg)

    out["price_chg_window"] = price_chg
    out["flow_chg_window"] = flow_chg
    out["price_flow_div"] = price_z - flow_z
    return out


def compute_atr_percentile(df: pd.DataFrame, window: int = 5000) -> pd.DataFrame:
    """H3: ATR percentile (simple cross-sectional percentile)."""
    out = df.copy()
    if "atr" not in out.columns:
        raise KeyError("compute_atr_percentile expected 'atr' column to exist.")
    atr = out["atr"].astype(float)
    out["atr_pct"] = atr.rank(pct=True).astype(float)
    return out


def add_hypothesis_features(
    df: pd.DataFrame,
    prior_flow_window: int = 60,
    div_window: int = 60,
    atr_pct_window: int = 5000,
) -> pd.DataFrame:
    """Apply H1–H3 transforms on top of the core features."""
    out = compute_prior_flow_sign(df, window=prior_flow_window)
    out = compute_price_flow_divergence(out, window=div_window)
    out = compute_atr_percentile(out, window=atr_pct_window)
    return out


# --------------------------------------------------------------------------- #
# ShockFlip detection
# --------------------------------------------------------------------------- #

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


def detect_shockflip_signals(df: pd.DataFrame, cfg: ShockFlipConfig) -> pd.DataFrame:
    """
    Apply ShockFlip logic on the provided feature frame and add `shockflip_signal`.
    """
    df = df.copy()

    # Signal source
    if cfg.source == "imbalance":
        if "imbalance_z" not in df.columns:
            vol_ma = df["volume"].rolling(cfg.z_window).mean()
            imb_raw = (df["buy_qty"] - df["sell_qty"]) / (vol_ma + 1e-9)
            mean = imb_raw.rolling(cfg.z_window).mean()
            std = imb_raw.rolling(cfg.z_window).std()
            df["imbalance_z"] = (imb_raw - mean) / (std + 1e-9)
        signal_col = "imbalance_z"
    else:
        signal_col = "imbalance_z"

    current_z_band = cfg.z_band
    if cfg.dynamic_thresholds.get("enabled", False):
        roll_z_high = df[signal_col].abs().rolling(cfg.z_window).quantile(cfg.dynamic_thresholds.get("percentile", 0.99))
        # Optionally adjust threshold; for now we keep the configured band.
        if roll_z_high.notna().any():
            current_z_band = max(current_z_band, roll_z_high.iloc[-1])

    long_condition = df[signal_col] <= -current_z_band
    short_condition = df[signal_col] >= current_z_band

    if "atr" in df.columns:
        rel_range = (df["high"] - df["low"]) / (df["atr"] + 1e-9)
        shock_cond = rel_range > cfg.jump_band
        long_condition = long_condition & shock_cond
        short_condition = short_condition & shock_cond

    if cfg.location_filter.get("require_extreme", False):
        d_win = cfg.location_filter.get("donchian_window", 120)
        d_low = df["low"].rolling(d_win).min()
        d_high = df["high"].rolling(d_win).max()
        is_low = df["low"] <= d_low * 1.001
        is_high = df["high"] >= d_high * 0.999
        long_condition = long_condition & is_low
        short_condition = short_condition & is_high

    df["shockflip_signal"] = 0
    df.loc[long_condition, "shockflip_signal"] = 1
    df.loc[short_condition, "shockflip_signal"] = -1
    return df
