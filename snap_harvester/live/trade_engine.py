from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Literal, Optional

import pandas as pd


TradeStatus = Literal["open", "be_locked", "closed"]


@dataclass
class Trade:
    id: str
    symbol: str
    direction: int  # +1 long, -1 short
    entry_ts: pd.Timestamp
    entry_price: float
    atr: float
    r_unit: float
    sl_price: float
    be_price: float
    tp_price: float
    horizon_bars: int
    bars_held: int = 0
    state: TradeStatus = "open"
    hit_be: bool = False
    hit_tp: bool = False
    hit_sl: bool = False
    mfe_r: float = 0.0
    mae_r: float = 0.0
    r_final: Optional[float] = None
    exit_ts: Optional[pd.Timestamp] = None
    p_hat: Optional[float] = None
    meta: Optional[Dict[str, Any]] = None


class TradeEngine:
    """
    Streaming implementation of the Snap Harvester barrier logic.

    Mirrors the logic in snap_harvester.meta_builder._simulate_snap_trade:

    - Risk unit = risk_k_atr * ATR(event_bar)
    - SL distance  = sl_r_multiple * risk_unit
    - TP distance  = tp_r_multiple * risk_unit
    - BE distance  = be_r_multiple * risk_unit   (or be_k_atr * ATR)
    - Horizon      = horizon_bars
    - After BE: no losses (SL becomes BE).
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        risk_cfg = cfg["risk"]
        self.risk_k_atr = float(risk_cfg["risk_k_atr"])
        self.be_k_atr = float(risk_cfg.get("be_k_atr", 0.0))
        self.tp_k_atr = float(risk_cfg.get("tp_k_atr", 0.0))
        self.sl_r_mult = float(risk_cfg.get("sl_r_multiple", 1.0))
        self.tp_r_mult = risk_cfg.get("tp_r_multiple")
        self.be_r_mult = risk_cfg.get("be_r_multiple")
        self.horizon = int(risk_cfg["horizon_bars"])
        self.target_r = risk_cfg.get("target_r_multiple")

        data_cfg = cfg["data"]
        self.bars_time_col = data_cfg.get("bars_time_col", "timestamp")
        self.bars_high_col = data_cfg.get("bars_high_col", "high")
        self.bars_low_col = data_cfg.get("bars_low_col", "low")
        self.bars_close_col = data_cfg.get("bars_close_col", "close")

        self._trades: Dict[str, Trade] = {}

    # --- Trade lifecycle -------------------------------------------------

    def make_trade_id(self, event: Dict[str, Any]) -> str:
        """Public helper to create deterministic trade IDs for a raw event."""
        return self._make_trade_id(event)

    def _make_trade_id(self, event: Dict[str, Any]) -> str:
        ts = pd.to_datetime(event.get("event_bar_time", event.get("timestamp")), utc=True)
        sym = str(event.get("symbol", "BTCUSDT"))
        idx = int(event.get("idx", 0))
        return f"{sym}-{ts.isoformat()}-{idx}"

    def open_trade(self, event: Dict[str, Any], p_hat: float, meta_row: Dict[str, Any]) -> Trade:
        side = int(event.get("side", 0))
        if side not in (1, -1):
            raise ValueError(f"Expected side in {{+1,-1}}, got {side!r}")

        dir_sign = 1 if side > 0 else -1
        entry_ts = pd.to_datetime(event.get("event_bar_time", event.get("timestamp")), utc=True)
        entry_price = float(event.get("close", event.get("price")))
        atr = float(event["atr"])
        if atr <= 0:
            raise ValueError("ATR must be positive")

        risk_dist = self.risk_k_atr * atr
        sl_dist = risk_dist * self.sl_r_mult

        if self.be_r_mult is not None:
            be_dist = risk_dist * float(self.be_r_mult)
        else:
            be_dist = self.be_k_atr * atr

        if self.tp_r_mult is not None:
            tp_dist = risk_dist * float(self.tp_r_mult)
        else:
            tp_dist = self.tp_k_atr * atr

        sl_price = entry_price - dir_sign * sl_dist
        be_price = entry_price + dir_sign * be_dist
        tp_price = entry_price + dir_sign * tp_dist

        r_unit = risk_dist
        trade_id = self._make_trade_id(event)

        trade = Trade(
            id=trade_id,
            symbol=str(event.get("symbol", "BTCUSDT")),
            direction=dir_sign,
            entry_ts=entry_ts,
            entry_price=entry_price,
            atr=atr,
            r_unit=r_unit,
            sl_price=sl_price,
            be_price=be_price,
            tp_price=tp_price,
            horizon_bars=self.horizon,
            p_hat=p_hat,
            meta=meta_row,
        )
        self._trades[trade_id] = trade
        return trade

    def on_new_event(self, event: Dict[str, Any], p_hat: float, meta_row: Dict[str, Any]) -> Trade:
        """Create and register a new trade."""
        return self.open_trade(event, p_hat, meta_row)

    def open_trade_from_live_fill(
        self,
        event: Dict[str, Any],
        p_hat: float,
        fill_price: float,
        r_unit: float,
        sl_price: float,
        be_price: float,
        tp_price: float,
        meta_row: Dict[str, Any] | None = None,
    ) -> Trade:
        """
        Create a trade using the actual exchange fill + externally computed levels.

        This bypasses internal SL/TP recomputation so live execution uses the real
        fill price returned by Binance.
        """
        side = int(event.get("side", 0))
        if side not in (1, -1):
            raise ValueError(f"Expected side in {{+1,-1}}, got {side!r}")
        if r_unit <= 0:
            raise ValueError(f"Risk unit must be positive, got {r_unit!r}")

        dir_sign = 1 if side > 0 else -1
        entry_ts = pd.to_datetime(event.get("event_bar_time", event.get("timestamp")), utc=True)
        trade_id = self._make_trade_id(event)

        trade = Trade(
            id=trade_id,
            symbol=str(event.get("symbol", "BTCUSDT")),
            direction=dir_sign,
            entry_ts=entry_ts,
            entry_price=float(fill_price),
            atr=float(event["atr"]),
            r_unit=float(r_unit),
            sl_price=float(sl_price),
            be_price=float(be_price),
            tp_price=float(tp_price),
            horizon_bars=self.horizon,
            p_hat=p_hat,
            meta=meta_row,
        )
        self._trades[trade_id] = trade
        return trade

    # --- Streaming bar updates -------------------------------------------

    def on_new_bar(self, bar: Dict[str, Any]) -> List[Trade]:
        """
        Update all open trades with a new bar.

        Returns list of trades that were closed on this bar.
        """
        closed: List[Trade] = []
        ts = pd.to_datetime(bar[self.bars_time_col], utc=True)
        high = float(bar[self.bars_high_col])
        low = float(bar[self.bars_low_col])
        close = float(bar[self.bars_close_col])

        for trade in list(self._trades.values()):
            if trade.state == "closed":
                continue

            trade.bars_held += 1
            dir_sign = trade.direction
            risk_dist = trade.r_unit
            entry = trade.entry_price

            if dir_sign == 1:
                fav = (high - entry) / max(risk_dist, 1e-8)
                adv = (low - entry) / max(risk_dist, 1e-8)
                cross_tp = high >= trade.tp_price
                cross_be = high >= trade.be_price
                cross_sl = low <= trade.sl_price
                cross_be_stop = low <= entry
            else:
                fav = (entry - low) / max(risk_dist, 1e-8)
                adv = (entry - high) / max(risk_dist, 1e-8)
                cross_tp = low <= trade.tp_price
                cross_be = low <= trade.be_price
                cross_sl = high >= trade.sl_price
                cross_be_stop = high >= entry

            trade.mfe_r = max(trade.mfe_r, fav)
            trade.mae_r = min(trade.mae_r, adv)

            if trade.state == "open":
                if cross_tp:
                    trade.hit_tp = True
                    trade.r_final = (trade.tp_price - entry) * dir_sign / max(risk_dist, 1e-8)
                    trade.exit_ts = ts
                    trade.state = "closed"
                elif cross_sl:
                    trade.hit_sl = True
                    trade.r_final = (trade.sl_price - entry) * dir_sign / max(risk_dist, 1e-8)
                    trade.exit_ts = ts
                    trade.state = "closed"
                elif cross_be:
                    trade.hit_be = True
                    trade.state = "be_locked"
            elif trade.state == "be_locked":
                if cross_tp:
                    trade.hit_tp = True
                    trade.r_final = (trade.tp_price - entry) * dir_sign / max(risk_dist, 1e-8)
                    trade.exit_ts = ts
                    trade.state = "closed"
                elif cross_be_stop:
                    trade.hit_sl = True
                    trade.r_final = 0.0
                    trade.exit_ts = ts
                    trade.state = "closed"

            # Horizon exit
            if trade.state != "closed" and trade.bars_held >= trade.horizon_bars:
                if dir_sign == 1:
                    pnl_r = (close - entry) / max(risk_dist, 1e-8)
                else:
                    pnl_r = (entry - close) / max(risk_dist, 1e-8)
                if trade.hit_be and pnl_r < 0.0:
                    pnl_r = 0.0
                trade.r_final = pnl_r
                trade.exit_ts = ts
                trade.state = "closed"

            if trade.state == "closed":
                closed.append(trade)
                # Do not delete immediately; caller can decide. For now we keep it.

        return closed

    # --- Introspection / export -----------------------------------------

    def open_trades(self) -> Dict[str, Trade]:
        return {k: v for k, v in self._trades.items() if v.state != "closed"}

    @staticmethod
    def to_record(trade: Trade) -> Dict[str, Any]:
        rec = asdict(trade)
        # Flatten timestamps
        rec["entry_ts"] = trade.entry_ts.isoformat() if trade.entry_ts is not None else None
        rec["exit_ts"] = trade.exit_ts.isoformat() if trade.exit_ts is not None else None
        return rec
