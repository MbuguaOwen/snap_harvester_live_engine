from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from binance.um_futures import UMFutures

from snap_harvester.logging_utils import get_logger


@dataclass
class ExecutionConfig:
    symbol: str
    quote_risk_per_trade: float
    use_oco_brackets: bool
    recv_window_ms: int
    max_retries: int
    order_timeout_sec: int
    testnet: bool
    api_key_env: str
    api_secret_env: str
    base_url: str


class BinanceExecutionClient:
    """
    Thin wrapper around Binance UM futures REST for live order entry.

    - Places entries and exchange-hosted SL/TP brackets (separate orders for safety).
    - Uses deterministic clientOrderId for idempotency across retries.
    - Exposes a sync helper to flatten ambiguous positions on startup.
    """

    def __init__(self, cfg: dict) -> None:
        bcfg = cfg.get("binance", {})
        self.config = ExecutionConfig(
            symbol=str(bcfg.get("symbol", "BTCUSDT")).upper(),
            quote_risk_per_trade=float(bcfg.get("quote_risk_per_trade", 0.0)),
            use_oco_brackets=bool(bcfg.get("use_oco_brackets", True)),
            recv_window_ms=int(bcfg.get("recv_window_ms", 5000)),
            max_retries=int(bcfg.get("max_retries", 3)),
            order_timeout_sec=int(bcfg.get("order_timeout_sec", 5)),
            testnet=bool(bcfg.get("testnet", False)),
            api_key_env=str(bcfg.get("api_key_env", "BINANCE_API_KEY")),
            api_secret_env=str(bcfg.get("api_secret_env", "BINANCE_API_SECRET")),
            base_url=str(bcfg.get("base_url", "https://fapi.binance.com")),
        )
        self.logger = get_logger("binance_exec")

        self.api_key = os.getenv(self.config.api_key_env)
        self.api_secret = os.getenv(self.config.api_secret_env)
        if not self.api_key or not self.api_secret:
            raise ValueError(
                f"Missing Binance API credentials. Set {self.config.api_key_env} and {self.config.api_secret_env}."
            )

        base_url = self.config.base_url
        if self.config.testnet:
            base_url = "https://testnet.binancefuture.com"
        self.client = UMFutures(
            key=self.api_key,
            secret=self.api_secret,
            base_url=base_url,
        )
        self._exchange_filters: Optional[Dict[str, Any]] = None

    # --- Connectivity / metadata ----------------------------------------

    def ensure_connection(self) -> None:
        """Raise if the exchange is unreachable."""
        for attempt in range(self.config.max_retries):
            try:
                self.client.ping()
                return
            except Exception as exc:  # pylint: disable=broad-except
                self.logger.warning("Binance ping failed (attempt %d/%d): %s", attempt + 1, self.config.max_retries, exc)
                time.sleep(1.0)
        raise RuntimeError("Binance connectivity failed")

    def _get_symbol_filters(self) -> Dict[str, Any]:
        """
        Fetch and cache the exchange filters for our symbol.

        Binance UM futures `exchangeInfo` does NOT take a `symbol` parameter.
        We call it without args and filter client-side. Some connector versions
        wrap the payload under "data", so handle both shapes.
        """
        if self._exchange_filters is not None:
            return self._exchange_filters

        info = self.client.exchange_info()
        payload = info.get("data", info)

        symbols = payload.get("symbols", [])
        symbol_info = next((s for s in symbols if s.get("symbol") == self.config.symbol), None)
        if not symbol_info:
            raise RuntimeError(f"Exchange info missing symbol={self.config.symbol}")

        filters = {f["filterType"]: f for f in symbol_info.get("filters", [])}
        self._exchange_filters = filters
        return filters

    # --- Sizing helpers --------------------------------------------------

    def _round_step(self, qty: float) -> float:
        filters = self._get_symbol_filters()
        lot = filters.get("LOT_SIZE", {})
        step = float(lot.get("stepSize", 0.001))
        min_qty = float(lot.get("minQty", step))
        max_qty = float(lot.get("maxQty", 1e12))
        stepped = math.floor(qty / step) * step
        stepped = max(min_qty, min(stepped, max_qty))
        return float(f"{stepped:.8f}")

    def _round_price(self, price: float) -> float:
        filters = self._get_symbol_filters()
        price_filter = filters.get("PRICE_FILTER", {})
        tick = float(price_filter.get("tickSize", 0.01))
        stepped = math.floor(price / tick + 1e-9) * tick
        return float(f"{stepped:.8f}")

    def calculate_qty(self, entry_price: float, r_unit: float, quote_risk: Optional[float] = None) -> float:
        quote = quote_risk if quote_risk is not None else self.config.quote_risk_per_trade
        if quote <= 0:
            raise ValueError("quote_risk_per_trade must be positive")
        if r_unit <= 0:
            raise ValueError("r_unit must be positive")
        raw_qty = quote / r_unit
        qty = self._round_step(raw_qty)
        if qty <= 0:
            raise ValueError(f"Quantity rounded to zero (raw={raw_qty})")
        return qty

    # --- Order entry -----------------------------------------------------

    def submit_entry_and_brackets(
        self,
        side: int,
        quantity: float,
        sl_dist: float,
        tp_dist: float,
        be_dist: float,
        client_order_id: str,
    ) -> Dict[str, Any]:
        """
        Place entry then SL/TP brackets on Binance.

        Returns a dict with fill_price and bracket order IDs.
        """
        if side not in (1, -1):
            raise ValueError(f"side must be +1/-1, got {side!r}")
        entry_side = "BUY" if side > 0 else "SELL"
        close_side = "SELL" if side > 0 else "BUY"

        order_info = self._get_order_by_client_id(client_order_id)
        entry_order_id: Optional[int] = None
        if order_info:
            entry_order_id = order_info.get("orderId")
        else:
            existing = self._find_existing_order(client_order_id)
            if existing:
                entry_order_id = existing["orderId"]
                order_info = self._get_order_by_id(entry_order_id)
            else:
                order_info = self._place_market_order(entry_side, quantity, client_order_id)
                entry_order_id = order_info.get("orderId")

        fill_info = self._wait_for_fill(entry_order_id, client_order_id)
        fill_price = self._extract_fill_price(fill_info)

        sl_price = self._round_price(fill_price - sl_dist if side > 0 else fill_price + sl_dist)
        tp_price = self._round_price(fill_price + tp_dist if side > 0 else fill_price - tp_dist)
        be_price = self._round_price(fill_price + be_dist if side > 0 else fill_price - be_dist)

        sl_resp, tp_resp = self._place_brackets(close_side, sl_price, tp_price)

        return {
            "fill_price": fill_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "be_price": be_price,
            "entry_order_id": entry_order_id,
            "sl_order_id": sl_resp.get("orderId") if sl_resp else None,
            "tp_order_id": tp_resp.get("orderId") if tp_resp else None,
            "client_order_id": client_order_id,
        }

    def _place_market_order(self, side: str, quantity: float, client_order_id: str) -> Dict[str, Any]:
        params = {
            "symbol": self.config.symbol,
            "side": side,
            "type": "MARKET",
            "quantity": quantity,
            "newClientOrderId": client_order_id,
            "recvWindow": self.config.recv_window_ms,
        }
        for attempt in range(self.config.max_retries):
            try:
                return self.client.new_order(**params)
            except Exception as exc:  # pylint: disable=broad-except
                self.logger.warning(
                    "Entry order failed (attempt %d/%d): %s", attempt + 1, self.config.max_retries, exc
                )
                time.sleep(0.5)
        raise RuntimeError("Entry order failed after retries")

    def _find_existing_order(self, client_order_id: str) -> Optional[Dict[str, Any]]:
        try:
            orders = self.client.get_open_orders(
                symbol=self.config.symbol,
                recvWindow=self.config.recv_window_ms,
            )
        except Exception:  # pylint: disable=broad-except
            return None
        for order in orders:
            if order.get("clientOrderId") == client_order_id:
                return order
        return None

    def _get_order_by_client_id(self, client_order_id: str) -> Optional[Dict[str, Any]]:
        try:
            order = self.client.query_order(
                symbol=self.config.symbol,
                origClientOrderId=client_order_id,
                recvWindow=self.config.recv_window_ms,
            )
            return order
        except Exception:
            return None

    def _get_order_by_id(self, order_id: int) -> Dict[str, Any]:
        return self.client.query_order(
            symbol=self.config.symbol,
            orderId=order_id,
            recvWindow=self.config.recv_window_ms,
        )

    def _wait_for_fill(self, order_id: Optional[int], client_order_id: str) -> Dict[str, Any]:
        deadline = time.time() + self.config.order_timeout_sec
        last: Dict[str, Any] = {}
        while time.time() < deadline:
            try:
                if order_id is not None:
                    last = self._get_order_by_id(order_id)
                else:
                    last = self.client.query_order(
                        symbol=self.config.symbol,
                        origClientOrderId=client_order_id,
                        recvWindow=self.config.recv_window_ms,
                    )
            except Exception as exc:  # pylint: disable=broad-except
                self.logger.warning("Get order failed for %s: %s", client_order_id, exc)
                time.sleep(0.25)
                continue

            if last.get("status") in ("FILLED", "PARTIALLY_FILLED"):
                return last
            time.sleep(0.25)
        raise TimeoutError(f"Order {client_order_id} not filled within timeout")

    def _extract_fill_price(self, order: Dict[str, Any]) -> float:
        for key in ("avgPrice", "price", "stopPrice"):
            try:
                val = float(order.get(key, 0))
            except (TypeError, ValueError):
                continue
            if val > 0:
                return val
        try:
            cquote = float(order.get("cumQuote", 0))
            executed = float(order.get("executedQty", 0))
            if executed > 0:
                return cquote / executed
        except (TypeError, ValueError, ZeroDivisionError):
            pass

        try:
            trades = self.client.user_trades(
                symbol=self.config.symbol,
                orderId=order.get("orderId"),
                recvWindow=self.config.recv_window_ms,
            )
            if trades:
                return float(trades[-1].get("price"))
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.warning("Unable to fetch trade fills: %s", exc)
        raise RuntimeError("Could not extract fill price from order response")

    def _place_brackets(self, close_side: str, sl_price: float, tp_price: float) -> tuple[Dict[str, Any], Dict[str, Any]]:
        sl_params = {
            "symbol": self.config.symbol,
            "side": close_side,
            "type": "STOP_MARKET",
            "stopPrice": sl_price,
            "closePosition": True,
            "recvWindow": self.config.recv_window_ms,
        }
        tp_params = {
            "symbol": self.config.symbol,
            "side": close_side,
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": tp_price,
            "closePosition": True,
            "recvWindow": self.config.recv_window_ms,
        }
        sl_resp: Dict[str, Any] = {}
        tp_resp: Dict[str, Any] = {}
        try:
            sl_resp = self.client.new_order(**sl_params)
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.error("Failed to place stop-loss: %s", exc)
        try:
            tp_resp = self.client.new_order(**tp_params)
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.error("Failed to place take-profit: %s", exc)
        return sl_resp, tp_resp

    # --- State sync ------------------------------------------------------

    def sync_open_positions(self) -> Dict[str, Any]:
        """Return current net position and open orders for the symbol."""
        try:
            pos_info = self.client.get_position_risk(
                symbol=self.config.symbol,
                recvWindow=self.config.recv_window_ms,
            )
            position_amt = float(pos_info[0]["positionAmt"]) if pos_info else 0.0
            entry_price = float(pos_info[0].get("entryPrice", 0.0)) if pos_info else 0.0
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.error("Failed to fetch position risk: %s", exc)
            position_amt = 0.0
            entry_price = 0.0
        try:
            open_orders = self.client.get_open_orders(
                symbol=self.config.symbol,
                recvWindow=self.config.recv_window_ms,
            )
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.error("Failed to fetch open orders: %s", exc)
            open_orders = []
        return {"position_amt": position_amt, "entry_price": entry_price, "open_orders": open_orders}

    def flatten_all_positions(self) -> None:
        """Market-close any residual position and cancel open orders."""
        state = self.sync_open_positions()
        pos = state["position_amt"]
        if abs(pos) > 0:
            side = "SELL" if pos > 0 else "BUY"
            qty = abs(pos)
            self.logger.warning("Flattening stray position %.6f via %s", pos, side)
            try:
                self.client.new_order(
                    symbol=self.config.symbol,
                    side=side,
                    type="MARKET",
                    quantity=qty,
                    reduceOnly=True,
                    recvWindow=self.config.recv_window_ms,
                )
            except Exception as exc:  # pylint: disable=broad-except
                self.logger.error("Failed to flatten position: %s", exc)
        if state["open_orders"]:
            for order in state["open_orders"]:
                try:
                    self.client.cancel_order(
                        symbol=self.config.symbol,
                        orderId=order["orderId"],
                        recvWindow=self.config.recv_window_ms,
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    self.logger.error("Failed to cancel order %s: %s", order.get("orderId"), exc)
