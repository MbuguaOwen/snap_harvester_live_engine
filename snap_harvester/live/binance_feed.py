from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import datetime, timezone
from queue import Empty, Queue
from typing import Optional

import pandas as pd
import websockets
from binance.um_futures import UMFutures

from snap_harvester.logging_utils import get_logger
from .feed import Bar


class BinanceBarFeed:
    """
    Streams 1m Binance futures klines and exposes them as Bar objects.

    - Uses the public futures websocket for the configured symbol/interval.
    - Backfills a small recent window via REST on reconnect to avoid gaps.
    - Tracks staleness so the orchestrator can pause entries when the feed is cold.
    """

    def __init__(self, cfg: dict) -> None:
        binance_cfg = cfg.get("binance", {})
        self.symbol = str(binance_cfg.get("symbol", "BTCUSDT")).upper()
        self.base_url = binance_cfg.get("base_url", "https://fapi.binance.com")
        self.ws_url = binance_cfg.get("ws_url", "wss://fstream.binance.com/ws")
        self.interval = binance_cfg.get("kline_interval", "1m")
        self.heartbeat_sec = int(binance_cfg.get("heartbeat_sec", 5))
        self.stale_timeout_sec = int(binance_cfg.get("stale_timeout_sec", 30))
        self.use_kline_stream = bool(binance_cfg.get("use_kline_stream", True))
        self.backfill_bars = int(binance_cfg.get("backfill_bars", 120))

        self._logger = get_logger("binance_feed")
        self._rest = UMFutures(base_url=self.base_url)
        self._queue: Queue[Bar] = Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_bar_time: Optional[pd.Timestamp] = None
        self._last_heartbeat = time.time()
        self.last_bar_ts: Optional[int] = None  # unix seconds of latest closed bar
        self.feed_stale = False
        self._seen_bars: set[pd.Timestamp] = set()

    # Public API ----------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._logger.info("BinanceBarFeed started for %s %s", self.symbol, self.interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._logger.info("BinanceBarFeed stopped")

    def get_bar(self, timeout: float | None = 0.0) -> Optional[Bar]:
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    def is_stale(self) -> bool:
        if self._last_bar_time is None:
            self.feed_stale = True
            return True
        now = datetime.now(timezone.utc)
        delta = now - self._last_bar_time.to_pydatetime()
        self.feed_stale = delta.total_seconds() > self.stale_timeout_sec
        return self.feed_stale

    # Internal ------------------------------------------------------------

    def _run(self) -> None:
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        while not self._stop.is_set():
            try:
                await self._connect_and_stream()
            except Exception as exc:  # pylint: disable=broad-except
                self._logger.exception("Binance feed error: %s", exc)
                await asyncio.sleep(2.0)

    async def _connect_and_stream(self) -> None:
        stream = f"{self.symbol.lower()}@kline_{self.interval}"
        url = f"{self.ws_url.rstrip('/')}/{stream}"

        # Backfill before each connect to close gaps.
        self._backfill_recent_bars()

        self._logger.info("Connecting websocket: %s", url)
        async with websockets.connect(
            url,
            ping_interval=self.heartbeat_sec,
            ping_timeout=self.heartbeat_sec * 2,
        ) as ws:
            async for msg in ws:
                if self._stop.is_set():
                    break
                self._last_heartbeat = time.time()
                try:
                    payload = json.loads(msg)
                except json.JSONDecodeError:
                    continue
                kline = payload.get("k")
                if not kline:
                    continue
                # Only emit closed klines to avoid duplicates.
                if not kline.get("x"):
                    continue
                bar_time = pd.to_datetime(kline["t"], unit="ms", utc=True)
                bar_ts = int(bar_time.timestamp())
                close_price = float(kline["c"])
                self.last_bar_ts = bar_ts
                self._logger.info(
                    "KLINE_TICK symbol=%s ts=%s close=%.2f",
                    self.symbol,
                    bar_ts,
                    close_price,
                )
                bar = Bar(
                    timestamp=bar_time,
                    open=float(kline["o"]),
                    high=float(kline["h"]),
                    low=float(kline["l"]),
                    close=close_price,
                    volume=float(kline.get("v", 0.0)),
                )
                self._enqueue_bar(bar)

    def _backfill_recent_bars(self) -> None:
        try:
            data = self._rest.klines(
                symbol=self.symbol,
                interval=self.interval,
                limit=self.backfill_bars,
            )
        except Exception as exc:  # pylint: disable=broad-except
            self._logger.warning("Kline backfill failed: %s", exc)
            return

        for raw in data or []:
            bar_time = pd.to_datetime(raw[0], unit="ms", utc=True)
            bar = Bar(
                timestamp=bar_time,
                open=float(raw[1]),
                high=float(raw[2]),
                low=float(raw[3]),
                close=float(raw[4]),
                volume=float(raw[5]),
            )
            self._enqueue_bar(bar)

    def _enqueue_bar(self, bar: Bar) -> None:
        if bar.timestamp in self._seen_bars:
            return
        self._seen_bars.add(bar.timestamp)
        # Keep the seen set bounded.
        if len(self._seen_bars) > 2000:
            self._seen_bars = set(list(self._seen_bars)[-1000:])

        self._last_bar_time = bar.timestamp
        self.feed_stale = False
        self._queue.put(bar)
