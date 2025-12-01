from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict

# Ensure repository root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient  # type: ignore

from snap_harvester.live.app import ShockFlipEventFeed, run_live_engine  # type: ignore
from snap_harvester.live.shockflip_live import ShockFlipDetector, Tick  # type: ignore


def main() -> None:
    config_path = "configs/snap_harvester_live_btc.yaml"
    symbol = "BTCUSDT"

    # Shared event feed between detector and trading engine
    event_feed = ShockFlipEventFeed()

    # Start live engine in background
    engine_thread = threading.Thread(
        target=run_live_engine,
        args=(config_path, event_feed),
        daemon=True,
    )
    engine_thread.start()
    print("Snap Harvester live engine started.")

    detector = ShockFlipDetector(symbol=symbol)
    tick_count = 10

    # UMFuturesWebsocketClient on_message signature is (client, message)
    def handle_aggtrade(_, msg: Dict[str, Any]) -> None:
        nonlocal tick_count
        try:
            tick = Tick(
                timestamp=msg["T"],
                price=float(msg["p"]),
                qty=float(msg["q"]),
                is_buyer_maker=msg["m"],
            )
        except (KeyError, TypeError, ValueError):
            return

        tick_count += 1
        if tick_count % 100 == 0:
            print(f"[DEBUG] Received {tick_count} aggTrades so far")

        event = detector.update(tick)
        if event is not None:
            print("ShockFlip LIVE event:", event)
            event_feed.put(event)

    # Websocket client (no testnet kwarg in this version)
    ws = UMFuturesWebsocketClient(on_message=handle_aggtrade)

    # Subscribe to aggTrade for BTCUSDT
    ws.agg_trade(symbol=symbol.lower())

    # Keep process alive while engine + WS threads run
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping websocket...")
        ws.stop()
        print("Done.")


if __name__ == "__main__":
    main()
