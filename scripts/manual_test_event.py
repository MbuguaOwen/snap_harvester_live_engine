from __future__ import annotations

import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure repository root is on sys.path when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snap_harvester.live.app import ShockFlipEventFeed, run_live_engine  # type: ignore


def main() -> None:
    config_path = "configs/snap_harvester_live_btc.yaml"

    # Shared queue: we push fake ShockFlip events into this
    feed = ShockFlipEventFeed()

    # Start live engine in a background thread
    t = threading.Thread(
        target=run_live_engine,
        args=(config_path, feed),
        daemon=True,
    )
    t.start()
    print("Live engine thread started. Waiting for Binance feed to warm up...")

    # Give the feed time to backfill + receive at least one live kline.
    warmup_sec = 45
    for i in range(warmup_sec):
        time.sleep(1)
        if i in (10, 20, 30):
            print(f"... still warming up ({i}s)")

    # Synthetic ShockFlip event matching the live engine schema
    now = datetime.now(timezone.utc)
    fake_event = {
        "timestamp": now.isoformat(),
        "symbol": "BTCUSDT",
        "side": 1,  # +1 long, -1 short
        "close": 50000.0,  # rough price placeholder for sizing
        "atr": 200.0,  # positive ATR for risk unit
    }

    print("Pushing synthetic ShockFlip event to live engine...")
    feed.put(fake_event)

    # Keep process alive to observe open/close + Telegram HUD
    time.sleep(120)
    print("Manual test finished; check HUD / Binance testnet logs.")


if __name__ == "__main__":
    main()
