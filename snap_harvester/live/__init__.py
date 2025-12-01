"""
Live Snap Harvester v2 engine.

This package provides:
- Data feeds (replay + live stubs)
- Meta-feature construction for live events
- Router that applies the frozen 2024 BTC meta-model
- Trade engine for paper/live trading
- A runner entrypoint used via: python -m snap_harvester.live.runner
"""
from .feed import ReplayBarFeed, ReplayEventFeed
from .meta_builder import LiveMetaBuilder
from .router import LiveRouter
from .trade_engine import Trade, TradeEngine
from .binance_feed import BinanceBarFeed
from .execution import BinanceExecutionClient
from .telegram_hud import TelegramHUD
from .app import run_live_engine

__all__ = [
    "ReplayBarFeed",
    "ReplayEventFeed",
    "LiveMetaBuilder",
    "LiveRouter",
    "Trade",
    "TradeEngine",
    "BinanceBarFeed",
    "BinanceExecutionClient",
    "TelegramHUD",
    "run_live_engine",
]
