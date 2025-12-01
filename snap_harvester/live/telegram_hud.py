from __future__ import annotations

import os
from typing import Any, Dict

import requests

from snap_harvester.logging_utils import get_logger
from .trade_engine import Trade


class TelegramHUD:
    """
    Minimal Telegram alerting layer used to surface curated live events.
    """

    def __init__(self, cfg: dict) -> None:
        tcfg = cfg.get("telegram", {})
        self.enabled = bool(tcfg.get("enabled", True))
        self.bot_token_env = str(tcfg.get("bot_token_env", "TELEGRAM_BOT_TOKEN"))
        self.chat_id_env = str(tcfg.get("chat_id_env", "TELEGRAM_CHAT_ID"))
        self.send_events = bool(tcfg.get("send_events", False))
        self.send_trade_open = bool(tcfg.get("send_trade_open", True))
        self.send_trade_close = bool(tcfg.get("send_trade_close", True))
        self.send_health = bool(tcfg.get("send_health", True))
        self.daily_summary_hour_utc = int(tcfg.get("daily_summary_hour_utc", 21))

        self.bot_token = os.getenv(self.bot_token_env)
        self.chat_id = os.getenv(self.chat_id_env)
        self.logger = get_logger("telegram_hud")

    # --- Core send -------------------------------------------------------

    def send_message(self, text: str, parse_mode: str = "Markdown") -> None:
        if not self.enabled:
            return
        if not self.bot_token or not self.chat_id:
            self.logger.warning("Telegram credentials missing; skipping message")
            return
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }
        try:
            resp = requests.post(url, json=payload, timeout=5)
            if not resp.ok:
                self.logger.warning("Telegram send failed: %s %s", resp.status_code, resp.text)
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.warning("Telegram send exception: %s", exc)

    # --- High-level notifications ---------------------------------------

    def notify_health(self, status: str, details: Dict[str, Any]) -> None:
        if not self.send_health:
            return
        lines = [
            f"[HEALTH] {status}",
            f"feed_stale={details.get('feed_stale')}",
            f"binance_ok={details.get('binance_ok')}",
            f"open_trades={details.get('open_trades')}",
        ]
        text = "```\n" + "\n".join(lines) + "\n```"
        self.send_message(text)

    def notify_event_decision(self, event: Dict[str, Any], p_hat: float, routed: bool) -> None:
        if not self.send_events:
            return
        side = int(event.get("side", 0))
        side_str = "LONG" if side > 0 else "SHORT"
        reason = "ROUTED" if routed else "SKIPPED"
        text = "```\n" + "\n".join(
            [
                f"[SNAP EVENT] {self._symbol(event)} {side_str} -> {reason}",
                f"p_hat={p_hat:.3f}",
                f"shock_z={event.get('shock_z')}, rel_vol={event.get('rel_vol_rank')}, div={event.get('divergence_rank')}",
            ]
        ) + "\n```"
        self.send_message(text)

    def notify_trade_open(self, trade: Trade) -> None:
        if not self.send_trade_open:
            return
        direction = "LONG" if trade.direction > 0 else "SHORT"
        p_hat_val = trade.p_hat if trade.p_hat is not None else 0.0
        text = "```\n" + "\n".join(
            [
                f"[SNAP OPENED] {trade.symbol} {direction}",
                f"p_hat={p_hat_val:.3f} | R_unit={trade.r_unit:.2f} | H={trade.horizon_bars}",
                f"Entry={trade.entry_price:.2f} | SL={trade.sl_price:.2f} | TP={trade.tp_price:.2f}",
                f"Levels: BE={trade.be_price:.2f}",
            ]
        ) + "\n```"
        self.send_message(text)

    def notify_trade_close(self, trade: Trade) -> None:
        if not self.send_trade_close:
            return
        direction = "LONG" if trade.direction > 0 else "SHORT"
        text = "```\n" + "\n".join(
            [
                f"[SNAP CLOSED] {trade.symbol} {direction}",
                f"R_final={trade.r_final}, hit_tp={trade.hit_tp}, hit_sl={trade.hit_sl}, hit_be={trade.hit_be}",
                f"Entry={trade.entry_price:.2f} -> Exit_ts={trade.exit_ts}",
            ]
        ) + "\n```"
        self.send_message(text)

    # --- Helpers ---------------------------------------------------------

    @staticmethod
    def _symbol(event: Dict[str, Any]) -> str:
        return str(event.get("symbol", "BTCUSDT"))
