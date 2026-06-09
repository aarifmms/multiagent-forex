from __future__ import annotations

"""
Monitor Agent — Observability, logging, and alerts.

Subscribes to ALL signals on the bus. Logs everything. Sends Telegram
alerts for trade decisions, errors, and risk breaches.

This is the system's black box — every decision, signal, and error
is recorded here for post-trade analysis.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path

from agents.base_agent import BaseAgent
from core.signal_bus import Signal, SignalDirection


class MonitorAgent(BaseAgent):
    """
    Passive observer that logs every signal and sends alerts.

    Responsibilities:
    - Log all signals to structured log file (JSONL)
    - Log trade decisions to separate trade journal
    - Send Telegram alerts for: new trades, errors, risk breaches
    - Track daily P&L via account snapshots
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_dir = Path("data")
        self.log_dir.mkdir(exist_ok=True)

        self.signal_log = self.log_dir / "signals.jsonl"
        self.trade_log = self.log_dir / "trades.jsonl"
        self.error_log = self.log_dir / "errors.jsonl"

        self._tg_token = self.config.get("telegram_bot_token") or os.getenv("TELEGRAM_BOT_TOKEN")
        self._tg_chat_id = self.config.get("telegram_chat_id") or os.getenv("TELEGRAM_CHAT_ID")

        self._alerted_tickets: set[int] = set()  # avoid duplicate alerts

    async def analyze(self) -> Signal | None:
        """Monitor doesn't produce trading signals — it observes."""
        # Log account snapshot periodically
        if int(time.time()) % 60 < self.scan_interval:  # ~once per minute
            self._log_account_snapshot()

        # Check for stale agents
        await self._check_agent_health()

        return None  # Monitor never votes

    async def on_signal(self, signal: Signal) -> None:
        """Called by the signal bus for every signal published."""
        self._write_jsonl(self.signal_log, {
            "timestamp": signal.timestamp,
            "iso_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(signal.timestamp)),
            "agent": signal.agent_name,
            "direction": signal.direction.value,
            "confidence": signal.confidence,
            "reason": signal.reason,
            "entry_price": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "metadata": signal.metadata,
        })

        # Trade decisions get special treatment
        if signal.agent_name == "decision":
            await self._on_trade_decision(signal)

    async def _on_trade_decision(self, signal: Signal) -> None:
        """Log trade decisions and send alerts."""
        decision = signal.metadata.get("trade_decision", {})
        self._write_jsonl(self.trade_log, decision)

        action = decision.get("action", "HOLD")
        if action in ("BUY", "SELL"):
            await self._send_telegram(
                f"🔔 TRADE SIGNAL\n"
                f"{action} XAUUSD @ {decision.get('entry_price', 'MKT')}\n"
                f"SL: {decision.get('stop_loss')}  TP: {decision.get('take_profit')}\n"
                f"Lot: {decision.get('lot_size')}  Confidence: {decision.get('confidence', 0):.0%}\n"
                f"Reason: {decision.get('reason', 'N/A')[:200]}"
            )

    async def _check_agent_health(self) -> None:
        """Alert if any agent is stale or has errors."""
        now = time.time()
        for name, last_run in list(self.store.agent_last_run.items()):
            if name == "monitor":
                continue
            stale_seconds = now - last_run
            if stale_seconds > 30:
                self.logger.warning("Agent %s is stale (%.0fs since last run)", name, stale_seconds)
                if stale_seconds > 120:
                    await self._send_telegram(f"⚠️ Agent {name} is stale: {stale_seconds:.0f}s since last run")

        for name, error in list(self.store.agent_errors.items()):
            self._write_jsonl(self.error_log, {
                "timestamp": now,
                "agent": name,
                "error": error,
            })

    def _log_account_snapshot(self) -> None:
        self.logger.info(
            "Account | Balance=%.2f Equity=%.2f Margin=%.2f Free=%.2f Positions=%d",
            self.store.balance, self.store.equity,
            self.store.margin, self.store.free_margin,
            len(self.store.open_positions),
        )

    def _write_jsonl(self, path: Path, data: dict) -> None:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(data, default=str) + "\n")
        except Exception:
            pass  # Don't let logging failures crash the system

    async def _send_telegram(self, message: str) -> None:
        if not self._tg_token or not self._tg_chat_id:
            return
        try:
            import requests
            url = f"https://api.telegram.org/bot{self._tg_token}/sendMessage"
            requests.post(url, json={
                "chat_id": self._tg_chat_id,
                "text": message,
                "parse_mode": "HTML",
            }, timeout=5)
        except Exception:
            pass  # Telegram is best-effort
