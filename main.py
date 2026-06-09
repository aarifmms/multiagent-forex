#!/usr/bin/env python3
"""
Multi-Agent Forex Trading System — XAUUSD
===========================================

Entry point. Starts the orchestrator which runs all specialist agents
in parallel, aggregates their signals through the decision agent,
and executes trades through MT5 or paper trading simulator.

Usage:
    python main.py                    # Paper trading (default)
    python main.py --live             # Live trading with MT5
    python main.py --config custom.yaml

Environment variables:
    MT5_ACCOUNT      — MT5 account number
    MT5_PASSWORD     — MT5 password
    TELEGRAM_BOT_TOKEN — Telegram bot token for alerts
    TELEGRAM_CHAT_ID   — Telegram chat ID for alerts
"""

import argparse
import asyncio
import logging
import signal
import sys

from core.orchestrator import Orchestrator


def parse_args():
    parser = argparse.ArgumentParser(
        description="XAUUSD Multi-Agent Trading System",
    )
    parser.add_argument(
        "--config", "-c",
        default="config/settings.yaml",
        help="Path to config file (default: config/settings.yaml)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run in live trading mode (default: paper trading)",
    )
    parser.add_argument(
        "--interval", "-i",
        type=float,
        default=None,
        help="Agent scan interval in seconds (overrides config)",
    )
    parser.add_argument(
        "--force-session",
        action="store_true",
        help="Ignore market session checks (for off-hours testing)",
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    orch = Orchestrator(config_path=args.config, force_session=args.force_session)

    # Override mode if --live flag
    if args.live:
        orch.config.setdefault("system", {})["mode"] = "live"
        logging.getLogger("orchestrator").warning(
            "LIVE TRADING MODE — real orders will be placed!"
        )

    # Override scan interval
    if args.interval:
        orch.config.setdefault("agents", {})["scan_interval_seconds"] = args.interval

    # Signal handling for graceful shutdown
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, orch.handle_signal)
        except NotImplementedError:
            # Windows compatibility
            signal.signal(sig, lambda s, f: orch.handle_signal())

    try:
        await orch.start()
    except KeyboardInterrupt:
        pass
    finally:
        await orch.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
