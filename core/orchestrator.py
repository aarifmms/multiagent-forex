from __future__ import annotations

"""
Orchestrator — Central coordinator that wires all agents together.

Lifecycle:
1. Load config
2. Initialize DataStore, SignalBus, TradingClock
3. Create all agents with their configs
4. Start MT5 Bridge (data pipeline)
5. Start Order Manager (execution pipeline)
6. Start all agents in parallel
7. Route decision signals → Order Manager
8. Handle graceful shutdown
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

import yaml

from core.signal_bus import SignalBus
from core.data_store import DataStore
from core.clock import TradingClock

from data.price_feed import create_price_feed, BasePriceFeed
from execution.mt5_bridge import MT5Bridge
from execution.order_manager import OrderManager

from agents.news_agent import NewsAgent
from agents.sentiment_agent import SentimentAgent
from agents.strategy_agent import StrategyAgent
from agents.pattern_agent import PatternAgent
from agents.correlation_agent import CorrelationAgent
from agents.risk_agent import RiskAgent
from agents.decision_agent import DecisionAgent
from agents.monitor_agent import MonitorAgent
from agents.base_agent import BaseAgent


class Orchestrator:
    """
    The conductor. Starts everything, routes signals, handles shutdown.
    """

    def __init__(self, config_path: str = "config/settings.yaml", force_session: bool = False):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._force_session = force_session

        log_level = self.config.get("system", {}).get("log_level", "INFO")
        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO),
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        self.logger = logging.getLogger("orchestrator")

        # Core infrastructure
        self.bus = SignalBus()
        self.store = DataStore()
        self.clock = TradingClock(
            cooldown_seconds=self.config.get("decision_agent", {}).get("cooldown_minutes", 5) * 60,
            force_session=self._force_session,
        )

        # Broker backend selection
        broker = self.config.get("broker", {}).get("provider", "mt5").lower()
        if broker == "icmarkets":
            from execution.icmarkets_bridge import ICMarketsBridge
            self.broker_bridge = ICMarketsBridge(self.store, self.config)
        else:
            self.broker_bridge = MT5Bridge(self.store, self.config)

        # Data & Execution layer
        self.price_feed = create_price_feed(self.store, self.config)
        self.order_manager = OrderManager(self.store, self.clock, self.config,
                                          broker_bridge=self.broker_bridge)

        # Agents (created in _create_agents)
        self.agents: list[BaseAgent] = []
        self._decision_agent: DecisionAgent | None = None
        self._monitor_agent: MonitorAgent | None = None

        # Shutdown
        self._shutdown_event = asyncio.Event()

    def _load_config(self) -> dict:
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def _create_agents(self) -> None:
        agent_cfg = self.config.get("agents", {})
        scan_interval = agent_cfg.get("scan_interval_seconds", 5)
        decision_cfg = self.config.get("decision_agent", {})

        agent_classes = {
            "news": (NewsAgent, agent_cfg.get("news_agent", {})),
            "sentiment": (SentimentAgent, agent_cfg.get("sentiment_agent", {})),
            "strategy": (StrategyAgent, agent_cfg.get("strategy_agent", {})),
            "pattern": (PatternAgent, agent_cfg.get("pattern_agent", {})),
            "correlation": (CorrelationAgent, agent_cfg.get("correlation_agent", {})),
            "risk": (RiskAgent, agent_cfg.get("risk_agent", {})),
            "decision": (DecisionAgent, {**decision_cfg, "weights": decision_cfg.get("weights", {})}),
            "monitor": (MonitorAgent, self.config.get("monitor", {})),
        }

        for name, (cls, cfg) in agent_classes.items():
            enabled = cfg.get("enabled", True) if name != "decision" and name != "monitor" else True
            if not enabled:
                continue

            agent = cls(
                name=name,
                signal_bus=self.bus,
                data_store=self.store,
                clock=self.clock,
                config=cfg,
                scan_interval=scan_interval,
            )
            self.agents.append(agent)

            if name == "decision":
                self._decision_agent = agent
            elif name == "monitor":
                self._monitor_agent = agent

    # ── Lifecycle ──────────────────────────────────────────────

    async def start(self) -> None:
        mode = self.config.get("system", {}).get("mode", "paper")
        self.logger.info("╔══════════════════════════════════════════════╗")
        self.logger.info("║  XAUUSD Multi-Agent Trading System v1.0     ║")
        self.logger.info("║  Mode: %-36s ║", mode)
        self.logger.info("╚══════════════════════════════════════════════╝")

        self._create_agents()
        self.store.is_running = True
        self.store.start_time = self.clock.now

        # Data feed: Yahoo in paper mode, broker provides ticks in live mode
        if mode == "live":
            self.logger.info("Live mode — broker bridge provides price data, skipping external feed")
            await self.broker_bridge.start()
        else:
            await self.price_feed.start()
            self.logger.info("Price feed: %s", self.price_feed.name)

        # Set paper trading balance
        if mode != "live" and self.store.balance == 0:
            self.store.balance = 10000.0
            self.store.equity = 10000.0
            self.store.free_margin = 10000.0

        # Start execution layer
        await self.order_manager.start()

        # Subscribe monitor agent to all signals
        if self._monitor_agent:
            self.bus.subscribe("signals", self._monitor_agent.on_signal)

        # Subscribe to decision signals for execution
        async def on_decision(signal):
            decision_data = signal.metadata.get("trade_decision")
            if decision_data:
                from agents.decision_agent import TradeDecision, TradeAction
                decision = TradeDecision(
                    action=TradeAction(decision_data["action"]),
                    symbol=decision_data.get("symbol", "XAUUSD"),
                    entry_price=decision_data.get("entry_price", 0),
                    stop_loss=decision_data.get("stop_loss", 0),
                    take_profit=decision_data.get("take_profit", 0),
                    lot_size=decision_data.get("lot_size", 0.01),
                    confidence=decision_data.get("confidence", 0),
                    reason=decision_data.get("reason", ""),
                    signals_used=decision_data.get("signals_used", []),
                )
                await self.order_manager.execute(decision)

        self.bus.subscribe("signals", on_decision)

        # Start all agents
        for agent in self.agents:
            await agent.start()

        self.logger.info("All agents started. Trading system is LIVE.")
        self.logger.info("Agents: %s", [a.name for a in self.agents])
        self.logger.info("Data source: %s", self.price_feed.name)

        # Wait for shutdown signal
        await self._shutdown_event.wait()

    async def shutdown(self) -> None:
        self.logger.info("Shutting down...")
        self._shutdown_event.set()

        for agent in self.agents:
            await agent.stop()

        await self.order_manager.stop()
        if self.config.get("system", {}).get("mode") == "live":
            await self.broker_bridge.stop()
        await self.price_feed.stop()
        self.store.is_running = False

        self.logger.info("Shutdown complete.")

    def handle_signal(self) -> None:
        """Called by signal handlers to trigger graceful shutdown."""
        self._shutdown_event.set()
