from __future__ import annotations

"""
Base Agent — Every specialist agent inherits from this.

Provides: lifecycle (start/stop/run loop), signal publishing, error handling,
logging, and heartbeat tracking.
"""

import asyncio
import logging
import time
import traceback
from abc import ABC, abstractmethod

from core.signal_bus import Signal, SignalBus, SignalDirection
from core.data_store import DataStore
from core.clock import TradingClock


class BaseAgent(ABC):
    """
    Abstract base for all agents.

    Subclasses implement `analyze()` which returns a Signal.
    The run loop calls `analyze()` on each tick and publishes results.
    """

    def __init__(
        self,
        name: str,
        signal_bus: SignalBus,
        data_store: DataStore,
        clock: TradingClock,
        config: dict | None = None,
        scan_interval: float = 5.0,
    ):
        self.name = name
        self.bus = signal_bus
        self.store = data_store
        self.clock = clock
        self.config = config or {}
        self.scan_interval = scan_interval
        self.logger = logging.getLogger(f"agent.{name}")
        self._running = False
        self._task: asyncio.Task | None = None

    # ── Lifecycle ──────────────────────────────────────────────

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        self.logger.info("Started (interval=%ss)", self.scan_interval)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.logger.info("Stopped")

    async def _run_loop(self) -> None:
        while self._running:
            try:
                signal = await self.analyze()
                if signal is not None:
                    await self.bus.publish("signals", signal)
                await self.store.set_agent_status(self.name)
            except asyncio.CancelledError:
                break
            except Exception:
                err = traceback.format_exc()
                self.logger.error("Error in analyze: %s", err)
                await self.store.set_agent_status(self.name, error=err)
            await asyncio.sleep(self.scan_interval)

    # ── Interface ──────────────────────────────────────────────

    @abstractmethod
    async def analyze(self) -> Signal | None:
        """Run one analysis cycle. Return a Signal or None if no opinion."""
        ...

    # ── Helpers ─────────────────────────────────────────────────

    def _signal(self, direction: SignalDirection, confidence: float, reason: str = "", **meta) -> Signal:
        return Signal(
            agent_name=self.name,
            direction=direction,
            confidence=min(max(confidence, 0.0), 1.0),
            reason=reason,
            metadata=meta,
        )

    def _buy(self, confidence: float, reason: str = "", **meta) -> Signal:
        return self._signal(SignalDirection.BUY, confidence, reason, **meta)

    def _sell(self, confidence: float, reason: str = "", **meta) -> Signal:
        return self._signal(SignalDirection.SELL, confidence, reason, **meta)

    def _neutral(self, reason: str = "", **meta) -> Signal:
        return self._signal(SignalDirection.NEUTRAL, 0.0, reason, **meta)

    @property
    def latest_tick(self):
        return self.store.current_tick

    async def get_candles(self, tf: str, count: int = 200):
        return await self.store.get_candles(tf, count)
