from __future__ import annotations

"""
Shared Data Store — Thread-safe dictionary for cross-agent state.

Holds latest prices, open positions, account info, and agent status.
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Candle:
    time: float
    open: float
    high: float
    low: float
    close: float
    tick_volume: int = 0
    spread: int = 0


@dataclass
class Tick:
    time: float
    bid: float
    ask: float
    volume: int = 0


@dataclass
class Position:
    ticket: int
    symbol: str
    direction: str   # BUY / SELL
    volume: float
    open_price: float
    sl: float
    tp: float
    open_time: float
    comment: str = ""


class DataStore:
    """
    Central shared-state container. All agents read/write through this.
    Uses asyncio locks for thread safety.
    """

    def __init__(self, max_candles: int = 5000, max_ticks: int = 10000):
        self._lock = asyncio.Lock()

        # Market data
        self.current_tick: Tick | None = None
        self.candles: dict[str, deque[Candle]] = {}  # tf -> deque of candles
        self.max_candles = max_candles
        self.ticks_buffer: deque[Tick] = deque(maxlen=max_ticks)

        # Account
        self.balance: float = 0.0
        self.equity: float = 0.0
        self.margin: float = 0.0
        self.free_margin: float = 0.0

        # Positions
        self.open_positions: dict[int, Position] = {}

        # Agent status
        self.agent_last_run: dict[str, float] = {}
        self.agent_errors: dict[str, str] = {}

        # News & calendar
        self.active_news_blackout: bool = False
        self.blackout_until: float = 0.0
        self.upcoming_events: list[dict] = []

        # System state
        self.is_running: bool = False
        self.start_time: float = 0.0

    async def update_tick(self, tick: Tick) -> None:
        async with self._lock:
            self.current_tick = tick
            self.ticks_buffer.append(tick)

    async def update_candle(self, timeframe: str, candle: Candle) -> None:
        async with self._lock:
            if timeframe not in self.candles:
                self.candles[timeframe] = deque(maxlen=self.max_candles)
            candles = self.candles[timeframe]
            if candles and candles[-1].time == candle.time:
                candles[-1] = candle  # update in-place
            else:
                candles.append(candle)

    async def get_candles(self, timeframe: str, count: int = 200) -> list[Candle]:
        async with self._lock:
            c = self.candles.get(timeframe, deque(maxlen=self.max_candles))
            return list(c)[-count:]

    async def update_position(self, pos: Position) -> None:
        async with self._lock:
            self.open_positions[pos.ticket] = pos

    async def remove_position(self, ticket: int) -> None:
        async with self._lock:
            self.open_positions.pop(ticket, None)

    async def set_agent_status(self, name: str, error: str | None = None) -> None:
        async with self._lock:
            self.agent_last_run[name] = time.time()
            if error:
                self.agent_errors[name] = error
            else:
                self.agent_errors.pop(name, None)
