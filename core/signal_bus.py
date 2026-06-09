from __future__ import annotations

"""
Signal Bus — Pub/Sub message backbone for inter-agent communication.

Every specialist agent publishes Signal objects. The decision agent subscribes
to all signals and aggregates them. The monitor agent subscribes to everything.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Awaitable


class SignalDirection(Enum):
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"


@dataclass
class Signal:
    """Standardized signal produced by every agent."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent_name: str = ""
    direction: SignalDirection = SignalDirection.NEUTRAL
    confidence: float = 0.0        # 0.0 — 1.0
    reason: str = ""
    timestamp: float = field(default_factory=time.time)

    # Optional trade details filled by strategy/pattern agents
    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None

    # Metadata for debugging
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "agent_name": self.agent_name,
            "direction": self.direction.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "metadata": self.metadata,
        }


Subscriber = Callable[[Signal], Awaitable[None]]


class SignalBus:
    """
    Thread-safe async pub/sub bus.

    Agents publish signals to named topics. Subscribers receive every signal
    published to topics they subscribe to.
    """

    def __init__(self):
        self._subscribers: dict[str, list[Subscriber]] = {}
        self._signal_history: list[Signal] = []
        self._lock = asyncio.Lock()
        self._max_history = 10_000

    def subscribe(self, topic: str, callback: Subscriber) -> None:
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(callback)

    async def publish(self, topic: str, signal: Signal) -> None:
        async with self._lock:
            self._signal_history.append(signal)
            if len(self._signal_history) > self._max_history:
                self._signal_history = self._signal_history[-self._max_history:]

        subscribers = self._subscribers.get(topic, [])
        await asyncio.gather(*[sub(signal) for sub in subscribers], return_exceptions=True)

    def get_recent_signals(self, agent_name: str | None = None, seconds: float = 30.0) -> list[Signal]:
        cutoff = time.time() - seconds
        signals = self._signal_history
        if agent_name:
            signals = [s for s in signals if s.agent_name == agent_name]
        return [s for s in signals if s.timestamp >= cutoff]

    def get_latest_signal(self, agent_name: str) -> Signal | None:
        for s in reversed(self._signal_history):
            if s.agent_name == agent_name:
                return s
        return None
