from __future__ import annotations

"""
Decision Agent — The Brain.

Aggregates signals from all specialist agents, applies weighted voting,
enforces confidence thresholds, and outputs the final trading decision.

Flow:
1. Collect the latest signal from each specialist agent
2. Check the risk agent's approval (hard gate)
3. Compute weighted vote across news, sentiment, strategy, pattern, correlation
4. If confidence >= threshold → output BUY/SELL with entry/SL/TP
5. Otherwise → HOLD

Only the decision agent's output goes to the execution layer.
"""

import time
from dataclasses import dataclass, field
from enum import Enum

from agents.base_agent import BaseAgent
from core.signal_bus import Signal, SignalBus, SignalDirection
from core.data_store import DataStore
from core.clock import TradingClock


class TradeAction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradeDecision:
    """Final decision ready for execution."""
    action: TradeAction
    symbol: str = "XAUUSD"
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    lot_size: float = 0.01
    confidence: float = 0.0
    reason: str = ""
    signals_used: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "action": self.action.value,
            "symbol": self.symbol,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "lot_size": self.lot_size,
            "confidence": self.confidence,
            "reason": self.reason,
            "signals_used": self.signals_used,
            "timestamp": self.timestamp,
        }


class DecisionAgent(BaseAgent):
    """
    Weighted voting aggregator with risk gating.

    Weights (configurable):
      news:        0.15
      sentiment:   0.15
      strategy:    0.25
      pattern:     0.15
      correlation: 0.10
      risk:        VETO (1.0 — not additive, binary gate)

    Confidence threshold: 0.70 (70%) required to fire a trade.
    """

    # Agent names that contribute to the weighted vote
    VOTING_AGENTS = ["news", "sentiment", "strategy", "pattern", "correlation"]
    RISK_AGENT = "risk"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.70)
        self.weights = self.config.get("weights", {
            "news": 0.15, "sentiment": 0.15, "strategy": 0.25,
            "pattern": 0.15, "correlation": 0.10,
        })
        self._last_decision_time: float = 0.0
        self._signal_window: float = 15.0  # signals must be within 15s to count

    async def analyze(self) -> Signal | None:
        """
        Collect signals, compute weighted vote, produce TradeDecision.

        This runs every scan_interval. A non-None return means a trade
        decision was made and should be forwarded to execution.
        """
        # ── 1. Collect latest signals from all voting agents ──
        agent_signals: dict[str, Signal] = {}
        for agent_name in self.VOTING_AGENTS:
            sig = self.bus.get_latest_signal(agent_name)
            if sig and (time.time() - sig.timestamp) < self._signal_window:
                agent_signals[agent_name] = sig

        if not agent_signals:
            return None  # No fresh signals to aggregate

        # ── 2. Check risk gate ──
        risk_signal = self.bus.get_latest_signal(self.RISK_AGENT)
        if risk_signal is None:
            return None  # Risk agent hasn't run yet

        risk_approved = risk_signal.metadata.get("risk_approved", False)
        if not risk_approved:
            self.logger.info("Risk gate closed: %s", risk_signal.reason)
            return None  # Risk veto — no trade

        lot_size = risk_signal.metadata.get("lot_size", 0.01)

        # ── 3. Weighted vote ──
        total_weight = 0.0
        weighted_score = 0.0
        vote_details = []

        for agent_name, sig in agent_signals.items():
            w = self.weights.get(agent_name, 0.1)
            direction = 0
            if sig.direction == SignalDirection.BUY:
                direction = 1
            elif sig.direction == SignalDirection.SELL:
                direction = -1

            # Ignore low-confidence votes — they add noise, not signal
            if direction != 0 and sig.confidence < 0.35:
                direction = 0

            contribution = direction * sig.confidence * w
            weighted_score += contribution
            # Only count agents that actually voted (non-neutral)
            if direction != 0:
                total_weight += w
            vote_details.append(f"{agent_name}:{direction:+d}(c={sig.confidence:.2f})")

        if total_weight == 0:
            return None

        normalized_score = weighted_score / total_weight
        confidence = abs(normalized_score)

        # ── 4. Threshold check ──
        if confidence < self.confidence_threshold:
            self.logger.info(
                "Confidence below threshold: %.2f < %.2f | %s",
                confidence, self.confidence_threshold, " | ".join(vote_details),
            )
            return None

        self._last_decision_time = time.time()

        # ── 5. Build trade decision ──
        direction = SignalDirection.BUY if normalized_score > 0 else SignalDirection.SELL

        # Use strategy agent's entry/SL/TP if available
        strat_sig = agent_signals.get("strategy")
        entry = strat_sig.entry_price if strat_sig and strat_sig.entry_price else 0.0
        sl = strat_sig.stop_loss if strat_sig and strat_sig.stop_loss else 0.0
        tp = strat_sig.take_profit if strat_sig and strat_sig.take_profit else 0.0

        # Fallback: use current tick
        if entry == 0 and self.latest_tick:
            entry = self.latest_tick.bid if direction == SignalDirection.BUY else self.latest_tick.ask

        reason = f"Decision: {direction.value} ({confidence:.1%}) | " + " | ".join(vote_details)

        signal = Signal(
            agent_name=self.name,
            direction=direction,
            confidence=confidence,
            reason=reason,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            metadata={
                "trade_decision": TradeDecision(
                    action=TradeAction.BUY if direction == SignalDirection.BUY else TradeAction.SELL,
                    entry_price=entry,
                    stop_loss=sl,
                    take_profit=tp,
                    lot_size=lot_size,
                    confidence=confidence,
                    reason=reason,
                    signals_used=list(agent_signals.keys()),
                ).to_dict(),
                "vote_details": vote_details,
                "risk_checks": risk_signal.metadata.get("risk_checks", []),
            },
        )

        self.logger.info("TRADE DECISION: %s", reason)
        return signal
