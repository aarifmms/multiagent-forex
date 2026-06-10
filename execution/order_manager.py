from __future__ import annotations

"""
Order Manager — Trade execution and lifecycle management.

Receives TradeDecision from the decision agent and:
1. Places the order (market or pending)
2. Manages partial closes (50% at 1:1 RR, 25% at 2:1 RR, 25% runner)
3. Implements ATR-based trailing stop
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum

from core.data_store import DataStore
from core.clock import TradingClock
from agents.decision_agent import TradeDecision, TradeAction


class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_CLOSED = "PARTIALLY_CLOSED"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


@dataclass
class ActiveTrade:
    ticket: int
    direction: TradeAction
    entry_price: float
    original_sl: float
    original_tp: float
    current_sl: float
    current_tp: float
    lot_size: float
    remaining_lots: float
    status: OrderStatus
    open_time: float
    partial_closes: list[dict]
    trail_activated: bool

    @property
    def profit_pips(self) -> float:
        if self.store.current_tick is None:
            return 0
        mid = (self.store.current_tick.bid + self.store.current_tick.ask) / 2
        if self.direction == TradeAction.BUY:
            return mid - self.entry_price
        else:
            return self.entry_price - mid


class OrderManager:
    """
    Manages trade execution and lifecycle.

    In paper mode: simulates order execution with spread and slippage.
    In live mode: sends orders to MT5.

    Handles:
    - Market order placement
    - Partial take-profit closes
    - Trailing stop updates
    - Trade monitoring and logging
    """

    def __init__(self, data_store: DataStore, clock: TradingClock, config: dict,
                 broker_bridge=None):
        self.store = data_store
        self.clock = clock
        self.config = config
        self.logger = logging.getLogger("order.manager")
        self.mode = config.get("system", {}).get("mode", "paper")
        self._bridge = broker_bridge  # ICMarketsBridge or MT5Bridge

        exec_cfg = config.get("execution", {})
        self.trail_enabled = exec_cfg.get("trailing_stop", {}).get("enabled", True)
        self.trail_activation_pips = exec_cfg.get("trailing_stop", {}).get("activation_pips", 30)
        self.trail_distance_pips = exec_cfg.get("trailing_stop", {}).get("distance_pips", 15)

        self.partial_enabled = exec_cfg.get("partial_close", {}).get("enabled", True)
        self.close_50_at_rr = exec_cfg.get("partial_close", {}).get("close_50_at_rr", 1.0)
        self.close_25_at_rr = exec_cfg.get("partial_close", {}).get("close_25_at_rr", 2.0)

        self._active_trades: dict[int, ActiveTrade] = {}
        self._ticket_counter = 1000
        self._monitor_task: asyncio.Task | None = None
        self._realized_pnl: float = 0.0
        self._closed_trades: list[dict] = []  # P&L history

    async def start(self) -> None:
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        self.logger.info("Order Manager started (%s mode)", self.mode)

    async def stop(self) -> None:
        if self._monitor_task:
            self._monitor_task.cancel()

    async def execute(self, decision: TradeDecision) -> ActiveTrade | None:
        """
        Place a trade based on the decision agent's output.

        Returns the ActiveTrade if filled, None if rejected.
        """
        if decision.action == TradeAction.HOLD:
            return None

        if not self.clock.can_trade:
            self.logger.info("Trade rejected: market/clock conditions prevent trading")
            return None

        ticket = self._ticket_counter
        self._ticket_counter += 1

        # Apply spread/slippage in paper mode
        slippage = 0
        if self.mode == "paper" and self.store.current_tick:
            if decision.action == TradeAction.BUY:
                filled_price = self.store.current_tick.ask + 0.05  # minimal slippage
            else:
                filled_price = self.store.current_tick.bid - 0.05
        else:
            filled_price = decision.entry_price

        trade = ActiveTrade(
            ticket=ticket,
            direction=decision.action,
            entry_price=filled_price,
            original_sl=decision.stop_loss,
            original_tp=decision.take_profit,
            current_sl=decision.stop_loss,
            current_tp=decision.take_profit,
            lot_size=decision.lot_size,
            remaining_lots=decision.lot_size,
            status=OrderStatus.FILLED,
            open_time=time.time(),
            partial_closes=[],
            trail_activated=False,
        )

        self._active_trades[ticket] = trade

        # In live mode, send to broker FIRST to get real ticket
        if self.mode == "live":
            await self._place_live_order(trade)
            if trade.status == OrderStatus.CANCELLED:
                del self._active_trades[ticket]
                return None

        self.clock.record_trade()

        self.logger.info(
            "TRADE OPENED | Ticket=%d %s @ %.2f SL=%.2f TP=%.2f Lot=%.2f",
            trade.ticket, decision.action.value, filled_price,
            decision.stop_loss, decision.take_profit, decision.lot_size,
        )

        # Sync to DataStore with the REAL broker ticket
        from core.data_store import Position
        await self.store.update_position(Position(
            ticket=trade.ticket,
            symbol="XAUUSD",
            direction="BUY" if decision.action == TradeAction.BUY else "SELL",
            volume=decision.lot_size,
            open_price=filled_price,
            sl=decision.stop_loss,
            tp=decision.take_profit,
            open_time=time.time(),
            comment="MultiAgent Live" if self.mode == "live" else "MultiAgent Paper",
        ))

        return trade

    async def _place_live_order(self, trade: ActiveTrade) -> None:
        """Send order to the configured broker bridge."""
        if self._bridge is None:
            self.logger.warning("No broker bridge — trade logged in paper mode")
            return

        direction = "BUY" if trade.direction == TradeAction.BUY else "SELL"
        try:
            order_id = await self._bridge.place_order(
                direction=direction,
                volume_lots=trade.lot_size,
                sl=trade.current_sl,
                tp=trade.current_tp,
            )
            if order_id:
                trade.ticket = order_id
                self.logger.info("Live order placed: ticket=%d", order_id)
            else:
                trade.status = OrderStatus.CANCELLED
                self.logger.error("Live order rejected")
        except Exception as e:
            self.logger.error("Live order error: %s", e)
            trade.status = OrderStatus.CANCELLED

    async def close_trade(self, ticket: int, lots: float | None = None, reason: str = "") -> None:
        """Close all or part of a trade."""
        trade = self._active_trades.get(ticket)
        if not trade:
            return

        close_lots = lots or trade.remaining_lots
        close_lots = min(close_lots, trade.remaining_lots)

        trade.remaining_lots -= close_lots
        trade.partial_closes.append({
            "time": time.time(),
            "lots": close_lots,
            "reason": reason,
        })

        # Calculate P&L for this close
        tick = self.store.current_tick
        close_price = (tick.bid + tick.ask) / 2 if tick else trade.entry_price
        if trade.direction == TradeAction.BUY:
            pnl = (close_price - trade.entry_price) * close_lots * 100
        else:
            pnl = (trade.entry_price - close_price) * close_lots * 100

        if trade.remaining_lots - close_lots <= 0.001:
            trade.status = OrderStatus.CLOSED
            trade.remaining_lots = 0
            await self.store.remove_position(ticket)
            self._realized_pnl += pnl
            self._closed_trades.append({
                "ticket": ticket, "direction": trade.direction.value,
                "entry": trade.entry_price, "exit": round(close_price, 2),
                "lots": trade.lot_size, "pnl": round(pnl, 2),
                "reason": reason, "time": time.time(),
            })
            self.store.balance += pnl
            self.logger.info("TRADE CLOSED | Ticket=%d Reason=%s PnL=$%.2f | Realized=$%.2f",
                             ticket, reason, pnl, self._realized_pnl)
        else:
            trade.status = OrderStatus.PARTIALLY_CLOSED
            self._realized_pnl += pnl
            self._closed_trades.append({
                "ticket": ticket, "direction": trade.direction.value,
                "entry": trade.entry_price, "exit": round(close_price, 2),
                "lots": close_lots, "pnl": round(pnl, 2),
                "reason": reason, "time": time.time(),
            })
            self.store.balance += pnl
            self.logger.info("PARTIAL CLOSE | Ticket=%d Lots=%.2f Remaining=%.2f Reason=%s PnL=$%.2f",
                             ticket, close_lots, trade.remaining_lots, reason, pnl)

        if self.mode == "live" and self._bridge:
            try:
                await self._bridge.close_position(ticket, close_lots)
            except Exception as e:
                self.logger.error("Live close error: %s", e)

    # ── Trade Monitor Loop ─────────────────────────────────────

    async def _monitor_loop(self) -> None:
        """Continuously monitor open trades for SL/TP hits, partial closes, trailing stops."""
        while True:
            try:
                await self._check_stop_levels()
                await self._check_partial_close_targets()
                await self._update_trailing_stops()
                self._update_equity()
            except Exception as e:
                self.logger.error("Monitor loop error: %s", e)
            await asyncio.sleep(0.5)

    def _update_equity(self) -> None:
        """Update store equity/balance based on open positions + realized P&L."""
        tick = self.store.current_tick
        if not tick:
            return
        mid = (tick.bid + tick.ask) / 2
        unrealized = 0.0
        for trade in self._active_trades.values():
            if trade.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_CLOSED):
                if trade.direction == TradeAction.BUY:
                    unrealized += (mid - trade.entry_price) * trade.remaining_lots * 100
                else:
                    unrealized += (trade.entry_price - mid) * trade.remaining_lots * 100
        self.store.equity = self.store.balance + unrealized
        self.store.free_margin = self.store.equity - self.store.margin

    async def _check_stop_levels(self) -> None:
        """Check if any trade has hit SL or TP."""
        tick = self.store.current_tick
        if not tick:
            return

        for ticket, trade in list(self._active_trades.items()):
            if trade.status not in (OrderStatus.FILLED, OrderStatus.PARTIALLY_CLOSED):
                continue

            bid, ask = tick.bid, tick.ask
            if trade.direction == TradeAction.BUY:
                if bid <= trade.current_sl:
                    await self.close_trade(ticket, reason="SL-hit")
                elif ask >= trade.current_tp:
                    await self.close_trade(ticket, reason="TP-hit")
            else:  # SELL
                if ask >= trade.current_sl:
                    await self.close_trade(ticket, reason="SL-hit")
                elif bid <= trade.current_tp:
                    await self.close_trade(ticket, reason="TP-hit")

    async def _check_partial_close_targets(self) -> None:
        """Check if partial close targets are hit."""
        if not self.partial_enabled:
            return

        tick = self.store.current_tick
        if not tick:
            return

        for ticket, trade in list(self._active_trades.items()):
            if trade.status != OrderStatus.FILLED:
                continue

            sl_distance = abs(trade.entry_price - trade.original_sl)
            current_profit = abs(
                (tick.bid + tick.ask) / 2 - trade.entry_price
            )

            rr_achieved = current_profit / sl_distance if sl_distance > 0 else 0

            # Close 50% at 1:1 RR
            if rr_achieved >= self.close_50_at_rr and trade.remaining_lots > trade.lot_size * 0.5:
                await self.close_trade(ticket, lots=trade.lot_size * 0.5, reason=f"Partial-50%@{rr_achieved:.1f}RR")
                # Move SL to breakeven on remaining
                trade.current_sl = trade.entry_price
                self.logger.info("SL moved to breakeven for ticket=%d", ticket)

            # Close 25% at 2:1 RR
            elif rr_achieved >= self.close_25_at_rr and trade.remaining_lots > trade.lot_size * 0.25:
                close_lots = min(trade.lot_size * 0.25, trade.remaining_lots - 0.01)
                if close_lots > 0:
                    await self.close_trade(ticket, lots=close_lots, reason=f"Partial-25%@{rr_achieved:.1f}RR")

    async def _update_trailing_stops(self) -> None:
        """Update trailing stops based on ATR."""
        if not self.trail_enabled:
            return

        tick = self.store.current_tick
        if not tick:
            return

        for ticket, trade in list(self._active_trades.items()):
            if trade.status not in (OrderStatus.FILLED, OrderStatus.PARTIALLY_CLOSED):
                continue

            mid = (tick.bid + tick.ask) / 2
            profit_pips = (
                mid - trade.entry_price if trade.direction == TradeAction.BUY
                else trade.entry_price - mid
            )

            # Activate trailing stop once profit exceeds threshold
            if profit_pips >= self.trail_activation_pips:
                trade.trail_activated = True

            if trade.trail_activated:
                if trade.direction == TradeAction.BUY:
                    new_sl = mid - self.trail_distance_pips
                    if new_sl > trade.current_sl:
                        trade.current_sl = round(new_sl, 2)
                else:
                    new_sl = mid + self.trail_distance_pips
                    if new_sl < trade.current_sl:
                        trade.current_sl = round(new_sl, 2)
