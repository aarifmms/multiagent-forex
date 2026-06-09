from __future__ import annotations

"""
MT5 Bridge — MetaTrader 5 connection and data pipeline.

Handles:
- Connection lifecycle (connect, heartbeat, reconnect)
- Price streaming (ticks → DataStore)
- Historical candle fetching (for all timeframes)
- Account state synchronization

Paper trading mode: runs without MT5, using simulated prices and account.
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from core.data_store import DataStore, Tick, Candle


class MT5Bridge:
    """
    Connects to MetaTrader 5 for live data and execution.

    In paper mode, runs a simulated price feed based on realistic
    XAUUSD price movement (mean-reverting random walk with volatility
    calibrated to gold's typical behavior).
    """

    TIMEFRAMES = {
        "M1": 1, "M5": 5, "M15": 15, "M30": 30,
        "H1": 60, "H4": 240, "D1": 1440,
    }

    def __init__(self, data_store: DataStore, config: dict):
        self.store = data_store
        self.config = config
        self.logger = logging.getLogger("mt5.bridge")
        self.mode = config.get("system", {}).get("mode", "paper")
        self.symbol = config.get("broker", {}).get("symbol", "XAUUSD")
        self._mt5 = None
        self._running = False
        self._sim_price = 2650.0  # realistic XAUUSD starting price
        self._sim_spread = 0.30   # 30 cents typical spread

    # ── Lifecycle ──────────────────────────────────────────────

    async def start(self) -> None:
        self._running = True
        if self.mode == "live":
            await self._connect_mt5()
            await self._seed_history()  # Seed candles before streaming
            asyncio.create_task(self._price_stream())
        # In paper mode, Yahoo feed handles price streaming & seeding
        asyncio.create_task(self._account_sync())
        self.logger.info("MT5 Bridge started in %s mode", self.mode)

    async def stop(self) -> None:
        self._running = False
        if self.mode == "live" and self._mt5:
            self._mt5.shutdown()
            self._mt5 = None

    async def _connect_mt5(self) -> bool:
        try:
            import MetaTrader5 as mt5
            self._mt5 = mt5

            account = os.getenv("MT5_ACCOUNT")
            password = os.getenv("MT5_PASSWORD")
            server = self.config.get("broker", {}).get("server", "")

            if not mt5.initialize():
                self.logger.error("MT5 init failed: %s", mt5.last_error())
                return False

            if account and password:
                authorized = mt5.login(int(account), password=password, server=server)
                if not authorized:
                    self.logger.error("MT5 login failed: %s", mt5.last_error())
                    return False

            self.logger.info("MT5 connected. Account: %s", mt5.account_info().login if mt5.account_info() else "N/A")
            return True
        except ImportError:
            self.logger.warning("MetaTrader5 package not installed — falling back to paper mode")
            self.mode = "paper"
            return False
        except Exception as e:
            self.logger.error("MT5 connection error: %s", e)
            self.mode = "paper"
            return False

    # ── Price Streaming ────────────────────────────────────────

    async def _price_stream(self) -> None:
        """Continuous price feed → DataStore."""
        while self._running:
            try:
                if self.mode == "live":
                    await self._stream_live_tick()
                else:
                    await self._stream_simulated_tick()
            except Exception as e:
                self.logger.error("Price stream error: %s", e)
            await asyncio.sleep(0.2)  # 5 ticks/sec

    async def _stream_live_tick(self) -> None:
        """Fetch latest tick from MT5."""
        if not self._mt5:
            return
        tick_info = self._mt5.symbol_info_tick(self.symbol)
        if tick_info is None:
            return
        tick = Tick(
            time=time.time(),
            bid=tick_info.bid,
            ask=tick_info.ask,
        )
        await self.store.update_tick(tick)
        await self._aggregate_candles(tick)

    async def _stream_simulated_tick(self) -> None:
        """Realistic simulated price for paper trading."""
        import random
        import math

        # Volatility by session (approximate)
        hour = datetime.now(timezone.utc).hour
        if 8 <= hour < 17:    # London
            vol = 0.15
        elif 13 <= hour < 22:  # NY
            vol = 0.18
        elif 1 <= hour < 10:   # Asian
            vol = 0.06
        else:
            vol = 0.03

        # Mean-reverting random walk
        mean_price = 2650.0
        reversion = (mean_price - self._sim_price) * 0.0001
        noise = random.gauss(0, vol)
        self._sim_price += reversion + noise
        self._sim_price = max(2400, min(2900, self._sim_price))

        spread = self._sim_spread * (1.0 + random.random() * 0.5)
        bid = round(self._sim_price, 2)
        ask = round(self._sim_price + spread, 2)

        tick = Tick(time=time.time(), bid=bid, ask=ask)
        await self.store.update_tick(tick)
        await self._aggregate_candles(tick)

    async def _aggregate_candles(self, tick: Tick) -> None:
        """Build candles from tick data for all timeframes."""
        for tf_name, tf_minutes in self.TIMEFRAMES.items():
            candle_time = int(tick.time // (tf_minutes * 60)) * (tf_minutes * 60)
            candles = await self.store.get_candles(tf_name, count=1)
            mid = (tick.bid + tick.ask) / 2

            if candles and candles[-1].time == candle_time:
                # Update current candle
                c = candles[-1]
                c.high = max(c.high, mid)
                c.low = min(c.low, mid)
                c.close = mid
                c.tick_volume += 1
            else:
                # New candle
                await self.store.update_candle(tf_name, Candle(
                    time=candle_time,
                    open=mid, high=mid, low=mid, close=mid,
                    tick_volume=1,
                ))

    # ── Account Sync ───────────────────────────────────────────

    async def _account_sync(self) -> None:
        """Periodically sync account state to DataStore."""
        while self._running:
            try:
                if self.mode == "live" and self._mt5:
                    info = self._mt5.account_info()
                    if info:
                        self.store.balance = info.balance
                        self.store.equity = info.equity
                        self.store.margin = info.margin
                        self.store.free_margin = info.margin_free

                    # Sync positions
                    positions = self._mt5.positions_get(symbol=self.symbol)
                    if positions:
                        for pos in positions:
                            from core.data_store import Position
                            await self.store.update_position(Position(
                                ticket=pos.ticket,
                                symbol=pos.symbol,
                                direction="BUY" if pos.type == 0 else "SELL",
                                volume=pos.volume,
                                open_price=pos.price_open,
                                sl=pos.sl,
                                tp=pos.tp,
                                open_time=pos.time,
                                comment=pos.comment,
                            ))
                else:
                    # Paper: maintain a static balance
                    if self.store.balance == 0:
                        self.store.balance = 10000.0
                        self.store.equity = 10000.0
                        self.store.free_margin = 10000.0

            except Exception as e:
                self.logger.error("Account sync error: %s", e)
            await asyncio.sleep(5)

    # ── Order Execution ──────────────────────────────────────

    async def place_order(self, direction: str, volume_lots: float,
                          sl: float = 0, tp: float = 0) -> int | None:
        """Place a market order via MT5 or simulate in paper mode."""
        if self.mode != "live" or not self._mt5:
            # Paper mode: generate a synthetic ticket ID
            ticket = int(time.time() * 1000) % 100000
            self.logger.info("Paper order: %s XAUUSD lot=%.2f ticket=%d",
                             direction, volume_lots, ticket)
            return ticket

        try:
            trade_type = self._mt5.ORDER_TYPE_BUY if direction == "BUY" else self._mt5.ORDER_TYPE_SELL
            tick = self._mt5.symbol_info_tick(self.symbol)
            if not tick:
                return None

            request = {
                "action": self._mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": volume_lots,
                "type": trade_type,
                "price": tick.ask if direction == "BUY" else tick.bid,
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "magic": 123456,
                "comment": "MultiAgent XAUUSD",
                "type_time": self._mt5.ORDER_TIME_GTC,
                "type_filling": self._mt5.ORDER_FILLING_IOC,
            }

            result = self._mt5.order_send(request)
            if result and result.retcode == self._mt5.TRADE_RETCODE_DONE:
                self.logger.info("MT5 order placed: ticket=%d", result.order)
                return int(result.order)
            else:
                self.logger.error("MT5 order failed: %s", result.comment if result else "no response")
                return None
        except Exception as e:
            self.logger.error("MT5 order error: %s", e)
            return None

    async def close_position(self, position_id: int, volume_lots: float) -> bool:
        """Close a position via MT5 or simulate in paper mode."""
        if self.mode != "live" or not self._mt5:
            self.logger.info("Paper close: position=%d lots=%.2f", position_id, volume_lots)
            return True

        try:
            pos_info = self._mt5.positions_get(ticket=position_id)
            if not pos_info:
                self.logger.warning("Position %d not found", position_id)
                return False

            pos = pos_info[0]
            tick = self._mt5.symbol_info_tick(self.symbol)
            if not tick:
                return False

            request = {
                "action": self._mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": volume_lots,
                "type": self._mt5.ORDER_TYPE_SELL if pos.type == 0 else self._mt5.ORDER_TYPE_BUY,
                "position": pos.ticket,
                "price": tick.bid if pos.type == 0 else tick.ask,
                "deviation": 20,
                "magic": 123456,
                "comment": "MultiAgent Close",
            }
            result = self._mt5.order_send(request)
            return result is not None and result.retcode == self._mt5.TRADE_RETCODE_DONE
        except Exception as e:
            self.logger.error("MT5 close error: %s", e)
            return False

    # ── Historical Data ────────────────────────────────────────

    async def fetch_history(self, tf: str, count: int = 200) -> list[Candle]:
        """Fetch historical candles for a timeframe."""
        if self.mode == "live" and self._mt5:
            mt5_tf = getattr(self._mt5, f"TIMEFRAME_{tf}", self._mt5.TIMEFRAME_H1)
            rates = self._mt5.copy_rates_from_pos(self.symbol, mt5_tf, 0, count)
            if rates is not None:
                return [
                    Candle(time=r["time"], open=r["open"], high=r["high"],
                           low=r["low"], close=r["close"], tick_volume=r["tick_volume"],
                           spread=r["spread"])
                    for r in rates
                ]
        # Return cached candles
        return await self.store.get_candles(tf, count)

    async def _seed_history(self) -> None:
        """Seed all timeframes with MT5 historical data on startup."""
        if not self._mt5:
            return
        try:
            for tf_name, tf_val in self.TIMEFRAMES.items():
                count = {"D1": 60, "H4": 250, "H1": 200, "M30": 200,
                         "M15": 200, "M5": 200, "M1": 200}.get(tf_name, 100)
                mt5_tf = getattr(self._mt5, f"TIMEFRAME_{tf_name}", self._mt5.TIMEFRAME_H1)
                rates = self._mt5.copy_rates_from_pos(self.symbol, mt5_tf, 0, count)
                if rates is not None and len(rates) > 0:
                    for r in rates:
                        c = Candle(
                            time=r["time"], open=r["open"], high=r["high"],
                            low=r["low"], close=r["close"], tick_volume=r["tick_volume"],
                            spread=r["spread"] if "spread" in r.dtype.names else 0,
                        )
                        await self.store.update_candle(tf_name, c)
                    self.logger.info("Seeded %d %s candles from MT5", len(rates), tf_name)
        except Exception as e:
            self.logger.error("History seeding failed: %s", e)
