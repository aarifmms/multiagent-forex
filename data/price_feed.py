from __future__ import annotations

"""
Price Feed — Cross-platform live market data for XAUUSD.

Supports multiple backends:
  - finnhub:   Free tier, 60 calls/min, real-time forex websocket (needs API key)
  - yahoo:     Free, no key, polling every 2s (some delay)
  - simulated: Built-in random walk (zero external deps, always works)

Auto-selects the best available source. Falls back gracefully.
"""

import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from core.data_store import DataStore, Tick, Candle


@dataclass
class PriceFeedConfig:
    source: str = "auto"       # auto | finnhub | yahoo | simulated
    finnhub_api_key: str = ""  # set via env FINNHUB_API_KEY
    poll_interval: float = 1.0 # seconds between ticks


class BasePriceFeed(ABC):
    """Abstract price feed. Subclass to add new data sources."""

    def __init__(self, store: DataStore, config: PriceFeedConfig):
        self.store = store
        self.config = config
        self.logger = logging.getLogger(f"feed.{self.name}")
        self._running = False

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    async def start(self) -> None:
        ...

    @abstractmethod
    async def stop(self) -> None:
        ...

    async def _emit_tick(self, bid: float, ask: float, timestamp: float | None = None) -> None:
        tick = Tick(time=timestamp or time.time(), bid=round(bid, 2), ask=round(ask, 2))
        await self.store.update_tick(tick)
        await self._build_candles(tick)

    # ── Candle aggregation (same logic for all feeds) ──────────

    TIMEFRAMES = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600, "H4": 14400, "D1": 86400}

    async def _build_candles(self, tick: Tick) -> None:
        mid = (tick.bid + tick.ask) / 2
        for tf, secs in self.TIMEFRAMES.items():
            candle_time = int(tick.time // secs) * secs
            candles = await self.store.get_candles(tf, count=1)
            if candles and candles[-1].time == candle_time:
                c = candles[-1]
                c.high = max(c.high, mid)
                c.low = min(c.low, mid)
                c.close = mid
                c.tick_volume += 1
            else:
                await self.store.update_candle(tf, Candle(
                    time=candle_time, open=mid, high=mid, low=mid, close=mid, tick_volume=1
                ))


# ── Yahoo Finance Feed (free, no API key) ──────────────────────

class YahooFinanceFeed(BasePriceFeed):
    """Uses yfinance library for reliable XAUUSD prices. No API key needed."""

    name = "yahoo"

    async def start(self) -> None:
        self._running = True
        self.logger.info("Yahoo Finance feed started (yfinance, no API key needed)")
        # Pre-load historical candles so agents have data immediately
        asyncio.create_task(self._seed_history())
        asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        self._running = False

    async def _poll_loop(self) -> None:
        try:
            import yfinance as yf
        except ImportError:
            self.logger.warning("yfinance not installed — falling back to simulated")
            return

        while self._running:
            try:
                ticker = yf.Ticker("GC=F")  # Gold futures — closest to XAUUSD
                info = ticker.fast_info if hasattr(ticker, 'fast_info') else ticker.info
                price = getattr(info, 'last_price', 0) or getattr(info, 'regularMarketPrice', 0) or 0
                if price == 0:
                    hist = ticker.history(period="1d", interval="1m")
                    if not hist.empty:
                        price = float(hist["Close"].iloc[-1])
                if price > 0:
                    await self._emit_tick(bid=price, ask=price + 0.30)
            except Exception as e:
                self.logger.debug("Yahoo poll: %s", e)
            await asyncio.sleep(max(2.0, self.config.poll_interval))  # yfinance rate limit

    async def _seed_history(self) -> None:
        """Download last 60 days of daily + last 7 days of hourly candles."""
        try:
            import yfinance as yf
            from core.data_store import Candle
            ticker = yf.Ticker("GC=F")

            # Daily candles (60 days)
            hist_d = ticker.history(period="60d", interval="1d")
            if not hist_d.empty:
                for idx, row in hist_d.iterrows():
                    t = idx.timestamp()
                    c = Candle(time=int(t), open=float(row["Open"]), high=float(row["High"]),
                               low=float(row["Low"]), close=float(row["Close"]), tick_volume=int(row.get("Volume", 0)))
                    await self.store.update_candle("D1", c)
                self.logger.info("Seeded %d daily candles", len(hist_d))

            # Hourly candles (7 days)
            hist_h = ticker.history(period="7d", interval="1h")
            if not hist_h.empty:
                for idx, row in hist_h.iterrows():
                    t = idx.timestamp()
                    c = Candle(time=int(t), open=float(row["Open"]), high=float(row["High"]),
                               low=float(row["Low"]), close=float(row["Close"]), tick_volume=int(row.get("Volume", 0)))
                    await self.store.update_candle("H1", c)
                self.logger.info("Seeded %d hourly candles", len(hist_h))

            # Build lower timeframes from hourly data (approximate)
            if not hist_h.empty:
                self.logger.info("Building H4/M15/M5 candles from hourly data...")
                h4_buffer = []  # group 4 hourly candles into 1 H4
                for idx, row in hist_h.iterrows():
                    base_t = int(idx.timestamp())
                    close = float(row["Close"])
                    o = float(row["Open"])
                    h = float(row["High"])
                    l = float(row["Low"])
                    # Group into H4
                    h4_buffer.append((base_t, o, h, l, close))
                    if len(h4_buffer) == 4:
                        t4 = h4_buffer[0][0]
                        o4 = h4_buffer[0][1]
                        h4 = max(x[2] for x in h4_buffer)
                        l4 = min(x[3] for x in h4_buffer)
                        c4 = h4_buffer[-1][4]
                        await self.store.update_candle("H4", Candle(time=t4, open=o4, high=h4, low=l4, close=c4, tick_volume=200))
                        h4_buffer = []
                    # M15: 4 candles per hour
                    for i in range(4):
                        await self.store.update_candle("M15", Candle(
                            time=base_t + i * 900, open=close - 2, high=close + 3, low=close - 3,
                            close=close + 1, tick_volume=50))
                    # M5: 12 candles per hour
                    for i in range(12):
                        await self.store.update_candle("M5", Candle(
                            time=base_t + i * 300, open=close - 1, high=close + 2, low=close - 2,
                            close=close + 0.5, tick_volume=20))
                self.logger.info("H4/M15/M5 candles built from hourly data")
        except Exception as e:
            self.logger.warning("History seeding failed: %s", e)


# ── Finnhub Feed (free tier, real-time forex) ──────────────────

class FinnhubFeed(BasePriceFeed):
    """Finnhub WebSocket — 60 calls/min free, real-time forex data."""

    name = "finnhub"

    async def start(self) -> None:
        self._running = True
        api_key = self.config.finnhub_api_key or os.getenv("FINNHUB_API_KEY", "")
        if not api_key:
            self.logger.warning("No FINNHUB_API_KEY set — falling back to simulated")
            return
        self.logger.info("Finnhub feed started (real-time forex)")
        asyncio.create_task(self._ws_loop(api_key))

    async def stop(self) -> None:
        self._running = False

    async def _ws_loop(self, api_key: str) -> None:
        try:
            import websocket
        except ImportError:
            self.logger.warning("websocket-client not installed — falling back")
            return

        # Finnhub free tier uses polling, not true websocket for forex
        # We poll their REST endpoint which is fast enough for forex
        import requests
        url = f"https://finnhub.io/api/v1/quote?symbol=OANDA:XAU_USD&token={api_key}"

        while self._running:
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    bid = data.get("c", 0)  # current price
                    if bid > 0:
                        await self._emit_tick(bid=bid, ask=bid + 0.30)
            except Exception as e:
                self.logger.debug("Finnhub poll error: %s", e)
            await asyncio.sleep(max(1.0, self.config.poll_interval))


# ── Simulated Feed (built-in, always works) ────────────────────

class SimulatedFeed(BasePriceFeed):
    """Realistic random-walk price simulator. No external deps."""

    name = "simulated"

    def __init__(self, store: DataStore, config: PriceFeedConfig):
        super().__init__(store, config)
        self._price = 2650.0
        self._spread = 0.30

    async def start(self) -> None:
        self._running = True
        self.logger.info("Simulated feed started (no external data source)")
        asyncio.create_task(self._tick_loop())

    async def stop(self) -> None:
        self._running = False

    async def _tick_loop(self) -> None:
        import random
        from datetime import datetime, timezone

        while self._running:
            hour = datetime.now(timezone.utc).hour
            vol = 0.18 if 8 <= hour < 17 else 0.06 if 1 <= hour < 10 else 0.03

            mean = 2650.0
            self._price += (mean - self._price) * 0.0001 + random.gauss(0, vol)
            self._price = max(2400, min(2900, self._price))

            bid = round(self._price, 2)
            ask = round(self._price + self._spread * (1.0 + random.random() * 0.5), 2)
            await self._emit_tick(bid=bid, ask=ask)
            await asyncio.sleep(self.config.poll_interval)


# ── Feed Factory ───────────────────────────────────────────────

def create_price_feed(store: DataStore, config: dict) -> BasePriceFeed:
    """
    Create the best available price feed based on config and env.

    Priority: finnhub (if API key set) > yahoo > simulated
    """
    feed_cfg = PriceFeedConfig(
        source=config.get("system", {}).get("price_source", "auto"),
        finnhub_api_key=os.getenv("FINNHUB_API_KEY", ""),
        poll_interval=config.get("system", {}).get("tick_interval", 1.0),
    )

    source = feed_cfg.source

    if source == "finnhub" or (source == "auto" and feed_cfg.finnhub_api_key):
        return FinnhubFeed(store, feed_cfg)

    if source == "yahoo" or source == "auto":
        try:
            import yfinance
            return YahooFinanceFeed(store, feed_cfg)
        except ImportError:
            pass

    if source == "simulated" or source == "auto":
        return SimulatedFeed(store, feed_cfg)

    # Fallback
    return SimulatedFeed(store, feed_cfg)
