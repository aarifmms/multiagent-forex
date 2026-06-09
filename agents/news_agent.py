from __future__ import annotations

"""
News Agent — Economic calendar monitor for XAUUSD.

Generates upcoming high-impact USD events from known schedules (NFP, CPI, FOMC, etc.)
and fetches real-time Fed press releases/speeches from federalreserve.gov JSON.

Strategy:
- Before high-impact news: NEUTRAL to suppress trades (blackout window)
- After bullish gold news (dovish Fed, weak data): BUY
- After bearish gold news (hawkish Fed, strong data): SELL
"""

import asyncio
import hashlib
import logging
import time
from datetime import datetime, timezone, timedelta
from calendar import monthrange

from agents.base_agent import BaseAgent
from core.signal_bus import SignalDirection


# ── Recurring Economic Event Schedule ──────────────────────────

def _nth_weekday(year: int, month: int, weekday: int, n: int) -> int:
    """Return day-of-month for the nth weekday of a month (e.g., 1st Friday)."""
    first_day, days_in_month = monthrange(year, month)
    first_occurrence = (weekday - first_day) % 7 + 1
    day = first_occurrence + (n - 1) * 7
    return day if day <= days_in_month else day - 7


def _generate_upcoming_events() -> list[dict]:
    """Generate the next 14 days of high-impact recurring USD economic events."""
    now = datetime.now(timezone.utc)
    events = []

    for offset in range(14):
        day = now + timedelta(days=offset)
        y, m, d = day.year, day.month, day.day
        dow = day.weekday()  # 0=Monday, 6=Sunday
        dom = day.day  # day of month

        # ── Weekly events ──
        # Initial Jobless Claims (every Thursday)
        if dow == 3:
            events.append(_make_event("Initial Jobless Claims", "HIGH", y, m, d, 8, 30, "USD"))

        # ── Monthly events (specific weekdays) ──
        # Non-Farm Payrolls: 1st Friday of each month
        nfp_day = _nth_weekday(y, m, 4, 1)  # Friday=4, 1st
        if dow == 4 and dom == nfp_day:
            events.append(_make_event("Non-Farm Employment Change", "HIGH", y, m, d, 8, 30, "USD"))
            events.append(_make_event("Unemployment Rate", "HIGH", y, m, d, 8, 30, "USD"))
            events.append(_make_event("Average Hourly Earnings m/m", "MEDIUM", y, m, d, 8, 30, "USD"))

        # CPI m/m: ~10th of each month (typically 2nd Wednesday)
        cpi_day = min(14, max(8, _nth_weekday(y, m, 2, 2)))  # 2nd Wed, clamped 8-14
        if dom == cpi_day:
            events.append(_make_event("CPI m/m", "HIGH", y, m, d, 8, 30, "USD"))
            events.append(_make_event("Core CPI m/m", "HIGH", y, m, d, 8, 30, "USD"))

        # PPI m/m: ~11th (typically a day after CPI)
        ppi_day = min(15, cpi_day + 1)
        if dom == ppi_day:
            events.append(_make_event("PPI m/m", "MEDIUM", y, m, d, 8, 30, "USD"))
            events.append(_make_event("Core PPI m/m", "MEDIUM", y, m, d, 8, 30, "USD"))

        # Retail Sales m/m: ~15th
        retail_day = min(18, _nth_weekday(y, m, 3, 3))  # ~3rd Wed
        if dom == retail_day:
            events.append(_make_event("Retail Sales m/m", "MEDIUM", y, m, d, 8, 30, "USD"))

        # ISM Manufacturing PMI: 1st business day of month
        ism_mfg_day = 1
        first_dow = datetime(y, m, 1).weekday()
        if first_dow == 5: ism_mfg_day = 3  # Sat → Mon
        elif first_dow == 6: ism_mfg_day = 2  # Sun → Mon
        if dom == ism_mfg_day:
            events.append(_make_event("ISM Manufacturing PMI", "MEDIUM", y, m, d, 10, 0, "USD"))

        # ISM Services PMI: 3rd business day
        ism_svc_day = ism_mfg_day + 2
        if dom == ism_svc_day:
            events.append(_make_event("ISM Services PMI", "MEDIUM", y, m, d, 10, 0, "USD"))

        # GDP q/q (advance): ~27th of Jan/Apr/Jul/Oct
        if m in (1, 4, 7, 10) and 25 <= dom <= 30 and dow in (3, 4):
            events.append(_make_event("GDP q/q", "HIGH", y, m, d, 8, 30, "USD"))

        # Consumer Confidence: last Tuesday of month
        if dow == 1 and dom >= 24:
            events.append(_make_event("Consumer Confidence", "MEDIUM", y, m, d, 10, 0, "USD"))

    # ── FOMC Meetings (2026 schedule) ──
    # Jan 27-28, Mar 17-18, May 5-6, Jun 16-17, Jul 28-29, Sep 15-16, Nov 3-4, Dec 15-16
    fomc_dates_2026 = [
        (1, 27), (1, 28), (3, 17), (3, 18), (5, 5), (5, 6),
        (6, 16), (6, 17), (7, 28), (7, 29), (9, 15), (9, 16),
        (11, 3), (11, 4), (12, 15), (12, 16),
    ]
    for fm, fd in fomc_dates_2026:
        ev = _make_event("FOMC Statement", "HIGH", 2026, fm, fd, 14, 0, "USD")
        ev["minutes_until"] = (
            datetime(2026, fm, fd, 14, 0, tzinfo=timezone.utc) - now
        ).total_seconds() / 60
        if -120 < ev["minutes_until"] < 1440:  # Within 1 day window
            events.append(ev)
        # Also add press conferences (day 2 of meeting)
        pc = _make_event("FOMC Press Conference", "HIGH", 2026, fm, fd, 14, 30, "USD")
        pc["minutes_until"] = ev["minutes_until"]
        if -120 < pc["minutes_until"] < 1440:
            events.append(pc)

    return events


def _make_event(name: str, impact: str, year: int, month: int, day: int,
                hour: int, minute: int, currency: str) -> dict:
    """Create an event dict with minutes_until."""
    dt = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    minutes_until = (dt - now).total_seconds() / 60
    return {
        "name": name,
        "impact": impact,
        "currency": currency,
        "minutes_until": round(minutes_until, 1),
        "date": dt.strftime("%a %b %d"),
        "time": dt.strftime("%H:%M UTC"),
        "source": "schedule",
    }


# ── Fed Events (Live JSON) ─────────────────────────────────────

_fed_cache: list[dict] = []
_fed_cache_time = 0.0
FED_CACHE_TTL = 3600  # 1 hour


async def _fetch_fed_events() -> list[dict]:
    """Fetch recent Fed press releases and speeches from federalreserve.gov."""
    global _fed_cache, _fed_cache_time

    if _fed_cache and time.time() - _fed_cache_time < FED_CACHE_TTL:
        return _fed_cache

    try:
        import requests

        url = "https://www.federalreserve.gov/json/ne-press.json"
        text = await asyncio.to_thread(
            lambda: requests.get(url, timeout=10).text
        )
        # Handle BOM
        if text.startswith("﻿"):
            text = text[1:]

        import json
        data = json.loads(text)
        now = datetime.now(timezone.utc)

        events = []
        for item in data:
            # Parse date: "6/2/2026 11:00:00 AM"
            try:
                dt = datetime.strptime(item["d"], "%m/%d/%Y %I:%M:%S %p")
                dt = dt.replace(tzinfo=timezone.utc)
            except (ValueError, KeyError):
                continue

            hours_ago = (now - dt).total_seconds() / 3600

            # Only recent events (last 48h) or upcoming (next 7 days)
            if not (-48 < hours_ago < 168):
                continue

            name = item.get("t", "")
            topic = item.get("pt", "")

            # Filter for gold-relevant events
            relevant_keywords = [
                "monetary", "rate", "fomc", "chair", "inflation",
                "economic", "financial stability", "regulation",
                "supervision", "stress test",
            ]
            is_relevant = any(kw in name.lower() or kw in topic.lower()
                              for kw in relevant_keywords)

            if not is_relevant:
                continue

            impact = "HIGH" if any(k in name.lower() for k in
                                   ["rate", "fomc", "chair", "powell",
                                    "monetary policy", "inflation"]) else "MEDIUM"

            events.append({
                "name": name,
                "impact": impact,
                "currency": "USD",
                "minutes_until": round(-hours_ago * 60, 1),
                "date": dt.strftime("%a %b %d"),
                "time": dt.strftime("%H:%M UTC"),
                "source": "fed",
            })

        _fed_cache = events
        _fed_cache_time = time.time()
        return events
    except Exception:
        return []


# ── News Agent ──────────────────────────────────────────────────

class NewsAgent(BaseAgent):
    """
    Watches economic calendar for high-impact events affecting gold.

    Data sources:
      1. Programmatic recurring event schedule (NFP, CPI, FOMC, etc.)
      2. Federal Reserve press releases & speeches (live JSON feed)

    Strategy:
      - Blackout window before HIGH impact events (15 min default)
      - Score recent events based on bullish/bearish gold implications
    """

    BULLISH_GOLD = {
        # Weaker data → lower rates → weaker USD → bullish gold
        "cpi m/m": -1, "core cpi m/m": -1,
        "ppi m/m": -1, "core ppi m/m": -1,
        "gdp q/q": -1,
        "ism manufacturing pmi": -1, "ism services pmi": -1,
        "retail sales m/m": -1,
        "unemployment rate": 1,
        "initial jobless claims": 1,
        "consumer confidence": -1,
        "average hourly earnings m/m": -1,
        "non-farm employment change": -1,
    }

    BEARISH_GOLD = {
        "cpi m/m": 1, "core cpi m/m": 1,
        "non-farm employment change": 1,
        "federal funds rate": 1,
        "fomc statement": 0, "fomc press conference": 0,
        "consumer confidence": 1,
        "average hourly earnings m/m": 1,
        "retail sales m/m": 1,
        "gdp q/q": 1,
    }

    BLACKOUT_EVENTS = {
        "fomc statement", "federal funds rate",
        "non-farm employment change", "cpi m/m", "core cpi m/m",
        "fomc press conference",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("agent.news")
        self.blackout_minutes = self.config.get("blackout_minutes", 15)

    async def analyze(self):
        # ── Fetch events from both sources in parallel ──
        scheduled, fed_events = await asyncio.gather(
            asyncio.to_thread(_generate_upcoming_events),
            _fetch_fed_events(),
        )

        # Merge: scheduled + Fed events
        all_events = scheduled + fed_events

        # Deduplicate by name
        seen = set()
        unique_events = []
        for ev in all_events:
            key = ev["name"].lower()
            if key not in seen:
                seen.add(key)
                unique_events.append(ev)

        # Update DataStore
        self.store.upcoming_events = unique_events

        # ── Check for active blackout ──
        for ev in unique_events:
            ev_name = ev.get("name", "").lower()
            minutes_until = ev.get("minutes_until", 9999)

            for be in self.BLACKOUT_EVENTS:
                if be in ev_name and 0 < minutes_until < self.blackout_minutes:
                    self.store.active_news_blackout = True
                    self.store.blackout_until = self.clock.now + (self.blackout_minutes * 60)
                    return self._neutral(
                        reason=f"Blackout: {ev_name} in {minutes_until:.0f}min",
                        blackout=True,
                    )

        self.store.active_news_blackout = False
        self.store.blackout_until = 0.0

        # ── Score recent events (last 3 hours) ──
        score = 0.0
        count = 0

        for ev in unique_events:
            minutes_until = ev.get("minutes_until", 9999)
            hours_ago = -minutes_until / 60

            if hours_ago < 0 or hours_ago > 3:
                continue

            ev_name = ev.get("name", "").lower()
            impact = ev.get("impact", "").upper()
            multiplier = {"HIGH": 1.0, "MEDIUM": 0.5, "LOW": 0.2}.get(impact, 0.1)
            decay = max(0, 1.0 - hours_ago / 3)

            # Check matched events
            for pattern, direction in self.BULLISH_GOLD.items():
                if pattern in ev_name:
                    score += direction * multiplier * decay
                    count += 1
                    break
            else:
                for pattern, direction in self.BEARISH_GOLD.items():
                    if pattern in ev_name:
                        score += direction * multiplier * decay
                        count += 1
                        break

        if count == 0:
            return self._neutral(
                reason=f"No recent USD events ({len(unique_events)} upcoming tracked)"
            )

        confidence = min(0.80, abs(score) / max(count, 1))
        if score > 0.15:
            return self._buy(confidence, reason=f"Bullish news (score={score:.2f})")
        elif score < -0.15:
            return self._sell(confidence, reason=f"Bearish news (score={score:.2f})")
        else:
            return self._neutral(reason=f"Neutral news (score={score:.2f})")
