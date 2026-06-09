from __future__ import annotations

"""
Sentiment Agent — Analyzes market positioning and sentiment for XAUUSD.

Data sources (in priority order):
1. COT (Commitment of Traders) — CFTC public API, real institutional positioning
2. Retail sentiment — IG Client Sentiment / price-based proxy
3. Price-action sentiment — momentum, RSI, distance from extremes

COT data: fetched from CFTC public API, updated every Friday.
Gold futures COT report code: 088691 (CME Gold Futures).
"""

import asyncio
import json
import logging
import time

from agents.base_agent import BaseAgent
from core.signal_bus import SignalDirection
from core.data_store import Candle


# COT cache (CFTC data updates weekly, no need to refetch)
_cot_cache: dict = {}
_cot_cache_time = 0.0
COT_CACHE_TTL = 3600 * 6  # 6 hours


async def _fetch_cot_data() -> dict | None:
    """Fetch latest COT report for gold futures from CFTC API."""
    global _cot_cache, _cot_cache_time

    if _cot_cache and time.time() - _cot_cache_time < COT_CACHE_TTL:
        return _cot_cache

    try:
        import requests

        # CFTC public API — Gold Futures (CME, code 088691)
        # Fields: market_and_exchange_names, report_date_as_yyyy_mm_dd,
        #         noncomm_positions_long_all, noncomm_positions_short_all,
        #         comm_positions_long_all, comm_positions_short_all,
        #         nonrept_positions_long_all, nonrept_positions_short_all
        url = (
            "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
            "?commodity=GOLD"
            "&$order=report_date_as_yyyy_mm_dd DESC"
            "&$limit=3"
        )
        resp = await asyncio.to_thread(lambda: requests.get(url, timeout=10))
        if resp.status_code != 200:
            return None

        data = resp.json()
        if not data:
            return None

        _cot_cache = {"reports": data, "fetched_at": time.time()}
        _cot_cache_time = time.time()
        return _cot_cache
    except Exception:
        return None


def _analyze_cot(reports: list[dict]) -> dict | None:
    """Analyze COT data for sentiment signal."""
    if not reports:
        return None

    latest = reports[0]
    noncomm_long = int(latest.get("noncomm_positions_long_all", 0))
    noncomm_short = int(latest.get("noncomm_positions_short_all", 0))
    comm_long = int(latest.get("comm_positions_long_all", 0))
    comm_short = int(latest.get("comm_positions_short_all", 0))

    if noncomm_long + noncomm_short == 0:
        return None

    # Non-commercial = speculators (trend followers)
    # Commercial = hedgers (smart money, usually contrarian)
    # Extreme non-commercial long = vulnerable to reversal
    # Extreme commercial short = hedgers expect lower prices

    noncomm_total = noncomm_long + noncomm_short
    noncomm_net = noncomm_long - noncomm_short
    noncomm_net_pct = noncomm_net / noncomm_total

    comm_total = comm_long + comm_short
    comm_net_pct = (comm_long - comm_short) / comm_total if comm_total > 0 else 0

    # Track weekly change if we have multiple reports
    net_change = 0.0
    if len(reports) >= 2:
        prev = reports[1]
        prev_nl = int(prev.get("noncomm_positions_long_all", 0))
        prev_ns = int(prev.get("noncomm_positions_short_all", 0))
        prev_total = prev_nl + prev_ns
        if prev_total > 0:
            prev_net = (prev_nl - prev_ns) / prev_total
            net_change = noncomm_net_pct - prev_net

    return {
        "noncomm_net_pct": round(noncomm_net_pct, 4),
        "comm_net_pct": round(comm_net_pct, 4),
        "noncomm_net_change": round(net_change, 4),
        "report_date": latest.get("report_date_as_yyyy_mm_dd", "unknown"),
    }


class SentimentAgent(BaseAgent):
    """
    Measures whether the market is overly bullish or bearish.

    Primary: CFTC COT data (institutional positioning)
    Fallback: Price-action sentiment proxies (momentum, RSI, range position)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("agent.sentiment")

    async def analyze(self):
        # ── Try COT data first ──
        cot_raw = await _fetch_cot_data()
        if cot_raw and cot_raw.get("reports"):
            cot_analysis = _analyze_cot(cot_raw["reports"])
            if cot_analysis:
                return self._cot_signal(cot_analysis)

        # ── Fallback: price-action sentiment ──
        return await self._price_based_sentiment()

    def _cot_signal(self, cot: dict):
        """
        Convert COT data into a sentiment signal.

        Rules:
        - Extreme speculator long (>60% net) = bullish but cautious (crowded trade)
        - Extreme speculator short (<-30% net) = bearish but possibly oversold
        - Commercials heavily short = smart money bearish (negative signal)
        - Net change: speculators adding longs = bullish momentum
        """
        net = cot["noncomm_net_pct"]
        comm_net = cot["comm_net_pct"]
        change = cot["noncomm_net_change"]

        score = 0.0
        reasons = []

        # Speculator positioning
        if net > 0.50:
            # Very crowded long — still bullish but reduce confidence
            score += 0.25
            reasons.append(f"Spec-long-extreme({net:.1%})")
        elif net > 0.20:
            score += 0.35
            reasons.append(f"Spec-long({net:.1%})")
        elif net > 0.05:
            score += 0.15
            reasons.append(f"Spec-slightly-long({net:.1%})")
        elif net < -0.30:
            score -= 0.25
            reasons.append(f"Spec-short-extreme({net:.1%})")
        elif net < -0.10:
            score -= 0.35
            reasons.append(f"Spec-short({net:.1%})")
        elif net < -0.05:
            score -= 0.15
            reasons.append(f"Spec-slightly-short({net:.1%})")
        else:
            reasons.append(f"Spec-neutral({net:.1%})")

        # Commercial (smart money) check — inverse weight
        if comm_net < -0.30:
            score -= 0.20  # hedgers heavily short = bearish
            reasons.append("Commercials-heavy-short")
        elif comm_net > 0.30:
            score += 0.20
            reasons.append("Commercials-heavy-long")

        # Momentum of change
        if change > 0.10:
            score += 0.15
            reasons.append(f"Specs-adding-longs({change:.1%})")
        elif change < -0.10:
            score -= 0.15
            reasons.append(f"Specs-covering-longs({change:.1%})")

        confidence = min(0.80, abs(score) * 1.2 + 0.10)
        reason_str = "COT: " + " | ".join(reasons)

        if score > 0.08:
            return self._buy(confidence, reason=reason_str, cot=cot)
        elif score < -0.08:
            return self._sell(confidence, reason=reason_str, cot=cot)
        else:
            return self._neutral(reason=reason_str, cot=cot)

    async def _price_based_sentiment(self):
        """Fallback: derive sentiment from price action across multiple timeframes."""
        candles_d1 = await self.get_candles("D1", count=20)
        if len(candles_d1) < 20:
            return self._neutral(reason="Insufficient data for sentiment analysis")

        closes = [c.close for c in candles_d1]
        highs = [c.high for c in candles_d1]
        lows = [c.low for c in candles_d1]
        current = closes[-1]

        sma_50 = sum(closes[-20:]) / min(len(closes), 20)
        trend_bias = 1 if current > sma_50 else -1

        rsi = self._calc_rsi(closes, 14)

        momentum_5d = (closes[-1] - closes[-6]) / closes[-6] if len(closes) >= 6 else 0

        hh_20 = max(highs[-20:])
        ll_20 = min(lows[-20:])
        range_20 = hh_20 - ll_20
        position_in_range = (current - ll_20) / range_20 if range_20 > 0 else 0.5

        score = 0.0
        score += trend_bias * 0.30

        if rsi > 70:
            score -= 0.20
        elif rsi < 30:
            score += 0.20
        elif rsi > 60:
            score += 0.10
        elif rsi < 40:
            score -= 0.10

        score += max(-0.20, min(0.20, momentum_5d * 10))

        if position_in_range > 0.80:
            score -= 0.15
        elif position_in_range < 0.20:
            score += 0.15

        confidence = min(0.75, abs(score))
        if score > 0.10:
            return self._buy(confidence,
                             reason=f"Bullish proxy (RSI={rsi:.0f})",
                             rsi=rsi, momentum=momentum_5d)
        elif score < -0.10:
            return self._sell(confidence,
                              reason=f"Bearish proxy (RSI={rsi:.0f})",
                              rsi=rsi, momentum=momentum_5d)
        else:
            return self._neutral(reason=f"Neutral proxy (RSI={rsi:.0f})", rsi=rsi)

    @staticmethod
    def _calc_rsi(prices: list[float], period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        gains = [d for d in deltas if d > 0]
        losses = [-d for d in deltas if d < 0]
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))
