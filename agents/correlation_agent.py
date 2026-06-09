from __future__ import annotations

"""
Correlation Agent — Inter-market analysis for XAUUSD.

Key relationships:
- DXY (US Dollar Index) ↑ → Gold ↓ (inverse correlation, ~ -0.8)
- US10Y (10Y Treasury Yield) ↑ → Gold ↓ (opportunity cost)
- SPX500 ↑ → Gold ↓ during risk-on, Gold ↑ during uncertainty

Fetches real DXY, US10Y, and SPX500 data via yfinance.
"""

import asyncio
import time

from agents.base_agent import BaseAgent
from core.signal_bus import SignalDirection


# Cache for yfinance data to avoid rate limits
_cache: dict[str, tuple[float, list[float]]] = {}
CACHE_TTL = 3600  # 1 hour


def _get_cached(ticker: str):
    """Return cached closes if fresh, else None."""
    ts, data = _cache.get(ticker, (0, []))
    if time.time() - ts < CACHE_TTL and data:
        return data
    return None


def _set_cache(ticker: str, closes: list[float]):
    _cache[ticker] = (time.time(), closes)


async def _fetch_yahoo(ticker: str, days: int = 60) -> list[float]:
    """Fetch daily closes for a ticker from yfinance. Runs in thread to avoid blocking."""
    cached = _get_cached(ticker)
    if cached:
        return cached

    try:
        import yfinance as yf
        data = await asyncio.to_thread(
            lambda: yf.Ticker(ticker).history(period=f"{days}d")
        )
        if data is None or len(data) < 5:
            return []
        closes = data["Close"].dropna().tolist()
        _set_cache(ticker, closes)
        return closes
    except Exception:
        return []


class CorrelationAgent(BaseAgent):
    """
    Reads correlated instruments to confirm or weaken gold-direction signals.

    Real data sources (via yfinance):
      - DXY:  DX-Y.NYB  (US Dollar Index futures)
      - 10Y:  ^TNX      (CBOE 10-Year Treasury Yield)
      - SPX:  ^GSPC     (S&P 500 Index)

    Weights: DXY=50%, Yields=30%, Equities=20%
    """

    # yfinance ticker symbols
    TICKERS = {
        "dxy": "DX-Y.NYB",
        "us10y": "^TNX",
        "spx": "^GSPC",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_fetch = 0.0

    async def analyze(self):
        # Fetch all three correlated instruments in parallel
        dxy_closes, yield_closes, spx_closes = await asyncio.gather(
            _fetch_yahoo(self.TICKERS["dxy"], 60),
            _fetch_yahoo(self.TICKERS["us10y"], 60),
            _fetch_yahoo(self.TICKERS["spx"], 60),
        )

        if not dxy_closes and not yield_closes and not spx_closes:
            # No external data available — fall back to gold momentum proxy
            return await self._gold_proxy_fallback()

        score = 0.0
        reasons = []
        weight_total = 0.0

        # ── DXY Analysis (50% weight, inverse to gold) ──
        if dxy_closes and len(dxy_closes) >= 20:
            dxy_score = self._score_inverse(dxy_closes, "DXY")
            score += dxy_score * 0.50
            weight_total += 0.50
            reasons.append(f"DXY={dxy_score:+.2f}")

        # ── US10Y Analysis (30% weight, inverse to gold) ──
        if yield_closes and len(yield_closes) >= 20:
            yld_score = self._score_inverse(yield_closes, "US10Y")
            score += yld_score * 0.30
            weight_total += 0.30
            reasons.append(f"US10Y={yld_score:+.2f}")

        # ── SPX500 Analysis (20% weight, context-dependent) ──
        if spx_closes and len(spx_closes) >= 20:
            spx_score = self._score_equities(spx_closes)
            score += spx_score * 0.20
            weight_total += 0.20
            reasons.append(f"SPX={spx_score:+.2f}")

        if weight_total == 0:
            return await self._gold_proxy_fallback()

        # Normalize score if some data sources were missing
        if weight_total < 1.0 and weight_total > 0:
            score = score / weight_total

        confidence = min(0.85, abs(score) * 1.3)

        if score > 0.08:
            return self._buy(confidence,
                             reason=" | ".join(reasons),
                             composite=round(score, 3))
        elif score < -0.08:
            return self._sell(confidence,
                              reason=" | ".join(reasons),
                              composite=round(score, 3))
        else:
            return self._neutral(reason=f"Mixed correlation ({' | '.join(reasons)})",
                                 composite=round(score, 3))

    # ── Scoring helpers ──────────────────────────────────────

    @staticmethod
    def _score_inverse(closes: list[float], label: str = "") -> float:
        """
        Score an inversely correlated instrument.
        Rising DXY/yields → bearish gold (negative score).
        Falling DXY/yields → bullish gold (positive score).
        """
        current = closes[-1]
        sma_20 = sum(closes[-20:]) / 20
        sma_50 = sum(closes[-min(50, len(closes)):]) / min(50, len(closes))

        score = 0.0

        # Trend: price vs SMAs
        if current < sma_20 < sma_50:
            score += 0.35  # strong downtrend → bullish gold
        elif current < sma_20:
            score += 0.20  # mild downtrend
        elif current > sma_20 > sma_50:
            score -= 0.35  # strong uptrend → bearish gold
        elif current > sma_20:
            score -= 0.20

        # Rate of change (5-day)
        if len(closes) >= 6:
            roc_5d = (closes[-1] - closes[-6]) / closes[-6]
            score -= roc_5d * 5  # rising instrument = negative for gold
            score = max(-0.5, min(0.5, score))

        # Acceleration (is trend strengthening?)
        if len(closes) >= 12:
            roc_10d = (closes[-1] - closes[-11]) / closes[-11]
            roc_5d_prev = (closes[-6] - closes[-11]) / closes[-11] if closes[-11] != 0 else 0
            if abs(roc_5d_prev) > 0.001:
                accel = (roc_10d - roc_5d_prev) / abs(roc_5d_prev)
                score -= accel * 0.15  # accelerating up = worse for gold

        return round(max(-0.5, min(0.5, score)), 3)

    @staticmethod
    def _score_equities(closes: list[float]) -> float:
        """
        Score S&P 500 for gold implications.
        Risk-on (SPX up) = mildly bearish gold.
        Risk-off (SPX down sharply) = bullish gold (safe haven).
        """
        current = closes[-1]
        sma_20 = sum(closes[-20:]) / 20

        score = 0.0

        # Trend direction
        if current > sma_20:
            score -= 0.10  # risk-on, mildly bearish gold
        else:
            score += 0.10

        # Drawdown / sell-off detection (sharp drops = gold safe haven bid)
        if len(closes) >= 21:
            high_20 = max(closes[-21:-1])  # exclude current
            drawdown = (current - high_20) / high_20
            if drawdown < -0.03:  # >3% drawdown
                score += 0.25  # fear bid for gold
            elif drawdown < -0.01:
                score += 0.10

        # VIX proxy: large daily moves = uncertainty
        if len(closes) >= 6:
            daily_changes = [abs((closes[i] - closes[i-1]) / closes[i-1])
                             for i in range(-5, 0)]
            avg_change = sum(daily_changes) / len(daily_changes)
            if avg_change > 0.015:  # >1.5% avg daily move
                score += 0.15  # high volatility = uncertainty = gold bid

        return round(max(-0.3, min(0.3, score)), 3)

    # ── Fallback (when no external data available) ───────────

    async def _gold_proxy_fallback(self):
        """Use gold's own momentum as a rough DXY inverse proxy."""
        gold_h4 = await self.get_candles("H4", count=50)
        gold_d1 = await self.get_candles("D1", count=30)

        if len(gold_d1) < 20:
            return self._neutral(reason="No correlation data available")

        closes_d1 = [c.close for c in gold_d1]
        closes_h4 = [c.close for c in gold_h4] if len(gold_h4) >= 12 else closes_d1

        ema_20_d1 = sum(closes_d1[-20:]) / 20
        ema_50_d1 = sum(closes_d1[-30:]) / min(len(closes_d1), 30)
        d1_trend = "UP" if ema_20_d1 > ema_50_d1 else "DOWN"

        ema_12_h4 = sum(closes_h4[-12:]) / 12
        h4_len = min(26, len(closes_h4))
        ema_26_h4 = sum(closes_h4[-h4_len:]) / h4_len if h4_len > 0 else 0
        h4_trend = "UP" if ema_12_h4 > ema_26_h4 else "DOWN"

        score = 0.0
        reasons = []

        if d1_trend == "UP":
            score += 0.30
            reasons.append("D1-gold-up(DXY-proxy-falling)")
        else:
            score -= 0.30
            reasons.append("D1-gold-down(DXY-proxy-rising)")

        if h4_trend == "UP":
            score += 0.20
            reasons.append("H4-gold-up")
        else:
            score -= 0.20
            reasons.append("H4-gold-down")

        confidence = min(0.75, abs(score) * 1.2)
        if score > 0.05:
            return self._buy(confidence, reason=" | ".join(reasons))
        elif score < -0.05:
            return self._sell(confidence, reason=" | ".join(reasons))
        return self._neutral(reason="Mixed correlation (proxy)")
