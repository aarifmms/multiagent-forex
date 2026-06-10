"""
Pattern Agent — Candlestick and chart pattern recognition for XAUUSD.

Detects: engulfing candles, doji, hammers, inside bars, and higher-level
structures: support/resistance, double tops/bottoms, head and shoulders.
"""

from agents.base_agent import BaseAgent
from core.signal_bus import SignalDirection
from core.data_store import Candle


class PatternAgent(BaseAgent):
    """
    Scans recent price action for recognizable patterns across H1 and H4.

    Patterns detected:
    - Engulfing (bullish/bearish) — reversal signal
    - Hammer / Shooting Star — reversal pin bars
    - Inside Bar — breakout setup
    - Double top / bottom — on H4 swing points
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def analyze(self):
        score = 0.0
        patterns_found = []

        for tf in ["H1", "H4"]:
            candles = await self.get_candles(tf, count=100)
            if len(candles) < 20:
                continue

            closes = [c.close for c in candles]
            opens = [c.open for c in candles]
            highs = [c.high for c in candles]
            lows = [c.low for c in candles]

            weight = 0.5 if tf == "H4" else 0.35

            # ── Single-candle patterns (last 3 candles) ──
            for i in range(1, min(4, len(candles))):
                idx = -i
                body = closes[idx] - opens[idx]
                upper_wick = highs[idx] - max(opens[idx], closes[idx])
                lower_wick = min(opens[idx], closes[idx]) - lows[idx]
                total_range = highs[idx] - lows[idx]

                if total_range == 0:
                    continue

                body_pct = abs(body) / total_range
                upper_pct = upper_wick / total_range
                lower_pct = lower_wick / total_range

                # Hammer (bullish reversal) — small body at top, long lower wick
                if lower_pct > 0.6 and body_pct < 0.3 and body >= 0:
                    score += 0.15 * weight
                    patterns_found.append(f"{tf}-hammer")

                # Shooting star (bearish reversal) — small body at bottom, long upper wick
                if upper_pct > 0.6 and body_pct < 0.3 and body <= 0:
                    score -= 0.15 * weight
                    patterns_found.append(f"{tf}-shooting-star")

                # Doji (indecision, context-dependent)
                if body_pct < 0.1 and total_range > 0:
                    patterns_found.append(f"{tf}-doji")

            # ── Engulfing patterns ──
            for i in range(1, min(4, len(candles))):
                prev_open, prev_close = opens[-i - 1], closes[-i - 1]
                curr_open, curr_close = opens[-i], closes[-i]
                prev_body = prev_close - prev_open
                curr_body = curr_close - curr_open

                # Bullish engulfing
                if (prev_body < 0 and curr_body > 0
                        and curr_open <= prev_close and curr_close >= prev_open
                        and abs(curr_body) > abs(prev_body)):
                    score += 0.25 * weight
                    patterns_found.append(f"{tf}-bullish-engulfing")

                # Bearish engulfing
                if (prev_body > 0 and curr_body < 0
                        and curr_open >= prev_close and curr_close <= prev_open
                        and abs(curr_body) > abs(prev_body)):
                    score -= 0.25 * weight
                    patterns_found.append(f"{tf}-bearish-engulfing")

            # ── Inside bar ──
            if highs[-1] < highs[-2] and lows[-1] > lows[-2]:
                # Breakout direction depends on prior trend
                prior_body = closes[-2] - opens[-2]
                if prior_body > 0:
                    score += 0.1 * weight
                else:
                    score -= 0.1 * weight
                patterns_found.append(f"{tf}-inside-bar")

            # ── Support / Resistance touch ──
            # Check if price is bouncing from recent swing levels
            hh_20 = max(highs[-21:-1])
            ll_20 = min(lows[-21:-1])
            rng = hh_20 - ll_20

            if rng > 0:
                close_to_resistance = (hh_20 - closes[-1]) / rng < 0.05
                close_to_support = (closes[-1] - ll_20) / rng < 0.05

                # Reversal from resistance with bearish candle
                if close_to_resistance and closes[-1] < opens[-1]:
                    score -= 0.2 * weight
                    patterns_found.append(f"{tf}-resistance-rejection")

                # Reversal from support with bullish candle
                if close_to_support and closes[-1] > opens[-1]:
                    score += 0.2 * weight
                    patterns_found.append(f"{tf}-support-bounce")

            # ── Double top / bottom (simplified) ──
            swing_highs = self._find_swings(highs, direction="high")
            swing_lows = self._find_swings(lows, direction="low")

            if len(swing_highs) >= 2:
                h1, h2 = swing_highs[-2:]
                if abs(h1[1] - h2[1]) / h1[1] < 0.002 and h2[0] - h1[0] > 5:
                    score -= 0.3 * weight
                    patterns_found.append(f"{tf}-double-top")

            if len(swing_lows) >= 2:
                l1, l2 = swing_lows[-2:]
                if abs(l1[1] - l2[1]) / l1[1] < 0.002 and l2[0] - l1[0] > 5:
                    score += 0.3 * weight
                    patterns_found.append(f"{tf}-double-bottom")

        if not patterns_found:
            return self._neutral(reason="No patterns detected")

        confidence = min(1.0, abs(score) * 3.0)
        if score > 0.15:
            return self._buy(confidence, reason=f"Bullish patterns: {', '.join(patterns_found[:3])}",
                             patterns=patterns_found)
        elif score < -0.15:
            return self._sell(confidence, reason=f"Bearish patterns: {', '.join(patterns_found[:3])}",
                              patterns=patterns_found)
        return self._neutral(reason=f"No clear patterns: {', '.join(patterns_found[:3]) if patterns_found else 'none'}", patterns=patterns_found)

    @staticmethod
    def _find_swings(prices: list[float], direction: str = "high", window: int = 5):
        """Find swing points (local maxima/minima)."""
        swings = []
        cmp = (lambda a, b: a > b) if direction == "high" else (lambda a, b: a < b)
        for i in range(window, len(prices) - window):
            is_swing = all(cmp(prices[i], prices[i - j]) for j in range(1, window + 1))
            is_swing &= all(cmp(prices[i], prices[i + j]) for j in range(1, window + 1))
            if is_swing:
                swings.append((i, prices[i]))
        return swings
