from __future__ import annotations

"""
Strategy Agent — Multi-timeframe technical analysis engine.

Runs EMA crossover, RSI, MACD, and ATR-based analysis across M5, M15, H1, H4, D1.
Each timeframe produces a sub-signal. Higher timeframes carry more weight.
The final signal is a weighted blend with entry/SL/TP levels derived from ATR.
"""

from agents.base_agent import BaseAgent
from core.signal_bus import SignalDirection
from core.data_store import Candle


class StrategyAgent(BaseAgent):
    """
    Core technical analysis agent for XAUUSD.

    Weight per timeframe:
      M5  → 0.10  (scalping context)
      M15 → 0.15  (entry timing)
      H1  → 0.25  (intraday trend)
      H4  → 0.25  (swing structure)
      D1  → 0.25  (macro trend)
    """

    TF_WEIGHTS = {"M5": 0.10, "M15": 0.15, "H1": 0.25, "H4": 0.25, "D1": 0.25}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ema_fast = self.config.get("ema_fast", 9)
        self.ema_slow = self.config.get("ema_slow", 21)
        self.ema_trend = self.config.get("ema_trend", 50)
        self.rsi_period = self.config.get("rsi_period", 14)
        self.rsi_os = self.config.get("rsi_oversold", 30)
        self.rsi_ob = self.config.get("rsi_overbought", 70)
        self.atr_period = self.config.get("atr_period", 14)

    async def analyze(self):
        timeframes = ["M5", "M15", "H1", "H4", "D1"]
        scores = []
        details = {}
        current_price = self.latest_tick.bid if self.latest_tick else None

        for tf in timeframes:
            candles = await self.get_candles(tf, count=200)
            if len(candles) < 60:
                continue
            tf_score, tf_detail = self._analyze_timeframe(tf, candles)
            weight = self.TF_WEIGHTS.get(tf, 0.1)
            scores.append(tf_score * weight)
            details[tf] = tf_detail

        if not scores:
            return self._neutral(reason="Insufficient data across timeframes")

        composite = sum(scores)
        confidence = min(1.0, abs(composite))

        # Calculate entry / SL / TP from H1 candles
        h1_candles = await self.get_candles("H1", count=20)
        atr = self._calc_atr(h1_candles, self.atr_period) if h1_candles else 0
        price = current_price or (h1_candles[-1].close if h1_candles else 0)

        if price == 0:
            return self._neutral(reason="No price data available")

        sl_distance = atr * 1.5
        tp_distance = atr * 2.5

        if composite > 0.05:
            signal = self._buy(
                confidence,
                reason=f"Bullish TA (composite={composite:.3f})",
                composite=composite,
                timeframe_signals=details,
            )
            signal.entry_price = price
            signal.stop_loss = round(price - sl_distance, 2)
            signal.take_profit = round(price + tp_distance, 2)
            return signal

        elif composite < -0.05:
            signal = self._sell(
                confidence,
                reason=f"Bearish TA (composite={composite:.3f})",
                composite=composite,
                timeframe_signals=details,
            )
            signal.entry_price = price
            signal.stop_loss = round(price + sl_distance, 2)
            signal.take_profit = round(price - tp_distance, 2)
            return signal

        return self._neutral(reason=f"Neutral TA (composite={composite:.3f})", composite=composite, timeframe_signals=details)

    def _analyze_timeframe(self, tf: str, candles: list[Candle]) -> tuple[float, dict]:
        """Score a single timeframe: -1 (strong sell) to +1 (strong buy)."""
        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        current = closes[-1]

        score = 0.0
        reasons = []

        # 1. EMA Crossover
        ema_f = self._ema(closes, self.ema_fast)
        ema_s = self._ema(closes, self.ema_slow)
        ema_t = self._ema(closes, self.ema_trend)

        if ema_f and ema_s and ema_t:
            if ema_f > ema_s > ema_t:
                score += 0.35
                reasons.append("EMA-bullish-aligned")
            elif ema_f < ema_s < ema_t:
                score -= 0.35
                reasons.append("EMA-bearish-aligned")
            elif ema_f > ema_s and current < ema_t:
                score += 0.15
                reasons.append("EMA-pullback-to-trend")
            elif ema_f < ema_s and current > ema_t:
                score -= 0.15
                reasons.append("EMA-rally-to-resistance")
            else:
                reasons.append("EMA-mixed")

        # 2. RSI
        rsi = self._calc_rsi(closes, self.rsi_period)
        if rsi < self.rsi_os:
            score += 0.25
            reasons.append(f"RSI-oversold({rsi:.0f})")
        elif rsi > self.rsi_ob:
            score -= 0.25
            reasons.append(f"RSI-overbought({rsi:.0f})")
        elif rsi > 55:
            score += 0.1
            reasons.append(f"RSI-bullish({rsi:.0f})")
        elif rsi < 45:
            score -= 0.1
            reasons.append(f"RSI-bearish({rsi:.0f})")
        else:
            reasons.append(f"RSI-neutral({rsi:.0f})")

        # 3. MACD
        macd, macd_signal_line, macd_hist = self._calc_macd(closes)
        if macd is not None:
            if macd_hist[-1] > 0 and macd_hist[-2] <= 0:
                score += 0.2
                reasons.append("MACD-bullish-cross")
            elif macd_hist[-1] < 0 and macd_hist[-2] >= 0:
                score -= 0.2
                reasons.append("MACD-bearish-cross")
            elif macd_hist[-1] > macd_hist[-2]:
                score += 0.1
                reasons.append("MACD-hist-rising")
            elif macd_hist[-1] < macd_hist[-2]:
                score -= 0.1
                reasons.append("MACD-hist-falling")

        # 4. Price relative to recent range
        hh = max(highs[-20:])
        ll = min(lows[-20:])
        rng = hh - ll
        if rng > 0:
            pos = (current - ll) / rng
            if pos > 0.7:
                score += 0.1
                reasons.append("price-near-high")
            elif pos < 0.3:
                score -= 0.1
                reasons.append("price-near-low")

        return max(-1.0, min(1.0, score)), {"score": round(score, 3), "reasons": reasons, "rsi": round(rsi, 1)}

    # ── TA Helpers ─────────────────────────────────────────────

    @staticmethod
    def _ema(prices: list[float], period: int) -> float | None:
        if len(prices) < period:
            return None
        k = 2 / (period + 1)
        ema = sum(prices[:period]) / period
        for p in prices[period:]:
            ema = p * k + ema * (1 - k)
        return ema

    @staticmethod
    def _calc_rsi(prices: list[float], period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        gains = [d for d in deltas[-period:] if d > 0]
        losses = [-d for d in deltas[-period:] if d < 0]
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss == 0:
            return 100.0
        return 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))

    @staticmethod
    def _calc_macd(prices: list[float], fast: int = 12, slow: int = 26, signal: int = 9):
        if len(prices) < slow + signal:
            return None, None, None
        ema_fast = [sum(prices[:fast]) / fast]
        kf = 2 / (fast + 1)
        for p in prices[fast:]:
            ema_fast.append(p * kf + ema_fast[-1] * (1 - kf))

        ema_slow = [sum(prices[:slow]) / slow]
        ks = 2 / (slow + 1)
        for p in prices[slow:]:
            ema_slow.append(p * ks + ema_slow[-1] * (1 - ks))

        offset = slow - fast
        macd_line = [ema_fast[i + offset] - ema_slow[i] for i in range(len(ema_slow))]

        sig_line = [sum(macd_line[:signal]) / signal]
        k = 2 / (signal + 1)
        for v in macd_line[signal:]:
            sig_line.append(v * k + sig_line[-1] * (1 - k))

        hist = [macd_line[i + signal - 1] - sig_line[i] for i in range(len(sig_line))]
        return macd_line, sig_line, hist

    @staticmethod
    def _calc_atr(candles: list[Candle], period: int = 14) -> float:
        if len(candles) < period + 1:
            return 0
        trs = []
        for i in range(1, min(len(candles), period + 1)):
            c = candles[-i]
            prev = candles[-i - 1]
            tr = max(c.high - c.low, abs(c.high - prev.close), abs(c.low - prev.close))
            trs.append(tr)
        return sum(trs) / len(trs) if trs else 0
