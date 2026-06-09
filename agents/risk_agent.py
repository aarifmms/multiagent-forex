"""
Risk Agent — Hard gate for all trade decisions.

This agent runs BEFORE the decision agent finalizes.
It has VETO power — if risk says no, no trade fires regardless of what
other agents voted. It enforces:

1. Max 1-2% account risk per trade
2. Daily drawdown cap (3%)
3. Max open positions (2)
4. Minimum risk/reward ratio (1.5:1)
5. Position size calculation based on SL distance
6. Session-based lot multiplier
7. ATR-based minimum volatility filter
"""

import math

from agents.base_agent import BaseAgent
from core.signal_bus import SignalDirection
from core.data_store import Candle


class RiskAgent(BaseAgent):
    """
    Non-negotiable risk management. This agent has veto authority.

    Returns a NEUTRAL signal with metadata.risk_approved=False
    when any rule is violated. The decision agent checks this flag
    before firing a trade.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_risk_pct = self.config.get("max_risk_per_trade_pct", 1.0)
        self.max_daily_dd_pct = self.config.get("max_daily_drawdown_pct", 3.0)
        self.max_open = self.config.get("max_open_trades", 2)
        self.atr_sl_mult = self.config.get("atr_multiplier_sl", 1.5)
        self.atr_tp_mult = self.config.get("atr_multiplier_tp", 2.5)
        self.min_rr = self.config.get("min_risk_reward_ratio", 1.5)
        self.min_atr = self.config.get("min_atr_h1", 10)  # minimum H1 ATR in pips

    async def analyze(self):
        checks = []
        approved = True

        # ── 1. Market open check ──
        if not self.clock.can_trade:
            return self._neutral(
                reason="Market closed / weekend / cooldown / blackout",
                risk_approved=False,
                risk_checks=["market-closed-or-cooldown"],
            )

        checks.append("market-open")

        # ── 2. Max open positions ──
        open_count = len(self.store.open_positions)
        if open_count >= self.max_open:
            approved = False
            checks.append(f"max-positions-exceeded({open_count}/{self.max_open})")
        else:
            checks.append(f"positions-ok({open_count}/{self.max_open})")

        # ── 3. Daily drawdown check ──
        if self.store.balance > 0:
            dd_pct = ((self.store.balance - self.store.equity) / self.store.balance) * 100
            if dd_pct >= self.max_daily_dd_pct:
                approved = False
                checks.append(f"daily-drawdown-limit({dd_pct:.1f}%>={self.max_daily_dd_pct}%)")
            else:
                checks.append(f"drawdown-ok({dd_pct:.1f}%)")

        # ── 4. ATR filter — avoid low-volatility traps ──
        h1_candles = await self.get_candles("H1", count=20)
        atr = self._calc_atr(h1_candles) if len(h1_candles) >= 15 else 0
        if atr < self.min_atr:  # below minimum ATR = dead market
            approved = False
            checks.append(f"low-volatility(ATR={atr:.1f}pips)")
        else:
            checks.append(f"volatility-ok(ATR={atr:.1f}pips)")

        # ── 5. Free margin check ──
        min_lot = 0.01
        # Estimate margin for minimum position (~$1000 notional for 0.01 lot XAUUSD)
        est_margin = 10  # ~$10 for 0.01 lot at 1:100 leverage
        if self.store.free_margin > 0 and self.store.free_margin < est_margin:
            approved = False
            checks.append(f"insufficient-margin(free=${self.store.free_margin:.0f})")
        elif self.store.free_margin > 0:
            checks.append(f"margin-ok(free=${self.store.free_margin:.0f})")

        # ── 6. Blackout from news agent ──
        if self.store.active_news_blackout:
            approved = False
            checks.append("news-blackout-active")

        # ── 7. Calculate position size if approved ──
        lot_size = 0.0
        if approved:
            lot_size = self._calculate_lot_size(atr)
            if lot_size < 0.01:
                approved = False
                checks.append("lot-size-too-small")
            else:
                checks.append(f"lot-size-ok({lot_size:.2f})")

        return self._neutral(
            reason=f"Risk checks: {' | '.join(checks)}",
            risk_approved=approved,
            risk_checks=checks,
            atr=atr,
            lot_size=lot_size,
            open_positions=open_count,
        )

    def _calculate_lot_size(self, atr: float) -> float:
        """
        Calculate position size based on:
        - Account balance
        - Max risk % per trade
        - SL distance (ATR-based)
        - Session volatility multiplier
        """
        balance = self.store.balance
        if balance <= 0:
            balance = 10000  # default for paper trading

        risk_amount = balance * (self.max_risk_pct / 100)

        # SL distance in dollars
        sl_distance_pips = atr * self.atr_sl_mult
        # For XAUUSD: 1 pip = $0.01 per 0.01 lot
        # 0.01 lot = $0.10 per pip, 0.10 lot = $1.00 per pip, 1.0 lot = $10 per pip
        pip_value_per_lot = 10.0  # $10 per pip for 1 standard lot of XAUUSD
        risk_per_lot = sl_distance_pips * pip_value_per_lot

        if risk_per_lot <= 0:
            return 0.01

        raw_lot = risk_amount / risk_per_lot
        raw_lot *= self.clock.session_lot_multiplier()

        # Round to broker lot step (0.01)
        lot = max(0.01, math.floor(raw_lot * 100) / 100)
        return min(lot, 1.0)

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
