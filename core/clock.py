"""
Trading Clock — Session-aware time management with timezone awareness.

Determines which trading sessions are active (Asian / London / New York),
whether the market is open, and enforces cooldown / blackout windows.
"""

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from zoneinfo import ZoneInfo


class Session(Enum):
    ASIAN = "ASIAN"
    LONDON = "LONDON"
    NEW_YORK = "NEW_YORK"
    LONDON_NY_OVERLAP = "LONDON_NY_OVERLAP"
    CLOSED = "CLOSED"


SESSION_HOURS = {
    Session.ASIAN: ("Asia/Tokyo", 0, 9),
    Session.LONDON: ("Europe/London", 8, 17),
    Session.NEW_YORK: ("America/New_York", 8, 17),
}


def _hour_in_session(tz_name: str, start_h: int, end_h: int) -> bool:
    now = datetime.now(ZoneInfo(tz_name))
    return start_h <= now.hour < end_h


def get_active_session() -> Session:
    london = _hour_in_session("Europe/London", 8, 17)
    new_york = _hour_in_session("America/New_York", 8, 17)
    asian = _hour_in_session("Asia/Tokyo", 0, 9)

    if london and new_york:
        return Session.LONDON_NY_OVERLAP
    if london:
        return Session.LONDON
    if new_york:
        return Session.NEW_YORK
    if asian:
        return Session.ASIAN
    return Session.CLOSED


def is_market_open() -> bool:
    return get_active_session() != Session.CLOSED


def is_weekend() -> bool:
    now = datetime.now(timezone.utc)
    return now.weekday() >= 5  # Saturday = 5, Sunday = 6


@dataclass
class TradingClock:
    cooldown_seconds: float = 300.0    # 5 min between trades
    force_session: bool = False        # override session checks (testing only)

    _last_trade_time: float = 0.0
    _blackout_until: float = 0.0

    @property
    def now(self) -> float:
        return time.time()

    @property
    def active_session(self) -> Session:
        return get_active_session()

    @property
    def is_in_cooldown(self) -> bool:
        return (self.now - self._last_trade_time) < self.cooldown_seconds

    @property
    def is_in_blackout(self) -> bool:
        return self.now < self._blackout_until

    @property
    def can_trade(self) -> bool:
        if self.force_session:
            return (not self.is_in_cooldown
                    and not self.is_in_blackout
                    and not is_weekend())
        return (is_market_open()
                and not self.is_in_cooldown
                and not self.is_in_blackout
                and not is_weekend())

    def record_trade(self) -> None:
        self._last_trade_time = self.now

    def set_blackout(self, duration_minutes: float) -> None:
        self._blackout_until = self.now + (duration_minutes * 60)

    def session_lot_multiplier(self) -> float:
        multipliers = {
            Session.ASIAN: 0.5,
            Session.LONDON: 1.0,
            Session.NEW_YORK: 1.0,
            Session.LONDON_NY_OVERLAP: 1.25,
            Session.CLOSED: 0.0,
        }
        return multipliers.get(self.active_session, 0.0)
