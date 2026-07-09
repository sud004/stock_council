#!/usr/bin/env python3
# ============================================================
# trading_calendar.py — Correct Trading Date Resolver
# ============================================================
# PROBLEM THIS SOLVES:
#   night_runner.py was dating every file by "when the code ran"
#   (datetime.now()), not "which trading day's data it analyzed".
#
#   A run at 3:34 AM on June 19 actually analyzed June 18's
#   closing prices (the most recent close available at that
#   moment) — but got saved as "2026-06-19", creating a mismatch
#   between the file date and the actual market data inside it.
#
# THE RULE:
#   NSE market hours: 9:15 AM – 3:30 PM IST, Mon–Fri, non-holiday.
#
#   If now() is BEFORE 3:30 PM on a trading day (or any time on
#   a non-trading day) → the most recent COMPLETE trading day's
#   close is the relevant data → use that as "trading_date".
#
#   If now() is AFTER 3:30 PM on a trading day → today's close
#   is now available → use today as "trading_date".
#
#   Weekends and NSE holidays are walked backward automatically.
#
# USAGE:
#   from trading_calendar import get_trading_date
#   trading_date = get_trading_date()   # '2026-06-18' or '2026-06-19'
# ============================================================

import sys
from pathlib import Path
from datetime import datetime, date, timedelta
import pytz

sys.path.insert(0, str(Path(__file__).parent))

from market_holidays import is_market_holiday

IST = pytz.timezone('Asia/Kolkata')

MARKET_CLOSE_HOUR   = 15   # 3 PM
MARKET_CLOSE_MINUTE = 30   # :30 → 3:30 PM IST


def is_trading_day(d: date) -> bool:
    """True if NSE is open on this date (weekday + not a holiday)."""
    if d.weekday() >= 5:          # Saturday=5, Sunday=6
        return False
    if is_market_holiday(d):
        return False
    return True


def last_trading_day(before: date = None) -> date:
    """
    Return the most recent trading day on or before `before`
    (default: yesterday). Walks backward past weekends/holidays.
    """
    d = before or (date.today() - timedelta(days=1))
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def get_trading_date(now: datetime = None) -> str:
    """
    Return the trading date (YYYY-MM-DD) that this run's data
    actually represents — NOT necessarily today's calendar date.

    Logic:
        - If today is a trading day AND current time ≥ 3:30 PM IST
          → today's close is available → return today
        - Otherwise (before close, or today is weekend/holiday)
          → return the most recent completed trading day
    """
    now = now or datetime.now(IST)
    today = now.date()

    market_closed_today = (
        now.hour > MARKET_CLOSE_HOUR or
        (now.hour == MARKET_CLOSE_HOUR and now.minute >= MARKET_CLOSE_MINUTE)
    )

    if is_trading_day(today) and market_closed_today:
        return today.isoformat()

    # Otherwise, the relevant data is the last completed trading day
    return last_trading_day(before=today - timedelta(days=1)
                            if today.weekday() < 7 else today).isoformat() \
           if is_trading_day(today) else last_trading_day(before=today).isoformat()


def get_trading_date_simple(now: datetime = None) -> str:
    """
    Simpler version with the same guarantee, easier to reason about:
    walks backward from "yesterday" unless market has already
    closed today.
    """
    now   = now or datetime.now(IST)
    today = now.date()

    market_closed_today = (
        now.hour > MARKET_CLOSE_HOUR or
        (now.hour == MARKET_CLOSE_HOUR and now.minute >= MARKET_CLOSE_MINUTE)
    )

    if is_trading_day(today) and market_closed_today:
        return today.isoformat()

    # Before close, or today isn't a trading day at all —
    # the data available is from the last trading day strictly
    # before today.
    return last_trading_day(before=today - timedelta(days=1)).isoformat()


# Use the simpler, more predictable version as the public API
get_trading_date = get_trading_date_simple


if __name__ == "__main__":
    now = datetime.now(IST)
    td  = get_trading_date(now)
    print(f"Current time:  {now.strftime('%Y-%m-%d %H:%M:%S IST (%A)')}")
    print(f"Trading date:  {td}")
    print(f"Is today a trading day: {is_trading_day(now.date())}")
    print(f"Last trading day before today: {last_trading_day(now.date() - timedelta(days=1))}")
