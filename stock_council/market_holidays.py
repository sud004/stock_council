#!/usr/bin/env python3
# ============================================================
# market_holidays.py — NSE trading holiday checker
# ============================================================
# Fetches the official NSE trading-holiday list from NSE's own
# (unofficial, no-key) JSON endpoint and caches it locally so
# night_runner.py can skip holidays automatically — not just
# weekends.
#
# Why caching matters:
#   The nseindia.com endpoint sometimes blocks server/script
#   traffic, rate-limits, or is briefly down. If the live fetch
#   fails, we fall back to the last good cache instead of
#   guessing — holidays are announced months ahead, so a cache
#   that's a few days/weeks old is still accurate.
#
# USAGE:
#   from market_holidays import is_market_holiday, get_holidays
#   if is_market_holiday(date.today()):
#       ...
# ============================================================

import json
from datetime import date, datetime
from pathlib import Path

import requests

CACHE_FILE = Path(__file__).parent / "data" / "nse_holidays_cache.json"
CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

# How long a cache is considered "fresh" before we try to refetch.
# Holidays don't change, so this is generous on purpose.
CACHE_MAX_AGE_DAYS = 7

NSE_HOLIDAY_URL = "https://www.nseindia.com/api/holiday-master?type=trading"

# NSE's endpoint wants browser-like headers or it returns 401/403.
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36"),
    "Accept": "application/json",
    "Referer": "https://www.nseindia.com/resources/exchange-communication-holidays",
}


def _fetch_live() -> set:
    """
    Hit NSE's holiday endpoint and return a set of 'YYYY-MM-DD' strings.
    Raises on any failure — caller decides how to handle it.
    """
    session = requests.Session()
    session.headers.update(_HEADERS)

    # NSE requires an initial visit to set cookies before the API call works.
    session.get("https://www.nseindia.com", timeout=10)
    resp = session.get(NSE_HOLIDAY_URL, timeout=10)
    resp.raise_for_status()
    payload = resp.json()

    # Response shape: {"CM": [{"tradingDate": "26-Jan-2026", "description": "..."}, ...], ...}
    # "CM" = Capital Market (equities) segment — what we care about.
    rows = payload.get("CM", [])
    dates = set()
    for row in rows:
        raw = row.get("tradingDate")
        if not raw:
            continue
        try:
            d = datetime.strptime(raw, "%d-%b-%Y").date()
            dates.add(d.isoformat())
        except ValueError:
            continue

    if not dates:
        raise ValueError("NSE holiday response parsed but contained no usable dates")

    return dates


def _load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(dates: set):
    data = {
        "fetched_at": datetime.now().isoformat(),
        "holidays": sorted(dates),
    }
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def get_holidays(force_refresh: bool = False) -> set:
    """
    Return the set of NSE trading-holiday dates ('YYYY-MM-DD' strings),
    covering whatever year(s) NSE currently publishes.

    Tries a live fetch first (unless the cache is fresh and
    force_refresh is False). Falls back to cache on any failure.
    If there's no usable cache either, returns an empty set so the
    caller can still fall back to weekend-only skipping.
    """
    cache = _load_cache()
    cached_dates = set(cache.get("holidays", []))

    if not force_refresh and cached_dates:
        fetched_at = cache.get("fetched_at")
        if fetched_at:
            try:
                age_days = (datetime.now() - datetime.fromisoformat(fetched_at)).days
                if age_days < CACHE_MAX_AGE_DAYS:
                    return cached_dates
            except ValueError:
                pass

    try:
        live_dates = _fetch_live()
        _save_cache(live_dates)
        return live_dates
    except Exception as e:
        if cached_dates:
            print(f"  [holidays] Live fetch failed ({e}) — using cached list "
                  f"from {cache.get('fetched_at', 'unknown date')}")
            return cached_dates
        print(f"  [holidays] Live fetch failed ({e}) and no cache available — "
              f"only weekends will be skipped")
        return set()


def is_market_holiday(d: date = None) -> bool:
    """True if the given date (default: today) is an NSE trading holiday."""
    d = d or date.today()
    holidays = get_holidays()
    return d.isoformat() in holidays


if __name__ == "__main__":
    # Quick manual check: python market_holidays.py
    hols = get_holidays(force_refresh=True)
    today = date.today()
    print(f"Loaded {len(hols)} holiday dates.")
    print(f"Today ({today}) is {'a HOLIDAY' if today.isoformat() in hols else 'a normal trading day'}.")
    upcoming = sorted(h for h in hols if h >= today.isoformat())[:5]
    print(f"Next upcoming holidays: {upcoming}")
