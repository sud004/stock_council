# ============================================================
# utils/symbol_mapper.py
# Maps NSE symbols → Yahoo Finance symbols reliably
# ============================================================
#
# THE PROBLEM:
#   NSE calls it:    BAJAJ-AUTO, M&M, BERGERPAINTS
#   Yahoo Finance:   BAJAJ-AUTO.NS, M&M.NS, BERGEPAINT.NS
#   Sometimes:       completely different (renamed, delisted)
#
# THE SOLUTION:
#   1. Known corrections map (handles renames, spelling diffs)
#   2. Auto-validate against Yahoo Finance API
#   3. Cache validated symbols to disk
#   4. Never retry a known-bad symbol
# ============================================================

import json
import time
import requests
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, VERBOSE_DEBUG

SYMBOL_MAP_FILE = DATA_DIR / "symbol_map.json"

# ── Known NSE → Yahoo Finance corrections ────────────────────
# Format: "NSE_SYMBOL": "YAHOO_SYMBOL" or None (if not available)
KNOWN_CORRECTIONS = {
    # Renamed companies
    "NIITTECH":      "COFORGE",        # NIIT Tech renamed to Coforge
    "HEXAWARE":      "HEXAWARE",       # Relisted in 2024
    "PIRAMALENT":    "PIRAMALENT",     # Check latest
    "ADANITRANS":    "ADANIENSOL",     # Adani Transmission → Adani Energy
    "TV18BRDCST":    "TV18BRDCST",
    "INDIAGRID":     "INDIAGRID",

    # Spelling differences (NSE vs Yahoo)
    "BERGERPAINTS":  "BERGEPAINT",     # NSE: BERGERPAINTS, Yahoo: BERGEPAINT
    "AJANTPHARMA":   "AJANTPHARM",     # NSE: AJANTPHARMA, Yahoo: AJANTPHARM
    "NATCOPHARMA":   "NATCOPHARM",     # NSE: NATCOPHARMA, Yahoo: NATCOPHARM
    "TVSMOTORS":     "TVSMOTOR",       # NSE: TVSMOTORS, Yahoo: TVSMOTOR
    "AMARARAJA":     "AMARAJABAT",     # Amara Raja Batteries correct Yahoo ticker
    "LEMONTRE":      "LEMONTREE",      # Spelling
    "CHAMBL":        "CHAMBLFERT",     # Full name
    "VINATI":        "VINATIORGA",     # Full name
    "AARTI":         "AARTIIND",       # Aarti Industries
    "ASTRA":         "ASTRAL",         # Astral Ltd (different company!)
    "KIRLOSKAR":     "KIRLOSKARELE",   # Kirloskar Electric
    "JSPL":          "JINDALSTEL",     # Jindal Steel & Power
    "DALMIACEME":    "DALMIABHARAT",   # Dalmia Bharat
    "KAVERI":        "KSCL",           # Kaveri Seed Company Ltd on Yahoo
    "BAYER":         "BAYERCROP",      # Bayer CropScience
    "INOX":          "INOXGREEN",      # Inox Green Energy
    "ASTER":         "ASTERDM",        # Aster DM Healthcare
    "IPCA":          "IPCALAB",        # IPCA Laboratories
    "TIPS":          "TIPSFILMS",      # Tips Films

    # Delisted / Not on NSE (return None)
    "GREENKO":       None,             # Private, not listed
    "ITNL":          None,             # IL&FS Transportation, suspended
    "ACMESOLAR":     None,             # Not yet listed
    "MAPDIGIT":      None,             # Very illiquid, often no data
    "BARBEQUE":      None,             # Barbeque Nation delisted
    "SUVENPHAR":     None,             # Suven Pharma changed structure
    "WATERBASE":     "WATERBASE",      # Small cap, may have data gaps
    "AVANTI":        "AVANTIFEED",     # Avanti Feeds
    "ICICISEC":      "ICICIPRULI",     # Different entity
    "NIPPONLIFE":    "NAM-INDIA",      # NAM India
    "LTFH":          "LTFH",          # L&T Finance (check)
    "LTIM":          "LTIM",           # LTIMindtree — Yahoo may be rate limiting, retry tomorrow
    "VARUNBEV":      "VARUNBEV",       # Varun Beverages — may need retry
    "TATAMOTORS":    "TATAMOTORS",     # Tata Motors — Yahoo rate limiting, retry
    "ZOMATO":        "ZOMATO",         # Zomato — Yahoo rate limiting, retry
    "VEDANT":        "VEDANT",         # Vedant Fashions
    "GMRINFRA":      "GMRINFRA",       # GMR Infra
    "PVRINOX":       "PVRINOX",        # PVR Inox merged
}

# ── Validated symbol cache ────────────────────────────────────
_cache: dict = {}
_loaded = False


def _load_cache():
    global _cache, _loaded
    if _loaded:
        return
    if SYMBOL_MAP_FILE.exists():
        try:
            with open(SYMBOL_MAP_FILE) as f:
                _cache = json.load(f)
        except Exception:
            _cache = {}
    _loaded = True


def _save_cache():
    SYMBOL_MAP_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SYMBOL_MAP_FILE, 'w') as f:
        json.dump(_cache, f, indent=2)


def nse_to_yahoo(nse_symbol: str, exchange: str = "NS") -> str | None:
    """
    Convert NSE symbol to Yahoo Finance ticker.

    Returns:
        "RELIANCE.NS"   → valid Yahoo ticker
        None            → symbol not available on Yahoo Finance

    Examples:
        nse_to_yahoo("RELIANCE")      → "RELIANCE.NS"
        nse_to_yahoo("BERGERPAINTS")  → "BERGEPAINT.NS"
        nse_to_yahoo("GREENKO")       → None
        nse_to_yahoo("TCS")           → "TCS.NS"
    """
    _load_cache()
    sym = nse_symbol.upper().strip()

    # Check cache first
    if sym in _cache:
        return _cache[sym]

    # Check known corrections
    if sym in KNOWN_CORRECTIONS:
        corrected = KNOWN_CORRECTIONS[sym]
        if corrected is None:
            _cache[sym] = None
            _save_cache()
            return None
        yahoo_sym = f"{corrected}.{exchange}"
        _cache[sym] = yahoo_sym
        _save_cache()
        return yahoo_sym

    # Try direct .NS append
    yahoo_sym = f"{sym}.{exchange}"
    if _validate_yahoo_symbol(yahoo_sym):
        _cache[sym] = yahoo_sym
        _save_cache()
        return yahoo_sym

    # Try .BO (BSE)
    bse_sym = f"{sym}.BO"
    if _validate_yahoo_symbol(bse_sym):
        _cache[sym] = bse_sym
        _save_cache()
        return bse_sym

    # Not found
    _cache[sym] = None
    _save_cache()
    return None


def _validate_yahoo_symbol(yahoo_ticker: str) -> bool:
    """
    Quick check if a Yahoo Finance symbol returns valid data.
    Uses fast_info to avoid downloading full history.
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(yahoo_ticker)
        info = ticker.fast_info
        price = getattr(info, 'last_price', None)
        if price and float(price) > 0:
            return True
        # Also check market cap
        mc = getattr(info, 'market_cap', None)
        if mc and float(mc) > 0:
            return True
        return False
    except Exception:
        return False


def get_all_yahoo_symbols(nse_symbols: list) -> dict:
    """
    Convert a list of NSE symbols to Yahoo Finance tickers.
    Returns {nse_sym: yahoo_ticker_or_None}

    Prints a clean summary at the end.
    """
    _load_cache()
    results = {}
    to_validate = []

    # First pass: use cache and known corrections
    for sym in nse_symbols:
        s = sym.upper().strip()
        if s in _cache:
            results[s] = _cache[s]
        elif s in KNOWN_CORRECTIONS:
            corrected = KNOWN_CORRECTIONS[s]
            results[s] = f"{corrected}.NS" if corrected else None
            _cache[s] = results[s]
        else:
            to_validate.append(s)

    # Second pass: validate unknowns against Yahoo
    if to_validate:
        print(f"[MAPPER] Validating {len(to_validate)} new symbols against Yahoo Finance...")
        for sym in to_validate:
            yahoo = nse_to_yahoo(sym)
            results[sym] = yahoo
            time.sleep(0.3)

    _save_cache()

    # Print summary
    valid   = [s for s, v in results.items() if v]
    invalid = [s for s, v in results.items() if not v]

    print(f"\n[MAPPER] Symbol mapping complete:")
    print(f"  ✅ Valid Yahoo symbols:  {len(valid)}")
    print(f"  ❌ Not available:        {len(invalid)}")
    if invalid:
        print(f"     {', '.join(invalid[:10])}{'...' if len(invalid)>10 else ''}")
    print(f"  Reason for failures:")
    print(f"    • Delisted from NSE (ITNL, GREENKO etc.)")
    print(f"    • Company renamed (symbol changed)")
    print(f"    • Private / unlisted company")
    print(f"    • Yahoo Finance doesn't cover this stock")

    return results


def print_symbol_report(nse_symbols: list):
    """
    Print a detailed report of symbol mapping.
    Shows which symbols work, which don't, and why.
    """
    results = get_all_yahoo_symbols(nse_symbols)

    print(f"\n{'═'*60}")
    print(f"  SYMBOL MAPPING REPORT")
    print(f"{'═'*60}")

    valid = {s: v for s, v in results.items() if v}
    invalid = {s: v for s, v in results.items() if not v}

    print(f"\n  ✅ VALID ({len(valid)} symbols):")
    for i, (nse, yahoo) in enumerate(list(valid.items())[:10]):
        diff = " ← renamed" if nse not in yahoo else ""
        print(f"     {nse:20} → {yahoo}{diff}")
    if len(valid) > 10:
        print(f"     ... and {len(valid)-10} more")

    print(f"\n  ❌ NOT AVAILABLE ({len(invalid)} symbols):")
    for nse in list(invalid.keys()):
        reason = "Delisted" if nse in KNOWN_CORRECTIONS and KNOWN_CORRECTIONS[nse] is None else "Not found"
        print(f"     {nse:20} → {reason}")

    print(f"\n{'═'*60}")


if __name__ == "__main__":
    # Test the mapper
    test_symbols = [
        "RELIANCE", "TCS", "BERGERPAINTS", "NIITTECH",
        "GREENKO", "AJANTPHARMA", "TATAMOTORS", "ZOMATO",
        "LTIM", "HEXAWARE", "M&M", "BAJAJ-AUTO"
    ]
    print_symbol_report(test_symbols)
