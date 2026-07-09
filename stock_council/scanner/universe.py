# ============================================================
# scanner/universe.py
# DYNAMIC NSE stock universe — fetches live from NSE every night
# Falls back to curated list if NSE is unavailable
# ============================================================

import requests
import pandas as pd
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
import pytz

IST = pytz.timezone('Asia/Kolkata')
BASE_DIR = Path(__file__).parent.parent
CACHE_FILE = BASE_DIR / "data" / "nse_universe.json"

# ── Sector classifications (used for analysis context) ────────
SECTORS = {
    "Information Technology": [
        "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM",
        "LTIM", "MPHASIS", "PERSISTENT", "COFORGE",
        "OFSS", "LTTS", "KPITTECH", "TATAELXSI",
        "MASTEK", "CYIENT", "RATEGAIN", "HAPPSTMNDS",
    ],
    "Financial Services": [
        "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK",
        "AXISBANK", "INDUSINDBK", "BANDHANBNK", "FEDERALBNK",
        "IDFCFIRSTB", "YESBANK", "PNB", "CANBK",
        "UNIONBANK", "BANKBARODA", "INDIANB",
        "BAJFINANCE", "BAJAJFINSV", "CHOLAFIN", "MUTHOOTFIN",
        "SHRIRAMFIN", "MANAPPURAM", "HDFCLIFE", "SBILIFE",
        "ICICIGI", "LICI", "BSE", "CDSL", "ANGELONE",
        "HDFCAMC", "NAM-INDIA",
    ],
    "Energy": [
        "RELIANCE", "ONGC", "IOC", "BPCL", "HINDPETRO",
        "GAIL", "MGL", "IGL", "PETRONET", "OIL",
        "NTPC", "POWERGRID", "TATAPOWER", "ADANIGREEN",
        "ADANIPOWER", "TORNTPOWER", "CESC", "NHPC",
        "SJVN", "RECLTD", "IRFC", "PFC",
    ],
    "Consumer Staples": [
        "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA",
        "DABUR", "MARICO", "GODREJCP", "COLPAL",
        "EMAMILTD", "JYOTHYLAB", "VBL", "RADICO",
        "TATACONSUM", "VARUNBEV", "BIKAJI",
    ],
    "Consumer Discretionary": [
        "MARUTI", "TATAMOTORS", "M&M", "HEROMOTOCO",
        "BAJAJ-AUTO", "EICHERMOT", "TVSMOTORS",
        "ASHOKLEY", "MOTHERSON", "BOSCHLTD", "BHARATFORG",
        "EXIDEIND", "AMARARAJA", "DMART", "TRENT",
        "JUBLFOOD", "DEVYANI", "WESTLIFE",
        "PAGEIND", "TITAN", "KALYANKJIL", "RAYMOND",
        "INDHOTEL", "IRCTC",
    ],
    "Healthcare": [
        "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB",
        "LUPIN", "BIOCON", "TORNTPHARM", "ALKEM",
        "GRANULES", "ABBOTINDIA", "PFIZER", "GLAXO",
        "APOLLOHOSP", "FORTIS", "MAXHEALTH",
        "METROPOLIS", "LALPATHLAB", "THYROCARE", "LAURUSLABS",
    ],
    "Industrials": [
        "LT", "SIEMENS", "ABB", "HAVELLS",
        "VOLTAS", "CUMMINSIND", "BHEL", "BEL",
        "HAL", "BEML", "ADANIPORTS", "CONCOR",
        "DELHIVERY", "BLUEDART", "ALLCARGO",
        "THERMAX", "ELGIEQUIP", "ESCORTS",
    ],
    "Basic Materials": [
        "JSWSTEEL", "TATASTEEL", "SAIL", "NMDC",
        "JSPL", "RATNAMANI", "HINDALCO", "VEDL",
        "NATIONALUM", "HINDCOPPER", "MOIL",
        "ULTRACEMCO", "AMBUJACEM", "ACC", "SHREECEM",
        "RAMCOCEM", "JKCEMENT", "PIDILITIND",
        "ASIANPAINT", "BERGEPAINT", "KANSAINER",
        "DEEPAKNTR", "NAVINFLUOR", "VINATI", "AARTI",
        "COROMANDEL", "GNFC",
    ],
    "Real Estate": [
        "DLF", "GODREJPROP", "PRESTIGE", "OBEROIRLTY",
        "PHOENIXLTD", "SOBHA", "BRIGADE", "LODHA",
    ],
    "Telecom": [
        "BHARTIARTL", "IDEA", "TATACOMM", "RAILTEL", "HFCL",
    ],
    "Media & Entertainment": [
        "ZEEL", "SUNTV", "PVRINOX", "NETWORK18", "SAREGAMA", "NAZARA",
    ],
    "New Age Tech": [
        "PAYTM", "POLICYBZR", "NYKAA", "ZOMATO",
        "CARTRADE", "EASEMYTRIP", "RATEGAIN",
        "NEWGEN", "INTELLECT",
    ],
    "Agriculture": [
        "UPL", "PIIND", "RALLIS", "KAVERI",
        "DHANUKA", "KRBL", "AVANTI",
    ],
}

# ── Priority stocks (always included, always analysed first) ──
PRIORITY_STOCKS = {
    s: stocks[:5] for s, stocks in SECTORS.items()
}

# ── Sector risk profiles ──────────────────────────────────────
SECTOR_RISK_PROFILE = {
    "Information Technology":  {"risk": "LOW",    "cyclical": False, "export_driven": True},
    "Financial Services":      {"risk": "MEDIUM", "cyclical": True,  "rbi_sensitive": True},
    "Energy":                  {"risk": "MEDIUM", "cyclical": True,  "crude_sensitive": True},
    "Consumer Staples":        {"risk": "LOW",    "cyclical": False, "defensive": True},
    "Consumer Discretionary":  {"risk": "MEDIUM", "cyclical": True},
    "Healthcare":              {"risk": "LOW",    "cyclical": False, "export_driven": True},
    "Industrials":             {"risk": "MEDIUM", "cyclical": True,  "capex_driven": True},
    "Basic Materials":         {"risk": "HIGH",   "cyclical": True,  "global_commodity": True},
    "Real Estate":             {"risk": "HIGH",   "cyclical": True,  "rbi_sensitive": True},
    "Telecom":                 {"risk": "HIGH",   "cyclical": False, "regulatory": True},
    "Media & Entertainment":   {"risk": "MEDIUM", "cyclical": True},
    "New Age Tech":            {"risk": "HIGH",   "cyclical": True,  "profitability_risk": True},
    "Agriculture":             {"risk": "MEDIUM", "cyclical": True,  "monsoon_sensitive": True},
}

# ── Index symbols ─────────────────────────────────────────────
INDICES = {
    "^NSEI":     "Nifty 50",
    "^NSEBANK":  "Nifty Bank",
    "^CNXIT":    "Nifty IT",
    "^CNXFMCG":  "Nifty FMCG",
    "^CNXPHARMA":"Nifty Pharma",
    "^CNXAUTO":  "Nifty Auto",
    "^CNXMETAL": "Nifty Metal",
    "^CNXREALTY":"Nifty Realty",
    "^CNXENERGY":"Nifty Energy",
    "^CNXMID50": "Nifty Midcap 50",
    "^BSESN":    "Sensex",
}


# ══════════════════════════════════════════════════════════════
# LIVE UNIVERSE FETCHER
# ══════════════════════════════════════════════════════════════

def fetch_nse_live_universe() -> list[str]:
    """
    Fetch ALL NSE-listed equity symbols dynamically from NSE India.
    Uses NSE's public equity list endpoint — no API key needed.
    Returns list of clean symbols (no .NS suffix).
    """
    sources = [
        # Source 1: NSE equity bhavcopy (most comprehensive)
        {
            "name": "NSE Equity List",
            "url": "https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O",
            "parser": _parse_nse_equity_api,
        },
        # Source 2: Nifty 500 (top 500 by market cap)
        {
            "name": "Nifty 500",
            "url": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500",
            "parser": _parse_nse_equity_api,
        },
        # Source 3: All NSE equities via bhavcopy CSV
        {
            "name": "NSE Bhavcopy",
            "url": "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
            "parser": _parse_nse_bhavcopy,
        },
    ]

    for source in sources:
        try:
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json, text/plain, */*',
                'Referer': 'https://www.nseindia.com/',
            })
            # NSE requires homepage hit first for cookies
            session.get("https://www.nseindia.com", timeout=10)
            time.sleep(1)

            r = session.get(source["url"], timeout=15)
            if r.status_code == 200:
                symbols = source["parser"](r)
                if symbols and len(symbols) > 50:
                    print(f"[UNIVERSE] ✅ Got {len(symbols)} symbols from {source['name']}")
                    return symbols
        except Exception as e:
            print(f"[UNIVERSE] {source['name']} failed: {e}")
            continue

    return []


def _parse_nse_equity_api(response) -> list[str]:
    """Parse NSE equity indices API response."""
    try:
        data = response.json()
        stocks = data.get('data', [])
        symbols = []
        for s in stocks:
            sym = s.get('symbol', '').strip().upper()
            if sym and len(sym) >= 2 and sym not in ('SYMBOL', ''):
                symbols.append(sym)
        return symbols
    except Exception:
        return []


def _parse_nse_bhavcopy(response) -> list[str]:
    """Parse NSE EQUITY_L.csv bhavcopy."""
    try:
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        # Column is 'SYMBOL' in the CSV
        col = None
        for c in df.columns:
            if 'SYMBOL' in c.upper():
                col = c
                break
        if col:
            symbols = df[col].str.strip().str.upper().dropna().tolist()
            # Filter: only equity symbols (no indices, ETFs etc.)
            symbols = [s for s in symbols if s.isalpha() or '-' in s or '&' in s]
            return symbols[:2000]  # cap at 2000
    except Exception:
        pass
    return []


# ══════════════════════════════════════════════════════════════
# SMART SYMBOL VALIDATOR
# ══════════════════════════════════════════════════════════════

def validate_and_correct_symbol(symbol: str) -> tuple[str | None, str]:
    """
    Try to find the correct Yahoo Finance symbol for an NSE stock.
    Returns (corrected_symbol, reason) or (None, reason) if not found.

    Tries: SYMBOL.NS → SYMBOL.BO → common name variations
    """
    import yfinance as yf

    # Common symbol corrections
    CORRECTIONS = {
        "BERGEPAINT":    "BERGEPAINT.NS",
        "BERGERPAINTS":  "BERGEPAINT.NS",
        "LEMONTRE":      "LEMONTREE.NS",
        "LEMONTREE":     "LEMONTREE.NS",
        "TVSMOTORS":     "TVSMOTOR.NS",
        "TVSMOTORS":     "TVSMOTOR.NS",
        "AMARARAJA":     "AMARARAJA.NS",     # renamed from AMARAJABAT (Amara Raja Energy & Mobility)
        "TATAMOTORS":    "TATAMOTORS.NS",
        "INOX":          "INOXWIND.NS",
        "INDIAGRID":     "INDIAGRID.NS",
        "VARUNBEV":      "VARUNBEV.NS",
        "ICICISEC":      "ICICIPRULI.NS",
        "NIPPONLIFE":    "NAM-INDIA.NS",
        "LTFH":          "LTFH.NS",
        "PIRAMALENT":    "PIRAMALENT.NS",
        "LTIM":          "LTIM.NS",          # LTIMindtree — correct NSE symbol
        "ZOMATO":        "ETERNAL.NS",       # Zomato rebranded to Eternal Ltd (Jan 2025)
        "GREENKO":       None,   # Not listed on NSE
        "ITNL":          None,   # Delisted
        "HEXAWARE":      "HEXAWARE.NS",
        "NIITTECH":      "COFORGE.NS",  # Renamed to Coforge
        "BARBEQUE":      "BARBEQUE.NS",
        "VEDANT":        "VEDANT.NS",
        "SUVENPHAR":     None,
        "ASTRA":         "ASTRAL.NS",
        "GMRINFRA":      "GMRINFRA.NS",
        "KIRLOSKAR":     "KIRLOSKARELE.NS",
        "JSPL":          "JINDALSTEL.NS",
        "DALMIACEME":    "DALMIA-OCL.NS",
        "CHAMBL":        "CHAMBLFERT.NS",
        "VINATI":        "VINATIORGA.NS",
        "AARTI":         "AARTIIND.NS",
        "MAPDIGIT":      None,
        "BAYER":         "BAYERCROP.NS",
        "KAVERI":        "KAVVERISED.NS",
        "AVANTI":        "AVANTIFEED.NS",
        "WATERBASE":     "WATERBASE.NS",
        "TIPS":          "TIPSFILMS.NS",
        "TV18BRDCST":    "TV18BRDCST.NS",
        "ADANITRANS":    "ADANIENSOL.NS",
        "INOX":          "INOXGREEN.NS",
        "AJANTPHARMA":   "AJANTPHARM.NS",
        "NATCOPHARMA":   "NATCOPHARM.NS",
        "ASTER":         "ASTERDM.NS",
        "IPCA":          "IPCALAB.NS",
    }

    sym = symbol.upper().replace('.NS', '').replace('.BO', '')

    # Check corrections map first
    if sym in CORRECTIONS:
        corrected = CORRECTIONS[sym]
        if corrected is None:
            return None, "Delisted or not on NSE"
        return corrected.replace('.NS', ''), "Corrected symbol"

    # Try .NS suffix
    try:
        ticker = yf.Ticker(f"{sym}.NS")
        info = ticker.fast_info
        price = getattr(info, 'last_price', None)
        if price and float(price) > 0:
            return sym, "Valid .NS symbol"
    except Exception:
        pass

    # Try .BO suffix
    try:
        ticker = yf.Ticker(f"{sym}.BO")
        info = ticker.fast_info
        price = getattr(info, 'last_price', None)
        if price and float(price) > 0:
            return sym + "_BSE", "Valid .BO symbol (BSE)"
    except Exception:
        pass

    return None, "Symbol not found on NSE or BSE"


# ══════════════════════════════════════════════════════════════
# CACHE MANAGEMENT
# ══════════════════════════════════════════════════════════════

def save_universe_cache(symbols: list, sector_map: dict):
    """Save universe to local cache file."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, 'w') as f:
        json.dump({
            'symbols': symbols,
            'sector_map': sector_map,
            'updated_at': datetime.now(IST).isoformat(),
            'count': len(symbols),
        }, f, indent=2)


def load_universe_cache() -> tuple[list, dict] | tuple[None, None]:
    """Load cached universe if fresh (< 24 hours)."""
    if not CACHE_FILE.exists():
        return None, None
    try:
        with open(CACHE_FILE) as f:
            data = json.load(f)
        updated = datetime.fromisoformat(data['updated_at'].replace('Z', '+00:00'))
        if datetime.now(IST) - updated.astimezone(IST) < timedelta(hours=24):
            return data['symbols'], data.get('sector_map', {})
    except Exception:
        pass
    return None, None


# ══════════════════════════════════════════════════════════════
# ERROR SUMMARY REPORTER
# ══════════════════════════════════════════════════════════════

def generate_error_summary(results: dict) -> str:
    """
    Generate a clean 5-6 line error summary from download results.
    Called at end of nightly job.

    results = {symbol: 'ok' | 'skipped' | 'error' | 'no_data' | 'delisted'}
    """
    ok       = [s for s, v in results.items() if v == 'ok']
    skipped  = [s for s, v in results.items() if v == 'skipped']
    no_data  = [s for s, v in results.items() if v == 'no_data']
    errors   = [s for s, v in results.items() if v.startswith('error')]
    delisted = [s for s, v in results.items() if v == 'delisted']

    lines = [
        f"  ✅ Downloaded:  {len(ok)} stocks",
        f"  ⏭  Skipped:     {len(skipped)} stocks (already fresh)",
        f"  ❌ No data:     {len(no_data)} stocks — " +
            (f"e.g. {', '.join(no_data[:3])}{'...' if len(no_data)>3 else ''}" if no_data else "none"),
        f"  🚫 Errors:      {len(errors)} stocks — " +
            (f"e.g. {', '.join(errors[:3])}{'...' if len(errors)>3 else ''}" if errors else "none"),
        f"  📋 Delisted:    {len(delisted)} stocks — " +
            (f"{', '.join(delisted[:5])}" if delisted else "none"),
        f"  📊 Total:       {len(results)} stocks attempted",
    ]

    # Possible reasons
    if no_data or errors:
        lines.append(f"\n  Possible reasons for failures:")
        if no_data:
            lines.append(f"    • Symbol renamed on NSE (e.g. NIITTECH → COFORGE)")
            lines.append(f"    • Stock delisted or suspended")
            lines.append(f"    • Yahoo Finance uses different ticker format")
        if errors:
            lines.append(f"    • Temporary Yahoo Finance rate limit (retry tomorrow)")
            lines.append(f"    • Network timeout")

    return '\n'.join(lines)


# ══════════════════════════════════════════════════════════════
# MAIN UNIVERSE GETTER (called by pipeline)
# ══════════════════════════════════════════════════════════════

def get_universe(force_refresh: bool = False) -> tuple[list, dict]:
    """
    Get the full stock universe.

    Returns:
        (all_symbols: list, stock_to_sector: dict)

    Priority:
        1. Live NSE fetch (if force_refresh or cache stale)
        2. Local cache (if fresh)
        3. Curated SECTORS list (fallback)
    """
    if not force_refresh:
        cached_syms, cached_map = load_universe_cache()
        if cached_syms:
            return cached_syms, cached_map

    # Try live NSE fetch
    live_symbols = fetch_nse_live_universe()

    if live_symbols and len(live_symbols) > 100:
        # Merge with our sector stocks (keep all sector stocks + live additions)
        curated = set()
        for stocks in SECTORS.values():
            curated.update(stocks)

        # Add live symbols not in our curated list
        all_symbols = list(curated)
        new_additions = []
        for sym in live_symbols:
            if sym not in curated:
                new_additions.append(sym)

        if new_additions:
            print(f"[UNIVERSE] {len(new_additions)} new symbols from NSE not in curated list")
            # Only add Nifty 500 stocks to keep list manageable
            # (live_symbols from Nifty 500 source are already filtered)
            all_symbols.extend(new_additions[:200])

        # Build sector map
        stock_to_sector = {}
        for sector, stocks in SECTORS.items():
            for s in stocks:
                stock_to_sector[s] = sector
        for sym in new_additions[:200]:
            stock_to_sector[sym] = "Other"

        all_symbols = sorted(list(set(all_symbols)))
        save_universe_cache(all_symbols, stock_to_sector)
        print(f"[UNIVERSE] Total universe: {len(all_symbols)} stocks")
        return all_symbols, stock_to_sector

    # Fallback to curated list
    print("[UNIVERSE] Using curated sector list (NSE fetch unavailable)")
    all_symbols = []
    stock_to_sector = {}
    for sector, stocks in SECTORS.items():
        for s in stocks:
            if s not in all_symbols:
                all_symbols.append(s)
            stock_to_sector[s] = sector

    return all_symbols, stock_to_sector


# ── Module-level constants (for backward compatibility) ───────
_all_stocks, _stock_to_sector = get_universe()

ALL_STOCKS      = _all_stocks
STOCK_TO_SECTOR = _stock_to_sector
TOTAL_STOCKS    = len(ALL_STOCKS)
TOTAL_SECTORS   = len(SECTORS)
