# ============================================================
# scanner/market_scanner.py
# Level 1: Scan the ENTIRE NSE/BSE market health
# ============================================================
# WHAT IT DOES:
#   - Fetches all major indices live (Nifty50, Bank, IT, FMCG…)
#   - Reads FII/DII net flows for the day
#   - Reads India VIX (fear index)
#   - Computes Advance/Decline ratio across the market
#   - Reads global cues: Dow, Nasdaq, SGX Nifty, crude oil
#   - LLM analyses all this → gives Market Outlook score 0-10
#   - OUTPUT: MarketSnapshot dataclass with everything needed
#     by the sector scanner downstream
# ============================================================

import sys
import time
import requests
import feedparser
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import pytz

sys.path.insert(0, str(Path(__file__).parent.parent))

from scanner.universe import INDICES, SECTORS
from utils.live_data import NSELive, FinnhubData
from utils.llm import stream_chat, extract_score
from config import ALLOW_INTERNET, VERBOSE_DEBUG

IST = pytz.timezone('Asia/Kolkata')

MARKET_BOT_PROMPT = """You are MARKET BOT — India's top macro analyst covering NSE/BSE markets.

You receive live data on:
- All major Nifty indices performance
- FII (Foreign Institutional Investor) net flows — critical signal
- DII (Domestic Institutional) net flows
- India VIX (fear index)
- Advance/Decline ratio (breadth)
- Global markets: Dow Jones, Nasdaq, SGX Nifty, Crude Oil

Your job: Give a precise market-level outlook that guides sector selection.

Your response MUST be 120-150 words and cover:
1. MARKET DIRECTION today (bull/bear/sideways)
2. FII intent — are they buying or running away?
3. Market breadth — are many stocks participating or just a few?
4. Key risk for today's session
5. Which BROAD theme is in favour today (IT/Banks/Infra/Defensive etc.)

End with exactly: "MARKET SCORE: X/10"
(10 = very bullish broad market, 1 = extreme bear market panic)
"""


@dataclass
class IndexData:
    symbol: str
    name: str
    last: float = None
    change: float = None
    pct_change: float = None
    year_high: float = None
    year_low: float = None
    pct_from_52w_high: float = None


@dataclass
class MarketSnapshot:
    """Everything about the current market state — passed to sector scanner."""
    timestamp: str = ""
    market_open: bool = False
    indices: list = field(default_factory=list)          # list of IndexData
    fii_net_cr: float = None                             # FII net ₹ crore
    dii_net_cr: float = None
    fii_dii_signal: str = "NEUTRAL"                      # BULLISH/BEARISH/NEUTRAL
    vix: float = None
    vix_signal: str = "NEUTRAL"
    advance_count: int = 0
    decline_count: int = 0
    unchanged_count: int = 0
    ad_ratio: float = None                               # advance/decline ratio
    breadth_signal: str = "NEUTRAL"
    global_cues: dict = field(default_factory=dict)      # Dow, Crude etc.
    market_score: float = 5.0                            # LLM 0-10 score
    market_outlook: str = "NEUTRAL"                      # BULLISH/BEARISH/NEUTRAL
    llm_analysis: str = ""
    top_gainers: list = field(default_factory=list)
    top_losers: list = field(default_factory=list)
    sector_performance: dict = field(default_factory=dict)  # sector → pct_change


# ── Fetch functions ────────────────────────────────────────────

def fetch_all_indices() -> list[IndexData]:
    """Fetch all NSE index values."""
    result = []
    raw = NSELive.get_all_indices() or []
    for item in raw:
        name = item.get('name', '')
        last = item.get('last')
        pct = item.get('pct_change', 0) or 0
        yh = item.get('year_high')
        yl = item.get('year_low')
        pct_from_ath = round((last - yh) / yh * 100, 2) if last and yh else None
        result.append(IndexData(
            symbol=name,
            name=name,
            last=last,
            change=item.get('change'),
            pct_change=pct,
            year_high=yh,
            year_low=yl,
            pct_from_52w_high=pct_from_ath
        ))
    return result


def fetch_global_cues() -> dict:
    """
    Fetch global market cues via Yahoo Finance or free APIs.
    SGX Nifty is best pre-market indicator for India.
    """
    global_symbols = {
        "^DJI":   "Dow Jones",
        "^IXIC":  "Nasdaq",
        "^GSPC":  "S&P 500",
        "^N225":  "Nikkei 225",
        "^HSI":   "Hang Seng",
        "CL=F":   "Crude Oil (WTI)",
        "GC=F":   "Gold",
        "EURINR=X":"EUR/INR",
        "USDINR=X":"USD/INR",
    }
    result = {}
    if not ALLOW_INTERNET:
        return result
    try:
        import yfinance as yf
        for sym, name in global_symbols.items():
            try:
                ticker = yf.Ticker(sym)
                info = ticker.fast_info
                price = getattr(info, 'last_price', None)
                prev = getattr(info, 'previous_close', None)
                if price and prev and prev != 0:
                    chg = round((price - prev) / prev * 100, 2)
                    result[name] = {"price": round(price, 2), "change_pct": chg}
            except Exception:
                pass
            time.sleep(0.2)  # gentle on Yahoo
    except Exception as e:
        if VERBOSE_DEBUG:
            print(f"[MARKET] Global cues error: {e}")
    return result


def fetch_advance_decline(snapshot_indices: list) -> dict:
    """
    Compute advance/decline from NSE's broader market.
    Uses NSE market status or approximates from index breadth.
    """
    if not ALLOW_INTERNET:
        return {"advance": 0, "decline": 0, "unchanged": 0}
    try:
        s = NSELive._get_session()
        r = s.get("https://www.nseindia.com/api/market-turnover", timeout=10)
        if r.status_code == 200:
            data = r.json()
            # NSE provides market-wide advance/decline
            total = data.get('allMarket', {})
            adv = int(total.get('advance', {}).get('noOfScrips', 0))
            dec = int(total.get('decline', {}).get('noOfScrips', 0))
            unc = int(total.get('unchanged', {}).get('noOfScrips', 0))
            return {"advance": adv, "decline": dec, "unchanged": unc}
    except Exception:
        pass
    # Fallback: estimate from Nifty indices direction
    gainers = sum(1 for idx in snapshot_indices if (idx.pct_change or 0) > 0)
    losers = sum(1 for idx in snapshot_indices if (idx.pct_change or 0) < 0)
    return {"advance": gainers * 100, "decline": losers * 100, "unchanged": 20}


def compute_market_signals(snapshot: MarketSnapshot) -> MarketSnapshot:
    """Derive signal labels from raw numbers."""

    # FII/DII signal
    fii = snapshot.fii_net_cr or 0
    dii = snapshot.dii_net_cr or 0
    combined = fii + dii
    if combined > 1000:
        snapshot.fii_dii_signal = "STRONG_BULLISH"
    elif combined > 0:
        snapshot.fii_dii_signal = "BULLISH"
    elif combined < -1000:
        snapshot.fii_dii_signal = "STRONG_BEARISH"
    elif combined < 0:
        snapshot.fii_dii_signal = "BEARISH"
    else:
        snapshot.fii_dii_signal = "NEUTRAL"

    # VIX signal
    vix = snapshot.vix or 15
    if vix < 12:
        snapshot.vix_signal = "COMPLACENCY"
    elif vix < 16:
        snapshot.vix_signal = "CALM_BULLISH"
    elif vix < 20:
        snapshot.vix_signal = "NORMAL"
    elif vix < 25:
        snapshot.vix_signal = "ELEVATED_FEAR"
    else:
        snapshot.vix_signal = "EXTREME_FEAR"

    # Breadth signal
    adv = snapshot.advance_count
    dec = snapshot.decline_count
    if adv + dec > 0:
        snapshot.ad_ratio = round(adv / (adv + dec), 3)
        if snapshot.ad_ratio > 0.65:
            snapshot.breadth_signal = "STRONG_BREADTH"
        elif snapshot.ad_ratio > 0.50:
            snapshot.breadth_signal = "POSITIVE_BREADTH"
        elif snapshot.ad_ratio > 0.35:
            snapshot.breadth_signal = "WEAK_BREADTH"
        else:
            snapshot.breadth_signal = "VERY_WEAK_BREADTH"

    # Market outlook
    score = 5.0
    if snapshot.fii_dii_signal in ("BULLISH", "STRONG_BULLISH"):
        score += 1.5 if "STRONG" in snapshot.fii_dii_signal else 1.0
    elif snapshot.fii_dii_signal in ("BEARISH", "STRONG_BEARISH"):
        score -= 1.5 if "STRONG" in snapshot.fii_dii_signal else 1.0

    if snapshot.breadth_signal in ("STRONG_BREADTH",):
        score += 1.0
    elif snapshot.breadth_signal in ("VERY_WEAK_BREADTH",):
        score -= 1.0

    if snapshot.vix_signal == "EXTREME_FEAR":
        score += 0.5  # contrarian
    elif snapshot.vix_signal == "ELEVATED_FEAR":
        score -= 0.5

    nifty = next((i for i in snapshot.indices if "NIFTY 50" in (i.name or "").upper()), None)
    if nifty and nifty.pct_change:
        if nifty.pct_change > 1.0:
            score += 1.0
        elif nifty.pct_change < -1.0:
            score -= 1.0

    snapshot.market_score = round(min(10, max(0, score)), 2)
    if snapshot.market_score >= 7:
        snapshot.market_outlook = "BULLISH"
    elif snapshot.market_score >= 5:
        snapshot.market_outlook = "NEUTRAL"
    else:
        snapshot.market_outlook = "BEARISH"

    return snapshot


def build_market_prompt(snapshot: MarketSnapshot) -> str:
    """Build the LLM prompt from all market data."""

    # Indices section
    idx_lines = []
    for idx in snapshot.indices:
        if idx.last and idx.pct_change is not None:
            arrow = "▲" if idx.pct_change >= 0 else "▼"
            ath_note = f" ({idx.pct_from_52w_high:+.1f}% vs 52w high)" if idx.pct_from_52w_high else ""
            idx_lines.append(
                f"  {idx.name:25} {idx.last:>10.2f}  {arrow}{abs(idx.pct_change):.2f}%{ath_note}"
            )
    indices_str = "\n".join(idx_lines) if idx_lines else "  (no index data)"

    # Global cues
    global_lines = []
    for name, data in snapshot.global_cues.items():
        arrow = "▲" if data.get('change_pct', 0) >= 0 else "▼"
        global_lines.append(
            f"  {name:20} {data.get('price', 'N/A'):>10}  {arrow}{abs(data.get('change_pct', 0)):.2f}%"
        )
    global_str = "\n".join(global_lines) if global_lines else "  (offline)"

    return f"""
=== NSE/BSE MARKET SCAN — {snapshot.timestamp} ===

--- INDIAN INDICES (Live) ---
{indices_str}

--- FII / DII FLOWS ---
FII Net Today:    ₹{snapshot.fii_net_cr or 'N/A'} Cr   Signal: {snapshot.fii_dii_signal}
DII Net Today:    ₹{snapshot.dii_net_cr or 'N/A'} Cr
Combined Net:     ₹{(snapshot.fii_net_cr or 0) + (snapshot.dii_net_cr or 0):.0f} Cr

--- INDIA VIX ---
VIX:              {snapshot.vix or 'N/A'}    Signal: {snapshot.vix_signal}

--- MARKET BREADTH ---
Advances:         {snapshot.advance_count}
Declines:         {snapshot.decline_count}
Unchanged:        {snapshot.unchanged_count}
A/D Ratio:        {snapshot.ad_ratio or 'N/A'}    Signal: {snapshot.breadth_signal}

--- GLOBAL CUES ---
{global_str}

--- PRE-COMPUTED MARKET SCORE: {snapshot.market_score}/10 ---
Outlook: {snapshot.market_outlook}

Give your market-level analysis and outlook. Identify the dominant theme and
which sectors are likely to outperform/underperform today.
"""


def run(print_output: bool = True) -> MarketSnapshot:
    """
    Run the full market-level scan.
    Returns a MarketSnapshot passed downstream to sector scanner.
    """
    if print_output:
        print(f"\n{'═'*65}")
        print("🌐  LEVEL 1: MARKET SCAN — NSE / BSE")
        print(f"{'═'*65}")

    snap = MarketSnapshot(
        timestamp=datetime.now(IST).strftime("%d %b %Y %H:%M IST"),
        market_open=NSELive.get_market_status().get('is_open', False)
                   if ALLOW_INTERNET else False
    )

    # Indices
    if print_output:
        print("  → Fetching all NSE indices...")
    snap.indices = fetch_all_indices()

    # FII/DII
    if print_output:
        print("  → Fetching FII/DII flows...")
    fii_dii = NSELive.get_fii_dii() or {}
    fii = fii_dii.get('fii', {})
    dii = fii_dii.get('dii', {})
    snap.fii_net_cr = _safe_float(fii.get('net_value'))
    snap.dii_net_cr = _safe_float(dii.get('net_value'))

    # VIX
    for idx in snap.indices:
        if 'VIX' in (idx.name or '').upper():
            snap.vix = idx.last
            break

    # Advance / Decline
    if print_output:
        print("  → Fetching breadth data...")
    ad = fetch_advance_decline(snap.indices)
    snap.advance_count = ad.get('advance', 0)
    snap.decline_count = ad.get('decline', 0)
    snap.unchanged_count = ad.get('unchanged', 0)

    # Global cues
    if print_output:
        print("  → Fetching global cues (Dow, Crude, USD/INR)...")
    snap.global_cues = fetch_global_cues()

    # Compute signals
    snap = compute_market_signals(snap)

    # LLM analysis
    prompt = build_market_prompt(snap)

    if print_output:
        print(f"\n{'─'*65}")
        print("🤖  MARKET BOT ANALYSIS:")
        print('─'*65)

    def on_tok(t):
        if print_output:
            print(t, end='', flush=True)

    llm_text = stream_chat(MARKET_BOT_PROMPT, prompt, on_token=on_tok)
    snap.llm_analysis = llm_text

    llm_score = extract_score(llm_text, default=snap.market_score)
    snap.market_score = round(0.5 * llm_score + 0.5 * snap.market_score, 2)

    if print_output:
        print(f"\n\n{'─'*65}")
        print(f"  🌐 MARKET VERDICT:  {snap.market_outlook}")
        print(f"  📊 MARKET SCORE:    {snap.market_score}/10")
        print(f"  💰 FII/DII:         {snap.fii_dii_signal}")
        print(f"  📉 VIX:             {snap.vix or 'N/A'} — {snap.vix_signal}")
        print(f"  📊 BREADTH:         {snap.ad_ratio or 'N/A'} A/D — {snap.breadth_signal}")

    return snap


def _safe_float(val):
    try:
        return float(str(val).replace(',', ''))
    except Exception:
        return None
