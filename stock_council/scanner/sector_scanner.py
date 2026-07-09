# ============================================================
# scanner/sector_scanner.py
# Level 2: Score & rank ALL 13 sectors
# ============================================================
# FLOW:
#   Takes MarketSnapshot from market_scanner
#   For each sector:
#     → Fetch price performance of ALL stocks in that sector
#     → Compute sector-level metrics:
#         Sector Momentum    = avg pct_change of stocks (1d, 1w, 1m)
#         Sector Breadth     = % stocks above 50 DMA
#         Relative Strength  = sector momentum vs Nifty50
#         Sector RSI         = avg RSI of all stocks in sector
#         Volume Surge       = avg volume ratio vs 20D avg
#         Valuation Band     = avg PE of sector vs historical
#     → LLM analyses sector: fundamentals + technicals + macro
#     → Gives sector score 0-10
#   Ranks all 13 sectors → top 3 passed to stock_scanner
# ============================================================

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from scanner.universe import SECTORS, PRIORITY_STOCKS, SECTOR_RISK_PROFILE
from scanner.market_scanner import MarketSnapshot
from utils.market_data import fetch_price_history, fetch_fundamentals, resolve_symbol
from utils.llm import stream_chat, extract_score
from config import VERBOSE_DEBUG, TA_PARAMS
from memory.storage import load_prices_csv

SECTOR_BOT_PROMPT = """You are SECTOR BOT — an Indian equity sector analyst at a top Mumbai fund house.

You receive quantitative data on ALL stocks in a sector.
Your job: determine if this sector is in FAVOUR or OUT OF FAVOUR right now.

Consider:
- Sector momentum vs Nifty (is it outperforming or lagging?)
- Breadth (are most stocks rising or just 1-2 leaders?)
- Macro tailwinds/headwinds specific to this sector
- FII preference (FIIs tend to buy IT/Banks, avoid small sectors)
- RBI policy sensitivity (NBFC, Banks, Real Estate most sensitive)
- Government policy: PLI schemes, capex, disinvestment targets

Your response MUST:
- Be 100-130 words
- Compare this sector's relative strength vs Nifty
- Identify the single biggest catalyst (positive or negative)
- Name 2-3 stocks within this sector that look strongest
- End with exactly: "SECTOR SCORE: X/10"

Scoring:
  9-10: Sector in strong uptrend, broad participation, macro tailwind
  7-8:  Sector outperforming with good momentum
  5-6:  In-line with market, no clear edge
  3-4:  Underperforming, headwinds
  1-2:  Sector in bear phase, avoid
"""


@dataclass
class SectorResult:
    """Complete analysis result for one sector."""
    name: str
    score: float = 5.0
    rank: int = 0
    verdict: str = "NEUTRAL"

    # Price metrics
    avg_1d_change: float = None
    avg_1w_change: float = None
    avg_1m_change: float = None
    vs_nifty_1m: float = None           # relative to Nifty 1M

    # Technical metrics
    pct_above_50dma: float = None
    pct_above_200dma: float = None
    avg_rsi: float = None
    avg_volume_ratio: float = None

    # Fundamental metrics
    avg_pe: float = None
    avg_roe: float = None
    avg_revenue_growth: float = None

    # Sub-scores
    momentum_score: float = 5.0
    breadth_score: float = 5.0
    technical_score: float = 5.0
    macro_score: float = 5.0

    # Stock-level detail
    stock_data: dict = field(default_factory=dict)   # symbol → {price, change, rsi, ...}
    top_stocks: list = field(default_factory=list)   # ranked top 3
    weak_stocks: list = field(default_factory=list)

    llm_analysis: str = ""
    quant_score: float = 5.0


# ── Price data helpers ─────────────────────────────────────────

def _safe(val, default=None):
    if val is None:
        return default
    try:
        f = float(val)
        return None if (f != f) else f   # NaN check
    except Exception:
        return default


def fetch_stock_quick_metrics(symbol: str, nifty_1m: float = 0) -> dict:
    """
    Fetch a stock's key metrics quickly.
    Reads from local CSV files first (populated by nightly download),
    so technical indicators are always calculated even when Yahoo Finance
    is slow or rate-limiting.
    Returns minimal dict for sector aggregation.
    """
    try:
        # ── Primary: load from local CSV (always fresh from nightly job) ──
        df = load_prices_csv(symbol, days=90)

        # ── Fallback: fetch from Yahoo Finance if no local data ──
        if df is None or df.empty or len(df) < 5:
            df = fetch_price_history(symbol, period="3mo", interval="1d")

        if df is None or df.empty or len(df) < 5:
            return {}

        close = df['Close']
        volume = df['Volume']
        price = float(close.iloc[-1])

        # Price changes
        c1d = round((close.iloc[-1]/close.iloc[-2] - 1)*100, 2) if len(close) >= 2 else None
        c1w = round((close.iloc[-1]/close.iloc[-6] - 1)*100, 2) if len(close) >= 6 else None
        c1m = round((close.iloc[-1]/close.iloc[-22] - 1)*100, 2) if len(close) >= 22 else None
        c3m = round((close.iloc[-1]/close.iloc[0] - 1)*100, 2) if len(close) >= 60 else None

        # Moving averages
        sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
        sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
        above_50dma = price > sma50 if sma50 else None
        above_200dma = price > sma200 if sma200 else None

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = float((100 - 100/(1+rs)).iloc[-1])

        # Volume
        vol_avg = float(volume.tail(20).mean())
        vol_today = float(volume.iloc[-1])
        vol_ratio = round(vol_today / vol_avg, 2) if vol_avg > 0 else None

        # Relative strength vs Nifty
        vs_nifty = round((c1m or 0) - nifty_1m, 2) if c1m is not None else None

        return {
            'symbol': symbol,
            'price': round(price, 2),
            'change_1d': c1d,
            'change_1w': c1w,
            'change_1m': c1m,
            'change_3m': c3m,
            'above_50dma': above_50dma,
            'above_200dma': above_200dma,
            'rsi': round(rsi, 1),
            'volume_ratio': vol_ratio,
            'vs_nifty_1m': vs_nifty,
        }
    except Exception as e:
        if VERBOSE_DEBUG:
            print(f"  [SECTOR] Error for {symbol}: {e}")
        return {}


def fetch_sector_fundamentals(symbols: list) -> dict:
    """
    Fetch fundamental metrics for a sample of stocks in the sector.
    Returns aggregated averages.
    """
    pe_list, roe_list, rev_growth_list = [], [], []
    # Sample up to 5 stocks to avoid too many API calls
    sample = symbols[:5]
    for sym in sample:
        try:
            fund = fetch_fundamentals(sym)
            if fund and not fund.get('error'):
                pe = _safe(fund.get('pe_ratio'))
                roe = _safe(fund.get('roe'))
                rg = _safe(fund.get('revenue_growth'))
                if pe and 0 < pe < 200:
                    pe_list.append(pe)
                if roe:
                    roe_list.append(roe)
                if rg:
                    rev_growth_list.append(rg)
            time.sleep(0.3)  # gentle on cache/API
        except Exception:
            pass

    return {
        'avg_pe': round(np.mean(pe_list), 1) if pe_list else None,
        'avg_roe': round(np.mean(roe_list), 1) if roe_list else None,
        'avg_revenue_growth': round(np.mean(rev_growth_list), 1) if rev_growth_list else None,
    }


# ── Sector scoring formulas ────────────────────────────────────

def score_sector_momentum(stocks: dict) -> tuple[float, float, float, float]:
    """
    Compute momentum scores from price changes.
    Returns: (score, avg_1d, avg_1w, avg_1m)
    """
    vals_1d = [v['change_1d'] for v in stocks.values() if v.get('change_1d') is not None]
    vals_1w = [v['change_1w'] for v in stocks.values() if v.get('change_1w') is not None]
    vals_1m = [v['change_1m'] for v in stocks.values() if v.get('change_1m') is not None]

    avg_1d = round(np.mean(vals_1d), 2) if vals_1d else None
    avg_1w = round(np.mean(vals_1w), 2) if vals_1w else None
    avg_1m = round(np.mean(vals_1m), 2) if vals_1m else None

    score = 5.0
    if avg_1m is not None:
        if avg_1m > 10:   score += 2.5
        elif avg_1m > 5:  score += 1.5
        elif avg_1m > 2:  score += 0.5
        elif avg_1m < -10: score -= 2.5
        elif avg_1m < -5:  score -= 1.5
        elif avg_1m < -2:  score -= 0.5

    if avg_1w is not None:
        if avg_1w > 3:   score += 1.0
        elif avg_1w < -3: score -= 1.0

    if avg_1d is not None:
        if avg_1d > 1:   score += 0.5
        elif avg_1d < -1: score -= 0.5

    return round(max(0, min(10, score)), 2), avg_1d, avg_1w, avg_1m


def score_sector_breadth(stocks: dict) -> tuple[float, float, float]:
    """
    Breadth = % stocks above 50/200 DMA.
    Returns: (score, pct_above_50dma, pct_above_200dma)
    """
    above_50 = [v['above_50dma'] for v in stocks.values() if v.get('above_50dma') is not None]
    above_200 = [v['above_200dma'] for v in stocks.values() if v.get('above_200dma') is not None]

    pct_50 = round(sum(above_50)/len(above_50)*100, 1) if above_50 else None
    pct_200 = round(sum(above_200)/len(above_200)*100, 1) if above_200 else None

    score = 5.0
    if pct_50 is not None:
        if pct_50 > 75:  score += 2.5
        elif pct_50 > 50: score += 1.0
        elif pct_50 < 25: score -= 2.0
        elif pct_50 < 40: score -= 1.0

    if pct_200 is not None:
        if pct_200 > 70: score += 1.5
        elif pct_200 < 30: score -= 1.5

    return round(max(0, min(10, score)), 2), pct_50, pct_200


def score_sector_technical(stocks: dict) -> tuple[float, float, float]:
    """
    Score RSI + volume signals.
    Returns: (score, avg_rsi, avg_volume_ratio)
    """
    rsi_vals = [v['rsi'] for v in stocks.values() if v.get('rsi') is not None]
    vol_vals = [v['volume_ratio'] for v in stocks.values() if v.get('volume_ratio') is not None]

    avg_rsi = round(np.mean(rsi_vals), 1) if rsi_vals else None
    avg_vol = round(np.mean(vol_vals), 2) if vol_vals else None

    score = 5.0
    if avg_rsi is not None:
        if 55 < avg_rsi < 70:  score += 2.0   # bullish momentum zone
        elif avg_rsi > 70:     score -= 0.5    # overbought
        elif 40 < avg_rsi < 55: score += 0.5
        elif avg_rsi < 30:     score -= 1.5    # sector oversold (could bounce)
        elif avg_rsi < 40:     score -= 1.0

    if avg_vol is not None:
        if avg_vol > 1.5:  score += 1.0   # high volume = conviction
        elif avg_vol < 0.5: score -= 0.5  # low volume = weak move

    return round(max(0, min(10, score)), 2), avg_rsi, avg_vol


def score_sector_macro(sector_name: str, market_snap: MarketSnapshot) -> tuple[float, str]:
    """
    Score macro environment for this sector given current market state.
    Each sector reacts differently to FII flows, VIX, RBI, etc.
    """
    profile = SECTOR_RISK_PROFILE.get(sector_name, {})
    score = 5.0
    notes = []

    fii_signal = market_snap.fii_dii_signal
    vix = market_snap.vix or 15

    # FII preference by sector
    # FII loves: IT (dollar earnings), Banks, Consumer Staples
    # FII avoids: Small caps, Real Estate, Telecom
    fii_positive_sectors = ["Information Technology", "Financial Services",
                             "Consumer Staples", "Healthcare"]
    fii_neutral_sectors = ["Consumer Discretionary", "Industrials", "Energy"]
    fii_negative_sectors = ["Real Estate", "Telecom", "Media & Entertainment",
                             "New Age / Tech Platform"]

    if "BULLISH" in fii_signal:
        if sector_name in fii_positive_sectors:
            score += 2.0
            notes.append(f"FII buying — {sector_name} is a primary FII favourite")
        elif sector_name in fii_neutral_sectors:
            score += 0.5
            notes.append(f"FII buying benefits {sector_name} indirectly")
        else:
            score += 0.0
            notes.append(f"FII buying has limited direct impact on {sector_name}")

    elif "BEARISH" in fii_signal:
        if sector_name in fii_positive_sectors:
            score -= 2.0
            notes.append(f"FII selling hits {sector_name} hard — direct outflow pressure")
        elif sector_name in fii_negative_sectors:
            score -= 0.5
        else:
            score -= 1.0

    # VIX impact
    if vix > 20:
        if profile.get('defensive'):
            score += 1.0
            notes.append(f"High VIX ({vix:.1f}) → defensive sector benefits")
        elif profile.get('risk') == 'HIGH':
            score -= 1.5
            notes.append(f"High VIX ({vix:.1f}) → high-risk sector hurt by fear")

    # Sector-specific macro
    global_cues = market_snap.global_cues
    crude = global_cues.get('Crude Oil (WTI)', {}).get('change_pct', 0) or 0
    usd_inr = global_cues.get('USD/INR', {}).get('change_pct', 0) or 0

    if profile.get('crude_sensitive'):
        if crude > 2:
            score -= 1.0
            notes.append(f"Crude up {crude:.1f}% — margin pressure for oil consumers")
        elif crude < -2:
            score += 1.0
            notes.append(f"Crude down {crude:.1f}% — cost relief")

    if profile.get('export_driven'):
        if usd_inr > 0:
            score += 0.5
            notes.append(f"Rupee weakening — export earnings benefit")
        elif usd_inr < -0.5:
            score -= 0.5
            notes.append(f"Rupee strengthening — export competitiveness hit")

    if profile.get('rbi_sensitive'):
        notes.append("Watch: RBI rate stance critical for this sector")

    return round(max(0, min(10, score)), 2), ' | '.join(notes) if notes else "No macro adjustment"


def rank_stocks_in_sector(stocks: dict) -> tuple[list, list]:
    """
    Rank all stocks in a sector.
    Top stocks: highest composite (momentum + RSI + volume).
    Weak stocks: lowest.
    Returns: (top_stocks, weak_stocks)
    """
    scored = []
    for sym, data in stocks.items():
        if not data:
            continue
        # Quick composite for ranking
        c1m = data.get('change_1m', 0) or 0
        rsi = data.get('rsi', 50) or 50
        vol = data.get('volume_ratio', 1) or 1
        vs_n = data.get('vs_nifty_1m', 0) or 0

        # Score: momentum 40%, relative strength 30%, rsi zone 20%, volume 10%
        m_score = min(10, max(0, 5 + c1m/5))
        rs_score = min(10, max(0, 5 + vs_n/4))
        rsi_score = 7 if 50 < rsi < 70 else 4 if rsi > 70 else 3 if rsi < 40 else 5
        vol_score = min(10, max(0, vol * 5))

        composite = m_score*0.40 + rs_score*0.30 + rsi_score*0.20 + vol_score*0.10
        scored.append((sym, round(composite, 2), data))

    scored.sort(key=lambda x: x[1], reverse=True)
    # Expanded from 3→10 so the orchestrator's GPS filter sees a larger candidate
    # pool per sector. High-GPS stocks (e.g. OFSS GPS 8.05) that have moderate
    # short-term momentum now survive into the GPS filter instead of being cut off.
    top = [(s[0], s[1], s[2]) for s in scored[:10]]
    weak = [(s[0], s[1], s[2]) for s in scored[-3:] if scored]
    return top, weak


def build_sector_prompt(sector: str, result: SectorResult,
                         market_snap: MarketSnapshot) -> str:
    """Build the LLM prompt for one sector."""

    # Top stocks detail
    top_lines = []
    for sym, score, data in result.top_stocks:
        c1m = data.get('change_1m', 0) or 0
        rsi = data.get('rsi', 0) or 0
        arrow = '▲' if c1m >= 0 else '▼'
        top_lines.append(
            f"  ★ {sym:15} {arrow}{abs(c1m):.1f}% (1M) | RSI {rsi:.0f} | "
            f"Vol {data.get('volume_ratio',1):.1f}x avg"
        )
    top_str = '\n'.join(top_lines) if top_lines else "  (no data)"

    # Weak stocks
    weak_lines = []
    for sym, score, data in result.weak_stocks:
        c1m = data.get('change_1m', 0) or 0
        rsi = data.get('rsi', 0) or 0
        weak_lines.append(f"  ↘ {sym:15} {c1m:+.1f}% (1M) | RSI {rsi:.0f}")
    weak_str = '\n'.join(weak_lines) if weak_lines else "  (no data)"

    # All stocks snapshot
    all_stocks_lines = []
    for sym, data in list(result.stock_data.items())[:15]:
        c1d = data.get('change_1d', 0) or 0
        c1m = data.get('change_1m', 0) or 0
        rsi = data.get('rsi', 0) or 0
        arrow = '▲' if c1d >= 0 else '▼'
        all_stocks_lines.append(
            f"  {sym:12} ₹{data.get('price',0):.0f}  "
            f"{arrow}{abs(c1d):.1f}%  1M:{c1m:+.1f}%  RSI:{rsi:.0f}"
        )
    stocks_str = '\n'.join(all_stocks_lines) if all_stocks_lines else "  (no data)"

    risk_profile = SECTOR_RISK_PROFILE.get(sector, {})

    return f"""
=== SECTOR ANALYSIS: {sector.upper()} ===

Market Context:
  Nifty50 Score:   {market_snap.market_score}/10 ({market_snap.market_outlook})
  FII/DII Signal:  {market_snap.fii_dii_signal}
  VIX:             {market_snap.vix or 'N/A'} ({market_snap.vix_signal})
  Market Breadth:  {market_snap.ad_ratio or 'N/A'} A/D ({market_snap.breadth_signal})

Sector Risk Profile:
  Risk Level:      {risk_profile.get('risk', 'MEDIUM')}
  Cyclical:        {risk_profile.get('cyclical', False)}
  Export Driven:   {risk_profile.get('export_driven', False)}
  RBI Sensitive:   {risk_profile.get('rbi_sensitive', False)}
  Crude Sensitive: {risk_profile.get('crude_sensitive', False)}

Sector Performance Metrics:
  Avg 1-Day:       {result.avg_1d_change:+.2f}%
  Avg 1-Week:      {result.avg_1w_change:+.2f}%
  Avg 1-Month:     {result.avg_1m_change:+.2f}%  (Nifty: reference)
  vs Nifty (1M):   {result.vs_nifty_1m:+.2f}%

Technical Breadth:
  Above 50 DMA:    {result.pct_above_50dma:.0f}% of stocks
  Above 200 DMA:   {result.pct_above_200dma:.0f}% of stocks
  Avg RSI:         {result.avg_rsi:.1f}
  Avg Volume Ratio:{result.avg_volume_ratio:.2f}x (1.0 = normal)

Fundamentals (sector sample):
  Avg P/E:         {result.avg_pe or 'N/A'}x
  Avg ROE:         {result.avg_roe or 'N/A'}%
  Avg Rev Growth:  {result.avg_revenue_growth or 'N/A'}%

Top Stocks in Sector:
{top_str}

Underperforming Stocks:
{weak_str}

All Stocks Snapshot:
{stocks_str}

Pre-computed Sub-scores:
  Momentum Score:  {result.momentum_score}/10
  Breadth Score:   {result.breadth_score}/10
  Technical Score: {result.technical_score}/10
  Macro Score:     {result.macro_score}/10
  QUANTITATIVE:    {result.quant_score}/10

Analyse this sector. Name the top 2-3 stocks to focus on.
"""


def analyse_one_sector(sector: str, symbols: list,
                        market_snap: MarketSnapshot,
                        nifty_1m: float,
                        print_output: bool = True) -> SectorResult:
    """Full analysis pipeline for one sector."""
    result = SectorResult(name=sector)

    if print_output:
        print(f"\n  {'─'*55}")
        print(f"  📂  {sector}  ({len(symbols)} stocks)")

    # Step 1: Fetch all stock metrics
    stock_data = {}
    for sym in symbols:
        metrics = fetch_stock_quick_metrics(sym, nifty_1m=nifty_1m)
        if metrics:
            stock_data[sym] = metrics
        time.sleep(0.05)  # avoid hammering cache

    result.stock_data = stock_data

    if not stock_data:
        if print_output:
            print(f"    ⚠ No data for {sector}")
        return result

    if print_output:
        print(f"    ✓ Got data for {len(stock_data)}/{len(symbols)} stocks")

    # Step 2: Rank stocks
    result.top_stocks, result.weak_stocks = rank_stocks_in_sector(stock_data)

    # Step 3: Score momentum
    result.momentum_score, result.avg_1d_change, \
        result.avg_1w_change, result.avg_1m_change = score_sector_momentum(stock_data)

    result.avg_1d_change = result.avg_1d_change or 0
    result.avg_1w_change = result.avg_1w_change or 0
    result.avg_1m_change = result.avg_1m_change or 0

    # Relative to Nifty
    result.vs_nifty_1m = round(result.avg_1m_change - nifty_1m, 2)

    # Step 4: Breadth
    result.breadth_score, result.pct_above_50dma, result.pct_above_200dma = \
        score_sector_breadth(stock_data)
    result.pct_above_50dma = result.pct_above_50dma or 50
    result.pct_above_200dma = result.pct_above_200dma or 50

    # Step 5: Technical
    result.technical_score, result.avg_rsi, result.avg_volume_ratio = \
        score_sector_technical(stock_data)
    result.avg_rsi = result.avg_rsi or 50
    result.avg_volume_ratio = result.avg_volume_ratio or 1.0

    # Step 6: Macro
    result.macro_score, macro_note = score_sector_macro(sector, market_snap)

    # Step 7: Fundamentals (sample)
    fund_data = fetch_sector_fundamentals(symbols[:5])
    result.avg_pe = fund_data.get('avg_pe')
    result.avg_roe = fund_data.get('avg_roe')
    result.avg_revenue_growth = fund_data.get('avg_revenue_growth')

    # Step 8: Quantitative composite
    # Weights: momentum 35%, breadth 25%, technical 20%, macro 20%
    result.quant_score = round(
        result.momentum_score * 0.35 +
        result.breadth_score * 0.25 +
        result.technical_score * 0.20 +
        result.macro_score * 0.20,
        2
    )

    # Step 9: Fast quant scoring (LLM reserved for stock debate only)
    result.llm_analysis = (
        f"{sector} | Score: {result.quant_score:.1f}/10 | "
        f"1M: {result.avg_1m_change:+.1f}% | RSI: {result.avg_rsi:.0f} | "
        f"Top: {', '.join(t[0] for t in result.top_stocks[:3])}"
    )
    result.score = result.quant_score

    # Verdict label
    if result.score >= 8:
        result.verdict = "STRONG BUY"
    elif result.score >= 7:
        result.verdict = "BUY"
    elif result.score >= 6:
        result.verdict = "ACCUMULATE"
    elif result.score >= 5:
        result.verdict = "HOLD"
    elif result.score >= 3:
        result.verdict = "AVOID"
    else:
        result.verdict = "BEARISH"

    if print_output:
        print(f"    Score: {result.score}/10  |  {result.verdict}")
        print(f"    Top: {', '.join(s[0] for s in result.top_stocks)}")

    return result


def run(market_snap: MarketSnapshot,
        priority_only: bool = False,
        print_output: bool = True) -> list[SectorResult]:
    """
    Scan ALL 13 sectors.
    Returns list of SectorResult sorted by score (best first).

    Args:
        market_snap: output from market_scanner.run()
        priority_only: if True, use only 5 priority stocks per sector (faster)
    """
    if print_output:
        print(f"\n{'═'*65}")
        print("📂  LEVEL 2: SECTOR SCAN — ALL 13 SECTORS")
        print('═'*65)
        print(f"  Market Backdrop: {market_snap.market_outlook} ({market_snap.market_score}/10)")
        print(f"  Scanning {'priority stocks only' if priority_only else 'all stocks'}...")

    # Get Nifty 1M performance as baseline
    nifty_df = fetch_price_history("^NSEI", period="2mo", interval="1d")
    nifty_1m = 0.0
    if nifty_df is not None and not nifty_df.empty and len(nifty_df) >= 22:
        close = nifty_df['Close']
        nifty_1m = round((close.iloc[-1]/close.iloc[-22] - 1)*100, 2)

    if print_output:
        print(f"  Nifty 1M baseline: {nifty_1m:+.2f}%")

    # Decide which stocks to scan per sector
    results = []
    for sector, all_stocks in SECTORS.items():
        if priority_only:
            stocks = PRIORITY_STOCKS.get(sector, all_stocks[:5])
        else:
            stocks = all_stocks

        try:
            result = analyse_one_sector(
                sector, stocks, market_snap,
                nifty_1m, print_output=print_output
            )
            results.append(result)
        except Exception as e:
            if print_output:
                print(f"    ❌ Error in {sector}: {e}")

    # Rank sectors
    results.sort(key=lambda r: r.score, reverse=True)
    for i, r in enumerate(results):
        r.rank = i + 1

    # Print ranking table
    if print_output:
        print(f"\n{'─'*65}")
        print("  📊  SECTOR RANKING:")
        print(f"  {'Rank':4} {'Sector':35} {'Score':7} {'1M':8} {'RSI':6} Verdict")
        print(f"  {'─'*60}")
        for r in results:
            arrow = '▲' if (r.avg_1m_change or 0) >= 0 else '▼'
            top_str = ', '.join(t[0] for t in r.top_stocks[:3])
            print(
                f"  #{r.rank:<3} {r.name:35} {r.score:>5.1f}/10  "
                f"{arrow}{abs(r.avg_1m_change or 0):.1f}%  "
                f"RSI:{r.avg_rsi:.0f}  {r.verdict}  | Top: {top_str}"
            )

    return results
