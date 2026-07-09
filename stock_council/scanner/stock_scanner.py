# ============================================================
# scanner/stock_scanner.py
# Level 3: Deep-dive every stock in the TOP sectors
# ============================================================
# FLOW:
#   Takes top N sector results from sector_scanner
#   For EACH stock in those sectors:
#     → Runs ALL 5 bots (Fundamental, Technical, News, Sentiment, Risk)
#     → Computes composite verdict score
#     → Ranks ALL stocks across all top sectors
#   Produces the final ranked list: "Best stocks to buy RIGHT NOW"
#
# OPTIMISATION FOR CPU:
#   - Light mode: uses fast quantitative scoring only (no LLM per stock)
#     Only the top 5 stocks get full LLM analysis
#   - This makes the full market scan feasible on CPU
#     (scanning 50 stocks in light mode takes ~5 min vs 4 hours with LLM each)
# ============================================================

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from scanner.universe import SECTORS, PRIORITY_STOCKS, STOCK_TO_SECTOR
from scanner.sector_scanner import SectorResult, fetch_stock_quick_metrics
from scanner.market_scanner import MarketSnapshot
from utils.market_data import fetch_price_history, fetch_fundamentals, resolve_symbol
from utils.live_data import NSELive, FinnhubData, get_live_quote
from utils.llm import stream_chat, extract_score
from config import FA_PARAMS, TA_PARAMS, VERDICT_WEIGHTS, VERBOSE_DEBUG


STOCK_VERDICT_PROMPT = """You are the STOCK VERDICT BOT — final decision maker at an Indian equity fund.

You receive comprehensive quant data on a stock across 5 dimensions.
Your job: deliver a crisp, actionable verdict for an Indian retail investor.

Your response MUST be 120-150 words covering:
1. VERDICT in first line: STRONG BUY / BUY / ACCUMULATE / HOLD / REDUCE / SELL
2. Single best reason to buy (if applicable)
3. Single biggest risk right now
4. Suggested entry range or "wait for dip to ₹X"
5. Time horizon: short-term (days), medium (weeks), long-term (months)

End with: "FINAL SCORE: X/10"
"""


@dataclass
class StockVerdict:
    """Final verdict for one stock from the full council."""
    symbol: str
    sector: str
    company_name: str = ""
    current_price: float = None
    verdict: str = "HOLD"
    final_score: float = 5.0
    rank: int = 0

    # Component scores
    fundamental_score: float = 5.0
    technical_score: float = 5.0
    news_score: float = 5.0
    sentiment_score: float = 5.0
    risk_score: float = 5.0   # risk level — inverted in composite

    # Key metrics for display
    pe_ratio: float = None
    roe: float = None
    rsi: float = None
    change_1m: float = None
    change_1d: float = None
    volume_ratio: float = None
    above_50dma: bool = None
    macd_bullish: bool = None
    beta: float = None
    max_drawdown: float = None

    # LLM output
    llm_analysis: str = ""
    quant_composite: float = 5.0


# ── Quantitative fast-score (no LLM) ─────────────────────────

def quant_fundamental_score(fund: dict) -> float:
    """Fast fundamental score from fetched data. Same formulas as fundamental_bot."""
    if not fund or fund.get('error'):
        return 5.0
    score = 5.0
    p = FA_PARAMS

    pe = fund.get('pe_ratio')
    if pe:
        if pe < p['PE_LOW']:   score += 2.0
        elif pe < p['PE_HIGH']: score += 0.5
        else:                   score -= 1.5

    roe = fund.get('roe')
    if roe:
        if roe >= p['ROE_GREAT']:  score += 2.0
        elif roe >= p['ROE_GOOD']: score += 1.0
        elif roe < 8:              score -= 1.5

    de = fund.get('debt_equity')
    if de is not None:
        if de < 0.3:   score += 2.0
        elif de < 0.7: score += 0.5
        elif de > 1.5: score -= 2.0

    rg = fund.get('revenue_growth')
    if rg:
        if rg >= p['REVENUE_GROWTH_GOOD']: score += 2.0
        elif rg < 0:                        score -= 2.0

    nm = fund.get('net_margin')
    if nm:
        if nm >= p['NET_MARGIN_GOOD']: score += 1.0
        elif nm < 5:                    score -= 1.0

    return round(max(0, min(10, score)), 2)


def quant_technical_score(price_metrics: dict) -> float:
    """Fast technical score from quick price metrics."""
    score = 5.0
    if not price_metrics:
        return score

    if price_metrics.get('above_50dma') is True:   score += 1.5
    elif price_metrics.get('above_50dma') is False: score -= 1.5

    if price_metrics.get('above_200dma') is True:   score += 1.5
    elif price_metrics.get('above_200dma') is False: score -= 1.5

    rsi = price_metrics.get('rsi')
    if rsi:
        if 55 < rsi < 70:  score += 2.0
        elif rsi > 70:     score -= 0.5
        elif rsi < 30:     score -= 1.0
        elif rsi < 45:     score -= 0.5

    vol = price_metrics.get('volume_ratio')
    if vol:
        if vol > 1.5: score += 1.0
        elif vol < 0.5: score -= 0.5

    c1m = price_metrics.get('change_1m')
    if c1m:
        if c1m > 10:   score += 1.0
        elif c1m > 5:  score += 0.5
        elif c1m < -10: score -= 1.0
        elif c1m < -5:  score -= 0.5

    return round(max(0, min(10, score)), 2)


def quant_risk_score_fast(fund: dict, price_metrics: dict) -> float:
    """
    Fast risk score (0-10, higher = more risk).
    Uses HV proxy from 1M change spread and DE ratio.
    """
    score = 5.0
    if fund:
        de = fund.get('debt_equity')
        if de is not None:
            if de < 0.3:   score -= 2.0
            elif de < 0.7: score -= 1.0
            elif de > 1.5: score += 2.0

        beta = fund.get('beta')
        if beta:
            if beta > 1.5:  score += 1.5
            elif beta < 0.8: score -= 1.0

    if price_metrics:
        c3m = price_metrics.get('change_3m')
        if c3m is not None:
            # Proxy for volatility: if 3M range is huge, risky
            if abs(c3m) > 40:   score += 1.5
            elif abs(c3m) < 10: score -= 0.5

    return round(max(0, min(10, score)), 2)


def compute_stock_composite(f_score, t_score, n_score, s_score, r_score) -> float:
    """
    Final composite:
      fundamental × 0.30
      technical   × 0.25
      news        × 0.20
      sentiment   × 0.15
      (10-risk)   × 0.10   ← risk is inverted
    """
    return round(
        f_score * 0.30 +
        t_score * 0.25 +
        n_score * 0.20 +
        s_score * 0.15 +
        (10 - r_score) * 0.10,
        2
    )


def score_to_verdict(score: float) -> str:
    if score >= 8.5:   return "STRONG BUY 🚀"
    elif score >= 7.5: return "BUY 📈"
    elif score >= 6.5: return "ACCUMULATE 💚"
    elif score >= 5.5: return "HOLD ⚖️"
    elif score >= 4.0: return "REDUCE 🟡"
    elif score >= 2.5: return "SELL 📉"
    else:              return "STRONG SELL 🔴"


# ── Full 5-bot analysis (for top stocks only) ─────────────────

def run_full_bots_on_stock(symbol: str, market_snap: MarketSnapshot,
                            print_output: bool = True) -> dict:
    """Run all 5 bots on a single stock. Used for top-ranked stocks only."""
    from bots import fundamental_bot, technical_bot, news_bot, sentiment_bot, risk_bot

    fund_r = fundamental_bot.run(symbol, print_output=print_output)
    tech_r = technical_bot.run(symbol, print_output=print_output)
    news_r = news_bot.run(symbol, print_output=print_output)
    news_sent = news_r.get('sentiment_summary', {}).get('avg_sentiment', 0.0)
    sent_r = sentiment_bot.run(symbol, news_sentiment=news_sent, print_output=print_output)
    risk_r = risk_bot.run(symbol, print_output=print_output)

    return {
        'fundamental': fund_r,
        'technical': tech_r,
        'news': news_r,
        'sentiment': sent_r,
        'risk': risk_r,
    }


def fast_scan_stock(symbol: str, sector: str,
                    market_snap: MarketSnapshot,
                    nifty_1m: float = 0) -> StockVerdict:
    """
    Fast quantitative scan (no LLM) for a single stock.
    Used to rank all stocks before selecting top N for full LLM analysis.
    """
    verdict = StockVerdict(symbol=symbol, sector=sector)

    # Quick price metrics
    pm = fetch_stock_quick_metrics(symbol, nifty_1m=nifty_1m)
    # Fundamentals (from cache)
    fund = fetch_fundamentals(symbol) or {}

    # Populate key metrics
    verdict.current_price = pm.get('price')
    verdict.company_name = fund.get('company_name', symbol)
    verdict.pe_ratio = fund.get('pe_ratio')
    verdict.roe = fund.get('roe')
    verdict.rsi = pm.get('rsi')
    verdict.change_1m = pm.get('change_1m')
    verdict.change_1d = pm.get('change_1d')
    verdict.volume_ratio = pm.get('volume_ratio')
    verdict.above_50dma = pm.get('above_50dma')
    verdict.beta = fund.get('beta')

    # Compute fast scores
    verdict.fundamental_score = quant_fundamental_score(fund)
    verdict.technical_score = quant_technical_score(pm)
    verdict.news_score = 5.0   # no news fetch in fast mode
    verdict.sentiment_score = 5.0 + (market_snap.market_score - 5) * 0.3
    verdict.risk_score = quant_risk_score_fast(fund, pm)

    verdict.quant_composite = compute_stock_composite(
        verdict.fundamental_score,
        verdict.technical_score,
        verdict.news_score,
        verdict.sentiment_score,
        verdict.risk_score
    )
    verdict.final_score = verdict.quant_composite
    verdict.verdict = score_to_verdict(verdict.final_score)

    return verdict


def run_llm_verdict(verdict: StockVerdict, bot_results: dict) -> StockVerdict:
    """
    Run LLM verdict bot on a stock that already has full bot analysis.
    Updates verdict with LLM scores and analysis.
    """
    fund_r = bot_results.get('fundamental', {})
    tech_r = bot_results.get('technical', {})
    news_r = bot_results.get('news', {})
    sent_r = bot_results.get('sentiment', {})
    risk_r = bot_results.get('risk', {})

    # Update scores from full bot results
    verdict.fundamental_score = fund_r.get('score', verdict.fundamental_score)
    verdict.technical_score = tech_r.get('score', verdict.technical_score)
    verdict.news_score = news_r.get('score', verdict.news_score)
    verdict.sentiment_score = sent_r.get('score', verdict.sentiment_score)
    verdict.risk_score = risk_r.get('score', verdict.risk_score)

    # Composite
    composite = compute_stock_composite(
        verdict.fundamental_score,
        verdict.technical_score,
        verdict.news_score,
        verdict.sentiment_score,
        verdict.risk_score
    )

    # Build LLM prompt
    prompt = f"""
STOCK: {verdict.symbol}
Company: {verdict.company_name}
Sector: {verdict.sector}
Price: ₹{verdict.current_price}

BOT SCORES:
  Fundamental: {verdict.fundamental_score}/10
  Technical:   {verdict.technical_score}/10
  News:        {verdict.news_score}/10
  Sentiment:   {verdict.sentiment_score}/10
  Risk Level:  {verdict.risk_score}/10 (higher = more risk)
  COMPOSITE:   {composite}/10

KEY METRICS:
  P/E:         {verdict.pe_ratio}x
  ROE:         {verdict.roe}%
  RSI(14):     {verdict.rsi}
  1M Return:   {verdict.change_1m}%
  Volume:      {verdict.volume_ratio}x average
  Above 50DMA: {verdict.above_50dma}

FUNDAMENTAL BOT SAYS:
{fund_r.get('text', '')[:400]}

TECHNICAL BOT SAYS:
{tech_r.get('text', '')[:400]}

NEWS BOT SAYS:
{news_r.get('text', '')[:300]}

RISK BOT SAYS:
{risk_r.get('text', '')[:300]}

Deliver your final verdict for Indian retail investors.
"""
    tokens = []
    llm_text = stream_chat(
        STOCK_VERDICT_PROMPT, prompt,
        on_token=lambda t: tokens.append(t)
    )
    verdict.llm_analysis = llm_text

    llm_score = extract_score(llm_text, default=composite)
    verdict.final_score = round(0.6 * composite + 0.4 * llm_score, 2)
    verdict.verdict = score_to_verdict(verdict.final_score)
    verdict.quant_composite = composite

    return verdict


# ── Main runner ───────────────────────────────────────────────

def run(sector_results: list[SectorResult],
        market_snap: MarketSnapshot,
        top_n_sectors: int = 3,
        full_llm_for_top: int = 5,
        print_output: bool = True) -> list[StockVerdict]:
    """
    Level 3: Deep-dive stocks in top sectors.

    Args:
        sector_results: ranked list from sector_scanner.run()
        market_snap: from market_scanner.run()
        top_n_sectors: how many top sectors to deep-dive (default 3)
        full_llm_for_top: top N stocks get full 5-bot LLM analysis
        print_output: print progress

    Returns:
        list[StockVerdict] sorted by final_score descending
    """
    if print_output:
        print(f"\n{'═'*65}")
        print(f"📈  LEVEL 3: STOCK SCAN — TOP {top_n_sectors} SECTORS")
        print('═'*65)

    # Get nifty 1M
    nifty_df = fetch_price_history("^NSEI", period="2mo")
    nifty_1m = 0.0
    if nifty_df is not None and not nifty_df.empty and len(nifty_df) >= 22:
        c = nifty_df['Close']
        nifty_1m = round((c.iloc[-1]/c.iloc[-22] - 1)*100, 2)

    # Select top sectors
    top_sectors = sector_results[:top_n_sectors]
    if print_output:
        print(f"  Top sectors selected:")
        for sr in top_sectors:
            print(f"    #{sr.rank} {sr.name} — {sr.score:.1f}/10  ({sr.verdict})")

    # ── PHASE 1: Fast-scan ALL stocks in top sectors ──────────
    if print_output:
        print(f"\n  Phase 1: Fast quantitative scan of all stocks...")

    all_verdicts: list[StockVerdict] = []
    for sr in top_sectors:
        sector = sr.name
        symbols = SECTORS.get(sector, [])
        if print_output:
            print(f"\n  [{sector}] — {len(symbols)} stocks")

        for sym in symbols:
            try:
                v = fast_scan_stock(sym, sector, market_snap, nifty_1m)
                all_verdicts.append(v)
                if print_output and v.current_price:
                    arrow = '▲' if (v.change_1d or 0) >= 0 else '▼'
                    print(
                        f"    {sym:14} ₹{v.current_price:>8.2f}  "
                        f"{arrow}{abs(v.change_1d or 0):.1f}%  "
                        f"Score:{v.final_score:.1f}  {v.verdict}"
                    )
                time.sleep(0.1)
            except Exception as e:
                if VERBOSE_DEBUG:
                    print(f"    ⚠ {sym}: {e}")

    # Sort by composite score
    all_verdicts.sort(key=lambda v: v.final_score, reverse=True)

    # ── PHASE 2: Full 5-bot LLM on top N stocks ───────────────
    if print_output:
        print(f"\n  Phase 2: Full 5-bot analysis on top {full_llm_for_top} stocks...")

    top_stocks = all_verdicts[:full_llm_for_top]
    for i, verdict in enumerate(top_stocks):
        if print_output:
            print(f"\n  {'─'*55}")
            print(f"  🔬 FULL ANALYSIS #{i+1}: {verdict.symbol} ({verdict.sector})")
            print(f"  {'─'*55}")
        try:
            bot_results = run_full_bots_on_stock(
                verdict.symbol, market_snap, print_output=print_output
            )
            verdict = run_llm_verdict(verdict, bot_results)
            all_verdicts[i] = verdict   # update in place
        except Exception as e:
            if print_output:
                print(f"  ❌ Error in full analysis for {verdict.symbol}: {e}")

    # Re-sort after LLM updates
    all_verdicts.sort(key=lambda v: v.final_score, reverse=True)
    for i, v in enumerate(all_verdicts):
        v.rank = i + 1

    # ── Print final ranked list ────────────────────────────────
    if print_output:
        print(f"\n{'═'*65}")
        print(f"🏆  FINAL STOCK RANKINGS — TOP PICKS FROM NSE")
        print('═'*65)
        print(
            f"  {'#':3} {'Symbol':12} {'Sector':25} {'Price':8} "
            f"{'1M':7} {'Score':7} Verdict"
        )
        print(f"  {'─'*70}")
        for v in all_verdicts[:20]:
            arrow = '▲' if (v.change_1m or 0) >= 0 else '▼'
            price_str = f"₹{v.current_price:.0f}" if v.current_price else "N/A"
            print(
                f"  #{v.rank:<2} {v.symbol:12} {v.sector[:24]:25} "
                f"{price_str:8} {arrow}{abs(v.change_1m or 0):.1f}%  "
                f"{v.final_score:>4.1f}/10  {v.verdict}"
            )
        print('═'*65)

    return all_verdicts
