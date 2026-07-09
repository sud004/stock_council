# ============================================================
# bots/fundamental_bot.py — Fundamental Analysis Bot
# ============================================================
#
# FORMULAS USED:
#   P/E Ratio         = Market Price / EPS (Trailing Twelve Months)
#   Forward P/E       = Market Price / Forward EPS
#   P/B Ratio         = Market Price / Book Value per Share
#   PEG Ratio         = P/E / Annual EPS Growth Rate
#   EV/EBITDA         = Enterprise Value / EBITDA
#   EV/Revenue        = Enterprise Value / Revenue
#   ROE               = Net Income / Shareholders' Equity × 100
#   ROCE              = EBIT / Capital Employed × 100
#   Net Margin        = Net Income / Revenue × 100
#   Gross Margin      = Gross Profit / Revenue × 100
#   Operating Margin  = Operating Income / Revenue × 100
#   D/E Ratio         = Total Debt / Shareholders' Equity
#   Current Ratio     = Current Assets / Current Liabilities
#   Quick Ratio       = (Current Assets - Inventory) / Current Liabilities
#   Revenue Growth    = (Current Revenue - Prior Revenue) / |Prior Revenue| × 100
#   EPS Growth        = (Current EPS - Prior EPS) / |Prior EPS| × 100
#   FCF Yield         = Free Cash Flow / Market Cap × 100
#   Dividend Yield    = Annual Dividend / Market Price × 100
#   Promoter Holding  = Promoter Shares / Total Shares × 100
#
# COMPOSITE SCORE LOGIC:
#   Each metric is scored 0-10:
#     Valuation Score (P/E, P/B, EV/EBITDA)    weight 0.25
#     Profitability Score (ROE, ROCE, margins)   weight 0.25
#     Growth Score (revenue, EPS, earnings)      weight 0.25
#     Financial Health (D/E, current ratio)      weight 0.15
#     Shareholding Quality (promoter, FII)       weight 0.10
#   Final = weighted average
# ============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import FA_PARAMS, VERBOSE_DEBUG
from utils.market_data import fetch_fundamentals, fetch_peer_data
from utils.llm import stream_chat, extract_score, format_currency, fmt
from utils.database import save_analysis


SYSTEM_PROMPT = """You are FUNDAMENTAL BOT — an expert Indian stock market fundamental analyst.
You work at a top BSE/NSE-registered advisory firm in India.

Your job:
1. Analyse the quantitative fundamental data provided
2. Compare against Indian sector peers
3. Identify value traps vs genuine opportunities
4. Consider India-specific factors: promoter holding, pledging, SEBI regulations, GST impact, RBI policy
5. Give a balanced bull/bear case

Your response MUST:
- Be 150-200 words
- Highlight 3 key positives and 2 key concerns
- End with exactly: "FUNDAMENTAL SCORE: X/10" where X is your assessment

Scoring guide:
  9-10: Exceptional value, strong growth, clean balance sheet
  7-8:  Good fundamentals with minor concerns
  5-6:  Mixed — some good, some bad
  3-4:  Weak fundamentals, overvalued or declining
  1-2:  Extremely poor, avoid
"""


def score_valuation(fund: dict) -> tuple[float, list[str]]:
    """Score valuation metrics 0-10. Lower P/E etc. = higher score."""
    score = 5.0
    notes = []
    p = FA_PARAMS

    pe = fund.get('pe_ratio')
    if pe is not None:
        if pe < p['PE_LOW']:
            score += 2.0
            notes.append(f"P/E {fmt(pe)}x — attractive valuation (below {p['PE_LOW']}x)")
        elif pe < p['PE_HIGH']:
            score += 0.5
            notes.append(f"P/E {fmt(pe)}x — fair valuation")
        else:
            score -= 1.5
            notes.append(f"P/E {fmt(pe)}x — expensive (above {p['PE_HIGH']}x)")

    pb = fund.get('pb_ratio')
    if pb is not None:
        if pb < p['PB_LOW']:
            score += 1.0
            notes.append(f"P/B {fmt(pb)}x — trading below sector average")
        elif pb > p['PB_HIGH']:
            score -= 1.0
            notes.append(f"P/B {fmt(pb)}x — premium valuation")
        else:
            notes.append(f"P/B {fmt(pb)}x — in-line with peers")

    ev_ebitda = fund.get('ev_ebitda')
    if ev_ebitda is not None:
        if ev_ebitda < p['EV_EBITDA_LOW']:
            score += 1.0
            notes.append(f"EV/EBITDA {fmt(ev_ebitda)}x — undervalued on EV basis")
        elif ev_ebitda > p['EV_EBITDA_HIGH']:
            score -= 1.0
            notes.append(f"EV/EBITDA {fmt(ev_ebitda)}x — premium multiple")

    peg = fund.get('peg_ratio')
    if peg is not None:
        if peg < 1.0:
            score += 1.0
            notes.append(f"PEG {fmt(peg)}x — growth underpriced")
        elif peg > 2.0:
            score -= 0.5
            notes.append(f"PEG {fmt(peg)}x — growth expensive")

    return max(0, min(10, score)), notes


def score_profitability(fund: dict) -> tuple[float, list[str]]:
    """Score profitability metrics 0-10."""
    score = 5.0
    notes = []
    p = FA_PARAMS

    roe = fund.get('roe')
    if roe is not None:
        if roe >= p['ROE_GREAT']:
            score += 2.0
            notes.append(f"ROE {fmt(roe)}% — excellent capital efficiency")
        elif roe >= p['ROE_GOOD']:
            score += 1.0
            notes.append(f"ROE {fmt(roe)}% — good return on equity")
        elif roe < 8:
            score -= 1.5
            notes.append(f"ROE {fmt(roe)}% — poor capital allocation")

    roce = fund.get('roce')
    if roce is not None:
        if roce >= p['ROCE_GOOD']:
            score += 1.0
            notes.append(f"ROCE {fmt(roce)}% — efficient capital deployment")
        elif roce < 8:
            score -= 1.0
            notes.append(f"ROCE {fmt(roce)}% — below cost of capital concern")

    net_margin = fund.get('net_margin')
    if net_margin is not None:
        if net_margin >= p['NET_MARGIN_GOOD']:
            score += 1.0
            notes.append(f"Net margin {fmt(net_margin)}% — strong profitability")
        elif net_margin < 5:
            score -= 1.0
            notes.append(f"Net margin {fmt(net_margin)}% — thin margins")

    op_margin = fund.get('operating_margin')
    if op_margin is not None:
        notes.append(f"Operating margin {fmt(op_margin)}%")

    return max(0, min(10, score)), notes


def score_growth(fund: dict) -> tuple[float, list[str]]:
    """Score growth metrics 0-10."""
    score = 5.0
    notes = []
    p = FA_PARAMS

    rev_growth = fund.get('revenue_growth')
    if rev_growth is not None:
        if rev_growth >= p['REVENUE_GROWTH_GOOD']:
            score += 2.0
            notes.append(f"Revenue growth {fmt(rev_growth)}% YoY — strong topline expansion")
        elif rev_growth >= 8:
            score += 0.5
            notes.append(f"Revenue growth {fmt(rev_growth)}% YoY — moderate")
        elif rev_growth < 0:
            score -= 2.0
            notes.append(f"Revenue DECLINING {fmt(rev_growth)}% YoY — red flag")

    eps_growth = fund.get('eps_growth')
    if eps_growth is not None:
        if eps_growth >= p['EPS_GROWTH_GOOD']:
            score += 2.0
            notes.append(f"EPS growth {fmt(eps_growth)}% — strong earnings acceleration")
        elif eps_growth < 0:
            score -= 1.5
            notes.append(f"EPS declining {fmt(eps_growth)}% — earnings pressure")

    earnings_growth = fund.get('earnings_growth')
    if earnings_growth is not None and eps_growth is None:
        if earnings_growth >= 15:
            score += 1.0
            notes.append(f"Earnings growth {fmt(earnings_growth)}%")

    return max(0, min(10, score)), notes


def score_financial_health(fund: dict) -> tuple[float, list[str]]:
    """Score balance sheet / financial health 0-10."""
    score = 5.0
    notes = []
    p = FA_PARAMS

    de_ratio = fund.get('debt_equity')
    if de_ratio is not None:
        if de_ratio < p['DEBT_EQUITY_LOW']:
            score += 2.0
            notes.append(f"D/E ratio {fmt(de_ratio)}x — low leverage, strong balance sheet")
        elif de_ratio < p['DEBT_EQUITY_HIGH']:
            score += 0.0
            notes.append(f"D/E ratio {fmt(de_ratio)}x — manageable debt")
        else:
            score -= 2.0
            notes.append(f"D/E ratio {fmt(de_ratio)}x — HIGH DEBT — risk factor")

    current_ratio = fund.get('current_ratio')
    if current_ratio is not None:
        if current_ratio >= 2.0:
            score += 1.0
            notes.append(f"Current ratio {fmt(current_ratio)}x — strong liquidity")
        elif current_ratio >= 1.0:
            score += 0.0
            notes.append(f"Current ratio {fmt(current_ratio)}x — adequate liquidity")
        else:
            score -= 1.5
            notes.append(f"Current ratio {fmt(current_ratio)}x — liquidity stress")

    fcf = fund.get('free_cash_flow')
    market_cap = fund.get('market_cap')
    if fcf is not None and market_cap and market_cap > 0:
        fcf_yield = (fcf / market_cap) * 100
        if fcf_yield > 5:
            score += 1.0
            notes.append(f"FCF yield {fmt(fcf_yield)}% — strong cash generation")
        elif fcf < 0:
            score -= 1.0
            notes.append(f"Negative FCF — company burning cash")

    div_yield = fund.get('dividend_yield')
    if div_yield is not None and div_yield > 0:
        notes.append(f"Dividend yield {fmt(div_yield)}%")
        if div_yield >= p['DIVIDEND_YIELD_GOOD']:
            score += 0.5

    return max(0, min(10, score)), notes


def score_shareholding(fund: dict) -> tuple[float, list[str]]:
    """Score shareholding pattern 0-10."""
    score = 5.0
    notes = []
    p = FA_PARAMS

    promoter = fund.get('promoter_holding_pct')
    if promoter is not None:
        if promoter >= p['PROMOTER_HOLD_GOOD']:
            score += 2.0
            notes.append(f"Promoter holding {fmt(promoter)}% — high skin in the game")
        elif promoter >= p['PROMOTER_HOLD_WARN']:
            score += 0.0
            notes.append(f"Promoter holding {fmt(promoter)}%")
        else:
            score -= 2.0
            notes.append(f"Low promoter holding {fmt(promoter)}% — governance concern")

    fii = fund.get('institutional_holding_pct')
    if fii is not None:
        if fii >= p['FII_GOOD']:
            score += 1.5
            notes.append(f"Institutional holding {fmt(fii)}% — strong institutional confidence")
        elif fii < 5:
            score -= 0.5
            notes.append(f"Low institutional interest {fmt(fii)}%")

    return max(0, min(10, score)), notes


def compute_composite_fundamental_score(fund: dict) -> tuple[float, dict]:
    """
    Compute overall fundamental score using weighted sub-scores.
    
    Returns:
        (composite_score: float, breakdown: dict)
    """
    val_score, val_notes = score_valuation(fund)
    prof_score, prof_notes = score_profitability(fund)
    growth_score, growth_notes = score_growth(fund)
    health_score, health_notes = score_financial_health(fund)
    share_score, share_notes = score_shareholding(fund)

    # Weights
    weights = {
        'valuation': 0.25,
        'profitability': 0.25,
        'growth': 0.25,
        'health': 0.15,
        'shareholding': 0.10
    }

    composite = (
        val_score * weights['valuation'] +
        prof_score * weights['profitability'] +
        growth_score * weights['growth'] +
        health_score * weights['health'] +
        share_score * weights['shareholding']
    )

    breakdown = {
        'valuation': {'score': round(val_score, 2), 'notes': val_notes, 'weight': weights['valuation']},
        'profitability': {'score': round(prof_score, 2), 'notes': prof_notes, 'weight': weights['profitability']},
        'growth': {'score': round(growth_score, 2), 'notes': growth_notes, 'weight': weights['growth']},
        'health': {'score': round(health_score, 2), 'notes': health_notes, 'weight': weights['health']},
        'shareholding': {'score': round(share_score, 2), 'notes': share_notes, 'weight': weights['shareholding']},
    }

    return round(composite, 2), breakdown


def build_fundamental_prompt(symbol: str, fund: dict, breakdown: dict, peers: dict) -> str:
    """Build the detailed prompt for the LLM with all fundamental data."""
    
    mc_str = format_currency(fund.get('market_cap'))
    price = fmt(fund.get('current_price'), "₹", na="N/A")

    # Peer comparison string
    peer_lines = []
    for peer_sym, peer_data in peers.items():
        peer_lines.append(
            f"  {peer_sym}: P/E={fmt(peer_data.get('pe_ratio'))}x | "
            f"ROE={fmt(peer_data.get('roe'))}% | "
            f"Revenue Growth={fmt(peer_data.get('revenue_growth'))}%"
        )
    peer_str = "\n".join(peer_lines) if peer_lines else "  (No peer data available)"

    prompt = f"""
=== FUNDAMENTAL ANALYSIS REQUEST ===
Stock: {symbol}
Company: {fund.get('company_name', 'N/A')}
Sector: {fund.get('sector', 'N/A')} | Industry: {fund.get('industry', 'N/A')}
Current Price: {price} | Market Cap: {mc_str}

--- VALUATION METRICS ---
P/E Ratio (TTM):     {fmt(fund.get('pe_ratio'))}x
Forward P/E:         {fmt(fund.get('forward_pe'))}x
P/B Ratio:           {fmt(fund.get('pb_ratio'))}x
P/S Ratio:           {fmt(fund.get('ps_ratio'))}x
PEG Ratio:           {fmt(fund.get('peg_ratio'))}x
EV/EBITDA:           {fmt(fund.get('ev_ebitda'))}x
52W High:            {fmt(fund.get('52w_high'), '₹')}
52W Low:             {fmt(fund.get('52w_low'), '₹')}

--- PROFITABILITY ---
ROE:                 {fmt(fund.get('roe'), '%')}
ROCE:                {fmt(fund.get('roce'), '%')}
ROA:                 {fmt(fund.get('roa'), '%')}
Net Margin:          {fmt(fund.get('net_margin'), '%')}
Gross Margin:        {fmt(fund.get('gross_margin'), '%')}
Operating Margin:    {fmt(fund.get('operating_margin'), '%')}
EBITDA Margin:       {fmt(fund.get('ebitda_margin'), '%')}

--- GROWTH ---
Revenue Growth YoY:  {fmt(fund.get('revenue_growth'), '%')}
EPS (TTM):           {fmt(fund.get('eps_ttm'), '₹')}
EPS (Forward):       {fmt(fund.get('eps_forward'), '₹')}
EPS Growth:          {fmt(fund.get('eps_growth'), '%')}
Earnings Growth:     {fmt(fund.get('earnings_growth'), '%')}

--- BALANCE SHEET ---
D/E Ratio:           {fmt(fund.get('debt_equity'))}x
Current Ratio:       {fmt(fund.get('current_ratio'))}x
Quick Ratio:         {fmt(fund.get('quick_ratio'))}x
Total Cash:          {format_currency(fund.get('total_cash'))}
Total Debt:          {format_currency(fund.get('total_debt'))}
Free Cash Flow:      {format_currency(fund.get('free_cash_flow'))}

--- DIVIDENDS ---
Dividend Yield:      {fmt(fund.get('dividend_yield'), '%')}
Dividend Rate:       {fmt(fund.get('dividend_rate'), '₹')}
Payout Ratio:        {fmt(fund.get('payout_ratio'), '%')}

--- SHAREHOLDING ---
Promoter Holding:    {fmt(fund.get('promoter_holding_pct'), '%')}
Institutional:       {fmt(fund.get('institutional_holding_pct'), '%')}
Analyst Target:      {fmt(fund.get('analyst_target_price'), '₹')}  (n={fund.get('analyst_count', 'N/A')})

--- QUANTITATIVE SCORES (pre-computed) ---
Valuation Score:     {breakdown['valuation']['score']}/10
Profitability Score: {breakdown['profitability']['score']}/10
Growth Score:        {breakdown['growth']['score']}/10
Financial Health:    {breakdown['health']['score']}/10
Shareholding:        {breakdown['shareholding']['score']}/10
COMPOSITE:           {sum(v['score']*v['weight'] for v in breakdown.values()):.1f}/10

--- SECTOR PEERS (Indian Market) ---
{peer_str}

Based on ALL the above data, provide your fundamental analysis. Consider India-specific context.
"""
    return prompt


def run(symbol: str, print_output: bool = True) -> dict:
    """
    Run the Fundamental Bot analysis.
    
    Returns:
        {
            "bot": "fundamental",
            "symbol": str,
            "score": float,
            "text": str,
            "breakdown": dict,
            "fundamentals": dict
        }
    """
    if print_output:
        print(f"\n{'='*60}")
        print(f"📊  FUNDAMENTAL BOT — {symbol}")
        print('='*60)

    # Fetch data
    fund = fetch_fundamentals(symbol)
    if not fund or fund.get('error'):
        error_msg = fund.get('error', 'No data') if fund else 'No data'
        return {"bot": "fundamental", "symbol": symbol, "score": 5.0,
                "text": f"Could not fetch fundamental data: {error_msg}", "breakdown": {}, "fundamentals": {}}

    # Compute scores
    composite_score, breakdown = compute_composite_fundamental_score(fund)

    # Fetch peer data for comparison
    sector = fund.get('sector', '')
    peers = fetch_peer_data(symbol, sector)

    # Build LLM prompt
    prompt = build_fundamental_prompt(symbol, fund, breakdown, peers)

    # Call LLM
    def on_token(t):
        if print_output:
            print(t, end='', flush=True)

    if print_output:
        print()
    
    llm_response = stream_chat(SYSTEM_PROMPT, prompt, on_token=on_token)

    if print_output:
        print()

    # Extract score from LLM response
    llm_score = extract_score(llm_response, default=composite_score)
    # Blend: 60% LLM judgment, 40% quantitative
    final_score = round(0.6 * llm_score + 0.4 * composite_score, 2)

    result = {
        "bot": "fundamental",
        "symbol": symbol,
        "score": final_score,
        "quant_score": composite_score,
        "llm_score": llm_score,
        "text": llm_response,
        "breakdown": breakdown,
        "fundamentals": fund
    }

    save_analysis(symbol, "fundamental", final_score, llm_response, breakdown)
    return result
