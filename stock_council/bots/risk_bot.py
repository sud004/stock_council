# ============================================================
# bots/risk_bot.py — Risk Assessment Bot
# ============================================================
#
# RISK FORMULAS:
#
# 1. VOLATILITY RISK
#    Historical Volatility (HV) = σ(daily log returns) × √252 × 100
#    Log Return = ln(Close_t / Close_{t-1})
#    Risk: HV > 40% = HIGH | 20-40% = MEDIUM | < 20% = LOW
#
# 2. BETA (Market Risk)
#    Beta = Cov(Stock, Market) / Var(Market)
#    Cov(Stock, Market) = Σ[(r_stock - μ_stock)(r_market - μ_market)] / (n-1)
#    Beta > 1.5 = HIGH | 1.0-1.5 = MEDIUM | < 1.0 = LOW market risk
#
# 3. VALUE AT RISK (VaR)
#    Historical VaR (95%) = 5th percentile of daily returns × √holding_period
#    Parametric VaR = μ - 1.645 × σ   (for 95% confidence)
#    CVaR (Expected Shortfall) = Mean of returns below VaR threshold
#
# 4. MAXIMUM DRAWDOWN
#    Peak = max(Close, running)
#    Drawdown_t = (Close_t - Peak_t) / Peak_t × 100
#    Max Drawdown = min(Drawdown_t)
#    Risk: MDD > 50% = EXTREME | 30-50% = HIGH | 15-30% = MEDIUM | < 15% = LOW
#
# 5. SHARPE RATIO (Risk-adjusted return)
#    Sharpe = (Annualized Return - Risk Free Rate) / HV
#    Risk Free Rate = 6.5% (India 10Y G-Sec approximate)
#    Sharpe > 1.5 = GREAT | 1.0-1.5 = GOOD | 0.5-1.0 = FAIR | < 0.5 = POOR
#
# 6. SORTINO RATIO (Downside risk)
#    Downside Deviation = σ(negative returns only) × √252
#    Sortino = (Annualized Return - Risk Free Rate) / Downside Deviation
#
# 7. DEBT RISK
#    Interest Coverage = EBIT / Interest Expense
#    ICR < 2 = DANGEROUS | 2-4 = RISKY | 4-8 = OKAY | > 8 = SAFE
#    D/E > 2.0 = HIGH | 1.0-2.0 = MEDIUM | 0.3-1.0 = LOW | < 0.3 = MINIMAL
#
# 8. PROMOTER PLEDGE RISK
#    Pledge % = Pledged Shares / Promoter Shares × 100
#    > 50% = EXTREME | 30-50% = HIGH | 10-30% = MEDIUM | < 10% = LOW
#    (Pledged shares can be sold forcibly by lenders if price falls)
#
# 9. LIQUIDITY RISK
#    Avg Daily Volume (ADV) = Mean(Volume, 20 days)
#    Market Impact Cost = (Price × Lot Size) / ADV
#    Low volume = high impact cost = harder to exit
#
# 10. CONCENTRATION RISK (Sector)
#    High-risk sectors get penalty score
#
# COMPOSITE RISK SCORE (0-10, HIGHER = MORE RISK):
#    Volatility:        0.20
#    Beta/Market risk:  0.15
#    Drawdown:          0.15
#    Debt risk:         0.20
#    Promoter pledge:   0.15
#    Liquidity:         0.10
#    Governance/Sector: 0.05
# ============================================================

import sys
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.market_data import fetch_price_history, fetch_fundamentals, fetch_nifty_data
from utils.llm import stream_chat, extract_score, fmt
from utils.database import save_analysis
from config import RISK_WEIGHTS, HIGH_RISK_SECTORS, DEFENSIVE_SECTORS, VERBOSE_DEBUG


RISK_FREE_RATE = 0.065   # 6.5% — India 10Y G-Sec (approximate)

SYSTEM_PROMPT = """You are RISK BOT — a conservative Indian stock market risk analyst and devil's advocate.
Your job is to find what can go WRONG with this investment.

You assess:
1. Quantitative risks: volatility, beta, drawdown, VaR
2. Balance sheet: debt levels, interest coverage, promoter pledging
3. Business risks: competitive threats, margin pressure, customer concentration
4. India-specific: SEBI actions, RBI policy sensitivity, regulatory risk
5. Global risks: FII flight risk, dollar strengthening, crude oil
6. Governance: related-party transactions, audit qualifications, management integrity

Your response MUST:
- Be 150-200 words
- List the TOP 3 specific risks for this stock RIGHT NOW
- Mention quantitative risk metrics (VaR, drawdown, beta)
- Be honest — do NOT sugarcoat risks
- End with exactly: "RISK SCORE: X/10" where X is the RISK LEVEL (10 = extreme risk, 1 = very safe)

Note: RISK SCORE is different from other bots.
  1-2: Very safe, low risk stock
  3-4: Below-average risk
  5-6: Average market risk
  7-8: Above-average risk, caution needed
  9-10: Extremely risky, potential capital loss
"""


def calculate_historical_volatility(close: pd.Series, period: int = 252) -> float:
    """
    HV = σ(log returns) × √252 × 100
    Annualized historical volatility as percentage.
    """
    log_returns = np.log(close / close.shift(1)).dropna()
    if len(log_returns) < 20:
        return None
    hv = log_returns.tail(period).std() * np.sqrt(252) * 100
    return round(float(hv), 2)


def calculate_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
    """
    Beta = Cov(Stock, Market) / Var(Market)
    """
    # Align indices
    aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
    if len(aligned) < 30:
        return None
    aligned.columns = ['stock', 'market']
    cov = aligned.cov()
    beta = cov.loc['stock', 'market'] / cov.loc['market', 'market']
    return round(float(beta), 3)


def calculate_var(returns: pd.Series, confidence: float = 0.95) -> dict:
    """
    Historical VaR and CVaR at 95% confidence.
    Returns daily figures.
    """
    clean = returns.dropna()
    if len(clean) < 30:
        return {}

    # Historical VaR (non-parametric)
    hist_var = np.percentile(clean, (1 - confidence) * 100)

    # Parametric VaR (assumes normality)
    mu = clean.mean()
    sigma = clean.std()
    param_var = mu - 1.645 * sigma  # 95% one-tailed

    # CVaR (Expected Shortfall) = mean of losses beyond VaR
    cvar = clean[clean <= hist_var].mean()

    return {
        'var_95_hist': round(float(hist_var) * 100, 2),  # as percentage
        'var_95_param': round(float(param_var) * 100, 2),
        'cvar_95': round(float(cvar) * 100, 2) if not np.isnan(cvar) else None,
        'daily_vol_pct': round(float(sigma) * 100, 2),
        'mean_daily_return': round(float(mu) * 100, 4),
    }


def calculate_max_drawdown(close: pd.Series) -> dict:
    """
    Max Drawdown = (Trough - Peak) / Peak × 100
    Also calculates drawdown duration.
    """
    rolling_max = close.cummax()
    drawdown = (close - rolling_max) / rolling_max * 100
    max_dd = float(drawdown.min())

    # Find duration of worst drawdown
    peak_idx = close[:drawdown.idxmin()].idxmax() if len(close) > 0 else None
    trough_idx = drawdown.idxmin()

    # Current drawdown from ATH
    current_dd = float(drawdown.iloc[-1])

    return {
        'max_drawdown_pct': round(max_dd, 2),
        'current_drawdown_pct': round(current_dd, 2),
        'peak_price': round(float(close.max()), 2),
        'current_price': round(float(close.iloc[-1]), 2),
        'recovery_needed_pct': round(-max_dd / (1 + max_dd / 100) * 100, 2) if max_dd < 0 else 0,
    }


def calculate_sharpe_sortino(close: pd.Series, risk_free: float = RISK_FREE_RATE) -> dict:
    """
    Sharpe = (Ann Return - Rf) / HV
    Sortino = (Ann Return - Rf) / Downside Deviation
    """
    returns = close.pct_change().dropna()
    if len(returns) < 30:
        return {}

    ann_return = (1 + returns.mean()) ** 252 - 1
    hv = returns.std() * np.sqrt(252)
    sharpe = (ann_return - risk_free) / hv if hv > 0 else None

    # Sortino
    negative_returns = returns[returns < 0]
    downside_dev = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 5 else hv
    sortino = (ann_return - risk_free) / downside_dev if downside_dev > 0 else None

    return {
        'annual_return_pct': round(float(ann_return) * 100, 2),
        'sharpe_ratio': round(float(sharpe), 3) if sharpe else None,
        'sortino_ratio': round(float(sortino), 3) if sortino else None,
        'hv_pct': round(float(hv) * 100, 2),
    }


def calculate_liquidity_risk(volume: pd.Series, price: float) -> dict:
    """
    Liquidity risk based on average daily volume and market impact.
    """
    adv = float(volume.tail(20).mean())  # 20-day ADV
    adv_value = adv * price  # in rupees

    # Days to liquidate ₹1 Cr position (assuming 10% of ADV max)
    position_size = 1e7  # ₹1 Cr
    max_daily_exit = adv_value * 0.10  # 10% of ADV
    days_to_exit = position_size / max_daily_exit if max_daily_exit > 0 else 999

    return {
        'adv_shares': round(adv),
        'adv_value_cr': round(adv_value / 1e7, 2),
        'days_to_exit_1cr': round(days_to_exit, 1),
        'liquidity_level': (
            'HIGH' if adv_value > 1e8    # > ₹10 Cr/day
            else 'MEDIUM' if adv_value > 1e7  # ₹1-10 Cr/day
            else 'LOW'
        )
    }


def score_volatility_risk(hv: float) -> tuple[float, str]:
    """Higher HV = higher risk score."""
    if hv is None:
        return 5.0, "Volatility data unavailable"
    if hv < 15:
        return 2.0, f"HV {hv:.1f}% — very low volatility, stable stock"
    elif hv < 25:
        return 4.0, f"HV {hv:.1f}% — below-average volatility"
    elif hv < 35:
        return 6.0, f"HV {hv:.1f}% — average market volatility"
    elif hv < 50:
        return 7.5, f"HV {hv:.1f}% — HIGH volatility, larger position risk"
    else:
        return 9.0, f"HV {hv:.1f}% — EXTREME volatility, speculative stock"


def score_beta_risk(beta: float) -> tuple[float, str]:
    if beta is None:
        return 5.0, "Beta unavailable"
    if beta < 0:
        return 3.0, f"Beta {beta:.2f} — negative beta (defensive/counter-cyclical)"
    elif beta < 0.7:
        return 2.0, f"Beta {beta:.2f} — low market risk"
    elif beta < 1.0:
        return 4.0, f"Beta {beta:.2f} — slightly defensive"
    elif beta < 1.3:
        return 5.5, f"Beta {beta:.2f} — in-line with market"
    elif beta < 1.7:
        return 7.0, f"Beta {beta:.2f} — high sensitivity to market moves"
    else:
        return 9.0, f"Beta {beta:.2f} — very high market risk"


def score_drawdown_risk(dd: dict) -> tuple[float, str]:
    mdd = dd.get('max_drawdown_pct', 0)
    curr = dd.get('current_drawdown_pct', 0)
    if mdd >= 0:
        return 5.0, "Drawdown data insufficient"
    if mdd > -15:
        score = 3.0
        note = f"Max drawdown {mdd:.1f}% — resilient stock"
    elif mdd > -30:
        score = 5.5
        note = f"Max drawdown {mdd:.1f}% — moderate historical loss"
    elif mdd > -50:
        score = 7.5
        note = f"Max drawdown {mdd:.1f}% — HIGH historical loss potential"
    else:
        score = 9.0
        note = f"Max drawdown {mdd:.1f}% — EXTREME historical crash"

    if curr < -20:
        score += 0.5
        note += f" | Currently {curr:.1f}% below peak"
    return round(score, 2), note


def score_debt_risk(fund: dict) -> tuple[float, str]:
    de = fund.get('debt_equity')
    current = fund.get('current_ratio')
    score = 5.0
    notes = []

    if de is not None:
        if de < 0.3:
            score -= 2.0
            notes.append(f"D/E {de:.2f}x — very low debt, safe")
        elif de < 0.7:
            score -= 1.0
            notes.append(f"D/E {de:.2f}x — manageable debt")
        elif de < 1.5:
            score += 1.0
            notes.append(f"D/E {de:.2f}x — moderate leverage")
        elif de < 2.5:
            score += 2.5
            notes.append(f"D/E {de:.2f}x — HIGH leverage risk")
        else:
            score += 4.0
            notes.append(f"D/E {de:.2f}x — DANGEROUS leverage")

    if current is not None:
        if current < 0.8:
            score += 2.0
            notes.append(f"Current ratio {current:.2f}x — liquidity stress")
        elif current < 1.2:
            score += 0.5
            notes.append(f"Current ratio {current:.2f}x — borderline")
        else:
            score -= 0.5
            notes.append(f"Current ratio {current:.2f}x — adequate")

    return max(0, min(10, round(score, 2))), ' | '.join(notes) or "Debt data unavailable"


def score_promoter_pledge_risk(fund: dict) -> tuple[float, str]:
    """
    Promoter pledging is a major India-specific risk.
    If stock falls, lenders can sell pledged shares = downward spiral.
    """
    # Yahoo Finance doesn't directly give pledge data.
    # We use promoter holding as proxy (low holding = higher governance risk).
    promoter = fund.get('promoter_holding_pct')
    if promoter is None:
        return 5.0, "Promoter pledge data unavailable (check BSE disclosures manually)"

    if promoter > 65:
        score = 2.0
        note = f"Promoter holding {promoter:.1f}% — high conviction, low exit risk"
    elif promoter > 50:
        score = 3.5
        note = f"Promoter holding {promoter:.1f}% — strong promoter commitment"
    elif promoter > 35:
        score = 5.0
        note = f"Promoter holding {promoter:.1f}% — moderate promoter stake"
    elif promoter > 20:
        score = 7.0
        note = f"Promoter holding {promoter:.1f}% — LOW promoter stake, governance risk"
    else:
        score = 9.0
        note = f"Promoter holding {promoter:.1f}% — VERY LOW, major governance red flag"

    return round(score, 2), note


def score_liquidity_risk(liq: dict) -> tuple[float, str]:
    if not liq:
        return 5.0, "Liquidity data unavailable"
    level = liq.get('liquidity_level', 'MEDIUM')
    adv = liq.get('adv_value_cr', 0)
    days = liq.get('days_to_exit_1cr', 0)

    if level == 'HIGH':
        return 2.0, f"High liquidity: ₹{adv:.1f} Cr/day ADV, exit ₹1Cr in {days:.1f} days"
    elif level == 'MEDIUM':
        return 5.0, f"Medium liquidity: ₹{adv:.1f} Cr/day ADV"
    else:
        return 8.0, f"LOW liquidity: ₹{adv:.1f} Cr/day — hard to exit large positions"


def score_sector_governance_risk(fund: dict) -> tuple[float, str]:
    sector = fund.get('sector', '')
    industry = fund.get('industry', '')

    if any(s.lower() in sector.lower() for s in HIGH_RISK_SECTORS):
        return 7.5, f"HIGH-RISK sector: {sector} — regulatory exposure, capex-heavy"
    elif any(s.lower() in sector.lower() for s in DEFENSIVE_SECTORS):
        return 3.0, f"DEFENSIVE sector: {sector} — stable demand, lower systemic risk"
    else:
        return 5.0, f"Sector: {sector} — average regulatory risk"


def compute_composite_risk_score(scores_dict: dict) -> tuple[float, dict]:
    """
    Weighted composite risk score.
    Unlike other bots, HIGHER = MORE RISK.
    """
    weights = RISK_WEIGHTS

    composite = (
        scores_dict['volatility'][0] * weights.get('volatility', 0.20) +
        scores_dict['beta'][0] * weights.get('beta', 0.15) +
        scores_dict['drawdown'][0] * weights.get('drawdown', 0.15) +
        scores_dict['debt'][0] * weights.get('debt_risk', 0.20) +
        scores_dict['pledge'][0] * weights.get('promoter_pledge', 0.15) +
        scores_dict['liquidity'][0] * weights.get('liquidity_risk', 0.10) +
        scores_dict['sector'][0] * weights.get('sector_risk', 0.05)
    )

    breakdown = {k: {'score': v[0], 'note': v[1]} for k, v in scores_dict.items()}
    return round(composite, 2), breakdown


def build_risk_prompt(symbol: str, fund: dict, hv: float, beta: float,
                       var_data: dict, dd: dict, sharpe: dict,
                       liq: dict, breakdown: dict) -> str:

    prompt = f"""
=== RISK ASSESSMENT REQUEST ===
Stock: {symbol}
Company: {fund.get('company_name', 'N/A')}
Sector: {fund.get('sector', 'N/A')} | Industry: {fund.get('industry', 'N/A')}

--- VOLATILITY METRICS ---
Historical Volatility (1Y): {fmt(hv)}%
VaR 95% Daily (Historical): {fmt(var_data.get('var_95_hist'))}%
VaR 95% Daily (Parametric): {fmt(var_data.get('var_95_param'))}%
CVaR 95% (Expected Loss):   {fmt(var_data.get('cvar_95'))}%
Daily Volatility:            {fmt(var_data.get('daily_vol_pct'))}%

--- MARKET RISK ---
Beta (vs Nifty 50):     {fmt(beta)}
Interpretation:         {'High sensitivity' if beta and beta > 1.3 else 'Low sensitivity' if beta and beta < 0.8 else 'In-line with market'}

--- DRAWDOWN ANALYSIS ---
Maximum Drawdown:       {fmt(dd.get('max_drawdown_pct'))}%
Current vs ATH:         {fmt(dd.get('current_drawdown_pct'))}%
All-Time High:          ₹{fmt(dd.get('peak_price'))}
Current Price:          ₹{fmt(dd.get('current_price'))}
Recovery Needed:        {fmt(dd.get('recovery_needed_pct'))}% to reach ATH

--- RISK-ADJUSTED RETURNS ---
1Y Annualized Return:   {fmt(sharpe.get('annual_return_pct'))}%
Sharpe Ratio:           {fmt(sharpe.get('sharpe_ratio'))}
Sortino Ratio:          {fmt(sharpe.get('sortino_ratio'))}
Interpretation:         {'Excellent' if sharpe.get('sharpe_ratio') and sharpe['sharpe_ratio'] > 1.5 else 'Good' if sharpe.get('sharpe_ratio') and sharpe['sharpe_ratio'] > 1.0 else 'Poor risk-return'}

--- BALANCE SHEET RISK ---
Debt/Equity:            {fmt(fund.get('debt_equity'))}x
Current Ratio:          {fmt(fund.get('current_ratio'))}x
Quick Ratio:            {fmt(fund.get('quick_ratio'))}x
Total Debt:             ₹{fmt(fund.get('total_debt'))}
Free Cash Flow:         ₹{fmt(fund.get('free_cash_flow'))}

--- PROMOTER / GOVERNANCE ---
Promoter Holding:       {fmt(fund.get('promoter_holding_pct'))}%
Institutional Holding:  {fmt(fund.get('institutional_holding_pct'))}%
Short Ratio:            {fmt(fund.get('short_ratio'))}
Analyst Recommendation: {fund.get('recommendation', 'N/A').upper()}

--- LIQUIDITY ---
ADV Value:              ₹{fmt(liq.get('adv_value_cr'))} Cr/day
Liquidity Level:        {liq.get('liquidity_level', 'N/A')}
Exit ₹1Cr position in:  {fmt(liq.get('days_to_exit_1cr'))} trading days

--- RISK BREAKDOWN SCORES (0-10, higher = more risk) ---
Volatility Risk:    {breakdown.get('volatility', {}).get('score', 'N/A')}/10
Beta/Market Risk:   {breakdown.get('beta', {}).get('score', 'N/A')}/10
Drawdown Risk:      {breakdown.get('drawdown', {}).get('score', 'N/A')}/10
Debt Risk:          {breakdown.get('debt', {}).get('score', 'N/A')}/10
Governance Risk:    {breakdown.get('pledge', {}).get('score', 'N/A')}/10
Liquidity Risk:     {breakdown.get('liquidity', {}).get('score', 'N/A')}/10
Sector Risk:        {breakdown.get('sector', {}).get('score', 'N/A')}/10

Identify the top 3 specific risks for this stock. Be thorough and conservative.
Remember: your score is RISK LEVEL (10=extremely risky, 1=very safe).
"""
    return prompt


def run(symbol: str, print_output: bool = True) -> dict:
    """Run the Risk Bot analysis."""
    if print_output:
        print(f"\n{'='*60}")
        print(f"⚠️   RISK BOT — {symbol}")
        print('='*60)

    # Fetch price data and fundamentals
    df = fetch_price_history(symbol)
    fund = fetch_fundamentals(symbol) or {}

    if df.empty:
        return {"bot": "risk", "symbol": symbol, "score": 5.0,
                "text": "Could not fetch price data for risk analysis.", "breakdown": {}}

    close = df['Close']
    volume = df['Volume']
    price = float(close.iloc[-1])
    returns = close.pct_change().dropna()

    # Calculate all risk metrics
    hv = calculate_historical_volatility(close)
    var_data = calculate_var(returns)
    dd = calculate_max_drawdown(close)
    sharpe = calculate_sharpe_sortino(close)
    liq = calculate_liquidity_risk(volume, price)

    # Beta vs Nifty
    beta = fund.get('beta')
    if beta is None:
        nifty_df = fetch_nifty_data()
        if not nifty_df.empty:
            stock_ret = close.pct_change().dropna()
            nifty_ret = nifty_df['Close'].pct_change().dropna()
            beta = calculate_beta(stock_ret, nifty_ret)

    # Score each risk dimension
    scores_dict = {
        'volatility': score_volatility_risk(hv),
        'beta': score_beta_risk(beta),
        'drawdown': score_drawdown_risk(dd),
        'debt': score_debt_risk(fund),
        'pledge': score_promoter_pledge_risk(fund),
        'liquidity': score_liquidity_risk(liq),
        'sector': score_sector_governance_risk(fund),
    }

    quant_risk_score, breakdown = compute_composite_risk_score(scores_dict)

    # Build prompt
    prompt = build_risk_prompt(symbol, fund, hv, beta, var_data, dd, sharpe, liq, breakdown)

    def on_token(t):
        if print_output:
            print(t, end='', flush=True)

    if print_output:
        print()

    llm_response = stream_chat(SYSTEM_PROMPT, prompt, on_token=on_token)

    if print_output:
        print()

    llm_score = extract_score(llm_response, default=quant_risk_score)
    final_score = round(0.5 * llm_score + 0.5 * quant_risk_score, 2)

    result = {
        "bot": "risk",
        "symbol": symbol,
        "score": final_score,           # RISK level (higher = more dangerous)
        "quant_score": quant_risk_score,
        "llm_score": llm_score,
        "text": llm_response,
        "breakdown": breakdown,
        "metrics": {
            "hv": hv, "beta": beta,
            "var_95": var_data, "max_drawdown": dd,
            "sharpe": sharpe, "liquidity": liq
        }
    }

    save_analysis(symbol, "risk", final_score, llm_response, breakdown)
    return result
