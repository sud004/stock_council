# ============================================================
# utils/market_data.py — Fetch prices & fundamentals
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DEFAULT_EXCHANGE_SUFFIX, HISTORICAL_PERIOD, HISTORICAL_INTERVAL,
    NIFTY_SYMBOL, SENSEX_SYMBOL, NIFTY_BANK_SYMBOL,
    CACHE_TTL_PRICE_HOURS, CACHE_TTL_FUNDAMENTALS_HOURS,
    ALLOW_INTERNET, VERBOSE_DEBUG
)
from utils.database import (
    save_prices, load_prices,
    save_fundamentals, load_fundamentals
)


def resolve_symbol(symbol: str) -> str:
    """
    Normalize a stock symbol for Yahoo Finance.
    Uses symbol_mapper to handle NSE→Yahoo differences.
    Examples:
        RELIANCE     → RELIANCE.NS
        BERGERPAINTS → BERGEPAINT.NS  (corrected)
        NIITTECH     → COFORGE.NS     (renamed)
        GREENKO      → None           (not listed)
        ^NSEI        → ^NSEI
    """
    symbol = symbol.strip().upper()
    if symbol.startswith("^"):
        return symbol
    if symbol.endswith(".NS") or symbol.endswith(".BO"):
        return symbol
    # Use mapper for clean NSE → Yahoo conversion
    try:
        from utils.symbol_mapper import nse_to_yahoo
        mapped = nse_to_yahoo(symbol)
        if mapped:
            return mapped
        # Return with .NS anyway so caller gets a proper 404
        return symbol + DEFAULT_EXCHANGE_SUFFIX
    except Exception:
        return symbol + DEFAULT_EXCHANGE_SUFFIX


def fetch_price_history(symbol: str, period: str = None, interval: str = None) -> pd.DataFrame:
    """
    Fetch OHLCV price history from Yahoo Finance.
    Falls back to SQLite cache when offline.

    Returns a DataFrame with columns:
        Open, High, Low, Close, Volume
    indexed by date.
    """
    period = period or HISTORICAL_PERIOD
    interval = interval or HISTORICAL_INTERVAL
    yf_symbol = resolve_symbol(symbol)

    # Try cache first
    cached = load_prices(yf_symbol, max_age_hours=CACHE_TTL_PRICE_HOURS)
    if cached:
        if VERBOSE_DEBUG:
            print(f"[DATA] Using cached prices for {yf_symbol} ({len(cached)} rows)")
        df = pd.DataFrame(cached)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df.columns = [c.capitalize() for c in df.columns if c != 'date']
        # Rename to standard OHLCV
        col_map = {'Open':'Open','High':'High','Low':'Low','Close':'Close','Volume':'Volume'}
        df = df.rename(columns={c.lower(): c for c in ['Open','High','Low','Close','Volume']
                                  if c.lower() in df.columns})
        return df

    if not ALLOW_INTERNET:
        print(f"[DATA] Offline mode — no cached data for {yf_symbol}")
        return pd.DataFrame()

    try:
        if VERBOSE_DEBUG:
            print(f"[DATA] Fetching {yf_symbol} from Yahoo Finance ({period}, {interval})")
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=True)
        if df.empty:
            print(f"[DATA] No data returned for {yf_symbol}")
            return pd.DataFrame()

        df.index = pd.to_datetime(df.index)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.dropna(inplace=True)

        # Save to cache
        save_prices(yf_symbol, df)
        return df

    except Exception as e:
        print(f"[DATA] Error fetching price history for {yf_symbol}: {e}")
        return pd.DataFrame()


def fetch_fundamentals(symbol: str) -> dict:
    """
    Fetch fundamental / company info from Yahoo Finance.

    Returns a dict with all fundamental metrics we need:
    - Valuation: pe_ratio, pb_ratio, ev_ebitda, market_cap
    - Profitability: roe, roce, net_margin, gross_margin, operating_margin
    - Growth: revenue_growth, earnings_growth, eps_growth
    - Balance sheet: debt_equity, current_ratio, quick_ratio
    - Income: revenue, net_income, ebitda, eps
    - Dividends: dividend_yield, dividend_rate
    - Shareholding: promoter_holding (approximate via insiders)
    - Meta: sector, industry, company_name, currency
    """
    yf_symbol = resolve_symbol(symbol)

    # Try cache
    cached = load_fundamentals(yf_symbol, max_age_hours=CACHE_TTL_FUNDAMENTALS_HOURS)
    if cached:
        if VERBOSE_DEBUG:
            print(f"[DATA] Using cached fundamentals for {yf_symbol}")
        return cached

    if not ALLOW_INTERNET:
        print(f"[DATA] Offline mode — no cached fundamentals for {yf_symbol}")
        return {}

    try:
        if VERBOSE_DEBUG:
            print(f"[DATA] Fetching fundamentals for {yf_symbol}")
        ticker = yf.Ticker(yf_symbol)
        info = ticker.info

        def safe(key, default=None):
            val = info.get(key)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return default
            return val

        # ── Quarterly financial statements ─────────────────────
        try:
            income_stmt = ticker.quarterly_financials
            balance_sheet = ticker.quarterly_balance_sheet
            cashflow = ticker.quarterly_cashflow
        except Exception:
            income_stmt = pd.DataFrame()
            balance_sheet = pd.DataFrame()
            cashflow = pd.DataFrame()

        # ── Revenue growth (last 4Q vs prior 4Q) ──────────────
        revenue_growth = _calc_revenue_growth(income_stmt)
        eps_growth = _calc_eps_growth(ticker)
        roce = _calc_roce(income_stmt, balance_sheet)
        promoter_holding = _estimate_promoter_holding(ticker, info)
        free_cash_flow = _calc_free_cash_flow(cashflow, income_stmt)

        fundamentals = {
            # Identity
            "symbol": yf_symbol,
            "company_name": safe("longName", safe("shortName", symbol)),
            "sector": safe("sector", "Unknown"),
            "industry": safe("industry", "Unknown"),
            "currency": safe("currency", "INR"),
            "exchange": safe("exchange", "NSI"),

            # Valuation
            "pe_ratio": safe("trailingPE"),
            "forward_pe": safe("forwardPE"),
            "pb_ratio": safe("priceToBook"),
            "ps_ratio": safe("priceToSalesTrailing12Months"),
            "peg_ratio": safe("pegRatio"),
            "ev_ebitda": safe("enterpriseToEbitda"),
            "ev_revenue": safe("enterpriseToRevenue"),
            "market_cap": safe("marketCap"),
            "enterprise_value": safe("enterpriseValue"),

            # Price
            "current_price": safe("currentPrice", safe("regularMarketPrice")),
            "52w_high": safe("fiftyTwoWeekHigh"),
            "52w_low": safe("fiftyTwoWeekLow"),
            "beta": safe("beta"),

            # Profitability
            "roe": _pct(safe("returnOnEquity")),
            "roa": _pct(safe("returnOnAssets")),
            "roce": roce,
            "net_margin": _pct(safe("profitMargins")),
            "gross_margin": _pct(safe("grossMargins")),
            "operating_margin": _pct(safe("operatingMargins")),
            "ebitda_margin": _pct(safe("ebitdaMargins")),

            # Growth
            "revenue_growth": revenue_growth,
            "earnings_growth": _pct(safe("earningsGrowth")),
            "earnings_quarterly_growth": _pct(safe("earningsQuarterlyGrowth")),
            "eps_ttm": safe("trailingEps"),
            "eps_forward": safe("forwardEps"),
            "eps_growth": eps_growth,

            # Balance Sheet
            "debt_equity": safe("debtToEquity"),
            "current_ratio": safe("currentRatio"),
            "quick_ratio": safe("quickRatio"),
            "total_cash": safe("totalCash"),
            "total_debt": safe("totalDebt"),
            "book_value_per_share": safe("bookValue"),

            # Income
            "revenue": safe("totalRevenue"),
            "net_income": safe("netIncomeToCommon"),
            "gross_profit": safe("grossProfits"),
            "ebitda": safe("ebitda"),
            "free_cash_flow": free_cash_flow,
            "operating_cash_flow": safe("operatingCashflow"),

            # Dividends
            "dividend_yield": _pct(safe("dividendYield")),
            "dividend_rate": safe("dividendRate"),
            "payout_ratio": _pct(safe("payoutRatio")),
            "ex_dividend_date": str(safe("exDividendDate", "")),

            # Shares
            "shares_outstanding": safe("sharesOutstanding"),
            "float_shares": safe("floatShares"),
            "shares_short": safe("sharesShort"),
            "short_ratio": safe("shortRatio"),

            # Institutional / Promoter (approximate)
            "institutional_holding_pct": _pct(safe("heldPercentInstitutions")),
            "insider_holding_pct": _pct(safe("heldPercentInsiders")),
            "promoter_holding_pct": promoter_holding,

            # Analyst
            "recommendation": safe("recommendationKey", "none"),
            "analyst_target_price": safe("targetMeanPrice"),
            "analyst_count": safe("numberOfAnalystOpinions"),

            # Fetched
            "fetched_at": datetime.now().isoformat()
        }

        save_fundamentals(yf_symbol, fundamentals)
        return fundamentals

    except Exception as e:
        print(f"[DATA] Error fetching fundamentals for {yf_symbol}: {e}")
        return {"symbol": yf_symbol, "error": str(e)}


# ── Helper Calculators ─────────────────────────────────────────

def _pct(val):
    """Convert 0.15 → 15.0 (percentage). Leave None as None."""
    if val is None:
        return None
    try:
        return round(float(val) * 100, 2)
    except Exception:
        return None


def _calc_revenue_growth(income_stmt: pd.DataFrame) -> float | None:
    """YoY revenue growth % from quarterly income statement."""
    try:
        if income_stmt.empty:
            return None
        rev_row = income_stmt.loc['Total Revenue'] if 'Total Revenue' in income_stmt.index else None
        if rev_row is None:
            return None
        vals = rev_row.dropna().values
        if len(vals) >= 5:
            recent = sum(vals[:4])
            prior = sum(vals[4:8]) if len(vals) >= 8 else vals[4]
            if prior and prior != 0:
                return round((recent - prior) / abs(prior) * 100, 2)
    except Exception:
        pass
    return None


def _calc_eps_growth(ticker) -> float | None:
    """EPS growth from earnings history."""
    try:
        hist = ticker.earnings_history
        if hist is not None and not hist.empty and 'epsActual' in hist.columns:
            eps_vals = hist['epsActual'].dropna().values
            if len(eps_vals) >= 2:
                recent, prior = eps_vals[0], eps_vals[-1]
                if prior and prior != 0:
                    return round((recent - prior) / abs(prior) * 100, 2)
    except Exception:
        pass
    return None


def _calc_roce(income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame) -> float | None:
    """ROCE = EBIT / Capital Employed × 100"""
    try:
        if income_stmt.empty or balance_sheet.empty:
            return None
        ebit = None
        for key in ['EBIT', 'Operating Income', 'Operating Profit']:
            if key in income_stmt.index:
                ebit = income_stmt.loc[key].iloc[0]
                break
        if ebit is None:
            return None
        total_assets = None
        current_liab = None
        for key in ['Total Assets', 'TotalAssets']:
            if key in balance_sheet.index:
                total_assets = balance_sheet.loc[key].iloc[0]
                break
        for key in ['Current Liabilities', 'CurrentLiabilities', 'Total Current Liabilities']:
            if key in balance_sheet.index:
                current_liab = balance_sheet.loc[key].iloc[0]
                break
        if total_assets is None:
            return None
        capital_employed = total_assets - (current_liab or 0)
        if capital_employed and capital_employed != 0:
            return round(float(ebit) / float(capital_employed) * 100, 2)
    except Exception:
        pass
    return None


def _calc_free_cash_flow(cashflow: pd.DataFrame, income_stmt: pd.DataFrame) -> float | None:
    """FCF = Operating Cash Flow - Capital Expenditure"""
    try:
        if cashflow.empty:
            return None
        ocf = None
        capex = None
        for key in ['Operating Cash Flow', 'Total Cash From Operating Activities']:
            if key in cashflow.index:
                ocf = cashflow.loc[key].iloc[0]
                break
        for key in ['Capital Expenditure', 'Capital Expenditures', 'Purchase Of PPE']:
            if key in cashflow.index:
                capex = cashflow.loc[key].iloc[0]
                break
        if ocf is not None and capex is not None:
            return float(ocf) + float(capex)   # capex is usually negative
        return float(ocf) if ocf is not None else None
    except Exception:
        pass
    return None


def _estimate_promoter_holding(ticker, info: dict) -> float | None:
    """
    For Indian stocks, 'insiders' on Yahoo Finance roughly approximates promoters.
    Returns percentage.
    """
    try:
        insider_pct = info.get('heldPercentInsiders')
        if insider_pct is not None and not np.isnan(insider_pct):
            return round(float(insider_pct) * 100, 2)
    except Exception:
        pass
    return None


def fetch_nifty_data() -> pd.DataFrame:
    """Fetch Nifty 50 index data for beta/relative performance."""
    return fetch_price_history(NIFTY_SYMBOL, period=HISTORICAL_PERIOD)


def fetch_peer_data(symbol: str, sector: str) -> dict:
    """
    Fetch a small set of sector peers for comparison.
    Returns dict of {symbol: fundamentals}.
    """
    # Known Indian sector peers
    SECTOR_PEERS = {
        "Technology": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
        "Financial Services": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS"],
        "Consumer Cyclical": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "HEROMOTOCO.NS"],
        "Energy": ["RELIANCE.NS", "ONGC.NS", "IOC.NS", "BPCL.NS"],
        "Consumer Defensive": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS"],
        "Healthcare": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS"],
        "Basic Materials": ["JSWSTEEL.NS", "TATASTEEL.NS", "HINDALCO.NS", "VEDL.NS"],
        "Industrials": ["LT.NS", "SIEMENS.NS", "ABB.NS", "BEL.NS"],
        "Utilities": ["NTPC.NS", "POWERGRID.NS", "ADANIGREEN.NS"],
        "Communication Services": ["BHARTIARTL.NS", "IDEA.NS"],
        "Real Estate": ["DLF.NS", "GODREJPROP.NS", "PRESTIGE.NS"],
    }
    peers = SECTOR_PEERS.get(sector, [])
    yf_sym = resolve_symbol(symbol)
    peers = [p for p in peers if p != yf_sym][:4]   # max 4 peers
    result = {}
    for p in peers:
        data = fetch_fundamentals(p)
        if data and not data.get('error'):
            result[p] = data
    return result
