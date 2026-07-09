# ============================================================
# pipeline/orchestrator.py  — FIXED VERSION
# ============================================================
# FIXES vs original:
#   1. --fast flag propagated: run_full_pipeline(fast_mode=True)
#      skips 10-bot debate entirely; uses 5-bot quant-only council
#      with fast prompts. ~8 min for 7 stocks instead of 7 hours.
#   2. GPS threshold passed as param to run_gps_filter — no monkey-patch
#   3. earnings_trend guard: no division when fund_scores < 2 entries
#   4. Risk bot error score = 3.0 penalty (not 5.0 neutral)
#   5. dead `from dataclasses import asdict` removed
#   6. StockVerdict rank assigned cleanly in a single pass
#   7. self._nifty_1d captured from MarketSnapshot and passed to
#      prediction_bot so market_beta signal is real (was always 0.0)
#   8. prediction_bot called after each council with stop/target
#   9. All bot chain analyze() calls pass fast_mode flag through
#  10. Per-stock wall clock logged so stalls are visible immediately
# ============================================================

import sys
import time
import json
import subprocess
import requests
import concurrent.futures as _cf
from pathlib import Path
from datetime import datetime
import pytz


def _safe_net(fn, timeout=12, default=None):
    """Call a live-data network function with a hard timeout; return default on hang."""
    with _cf.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn)
        try:
            return fut.result(timeout=timeout)
        except Exception:
            return default


def _restart_ollama_inline(label: str = ""):
    """
    Kill Ollama (already dead or hung) and restart it fresh.
    Called after a per-stock timeout so the zombie thread dies quickly
    and the next stock starts with a clean Ollama process and full RAM.
    """
    tag = f"[OLLAMA RESTART{' ' + label if label else ''}]"
    print(f"\n  {tag} Killing Ollama...", flush=True)
    subprocess.run(["taskkill", "/f", "/im", "ollama.exe"],
                   shell=True, capture_output=True)
    time.sleep(5)
    print(f"  {tag} Starting fresh Ollama...", flush=True)
    subprocess.Popen(["ollama", "serve"], shell=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Wait up to 30s for Ollama to be responsive
    from utils.llm import check_ollama, warmup_ollama
    for _i in range(10):
        time.sleep(3)
        if check_ollama():
            warmup_ollama()   # pre-load model; switches to 3b globally if OOM
            print(f"  {tag} Ready ✓\n", flush=True)
            return True
    print(f"  {tag} Not responding yet — continuing anyway\n", flush=True)
    return False

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import VERBOSE_DEBUG, ALLOW_INTERNET
from scanner import market_scanner, sector_scanner
from scanner.universe import SECTORS, PRIORITY_STOCKS
from scanner.sector_scanner import SectorResult
from scanner.stock_scanner import fast_scan_stock, score_to_verdict
from utils.market_data import fetch_price_history, fetch_fundamentals
from utils.live_data import get_live_quote, get_all_news, NSELive, FinnhubData
from utils.llm import extract_score
from memory.storage import (
    save_prices_csv, load_prices_csv, is_prices_fresh,
    save_fundamentals_json, load_fundamentals_json,
    save_news_text, load_news_text,
    save_bot_scores, load_bot_scores_history, get_score_trend,
    save_market_snapshot, save_council_session,
    save_master_excel, today_str, now_str
)
from memory.vector_store import get_vector_store

IST = pytz.timezone('Asia/Kolkata')


# ══════════════════════════════════════════════════════════════
# GROWTH PROBABILITY SCORE (GPS)
# ══════════════════════════════════════════════════════════════

def compute_gps(symbol: str, pm: dict, fund: dict,
                nifty_1m: float, sector_avg_1m: float) -> tuple:
    """
    Growth Probability Score (GPS) — 0 to 10.
    Only stocks scoring >= threshold get full council debate.
    """
    components = {}

    # 1. TREND SCORE
    above_50  = pm.get('above_50dma', False) or False
    above_200 = pm.get('above_200dma', False) or False
    c1w = pm.get('change_1w', 0) or 0
    c1m = pm.get('change_1m', 0) or 0

    trend = 5.0
    if above_50 and above_200: trend = 8.0
    elif above_50:              trend = 6.0
    elif above_200:             trend = 4.0
    else:                       trend = 2.0
    if c1w > 3:  trend += 1.0
    if c1m > 8:  trend += 1.0
    components['trend'] = round(min(10, max(0, trend)), 2)

    # 2. MOMENTUM SCORE
    rsi = pm.get('rsi', 50) or 50
    if   50 <= rsi <= 65: momentum = 9.0
    elif 65 <  rsi <= 70: momentum = 7.5
    elif 45 <= rsi < 50:  momentum = 6.0
    elif 40 <= rsi < 45:  momentum = 4.0
    elif 70 <  rsi:       momentum = 5.0
    elif 30 <= rsi < 40:  momentum = 3.0
    else:                 momentum = 1.5
    components['momentum'] = round(momentum, 2)

    # 3. RELATIVE STRENGTH
    vs_nifty  = (c1m - nifty_1m)
    vs_sector = (c1m - sector_avg_1m)
    rel = 5.0
    if vs_nifty > 5:    rel += 2.5
    elif vs_nifty > 2:  rel += 1.5
    elif vs_nifty > 0:  rel += 0.5
    elif vs_nifty < -5: rel -= 2.0
    elif vs_nifty < -2: rel -= 1.0
    if vs_sector > 3:   rel += 1.5
    elif vs_sector > 0: rel += 0.5
    elif vs_sector < -3: rel -= 1.0
    components['relative_strength'] = round(min(10, max(0, rel)), 2)

    # 4. VOLUME SCORE
    vol_ratio = pm.get('volume_ratio', 1.0) or 1.0
    if vol_ratio >= 2.0:   vol_score = 9.5
    elif vol_ratio >= 1.5: vol_score = 8.0
    elif vol_ratio >= 1.2: vol_score = 7.0
    elif vol_ratio >= 0.8: vol_score = 5.5
    elif vol_ratio >= 0.5: vol_score = 3.5
    else:                  vol_score = 2.0
    components['volume'] = round(vol_score, 2)

    # 5. FUNDAMENTAL QUALITY
    if fund and not fund.get('error'):
        fq  = 5.0
        roe = fund.get('roe') or 0
        de  = fund.get('debt_equity') or 1
        nm  = fund.get('net_margin') or 0
        rg  = fund.get('revenue_growth') or 0
        if roe >= 20: fq += 2.0
        elif roe >= 12: fq += 1.0
        elif roe < 5:   fq -= 1.5
        if de < 0.3: fq += 1.5
        elif de < 0.7: fq += 0.5
        elif de > 1.5: fq -= 2.0
        if nm >= 15: fq += 1.0
        elif nm >= 8: fq += 0.5
        elif nm < 3:  fq -= 1.0
        if rg >= 15: fq += 1.0
        elif rg < 0: fq -= 1.0
        components['fundamental_quality'] = round(min(10, max(0, fq)), 2)
    else:
        components['fundamental_quality'] = 5.0

    # 6. EARNINGS TREND — FIX: guard n_pairs < 1 to avoid zero-division
    score_history = load_bot_scores_history(symbol, days=30)
    if len(score_history) >= 3:
        fund_scores = [h['scores'].get('fundamental', 5) for h in score_history[-5:]]
        n_pairs = len(fund_scores) - 1
        if n_pairs >= 1:
            improving = sum(1 for i in range(1, len(fund_scores))
                           if fund_scores[i] > fund_scores[i-1])
            et = 5.0 + (improving / n_pairs - 0.5) * 8
        else:
            et = 5.0   # single data point — stay neutral
        components['earnings_trend'] = round(min(10, max(0, et)), 2)
    else:
        components['earnings_trend'] = 5.0

    weights = {
        'trend': 0.20, 'momentum': 0.20, 'relative_strength': 0.20,
        'volume': 0.15, 'fundamental_quality': 0.15, 'earnings_trend': 0.10,
    }
    gps = sum(components[k] * weights[k] for k in weights)
    return round(gps, 2), components


# ══════════════════════════════════════════════════════════════
# DATA TEXT BUILDERS
# ══════════════════════════════════════════════════════════════

def build_fundamental_data_text(symbol: str, fund: dict) -> str:
    from utils.llm import format_currency, fmt
    if not fund or fund.get('error'):
        return f"Fundamental data unavailable for {symbol}"
    return f"""
Company:         {fund.get('company_name', symbol)}
Sector:          {fund.get('sector', 'N/A')} | {fund.get('industry', 'N/A')}
Price:           ₹{fmt(fund.get('current_price'))}
Market Cap:      {format_currency(fund.get('market_cap'))}

VALUATION:
  P/E (TTM):     {fmt(fund.get('pe_ratio'))}x  |  Forward P/E: {fmt(fund.get('forward_pe'))}x
  P/B:           {fmt(fund.get('pb_ratio'))}x  |  EV/EBITDA: {fmt(fund.get('ev_ebitda'))}x

PROFITABILITY:
  ROE:           {fmt(fund.get('roe'))}%  |  ROCE: {fmt(fund.get('roce'))}%
  Net Margin:    {fmt(fund.get('net_margin'))}%  |  EBITDA Margin: {fmt(fund.get('ebitda_margin'))}%

GROWTH:
  Revenue YoY:   {fmt(fund.get('revenue_growth'))}%
  EPS Growth:    {fmt(fund.get('eps_growth'))}%

BALANCE SHEET:
  D/E Ratio:     {fmt(fund.get('debt_equity'))}x  |  Current Ratio: {fmt(fund.get('current_ratio'))}x

SHAREHOLDING:
  Promoter:      {fmt(fund.get('promoter_holding_pct'))}%
  Institutional: {fmt(fund.get('institutional_holding_pct'))}%

ANALYST:
  Target Price:  ₹{fmt(fund.get('analyst_target_price'))}
  Recommendation: {(fund.get('recommendation') or 'N/A').upper()}
""".strip()


def build_technical_data_text(symbol: str, ta: dict) -> str:
    from utils.llm import fmt
    if not ta:
        return f"Technical data unavailable for {symbol}"
    ich = ta.get('ichimoku', {})
    fib = ta.get('fibonacci', {})
    piv = ta.get('pivot_points', {})
    return f"""
Price:    ₹{fmt(ta.get('current_price'))}
1D: {ta.get('price_change_1d',0):+.2f}%  1W: {ta.get('price_change_1w',0):+.2f}%  1M: {ta.get('price_change_1m',0):+.2f}%

MOVING AVERAGES:
  SMA20: ₹{fmt(ta.get('sma20'))} ({'ABOVE' if ta.get('above_sma20') else 'BELOW'})
  SMA50: ₹{fmt(ta.get('sma50'))} ({'ABOVE' if ta.get('above_sma50') else 'BELOW'})
  SMA200:₹{fmt(ta.get('sma200'))} ({'ABOVE' if ta.get('above_sma200') else 'BELOW'})
  Golden Cross: {'YES' if ta.get('golden_cross') else 'NO'}

MOMENTUM:
  RSI(14):  {fmt(ta.get('rsi'))}  {'OB' if ta.get('rsi_overbought') else 'OS' if ta.get('rsi_oversold') else 'neutral'}
  MACD:     {fmt(ta.get('macd'))} vs Signal {fmt(ta.get('macd_signal'))}  {'BULLISH' if ta.get('macd_bullish') else 'BEARISH'}

VOLATILITY:
  ATR:      ₹{fmt(ta.get('atr'))} ({fmt(ta.get('atr_pct'))}%)
  BB %B:    {fmt(ta.get('bb_pct_b'))}  {'SQUEEZE' if ta.get('bb_squeeze') else ''}

TREND:
  ADX:      {fmt(ta.get('adx'))}  {'STRONG' if ta.get('strong_trend') else 'WEAK'}

VOLUME:
  Ratio:    {fmt(ta.get('volume_ratio'))}x avg

ICHIMOKU:
  Cloud:    {'ABOVE' if ich.get('price_above_cloud') else 'BELOW'}

PIVOT: R1:{piv.get('R1','?')} PP:{piv.get('PP','?')} S1:{piv.get('S1','?')}
""".strip()


def build_sentiment_data_text(symbol: str, fii_dii: dict,
                               options: dict, vix: dict, indices: list) -> str:
    fii = (fii_dii or {}).get('fii', {})
    dii = (fii_dii or {}).get('dii', {})
    idx_lines = '\n'.join(
        f"  {i.get('name','')}: {i.get('last','N/A')} ({i.get('pct_change',0):+.2f}%)"
        for i in (indices or [])[:5]
    )
    return f"""
FII Net:    ₹{fii.get('net_value','N/A')} Cr
DII Net:    ₹{dii.get('net_value','N/A')} Cr
PCR:        {(options or {}).get('pcr','N/A')}  Signal: {(options or {}).get('pcr_signal','N/A')}
India VIX:  {(vix or {}).get('vix','N/A')}  ({(vix or {}).get('level','N/A')})
Indices:
{idx_lines}
""".strip()


def build_risk_data_text(symbol: str, hv, beta, var_data, dd, sharpe, fund, liq) -> str:
    from utils.llm import fmt, format_currency
    return f"""
VOLATILITY:
  HV (1Y):    {fmt(hv)}%
  VaR 95%:    {fmt(var_data.get('var_95_hist') if var_data else None)}% daily
  Daily Vol:  {fmt(var_data.get('daily_vol_pct') if var_data else None)}%

MARKET RISK:
  Beta:       {fmt(beta)} vs Nifty
  Max DD:     {fmt(dd.get('max_drawdown_pct') if dd else None)}%

RETURNS:
  Sharpe:     {fmt(sharpe.get('sharpe_ratio') if sharpe else None)}
  Sortino:    {fmt(sharpe.get('sortino_ratio') if sharpe else None)}

BALANCE SHEET:
  D/E:        {fmt(fund.get('debt_equity') if fund else None)}x
  Current:    {fmt(fund.get('current_ratio') if fund else None)}x

LIQUIDITY:
  ADV:        ₹{fmt(liq.get('adv_value_cr') if liq else None)} Cr/day
  Promoter:   {fmt(fund.get('promoter_holding_pct') if fund else None)}%
""".strip()


# ══════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════

class MarketOrchestrator:
    """
    Runs the full unified pipeline.
    Pass fast_mode=True to run_full_pipeline() for --fast invocations.
    """

    def __init__(self):
        from trading_calendar import get_trading_date
        self.vector_store = get_vector_store()
        self.date         = get_trading_date()   # trading date, not calendar date
        self._nifty_1m    = 0.0
        self._nifty_1d    = 0.0   # live 1D % — fed into prediction_bot

    # ── LEVEL 1: MARKET ──────────────────────────────────────

    def run_market_scan(self, print_output: bool = True) -> tuple:
        if print_output:
            print(f"\n{'═'*65}")
            print("🌐  LEVEL 1 — MARKET SCAN")
            print('═'*65)

        snap = market_scanner.run(print_output=print_output)

        snap_dict = {
            'market_score':   snap.market_score,
            'market_outlook': snap.market_outlook,
            'fii_net_cr':     snap.fii_net_cr,
            'dii_net_cr':     snap.dii_net_cr,
            'fii_dii_signal': snap.fii_dii_signal,
            'vix':            snap.vix,
            'vix_signal':     snap.vix_signal,
            'advance_count':  snap.advance_count,
            'decline_count':  snap.decline_count,
            'ad_ratio':       snap.ad_ratio,
            'breadth_signal': snap.breadth_signal,
            'indices':        [{'name': i.name, 'last': i.last, 'pct_change': i.pct_change}
                               for i in snap.indices],
            'global_cues':    snap.global_cues,
            'llm_analysis':   snap.llm_analysis,
        }

        save_market_snapshot(snap_dict, self.date)
        self.vector_store.add_market_analysis(
            snap.llm_analysis, snap.market_score, snap.market_outlook, self.date
        )
        return snap, snap_dict

    # ── LEVEL 2: SECTOR SCAN ─────────────────────────────────

    def run_sector_scan(self, market_snap,
                        print_output: bool = True) -> tuple:
        if print_output:
            print(f"\n{'═'*65}")
            print("📂  LEVEL 2 — SECTOR SCAN")
            print('═'*65)

        # Get nifty 1M baseline AND live 1D %
        nifty_df = fetch_price_history("^NSEI", period="2mo")
        if nifty_df is not None and not nifty_df.empty and len(nifty_df) >= 22:
            c = nifty_df['Close']
            self._nifty_1m = round((c.iloc[-1]/c.iloc[-22]-1)*100, 2)
            if len(c) >= 2:
                self._nifty_1d = round((c.iloc[-1]/c.iloc[-2]-1)*100, 2)

        # Prefer live pct_change from already-fetched MarketSnapshot
        if market_snap and market_snap.indices:
            nifty_idx = next(
                (i for i in market_snap.indices
                 if "NIFTY 50" in (i.name or "").upper()), None
            )
            if nifty_idx and nifty_idx.pct_change is not None:
                self._nifty_1d = nifty_idx.pct_change

        sector_results = sector_scanner.run(
            market_snap, priority_only=True, print_output=print_output
        )

        stocks_per_sector = {}

        if print_output:
            print(f"\n{'─'*65}")
            print("  ⚡ Sector stock selection (quant)...")

        for sr in sector_results:
            analysis_text = f"Sector {sr.name}: Score {sr.score:.1f}/10 | {sr.verdict}"
            n_stocks = (
                8 if sr.score >= 8.0 else
                6 if sr.score >= 7.0 else
                4 if sr.score >= 6.0 else
                2 if sr.score >= 5.0 else 0
            )
            chosen_stocks = [t[0] for t in sr.top_stocks[:n_stocks]]
            stocks_per_sector[sr.name] = chosen_stocks

            if print_output:
                print(
                    f"  {sr.name:35} Score:{sr.score:.1f}  "
                    f"→ {len(chosen_stocks)} stocks: {', '.join(chosen_stocks[:6])}"
                )

            self.vector_store.add_sector_analysis(
                sr.name, analysis_text, sr.score, chosen_stocks, self.date
            )
            sr.llm_analysis = analysis_text

        stocks_per_sector = {k: v for k, v in stocks_per_sector.items() if v}
        total = sum(len(v) for v in stocks_per_sector.values())
        if print_output:
            print(f"\n  Total stocks entering filter: {total}")

        return sector_results, stocks_per_sector

    # ── LEVEL 3: GPS FILTER ──────────────────────────────────

    def run_gps_filter(self, stocks_per_sector: dict,
                        sector_results: list,
                        gps_threshold: float = 6.5,
                        print_output: bool = True) -> list:
        """
        FIX: gps_threshold is a proper parameter now (not monkey-patched).
        Compute GPS for every candidate stock; return those >= threshold.
        """
        if print_output:
            print(f"\n{'═'*65}")
            print(f"🎯  LEVEL 3 — GROWTH PROBABILITY FILTER (GPS ≥ {gps_threshold})")
            print('═'*65)

        sector_avg = {sr.name: sr.avg_1m_change or 0 for sr in sector_results}
        qualified  = []

        for sector, symbols in stocks_per_sector.items():
            sec_avg = sector_avg.get(sector, 0)
            if print_output:
                print(f"\n  [{sector}]")

            for sym in symbols:
                from scanner.sector_scanner import fetch_stock_quick_metrics
                pm   = fetch_stock_quick_metrics(sym, nifty_1m=self._nifty_1m)
                fund = load_fundamentals_json(sym)
                if fund is None and ALLOW_INTERNET:
                    fund = fetch_fundamentals(sym)
                    if fund and not fund.get('error'):
                        save_fundamentals_json(sym, fund)

                gps, components = compute_gps(sym, pm, fund or {}, self._nifty_1m, sec_avg)
                status = "✅ QUALIFIED" if gps >= gps_threshold else "❌ filtered"
                if print_output:
                    rsi = pm.get('rsi', 0) or 0
                    c1m = pm.get('change_1m', 0) or 0
                    print(f"    {sym:14} GPS:{gps:.1f}  RSI:{rsi:.0f}  1M:{c1m:+.1f}%  {status}")

                if gps >= gps_threshold:
                    qualified.append({
                        'symbol':       sym,
                        'sector':       sector,
                        'gps':          gps,
                        'components':   components,
                        'price_metrics': pm,
                        'fundamentals': fund or {},
                    })

        qualified.sort(key=lambda x: x['gps'], reverse=True)
        if print_output:
            print(f"\n  ✅ {len(qualified)} stocks qualified for council debate")
        return qualified

    # ── LEVEL 4: BOT COUNCIL DEBATE (ONE STOCK) ──────────────

    @staticmethod
    def _save_bot_cache(path: str, cache: dict):
        """Persist per-bot output so a restart can skip completed bots."""
        if not path:
            return
        try:
            with open(path, 'w') as _bcf:
                json.dump(cache, _bcf, default=str, indent=2)
        except Exception:
            pass  # non-fatal — worst case the bot reruns

    def run_council_for_stock(self, stock_info: dict,
                               market_snap_dict: dict,
                               print_output: bool = True,
                               fast_mode: bool = False,
                               bot_cache_path: str = None,
                               bot_cache: dict = None) -> dict:
        """
        Run full bot council debate for ONE stock.

        fast_mode=True: 5-bot chains with 60-word prompts (no 10-bot debate).
                        Target ~5 min/stock.
        fast_mode=False: 5-bot chains (full) + 10-bot cross-debate.
                         Target ~20 min/stock with good hardware.
        """
        from pipeline.chains import (
            get_fundamental_chain, get_technical_chain,
            get_news_chain, get_sentiment_chain,
            get_risk_chain
        )
        from bots.technical_bot import compute_all_indicators
        from bots.risk_bot import (
            calculate_historical_volatility, calculate_var,
            calculate_max_drawdown, calculate_sharpe_sortino,
            calculate_liquidity_risk
        )

        sym     = stock_info['symbol']
        sector  = stock_info['sector']
        fund    = stock_info['fundamentals']
        company = fund.get('company_name', sym)
        stock_start = time.time()

        if print_output:
            mode_label = "[FAST]" if fast_mode else "[FULL]"
            print(f"\n{'═'*65}")
            print(f"🏛  COUNCIL DEBATE {mode_label}: {sym} — {company}")
            print(f"    Sector: {sector} | GPS: {stock_info['gps']:.1f}/10")
            print('═'*65)

        # ── Price data ─────────────────────────────────────────
        # Always load from local CSV first (populated by nightly download).
        # Only fall back to Yahoo Finance if no local data exists at all.
        # This prevents empty technical indicators when Yahoo rate-limits.
        df = load_prices_csv(sym)
        if df is None or df.empty:
            df = fetch_price_history(sym)
            if df is not None and not df.empty:
                save_prices_csv(sym, df)

        # ── News ──────────────────────────────────────────────
        # days_back=2 handles overnight/late runs that cross midnight
        cached_news = load_news_text(sym, days_back=2)
        if not cached_news and ALLOW_INTERNET:
            articles = get_all_news(sym, company)
            if articles:
                save_news_text(sym, articles)
                news_text = '\n'.join(
                    f"{a.get('title','')} — {a.get('summary','')[:150]}"
                    for a in articles[:10]
                )
                avg_sent = sum(a.get('sentiment',0) for a in articles)/len(articles) if articles else 0
                self.vector_store.add_news(sym, news_text, avg_sent, self.date)
                cached_news = articles

        news_text_for_bot = '\n'.join(
            f"• {a.get('title','')} [{a.get('source','')}] sent:{a.get('sentiment',0):.2f}"
            for a in (cached_news or [])[:10]   # cap at 10 in fast mode
        ) or "No recent news found"

        # ── Live quote ────────────────────────────────────────
        quote = _safe_net(lambda: get_live_quote(sym), timeout=12, default={})
        price = quote.get('last_price') or quote.get('current') or fund.get('current_price')

        # ── Technical indicators ──────────────────────────────
        ta = {}
        if df is not None and not df.empty and len(df) >= 30:
            try:
                ta = compute_all_indicators(df)
            except Exception as e:
                if VERBOSE_DEBUG:
                    print(f"  [TA] Error: {e}")

        # ── Risk metrics ──────────────────────────────────────
        hv, beta, var_data, dd, sharpe, liq = None, None, {}, {}, {}, {}
        if df is not None and not df.empty:
            try:
                close  = df['Close']
                volume = df['Volume']
                hv       = calculate_historical_volatility(close)
                var_data = calculate_var(close.pct_change().dropna())
                dd       = calculate_max_drawdown(close)
                sharpe   = calculate_sharpe_sortino(close)
                liq      = calculate_liquidity_risk(volume, float(close.iloc[-1]))
                beta     = fund.get('beta')
            except Exception as e:
                if VERBOSE_DEBUG:
                    print(f"  [RISK] Error: {e}")

        # ── Sentiment data ────────────────────────────────────
        fii_dii = _safe_net(lambda: NSELive.get_fii_dii(), timeout=12, default={}) or {}
        options = _safe_net(lambda: NSELive.get_option_chain(sym), timeout=15, default={}) or {}
        from bots.sentiment_bot import fetch_india_vix
        vix_data = _safe_net(lambda: fetch_india_vix(), timeout=12, default={})
        indices  = _safe_net(lambda: NSELive.get_all_indices(), timeout=12, default=[]) or []

        # ── Build text inputs ─────────────────────────────────
        fund_text_input = build_fundamental_data_text(sym, fund)
        tech_text_input = build_technical_data_text(sym, ta)
        sent_text_input = build_sentiment_data_text(sym, fii_dii, options, vix_data, indices)
        risk_text_input = build_risk_data_text(sym, hv, beta, var_data, dd, sharpe, fund, liq)

        # ════════════════════════════════════════════════════
        # RUN ALL 5 BOT CHAINS (fast_mode flag propagated)
        # ════════════════════════════════════════════════════

        # ── initialise bot_cache dict if caller didn't provide one ──
        if bot_cache is None:
            bot_cache = {}

        # ── Bot 1: FUNDAMENTAL ────────────────────────────────
        if 'fundamental' in bot_cache:
            fund_analysis = bot_cache['fundamental']['text']
            fund_score    = bot_cache['fundamental']['score']
            if print_output:
                print(f"\n  🤖 Bot 1/5 — FUNDAMENTAL BOT  [cached ✓ score={fund_score:.1f}]")
        else:
            if print_output:
                print(f"\n  🤖 Bot 1/5 — FUNDAMENTAL BOT")
            fund_analysis = get_fundamental_chain().analyze(
                sym, sector, fund_text_input,
                self.vector_store, print_output=print_output,
                fast_mode=fast_mode
            )
            fund_score = extract_score(fund_analysis, default=5.0)
            bot_cache['fundamental'] = {'text': fund_analysis, 'score': fund_score}
            self._save_bot_cache(bot_cache_path, bot_cache)

        # ── Bot 2: TECHNICAL ──────────────────────────────────
        if 'technical' in bot_cache:
            tech_analysis = bot_cache['technical']['text']
            tech_score    = bot_cache['technical']['score']
            if print_output:
                print(f"\n  🤖 Bot 2/5 — TECHNICAL BOT    [cached ✓ score={tech_score:.1f}]")
        else:
            if print_output:
                print(f"\n  🤖 Bot 2/5 — TECHNICAL BOT")
            tech_analysis = get_technical_chain().analyze(
                sym, tech_text_input,
                self.vector_store, print_output=print_output,
                fast_mode=fast_mode
            )
            tech_score = extract_score(tech_analysis, default=5.0)
            bot_cache['technical'] = {'text': tech_analysis, 'score': tech_score}
            self._save_bot_cache(bot_cache_path, bot_cache)

        # ── Bot 3: NEWS ───────────────────────────────────────
        if 'news' in bot_cache:
            news_analysis = bot_cache['news']['text']
            news_score    = bot_cache['news']['score']
            if print_output:
                print(f"\n  🤖 Bot 3/5 — NEWS BOT         [cached ✓ score={news_score:.1f}]")
        else:
            if print_output:
                print(f"\n  🤖 Bot 3/5 — NEWS BOT")
            news_analysis = get_news_chain().analyze(
                sym, sector, news_text_for_bot,
                self.vector_store, print_output=print_output,
                fast_mode=fast_mode
            )
            news_score = extract_score(news_analysis, default=5.0)
            bot_cache['news'] = {'text': news_analysis, 'score': news_score}
            self._save_bot_cache(bot_cache_path, bot_cache)

        # ── Bot 4: SENTIMENT ──────────────────────────────────
        if 'sentiment' in bot_cache:
            sent_analysis = bot_cache['sentiment']['text']
            sent_score    = bot_cache['sentiment']['score']
            if print_output:
                print(f"\n  🤖 Bot 4/5 — SENTIMENT BOT    [cached ✓ score={sent_score:.1f}]")
        else:
            if print_output:
                print(f"\n  🤖 Bot 4/5 — SENTIMENT BOT")
            sent_analysis = get_sentiment_chain().analyze(
                sym, sent_text_input,
                self.vector_store, print_output=print_output,
                fast_mode=fast_mode
            )
            sent_score = extract_score(sent_analysis, default=5.0)
            bot_cache['sentiment'] = {'text': sent_analysis, 'score': sent_score}
            self._save_bot_cache(bot_cache_path, bot_cache)

        # ── Bot 5: RISK ───────────────────────────────────────
        if 'risk' in bot_cache:
            risk_analysis = bot_cache['risk']['text']
            risk_score    = bot_cache['risk']['score']
            if print_output:
                print(f"\n  🤖 Bot 5/5 — RISK BOT         [cached ✓ score={risk_score:.1f}]")
        else:
            if print_output:
                print(f"\n  🤖 Bot 5/5 — RISK BOT")
            try:
                risk_analysis = get_risk_chain().analyze(
                    sym, sector, risk_text_input,
                    self.vector_store, print_output=print_output,
                    fast_mode=fast_mode
                )
                risk_score = extract_score(risk_analysis, default=5.0)
            except Exception as e:
                print(f"  [RISK BOT] Error: {e} — applying 3.0 penalty score")
                risk_analysis = f"Risk analysis failed: {str(e)[:100]}"
                # FIX: use 3.0 penalty, NOT neutral 5.0 — broken risk should not boost stocks
                risk_score = 3.0
            bot_cache['risk'] = {'text': risk_analysis, 'score': risk_score}
            self._save_bot_cache(bot_cache_path, bot_cache)

        scores = {
            'fundamental': fund_score,
            'technical':   tech_score,
            'news':        news_score,
            'sentiment':   sent_score,
            'risk':        risk_score,
        }

        # ── 10-Bot Cross Debate (FULL mode only) ─────────────
        council_verdict = ""
        final_score     = 5.0
        verdict_label   = "HOLD"
        debate_stop     = None
        debate_target   = None

        if not fast_mode:
            from pipeline.debate_council import get_debate_council
            combined_data = f"""
FUNDAMENTAL: {fund_analysis[:400]}
TECHNICAL: {tech_analysis[:400]}
NEWS: {news_analysis[:300]}
SENTIMENT: {sent_text_input[:300]}
RISK: {risk_text_input[:300]}
"""
            debate = get_debate_council()
            debate_result = debate.run(
                sym, company, sector, price or 0,
                combined_data, print_output=print_output
            )
            council_verdict = debate_result.get('verdict_text', '')
            final_score     = debate_result.get('score', 5.0)
            verdict_label   = debate_result.get('verdict', 'HOLD')
            debate_stop     = debate_result.get('stop_loss')
            debate_target   = debate_result.get('target')
            scores['bull_avg'] = debate_result.get('bull_avg_score', 5.0)
            scores['bear_avg'] = debate_result.get('bear_avg_score', 5.0)
        else:
            # Fast mode: weighted composite without 10-bot debate
            from config import VERDICT_WEIGHTS
            w = VERDICT_WEIGHTS
            composite = (
                fund_score  * w.get('fundamental', 0.30) +
                tech_score  * w.get('technical',   0.25) +
                news_score  * w.get('news',        0.20) +
                sent_score  * w.get('sentiment',   0.15) +
                (10 - risk_score) * w.get('risk',  0.10)
            )
            final_score   = round(min(10, max(0, composite)), 2)
            verdict_label = score_to_verdict(final_score)
            council_verdict = (
                f"[FAST MODE] F:{fund_score:.1f} T:{tech_score:.1f} "
                f"N:{news_score:.1f} S:{sent_score:.1f} Risk:{risk_score:.1f} "
                f"→ {verdict_label} {final_score:.1f}/10"
            )

        elapsed_stock = round(time.time() - stock_start)
        if print_output:
            print(f"\n  {'─'*55}")
            print(f"  VERDICT:   {verdict_label}")
            print(f"  SCORE:     {final_score}/10")
            print(f"  F:{fund_score:.1f} T:{tech_score:.1f} "
                  f"N:{news_score:.1f} S:{sent_score:.1f} Risk:{risk_score:.1f}")
            print(f"  GPS:       {stock_info['gps']:.1f}/10")
            print(f"  Time:      {elapsed_stock}s for this stock")

        # ── Prediction bot (wired with live Nifty 1D + stop/target) ──
        try:
            from bots.prediction_bot import run as run_prediction
            run_prediction(
                sym, float(price or 0), scores, ta=ta,
                print_output=False,
                nifty_1d_pct=self._nifty_1d,
                stop_loss=debate_stop,
                target=debate_target,
                verdict=verdict_label,
            )
        except Exception as pred_err:
            if VERBOSE_DEBUG:
                print(f"  [PRED BOT] Skipped: {pred_err}")

        # ── Store ─────────────────────────────────────────────
        save_bot_scores(
            sym, self.date, scores, verdict_label, final_score,
            analysis_texts={
                'fundamental': fund_analysis,
                'technical':   tech_analysis,
                'news':        news_analysis,
                'sentiment':   sent_analysis,
                'risk':        risk_analysis,
                'council':     council_verdict,
            }
        )

        for bot_name, text, score in [
            ('fundamental', fund_analysis, fund_score),
            ('technical',   tech_analysis, tech_score),
            ('news',        news_analysis, news_score),
            ('sentiment',   sent_analysis, sent_score),
            ('risk',        risk_analysis, risk_score),
        ]:
            self.vector_store.add_analysis(
                sym, bot_name, text, score, self.date,
                extra_meta={'sector': sector, 'price': str(price or '')}
            )

        full_debate = '\n\n'.join([
            f"FUNDAMENTAL:\n{fund_analysis}",
            f"TECHNICAL:\n{tech_analysis}",
            f"NEWS:\n{news_analysis}",
            f"SENTIMENT:\n{sent_analysis}",
            f"RISK:\n{risk_analysis}",
            f"COUNCIL:\n{council_verdict}",
        ])
        self.vector_store.add_council_debate(
            sym, full_debate, verdict_label, final_score, self.date
        )

        return {
            'symbol':         sym,
            'company':        company,
            'sector':         sector,
            'price':          price,
            'gps':            stock_info['gps'],
            'verdict':        verdict_label,
            'final_score':    final_score,
            'composite':      final_score,
            'scores':         scores,
            'council_verdict': council_verdict,
            'stop_loss':      debate_stop,
            'target':         debate_target,
            'elapsed_s':      elapsed_stock,
            'analyses': {
                'fundamental': fund_analysis,
                'technical':   tech_analysis,
                'news':        news_analysis,
                'sentiment':   sent_analysis,
                'risk':        risk_analysis,
            }
        }

    # ── FULL PIPELINE ─────────────────────────────────────────

    def _warmup_ollama(self, print_output: bool = True):
        """
        Send a trivial prompt to force the model into RAM before any
        stock debate begins. If the primary model OOMs, switches
        config.OLLAMA_MODEL to qwen2.5:3b for the entire run.
        """
        from utils.llm import warmup_ollama
        warmup_ollama()   # handles OOM detection + 3b fallback + prints status

    def run_full_pipeline(self, print_output: bool = True,
                          fast_mode: bool = False,
                          gps_threshold: float = 6.5) -> dict:
        """
        Run the complete unified pipeline end-to-end.

        Args:
            fast_mode     : Skip 10-bot debate; use fast chain prompts.
                            For --fast flag. ~5 min/stock vs ~20 min.
            gps_threshold : GPS cutoff (default 6.5). Passed as param
                            not monkey-patch — fixes --gps override bug.
        """
        start = time.time()
        from trading_calendar import get_trading_date
        self.date = get_trading_date()   # trading date (e.g. Jul 3 when run on Jul 4)

        mode_label = "⚡ FAST MODE" if fast_mode else "🔬 FULL MODE"
        print(f"\n{'═'*65}")
        print(f"🚀  UNIFIED MARKET COUNCIL — {mode_label}")
        print(f"    {datetime.now(IST).strftime('%d %b %Y %H:%M IST')}")
        if fast_mode:
            print("    --fast: 5-bot quant chains, no 10-bot debate")
            print("    Expected: ~5 min/stock on CPU")
        print('═'*65)

        # L1: Market
        market_snap, market_snap_dict = self.run_market_scan(print_output)

        # L2: Sectors
        sector_results, stocks_per_sector = self.run_sector_scan(
            market_snap, print_output
        )

        # L3: GPS filter — threshold now a proper parameter
        qualified = self.run_gps_filter(
            stocks_per_sector, sector_results,
            gps_threshold=gps_threshold,
            print_output=print_output
        )

        if not qualified:
            print("\n  ⚠ No stocks passed GPS filter today.")
            print(f"  Threshold was {gps_threshold} — try --gps 5.5 to lower it.")
            return {'market': market_snap_dict, 'sectors': [], 'stocks': []}

        # L4: Council debates
        if print_output:
            print(f"\n{'═'*65}")
            print(f"🏛  LEVEL 4 — BOT COUNCIL ({len(qualified)} stocks, {mode_label})")
            est_min = len(qualified) * (5 if fast_mode else 20)
            print(f"    Estimated time: ~{est_min} min")
            print('═'*65)

        # Warm up the model once before the loop so all stocks run
        # against an already-loaded model (avoids 10-min cold-load hangs)
        self._warmup_ollama(print_output=print_output)

        import concurrent.futures
        STOCK_TIMEOUT_S = 15 * 60   # 15 min hard ceiling per stock

        # ── CHECKPOINT: resume from crash without re-debating done stocks ──
        _chk_path = Path(__file__).parent.parent / 'data' / f'council_checkpoint_{self.date}.json'
        _checkpoint = {}
        if _chk_path.exists():
            try:
                with open(_chk_path) as _f:
                    _checkpoint = json.load(_f)
                if _checkpoint and print_output:
                    print(f"\n  [CHECKPOINT] Resuming run — {len(_checkpoint)} stocks already done: "
                          f"{', '.join(_checkpoint.keys())}")
            except Exception as _ce:
                print(f"  [CHECKPOINT] Could not load checkpoint: {_ce} — starting fresh")
                _checkpoint = {}

        all_results = list(_checkpoint.values())   # pre-fill with already-done stocks

        for i, stock_info in enumerate(qualified):
            sym = stock_info['symbol']

            # Skip stocks already completed in a prior attempt this session
            if sym in _checkpoint:
                if print_output:
                    r = _checkpoint[sym]
                    print(f"\n  [{i+1}/{len(qualified)}] {sym} — ✅ checkpoint ({r.get('verdict','?')} {r.get('final_score',0):.1f})")
                continue

            remaining = sum(1 for s in qualified[i:] if s['symbol'] not in _checkpoint)
            if print_output:
                print(f"\n  [{i+1}/{len(qualified)}] Debating: {sym} "
                      f"({remaining} remaining)")

            # ── PER-BOT cache: resume from a partially-done stock ──
            _bot_cache_path = str(
                Path(__file__).parent.parent / 'data' / f'bot_cache_{self.date}_{sym}.json'
            )
            _bot_cache: dict = {}
            if Path(_bot_cache_path).exists():
                try:
                    with open(_bot_cache_path) as _bcf:
                        _bot_cache = json.load(_bcf)
                    _done_bots = [k for k in _bot_cache
                                  if k in ('fundamental','technical','news','sentiment','risk')]
                    if _done_bots and print_output:
                        print(f"  [BOT CACHE] {sym}: {len(_done_bots)}/5 bots already done "
                              f"({', '.join(_done_bots)}) — skipping those")
                except Exception:
                    _bot_cache = {}


            _stock_timed_out = False
            _stock_crashed   = False
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _ex:
                _fut = _ex.submit(
                    self.run_council_for_stock,
                    stock_info, market_snap_dict,
                    print_output=print_output,
                    fast_mode=fast_mode,
                    bot_cache_path=_bot_cache_path,
                    bot_cache=_bot_cache,
                )
                try:
                    result = _fut.result(timeout=STOCK_TIMEOUT_S)
                except concurrent.futures.TimeoutError:
                    # ── Kill Ollama NOW so the zombie thread gets a connection
                    #    error and dies quickly — this lets shutdown(wait=True)
                    #    return in seconds instead of blocking for 12+ minutes
                    #    (which is what caused multiple parallel Phase 3 instances
                    #    and the OOM crashes). ──
                    _done_bots_now = [k for k in _bot_cache
                                      if k in ('fundamental','technical','news','sentiment','risk')]
                    print(f"\n  ⚠️  {sym}: hard timeout {STOCK_TIMEOUT_S//60}m — "
                          f"{len(_done_bots_now)}/5 bots cached — killing Ollama to unblock thread",
                          flush=True)
                    subprocess.run(["taskkill", "/f", "/im", "ollama.exe"],
                                   shell=True, capture_output=True)
                    _stock_timed_out = True
                    # Fall through — with-block __exit__ calls shutdown(wait=True)
                    # which now returns quickly because the thread gets a conn error
                except Exception as _e:
                    # Ollama crash (WinError 10054, ConnectionError, etc.)
                    _done_bots_now = [k for k in _bot_cache
                                      if k in ('fundamental','technical','news','sentiment','risk')]
                    print(f"\n  ⚠️  {sym}: LLM crash ({_e.__class__.__name__}) — "
                          f"{len(_done_bots_now)}/5 bots cached",
                          flush=True)
                    _stock_crashed = True

            # ── OUTSIDE the with block (thread is definitely dead now) ──
            if _stock_timed_out or _stock_crashed:
                # Always restart Ollama before next stock:
                # - timeout: Ollama was killed above, must restart
                # - crash: Ollama may have died (watchdog kill, OOM, WinError) — restart to be safe
                _restart_ollama_inline(sym)
                continue   # ← safe: no zombie threads alive at this point

            # ── Only reached on SUCCESS (no TimeoutError, no LLM crash) ────
            # Save to checkpoint immediately so a crash/restart skips this stock
            _checkpoint[sym] = result
            try:
                with open(_chk_path, 'w') as _f:
                    json.dump(_checkpoint, _f, default=str, indent=2)
            except Exception as _ce:
                print(f"  [CHECKPOINT] Warning: could not save: {_ce}")

            all_results.append(result)

        all_results.sort(key=lambda x: x['final_score'], reverse=True)

        # L5: Save
        if print_output:
            print(f"\n{'─'*65}")
            print("  💾 Saving all data...")

        session_path = save_council_session(self.date, [], sector_results)

        from scanner.stock_scanner import StockVerdict
        stock_verdicts = []
        for rank_i, r in enumerate(all_results, start=1):
            v = StockVerdict(
                symbol=r['symbol'],
                sector=r['sector'],
                company_name=r.get('company', r['symbol']),
                current_price=r.get('price'),
                verdict=r['verdict'],
                final_score=r['final_score'],
                rank=rank_i,
                fundamental_score=r['scores'].get('fundamental', 5),
                technical_score=r['scores'].get('technical', 5),
                news_score=r['scores'].get('news', 5),
                sentiment_score=r['scores'].get('sentiment', 5),
                risk_score=r['scores'].get('risk', 5),
                llm_analysis=r.get('council_verdict', ''),
            )
            stock_verdicts.append(v)

        excel_path = save_master_excel(
            stock_verdicts, sector_results, market_snap_dict, self.date
        )

        elapsed      = round(time.time() - start)
        mins, secs   = divmod(elapsed, 60)
        total_llm_s  = sum(r.get('elapsed_s', 0) for r in all_results)

        if print_output:
            print(f"\n{'═'*65}")
            print(f"✅  PIPELINE COMPLETE  ({mins}m {secs}s total)")
            print(f"    LLM time: {round(total_llm_s/60)}m | "
                  f"Overhead: {round((elapsed-total_llm_s)/60)}m")
            if excel_path:
                print(f"  📊 Excel:   {excel_path}")
            print(f"\n  🏆 TOP STOCKS TODAY:")
            for r in all_results[:10]:
                v = r.get('verdict', '?')
                sc = r.get('final_score', 0)
                gps = r.get('gps', 0)
                sym = r.get('symbol', '?')
                sec = r.get('sector', '')[:18]
                print(f"    #{all_results.index(r)+1:<2} {sym:<12} {sec:<18} Score:{sc:.1f}  GPS:{gps:.1f}  {v}")
        print(f"\n  Phase 3: {len(all_results)} stocks debated in {mins}m {secs}s")
        return {
            'stocks':  all_results,
            'market':  market_snap_dict,
            'sectors': sector_results,
        }
