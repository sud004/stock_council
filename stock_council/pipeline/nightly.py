# ============================================================
# pipeline/nightly.py — Nightly Data Download Job
# ============================================================

import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import pytz

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Load .env and show key status at startup ──────────────────
from dotenv import load_dotenv
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(_env_path)

def _show_startup_status():
    """Print .env load status and all API key status at startup."""
    IST_tz = pytz.timezone('Asia/Kolkata')
    ts = datetime.now(IST_tz).strftime('%H:%M:%S')

    print(f"\n{'═'*55}")
    print(f"  🌙 NIGHTLY JOB STARTUP — {datetime.now(IST_tz).strftime('%d %b %Y %H:%M IST')}")
    print(f"{'═'*55}")

    # .env file status
    if _env_path.exists():
        print(f"  [.env]    ✅ Loaded from {_env_path}")
    else:
        print(f"  [.env]    ⚠️  Not found at {_env_path}")
        print(f"            → Copy .env.example to .env and fill in keys")

    print(f"\n  [API KEYS]")

    # Check each key
    keys = {
        "Finnhub":       os.getenv("FINNHUB_KEY", ""),
        "Alpha Vantage": os.getenv("ALPHA_VANTAGE_KEY", ""),
        "NewsAPI":       os.getenv("NEWSAPI_KEY", ""),
        "GNews":         os.getenv("GNEWS_KEY", ""),
        "EaseAPI":       os.getenv("EASEAPI_APP_KEY", ""),
        "Reddit":        os.getenv("REDDIT_CLIENT_ID", ""),
    }

    for name, val in keys.items():
        is_set = bool(val) and not val.startswith("your_") and val != ""
        if is_set:
            # Show first 6 chars only for security
            masked = val[:6] + "..." + val[-4:] if len(val) > 10 else val[:4] + "..."
            print(f"             {name:15} ✅ set  ({masked})")
        else:
            fallback = {
                "Finnhub":       "Google News RSS",
                "Alpha Vantage": "Yahoo Finance",
                "NewsAPI":       "Google News RSS",
                "GNews":         "Google News RSS",
                "EaseAPI":       "Yahoo Finance + NSE scraper",
                "Reddit":        "skipped",
            }.get(name, "fallback")
            print(f"             {name:15} ❌ not set → using {fallback}")

    # Ollama model
    model = os.getenv("OLLAMA_MODEL", "mistral")
    print(f"\n  [OLLAMA]   Model: {model}")
    print(f"  [INTERNET] {'✅ enabled' if os.getenv('ALLOW_INTERNET','true') == 'true' else '❌ offline mode'}")
    print(f"{'═'*55}\n")

# Show status immediately when module loads
_show_startup_status()

from scanner.universe import get_universe, generate_error_summary, validate_and_correct_symbol, SECTORS, STOCK_TO_SECTOR
from utils.market_data import fetch_price_history, fetch_fundamentals, resolve_symbol
from memory.storage import (
    save_prices_csv, load_prices_csv, is_prices_fresh,
    save_fundamentals_json, load_fundamentals_json,
    save_news_text, load_news_text,
    save_master_excel, get_storage_summary, cleanup_old_files,
    export_all_prices_excel, today_str, PRICES_DIR, FUND_DIR
)
from memory.vector_store import get_vector_store
from config import ALLOW_INTERNET, VERBOSE_DEBUG

IST = pytz.timezone('Asia/Kolkata')


def log(msg: str):
    ts = datetime.now(IST).strftime('%H:%M:%S')
    print(f"[{ts}] {msg}")


# ══════════════════════════════════════════════════════════════
# 1. PRICES
# ══════════════════════════════════════════════════════════════

def download_all_prices(symbols: list = None, force: bool = False) -> dict:
    symbols = symbols or all_stocks
    log(f"📈 Downloading prices for {len(symbols)} stocks...")
    results = {}

    with tqdm(symbols, desc="Prices", unit="stock") as pbar:
        for sym in pbar:
            pbar.set_postfix(sym=sym)
            if not force and is_prices_fresh(sym, max_age_hours=20):
                results[sym] = 'skipped'
                continue
            if not ALLOW_INTERNET:
                results[sym] = 'offline'
                continue
            try:
                df = fetch_price_history(sym, period="2y", interval="1d")
                if df is not None and not df.empty:
                    save_prices_csv(sym, df)
                    results[sym] = 'ok'
                else:
                    results[sym] = 'no_data'
            except Exception as e:
                results[sym] = f'error'
            time.sleep(0.3)

    ok = sum(1 for v in results.values() if v == 'ok')
    skipped = sum(1 for v in results.values() if v == 'skipped')
    errors = sum(1 for v in results.values() if v.startswith('error') or v == 'no_data')
    log(f"  ✓ Prices: {ok} downloaded, {skipped} skipped, {errors} errors")
    return results


# ══════════════════════════════════════════════════════════════
# 2. FUNDAMENTALS
# ══════════════════════════════════════════════════════════════

def download_all_fundamentals(symbols: list = None, force: bool = False) -> dict:
    symbols = symbols or all_stocks
    log(f"📊 Downloading fundamentals for {len(symbols)} stocks...")
    results = {}

    with tqdm(symbols, desc="Fundamentals", unit="stock") as pbar:
        for sym in pbar:
            pbar.set_postfix(sym=sym)
            if not force:
                cached = load_fundamentals_json(sym, max_age_hours=24)
                if cached:
                    results[sym] = 'skipped'
                    continue
            if not ALLOW_INTERNET:
                results[sym] = 'offline'
                continue
            try:
                fund = fetch_fundamentals(sym)
                if fund and not fund.get('error'):
                    save_fundamentals_json(sym, fund)
                    results[sym] = 'ok'
                else:
                    results[sym] = 'no_data'
            except Exception as e:
                results[sym] = 'error'
            time.sleep(0.5)

    ok = sum(1 for v in results.values() if v == 'ok')
    skipped = sum(1 for v in results.values() if v == 'skipped')
    log(f"  ✓ Fundamentals: {ok} downloaded, {skipped} skipped")
    return results


# ══════════════════════════════════════════════════════════════
# 3. NEWS — FAST METHOD using Google News RSS
# ══════════════════════════════════════════════════════════════

def fetch_news_fast(symbol: str, company_name: str = "") -> list:
    """
    Fast news fetch using Google News RSS — no API key, very reliable.
    Returns list of article dicts in under 2 seconds per stock.
    """
    import requests
    import feedparser
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    vader = SentimentIntensityAnalyzer()
    articles = []
    sym_clean = symbol.replace('.NS', '').replace('.BO', '').upper()

    # Build search query — company name is more accurate than symbol
    query = company_name if company_name and len(company_name) > 3 else sym_clean
    # Remove common suffixes that confuse news search
    query = query.replace(' Limited', '').replace(' Ltd', '').replace(' Ltd.', '').strip()

    # Google News RSS — fastest and most reliable free source
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(query + ' NSE stock India')}&hl=en-IN&gl=IN&ceid=IN:en"

    try:
        feed = feedparser.parse(url)
        for entry in feed.entries[:8]:   # max 8 articles per stock
            title = entry.get('title', '')
            summary = entry.get('summary', '')

            # Clean HTML tags
            import re
            summary = re.sub(r'<[^>]+>', '', summary).strip()

            # Score sentiment
            text = title + ' ' + summary
            sent = vader.polarity_scores(text).get('compound', 0.0)

            articles.append({
                'title': title,
                'summary': summary[:300],
                'source': 'Google News',
                'url': entry.get('link', ''),
                'published': entry.get('published', ''),
                'sentiment': round(sent, 4),
                'relevance': 3,
                'symbol': symbol,
            })
    except Exception as e:
        if VERBOSE_DEBUG:
            print(f"[NEWS] Google RSS error for {sym_clean}: {e}")

    return articles


def download_all_news(symbols: list = None) -> dict:
    """
    Download news for all stocks using Google News RSS.
    Fast: ~1-2 seconds per stock = ~10 minutes for all 286.
    """
    symbols = symbols or all_stocks
    log(f"📰 Downloading news for {len(symbols)} stocks (Google News RSS)...")
    results = {}
    total_articles = 0

    with tqdm(symbols, desc="News", unit="stock") as pbar:
        for sym in pbar:
            pbar.set_postfix(sym=sym)

            # Skip if we already have good news within last 2 days
            # (days_back=2 handles late/overnight runs that cross midnight)
            existing = load_news_text(sym, days_back=2)
            if existing and len(existing) >= 5:
                results[sym] = 'skipped'
                continue

            if not ALLOW_INTERNET:
                results[sym] = 'offline'
                continue

            try:
                # Get company name from cached fundamentals
                fund = load_fundamentals_json(sym)
                company = (fund or {}).get('company_name', sym)

                articles = fetch_news_fast(sym, company)

                if articles:
                    save_news_text(sym, articles)
                    total_articles += len(articles)
                    results[sym] = f'ok:{len(articles)}'
                else:
                    results[sym] = 'no_news'

            except Exception as e:
                results[sym] = 'error'

            # Small delay to be polite to Google
            time.sleep(1.0)

    ok = sum(1 for v in results.values() if v.startswith('ok'))
    log(f"  ✓ News: {ok} stocks, {total_articles} total articles")
    return results


# ══════════════════════════════════════════════════════════════
# 4. VECTOR INDEX
# ══════════════════════════════════════════════════════════════

def build_vector_index(force_rebuild: bool = False) -> dict:
    log("🧠 Building vector index...")
    vs = get_vector_store()
    count = vs.index_all_stored_data()
    stats = vs.get_stats()
    log(f"  ✓ Vector index: {count} documents")
    return stats


# ══════════════════════════════════════════════════════════════
# 5. NIGHTLY EXCEL
# ══════════════════════════════════════════════════════════════

def generate_nightly_excel() -> Path | None:
    log("📊 Generating nightly Excel report...")
    import pandas as pd
    from memory.storage import EXCEL_DIR

    all_rows = []
    all_stocks_list, stock_to_sector_map = get_universe()
    for sym in (all_stocks_list or []):
        fund = load_fundamentals_json(sym)
        if not fund or fund.get('error'):
            continue

        df = load_prices_csv(sym, days=90)
        c1d = c1w = c1m = None
        if df is not None and not df.empty:
            close = df['Close']
            if len(close) >= 2:  c1d = round((close.iloc[-1]/close.iloc[-2]-1)*100, 2)
            if len(close) >= 6:  c1w = round((close.iloc[-1]/close.iloc[-6]-1)*100, 2)
            if len(close) >= 22: c1m = round((close.iloc[-1]/close.iloc[-22]-1)*100, 2)

        all_rows.append({
            'Symbol':        sym,
            'Company':       fund.get('company_name', sym),
            'Sector':        stock_to_sector_map.get(sym, 'Unknown'),
            'Price (₹)':     fund.get('current_price'),
            '1D %':          c1d,
            '1W %':          c1w,
            '1M %':          c1m,
            'Market Cap':    fund.get('market_cap'),
            'P/E':           fund.get('pe_ratio'),
            'ROE %':         fund.get('roe'),
            'D/E Ratio':     fund.get('debt_equity'),
            'Rev Growth %':  fund.get('revenue_growth'),
            'Net Margin %':  fund.get('net_margin'),
            'Promoter %':    fund.get('promoter_holding_pct'),
            'Beta':          fund.get('beta'),
            '52W High':      fund.get('52w_high'),
            '52W Low':       fund.get('52w_low'),
        })

    if not all_rows:
        log("  No data to export yet")
        return None

    path = EXCEL_DIR / f"nightly_summary_{today_str()}.xlsx"
    try:
        df_all = pd.DataFrame(all_rows)
        df_all = df_all.sort_values('Market Cap', ascending=False, na_position='last')

        with pd.ExcelWriter(str(path), engine='openpyxl') as writer:
            df_all.to_excel(writer, sheet_name='All Stocks', index=False)

            # Top performers by 1M
            df_top = df_all.dropna(subset=['1M %']).sort_values('1M %', ascending=False).head(50)
            df_top.to_excel(writer, sheet_name='Top 50 (1M)', index=False)

            # Quality screen
            df_q = df_all[
                (df_all['ROE %'].fillna(0) >= 15) &
                (df_all['D/E Ratio'].fillna(999) <= 1.0) &
                (df_all['Rev Growth %'].fillna(-999) >= 10)
            ]
            df_q.to_excel(writer, sheet_name='Quality Screen', index=False)

            # By sector
            for sector in SECTORS:
                df_sec = df_all[df_all['Sector'] == sector]
                if not df_sec.empty:
                    safe_name = sector.replace('/', '-').replace('*', '-').replace('[', '').replace(']', '')[:31]
                    df_sec.to_excel(writer, sheet_name=safe_name, index=False)

        log(f"  ✓ Excel saved: {path.name} ({len(all_rows)} stocks)")
        return path
    except Exception as e:
        log(f"  Excel error: {e}")
        return None


# ══════════════════════════════════════════════════════════════
# MAIN NIGHTLY JOB
# ══════════════════════════════════════════════════════════════

def run_nightly_job(
    do_prices: bool = True,
    do_fundamentals: bool = True,
    do_news: bool = True,
    do_vector_index: bool = True,
    do_excel: bool = True,
    force: bool = False,
    symbols: list = None
):
    start = time.time()
    log("🌙 NIGHTLY DATA JOB STARTED")
    log(f"   Date: {today_str()}")
    # Get live universe first
    all_stocks, stock_to_sector = get_universe(force_refresh=force)
    syms = symbols or all_stocks
    log(f"   Stocks: {len(syms)} total (live NSE universe)")

    if do_prices:
        price_results = download_all_prices(syms, force=force)
        log("\n" + generate_error_summary(price_results))
    else:
        price_results = {}

    if do_fundamentals:
        fund_results = download_all_fundamentals(syms, force=force)
        log("\n" + generate_error_summary(fund_results))
    else:
        fund_results = {}

    if do_news:
        download_all_news(syms)

    if do_vector_index:
        build_vector_index(force_rebuild=force)

    if do_excel:
        generate_nightly_excel()

    cleanup_old_files(keep_days=90)

    elapsed = round(time.time() - start)
    mins, secs = divmod(elapsed, 60)
    summary = get_storage_summary()

    log(f"\n✅  NIGHTLY JOB COMPLETE ({mins}m {secs}s)")
    log(f"   Price CSVs:     {summary['price_csvs']}")
    log(f"   Fundamentals:   {summary['fundamental_jsons']}")
    log(f"   News archives:  {summary['news_stock_dirs']}")
    log(f"   Excel reports:  {summary['excel_reports']}")
    log(f"   Total storage:  {summary['total_size_mb']} MB")

    return summary


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prices',   action='store_true')
    parser.add_argument('--fund',     action='store_true')
    parser.add_argument('--news',     action='store_true')
    parser.add_argument('--index',    action='store_true')
    parser.add_argument('--excel',    action='store_true')
    parser.add_argument('--force',    action='store_true')
    parser.add_argument('--symbols',  nargs='+')
    args = parser.parse_args()

    syms = [s.upper() for s in args.symbols] if args.symbols else None
    specific = any([args.prices, args.fund, args.news, args.index, args.excel])

    if specific:
        if args.prices: download_all_prices(syms, force=args.force)
        if args.fund:   download_all_fundamentals(syms, force=args.force)
        if args.news:   download_all_news(syms)
        if args.index:  build_vector_index(force_rebuild=args.force)
        if args.excel:  generate_nightly_excel()
    else:
        run_nightly_job(symbols=syms, force=args.force)
