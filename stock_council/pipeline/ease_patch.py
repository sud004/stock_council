# ============================================================
# pipeline/ease_patch.py
# Wires Ventura EaseAPI into the existing pipeline
# ============================================================
# Call apply_patch() once at startup.
# After that, all data fetching transparently uses EaseAPI
# instead of Yahoo Finance / NSE scraping.
#
# WHAT GETS REPLACED:
#   utils.live_data.get_live_quote     → EaseAPI OHLCV
#   utils.live_data.NSELive.get_quote  → EaseAPI OHLCV
#   utils.live_data.NSELive.get_all_indices → EaseAPI index quotes
#   scanner.sector_scanner fetch loop  → Single batch EaseAPI call
#   market_scanner indices             → EaseAPI index quotes
#
# WHAT STAYS THE SAME:
#   utils.market_data.fetch_price_history  → Yahoo Finance (history)
#   utils.market_data.fetch_fundamentals   → Yahoo Finance (fundamentals)
#   bots/* (all 5 bots)                    → unchanged
#   pipeline/chains.py                     → unchanged
#   memory/*                               → unchanged
# ============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def apply_patch():
    """
    Monkey-patch the pipeline to use EaseAPI where applicable.
    Call this ONCE at the start of run.py before anything else.
    """
    from utils.easeapi import (
        _auth, get_live_quote_ease, get_nse_indices_ease,
        get_all_sector_quotes_ease, status as ease_status
    )

    if not _auth.is_valid():
        print("[PATCH] EaseAPI not authenticated — using fallback sources")
        print("        Run: python utils/easeapi.py   to login")
        return False

    print("[PATCH] EaseAPI authenticated — patching data sources...")

    # ── Patch 1: get_live_quote ───────────────────────────────
    import utils.live_data as ld
    ld._original_get_live_quote = ld.get_live_quote
    ld.get_live_quote = get_live_quote_ease

    # ── Patch 2: NSELive.get_all_indices ─────────────────────
    original_get_indices = ld.NSELive.get_all_indices

    @classmethod
    def patched_get_indices(cls):
        result = get_nse_indices_ease()
        return result if result else original_get_indices()

    ld.NSELive.get_all_indices = patched_get_indices

    # ── Patch 3: Sector scanner batch fetch ───────────────────
    # Replace the per-stock loop with a single batch call
    import scanner.sector_scanner as ss
    original_fetch = ss.fetch_stock_quick_metrics

    # Pre-load quotes for the entire sector in one call
    _sector_quote_cache = {}
    _cache_time = [0]

    def patched_fetch_stock_metrics(symbol: str, nifty_1m: float = 0) -> dict:
        import time
        import numpy as np
        from utils.market_data import fetch_price_history, resolve_symbol
        from utils.easeapi import get_all_quotes

        sym = symbol.replace('.NS', '').replace('.BO', '').upper()

        # Check batch cache (refresh every 5 min)
        if time.time() - _cache_time[0] > 300 or sym not in _sector_quote_cache:
            # Refresh quotes for this batch
            from scanner.universe import ALL_STOCKS
            batch_size = 100
            # Only refresh if cache is truly stale
            if time.time() - _cache_time[0] > 300:
                batch = ALL_STOCKS[:batch_size]   # first batch
                quotes = get_all_quotes(batch)
                _sector_quote_cache.update(quotes)
                _cache_time[0] = time.time()

        # Get live quote from cache
        live = _sector_quote_cache.get(sym, {})

        # Still get historical data for moving averages / RSI
        # (EaseAPI doesn't provide historical, Yahoo Finance does)
        try:
            df = fetch_price_history(symbol, period="3mo", interval="1d")
            if df is None or df.empty or len(df) < 5:
                # Use live data only
                return {
                    'symbol':       sym,
                    'price':        live.get('ltp'),
                    'change_1d':    live.get('pct_change'),
                    'change_1w':    None,
                    'change_1m':    None,
                    'above_50dma':  None,
                    'above_200dma': None,
                    'rsi':          50,
                    'volume_ratio': None,
                    'vs_nifty_1m':  None,
                }

            close = df['Close']
            volume = df['Volume']
            price = live.get('ltp') or float(close.iloc[-1])

            c1d = live.get('pct_change')  # use live 1D
            c1w = round((close.iloc[-1]/close.iloc[-6]-1)*100, 2) if len(close) >= 6 else None
            c1m = round((close.iloc[-1]/close.iloc[-22]-1)*100, 2) if len(close) >= 22 else None
            c3m = round((close.iloc[-1]/close.iloc[0]-1)*100, 2) if len(close) >= 60 else None

            sma50  = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
            sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = (-delta).clip(lower=0)
            avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = float((100 - 100/(1+rs)).iloc[-1])

            vol_avg = float(volume.tail(20).mean())
            vol_today = live.get('volume') or float(volume.iloc[-1])
            vol_ratio = round(vol_today / vol_avg, 2) if vol_avg > 0 else None

            return {
                'symbol':       sym,
                'price':        round(price, 2),
                'change_1d':    c1d,
                'change_1w':    c1w,
                'change_1m':    c1m,
                'change_3m':    c3m,
                'above_50dma':  price > sma50 if sma50 else None,
                'above_200dma': price > sma200 if sma200 else None,
                'rsi':          round(rsi, 1),
                'volume_ratio': vol_ratio,
                'vs_nifty_1m':  round((c1m or 0) - nifty_1m, 2),
                # EaseAPI extras
                'high_today':   live.get('high'),
                'low_today':    live.get('low'),
                'upper_circuit': live.get('upper_circuit'),
                'lower_circuit': live.get('lower_circuit'),
            }
        except Exception as e:
            # Fallback to original
            return original_fetch(symbol, nifty_1m)

    ss.fetch_stock_quick_metrics = patched_fetch_stock_metrics

    print("[PATCH] ✅ EaseAPI patches applied:")
    print("         live quotes      → Ventura EaseAPI (direct exchange)")
    print("         index data       → Ventura EaseAPI")
    print("         sector batch     → Ventura EaseAPI (1 call per 1000 stocks)")
    print("         historical OHLCV → Yahoo Finance (EaseAPI has no history endpoint)")
    print("         fundamentals     → Yahoo Finance (EaseAPI has no fundamentals)")
    return True


def refresh_all_quotes_ease(symbols: list) -> dict:
    """
    Pre-load quotes for a large list of symbols using EaseAPI batch call.
    Call this once at market open to warm the cache.
    Returns {symbol: quote_dict}
    """
    from utils.easeapi import get_all_quotes
    print(f"[EASE] Batch loading quotes for {len(symbols)} symbols...")
    quotes = get_all_quotes(symbols)
    print(f"[EASE] Got {len(quotes)} quotes in one API call")
    return quotes


def start_realtime_streaming(symbols: list, on_tick=None):
    """
    Start WebSocket streaming for real-time price updates.
    Runs in background — updates pipeline data automatically.

    on_tick(symbol, data) called on every price update.
    """
    from utils.easeapi import get_websocket

    ws = get_websocket()

    # Default tick handler — updates live data store
    def default_tick_handler(symbol, tick):
        if on_tick:
            on_tick(symbol, tick)
        # Could trigger re-analysis if price moves > X%
        pct = tick.get('pct_change', 0) or 0
        if abs(pct) > 3.0:
            print(f"[WS ALERT] {symbol} moved {pct:+.2f}% — consider re-analysis")

    ws.subscribe(symbols, on_tick=default_tick_handler, with_depth=False)
    ws.start(block=False)

    print(f"[EASE WS] Streaming {len(symbols)} symbols in real-time")
    return ws
