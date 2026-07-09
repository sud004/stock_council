#!/usr/bin/env python3
# ============================================================
# shadow_log.py — Passive Evidence Logging (Observes Only)
# ============================================================
# PURPOSE:
#   Two parallel "what-if" measurements that run alongside the
#   real portfolio every night, without ever changing a real
#   buy/sell/hold decision. Built in response to external review
#   feedback flagging two specific design risks:
#
#   1. ATR shadow-stop — is the fixed entry-price stop too tight
#      compared to a volatility-adjusted (ATR-based) stop?
#
#   2. Sector-override shadow log — how often does an individually
#      strong stock (GPS ≥ 7.0) get excluded purely because its
#      sector wasn't selected that night?
#
# DESIGN PRINCIPLE:
#   Nothing in this file ever buys, sells, or changes a real
#   position. It only writes hypothetical outcomes to a separate
#   log file, building an evidence base for the Day 14 weekly
#   review — where any resulting parameter change goes through
#   the normal human-approved governance process, not here.
#
# USAGE (called from night_runner.py, after Phase 4):
#   from shadow_log import run_shadow_logging
#   run_shadow_logging(state, sector_results, stocks_per_sector)
#
# STANDALONE TEST:
#   python shadow_log.py --report   # show accumulated findings
# ============================================================

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR
from trading_calendar import get_trading_date
from memory.storage import load_prices_csv, load_fundamentals_json

SHADOW_LOG_FILE = DATA_DIR / "shadow_log.json"

# ATR stop multiplier — how many ATRs below entry the hypothetical
# stop sits. 1.5x is a common starting point in technical trading
# (tight enough to still cut real losers, wide enough to absorb
# one day of normal noise for an average-volatility stock).
ATR_STOP_MULTIPLIER = 1.5

# Sector-override threshold — deliberately higher than the normal
# 6.0 GPS entry bar. We are looking for stocks that would clear
# debate easily, not borderline cases, to keep the evidence clean.
SECTOR_OVERRIDE_GPS_THRESHOLD = 7.0

# How many top stocks per excluded sector to check. Checking only
# the top 1-2 keeps this nearly free — these are exactly the
# stocks that would have been first in line if the sector had
# been selected.
SECTOR_OVERRIDE_CHECK_TOP_N = 2


def _load_log() -> dict:
    if SHADOW_LOG_FILE.exists():
        with open(SHADOW_LOG_FILE) as f:
            return json.load(f)
    return {
        'atr_shadow_stop': [],
        'sector_override': [],
        'started': get_trading_date(),
    }


def _save_log(data: dict):
    with open(SHADOW_LOG_FILE, 'w') as f:
        json.dump(data, f, indent=2, default=str)


# ══════════════════════════════════════════════════════════════
# SHADOW LOG 1: ATR-BASED STOP COMPARISON
# ══════════════════════════════════════════════════════════════

def _get_atr(symbol: str) -> float:
    """
    Get the stock's current ATR(14) in rupees.
    Reuses the existing technical_bot indicator — no new
    calculation, just retrieval.
    """
    try:
        from bots.technical_bot import compute_all_indicators
        df = load_prices_csv(symbol)
        if df is None or len(df) < 20:
            return None
        ta = compute_all_indicators(df)
        return ta.get('atr')
    except Exception:
        return None


def log_atr_shadow_stops(state: dict, print_output: bool = True) -> list:
    """
    For every currently open position, compute what an ATR-based
    stop would be, and whether that hypothetical stop has been
    hit — without touching the real entry-price stop in any way.

    Real stop-loss logic in portfolio_engine.py is completely
    untouched by this function. This only reads state, computes
    a parallel number, and writes it to the shadow log.
    """
    today = get_trading_date()
    entries = []

    for sym, pos in state.get('positions', {}).items():
        entry_price = pos.get('entry_price')
        current_price = pos.get('current_price', entry_price)
        real_stop = pos.get('stop_loss')
        real_stop_mode = pos.get('stop_mode', 'TIGHT')

        atr = _get_atr(sym)
        if atr is None or entry_price is None:
            continue

        atr_stop = round(entry_price - (ATR_STOP_MULTIPLIER * atr), 2)
        atr_stop_hit = current_price is not None and current_price <= atr_stop
        real_stop_hit = current_price is not None and real_stop is not None and current_price <= real_stop

        entry = {
            'date':            today,
            'symbol':          sym,
            'entry_price':     entry_price,
            'current_price':   current_price,
            'atr':             round(atr, 2),
            'atr_stop_price':  atr_stop,
            'atr_stop_hit':    atr_stop_hit,
            'real_stop_price': real_stop,
            'real_stop_mode':  real_stop_mode,
            'real_stop_hit':   real_stop_hit,
            'days_held':       pos.get('days_held', 0),
            # The interesting case: real stop would exit, ATR stop would not
            'divergence':      real_stop_hit and not atr_stop_hit,
        }
        entries.append(entry)

        if print_output and entry['divergence']:
            print(f"    [ATR SHADOW] {sym}: real stop ₹{real_stop:.2f} would exit, "
                  f"ATR stop ₹{atr_stop:.2f} (±{ATR_STOP_MULTIPLIER}×ATR ₹{atr:.2f}) would hold")

    # Also check positions closed TODAY via real stop-loss — these are
    # the cases that already happened and are most informative
    for trade in state.get('closed_trades', []):
        if trade.get('sell_date') != today or trade.get('reason') != 'SELL-STOP':
            continue
        sym = trade.get('symbol')
        atr = _get_atr(sym)
        if atr is None:
            continue
        entry_price = trade.get('entry_price')
        exit_price = trade.get('exit_price')
        atr_stop = round(entry_price - (ATR_STOP_MULTIPLIER * atr), 2)
        would_have_held = exit_price > atr_stop

        entries.append({
            'date':            today,
            'symbol':          sym,
            'entry_price':     entry_price,
            'current_price':   exit_price,
            'atr':             round(atr, 2),
            'atr_stop_price':  atr_stop,
            'atr_stop_hit':    not would_have_held,
            'real_stop_price': trade.get('entry_price'),
            'real_stop_mode':  'TIGHT',
            'real_stop_hit':   True,
            'days_held':       trade.get('days_held', 0),
            'divergence':      would_have_held,
            'note':            'closed_today_real_stop',
        })

        if print_output and would_have_held:
            print(f"    [ATR SHADOW] {sym} REAL EXIT today at ₹{exit_price:.2f} — "
                  f"ATR stop ₹{atr_stop:.2f} would NOT have exited (would still be held)")

    return entries


# ══════════════════════════════════════════════════════════════
# SHADOW LOG 2: SECTOR-OVERRIDE CANDIDATES
# ══════════════════════════════════════════════════════════════

def _recompute_stocks_per_sector(sector_results: list) -> dict:
    """
    Reconstruct which sectors were selected and contributed stocks
    tonight, using only sector_results (already returned by the
    orchestrator). Mirrors the exact selection rule in
    orchestrator.run_sector_scan() so this stays consistent with
    what actually happened tonight, without requiring any change
    to the orchestrator's return signature.
    """
    stocks_per_sector = {}
    for sr in sector_results:
        n_stocks = (
            8 if sr.score >= 8.0 else
            6 if sr.score >= 7.0 else
            4 if sr.score >= 6.0 else
            2 if sr.score >= 5.0 else
            0
        )
        chosen = [t[0] for t in sr.top_stocks[:n_stocks]]
        if chosen:
            stocks_per_sector[sr.name] = chosen
    return stocks_per_sector


def log_sector_override_candidates(sector_results: list,
                                    stocks_per_sector: dict,
                                    nifty_1m: float,
                                    print_output: bool = True) -> list:
    """
    For every sector that was EXCLUDED tonight (contributed zero
    stocks to stocks_per_sector), compute the GPS of its top 1-2
    candidate stocks. Log any that would have cleared the higher
    7.0 override bar.

    This never adds a stock to tonight's debate list. It only
    measures: "if we had let exceptional stocks through anyway,
    how often would that have mattered?"
    """
    from pipeline.orchestrator import compute_gps
    from scanner.sector_scanner import fetch_stock_quick_metrics

    today = get_trading_date()
    candidates = []

    excluded_sectors = [
        sr for sr in sector_results
        if sr.name not in stocks_per_sector or not stocks_per_sector[sr.name]
    ]

    if print_output and excluded_sectors:
        print(f"    [SECTOR OVERRIDE] Checking top stocks in "
              f"{len(excluded_sectors)} excluded sectors...")

    for sr in excluded_sectors:
        top_n = sr.top_stocks[:SECTOR_OVERRIDE_CHECK_TOP_N]
        sec_avg = sr.avg_1m_change or 0

        for sym, *_ in top_n:
            try:
                pm   = fetch_stock_quick_metrics(sym, nifty_1m=nifty_1m)
                fund = load_fundamentals_json(sym) or {}
                gps, components = compute_gps(sym, pm, fund, nifty_1m, sec_avg)
            except Exception as e:
                continue

            qualifies = gps >= SECTOR_OVERRIDE_GPS_THRESHOLD

            entry = {
                'date':           today,
                'symbol':         sym,
                'sector':         sr.name,
                'sector_score':   sr.score,
                'gps':            gps,
                'components':     components,
                'qualifies_for_override': qualifies,
                'threshold':      SECTOR_OVERRIDE_GPS_THRESHOLD,
            }
            candidates.append(entry)

            if print_output and qualifies:
                print(f"    [SECTOR OVERRIDE] {sym} ({sr.name}, sector score "
                      f"{sr.score:.1f}) — GPS {gps:.2f} clears {SECTOR_OVERRIDE_GPS_THRESHOLD} "
                      f"override bar but sector was excluded tonight")

    return candidates


# ══════════════════════════════════════════════════════════════
# MAIN ENTRY POINT — called from night_runner.py
# ══════════════════════════════════════════════════════════════

def run_shadow_logging(state: dict,
                       sector_results: list = None,
                       stocks_per_sector: dict = None,
                       nifty_1m: float = 0,
                       print_output: bool = True) -> dict:
    """
    Run both shadow logs for tonight and append to the rolling
    shadow_log.json history. Call this AFTER the real portfolio
    decisions for the night are already finalized — this function
    never feeds back into them.

    stocks_per_sector is optional — if not supplied (the normal
    case, since the orchestrator does not currently expose it),
    it is reconstructed from sector_results using the identical
    selection rule the orchestrator itself uses.
    """
    if print_output:
        print(f"\n{'='*60}")
        print("  SHADOW LOG — passive evidence collection (no live changes)")
        print('='*60)

    log = _load_log()

    atr_entries = log_atr_shadow_stops(state, print_output=print_output)
    log.setdefault('atr_shadow_stop', []).extend(atr_entries)
    log['atr_shadow_stop'] = log['atr_shadow_stop'][-500:]   # cap growth

    sector_entries = []
    if sector_results:
        if stocks_per_sector is None:
            stocks_per_sector = _recompute_stocks_per_sector(sector_results)
        sector_entries = log_sector_override_candidates(
            sector_results, stocks_per_sector, nifty_1m,
            print_output=print_output
        )
        log.setdefault('sector_override', []).extend(sector_entries)
        log['sector_override'] = log['sector_override'][-500:]

    _save_log(log)

    divergent_atr = sum(1 for e in atr_entries if e.get('divergence'))
    qualifying_sector = sum(1 for e in sector_entries if e.get('qualifies_for_override'))

    if print_output:
        print(f"\n  Tonight: {len(atr_entries)} ATR comparisons "
              f"({divergent_atr} divergent), "
              f"{len(sector_entries)} sector-override checks "
              f"({qualifying_sector} qualifying)")
        print(f"  Logged to: shadow_log.json")

    return {
        'atr_entries':          atr_entries,
        'sector_entries':       sector_entries,
        'atr_divergent_count':  divergent_atr,
        'sector_qualifying_count': qualifying_sector,
    }


# ══════════════════════════════════════════════════════════════
# REPORT — summarize accumulated evidence
# ══════════════════════════════════════════════════════════════

def show_shadow_report():
    log = _load_log()

    print(f"\n{'='*60}")
    print("  SHADOW LOG REPORT — accumulated evidence")
    print('='*60)

    atr_log = log.get('atr_shadow_stop', [])
    print(f"\n  ATR SHADOW STOP — {len(atr_log)} total observations")
    divergent = [e for e in atr_log if e.get('divergence')]
    print(f"  Divergent cases (real stop exited, ATR stop would have held): "
          f"{len(divergent)}")
    if divergent:
        print(f"\n  {'Date':12} {'Symbol':12} {'Entry':>10} {'RealStop':>10} {'ATRStop':>10}")
        for e in divergent[-15:]:
            print(f"  {e['date']:12} {e['symbol']:12} "
                  f"₹{e['entry_price']:>8.2f} ₹{e['real_stop_price']:>8.2f} "
                  f"₹{e['atr_stop_price']:>8.2f}")

    sector_log = log.get('sector_override', [])
    print(f"\n  SECTOR OVERRIDE — {len(sector_log)} total candidate checks")
    qualifying = [e for e in sector_log if e.get('qualifies_for_override')]
    print(f"  Stocks clearing {SECTOR_OVERRIDE_GPS_THRESHOLD} GPS but excluded by sector: "
          f"{len(qualifying)}")
    if qualifying:
        print(f"\n  {'Date':12} {'Symbol':12} {'Sector':22} {'GPS':>6}")
        for e in qualifying[-15:]:
            print(f"  {e['date']:12} {e['symbol']:12} {e['sector'][:22]:22} "
                  f"{e['gps']:>6.2f}")

    print(f"\n  Use this evidence at the Day 14 weekly review.")
    print(f"  Nothing here has changed any live trading decision.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Shadow logging report')
    parser.add_argument('--report', action='store_true', help='Show accumulated findings')
    args = parser.parse_args()

    if args.report:
        show_shadow_report()
    else:
        print("Use --report to view accumulated shadow log findings.")
        print("This module is normally called automatically from night_runner.py.")
