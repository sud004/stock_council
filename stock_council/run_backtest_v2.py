# -*- coding: utf-8 -*-
"""
run_backtest_v2.py — Backtest Model B on historical Days 1-13
==============================================================
Reads saved council score files (data/scores/*.json) and price CSVs
(data/prices/*.csv) to simulate what Model B would have done on each
trading day already completed.

At the end, prints a side-by-side comparison of Model A vs Model B,
and saves the Model B state to data/portfolio_state_v2.json so it can
continue live from tonight's Day 14 run.

Usage:
    python run_backtest_v2.py
    python run_backtest_v2.py --reset       # wipe existing v2 state first
    python run_backtest_v2.py --show-only   # print summary without re-running
"""

import sys
import json
import argparse
import os
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR, BASE_DIR
from portfolio_engine_v2 import ModelBEngine, STATE_FILE

SCORES_DIR  = DATA_DIR / "scores"
PRICES_DIR  = DATA_DIR / "prices"
NIGHT_DIR   = BASE_DIR / "reports" / "nightly"


# ══════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════

def load_council_history() -> dict:
    """
    Read all data/scores/{SYM}.json files.
    Returns {date_str: [{'symbol', 'composite', 'scores', 'verdict'}, ...]}
    grouped by trading date.
    """
    daily = defaultdict(list)
    errors = 0

    for fname in os.listdir(SCORES_DIR):
        if not fname.endswith('.json'):
            continue
        sym = fname[:-5]
        try:
            with open(SCORES_DIR / fname) as f:
                entries = json.load(f)
            for e in entries:
                if not isinstance(e, dict) or 'date' not in e:
                    continue
                daily[e['date']].append({
                    'symbol':    sym,
                    'composite': e.get('composite', 0.0),
                    'scores':    e.get('scores', {}),
                    'verdict':   e.get('verdict', ''),
                    'gps':       e.get('gps', 0.0),
                })
        except Exception:
            errors += 1

    if errors:
        print(f"  [DATA] Skipped {errors} corrupt score files")

    return dict(daily)


def load_all_prices() -> dict:
    """
    Read all data/prices/{SYM}.csv files.
    Returns {symbol: pd.DataFrame} indexed by date string.
    """
    prices = {}
    for fname in os.listdir(PRICES_DIR):
        if not fname.endswith('.csv'):
            continue
        sym = fname[:-4]
        try:
            df = pd.read_csv(PRICES_DIR / fname)
            # Drop corrupt rows (non-date values like "202")
            df = df[pd.to_numeric(df['Open'], errors='coerce').notna()].copy()
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date']).sort_values('Date')
            df['date_str'] = df['Date'].dt.strftime('%Y-%m-%d')
            prices[sym] = df.set_index('date_str')
        except Exception:
            pass
    return prices


def get_price(prices: dict, symbol: str, date_str: str, field: str) -> float:
    """Get open or close price for a symbol on a date. Returns None if missing."""
    df = prices.get(symbol)
    if df is None or date_str not in df.index:
        return None
    val = df.loc[date_str, field]
    try:
        return float(val)
    except Exception:
        return None


def next_trading_date(prices: dict, after_date: str) -> str:
    """
    Return the first date > after_date that has price data for at least
    one stock. Used to find the execution date (next-day open).
    """
    # Collect all dates across all price files
    all_dates = set()
    ref_sym = next(iter(prices))  # use any stock as reference
    for d in prices[ref_sym].index:
        if d > after_date:
            all_dates.add(d)
    if not all_dates:
        return None
    return sorted(all_dates)[0]


def load_model_a_log() -> list:
    """Load Model A's daily log for comparison. Returns [] on failure."""
    state_path = DATA_DIR / "portfolio_state.json"
    # Try backup if main is corrupt
    for path in [state_path,
                 DATA_DIR / "portfolio_state_backup_20260702_195909.json"]:
        try:
            with open(path) as f:
                state = json.load(f)
            log = state.get('daily_log', [])
            if log:
                return log
        except Exception:
            continue
    # Try to reconstruct from night reports
    log = []
    if NIGHT_DIR.exists():
        for fname in sorted(NIGHT_DIR.iterdir()):
            try:
                with open(fname) as f:
                    r = json.load(f)
                p4 = r.get('phase4', {})
                if p4:
                    log.append({
                        'date':            r.get('date'),
                        'day':             r.get('day'),
                        'portfolio_value': p4.get('portfolio_value'),
                        'total_pl':        p4.get('total_pl'),
                        'total_return_pct': p4.get('total_return_pct'),
                    })
            except Exception:
                pass
    return log


# ══════════════════════════════════════════════════════════════
# BACKTEST RUNNER
# ══════════════════════════════════════════════════════════════

def run_backtest(reset: bool = False):
    print(f"\n{'='*60}")
    print("  MODEL B BACKTEST  —  Days 1-13")
    print(f"{'='*60}")

    # ── Load data ──────────────────────────────────────────────
    print("\n  Loading council history...", end=' ', flush=True)
    council_hist = load_council_history()
    print(f"  {len(council_hist)} days found")

    print("  Loading price data...", end=' ', flush=True)
    prices = load_all_prices()
    print(f"  {len(prices)} stocks")

    # ── Determine trading days (sorted council dates) ──────────
    # Only use dates that appear in night reports (actual game days)
    # Filter out weekends — the night runner sometimes stores scores on
    # Saturdays/Sundays when it runs overnight after a Friday session,
    # but those are NOT trading days (no market open/close prices exist).
    from datetime import datetime as _dt
    all_council_dates = sorted(council_hist.keys())
    weekend_skipped   = [d for d in all_council_dates
                         if _dt.strptime(d, '%Y-%m-%d').weekday() >= 5]
    game_dates        = [d for d in all_council_dates
                         if _dt.strptime(d, '%Y-%m-%d').weekday() < 5]

    if weekend_skipped:
        print(f"\n  ⚠️  Skipped {len(weekend_skipped)} weekend date(s): {weekend_skipped}")

    print(f"\n  Trading days to backtest: {len(game_dates)}")
    for d in game_dates:
        n = len(council_hist[d])
        print(f"    {d}  ({n} stocks debated)")

    # ── Initialise engine ──────────────────────────────────────
    if reset and STATE_FILE.exists():
        STATE_FILE.unlink()
        print("\n  State reset ✓")

    engine = ModelBEngine()
    if engine.state['current_day'] > 0 and not reset:
        print(f"\n  Resuming from Day {engine.state['current_day']} "
              f"(use --reset to start fresh)")
        already_done = {d['date'] for d in engine.state['daily_log']}
        game_dates   = [d for d in game_dates if d not in already_done]
        print(f"  Remaining days to process: {len(game_dates)}")

    # ── Main simulation loop ───────────────────────────────────
    print(f"\n{'─'*60}")

    for day_n, date_str in enumerate(game_dates, start=1):
        stocks_today = council_hist[date_str]

        # next trading date for buy execution
        next_date = next_trading_date(prices, date_str)

        # Price maps
        today_close_map = {}
        next_open_map   = {}
        today_open_map  = {}
        for sym in prices:
            c = get_price(prices, sym, date_str, 'Close')
            if c is not None:
                today_close_map[sym] = c
            if next_date:
                no = get_price(prices, sym, next_date, 'Open')
                if no is not None:
                    next_open_map[sym] = no
            to = get_price(prices, sym, date_str, 'Open')
            if to is not None:
                today_open_map[sym] = to

        engine.run_day(
            date_str       = date_str,
            council_stocks = stocks_today,
            price_map      = today_close_map,
            next_open_map  = next_open_map,
            prev_open_map  = today_open_map,
            trading_day_n  = engine.state['current_day'] + 1,
            print_output   = True,
        )

    # ── Mark any still-open positions at last price ────────────
    # (MTM using the most recent close available)
    last_date = game_dates[-1] if game_dates else None
    if last_date:
        final_price_map = {}
        for sym in prices:
            # Walk forward from last council date to get freshest close
            df = prices.get(sym)
            if df is not None:
                future = df[df.index >= last_date]
                if not future.empty:
                    final_price_map[sym] = float(future['Close'].iloc[-1])
        engine.state['portfolio_value'] = engine.portfolio_value(final_price_map)

    # ── Save state ─────────────────────────────────────────────
    engine.save()
    print(f"\n  State saved → {STATE_FILE}")

    # ── Print Model B summary ──────────────────────────────────
    engine.print_summary()

    # ── Side-by-side comparison with Model A ──────────────────
    model_a_log = load_model_a_log()
    if model_a_log:
        print(f"\n{'═'*60}")
        print(f"  A vs B  COMPARISON  (per day)")
        print(f"{'═'*60}")
        print(f"  {'Date':<12} {'Day':>4}  {'Model A Value':>14}  {'Model B Value':>14}  {'A Return':>9}  {'B Return':>9}")
        print(f"  {'─'*12} {'─'*4}  {'─'*14}  {'─'*14}  {'─'*9}  {'─'*9}")

        b_log_by_date = {d['date']: d for d in engine.state['daily_log']}

        for a_day in model_a_log:
            date = a_day.get('date') or a_day.get('trading_date', '')
            b_day = b_log_by_date.get(date, {})
            a_val = a_day.get('portfolio_value', 0)
            b_val = b_day.get('portfolio_value', 0)
            a_ret = a_day.get('total_return_pct', 0)
            b_ret = b_day.get('total_return_pct', 0)
            day_n = a_day.get('day', '?')

            a_sign = '+' if a_ret >= 0 else ''
            b_sign = '+' if b_ret >= 0 else ''
            print(f"  {date:<12} {str(day_n):>4}  "
                  f"₹{a_val:>12,.0f}  ₹{b_val:>12,.0f}  "
                  f"{a_sign}{a_ret:>7.2f}%  {b_sign}{b_ret:>7.2f}%")

        # Final comparison
        last_a = model_a_log[-1]
        last_b = engine.state['daily_log'][-1] if engine.state['daily_log'] else {}
        a_final_val = last_a.get('portfolio_value', 100000)
        b_final_val = last_b.get('portfolio_value', 0)
        a_final_ret = last_a.get('total_return_pct', 0)
        b_final_ret = last_b.get('total_return_pct', 0)

        print(f"\n  {'─'*60}")
        print(f"  Model A: ₹{a_final_val:>10,.0f}   {'+' if a_final_ret>=0 else ''}{a_final_ret:.2f}% return on ₹1,00,000 corpus")
        b_injected = engine.state['total_injected']
        print(f"  Model B: ₹{b_final_val:>10,.0f}   {'+' if b_final_ret>=0 else ''}{b_final_ret:.2f}% return on ₹{b_injected:,.0f} deployed")
        print(f"  Cash Reserve (B): ₹{engine.state['cash']:,.0f}")
        print(f"{'═'*60}")

    print(f"\n  ✅  Backtest complete. Model B state ready for live Day 14+.")
    print(f"  Tonight's night_runner.py will pick up from here automatically.\n")


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════�