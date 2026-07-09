#!/usr/bin/env python3
# ============================================================
# missed_opportunities.py — Missed Gainer/Loser Analysis
# ============================================================
# PURPOSE:
#   Every night's main pipeline only debates ~15-20 GPS-qualified
#   stocks out of the full 406-stock universe. This module finds
#   the day's biggest movers (top 5 gainers + bottom 5 losers)
#   and checks: did our system already analyze them tonight?
#
#   For genuine misses, it runs the EXACT SAME 5-bot council
#   (same prompts, same scoring, same fast_mode) so the result
#   is directly comparable to tonight's actual debate output.
#   This answers: "would we have caught this move if we'd looked?"
#
# DESIGN (confirmed):
#   - Top 5 gainers + bottom 5 losers by daily % change (10 total)
#   - Cross-referenced against tonight's debated symbol list
#   - Genuine misses get the FULL 5-bot fast-mode analysis
#     (identical methodology to the main pipeline — Option A)
#   - ANALYTICAL ONLY — never triggers a portfolio buy/sell
#   - Runs as a SEPARATE phase, after the main pipeline finishes,
#     with its own fresh Ollama restart first
#   - Results logged to a new "Missed Opportunities" Excel sheet
#   - Data feeds into Day 7/14 weekly review (NOT acted on
#     automatically before Day 7 — purely collected first)
#
# USAGE (standalone test, before wiring into night_runner.py):
#   python missed_opportunities.py
#   python missed_opportunities.py --top-n 5
#   python missed_opportunities.py --no-restart   (skip ollama restart)
#   python missed_opportunities.py --dry-run      (scan only, no debate)
# ============================================================

import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR, BASE_DIR
from trading_calendar import get_trading_date
from memory.storage import load_prices_csv, load_fundamentals_json
from scanner.universe import ALL_STOCKS, STOCK_TO_SECTOR

MISSED_OPP_FILE = DATA_DIR / "missed_opportunities.json"
MISSED_OPP_DIR  = BASE_DIR / "reports" / "missed_opportunities"
MISSED_OPP_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# STEP 1: SCAN FULL UNIVERSE FOR TODAY'S MOVERS
# ══════════════════════════════════════════════════════════════

def scan_daily_movers(top_n: int = 5, curated_only: bool = True) -> dict:
    """
    Scan stocks in the universe for today's % change.
    Pure price-based, no LLM, no extra API calls — uses
    the price CSVs already downloaded in Phase 1.

    Args:
        curated_only: If True (default), only scans the ~250
            stocks in the curated SECTORS dict — excludes the
            "Other" sector stocks pulled from the live NSE
            universe fetch. Those are often illiquid micro-caps
            where a small Rs move shows as a large % swing --
            noise, not genuine institutional-grade opportunities.
            Set False to scan the full 406-stock universe instead.

    Returns dict with 'gainers' and 'losers' lists, each
    containing {symbol, sector, pct_change, close_price}.
    """
    from scanner.universe import SECTORS

    if curated_only:
        scan_list = set()
        for stocks in SECTORS.values():
            scan_list.update(stocks)
        scan_list = sorted(scan_list)
    else:
        scan_list = ALL_STOCKS

    print(f"\n{'='*60}")
    print(f"  SCANNING {len(scan_list)} STOCKS FOR TODAY'S MOVERS "
          f"({'curated universe only' if curated_only else 'full universe'})")
    print('='*60)

    movers = []
    skipped = 0

    for sym in scan_list:
        # FIX: load_prices_csv(days=N) compares against datetime.now()
        # which includes a time-of-day component. A date-only CSV row
        # like "2026-06-19 00:00:00" can fall BEFORE a cutoff of
        # "2026-06-19 14:30:00", silently excluding the most recent
        # trading day. Load full history instead and slice the last
        # 2 rows directly — avoids the time-of-day comparison entirely.
        df = load_prices_csv(sym)
        if df is None or df.empty or len(df) < 2:
            skipped += 1
            continue

        try:
            close_today     = float(df['Close'].iloc[-1])
            close_yesterday = float(df['Close'].iloc[-2])
            pct_change = round(
                (close_today - close_yesterday) / close_yesterday * 100, 2)

            movers.append({
                'symbol':      sym,
                'sector':      STOCK_TO_SECTOR.get(sym, 'Unknown'),
                'pct_change':  pct_change,
                'close_price': close_today,
            })
        except Exception:
            skipped += 1
            continue

    movers.sort(key=lambda x: x['pct_change'], reverse=True)
    gainers = movers[:top_n]
    losers  = movers[-top_n:][::-1]   # most negative first

    print(f"\n  Scanned: {len(movers)} stocks | Skipped (no data): {skipped}")
    print(f"\n  Top {top_n} GAINERS today:")
    for g in gainers:
        print(f"    {g['symbol']:14} {g['sector'][:20]:20} {g['pct_change']:+.2f}%")

    print(f"\n  Top {top_n} LOSERS today:")
    for l in losers:
        print(f"    {l['symbol']:14} {l['sector'][:20]:20} {l['pct_change']:+.2f}%")

    return {'gainers': gainers, 'losers': losers}


# ══════════════════════════════════════════════════════════════
# STEP 2: CROSS-REFERENCE AGAINST TONIGHT'S DEBATED STOCKS
# ══════════════════════════════════════════════════════════════

def find_genuine_misses(movers: dict, debated_symbols: set) -> list:
    """
    Filter movers down to only those NOT already debated tonight.
    Returns a flat list tagged with 'mover_type': 'gainer'|'loser'.
    """
    misses = []

    for g in movers['gainers']:
        if g['symbol'] not in debated_symbols:
            misses.append({**g, 'mover_type': 'gainer'})

    for l in movers['losers']:
        if l['symbol'] not in debated_symbols:
            misses.append({**l, 'mover_type': 'loser'})

    print(f"\n{'='*60}")
    print(f"  GENUINE MISSES: {len(misses)} stocks "
          f"(not in tonight's {len(debated_symbols)}-stock debate list)")
    print('='*60)
    for m in misses:
        tag = "📈 GAINER" if m['mover_type'] == 'gainer' else "📉 LOSER"
        print(f"    {tag}  {m['symbol']:14} {m['pct_change']:+.2f}%  "
              f"({m['sector']})")

    return misses


# ══════════════════════════════════════════════════════════════
# STEP 3: RETROACTIVE GPS — WOULD THIS STOCK HAVE QUALIFIED?
# ══════════════════════════════════════════════════════════════

def compute_retroactive_gps(symbol: str, sector: str) -> dict:
    """
    Compute what this stock's GPS score would have been tonight,
    using the exact same formula as the main pipeline. Tells us
    WHY it was missed: low GPS (correctly excluded) vs high GPS
    but excluded by sector caps (genuine blind spot).
    """
    from pipeline.orchestrator import compute_gps
    from scanner.sector_scanner import fetch_stock_quick_metrics

    try:
        pm   = fetch_stock_quick_metrics(symbol, nifty_1m=0)
        fund = load_fundamentals_json(symbol) or {}
        gps, components = compute_gps(symbol, pm, fund, nifty_1m=0,
                                      sector_avg_1m=0)
        return {'gps': gps, 'components': components, 'error': None}
    except Exception as e:
        return {'gps': None, 'components': {}, 'error': str(e)}


# ══════════════════════════════════════════════════════════════
# STEP 4: FULL 5-BOT DEBATE (Option A — identical methodology)
# ══════════════════════════════════════════════════════════════

def debate_missed_stock(miss: dict, market_snap_dict: dict,
                        print_output: bool = True) -> dict:
    """
    Run the EXACT SAME 5-bot fast-mode analysis used by the main
    pipeline on this missed stock. Returns the same shape of
    result as run_council_for_stock() so it's directly comparable.

    This deliberately reuses MarketOrchestrator's own method
    rather than reimplementing bot calls — guarantees identical
    prompts, identical scoring, identical weighting (Option A).
    """
    from pipeline.orchestrator import MarketOrchestrator
    from memory.storage import load_fundamentals_json

    sym    = miss['symbol']
    sector = miss['sector']
    fund   = load_fundamentals_json(sym) or {}

    if print_output:
        print(f"\n  --- Debating missed {miss['mover_type'].upper()}: "
              f"{sym} ({miss['pct_change']:+.2f}%) ---")

    orch = MarketOrchestrator()
    # Build the same stock_info shape the main pipeline uses
    retro = compute_retroactive_gps(sym, sector)

    stock_info = {
        'symbol':       sym,
        'sector':       sector,
        'gps':          retro['gps'] or 0,
        'components':   retro['components'],
        'price_metrics': {},
        'fundamentals': fund,
    }

    try:
        result = orch.run_council_for_stock(
            stock_info, market_snap_dict,
            print_output=print_output,
            fast_mode=True,   # same mode as tonight's main run
        )
        result['retroactive_gps']  = retro['gps']
        result['actual_pct_change'] = miss['pct_change']
        result['mover_type']        = miss['mover_type']
        return result
    except Exception as e:
        print(f"  [MISSED OPP] Error debating {sym}: {e}")
        return {
            'symbol': sym, 'sector': sector,
            'error': str(e),
            'retroactive_gps': retro['gps'],
            'actual_pct_change': miss['pct_change'],
            'mover_type': miss['mover_type'],
        }


# ══════════════════════════════════════════════════════════════
# OLLAMA RESTART (reuses the same logic as night_runner.py)
# ══════════════════════════════════════════════════════════════

def restart_ollama_for_missed_opp() -> bool:
    """
    Fresh Ollama restart before this second debate batch.
    The main 5 PM pipeline already used Ollama heavily —
    RAM needs clearing again before debating 10 more stocks.
    """
    bat_path = Path(__file__).parent / "restart_ollama.bat"
    print(f"\n  Restarting Ollama before missed-opportunity debate...")

    try:
        if bat_path.exists():
            subprocess.run([str(bat_path)], shell=True, timeout=40)
        else:
            subprocess.run(["taskkill", "/f", "/im", "ollama.exe"],
                          shell=True, capture_output=True)
            time.sleep(5)
            subprocess.Popen(["ollama", "serve"], shell=True)
            time.sleep(8)

        from utils.llm import check_ollama
        ready = check_ollama()
        print(f"  {'Ollama ready' if ready else 'Ollama may still be starting'}\n")
        return ready
    except Exception as e:
        print(f"  Restart failed: {e} — continuing anyway\n")
        return False


# ══════════════════════════════════════════════════════════════
# STORAGE
# ══════════════════════════════════════════════════════════════

def _load_history() -> dict:
    if MISSED_OPP_FILE.exists():
        with open(MISSED_OPP_FILE) as f:
            return json.load(f)
    return {'daily_records': []}


def _save_history(data: dict):
    with open(MISSED_OPP_FILE, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def save_tonight_record(misses: list, debate_results: list,
                        debated_count: int, trading_date: str):
    """Save tonight's missed-opportunity analysis to history."""
    history = _load_history()

    record = {
        'date':            trading_date,
        'debated_tonight': debated_count,
        'movers_scanned':  len(misses),
        'genuine_misses':  len(misses),
        'results':         debate_results,
    }
    history.setdefault('daily_records', []).append(record)
    history['daily_records'] = history['daily_records'][-21:]
    _save_history(history)

    report_path = MISSED_OPP_DIR / f"missed_opp_{trading_date}.json"
    with open(report_path, 'w') as f:
        json.dump(record, f, indent=2, default=str)

    print(f"\n  Saved: {report_path.name}")


# ══════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════

def run_missed_opportunity_analysis(
    debated_symbols: set = None,
    market_snap_dict: dict = None,
    top_n: int = 5,
    curated_only: bool = True,
    skip_restart: bool = False,
    dry_run: bool = False,
    print_output: bool = True
) -> dict:
    """
    Full missed-opportunity pipeline. Called either standalone
    (for testing) or from night_runner.py as a post-main-pipeline
    phase.

    Args:
        debated_symbols: set of symbols already debated tonight.
                          If None, tries to load from today's
                          council session file.
        market_snap_dict: market context dict for the debate
                          prompts. If None, fetches fresh.
        top_n:            how many gainers/losers to scan (default 5)
        skip_restart:     skip the Ollama restart (for testing)
        dry_run:          scan and identify misses only, no debate
    """
    trading_date = get_trading_date()

    print(f"\n{'#'*60}")
    print(f"  MISSED OPPORTUNITY ANALYSIS — {trading_date}")
    print(f"{'#'*60}")
    start = time.time()

    # Resolve tonight's debated symbols if not provided
    if debated_symbols is None:
        debated_symbols = _load_tonight_debated_symbols(trading_date)

    # Step 1: scan
    movers = scan_daily_movers(top_n=top_n, curated_only=curated_only)

    # Step 2: cross-reference
    misses = find_genuine_misses(movers, debated_symbols)

    if not misses:
        print(f"\n  No genuine misses tonight — all top movers were "
              f"already debated. Nothing to analyze.")
        return {'misses': [], 'debate_results': []}

    if dry_run:
        print(f"\n  --dry-run: skipping debate. {len(misses)} misses identified.")
        return {'misses': misses, 'debate_results': []}

    # Step 3 + 4: restart Ollama, then debate each miss
    if not skip_restart:
        restart_ollama_for_missed_opp()

    if market_snap_dict is None:
        from scanner import market_scanner
        snap = market_scanner.run(print_output=False)
        market_snap_dict = {
            'market_score': snap.market_score,
            'market_outlook': snap.market_outlook,
        }

    debate_results = []
    for i, miss in enumerate(misses, 1):
        print(f"\n  [{i}/{len(misses)}] Analyzing missed stock...")
        result = debate_missed_stock(miss, market_snap_dict,
                                     print_output=print_output)
        debate_results.append(result)

    # Save
    save_tonight_record(misses, debate_results,
                        len(debated_symbols), trading_date)

    elapsed = round(time.time() - start)
    print(f"\n{'#'*60}")
    print(f"  MISSED OPPORTUNITY ANALYSIS COMPLETE — "
          f"{elapsed//60}m {elapsed%60}s")
    print(f"  {len(misses)} stocks analyzed | Results saved")
    print(f"{'#'*60}\n")

    return {'misses': misses, 'debate_results': debate_results,
            'elapsed_s': elapsed}


def _load_tonight_debated_symbols(trading_date: str) -> set:
    """
    Try to load which symbols were debated tonight from the
    council session file. Falls back to empty set (treats
    everything as a potential miss) if not found.
    """
    session_file = Path(BASE_DIR) / "memory" / f"council_{trading_date}.json"
    if session_file.exists():
        try:
            with open(session_file) as f:
                data = json.load(f)
            symbols = {s.get('symbol') for s in data.get('stocks', [])
                      if s.get('symbol')}
            print(f"  Loaded {len(symbols)} debated symbols from "
                  f"council_{trading_date}.json")
            return symbols
        except Exception as e:
            print(f"  Could not load council session: {e}")

    # Fallback: check portfolio_state for today's actions
    try:
        from portfolio_engine import load_state
        state = load_state()
        today_log = next(
            (d for d in state.get('daily_log', [])
             if d.get('date') == trading_date), None)
        if today_log:
            symbols = set()
            for action in today_log.get('actions', []):
                parts = action.split()
                if len(parts) > 1:
                    symbols.add(parts[1])
            print(f"  Fallback: found {len(symbols)} symbols from "
                  f"portfolio actions today")
            return symbols
    except Exception:
        pass

    print(f"  Warning: could not determine tonight's debated symbols — "
          f"treating all movers as potential misses")
    return set()


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Missed Opportunity Analysis — gainer/loser scan + debate',
        epilog="""
EXAMPLES:
  python missed_opportunities.py                # full run, top 5 each side
  python missed_opportunities.py --top-n 10      # top 10 gainers + losers
  python missed_opportunities.py --dry-run       # scan only, no debate
  python missed_opportunities.py --no-restart    # skip Ollama restart
        """
    )
    parser.add_argument('--top-n', type=int, default=5,
                        help='Number of gainers/losers to scan (default 5)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Scan and identify misses only, skip debate')
    parser.add_argument('--no-restart', action='store_true',
                        help='Skip the Ollama restart step')
    parser.add_argument('--full-universe', action='store_true',
                        help='Scan all 406 stocks instead of the curated '
                             '~250-stock list (includes noisy micro-caps)')

    args = parser.parse_args()

    run_missed_opportunity_analysis(
        top_n=args.top_n,
        curated_only=not args.full_universe,
        skip_restart=args.no_restart,
        dry_run=args.dry_run,
    )