#!/usr/bin/env python3
# ============================================================
# night_runner.py — 3-Week Learning Mode + Portfolio Engine
# ============================================================
# WHAT RUNS EACH DAY (5:00 PM IST, weekdays only):
#   Pre-step — Restart Ollama (clears RAM, prevents 5hr+ stalls)
#   Phase 1  — Data refresh           (~45 min)
#   Phase 2  — Backtest + learn       (~2 min)
#   Phase 3  — Council run            (~50 min)
#   Phase 4  — Portfolio engine       (~2 min)
#   Phase 5  — Night report           (~1 min)
#   Phase 6  — Missed opportunities   (~30-45 min, own Ollama restart)
#             Scans top 5 gainers/losers in curated universe, debates
#             genuine misses with identical 5-bot methodology.
#             ANALYTICAL ONLY -- no portfolio impact. Skip with
#             --skip-missed-opp if you need a faster run.
#
#   5 PM IST is chosen because NSE closes at 3:30 PM — by 5 PM
#   the day's closing prices are fully settled on Yahoo Finance,
#   giving 90 minutes of buffer before any data is pulled.
#
# WEEKLY REVIEW (Sunday 7 PM IST):
#   Auto runs before the night run.
#   Prediction weights auto-adjusted (no input needed).
#   GPS threshold + deployment % need your Y/N (55-min window).
#   No input = freeze weights and proceed.
#
# TRADING DATE LOGIC:
#   Every file is dated by which trading day's closing prices
#   it actually analyzed -- NOT by the wall-clock date the code
#   happened to execute on. See trading_calendar.py for the
#   exact rule (handles late/manual runs, weekends, holidays).
#
# DAY 21 REVIEW:
#   python night_runner.py --day21
#   Live prices, decide sell/hold/partial for each position.
#
# USAGE:
#   python night_runner.py              # run now (manual)
#   python night_runner.py --schedule   # auto every day 5 PM IST
#   python night_runner.py --data-only  # Phase 1 only
#   python night_runner.py --report     # learning progress
#   python night_runner.py --portfolio  # portfolio status
#   python night_runner.py --day21      # Day 21 interactive review
#   python night_runner.py --week 1     # Week 1 progress
#   python night_runner.py --no-restart # skip Ollama restart this run
#   python night_runner.py --skip-missed-opp  # skip Phase 6 this run
# ============================================================

import sys
import time
import json
import argparse
import subprocess
import schedule
import threading
import io
import pytz
from pathlib import Path
from datetime import datetime, date, timedelta

sys.path.insert(0, str(Path(__file__).parent))

from market_holidays import is_market_holiday
from trading_calendar import get_trading_date
from config import DATA_DIR, BASE_DIR
from portfolio_engine import (
    load_state, save_state,
    run_nightly_portfolio, run_weekly_review,
    run_day21_review, _portfolio_value,
    STARTING_CAPITAL
)
from portfolio_excel import save_portfolio_excel
from missed_opportunities import run_missed_opportunity_analysis
from shadow_log import run_shadow_logging

IST = pytz.timezone('Asia/Kolkata')

# Schedule timing
RUN_HOUR        = 17    # 5:00 PM IST -- 90 min after NSE close (3:30 PM)
RUN_MINUTE      = 0
REVIEW_HOUR     = 19    # 7:00 PM IST Sunday
REVIEW_MINUTE   = 0
REVIEW_TIMEOUT  = 55    # minutes

OLLAMA_RESTART_WAIT_KILL  = 5
OLLAMA_RESTART_WAIT_READY = 8

# ── Watchdog — auto-recover from Ollama hangs ─────────────────
# If stdout goes silent for this many minutes, Ollama is assumed
# frozen. The watchdog kills Ollama, restarts it, and retries
# Phase 3 from scratch. Phase 1 data (CSVs, fundamentals, news)
# is always preserved — only the in-progress council is retried.
WATCHDOG_SILENCE_MIN  = 12   # minutes of silence → declare hung
COUNCIL_MAX_RETRIES   = 12   # max retry attempts for Phase 3 (each resumes from bot-cache)


class _StdoutWatchdog(io.TextIOBase):
    """
    Wraps sys.stdout. Every write() updates last_print_time so
    the watchdog thread can detect when Ollama has gone silent.
    """
    def __init__(self, real_stdout):
        self._real  = real_stdout
        self.last_print_time = time.time()

    def write(self, text):
        self.last_print_time = time.time()
        return self._real.write(text)

    def flush(self):
        return self._real.flush()

    def fileno(self):
        return self._real.fileno()


NIGHT_REPORTS_DIR = BASE_DIR / "reports" / "nightly"
NIGHT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS_FILE = DATA_DIR / "learning_progress.json"


def print_banner(phase: str = ""):
    now = datetime.now(IST).strftime('%A, %d %b %Y  %H:%M IST')
    print(f"""
============================================================
  NIGHT RUNNER -- 3-WEEK LEARNING + PORTFOLIO ENGINE
  Ollama Restart -> Data -> Backtest -> Council -> Portfolio
============================================================
  {now}
  {phase}
""")


def restart_ollama() -> bool:
    """
    Kill and restart Ollama before every run to clear RAM
    fragmentation left over from the previous session.

    This fixes the 5+ hour stalls observed on stocks 15-20 of
    a run -- Ollama's memory fragments across many sequential
    LLM calls, and a fresh process avoids that entirely.

    Tries restart_ollama.bat first (project root). Falls back
    to direct taskkill + ollama serve commands.
    """
    bat_path = Path(__file__).parent / "restart_ollama.bat"
    print(f"\n  Restarting Ollama (clearing RAM before tonight's run)...")

    try:
        if bat_path.exists():
            subprocess.run([str(bat_path)], shell=True, timeout=40)
        else:
            print(f"  [OLLAMA] restart_ollama.bat not found -- using fallback")
            subprocess.run(
                ["taskkill", "/f", "/im", "ollama.exe"],
                shell=True, capture_output=True
            )
            time.sleep(OLLAMA_RESTART_WAIT_KILL)
            subprocess.Popen(["ollama", "serve"], shell=True)
            time.sleep(OLLAMA_RESTART_WAIT_READY)

        from utils.llm import check_ollama, warmup_ollama
        if check_ollama():
            warmup_ollama()   # pre-load model into RAM; switches to 3b if OOM
            print(f"  Ollama restarted and ready\n")
            return True
        else:
            print(f"  Ollama restarted but not responding yet -- "
                  f"continuing anyway\n")
            return False

    except Exception as e:
        print(f"  Ollama restart failed: {e} -- continuing without restart\n")
        return False


def _build_vector_index_with_timeout(build_fn, timeout_min: int = 15) -> dict:
    """
    Run build_vector_index() in a thread with a timeout.
    ChromaDB can hang on large document sets. If it exceeds
    timeout_min minutes, we skip it and return empty stats —
    Phase 3 still runs fine without the vector index being fresh.
    """
    result_box = {'data': None, 'done': False}

    def _run():
        try:
            result_box['data'] = build_fn()
        except Exception as e:
            print(f"  [VECTOR] Index build error: {e}")
        finally:
            result_box['done'] = True

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout_min * 60)

    if not result_box['done']:
        print(f"\n  ⚠️  VECTOR INDEX: Still running after {timeout_min} min — skipping.")
        print(f"  ⚠️  Council will use cached index from previous run. Continuing...")
        return {'total_documents': 0, 'skipped': True}

    return result_box['data'] or {}


def phase1_data_refresh(force: bool = False) -> dict:
    print(f"\n{'='*60}")
    print("  PHASE 1 -- DATA REFRESH")
    print('='*60)
    start = time.time()

    from pipeline.nightly import (
        download_all_prices, download_all_fundamentals,
        download_all_news, build_vector_index
    )
    from scanner.universe import get_universe

    all_stocks, _ = get_universe()
    print(f"\n  Stocks: {len(all_stocks)} total\n")

    price_r  = download_all_prices(all_stocks, force=force)
    fund_r   = download_all_fundamentals(all_stocks, force=force)
    download_all_news(all_stocks)

    # Skip vector rebuild if index was built within the last 20 hours
    _vector_dir = Path(__file__).parent / "data" / "vectors"
    _vector_age_h = None
    if _vector_dir.exists():
        import os
        _newest = max(
            (os.path.getmtime(str(p)) for p in _vector_dir.rglob("*") if p.is_file()),
            default=0
        )
        _vector_age_h = (time.time() - _newest) / 3600 if _newest else None

    if _vector_age_h is not None and _vector_age_h < 20:
        print(f"\n  [VECTOR] Index is {_vector_age_h:.1f}h old — skipping rebuild (< 20h)")
        vs_stats = {'total_documents': 0, 'skipped': True}
    else:
        vs_stats = _build_vector_index_with_timeout(build_vector_index)

    elapsed = round(time.time() - start)
    print(f"\n  Phase 1 complete in {elapsed//60}m {elapsed%60}s")
    return {
        'prices_ok': sum(1 for v in price_r.values() if v == 'ok'),
        'fund_ok':   sum(1 for v in fund_r.values() if v == 'ok'),
        'vs_docs':   vs_stats.get('total_documents', 0),
        'elapsed_s': elapsed,
    }


def phase2_backtest() -> dict:
    print(f"\n{'='*60}")
    print("  PHASE 2 -- BACKTEST & WEIGHT LEARNING")
    print('='*60)

    from bots.prediction_bot import (
        run_daily_backtest, load_learned_weights,
        DEFAULT_WEIGHTS, DEFAULT_PRED_WEIGHTS
    )

    results = run_daily_backtest(print_output=True)
    learned = load_learned_weights()
    sw      = learned.get('score_weights', DEFAULT_WEIGHTS)
    pw      = learned.get('pred_weights', DEFAULT_PRED_WEIGHTS)
    opt     = learned.get('optimization_count', 0)

    print(f"\n  Weight optimizer runs: {opt}")
    print(f"\n  Verdict weights (vs default):")
    for k, v in sw.items():
        diff  = v - DEFAULT_WEIGHTS.get(k, 0)
        arrow = (f"+{diff:+.3f}" if diff > 0.001 else
                 f"{diff:+.3f}" if diff < -0.001 else "  same")
        print(f"    {k:15} {v:.3f}  {arrow}")

    print(f"\n  Prediction weights (vs default):")
    for k, v in pw.items():
        diff  = v - DEFAULT_PRED_WEIGHTS.get(k, 0)
        arrow = (f"+{diff:+.3f}" if diff > 0.001 else
                 f"{diff:+.3f}" if diff < -0.001 else "  same")
        print(f"    {k:20} {v:.3f}  {arrow}")

    acc = results.get('accuracy', {}).get('overall_1d')
    if acc:
        print(f"\n  1D accuracy: {acc:.1f}%  "
              f"{'GOOD' if acc >= 60 else 'LEARNING'}")

    return {
        'accuracy_1d': acc,
        'checked':     results.get('checked', 0),
        'correct':     results.get('correct', 0),
        'opt_runs':    opt,
    }


def phase3_council(gps_threshold: float = None,
                   model: str = None) -> dict:
    print(f"\n{'='*60}")
    print("  PHASE 3 -- COUNCIL RUN (EOD DATA)")
    print('='*60)

    import config
    if model:
        config.OLLAMA_MODEL = model

    if gps_threshold is None:
        state = load_state()
        gps_threshold = state.get('params', {}).get('gps_threshold', 6.0)

    print(f"  GPS threshold: {gps_threshold} | Mode: fast (5-bot chains)")

    from utils.llm import check_ollama
    if not check_ollama():
        print("\n  Ollama not running -- skipping council")
        print("     Start: ollama serve")
        return {'skipped': True, 'stocks': [], 'market': {}}
    # Orchestrator's _warmup_ollama() handles pre-loading with OOM fallback

    from pipeline.orchestrator import MarketOrchestrator, compute_gps
    start   = time.time()
    orch    = MarketOrchestrator()
    results = orch.run_full_pipeline(
        print_output=True,
        fast_mode=True,
        gps_threshold=gps_threshold,
    )

    # ── GPS Rescue Pass ────────────────────────────────────────
    # Sector caps limit stocks per sector to 2-8 based on sector score,
    # drawn from a momentum-ranked pool. High-GPS stocks with moderate
    # short-term momentum can be excluded even when GPS ≥ 7.5.
    # This pass scans every stock that WAS in the sector scan pool
    # (sr.top_stocks, now expanded to 10) but was NOT debated, and
    # debates any with GPS ≥ GPS_RESCUE_THRESHOLD.
    GPS_RESCUE_THRESHOLD = 7.5
    try:
        debated_symbols  = {s['symbol'] for s in results.get('stocks', [])}
        sector_results   = results.get('sectors', [])
        market_snap_dict = results.get('market', {})
        rescued_results  = []

        from memory.storage import load_fundamentals_json
        from scanner.sector_scanner import fetch_stock_quick_metrics

        rescue_candidates = []
        for sr in sector_results:
            sector_avg_1m = sr.avg_1m_change or 0
            for sym, _comp, _data in sr.top_stocks:
                if sym not in debated_symbols:
                    rescue_candidates.append((sym, sr.name, sector_avg_1m))

        if rescue_candidates:
            print(f"\n{'─'*60}")
            print(f"  GPS RESCUE: checking {len(rescue_candidates)} sector-capped stocks "
                  f"(threshold GPS ≥ {GPS_RESCUE_THRESHOLD})")

        for sym, sector_name, sector_avg_1m in rescue_candidates:
            try:
                pm   = fetch_stock_quick_metrics(sym, nifty_1m=0)
                fund = load_fundamentals_json(sym) or {}
                gps, components = compute_gps(sym, pm, fund, 0, sector_avg_1m)
                status = f"GPS={gps:.1f}"
                if gps >= GPS_RESCUE_THRESHOLD:
                    print(f"  [RESCUE] {sym:14} {status} ≥ {GPS_RESCUE_THRESHOLD} → debating now")
                    stock_info = {
                        'symbol':        sym,
                        'sector':        sector_name,
                        'gps':           gps,
                        'components':    components,
                        'price_metrics': pm,
                        'fundamentals':  fund,
                    }
                    res = orch.run_council_for_stock(
                        stock_info, market_snap_dict,
                        print_output=True, fast_mode=True,
                    )
                    res['rescued_by_gps'] = True
                    rescued_results.append(res)
                    debated_symbols.add(sym)
                else:
                    print(f"  [RESCUE] {sym:14} {status} < {GPS_RESCUE_THRESHOLD} — skip")
            except Exception as _re:
                print(f"  [RESCUE] {sym} error: {_re}")

        if rescued_results:
            print(f"\n  GPS Rescue added {len(rescued_results)} stock(s): "
                  f"{[r['symbol'] for r in rescued_results]}")
            results['stocks'] = results.get('stocks', []) + rescued_results

    except Exception as _e:
        print(f"  [GPS RESCUE] Error (non-fatal): {_e}")
    # ── end GPS Rescue ─────────────────────────────────────────

    elapsed = round(time.time() - start)
    stocks  = results.get('stocks', [])
    print(f"\n  Phase 3: {len(stocks)} stocks debated in "
          f"{elapsed//60}m {elapsed%60}s")

    return {
        **results,
        'stocks_debated': len(stocks),
        'elapsed_s':      elapsed,
    }


def phase4_portfolio(council_results: dict) -> dict:
    """
    Run the self-managing portfolio engine.
    Uses tonight's council results to make buy/sell/hold decisions.
    Updates portfolio_tracker.xlsx.
    """
    print(f"\n{'='*60}")
    print("  PHASE 4 -- PORTFOLIO ENGINE")
    print('='*60)

    if council_results.get('skipped'):
        print("  Council was skipped -- portfolio engine skipped too")
        return {'skipped': True}

    state = load_state()

    state = run_nightly_portfolio(
        council_results=council_results,
        state=state,
        print_output=True,
    )

    save_state(state)
    save_portfolio_excel(state, print_output=True)

    # Shadow logging — passive evidence collection only, runs AFTER
    # all real decisions above are finalized. Never feeds back into
    # tonight's buy/sell/hold logic. See shadow_log.py for design.
    try:
        run_shadow_logging(
            state=state,
            sector_results=council_results.get('sectors', []),
            nifty_1m=council_results.get('market', {}).get('nifty_1m', 0),
            print_output=True,
        )
    except Exception as e:
        import traceback
        print(f"  [SHADOW LOG] Error (non-critical, real portfolio unaffected): {e}")
        traceback.print_exc()

    pv  = _portfolio_value(state)
    pl  = pv - STARTING_CAPITAL
    day = state.get('day', 0)

    return {
        'day':              day,
        'portfolio_value':  pv,
        'total_pl':         pl,
        'total_return_pct': pl / STARTING_CAPITAL * 100,
        'open_positions':   len(state['positions']),
        'pending_orders':   len(state.get('pending_orders', [])),
        'cash':             state['cash'],
    }


def phase5_night_report(p1: dict, p2: dict, p3: dict,
                         p4: dict, day_num: int) -> dict:
    print(f"\n{'='*60}")
    print(f"  PHASE 5 -- NIGHT REPORT  (Learning Day {day_num})")
    print('='*60)

    from memory.storage import get_storage_summary
    storage  = get_storage_summary()
    progress = _load_progress()

    today = get_trading_date()
    acc   = p2.get('accuracy_1d')
    progress.setdefault('daily_log', []).append({
        'date':            today,
        'day':             day_num,
        'accuracy_1d':     acc,
        'stocks_debated':  p3.get('stocks_debated', 0),
        'opt_runs':        p2.get('opt_runs', 0),
        'prices_ok':       p1.get('prices_ok', 0),
        'portfolio_value': p4.get('portfolio_value', STARTING_CAPITAL),
        'portfolio_pl':    p4.get('total_pl', 0),
        'portfolio_ret':   p4.get('total_return_pct', 0),
    })
    progress['daily_log'] = progress['daily_log'][-21:]
    _save_progress(progress)

    recent = [d for d in progress['daily_log']
              if d.get('accuracy_1d') is not None][-7:]
    if len(recent) >= 2:
        accs  = [d['accuracy_1d'] for d in recent]
        arrow = ("UP" if accs[-1] > accs[0] else
                 "DOWN" if accs[-1] < accs[0] else "FLAT")
        print(f"\n  Accuracy: {' -> '.join(f'{a:.0f}%' for a in accs)} ({arrow})")
    elif acc:
        print(f"\n  Accuracy: {acc:.1f}%")
    else:
        print(f"\n  Accuracy: building... (need Day 2+)")

    if not p4.get('skipped'):
        ret = p4.get('total_return_pct', 0)
        print(f"\n  Portfolio Day {p4.get('day',0)}/21:")
        print(f"     Value:  Rs.{p4.get('portfolio_value',STARTING_CAPITAL):,.0f}")
        print(f"     P&L:    Rs.{p4.get('total_pl',0):+,.0f} "
              f"({ret:+.1f}%)")
        print(f"     Cash:   Rs.{p4.get('cash',0):,.0f}")
        print(f"     Open:   {p4.get('open_positions',0)} positions")
        print(f"     Pending:{p4.get('pending_orders',0)} orders "
              f"(fill at tomorrow's open)")

    print(f"\n  Storage: {storage['price_csvs']}/286 stocks | "
          f"{storage['council_sessions']} sessions | "
          f"{storage['total_size_mb']} MB")

    stocks = p3.get('stocks', [])
    if stocks:
        print(f"\n  Tonight's top picks:")
        for i, s in enumerate(stocks[:5], 1):
            print(f"    {i}. {s['symbol']:12} "
                  f"{s.get('verdict',''):12} "
                  f"score:{s.get('final_score',0):.1f}  "
                  f"GPS:{s.get('gps',0):.1f}")

    print(f"\n  Learning milestones:")
    _print_milestones(day_num, p2, recent)

    report = {
        'date':   today, 'day': day_num,
        'phase1': p1, 'phase2': p2,
        'phase3': {'stocks_debated': p3.get('stocks_debated', 0),
                   'elapsed_s': p3.get('elapsed_s', 0)},
        'phase4': p4,
        'storage': storage,
    }
    report_path = NIGHT_REPORTS_DIR / f"night_report_{today}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report: {report_path.name}")

    return report


def phase6_missed_opportunities(debated_symbols: set,
                                market_snap_dict: dict,
                                top_n: int = 5,
                                skip: bool = False) -> dict:
    """
    Phase 6 -- Missed Opportunity Analysis.

    Scans the curated ~250-stock universe for today's top gainers
    and losers, identifies genuine misses (movers not already
    debated tonight), and runs the EXACT SAME 5-bot fast-mode
    council on them (Option A -- identical methodology to the
    main pipeline, so results are directly comparable).

    ANALYTICAL ONLY -- never triggers a portfolio buy/sell.
    Restarts Ollama fresh first since the main pipeline already
    used significant RAM debating tonight's qualified stocks.

    Adds ~30-45 min per night. Findings accumulate in
    data/missed_opportunities.json and feed into the Day 14
    weekly review as evidence for structural parameter changes
    (e.g. should sector caps flex for very high individual GPS).
    """
    print(f"\n{'='*60}")
    print("  PHASE 6 -- MISSED OPPORTUNITY ANALYSIS")
    print('='*60)

    if skip:
        print("  Skipped (--skip-missed-opp flag)")
        return {'skipped': True}

    try:
        result = run_missed_opportunity_analysis(
            debated_symbols=debated_symbols,
            market_snap_dict=market_snap_dict,
            top_n=top_n,
            curated_only=True,
            skip_restart=False,   # always fresh Ollama for this phase
            dry_run=False,
            print_output=True,
        )
        return result
    except Exception as e:
        print(f"  [PHASE 6] Error: {e} -- continuing, main pipeline unaffected")
        return {'skipped': True, 'error': str(e)}


def _print_milestones(day_num: int, p2: dict, recent: list):
    milestones = [
        (7,  "ChromaDB 7+ sessions -- bots read own history"),
        (10, "Weight optimizer -- first auto-adjustment"),
        (14, "Score trends active -- GPS earnings_trend live"),
        (21, "Full baseline -- ready for live market mode"),
    ]
    for md, text in milestones:
        mark = "DONE" if day_num >= md else "....."
        print(f"    [{mark}]  Day {md:2}: {text}")

    days_left = max(0, 21 - day_num)
    if days_left > 0:
        print(f"\n    {days_left} nights remaining in learning mode.")
    else:
        acc = p2.get('accuracy_1d', 0) or 0
        opt = p2.get('opt_runs', 0)
        if acc >= 55 and opt >= 2:
            print(f"\n    READY FOR LIVE MODE")
            print(f"       Run: python run.py --schedule --model qwen2.5:7b")
        else:
            print(f"\n    Keep running (accuracy {acc:.0f}% | "
                  f"opt runs {opt})")


def run_phase3_with_watchdog(gps_threshold: float = None,
                              model: str = None) -> dict:
    """
    Run Phase 3 (council) with an Ollama hang watchdog.

    Strategy:
      - Wrap sys.stdout so every print() updates last_print_time.
      - A background watchdog thread checks every 60 s whether
        stdout has been silent for WATCHDOG_SILENCE_MIN minutes.
      - Silence ≥ threshold → Ollama is hung. The watchdog:
          1. Kills ollama.exe (this causes pending API calls to
             raise exceptions, which terminates the council thread).
          2. Waits up to 30 s for the council thread to die.
          3. Signals the main loop to restart Ollama and retry.
      - Phase 1 data (CSVs, fundamentals, news) is NEVER touched —
        only the in-progress council run is retried.
      - Retries up to COUNCIL_MAX_RETRIES times before giving up
        and returning an empty result so portfolio + report still run.
    """
    original_stdout = sys.stdout
    silence_secs    = WATCHDOG_SILENCE_MIN * 60

    for attempt in range(1, COUNCIL_MAX_RETRIES + 1):

        if attempt > 1:
            print(f"\n  🔄 WATCHDOG RETRY {attempt}/{COUNCIL_MAX_RETRIES} "
                  f"— restarting Ollama and retrying Phase 3...")
            restart_ollama()

        # Install stdout watchdog
        tracker     = _StdoutWatchdog(original_stdout)
        sys.stdout  = tracker

        result_box  = {'data': None, 'exc': None, 'done': False}
        stop_event  = threading.Event()

        def _council_worker():
            try:
                result_box['data'] = phase3_council(
                    gps_threshold=gps_threshold, model=model)
            except Exception as e:
                result_box['exc'] = e
            finally:
                result_box['done'] = True
                stop_event.set()

        def _watchdog_worker():
            while not stop_event.is_set():
                stop_event.wait(timeout=60)   # check every 60 s
                if stop_event.is_set():
                    break
                silence = time.time() - tracker.last_print_time
                if silence >= silence_secs:
                    sys.stdout = original_stdout  # restore before printing
                    print(f"\n  ⚠️  WATCHDOG: No output for "
                          f"{silence/60:.1f} min — Ollama appears hung.")
                    print(f"  ⚠️  Killing ollama.exe and scheduling retry...")
                    subprocess.run(
                        ["taskkill", "/f", "/im", "ollama.exe"],
                        shell=True, capture_output=True
                    )
                    stop_event.set()   # unblock watchdog loop
                    break

        council_thread  = threading.Thread(target=_council_worker,  daemon=True)
        watchdog_thread = threading.Thread(target=_watchdog_worker, daemon=True)

        council_thread.start()
        watchdog_thread.start()

        # Wait for council up to 5 hours.
        # The orchestrator now self-heals: each per-stock timeout kills Ollama
        # immediately (unblocking the zombie thread within seconds) and restarts
        # it before the next stock. ChromaDB and NSE calls also have hard timeouts
        # (_safe_vec / _safe_net). So council_thread will always finish eventually.
        # The old 780s (13 min) limit was shorter than STOCK_TIMEOUT_S (15 min),
        # causing new Phase 3 instances to spawn before the old one recovered —
        # multiple parallel instances → OOM. 5 hours covers any realistic run.
        council_thread.join(timeout=5 * 60 * 60)
        stop_event.set()               # stop watchdog whether we finished or not
        watchdog_thread.join(timeout=5)

        # Restore stdout
        sys.stdout = original_stdout

        if result_box['done'] and result_box['data'] is not None:
            return result_box['data']  # ✅ success

        if result_box['exc']:
            # Real exception (not hang-kill) — still retry, LLM errors are transient
            print(f"  [WATCHDOG] Phase 3 error on attempt {attempt}: "
                  f"{result_box['exc']}")

        if attempt < COUNCIL_MAX_RETRIES:
            print(f"  [WATCHDOG] Will retry Phase 3 in 10 s...")
            time.sleep(10)

    print(f"\n  ⚠️  WATCHDOG: Phase 3 failed after {COUNCIL_MAX_RETRIES} attempts.")
    print(f"  ⚠️  Returning empty council — portfolio + report will still run.")
    return {'skipped': True, 'stocks': [], 'market': {}, 'sectors': []}


def run_tonight(model: str = None, data_only: bool = False,
                gps: float = None, force: bool = False,
                skip_ollama_restart: bool = False,
                skip_missed_opp: bool = False):
    """
    Run all 6 phases for one trading day.

    Always restarts Ollama first (unless skip_ollama_restart=True
    or data_only=True, since data-only never touches the LLM).
    Phase 6 (missed opportunities) restarts Ollama again on its
    own before it starts -- the main pipeline already used
    significant RAM debating tonight's qualified stocks.
    """
    progress = _load_progress()
    day_num  = len(progress.get('daily_log', [])) + 1

    # Duplicate guard: if this trading date already has a completed entry,
    # refuse to run again (would create a second entry for the same date,
    # advancing the day counter incorrectly). Use --force to override.
    trading_date = get_trading_date()
    already_ran = any(e.get('date') == trading_date
                      for e in progress.get('daily_log', []))
    if already_ran and not force:
        print(f"\n  ⚠️  Already ran for trading date {trading_date} (Day "
              f"{next((e['day'] for e in progress['daily_log'] if e.get('date') == trading_date), '?')}) "
              f"— skipping to avoid duplicate entry.")
        print(f"  Use --force to override and run again.")
        return

    print_banner(f"Learning Day {day_num} / 21")

    if not skip_ollama_restart and not data_only:
        restart_ollama()

    p1 = phase1_data_refresh(force=force)

    if data_only:
        print("\n  --data-only: skipping phases 2-6")
        return

    p2 = phase2_backtest()
    p3 = run_phase3_with_watchdog(gps_threshold=gps, model=model)
    p4 = phase4_portfolio(p3)
    phase5_night_report(p1, p2, p3, p4, day_num)

    # ── Phase 4B: Model B (Daily Conviction Budget) ───────────
    # Runs independently of Model A — same council output, separate state.
    # Backtest (Days 1-13) pre-loaded via run_backtest_v2.py.
    # From Day 14 onward this updates portfolio_state_v2.json live.
    try:
        from portfolio_engine_v2 import ModelBEngine
        from memory.storage import load_prices_csv
        import pandas as pd

        _b_stocks = p3.get('stocks', [])
        if _b_stocks and not p3.get('skipped'):
            print(f"\n{'='*60}")
            print("  PHASE 4B -- MODEL B (Daily Conviction Budget)")
            print(f"{'='*60}")

            # Build price maps from cached CSVs
            _b_price_close = {}   # today's close (MTM)
            _b_next_open   = {}   # next trading day open (buy execution)
            _today_str     = p3['stocks'][0].get('date', get_trading_date()) if p3['stocks'] else get_trading_date()

            for _sym in {s['symbol'] for s in _b_stocks if 'symbol' in s}:
                try:
                    _df = load_prices_csv(_sym)
                    if _df is None or _df.empty:
                        continue
                    _df = _df[pd.to_numeric(_df['Open'], errors='coerce').notna()].copy()
                    _df['Date'] = pd.to_datetime(_df['Date'], errors='coerce')
                    _df = _df.dropna(subset=['Date']).sort_values('Date')
                    _df['ds'] = _df['Date'].dt.strftime('%Y-%m-%d')
                    _df = _df.set_index('ds')
                    # Today's close
                    if _today_str in _df.index:
                        _b_price_close[_sym] = float(_df.loc[_today_str, 'Close'])
                    # Next day's open (the last row after today)
                    _future = _df[_df.index > _today_str]
                    if not _future.empty:
                        _b_next_open[_sym] = float(_future.iloc[0]['Open'])
                except Exception:
                    pass

            _engine = ModelBEngine()
            _engine.run_day(
                date_str       = _today_str,
                council_stocks = _b_stocks,
                price_map      = _b_price_close,
                next_open_map  = _b_next_open,
                prev_open_map  = _b_price_close,   # use close as proxy if no separate open
                trading_day_n  = day_num,
                print_output   = True,
            )
            _engine.save()

            # Quick A vs B comparison
            _a_val  = p4.get('portfolio_value', 0)
            _a_ret  = p4.get('total_return_pct', 0)
            _b_val  = _engine.state['portfolio_value']
            _b_ret  = _engine.state['daily_log'][-1].get('total_return_pct', 0) if _engine.state['daily_log'] else 0
            print(f"\n  📊 A vs B:  Model A ₹{_a_val:,.0f} ({_a_ret:+.2f}%)  |  Model B ₹{_b_val:,.0f} ({_b_ret:+.2f}%)")
    except Exception as _b_err:
        print(f"\n  [MODEL B] Non-critical error: {_b_err}")

    # Phase 6 -- runs after everything else, with its own Ollama
    # restart. Adds ~30-45 min. Analytical only, no portfolio impact.
    debated_symbols = {s.get('symbol') for s in p3.get('stocks', [])
                       if s.get('symbol')}
    p6 = phase6_missed_opportunities(
        debated_symbols=debated_symbols,
        market_snap_dict=p3.get('market', {}),
        skip=skip_missed_opp,
    )

    total_s  = (p1.get('elapsed_s', 0) + p3.get('elapsed_s', 0) +
                p6.get('elapsed_s', 0))
    next_run = f"{RUN_HOUR:02d}:{RUN_MINUTE:02d}"
    print(f"\n  Total: {total_s//60}m {total_s%60}s | "
          f"Next run: tomorrow {next_run} IST\n")


def _run_sunday_review():
    """Called by scheduler at 7 PM Sunday."""
    progress = _load_progress()
    day_num  = len(progress.get('daily_log', []))

    if day_num < 7:
        print(f"  [REVIEW] Only {day_num} days in -- skipping "
              f"(need 7+ days)")
        return

    week_num = min((day_num // 7), 2)   # week 3 is frozen (validation)
    if week_num == 0:
        return

    print(f"\n  Sunday Review starting for Week {week_num}...")
    state = load_state()
    state = run_weekly_review(state, week_num,
                              timeout_minutes=REVIEW_TIMEOUT)
    save_state(state)


def run_scheduled(model: str = None, gps: float = None,
                  skip_missed_opp: bool = False):
    """
    Auto-run every weekday at 5:00 PM IST + Sunday review at 7 PM IST.
    Skips weekends and NSE holidays automatically.
    """
    run_time    = f"{RUN_HOUR:02d}:{RUN_MINUTE:02d}"
    review_time = f"{REVIEW_HOUR:02d}:{REVIEW_MINUTE:02d}"

    print(f"""
  Auto scheduler active.
  Weekday council + portfolio: {run_time} IST  (90 min after NSE close)
  Sunday parameter review:     {review_time} IST
  Ollama auto-restart:         before every run
  NSE holidays:                skipped automatically
  Press Ctrl+C to stop
""")

    def _daily_job():
        now = datetime.now(IST)
        if now.weekday() >= 5:
            print(f"  Weekend -- skipping")
            return
        if is_market_holiday(now.date()):
            print(f"  Market holiday ({now.date()}) -- skipping")
            return
        run_tonight(model=model, gps=gps, skip_missed_opp=skip_missed_opp)

    def _sunday_review():
        now = datetime.now(IST)
        if now.weekday() != 6:
            return
        _run_sunday_review()

    schedule.every().day.at(run_time).do(_daily_job)
    schedule.every().day.at(review_time).do(_sunday_review)

    now = datetime.now(IST)
    past_run_time = (now.hour > RUN_HOUR or
                     (now.hour == RUN_HOUR and now.minute >= RUN_MINUTE))

    if now.weekday() < 5 and past_run_time and not is_market_holiday(now.date()):
        print(f"  Past {run_time} IST -- running today's job now...")
        run_tonight(model=model, gps=gps, skip_missed_opp=skip_missed_opp)
    elif now.weekday() < 5 and past_run_time:
        print(f"  Today is a market holiday -- skipping immediate run")
    else:
        print(f"  Waiting for {run_time} IST...")

    try:
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n  Scheduler stopped.")


def show_learning_report(week: int = None):
    """Print full learning + portfolio progress."""
    progress = _load_progress()
    log      = progress.get('daily_log', [])

    if not log:
        print("\n  No data yet. Run: python night_runner.py")
        return

    print(f"\n{'='*60}")
    print(f"  LEARNING + PORTFOLIO PROGRESS REPORT")
    print('='*60)

    if week:
        s   = (week - 1) * 7 + 1
        e   = week * 7
        log = [d for d in log if s <= d.get('day', 0) <= e]
        print(f"  Week {week} (Days {s}-{e})")

    print(f"\n  {'Date':12} {'Day':4} {'Acc':8} "
          f"{'Stocks':8} {'Portfolio':12} {'P&L':10}")
    print(f"  {'-'*58}")
    for d in log:
        acc_s = f"{d['accuracy_1d']:.0f}%" if d.get('accuracy_1d') else "--"
        stk_s = str(d.get('stocks_debated', 0))
        pv_s  = f"Rs.{d.get('portfolio_value',STARTING_CAPITAL):,.0f}"
        pl_s  = (f"Rs.{d.get('portfolio_pl',0):+,.0f} "
                 f"({d.get('portfolio_ret',0):+.1f}%)")
        print(f"  {d['date']:12} {d['day']:4} {acc_s:8} "
              f"{stk_s:8} {pv_s:12} {pl_s}")

    state = load_state()
    pv    = _portfolio_value(state)
    pl    = pv - STARTING_CAPITAL
    print(f"\n  Current portfolio: Rs.{pv:,.0f} | "
          f"P&L: Rs.{pl:+,.0f} ({pl/STARTING_CAPITAL*100:+.1f}%)")
    print(f"  Open: {len(state['positions'])} | "
          f"Closed: {len(state.get('closed_trades',[]))} trades | "
          f"Day: {state.get('day',0)}/21")

    from bots.prediction_bot import load_learned_weights
    learned  = load_learned_weights()
    opt_runs = learned.get('optimization_count', 0)
    print(f"\n  Weight optimizer runs: {opt_runs}")

    wc = state.get('weight_changes', [])
    if wc:
        print(f"  Last param change:")
        last = wc[-1]
        print(f"    {last['param']}: {last['old']} -> {last['new']}")
        print(f"    Reason: {last['reason']}")


def _load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {'daily_log': [], 'started': get_trading_date()}


def _save_progress(data: dict):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(
        description='Night Runner -- 3-Week Learning + Portfolio',
        epilog="""
COMMANDS:
  python night_runner.py                     # run now (manual)
  python night_runner.py --sche
  python night_runner.py --schedule          # auto every day 5 PM IST
  python night_runner.py --data-only         # data refresh only (no LLM)
  python night_runner.py --report            # learning + portfolio progress
  python night_runner.py --week 1            # Week 1 summary
  python night_runner.py --model qwen2.5:7b  # override LLM model
  python night_runner.py --gps 5.5           # override GPS threshold
  python night_runner.py --no-restart        # skip Ollama restart this run
  python night_runner.py --skip-missed-opp   # skip Phase 6 (~30-45 min saved)
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--schedule',        action='store_true',
                        help='Run on auto schedule (5 PM IST weekdays)')
    parser.add_argument('--report',          action='store_true',
                        help='Show learning + portfolio progress report')
    parser.add_argument('--portfolio',       action='store_true',
                        help='Show portfolio summary')
    parser.add_argument('--model',           type=str, default=None,
                        help='Override LLM model (e.g. qwen2.5:7b)')
    parser.add_argument('--gps',             type=float, default=None,
                        help='Override GPS threshold (default: 6.0)')
    parser.add_argument('--force',           action='store_true',
                        help='Force run even if already ran today')
    parser.add_argument('--data-only',       action='store_true',
                        dest='data_only',
                        help='Data refresh only -- skip LLM council and portfolio')
    parser.add_argument('--no-restart',      action='store_true',
                        dest='no_restart',
                        help='Skip Ollama restart this run')
    parser.add_argument('--skip-missed-opp', action='store_true',
                        dest='skip_missed_opp',
                        help='Skip Phase 6 (missed opportunity analysis) -- saves ~30-45 min')
    parser.add_argument('--week',            type=int, default=None,
                        help='Week number for --report')

    args = parser.parse_args()

    if args.report:
        show_learning_report(week=args.week)
        return

    if args.portfolio:
        state = load_state()
        pv = _portfolio_value(state)
        pl = pv - STARTING_CAPITAL
        print(f"\n  Portfolio: Rs.{pv:,.0f} | P&L: Rs.{pl:+,.0f} ({pl/STARTING_CAPITAL*100:+.1f}%)")
        print(f"  Open positions: {len(state['positions'])}")
        return

    if args.model:
        print(f"  [CONFIG] Model: {args.model}")

    if args.schedule:
        run_scheduled(model=args.model, gps=args.gps,
                      skip_missed_opp=args.skip_missed_opp)
    else:
        run_tonight(model=args.model, gps=args.gps,
                    data_only=args.data_only,
                    force=args.force,
                    skip_ollama_restart=args.no_restart,
                    skip_missed_opp=args.skip_missed_opp)


if __name__ == '__main__':
    main()

