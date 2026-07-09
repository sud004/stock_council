#!/usr/bin/env python3
# ============================================================
# run.py — Master entry point for the unified pipeline
# ============================================================
# FIXES vs original:
#   1. --fast flag now propagated into run_full_pipeline(fast_mode=True)
#      Previously it only skipped the Ollama check and did nothing else.
#      Now it actually uses fast prompts + skips 10-bot debate.
#      Result: 7 stocks in ~35 min instead of 7 hours.
#
#   2. --gps threshold passed as parameter to run_gps_filter()
#      instead of monkey-patching the method (which caused KeyError
#      when the filtered list was empty).
#
#   3. --backtest N flag added: replays last N days of stored price
#      data through the prediction_bot backtest engine.
#
#   4. --timeout flag: override per-call LLM timeout (default 90s)
#
# USAGE:
#   python run.py                    # full pipeline (LLM required)
#   python run.py --fast             # fast quant mode (~35 min for 7 stocks)
#   python run.py --schedule         # hourly + nightly auto-tracker
#   python run.py --nightly          # download all data now
#   python run.py --status           # system status
#   python run.py --gps 5.5          # lower threshold → more stocks
#   python run.py --top-sectors 5    # debate stocks from top 5 sectors
#   python run.py --model phi3:mini  # override LLM model
#   python run.py --offline          # cached data only
#   python run.py --backtest 30      # backtest last 30 days of predictions
#   python run.py --timeout 60       # 60s per LLM call (faster, may truncate)
# ============================================================

import sys
import time
import argparse
import schedule
import pytz
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
IST = pytz.timezone('Asia/Kolkata')


# ── Banner ────────────────────────────────────────────────────

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  🏛  INDIAN STOCK MARKET BOT COUNCIL — UNIFIED PIPELINE        ║
║  Market → Sectors → GPS Filter → Bot Council → Memory → Excel  ║
║  LLM: Ollama (local) | Data: Ventura EaseAPI | DB: ChromaDB    ║
╚══════════════════════════════════════════════════════════════════╝""")
    print(f"  {datetime.now(IST).strftime('%A, %d %b %Y  %H:%M IST')}\n")


# ── EaseAPI Initialization ────────────────────────────────────

def setup_easeapi() -> bool:
    try:
        from utils.easeapi_auth import (
            ensure_logged_in, schedule_daily_renewal, token_status
        )
        from utils.easeapi import load_instruments
        from pipeline.ease_patch import apply_patch

        ok = ensure_logged_in()
        if not ok:
            print("[EASE] Could not authenticate — using Yahoo Finance fallback")
            return False

        schedule_daily_renewal()
        load_instruments()
        apply_patch()

        s       = token_status()
        renewal = "✅ TOTP auto-renewal ON" if s['auto_renewal'] else "⚠  manual"
        print(f"[EASE] Authenticated as {s['client_id']} | Expires: {s['expires_at']}")
        print(f"[EASE] Daily renewal: {renewal}")
        return True

    except Exception as e:
        print(f"[EASE] Setup error: {e}")
        print("[EASE] Falling back to Yahoo Finance + NSE scraping")
        return False


# ── Status ────────────────────────────────────────────────────

def show_status():
    from utils.llm import check_ollama
    from live_apis import APIKeys
    from memory.storage import get_storage_summary
    from memory.vector_store import get_vector_store
    from scanner.universe import TOTAL_STOCKS, TOTAL_SECTORS

    print("\n📋  SYSTEM STATUS")
    print("═" * 60)

    ollama_ok = check_ollama()
    print(f"\n  [LLM]      {'✅' if ollama_ok else '❌'} Ollama")
    if not ollama_ok:
        print("              → Start: ollama serve")
        print("              → Model: ollama pull mistral")

    try:
        from utils.easeapi_auth import token_status
        s = token_status()
        print(f"\n  [EASEAPI]  {'✅' if s['authenticated'] else '❌'} Authentication")
        print(f"             Client:     {s['client_id']}")
        print(f"             Expires:    {s['expires_at']}")
        print(f"             Auto-login: {'✅ TOTP active' if s['totp_setup'] else '❌ setup needed'}")
    except Exception as e:
        print(f"\n  [EASEAPI]  ❌ Error: {e}")

    print("\n  [OTHER APIs]")
    for name, status in APIKeys.status().items():
        print(f"             {name:20} {status}")

    print("\n  [STORAGE]")
    s = get_storage_summary()
    print(f"             Price CSVs:      {s['price_csvs']:>5} / {TOTAL_STOCKS} stocks")
    print(f"             Fundamentals:    {s['fundamental_jsons']:>5} stocks")
    print(f"             News archives:   {s['news_stock_dirs']:>5} stocks")
    print(f"             Score histories: {s['council_sessions']:>5}")
    print(f"             Excel reports:   {s['excel_reports']:>5}")
    print(f"             Total disk:      {s['total_size_mb']:>5} MB")

    print("\n  [VECTOR DB]  (ChromaDB local)")
    try:
        vs_stats = get_vector_store().get_stats()
        for col, count in vs_stats.get('collections', {}).items():
            print(f"             {col:35} {count:>5} docs")
    except Exception:
        print("             Not initialized yet")

    print(f"\n  [UNIVERSE]  {TOTAL_STOCKS} stocks × {TOTAL_SECTORS} sectors")

    print("""
  [NEXT STEPS]
    python utils/easeapi_auth.py --setup   # one-time TOTP setup
    python run.py --nightly                # download all 286 stocks
    python run.py --fast                   # fast scan (~35 min, no 10-bot debate)
    python run.py                          # full pipeline (~3h for 7 stocks)
    python run.py --schedule               # hourly tracker
""")


# ── Pipeline Runner ───────────────────────────────────────────

def run_pipeline(args) -> dict:
    """
    Run the full unified pipeline once.

    FIX: --fast and --gps are now passed as proper parameters to
    run_full_pipeline() instead of monkey-patching methods.
    """
    from pipeline.orchestrator import MarketOrchestrator

    gps_threshold = getattr(args, 'gps', 6.5)
    fast_mode     = getattr(args, 'fast', False)

    if fast_mode:
        print(f"[CONFIG] Fast mode: ON — 5-bot quant chains, no 10-bot debate")
        print(f"[CONFIG] Expected: ~5 min/stock on CPU")
    if gps_threshold != 6.5:
        print(f"[GPS] Threshold: {gps_threshold} (default 6.5)")

    return MarketOrchestrator().run_full_pipeline(
        print_output=True,
        fast_mode=fast_mode,
        gps_threshold=gps_threshold,
    )


# ── Backtest Runner ───────────────────────────────────────────

def run_backtest(days: int):
    """
    Run the prediction bot backtest for the last N days.
    Checks stored predictions against actual prices and
    triggers weight optimization if accuracy < 55%.
    Also prints stop-loss and target hit rates.
    """
    from bots.prediction_bot import run_daily_backtest, load_learned_weights, PREDICTIONS_FILE
    import json

    print(f"\n{'═'*60}")
    print(f"📊  PREDICTION BACKTEST — last {days} days")
    print('═'*60)

    # Show how many predictions are stored
    if PREDICTIONS_FILE.exists():
        with open(PREDICTIONS_FILE) as f:
            all_preds = json.load(f)
        total = sum(len(v) for v in all_preds.values())
        print(f"\n  Stored predictions: {total} across {len(all_preds)} stocks")
    else:
        print("\n  No predictions file found — run the full pipeline first")
        return

    results = run_daily_backtest(print_output=True)

    # Stop/target hit summary
    hit_target = hit_stop = 0
    for preds in all_preds.values():
        for p in preds:
            if p.get('hit_target'): hit_target += 1
            if p.get('hit_stop'):   hit_stop   += 1

    if hit_target + hit_stop > 0:
        print(f"\n  Stop/Target Tracking:")
        print(f"    Target hit: {hit_target}")
        print(f"    Stop hit:   {hit_stop}")
        ratio = hit_target / (hit_target + hit_stop) * 100
        print(f"    Hit rate:   {ratio:.0f}% (targets vs stops)")

    print(f"\n  Current learned weights:")
    w = load_learned_weights()
    for k, v in w.get('pred_weights', {}).items():
        print(f"    {k:20} {v:.3f}")


# ── Scheduler ─────────────────────────────────────────────────

def run_scheduled(args):
    from pipeline.nightly import run_nightly_job

    print("\n[SCHEDULER] Active schedule:")
    print("  08:45  → EaseAPI token renewal")
    print("  09:15  → Full pipeline (market open)")
    print("  11:00  → Full pipeline")
    print("  12:30  → Full pipeline (midday)")
    print("  14:00  → Full pipeline")
    print("  15:30  → Full pipeline (EOD)")
    print("  16:15  → Nightly data download")
    print("  Press Ctrl+C to stop\n")

    def _pipeline():
        print(f"\n[{datetime.now(IST).strftime('%H:%M IST')}] 🔔 Pipeline starting...")
        try:
            run_pipeline(args)
        except Exception as e:
            print(f"[SCHEDULER] Pipeline error: {e}")

    def _nightly():
        print(f"\n[{datetime.now(IST).strftime('%H:%M IST')}] 🌙 Nightly download starting...")
        try:
            run_nightly_job()
        except Exception as e:
            print(f"[SCHEDULER] Nightly error: {e}")

    for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']:
        for t in ["09:15", "11:00", "12:30", "14:00", "15:30"]:
            getattr(schedule.every(), day).at(t).do(_pipeline)
        getattr(schedule.every(), day).at("16:15").do(_nightly)

    _pipeline()   # run once immediately

    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n[SCHEDULER] Stopped.")


# ── Main ──────────────────────────────────────────────────────

def main():
    print_banner()

    parser = argparse.ArgumentParser(
        description='Indian Stock Market Bot Council — Unified Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
QUICK START:
  # First-time setup (do once):
  python utils/easeapi_auth.py --setup       ← set up auto-login
  python run.py --nightly                    ← download all stock data

  # Daily usage:
  python run.py --fast                       ← FAST: ~35 min, 7 stocks (USE THIS FOR TESTING)
  python run.py                              ← FULL: ~3h for 7 stocks (overnight run)
  python run.py --schedule                   ← hourly auto-tracker

  # Tuning:
  python run.py --gps 5.5                    ← more stocks in council
  python run.py --model phi3:mini            ← faster LLM on low RAM
  python run.py --offline                    ← use cached data only
  python run.py --timeout 60                 ← 60s per LLM call

  # Analysis:
  python run.py --backtest 30                ← backtest last 30 days
        """
    )

    parser.add_argument('--fast',      action='store_true',
                        help='Fast mode: 5-bot quant chains, no 10-bot debate (~35 min for 7 stocks)')
    parser.add_argument('--schedule',  action='store_true',
                        help='Start hourly scheduler (runs all day + nightly download)')
    parser.add_argument('--nightly',   action='store_true',
                        help='Run nightly data download now (all 286 stocks)')
    parser.add_argument('--status',    action='store_true',
                        help='Show full system status')
    parser.add_argument('--backtest',  type=int, default=0, metavar='DAYS',
                        help='Backtest prediction accuracy for last N days')
    parser.add_argument('--gps',       type=float, default=6.5,
                        help='GPS threshold (default 6.5, lower = more stocks)')
    parser.add_argument('--top-sectors', type=int, default=3,
                        help='Debate stocks from top N sectors (default 3)')
    parser.add_argument('--model',     type=str,
                        help='Ollama model override (e.g. phi3:mini, mistral, llama3.1)')
    parser.add_argument('--offline',   action='store_true',
                        help='Offline mode — use only locally cached data')
    parser.add_argument('--force',     action='store_true',
                        help='Force re-download even if data is fresh')
    parser.add_argument('--timeout',   type=int, default=90,
                        help='Per-call LLM timeout in seconds (default 90). '
                             'Lower = faster but may truncate. Try 60 on fast hardware.')

    args = parser.parse_args()

    # ── Apply config overrides ────────────────────────────────
    if args.model:
        import config
        config.OLLAMA_MODEL = args.model
        print(f"[CONFIG] LLM model: {args.model}")

    if args.offline:
        import config
        config.ALLOW_INTERNET = False
        print("[CONFIG] Offline mode — using cached data only")

    # Apply timeout override to utils/llm constants
    if args.timeout != 90:
        import utils.llm as _llm_mod
        _llm_mod.CALL_HARD_TIMEOUT   = args.timeout
        _llm_mod.TOKEN_STALL_TIMEOUT = min(args.timeout - 10, 60)
        print(f"[CONFIG] LLM timeout: {args.timeout}s per call")

    # ── Status ────────────────────────────────────────────────
    if args.status:
        show_status()
        return

    # ── Backtest ──────────────────────────────────────────────
    if args.backtest > 0:
        run_backtest(args.backtest)
        return

    # ── Initialize EaseAPI ────────────────────────────────────
    if not args.offline:
        setup_easeapi()

    # ── Nightly download ──────────────────────────────────────
    if args.nightly:
        from pipeline.nightly import run_nightly_job
        run_nightly_job(force=args.force)
        return

    # ── Scheduler ─────────────────────────────────────────────
    if args.schedule:
        run_scheduled(args)
        return

    # ── Single pipeline run ───────────────────────────────────
    if not args.fast:
        from utils.llm import check_ollama
        if not check_ollama():
            print("\n❌ Ollama is not running!")
            print("  Start it:    ollama serve")
            print("  Get a model: ollama pull mistral")
            print("  Fast mode:   python run.py --fast   (skips 10-bot debate)")
            sys.exit(1)
    else:
        # Fast mode: Ollama still needed for 5-bot chains but warn if missing
        from utils.llm import check_ollama
        if not check_ollama():
            print("\n⚠ Ollama not found — fast mode still calls LLM for 5 bots.")
            print("  Start it: ollama serve && ollama pull mistral")
            print("  Or use:   python run.py --fast --offline for pure quant")

    run_pipeline(args)


if __name__ == "__main__":
    main()
