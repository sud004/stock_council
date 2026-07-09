#!/usr/bin/env python3
# ============================================================
# scan.py — Master top-down market scanner
# ============================================================
# USAGE:
#   python scan.py                    # full scan (all sectors → top stocks)
#   python scan.py --fast             # fast mode (no LLM per stock, just quant)
#   python scan.py --sectors 5        # scan top 5 sectors instead of 3
#   python scan.py --top 10           # full analysis on top 10 stocks
#   python scan.py --schedule         # run every hour during market hours
#   python scan.py --sector "IT"      # scan only one sector
#   python scan.py --no-save          # don't save report
#
# FLOW:
#   Level 1: Market scan (NSE + global cues + FII/DII + VIX)
#   Level 2: All 13 sectors scored and ranked
#   Level 3: All stocks in top 3 sectors deep-scanned
#             → Top 5 get full 5-bot LLM analysis
#   Report: Terminal + JSON saved to reports/
# ============================================================

import sys
import time
import argparse
import schedule
import pytz
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from scanner import market_scanner, sector_scanner, stock_scanner, report
from utils.llm import check_ollama
from live_apis import APIKeys

IST = pytz.timezone('Asia/Kolkata')


def print_startup_info(args):
    print("""
╔══════════════════════════════════════════════════════════════╗
║   🌐  NSE / BSE MARKET SCANNER — TOP-DOWN AI ANALYSIS      ║
║   Market → All 13 Sectors → Top Stocks                     ║
╚══════════════════════════════════════════════════════════════╝""")
    print(f"\n  Started:     {datetime.now(IST).strftime('%d %b %Y %H:%M IST')}")
    print(f"  Mode:        {'FAST (quant only)' if args.fast else 'FULL (LLM + quant)'}")
    print(f"  Top sectors: {args.sectors}")
    print(f"  Full LLM on: top {args.top} stocks")

    # API status
    print("\n  API Status:")
    for name, status in APIKeys.status().items():
        print(f"    {name:20} {status}")

    print()


def run_full_scan(args) -> dict:
    """
    Run the complete top-down scan.
    Returns dict with market, sectors, stocks results.
    """
    scan_start = time.time()

    # ── Level 1: Market ────────────────────────────────────────
    print(f"\n⏱  [{datetime.now(IST).strftime('%H:%M')}] Starting Level 1: Market scan...")
    market_snap = market_scanner.run(print_output=True)

    # ── Level 2: All sectors ───────────────────────────────────
    print(f"\n⏱  [{datetime.now(IST).strftime('%H:%M')}] Starting Level 2: Sector scan...")
    sector_results = sector_scanner.run(
        market_snap,
        priority_only=args.fast,     # fast mode uses fewer stocks per sector
        print_output=True
    )

    # Filter to specific sector if requested
    if args.sector:
        query = args.sector.lower()
        sector_results = [s for s in sector_results
                          if query in s.name.lower()]
        if not sector_results:
            print(f"  ⚠ No sector matching '{args.sector}' found")
            sector_results = sector_scanner.run(market_snap)[:3]

    # ── Level 3: Stocks ────────────────────────────────────────
    print(f"\n⏱  [{datetime.now(IST).strftime('%H:%M')}] Starting Level 3: Stock scan...")
    stock_verdicts = stock_scanner.run(
        sector_results,
        market_snap,
        top_n_sectors=args.sectors,
        full_llm_for_top=0 if args.fast else args.top,
        print_output=True
    )

    # ── Report ────────────────────────────────────────────────
    print(f"\n⏱  [{datetime.now(IST).strftime('%H:%M')}] Generating report...")
    report_path = report.generate(
        market_snap, sector_results, stock_verdicts,
        save=not args.no_save
    )

    elapsed = round(time.time() - scan_start)
    mins, secs = divmod(elapsed, 60)
    print(f"\n  ✅ Scan complete in {mins}m {secs}s")

    return {
        "market": market_snap,
        "sectors": sector_results,
        "stocks": stock_verdicts,
        "report_path": report_path,
        "elapsed_seconds": elapsed,
    }


def run_scheduled(args):
    """Run scan on schedule during market hours."""
    print(f"\n[SCHEDULER] Market scan schedule:")
    print(f"  Pre-market:  9:00 AM IST")
    print(f"  Market scan: 9:30, 11:00, 12:30, 2:00, 3:30 PM IST")
    print(f"  Post-market: 4:00 PM IST")
    print(f"\n[SCHEDULER] Running now and then on schedule...")
    print(f"  Press Ctrl+C to stop\n")

    def _scan():
        now = datetime.now(IST)
        print(f"\n[{now.strftime('%H:%M IST')}] 🔔 Scheduled scan starting...")
        try:
            run_full_scan(args)
        except Exception as e:
            print(f"[SCHEDULER] Error: {e}")

    # Run immediately first
    _scan()

    # Schedule for market hours (IST)
    scan_times = ["09:00", "09:30", "11:00", "12:30", "14:00", "15:30", "16:00"]
    for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']:
        for t in scan_times:
            getattr(schedule.every(), day).at(t).do(_scan)

    print(f"\n[SCHEDULER] Next scan: {schedule.next_run()}")

    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n[SCHEDULER] Stopped.")


def main():
    parser = argparse.ArgumentParser(
        description='NSE/BSE Top-Down Market Scanner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scan.py                     # Full scan
  python scan.py --fast              # Fast mode (5 min on CPU)
  python scan.py --sectors 5 --top 10   # Scan 5 sectors, analyse top 10 stocks
  python scan.py --sector IT         # Only IT sector deep dive
  python scan.py --schedule          # Hourly scanner
        """
    )
    parser.add_argument('--fast', action='store_true',
                        help='Fast mode: quant only, no LLM per stock (~5 min on CPU)')
    parser.add_argument('--sectors', type=int, default=3,
                        help='How many top sectors to deep-dive (default: 3)')
    parser.add_argument('--top', type=int, default=5,
                        help='Full LLM analysis for top N stocks (default: 5)')
    parser.add_argument('--sector', type=str,
                        help='Scan only this sector (e.g. "IT" or "Banks")')
    parser.add_argument('--schedule', action='store_true',
                        help='Run every hour during market hours (9 AM–4 PM IST)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save report to disk')
    parser.add_argument('--offline', action='store_true',
                        help='Offline mode: use cached data only')
    parser.add_argument('--model', type=str,
                        help='Ollama model to use (e.g. phi3:mini, mistral)')

    args = parser.parse_args()

    # Apply overrides
    if args.offline:
        import config
        config.ALLOW_INTERNET = False
        print("[CONFIG] Offline mode — using cached data")

    if args.model:
        import config
        config.OLLAMA_MODEL = args.model
        print(f"[CONFIG] Model: {args.model}")

    # Check Ollama
    if not args.fast:
        if not check_ollama():
            print("\n[ERROR] Ollama is not running!")
            print("  Start it: ollama serve")
            print("  OR run fast mode: python scan.py --fast")
            sys.exit(1)

    print_startup_info(args)

    if args.schedule:
        run_scheduled(args)
    else:
        run_full_scan(args)


if __name__ == "__main__":
    main()
