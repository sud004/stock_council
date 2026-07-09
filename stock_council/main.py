#!/usr/bin/env python3
# ============================================================
# main.py — Indian Stock Market Bot Council
# ============================================================
# USAGE:
#   python main.py RELIANCE              # single stock analysis
#   python main.py TCS INFY HDFC        # multiple stocks
#   python main.py --schedule            # start hourly tracker
#   python main.py --watchlist           # analyse default watchlist
#   python main.py ZOMATO --bot tech     # run only one bot
#   python main.py --status              # check API/Ollama status
#   python main.py --history RELIANCE    # show past verdicts
# ============================================================

import sys
import argparse
from pathlib import Path
from datetime import datetime
import pytz

sys.path.insert(0, str(Path(__file__).parent))


def print_banner():
    print("""
╔═══════════════════════════════════════════════════════════════╗
║     🏛  INDIAN STOCK MARKET BOT COUNCIL — LOCAL AI SYSTEM    ║
║     Fundamental • Technical • News • Sentiment • Risk         ║
║     Powered by Ollama (Local LLM) + Free Market APIs         ║
╚═══════════════════════════════════════════════════════════════╝
""")


def check_status():
    """Check Ollama + all API keys + NSE connectivity."""
    print("\n📋  SYSTEM STATUS CHECK")
    print("─" * 50)

    from utils.llm import check_ollama
    from live_apis import APIKeys
    from utils.live_data import NSELive

    # Ollama
    ollama_ok = check_ollama()
    print(f"\n[LLM]  Ollama: {'✅ Running' if ollama_ok else '❌ Not running'}")
    if not ollama_ok:
        print("       → Start with: ollama serve")
        print("       → Install model: ollama pull mistral")

    # API keys
    print("\n[APIS] API Key Status:")
    for name, status in APIKeys.status().items():
        print(f"       {name:20} {status}")

    # NSE connectivity
    print("\n[NSE]  Testing NSE India connection...")
    try:
        mkt = NSELive.get_market_status()
        if mkt:
            print(f"       ✅ Connected | Market: {mkt.get('status', 'N/A')}")
        else:
            print("       ⚠️  Could not get market status (may be weekend/holiday)")
    except Exception as e:
        print(f"       ❌ NSE connection failed: {e}")

    # Database
    from utils.database import DB_PATH
    print(f"\n[DB]   Database: {'✅' if DB_PATH.exists() else '📝 Will be created'} {DB_PATH}")

    print("\n[CONFIG] Setup:")
    from config import OLLAMA_MODEL, ALLOW_INTERNET, HISTORICAL_PERIOD
    print(f"         LLM Model:      {OLLAMA_MODEL}")
    print(f"         Internet:       {'Enabled' if ALLOW_INTERNET else 'OFFLINE MODE'}")
    print(f"         Price History:  {HISTORICAL_PERIOD}")
    print(f"         Current Time:   {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%d %b %Y %H:%M IST')}")


def run_single_bot(symbol: str, bot_name: str):
    """Run only one specific bot."""
    bot_name = bot_name.lower()
    if bot_name in ('fundamental', 'fund', 'f'):
        from bots import fundamental_bot
        fundamental_bot.run(symbol)
    elif bot_name in ('technical', 'tech', 't'):
        from bots import technical_bot
        technical_bot.run(symbol)
    elif bot_name in ('news', 'n'):
        from bots import news_bot
        news_bot.run(symbol)
    elif bot_name in ('sentiment', 'sent', 's'):
        from bots import sentiment_bot
        sentiment_bot.run(symbol)
    elif bot_name in ('risk', 'r'):
        from bots import risk_bot
        risk_bot.run(symbol)
    else:
        print(f"Unknown bot: {bot_name}")
        print("Valid: fundamental, technical, news, sentiment, risk")


def main():
    print_banner()

    parser = argparse.ArgumentParser(
        description='Indian Stock Market Bot Council',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py RELIANCE                # full analysis
  python main.py TCS INFY               # analyse 2 stocks
  python main.py HDFC --bot technical   # only tech analysis
  python main.py --schedule             # start hourly tracker
  python main.py --status               # check all connections
  python main.py --history TATAMOTORS   # past verdicts
        """
    )

    parser.add_argument('symbols', nargs='*', help='NSE stock symbols (e.g. RELIANCE TCS)')
    parser.add_argument('--schedule', action='store_true', help='Start hourly market tracker')
    parser.add_argument('--watchlist', action='store_true', help='Analyse default watchlist')
    parser.add_argument('--status', action='store_true', help='Check system status')
    parser.add_argument('--bot', type=str, help='Run specific bot only: fundamental/technical/news/sentiment/risk')
    parser.add_argument('--history', type=str, help='Show verdict history for a symbol')
    parser.add_argument('--model', type=str, help='Override LLM model (e.g. llama3.1, mistral, gemma2)')
    parser.add_argument('--offline', action='store_true', help='Run in offline mode (cache only)')
    parser.add_argument('--quick', action='store_true', help='Quick price check only (no LLM)')

    args = parser.parse_args()

    # Override config if flags set
    if args.model:
        import config
        config.OLLAMA_MODEL = args.model
        print(f"[CONFIG] Using model: {args.model}")

    if args.offline:
        import config
        config.ALLOW_INTERNET = False
        print("[CONFIG] Running in OFFLINE mode")

    # ── Handle commands ──────────────────────────────────────

    if args.status:
        check_status()
        return

    if args.history:
        from utils.database import get_verdict_history
        history = get_verdict_history(args.history.upper(), limit=20)
        if not history:
            print(f"No history for {args.history.upper()}")
            return
        print(f"\n📅  Verdict History — {args.history.upper()}")
        print("─" * 55)
        for h in history:
            ts = datetime.fromtimestamp(h['created_at']).strftime('%d %b %Y %H:%M')
            scores = h.get('scores_json', '{}')
            print(f"  [{ts}]  {h['verdict']:25}  Score: {h['composite_score']:.1f}/10")
        return

    if args.schedule:
        from scheduler import start_scheduler
        watchlist = args.symbols if args.symbols else None
        start_scheduler(watchlist=watchlist)
        return

    if args.watchlist:
        from scheduler import DEFAULT_WATCHLIST, run_full_council
        print(f"[WATCHLIST] Analysing: {', '.join(DEFAULT_WATCHLIST[:5])}...")
        for sym in DEFAULT_WATCHLIST[:5]:
            run_full_council(sym)
        return

    if args.quick:
        symbols = [s.upper() for s in args.symbols] if args.symbols else ['NIFTY50']
        from scheduler import run_quick_price_check
        run_quick_price_check(symbols)
        return

    if not args.symbols:
        parser.print_help()
        print("\n💡 Quick start:")
        print("   python main.py RELIANCE")
        print("   python main.py --status")
        return

    # ── Run analysis ─────────────────────────────────────────

    symbols = [s.upper() for s in args.symbols]

    for symbol in symbols:
        if args.bot:
            run_single_bot(symbol, args.bot)
        else:
            from scheduler import run_full_council
            run_full_council(symbol)
            if len(symbols) > 1:
                print(f"\n⏱  Waiting 10s before next stock...")
                import time
                time.sleep(10)


if __name__ == "__main__":
    main()
