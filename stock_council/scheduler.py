# ============================================================
# scheduler.py — Hourly Bot Runner for Live Market Tracking
# ============================================================
#
# MARKET HOURS: NSE/BSE trade 9:15 AM to 3:30 PM IST Monday-Friday
#
# SCHEDULE:
#   Pre-market  (9:00 AM):  Pre-market sentiment + global cues
#   Market open (9:15 AM):  First full analysis
#   Hourly      (every 1h): Price + sentiment update
#   Mid-day     (12:00 PM): Full council meeting
#   Close       (3:30 PM):  EOD summary + next-day outlook
#   Post-market (4:00 PM):  FII/DII data + final verdict
#
# WATCHLIST: Add symbols to WATCHLIST in config or pass via CLI
# ============================================================

import sys
import time
import json
import pytz
import schedule
import threading
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from config import VERBOSE_DEBUG
from utils.database import save_verdict, get_verdict_history
from utils.live_data import NSELive, get_live_quote, get_all_news
from utils.market_data import fetch_fundamentals
from live_apis import APIKeys


IST = pytz.timezone('Asia/Kolkata')
MARKET_OPEN = (9, 15)    # 9:15 AM IST
MARKET_CLOSE = (15, 30)  # 3:30 PM IST


# ── Watchlist ──────────────────────────────────────────────────
# Add any NSE symbols you want to track here
DEFAULT_WATCHLIST = [
    "RELIANCE", "TCS", "INFOSYS", "HDFCBANK", "ICICIBANK",
    "SBIN", "WIPRO", "TATAMOTORS", "ADANIENT", "BAJFINANCE",
    "ZOMATO", "NYKAA", "PAYTM", "INFY", "LT"
]


class MarketSession:
    """Tracks the current trading session state."""

    def __init__(self):
        self.session_results = defaultdict(list)  # symbol → [hourly results]
        self.current_prices = {}
        self.alert_thresholds = {}   # symbol → {'up': 5%, 'down': -3%}
        self._lock = threading.Lock()

    def is_market_open(self) -> bool:
        """Check if NSE market is currently open."""
        now = datetime.now(IST)
        if now.weekday() >= 5:  # Saturday/Sunday
            return False
        market_open_time = now.replace(
            hour=MARKET_OPEN[0], minute=MARKET_OPEN[1], second=0, microsecond=0
        )
        market_close_time = now.replace(
            hour=MARKET_CLOSE[0], minute=MARKET_CLOSE[1], second=0, microsecond=0
        )
        return market_open_time <= now <= market_close_time

    def is_pre_market(self) -> bool:
        now = datetime.now(IST)
        if now.weekday() >= 5:
            return False
        pre = now.replace(hour=9, minute=0, second=0, microsecond=0)
        open_ = now.replace(hour=MARKET_OPEN[0], minute=MARKET_OPEN[1], second=0, microsecond=0)
        return pre <= now < open_

    def time_to_close(self) -> float:
        """Minutes remaining to market close."""
        now = datetime.now(IST)
        close = now.replace(hour=MARKET_CLOSE[0], minute=MARKET_CLOSE[1], second=0)
        return max(0, (close - now).total_seconds() / 60)


session = MarketSession()


# ── Price Alert System ──────────────────────────────────────────

class PriceAlertTracker:
    """Track intraday price movements and trigger alerts."""

    def __init__(self):
        self._session_open_prices = {}  # symbol → opening price for day
        self._last_prices = {}

    def set_open_price(self, symbol: str, price: float):
        if symbol not in self._session_open_prices:
            self._session_open_prices[symbol] = price
        self._last_prices[symbol] = price

    def check_alert(self, symbol: str, current_price: float, threshold_pct: float = 2.0) -> dict | None:
        """
        Return alert if price moved more than threshold from open.
        """
        open_price = self._session_open_prices.get(symbol)
        if not open_price or open_price == 0:
            return None

        change_pct = (current_price - open_price) / open_price * 100
        last = self._last_prices.get(symbol, open_price)
        tick_change = (current_price - last) / last * 100 if last else 0
        self._last_prices[symbol] = current_price

        alert = None
        if abs(change_pct) >= threshold_pct:
            direction = "UP 🟢" if change_pct > 0 else "DOWN 🔴"
            alert = {
                'symbol': symbol,
                'current': current_price,
                'open': open_price,
                'change_pct': round(change_pct, 2),
                'direction': direction,
                'type': 'PRICE_ALERT',
                'timestamp': datetime.now(IST).isoformat()
            }

        # Circuit breaker alert (5% move)
        if abs(change_pct) >= 5.0:
            alert['severity'] = 'CIRCUIT_RISK'

        return alert


alert_tracker = PriceAlertTracker()


# ── Core Run Functions ──────────────────────────────────────────

def run_quick_price_check(symbols: list, print_fn=print) -> dict:
    """
    Lightweight: only fetch live price + basic sentiment for each symbol.
    Runs every 30 minutes between full analyses.
    """
    print_fn(f"\n[{datetime.now(IST).strftime('%H:%M IST')}] ⚡ QUICK PRICE CHECK")
    print_fn("─" * 50)

    results = {}
    for sym in symbols:
        quote = get_live_quote(sym)
        price = quote.get('last_price') or quote.get('current')
        if price:
            alert = alert_tracker.check_alert(sym, float(price))
            change = quote.get('pct_change') or quote.get('change_pct', 0)
            arrow = '▲' if (change or 0) >= 0 else '▼'
            print_fn(f"  {sym:15} ₹{price:<10.2f} {arrow}{abs(change or 0):.2f}%"
                     f"  [{quote.get('source', '?')}]")
            if alert:
                print_fn(f"  🚨 ALERT: {sym} moved {alert['change_pct']:+.2f}% from open!")
            results[sym] = {'price': price, 'change_pct': change, 'alert': alert}
        else:
            print_fn(f"  {sym:15} Price unavailable")

    return results


def run_full_council(symbol: str, print_fn=print, save_to_db: bool = True) -> dict:
    """
    Run all 5 bots for one symbol. This is the main analysis function.
    Takes 2-5 minutes per symbol depending on model speed.
    """
    from bots import fundamental_bot, technical_bot, news_bot, sentiment_bot, risk_bot
    from utils.llm import check_ollama

    timestamp = datetime.now(IST)
    print_fn(f"\n{'═'*65}")
    print_fn(f"🏛  STOCK COUNCIL SESSION — {symbol}")
    print_fn(f"📅  {timestamp.strftime('%d %b %Y %H:%M IST')}")
    print_fn('═'*65)

    if not check_ollama():
        print_fn("[ERROR] Ollama not running! Start with: ollama serve")
        return {}

    # ── Bot 1: Fundamentals ────────────────────────────────────
    print_fn("\n🤖 Bot 1/5: Fundamental Bot analyzing...")
    fund_result = fundamental_bot.run(symbol, print_output=True)

    # ── Bot 2: Technical ──────────────────────────────────────
    print_fn("\n🤖 Bot 2/5: Technical Bot analyzing...")
    tech_result = technical_bot.run(symbol, print_output=True)

    # ── Bot 3: News ───────────────────────────────────────────
    print_fn("\n🤖 Bot 3/5: News Bot analyzing...")
    news_result = news_bot.run(symbol, print_output=True)
    news_sentiment_compound = news_result.get('sentiment_summary', {}).get('avg_sentiment', 0.0)

    # ── Bot 4: Sentiment ──────────────────────────────────────
    print_fn("\n🤖 Bot 4/5: Sentiment Bot analyzing...")
    sent_result = sentiment_bot.run(symbol, news_sentiment=news_sentiment_compound, print_output=True)

    # ── Bot 5: Risk ───────────────────────────────────────────
    print_fn("\n🤖 Bot 5/5: Risk Bot analyzing...")
    risk_result = risk_bot.run(symbol, print_output=True)

    # ── Council Verdict ───────────────────────────────────────
    verdict = generate_final_verdict(
        symbol, fund_result, tech_result,
        news_result, sent_result, risk_result,
        print_fn=print_fn
    )

    full_result = {
        'symbol': symbol,
        'timestamp': timestamp.isoformat(),
        'fundamental': fund_result,
        'technical': tech_result,
        'news': news_result,
        'sentiment': sent_result,
        'risk': risk_result,
        'verdict': verdict,
    }

    if save_to_db:
        save_verdict(
            symbol,
            verdict['label'],
            verdict['composite_score'],
            verdict['summary'],
            {
                'fundamental': fund_result.get('score'),
                'technical': tech_result.get('score'),
                'news': news_result.get('score'),
                'sentiment': sent_result.get('score'),
                'risk': risk_result.get('score'),
            }
        )

    return full_result


def generate_final_verdict(symbol: str,
                            fund_r: dict, tech_r: dict,
                            news_r: dict, sent_r: dict, risk_r: dict,
                            print_fn=print) -> dict:
    """
    Generate the final council verdict from all bot scores.

    VERDICT FORMULA:
      Composite = (
          fund_score × 0.30 +
          tech_score × 0.25 +
          news_score × 0.20 +
          sent_score × 0.15 +
          (10 - risk_score) × 0.10   ← risk is INVERTED
      )

    VERDICT LABELS:
      9.0-10.0 = STRONG BUY
      7.5-9.0  = BUY
      6.0-7.5  = ACCUMULATE
      5.0-6.0  = HOLD
      3.5-5.0  = REDUCE
      2.0-3.5  = SELL
      0.0-2.0  = STRONG SELL
    """
    from utils.llm import stream_chat

    JUDGE_SYSTEM = """You are the COUNCIL CHAIR of an Indian stock advisory firm.
You receive analysis from 5 expert bots and must deliver a final verdict.
Keep your summary to 80-100 words. Be direct and actionable for an Indian retail investor.
Mention: (1) the consensus view, (2) the single biggest reason to buy/sell, 
(3) the single biggest risk, (4) a suggested action."""

    f = fund_r.get('score', 5)
    t = tech_r.get('score', 5)
    n = news_r.get('score', 5)
    s = sent_r.get('score', 5)
    r = risk_r.get('score', 5)

    # Composite: risk score is inverted
    composite = (
        f * 0.30 +
        t * 0.25 +
        n * 0.20 +
        s * 0.15 +
        (10 - r) * 0.10
    )
    composite = round(composite, 2)

    # Label
    if composite >= 9.0:
        label = "STRONG BUY 🚀"
    elif composite >= 7.5:
        label = "BUY 📈"
    elif composite >= 6.0:
        label = "ACCUMULATE 💚"
    elif composite >= 5.0:
        label = "HOLD ⚖️"
    elif composite >= 3.5:
        label = "REDUCE 🟡"
    elif composite >= 2.0:
        label = "SELL 📉"
    else:
        label = "STRONG SELL 🔴"

    # Build council debate summary for judge
    council_brief = f"""
COUNCIL RESULTS FOR: {symbol}

FUNDAMENTAL BOT (score {f}/10):
{fund_r.get('text', 'No analysis')[:300]}

TECHNICAL BOT (score {t}/10):
{tech_r.get('text', 'No analysis')[:300]}

NEWS BOT (score {n}/10):
{news_r.get('text', 'No analysis')[:300]}

SENTIMENT BOT (score {s}/10):
{sent_r.get('text', 'No analysis')[:300]}

RISK BOT (risk level {r}/10):
{risk_r.get('text', 'No analysis')[:300]}

Composite Score: {composite}/10
Suggested verdict: {label}

Write the final verdict summary for the council.
"""

    print_fn(f"\n{'═'*65}")
    print_fn(f"🏛  COUNCIL VERDICT — {symbol}")
    print_fn('─'*65)

    verdict_text = stream_chat(JUDGE_SYSTEM, council_brief, on_token=lambda t: print_fn(t, end='', flush=True))

    print_fn(f"\n\n{'═'*65}")
    print_fn(f"  FINAL VERDICT: {label}")
    print_fn(f"  COMPOSITE SCORE: {composite}/10")
    print_fn(f"  {'─'*50}")
    print_fn(f"  📊 Fundamental: {f}/10  |  📈 Technical: {t}/10")
    print_fn(f"  📰 News:        {n}/10  |  💬 Sentiment: {s}/10")
    print_fn(f"  ⚠️  Risk Level:  {r}/10  (inverted for composite)")
    print_fn('═'*65)

    return {
        'label': label,
        'composite_score': composite,
        'summary': verdict_text,
        'scores': {
            'fundamental': f,
            'technical': t,
            'news': n,
            'sentiment': s,
            'risk': r,
        }
    }


# ── Daily History Report ────────────────────────────────────────

def print_daily_history(symbol: str, print_fn=print):
    """Print today's analysis history for a symbol."""
    history = get_verdict_history(symbol, limit=10)
    if not history:
        print_fn(f"No history for {symbol}")
        return

    print_fn(f"\n{'─'*55}")
    print_fn(f"📅  VERDICT HISTORY — {symbol}")
    print_fn('─'*55)
    for h in history:
        ts = datetime.fromtimestamp(h['created_at']).strftime('%d %b %H:%M')
        print_fn(f"  [{ts}] {h['verdict']:20} Score: {h['composite_score']:.1f}/10")


# ── Scheduler Setup ────────────────────────────────────────────

class HourlyScheduler:
    """
    Manages hourly analysis schedule during market hours.
    """

    def __init__(self, watchlist: list = None):
        self.watchlist = watchlist or DEFAULT_WATCHLIST
        self.running = False

    def _log(self, msg: str):
        ts = datetime.now(IST).strftime('%H:%M:%S IST')
        print(f"[{ts}] {msg}")

    def run_pre_market(self):
        """9:00 AM — Pre-market briefing."""
        self._log("🌅 PRE-MARKET BRIEFING")
        # Get global cues
        indices = NSELive.get_all_indices()
        fii_dii = NSELive.get_fii_dii()
        if indices:
            for idx in indices[:5]:
                chg = idx.get('pct_change', 0) or 0
                print(f"  {idx['name']}: {idx.get('last', 'N/A')} ({chg:+.2f}%)")
        if fii_dii:
            fii_net = fii_dii.get('fii', {}).get('net_value', 'N/A')
            print(f"  FII Net (prev day): ₹{fii_net} Cr")

    def run_market_open(self):
        """9:15 AM — Full analysis for top 2 watchlist stocks."""
        if not session.is_market_open():
            return
        self._log("🔔 MARKET OPEN — Running opening analysis")
        top_stocks = self.watchlist[:2]  # analyse top 2 at open
        for sym in top_stocks:
            try:
                result = run_full_council(sym)
                if result:
                    quote = get_live_quote(sym)
                    price = quote.get('last_price') or quote.get('current', 0)
                    if price:
                        alert_tracker.set_open_price(sym, float(price))
            except Exception as e:
                self._log(f"Error analysing {sym}: {e}")

    def run_hourly_update(self):
        """Every hour during market — quick check + 1 deep analysis."""
        if not session.is_market_open():
            self._log("Market closed — skipping hourly update")
            return

        time_remaining = session.time_to_close()
        self._log(f"⏰ HOURLY UPDATE (market closes in {time_remaining:.0f} min)")

        # Quick price check for all watchlist stocks
        run_quick_price_check(self.watchlist, print_fn=self._log)

        # Do full analysis for one stock (rotating)
        hour_idx = datetime.now(IST).hour - MARKET_OPEN[0]
        if 0 <= hour_idx < len(self.watchlist):
            sym = self.watchlist[hour_idx]
            self._log(f"\n📊 Deep analysis: {sym}")
            try:
                run_full_council(sym)
            except Exception as e:
                self._log(f"Error: {e}")

    def run_midday_council(self):
        """12:00 PM — Mid-session council for priority stocks."""
        if not session.is_market_open():
            return
        self._log("☀️  MIDDAY COUNCIL SESSION")
        # Analyse top 3 by priority
        for sym in self.watchlist[:3]:
            try:
                run_full_council(sym)
            except Exception as e:
                self._log(f"Error analysing {sym}: {e}")

    def run_market_close(self):
        """3:30 PM — End of day summary."""
        self._log("🔔 MARKET CLOSING — EOD Summary")
        print_fn = self._log

        # Quick price check
        prices = run_quick_price_check(self.watchlist, print_fn=print_fn)

        # Print winners/losers
        changes = [(sym, data.get('change_pct', 0))
                   for sym, data in prices.items()
                   if data.get('change_pct') is not None]
        changes.sort(key=lambda x: x[1], reverse=True)

        if changes:
            print_fn("\n📈 TOP MOVERS TODAY:")
            for sym, chg in changes[:3]:
                arrow = '▲' if chg >= 0 else '▼'
                print_fn(f"  {sym}: {arrow}{abs(chg):.2f}%")

    def run_post_market(self):
        """4:00 PM — Full council for 2 stocks with EOD data."""
        self._log("🌆 POST-MARKET ANALYSIS")
        for sym in self.watchlist[:2]:
            try:
                run_full_council(sym)
                print_daily_history(sym, print_fn=self._log)
            except Exception as e:
                self._log(f"Error: {e}")

    def setup_schedule(self):
        """Register all scheduled jobs."""
        # Pre-market
        schedule.every().monday.at("09:00").do(self.run_pre_market)
        schedule.every().tuesday.at("09:00").do(self.run_pre_market)
        schedule.every().wednesday.at("09:00").do(self.run_pre_market)
        schedule.every().thursday.at("09:00").do(self.run_pre_market)
        schedule.every().friday.at("09:00").do(self.run_pre_market)

        # Market open
        schedule.every().monday.at("09:15").do(self.run_market_open)
        schedule.every().tuesday.at("09:15").do(self.run_market_open)
        schedule.every().wednesday.at("09:15").do(self.run_market_open)
        schedule.every().thursday.at("09:15").do(self.run_market_open)
        schedule.every().friday.at("09:15").do(self.run_market_open)

        # Hourly (10:15, 11:15, 12:15, 1:15, 2:15)
        for hour in ["10:15", "11:15", "12:15", "13:15", "14:15"]:
            for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']:
                getattr(schedule.every(), day).at(hour).do(self.run_hourly_update)

        # Midday deep dive
        schedule.every().monday.at("12:00").do(self.run_midday_council)
        schedule.every().tuesday.at("12:00").do(self.run_midday_council)
        schedule.every().wednesday.at("12:00").do(self.run_midday_council)
        schedule.every().thursday.at("12:00").do(self.run_midday_council)
        schedule.every().friday.at("12:00").do(self.run_midday_council)

        # EOD
        for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']:
            getattr(schedule.every(), day).at("15:30").do(self.run_market_close)
            getattr(schedule.every(), day).at("16:00").do(self.run_post_market)

        print(f"[SCHEDULER] {len(schedule.jobs)} jobs registered")
        print("[SCHEDULER] Schedule:")
        for job in schedule.jobs[:8]:
            print(f"  {job}")

    def start(self):
        """Start the scheduler loop."""
        self.running = True
        self.setup_schedule()
        print("\n[SCHEDULER] Started. Press Ctrl+C to stop.")
        print(f"[SCHEDULER] Watchlist: {', '.join(self.watchlist)}")
        print(f"[SCHEDULER] Market hours: 9:15 AM - 3:30 PM IST (Mon-Fri)")
        print(f"[SCHEDULER] Current time: {datetime.now(IST).strftime('%d %b %Y %H:%M IST')}")
        print(f"[SCHEDULER] Market open: {session.is_market_open()}")

        try:
            while self.running:
                schedule.run_pending()
                time.sleep(30)  # check every 30 seconds
        except KeyboardInterrupt:
            print("\n[SCHEDULER] Stopped by user.")
            self.running = False

    def run_now(self, symbols: list = None):
        """Run full council immediately (for testing / manual run)."""
        syms = symbols or self.watchlist[:1]
        for sym in syms:
            run_full_council(sym)


# ── Entry point for scheduler ─────────────────────────────────

def start_scheduler(watchlist: list = None):
    """Start the hourly market tracker."""
    sched = HourlyScheduler(watchlist=watchlist or DEFAULT_WATCHLIST)
    sched.start()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Stock Council Scheduler')
    parser.add_argument('--watchlist', nargs='+', default=DEFAULT_WATCHLIST,
                        help='List of NSE symbols to track')
    parser.add_argument('--run-now', action='store_true',
                        help='Run analysis immediately instead of scheduling')
    parser.add_argument('--symbol', type=str, help='Single symbol for immediate run')
    args = parser.parse_args()

    if args.run_now or args.symbol:
        syms = [args.symbol] if args.symbol else args.watchlist[:2]
        for sym in syms:
            run_full_council(sym)
    else:
        start_scheduler(watchlist=args.watchlist)
