# ============================================================
# notifier/telegram_notifier.py
# Send Telegram alerts when the council delivers a high-conviction
# STRONG BUY or STRONG SELL verdict.
# ============================================================
#
# SETUP (one-time, 5 minutes):
#   1. Open Telegram → search @BotFather → /newbot
#      Follow prompts → get BOT_TOKEN
#   2. Start a chat with your new bot, then visit:
#      https://api.telegram.org/bot<BOT_TOKEN>/getUpdates
#      to find your CHAT_ID
#   3. Add to .env:
#      TELEGRAM_BOT_TOKEN=123456789:AAbbccdd...
#      TELEGRAM_CHAT_ID=-100123456789
#
# USAGE (auto-called from orchestrator):
#   from notifier.telegram_notifier import notify_verdict
#   notify_verdict(result_dict)
#
# STANDALONE TEST:
#   python notifier/telegram_notifier.py --test
#
# TRIGGER CONDITIONS (all must be true):
#   - verdict in (STRONG BUY, BUY, STRONG SELL)
#   - conviction == HIGH
#   - GPS >= 7.0
#   - final_score >= 7.5 (for buy) or <= 3.0 (for sell alert)
# ============================================================

import os
import sys
import json
import requests
from pathlib import Path
from datetime import datetime
import pytz

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

IST = pytz.timezone('Asia/Kolkata')

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

# Alert trigger thresholds
STRONG_BUY_VERDICTS  = {"STRONG BUY", "BUY"}
STRONG_SELL_VERDICTS = {"STRONG SELL", "SELL"}
MIN_GPS_FOR_ALERT    = 7.0
MIN_BUY_SCORE        = 7.5
MAX_SELL_SCORE       = 3.5


def _is_configured() -> bool:
    """Check if Telegram credentials are set."""
    return bool(BOT_TOKEN and CHAT_ID and
                not BOT_TOKEN.startswith("your_") and
                not CHAT_ID.startswith("your_"))


def _send_message(text: str, parse_mode: str = "HTML") -> bool:
    """
    Send a message via Telegram Bot API.
    Returns True on success, False on failure.
    """
    if not _is_configured():
        return False

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id":    CHAT_ID,
        "text":       text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code == 200:
            return True
        else:
            print(f"[TELEGRAM] API error {r.status_code}: {r.text[:200]}")
            return False
    except Exception as e:
        print(f"[TELEGRAM] Send failed: {e}")
        return False


def _should_alert(result: dict) -> tuple:
    """
    Decide whether to send an alert for this council result.

    Returns (should_alert: bool, alert_type: str)
    """
    verdict    = (result.get('verdict') or "").upper().strip()
    conviction = (result.get('conviction') or "MEDIUM").upper().strip()
    gps        = result.get('gps', 0)
    score      = result.get('final_score', 5)

    is_strong_buy  = verdict in STRONG_BUY_VERDICTS
    is_strong_sell = verdict in STRONG_SELL_VERDICTS
    is_high_conv   = conviction == "HIGH"
    has_good_gps   = gps >= MIN_GPS_FOR_ALERT

    if is_strong_buy and is_high_conv and has_good_gps and score >= MIN_BUY_SCORE:
        return True, "BUY"
    if is_strong_sell and is_high_conv and score <= MAX_SELL_SCORE:
        return True, "SELL"

    return False, ""


def _format_message(result: dict, alert_type: str) -> str:
    """Format a rich Telegram HTML message for the council result."""
    sym     = result.get('symbol', 'N/A')
    company = result.get('company', sym)
    sector  = result.get('sector', 'N/A')
    price   = result.get('price')
    gps     = result.get('gps', 0)
    score   = result.get('final_score', 5)
    verdict = result.get('verdict', 'N/A')
    conv    = result.get('conviction', 'N/A')
    sl      = result.get('stop_loss')
    tgt     = result.get('target')
    scores  = result.get('scores', {})
    now_str = datetime.now(IST).strftime('%d %b %Y %H:%M IST')

    emoji = "📈" if alert_type == "BUY" else "📉"
    alert_header = f"{emoji} <b>COUNCIL ALERT — {verdict}</b>"

    price_str = f"₹{price:.2f}" if price else "N/A"
    sl_str    = f"₹{sl:.2f}"   if sl    else "N/A"
    tgt_str   = f"₹{tgt:.2f}" if tgt   else "N/A"

    component_scores = (
        f"F:{scores.get('fundamental',5):.1f} "
        f"T:{scores.get('technical',5):.1f} "
        f"N:{scores.get('news',5):.1f} "
        f"S:{scores.get('sentiment',5):.1f} "
        f"Risk:{scores.get('risk',5):.1f}"
    )

    # Bull/bear debate scores if available
    debate_line = ""
    if scores.get('bull_avg') and scores.get('bear_avg'):
        debate_line = (
            f"\n🐂 Bull avg: <b>{scores['bull_avg']:.1f}</b>  "
            f"🐻 Bear avg: <b>{scores['bear_avg']:.1f}</b>"
        )

    # Council verdict excerpt
    council_text = result.get('council_verdict', '')
    excerpt      = council_text[:300].replace('<', '&lt;').replace('>', '&gt;') if council_text else ""
    excerpt_block = f"\n\n<i>{excerpt}...</i>" if excerpt else ""

    return f"""{alert_header}

🏢 <b>{sym}</b> — {company}
📂 {sector}
🕐 {now_str}

💰 Price:      <b>{price_str}</b>
🎯 GPS:        <b>{gps:.1f}/10</b>
⭐ Score:      <b>{score:.1f}/10</b>
🔒 Conviction: <b>{conv}</b>

📊 Bot scores: {component_scores}{debate_line}

🛑 Stop Loss:  <b>{sl_str}</b>
🎯 Target:     <b>{tgt_str}</b>
{excerpt_block}

#StockCouncil #{sym} #{alert_type}"""


def notify_verdict(result: dict, force: bool = False) -> bool:
    """
    Send a Telegram alert if the result meets alert criteria.

    Args:
        result: The dict returned by run_council_for_stock()
        force:  Send regardless of thresholds (for testing)

    Returns True if message was sent.
    """
    if not _is_configured():
        if os.getenv("TELEGRAM_BOT_TOKEN"):
            print("[TELEGRAM] Bot token set but chat_id missing — "
                  "add TELEGRAM_CHAT_ID to .env")
        return False

    should_send, alert_type = _should_alert(result)
    if not should_send and not force:
        return False

    alert_type = alert_type or result.get('verdict', 'INFO')
    msg        = _format_message(result, alert_type)
    ok         = _send_message(msg)

    if ok:
        print(f"  [TELEGRAM] Alert sent: {result.get('symbol')} {result.get('verdict')}")
    return ok


def notify_session_summary(all_results: list, elapsed_min: int) -> bool:
    """
    Send an end-of-session summary with the top 3 stocks.
    Called at the end of run_full_pipeline().
    """
    if not _is_configured() or not all_results:
        return False

    now_str = datetime.now(IST).strftime('%d %b %Y %H:%M IST')
    top3    = sorted(all_results, key=lambda r: r.get('final_score', 0), reverse=True)[:3]

    lines = [f"🏛 <b>COUNCIL SESSION COMPLETE</b> — {now_str}"]
    lines.append(f"⏱ Duration: {elapsed_min} min | {len(all_results)} stocks debated\n")
    lines.append("🏆 <b>Top picks:</b>")

    for i, r in enumerate(top3, 1):
        sym     = r.get('symbol', '?')
        verdict = r.get('verdict', '?')
        score   = r.get('final_score', 0)
        gps     = r.get('gps', 0)
        tgt     = r.get('target')
        tgt_str = f" → ₹{tgt:.0f}" if tgt else ""
        lines.append(f"  {i}. <b>{sym}</b> {verdict} {score:.1f}/10 GPS:{gps:.1f}{tgt_str}")

    return _send_message('\n'.join(lines))


def send_test_message() -> bool:
    """Send a test message to verify the bot is configured correctly."""
    if not _is_configured():
        print("[TELEGRAM] Not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
        return False

    msg = (
        "✅ <b>Stock Council Bot — Test Message</b>\n\n"
        "Your Telegram notifier is working correctly.\n"
        "You will receive alerts when the council delivers\n"
        "HIGH conviction STRONG BUY/SELL signals.\n\n"
        f"🕐 {datetime.now(IST).strftime('%d %b %Y %H:%M IST')}"
    )
    ok = _send_message(msg)
    if ok:
        print("[TELEGRAM] Test message sent successfully!")
    return ok


# ── CLI ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Telegram notifier for Stock Council')
    parser.add_argument('--test', action='store_true',
                        help='Send a test message to verify configuration')
    parser.add_argument('--status', action='store_true',
                        help='Show notifier configuration status')
    args = parser.parse_args()

    if args.status or not any(vars(args).values()):
        print(f"Telegram configured: {'✅' if _is_configured() else '❌'}")
        if BOT_TOKEN:
            print(f"  Bot token: {BOT_TOKEN[:10]}...{BOT_TOKEN[-4:]}")
        else:
            print("  Bot token: NOT SET → add TELEGRAM_BOT_TOKEN to .env")
        if CHAT_ID:
            print(f"  Chat ID:   {CHAT_ID}")
        else:
            print("  Chat ID:   NOT SET → add TELEGRAM_CHAT_ID to .env")
        print("\nAlert triggers:")
        print(f"  Verdicts:   {STRONG_BUY_VERDICTS | STRONG_SELL_VERDICTS}")
        print(f"  Conviction: HIGH required")
        print(f"  Min GPS:    {MIN_GPS_FOR_ALERT}")
        print(f"  Min score:  {MIN_BUY_SCORE} (buy) / ≤{MAX_SELL_SCORE} (sell)")

    if args.test:
        send_test_message()
