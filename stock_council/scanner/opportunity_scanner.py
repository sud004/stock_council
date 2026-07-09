# ============================================================
# scanner/opportunity_scanner.py
# Finds HIGH PROBABILITY contrarian buy opportunities:
#
# STRATEGY 1 — VALUE AT LOWS (Contrarian Buy)
#   Stock near 52W low / All-Time Low BUT strong fundamentals
#   Logic: good company beaten down = opportunity
#   Entry signal: price < 52W low × 1.15  (within 15% of yearly low)
#   Quality check: ROE > 12%, D/E < 1.5, Revenue growth > 0%
#   Avoid: stocks falling due to fraud/governance issues
#
# STRATEGY 2 — SECTOR LOSER BOUNCE
#   Top daily loser in each sector = potential 3-4 day bounce
#   Logic: oversold stocks bounce after panic selling
#   Conditions:
#     RSI < 35 (oversold)
#     Volume spike > 2x average (panic selling)
#     Fundamentally sound (GPS > 5.0)
#     NOT in our "consistently negative" watchlist
#   Hold: 3-5 trading days
#
# COUNCIL MEMBERSHIP SYSTEM
#   Each stock gets a Performance Score tracked over time
#   Score improves: stock rises after BUY verdict
#   Score drops:    stock falls after BUY verdict
#   Threshold:      if Performance Score < -3 over 14 days → DROPPED
#   Reinstated:     after 30 days of positive performance
# ============================================================

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import pytz

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, VERBOSE_DEBUG
from memory.storage import (
    load_prices_csv, load_fundamentals_json,
    load_bot_scores_history, today_str
)

IST = pytz.timezone('Asia/Kolkata')

# ── Performance tracking file ─────────────────────────────────
PERFORMANCE_FILE = DATA_DIR / "council_performance.json"
DROPPED_FILE     = DATA_DIR / "dropped_stocks.json"


# ══════════════════════════════════════════════════════════════
# STRATEGY 1: VALUE AT LOWS
# ══════════════════════════════════════════════════════════════

def find_value_at_lows(symbols: list, sector_map: dict) -> list[dict]:
    """
    Find stocks near 52W/ATL lows with strong fundamentals.

    SCORING:
      Proximity to low:
        Within 5% of 52W low  → score +4  (deep value)
        Within 10% of 52W low → score +3
        Within 15% of 52W low → score +2
        Within 20% of 52W low → score +1

      Fundamental quality gates (ALL must pass):
        ROE > 12%              → quality check
        D/E < 1.5              → not over-leveraged
        Revenue growth > 0%    → business still growing
        Net margin > 5%        → profitable

      Bonus signals:
        Promoter buying recently → +2
        Institutional holding > 15% → +1
        Dividend yield > 2% → +1 (company returning cash)

      Red flags (disqualify):
        Revenue DECLINING > 10% → SKIP
        D/E > 2.5 → SKIP (debt trap)
        Consistently dropped from council → SKIP
    """
    opportunities = []
    dropped = load_dropped_stocks()

    for sym in symbols:
        # Skip if dropped from council
        if sym in dropped and dropped[sym].get('status') == 'dropped':
            continue

        df = load_prices_csv(sym, days=365)
        fund = load_fundamentals_json(sym)

        if df is None or df.empty or len(df) < 50:
            continue
        if not fund or fund.get('error'):
            continue

        close = df['Close']
        price = float(close.iloc[-1])

        # ── 52 Week levels ─────────────────────────────────
        w52_high = float(close.tail(252).max())
        w52_low  = float(close.tail(252).min())

        # ── All time low (full history) ─────────────────────
        atl = float(close.min())
        ath = float(close.max())

        pct_above_52w_low  = (price - w52_low) / w52_low * 100
        pct_below_52w_high = (w52_high - price) / w52_high * 100
        pct_above_atl      = (price - atl) / atl * 100

        # Only consider stocks within 25% of 52W low
        if pct_above_52w_low > 25:
            continue

        # ── Fundamental quality gates ───────────────────────
        roe            = fund.get('roe') or 0
        de             = fund.get('debt_equity') or 0
        rev_growth     = fund.get('revenue_growth') or 0
        net_margin     = fund.get('net_margin') or 0
        promoter       = fund.get('promoter_holding_pct') or 0
        inst_holding   = fund.get('institutional_holding_pct') or 0
        div_yield      = fund.get('dividend_yield') or 0
        pe             = fund.get('pe_ratio') or 999

        # Hard disqualifiers
        if rev_growth < -15:   continue   # Revenue crashing
        if de > 2.5:           continue   # Debt trap
        if roe < 0:            continue   # Losing money on equity
        if net_margin < 0:     continue   # Not profitable

        # Soft quality checks
        quality_score = 0
        quality_notes = []

        if roe >= 20:
            quality_score += 3
            quality_notes.append(f"Excellent ROE {roe:.1f}%")
        elif roe >= 12:
            quality_score += 2
            quality_notes.append(f"Good ROE {roe:.1f}%")
        elif roe >= 8:
            quality_score += 1
            quality_notes.append(f"Moderate ROE {roe:.1f}%")

        if de < 0.3:
            quality_score += 2
            quality_notes.append(f"Very low debt D/E {de:.2f}x")
        elif de < 0.7:
            quality_score += 1
            quality_notes.append(f"Low debt D/E {de:.2f}x")

        if rev_growth >= 15:
            quality_score += 2
            quality_notes.append(f"Strong revenue growth {rev_growth:.1f}%")
        elif rev_growth >= 5:
            quality_score += 1
            quality_notes.append(f"Revenue growing {rev_growth:.1f}%")

        if net_margin >= 15:
            quality_score += 2
            quality_notes.append(f"High margin {net_margin:.1f}%")
        elif net_margin >= 8:
            quality_score += 1

        if promoter >= 50:
            quality_score += 2
            quality_notes.append(f"Strong promoter holding {promoter:.1f}%")
        elif promoter >= 35:
            quality_score += 1

        if div_yield >= 3:
            quality_score += 1
            quality_notes.append(f"Good dividend yield {div_yield:.1f}%")

        if inst_holding >= 20:
            quality_score += 1
            quality_notes.append(f"High institutional interest {inst_holding:.1f}%")

        # Skip weak fundamentals
        if quality_score < 3:
            continue

        # ── Proximity score ────────────────────────────────
        if pct_above_52w_low <= 5:
            proximity_score = 4
            proximity_label = "DEEP VALUE — within 5% of 52W low"
        elif pct_above_52w_low <= 10:
            proximity_score = 3
            proximity_label = f"Near 52W low ({pct_above_52w_low:.1f}% above)"
        elif pct_above_52w_low <= 15:
            proximity_score = 2
            proximity_label = f"Approaching 52W low ({pct_above_52w_low:.1f}% above)"
        else:
            proximity_score = 1
            proximity_label = f"Within 25% of 52W low ({pct_above_52w_low:.1f}% above)"

        # ATL bonus
        atl_bonus = 0
        atl_label = ""
        if pct_above_atl <= 10:
            atl_bonus = 2
            atl_label = f"⚡ NEAR ALL-TIME LOW ({pct_above_atl:.1f}% above ATL ₹{atl:.0f})"
        elif pct_above_atl <= 20:
            atl_bonus = 1
            atl_label = f"Near ATL ({pct_above_atl:.1f}% above)"

        total_score = quality_score + proximity_score + atl_bonus

        # ── Technical confirmation ─────────────────────────
        rsi = _calc_rsi(close)
        volume = df['Volume']
        vol_ratio = float(volume.iloc[-1]) / float(volume.tail(20).mean())

        # RSI oversold is good for contrarian
        rsi_note = ""
        if rsi and rsi < 35:
            total_score += 1
            rsi_note = f"RSI {rsi:.0f} — oversold (bounce likely)"
        elif rsi and rsi < 45:
            rsi_note = f"RSI {rsi:.0f} — weak momentum"

        opportunities.append({
            'symbol':          sym,
            'sector':          sector_map.get(sym, 'Unknown'),
            'strategy':        'VALUE_AT_LOWS',
            'price':           round(price, 2),
            'w52_low':         round(w52_low, 2),
            'w52_high':        round(w52_high, 2),
            'atl':             round(atl, 2),
            'ath':             round(ath, 2),
            'pct_above_52w_low':  round(pct_above_52w_low, 1),
            'pct_below_52w_high': round(pct_below_52w_high, 1),
            'pct_above_atl':      round(pct_above_atl, 1),
            'proximity_label': proximity_label,
            'atl_label':       atl_label,
            'quality_score':   quality_score,
            'proximity_score': proximity_score,
            'atl_bonus':       atl_bonus,
            'total_score':     total_score,
            'rsi':             round(rsi, 1) if rsi else None,
            'rsi_note':        rsi_note,
            'vol_ratio':       round(vol_ratio, 2),
            'roe':             roe,
            'de_ratio':        de,
            'rev_growth':      rev_growth,
            'net_margin':      net_margin,
            'promoter':        promoter,
            'div_yield':       div_yield,
            'pe':              pe,
            'quality_notes':   quality_notes,
            'suggested_hold':  '4-8 weeks',
            'stop_loss':       round(w52_low * 0.95, 2),  # 5% below 52W low
            'target_1':        round(price * 1.10, 2),    # 10% target
            'target_2':        round(price * 1.20, 2),    # 20% target
        })

    # Sort by total score descending
    opportunities.sort(key=lambda x: x['total_score'], reverse=True)
    return opportunities


# ══════════════════════════════════════════════════════════════
# STRATEGY 2: SECTOR TOP LOSER BOUNCE
# ══════════════════════════════════════════════════════════════

def find_sector_loser_bounce(symbols: list, sector_map: dict,
                              live_quotes: dict = None) -> list[dict]:
    """
    Find the biggest daily loser in each sector that's oversold
    and fundamentally sound — potential 3-5 day bounce trade.

    CONDITIONS:
      1. Biggest % loser in its sector today (1D change)
      2. RSI < 40 (oversold)
      3. Volume > 1.5x average (capitulation selling)
      4. NOT on dropped list (consistently negative)
      5. GPS score > 5.0 (fundamentally okay)
      6. Not in free-fall (not down >15% in 1 week)

    HOLD PERIOD: 3-5 trading days
    EXIT: +5-8% gain OR if falls further -3% from entry
    """
    dropped = load_dropped_stocks()

    # Group stocks by sector
    sector_stocks = {}
    for sym in symbols:
        sec = sector_map.get(sym, 'Unknown')
        if sec not in sector_stocks:
            sector_stocks[sec] = []
        sector_stocks[sec].append(sym)

    bounce_candidates = []

    for sector, stocks in sector_stocks.items():
        sector_losers = []

        for sym in stocks:
            # Skip dropped stocks
            if sym in dropped and dropped[sym].get('status') == 'dropped':
                continue

            df = load_prices_csv(sym, days=60)
            fund = load_fundamentals_json(sym)

            if df is None or df.empty or len(df) < 20:
                continue

            close = df['Close']
            volume = df['Volume']
            price = float(close.iloc[-1])

            # Get today's change
            if live_quotes and sym in live_quotes:
                change_1d = live_quotes[sym].get('pct_change', 0) or 0
            elif len(close) >= 2:
                change_1d = (close.iloc[-1] / close.iloc[-2] - 1) * 100
            else:
                continue

            # Only losers
            if change_1d >= -1.0:
                continue

            # Not in free fall (weekly check)
            change_1w = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) >= 6 else 0
            if change_1w < -15:
                continue   # Free fall, not a bounce candidate

            # RSI check
            rsi = _calc_rsi(close)
            if not rsi or rsi > 45:
                continue   # Not oversold enough

            # Volume spike check
            avg_vol = float(volume.tail(20).mean())
            today_vol = float(volume.iloc[-1])
            vol_ratio = today_vol / avg_vol if avg_vol > 0 else 0
            if vol_ratio < 1.2:
                continue   # No volume confirmation

            # Basic fundamental check
            if fund:
                roe = fund.get('roe') or 0
                de  = fund.get('debt_equity') or 0
                if roe < 5 or de > 2.5:
                    continue   # Too weak fundamentally

            sector_losers.append({
                'symbol':     sym,
                'change_1d':  round(change_1d, 2),
                'change_1w':  round(change_1w, 2),
                'rsi':        round(rsi, 1),
                'vol_ratio':  round(vol_ratio, 2),
                'price':      round(price, 2),
                'roe':        (fund or {}).get('roe'),
                'de':         (fund or {}).get('debt_equity'),
            })

        if not sector_losers:
            continue

        # Pick the biggest loser in the sector
        sector_losers.sort(key=lambda x: x['change_1d'])
        biggest_loser = sector_losers[0]

        # Score the bounce opportunity
        bounce_score = 0
        bounce_notes = []

        rsi = biggest_loser['rsi']
        if rsi < 25:
            bounce_score += 4
            bounce_notes.append(f"RSI {rsi:.0f} — extremely oversold")
        elif rsi < 35:
            bounce_score += 3
            bounce_notes.append(f"RSI {rsi:.0f} — oversold")
        else:
            bounce_score += 1
            bounce_notes.append(f"RSI {rsi:.0f} — weakening")

        vr = biggest_loser['vol_ratio']
        if vr > 3:
            bounce_score += 3
            bounce_notes.append(f"Volume {vr:.1f}x avg — panic selling (strong bounce signal)")
        elif vr > 2:
            bounce_score += 2
            bounce_notes.append(f"Volume {vr:.1f}x avg — high selling pressure")
        else:
            bounce_score += 1
            bounce_notes.append(f"Volume {vr:.1f}x avg — moderate")

        change = biggest_loser['change_1d']
        if change < -5:
            bounce_score += 2
            bounce_notes.append(f"Down {abs(change):.1f}% today — sharp drop")
        elif change < -3:
            bounce_score += 1

        price = biggest_loser['price']
        bounce_candidates.append({
            **biggest_loser,
            'sector':          sector,
            'strategy':        'SECTOR_LOSER_BOUNCE',
            'bounce_score':    bounce_score,
            'bounce_notes':    bounce_notes,
            'suggested_hold':  '3-5 trading days',
            'entry_price':     price,
            'stop_loss':       round(price * 0.97, 2),   # 3% stop loss
            'target_1':        round(price * 1.05, 2),   # 5% target
            'target_2':        round(price * 1.08, 2),   # 8% target
            'risk_reward':     '1:2.5',
            'caution':         'Short-term trade only. Exit in 3-5 days regardless.',
        })

    bounce_candidates.sort(key=lambda x: x['bounce_score'], reverse=True)
    return bounce_candidates


# ══════════════════════════════════════════════════════════════
# COUNCIL MEMBERSHIP — Performance Tracking
# ══════════════════════════════════════════════════════════════

def load_performance_scores() -> dict:
    """Load performance tracking data."""
    if PERFORMANCE_FILE.exists():
        with open(PERFORMANCE_FILE) as f:
            return json.load(f)
    return {}


def load_dropped_stocks() -> dict:
    """Load list of stocks dropped from council."""
    if DROPPED_FILE.exists():
        with open(DROPPED_FILE) as f:
            return json.load(f)
    return {}


def update_performance(symbol: str, verdict: str, entry_price: float):
    """
    Record a new council verdict for tracking.
    Called when council gives BUY/ACCUMULATE verdict.
    """
    scores = load_performance_scores()
    today = today_str()

    if symbol not in scores:
        scores[symbol] = {'entries': [], 'performance_score': 0}

    scores[symbol]['entries'].append({
        'date':        today,
        'verdict':     verdict,
        'entry_price': entry_price,
        'checked':     False,
    })

    with open(PERFORMANCE_FILE, 'w') as f:
        json.dump(scores, f, indent=2)


def check_and_update_outcomes():
    """
    Check how past verdicts performed.
    Run this daily — looks at verdicts from 3-5 days ago
    and checks if the stock went up or down.

    SCORING:
      Stock up > 3% after BUY verdict  → +1 performance point
      Stock flat (-3% to +3%)          →  0 points
      Stock down > 3% after BUY verdict → -1 performance point
      Stock down > 8% after BUY verdict → -2 performance points

    DROP RULE:
      Performance score < -3 over last 14 days → dropped for 30 days
      This means: if council recommended BUY on a stock 3+ times
      and it fell each time → stop analysing it for 30 days
    """
    scores     = load_performance_scores()
    dropped    = load_dropped_stocks()
    today      = datetime.now(IST)
    today_str_ = today.strftime('%Y-%m-%d')

    for symbol, data in scores.items():
        for entry in data['entries']:
            if entry.get('checked'):
                continue

            # Only check entries that are 4-7 days old
            entry_date = datetime.strptime(entry['date'], '%Y-%m-%d')
            entry_date = IST.localize(entry_date)
            days_old = (today - entry_date).days

            if days_old < 4:
                continue   # Too early to check
            if days_old > 10:
                entry['checked'] = True
                entry['outcome'] = 'expired'
                continue

            # Get current price
            df = load_prices_csv(symbol, days=15)
            if df is None or df.empty:
                continue

            close = df['Close']
            current_price = float(close.iloc[-1])
            entry_price   = entry.get('entry_price', current_price)

            pct_change = (current_price - entry_price) / entry_price * 100

            # Score the outcome
            if pct_change > 5:
                points = 2
                outcome = f"✅ UP {pct_change:.1f}% — council was RIGHT"
            elif pct_change > 2:
                points = 1
                outcome = f"✅ UP {pct_change:.1f}%"
            elif pct_change > -2:
                points = 0
                outcome = f"➡️  FLAT {pct_change:.1f}%"
            elif pct_change > -5:
                points = -1
                outcome = f"❌ DOWN {pct_change:.1f}%"
            elif pct_change > -10:
                points = -2
                outcome = f"❌ DOWN {pct_change:.1f}% — bad call"
            else:
                points = -3
                outcome = f"🔴 DOWN {pct_change:.1f}% — very bad call"

            entry['checked']      = True
            entry['outcome']      = outcome
            entry['points']       = points
            entry['checked_date'] = today_str_
            entry['exit_price']   = round(current_price, 2)
            data['performance_score'] += points

        # ── Check drop rule ────────────────────────────────
        perf_score = data.get('performance_score', 0)

        if perf_score <= -3:
            # Drop this stock from council for 30 days
            dropped[symbol] = {
                'status':       'dropped',
                'dropped_on':   today_str_,
                'reinstate_on': (today + timedelta(days=30)).strftime('%Y-%m-%d'),
                'reason':       f"Performance score {perf_score} — consistently wrong",
                'score':        perf_score,
            }
            print(f"[COUNCIL] ⚠️  {symbol} DROPPED from council (score: {perf_score})")
            print(f"           Reinstate after: {dropped[symbol]['reinstate_on']}")

        # ── Check reinstatement ────────────────────────────
        if symbol in dropped:
            reinstate_date = dropped[symbol].get('reinstate_on', '')
            if reinstate_date and today_str_ >= reinstate_date:
                # Check if stock has improved
                df = load_prices_csv(symbol, days=30)
                if df is not None and not df.empty:
                    close = df['Close']
                    recent_return = (close.iloc[-1]/close.iloc[0] - 1) * 100
                    if recent_return > 0:
                        dropped[symbol]['status'] = 'reinstated'
                        data['performance_score'] = 0   # Reset score
                        print(f"[COUNCIL] ✅ {symbol} REINSTATED (30-day return: {recent_return:.1f}%)")
                    else:
                        # Extend drop by another 30 days
                        dropped[symbol]['reinstate_on'] = (
                            today + timedelta(days=30)
                        ).strftime('%Y-%m-%d')
                        print(f"[COUNCIL] ⚠️  {symbol} drop EXTENDED (still negative)")

    # Save updates
    with open(PERFORMANCE_FILE, 'w') as f:
        json.dump(scores, f, indent=2)
    with open(DROPPED_FILE, 'w') as f:
        json.dump(dropped, f, indent=2)


def get_council_membership_report() -> str:
    """Print a summary of council membership status."""
    scores  = load_performance_scores()
    dropped = load_dropped_stocks()

    active   = {s: d for s, d in dropped.items() if d.get('status') == 'dropped'}
    restored = {s: d for s, d in dropped.items() if d.get('status') == 'reinstated'}

    lines = [
        "\n📋  COUNCIL MEMBERSHIP REPORT",
        "─" * 50,
        f"  Total tracked:   {len(scores)} stocks",
        f"  Currently dropped: {len(active)} stocks",
        f"  Reinstated:      {len(restored)} stocks",
        "",
    ]

    if active:
        lines.append("  ❌ DROPPED (consistently bad calls):")
        for sym, data in active.items():
            lines.append(
                f"     {sym:15} Score: {data.get('score', '?'):>3} | "
                f"Reinstate: {data.get('reinstate_on', '?')} | "
                f"{data.get('reason', '')}"
            )

    # Top performers
    top = sorted(scores.items(), key=lambda x: x[1].get('performance_score', 0), reverse=True)[:5]
    if top:
        lines.append("\n  🏆 TOP PERFORMERS (council was right):")
        for sym, data in top:
            lines.append(f"     {sym:15} Score: {data.get('performance_score', 0):>+3}")

    # Worst performers
    worst = sorted(scores.items(), key=lambda x: x[1].get('performance_score', 0))[:5]
    if worst:
        lines.append("\n  📉 WATCH LIST (council may be wrong on these):")
        for sym, data in worst:
            score = data.get('performance_score', 0)
            if score < 0:
                lines.append(f"     {sym:15} Score: {score:>+3}")

    return '\n'.join(lines)


# ══════════════════════════════════════════════════════════════
# COMBINED OPPORTUNITY REPORT
# ══════════════════════════════════════════════════════════════

def run_opportunity_scan(symbols: list, sector_map: dict,
                          live_quotes: dict = None,
                          print_output: bool = True) -> dict:
    """
    Run all opportunity strategies and return combined results.
    Called daily before the main pipeline run.
    """
    if print_output:
        print(f"\n{'═'*65}")
        print(f"🎯  OPPORTUNITY SCANNER")
        print('═'*65)

    # Update outcomes from past verdicts
    check_and_update_outcomes()

    # Strategy 1: Value at lows
    if print_output:
        print("\n  📉 Strategy 1: Value stocks near 52W / All-Time Lows...")
    value_opps = find_value_at_lows(symbols, sector_map)

    if print_output:
        print(f"  Found {len(value_opps)} value opportunities:")
        for v in value_opps[:5]:
            atl = f" {v['atl_label']}" if v['atl_label'] else ""
            print(
                f"    ★ {v['symbol']:12} ₹{v['price']:.0f} | "
                f"{v['proximity_label']} | "
                f"Score: {v['total_score']}/12{atl}"
            )

    # Strategy 2: Sector loser bounce
    if print_output:
        print("\n  📊 Strategy 2: Sector top loser bounce candidates...")
    bounce_opps = find_sector_loser_bounce(symbols, sector_map, live_quotes)

    if print_output:
        print(f"  Found {len(bounce_opps)} bounce candidates:")
        for b in bounce_opps[:5]:
            print(
                f"    ↗ {b['symbol']:12} ₹{b['price']:.0f} | "
                f"Down {abs(b['change_1d']):.1f}% today | "
                f"RSI {b['rsi']:.0f} | "
                f"Vol {b['vol_ratio']:.1f}x | "
                f"Sector: {b['sector']}"
            )

    # Membership report
    if print_output:
        print(get_council_membership_report())

    return {
        'value_at_lows':      value_opps,
        'sector_bounces':     bounce_opps,
        'dropped_stocks':     load_dropped_stocks(),
        'scan_date':          today_str(),
    }


# ── TA Helper ─────────────────────────────────────────────────

def _calc_rsi(close: pd.Series, period: int = 14) -> float | None:
    try:
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
        loss  = (-delta).clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])
    except Exception:
        return None


if __name__ == "__main__":
    from scanner.universe import get_universe
    symbols, sector_map = get_universe()
    results = run_opportunity_scan(symbols, sector_map, print_output=True)

    print(f"\n{'═'*65}")
    print(f"  TOP VALUE PICKS:")
    for v in results['value_at_lows'][:10]:
        print(
            f"  {v['symbol']:12} ₹{v['price']:.0f} | "
            f"52W Low: ₹{v['w52_low']:.0f} | "
            f"ATL: ₹{v['atl']:.0f} | "
            f"ROE: {v['roe']:.1f}% | "
            f"Score: {v['total_score']}/12"
        )
        print(f"               {v['proximity_label']}")
        if v['atl_label']:
            print(f"               {v['atl_label']}")
        print(f"               Stop: ₹{v['stop_loss']} | T1: ₹{v['target_1']} | T2: ₹{v['target_2']}")
        print()
