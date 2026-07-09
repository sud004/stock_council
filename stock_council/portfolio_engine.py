#!/usr/bin/env python3
# ============================================================
# portfolio_engine.py — Self-Managing Paper Portfolio
# ============================================================
# DESIGN:
#   ₹1,00,000 starting capital. Bot decides everything:
#   how much to deploy, when to buy, when to sell, when
#   to accumulate. Runs every night inside night_runner.py.
#
# ENTRY PRICE (Option B):
#   Night N  → council decides BUY → logged as PENDING
#   Night N+1 → Phase 1 fetches Day N+1 open price
#             → PENDING becomes LIVE at that open price
#             → Gap filter: if open < yesterday_close × 0.98
#               → cancel order (CANCELLED-GAP)
#
# DEPLOYMENT:
#   Max 50% of available cash per night.
#   Actual % scales with avg GPS of tonight's picks:
#     GPS ≥ 7.5  → 50% (bumper day)
#     GPS 6.5–7.5 → 35% (good day)
#     GPS 5.5–6.5 → 20% (average day)
#     GPS < 5.5  → 0%  (hold cash)
#   Cash floor: always keep ₹10,000 in reserve.
#
# STOP LOSS:
#   Default: buy price (zero loss tolerance)
#   Risk mode (after 3 nights improving score):
#     stop = buy_price × 0.90 (10% room to run)
#
# SELL TRIGGERS (priority order):
#   1. Price hits stop loss → SELL-STOP
#   2. Score < 5.0 or REDUCE/SELL verdict → SELL-SCORE
#   3. Profit ≥ 8% AND new STRONG BUY available → SELL-ROTATE
#   4. Day 21 close-out → SELL-DAY21
#
# ACCUMULATE:
#   Stock reappears with score +0.5 higher → add up to
#   50% of original position (within 25% single-stock cap)
#
# WEEKLY REVIEW (Option C):
#   Prediction weights → auto-adjust (small, data-driven)
#   GPS threshold + deployment % → your confirmation
#   Sunday 7 PM IST → 55-min window → freeze if no input
# ============================================================

import json
import time
import sys
import threading
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Optional
import pytz

sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR, BASE_DIR
from memory.storage import (
    load_prices_csv, load_fundamentals_json,
    today_str, now_str
)
from utils.live_data import get_live_quote

IST = pytz.timezone('Asia/Kolkata')

# ── Portfolio constants ───────────────────────────────────────
STARTING_CAPITAL     = 100_000.0   # ₹1,00,000
CASH_FLOOR           = 10_000.0    # always keep ₹10k reserve
MAX_SINGLE_STOCK_PCT = 0.25        # 25% max per stock
ACCUMULATE_MIN_DELTA = 0.5         # score must improve by 0.5+ to accumulate
RISK_MODE_NIGHTS     = 3           # nights of improving score before risk mode
DEFAULT_STOP_PCT     = 0.03        # 3% below entry (default tight stop)
RISK_MODE_STOP_PCT   = 0.07        # 7% below entry in risk mode (wider room to run)
ROTATE_PROFIT_PCT    = 0.08        # 8% profit triggers rotation check
GAP_FILTER_PCT       = 0.02        # cancel if open < prev_close × (1 - 0.02)
STOP_COOLDOWN_DAYS   = 3           # days to block re-entry after a stop-out

# GPS → deployment percentage mapping
GPS_DEPLOYMENT_MAP = {
    7.5: 0.50,   # bumper day
    6.5: 0.35,   # good day
    5.5: 0.20,   # average day
    0.0: 0.00,   # hold cash
}

# ── File paths ────────────────────────────────────────────────
PORTFOLIO_FILE   = DATA_DIR / "portfolio_state.json"
PORTFOLIO_EXCEL  = DATA_DIR / "excel" / "portfolio_tracker.xlsx"
WEEKLY_REVIEW_DIR = BASE_DIR / "reports" / "weekly_reviews"
WEEKLY_REVIEW_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ══════════════════════════════════════════════════════════════

def _empty_state() -> dict:
    return {
        'started':          today_str(),
        'day':              0,
        'cash':             STARTING_CAPITAL,
        'total_invested':   0.0,
        'starting_capital': STARTING_CAPITAL,
        'positions':        {},   # symbol → position dict
        'pending_orders':   [],   # orders waiting for next open
        'closed_trades':    [],   # completed trades
        'daily_log':        [],   # one entry per day
        'weight_changes':   [],   # parameter change audit trail
        # Tunable params (Option C — weekly review adjusts these)
        'params': {
            'gps_threshold':     6.0,
            'gps_deployment_map': {str(k): v for k, v in GPS_DEPLOYMENT_MAP.items()},
            'cash_floor':        CASH_FLOOR,
            'max_single_pct':    MAX_SINGLE_STOCK_PCT,
            'rotate_profit_pct': ROTATE_PROFIT_PCT,
            'gap_filter_pct':    GAP_FILTER_PCT,
            'risk_mode_nights':  RISK_MODE_NIGHTS,
        },
        'week_summaries':   {},   # week_1, week_2, week_3
    }


def load_state() -> dict:
    if PORTFOLIO_FILE.exists():
        try:
            with open(PORTFOLIO_FILE) as f:
                state = json.load(f)
            # Backfill missing keys lost to file corruption/recovery
            if 'params' not in state:
                state['params'] = _empty_state()['params']
            return state
        except Exception:
            pass
    return _empty_state()


def save_state(state: dict):
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)


# ══════════════════════════════════════════════════════════════
# PRICE HELPERS
# ══════════════════════════════════════════════════════════════

def get_eod_price(symbol: str) -> Optional[float]:
    """Get today's closing price from local CSV."""
    df = load_prices_csv(symbol, days=3)
    if df is not None and not df.empty:
        return float(df['Close'].iloc[-1])
    return None


def get_opening_price(symbol: str, target_date: str = None) -> Optional[float]:
    """
    Get opening price for target_date (default: today).
    Used for Option B entry price fill.
    """
    df = load_prices_csv(symbol, days=5)
    if df is None or df.empty:
        return None
    df.index = df.index.normalize()
    if target_date:
        dt = datetime.strptime(target_date, '%Y-%m-%d').date()
        matches = df[df.index.date == dt]
        if not matches.empty:
            return float(matches['Open'].iloc[0])
    else:
        return float(df['Open'].iloc[-1])
    return None


def get_live_price(symbol: str) -> Optional[float]:
    """
    Get live price for Day 21 review.
    Falls back to last closing price if market closed.
    """
    quote = get_live_quote(symbol)
    price = quote.get('last_price') or quote.get('current')
    if price:
        return float(price)
    return get_eod_price(symbol)


def _portfolio_value(state: dict) -> float:
    """
    Total portfolio value = cash + position values + pending reserved cash.

    FIX: Pending orders have cash already deducted from state['cash']
    when created. Must add it back so portfolio value is accurate.
    Without this, pending orders show as phantom losses.
    """
    total = state['cash']
    # Add market value of open positions
    for sym, pos in state['positions'].items():
        price = get_eod_price(sym) or pos['entry_price']
        total += price * pos['qty']
    # Add cash reserved for pending orders (already deducted from cash)
    for order in state.get('pending_orders', []):
        total += order.get('allocated', 0)
    return round(total, 2)


# ══════════════════════════════════════════════════════════════
# DEPLOYMENT LOGIC
# ══════════════════════════════════════════════════════════════

def _get_deployment_pct(avg_gps: float, params: dict) -> float:
    """
    Return deployment % based on avg GPS of tonight's picks.
    Uses tunable GPS_DEPLOYMENT_MAP from params.
    """
    gps_map = {float(k): v for k, v in
               params.get('gps_deployment_map',
               {str(k): v for k, v in GPS_DEPLOYMENT_MAP.items()}).items()}
    thresholds = sorted(gps_map.keys(), reverse=True)
    for threshold in thresholds:
        if avg_gps >= threshold:
            return gps_map[threshold]
    return 0.0


def _available_to_deploy(state: dict) -> float:
    """Cash available after respecting the floor."""
    floor = state['params'].get('cash_floor', CASH_FLOOR)
    return max(0.0, state['cash'] - floor)


def _max_deploy_tonight(state: dict, avg_gps: float) -> float:
    """Max ₹ to deploy tonight = 50% of available × GPS scaling."""
    available = _available_to_deploy(state)
    gps_pct   = _get_deployment_pct(avg_gps, state['params'])
    return round(available * gps_pct, 2)


def _score_weighted_allocation(stocks: list, total_budget: float,
                                max_single: float) -> dict:
    """
    Allocate total_budget across stocks proportionally by score.
    Caps any single stock at max_single (₹).
    Returns {symbol: allocated_₹}
    """
    if not stocks or total_budget <= 0:
        return {}

    total_score = sum(s['final_score'] for s in stocks)
    if total_score == 0:
        return {}

    allocations = {}
    for s in stocks:
        raw = (s['final_score'] / total_score) * total_budget
        allocations[s['symbol']] = round(min(raw, max_single), 2)

    # Redistribute any capped excess
    total_allocated = sum(allocations.values())
    if total_allocated < total_budget * 0.95:
        uncapped = [s for s in stocks
                    if allocations[s['symbol']] < max_single]
        if uncapped:
            leftover   = total_budget - total_allocated
            extra_score = sum(s['final_score'] for s in uncapped)
            for s in uncapped:
                extra = (s['final_score'] / extra_score) * leftover
                allocations[s['symbol']] = round(
                    min(allocations[s['symbol']] + extra, max_single), 2)

    return allocations


# ══════════════════════════════════════════════════════════════
# NIGHTLY DECISION ENGINE
# ══════════════════════════════════════════════════════════════

def run_nightly_portfolio(council_results: dict,
                           state: dict,
                           print_output: bool = True) -> dict:
    """
    Main nightly portfolio decision function.
    Called from night_runner.py after Phase 3 council run.

    Args:
        council_results: output of run_full_pipeline()
        state:           current portfolio state (loaded before call)

    Returns updated state dict (caller must save_state()).
    """
    now         = datetime.now(IST)
    today       = today_str()
    state['day'] = state.get('day', 0) + 1
    day_num     = state['day']
    # Safety guard: backfill 'params' if missing (e.g. recovered from corruption)
    if 'params' not in state:
        state['params'] = _empty_state()['params']
    params      = state['params']

    stocks      = council_results.get('stocks', [])
    market_snap = council_results.get('market', {})
    avg_gps     = (sum(s.get('gps', 0) for s in stocks) / len(stocks)
                   if stocks else 0)

    if print_output:
        pv = _portfolio_value(state)
        pl = pv - STARTING_CAPITAL
        print(f"\n{'═'*60}")
        print(f"  💼  PORTFOLIO ENGINE — Day {day_num}/21")
        print(f"  Portfolio: ₹{pv:,.0f}  P&L: ₹{pl:+,.0f} "
              f"({pl/STARTING_CAPITAL*100:+.1f}%)")
        print(f"  Cash: ₹{state['cash']:,.0f}  "
              f"Positions: {len(state['positions'])}")
        print('═'*60)

    day_log = {
        'date':            today,
        'day':             day_num,
        'portfolio_value': _portfolio_value(state),
        'cash':            state['cash'],
        'avg_gps':         round(avg_gps, 2),
        'market_score':    market_snap.get('market_score', 0),
        'actions':         [],
        'buys':            0,
        'sells':           0,
        'skipped':         False,
    }

    # ── Step 1: Fill pending orders (Option B) ────────────────
    _fill_pending_orders(state, today, day_log, print_output)

    # ── Step 2: Mark all positions to market ──────────────────
    _mark_to_market(state, print_output)

    # ── Step 3: Check stop losses ─────────────────────────────
    _check_stop_losses(state, day_log, print_output)

    # ── Step 4: Check score collapses ────────────────────────
    _check_score_collapses(state, stocks, day_log, print_output)

    # ── Step 5: Upgrade stop modes ───────────────────────────
    _upgrade_stop_modes(state, stocks, print_output)

    # ── Step 6: Check accumulations ──────────────────────────
    _check_accumulations(state, stocks, day_log, print_output)

    # ── Step 7: Rotation check ───────────────────────────────
    _check_rotation(state, stocks, day_log, print_output)

    # ── Step 8: New deployments ───────────────────────────────
    _deploy_new_capital(state, stocks, avg_gps, day_log,
                        print_output, params)

    # ── Step 9: Day summary ───────────────────────────────────
    pv_end   = _portfolio_value(state)
    pl_today = pv_end - day_log['portfolio_value']

    day_log.update({
        'portfolio_value_end': pv_end,
        'daily_pl':            round(pl_today, 2),
        'daily_pl_pct':        round(pl_today / day_log['portfolio_value'] * 100
                                     if day_log['portfolio_value'] else 0, 2),
        'total_pl':            round(pv_end - STARTING_CAPITAL, 2),
        'total_return_pct':    round((pv_end - STARTING_CAPITAL)
                                     / STARTING_CAPITAL * 100, 2),
        'positions_held':      len(state['positions']),
    })
    state.setdefault('daily_log', []).append(day_log)

    # Determine day mode for report
    if avg_gps >= 7.5:   day_mode = 'AGGRESSIVE'
    elif avg_gps >= 6.5: day_mode = 'NORMAL'
    elif avg_gps >= 5.5: day_mode = 'CAUTIOUS'
    else:                day_mode = 'CASH'
    day_log['day_mode'] = day_mode

    if print_output:
        print(f"\n  Day {day_num} complete | Mode: {day_mode}")
        print(f"  Portfolio: ₹{pv_end:,.0f}  "
              f"Daily P&L: ₹{pl_today:+,.0f} "
              f"({pl_today/day_log['portfolio_value']*100:+.1f}%)")
        print(f"  Total return: {day_log['total_return_pct']:+.1f}%")

    return state


# ══════════════════════════════════════════════════════════════
# STEP FUNCTIONS
# ══════════════════════════════════════════════════════════════

def _fill_pending_orders(state: dict, today: str,
                          day_log: dict, print_output: bool):
    """
    Fill yesterday's PENDING orders at today's open price.
    Apply gap filter: cancel if open < prev_close × (1 - GAP_FILTER_PCT).
    """
    filled   = []
    cancelled = []
    gap_pct  = state['params'].get('gap_filter_pct', GAP_FILTER_PCT)

    for order in state.get('pending_orders', []):
        sym         = order['symbol']
        open_price  = get_opening_price(sym)
        prev_close  = order.get('ref_close')

        if open_price is None:
            cancelled.append({**order, 'reason': 'NO_OPEN_PRICE'})
            continue

        # Gap filter
        if prev_close and open_price < prev_close * (1 - gap_pct):
            gap = (prev_close - open_price) / prev_close * 100
            if print_output:
                print(f"  ❌ CANCELLED-GAP: {sym} opened ₹{open_price:.2f} "
                      f"({gap:.1f}% below prev close ₹{prev_close:.2f})")
            cancelled.append({**order, 'reason': 'CANCELLED-GAP',
                               'open_price': open_price})
            day_log['actions'].append(
                f"CANCELLED-GAP {sym} gap={gap:.1f}%")
            state['cash'] += order['allocated']   # return cash
            continue

        # Fill the order
        qty         = int(order['allocated'] / open_price)
        if qty < 1:
            cancelled.append({**order, 'reason': 'INSUFFICIENT-FUNDS'})
            state['cash'] += order['allocated']
            continue

        actual_cost = round(qty * open_price, 2)
        # Return unspent portion
        state['cash'] += order['allocated'] - actual_cost

        position = {
            'symbol':        sym,
            'entry_date':    today,
            'entry_price':   round(open_price, 2),
            'qty':           qty,
            'invested':      actual_cost,
            'stop_loss':     round(open_price * (1 - DEFAULT_STOP_PCT), 2),  # 3% below entry
            'stop_mode':     'TIGHT',
            'improving_nights': 0,
            'entry_score':   order.get('score', 5.0),
            'latest_score':  order.get('score', 5.0),
            'score_history': [order.get('score', 5.0)],
            'sector':        order.get('sector', ''),
            'verdict':       order.get('verdict', ''),
            'gps':           order.get('gps', 0),
            'pred_1d':       order.get('pred_1d'),
            'pred_3d':       order.get('pred_3d'),
            'pred_7d':       order.get('pred_7d'),
            'pred_14d':      order.get('pred_14d'),
            'pred_21d':      order.get('pred_21d'),
        }

        # If symbol already held (accumulation from pending)
        if sym in state['positions']:
            existing = state['positions'][sym]
            total_qty    = existing['qty'] + qty
            total_cost   = existing['invested'] + actual_cost
            avg_price    = total_cost / total_qty
            existing.update({
                'qty':       total_qty,
                'invested':  total_cost,
                'entry_price': round(avg_price, 2),
                'stop_loss': round(avg_price * (1 - DEFAULT_STOP_PCT), 2),
            })
            if print_output:
                print(f"  ✅ ACCUMULATED: {sym} +{qty} @ ₹{open_price:.2f} "
                      f"avg ₹{avg_price:.2f}")
            day_log['actions'].append(
                f"ACCUMULATED {sym} qty={qty} @ ₹{open_price:.2f}")
        else:
            state['positions'][sym] = position
            if print_output:
                print(f"  ✅ FILLED: {sym} {qty} shares @ ₹{open_price:.2f} "
                      f"= ₹{actual_cost:,.0f}")
            day_log['actions'].append(
                f"BUY-FILLED {sym} qty={qty} @ ₹{open_price:.2f}")
            day_log['buys'] += 1

        filled.append(sym)

    # Clear pending orders
    state['pending_orders'] = []


def _mark_to_market(state: dict, print_output: bool):
    """Update current_price and unrealised P&L for all positions."""
    for sym, pos in state['positions'].items():
        price = get_eod_price(sym) or pos['entry_price']
        pos['current_price']  = round(price, 2)
        pos['current_value']  = round(price * pos['qty'], 2)
        pos['unrealised_pl']  = round(pos['current_value'] - pos['invested'], 2)
        pos['unrealised_pct'] = round(pos['unrealised_pl'] / pos['invested'] * 100, 2)
        pos['days_held'] = (
            datetime.strptime(today_str(), '%Y-%m-%d') -
            datetime.strptime(pos['entry_date'], '%Y-%m-%d')
        ).days + 1


def _close_position(state: dict, sym: str, reason: str,
                    day_log: dict, print_output: bool):
    """Close a position and log the closed trade."""
    pos   = state['positions'].pop(sym, None)
    if not pos:
        return

    exit_price = pos.get('current_price', pos['entry_price'])
    proceeds   = round(exit_price * pos['qty'], 2)
    pl         = round(proceeds - pos['invested'], 2)
    pl_pct     = round(pl / pos['invested'] * 100, 2)

    state['cash'] += proceeds

    trade = {
        'symbol':       sym,
        'buy_date':     pos['entry_date'],
        'sell_date':    today_str(),
        'entry_price':  pos['entry_price'],
        'exit_price':   exit_price,
        'qty':          pos['qty'],
        'invested':     pos['invested'],
        'proceeds':     proceeds,
        'pl':           pl,
        'pl_pct':       pl_pct,
        'reason':       reason,
        'days_held':    pos.get('days_held', 0),
        'entry_score':  pos.get('entry_score', 0),
        'exit_score':   pos.get('latest_score', 0),
        'sector':       pos.get('sector', ''),
    }
    state.setdefault('closed_trades', []).append(trade)

    icon = "💚" if pl > 0 else "🔴"
    if print_output:
        print(f"  {icon} SOLD {sym} @ ₹{exit_price:.2f} | "
              f"P&L: ₹{pl:+,.0f} ({pl_pct:+.1f}%) | {reason}")

    day_log['actions'].append(
        f"{reason} {sym} pl=₹{pl:+,.0f} ({pl_pct:+.1f}%)")
    day_log['sells'] += 1


def _check_stop_losses(state: dict, day_log: dict, print_output: bool):
    """Close any position where current price ≤ stop loss."""
    to_close = []
    for sym, pos in state['positions'].items():
        if pos.get('current_price', 999999) <= pos['stop_loss']:
            to_close.append(sym)

    for sym in to_close:
        _close_position(state, sym, 'SELL-STOP', day_log, print_output)


def _check_score_collapses(state: dict, stocks: list,
                             day_log: dict, print_output: bool):
    """Close positions where tonight's council gave score < 5.0 or REDUCE/SELL."""
    tonight = {s['symbol']: s for s in stocks}
    to_close = []

    for sym in list(state['positions'].keys()):
        if sym not in tonight:
            continue
        result = tonight[sym]
        verdict = result.get('verdict', '').upper()
        score   = result.get('final_score', 5.0)

        if score < 5.0 or verdict in ('REDUCE', 'SELL', 'STRONG SELL'):
            to_close.append(sym)

    for sym in to_close:
        _close_position(state, sym, 'SELL-SCORE', day_log, print_output)


def _upgrade_stop_modes(state: dict, stocks: list, print_output: bool):
    """
    Update score history per position.
    After RISK_MODE_NIGHTS consecutive improving scores:
    upgrade to risk mode (stop = entry × 0.90).
    """
    tonight = {s['symbol']: s for s in stocks}

    for sym, pos in state['positions'].items():
        if sym not in tonight:
            continue

        new_score = tonight[sym].get('final_score', pos['latest_score'])
        history   = pos.setdefault('score_history', [pos['entry_score']])
        history.append(new_score)
        pos['latest_score'] = new_score

        # Count consecutive improvements
        if len(history) >= 2 and history[-1] > history[-2]:
            pos['improving_nights'] = pos.get('improving_nights', 0) + 1
        else:
            pos['improving_nights'] = 0

        nights_needed = state['params'].get('risk_mode_nights', RISK_MODE_NIGHTS)
        if (pos.get('stop_mode') == 'TIGHT' and
                pos['improving_nights'] >= nights_needed):
            pos['stop_mode'] = 'RISK'
            pos['stop_loss'] = round(pos['entry_price'] * (1 - RISK_MODE_STOP_PCT), 2)
            if print_output:
                print(f"  🔓 RISK MODE: {sym} stop → ₹{pos['stop_loss']:.2f} "
                      f"(−{RISK_MODE_STOP_PCT*100:.0f}% from entry ₹{pos['entry_price']:.2f})")


def _check_accumulations(state: dict, stocks: list,
                          day_log: dict, print_output: bool):
    """
    If a held stock reappears with score +0.5 higher → accumulate.
    Creates a PENDING order for tomorrow's open (Option B).
    """
    tonight   = {s['symbol']: s for s in stocks}
    pv        = _portfolio_value(state)
    max_single = pv * state['params'].get('max_single_pct', MAX_SINGLE_STOCK_PCT)
    min_delta  = state['params'].get('accumulate_min_delta',
                                     ACCUMULATE_MIN_DELTA)

    for sym, pos in state['positions'].items():
        if sym not in tonight:
            continue

        new_score   = tonight[sym].get('final_score', 0)
        score_delta = new_score - pos.get('latest_score', pos['entry_score'])

        if score_delta < min_delta:
            continue

        current_val = pos.get('current_value', pos['invested'])
        room        = max_single - current_val
        if room < 1000:   # need at least ₹1000 room
            continue

        add_amount  = min(room, pos['invested'] * 0.5,
                          _available_to_deploy(state) * 0.3)
        add_amount  = round(add_amount, 2)

        if add_amount < 500:
            continue

        # Reserve cash now, fill at tomorrow's open
        state['cash'] -= add_amount
        state['pending_orders'].append({
            'symbol':    sym,
            'allocated': add_amount,
            'score':     new_score,
            'sector':    pos.get('sector', ''),
            'verdict':   tonight[sym].get('verdict', ''),
            'gps':       tonight[sym].get('gps', 0),
            'ref_close': pos.get('current_price'),
            'reason':    'ACCUMULATE',
        })

        if print_output:
            print(f"  📈 ACCUMULATE PENDING: {sym} +₹{add_amount:,.0f} "
                  f"score {pos['latest_score']:.1f}→{new_score:.1f} "
                  f"(+{score_delta:.1f}) fills tomorrow open")
        day_log['actions'].append(
            f"ACCUMULATE-PENDING {sym} +₹{add_amount:,.0f}")


def _check_rotation(state: dict, stocks: list,
                     day_log: dict, print_output: bool):
    """
    If any position has ≥ 8% profit AND a new STRONG BUY exists
    AND available cash < ₹5,000 → sell winner, create PENDING buy.
    """
    if _available_to_deploy(state) >= 5000:
        return   # enough cash, no need to rotate

    rotate_pct    = state['params'].get('rotate_profit_pct', ROTATE_PROFIT_PCT)
    strong_buys   = [s for s in stocks
                     if s.get('verdict', '').upper() in ('STRONG BUY', 'BUY')
                     and s['symbol'] not in state['positions']]

    if not strong_buys:
        return

    # Find best candidate to sell (highest profit %)
    candidates = [
        (sym, pos) for sym, pos in state['positions'].items()
        if pos.get('unrealised_pct', 0) >= rotate_pct * 100
    ]
    if not candidates:
        return

    # Sell the most profitable
    sell_sym = max(candidates, key=lambda x: x[1].get('unrealised_pct', 0))[0]
    _close_position(state, sell_sym, 'SELL-ROTATE', day_log, print_output)

    # Queue best strong buy as pending
    best_buy  = max(strong_buys, key=lambda s: s.get('final_score', 0))
    available = _available_to_deploy(state)
    allocated = round(min(available * 0.5, available), 2)

    if allocated >= 500:
        state['cash'] -= allocated
        state['pending_orders'].append({
            'symbol':    best_buy['symbol'],
            'allocated': allocated,
            'score':     best_buy.get('final_score', 5),
            'sector':    best_buy.get('sector', ''),
            'verdict':   best_buy.get('verdict', ''),
            'gps':       best_buy.get('gps', 0),
            'ref_close': get_eod_price(best_buy['symbol']),
            'pred_1d':   best_buy.get('pred_1d'),
            'pred_3d':   best_buy.get('pred_3d'),
            'pred_7d':   best_buy.get('pred_7d'),
            'reason':    'ROTATE-BUY',
        })
        if print_output:
            print(f"  🔄 ROTATE: Sold {sell_sym} → "
                  f"Buying {best_buy['symbol']} tomorrow open")
        day_log['actions'].append(
            f"ROTATE {sell_sym}→{best_buy['symbol']}")


def _deploy_new_capital(state: dict, stocks: list, avg_gps: float,
                         day_log: dict, print_output: bool, params: dict):
    """
    Deploy tonight's budget into new picks.
    Score-weighted allocation. Creates PENDING orders for tomorrow's open.
    """
    budget = _max_deploy_tonight(state, avg_gps)

    if print_output:
        dep_pct = _get_deployment_pct(avg_gps, params)
        if dep_pct == 0:
            print(f"\n  💤 CASH DAY — avg GPS {avg_gps:.1f} < 5.5, holding cash")
            day_log['skipped'] = True
            return
        print(f"\n  📊 Deployment: avg GPS {avg_gps:.1f} → "
              f"{dep_pct*100:.0f}% = ₹{budget:,.0f} to deploy")

    if budget < 500:
        if print_output:
            print(f"  💤 Budget ₹{budget:,.0f} too small — skipping")
        day_log['skipped'] = True
        return

    # Filter: new stocks only (not already held, pending, or recently stopped out)
    held_syms    = set(state['positions'].keys())
    pending_syms = {o['symbol'] for o in state.get('pending_orders', [])}
    min_score    = params.get('gps_threshold', 6.0)

    # Build cooldown set: stocks stopped out within last STOP_COOLDOWN_DAYS days
    from datetime import datetime, timedelta
    cutoff_date = (datetime.now() - timedelta(days=STOP_COOLDOWN_DAYS)).strftime('%Y-%m-%d')
    recently_stopped = {
        t['symbol'] for t in state.get('closed_trades', [])
        if t.get('reason') == 'SELL-STOP'
        and str(t.get('sell_date', '')) >= cutoff_date
    }

    candidates = [
        s for s in stocks
        if s['symbol'] not in held_syms
        and s['symbol'] not in pending_syms
        and s['symbol'] not in recently_stopped
        and s.get('final_score', 0) >= min_score
        and s.get('verdict', '').upper() not in ('REDUCE', 'SELL', 'STRONG SELL')
    ]

    if not candidates:
        if print_output:
            print(f"  💤 No new qualifying stocks tonight")
        day_log['skipped'] = True
        return

    # Score-weighted allocation
    pv         = _portfolio_value(state)
    max_single = pv * params.get('max_single_pct', MAX_SINGLE_STOCK_PCT)
    allocs     = _score_weighted_allocation(candidates[:5], budget, max_single)

    for sym, amount in allocs.items():
        if amount < 500:
            continue

        stock   = next(s for s in candidates if s['symbol'] == sym)
        eod_ref = get_eod_price(sym)

        state['cash'] -= amount
        state['pending_orders'].append({
            'symbol':    sym,
            'allocated': amount,
            'score':     stock.get('final_score', 5),
            'sector':    stock.get('sector', ''),
            'verdict':   stock.get('verdict', ''),
            'gps':       stock.get('gps', 0),
            'ref_close': eod_ref,
            'pred_1d':   stock.get('analyses', {}).get('pred_1d'),
            'pred_3d':   stock.get('analyses', {}).get('pred_3d'),
            'pred_7d':   stock.get('analyses', {}).get('pred_7d'),
            'reason':    'NEW-BUY',
        })

        if print_output:
            print(f"  🛒 BUY PENDING: {sym} ₹{amount:,.0f} "
                  f"(score:{stock.get('final_score',0):.1f} "
                  f"GPS:{stock.get('gps',0):.1f}) fills tomorrow open")
        day_log['actions'].append(
            f"BUY-PENDING {sym} ₹{amount:,.0f}")
        day_log['buys'] += 1


# ══════════════════════════════════════════════════════════════
# DAY 21 REVIEW MODE
# ══════════════════════════════════════════════════════════════

def run_day21_review(state: dict):
    """
    Interactive Day 21 review — shows live prices for all positions
    and lets you decide: SELL / HOLD / PARTIAL SELL for each.
    Target: complete in 15 minutes.
    """
    print(f"\n{'═'*60}")
    print(f"  🏁  DAY 21 REVIEW — FINAL POSITIONS")
    print(f"  Live prices via NSE/Finnhub/Yahoo Finance")
    print('═'*60)

    if not state['positions']:
        print("  No open positions to review.")
        _print_final_summary(state)
        return

    pv_start = _portfolio_value(state)

    for sym, pos in list(state['positions'].items()):
        live_price = get_live_price(sym)
        if live_price:
            pos['current_price'] = live_price
            pos['current_value'] = round(live_price * pos['qty'], 2)
            pos['unrealised_pl'] = round(
                pos['current_value'] - pos['invested'], 2)
            pos['unrealised_pct'] = round(
                pos['unrealised_pl'] / pos['invested'] * 100, 2)

        pl_icon = "💚" if pos.get('unrealised_pl', 0) > 0 else "🔴"
        print(f"""
  {sym} ({pos.get('sector','')})
    Held {pos.get('days_held',0)} days | Entry ₹{pos['entry_price']:.2f} → Live ₹{live_price:.2f if live_price else '?'}
    {pl_icon} P&L: ₹{pos.get('unrealised_pl',0):+,.0f} ({pos.get('unrealised_pct',0):+.1f}%)
    Score: {pos.get('entry_score',0):.1f} → {pos.get('latest_score',0):.1f}
    Google Finance: https://www.google.com/finance/quote/{sym}:NSE

  Options:
    [1] Sell all now  @ ₹{live_price:.2f if live_price else '?'}
    [2] Hold (extend beyond Day 21)
    [3] Partial sell (50%)
""")

        choice = _timed_input("  Your choice (1/2/3): ", timeout=120)

        if choice == '1':
            _close_position(state, sym, 'SELL-DAY21',
                            {'actions': [], 'sells': 0}, True)
        elif choice == '3':
            half_qty = pos['qty'] // 2
            if half_qty > 0:
                proceeds      = round(half_qty * (live_price or pos['entry_price']), 2)
                state['cash'] += proceeds
                pos['qty']    -= half_qty
                pos['invested'] = round(pos['qty'] * pos['entry_price'], 2)
                print(f"  Partial sold {half_qty} shares → ₹{proceeds:,.0f} to cash")
            else:
                print("  Too few shares for partial — holding all")
        else:
            print(f"  Holding {sym}")

    _print_final_summary(state)


def _timed_input(prompt: str, timeout: int = 120) -> str:
    """Input with timeout. Returns empty string on timeout."""
    result = ['']
    done   = threading.Event()

    def _get():
        try:
            result[0] = input(prompt).strip()
        except Exception:
            pass
        done.set()

    t = threading.Thread(target=_get, daemon=True)
    t.start()
    done.wait(timeout=timeout)
    if not done.is_set():
        print(f"\n  (No input — defaulting to Hold)")
    return result[0]


def _print_final_summary(state: dict):
    """Print complete 21-day portfolio performance summary."""
    pv_final   = _portfolio_value(state)
    total_pl   = pv_final - STARTING_CAPITAL
    total_ret  = total_pl / STARTING_CAPITAL * 100
    closed     = state.get('closed_trades', [])
    wins       = [t for t in closed if t['pl'] > 0]
    losses     = [t for t in closed if t['pl'] <= 0]

    print(f"\n{'═'*60}")
    print(f"  🏆  21-DAY PAPER PORTFOLIO FINAL RESULTS")
    print('═'*60)
    print(f"  Starting capital:  ₹{STARTING_CAPITAL:,.0f}")
    print(f"  Final value:       ₹{pv_final:,.0f}")
    print(f"  Total P&L:         ₹{total_pl:+,.0f} ({total_ret:+.1f}%)")
    print(f"\n  Closed trades: {len(closed)}")
    print(f"  Wins:  {len(wins)}  | Avg: "
          f"₹{sum(t['pl'] for t in wins)/len(wins):,.0f}" if wins else "  Wins:  0")
    print(f"  Losses:{len(losses)}  | Avg: "
          f"₹{sum(t['pl'] for t in losses)/len(losses):,.0f}" if losses else
          "  Losses:0")
    if closed:
        print(f"  Win rate: {len(wins)/len(closed)*100:.0f}%")
        best  = max(closed, key=lambda t: t['pl_pct'])
        worst = min(closed, key=lambda t: t['pl_pct'])
        print(f"  Best:  {best['symbol']} {best['pl_pct']:+.1f}%")
        print(f"  Worst: {worst['symbol']} {worst['pl_pct']:+.1f}%")

    print(f"\n  Open positions remaining: {len(state['positions'])}")
    for sym, pos in state['positions'].items():
        print(f"    {sym}: {pos.get('unrealised_pct',0):+.1f}%  "
              f"({pos.get('days_held',0)} days held)")
    print('═'*60)


# ══════════════════════════════════════════════════════════════
# WEEKLY PARAMETER REVIEW (Option C)
# ══════════════════════════════════════════════════════════════

def run_weekly_review(state: dict, week_num: int,
                       timeout_minutes: int = 55) -> dict:
    """
    Sunday 7 PM review. Auto-approves small signal weights.
    GPS threshold + deployment % need your Y/N.
    Freezes if no input within timeout.
    """
    print(f"\n{'╔'+'═'*58+'╗'}")
    print(f"║  📊 WEEKLY PARAMETER REVIEW — Week {week_num} Complete"
          + " " * (24 - len(str(week_num))) + "║")
    print(f"║  You have {timeout_minutes} min to respond "
          f"(deadline: auto-freeze)       ║")
    print(f"{'╚'+'═'*58+'╝'}")

    # ── Week performance ──────────────────────────────────────
    week_start_day = (week_num - 1) * 7 + 1
    week_log = [d for d in state.get('daily_log', [])
                if week_start_day <= d.get('day', 0) <= week_num * 7]
    week_trades = [t for t in state.get('closed_trades', [])
                   if _trade_in_week(t, week_num)]

    pv_start = (week_log[0]['portfolio_value']
                if week_log else STARTING_CAPITAL)
    pv_end   = (week_log[-1].get('portfolio_value_end', pv_start)
                if week_log else pv_start)
    week_pl  = pv_end - pv_start
    week_ret = week_pl / pv_start * 100 if pv_start else 0

    wins   = [t for t in week_trades if t['pl'] > 0]
    losses = [t for t in week_trades if t['pl'] <= 0]

    print(f"\n  WEEK {week_num} PERFORMANCE:")
    print(f"  Portfolio: ₹{pv_start:,.0f} → ₹{pv_end:,.0f} "
          f"({week_ret:+.1f}%)")
    print(f"  Trades closed: {len(week_trades)}  |  "
          f"Win rate: {len(wins)/len(week_trades)*100:.0f}%"
          if week_trades else
          f"  Trades closed: 0")

    if week_trades:
        avg_hold = sum(t.get('days_held', 0) for t in week_trades) / len(week_trades)
        print(f"  Avg hold: {avg_hold:.1f} days")
        if wins:
            best = max(week_trades, key=lambda t: t['pl_pct'])
            print(f"  Best: {best['symbol']} {best['pl_pct']:+.1f}%")
        if losses:
            worst = min(week_trades, key=lambda t: t['pl_pct'])
            print(f"  Worst: {worst['symbol']} {worst['pl_pct']:+.1f}%")

    # Save review to file
    review_data = {
        'week':       week_num,
        'date':       today_str(),
        'week_pl':    round(week_pl, 2),
        'week_ret':   round(week_ret, 2),
        'trades':     len(week_trades),
        'win_rate':   round(len(wins)/len(week_trades)*100, 0) if week_trades else 0,
        'decisions':  {},
    }

    # ── AUTO-APPROVED: prediction weights ────────────────────
    print(f"\n  AUTO-APPROVED (prediction signal weights):")
    _auto_adjust_pred_weights(state, week_trades, review_data)

    if week_num == 3:
        print(f"\n  WEEK 3: Parameter changes FROZEN (validation week)")
        _save_weekly_review(review_data, week_num)
        return state

    # ── YOUR APPROVAL: GPS threshold ──────────────────────────
    params     = state['params']
    current_gps = params.get('gps_threshold', 6.0)
    proposed_gps = current_gps

    # Propose adjustment based on week P&L
    if week_ret < -2:
        proposed_gps = round(current_gps + 0.5, 1)
        reason_gps = f"Week P&L {week_ret:+.1f}% → raise bar"
    elif week_ret > 10 and len(week_trades) >= 5:
        proposed_gps = round(current_gps - 0.3, 1)
        reason_gps = f"Strong week {week_ret:+.1f}% → slightly more aggressive"
    else:
        reason_gps  = "Performance neutral — no change proposed"

    if proposed_gps != current_gps:
        print(f"\n  [1] GPS Threshold: {current_gps} → {proposed_gps}")
        print(f"      Reason: {reason_gps}")
        print(f"      Impact: {'More selective' if proposed_gps > current_gps else 'More stocks'}")
        choice = _timed_input(
            f"      Approve? (Y/N) [{timeout_minutes}min timeout = auto-freeze]: ",
            timeout=timeout_minutes * 60)
        if choice.upper() == 'Y':
            params['gps_threshold'] = proposed_gps
            review_data['decisions']['gps_threshold'] = {
                'old': current_gps, 'new': proposed_gps, 'approved': True}
            print(f"      ✅ GPS threshold → {proposed_gps}")
            _log_weight_change(state, 'gps_threshold',
                               current_gps, proposed_gps, reason_gps)
        else:
            print(f"      ❌ Kept at {current_gps} (frozen)")
            review_data['decisions']['gps_threshold'] = {
                'old': current_gps, 'new': current_gps, 'approved': False}
    else:
        print(f"\n  [1] GPS Threshold: {current_gps} (no change proposed)")

    # ── YOUR APPROVAL: Deployment % ───────────────────────────
    cash_days = sum(1 for d in week_log if d.get('skipped'))
    if cash_days >= 4 and week_ret > 2:
        print(f"\n  [2] Deployment (GPS 6.5–7.5): 35% → 40%")
        print(f"      Reason: Bot skipped {cash_days} nights but market was up")
        choice = _timed_input(
            f"      Approve? (Y/N) [auto-freeze in {timeout_minutes}min]: ",
            timeout=timeout_minutes * 60)
        if choice.upper() == 'Y':
            gps_map = params.get('gps_deployment_map', {})
            gps_map['6.5'] = 0.40
            params['gps_deployment_map'] = gps_map
            review_data['decisions']['deploy_65'] = {
                'old': 0.35, 'new': 0.40, 'approved': True}
            print(f"      ✅ Deployment (GPS 6.5+) → 40%")
            _log_weight_change(state, 'deploy_gps_65', 0.35, 0.40,
                               f"Too many cash days ({cash_days})")
        else:
            print(f"      ❌ Kept at 35% (frozen)")

    review_data['frozen_at'] = now_str()
    _save_weekly_review(review_data, week_num)

    print(f"\n  ✅ Week {week_num} review complete. "
          f"Night runner starts at 8:00 PM IST.")
    return state


def _auto_adjust_pred_weights(state: dict, week_trades: list,
                               review_data: dict):
    """Auto-adjust prediction signal weights based on closed trade accuracy."""
    try:
        from bots.prediction_bot import load_learned_weights, DEFAULT_PRED_WEIGHTS
        learned = load_learned_weights()
        pw      = learned.get('pred_weights', dict(DEFAULT_PRED_WEIGHTS))
        changed = []

        # Simple accuracy proxy: if week was profitable, boost tech_momentum
        # (the strongest signal); if unprofitable, trim it
        if week_trades:
            week_pl = sum(t['pl'] for t in week_trades)
            wins    = [t for t in week_trades if t['pl'] > 0]
            win_rate = len(wins) / len(week_trades)

            if win_rate > 0.65 and pw.get('tech_momentum', 0.35) < 0.45:
                old = pw['tech_momentum']
                pw['tech_momentum'] = round(min(0.45, old * 1.10), 3)
                changed.append(
                    f"  tech_momentum: {old:.3f} → {pw['tech_momentum']:.3f} ↑")
            elif win_rate < 0.40 and pw.get('tech_momentum', 0.35) > 0.25:
                old = pw['tech_momentum']
                pw['tech_momentum'] = round(max(0.25, old * 0.90), 3)
                changed.append(
                    f"  tech_momentum: {old:.3f} → {pw['tech_momentum']:.3f} ↓")

            # Re-normalize
            total = sum(pw.values())
            if total > 0:
                pw = {k: round(v/total, 4) for k, v in pw.items()}

            # Re-clamp and re-normalize
            pw = {k: round(min(0.50, max(0.05, v)), 4) for k, v in pw.items()}
            total2 = sum(pw.values())
            if total2 > 0:
                pw = {k: round(v/total2, 4) for k, v in pw.items()}

            learned['pred_weights'] = pw
            _save_learned_weights(learned)

        for c in changed:
            print(c)
        if not changed:
            print(f"  No auto-adjustments needed (win rate stable)")

        review_data['auto_weight_changes'] = changed

    except Exception as e:
        print(f"  Auto-adjust skipped: {e}")


def _save_learned_weights(data: dict):
    from config import DATA_DIR
    path = DATA_DIR / "learned_weights.json"
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def _log_weight_change(state: dict, param: str,
                        old_val, new_val, reason: str):
    state.setdefault('weight_changes', []).append({
        'date':   today_str(),
        'param':  param,
        'old':    old_val,
        'new':    new_val,
        'reason': reason,
    })


def _trade_in_week(trade: dict, week_num: int) -> bool:
    try:
        sell_date = datetime.strptime(trade['sell_date'], '%Y-%m-%d').date()
        started   = datetime.strptime(
            trade.get('buy_date', today_str()), '%Y-%m-%d').date()
        # Use sell date to assign to week
        state_start = date.today() - timedelta(days=21)
        day_num  = (sell_date - state_start).days + 1
        return (week_num - 1) * 7 < day_num <= week_num * 7
    except Exception:
        return False


def _save_weekly_review(data: dict, week_num: int):
    path = WEEKLY_REVIEW_DIR / f"week_{week_num}_review.json"
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\n  📄 Review saved: {path.name}")


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Portfolio Engine')
    parser.add_argument('--status',   action='store_true',
                        help='Show current portfolio status')
    parser.add_argument('--day21',    action='store_true',
                        help='Run Day 21 interactive review')
    parser.add_argument('--review',   type=int, metavar='WEEK',
                        help='Run weekly review for WEEK (1/2/3)')
    args = parser.parse_args()

    state = load_state()

    if args.status:
        pv  = _portfolio_value(state)
        pl  = pv - STARTING_CAPITAL
        print(f"\n  Portfolio: ₹{pv:,.0f}  P&L: ₹{pl:+,.0f} "
              f"({pl/STARTING_CAPITAL*100:+.1f}%)")
        print(f"  Cash: ₹{state['cash']:,.0f}  "
              f"Day: {state.get('day',0)}/21")
        print(f"  Open positions: {len(state['positions'])}")
        for sym, pos in state['positions'].items():
            print(f"    {sym}: {pos.get('unrealised_pct',0):+.1f}%  "
                  f"₹{pos.get('unrealised_pl',0):+,.0f}")

    elif args.day21:
        run_day21_review(state)
        save_state(state)

    elif args.review:
        state = run_weekly_review(state, args.review)
        save_state(state)