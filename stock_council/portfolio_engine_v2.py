"""
portfolio_engine_v2.py — Model B: Daily Conviction Budget
============================================================
Strategy:
  - ₹10,000 injected as new capital every trading day
  - Day Confidence computed from council output:
      avg composite score + % stocks above 6.5 + strong-buy count
  - Daily budget deployed in tiers based on confidence:
      < 35%  → ₹0       (skip day, save budget)
      35-55% → ₹5,000   (half budget)
      55-70% → ₹7,500   (75% budget)
      > 70%  → ₹10,000  (full budget)
  - Top 1-2 stocks by council score are bought at next-day open
  - Position exits: when score drops below SELL_SCORE on next council,
    OR after MAX_HOLD_DAYS trading days (whichever comes first)
  - Unspent budget accumulates as cash reserve — deployed as bonus
    on the next ≥70% confidence day (up to 2× daily budget)
  - Runs fully independently of Model A; no shared state

Comparable metric:
  Return on Deployed Capital = (gains) / (total capital ever invested)
  vs Model A's Total Return = (value - 100000) / 100000

Usage (live, called from night_runner.py after Phase 3):
  from portfolio_engine_v2 import ModelBEngine
  engine = ModelBEngine()
  engine.run_day(date_str, council_stocks, price_map, print_output=True)
  engine.save()

Usage (backtest):
  See run_backtest_v2.py
"""

import json
import os
from pathlib import Path
from datetime import datetime

# ── Constants ─────────────────────────────────────────────────
DAILY_BUDGET      = 10_000    # ₹ injected each trading day
MAX_POSITIONS     = 2         # max concurrent open positions
MIN_BUY_SCORE     = 6.5       # minimum council score to buy
SELL_SCORE        = 6.0       # exit if score drops below this
MAX_HOLD_DAYS     = 5         # force-exit after this many trading days
MAX_RESERVE_USE   = 2.0       # can deploy up to 2× daily budget on very high conf days

# Confidence thresholds → fraction of daily budget to deploy
CONF_TIERS = [
    (0.35, 0.00),    # < 35%  → skip
    (0.55, 0.50),    # 35-55% → 50%  (₹5,000)
    (0.70, 0.75),    # 55-70% → 75%  (₹7,500)
    (1.01, 1.00),    # > 70%  → 100% (₹10,000)
]

STATE_FILE = Path(__file__).parent / "data" / "portfolio_state_v2.json"


# ══════════════════════════════════════════════════════════════
# DAY CONFIDENCE SCORER
# ══════════════════════════════════════════════════════════════

def compute_day_confidence(stocks: list) -> float:
    """
    0–1 score from council output for one day.

    Components (all normalised to 0–1):
      - avg composite score   (weight 0.5)   5.0=0  7.0=1
      - % stocks scoring ≥6.5 (weight 0.3)
      - % STRONG BUY verdicts (weight 0.2)
    """
    if not stocks:
        return 0.0

    composites = [s.get('composite', s.get('final_score', 0)) for s in stocks]
    avg = sum(composites) / len(composites)

    avg_norm   = min(1.0, max(0.0, (avg - 5.0) / 2.0))   # 5→0  7→1
    pct_high   = sum(1 for c in composites if c >= 6.5) / len(composites)
    pct_strong = sum(
        1 for s in stocks
        if 'STRONG' in str(s.get('verdict', '')).upper()
    ) / len(stocks)

    return round(avg_norm * 0.5 + pct_high * 0.3 + pct_strong * 0.2, 4)


def confidence_to_budget(conf: float, reserve: float) -> float:
    """
    Return how much cash to deploy given confidence score and reserve.
    On very high confidence (>70%), draws from reserve too (up to 2× budget).
    """
    fraction = 0.0
    for threshold, frac in CONF_TIERS:
        if conf < threshold:
            fraction = frac
            break

    base_deploy = DAILY_BUDGET * fraction

    # Bonus: on full-confidence days, tap reserve (up to MAX_RESERVE_USE × budget)
    if fraction == 1.0 and reserve > 0:
        max_bonus = DAILY_BUDGET * (MAX_RESERVE_USE - 1.0)
        bonus = min(reserve, max_bonus)
        base_deploy += bonus

    return round(base_deploy, 2)


# ══════════════════════════════════════════════════════════════
# STOCK PICKER
# ══════════════════════════════════════════════════════════════

def pick_stocks(stocks: list, budget: float, held_symbols: set) -> list:
    """
    Pick up to MAX_POSITIONS stocks to buy, excluding already-held symbols.
    Returns list of {'symbol', 'score', 'alloc'} — alloc is ₹ amount.
    """
    if budget <= 0:
        return []

    candidates = [
        s for s in stocks
        if s.get('composite', s.get('final_score', 0)) >= MIN_BUY_SCORE
        and s.get('symbol') not in held_symbols
    ]
    candidates.sort(key=lambda x: x.get('composite', x.get('final_score', 0)), reverse=True)
    candidates = candidates[:MAX_POSITIONS]

    if not candidates:
        return []

    alloc_each = round(budget / len(candidates), 2)
    return [
        {
            'symbol': c['symbol'],
            'score':  round(c.get('composite', c.get('final_score', 0)), 2),
            'alloc':  alloc_each,
        }
        for c in candidates
    ]


# ══════════════════════════════════════════════════════════════
# MODEL B ENGINE
# ══════════════════════════════════════════════════════════════

class ModelBEngine:
    """
    Stateful engine for Model B.  Call run_day() once per trading day.
    State is persisted to data/portfolio_state_v2.json.
    """

    def __init__(self, state_file: Path = STATE_FILE):
        self.state_file = state_file
        self._load()

    # ── Persistence ───────────────────────────────────────────

    def _load(self):
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    self.state = json.load(f)
                return
            except Exception:
                pass
        self.state = self._blank_state()

    def _blank_state(self) -> dict:
        return {
            'model':            'B_conviction_budget',
            'daily_budget':     DAILY_BUDGET,
            'cash':             0.0,          # unspent / accumulated reserve
            'total_injected':   0.0,          # cumulative capital added
            'total_deployed':   0.0,          # cumulative capital actually invested
            'total_gains':      0.0,          # realised gains
            'positions':        [],           # open positions
            'closed_positions': [],           # closed with P&L
            'daily_log':        [],           # per-day summary
            'current_day':      0,
            'portfolio_value':  0.0,
        }

    def save(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)

    # ── Portfolio value ───────────────────────────────────────

    def portfolio_value(self, price_map: dict) -> float:
        """Cash + market value of all open positions."""
        pos_value = sum(
            p['shares'] * price_map.get(p['symbol'], p['buy_price'])
            for p in self.state['positions']
        )
        return round(self.state['cash'] + pos_value, 2)

    # ── Main entry point ──────────────────────────────────────

    def run_day(
        self,
        date_str:       str,
        council_stocks: list,
        price_map:      dict,    # {symbol: today_close}  (for MTM valuation)
        next_open_map:  dict,    # {symbol: next_day_open}  (for buys)
        prev_open_map:  dict,    # {symbol: prev exec day open}  (for sells)
        trading_day_n:  int = None,
        print_output:   bool = True,
    ) -> dict:
        """
        Process one trading day for Model B.

        Args:
            date_str:       'YYYY-MM-DD' of today's council
            council_stocks: list of {'symbol', 'composite'/'final_score', 'verdict'}
            price_map:      today's closing prices (MTM)
            next_open_map:  next trading day's open prices (for new buys)
            prev_open_map:  today's open prices (for selling yesterday's buys)
            trading_day_n:  day number (1-based); auto-increments if None
        """
        s = self.state
        if trading_day_n is None:
            trading_day_n = s['current_day'] + 1
        s['current_day'] = trading_day_n

        # 1. Inject daily budget (does NOT add to carry-over cash)
        #    Unused daily budget VANISHES at end of day.
        #    Only closed-position proceeds (s['cash']) carry forward.
        daily_budget_today   = DAILY_BUDGET
        s['total_injected'] += DAILY_BUDGET

        # 2. Check exits — sell positions that are stale or score-dropped
        score_map = {
            st['symbol']: st.get('composite', st.get('final_score', 0))
            for st in council_stocks
        }
        sells = []
        keeps = []
        for pos in s['positions']:
            sym       = pos['symbol']
            held_days = trading_day_n - pos['bought_day']
            cur_score = score_map.get(sym, None)
            sell_price = prev_open_map.get(sym, pos['buy_price'])  # today's open

            should_sell = (
                held_days >= MAX_HOLD_DAYS
                or (cur_score is not None and cur_score < SELL_SCORE)
            )

            if should_sell:
                gain       = round((sell_price - pos['buy_price']) * pos['shares'], 2)
                gain_pct   = round((sell_price - pos['buy_price']) / pos['buy_price'] * 100, 2)
                closed_pos = {**pos, 'sell_price': sell_price, 'sell_date': date_str,
                              'gain': gain, 'gain_pct': gain_pct, 'held_days': held_days}
                sells.append(closed_pos)
                s['closed_positions'].append(closed_pos)
                s['cash']        += round(sell_price * pos['shares'], 2)
                s['total_gains'] += gain
            else:
                keeps.append(pos)

        s['positions'] = keeps

        # 3. Compute day confidence and budget to deploy
        #    Available = today's fresh budget + carry-over cash from closed positions
        day_conf         = compute_day_confidence(council_stocks)
        carry_cash       = s['cash']                          # only closed-position proceeds
        available_today  = daily_budget_today + carry_cash    # total spendable this session
        reserve          = carry_cash                         # reserve = closed proceeds only
        to_deploy        = confidence_to_budget(day_conf, reserve)
        to_deploy        = min(to_deploy, available_today)    # can't spend more than available

        # 4. Pick stocks to buy
        held_syms = {p['symbol'] for p in s['positions']}
        picks     = pick_stocks(council_stocks, to_deploy, held_syms)

        # 5. Execute buys at next day's open
        buys = []
        total_spent = 0.0
        for pick in picks:
            sym        = pick['symbol']
            buy_price  = next_open_map.get(sym)
            if buy_price is None or buy_price <= 0:
                continue
            shares     = int(pick['alloc'] // buy_price)
            if shares < 1:
                continue
            cost       = round(shares * buy_price, 2)
            pos = {
                'symbol':    sym,
                'buy_price': buy_price,
                'shares':    shares,
                'cost':      cost,
                'score':     pick['score'],
                'buy_date':  date_str,
                'bought_day': trading_day_n,
            }
            s['positions'].append(pos)
            buys.append(pos)
            total_spent     += cost
            s['total_deployed'] += cost

        # Spend from daily budget first, then from carry-over cash.
        # Unused portion of daily_budget_today VANISHES (not added to s['cash']).
        from_cash_used = max(0.0, total_spent - daily_budget_today)
        s['cash'] = round(carry_cash - from_cash_used, 2)

        # 6. Portfolio MTM value
        port_val = self.portfolio_value(price_map)
        s['portfolio_value'] = port_val
        # P&L = gain/loss on deployed capital (vanished budget is not 'lost money')
        total_pl     = round(port_val - s['total_deployed'], 2)
        deployed_pl  = round(s['total_gains'] + sum(
            (price_map.get(p['symbol'], p['buy_price']) - p['buy_price']) * p['shares']
            for p in s['positions']
        ), 2)

        # 7. Log the day
        day_log = {
            'date':          date_str,
            'day':           trading_day_n,
            'day_confidence': day_conf,
            'budget_deployed': round(total_spent, 2),
            'cash_reserve':  round(s['cash'], 2),
            'total_injected': s['total_injected'],
            'portfolio_value': port_val,
            'total_pl':      total_pl,
            'total_return_pct': round(total_pl / s['total_deployed'] * 100, 2) if s['total_deployed'] else 0,
            'deployed_pl':   deployed_pl,
            'buys':          [{'symbol': b['symbol'], 'shares': b['shares'],
                               'price': b['buy_price'], 'cost': b['cost']} for b in buys],
            'sells':         [{'symbol': sv['symbol'], 'gain': sv['gain'],
                               'gain_pct': sv['gain_pct']} for sv in sells],
            'open_positions': len(s['positions']),
        }
        s['daily_log'].append(day_log)

        # 8. Print summary
        if print_output:
            conf_pct = round(day_conf * 100, 1)
            print(f"\n{'─'*55}")
            print(f"  MODEL B  Day {trading_day_n} | {date_str}")
            print(f"  Day Confidence : {conf_pct}%")
            print(f"  Budget Deployed: ₹{total_spent:,.0f} of ₹{DAILY_BUDGET:,}")
            print(f"  Cash Reserve   : ₹{s['cash']:,.0f}")
            if buys:
                for b in buys:
                    print(f"    🟢 BUY  {b['symbol']:<12} {b['shares']} sh @ ₹{b['buy_price']:,.0f}  = ₹{b['cost']:,.0f}")
            else:
                print(f"    ⏸️  No buys today (conf={conf_pct}% or no qualifying stocks)")
            if sells:
                for sv in sells:
                    sign = '+' if sv['gain'] >= 0 else ''
                    print(f"    🔴 SELL {sv['symbol']:<12}  {sign}₹{sv['gain']:,.0f} ({sign}{sv['gain_pct']}%)")
            print(f"  Portfolio Value: ₹{port_val:,.0f}  |  P&L: ₹{total_pl:+,.0f}")
            print(f"{'─'*55}")

        return day_log

    def print_summary(self):
        """Print full backtest / run summary."""
        s = self.state
        log = s['daily_log']
        if not log:
            print("  No days run yet.")
            return

        total_injected  = s['total_injected']
        total_deployed  = s['total_deployed']
        portfolio_val   = s['portfolio_value']
        total_pl        = round(portfolio_val - total_deployed, 2)
        return_pct      = round(total_pl / total_deployed * 100, 2) if total_deployed else 0
        days_invested   = sum(1 for d in log if d['budget_deployed'] > 0)
        days_skipped    = sum(1 for d in log if d['budget_deployed'] == 0)
        avg_conf        = round(sum(d['day_confidence'] for d in log) / len(log) * 100, 1)

        closed = s['closed_positions']
        winners = [p for p in closed if p.get('gain', 0) > 0]
        losers  = [p for p in closed if p.get('gain', 0) <= 0]

        print(f"\n{'═'*55}")
        print(f"  MODEL B  SUMMARY  ({len(log)} trading days)")
        print(f"{'═'*55}")
        print(f"  Total Capital Injected : ₹{total_injected:>10,.0f}")
        print(f"  Total Capital Deployed : ₹{total_deployed:>10,.0f}")
        print(f"  Cash Reserve           : ₹{s['cash']:>10,.0f}")
        print(f"  Portfolio Value        : ₹{portfolio_val:>10,.0f}")
        print(f"  Total P&L              : \u20b9{total_pl:>+10,.0f}  ({return_pct:+.2f}%)")
        print(f"  Return on Deployed     : {round(total_pl/total_deployed*100,2) if total_deployed else 0:+.2f}%")
        print(f"")
        print(f"  Days Invested : {days_invested}  |  Days Skipped: {days_skipped}")
        print(f"  Avg Day Confidence: {avg_conf}%")
        print(f"  Closed Trades   : {len(closed)}  "
              f"(W:{len(winners)}  L:{len(losers)}  "
              f"WR:{round(len(winners)/len(closed)*100) if closed else 0}%)")
        if closed:
            avg_gain = round(sum(p.get('gain',0) for p in closed)/len(closed), 2)
            print(f"  Avg Trade Gain  : \u20b9{avg_gain:+,.2f}")
        print(f"{chr(9552)*55}")
