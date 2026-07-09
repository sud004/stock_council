# ============================================================
# bots/prediction_bot.py — Prediction & Self-Learning Bot
# ============================================================
#
# WHAT IT DOES:
#   1. DAILY PREDICTION: At market close, predicts price for:
#      - Tomorrow (1 day)
#      - 3 days from now
#      - 7 days from now
#      Uses: current TA indicators + fundamental score + sentiment
#
#   2. BACKTEST: Next morning compares prediction vs actual price
#      Calculates prediction accuracy per timeframe
#
#   3. WEIGHT OPTIMIZER: When a stock is DROPPED or a BUY call fails:
#      - Analyses WHICH scoring parameter was wrong
#      - Adjusts weights in config.py automatically
#      - Logs every weight change with reason
#
# PREDICTION FORMULA:
#   base_target = current_price × (1 + expected_return)
#
#   expected_return = (
#       tech_momentum   × W_tech   +   # RSI, MACD signal
#       fund_quality    × W_fund   +   # ROE, growth
#       news_sentiment  × W_news   +   # avg news sentiment
#       market_beta     × W_market +   # market direction × beta
#       mean_reversion  × W_mean       # distance from 50 DMA
#   )
#
# WEIGHT LEARNING:
#   After N predictions, compare accuracy per parameter:
#   If technical was consistently right → increase W_tech
#   If sentiment was consistently wrong → decrease W_sent
#   Weights saved to data/learned_weights.json
#   Applied next day automatically
#
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
from utils.llm import stream_chat, extract_score

IST = pytz.timezone('Asia/Kolkata')

# ── Storage files ─────────────────────────────────────────────
PREDICTIONS_FILE    = DATA_DIR / "predictions.json"
ACCURACY_FILE       = DATA_DIR / "prediction_accuracy.json"
LEARNED_WEIGHTS_FILE = DATA_DIR / "learned_weights.json"

# ── Default scoring weights ────────────────────────────────────
DEFAULT_WEIGHTS = {
    "fundamental": 0.30,
    "technical":   0.25,
    "news":        0.20,
    "sentiment":   0.15,
    "risk":        0.10,   # inverted
}

# ── Prediction model weights (for price movement) ─────────────
DEFAULT_PRED_WEIGHTS = {
    "tech_momentum":  0.35,
    "fund_quality":   0.20,
    "news_sentiment": 0.15,
    "market_beta":    0.20,
    "mean_reversion": 0.10,
}

PREDICTION_BOT_PROMPT = """You are PREDICTION BOT — a quantitative analyst who makes specific price predictions.

You receive technical indicators, fundamental scores and sentiment data for a stock.
Your job: make SPECIFIC price predictions for 1-day, 3-day, and 7-day timeframes.

Be specific about:
1. Direction: UP or DOWN
2. Expected % move
3. Predicted price
4. Confidence: HIGH / MEDIUM / LOW
5. Key driver of your prediction

Format your response EXACTLY like this:
1D PREDICTION: ₹[price] ([+/-X.X]%) | Confidence: [HIGH/MEDIUM/LOW] | Driver: [reason]
3D PREDICTION: ₹[price] ([+/-X.X]%) | Confidence: [HIGH/MEDIUM/LOW] | Driver: [reason]
7D PREDICTION: ₹[price] ([+/-X.X]%) | Confidence: [HIGH/MEDIUM/LOW] | Driver: [reason]
PREDICTION SCORE: X/10"""


# ══════════════════════════════════════════════════════════════
# PREDICTION ENGINE
# ══════════════════════════════════════════════════════════════

def compute_quant_prediction(symbol: str, current_price: float,
                              scores: dict, ta: dict = None,
                              nifty_1d_pct: float = None) -> dict:
    """
    Quantitative price prediction using weighted signals.

    Args:
        nifty_1d_pct: Live Nifty 1-day % change (from MarketSnapshot).
                      Pass this so the market_beta signal is real.
                      If None, fetched from cached prices as fallback.

    Returns predicted prices for 1d, 3d, 7d timeframes.
    """
    weights = load_learned_weights().get('pred_weights', DEFAULT_PRED_WEIGHTS)

    # ── Signal 1: Technical momentum ─────────────────────────
    rsi = (ta or {}).get('rsi', 50) or 50
    macd_bullish = (ta or {}).get('macd_bullish', None)
    above_50dma  = (ta or {}).get('above_sma50', None)

    tech_signal = 0
    if 50 < rsi < 65:    tech_signal += 0.015   # bullish zone
    elif rsi > 70:        tech_signal -= 0.005   # overbought
    elif rsi < 35:        tech_signal -= 0.010   # oversold
    if macd_bullish:      tech_signal += 0.010
    elif macd_bullish is False: tech_signal -= 0.010
    if above_50dma:       tech_signal += 0.005

    # ── Signal 2: Fundamental quality ─────────────────────────
    fund_score = scores.get('fundamental', 5.0)
    fund_signal = (fund_score - 5.0) * 0.003   # +/- 0.015 max

    # ── Signal 3: News sentiment ──────────────────────────────
    news_score = scores.get('news', 5.0)
    news_signal = (news_score - 5.0) * 0.002   # +/- 0.010 max

    # ── Signal 4: Market beta adjustment ─────────────────────
    beta = 1.0  # default
    fund = load_fundamentals_json(symbol)
    if fund:
        beta = fund.get('beta') or 1.0

    # Use live Nifty 1D % passed from orchestrator (MarketSnapshot).
    # Fallback: compute from cached Nifty price CSV if not provided.
    market_drift = 0.0
    if nifty_1d_pct is not None:
        market_drift = nifty_1d_pct / 100.0   # convert % → decimal
    else:
        # Fallback: read last 2 rows of cached Nifty prices
        try:
            nifty_df = load_prices_csv('^NSEI', days=5)
            if nifty_df is not None and len(nifty_df) >= 2:
                c = nifty_df['Close']
                market_drift = float((c.iloc[-1] - c.iloc[-2]) / c.iloc[-2])
        except Exception:
            market_drift = 0.0

    market_signal = beta * market_drift

    # ── Signal 5: Mean reversion ──────────────────────────────
    mean_rev_signal = 0
    df = load_prices_csv(symbol, days=60)
    if df is not None and not df.empty and len(df) >= 20:
        close = df['Close']
        sma20 = float(close.rolling(20).mean().iloc[-1])
        if sma20 > 0:
            deviation = (current_price - sma20) / sma20
            # If far above SMA → expect reversion down
            # If far below SMA → expect reversion up
            mean_rev_signal = -deviation * 0.3  # dampened

    # ── Combine signals ────────────────────────────────────────
    combined_1d = (
        tech_signal   * weights['tech_momentum'] +
        fund_signal   * weights['fund_quality'] +
        news_signal   * weights['news_sentiment'] +
        market_signal * weights['market_beta'] +
        mean_rev_signal * weights['mean_reversion']
    )

    # Scale for timeframes (not perfectly linear but approximate)
    combined_3d = combined_1d * 2.2
    combined_7d = combined_1d * 3.8

    return {
        'current_price': current_price,
        'pred_1d':   round(current_price * (1 + combined_1d), 2),
        'pred_3d':   round(current_price * (1 + combined_3d), 2),
        'pred_7d':   round(current_price * (1 + combined_7d), 2),
        'pct_1d':    round(combined_1d * 100, 2),
        'pct_3d':    round(combined_3d * 100, 2),
        'pct_7d':    round(combined_7d * 100, 2),
        'signals': {
            'tech':        round(tech_signal * 100, 3),
            'fundamental': round(fund_signal * 100, 3),
            'news':        round(news_signal * 100, 3),
            'market':      round(market_signal * 100, 3),
            'mean_rev':    round(mean_rev_signal * 100, 3),
        }
    }


def run(symbol: str, current_price: float, scores: dict,
        ta: dict = None, print_output: bool = True,
        nifty_1d_pct: float = None,
        stop_loss: float = None, target: float = None,
        verdict: str = None) -> dict:
    """
    Run prediction bot for one stock.
    Called at end of each council session.

    Args:
        nifty_1d_pct : Live Nifty 1-day % change from MarketSnapshot.
        stop_loss    : Stop-loss price from debate council verdict.
        target       : Target price from debate council verdict.
        verdict      : Council verdict label (BUY / SELL etc.) for tracking.
    """
    if print_output:
        print(f"\n{'='*60}")
        print(f"🔮  PREDICTION BOT — {symbol}")
        print('='*60)

    # Quantitative prediction
    quant_pred = compute_quant_prediction(symbol, current_price, scores, ta,
                                          nifty_1d_pct=nifty_1d_pct)

    # Build LLM prompt
    fund = load_fundamentals_json(symbol) or {}
    beta_used = fund.get('beta') or 1.0
    nifty_drift_pct = nifty_1d_pct or 0.0
    stop_target_section = ""
    if stop_loss or target:
        sl_str = f"₹{stop_loss:.2f}" if stop_loss else "N/A"
        tgt_str = f"₹{target:.2f}" if target else "N/A"
        stop_target_section = f"\nCOUNCIL LEVELS:\n  Stop Loss: {sl_str}  |  Target: {tgt_str}"

    prompt = f"""
Stock: {symbol}
Current Price: ₹{current_price:.2f}
Date: {today_str()}

COUNCIL SCORES:
  Fundamental: {scores.get('fundamental', 5):.1f}/10
  Technical:   {scores.get('technical', 5):.1f}/10
  News:        {scores.get('news', 5):.1f}/10
  Sentiment:   {scores.get('sentiment', 5):.1f}/10
  Risk Level:  {scores.get('risk', 5):.1f}/10
  Verdict:     {verdict or 'N/A'}
{stop_target_section}

QUANTITATIVE SIGNALS:
  Tech momentum:    {quant_pred['signals']['tech']:+.3f}%
  Fund quality:     {quant_pred['signals']['fundamental']:+.3f}%
  News sentiment:   {quant_pred['signals']['news']:+.3f}%
  Market (β={beta_used:.2f}×Nifty {nifty_drift_pct:+.2f}%): {quant_pred['signals']['market']:+.3f}%
  Mean reversion:   {quant_pred['signals']['mean_rev']:+.3f}%

QUANT PREDICTIONS:
  1D target: ₹{quant_pred['pred_1d']} ({quant_pred['pct_1d']:+.2f}%)
  3D target: ₹{quant_pred['pred_3d']} ({quant_pred['pct_3d']:+.2f}%)
  7D target: ₹{quant_pred['pred_7d']} ({quant_pred['pct_7d']:+.2f}%)

KEY FUNDAMENTALS:
  P/E: {fund.get('pe_ratio', 'N/A')}x | ROE: {fund.get('roe', 'N/A')}% | Beta: {fund.get('beta', 'N/A')}

Make your price predictions with reasoning.
"""

    def on_tok(t):
        if print_output:
            print(t, end='', flush=True)

    if print_output:
        print()

    llm_text = stream_chat(PREDICTION_BOT_PROMPT, prompt, on_token=on_tok)

    if print_output:
        print()

    # Parse LLM predictions
    llm_pred = _parse_llm_predictions(llm_text, current_price)

    # Blend: 60% quant + 40% LLM
    final = {
        'symbol':        symbol,
        'date':          today_str(),
        'current_price': current_price,
        'pred_1d':  round(0.6*quant_pred['pred_1d'] + 0.4*llm_pred.get('pred_1d', quant_pred['pred_1d']), 2),
        'pred_3d':  round(0.6*quant_pred['pred_3d'] + 0.4*llm_pred.get('pred_3d', quant_pred['pred_3d']), 2),
        'pred_7d':  round(0.6*quant_pred['pred_7d'] + 0.4*llm_pred.get('pred_7d', quant_pred['pred_7d']), 2),
        'pct_1d':   round((0.6*quant_pred['pred_1d'] + 0.4*llm_pred.get('pred_1d', quant_pred['pred_1d']))/current_price - 1, 4)*100,
        'pct_3d':   round((0.6*quant_pred['pred_3d'] + 0.4*llm_pred.get('pred_3d', quant_pred['pred_3d']))/current_price - 1, 4)*100,
        'pct_7d':   round((0.6*quant_pred['pred_7d'] + 0.4*llm_pred.get('pred_7d', quant_pred['pred_7d']))/current_price - 1, 4)*100,
        'signals':       quant_pred['signals'],
        'llm_text':      llm_text,
        # Council context for backtest stop/target tracking
        'verdict':       verdict or 'N/A',
        'stop_loss':     stop_loss,
        'target':        target,
        'nifty_1d_pct':  nifty_1d_pct,
        # Backtest flags
        'checked_1d':    False,
        'checked_3d':    False,
        'checked_7d':    False,
        'hit_target':    None,    # True/False once target/stop reached
        'hit_stop':      None,
        'target_hit_date': None,
        'stop_hit_date':   None,
    }

    # Save prediction
    _save_prediction(symbol, final)

    if print_output:
        direction_1d = "📈" if final['pct_1d'] > 0 else "📉"
        direction_3d = "📈" if final['pct_3d'] > 0 else "📉"
        direction_7d = "📈" if final['pct_7d'] > 0 else "📉"
        print(f"\n  {direction_1d} 1D: ₹{final['pred_1d']} ({final['pct_1d']:+.2f}%)")
        print(f"  {direction_3d} 3D: ₹{final['pred_3d']} ({final['pct_3d']:+.2f}%)")
        print(f"  {direction_7d} 7D: ₹{final['pred_7d']} ({final['pct_7d']:+.2f}%)")

    return final


# ══════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════

def run_daily_backtest(print_output: bool = True) -> dict:
    """
    Run every morning — compare yesterday's predictions vs actual prices.
    Updates accuracy scores and triggers weight optimization if needed.
    """
    predictions = _load_all_predictions()
    today = datetime.now(IST)
    results = {'checked': 0, 'correct': 0, 'wrong': 0, 'accuracy': {}}

    if print_output:
        print(f"\n{'='*60}")
        print(f"📊  PREDICTION BACKTEST — {today_str()}")
        print('='*60)

    accuracy_by_param = {k: [] for k in DEFAULT_PRED_WEIGHTS.keys()}

    for symbol, pred_list in predictions.items():
        df = load_prices_csv(symbol, days=15)
        if df is None or df.empty:
            continue

        close = df['Close']

        for pred in pred_list:
            pred_date = datetime.strptime(pred['date'], '%Y-%m-%d')
            pred_date = IST.localize(pred_date)
            current_p = pred['current_price']

            # Check 1D prediction
            if not pred.get('checked_1d'):
                target_date = pred_date + timedelta(days=1)
                actual = _get_price_on_date(close, target_date)
                if actual:
                    error_pct = (actual - pred['pred_1d']) / pred['pred_1d'] * 100
                    direction_correct = (
                        (pred['pred_1d'] > current_p and actual > current_p) or
                        (pred['pred_1d'] < current_p and actual < current_p)
                    )
                    pred['actual_1d']     = actual
                    pred['error_1d_pct']  = round(error_pct, 2)
                    pred['direction_1d']  = direction_correct
                    pred['checked_1d']    = True
                    results['checked'] += 1
                    if direction_correct:
                        results['correct'] += 1
                    else:
                        results['wrong'] += 1

                    # Track which signals were right
                    signals = pred.get('signals', {})
                    for param, signal_val in signals.items():
                        actual_move = (actual - current_p) / current_p * 100
                        signal_correct = (signal_val > 0 and actual_move > 0) or \
                                        (signal_val < 0 and actual_move < 0)
                        accuracy_by_param.get(param, []).append(1 if signal_correct else 0)

            # Check 3D prediction
            if not pred.get('checked_3d'):
                target_date = pred_date + timedelta(days=3)
                actual = _get_price_on_date(close, target_date)
                if actual:
                    pred['actual_3d']    = actual
                    pred['error_3d_pct'] = round((actual - pred['pred_3d']) / pred['pred_3d'] * 100, 2)
                    pred['checked_3d']   = True

            # Check 7D prediction
            if not pred.get('checked_7d'):
                target_date = pred_date + timedelta(days=7)
                actual = _get_price_on_date(close, target_date)
                if actual:
                    pred['actual_7d']    = actual
                    pred['error_7d_pct'] = round((actual - pred['pred_7d']) / pred['pred_7d'] * 100, 2)
                    pred['checked_7d']   = True

            # ── Stop-loss / Target tracking (new) ─────────────
            # Scan up to 10 trading days from prediction date for a hit
            stop_price  = pred.get('stop_loss')
            target_price = pred.get('target')
            if (stop_price or target_price) and pred.get('hit_target') is None and pred.get('hit_stop') is None:
                for day_offset in range(1, 11):
                    scan_date = pred_date + timedelta(days=day_offset)
                    # Use intraday proxy: check close only (best we can do with daily data)
                    day_close = _get_price_on_date(close, scan_date)
                    if day_close is None:
                        continue
                    date_str = scan_date.strftime('%Y-%m-%d')
                    if target_price and day_close >= target_price:
                        pred['hit_target']      = True
                        pred['hit_stop']        = False
                        pred['target_hit_date'] = date_str
                        break
                    if stop_price and day_close <= stop_price:
                        pred['hit_stop']        = True
                        pred['hit_target']      = False
                        pred['stop_hit_date']   = date_str
                        break

    # Save updated predictions
    _save_all_predictions(predictions)

    # Compute overall accuracy
    if results['checked'] > 0:
        acc = results['correct'] / results['checked'] * 100
        results['accuracy']['overall_1d'] = round(acc, 1)

        if print_output:
            print(f"\n  Predictions checked: {results['checked']}")
            print(f"  Direction correct:   {results['correct']} ({acc:.1f}%)")
            print(f"  Direction wrong:     {results['wrong']}")

        # Trigger weight optimization if accuracy < 55%
        if acc < 55 and results['checked'] >= 10:
            if print_output:
                print(f"\n  ⚠️  Accuracy {acc:.1f}% below threshold — optimizing weights...")
            optimize_weights(accuracy_by_param, print_output=print_output)

    return results


# ══════════════════════════════════════════════════════════════
# WEIGHT OPTIMIZER
# ══════════════════════════════════════════════════════════════

def optimize_weights(accuracy_by_param: dict, print_output: bool = True):
    """
    Automatically adjust scoring weights based on prediction accuracy.

    ALGORITHM:
      1. Compute accuracy per signal parameter (tech, fund, news etc.)
      2. Parameters with high accuracy → increase weight
      3. Parameters with low accuracy → decrease weight
      4. Normalize so all weights sum to 1.0
      5. Save to learned_weights.json

    CONSTRAINTS:
      No weight below 0.05 (always some influence)
      No weight above 0.50 (never dominates completely)
      Changes are gradual: max 20% change per optimization
    """
    current = load_learned_weights()
    current_pred = current.get('pred_weights', dict(DEFAULT_PRED_WEIGHTS))
    current_score = current.get('score_weights', dict(DEFAULT_WEIGHTS))

    changes = []
    new_pred_weights = {}

    for param, accuracy_list in accuracy_by_param.items():
        if len(accuracy_list) < 5:
            new_pred_weights[param] = current_pred.get(param, 0.2)
            continue

        param_accuracy = sum(accuracy_list) / len(accuracy_list)
        current_w = current_pred.get(param, 0.2)

        # Adjust: good accuracy → higher weight, bad → lower
        if param_accuracy > 0.65:
            new_w = min(0.50, current_w * 1.15)   # increase 15%
            direction = "↑"
        elif param_accuracy > 0.55:
            new_w = min(0.50, current_w * 1.05)   # increase 5%
            direction = "↑"
        elif param_accuracy < 0.45:
            new_w = max(0.05, current_w * 0.85)   # decrease 15%
            direction = "↓"
        elif param_accuracy < 0.35:
            new_w = max(0.05, current_w * 0.75)   # decrease 25%
            direction = "↓↓"
        else:
            new_w = current_w   # unchanged
            direction = "→"

        new_pred_weights[param] = round(new_w, 4)
        if abs(new_w - current_w) > 0.001:
            changes.append(
                f"  {param:20} {current_w:.3f} → {new_w:.3f} {direction} "
                f"(accuracy: {param_accuracy*100:.0f}%)"
            )

    # Normalize to sum = 1.0
    total = sum(new_pred_weights.values())
    if total > 0:
        new_pred_weights = {k: round(v/total, 4) for k, v in new_pred_weights.items()}

    # Re-clamp AFTER normalization — normalization can push values back outside bounds
    new_pred_weights = {
        k: round(min(0.50, max(0.05, v)), 4)
        for k, v in new_pred_weights.items()
    }
    # Re-normalize after second clamp to guarantee sum == 1.0
    total2 = sum(new_pred_weights.values())
    if total2 > 0:
        new_pred_weights = {k: round(v/total2, 4) for k, v in new_pred_weights.items()}

    # Save
    learned = {
        'pred_weights':  new_pred_weights,
        'score_weights': current_score,
        'last_optimized': today_str(),
        'optimization_count': current.get('optimization_count', 0) + 1,
        'change_log': current.get('change_log', []) + [{
            'date': today_str(),
            'changes': changes,
        }]
    }
    with open(LEARNED_WEIGHTS_FILE, 'w') as f:
        json.dump(learned, f, indent=2)

    if print_output and changes:
        print(f"\n  WEIGHT OPTIMIZATION:")
        for c in changes:
            print(c)
        print(f"\n  New weights: {new_pred_weights}")
        print(f"  Saved to: {LEARNED_WEIGHTS_FILE}")


def analyze_failed_call(symbol: str, verdict: str, entry_price: float,
                         exit_price: float, scores: dict):
    """
    When a stock is DROPPED from council (consistently bad calls),
    analyze WHY the call was wrong and adjust weights.

    Called from opportunity_scanner when drop_rule triggers.
    """
    pct_change = (exit_price - entry_price) / entry_price * 100
    direction = "UP" if pct_change > 0 else "DOWN"
    was_correct = (verdict in ('BUY', 'STRONG BUY', 'ACCUMULATE') and pct_change > 0) or \
                  (verdict in ('SELL', 'STRONG SELL', 'REDUCE') and pct_change < 0)

    analysis = {
        'symbol':       symbol,
        'verdict':      verdict,
        'entry_price':  entry_price,
        'exit_price':   exit_price,
        'pct_change':   round(pct_change, 2),
        'was_correct':  was_correct,
        'date':         today_str(),
    }

    if not was_correct:
        print(f"\n[PREDICTION BOT] 🔍 Analyzing failed call: {symbol}")
        print(f"  Verdict was: {verdict} at ₹{entry_price:.0f}")
        print(f"  Stock went:  {direction} {abs(pct_change):.1f}% to ₹{exit_price:.0f}")

        # Find which parameter was most wrong
        high_score_params = {k: v for k, v in scores.items() if v > 6.5}
        low_actual_params = []

        # If fundamental was high but stock fell — overweighted fundamentals
        if 'fundamental' in high_score_params and pct_change < -5:
            low_actual_params.append('fundamental')
            print(f"  → Fundamental score {scores['fundamental']:.1f} was too optimistic")

        # If technical was high but stock fell — overweighted technicals
        if 'technical' in high_score_params and pct_change < -5:
            low_actual_params.append('technical')
            print(f"  → Technical score {scores['technical']:.1f} gave false signal")

        # If sentiment was high but stock fell — overweighted sentiment
        if 'sentiment' in high_score_params and pct_change < -5:
            low_actual_params.append('sentiment')
            print(f"  → Sentiment score {scores['sentiment']:.1f} was misleading")

        # If risk was LOW but stock fell a lot — underweighted risk
        risk = scores.get('risk', 5)
        if risk < 5 and pct_change < -10:
            print(f"  → Risk score {risk:.1f} underestimated actual risk")

        # Adjust weights: penalize wrong parameters
        if low_actual_params:
            current = load_learned_weights()
            sw = current.get('score_weights', dict(DEFAULT_WEIGHTS))

            for param in low_actual_params:
                if param in sw:
                    sw[param] = max(0.05, sw[param] * 0.90)  # reduce by 10%
                    print(f"  Reducing {param} weight: → {sw[param]:.3f}")

            # Compensate: increase risk weight
            sw['risk'] = min(0.25, sw.get('risk', 0.10) * 1.15)
            print(f"  Increasing risk weight: → {sw['risk']:.3f}")

            # Normalize
            total = sum(sw.values())
            sw = {k: round(v/total, 4) for k, v in sw.items()}

            current['score_weights'] = sw
            current['change_log'] = current.get('change_log', []) + [{
                'date': today_str(),
                'event': f'Failed call: {symbol} {verdict}',
                'adjustment': f'Reduced: {low_actual_params}',
            }]

            with open(LEARNED_WEIGHTS_FILE, 'w') as f:
                json.dump(current, f, indent=2)

            print(f"  Updated score weights: {sw}")

    return analysis


def load_learned_weights() -> dict:
    """Load learned weights from disk. Returns defaults if not found."""
    if LEARNED_WEIGHTS_FILE.exists():
        with open(LEARNED_WEIGHTS_FILE) as f:
            return json.load(f)
    return {
        'pred_weights':  dict(DEFAULT_PRED_WEIGHTS),
        'score_weights': dict(DEFAULT_WEIGHTS),
        'optimization_count': 0,
        'change_log': [],
    }


def get_current_score_weights() -> dict:
    """Get current scoring weights (may be different from defaults after learning)."""
    return load_learned_weights().get('score_weights', dict(DEFAULT_WEIGHTS))


def print_weight_history():
    """Print history of all weight changes."""
    data = load_learned_weights()
    log = data.get('change_log', [])

    print(f"\n{'='*60}")
    print(f"📊  WEIGHT LEARNING HISTORY")
    print(f"    Optimizations: {data.get('optimization_count', 0)}")
    print('='*60)

    print(f"\nCurrent Score Weights:")
    for k, v in data.get('score_weights', DEFAULT_WEIGHTS).items():
        default = DEFAULT_WEIGHTS.get(k, 0)
        diff = v - default
        arrow = f" ({diff:+.3f} vs default)" if abs(diff) > 0.001 else " (unchanged)"
        print(f"  {k:15} {v:.3f}{arrow}")

    print(f"\nCurrent Prediction Weights:")
    for k, v in data.get('pred_weights', DEFAULT_PRED_WEIGHTS).items():
        print(f"  {k:20} {v:.3f}")

    if log:
        print(f"\nChange Log (last 5):")
        for entry in log[-5:]:
            print(f"  [{entry.get('date')}] {entry.get('event', 'optimization')}")


# ── Storage helpers ───────────────────────────────────────────

def _save_prediction(symbol: str, pred: dict):
    all_preds = _load_all_predictions()
    if symbol not in all_preds:
        all_preds[symbol] = []
    # Only keep last 30 predictions per stock
    all_preds[symbol] = all_preds[symbol][-29:] + [pred]
    _save_all_predictions(all_preds)


def _load_all_predictions() -> dict:
    if not PREDICTIONS_FILE.exists():
        return {}
    try:
        with open(PREDICTIONS_FILE) as f:
            return json.load(f)
    except json.JSONDecodeError:
        # File truncated mid-write (pipeline killed during save).
        # Walk backwards to find last valid JSON boundary and recover.
        with open(PREDICTIONS_FILE) as f:
            content = f.read()
        for pos in range(len(content) - 1, 0, -1):
            if content[pos] in ('}', ']'):
                for suffix in ('', '}', ']}', ']\n}'):
                    try:
                        data = json.loads(content[:pos + 1] + suffix)
                        if isinstance(data, dict):
                            print(f"  [PREDICTIONS] Truncated file auto-recovered "
                                  f"({len(data)} stocks). Re-saving clean copy.")
                            _save_all_predictions(data)
                            return data
                    except Exception:
                        pass
        print("  [PREDICTIONS] Could not recover predictions.json — starting fresh.")
        return {}


def _save_all_predictions(data: dict):
    with open(PREDICTIONS_FILE, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def _get_price_on_date(close: pd.Series, target_date: datetime) -> float | None:
    """Get closing price on or near a target date."""
    try:
        close_tz_naive = close.copy()
        if close_tz_naive.index.tz is not None:
            close_tz_naive.index = close_tz_naive.index.tz_localize(None)

        target_naive = target_date.replace(tzinfo=None)

        # Find closest date
        diffs = abs(close_tz_naive.index - target_naive)
        if diffs.min().days <= 3:   # within 3 calendar days
            idx = diffs.argmin()
            return round(float(close_tz_naive.iloc[idx]), 2)
    except Exception:
        pass
    return None


def _parse_llm_predictions(text: str, current_price: float) -> dict:
    """Parse 1D/3D/7D predictions from LLM text."""
    import re
    result = {}

    patterns = {
        'pred_1d': r'1D PREDICTION[:\s]+₹?([\d,]+(?:\.\d+)?)',
        'pred_3d': r'3D PREDICTION[:\s]+₹?([\d,]+(?:\.\d+)?)',
        'pred_7d': r'7D PREDICTION[:\s]+₹?([\d,]+(?:\.\d+)?)',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                price = float(match.group(1).replace(',', ''))
                # Sanity check: within 20% of current price
                if 0.8 * current_price <= price <= 1.2 * current_price:
                    result[key] = price
            except Exception:
                pass

    return result
