# ============================================================
# bots/technical_bot.py — Technical Analysis Bot
# ============================================================
#
# ALL FORMULAS IMPLEMENTED:
#
# TREND INDICATORS:
#   SMA(n)       = Sum(Close[i], i=0..n-1) / n
#   EMA(n)       = Close × k + EMA_prev × (1-k),  k = 2/(n+1)
#   VWAP         = Sum(Volume × Typical Price) / Sum(Volume)
#   Typical Price = (High + Low + Close) / 3
#
# MOMENTUM:
#   RSI(n)       = 100 - 100/(1 + RS)
#                  RS = Avg Gain(n) / Avg Loss(n)
#   MACD         = EMA(12) - EMA(26)
#   Signal Line  = EMA(9) of MACD
#   MACD Histogram = MACD - Signal
#   Stochastic %K = (Close - Lowest Low(n)) / (Highest High(n) - Lowest Low(n)) × 100
#   Stochastic %D = SMA(3) of %K
#   Williams %R  = (Highest High(n) - Close) / (Highest High(n) - Lowest Low(n)) × -100
#   ROC(n)       = (Close - Close[n]) / Close[n] × 100   (Rate of Change)
#   CCI(n)       = (Typical Price - SMA(n)) / (0.015 × Mean Deviation)
#
# VOLATILITY:
#   ATR(n)       = RMA of TR
#   TR           = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
#   RMA(n)       = (TR + (n-1) × RMA_prev) / n   (Wilder's smoothing)
#   BB Upper     = SMA(20) + 2 × σ(Close, 20)
#   BB Lower     = SMA(20) - 2 × σ(Close, 20)
#   BB Width     = (BB_Upper - BB_Lower) / BB_Middle × 100
#   %B           = (Close - BB_Lower) / (BB_Upper - BB_Lower)
#   Keltner Upper= EMA(20) + 1.5 × ATR(10)
#   Keltner Lower= EMA(20) - 1.5 × ATR(10)
#
# TREND STRENGTH:
#   ADX(n)       = SMA(n) of |+DI - -DI| / (+DI + -DI) × 100
#   +DM          = High - Prev High (if positive and > Prev Low - Low, else 0)
#   -DM          = Prev Low - Low  (if positive and > High - Prev High, else 0)
#   +DI(n)       = RMA(+DM, n) / ATR(n) × 100
#   -DI(n)       = RMA(-DM, n) / ATR(n) × 100
#
# VOLUME:
#   OBV          = Cumulative(volume × sign(Close - PrevClose))
#   VWAP         = Cumulative(Close × Volume) / Cumulative(Volume)
#   Volume Ratio = Volume / Volume_SMA(20)
#   CMF(n)       = Sum(Money Flow Volume, n) / Sum(Volume, n)
#   Money Flow Volume = ((Close-Low)-(High-Close))/(High-Low) × Volume
#
# SUPPORT / RESISTANCE:
#   Pivot Point (Classic) = (High + Low + Close) / 3
#   R1 = 2×PP - Low    S1 = 2×PP - High
#   R2 = PP + (High-Low)  S2 = PP - (High-Low)
#   R3 = High + 2×(PP-Low)  S3 = Low - 2×(High-PP)
#
# FIBONACCI RETRACEMENT LEVELS:
#   Swing High = max(Close, lookback period)
#   Swing Low  = min(Close, lookback period)
#   Fib 23.6% = High - 0.236 × (High - Low)
#   Fib 38.2% = High - 0.382 × (High - Low)
#   Fib 50.0% = High - 0.500 × (High - Low)
#   Fib 61.8% = High - 0.618 × (High - Low)
#   Fib 78.6% = High - 0.786 × (High - Low)
#
# ICHIMOKU CLOUD:
#   Tenkan-sen (9)  = (High_9 + Low_9) / 2
#   Kijun-sen (26)  = (High_26 + Low_26) / 2
#   Senkou A        = (Tenkan + Kijun) / 2  [shifted 26 forward]
#   Senkou B        = (High_52 + Low_52) / 2  [shifted 26 forward]
#   Chikou Span     = Close  [shifted 26 back]
#
# COMPOSITE SCORE:
#   Trend score   (SMA cross, EMA, ADX)         weight 0.25
#   Momentum score (RSI, MACD, Stoch)            weight 0.30
#   Volatility score (ATR, BB, squeeze)          weight 0.20
#   Volume score  (OBV, CMF, volume ratio)       weight 0.15
#   S/R score     (price vs pivot/fib levels)    weight 0.10
# ============================================================

import sys
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TA_PARAMS, VERBOSE_DEBUG
from utils.market_data import fetch_price_history, fetch_nifty_data, resolve_symbol
from utils.llm import stream_chat, extract_score, fmt
from utils.database import save_analysis


SYSTEM_PROMPT = """You are TECHNICAL BOT — an expert Indian stock market technical analyst (CMT certified).
You specialize in BSE/NSE listed equities using price action, indicators, and chart patterns.

Your analysis must cover:
1. Trend direction (above/below key MAs, EMA alignment)
2. Momentum signals (RSI, MACD status, Stochastic)
3. Key support and resistance levels (with actual price levels)
4. Volume analysis (OBV trend, volume confirmation)
5. Chart patterns if identifiable (head & shoulders, double top/bottom, triangle, flag)
6. Fibonacci and pivot levels

Your response MUST:
- Be 150-200 words
- Cite actual price levels (support/resistance) from the data
- End with exactly: "TECHNICAL SCORE: X/10" where X is your assessment

Scoring guide:
  9-10: Strong uptrend, all indicators bullish, strong volume support
  7-8:  Bullish setup with minor concerns
  5-6:  Sideways/consolidation, mixed signals
  3-4:  Bearish trend, indicators negative
  1-2:  Strong downtrend, sell signals across all indicators
"""


# ── Pure numpy/pandas TA implementations ─────────────────────

def sma(series: pd.Series, n: int) -> pd.Series:
    """Simple Moving Average"""
    return series.rolling(window=n, min_periods=1).mean()


def ema(series: pd.Series, n: int) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=n, adjust=False, min_periods=1).mean()


def rma(series: pd.Series, n: int) -> pd.Series:
    """Wilder's RMA (used in RSI, ATR, ADX)"""
    return series.ewm(alpha=1/n, adjust=False, min_periods=n).mean()


def calculate_rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """RSI = 100 - 100 / (1 + RS)"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = rma(gain, n)
    avg_loss = rma(loss, n)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(close: pd.Series, fast=12, slow=26, signal=9) -> tuple:
    """MACD Line, Signal Line, Histogram"""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(close: pd.Series, n=20, std_mult=2.0) -> tuple:
    """Upper, Middle (SMA), Lower bands + Width + %B"""
    middle = sma(close, n)
    std = close.rolling(window=n).std()
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    bandwidth = (upper - lower) / middle * 100
    pct_b = (close - lower) / (upper - lower)
    return upper, middle, lower, bandwidth, pct_b


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, n=14) -> pd.Series:
    """Average True Range using Wilder's smoothing"""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return rma(tr, n)


def calculate_stochastic(high, low, close, k_period=14, d_period=3, smooth_k=3) -> tuple:
    """%K and %D"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    pct_k_raw = (close - lowest_low) / (highest_high - lowest_low) * 100
    pct_k = sma(pct_k_raw, smooth_k)
    pct_d = sma(pct_k, d_period)
    return pct_k, pct_d


def calculate_williams_r(high, low, close, period=14) -> pd.Series:
    """Williams %R = (Highest High - Close) / (Highest High - Lowest Low) × -100"""
    hh = high.rolling(window=period).max()
    ll = low.rolling(window=period).min()
    return (hh - close) / (hh - ll) * -100


def calculate_adx(high, low, close, n=14) -> tuple:
    """ADX, +DI, -DI"""
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    
    plus_dm = high - prev_high
    minus_dm = prev_low - low
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    atr = calculate_atr(high, low, close, n)
    
    plus_di = rma(plus_dm, n) / atr * 100
    minus_di = rma(minus_dm, n) / atr * 100
    
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    adx = rma(dx, n)
    
    return adx, plus_di, minus_di


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume"""
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    obv = (volume * direction).cumsum()
    return obv


def calculate_cmf(high, low, close, volume, n=20) -> pd.Series:
    """Chaikin Money Flow"""
    mf_multiplier = ((close - low) - (high - close)) / (high - low)
    mf_volume = mf_multiplier * volume
    cmf = mf_volume.rolling(window=n).sum() / volume.rolling(window=n).sum()
    return cmf


def calculate_roc(close: pd.Series, n=10) -> pd.Series:
    """Rate of Change"""
    return (close - close.shift(n)) / close.shift(n) * 100


def calculate_cci(high, low, close, n=20) -> pd.Series:
    """Commodity Channel Index"""
    tp = (high + low + close) / 3
    tp_sma = sma(tp, n)
    mean_dev = tp.rolling(window=n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - tp_sma) / (0.015 * mean_dev)


def calculate_vwap(high, low, close, volume) -> pd.Series:
    """Intraday VWAP (cumulative within available data)"""
    tp = (high + low + close) / 3
    return (tp * volume).cumsum() / volume.cumsum()


def calculate_fibonacci_levels(close: pd.Series, lookback: int = 120) -> dict:
    """Fibonacci retracement from swing high to swing low"""
    recent = close.tail(lookback)
    swing_high = recent.max()
    swing_low = recent.min()
    diff = swing_high - swing_low
    return {
        'swing_high': round(swing_high, 2),
        'swing_low': round(swing_low, 2),
        'fib_786': round(swing_high - 0.786 * diff, 2),
        'fib_618': round(swing_high - 0.618 * diff, 2),
        'fib_500': round(swing_high - 0.500 * diff, 2),
        'fib_382': round(swing_high - 0.382 * diff, 2),
        'fib_236': round(swing_high - 0.236 * diff, 2),
    }


def calculate_pivot_points(high, low, close) -> dict:
    """Classic Pivot Points from last completed candle"""
    h = float(high.iloc[-2]) if len(high) >= 2 else float(high.iloc[-1])
    l = float(low.iloc[-2]) if len(low) >= 2 else float(low.iloc[-1])
    c = float(close.iloc[-2]) if len(close) >= 2 else float(close.iloc[-1])

    pp = (h + l + c) / 3
    return {
        'PP':  round(pp, 2),
        'R1':  round(2 * pp - l, 2),
        'R2':  round(pp + (h - l), 2),
        'R3':  round(h + 2 * (pp - l), 2),
        'S1':  round(2 * pp - h, 2),
        'S2':  round(pp - (h - l), 2),
        'S3':  round(l - 2 * (h - pp), 2),
    }


def calculate_ichimoku(high, low, close) -> dict:
    """Ichimoku Cloud components"""
    def midpoint(h, l, n):
        return (h.rolling(n).max() + l.rolling(n).min()) / 2

    tenkan = midpoint(high, low, 9)
    kijun = midpoint(high, low, 26)
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = midpoint(high, low, 52).shift(26)
    chikou = close.shift(-26)

    last = lambda s: float(s.dropna().iloc[-1]) if not s.dropna().empty else None

    return {
        'tenkan': round(last(tenkan), 2) if last(tenkan) else None,
        'kijun': round(last(kijun), 2) if last(kijun) else None,
        'senkou_a': round(last(senkou_a), 2) if last(senkou_a) else None,
        'senkou_b': round(last(senkou_b), 2) if last(senkou_b) else None,
        'price_above_cloud': last(close) > max(last(senkou_a) or 0, last(senkou_b) or 0)
            if last(close) else None,
    }


def find_support_resistance(close: pd.Series, sensitivity=0.02, lookback=60) -> dict:
    """
    Find key S/R zones by clustering price points where reversals occurred.
    Returns top 3 support and resistance levels.
    """
    recent = close.tail(lookback)
    price = float(close.iloc[-1])
    
    # Find local minima (support) and maxima (resistance)
    levels = []
    for i in range(2, len(recent) - 2):
        val = float(recent.iloc[i])
        # Local max
        if val > float(recent.iloc[i-1]) and val > float(recent.iloc[i+1]):
            levels.append(('R', val))
        # Local min
        if val < float(recent.iloc[i-1]) and val < float(recent.iloc[i+1]):
            levels.append(('S', val))
    
    supports = sorted([l for t, l in levels if t == 'S' and l < price], reverse=True)[:3]
    resistances = sorted([l for t, l in levels if t == 'R' and l > price])[:3]
    
    return {
        'support_levels': [round(s, 2) for s in supports],
        'resistance_levels': [round(r, 2) for r in resistances]
    }


def compute_all_indicators(df: pd.DataFrame) -> dict:
    """Run all TA formulas on OHLCV dataframe. Returns dict of latest values."""
    p = TA_PARAMS
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    # Trend
    sma20 = sma(close, p['SMA_SHORT'])
    sma50 = sma(close, p['SMA_MID'])
    sma200 = sma(close, p['SMA_LONG'])
    ema9 = ema(close, p['EMA_SHORT'])
    ema21 = ema(close, p['EMA_MID'])
    ema55 = ema(close, p['EMA_LONG'])

    # Momentum
    rsi14 = calculate_rsi(close, p['RSI_PERIOD'])
    macd_line, macd_sig, macd_hist = calculate_macd(close, p['MACD_FAST'], p['MACD_SLOW'], p['MACD_SIGNAL'])
    stoch_k, stoch_d = calculate_stochastic(high, low, close, p['STOCH_K'], p['STOCH_D'], p['STOCH_SMOOTH'])
    williams = calculate_williams_r(high, low, close, p['WILLIAMS_PERIOD'])
    roc10 = calculate_roc(close, 10)
    cci20 = calculate_cci(high, low, close, 20)

    # Volatility
    atr14 = calculate_atr(high, low, close, p['ATR_PERIOD'])
    bb_upper, bb_mid, bb_lower, bb_width, bb_pct = calculate_bollinger_bands(close, p['BB_PERIOD'], p['BB_STD'])

    # Trend strength
    adx14, plus_di, minus_di = calculate_adx(high, low, close, p['ADX_PERIOD'])

    # Volume
    obv_series = calculate_obv(close, volume)
    cmf20 = calculate_cmf(high, low, close, volume, 20)
    vol_sma20 = sma(volume, p['VOL_SMA'])
    vwap = calculate_vwap(high, low, close, volume)

    # S/R
    sr = find_support_resistance(close, p['SR_SENSITIVITY'], p['SR_LOOKBACK'])
    fib = calculate_fibonacci_levels(close, p['FIB_LOOKBACK'])
    pivots = calculate_pivot_points(high, low, close)
    ichimoku = calculate_ichimoku(high, low, close)

    def last(s):
        """Get last non-NaN value from series."""
        v = s.dropna()
        return round(float(v.iloc[-1]), 4) if not v.empty else None

    current_price = last(close)
    current_vol = last(volume)
    avg_vol = last(vol_sma20)

    return {
        # Price context
        'current_price': current_price,
        'price_change_1d': round(float(close.pct_change().iloc[-1] * 100), 2) if len(close) > 1 else None,
        'price_change_1w': round(float((close.iloc[-1]/close.iloc[-5]-1)*100), 2) if len(close) >= 5 else None,
        'price_change_1m': round(float((close.iloc[-1]/close.iloc[-22]-1)*100), 2) if len(close) >= 22 else None,

        # Moving Averages
        'sma20': last(sma20),
        'sma50': last(sma50),
        'sma200': last(sma200),
        'ema9': last(ema9),
        'ema21': last(ema21),
        'ema55': last(ema55),
        'above_sma20': current_price > last(sma20) if current_price and last(sma20) else None,
        'above_sma50': current_price > last(sma50) if current_price and last(sma50) else None,
        'above_sma200': current_price > last(sma200) if current_price and last(sma200) else None,
        'golden_cross': last(sma50) > last(sma200) if last(sma50) and last(sma200) else None,

        # RSI
        'rsi': last(rsi14),
        'rsi_oversold': last(rsi14) < 30 if last(rsi14) else None,
        'rsi_overbought': last(rsi14) > 70 if last(rsi14) else None,

        # MACD
        'macd': last(macd_line),
        'macd_signal': last(macd_sig),
        'macd_histogram': last(macd_hist),
        'macd_bullish': last(macd_line) > last(macd_sig) if last(macd_line) and last(macd_sig) else None,

        # Stochastic
        'stoch_k': last(stoch_k),
        'stoch_d': last(stoch_d),
        'stoch_oversold': last(stoch_k) < 20 if last(stoch_k) else None,
        'stoch_overbought': last(stoch_k) > 80 if last(stoch_k) else None,

        # Williams %R
        'williams_r': last(williams),

        # ROC, CCI
        'roc_10': last(roc10),
        'cci_20': last(cci20),

        # ATR
        'atr': last(atr14),
        'atr_pct': round(last(atr14) / current_price * 100, 2) if last(atr14) and current_price else None,

        # Bollinger Bands
        'bb_upper': last(bb_upper),
        'bb_middle': last(bb_mid),
        'bb_lower': last(bb_lower),
        'bb_width': last(bb_width),
        'bb_pct_b': last(bb_pct),
        'bb_squeeze': last(bb_width) < 10 if last(bb_width) else None,

        # ADX
        'adx': last(adx14),
        'plus_di': last(plus_di),
        'minus_di': last(minus_di),
        'strong_trend': last(adx14) > 25 if last(adx14) else None,
        'bullish_di': last(plus_di) > last(minus_di) if last(plus_di) and last(minus_di) else None,

        # Volume
        'volume': current_vol,
        'volume_avg': avg_vol,
        'volume_ratio': round(current_vol / avg_vol, 2) if current_vol and avg_vol else None,
        'obv_trend': 'rising' if len(obv_series.dropna()) > 5 and
                     obv_series.dropna().iloc[-1] > obv_series.dropna().iloc[-5] else 'falling',
        'cmf': last(cmf20),
        'vwap': last(vwap),

        # Ichimoku
        'ichimoku': ichimoku,

        # S/R
        'support_levels': sr['support_levels'],
        'resistance_levels': sr['resistance_levels'],

        # Fibonacci
        'fibonacci': fib,

        # Pivot Points
        'pivot_points': pivots,
    }


def score_trend(ta: dict) -> tuple[float, list[str]]:
    """Score trend direction and strength 0-10."""
    score = 5.0
    notes = []

    if ta.get('above_sma200'):
        score += 1.5
        notes.append(f"Price above 200 DMA ({fmt(ta.get('sma200'))}) — long-term uptrend intact")
    else:
        score -= 1.5
        notes.append(f"Price BELOW 200 DMA ({fmt(ta.get('sma200'))}) — long-term downtrend")

    if ta.get('above_sma50'):
        score += 1.0
        notes.append(f"Above 50 DMA ({fmt(ta.get('sma50'))}) — medium-term bullish")
    else:
        score -= 1.0
        notes.append(f"Below 50 DMA ({fmt(ta.get('sma50'))}) — medium-term bearish")

    if ta.get('golden_cross'):
        score += 1.5
        notes.append("Golden cross (50>200 DMA) — major bullish signal")
    else:
        score -= 0.5
        notes.append("Death cross / no golden cross")

    if ta.get('strong_trend'):
        adx = ta.get('adx', 0)
        if ta.get('bullish_di'):
            score += 1.0
            notes.append(f"ADX {fmt(adx)} — strong bullish trend (+DI>{ta.get('plus_di', 0):.1f} > -DI{ta.get('minus_di', 0):.1f})")
        else:
            score -= 1.0
            notes.append(f"ADX {fmt(adx)} — strong bearish trend (-DI dominant)")

    ich = ta.get('ichimoku', {})
    if ich.get('price_above_cloud'):
        score += 0.5
        notes.append("Price above Ichimoku cloud — bullish")
    elif ich.get('price_above_cloud') is False:
        score -= 0.5
        notes.append("Price below Ichimoku cloud — bearish")

    return max(0, min(10, score)), notes


def score_momentum(ta: dict) -> tuple[float, list[str]]:
    """Score momentum indicators 0-10."""
    score = 5.0
    notes = []

    rsi = ta.get('rsi')
    if rsi is not None:
        if 50 < rsi < 70:
            score += 2.0
            notes.append(f"RSI {fmt(rsi)} — bullish momentum zone")
        elif rsi > 70:
            score -= 0.5
            notes.append(f"RSI {fmt(rsi)} — overbought, potential pullback")
        elif rsi < 30:
            score -= 1.5
            notes.append(f"RSI {fmt(rsi)} — oversold (could bounce)")
        elif 30 < rsi < 45:
            score -= 1.0
            notes.append(f"RSI {fmt(rsi)} — weak momentum")

    if ta.get('macd_bullish'):
        score += 2.0
        notes.append(f"MACD bullish crossover (MACD={fmt(ta.get('macd'))} > Signal={fmt(ta.get('macd_signal'))})")
    elif ta.get('macd_bullish') is False:
        score -= 2.0
        notes.append(f"MACD bearish (below signal line)")

    hist = ta.get('macd_histogram')
    if hist is not None:
        if hist > 0:
            notes.append(f"MACD histogram positive ({fmt(hist)}) — accelerating upward")
        else:
            notes.append(f"MACD histogram negative ({fmt(hist)}) — momentum weakening")

    stoch_k = ta.get('stoch_k')
    if stoch_k is not None:
        if 40 < stoch_k < 80:
            score += 0.5
            notes.append(f"Stochastic %K {fmt(stoch_k)} — healthy momentum")
        elif stoch_k > 80:
            score -= 0.5
            notes.append(f"Stochastic %K {fmt(stoch_k)} — overbought")
        elif stoch_k < 20:
            score -= 1.0
            notes.append(f"Stochastic %K {fmt(stoch_k)} — oversold")

    return max(0, min(10, score)), notes


def score_volatility(ta: dict) -> tuple[float, list[str]]:
    """Score volatility context 0-10."""
    score = 5.0
    notes = []

    atr_pct = ta.get('atr_pct')
    if atr_pct is not None:
        if atr_pct < 1.5:
            score += 1.0
            notes.append(f"ATR {fmt(atr_pct)}% — low volatility, stable trend")
        elif atr_pct > 4.0:
            score -= 1.0
            notes.append(f"ATR {fmt(atr_pct)}% — high volatility, wider stops needed")
        else:
            notes.append(f"ATR {fmt(atr_pct)}% — moderate volatility")

    bb_pct = ta.get('bb_pct_b')
    if bb_pct is not None:
        if 0.4 < bb_pct < 0.8:
            score += 1.0
            notes.append(f"BB %B {fmt(bb_pct)} — price in healthy zone of bands")
        elif bb_pct > 1.0:
            score -= 1.0
            notes.append(f"BB %B {fmt(bb_pct)} — above upper band, stretched")
        elif bb_pct < 0.0:
            score -= 1.5
            notes.append(f"BB %B {fmt(bb_pct)} — below lower band, very weak")

    if ta.get('bb_squeeze'):
        score += 0.5
        notes.append(f"BB squeeze detected (width {fmt(ta.get('bb_width'))}%) — big move incoming")

    return max(0, min(10, score)), notes


def score_volume(ta: dict) -> tuple[float, list[str]]:
    """Score volume indicators 0-10."""
    score = 5.0
    notes = []

    vol_ratio = ta.get('volume_ratio')
    if vol_ratio is not None:
        if vol_ratio > 1.5:
            score += 1.5
            notes.append(f"Volume {fmt(vol_ratio)}x average — strong participation")
        elif vol_ratio < 0.5:
            score -= 1.0
            notes.append(f"Volume {fmt(vol_ratio)}x average — very low participation")

    if ta.get('obv_trend') == 'rising':
        score += 2.0
        notes.append("OBV rising — accumulation underway")
    else:
        score -= 2.0
        notes.append("OBV falling — distribution phase")

    cmf = ta.get('cmf')
    if cmf is not None:
        if cmf > 0.1:
            score += 1.0
            notes.append(f"CMF {fmt(cmf)} — money flowing in (buying pressure)")
        elif cmf < -0.1:
            score -= 1.0
            notes.append(f"CMF {fmt(cmf)} — money flowing out (selling pressure)")

    return max(0, min(10, score)), notes


def score_support_resistance(ta: dict) -> tuple[float, list[str]]:
    """Score price vs key S/R levels 0-10."""
    score = 5.0
    notes = []
    price = ta.get('current_price')
    if not price:
        return score, notes

    fib = ta.get('fibonacci', {})
    supports = ta.get('support_levels', [])
    resistances = ta.get('resistance_levels', [])

    if supports:
        nearest_sup = supports[0] if supports else None
        if nearest_sup:
            pct_above = (price - nearest_sup) / nearest_sup * 100
            notes.append(f"Nearest support: ₹{nearest_sup} ({fmt(pct_above)}% below current)")
            if pct_above < 5:
                score += 1.0  # close to support — bounce opportunity
            elif pct_above > 15:
                score -= 0.5  # far from support — downside room

    if resistances:
        nearest_res = resistances[0] if resistances else None
        if nearest_res:
            pct_below = (nearest_res - price) / price * 100
            notes.append(f"Nearest resistance: ₹{nearest_res} ({fmt(pct_below)}% above)")
            if pct_below < 3:
                score -= 0.5  # very close to resistance

    piv = ta.get('pivot_points', {})
    if piv:
        pp = piv.get('PP')
        if pp:
            if price > pp:
                score += 0.5
                notes.append(f"Above pivot point ₹{pp} — bullish intraday bias")
            else:
                score -= 0.5
                notes.append(f"Below pivot point ₹{pp} — bearish intraday bias")

    return max(0, min(10, score)), notes


def compute_composite_technical_score(ta: dict) -> tuple[float, dict]:
    """Weighted composite of all TA sub-scores."""
    trend_score, trend_notes = score_trend(ta)
    momentum_score, momentum_notes = score_momentum(ta)
    vol_score, vol_notes = score_volatility(ta)
    volume_score, volume_notes = score_volume(ta)
    sr_score, sr_notes = score_support_resistance(ta)

    weights = {'trend': 0.25, 'momentum': 0.30, 'volatility': 0.20, 'volume': 0.15, 'sr': 0.10}

    composite = (
        trend_score * weights['trend'] +
        momentum_score * weights['momentum'] +
        vol_score * weights['volatility'] +
        volume_score * weights['volume'] +
        sr_score * weights['sr']
    )

    breakdown = {
        'trend': {'score': round(trend_score, 2), 'notes': trend_notes, 'weight': weights['trend']},
        'momentum': {'score': round(momentum_score, 2), 'notes': momentum_notes, 'weight': weights['momentum']},
        'volatility': {'score': round(vol_score, 2), 'notes': vol_notes, 'weight': weights['volatility']},
        'volume': {'score': round(volume_score, 2), 'notes': volume_notes, 'weight': weights['volume']},
        'sr': {'score': round(sr_score, 2), 'notes': sr_notes, 'weight': weights['sr']},
    }

    return round(composite, 2), breakdown


def build_technical_prompt(symbol: str, ta: dict, breakdown: dict) -> str:
    """Build the detailed TA prompt for the LLM."""
    ich = ta.get('ichimoku', {})
    fib = ta.get('fibonacci', {})
    piv = ta.get('pivot_points', {})

    prompt = f"""
=== TECHNICAL ANALYSIS REQUEST ===
Stock: {symbol}
Current Price: ₹{fmt(ta.get('current_price'))}
1D Change: {fmt(ta.get('price_change_1d'))}%  |  1W: {fmt(ta.get('price_change_1w'))}%  |  1M: {fmt(ta.get('price_change_1m'))}%

--- MOVING AVERAGES ---
SMA 20:    ₹{fmt(ta.get('sma20'))}   ({'ABOVE ✓' if ta.get('above_sma20') else 'BELOW ✗'})
SMA 50:    ₹{fmt(ta.get('sma50'))}   ({'ABOVE ✓' if ta.get('above_sma50') else 'BELOW ✗'})
SMA 200:   ₹{fmt(ta.get('sma200'))}  ({'ABOVE ✓' if ta.get('above_sma200') else 'BELOW ✗'})
EMA 9:     ₹{fmt(ta.get('ema9'))}
EMA 21:    ₹{fmt(ta.get('ema21'))}
Golden Cross: {'YES ✓' if ta.get('golden_cross') else 'NO ✗'}

--- MOMENTUM ---
RSI(14):         {fmt(ta.get('rsi'))}  {'🔴 OVERBOUGHT' if ta.get('rsi_overbought') else '🟢 OVERSOLD' if ta.get('rsi_oversold') else ''}
MACD Line:       {fmt(ta.get('macd'))}
MACD Signal:     {fmt(ta.get('macd_signal'))}
MACD Histogram:  {fmt(ta.get('macd_histogram'))}  ({'BULLISH ✓' if ta.get('macd_bullish') else 'BEARISH ✗'})
Stochastic %K:   {fmt(ta.get('stoch_k'))}  |  %D: {fmt(ta.get('stoch_d'))}
Williams %R:     {fmt(ta.get('williams_r'))}
ROC(10):         {fmt(ta.get('roc_10'))}%
CCI(20):         {fmt(ta.get('cci_20'))}

--- VOLATILITY ---
ATR(14):         ₹{fmt(ta.get('atr'))}  ({fmt(ta.get('atr_pct'))}% of price)
BB Upper:        ₹{fmt(ta.get('bb_upper'))}
BB Middle:       ₹{fmt(ta.get('bb_middle'))}
BB Lower:        ₹{fmt(ta.get('bb_lower'))}
BB Width:        {fmt(ta.get('bb_width'))}%  {'⚡ SQUEEZE' if ta.get('bb_squeeze') else ''}
BB %B:           {fmt(ta.get('bb_pct_b'))}

--- TREND STRENGTH ---
ADX(14):         {fmt(ta.get('adx'))}  ({'STRONG' if ta.get('strong_trend') else 'WEAK'})
+DI:             {fmt(ta.get('plus_di'))}  |  -DI: {fmt(ta.get('minus_di'))}
Direction:       {'BULLISH (+DI > -DI)' if ta.get('bullish_di') else 'BEARISH (-DI > +DI)'}

--- VOLUME ---
Volume:          {fmt(ta.get('volume'), na='N/A')}
Avg Volume(20):  {fmt(ta.get('volume_avg'), na='N/A')}
Volume Ratio:    {fmt(ta.get('volume_ratio'))}x
OBV Trend:       {ta.get('obv_trend', 'N/A').upper()}
CMF(20):         {fmt(ta.get('cmf'))}
VWAP:            ₹{fmt(ta.get('vwap'))}

--- ICHIMOKU CLOUD ---
Tenkan-sen(9):   ₹{fmt(ich.get('tenkan'))}
Kijun-sen(26):   ₹{fmt(ich.get('kijun'))}
Senkou A:        ₹{fmt(ich.get('senkou_a'))}
Senkou B:        ₹{fmt(ich.get('senkou_b'))}
Price vs Cloud:  {'ABOVE ✓ Bullish' if ich.get('price_above_cloud') else 'BELOW ✗ Bearish'}

--- SUPPORT & RESISTANCE ---
Resistance: {' | '.join([f'₹{r}' for r in ta.get('resistance_levels', [])]) or 'N/A'}
Support:    {' | '.join([f'₹{s}' for s in ta.get('support_levels', [])]) or 'N/A'}

--- FIBONACCI LEVELS (last 120 days) ---
Swing High: ₹{fib.get('swing_high', 'N/A')}  |  Swing Low: ₹{fib.get('swing_low', 'N/A')}
78.6%: ₹{fib.get('fib_786', 'N/A')}  |  61.8%: ₹{fib.get('fib_618', 'N/A')}
50.0%: ₹{fib.get('fib_500', 'N/A')}  |  38.2%: ₹{fib.get('fib_382', 'N/A')}
23.6%: ₹{fib.get('fib_236', 'N/A')}

--- PIVOT POINTS (Classic) ---
R3:{piv.get('R3','N/A')}  R2:{piv.get('R2','N/A')}  R1:{piv.get('R1','N/A')}
PP:{piv.get('PP','N/A')}
S1:{piv.get('S1','N/A')}  S2:{piv.get('S2','N/A')}  S3:{piv.get('S3','N/A')}

--- QUANTITATIVE SCORES ---
Trend Score:     {breakdown['trend']['score']}/10
Momentum Score:  {breakdown['momentum']['score']}/10
Volatility:      {breakdown['volatility']['score']}/10
Volume:          {breakdown['volume']['score']}/10
S/R Positioning: {breakdown['sr']['score']}/10

Analyse the above data for Indian stock context and give your technical assessment.
"""
    return prompt


def run(symbol: str, print_output: bool = True) -> dict:
    """Run the Technical Bot analysis."""
    if print_output:
        print(f"\n{'='*60}")
        print(f"📈  TECHNICAL BOT — {symbol}")
        print('='*60)

    df = fetch_price_history(symbol)
    if df.empty or len(df) < 50:
        return {"bot": "technical", "symbol": symbol, "score": 5.0,
                "text": "Insufficient price data for technical analysis.", "breakdown": {}, "ta": {}}

    ta = compute_all_indicators(df)
    composite_score, breakdown = compute_composite_technical_score(ta)
    prompt = build_technical_prompt(symbol, ta, breakdown)

    def on_token(t):
        if print_output:
            print(t, end='', flush=True)

    if print_output:
        print()

    llm_response = stream_chat(SYSTEM_PROMPT, prompt, on_token=on_token)

    if print_output:
        print()

    llm_score = extract_score(llm_response, default=composite_score)
    final_score = round(0.5 * llm_score + 0.5 * composite_score, 2)

    result = {
        "bot": "technical",
        "symbol": symbol,
        "score": final_score,
        "quant_score": composite_score,
        "llm_score": llm_score,
        "text": llm_response,
        "breakdown": breakdown,
        "ta": ta
    }

    save_analysis(symbol, "technical", final_score, llm_response, breakdown)
    return result
