# ============================================================
# bots/sentiment_bot.py — Market Sentiment Bot
# ============================================================
#
# SENTIMENT SOURCES & FORMULAS:
#
# 1. SOCIAL MEDIA SENTIMENT (Reddit, aggregated news)
#    VADER compound score ∈ [-1, +1]
#    FinBERT label: positive/negative/neutral
#    Social Score = 0.6 × avg_compound + 0.4 × bullish_pct/10
#
# 2. FII/DII FLOWS (NSE India)
#    FII Net = FII Buy Value - FII Sell Value
#    DII Net = DII Buy Value - DII Sell Value
#    Flow Signal:
#       FII Net > 0  AND DII Net > 0  → STRONG BULLISH
#       FII Net > 0  OR  DII Net > 0  → BULLISH
#       FII Net < 0  AND DII Net < 0  → STRONG BEARISH
#       else                           → MIXED
#    Flow Score = normalize(FII_Net + DII_Net) to 0-10
#
# 3. OPTIONS MARKET SENTIMENT
#    PCR (Put-Call Ratio) = Total PE OI / Total CE OI
#       PCR > 1.5 → Extreme bullish (contrarian: market is overly hedged)
#       PCR 1.0-1.5 → Bullish
#       PCR 0.7-1.0 → Neutral
#       PCR < 0.7 → Bearish
#    Max Pain = Strike price where option buyers lose maximum money
#    If Current Price > Max Pain → bullish for options sellers (smart money)
#
# 4. INDIA VIX (Fear Index)
#    VIX < 12    → Very low fear, complacency (can be warning)
#    VIX 12-20   → Normal range
#    VIX 20-25   → Elevated fear
#    VIX > 25    → High fear → contrarian buy signal
#    VIX Score = 10 - normalize(VIX, 12, 30) × 10
#
# 5. BULK/BLOCK DEALS
#    Large institutional buys in bulk/block deals = strong bullish signal
#    Heavy institutional selling = bearish
#
# 6. MUTUAL FUND ACTIVITY
#    SIP flows data from AMFI — sustained high SIP = market confidence
#
# COMPOSITE SENTIMENT SCORE:
#    Social sentiment:   0.20
#    FII/DII flows:      0.30
#    Options/PCR:        0.25
#    VIX:                0.15
#    News sentiment:     0.10
# ============================================================

import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.live_data import (
    NSELive, FinnhubData, RedditSentiment, get_all_news
)
from utils.market_data import fetch_fundamentals
from utils.llm import stream_chat, extract_score, fmt
from utils.database import save_analysis
from config import VERBOSE_DEBUG


SYSTEM_PROMPT = """You are SENTIMENT BOT — an expert at reading Indian stock market psychology and flows.

You analyse:
1. FII (Foreign Institutional) vs DII (Domestic Institutional) net flows
2. Options market: PCR, OI buildup, max pain, put/call skew
3. India VIX (fear index) interpretation  
4. Social media: Reddit r/IndianStockMarket, r/DalalStreetBets, financial Twitter India
5. Retail investor mood (herding behaviour, FOMO/panic)
6. Bulk deals and block deals by institutions
7. Delivery percentage (high delivery = conviction buying)

Your response MUST:
- Be 150-200 words
- Clearly state FII/DII stance: BUYING or SELLING and by how much
- Interpret the options PCR and what smart money is positioning for
- Comment on retail sentiment vs institutional
- End with exactly: "SENTIMENT SCORE: X/10" where X is your assessment

Scoring:
  9-10: FII buying, PCR bullish, VIX falling, positive social sentiment
  7-8:  Net positive institutional flows, decent retail confidence
  5-6:  Mixed signals, FII cautious, PCR neutral
  3-4:  FII selling, negative sentiment, VIX rising
  1-2:  Heavy FII outflows, panic in market, VIX spiking
"""


def fetch_india_vix() -> dict:
    """
    Fetch India VIX from NSE indices.
    VIX < 12 = complacency | 12-20 = normal | >25 = fear
    """
    try:
        indices = NSELive.get_all_indices()
        if indices:
            for idx in indices:
                if 'VIX' in (idx.get('name') or '').upper():
                    vix = idx.get('last')
                    change = idx.get('pct_change', 0)
                    return {
                        'vix': vix,
                        'vix_change_pct': change,
                        'level': (
                            'EXTREME_FEAR' if vix and vix > 25
                            else 'HIGH_FEAR' if vix and vix > 20
                            else 'NORMAL' if vix and vix > 12
                            else 'COMPLACENCY'
                        ),
                        'contrarian_signal': 'BUY' if vix and vix > 25 else 'NEUTRAL',
                        'source': 'NSE'
                    }
    except Exception as e:
        if VERBOSE_DEBUG:
            print(f"[SENTIMENT] VIX error: {e}")
    return {'vix': None, 'level': 'unknown', 'source': 'failed'}


def score_vix(vix_data: dict) -> tuple[float, str]:
    """
    Score VIX contribution to sentiment.
    High VIX = fear = contrarian bullish (counter-intuitive but proven).
    Low VIX = complacency = mild negative (overconfidence).
    """
    vix = vix_data.get('vix')
    if vix is None:
        return 5.0, "VIX data unavailable"

    change = vix_data.get('vix_change_pct', 0) or 0

    if vix < 12:
        score = 5.5  # complacency — slight concern
        note = f"India VIX {vix:.1f} — dangerously low, market complacency"
    elif vix < 16:
        score = 7.0  # healthy low volatility
        note = f"India VIX {vix:.1f} — calm markets, bullish backdrop"
    elif vix < 20:
        score = 6.0
        note = f"India VIX {vix:.1f} — moderate volatility, normal range"
    elif vix < 25:
        score = 5.0
        note = f"India VIX {vix:.1f} — elevated fear, caution advised"
    elif vix < 30:
        score = 6.5  # contrarian: high fear = buy opportunity
        note = f"India VIX {vix:.1f} — HIGH FEAR = contrarian buy zone"
    else:
        score = 7.0  # extreme fear = strong contrarian buy
        note = f"India VIX {vix:.1f} — EXTREME FEAR, historically strong contrarian buy"

    # VIX rising = increasing fear (slightly negative current)
    if change > 10:
        score -= 0.5
        note += f" (VIX up {change:.1f}% — fear increasing)"
    elif change < -10:
        score += 0.5
        note += f" (VIX down {change:.1f}% — fear easing)"

    return round(score, 2), note


def score_fii_dii(fii_dii: dict) -> tuple[float, str]:
    """
    Score FII/DII flows.
    FII net positive = bullish, FII net negative = bearish.
    DII tends to be counter-cyclical but still matters.
    """
    if not fii_dii:
        return 5.0, "FII/DII data unavailable"

    fii = fii_dii.get('fii', {})
    dii = fii_dii.get('dii', {})

    fii_net = _safe_float(fii.get('net_value')) or 0
    dii_net = _safe_float(dii.get('net_value')) or 0

    score = 5.0
    combined_net = fii_net + dii_net

    # FII flows (weight: 70% of this sub-score)
    if fii_net > 2000:
        score += 2.5
        fii_note = f"FII buying ₹{fii_net:.0f} Cr — heavy accumulation"
    elif fii_net > 500:
        score += 1.5
        fii_note = f"FII net buying ₹{fii_net:.0f} Cr"
    elif fii_net > 0:
        score += 0.5
        fii_note = f"FII marginal buying ₹{fii_net:.0f} Cr"
    elif fii_net < -2000:
        score -= 2.5
        fii_note = f"FII SELLING ₹{abs(fii_net):.0f} Cr — heavy outflows"
    elif fii_net < -500:
        score -= 1.5
        fii_note = f"FII net selling ₹{abs(fii_net):.0f} Cr"
    else:
        fii_note = f"FII neutral ₹{fii_net:.0f} Cr"

    # DII flows (weight: 30%)
    if dii_net > 1000:
        score += 1.0
        dii_note = f"DII buying ₹{dii_net:.0f} Cr — domestic support"
    elif dii_net > 0:
        score += 0.3
        dii_note = f"DII marginal buying ₹{dii_net:.0f} Cr"
    elif dii_net < -1000:
        score -= 1.0
        dii_note = f"DII selling ₹{abs(dii_net):.0f} Cr"
    else:
        dii_note = f"DII neutral ₹{dii_net:.0f} Cr"

    note = f"{fii_note} | {dii_note} | Net combined: ₹{combined_net:.0f} Cr"
    return max(0, min(10, round(score, 2))), note


def score_options_pcr(options: dict) -> tuple[float, str]:
    """
    Score options market sentiment via PCR and OI data.
    PCR interpretation:
        > 1.5 = Extreme bearishness priced in = contrarian bullish
        1.0-1.5 = Bullish
        0.8-1.0 = Neutral
        < 0.7 = Bullish options positioning = bearish for stock (calls dominate)
    """
    if not options:
        return 5.0, "Options data unavailable"

    pcr = options.get('pcr')
    if pcr is None:
        return 5.0, "PCR unavailable"

    signal = options.get('pcr_signal', 'NEUTRAL')

    if pcr > 1.5:
        score = 7.5
        note = f"PCR {pcr:.3f} — extreme put buying, contrarian BULLISH (market over-hedged)"
    elif pcr > 1.2:
        score = 7.0
        note = f"PCR {pcr:.3f} — {signal}: healthy bullish sentiment in derivatives"
    elif pcr > 0.9:
        score = 6.0
        note = f"PCR {pcr:.3f} — neutral derivatives positioning"
    elif pcr > 0.7:
        score = 5.0
        note = f"PCR {pcr:.3f} — mild call buying, slight caution"
    else:
        score = 3.5
        note = f"PCR {pcr:.3f} — heavy call buying = bearish signal (complacency)"

    # Add call/put OI context
    call_oi = options.get('call_oi')
    put_oi = options.get('put_oi')
    if call_oi and put_oi:
        note += f" | Call OI: {call_oi:,} | Put OI: {put_oi:,}"

    return round(score, 2), note


def score_social_sentiment(reddit: dict, finnhub_sentiment: dict, news_sentiment: float) -> tuple[float, str]:
    """
    Aggregate social media sentiment score.
    """
    scores = []
    notes = []

    # Reddit sentiment (0.4 weight)
    if reddit and reddit.get('mentions', 0) > 0:
        r_sent = reddit.get('sentiment', 0)
        r_bull = reddit.get('bullish_pct', 50)
        r_score = 5.0 + (r_sent * 3) + (r_bull - 50) / 25
        r_score = max(0, min(10, r_score))
        scores.append((r_score, 0.4))
        notes.append(
            f"Reddit: {reddit['mentions']} mentions | sentiment {r_sent:.3f} | "
            f"bullish {reddit.get('bullish_pct', 0):.0f}%"
        )

    # Finnhub news sentiment (0.35 weight)
    if finnhub_sentiment:
        fh_bull = finnhub_sentiment.get('sentiment_bullish', 0.5) or 0.5
        fh_score = 5.0 + (fh_bull - 0.5) * 10
        fh_score = max(0, min(10, fh_score))
        scores.append((fh_score, 0.35))
        company_score = finnhub_sentiment.get('company_score', 0)
        notes.append(
            f"Finnhub: {fh_bull*100:.0f}% bullish news | "
            f"company score {company_score:.3f}"
        )

    # News VADER aggregate (0.25 weight)
    news_score = 5.0 + (news_sentiment * 5)
    news_score = max(0, min(10, news_score))
    scores.append((news_score, 0.25))
    notes.append(f"News VADER: compound {news_sentiment:.3f}")

    if not scores:
        return 5.0, "No social sentiment data"

    # Weighted average
    total_weight = sum(w for _, w in scores)
    final = sum(s * w for s, w in scores) / total_weight
    return round(final, 2), ' | '.join(notes)


def compute_composite_sentiment_score(vix_score, vix_note,
                                       fii_score, fii_note,
                                       pcr_score, pcr_note,
                                       social_score, social_note,
                                       news_sent: float) -> tuple[float, dict]:
    """
    Final weighted composite sentiment score.
    
    Weights:
      FII/DII flows:    30%  (institutional money is smart money)
      Options/PCR:      25%  (derivatives = forward-looking)
      Social:           20%  (retail momentum matters)
      VIX:              15%  (macro fear/greed)
      News sentiment:   10%  (headline risk)
    """
    news_score = 5.0 + (news_sent * 5)
    news_score = max(0, min(10, news_score))

    weights = {
        'fii_dii': 0.30,
        'options': 0.25,
        'social': 0.20,
        'vix': 0.15,
        'news': 0.10,
    }

    composite = (
        fii_score * weights['fii_dii'] +
        pcr_score * weights['options'] +
        social_score * weights['social'] +
        vix_score * weights['vix'] +
        news_score * weights['news']
    )

    breakdown = {
        'fii_dii': {'score': fii_score, 'note': fii_note, 'weight': weights['fii_dii']},
        'options': {'score': pcr_score, 'note': pcr_note, 'weight': weights['options']},
        'social': {'score': social_score, 'note': social_note, 'weight': weights['social']},
        'vix': {'score': vix_score, 'note': vix_note, 'weight': weights['vix']},
        'news': {'score': round(news_score, 2), 'note': f"News sentiment: {news_sent:.4f}", 'weight': weights['news']},
    }

    return round(composite, 2), breakdown


def build_sentiment_prompt(symbol: str, company_name: str,
                            vix: dict, fii_dii: dict, options: dict,
                            reddit: dict, fh_sentiment: dict,
                            indices: list, breakdown: dict) -> str:

    # Indices summary
    idx_lines = []
    if indices:
        for idx in (indices or [])[:5]:
            chg = idx.get('pct_change', 0) or 0
            arrow = '▲' if chg >= 0 else '▼'
            idx_lines.append(f"  {idx['name']}: {idx.get('last', 'N/A')} {arrow}{abs(chg):.2f}%")
    idx_str = '\n'.join(idx_lines) if idx_lines else "  (unavailable)"

    # FII/DII
    fii = fii_dii.get('fii', {}) if fii_dii else {}
    dii = fii_dii.get('dii', {}) if fii_dii else {}

    # Options
    pcr = options.get('pcr', 'N/A') if options else 'N/A'
    pcr_signal = options.get('pcr_signal', 'N/A') if options else 'N/A'

    # Reddit
    reddit_mentions = reddit.get('mentions', 0) if reddit else 0
    reddit_sent = reddit.get('sentiment', 0) if reddit else 0
    reddit_bull = reddit.get('bullish_pct', 0) if reddit else 0

    # Top Reddit posts
    reddit_posts = []
    if reddit and reddit.get('posts'):
        for p in reddit['posts'][:3]:
            emoji = '🟢' if p.get('sentiment', 0) > 0.05 else '🔴' if p.get('sentiment', 0) < -0.05 else '⚪'
            reddit_posts.append(f"  {emoji} [{p['subreddit']}] {p['title'][:80]} (↑{p['score']})")
    reddit_str = '\n'.join(reddit_posts) if reddit_posts else "  No Reddit mentions found"

    prompt = f"""
=== SENTIMENT ANALYSIS REQUEST ===
Stock: {symbol}
Company: {company_name}

--- MARKET INDICES (Live) ---
{idx_str}

--- INDIA VIX ---
VIX Level:     {vix.get('vix', 'N/A')}
VIX Change:    {vix.get('vix_change_pct', 'N/A')}%
Status:        {vix.get('level', 'N/A')}
Signal:        {vix.get('contrarian_signal', 'N/A')}

--- FII / DII FLOWS ---
FII Net Flow:   ₹{fii.get('net_value', 'N/A')} Cr
FII Buy:        ₹{fii.get('buy_value', 'N/A')} Cr
FII Sell:       ₹{fii.get('sell_value', 'N/A')} Cr
DII Net Flow:   ₹{dii.get('net_value', 'N/A')} Cr
DII Buy:        ₹{dii.get('buy_value', 'N/A')} Cr
DII Sell:       ₹{dii.get('sell_value', 'N/A')} Cr
Date:           {fii.get('date', 'N/A')}

--- OPTIONS CHAIN ---
Put-Call Ratio (PCR): {pcr}
PCR Signal:           {pcr_signal}
Total Call OI:        {options.get('call_oi', 'N/A') if options else 'N/A'}
Total Put OI:         {options.get('put_oi', 'N/A') if options else 'N/A'}
Expiry:               {options.get('expiry', 'N/A') if options else 'N/A'}

--- SOCIAL SENTIMENT ---
Reddit Mentions (7 days): {reddit_mentions}
Reddit Avg Sentiment:     {reddit_sent:.4f}
Reddit Bullish:           {reddit_bull:.1f}%
Finnhub Bullish News:     {(fh_sentiment or {}).get('sentiment_bullish', 'N/A')}
Finnhub Company Score:    {(fh_sentiment or {}).get('company_score', 'N/A')}
Buzz Articles (7d):       {(fh_sentiment or {}).get('buzz_articles', 'N/A')}

Top Reddit Posts:
{reddit_str}

--- SENTIMENT BREAKDOWN SCORES ---
FII/DII Score:  {breakdown.get('fii_dii', {}).get('score', 'N/A')}/10
Options Score:  {breakdown.get('options', {}).get('score', 'N/A')}/10
Social Score:   {breakdown.get('social', {}).get('score', 'N/A')}/10
VIX Score:      {breakdown.get('vix', {}).get('score', 'N/A')}/10
News Score:     {breakdown.get('news', {}).get('score', 'N/A')}/10
COMPOSITE:      {sum(v.get('score',0)*v.get('weight',0) for v in breakdown.values()):.1f}/10

Analyse the sentiment landscape for {symbol} in the current Indian market context.
Focus on what FII/DII flows tell us about institutional intent, and what the options
market signals about near-term expectations.
"""
    return prompt


def run(symbol: str, news_sentiment: float = 0.0, print_output: bool = True) -> dict:
    """Run the Sentiment Bot analysis."""
    if print_output:
        print(f"\n{'='*60}")
        print(f"💬  SENTIMENT BOT — {symbol}")
        print('='*60)

    fund = fetch_fundamentals(symbol)
    company_name = (fund or {}).get('company_name', symbol)

    # Fetch all sentiment data in parallel (sequentially for simplicity)
    if print_output:
        print("[SENTIMENT] Fetching India VIX...")
    vix_data = fetch_india_vix()

    if print_output:
        print("[SENTIMENT] Fetching FII/DII flows...")
    fii_dii = NSELive.get_fii_dii()

    if print_output:
        print("[SENTIMENT] Fetching Option chain (PCR)...")
    options = NSELive.get_option_chain(symbol)

    if print_output:
        print("[SENTIMENT] Fetching Reddit sentiment...")
    reddit = RedditSentiment.search_mentions(symbol, company_name)

    if print_output:
        print("[SENTIMENT] Fetching Finnhub sentiment...")
    fh_sentiment = FinnhubData.get_news_sentiment(symbol)

    if print_output:
        print("[SENTIMENT] Fetching market indices...")
    indices = NSELive.get_all_indices()

    # Score each component
    vix_score, vix_note = score_vix(vix_data)
    fii_score, fii_note = score_fii_dii(fii_dii or {})
    pcr_score, pcr_note = score_options_pcr(options or {})
    social_score, social_note = score_social_sentiment(reddit, fh_sentiment, news_sentiment)

    # Composite score
    quant_score, breakdown = compute_composite_sentiment_score(
        vix_score, vix_note,
        fii_score, fii_note,
        pcr_score, pcr_note,
        social_score, social_note,
        news_sentiment
    )

    # Build prompt and call LLM
    prompt = build_sentiment_prompt(
        symbol, company_name,
        vix_data, fii_dii or {}, options or {},
        reddit, fh_sentiment,
        indices or [], breakdown
    )

    def on_token(t):
        if print_output:
            print(t, end='', flush=True)

    if print_output:
        print()

    llm_response = stream_chat(SYSTEM_PROMPT, prompt, on_token=on_token)

    if print_output:
        print()

    llm_score = extract_score(llm_response, default=quant_score)
    final_score = round(0.55 * llm_score + 0.45 * quant_score, 2)

    result = {
        "bot": "sentiment",
        "symbol": symbol,
        "score": final_score,
        "quant_score": quant_score,
        "llm_score": llm_score,
        "text": llm_response,
        "breakdown": breakdown,
        "vix": vix_data,
        "fii_dii": fii_dii,
        "options": options,
        "reddit": reddit,
    }

    save_analysis(symbol, "sentiment", final_score, llm_response, breakdown)
    return result


def _safe_float(val):
    try:
        return float(str(val).replace(',', ''))
    except Exception:
        return 0.0
