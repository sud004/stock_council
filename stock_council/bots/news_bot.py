# ============================================================
# bots/news_bot.py — News & Macro Analysis Bot
# ============================================================
#
# DATA SOURCES:
#   - RSS feeds (Moneycontrol, ET, Business Standard, LiveMint)
#   - Yahoo Finance news
#   - NSE corporate actions (dividends, splits, bonus)
#
# SENTIMENT SCORING:
#   - VADER sentiment (offline, lexicon-based):
#       compound score ∈ [-1, 1]
#       compound > 0.05  → positive
#       compound < -0.05 → negative
#       else             → neutral
#
#   - Transformer-based (optional, local HuggingFace model):
#       Model: ProsusAI/finbert (financial BERT)
#       OR: yiyanghkust/finbert-tone
#       Outputs: positive / negative / neutral + confidence
#
#   - Final news sentiment:
#       score = weighted avg of:
#           title_sentiment × 0.4 (titles matter more)
#           body_sentiment × 0.6
#
# RELEVANCE SCORING:
#   Articles filtered by keyword match to stock symbol/company name.
#   Relevance = keyword_match_count / total_keywords × 100
#
# NEWS COMPOSITE SCORE:
#   Volume of coverage: 0.20
#   Sentiment score:    0.40
#   Macro alignment:    0.20
#   Corporate actions:  0.20
# ============================================================

import sys
import re
import time
import json
import requests
import feedparser
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import NEWS_SOURCES, ALLOW_INTERNET, CACHE_TTL_NEWS_HOURS, VERBOSE_DEBUG
from utils.market_data import fetch_fundamentals, resolve_symbol
from utils.llm import stream_chat, extract_score, fmt
from utils.database import save_analysis, save_news, load_news


SYSTEM_PROMPT = """You are NEWS BOT — an expert Indian financial news analyst who monitors BSE/NSE listed companies.

You track:
- Corporate announcements: earnings, management changes, M&A, capex plans
- Regulatory: SEBI actions, RBI policy, government schemes (PLI, Make in India), taxation
- Macro: FII/DII flows, rupee movement, crude oil, US Fed impact on Indian markets
- Sector: tailwinds and headwinds specific to the company's industry
- Global: supply chain, China factor, export markets

Your response MUST:
- Be 150-200 words
- Categorize news as: BULLISH catalyst / BEARISH catalyst / NEUTRAL
- Identify the single most important news item and its likely price impact
- Comment on macro environment for this specific sector
- End with exactly: "NEWS SCORE: X/10" where X is your assessment

Scoring:
  9-10: Very positive news flow, strong catalysts, macro tailwinds
  7-8:  Net positive news environment
  5-6:  Mixed news, no clear direction
  3-4:  Negative news flow
  1-2:  Very negative — earnings miss, scandal, regulatory action
"""


# ── Offline VADER Sentiment ────────────────────────────────────

def get_vader():
    """Get VADER sentiment analyzer (offline, no model download needed)."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except ImportError:
        print("[NEWS] VADER not installed. Run: pip install vaderSentiment")
        return None


# Financial domain words to boost VADER scores
FINANCIAL_LEXICON = {
    # Bullish
    'outperform': 2.0, 'upgrade': 2.0, 'buy': 1.5, 'strong': 1.5,
    'beat': 2.0, 'exceed': 2.0, 'record': 1.5, 'growth': 1.5,
    'acquisition': 0.5, 'expansion': 1.5, 'profit': 1.5, 'rally': 2.0,
    'breakout': 2.0, 'surge': 2.0, 'bullish': 2.5, 'accumulate': 1.5,
    'dividend': 1.0, 'bonus': 1.0, 'buyback': 1.5, 'order': 0.5,
    'win': 1.5, 'contract': 0.3, 'launch': 0.5, 'capex': 0.3,

    # Bearish
    'miss': -2.0, 'loss': -2.0, 'downgrade': -2.0, 'sell': -1.5,
    'weak': -1.5, 'decline': -1.5, 'fall': -1.0, 'crash': -3.0,
    'fraud': -3.0, 'probe': -2.0, 'investigation': -2.0, 'sebi': -0.5,
    'bearish': -2.5, 'debt': -0.5, 'default': -3.0, 'insolvency': -3.0,
    'penalty': -2.0, 'fine': -1.5, 'layoff': -1.5, 'restructure': -0.5,
    'slowdown': -1.5, 'headwind': -1.5, 'concern': -1.0,
}


def vader_sentiment(text: str, vader=None) -> dict:
    """
    Analyse text sentiment using VADER with financial lexicon augmentation.
    Returns: {compound, pos, neg, neu, label}
    """
    if vader is None:
        vader = get_vader()
    if vader is None:
        return {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 1.0, 'label': 'neutral'}

    # Augment VADER with financial terms
    for word, score in FINANCIAL_LEXICON.items():
        vader.lexicon[word] = score

    scores = vader.polarity_scores(text.lower())
    compound = scores['compound']
    label = 'positive' if compound > 0.05 else 'negative' if compound < -0.05 else 'neutral'
    scores['label'] = label
    return scores


def try_finbert_sentiment(texts: list) -> list:
    """
    Optional: Use FinBERT for more accurate financial sentiment.
    Falls back gracefully if transformers/model not available.
    
    Model: ProsusAI/finbert (downloaded to models/ directory on first run)
    """
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
        model_path = Path(__file__).parent.parent / "models" / "finbert"

        if not model_path.exists():
            print("[NEWS] Downloading FinBERT model (first time only, ~438MB)...")
            model_path.mkdir(parents=True, exist_ok=True)
            tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            tokenizer.save_pretrained(str(model_path))
            model.save_pretrained(str(model_path))

        pipe = pipeline("text-classification",
                       model=str(model_path),
                       tokenizer=str(model_path),
                       truncation=True,
                       max_length=512)
        results = []
        for text in texts:
            out = pipe(text[:512])
            label = out[0]['label'].lower()
            score = out[0]['score']
            # FinBERT: positive=+score, negative=-score, neutral=0
            compound = score if label == 'positive' else -score if label == 'negative' else 0.0
            results.append({'compound': compound, 'label': label, 'confidence': score})
        return results
    except Exception as e:
        if VERBOSE_DEBUG:
            print(f"[NEWS] FinBERT unavailable ({e}), using VADER")
        return []


# ── News Fetching ─────────────────────────────────────────────

def clean_html(text: str) -> str:
    """Strip HTML tags from text."""
    return re.sub(r'<[^>]+>', '', text or '').strip()


def fetch_rss_news(symbol: str, company_name: str) -> list:
    """
    Fetch news articles from RSS feeds and filter by stock relevance.
    Returns list of article dicts.
    """
    # Build search keywords
    short_sym = symbol.replace('.NS', '').replace('.BO', '').upper()
    company_words = [w for w in (company_name or '').split() if len(w) > 2]
    keywords = [short_sym] + company_words[:3]

    articles = []
    vader = get_vader()

    for source in NEWS_SOURCES:
        if not ALLOW_INTERNET:
            break
        try:
            if VERBOSE_DEBUG:
                print(f"[NEWS] Fetching RSS: {source['name']}")
            feed = feedparser.parse(source['url'], request_headers={
                'User-Agent': 'Mozilla/5.0 (research bot)'
            })
            for entry in feed.entries[:30]:
                title = clean_html(getattr(entry, 'title', ''))
                summary = clean_html(getattr(entry, 'summary', getattr(entry, 'description', '')))
                url = getattr(entry, 'link', '')
                published = getattr(entry, 'published', str(datetime.now()))

                # Relevance check
                combined = (title + ' ' + summary).lower()
                match_count = sum(1 for kw in keywords if kw.lower() in combined)
                if match_count == 0:
                    continue

                # Sentiment
                title_sent = vader_sentiment(title, vader)
                body_sent = vader_sentiment(summary, vader)
                # Weighted sentiment: title 40%, body 60%
                compound = 0.4 * title_sent['compound'] + 0.6 * body_sent['compound']

                articles.append({
                    'title': title,
                    'summary': summary[:500],
                    'source': source['name'],
                    'url': url,
                    'published': published,
                    'sentiment': round(compound, 4),
                    'relevance': match_count,
                    'symbol': symbol
                })
        except Exception as e:
            if VERBOSE_DEBUG:
                print(f"[NEWS] Error fetching {source['name']}: {e}")

    # Sort by relevance then recency
    articles.sort(key=lambda x: (x['relevance'], x['published']), reverse=True)
    return articles[:20]


def fetch_yahoo_news(symbol: str) -> list:
    """Fetch news from Yahoo Finance for the ticker."""
    if not ALLOW_INTERNET:
        return []
    try:
        import yfinance as yf
        yf_sym = resolve_symbol(symbol)
        ticker = yf.Ticker(yf_sym)
        news = ticker.news or []
        vader = get_vader()
        result = []
        for item in news[:10]:
            title = item.get('title', '')
            summary = item.get('summary', item.get('description', ''))
            compound = vader_sentiment(title + ' ' + summary, vader).get('compound', 0.0)
            result.append({
                'title': title,
                'summary': summary[:400],
                'source': item.get('publisher', 'Yahoo Finance'),
                'url': item.get('link', ''),
                'published': datetime.fromtimestamp(item.get('providerPublishTime', time.time())).isoformat(),
                'sentiment': round(compound, 4),
                'relevance': 3,
                'symbol': symbol
            })
        return result
    except Exception as e:
        if VERBOSE_DEBUG:
            print(f"[NEWS] Yahoo news error: {e}")
        return []


def fetch_corporate_actions(symbol: str) -> dict:
    """
    Fetch upcoming/recent corporate actions (dividends, splits, bonus, AGM).
    Uses Yahoo Finance calendar data.
    """
    if not ALLOW_INTERNET:
        return {}
    try:
        import yfinance as yf
        yf_sym = resolve_symbol(symbol)
        ticker = yf.Ticker(yf_sym)
        actions = {}

        # Dividends (last 4)
        try:
            div = ticker.dividends
            if not div.empty:
                actions['recent_dividends'] = [
                    {'date': str(d.date()), 'amount': round(float(v), 2)}
                    for d, v in list(div.items())[-4:]
                ]
        except Exception:
            pass

        # Splits
        try:
            splits = ticker.splits
            if not splits.empty:
                actions['recent_splits'] = [
                    {'date': str(d.date()), 'ratio': f"{v}:1"}
                    for d, v in list(splits.items())[-2:]
                ]
        except Exception:
            pass

        # Calendar
        try:
            cal = ticker.calendar
            if cal is not None:
                earnings_date = cal.get('Earnings Date', [])
                if earnings_date:
                    actions['next_earnings'] = str(earnings_date[0]) if hasattr(earnings_date[0], 'strftime') else str(earnings_date[0])
        except Exception:
            pass

        return actions
    except Exception as e:
        if VERBOSE_DEBUG:
            print(f"[NEWS] Corporate actions error: {e}")
        return {}


def aggregate_sentiment(articles: list) -> dict:
    """
    Aggregate sentiment from all articles.
    Returns summary statistics.
    """
    if not articles:
        return {'avg_sentiment': 0.0, 'positive_pct': 0.0, 'negative_pct': 0.0,
                'neutral_pct': 100.0, 'article_count': 0, 'sentiment_label': 'neutral'}

    sentiments = [a['sentiment'] for a in articles]
    avg = np.mean(sentiments)
    positive = sum(1 for s in sentiments if s > 0.05)
    negative = sum(1 for s in sentiments if s < -0.05)
    neutral = len(sentiments) - positive - negative
    n = len(sentiments)

    label = 'positive' if avg > 0.05 else 'negative' if avg < -0.05 else 'neutral'

    return {
        'avg_sentiment': round(avg, 4),
        'sentiment_std': round(np.std(sentiments), 4),
        'positive_pct': round(positive / n * 100, 1),
        'negative_pct': round(negative / n * 100, 1),
        'neutral_pct': round(neutral / n * 100, 1),
        'article_count': n,
        'sentiment_label': label
    }


def compute_news_score(articles: list, agg: dict, actions: dict) -> tuple[float, dict]:
    """
    Compute news score 0-10 from:
      - Coverage volume (more relevant coverage = more important stock)
      - Sentiment direction and strength
      - Corporate action quality
      - Macro environment
    """
    score = 5.0
    breakdown = {}

    # Volume score
    n = agg['article_count']
    if n > 10:
        score += 0.5
        vol_note = f"High coverage: {n} articles found"
    elif n > 5:
        vol_note = f"Moderate coverage: {n} articles"
    elif n == 0:
        score -= 0.5
        vol_note = "No recent news found"
    else:
        vol_note = f"Low coverage: {n} articles"
    breakdown['volume'] = vol_note

    # Sentiment score (primary driver)
    avg_sent = agg['avg_sentiment']
    pos_pct = agg['positive_pct']
    neg_pct = agg['negative_pct']

    if avg_sent > 0.3:
        score += 2.5
    elif avg_sent > 0.1:
        score += 1.5
    elif avg_sent > 0.0:
        score += 0.5
    elif avg_sent < -0.3:
        score -= 2.5
    elif avg_sent < -0.1:
        score -= 1.5
    elif avg_sent < 0.0:
        score -= 0.5

    breakdown['sentiment'] = f"Avg sentiment: {avg_sent:.3f} | Positive: {pos_pct}% | Negative: {neg_pct}%"

    # Corporate actions
    if 'next_earnings' in actions:
        breakdown['earnings'] = f"Next earnings: {actions['next_earnings']}"
    if actions.get('recent_dividends'):
        last_div = actions['recent_dividends'][-1]
        score += 0.3
        breakdown['dividend'] = f"Recent dividend: ₹{last_div['amount']} ({last_div['date']})"
    if actions.get('recent_splits'):
        score += 0.2
        breakdown['splits'] = f"Recent split: {actions['recent_splits'][-1]['ratio']}"

    breakdown['final'] = f"Computed news score: {round(score, 2)}/10"

    return max(0, min(10, round(score, 2))), breakdown


def build_news_prompt(symbol: str, fund: dict, articles: list, agg: dict,
                      actions: dict, score: float) -> str:
    """Build the LLM prompt with all news data."""

    # Top 5 articles
    top_articles = articles[:5]
    article_lines = []
    for i, a in enumerate(top_articles, 1):
        sent_emoji = '🟢' if a['sentiment'] > 0.05 else '🔴' if a['sentiment'] < -0.05 else '⚪'
        article_lines.append(
            f"{i}. {sent_emoji} [{a['source']}] {a['title']}\n"
            f"   Sentiment: {a['sentiment']:.3f} | {a['summary'][:150]}..."
        )
    articles_str = '\n'.join(article_lines) if article_lines else "No relevant articles found."

    # Corporate actions
    action_lines = []
    if actions.get('next_earnings'):
        action_lines.append(f"Next Earnings Date: {actions['next_earnings']}")
    if actions.get('recent_dividends'):
        for d in actions['recent_dividends'][-2:]:
            action_lines.append(f"Dividend: ₹{d['amount']} on {d['date']}")
    if actions.get('recent_splits'):
        for s in actions['recent_splits']:
            action_lines.append(f"Stock Split {s['ratio']} on {s['date']}")
    actions_str = '\n'.join(action_lines) if action_lines else "None found"

    prompt = f"""
=== NEWS & MACRO ANALYSIS REQUEST ===
Stock: {symbol}
Company: {fund.get('company_name', 'N/A')}
Sector: {fund.get('sector', 'N/A')} | Industry: {fund.get('industry', 'N/A')}

--- NEWS SENTIMENT SUMMARY ---
Total Articles Found:  {agg['article_count']}
Average Sentiment:     {agg['avg_sentiment']:.4f} ({agg['sentiment_label'].upper()})
Positive Articles:     {agg['positive_pct']}%
Negative Articles:     {agg['negative_pct']}%
Neutral Articles:      {agg['neutral_pct']}%

--- TOP NEWS ARTICLES ---
{articles_str}

--- CORPORATE ACTIONS ---
{actions_str}

--- MACRO CONTEXT (India, June 2026) ---
Key macro factors to consider:
- RBI monetary policy stance and interest rate trajectory
- FII/DII flow trends in Indian markets
- INR/USD movement impact on this sector
- Government capex and policy support for the sector
- Crude oil prices (affects many Indian industries)
- Global slowdown/growth impact on Indian exports
- GST collections and domestic consumption health

--- QUANTITATIVE NEWS SCORE: {score}/10 ---

Provide your news and macro analysis for {symbol} in the Indian market context.
Be specific about recent catalysts and the macro environment for this sector.
"""
    return prompt


def run(symbol: str, print_output: bool = True) -> dict:
    """Run the News Bot analysis."""
    if print_output:
        print(f"\n{'='*60}")
        print(f"📰  NEWS BOT — {symbol}")
        print('='*60)

    # Get company info
    fund = fetch_fundamentals(symbol)
    company_name = fund.get('company_name', symbol) if fund else symbol

    # Try cache first
    cached = load_news(symbol, max_age_hours=CACHE_TTL_NEWS_HOURS)
    if cached:
        articles = cached
        if VERBOSE_DEBUG:
            print(f"[NEWS] Using {len(articles)} cached articles")
    else:
        # Fetch fresh news
        if VERBOSE_DEBUG:
            print(f"[NEWS] Fetching news for {symbol}")
        rss_articles = fetch_rss_news(symbol, company_name)
        yahoo_articles = fetch_yahoo_news(symbol)

        # Combine and deduplicate
        seen_titles = set()
        articles = []
        for a in rss_articles + yahoo_articles:
            title_key = a['title'][:50].lower()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                articles.append(a)

        if articles:
            save_news(symbol, articles)

    if VERBOSE_DEBUG:
        print(f"[NEWS] Processing {len(articles)} articles")

    # Aggregate sentiment
    agg = aggregate_sentiment(articles)

    # Corporate actions
    actions = fetch_corporate_actions(symbol)

    # Compute score
    quant_score, breakdown = compute_news_score(articles, agg, actions)

    # Build prompt and call LLM
    prompt = build_news_prompt(symbol, fund or {}, articles, agg, actions, quant_score)

    def on_token(t):
        if print_output:
            print(t, end='', flush=True)

    if print_output:
        print()

    llm_response = stream_chat(SYSTEM_PROMPT, prompt, on_token=on_token)

    if print_output:
        print()

    llm_score = extract_score(llm_response, default=quant_score)
    final_score = round(0.6 * llm_score + 0.4 * quant_score, 2)

    result = {
        "bot": "news",
        "symbol": symbol,
        "score": final_score,
        "quant_score": quant_score,
        "llm_score": llm_score,
        "text": llm_response,
        "articles": articles[:5],
        "sentiment_summary": agg,
        "corporate_actions": actions,
        "breakdown": breakdown
    }

    save_analysis(symbol, "news", final_score, llm_response, breakdown)
    return result
