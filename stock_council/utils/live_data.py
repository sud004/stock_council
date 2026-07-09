# ============================================================
# utils/live_data.py — Unified Live Data from All Free APIs
# ============================================================
#
# PRIORITY ORDER (per data type):
#   Live Quote:     NSE India → Finnhub → Alpha Vantage → Yahoo Finance
#   News:           Finnhub → NewsAPI → GNews → MediaStack → Google RSS → RSS feeds
#   Fundamentals:   Alpha Vantage → Finnhub → Yahoo Finance
#   FII/DII:        NSE India (only source)
#   Option Chain:   NSE India
#   Sentiment:      Reddit → Finnhub → VADER on scraped news
#
# RATE LIMIT MANAGEMENT:
#   Each API class tracks its own call count and timestamps.
#   Automatic backoff when limits approached.
# ============================================================

import requests
import json
import time
import re
import feedparser
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
from threading import Lock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from live_apis import APIKeys, Endpoints
from config import ALLOW_INTERNET, VERBOSE_DEBUG, CACHE_TTL_PRICE_HOURS


# ── Rate limiter ──────────────────────────────────────────────

class RateLimiter:
    """Track API calls to stay within free tier limits."""
    def __init__(self, calls_per_minute: int, calls_per_day: int = 999999):
        self.cpm = calls_per_minute
        self.cpd = calls_per_day
        self._minute_calls = deque()
        self._day_calls = deque()
        self._lock = Lock()

    def wait_if_needed(self):
        with self._lock:
            now = time.time()
            # Clean old entries
            while self._minute_calls and now - self._minute_calls[0] > 60:
                self._minute_calls.popleft()
            while self._day_calls and now - self._day_calls[0] > 86400:
                self._day_calls.popleft()
            # Check day limit
            if len(self._day_calls) >= self.cpd:
                raise Exception("Daily API limit reached")
            # Wait for minute limit
            if len(self._minute_calls) >= self.cpm:
                wait = 61 - (now - self._minute_calls[0])
                if wait > 0:
                    if VERBOSE_DEBUG:
                        print(f"[RATE] Waiting {wait:.1f}s for rate limit...")
                    time.sleep(wait)
            self._minute_calls.append(time.time())
            self._day_calls.append(time.time())


# Rate limiters per API
_av_limiter = RateLimiter(calls_per_minute=5, calls_per_day=25)
_finnhub_limiter = RateLimiter(calls_per_minute=60)
_newsapi_limiter = RateLimiter(calls_per_minute=10, calls_per_day=100)
_gnews_limiter = RateLimiter(calls_per_minute=10, calls_per_day=100)
_mediastack_limiter = RateLimiter(calls_per_minute=5, calls_per_day=17)  # 500/month ≈ 17/day


# ── Shared HTTP session ───────────────────────────────────────

def _get_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-IN,en;q=0.9',
    })
    return s


_SESSION = _get_session()


def _get(url: str, params: dict = None, headers: dict = None,
         timeout: int = 10, retries: int = 3) -> dict | None:
    """Safe GET with retries and error handling."""
    for attempt in range(retries):
        try:
            r = _SESSION.get(url, params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            if r.status_code == 429:
                wait = 60 * (attempt + 1)
                print(f"[HTTP] Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            elif r.status_code in [401, 403]:
                print(f"[HTTP] Auth error for {url} — check API key")
                return None
            else:
                if VERBOSE_DEBUG:
                    print(f"[HTTP] Error {r.status_code} for {url}: {e}")
                time.sleep(2 ** attempt)
        except requests.exceptions.ConnectionError:
            if VERBOSE_DEBUG:
                print(f"[HTTP] Connection error (attempt {attempt+1}): {url}")
            time.sleep(2 ** attempt)
        except Exception as e:
            if VERBOSE_DEBUG:
                print(f"[HTTP] Unexpected error: {e}")
            time.sleep(1)
    return None


# ═══════════════════════════════════════════════════════════
# 1. NSE INDIA — Live Quotes (No API key needed)
# ═══════════════════════════════════════════════════════════

class NSELive:
    """
    Fetch live data from NSE India's public API.
    No API key required. Uses session cookie.
    """
    BASE = "https://www.nseindia.com"
    _session = None
    _last_cookie_time = 0

    @classmethod
    def _get_session(cls) -> requests.Session:
        """Get NSE session with valid cookies (required by NSE)."""
        if cls._session and (time.time() - cls._last_cookie_time) < 300:
            return cls._session
        s = requests.Session()
        s.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://www.nseindia.com/',
            'Connection': 'keep-alive',
        })
        try:
            # Must hit homepage first to get cookies
            s.get(cls.BASE, timeout=10)
            time.sleep(1)
            cls._session = s
            cls._last_cookie_time = time.time()
        except Exception as e:
            if VERBOSE_DEBUG:
                print(f"[NSE] Session init error: {e}")
        return s

    @classmethod
    def get_quote(cls, symbol: str) -> dict | None:
        """
        Get live NSE quote for a symbol.
        symbol: e.g. "RELIANCE" (no .NS suffix)
        Returns live price, volumes, circuit limits, etc.
        """
        if not ALLOW_INTERNET:
            return None
        sym = symbol.replace('.NS', '').replace('.BO', '').upper()
        try:
            s = cls._get_session()
            url = f"{cls.BASE}/api/quote-equity?symbol={sym}"
            r = s.get(url, timeout=10)
            if r.status_code == 401:
                cls._last_cookie_time = 0  # force re-init
                s = cls._get_session()
                r = s.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                price_info = data.get('priceInfo', {})
                meta = data.get('metadata', {})
                return {
                    'symbol': sym,
                    'last_price': price_info.get('lastPrice'),
                    'open': price_info.get('open'),
                    'high': price_info.get('intraDayHighLow', {}).get('max'),
                    'low': price_info.get('intraDayHighLow', {}).get('min'),
                    'prev_close': price_info.get('previousClose'),
                    'change': price_info.get('change'),
                    'pct_change': price_info.get('pChange'),
                    'volume': data.get('marketDeptOrderBook', {}).get('tradeInfo', {}).get('totalTradedVolume'),
                    'total_buy_qty': data.get('marketDeptOrderBook', {}).get('totalBuyQuantity'),
                    'total_sell_qty': data.get('marketDeptOrderBook', {}).get('totalSellQuantity'),
                    'upper_circuit': price_info.get('upperCP'),
                    'lower_circuit': price_info.get('lowerCP'),
                    'year_high': price_info.get('weekHighLow', {}).get('max'),
                    'year_low': price_info.get('weekHighLow', {}).get('min'),
                    'market_cap': meta.get('pdMarketCap'),
                    'face_value': meta.get('pdFaceValue'),
                    'sector': meta.get('pdSectorPe'),
                    'timestamp': datetime.now().isoformat(),
                    'source': 'NSE'
                }
        except Exception as e:
            if VERBOSE_DEBUG:
                print(f"[NSE] Quote error for {sym}: {e}")
        return None

    @classmethod
    def get_fii_dii(cls) -> dict | None:
        """
        Get today's FII/DII buying and selling data.
        Critical indicator for Indian market direction.
        """
        if not ALLOW_INTERNET:
            return None
        try:
            s = cls._get_session()
            r = s.get(Endpoints.NSE_FII_DII, timeout=10)
            if r.status_code == 200:
                data = r.json()
                # Parse FII and DII data
                fii = {}
                dii = {}
                for item in data:
                    category = item.get('category', '').upper()
                    if 'FII' in category or 'FPI' in category:
                        fii = {
                            'buy_value': item.get('buyValue'),
                            'sell_value': item.get('sellValue'),
                            'net_value': item.get('netValue'),
                            'date': item.get('date')
                        }
                    elif 'DII' in category:
                        dii = {
                            'buy_value': item.get('buyValue'),
                            'sell_value': item.get('sellValue'),
                            'net_value': item.get('netValue'),
                            'date': item.get('date')
                        }
                return {'fii': fii, 'dii': dii, 'timestamp': datetime.now().isoformat()}
        except Exception as e:
            if VERBOSE_DEBUG:
                print(f"[NSE] FII/DII error: {e}")
        return None

    @classmethod
    def get_option_chain(cls, symbol: str) -> dict | None:
        """
        Get NSE option chain data for PCR (Put-Call Ratio) calculation.
        PCR > 1.2 = bullish  |  PCR < 0.7 = bearish
        """
        if not ALLOW_INTERNET:
            return None
        sym = symbol.replace('.NS', '').replace('.BO', '').upper()
        try:
            s = cls._get_session()
            url = f"{cls.BASE}/api/option-chain-equities?symbol={sym}"
            r = s.get(url, timeout=15)
            if r.status_code == 200:
                data = r.json()
                filtered = data.get('filtered', {})
                ce_oi = filtered.get('CE', {}).get('totOI', 0)
                pe_oi = filtered.get('PE', {}).get('totOI', 0)
                pcr = round(pe_oi / ce_oi, 3) if ce_oi and ce_oi > 0 else None
                return {
                    'symbol': sym,
                    'pcr': pcr,
                    'call_oi': ce_oi,
                    'put_oi': pe_oi,
                    'pcr_signal': (
                        'BULLISH' if pcr and pcr > 1.2
                        else 'BEARISH' if pcr and pcr < 0.7
                        else 'NEUTRAL'
                    ),
                    'expiry': filtered.get('expiryDate'),
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            if VERBOSE_DEBUG:
                print(f"[NSE] Option chain error for {sym}: {e}")
        return None

    @classmethod
    def get_all_indices(cls) -> list | None:
        """Get live Nifty, Sensex, Bank Nifty values."""
        if not ALLOW_INTERNET:
            return None
        try:
            s = cls._get_session()
            r = s.get(Endpoints.NSE_INDEX, timeout=10)
            if r.status_code == 200:
                data = r.json()
                indices = []
                for item in data.get('data', []):
                    name = item.get('index', '')
                    if any(x in name.upper() for x in ['NIFTY 50', 'NIFTY BANK', 'INDIA VIX', 'NIFTY IT', 'NIFTY MIDCAP']):
                        indices.append({
                            'name': name,
                            'last': item.get('last'),
                            'change': item.get('change'),
                            'pct_change': item.get('percentChange'),
                            'year_high': item.get('yearHigh'),
                            'year_low': item.get('yearLow'),
                        })
                return indices
        except Exception as e:
            if VERBOSE_DEBUG:
                print(f"[NSE] Indices error: {e}")
        return None

    @classmethod
    def get_corporate_actions(cls, symbol: str) -> list:
        """Get upcoming dividends, splits, bonus for a stock."""
        if not ALLOW_INTERNET:
            return []
        sym = symbol.replace('.NS', '').replace('.BO', '').upper()
        try:
            s = cls._get_session()
            url = f"{cls.BASE}/api/corporates-corporateActions?index=equities&symbol={sym}"
            r = s.get(url, timeout=10)
            if r.status_code == 200:
                return r.json() or []
        except Exception as e:
            if VERBOSE_DEBUG:
                print(f"[NSE] Corporate actions error: {e}")
        return []

    @classmethod
    def get_market_status(cls) -> dict | None:
        """Check if NSE market is open right now."""
        if not ALLOW_INTERNET:
            return None
        try:
            s = cls._get_session()
            r = s.get(Endpoints.NSE_MARKET_STATUS, timeout=5)
            if r.status_code == 200:
                data = r.json()
                mkt = data.get('marketState', [{}])[0] if data.get('marketState') else {}
                return {
                    'is_open': mkt.get('marketStatus', '').upper() == 'OPEN',
                    'status': mkt.get('marketStatus'),
                    'trade_date': mkt.get('tradeDate'),
                    'index': mkt.get('index'),
                }
        except Exception:
            pass
        return {'is_open': False, 'status': 'unknown'}


# ═══════════════════════════════════════════════════════════
# 2. FINNHUB — Real-time quotes + news + sentiment
# ═══════════════════════════════════════════════════════════

class FinnhubData:
    """
    Finnhub free tier: 60 requests/min
    Best for: real-time quotes, news, earnings, insider trades
    Note: Indian stocks use BSE/NSE format: "NSE:RELIANCE"
    """

    @staticmethod
    def _sym(symbol: str) -> str:
        """Convert RELIANCE.NS → NSE:RELIANCE"""
        s = symbol.replace('.NS', '').replace('.BO', '').upper()
        suffix = 'BSE' if '.BO' in symbol else 'NSE'
        return f"{suffix}:{s}"

    @classmethod
    def get_quote(cls, symbol: str) -> dict | None:
        if not ALLOW_INTERNET or not APIKeys.is_configured('FINNHUB'):
            return None
        try:
            _finnhub_limiter.wait_if_needed()
            fh_sym = cls._sym(symbol)
            data = _get(Endpoints.FINNHUB_QUOTE,
                        params={'symbol': fh_sym, 'token': APIKeys.FINNHUB})
            if data and data.get('c'):
                return {
                    'symbol': symbol,
                    'current': data.get('c'),
                    'high': data.get('h'),
                    'low': data.get('l'),
                    'open': data.get('o'),
                    'prev_close': data.get('pc'),
                    'change': data.get('d'),
                    'pct_change': data.get('dp'),
                    'timestamp': data.get('t'),
                    'source': 'Finnhub'
                }
        except Exception as e:
            if VERBOSE_DEBUG:
                print(f"[FINNHUB] Quote error: {e}")
        return None

    @classmethod
    def get_news(cls, symbol: str, days_back: int = 7) -> list:
        """Fetch company news from Finnhub."""
        if not ALLOW_INTERNET or not APIKeys.is_configured('FINNHUB'):
            return []
        try:
            _finnhub_limiter.wait_if_needed()
            fh_sym = cls._sym(symbol)
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            to_date = datetime.now().strftime('%Y-%m-%d')
            data = _get(Endpoints.FINNHUB_NEWS, params={
                'symbol': fh_sym,
                'from': from_date,
                'to': to_date,
                'token': APIKeys.FINNHUB
            })
            if data and isinstance(data, list):
                articles = []
                for item in data[:15]:
                    articles.append({
                        'title': item.get('headline', ''),
                        'summary': item.get('summary', '')[:400],
                        'source': item.get('source', 'Finnhub'),
                        'url': item.get('url', ''),
                        'published': datetime.fromtimestamp(item.get('datetime', time.time())).isoformat(),
                        'sentiment': 0.0,  # will be scored separately
                        'relevance': 5,
                        'symbol': symbol
                    })
                return articles
        except Exception as e:
            if VERBOSE_DEBUG:
                print(f"[FINNHUB] News error: {e}")
        return []

    @classmethod
    def get_news_sentiment(cls, symbol: str) -> dict | None:
        """Get Finnhub's pre-computed news sentiment score."""
        if not ALLOW_INTERNET or not APIKeys.is_configured('FINNHUB'):
            return None
        try:
            _finnhub_limiter.wait_if_needed()
            fh_sym = cls._sym(symbol)
            data = _get(Endpoints.FINNHUB_SENTIMENT,
                        params={'symbol': fh_sym, 'token': APIKeys.FINNHUB})
            if data:
                return {
                    'buzz_articles': data.get('buzz', {}).get('articlesInLastWeek'),
                    'buzz_weekly_avg': data.get('buzz', {}).get('weeklyAverage'),
                    'buzz_score': data.get('buzz', {}).get('buzz'),
                    'company_score': data.get('companyNewsScore'),
                    'sector_avg_bullish': data.get('sectorAverageBullishPercent'),
                    'sector_avg_score': data.get('sectorAverageNewsScore'),
                    'sentiment_bearish': data.get('sentiment', {}).get('bearishPercent'),
                    'sentiment_bullish': data.get('sentiment', {}).get('bullishPercent'),
                    'source': 'Finnhub'
                }
        except Exception as e:
            if VERBOSE_DEBUG:
                print(f"[FINNHUB] Sentiment error: {e}")
        return None

    @classmethod
    def get_basic_financials(cls, symbol: str) -> dict | None:
        """Get key financial metrics from Finnhub."""
        if not ALLOW_INTERNET or not APIKeys.is_configured('FINNHUB'):
            return None
        try:
            _finnhub_limiter.wait_if_needed()
            fh_sym = cls._sym(symbol)
            data = _get(Endpoints.FINNHUB_METRICS, params={
                'symbol': fh_sym,
                'metric': 'all',
                'token': APIKeys.FINNHUB
            })
            if data and data.get('metric'):
                m = data['metric']
                return {
                    'pe_ttm': m.get('peNormalizedAnnual'),
                    'pb': m.get('pbQuarterly'),
                    'ps_ttm': m.get('psTTM'),
                    'roe_ttm': m.get('roeTTM'),
                    'net_margin_ttm': m.get('netProfitMarginTTM'),
                    'revenue_growth_5y': m.get('revenueGrowth5Y'),
                    'eps_growth_5y': m.get('epsGrowth5Y'),
                    'dividend_yield_ind': m.get('dividendYieldIndicatedAnnual'),
                    'beta': m.get('beta'),
                    '52w_high': m.get('52WeekHigh'),
                    '52w_low': m.get('52WeekLow'),
                    'source': 'Finnhub'
                }
        except Exception as e:
            if VERBOSE_DEBUG:
                print(f"[FINNHUB] Metrics error: {e}")
        return None

    @classmethod
    def get_earnings(cls, symbol: str) -> list:
        """Get historical earnings surprises."""
        if not ALLOW_INTERNET or not APIKeys.is_configured('FINNHUB'):
            return []
        try:
            _finnhub_limiter.wait_if_needed()
            fh_sym = cls._sym(symbol)
            data = _get(Endpoints.FINNHUB_EARNINGS, params={
                'symbol': fh_sym,
                'token': APIKeys.FINNHUB
            })
            if data and isinstance(data, list):
                return [{
                    'date': item.get('period'),
                    'eps_actual': item.get('actual'),
                    'eps_estimate': item.get('estimate'),
                    'surprise': item.get('surprise'),
                    'surprise_pct': item.get('surprisePercent'),
                } for item in data[:4]]
        except Exception as e:
            if VERBOSE_DEBUG:
                print(f"[FINNHUB] Earnings error: {e}")
        return []


# ═══════════════════════════════════════════════════════════
# 3. ALPHA VANTAGE — Fundamentals + Historical Data
# ═══════════════════════════════════════════════════════════

class AlphaVantageData:
    """
    Alpha Vantage free: 25 requests/day, 5/min
    Best for: fundamentals, income statement, balance sheet
    Note: Indian stocks: symbol = "RELIANCE.BSE"
    """

    @staticmethod
    def _sym(symbol: str) -> str:
        """Convert RELIANCE.NS → RELIANCE.BSE (AV format for Indian stocks)"""
        s = symbol.replace('.NS', '').replace('.BO', '').upper()
        return f"{s}.BSE"

    @classmethod
    def get_overview(cls, symbol: str) -> dict | None:
        """Company overview with all fundamental metrics."""
        if not ALLOW_INTERNET or not APIKeys.is_configured('ALPHA_VANTAGE'):
            return None
        try:
            _av_limiter.wait_if_needed()
            data = _get(Endpoints.AV_BASE, params={
                'function': 'OVERVIEW',
                'symbol': cls._sym(symbol),
                'apikey': APIKeys.ALPHA_VANTAGE
            })
            if data and data.get('Symbol'):
                return {
                    'company_name': data.get('Name'),
                    'sector': data.get('Sector'),
                    'industry': data.get('Industry'),
                    'market_cap': _safe_float(data.get('MarketCapitalization')),
                    'pe_ratio': _safe_float(data.get('TrailingPE')),
                    'forward_pe': _safe_float(data.get('ForwardPE')),
                    'pb_ratio': _safe_float(data.get('PriceToBookRatio')),
                    'peg_ratio': _safe_float(data.get('PEGRatio')),
                    'ev_ebitda': _safe_float(data.get('EVToEBITDA')),
                    'revenue_ttm': _safe_float(data.get('RevenueTTM')),
                    'gross_profit_ttm': _safe_float(data.get('GrossProfitTTM')),
                    'ebitda': _safe_float(data.get('EBITDA')),
                    'eps': _safe_float(data.get('EPS')),
                    'roe': _safe_float(data.get('ReturnOnEquityTTM')),
                    'roa': _safe_float(data.get('ReturnOnAssetsTTM')),
                    'revenue_growth_yoy': _safe_float(data.get('QuarterlyRevenueGrowthYOY')),
                    'eps_growth_yoy': _safe_float(data.get('QuarterlyEarningsGrowthYOY')),
                    'operating_margin': _safe_float(data.get('OperatingMarginTTM')),
                    'net_margin': _safe_float(data.get('ProfitMargin')),
                    'debt_equity': _safe_float(data.get('DebtToEquityRatio')),
                    'current_ratio': _safe_float(data.get('CurrentRatio')),
                    'book_value': _safe_float(data.get('BookValue')),
                    'dividend_yield': _safe_float(data.get('DividendYield')),
                    'dividend_per_share': _safe_float(data.get('DividendPerShare')),
                    '52w_high': _safe_float(data.get('52WeekHigh')),
                    '52w_low': _safe_float(data.get('52WeekLow')),
                    'beta': _safe_float(data.get('Beta')),
                    'shares_outstanding': _safe_float(data.get('SharesOutstanding')),
                    'analyst_target': _safe_float(data.get('AnalystTargetPrice')),
                    'source': 'AlphaVantage'
                }
        except Exception as e:
            if VERBOSE_DEBUG:
                print(f"[AV] Overview error: {e}")
        return None

    @classmethod
    def get_income_statement(cls, symbol: str) -> list:
        """Get quarterly income statements (last 4 quarters)."""
        if not ALLOW_INTERNET or not APIKeys.is_configured('ALPHA_VANTAGE'):
            return []
        try:
            _av_limiter.wait_if_needed()
            data = _get(Endpoints.AV_BASE, params={
                'function': 'INCOME_STATEMENT',
                'symbol': cls._sym(symbol),
                'apikey': APIKeys.ALPHA_VANTAGE
            })
            if data and data.get('quarterlyReports'):
                result = []
                for q in data['quarterlyReports'][:4]:
                    result.append({
                        'period': q.get('fiscalDateEnding'),
                        'revenue': _safe_float(q.get('totalRevenue')),
                        'gross_profit': _safe_float(q.get('grossProfit')),
                        'ebitda': _safe_float(q.get('ebitda')),
                        'net_income': _safe_float(q.get('netIncome')),
                        'eps': _safe_float(q.get('reportedEPS')),
                        'operating_income': _safe_float(q.get('operatingIncome')),
                    })
                return result
        except Exception as e:
            if VERBOSE_DEBUG:
                print(f"[AV] Income statement error: {e}")
        return []

    @classmethod
    def get_news_sentiment(cls, symbol: str, topics: str = "financial_markets,earnings") -> dict | None:
        """Alpha Vantage news sentiment API."""
        if not ALLOW_INTERNET or not APIKeys.is_configured('ALPHA_VANTAGE'):
            return None
        try:
            _av_limiter.wait_if_needed()
            sym = symbol.replace('.NS', '').replace('.BO', '').upper()
            data = _get(Endpoints.AV_BASE, params={
                'function': 'NEWS_SENTIMENT',
                'tickers': sym,
                'topics': topics,
                'time_from': (datetime.now() - timedelta(days=7)).strftime('%Y%m%dT0000'),
                'sort': 'RELEVANCE',
                'limit': 20,
                'apikey': APIKeys.ALPHA_VANTAGE
            })
            if data and data.get('feed'):
                articles = []
                overall_sentiments = []
                for item in data['feed'][:10]:
                    ticker_sentiments = item.get('ticker_sentiment', [])
                    ticker_score = 0.0
                    for ts in ticker_sentiments:
                        if sym in ts.get('ticker', ''):
                            ticker_score = float(ts.get('ticker_sentiment_score', 0))
                            break
                    articles.append({
                        'title': item.get('title', ''),
                        'summary': item.get('summary', '')[:300],
                        'source': item.get('source', 'AlphaVantage'),
                        'url': item.get('url', ''),
                        'published': item.get('time_published', ''),
                        'sentiment': ticker_score,
                        'relevance': 5,
                        'symbol': symbol
                    })
                    overall_sentiments.append(ticker_score)
                avg = sum(overall_sentiments) / len(overall_sentiments) if overall_sentiments else 0
                return {
                    'articles': articles,
                    'average_sentiment': round(avg, 4),
                    'article_count': len(articles),
                    'source': 'AlphaVantage'
                }
        except Exception as e:
            if VERBOSE_DEBUG:
                print(f"[AV] News sentiment error: {e}")
        return None


# ═══════════════════════════════════════════════════════════
# 4. NEWSAPI — Global news search
# ═══════════════════════════════════════════════════════════

class NewsAPIData:
    """
    NewsAPI free: 100 requests/day
    Best for: broad news search about a company
    """

    @classmethod
    def search_news(cls, symbol: str, company_name: str, days_back: int = 7) -> list:
        if not ALLOW_INTERNET or not APIKeys.is_configured('NEWSAPI'):
            return []
        try:
            _newsapi_limiter.wait_if_needed()
            short_sym = symbol.replace('.NS', '').replace('.BO', '').upper()
            # Build query: company name OR stock symbol
            query = f'"{company_name or short_sym}" OR "{short_sym}"'
            if len(query) > 100:
                query = f'"{short_sym}"'

            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            data = _get(Endpoints.NEWSAPI_EVERYTHING, params={
                'q': query,
                'from': from_date,
                'sortBy': 'relevancy',
                'language': 'en',
                'pageSize': 15,
                'apiKey': APIKeys.NEWSAPI
            })

            if data and data.get('articles'):
                from bots.news_bot import vader_sentiment
                vader = None
                try:
                    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                    vader = SentimentIntensityAnalyzer()
                except Exception:
                    pass
                articles = []
                for a in data['articles']:
                    title = a.get('title', '') or ''
                    desc = a.get('description', '') or ''
                    content = a.get('content', '') or ''
                    text = title + ' ' + desc
                    sent = vader_sentiment(text, vader).get('compound', 0.0) if vader else 0.0
                    articles.append({
                        'title': title,
                        'summary': (desc or content)[:400],
                        'source': a.get('source', {}).get('name', 'NewsAPI'),
                        'url': a.get('url', ''),
                        'published': a.get('publishedAt', ''),
                        'sentiment': round(sent, 4),
                        'relevance': 4,
                        'symbol': symbol
                    })
                return articles
        except Exception as e:
            if VERBOSE_DEBUG:
                print(f"[NEWSAPI] Search error: {e}")
        return []


# ═══════════════════════════════════════════════════════════
# 5. GNEWS — Google News
# ═══════════════════════════════════════════════════════════

class GNewsData:
    """
    GNews free: 100 requests/day
    Best for: latest Indian news
    """

    @classmethod
    def search_news(cls, symbol: str, company_name: str) -> list:
        if not ALLOW_INTERNET or not APIKeys.is_configured('GNEWS'):
            return []
        try:
            _gnews_limiter.wait_if_needed()
            short_sym = symbol.replace('.NS', '').replace('.BO', '').upper()
            query = f"{company_name or short_sym} stock NSE BSE"
            data = _get(Endpoints.GNEWS_SEARCH, params={
                'q': query,
                'lang': 'en',
                'country': 'in',
                'max': 10,
                'apikey': APIKeys.GNEWS
            })
            if data and data.get('articles'):
                from bots.news_bot import vader_sentiment
                articles = []
                for a in data['articles']:
                    title = a.get('title', '') or ''
                    desc = a.get('description', '') or ''
                    sent = vader_sentiment(title + ' ' + desc).get('compound', 0.0)
                    articles.append({
                        'title': title,
                        'summary': desc[:400],
                        'source': a.get('source', {}).get('name', 'GNews'),
                        'url': a.get('url', ''),
                        'published': a.get('publishedAt', ''),
                        'sentiment': round(sent, 4),
                        'relevance': 4,
                        'symbol': symbol
                    })
                return articles
        except Exception as e:
            if VERBOSE_DEBUG:
                print(f"[GNEWS] Search error: {e}")
        return []


# ═══════════════════════════════════════════════════════════
# 6. GOOGLE FINANCE RSS — No key needed
# ═══════════════════════════════════════════════════════════

class GoogleFinanceRSS:
    """
    Google Finance RSS — completely free, no API key.
    Provides real-time Google News for stocks.
    """

    @classmethod
    def get_news(cls, symbol: str, company_name: str) -> list:
        if not ALLOW_INTERNET:
            return []
        try:
            short_sym = symbol.replace('.NS', '').replace('.BO', '').upper()
            query = f"{company_name or short_sym} stock NSE"
            # Google News RSS
            url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"
            feed = feedparser.parse(url)
            from bots.news_bot import vader_sentiment
            articles = []
            for entry in feed.entries[:15]:
                title = entry.get('title', '')
                summary = entry.get('summary', '') or ''
                # Clean HTML from summary
                summary = re.sub(r'<[^>]+>', '', summary)
                sent = vader_sentiment(title + ' ' + summary).get('compound', 0.0)
                articles.append({
                    'title': title,
                    'summary': summary[:400],
                    'source': 'Google News',
                    'url': entry.get('link', ''),
                    'published': entry.get('published', ''),
                    'sentiment': round(sent, 4),
                    'relevance': 3,
                    'symbol': symbol
                })
            return articles
        except Exception as e:
            if VERBOSE_DEBUG:
                print(f"[GOOGLE RSS] Error: {e}")
        return []


# ═══════════════════════════════════════════════════════════
# 7. REDDIT — Indian Stock Market Sentiment
# ═══════════════════════════════════════════════════════════

class RedditSentiment:
    """
    Reddit API — Free with OAuth
    Subreddits: r/IndianStockMarket, r/DalalStreetBets, r/IndiaInvestments
    """
    _access_token = None
    _token_expiry = 0

    @classmethod
    def _get_token(cls) -> str | None:
        if not APIKeys.REDDIT_CLIENT_ID or not APIKeys.REDDIT_CLIENT_SECRET:
            return None
        if cls._access_token and time.time() < cls._token_expiry:
            return cls._access_token
        try:
            r = requests.post(
                Endpoints.REDDIT_TOKEN,
                data={'grant_type': 'client_credentials'},
                auth=(APIKeys.REDDIT_CLIENT_ID, APIKeys.REDDIT_CLIENT_SECRET),
                headers={'User-Agent': APIKeys.REDDIT_USER_AGENT},
                timeout=10
            )
            if r.status_code == 200:
                data = r.json()
                cls._access_token = data.get('access_token')
                cls._token_expiry = time.time() + data.get('expires_in', 3600) - 60
                return cls._access_token
        except Exception as e:
            if VERBOSE_DEBUG:
                print(f"[REDDIT] Token error: {e}")
        return None

    @classmethod
    def search_mentions(cls, symbol: str, company_name: str) -> dict:
        """Search Reddit for stock mentions and sentiment."""
        if not ALLOW_INTERNET:
            return {'mentions': 0, 'sentiment': 0.0, 'posts': []}

        short_sym = symbol.replace('.NS', '').replace('.BO', '').upper()
        subreddits = ['IndianStockMarket', 'DalalStreetBets', 'IndiaInvestments', 'IndianStreetBets']
        all_posts = []
        token = cls._get_token()
        headers = {'User-Agent': APIKeys.REDDIT_USER_AGENT}
        if token:
            headers['Authorization'] = f'Bearer {token}'

        for subreddit in subreddits:
            try:
                # Search without auth (public)
                url = f"https://www.reddit.com/r/{subreddit}/search.json"
                r = requests.get(url, params={
                    'q': short_sym,
                    'sort': 'new',
                    'limit': 10,
                    't': 'week',
                    'restrict_sr': True
                }, headers=headers, timeout=10)

                if r.status_code == 200:
                    data = r.json()
                    posts = data.get('data', {}).get('children', [])
                    for post in posts:
                        p = post.get('data', {})
                        title = p.get('title', '')
                        body = p.get('selftext', '')[:200]
                        # Check relevance
                        if short_sym.lower() not in (title + body).lower() and \
                           (company_name or '').lower()[:6] not in (title + body).lower():
                            continue
                        all_posts.append({
                            'title': title,
                            'body': body,
                            'score': p.get('score', 0),
                            'comments': p.get('num_comments', 0),
                            'subreddit': subreddit,
                            'created': p.get('created_utc', 0),
                            'url': f"https://reddit.com{p.get('permalink', '')}"
                        })
                time.sleep(1)  # be polite to Reddit
            except Exception as e:
                if VERBOSE_DEBUG:
                    print(f"[REDDIT] {subreddit} error: {e}")

        if not all_posts:
            return {'mentions': 0, 'sentiment': 0.0, 'posts': [], 'source': 'Reddit'}

        # Score sentiment on posts
        from bots.news_bot import vader_sentiment
        sentiments = []
        for post in all_posts:
            text = post['title'] + ' ' + post['body']
            sent = vader_sentiment(text).get('compound', 0.0)
            post['sentiment'] = round(sent, 4)
            # Weight by upvotes (more upvoted = more representative)
            weight = max(1, min(post['score'], 100))  # cap at 100
            sentiments.extend([sent] * weight)

        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
        bullish = sum(1 for s in sentiments if s > 0.05) / len(sentiments) * 100 if sentiments else 0
        bearish = sum(1 for s in sentiments if s < -0.05) / len(sentiments) * 100 if sentiments else 0

        return {
            'mentions': len(all_posts),
            'sentiment': round(avg_sentiment, 4),
            'bullish_pct': round(bullish, 1),
            'bearish_pct': round(bearish, 1),
            'posts': sorted(all_posts, key=lambda x: x['score'], reverse=True)[:5],
            'source': 'Reddit'
        }


# ═══════════════════════════════════════════════════════════
# 8. UNIFIED LIVE QUOTE FETCHER
# ═══════════════════════════════════════════════════════════

def get_live_quote(symbol: str) -> dict:
    """
    Get the best available live quote.
    Priority: NSE India → Finnhub → Alpha Vantage
    """
    # 1. Try NSE (most accurate for Indian stocks, no key)
    quote = NSELive.get_quote(symbol)
    if quote and quote.get('last_price'):
        if VERBOSE_DEBUG:
            print(f"[QUOTE] Got live NSE quote: ₹{quote['last_price']}")
        return quote

    # 2. Try Finnhub
    quote = FinnhubData.get_quote(symbol)
    if quote and quote.get('current'):
        if VERBOSE_DEBUG:
            print(f"[QUOTE] Got Finnhub quote: {quote['current']}")
        return quote

    # 3. Fallback: Yahoo Finance
    try:
        import yfinance as yf
        from utils.market_data import resolve_symbol
        ticker = yf.Ticker(resolve_symbol(symbol))
        info = ticker.fast_info
        price = getattr(info, 'last_price', None) or getattr(info, 'regular_market_price', None)
        if price:
            return {
                'symbol': symbol,
                'last_price': price,
                'source': 'Yahoo Finance (fallback)'
            }
    except Exception:
        pass

    return {'symbol': symbol, 'last_price': None, 'source': 'none — all sources failed'}


def get_all_news(symbol: str, company_name: str = "",
                  max_age_days: int = 3) -> list:
    """
    Aggregate news from ALL available sources.
    Returns deduplicated, date-filtered, sorted article list.

    FIX: Added max_age_days filter (default 3 days).
    Google News RSS was returning articles 2-3 weeks old which
    caused the news bot to score stale negative sentiment as
    today's signal — corrupting the council verdict.

    Articles older than max_age_days are excluded unless no
    fresh articles exist for that stock (fallback to 7 days).
    """
    all_articles = []
    seen         = set()
    cutoff_3d    = datetime.now() - timedelta(days=max_age_days)
    cutoff_7d    = datetime.now() - timedelta(days=7)

    def _parse_date(article: dict):
        """Parse published date from article dict."""
        raw = article.get('published') or article.get('date') or ''
        if not raw:
            return None
        # Try common formats
        for fmt in [
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M:%S',
            '%a, %d %b %Y %H:%M:%S %Z',
            '%a, %d %b %Y %H:%M:%S %z',
            '%Y-%m-%d',
        ]:
            try:
                return datetime.strptime(raw[:25], fmt)
            except ValueError:
                continue
        # Try ISO format
        try:
            return datetime.fromisoformat(raw[:19])
        except Exception:
            return None

    def _is_fresh(article: dict, cutoff: datetime) -> bool:
        """Return True if article is newer than cutoff."""
        pub = _parse_date(article)
        if pub is None:
            return True   # no date = assume fresh (don't discard)
        # Make both naive for comparison
        if pub.tzinfo is not None:
            pub = pub.replace(tzinfo=None)
        return pub >= cutoff

    # Priority sources
    sources = [
        ("Finnhub",      lambda: FinnhubData.get_news(symbol)),
        ("AlphaVantage", lambda: AlphaVantageData.get_news_sentiment(symbol)
                                 and AlphaVantageData.get_news_sentiment(symbol)
                                 .get('articles', []) or []),
        ("NewsAPI",      lambda: NewsAPIData.search_news(symbol, company_name)),
        ("GNews",        lambda: GNewsData.search_news(symbol, company_name)),
        ("Google RSS",   lambda: GoogleFinanceRSS.get_news(symbol, company_name)),
    ]

    for src_name, fetcher in sources:
        try:
            articles = fetcher() or []
            for a in articles:
                key = a.get('title', '')[:50].lower().strip()
                if key and key not in seen:
                    seen.add(key)
                    all_articles.append(a)
            if VERBOSE_DEBUG:
                print(f"[NEWS] {src_name}: {len(articles)} articles")
        except Exception as e:
            if VERBOSE_DEBUG:
                print(f"[NEWS] {src_name} failed: {e}")

    # Also add RSS feeds from news_bot
    try:
        from bots.news_bot import fetch_rss_news
        rss = fetch_rss_news(symbol, company_name)
        for a in rss:
            key = a.get('title', '')[:50].lower().strip()
            if key and key not in seen:
                seen.add(key)
                all_articles.append(a)
    except Exception as e:
        if VERBOSE_DEBUG:
            print(f"[NEWS] RSS feeds failed: {e}")

    # ── Date filter ───────────────────────────────────────────
    # Try 3-day fresh articles first
    fresh_3d = [a for a in all_articles if _is_fresh(a, cutoff_3d)]

    if fresh_3d:
        # Good — we have recent news
        filtered = fresh_3d
        if VERBOSE_DEBUG:
            stale = len(all_articles) - len(fresh_3d)
            print(f"[NEWS] {symbol}: {len(fresh_3d)} fresh articles "
                  f"(removed {stale} older than {max_age_days} days)")
    else:
        # No fresh news — fall back to 7-day window
        fresh_7d = [a for a in all_articles if _is_fresh(a, cutoff_7d)]
        if fresh_7d:
            filtered = fresh_7d
            if VERBOSE_DEBUG:
                print(f"[NEWS] {symbol}: No {max_age_days}d news — "
                      f"using {len(fresh_7d)} articles from last 7 days")
        else:
            # Still nothing — use all articles (better than nothing)
            filtered = all_articles
            if VERBOSE_DEBUG:
                print(f"[NEWS] {symbol}: No recent news — "
                      f"using all {len(filtered)} articles (may be stale)")

    # Sort by date descending (newest first) then relevance
    def _sort_key(a):
        pub = _parse_date(a)
        pub_str = pub.isoformat() if pub else ''
        return (pub_str, a.get('relevance', 0))

    filtered.sort(key=_sort_key, reverse=True)
    return filtered[:20]   # cap at 20 (was 30 — fresh is better than volume)


# ── Helper ────────────────────────────────────────────────────

def _safe_float(val) -> float | None:
    try:
        f = float(val)
        return None if (f != f) else f  # NaN check
    except (TypeError, ValueError):
        return None