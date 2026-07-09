# ============================================================
# live_apis.py — FREE API Keys & Live Data Sources
# ============================================================
#
# ALL APIs BELOW ARE FREE TIER (no credit card needed):
#
# 1. ALPHA VANTAGE    → https://www.alphavantage.co/support/#api-key
#    Free: 25 requests/day, 5/min
#    Data: OHLCV, fundamentals, news sentiment
#
# 2. POLYGON.IO       → https://polygon.io/dashboard/signup
#    Free: 5 calls/min, delayed 15min data
#    Data: stocks, news, options
#
# 3. FINNHUB          → https://finnhub.io/register
#    Free: 60 calls/min
#    Data: real-time quotes, news, earnings, sentiment
#
# 4. NEWSAPI          → https://newsapi.org/register
#    Free: 100 requests/day (developer plan)
#    Data: global news headlines
#
# 5. GNEWS            → https://gnews.io/
#    Free: 100 requests/day
#    Data: Google News aggregator
#
# 6. MEDIASTACK       → https://mediastack.com/signup/free
#    Free: 500 requests/month
#    Data: live news, 50+ countries
#
# 7. NSE INDIA        → No key needed (public endpoints)
#    Data: NSE live quotes, option chain, FII/DII
#
# 8. BSE INDIA        → No key needed (public endpoints)
#    Data: BSE live quotes, corporate announcements
#
# 9. GOOGLE RSS       → No key needed
#    Data: Google Finance news per stock
#
# 10. REDDIT API      → https://www.reddit.com/prefs/apps
#     Free: OAuth2, 60 requests/min
#     Data: r/IndianStockMarket, r/DalalStreetBets sentiment
#
# HOW TO SET UP:
#   Option A: Set environment variables in .env file
#   Option B: Edit the values directly below
#
# SETUP STEPS:
#   1. Copy this file, fill in your keys
#   2. Run: pip install -r requirements.txt
#   3. Run: python main.py RELIANCE
# ============================================================

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv(Path(__file__).parent / ".env")


# ═══════════════════════════════════════════════════════
# PASTE YOUR FREE API KEYS HERE
# (or set them in .env file)
# ═══════════════════════════════════════════════════════

class APIKeys:

    # ── Alpha Vantage ──────────────────────────────────
    # Get free key: https://www.alphavantage.co/support/#api-key
    # Free tier: 25 req/day, 5 req/min
    ALPHA_VANTAGE = os.getenv("ALPHA_VANTAGE_KEY", "YOUR_ALPHA_VANTAGE_KEY_HERE")

    # ── Finnhub ────────────────────────────────────────
    # Get free key: https://finnhub.io/register
    # Free tier: 60 req/min — BEST FOR REAL-TIME INDIAN STOCKS
    FINNHUB = os.getenv("FINNHUB_KEY", "YOUR_FINNHUB_KEY_HERE")

    # ── Polygon.io ─────────────────────────────────────
    # Get free key: https://polygon.io/dashboard/signup
    # Free tier: 5 req/min, 15-min delayed
    POLYGON = os.getenv("POLYGON_KEY", "YOUR_POLYGON_KEY_HERE")

    # ── NewsAPI ────────────────────────────────────────
    # Get free key: https://newsapi.org/register
    # Free tier: 100 req/day
    NEWSAPI = os.getenv("NEWSAPI_KEY", "YOUR_NEWSAPI_KEY_HERE")

    # ── GNews ──────────────────────────────────────────
    # Get free key: https://gnews.io/
    # Free tier: 100 req/day
    GNEWS = os.getenv("GNEWS_KEY", "YOUR_GNEWS_KEY_HERE")

    # ── MediaStack ─────────────────────────────────────
    # Get free key: https://mediastack.com/signup/free
    # Free tier: 500 req/month
    MEDIASTACK = os.getenv("MEDIASTACK_KEY", "YOUR_MEDIASTACK_KEY_HERE")

    # ── Reddit OAuth2 (optional) ───────────────────────
    # Create app: https://www.reddit.com/prefs/apps
    # Select "script" type
    REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
    REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
    REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "IndianStockBot/1.0")

    # ── Twelve Data (alternative to Alpha Vantage) ─────
    # Get free key: https://twelvedata.com/apikey
    # Free tier: 800 req/day, 8 req/min
    TWELVE_DATA = os.getenv("TWELVE_DATA_KEY", "")

    @classmethod
    def is_configured(cls, key_name: str) -> bool:
        """Check if an API key is actually set (not placeholder)."""
        val = getattr(cls, key_name, "")
        return bool(val) and not val.startswith("YOUR_") and val != ""

    @classmethod
    def status(cls) -> dict:
        """Print status of all API keys."""
        keys = {
            "Alpha Vantage": cls.ALPHA_VANTAGE,
            "Finnhub": cls.FINNHUB,
            "Polygon": cls.POLYGON,
            "NewsAPI": cls.NEWSAPI,
            "GNews": cls.GNEWS,
            "MediaStack": cls.MEDIASTACK,
            "Reddit": cls.REDDIT_CLIENT_ID,
            "Twelve Data": cls.TWELVE_DATA,
        }
        status = {}
        for name, val in keys.items():
            configured = bool(val) and not val.startswith("YOUR_") and val != ""
            status[name] = "✅ configured" if configured else "❌ not set (using fallback)"
        return status


# ── API Endpoints ─────────────────────────────────────────────
class Endpoints:
    # Alpha Vantage
    AV_BASE = "https://www.alphavantage.co/query"

    # Finnhub
    FINNHUB_BASE = "https://finnhub.io/api/v1"
    FINNHUB_QUOTE = f"{FINNHUB_BASE}/quote"
    FINNHUB_NEWS = f"{FINNHUB_BASE}/company-news"
    FINNHUB_SENTIMENT = f"{FINNHUB_BASE}/news-sentiment"
    FINNHUB_INSIDER = f"{FINNHUB_BASE}/stock/insider-transactions"
    FINNHUB_EARNINGS = f"{FINNHUB_BASE}/stock/earnings"
    FINNHUB_METRICS = f"{FINNHUB_BASE}/stock/metric"

    # NewsAPI
    NEWSAPI_BASE = "https://newsapi.org/v2"
    NEWSAPI_EVERYTHING = f"{NEWSAPI_BASE}/everything"
    NEWSAPI_TOP = f"{NEWSAPI_BASE}/top-headlines"

    # GNews
    GNEWS_BASE = "https://gnews.io/api/v4"
    GNEWS_SEARCH = f"{GNEWS_BASE}/search"

    # MediaStack
    MEDIASTACK_BASE = "http://api.mediastack.com/v1"
    MEDIASTACK_NEWS = f"{MEDIASTACK_BASE}/news"

    # NSE India (no key)
    NSE_BASE = "https://www.nseindia.com"
    NSE_QUOTE = "https://www.nseindia.com/api/quote-equity"
    NSE_OPTION_CHAIN = "https://www.nseindia.com/api/option-chain-equities"
    NSE_FII_DII = "https://www.nseindia.com/api/fiidiiTradeReact"
    NSE_CORPORATE_ACTIONS = "https://www.nseindia.com/api/corporates-corporateActions"
    NSE_MARKET_STATUS = "https://www.nseindia.com/api/marketStatus"
    NSE_INDEX = "https://www.nseindia.com/api/allIndices"

    # BSE India (no key)
    BSE_BASE = "https://api.bseindia.com/BseIndiaAPI/api"
    BSE_QUOTE = f"{BSE_BASE}/StockReachGraph/w"

    # Google Finance RSS (no key)
    GOOGLE_FINANCE_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline"

    # Reddit
    REDDIT_BASE = "https://oauth.reddit.com"
    REDDIT_TOKEN = "https://www.reddit.com/api/v1/access_token"
