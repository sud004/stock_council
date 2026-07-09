# ============================================================
# config.py — Central configuration for Stock Council
# ============================================================
# TOKEN FIX:
#   OLD: LLM_MAX_TOKENS = 1500 applied to EVERY call regardless of task.
#        Model was generating 1500 tokens even when asked for 80 words.
#        29 calls × 1500 = 43,500 tokens per stock = 7+ hours for 7 stocks.
#
#   NEW: Per-call token budgets sized to exactly what is asked for:
#        Opening argument  → 150 tokens  (100 words)
#        Cross question    →  80 tokens  ( 50 words)
#        Cross answer      → 110 tokens  ( 70 words)
#        Chair questions   → 100 tokens  ( 60 words)
#        Bot chain analysis→ 250 tokens  (150 words full / 60 words fast)
#        Final verdict     → 400 tokens  (250 words)
#        Result: 4,400 tokens per stock = ~50 min for 7 stocks at 10 tok/s
# ============================================================

import os
from pathlib import Path

# ── Project Paths ────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.resolve()
DATA_DIR    = BASE_DIR / "data"
CACHE_DIR   = BASE_DIR / "cache"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR  = BASE_DIR / "models"

for d in [DATA_DIR, CACHE_DIR, REPORTS_DIR, MODELS_DIR]:
    d.mkdir(exist_ok=True)

# ── Local LLM (Ollama) ───────────────────────────────────────
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# RECOMMENDED MODELS (2025-2026):
#   "qwen2.5:7b"     → best speed/quality for 8 GB RAM  (~10 tok/s CPU)
#   "qwen2.5:14b"    → best for 16 GB RAM               (~6 tok/s CPU)
#   "qwen2.5:3b"     → fastest for 4 GB RAM             (~15 tok/s CPU)
#   "deepseek-r1:7b" → best reasoning, slower (overnight full runs only)
#   "mistral"        → original default, still fine but older
#   "phi4-mini"      → very fast, good structured output
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# LLM generation parameters
LLM_TEMPERATURE  = 0.3    # low = more factual/consistent
LLM_TOP_P        = 0.9
LLM_CONTEXT_WINDOW = 2048  # lowered to 2048 to reduce RAM pressure (OOM hangs)

# ── Per-call token budgets ────────────────────────────────────
# CRITICAL: These replace the single LLM_MAX_TOKENS=1500.
# Each value is set to just above what the prompt requests,
# plus a 30% buffer for numerical data and formatting.
# Ollama generates UP TO num_predict tokens — if the model
# finishes early it stops, but if num_predict is 1500 and the
# model doesn't know when to stop, it will use all 1500.
LLM_TOKENS = {
    # Debate council calls
    "opening":       150,   # 100 word opening argument
    "question":       80,   # 50 word cross-question
    "answer":        110,   # 70 word cross-answer
    "chair_q":       100,   # 60 word chair questions (3 questions)
    "chair_ans":     100,   # 60 word chair answer (3 Q answers)
    "verdict":       450,   # 250 word verdict with 10 structured fields

    # Bot chain calls (LangChain)
    "bot_chain_full": 280,  # 150-200 word full analysis
    "bot_chain_fast": 100,  # 60 word fast analysis

    # Sector bot
    "sector_bot":    200,   # sector analysis + stock list

    # Market scanner LLM
    "market_bot":    200,   # market overview
}

# Legacy alias — used by any code still referencing LLM_MAX_TOKENS directly
# Set to the largest single value (verdict) so nothing breaks
LLM_MAX_TOKENS = LLM_TOKENS["verdict"]

# Stop sequences — model stops generating when it hits these
# This is the most reliable way to prevent over-generation
LLM_STOP_SEQUENCES = {
    "opening":  ["OPENING SCORE:", "---", "==="],
    "question": ["QUESTION TO", "---"],
    "answer":   ["ANSWER:", "---"],
    "verdict":  ["#StockCouncil", "---"],
    "default":  [],
}

# ── Internet / Connectivity ───────────────────────────────────
ALLOW_INTERNET = True

CACHE_TTL_PRICE_HOURS        = 1
CACHE_TTL_FUNDAMENTALS_HOURS = 24
CACHE_TTL_NEWS_HOURS         = 2

# ── Market Data Sources ───────────────────────────────────────
DEFAULT_EXCHANGE_SUFFIX = ".NS"
HISTORICAL_PERIOD       = "2y"
HISTORICAL_INTERVAL     = "1d"
NIFTY_SYMBOL            = "^NSEI"
SENSEX_SYMBOL           = "^BSESN"
NIFTY_BANK_SYMBOL       = "^NSEBANK"

# ── Technical Analysis Parameters ────────────────────────────
TA_PARAMS = {
    "SMA_SHORT": 20, "SMA_MID": 50, "SMA_LONG": 200,
    "EMA_SHORT": 9,  "EMA_MID": 21, "EMA_LONG": 55,
    "RSI_PERIOD": 14, "RSI_OVERBOUGHT": 70, "RSI_OVERSOLD": 30,
    "MACD_FAST": 12,  "MACD_SLOW": 26, "MACD_SIGNAL": 9,
    "BB_PERIOD": 20,  "BB_STD": 2.0,
    "ATR_PERIOD": 14,
    "STOCH_K": 14, "STOCH_D": 3, "STOCH_SMOOTH": 3,
    "WILLIAMS_PERIOD": 14,
    "ADX_PERIOD": 14,
    "VOL_SMA": 20,
    "FIB_LOOKBACK": 120,
    "PIVOT_LOOKBACK": 30,
    "SR_LOOKBACK": 60,
    "SR_SENSITIVITY": 0.02,
}

# ── Fundamental Analysis Thresholds ──────────────────────────
FA_PARAMS = {
    "PE_LOW": 15, "PE_HIGH": 40,
    "PB_LOW": 1.5, "PB_HIGH": 5.0,
    "EV_EBITDA_LOW": 8, "EV_EBITDA_HIGH": 20,
    "ROE_GOOD": 15, "ROE_GREAT": 20,
    "ROCE_GOOD": 15,
    "NET_MARGIN_GOOD": 10,
    "DEBT_EQUITY_LOW": 0.5, "DEBT_EQUITY_HIGH": 1.5,
    "REVENUE_GROWTH_GOOD": 15,
    "EPS_GROWTH_GOOD": 20,
    "DIVIDEND_YIELD_GOOD": 2.0,
    "PROMOTER_HOLD_GOOD": 50, "PROMOTER_HOLD_WARN": 30,
    "FII_GOOD": 15,
}

# ── Sentiment Sources ─────────────────────────────────────────
NEWS_SOURCES = [
    {"name": "Moneycontrol",    "url": "https://www.moneycontrol.com/rss/MCtopnews.xml"},
    {"name": "Economic Times",  "url": "https://economictimes.indiatimes.com/markets/rss.cms"},
    {"name": "Business Standard","url": "https://www.business-standard.com/rss/markets-106.rss"},
    {"name": "LiveMint",        "url": "https://www.livemint.com/rss/markets"},
    {"name": "NSE Official",    "url": "https://www.nseindia.com/"},
]

# ── Risk Scoring Weights ──────────────────────────────────────
RISK_WEIGHTS = {
    "volatility":      0.20,
    "beta":            0.15,
    "debt_risk":       0.20,
    "promoter_pledge": 0.15,
    "governance":      0.10,
    "sector_risk":     0.10,
    "liquidity_risk":  0.10,
}

# ── Bot Scoring Weights for Final Verdict ────────────────────
VERDICT_WEIGHTS = {
    "fundamental": 0.30,
    "technical":   0.25,
    "news":        0.20,
    "sentiment":   0.15,
    "risk":        0.10,
}

# ── Output ────────────────────────────────────────────────────
REPORT_FORMAT = "terminal"
SAVE_REPORTS  = True
VERBOSE_DEBUG = False

# ── Database ─────────────────────────────────────────────────
DB_PATH = DATA_DIR / "stock_council.db"

# ── High-Risk Sectors ─────────────────────────────────────────
HIGH_RISK_SECTORS = [
    "Real Estate", "Construction", "Telecom",
    "Aviation", "Power", "Infrastructure"
]

DEFENSIVE_SECTORS = [
    "FMCG", "Pharmaceuticals", "IT Services",
    "Consumer Staples", "Healthcare"
]
