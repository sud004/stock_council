# 🏛 Indian Stock Market Bot Council

**5 AI bots debate every stock that has high growth probability and deliver a verdict.**

---

## Architecture

```
Market Scan (NSE indices + FII/DII + VIX + global cues)
       ↓
All 13 Sectors scored & ranked
Sector Bot decides: HOT sector → 8 stocks, WARM → 4, COLD → skip
       ↓
Growth Probability Score (GPS) filter
Only stocks scoring ≥ 6.5/10 enter the council
       ↓
Bot Council debates each qualified stock:
  📊 Fundamental Bot  — P/E, ROE, debt, growth (reads past history via ChromaDB)
  📈 Technical Bot    — RSI, MACD, Bollinger, ADX, Fibonacci, Ichimoku
  📰 News Bot         — headlines, RBI, FII/DII, macro catalysts
  💬 Sentiment Bot    — PCR, India VIX, Reddit, institutional flows
  ⚠️  Risk Bot         — VaR, beta, max drawdown, debt risk, governance
       ↓
Council Chair synthesises → Final Verdict + Score
       ↓
Everything saved: CSV + JSON + Excel + ChromaDB vectors
Bots improve daily by reading their own past analyses
```

---

## Setup (15 minutes total)

### 1. Install
```bash
bash setup.sh
```

### 2. Set up Ventura EaseAPI (5 min, once ever)
```bash
python utils/easeapi_auth.py --setup
```
This guides you through:
- Getting your app key from easeapi.venturasecurities.com/portal
- Enabling TOTP in Ventura app (Profile → Security → Authenticator)
- Testing the connection

**After this, the system auto-logins every morning at 8:45 AM. No manual action needed.**

### 3. Download all stock data (run tonight)
```bash
python run.py --nightly
```
Downloads: prices (2 years), fundamentals, news for all 286 NSE stocks.
Stored locally — never need to re-download unless you want fresh history.

### 4. Run
```bash
python run.py            # Full pipeline with LLM
python run.py --fast     # Fast quantitative mode (~5 min)
python run.py --schedule # All-day hourly tracker
```

---

## Token / Login — Answered

**Q: Do I need to generate a new API key every day?**

No. The `app_key` and `app_secret` are permanent — you get them once from the EaseAPI portal and never change them.

The `auth_token` expires daily (this is SEBI's requirement for all Indian brokers — Zerodha, Upstox, Angel all do the same). But after the one-time TOTP setup, the system renews it automatically every morning at 8:45 AM using your TOTP secret. You never touch it.

**Q: What if I don't set up TOTP?**

The pipeline still works but you need to manually paste a token once a day before running. After TOTP setup that step disappears entirely.

---

## Commands

```bash
# One-time setup
python utils/easeapi_auth.py --setup    # set up auto-login (5 min)
python run.py --nightly                 # download all data

# Daily usage
python run.py                           # full pipeline
python run.py --fast                    # quant-only, no LLM needed
python run.py --schedule                # hourly all-day tracker

# Tuning
python run.py --gps 5.5                 # lower bar → more stocks debated
python run.py --model phi3:mini         # faster model (low RAM)
python run.py --offline                 # use cached data only

# Check status
python run.py --status
python utils/easeapi_auth.py --status
```

---

## Data stored locally

| Location | Contents | Size |
|----------|----------|------|
| `data/prices/*.csv` | 2-year OHLCV per stock | ~50MB for 286 stocks |
| `data/fundamentals/*.json` | P/E, ROE, debt etc. | ~5MB |
| `data/news/{SYMBOL}/*.txt` | Daily news archive | grows over time |
| `data/scores/*.json` | Bot score history 90 days | ~2MB |
| `data/vectors/` | ChromaDB embeddings | ~100MB after 2 weeks |
| `data/excel/nightly_*.xlsx` | Full market Excel report | ~3MB per day |
| `memory/council_*.json` | Full session history | ~1MB per day |

---

## Hardware Requirements

| RAM | Best model | Speed |
|-----|-----------|-------|
| 4GB | phi3:mini | ~3 min/stock |
| 8GB | mistral | ~6 min/stock |
| 16GB | llama3.1 | ~8 min/stock |
| GPU | any | 5-10× faster |

Use `--fast` mode for pure quantitative analysis — no LLM, scans all 286 stocks in ~5 minutes.

---

## Disclaimer

For educational and research purposes only. Not SEBI-registered investment advice. Always consult a qualified advisor before trading.
