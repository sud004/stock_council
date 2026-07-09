# 🏛️ Stock Council — AI-Powered NSE Trading System

> **5 specialized LLM bots debate every qualifying stock nightly and deliver a structured buy/sell verdict — with a live paper-trading portfolio tracking real performance over 21 days.**

---

## What It Does

Stock Council runs a fully automated nightly pipeline that:

1. **Scans all 13 NSE sectors** and scores them by momentum, RSI, FII/DII flows, and volatility
2. **Computes a Growth Probability Score (GPS)** for every stock in the universe (~286 NSE stocks)
3. **Routes high-GPS stocks into a 5-bot AI council** where specialist bots debate fundamentals, technicals, news, sentiment, and risk
4. **Produces structured verdicts** (STRONG BUY / BUY / HOLD / REDUCE / SELL) with confidence scores
5. **Executes paper trades** on Model A (₹1L lump sum) and Model B (₹10K/day conviction budget) simultaneously
6. **Logs missed opportunities** nightly — stocks that moved big but weren't debated — to improve future scans

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1 — Data Refresh                                         │
│  Price CSVs · Fundamentals · News · FII/DII · India VIX        │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2 — Prediction Accuracy Review                           │
│  Checks yesterday's 1D predictions vs actual closes            │
│  Auto-updates learned weights (Sunday 7 PM IST weekly review)  │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 3 — Sector Scan + Council                                │
│                                                                 │
│  Sector Scanner  →  GPS Filter (≥6.0)  →  Bot Council          │
│  13 sectors          Top 10/sector          5 specialist bots  │
│  scored 0-10         GPS Rescue Pass        per qualifying      │
│                       (GPS ≥7.5 override)   stock              │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Fundamental  │  │  Technical   │  │   News Bot   │          │
│  │     Bot      │  │     Bot      │  │              │          │
│  │ P/E·ROE·Debt │  │RSI·MACD·ADX  │  │Headlines·RBI │          │
│  │ ChromaDB mem │  │Bollinger·Ichi│  │FII/DII·Macro │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│  ┌──────────────┐  ┌──────────────┐                            │
│  │  Sentiment   │  │   Risk Bot   │  Council Chair             │
│  │     Bot      │  │              │  synthesises all 5         │
│  │PCR·VIX·Reddit│  │VaR·Beta·Draw │  → Final verdict + score  │
│  └──────────────┘  └──────────────┘                            │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 4 — Portfolio Engine                                     │
│  Model A: ₹1,00,000 lump sum · GPS-tiered deployment %         │
│  Model B: ₹10,000/day conviction budget · vanishes if unused   │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 5 — Night Report + Learning Progress                     │
│  Accuracy tracking · Milestone log · Excel trackers updated     │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 6 — Missed Opportunity Analysis (analytical only)        │
│  Top daily movers not debated → fast council → logs to JSON     │
│  Feeds weekly parameter review; never triggers real trades      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Features

| Feature | Detail |
|---|---|
| **GPS Rescue Pass** | Any stock with GPS ≥ 7.5 skipped by sector cap gets force-debated |
| **Bot Memory** | ChromaDB vector store; each bot reads its own past analyses before responding |
| **Per-bot Checkpointing** | Crash recovery resumes from last completed bot — no full restart needed |
| **Dual Portfolio Models** | A/B comparison: lump-sum vs daily-conviction allocation strategies |
| **Weekly Parameter Review** | Auto-approves prediction weights Sunday 7 PM IST; GPS threshold needs Y/N |
| **Missed Opportunity Log** | Nightly JSON log of high-movers not debated; feeds structural improvements |
| **NSE Trading Calendar** | Holiday-aware day counter; correct trading date even after weekends/gaps |

---

## Tech Stack

- **LLM Backend** — Ollama (local, runs on your GPU)
- **Vector Memory** — ChromaDB
- **Data Sources** — Yahoo Finance (prices), NSE APIs (fundamentals/FII/DII), NewsAPI
- **Brokerage** — Ventura EaseAPI (paper trading integration)
- **Portfolio Engine** — Custom Python (openpyxl trackers, JSON state)
- **Scheduler** — Windows Task Scheduler / cron

---

## Setup

### Requirements

- Python 3.10+
- [Ollama](https://ollama.ai) installed with a model (e.g. `llama3.1:8b` or `mistral`)
- Windows (for Task Scheduler) or Linux/Mac (cron)

### Install

```bash
git clone https://github.com/sud004/stock-council.git
cd stock-council
pip install -r stock_council/requirements.txt
```

### Configure

```bash
# Set up Ventura EaseAPI (once)
python stock_council/utils/easeapi_auth.py --setup

# Download all historical data (run once, takes ~30 min)
python stock_council/run.py --nightly
```

### Run

```bash
# Manual run (any time)
python stock_council/night_runner.py

# Schedule nightly (e.g. 9 PM IST)
python stock_council/night_runner.py --schedule

# Force re-run same day
python stock_council/night_runner.py --force

# Skip Missed Opportunity phase (faster)
python stock_council/night_runner.py --skip-missed-opp
```

---

## Project Structure

```
stock_council/
├── night_runner.py          # Main orchestrator — 6-phase nightly pipeline
├── portfolio_engine.py      # Model A: lump-sum portfolio logic
├── portfolio_engine_v2.py   # Model B: daily conviction budget logic
├── trading_calendar.py      # NSE holiday-aware date logic
├── config.py                # All thresholds, paths, constants
│
├── bots/                    # The 5 specialist council bots
│   ├── fundamental_bot.py
│   ├── technical_bot.py
│   ├── news_bot.py
│   ├── sentiment_bot.py
│   └── risk_bot.py
│
├── scanner/                 # Sector scan + GPS computation
│   └── sector_scanner.py
│
├── pipeline/                # Council orchestration (binary — compiled)
│   └── orchestrator.py
│
├── memory/                  # ChromaDB vector store interface
│   └── storage.py
│
├── data/                    # Runtime state (gitignored except key JSON)
│   ├── learned_weights.json       # Bot weight history
│   ├── learning_progress.json     # Day counter + accuracy log
│   ├── missed_opportunities.json  # Nightly missed-opportunity log
│   └── predictions.json           # 1D/3D/7D price predictions
│
└── utils/                   # Auth, notifiers, helpers
```

---

## Live Results (Day 15 / 21)

> Paper trading started June 17, 2026. Results updated nightly.

| Metric | Model A (Lump Sum) | Model B (Daily Budget) |
|---|---|---|
| Capital | ₹1,00,000 | ₹10,000/day × 14 days |
| Open Positions | 17 | 5 |
| Win Rate | — | 80% (8W / 2L) |
| Total P&L | tracked nightly | +₹280 (+0.95%) |

---

## Model Improvement Log

All structural changes are tracked in `portfolio_tracker.xlsx → Model Improvements` sheet.

Key changes to date:
- `COUNCIL_MAX_RETRIES` raised 3 → 12 (crash resilience)
- Per-bot checkpoint cache (resume on GPU OOM)
- Sector scanner pool expanded 3 → 10 stocks/sector
- GPS Rescue Pass for GPS ≥ 7.5 stocks bypassed by sector cap
- Phase 6 Missed Opportunity Analysis added
- Model B daily conviction budget engine

---

## Author

**Sudhanshu Kumar Sharma**  
Deputy Manager · AI & Deep Learning · IFFCO  
IIT Roorkee — Executive PG, AI & Deep Learning (2023-24)  
[LinkedIn](https://linkedin.com/in/ssafreak) · [GitHub](https://github.com/sud004)

---

> *This is a research/learning project. Not financial advice. Paper trading only.*
