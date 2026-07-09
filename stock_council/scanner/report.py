# ============================================================
# scanner/report.py — Generate full market scan reports
# ============================================================

import json
import sys
from pathlib import Path
from datetime import datetime
import pytz

sys.path.insert(0, str(Path(__file__).parent.parent))
from scanner.market_scanner import MarketSnapshot
from scanner.sector_scanner import SectorResult
from scanner.stock_scanner import StockVerdict
from config import REPORTS_DIR

IST = pytz.timezone('Asia/Kolkata')


def save_json_report(market: MarketSnapshot,
                     sectors: list[SectorResult],
                     stocks: list[StockVerdict]) -> Path:
    """Save full report as JSON."""
    ts = datetime.now(IST).strftime('%Y%m%d_%H%M')
    path = REPORTS_DIR / f"report_{ts}.json"

    report = {
        "generated_at": datetime.now(IST).isoformat(),
        "market": {
            "score": market.market_score,
            "outlook": market.market_outlook,
            "fii_net": market.fii_net_cr,
            "dii_net": market.dii_net_cr,
            "vix": market.vix,
            "ad_ratio": market.ad_ratio,
            "analysis": market.llm_analysis,
        },
        "sectors": [
            {
                "rank": s.rank,
                "name": s.name,
                "score": s.score,
                "verdict": s.verdict,
                "avg_1m": s.avg_1m_change,
                "rsi": s.avg_rsi,
                "above_50dma_pct": s.pct_above_50dma,
                "top_stocks": [t[0] for t in s.top_stocks],
            }
            for s in sectors
        ],
        "stocks": [
            {
                "rank": v.rank,
                "symbol": v.symbol,
                "company": v.company_name,
                "sector": v.sector,
                "price": v.current_price,
                "verdict": v.verdict,
                "score": v.final_score,
                "scores": {
                    "fundamental": v.fundamental_score,
                    "technical": v.technical_score,
                    "news": v.news_score,
                    "sentiment": v.sentiment_score,
                    "risk": v.risk_score,
                },
                "change_1m": v.change_1m,
                "rsi": v.rsi,
                "pe": v.pe_ratio,
                "roe": v.roe,
                "analysis": v.llm_analysis[:500] if v.llm_analysis else "",
            }
            for v in stocks[:30]
        ]
    }

    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    return path


def print_full_terminal_report(market: MarketSnapshot,
                                sectors: list[SectorResult],
                                stocks: list[StockVerdict]):
    """Print beautifully formatted terminal report."""
    ts = datetime.now(IST).strftime('%d %b %Y %H:%M IST')
    W = 70

    def line(char='─'):
        print(char * W)

    def header(text):
        print(f"\n{'═'*W}")
        print(f"  {text}")
        print('═'*W)

    def section(text):
        print(f"\n{'─'*W}")
        print(f"  {text}")
        print('─'*W)

    # ── Cover ─────────────────────────────────────────────────
    print(f"\n{'═'*W}")
    print(f"{'🏛  INDIAN STOCK MARKET — BOT COUNCIL FULL REPORT':^{W}}")
    print(f"{'Generated: ' + ts:^{W}}")
    print('═'*W)

    # ── Market Overview ────────────────────────────────────────
    header("🌐  LEVEL 1: MARKET OVERVIEW")
    sentiment_bar = '█' * int(market.market_score) + '░' * (10 - int(market.market_score))
    print(f"  Market Score:  [{sentiment_bar}] {market.market_score}/10")
    print(f"  Outlook:       {market.market_outlook}")
    print(f"  FII Net:       ₹{market.fii_net_cr or 'N/A'} Cr  ({market.fii_dii_signal})")
    print(f"  DII Net:       ₹{market.dii_net_cr or 'N/A'} Cr")
    print(f"  India VIX:     {market.vix or 'N/A'}  ({market.vix_signal})")
    print(f"  A/D Ratio:     {market.ad_ratio or 'N/A'}  ({market.breadth_signal})")
    if market.indices:
        section("Live Indices")
        for idx in market.indices[:8]:
            if idx.last and idx.pct_change is not None:
                arrow = '▲' if idx.pct_change >= 0 else '▼'
                bar = '█' * min(10, max(0, int(abs(idx.pct_change) * 2)))
                print(f"  {idx.name:28} {idx.last:>10.2f}  {arrow}{abs(idx.pct_change):.2f}%  {bar}")

    # ── Sector Rankings ────────────────────────────────────────
    header("📂  LEVEL 2: ALL SECTOR RANKINGS")
    print(f"  {'#':3} {'Sector':32} {'Score':7} {'1M':8} {'RSI':6} {'50DMA':7} Verdict")
    line()
    for s in sectors:
        arrow = '▲' if s.avg_1m_change >= 0 else '▼'
        star = '★ ' if s.rank <= 3 else '  '
        dma = f"{s.pct_above_50dma:.0f}%" if s.pct_above_50dma else " N/A"
        print(
            f"  {star}#{s.rank:<2} {s.name:32} {s.score:>4.1f}/10 "
            f" {arrow}{abs(s.avg_1m_change):.1f}%  "
            f"RSI:{s.avg_rsi:<5.0f} {dma:>5}   {s.verdict}"
        )
        if s.rank <= 3 and s.top_stocks:
            top_syms = ', '.join(t[0] for t in s.top_stocks)
            print(f"         Top picks: {top_syms}")

    # ── Stock Rankings ─────────────────────────────────────────
    header("🏆  LEVEL 3: TOP STOCK PICKS (RANKED)")
    print(f"  {'#':3} {'Symbol':12} {'Company':22} {'Price':9} {'1M':8} {'RSI':6} {'Score':7} Verdict")
    line()

    for v in stocks[:25]:
        arrow = '▲' if (v.change_1m or 0) >= 0 else '▼'
        price = f"₹{v.current_price:.0f}" if v.current_price else "N/A"
        company = (v.company_name or v.symbol)[:21]
        rsi_str = f"{v.rsi:.0f}" if v.rsi else "N/A"
        star = "🥇" if v.rank == 1 else "🥈" if v.rank == 2 else "🥉" if v.rank == 3 else f"#{v.rank:<2}"
        print(
            f"  {star} {v.symbol:12} {company:22} {price:9} "
            f"{arrow}{abs(v.change_1m or 0):.1f}%  "
            f"RSI:{rsi_str:4}  {v.final_score:>3.1f}/10  {v.verdict}"
        )

    # ── Top 5 Deep Analysis ────────────────────────────────────
    deep = [v for v in stocks[:10] if v.llm_analysis]
    if deep:
        header("🔬  TOP 5 STOCKS — FULL COUNCIL ANALYSIS")
        for v in deep[:5]:
            section(f"#{v.rank} {v.symbol} — {v.company_name} | {v.sector}")
            print(f"  Price: ₹{v.current_price}  |  Score: {v.final_score}/10  |  {v.verdict}")
            print(f"  F:{v.fundamental_score}  T:{v.technical_score}  "
                  f"N:{v.news_score}  S:{v.sentiment_score}  Risk:{v.risk_score}")
            line()
            # Wrap LLM text
            text = v.llm_analysis
            for para in text.split('\n'):
                if para.strip():
                    print(f"  {para.strip()}")

    # ── Footer ─────────────────────────────────────────────────
    print(f"\n{'═'*W}")
    print(f"  ⚠  DISCLAIMER: For educational purposes only.")
    print(f"  Not SEBI-registered advice. Always do your own research.")
    print(f"{'═'*W}\n")


def generate(market: MarketSnapshot,
             sectors: list[SectorResult],
             stocks: list[StockVerdict],
             save: bool = True) -> Path | None:
    """Generate all report formats."""
    print_full_terminal_report(market, sectors, stocks)
    if save:
        path = save_json_report(market, sectors, stocks)
        print(f"\n  💾 Report saved: {path}")
        return path
    return None
