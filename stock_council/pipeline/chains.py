# ============================================================
# pipeline/chains.py
# LangChain chains for each bot with RAG memory
# ============================================================
#
# EACH BOT IS A LANGCHAIN CHAIN:
#   Input  → RAG retrieval (vector store) → LLM → Output
#
# CHAIN TYPES:
#   FundamentalChain   : RetrievalQA with stock fundamental context
#   TechnicalChain     : RetrievalQA with price history context
#   NewsChain          : RetrievalQA with news summaries context
#   SentimentChain     : RetrievalQA with FII/DII + social context
#   RiskChain          : RetrievalQA with risk history context
#   CouncilChain       : Sequential chain — all 5 bots debate in order
#   VerdictChain       : Final judge with full debate as input
#
# MEMORY:
#   Each bot has ConversationBufferWindowMemory (last 5 exchanges)
#   Vector store provides long-term RAG context
#   Bots see their OWN past analyses before speaking
# ============================================================

import sys
from pathlib import Path
from datetime import datetime
import pytz

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OLLAMA_MODEL, OLLAMA_HOST, LLM_TEMPERATURE, VERBOSE_DEBUG

IST = pytz.timezone('Asia/Kolkata')

import concurrent.futures as _cf

def _safe_vec(fn, timeout=8, default=""):
    """Wrap a ChromaDB vector call with a hard timeout to prevent SQLite lock hangs."""
    with _cf.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn)
        try:
            return fut.result(timeout=timeout)
        except Exception:
            return default


# ── LangChain LLM wrapper for Ollama ──────────────────────────

def get_llm(temperature: float = None, model: str = None):
    """
    Get LangChain-wrapped Ollama LLM.
    Uses langchain-ollama for clean integration.
    """
    try:
        from langchain_ollama import OllamaLLM
        return OllamaLLM(
            model=model or OLLAMA_MODEL,
            base_url=OLLAMA_HOST,
            temperature=temperature if temperature is not None else LLM_TEMPERATURE,
            num_predict=1500,
        )
    except ImportError:
        try:
            from langchain_community.llms import Ollama
            return Ollama(
                model=model or OLLAMA_MODEL,
                base_url=OLLAMA_HOST,
                temperature=temperature if temperature is not None else LLM_TEMPERATURE,
            )
        except ImportError:
            raise ImportError(
                "Install langchain-ollama: pip install langchain-ollama"
            )


def get_embeddings():
    """
    Get local sentence-transformer embeddings for LangChain.
    No API key needed.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embed_path = Path(__file__).parent.parent / "models" / "embeddings" / "minilm"
    model_name = str(embed_path) if embed_path.exists() else "sentence-transformers/all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


def get_chroma_retriever(collection_name: str, filter_dict: dict = None, top_k: int = 4):
    """
    Get a LangChain retriever backed by local ChromaDB.
    """
    try:
        from langchain_chroma import Chroma
        from config import DATA_DIR

        vectordb = Chroma(
            collection_name=collection_name,
            embedding_function=get_embeddings(),
            persist_directory=str(DATA_DIR / "vectors"),
            collection_metadata={"hnsw:space": "cosine"}
        )
        search_kwargs = {"k": top_k}
        if filter_dict:
            search_kwargs["filter"] = filter_dict

        return vectordb.as_retriever(search_kwargs=search_kwargs)
    except Exception as e:
        if VERBOSE_DEBUG:
            print(f"[CHAIN] Retriever error: {e}")
        return None


# ══════════════════════════════════════════════════════════════
# BOT PROMPTS (LangChain PromptTemplates)
# ══════════════════════════════════════════════════════════════

from langchain.prompts import PromptTemplate

FUNDAMENTAL_PROMPT = PromptTemplate(
    input_variables=["symbol", "sector", "live_data", "history_context", "score_trend"],
    template="""You are FUNDAMENTAL BOT — expert Indian equity fundamental analyst.

HISTORICAL CONTEXT (your past analyses):
{history_context}

YOUR SCORE TREND: {score_trend}

LIVE FUNDAMENTAL DATA:
{live_data}

TASK: Analyse {symbol} ({sector}) fundamentals.
- Compare today vs your historical view — has anything CHANGED?
- Call out if the stock is improving or deteriorating vs last time
- Give specific Indian market context (GST, promoter quality, RBI impact)
- Highlight 3 positives, 2 concerns
- End with: "FUNDAMENTAL SCORE: X/10"

Your analysis (150-200 words):"""
)

TECHNICAL_PROMPT = PromptTemplate(
    input_variables=["symbol", "live_data", "history_context", "score_trend"],
    template="""You are TECHNICAL BOT — expert Indian stock technical analyst (CMT).

HISTORICAL CONTEXT (your past chart readings):
{history_context}

YOUR SCORE TREND: {score_trend}

LIVE TECHNICAL DATA:
{live_data}

TASK: Analyse {symbol} chart setup.
- Has the technical picture IMPROVED or DETERIORATED since last analysis?
- Are we breaking out or breaking down? What is the key level?
- RSI, MACD, volume tell what story today?
- Give specific price targets and stop levels in ₹
- End with: "TECHNICAL SCORE: X/10"

Your analysis (150-200 words):"""
)

NEWS_PROMPT = PromptTemplate(
    input_variables=["symbol", "sector", "live_news", "history_context", "news_context"],
    template="""You are NEWS BOT — expert Indian financial news analyst.

PAST NEWS CONTEXT (stored locally):
{history_context}

RECENT NEWS ARCHIVE:
{news_context}

TODAY'S LIVE NEWS:
{live_news}

TASK: Analyse news flow for {symbol} ({sector}).
- What is NEW today vs stored history?
- Is the narrative improving or worsening?
- What is the single biggest catalyst (positive or negative)?
- Macro: RBI, FII flows, sector tailwinds specific to India
- End with: "NEWS SCORE: X/10"

Your analysis (150-200 words):"""
)

SENTIMENT_PROMPT = PromptTemplate(
    input_variables=["symbol", "live_sentiment_data", "history_context", "market_context"],
    template="""You are SENTIMENT BOT — expert at reading Indian market psychology.

MARKET CONTEXT (recent history):
{market_context}

YOUR HISTORICAL READINGS:
{history_context}

LIVE SENTIMENT DATA:
{live_sentiment_data}

TASK: Assess sentiment for {symbol}.
- FII/DII: are institutions accumulating or distributing?
- PCR and options positioning — what is smart money doing?
- India VIX: fear vs greed reading
- Reddit/social: retail mood
- Has sentiment SHIFTED since last reading?
- End with: "SENTIMENT SCORE: X/10"

Your analysis (150-200 words):"""
)

RISK_PROMPT = PromptTemplate(
    input_variables=["symbol", "sector", "live_risk_data", "history_context", "score_trend"],
    template="""You are RISK BOT — conservative Indian stock risk analyst. Devil's advocate.

PAST RISK ASSESSMENTS:
{history_context}

RISK SCORE TREND: {score_trend}

LIVE RISK DATA:
{live_risk_data}

TASK: Identify risks for {symbol} ({sector}).
- Have risks INCREASED or DECREASED vs last assessment?
- List TOP 3 specific risks RIGHT NOW
- Quantify: VaR, beta, max drawdown, debt risk
- India-specific: promoter pledge, SEBI risk, RBI sensitivity
- Give risk SCORE (10 = extreme risk, 1 = very safe)
- End with: "RISK SCORE: X/10"

Your analysis (150-200 words):"""
)

COUNCIL_DEBATE_PROMPT = PromptTemplate(
    input_variables=[
        "symbol", "sector", "company", "price",
        "fundamental_analysis", "technical_analysis",
        "news_analysis", "sentiment_analysis", "risk_analysis",
        "scores_line", "past_verdict", "market_context"
    ],
    template="""You are the COUNCIL CHAIR of an Indian equity research firm.

5 expert bots have debated {symbol} ({company}, {sector}) at ₹{price}.

MARKET CONTEXT:
{market_context}

PAST VERDICT (if any):
{past_verdict}

TODAY'S BOT ANALYSES:
━━━ FUNDAMENTAL BOT ━━━
{fundamental_analysis}

━━━ TECHNICAL BOT ━━━
{technical_analysis}

━━━ NEWS BOT ━━━
{news_analysis}

━━━ SENTIMENT BOT ━━━
{sentiment_analysis}

━━━ RISK BOT (devil's advocate) ━━━
{risk_analysis}

SCORES: {scores_line}

YOUR JOB:
1. Summarise where bots AGREE and where they DISAGREE
2. Weigh the bull vs bear case
3. Give final VERDICT: STRONG BUY / BUY / ACCUMULATE / HOLD / REDUCE / SELL / STRONG SELL
4. Suggest entry price range or wait level
5. Time horizon: SHORT (days) / MEDIUM (weeks) / LONG (months)
6. Has the story IMPROVED or DECLINED vs past verdict?

Final council verdict (150 words):"""
)

SECTOR_HOT_PROMPT = PromptTemplate(
    input_variables=[
        "sector", "sector_score", "sector_metrics",
        "all_stocks_data", "market_context", "past_sector_context"
    ],
    template="""You are SECTOR BOT — Indian equity sector specialist.

SECTOR: {sector} | SCORE: {sector_score}/10

PAST SECTOR ANALYSIS:
{past_sector_context}

MARKET CONTEXT:
{market_context}

SECTOR METRICS:
{sector_metrics}

ALL STOCKS IN THIS SECTOR:
{all_stocks_data}

TASK:
1. Is this sector HOT or COLD right now? Why?
2. What is the dominant theme driving this sector?
3. Rank the TOP stocks within this sector by opportunity
4. How many stocks from this sector should the council debate?
   - VERY HOT (score 8+): take top 8 stocks
   - HOT (score 7-8):     take top 6 stocks
   - WARM (score 6-7):    take top 4 stocks
   - NEUTRAL (5-6):       take top 2 stocks
   - COLD (<5):           skip sector
5. Name the stocks explicitly

End with: "SECTOR SCORE: X/10 | STOCKS TO DEBATE: [SYM1, SYM2, ...]"

Your sector analysis (130 words):"""
)


# ══════════════════════════════════════════════════════════════
# LANGCHAIN CHAINS
# ══════════════════════════════════════════════════════════════

class BotChain:
    """
    Base class for all bot chains.
    Each chain = PromptTemplate + Ollama LLM + Vector RAG
    """

    def __init__(self, bot_name: str, prompt_template: PromptTemplate):
        self.bot_name = bot_name
        self.prompt = prompt_template
        self._llm = None
        self._chain = None

    def _get_chain(self):
        if self._chain is None:
            from langchain.chains import LLMChain
            self._llm = get_llm()
            self._chain = LLMChain(llm=self._llm, prompt=self.prompt, verbose=VERBOSE_DEBUG)
        return self._chain

    def run(self, inputs: dict, print_output: bool = True) -> str:
        """Run the chain with given inputs."""
        chain = self._get_chain()
        if print_output:
            print(f"\n[{self.bot_name.upper()}] Analyzing...")
        try:
            result = chain.invoke(inputs)
            text = result.get('text', str(result))
            if print_output:
                print(text)
            return text
        except Exception as e:
            err_str = str(e)
            # OOM: model can't load into RAM — retry once with 3b
            if "allocate" in err_str.lower() and "3b" not in str(self._llm):
                print(f"\n[{self.bot_name.upper()}] ⚠ OOM → retrying with qwen2.5:3b",
                      flush=True)
                try:
                    from langchain.chains import LLMChain
                    _fallback_llm = get_llm(model="qwen2.5:3b")
                    _fallback_chain = LLMChain(llm=_fallback_llm, prompt=self.prompt,
                                               verbose=VERBOSE_DEBUG)
                    result = _fallback_chain.invoke(inputs)
                    text = result.get('text', str(result))
                    if print_output:
                        print(text)
                    return text
                except Exception as e2:
                    err = f"[{self.bot_name} ERROR (3b fallback): {e2}]"
                    print(err)
                    return err
            err = f"[{self.bot_name} ERROR: {e}]"
            print(err)
            return err


class FundamentalChain(BotChain):
    def __init__(self):
        super().__init__("fundamental", FUNDAMENTAL_PROMPT)

    def analyze(self, symbol: str, sector: str, live_data: str,
                 vector_store, print_output: bool = True,
                 fast_mode: bool = False) -> str:
        # Get history from vector store
        hist_ctx = _safe_vec(lambda: vector_store.get_context_for_stock(
            symbol, "fundamental valuation earnings growth"
        )) if vector_store else ""

        trend = _safe_vec(lambda: vector_store.query_stock_history(
            symbol, "score", bot="fundamental", top_k=5
        ), default=[]) if vector_store else []
        score_vals = [r['metadata'].get('score') for r in trend if r['metadata'].get('score')]
        if len(score_vals) >= 2:
            delta = round(score_vals[-1] - score_vals[0], 1)
            trend_str = f"Last {len(score_vals)} sessions: avg {sum(score_vals)/len(score_vals):.1f} | trend {delta:+.1f}"
        else:
            trend_str = "No history yet"

        return self.run({
            'symbol': symbol,
            'sector': sector,
            'live_data': live_data,
            'history_context': hist_ctx or "No past analyses stored yet",
            'score_trend': trend_str,
        }, print_output=print_output)


class TechnicalChain(BotChain):
    def __init__(self):
        super().__init__("technical", TECHNICAL_PROMPT)

    def analyze(self, symbol: str, live_data: str,
                 vector_store, print_output: bool = True,
                 fast_mode: bool = False) -> str:
        hist_ctx = _safe_vec(lambda: vector_store.get_context_for_stock(
            symbol, "technical RSI MACD trend support resistance"
        )) if vector_store else ""

        trend = _safe_vec(lambda: vector_store.query_stock_history(
            symbol, "technical score", bot="technical", top_k=5
        ), default=[]) if vector_store else []
        score_vals = [r['metadata'].get('score') for r in trend if r['metadata'].get('score')]
        trend_str = f"Scores: {score_vals}" if score_vals else "No history yet"

        return self.run({
            'symbol': symbol,
            'live_data': live_data,
            'history_context': hist_ctx or "No past analyses stored yet",
            'score_trend': trend_str,
        }, print_output=print_output)


class NewsChain(BotChain):
    def __init__(self):
        super().__init__("news", NEWS_PROMPT)

    def analyze(self, symbol: str, sector: str, live_news: str,
                 vector_store, print_output: bool = True,
                 fast_mode: bool = False) -> str:
        hist_ctx = _safe_vec(lambda: vector_store.get_context_for_stock(
            symbol, "news sentiment catalyst announcement"
        )) if vector_store else ""
        news_ctx = _safe_vec(lambda: vector_store._query_collection(
            'news_summaries', f"{symbol} news", where={'symbol': symbol}, top_k=3
        ), default=[]) if vector_store else []
        news_ctx_str = '\n'.join(r['text'][:300] for r in news_ctx)

        return self.run({
            'symbol': symbol,
            'sector': sector,
            'live_news': live_news,
            'history_context': hist_ctx or "No past analyses stored yet",
            'news_context': news_ctx_str or "No past news stored yet",
        }, print_output=print_output)


class SentimentChain(BotChain):
    def __init__(self):
        super().__init__("sentiment", SENTIMENT_PROMPT)

    def analyze(self, symbol: str, live_sentiment: str,
                 vector_store, print_output: bool = True,
                 fast_mode: bool = False) -> str:
        hist_ctx = _safe_vec(lambda: vector_store.get_context_for_stock(
            symbol, "sentiment FII DII PCR social Reddit"
        )) if vector_store else ""
        mkt_ctx = _safe_vec(lambda: vector_store.get_market_context(top_k=2)) if vector_store else ""

        return self.run({
            'symbol': symbol,
            'live_sentiment_data': live_sentiment,
            'history_context': hist_ctx or "No past analyses stored yet",
            'market_context': mkt_ctx or "No market context stored yet",
        }, print_output=print_output)


class RiskChain(BotChain):
    def __init__(self):
        super().__init__("risk", RISK_PROMPT)

    def analyze(self, symbol: str, sector: str, live_risk: str,
                 vector_store, print_output: bool = True,
                 fast_mode: bool = False) -> str:
        hist_ctx = _safe_vec(lambda: vector_store.get_context_for_stock(
            symbol, "risk volatility drawdown debt governance"
        )) if vector_store else ""

        trend = _safe_vec(lambda: vector_store.query_stock_history(
            symbol, "risk score", bot="risk", top_k=5
        ), default=[]) if vector_store else []
        score_vals = [r['metadata'].get('score') for r in trend if r['metadata'].get('score')]
        trend_str = f"Risk levels: {score_vals}" if score_vals else "No history yet"

        return self.run({
            'symbol': symbol,
            'sector': sector,
            'live_risk_data': live_risk,
            'history_context': hist_ctx or "No past risk data stored yet",
            'score_trend': trend_str,
        }, print_output=print_output)


class SectorHotChain(BotChain):
    def __init__(self):
        super().__init__("sector", SECTOR_HOT_PROMPT)

    def analyze(self, sector: str, sector_score: float,
                 sector_metrics: str, all_stocks_data: str,
                 vector_store, market_context: str = "",
                 print_output: bool = True) -> tuple[str, list[str]]:
        """
        Returns (analysis_text, list_of_stocks_to_debate)
        """
        past_ctx = vector_store.get_sector_context(sector, top_k=3) if vector_store else ""

        text = self.run({
            'sector': sector,
            'sector_score': round(sector_score, 1),
            'sector_metrics': sector_metrics,
            'all_stocks_data': all_stocks_data,
            'market_context': market_context,
            'past_sector_context': past_ctx or "No past sector data",
        }, print_output=print_output)

        # Extract stock list from LLM output
        stocks = _extract_stocks_from_text(text)
        return text, stocks


class CouncilDebateChain(BotChain):
    def __init__(self):
        super().__init__("council", COUNCIL_DEBATE_PROMPT)

    def debate(self, symbol: str, company: str, sector: str, price: float,
                fund_text: str, tech_text: str, news_text: str,
                sent_text: str, risk_text: str,
                scores: dict, vector_store,
                print_output: bool = True) -> str:

        scores_line = (
            f"Fund:{scores.get('fundamental',5):.1f} | "
            f"Tech:{scores.get('technical',5):.1f} | "
            f"News:{scores.get('news',5):.1f} | "
            f"Sent:{scores.get('sentiment',5):.1f} | "
            f"Risk:{scores.get('risk',5):.1f}"
        )

        # Past verdict
        past = vector_store._query_collection(
            'council_debates', f"{symbol} verdict",
            where={'symbol': symbol}, top_k=1
        ) if vector_store else []
        past_verdict = past[0]['text'][:300] if past else "No past verdict"

        mkt_ctx = vector_store.get_market_context(top_k=1) if vector_store else ""

        return self.run({
            'symbol': symbol,
            'company': company,
            'sector': sector,
            'price': f"{price:.2f}" if price else "N/A",
            'fundamental_analysis': fund_text[:600],
            'technical_analysis': tech_text[:600],
            'news_analysis': news_text[:500],
            'sentiment_analysis': sent_text[:500],
            'risk_analysis': risk_text[:500],
            'scores_line': scores_line,
            'past_verdict': past_verdict,
            'market_context': mkt_ctx[:300] if mkt_ctx else "N/A",
        }, print_output=print_output)


# ── Helper ────────────────────────────────────────────────────

def _extract_stocks_from_text(text: str) -> list[str]:
    """
    Parse the sector bot's output to find which stocks to debate.
    Looks for: "STOCKS TO DEBATE: [RELIANCE, TCS, INFY]"
    """
    import re
    from scanner.universe import ALL_STOCKS

    # Try explicit format first
    match = re.search(r'STOCKS TO DEBATE:\s*\[([^\]]+)\]', text, re.IGNORECASE)
    if match:
        raw = match.group(1)
        stocks = [s.strip().upper() for s in re.split(r'[,\s]+', raw) if s.strip()]
        return [s for s in stocks if s in ALL_STOCKS]

    # Fallback: find all known stock symbols mentioned in text
    found = []
    text_upper = text.upper()
    for sym in ALL_STOCKS:
        if sym in text_upper:
            found.append(sym)
    return found[:8]   # cap at 8


# ── Chain registry ────────────────────────────────────────────

_chains = {}

def get_fundamental_chain() -> FundamentalChain:
    if 'fundamental' not in _chains:
        _chains['fundamental'] = FundamentalChain()
    return _chains['fundamental']

def get_technical_chain() -> TechnicalChain:
    if 'technical' not in _chains:
        _chains['technical'] = TechnicalChain()
    return _chains['technical']

def get_news_chain() -> NewsChain:
    if 'news' not in _chains:
        _chains['news'] = NewsChain()
    return _chains['news']

def get_sentiment_chain() -> SentimentChain:
    if 'sentiment' not in _chains:
        _chains['sentiment'] = SentimentChain()
    return _chains['sentiment']

def get_risk_chain() -> RiskChain:
    if 'risk' not in _chains:
        _chains['risk'] = RiskChain()
    return _chains['risk']

def get_sector_chain() -> SectorHotChain:
    if 'sector' not in _chains:
        _chains['sector'] = SectorHotChain()
    return _chains['sector']

def get_council_chain() -> CouncilDebateChain:
    if 'council' not in _chains:
        _chains['council'] = CouncilDebateChain()
    return _chains['council']