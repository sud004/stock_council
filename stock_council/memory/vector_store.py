# ============================================================
# memory/vector_store.py
# Local vector database using ChromaDB + sentence-transformers
# NO OpenAI, NO API key — 100% local embeddings
# ============================================================
#
# COLLECTIONS:
#   "market_analyses"    — daily market bot outputs
#   "sector_analyses"    — sector bot outputs per sector per day
#   "stock_analyses"     — per-stock bot council outputs
#   "news_summaries"     — news sentiment summaries
#   "price_narratives"   — price action descriptions
#
# EMBEDDING MODEL:
#   sentence-transformers/all-MiniLM-L6-v2
#   - 384 dimensions, ~80MB download once
#   - Runs on CPU, ~50ms per embedding
#   - Stored locally in models/embeddings/
#
# USAGE:
#   store = VectorMemory()
#   store.add_analysis("RELIANCE", "fundamental", "Strong ROE of 18%...")
#   results = store.query_similar("RELIANCE", "fundamental outlook")
#   context = store.get_context_for_stock("RELIANCE", top_k=5)
# ============================================================

import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import pytz

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BASE_DIR, VERBOSE_DEBUG

IST = pytz.timezone('Asia/Kolkata')

VECTOR_DIR  = BASE_DIR / "data" / "vectors"
EMBED_DIR   = BASE_DIR / "models" / "embeddings"
VECTOR_DIR.mkdir(parents=True, exist_ok=True)
EMBED_DIR.mkdir(parents=True, exist_ok=True)

# Embedding model name — downloaded once, stored locally
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class VectorMemory:
    """
    Local vector store for all market intelligence.
    Bots query this before speaking — so they learn from history.
    """

    def __init__(self):
        self._client = None
        self._embedder = None
        self._collections = {}
        self._initialized = False

    def _init(self):
        """Lazy init — only load heavy models when first needed."""
        if self._initialized:
            return

        try:
            import chromadb
            from chromadb.config import Settings

            # Persistent local ChromaDB
            self._client = chromadb.PersistentClient(
                path=str(VECTOR_DIR),
                settings=Settings(anonymized_telemetry=False)
            )

            # Create collections
            collection_names = [
                "market_analyses",
                "sector_analyses",
                "stock_analyses",
                "news_summaries",
                "price_narratives",
                "council_debates",
            ]
            for name in collection_names:
                self._collections[name] = self._client.get_or_create_collection(
                    name=name,
                    metadata={"hnsw:space": "cosine"}
                )

            # Local embeddings — downloads once (~80MB)
            self._embedder = self._load_embedder()
            self._initialized = True

            if VERBOSE_DEBUG:
                print(f"[VECTOR] ChromaDB initialized at {VECTOR_DIR}")
                for name, col in self._collections.items():
                    print(f"  {name}: {col.count()} documents")

        except ImportError as e:
            print(f"[VECTOR] Missing package: {e}")
            print("[VECTOR] Run: pip install chromadb sentence-transformers")
            self._initialized = False
        except Exception as e:
            print(f"[VECTOR] Init error: {e}")
            self._initialized = False

    def _load_embedder(self):
        """Load sentence-transformer model locally."""
        try:
            from sentence_transformers import SentenceTransformer
            local_path = EMBED_DIR / "minilm"

            if local_path.exists() and any(local_path.iterdir()):
                if VERBOSE_DEBUG:
                    print(f"[VECTOR] Loading embedder from local cache...")
                return SentenceTransformer(str(local_path))
            else:
                print(f"[VECTOR] Downloading embedding model (once, ~80MB)...")
                model = SentenceTransformer(EMBED_MODEL)
                model.save(str(local_path))
                print(f"[VECTOR] Embedder saved to {local_path}")
                return model

        except ImportError:
            print("[VECTOR] sentence-transformers not installed")
            print("  Run: pip install sentence-transformers")
            return None
        except Exception as e:
            print(f"[VECTOR] Embedder load error: {e}")
            return None

    def _embed(self, text: str) -> list[float] | None:
        """Convert text to embedding vector."""
        if not self._embedder:
            return None
        try:
            vec = self._embedder.encode(text[:1000], normalize_embeddings=True)
            return vec.tolist()
        except Exception as e:
            if VERBOSE_DEBUG:
                print(f"[VECTOR] Embed error: {e}")
            return None

    def _make_id(self, *parts) -> str:
        """Create unique document ID from parts."""
        combined = "_".join(str(p) for p in parts)
        return hashlib.md5(combined.encode()).hexdigest()[:16]

    # ── ADD DOCUMENTS ─────────────────────────────────────────

    def add_analysis(self, symbol: str, bot_name: str,
                     analysis_text: str, score: float,
                     date: str = None, extra_meta: dict = None):
        """
        Store a bot's analysis text in the vector store.
        Each analysis is embedded and stored with metadata.
        """
        self._init()
        if not self._initialized or not self._embedder:
            return

        date = date or datetime.now(IST).strftime('%Y-%m-%d')
        doc_id = self._make_id(symbol, bot_name, date)

        # Rich document text for better retrieval
        doc_text = f"""
Symbol: {symbol}
Bot: {bot_name}
Date: {date}
Score: {score}/10
Analysis: {analysis_text}
""".strip()

        metadata = {
            'symbol': symbol,
            'bot': bot_name,
            'date': date,
            'score': float(score),
            'type': 'stock_analysis',
        }
        if extra_meta:
            metadata.update({k: str(v) for k, v in extra_meta.items()})

        embedding = self._embed(doc_text)
        if embedding is None:
            return

        try:
            self._collections['stock_analyses'].upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[doc_text],
                metadatas=[metadata]
            )
        except Exception as e:
            if VERBOSE_DEBUG:
                print(f"[VECTOR] Add analysis error: {e}")

    def add_sector_analysis(self, sector: str, analysis_text: str,
                             score: float, top_stocks: list,
                             date: str = None):
        """Store sector bot analysis."""
        self._init()
        if not self._initialized:
            return

        date = date or datetime.now(IST).strftime('%Y-%m-%d')
        doc_id = self._make_id("sector", sector, date)

        doc_text = f"""
Sector: {sector}
Date: {date}
Score: {score}/10
Top Stocks: {', '.join(top_stocks[:5])}
Analysis: {analysis_text}
""".strip()

        embedding = self._embed(doc_text)
        if embedding is None:
            return

        try:
            self._collections['sector_analyses'].upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[doc_text],
                metadatas={
                    'sector': sector,
                    'date': date,
                    'score': float(score),
                    'top_stocks': json.dumps(top_stocks[:5]),
                }
            )
        except Exception as e:
            if VERBOSE_DEBUG:
                print(f"[VECTOR] Sector add error: {e}")

    def add_market_analysis(self, analysis_text: str, score: float,
                             outlook: str, date: str = None):
        """Store market-level analysis."""
        self._init()
        if not self._initialized:
            return

        date = date or datetime.now(IST).strftime('%Y-%m-%d')
        doc_id = self._make_id("market", date)

        doc_text = f"Date: {date}\nMarket Score: {score}/10\nOutlook: {outlook}\n{analysis_text}"
        embedding = self._embed(doc_text)
        if embedding is None:
            return

        try:
            self._collections['market_analyses'].upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[doc_text],
                metadatas={'date': date, 'score': float(score), 'outlook': outlook}
            )
        except Exception as e:
            if VERBOSE_DEBUG:
                print(f"[VECTOR] Market add error: {e}")

    def add_news(self, symbol: str, news_text: str,
                 sentiment: float, date: str = None):
        """Store news summary for a stock."""
        self._init()
        if not self._initialized:
            return

        date = date or datetime.now(IST).strftime('%Y-%m-%d')
        doc_id = self._make_id("news", symbol, date)
        doc_text = f"Symbol: {symbol}\nDate: {date}\nSentiment: {sentiment:.3f}\n{news_text}"
        embedding = self._embed(doc_text)
        if embedding is None:
            return

        try:
            self._collections['news_summaries'].upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[doc_text],
                metadatas={'symbol': symbol, 'date': date, 'sentiment': float(sentiment)}
            )
        except Exception as e:
            if VERBOSE_DEBUG:
                print(f"[VECTOR] News add error: {e}")

    def add_council_debate(self, symbol: str, debate_text: str,
                            verdict: str, score: float, date: str = None):
        """Store the full council debate for a stock."""
        self._init()
        if not self._initialized:
            return

        date = date or datetime.now(IST).strftime('%Y-%m-%d')
        doc_id = self._make_id("council", symbol, date)
        doc_text = f"""
Council Debate: {symbol}
Date: {date}
Verdict: {verdict}
Score: {score}/10
{debate_text}
""".strip()

        embedding = self._embed(doc_text)
        if embedding is None:
            return

        try:
            self._collections['council_debates'].upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[doc_text],
                metadatas={
                    'symbol': symbol, 'date': date,
                    'verdict': verdict, 'score': float(score)
                }
            )
        except Exception as e:
            if VERBOSE_DEBUG:
                print(f"[VECTOR] Council add error: {e}")

    # ── QUERY / RETRIEVE ──────────────────────────────────────

    def query_stock_history(self, symbol: str, query: str,
                             bot: str = None, top_k: int = 5,
                             days_back: int = 30) -> list[dict]:
        """
        Query past analyses for a stock.
        Used by bots to read their own history before speaking.

        Example:
          query_stock_history("RELIANCE", "fundamental valuation trend")
          → returns last 5 relevant fundamental analyses
        """
        self._init()
        if not self._initialized or not self._embedder:
            return []

        embedding = self._embed(f"{symbol} {query}")
        if embedding is None:
            return []

        where = {'symbol': symbol}
        if bot:
            where['bot'] = bot

        try:
            results = self._collections['stock_analyses'].query(
                query_embeddings=[embedding],
                n_results=min(top_k, 10),
                where=where,
                include=['documents', 'metadatas', 'distances']
            )

            docs = []
            for doc, meta, dist in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ):
                # Filter by date
                doc_date = meta.get('date', '2000-01-01')
                cutoff = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
                if doc_date >= cutoff:
                    docs.append({
                        'text': doc,
                        'metadata': meta,
                        'relevance': round(1 - dist, 3)
                    })
            return docs

        except Exception as e:
            if VERBOSE_DEBUG:
                print(f"[VECTOR] Query error: {e}")
            return []

    def get_context_for_stock(self, symbol: str, query: str = "",
                               top_k: int = 5) -> str:
        """
        Get a formatted context string for a stock.
        Combines: past analyses + news + council debates.
        Fed directly into LLM prompts as historical context.
        """
        self._init()
        if not self._initialized:
            return ""

        q = f"{symbol} {query}".strip()
        contexts = []

        # Past stock analyses
        analyses = self.query_stock_history(symbol, q, top_k=3)
        if analyses:
            contexts.append("=== PAST BOT ANALYSES ===")
            for a in analyses:
                meta = a['metadata']
                contexts.append(
                    f"[{meta.get('date','')}] {meta.get('bot','').upper()} "
                    f"(Score: {meta.get('score','?')}/10): "
                    f"{a['text'][a['text'].find('Analysis:'):a['text'].find('Analysis:')+300]}"
                )

        # Past news
        news = self._query_collection('news_summaries', q,
                                       where={'symbol': symbol}, top_k=2)
        if news:
            contexts.append("\n=== RECENT NEWS CONTEXT ===")
            for n in news:
                contexts.append(n['text'][:400])

        # Past council debates
        debates = self._query_collection('council_debates', q,
                                          where={'symbol': symbol}, top_k=2)
        if debates:
            contexts.append("\n=== PAST COUNCIL VERDICTS ===")
            for d in debates:
                meta = d['metadata']
                contexts.append(
                    f"[{meta.get('date','')}] Verdict: {meta.get('verdict','')} "
                    f"Score: {meta.get('score','?')}/10"
                )

        return '\n'.join(contexts) if contexts else ""

    def get_sector_context(self, sector: str, top_k: int = 3) -> str:
        """Get past sector analyses as context."""
        self._init()
        if not self._initialized:
            return ""

        results = self._query_collection(
            'sector_analyses',
            sector,
            where={'sector': sector},
            top_k=top_k
        )
        if not results:
            return ""

        lines = ["=== PAST SECTOR ANALYSES ==="]
        for r in results:
            meta = r['metadata']
            lines.append(
                f"[{meta.get('date','')}] Score: {meta.get('score','?')}/10 | "
                f"Top: {meta.get('top_stocks','[]')}"
            )
            lines.append(r['text'][r['text'].find('Analysis:'):r['text'].find('Analysis:')+300])
        return '\n'.join(lines)

    def get_market_context(self, top_k: int = 3) -> str:
        """Get recent market analyses as context."""
        self._init()
        if not self._initialized:
            return ""

        results = self._query_collection('market_analyses', "market outlook India NSE", top_k=top_k)
        if not results:
            return ""

        lines = ["=== RECENT MARKET CONTEXT ==="]
        for r in results:
            meta = r['metadata']
            lines.append(f"[{meta.get('date','')}] Score: {meta.get('score','?')}/10 | {meta.get('outlook','')}")
            lines.append(r['text'][:300])
        return '\n'.join(lines)

    def _query_collection(self, collection_name: str, query: str,
                           where: dict = None, top_k: int = 5) -> list[dict]:
        """Generic collection query."""
        self._init()
        if not self._initialized or not self._embedder:
            return []

        embedding = self._embed(query)
        if embedding is None:
            return []

        try:
            col = self._collections.get(collection_name)
            if col is None or col.count() == 0:
                return []

            kwargs = {
                'query_embeddings': [embedding],
                'n_results': min(top_k, max(1, col.count())),
                'include': ['documents', 'metadatas', 'distances']
            }
            if where:
                kwargs['where'] = where

            results = col.query(**kwargs)
            return [
                {'text': doc, 'metadata': meta, 'relevance': round(1 - dist, 3)}
                for doc, meta, dist in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )
            ]
        except Exception as e:
            if VERBOSE_DEBUG:
                print(f"[VECTOR] Collection query error ({collection_name}): {e}")
            return []

    # ── BULK INDEXING ─────────────────────────────────────────

    def index_all_stored_data(self):
        """
        Bulk index all local files into vector store.
        Run this once after downloading data, or nightly.
        """
        from memory.storage import (
            list_stored_symbols, load_bot_scores_history,
            load_news_text, NEWS_DIR
        )

        print("[VECTOR] Bulk indexing all stored data...")
        indexed = 0

        symbols = list_stored_symbols()
        for sym in symbols:
            # Index past bot analyses
            history = load_bot_scores_history(sym, days=90)
            for entry in history:
                for bot_name, analysis in (entry.get('analysis') or {}).items():
                    if analysis:
                        self.add_analysis(
                            sym, bot_name, analysis,
                            entry['scores'].get(bot_name, 5.0),
                            date=entry.get('date')
                        )
                        indexed += 1

            # Index news
            news_dir = NEWS_DIR / sym
            if news_dir.exists():
                for json_file in sorted(news_dir.glob('*.json'))[-30:]:
                    try:
                        import json as _json
                        with open(json_file) as f:
                            news_data = _json.load(f)
                        articles = news_data.get('articles', [])
                        if articles:
                            text = '\n'.join(
                                f"{a.get('title','')} — {a.get('summary','')[:200]}"
                                for a in articles[:10]
                            )
                            avg_sent = sum(a.get('sentiment', 0) for a in articles) / len(articles)
                            self.add_news(sym, text, avg_sent, date=json_file.stem)
                            indexed += 1
                    except Exception:
                        pass

        print(f"[VECTOR] Indexed {indexed} documents")
        return indexed

    def get_stats(self) -> dict:
        """Get vector store statistics."""
        self._init()
        if not self._initialized:
            return {'status': 'not initialized'}

        stats = {'status': 'ready', 'collections': {}}
        for name, col in self._collections.items():
            stats['collections'][name] = col.count()
        return stats


# ── Singleton ─────────────────────────────────────────────────
_instance: VectorMemory = None

def get_vector_store() -> VectorMemory:
    global _instance
    if _instance is None:
        _instance = VectorMemory()
    return _instance
