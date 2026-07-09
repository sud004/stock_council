# ============================================================
# utils/database.py — SQLite caching and storage
# ============================================================

import sqlite3
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH, VERBOSE_DEBUG


def get_connection():
    """Return a thread-safe SQLite connection."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all tables if they don't exist."""
    conn = get_connection()
    c = conn.cursor()

    # Price / OHLCV cache
    c.execute("""
        CREATE TABLE IF NOT EXISTS price_cache (
            symbol      TEXT NOT NULL,
            date        TEXT NOT NULL,
            open        REAL,
            high        REAL,
            low         REAL,
            close       REAL,
            volume      INTEGER,
            fetched_at  REAL,
            PRIMARY KEY (symbol, date)
        )
    """)

    # Fundamental data cache
    c.execute("""
        CREATE TABLE IF NOT EXISTS fundamental_cache (
            symbol      TEXT PRIMARY KEY,
            data_json   TEXT,
            fetched_at  REAL
        )
    """)

    # News cache
    c.execute("""
        CREATE TABLE IF NOT EXISTS news_cache (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol      TEXT,
            title       TEXT,
            summary     TEXT,
            source      TEXT,
            url         TEXT,
            published   TEXT,
            sentiment   REAL,
            fetched_at  REAL
        )
    """)

    # Full analysis results
    c.execute("""
        CREATE TABLE IF NOT EXISTS analysis_results (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol          TEXT,
            bot             TEXT,
            score           REAL,
            analysis_text   TEXT,
            raw_data_json   TEXT,
            created_at      REAL
        )
    """)

    # Verdict history
    c.execute("""
        CREATE TABLE IF NOT EXISTS verdicts (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol          TEXT,
            verdict         TEXT,
            composite_score REAL,
            summary         TEXT,
            scores_json     TEXT,
            created_at      REAL
        )
    """)

    conn.commit()
    conn.close()
    if VERBOSE_DEBUG:
        print("[DB] Initialized database tables.")


# ── Price Cache ───────────────────────────────────────────────

def save_prices(symbol: str, df):
    """Save OHLCV dataframe to cache."""
    conn = get_connection()
    now = time.time()
    rows = []
    for idx, row in df.iterrows():
        date_str = str(idx.date()) if hasattr(idx, 'date') else str(idx)
        rows.append((
            symbol, date_str,
            float(row.get('Open', 0)),
            float(row.get('High', 0)),
            float(row.get('Low', 0)),
            float(row.get('Close', 0)),
            int(row.get('Volume', 0)),
            now
        ))
    conn.executemany("""
        INSERT OR REPLACE INTO price_cache
        (symbol, date, open, high, low, close, volume, fetched_at)
        VALUES (?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()
    conn.close()


def load_prices(symbol: str, max_age_hours: float = 1.0):
    """Load cached prices if fresh enough. Returns list of dicts or None."""
    conn = get_connection()
    cutoff = time.time() - max_age_hours * 3600
    rows = conn.execute("""
        SELECT * FROM price_cache
        WHERE symbol = ? AND fetched_at > ?
        ORDER BY date ASC
    """, (symbol, cutoff)).fetchall()
    conn.close()
    if not rows:
        return None
    return [dict(r) for r in rows]


# ── Fundamental Cache ─────────────────────────────────────────

def save_fundamentals(symbol: str, data: dict):
    conn = get_connection()
    conn.execute("""
        INSERT OR REPLACE INTO fundamental_cache (symbol, data_json, fetched_at)
        VALUES (?, ?, ?)
    """, (symbol, json.dumps(data), time.time()))
    conn.commit()
    conn.close()


def load_fundamentals(symbol: str, max_age_hours: float = 24.0):
    conn = get_connection()
    cutoff = time.time() - max_age_hours * 3600
    row = conn.execute("""
        SELECT * FROM fundamental_cache
        WHERE symbol = ? AND fetched_at > ?
    """, (symbol, cutoff)).fetchone()
    conn.close()
    if row:
        return json.loads(row['data_json'])
    return None


# ── News Cache ────────────────────────────────────────────────

def save_news(symbol: str, articles: list):
    conn = get_connection()
    now = time.time()
    for a in articles:
        conn.execute("""
            INSERT INTO news_cache
            (symbol, title, summary, source, url, published, sentiment, fetched_at)
            VALUES (?,?,?,?,?,?,?,?)
        """, (
            symbol,
            a.get('title', ''),
            a.get('summary', ''),
            a.get('source', ''),
            a.get('url', ''),
            a.get('published', ''),
            float(a.get('sentiment', 0.0)),
            now
        ))
    conn.commit()
    conn.close()


def load_news(symbol: str, max_age_hours: float = 2.0, limit: int = 20):
    conn = get_connection()
    cutoff = time.time() - max_age_hours * 3600
    rows = conn.execute("""
        SELECT * FROM news_cache
        WHERE symbol = ? AND fetched_at > ?
        ORDER BY fetched_at DESC
        LIMIT ?
    """, (symbol, cutoff, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Analysis Results ──────────────────────────────────────────

def save_analysis(symbol: str, bot: str, score: float, text: str, raw_data: dict = None):
    conn = get_connection()
    conn.execute("""
        INSERT INTO analysis_results
        (symbol, bot, score, analysis_text, raw_data_json, created_at)
        VALUES (?,?,?,?,?,?)
    """, (symbol, bot, score, text, json.dumps(raw_data or {}), time.time()))
    conn.commit()
    conn.close()


def save_verdict(symbol: str, verdict: str, score: float, summary: str, scores: dict):
    conn = get_connection()
    conn.execute("""
        INSERT INTO verdicts
        (symbol, verdict, composite_score, summary, scores_json, created_at)
        VALUES (?,?,?,?,?,?)
    """, (symbol, verdict, score, summary, json.dumps(scores), time.time()))
    conn.commit()
    conn.close()


def get_verdict_history(symbol: str, limit: int = 10):
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM verdicts
        WHERE symbol = ?
        ORDER BY created_at DESC
        LIMIT ?
    """, (symbol, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# Initialize on import
init_db()
