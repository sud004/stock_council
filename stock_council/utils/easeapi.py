# ============================================================
# utils/easeapi.py — Ventura EaseAPI Complete Integration
# ============================================================
#
# WHAT THIS REPLACES:
#   ❌ Yahoo Finance (unreliable, rate-limited, not Indian)
#   ❌ NSE India scraping (Cloudflare blocks, fragile)
#   ❌ Finnhub (limited Indian stock support)
#   ✅ Ventura EaseAPI — direct exchange data, your own account
#
# ENDPOINTS USED:
#   Auth:        POST /login/v1/authorization/token
#   OHLCV:       POST /instrument/v1/ohlcv          (up to 1000 stocks/call)
#   Depth:       POST /instrument/v1/ltp_depth       (order book + LTP)
#   Instruments: GET  /instrument/v1/instruments     (NSE token master)
#   Holdings:    GET  /portfolio/v1/holdings         (your demat holdings)
#   Positions:   GET  /portfolio/v1/positions        (intraday positions)
#   Funds:       GET  /user/v1/funds_details         (available margin)
#   WebSocket:   wss://easeapi-ws.venturasecurities.com/v1/easeapi_mktdata
#
# AUTHENTICATION FLOW:
#   1. Open login URL in browser → Ventura login page
#   2. After login, Ventura redirects to your redirect_url with request_token
#   3. POST request_token + HMAC-SHA256(request_token + app_secret) → get auth_token
#   4. Use auth_token as Bearer in all subsequent calls
#   5. Token valid for ~24 hours, refresh using refresh_token
#
# SETUP:
#   Add to your .env:
#     EASEAPI_APP_KEY=your_app_key
#     EASEAPI_APP_SECRET=your_app_secret
#     EASEAPI_CLIENT_ID=your_client_id   (e.g. AA0605)
#     EASEAPI_REDIRECT_URL=http://localhost:8080
# ============================================================

import os
import json
import time
import hashlib
import hmac
import asyncio
import threading
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Callable, Optional
import pytz

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BASE_DIR, DATA_DIR, VERBOSE_DEBUG
from dotenv import load_dotenv

load_dotenv(BASE_DIR / ".env")
IST = pytz.timezone('Asia/Kolkata')

# ── Config from .env ──────────────────────────────────────────
APP_KEY       = os.getenv("EASEAPI_APP_KEY", "")
APP_SECRET    = os.getenv("EASEAPI_APP_SECRET", "")
CLIENT_ID     = os.getenv("EASEAPI_CLIENT_ID", "")
REDIRECT_URL  = os.getenv("EASEAPI_REDIRECT_URL", "http://localhost:8080")

# ── API Base URLs ─────────────────────────────────────────────
BASE_URL      = "https://easeapi.venturasecurities.com"
WS_URL        = "wss://easeapi-ws.venturasecurities.com/v1/easeapi_mktdata"

# ── Token storage ─────────────────────────────────────────────
TOKEN_FILE    = DATA_DIR / "easeapi_token.json"

# ── Instrument token cache ────────────────────────────────────
INSTRUMENT_CACHE_FILE = DATA_DIR / "easeapi_instruments.csv"
_instrument_df: pd.DataFrame = None   # loaded once


# ══════════════════════════════════════════════════════════════
# AUTHENTICATION
# ══════════════════════════════════════════════════════════════

class EaseAuth:
    """
    Manages Ventura EaseAPI authentication.
    Tokens are cached to disk and reused until expiry.
    """

    def __init__(self):
        self._auth_token    = None
        self._refresh_token = None
        self._expiry        = None
        self._client_id     = CLIENT_ID
        self._load_token()

    def _load_token(self):
        """Load cached token from disk."""
        if TOKEN_FILE.exists():
            try:
                with open(TOKEN_FILE) as f:
                    data = json.load(f)
                self._auth_token    = data.get('auth_token')
                self._refresh_token = data.get('refresh_token')
                self._client_id     = data.get('client_id', CLIENT_ID)
                expiry_str          = data.get('auth_expiry', '')
                if expiry_str:
                    self._expiry = datetime.strptime(expiry_str, '%Y-%m-%d %H:%M:%S')
                    self._expiry = IST.localize(self._expiry)
                if VERBOSE_DEBUG:
                    print(f"[EASE] Loaded cached token, expires: {self._expiry}")
            except Exception as e:
                if VERBOSE_DEBUG:
                    print(f"[EASE] Token load error: {e}")

    def _save_token(self, data: dict):
        """Save token to disk."""
        with open(TOKEN_FILE, 'w') as f:
            json.dump(data, f, indent=2)

    def is_valid(self) -> bool:
        """Check if current token is valid and not expired."""
        if not self._auth_token:
            return False
        if self._expiry:
            now = datetime.now(IST)
            # Treat as expired 5 minutes before actual expiry
            return now < self._expiry - timedelta(minutes=5)
        return True

    def get_login_url(self) -> str:
        """
        Step 1: Get the login URL to open in browser.
        User logs in, Ventura redirects to REDIRECT_URL?request_token=xxx&state=xxx
        """
        import secrets
        state = secrets.token_hex(8)
        return (
            f"{BASE_URL}/auth/v1/login"
            f"?app_key={APP_KEY}&state={state}"
            f"&redirect_url={REDIRECT_URL}"
        )

    def exchange_token(self, request_token: str) -> bool:
        """
        Step 3: Exchange request_token for auth_token.
        HMAC-SHA256 of (request_token + app_secret) using app_secret as key.
        """
        # Build the checksum: SHA256(request_token + app_secret)
        checksum = hashlib.sha256(
            (request_token + APP_SECRET).encode('utf-8')
        ).hexdigest()

        url = f"{BASE_URL}/login/v1/authorization/token"
        headers = {
            'x-app-key': APP_KEY,
            'Content-Type': 'application/json'
        }
        payload = {
            'request_token': request_token,
            'data': checksum
        }

        try:
            r = requests.post(url, json=payload, headers=headers, timeout=15)
            r.raise_for_status()
            data = r.json()

            self._auth_token    = data['auth_token']
            self._refresh_token = data.get('refresh_token')
            self._client_id     = data['client_id']
            expiry_str          = data.get('auth_expiry', '')

            if expiry_str:
                self._expiry = IST.localize(
                    datetime.strptime(expiry_str, '%Y-%m-%d %H:%M:%S')
                )

            self._save_token({
                'auth_token':    self._auth_token,
                'refresh_token': self._refresh_token,
                'client_id':     self._client_id,
                'auth_expiry':   expiry_str,
            })

            print(f"[EASE] ✅ Logged in as {self._client_id}, expires {expiry_str}")
            return True

        except Exception as e:
            print(f"[EASE] ❌ Token exchange failed: {e}")
            return False

    @property
    def token(self) -> str:
        return self._auth_token

    @property
    def client_id(self) -> str:
        return self._client_id

    def get_headers(self) -> dict:
        """Return headers for authenticated API calls."""
        from utils.easeapi_auth import get_headers as _get_headers
        return _get_headers()


# ── Global auth singleton ─────────────────────────────────────
_auth = EaseAuth()


def login_interactive():
    """
    Interactive login flow.
    Opens browser → user logs in → paste request_token.
    """
    if _auth.is_valid():
        print(f"[EASE] Already logged in as {_auth.client_id}")
        return True

    print("\n[EASE] LOGIN REQUIRED")
    print("─" * 50)
    print(f"1. Open this URL in your browser:")
    print(f"\n   {_auth.get_login_url()}\n")
    print("2. Log in with your Ventura credentials")
    print("3. After login, copy the 'request_token' from the redirect URL")
    print("   (URL looks like: http://localhost:8080?request_token=XXXXX&state=...)\n")

    request_token = input("Paste request_token here: ").strip()
    if not request_token:
        print("[EASE] No token entered. Aborting.")
        return False

    return _auth.exchange_token(request_token)


def login_totp(client_id: str, password: str, totp: str) -> bool:
    """
    Programmatic login with TOTP (2FA).
    Use this for automated/scheduled runs without browser.

    Setup TOTP in Ventura app settings first.
    Use pyotp to generate TOTP codes automatically:
        pip install pyotp
        totp_secret = "YOUR_TOTP_SECRET_FROM_VENTURA"
        import pyotp
        totp = pyotp.TOTP(totp_secret).now()
    """
    if _auth.is_valid():
        return True

    url = f"{BASE_URL}/auth/v1/login-with-totp"
    payload = {
        'client_id': client_id,
        'password':  password,
        'totp':      totp,
        'app_key':   APP_KEY,
    }
    try:
        r = requests.post(url, json=payload, timeout=15,
                          headers={'Content-Type': 'application/json'})
        r.raise_for_status()
        data = r.json()
        request_token = data.get('request_token')
        if request_token:
            return _auth.exchange_token(request_token)
        else:
            print(f"[EASE] TOTP login failed: {data}")
            return False
    except Exception as e:
        print(f"[EASE] TOTP login error: {e}")
        return False


def require_auth(fn):
    """Decorator: auto-login if needed, then call function."""
    def wrapper(*args, **kwargs):
        from utils.easeapi_auth import ensure_logged_in
        ensure_logged_in()   # auto-renews via TOTP if expired
        return fn(*args, **kwargs)
    return wrapper


# ══════════════════════════════════════════════════════════════
# INSTRUMENT MASTER
# ══════════════════════════════════════════════════════════════

def load_instruments(force_refresh: bool = False) -> pd.DataFrame:
    """
    Download and cache the NSE/BSE instrument master CSV.
    Maps symbol → exchange_token (needed for all market data calls).

    Returns DataFrame with columns:
      exchange, exchange_token, trading_symbol, instrument_name,
      lot_size, tick_size, instrument_type, expiry, strike, option_type
    """
    global _instrument_df

    if _instrument_df is not None and not force_refresh:
        return _instrument_df

    # Check local cache (refresh daily)
    if INSTRUMENT_CACHE_FILE.exists() and not force_refresh:
        age_hours = (time.time() - INSTRUMENT_CACHE_FILE.stat().st_mtime) / 3600
        if age_hours < 24:
            _instrument_df = pd.read_csv(INSTRUMENT_CACHE_FILE, low_memory=False)
            if VERBOSE_DEBUG:
                print(f"[EASE] Instruments from cache: {len(_instrument_df)} rows")
            return _instrument_df

    # Download fresh
    url = f"{BASE_URL}/instrument/v1/instruments"
    try:
        # Instruments CSV is a public endpoint — no auth needed
        r = requests.get(url, timeout=30)
        r.raise_for_status()

        # Save raw CSV
        INSTRUMENT_CACHE_FILE.write_bytes(r.content)
        _instrument_df = pd.read_csv(INSTRUMENT_CACHE_FILE, low_memory=False)
        print(f"[EASE] Downloaded instruments: {len(_instrument_df)} rows")
        return _instrument_df

    except Exception as e:
        print(f"[EASE] Instrument download error: {e}")
        if INSTRUMENT_CACHE_FILE.exists():
            _instrument_df = pd.read_csv(INSTRUMENT_CACHE_FILE, low_memory=False)
            return _instrument_df
        return pd.DataFrame()


def get_token_for_symbol(symbol: str, exchange: str = 'NSE') -> str | None:
    """
    Get exchange_token for a trading symbol.
    e.g. get_token_for_symbol('RELIANCE', 'NSE') → '2885'
    """
    df = load_instruments()
    if df.empty:
        return None

    sym = symbol.replace('.NS', '').replace('.BO', '').upper()
    exch = 'BSE' if '.BO' in symbol else exchange

    # Try exact match first
    mask = (
        (df['trading_symbol'].str.upper() == sym) &
        (df['exchange'].str.upper() == exch) &
        (df['instrument_type'].str.upper().isin(['EQ', 'ETF', 'INDEX']) |
         df['instrument_type'].isna())
    )
    result = df[mask]

    if result.empty:
        # Fuzzy: starts with
        mask2 = (
            df['trading_symbol'].str.upper().str.startswith(sym) &
            (df['exchange'].str.upper() == exch)
        )
        result = df[mask2]

    if not result.empty:
        return str(result.iloc[0]['exchange_token'])
    return None


def build_token_map(symbols: list, exchange: str = 'NSE') -> dict:
    """
    Build {symbol: exchange_token} map for a list of symbols.
    e.g. {'RELIANCE': '2885', 'TCS': '2244', ...}
    """
    df = load_instruments()
    if df.empty:
        return {}

    token_map = {}
    for sym in symbols:
        clean = sym.replace('.NS', '').replace('.BO', '').upper()
        exch = 'BSE' if '.BO' in sym else exchange
        tok = get_token_for_symbol(clean, exch)
        if tok:
            token_map[clean] = tok
        elif VERBOSE_DEBUG:
            print(f"[EASE] No token for {clean}")
    return token_map


# ══════════════════════════════════════════════════════════════
# MARKET QUOTES — LIVE OHLCV
# ══════════════════════════════════════════════════════════════

@require_auth
def get_ohlcv_batch(tokens: list, exchange: str = 'NSE') -> list:
    """
    Get live OHLCV + LTP for up to 1000 instruments in one call.

    Args:
        tokens: list of exchange_tokens OR index names e.g. ['2885', 'Nifty 50']
        exchange: 'NSE' or 'BSE'

    Returns list of:
        [token, ltp, open, high, low, close, volume, timestamp,
         upper_circuit, lower_circuit]
    """
    url = f"{BASE_URL}/instrument/v1/ohlcv"
    payload = {'exchange': exchange, 'tokens': tokens}

    try:
        r = requests.post(url, json=payload, headers=_auth.get_headers(), timeout=15)
        r.raise_for_status()
        data = r.json()
        if data.get('success'):
            return data.get('data', [])
        else:
            print(f"[EASE] OHLCV error: {data.get('message')}")
            return []
    except Exception as e:
        print(f"[EASE] OHLCV request error: {e}")
        return []


@require_auth
def get_market_depth_batch(tokens: list, exchange: str = 'NSE') -> list:
    """
    Get live OHLCV + order book depth (5 bid/ask levels) for instruments.

    Returns list of:
        [token, ltp, open, high, low, close, volume, timestamp,
         upper_circuit, lower_circuit, total_buy_qty, total_sell_qty,
         [[buy_qty, sell_qty, buy_orders, sell_orders, buy_price, sell_price], ...]]
    """
    url = f"{BASE_URL}/instrument/v1/ltp_depth"
    payload = {'exchange': exchange, 'tokens': tokens}

    try:
        r = requests.post(url, json=payload, headers=_auth.get_headers(), timeout=15)
        r.raise_for_status()
        data = r.json()
        if data.get('success'):
            return data.get('data', [])
        else:
            print(f"[EASE] Depth error: {data.get('message')}")
            return []
    except Exception as e:
        print(f"[EASE] Depth request error: {e}")
        return []


def parse_ohlcv_response(raw: list) -> dict:
    """
    Parse the raw OHLCV list response into a clean dict.
    Input:  [token, ltp, open, high, low, close, volume, timestamp, uc, lc]
    Output: {'token': ..., 'ltp': ..., 'open': ..., ...}
    """
    if not raw or len(raw) < 8:
        return {}
    return {
        'token':          str(raw[0]),
        'ltp':            raw[1],
        'open':           raw[2],
        'high':           raw[3],
        'low':            raw[4],
        'close':          raw[5],
        'volume':         raw[6],
        'timestamp':      raw[7],
        'upper_circuit':  raw[8] if len(raw) > 8 else None,
        'lower_circuit':  raw[9] if len(raw) > 9 else None,
        'change':         round(raw[1] - raw[5], 2) if raw[1] and raw[5] else None,
        'pct_change':     round((raw[1] - raw[5]) / raw[5] * 100, 2)
                          if raw[1] and raw[5] and raw[5] != 0 else None,
    }


def parse_depth_response(raw: list) -> dict:
    """Parse market depth response."""
    base = parse_ohlcv_response(raw[:10])
    if not base:
        return {}
    if len(raw) >= 13:
        base['total_buy_qty']  = raw[10]
        base['total_sell_qty'] = raw[11]
        depth_levels = raw[12] if len(raw) > 12 else []
        base['depth'] = [
            {
                'buy_qty':    lvl[0], 'sell_qty':    lvl[1],
                'buy_orders': lvl[2], 'sell_orders': lvl[3],
                'buy_price':  lvl[4], 'sell_price':  lvl[5],
            }
            for lvl in depth_levels
        ]
        # Buy/sell ratio from depth
        total_b = sum(lvl[0] for lvl in depth_levels) if depth_levels else 0
        total_s = sum(lvl[1] for lvl in depth_levels) if depth_levels else 0
        base['buy_sell_ratio'] = round(total_b / total_s, 3) if total_s > 0 else None
    return base


# ══════════════════════════════════════════════════════════════
# HIGH-LEVEL: GET ALL STOCK QUOTES AT ONCE
# ══════════════════════════════════════════════════════════════

def get_all_quotes(symbols: list, exchange: str = 'NSE',
                   with_depth: bool = False) -> dict:
    """
    Get live quotes for ALL symbols in one or a few API calls.
    Handles batching (max 1000 per call).

    Args:
        symbols: list of NSE symbols e.g. ['RELIANCE', 'TCS', 'INFY']
        exchange: 'NSE' or 'BSE'
        with_depth: if True, also get order book (slower)

    Returns: {symbol: {ltp, open, high, low, close, volume, pct_change, ...}}
    """
    # Build token map
    token_map = build_token_map(symbols, exchange)
    if not token_map:
        print("[EASE] No tokens found. Run load_instruments() first.")
        return {}

    # Reverse map: token → symbol
    reverse_map = {v: k for k, v in token_map.items()}
    tokens = list(token_map.values())

    # Batch into 1000-token chunks
    results = {}
    for i in range(0, len(tokens), 1000):
        batch = tokens[i:i+1000]
        fetch_fn = get_market_depth_batch if with_depth else get_ohlcv_batch
        raw_list = fetch_fn(batch, exchange)

        for raw in raw_list:
            if not raw:
                continue
            parsed = parse_depth_response(raw) if with_depth else parse_ohlcv_response(raw)
            token_str = str(parsed.get('token', ''))
            sym = reverse_map.get(token_str, token_str)
            results[sym] = parsed

    if VERBOSE_DEBUG:
        print(f"[EASE] Got quotes for {len(results)}/{len(symbols)} symbols")
    return results


def get_index_quotes() -> dict:
    """
    Get live values for all major indices.
    Returns: {'Nifty 50': {ltp, change, pct_change, ...}, ...}
    """
    index_names = [
        'Nifty 50', 'Nifty Bank', 'Nifty IT', 'Nifty FMCG',
        'Nifty Pharma', 'Nifty Auto', 'Nifty Metal', 'Nifty Realty',
        'Nifty Energy', 'Nifty Midcap 50', 'India VIX', 'Sensex'
    ]

    raw_list = get_ohlcv_batch(index_names, exchange='NSE')
    results = {}
    for raw in raw_list:
        parsed = parse_ohlcv_response(raw)
        name = str(parsed.get('token', ''))
        results[name] = parsed
    return results


# ══════════════════════════════════════════════════════════════
# PORTFOLIO DATA
# ══════════════════════════════════════════════════════════════

@require_auth
def get_holdings() -> list:
    """
    Get demat holdings from your Ventura account.
    Returns list of {symbol, quantity, avg_price, ltp, pnl, pnl_pct}
    """
    url = f"{BASE_URL}/portfolio/v1/holdings"
    try:
        r = requests.get(url, headers=_auth.get_headers(), timeout=10)
        r.raise_for_status()
        data = r.json()
        holdings = data.get('data', [])
        result = []
        for h in holdings:
            qty = h.get('quantity', 0)
            avg = h.get('average_price', 0)
            ltp = h.get('ltp', 0)
            pnl = (ltp - avg) * qty if ltp and avg else 0
            pnl_pct = (ltp - avg) / avg * 100 if avg and avg != 0 else 0
            result.append({
                'symbol':        h.get('trading_symbol', ''),
                'exchange':      h.get('exchange', ''),
                'quantity':      qty,
                'avg_price':     avg,
                'ltp':           ltp,
                'current_value': ltp * qty if ltp else 0,
                'invested':      avg * qty,
                'pnl':           round(pnl, 2),
                'pnl_pct':       round(pnl_pct, 2),
                'exchange_token': h.get('exchange_token', ''),
            })
        return result
    except Exception as e:
        print(f"[EASE] Holdings error: {e}")
        return []


@require_auth
def get_positions() -> list:
    """Get intraday positions."""
    url = f"{BASE_URL}/portfolio/v1/positions"
    try:
        r = requests.get(url, headers=_auth.get_headers(), timeout=10)
        r.raise_for_status()
        return r.json().get('data', [])
    except Exception as e:
        print(f"[EASE] Positions error: {e}")
        return []


@require_auth
def get_funds() -> dict:
    """Get available funds / margin."""
    url = f"{BASE_URL}/user/v1/funds_details"
    try:
        r = requests.get(url, headers=_auth.get_headers(), timeout=10)
        r.raise_for_status()
        data = r.json()
        return data.get('data', {})
    except Exception as e:
        print(f"[EASE] Funds error: {e}")
        return {}


# ══════════════════════════════════════════════════════════════
# WEBSOCKET — REAL-TIME STREAMING
# ══════════════════════════════════════════════════════════════

class EaseWebSocket:
    """
    WebSocket client for real-time market data.
    Streams live LTP ticks for subscribed symbols.

    Usage:
        ws = EaseWebSocket()
        ws.subscribe(['RELIANCE', 'TCS'], on_tick=my_callback)
        ws.start()  # non-blocking, runs in background thread
    """

    def __init__(self):
        self._ws = None
        self._thread = None
        self._running = False
        self._token_map = {}
        self._reverse_map = {}
        self._callbacks: list[Callable] = []
        self._tick_store: dict = {}   # latest tick per symbol

    def subscribe(self, symbols: list, on_tick: Callable = None,
                  with_depth: bool = False, exchange: str = 'NSE'):
        """
        Subscribe to real-time ticks for symbols.

        on_tick(symbol, tick_data) called on every price update.
        tick_data = {symbol, ltp, open, high, low, close, volume, timestamp}
        """
        self._token_map = build_token_map(symbols, exchange)
        self._reverse_map = {v: k for k, v in self._token_map.items()}
        self._exchange = exchange.lower()
        self._with_depth = with_depth
        if on_tick:
            self._callbacks.append(on_tick)

        print(f"[EASE WS] Subscribed to {len(self._token_map)} symbols")

    def add_callback(self, fn: Callable):
        self._callbacks.append(fn)

    def get_latest(self, symbol: str) -> dict:
        """Get latest tick for a symbol (from memory)."""
        return self._tick_store.get(symbol.upper(), {})

    def get_all_latest(self) -> dict:
        """Get all latest ticks."""
        return dict(self._tick_store)

    def start(self, block: bool = False):
        """
        Start WebSocket connection.
        block=False: runs in background thread
        block=True:  blocks current thread
        """
        if not _auth.is_valid():
            print("[EASE WS] Not authenticated. Login first.")
            return

        if block:
            self._run()
        else:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            print("[EASE WS] Started in background")

    def stop(self):
        self._running = False
        if self._ws:
            self._ws.close()

    def _run(self):
        """Internal WebSocket loop."""
        try:
            import websocket as ws_lib
        except ImportError:
            print("[EASE WS] Install websocket-client: pip install websocket-client")
            return

        self._running = True
        action_type = 'ltp_depth' if self._with_depth else 'ltp'

        ws_url = (
            f"{WS_URL}"
            f"?app_key={APP_KEY}"
            f"&client_id={_auth.client_id}"
            f"&authorization={_auth.token}"
        )

        def on_open(ws):
            print(f"[EASE WS] Connected")
            # Subscribe to all tokens
            for token in self._token_map.values():
                msg = json.dumps({
                    "actions": [f"{self._exchange}:{action_type}"],
                    "token": [token],
                    "mode": "sub"
                })
                ws.send(msg)

        def on_message(ws, message):
            try:
                data = json.loads(message)
                if not isinstance(data, list) or len(data) < 3:
                    return

                action = data[0]
                token_str = str(data[1])
                symbol = self._reverse_map.get(token_str, token_str)

                if 'ltp_depth' in action:
                    tick = parse_depth_response(data[1:])
                else:
                    tick = parse_ohlcv_response(data[1:])

                tick['symbol'] = symbol
                self._tick_store[symbol] = tick

                for cb in self._callbacks:
                    try:
                        cb(symbol, tick)
                    except Exception as e:
                        if VERBOSE_DEBUG:
                            print(f"[EASE WS] Callback error: {e}")

            except Exception as e:
                if VERBOSE_DEBUG:
                    print(f"[EASE WS] Message parse error: {e}")

        def on_error(ws, error):
            print(f"[EASE WS] Error: {error}")

        def on_close(ws, code, msg):
            print(f"[EASE WS] Closed: {code} {msg}")
            self._running = False
            # Auto-reconnect after 5 seconds
            if self._running:
                time.sleep(5)
                self._run()

        self._ws = ws_lib.WebSocketApp(
            ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        self._ws.run_forever(
            ping_interval=30,
            ping_timeout=10,
            reconnect=5
        )


# ── Global WebSocket singleton ────────────────────────────────
_ws_instance: EaseWebSocket = None

def get_websocket() -> EaseWebSocket:
    global _ws_instance
    if _ws_instance is None:
        _ws_instance = EaseWebSocket()
    return _ws_instance


# ══════════════════════════════════════════════════════════════
# INTEGRATION WITH EXISTING PIPELINE
# ══════════════════════════════════════════════════════════════

def get_live_quote_ease(symbol: str) -> dict:
    """
    Drop-in replacement for get_live_quote() in live_data.py
    Uses EaseAPI instead of NSE scraping.
    Returns same format as the old function.
    """
    if not _auth.is_valid():
        # Fall back to old method
        from utils.live_data import get_live_quote as old_fn
        return old_fn(symbol)

    sym = symbol.replace('.NS', '').replace('.BO', '').upper()
    quotes = get_all_quotes([sym])
    q = quotes.get(sym, {})
    if not q:
        return {'symbol': symbol, 'last_price': None, 'source': 'EaseAPI (no data)'}

    return {
        'symbol':     symbol,
        'last_price': q.get('ltp'),
        'open':       q.get('open'),
        'high':       q.get('high'),
        'low':        q.get('low'),
        'prev_close': q.get('close'),
        'change':     q.get('change'),
        'pct_change': q.get('pct_change'),
        'volume':     q.get('volume'),
        'upper_circuit': q.get('upper_circuit'),
        'lower_circuit': q.get('lower_circuit'),
        'timestamp':  q.get('timestamp'),
        'source':     'Ventura EaseAPI',
    }


def get_all_sector_quotes_ease(symbols: list) -> dict:
    """
    Get live quotes for all stocks in a sector scan in ONE API call.
    This is the killer feature — instead of 286 separate calls,
    we do ceil(286/1000) = 1 call for the entire market.

    Returns {symbol: {ltp, pct_change, volume, ...}}
    """
    return get_all_quotes(symbols, with_depth=False)


def get_nse_indices_ease() -> list:
    """
    Get live index values from EaseAPI.
    Returns same format as NSELive.get_all_indices().
    """
    raw = get_index_quotes()
    result = []
    for name, data in raw.items():
        result.append({
            'name':       name,
            'last':       data.get('ltp'),
            'change':     data.get('change'),
            'pct_change': data.get('pct_change'),
            'year_high':  data.get('high'),
            'year_low':   data.get('low'),
        })
    return result


# ══════════════════════════════════════════════════════════════
# STATUS & DIAGNOSTICS
# ══════════════════════════════════════════════════════════════

def status() -> dict:
    """Check EaseAPI connection and auth status."""
    return {
        'authenticated':    _auth.is_valid(),
        'client_id':        _auth.client_id,
        'token_expiry':     str(_auth._expiry) if _auth._expiry else 'unknown',
        'app_key_set':      bool(APP_KEY) and not APP_KEY.startswith('YOUR'),
        'instruments_cached': INSTRUMENT_CACHE_FILE.exists(),
        'instruments_count': len(load_instruments()) if INSTRUMENT_CACHE_FILE.exists() else 0,
    }


if __name__ == "__main__":
    # Quick test
    print("EaseAPI Status:", json.dumps(status(), indent=2))
    if not _auth.is_valid():
        login_interactive()
    else:
        print("\nFetching RELIANCE quote...")
        q = get_live_quote_ease("RELIANCE")
        print(json.dumps(q, indent=2))


# ══════════════════════════════════════════════════════════════
# UPDATED AUTH FUNCTIONS — uses easeapi_auth.py
# ══════════════════════════════════════════════════════════════

def get_headers_auto() -> dict:
    """
    Get authenticated headers. Auto-refreshes token if expired.
    Uses the new TOTP-based auto-auth module.
    """
    from utils.easeapi_auth import get_headers
    return get_headers()


def is_authenticated() -> bool:
    """Check if authenticated without triggering login."""
    from utils.easeapi_auth import is_authenticated as _is_auth
    return _is_auth()


# ══════════════════════════════════════════════════════════════
# CONVENIENCE: complete status with auth check
# ══════════════════════════════════════════════════════════════

def full_status() -> dict:
    """Comprehensive status check including auth."""
    from utils.easeapi_auth import load_token, is_authenticated
    token = load_token()
    s = status()
    s['token_valid']  = is_authenticated()
    s['token_expiry'] = token.get('auth_expiry') if token else 'none'
    s['client_id']    = token.get('client_id') if token else 'not logged in'
    return s
