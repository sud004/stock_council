# ============================================================
# utils/llm.py — Local LLM via Ollama
# ============================================================
# TOKEN FIX:
#   stream_chat() now accepts `num_predict` and `stop` parameters
#   so each call type uses the exact token budget it needs.
#   The old global LLM_MAX_TOKENS=1500 caused every call —
#   including 50-word cross-questions — to generate up to 1500
#   tokens. That alone was the primary cause of 7-hour runtimes.
# ============================================================

import requests
import json
import time
import threading
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    OLLAMA_HOST, OLLAMA_MODEL, LLM_TEMPERATURE,
    LLM_TOP_P, LLM_MAX_TOKENS, LLM_CONTEXT_WINDOW,
    LLM_TOKENS, VERBOSE_DEBUG
)

# Stall guard: if no token arrives in this many seconds → abort call
TOKEN_STALL_TIMEOUT = 45    # seconds between tokens
CALL_HARD_TIMEOUT   = 120   # absolute wall-clock limit per call


def check_ollama() -> bool:
    """Check if Ollama is running and the model is available."""
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m['name'].split(':')[0] for m in r.json().get('models', [])]
            base = OLLAMA_MODEL.split(':')[0]
            if base in models or OLLAMA_MODEL in models:
                return True
            print(f"[LLM] WARNING: Model '{OLLAMA_MODEL}' not found.")
            print(f"[LLM] Available: {models}")
            print(f"[LLM] Run: ollama pull {OLLAMA_MODEL}")
            return False
        return False
    except requests.ConnectionError:
        print(f"[LLM] Cannot connect to Ollama at {OLLAMA_HOST}")
        print("[LLM] Start with: ollama serve")
        return False
    except Exception as e:
        print(f"[LLM] Ollama check error: {e}")
        return False


def warmup_ollama() -> str:
    """
    Send a 5-token dummy prompt to force the model to load into RAM.
    Returns the model name that actually loaded successfully.

    If the configured model OOMs (can't allocate buffer), automatically
    falls back to qwen2.5:3b and updates config.OLLAMA_MODEL globally
    so all subsequent calls in this process use the smaller model.

    Call this immediately after every Ollama restart — before starting
    any real analysis — so OOM is caught once at startup, not mid-run.
    """
    import config as _cfg

    def _try_load(model: str) -> bool:
        """Returns True if model loads successfully."""
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "Say OK"}],
                "stream": False,
                "options": {"num_predict": 5, "num_ctx": 512},
            }
            r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=90)
            if r.status_code == 200:
                return True
            body = r.text.lower()
            if "allocate" in body or "500" in str(r.status_code):
                return False
            return False
        except Exception as e:
            if "allocate" in str(e).lower():
                return False
            return False

    model = _cfg.OLLAMA_MODEL
    print(f"  [WARMUP] Pre-loading {model} into RAM...", end=" ", flush=True)

    if _try_load(model):
        print(f"✓  ({model} loaded)", flush=True)
        return model

    # OOM on primary model → try 3b fallback
    fallback = "qwen2.5:3b"
    print(f"\n  [WARMUP] ⚠ OOM on {model} → switching to {fallback}", flush=True)
    if _try_load(fallback):
        _cfg.OLLAMA_MODEL = fallback   # global switch for this process
        print(f"  [WARMUP] ✓ {fallback} loaded — using for entire run", flush=True)
        return fallback

    print(f"  [WARMUP] ⚠ Both models failed to load — continuing anyway", flush=True)
    return model


def stream_chat(system_prompt: str, user_message: str,
                model: str = None,
                temperature: float = None,
                on_token=None,
                num_predict: int = None,
                stop: list = None,
                timeout: int = None) -> str:
    """
    Send a prompt to Ollama and stream the response.

    Args:
        system_prompt : Bot persona / system instructions
        user_message  : The actual analysis request
        model         : Override default model
        temperature   : Override default temperature
        on_token      : Callback(str) per streamed token
        num_predict   : MAX TOKENS TO GENERATE for this call.
                        Pass the right value per call type — do NOT
                        leave this as the global 1500 default.
                        Use LLM_TOKENS["opening"], ["verdict"] etc.
        stop          : List of stop sequences (model halts on first match)
        timeout       : Hard wall-clock timeout in seconds (default 120)

    Returns:
        Full response string. May end with [STALL] or [TIMEOUT] if aborted.

    STALL GUARD:
        A daemon thread monitors time since last token.
        If no token arrives in TOKEN_STALL_TIMEOUT seconds the stream
        is closed and the partial result returned — pipeline continues.
    """
    model       = model or OLLAMA_MODEL
    temperature = temperature if temperature is not None else LLM_TEMPERATURE
    hard_limit  = timeout or CALL_HARD_TIMEOUT

    # Default to the verdict budget if not specified — safer than 1500
    if num_predict is None:
        num_predict = LLM_TOKENS.get("verdict", 450)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message}
        ],
        "stream": True,
        "options": {
            "temperature":    temperature,
            "top_p":          LLM_TOP_P,
            "num_predict":    num_predict,     # FIX: per-call budget
            "num_ctx":        LLM_CONTEXT_WINDOW,  # FIX: was 4096, now 8192
        }
    }

    # Add stop sequences if provided
    if stop:
        payload["options"]["stop"] = stop

    if VERBOSE_DEBUG:
        print(f"\n[LLM] {model} | num_predict={num_predict} | "
              f"stop={stop} | timeout={hard_limit}s")
        print(f"[LLM] Prompt (first 150 chars): {user_message[:150]}...")

    full_response  = ""
    _abort         = threading.Event()
    _last_tok_time = [time.time()]
    _response_ref  = [None]   # watchdog closes this to unblock iter_lines()

    def _watchdog():
        start = time.time()
        while not _abort.is_set():
            time.sleep(2)
            idle    = time.time() - _last_tok_time[0]
            elapsed = time.time() - start
            if idle > TOKEN_STALL_TIMEOUT:
                print(f"\n[LLM] ⚠ Stall: no token for {idle:.0f}s — aborting",
                      flush=True)
                _abort.set()
                if _response_ref[0]:
                    try: _response_ref[0].close()
                    except Exception: pass
                return
            if elapsed > hard_limit:
                print(f"\n[LLM] ⚠ Hard timeout {hard_limit}s — aborting",
                      flush=True)
                _abort.set()
                if _response_ref[0]:
                    try: _response_ref[0].close()
                    except Exception: pass
                return

    wdog = threading.Thread(target=_watchdog, daemon=True)
    wdog.start()

    try:
        r = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json=payload,
            stream=True,
            timeout=(10, None),   # no per-chunk timeout; watchdog handles stalls
        )
        _response_ref[0] = r      # give watchdog a handle to close it
        r.raise_for_status()

        for line in r.iter_lines():
            if _abort.is_set():
                full_response += "\n[STALL-ABORTED]"
                break
            if line:
                try:
                    data  = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        full_response     += token
                        _last_tok_time[0]  = time.time()
                        if on_token:
                            on_token(token)
                    if data.get("done"):
                        break
                except json.JSONDecodeError:
                    continue

    except requests.Timeout:
        print(f"[LLM] Request timed out after {hard_limit}s")
        full_response += "\n[TIMEOUT]"
    except Exception as e:
        err_str = str(e)
        # OOM: Ollama can't allocate model buffer → fall back to 3b
        if "allocate" in err_str.lower() and "3b" not in model:
            _fallback = "qwen2.5:3b"
            print(f"\n[LLM] ⚠ OOM on {model} → retrying with {_fallback}", flush=True)
            _abort.set()
            return stream_chat(
                system_prompt, user_message,
                model=_fallback,
                temperature=temperature,
                on_token=on_token,
                num_predict=num_predict,
                stop=stop,
                timeout=hard_limit,
            )
        print(f"[LLM] Error: {e}")
        full_response = f"[LLM Error: {e}]"
    finally:
        _abort.set()

    return full_response.strip()


def extract_score(text: str, default: float = 5.0) -> float:
    """Extract a numeric score from LLM response text."""
    import re
    patterns = [
        r'score[:\s]+(\d+(?:\.\d+)?)\s*/\s*10',
        r'rating[:\s]+(\d+(?:\.\d+)?)\s*/\s*10',
        r'(\d+(?:\.\d+)?)\s*/\s*10',
        r'score[:\s]+(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s+out\s+of\s+10',
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            val = float(match.group(1))
            if 0 <= val <= 10:
                return round(val, 1)
    return default


def format_currency(val, currency="₹"):
    """Format large numbers: 1500000000 → ₹150 Cr"""
    if val is None:
        return "N/A"
    try:
        val = float(val)
        if val >= 1e12:  return f"{currency}{val/1e12:.2f}L Cr"
        if val >= 1e9:   return f"{currency}{val/1e9:.2f}K Cr"
        if val >= 1e7:   return f"{currency}{val/1e7:.2f} Cr"
        if val >= 1e5:   return f"{currency}{val/1e5:.2f} L"
        return f"{currency}{val:.2f}"
    except Exception:
        return str(val)


def safe_round(val, decimals=2):
    if val is None: return None
    try:   return round(float(val), decimals)
    except: return None


def fmt(val, suffix="", decimals=2, na="N/A"):
    if val is None: return na
    try:   return f"{round(float(val), decimals)}{suffix}"
    except: return na
