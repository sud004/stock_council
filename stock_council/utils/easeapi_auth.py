# ============================================================
# utils/easeapi_auth.py — Fully Automated Ventura EaseAPI Auth
# ============================================================
#
# HOW IT WORKS:
#   1. First time: python utils/easeapi_auth.py --setup
#      → Guides you through TOTP setup (5 minutes, once ever)
#   2. Every morning at 8:45 AM (automatic, via scheduler):
#      auto_login() generates TOTP code → gets fresh auth_token
#   3. Token saved to data/easeapi_token.json
#   4. ensure_logged_in() called at every pipeline start
#      → loads cache if valid, re-logins if expired
#   5. You never manually do anything after setup
#
# TOTP AUTH FORMULA:
#   data  = SHA256(app_key + app_secret)     ← fixed, computed once
#   totp  = pyotp.TOTP(totp_secret).now()   ← 6-digit, changes every 30s
#   POST /login/v1/authorization/totp
#     headers: x-app-key, x-client-id, x-mac-address
#     body:    {password, data, totp}
#   → returns auth_token valid until midnight that day
#
# TOKEN EXPIRY:
#   Ventura tokens expire at end of trading day (typically ~11:59 PM IST)
#   The scheduler renews at 8:45 AM next day automatically
#   If you run the script outside market hours, it re-logins automatically
# ============================================================

import os
import sys
import json
import hashlib
import time
import uuid
import re
import requests
import schedule
from pathlib import Path
from datetime import datetime, timedelta
import pytz

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_DIR, BASE_DIR, VERBOSE_DEBUG
from dotenv import load_dotenv

load_dotenv(BASE_DIR / ".env")

IST       = pytz.timezone('Asia/Kolkata')
BASE_URL  = "https://easeapi.venturasecurities.com"
TOKEN_FILE = DATA_DIR / "easeapi_token.json"

# ── Read credentials from .env ────────────────────────────────
APP_KEY      = os.getenv("EASEAPI_APP_KEY", "")
APP_SECRET   = os.getenv("EASEAPI_APP_SECRET", "")
CLIENT_ID    = os.getenv("EASEAPI_CLIENT_ID", "")
PASSWORD     = os.getenv("EASEAPI_PASSWORD", "")
TOTP_SECRET  = os.getenv("EASEAPI_TOTP_SECRET", "")


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def _get_mac() -> str:
    """Read this machine's MAC address (required by Ventura as device ID)."""
    try:
        mac = uuid.getnode()
        return ':'.join(f'{(mac >> (8*i)) & 0xff:02x}' for i in range(5, -1, -1))
    except Exception:
        return '00:00:00:00:00:00'


def _data_hash() -> str:
    """SHA256(app_key + app_secret) — fixed value, computed once per session."""
    return hashlib.sha256((APP_KEY + APP_SECRET).encode()).hexdigest()


def _generate_totp() -> str:
    """Generate current 6-digit TOTP code. Changes every 30 seconds."""
    try:
        import pyotp
    except ImportError:
        raise ImportError("Run: pip install pyotp")
    return pyotp.TOTP(TOTP_SECRET).now()


def _seconds_until_next_totp() -> int:
    """Seconds remaining before TOTP code rotates (cycle = 30s)."""
    return 30 - (int(time.time()) % 30)


def _save_token(data: dict):
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TOKEN_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def _load_token() -> dict | None:
    """Load token from disk. Returns None if missing or expired."""
    if not TOKEN_FILE.exists():
        return None
    try:
        with open(TOKEN_FILE) as f:
            data = json.load(f)
        expiry_str = data.get('auth_expiry', '')
        if expiry_str:
            expiry = IST.localize(datetime.strptime(expiry_str, '%Y-%m-%d %H:%M:%S'))
            if datetime.now(IST) >= expiry - timedelta(minutes=10):
                if VERBOSE_DEBUG:
                    print(f"[AUTH] Token expired at {expiry_str}")
                return None
        return data if data.get('auth_token') else None
    except Exception as e:
        if VERBOSE_DEBUG:
            print(f"[AUTH] Token load error: {e}")
        return None


# ══════════════════════════════════════════════════════════════
# CORE LOGIN
# ══════════════════════════════════════════════════════════════

def auto_login(retries: int = 3) -> dict | None:
    """
    Fully automated login using TOTP. No browser, no human input.

    Handles edge cases:
    - TOTP code about to expire → waits for next cycle
    - Rate limit hit → backs off
    - Wrong TOTP → retries with fresh code
    """
    if not APP_KEY or not APP_SECRET or not CLIENT_ID:
        print("[AUTH] Missing EASEAPI credentials in .env")
        print("       Run: python utils/easeapi_auth.py --setup")
        return None

    if not TOTP_SECRET:
        print("[AUTH] EASEAPI_TOTP_SECRET not set — cannot auto-login")
        print("       Run: python utils/easeapi_auth.py --setup")
        return None

    if not PASSWORD:
        print("[AUTH] EASEAPI_PASSWORD not set in .env")
        return None

    url     = f"{BASE_URL}/login/v1/authorization/totp"
    mac     = _get_mac()
    data_h  = _data_hash()
    headers = {
        'x-app-key':     APP_KEY,
        'x-client-id':   CLIENT_ID,
        'x-mac-address': mac,
        'Content-Type':  'application/json',
    }

    for attempt in range(1, retries + 1):
        # If TOTP code expires in < 5s, wait for the next one
        secs = _seconds_until_next_totp()
        if secs < 5:
            wait = secs + 2
            print(f"[AUTH] Waiting {wait}s for fresh TOTP code...")
            time.sleep(wait)

        try:
            totp = _generate_totp()
            payload = {'password': PASSWORD, 'data': data_h, 'totp': totp}

            print(f"[AUTH] Logging in as {CLIENT_ID} (attempt {attempt}/{retries})...")
            r = requests.post(url, json=payload, headers=headers, timeout=15)

            if r.status_code == 200:
                resp = r.json()
                # Ventura may return success=True or just the token directly
                auth_token = resp.get('auth_token') or resp.get('data', {}).get('auth_token')
                if auth_token:
                    token_data = {
                        'auth_token':    auth_token,
                        'refresh_token': resp.get('refresh_token', ''),
                        'client_id':     resp.get('client_id', CLIENT_ID),
                        'auth_expiry':   resp.get('auth_expiry', ''),
                        'logged_in_at':  datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S'),
                        'mac_address':   mac,
                    }
                    _save_token(token_data)
                    print(f"[AUTH] ✅ Login successful | Expires: {token_data['auth_expiry']}")
                    return token_data
                else:
                    print(f"[AUTH] Unexpected response: {resp}")
                    # Might be wrong TOTP timing — wait full cycle
                    time.sleep(35)

            elif r.status_code == 429:
                print(f"[AUTH] Rate limited. Waiting 60s...")
                time.sleep(60)

            elif r.status_code in (400, 401):
                body = r.json() if r.content else {}
                msg  = body.get('message', r.text[:200])
                print(f"[AUTH] Auth failed ({r.status_code}): {msg}")
                if 'totp' in msg.lower() or 'otp' in msg.lower():
                    # TOTP mismatch — wait for next cycle and retry
                    time.sleep(35)
                else:
                    # PIN or other credential issue
                    print("[AUTH] Check EASEAPI_PASSWORD in .env")
                    return None

            else:
                print(f"[AUTH] HTTP {r.status_code}: {r.text[:200]}")
                time.sleep(5)

        except requests.ConnectionError:
            print(f"[AUTH] No internet connection (attempt {attempt})")
            time.sleep(15)
        except Exception as e:
            print(f"[AUTH] Unexpected error: {e}")
            time.sleep(5)

    print("[AUTH] ❌ All login attempts failed")
    return None


# ══════════════════════════════════════════════════════════════
# PUBLIC INTERFACE (called by pipeline)
# ══════════════════════════════════════════════════════════════

def ensure_logged_in() -> bool:
    """
    Ensure we have a valid token. Auto-logins if expired.
    Called at pipeline startup. Returns True if authenticated.
    """
    token = _load_token()
    if token:
        if VERBOSE_DEBUG:
            print(f"[AUTH] Valid cached token for {token.get('client_id')}")
        return True

    # Token missing or expired — auto-login
    result = auto_login()
    return result is not None


def get_auth_token() -> str | None:
    """Get the current auth_token string. Auto-refreshes if needed."""
    token = _load_token()
    if not token:
        token = auto_login()
    return token.get('auth_token') if token else None


def get_headers() -> dict:
    """
    Get complete authenticated headers for EaseAPI calls.
    Auto-refreshes token if expired.
    Raises RuntimeError if login fails.
    """
    token = _load_token()
    if not token:
        token = auto_login()
    if not token:
        raise RuntimeError(
            "[AUTH] Cannot authenticate. "
            "Run: python utils/easeapi_auth.py --setup"
        )
    return {
        'x-app-key':     APP_KEY,
        'x-client-id':   token['client_id'],
        'Content-Type':  'application/json',
        'Authorization': f"Bearer {token['auth_token']}",
    }


def get_client_id() -> str:
    """Return the Ventura client ID."""
    token = _load_token()
    return token.get('client_id', CLIENT_ID) if token else CLIENT_ID


def is_authenticated() -> bool:
    """Quick non-blocking check — does NOT trigger login."""
    return _load_token() is not None


def token_status() -> dict:
    """Return human-readable token status dict."""
    token = _load_token()
    has_totp = bool(TOTP_SECRET and TOTP_SECRET.strip())
    return {
        'authenticated':  token is not None,
        'client_id':      token.get('client_id', 'N/A') if token else 'not logged in',
        'expires_at':     token.get('auth_expiry', 'N/A') if token else 'N/A',
        'logged_in_at':   token.get('logged_in_at', 'N/A') if token else 'N/A',
        'auto_renewal':   has_totp,
        'totp_setup':     has_totp,
        'credentials_set': bool(APP_KEY and APP_SECRET and CLIENT_ID and PASSWORD),
    }


# ══════════════════════════════════════════════════════════════
# DAILY SCHEDULER INTEGRATION
# ══════════════════════════════════════════════════════════════

def schedule_daily_renewal():
    """
    Register a daily 8:45 AM IST job to renew the auth token.
    Called once from run.py --schedule at startup.
    Token expires daily at midnight — this ensures it's always fresh
    before market opens at 9:15 AM.
    """
    _retry_count = [0]

    def _renewal_job():
        _retry_count[0] = 0
        now = datetime.now(IST).strftime('%H:%M IST')
        print(f"\n[AUTH] 🔄 Daily token renewal at {now}")
        result = auto_login()
        if result:
            print("[AUTH] ✅ Token renewed successfully")
            _retry_count[0] = 0
        else:
            print("[AUTH] ❌ Renewal failed — will retry in 5 min")
            _retry_count[0] += 1

    def _retry_job():
        if _retry_count[0] >= 5:
            print("[AUTH] ❌ 5 retry attempts failed — check credentials")
            return schedule.CancelJob
        result = auto_login()
        if result:
            print("[AUTH] ✅ Token renewed on retry")
            _retry_count[0] = 0
            return schedule.CancelJob   # stop retrying
        _retry_count[0] += 1
        return None

    # Schedule at 8:45 AM every weekday
    for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']:
        getattr(schedule.every(), day).at("08:45").do(_renewal_job)

    # Also schedule retries at 8:50, 8:55, 9:00, 9:05, 9:10 as fallback
    for t in ["08:50", "08:55", "09:00", "09:05", "09:10"]:
        for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']:
            getattr(schedule.every(), day).at(t).do(
                lambda: _retry_job() if _retry_count[0] > 0 else None
            )

    print("[AUTH] Daily token renewal scheduled: 8:45 AM IST (Mon–Fri)")


# ══════════════════════════════════════════════════════════════
# SETUP WIZARD — run once ever
# ══════════════════════════════════════════════════════════════

def setup_wizard():
    """
    Interactive first-time TOTP setup.
    Run: python utils/easeapi_auth.py --setup
    Takes about 5 minutes. Never need to do it again.
    """
    print("""
╔══════════════════════════════════════════════════════════════╗
║    VENTURA EASEAPI — ONE-TIME SETUP (5 minutes)             ║
╚══════════════════════════════════════════════════════════════╝

After this setup, the system logs in automatically every morning.
You never need to do this again.
""")
    # Step 1: EaseAPI credentials
    print("─── STEP 1: EaseAPI App Credentials ─────────────────────────")
    print("1. Go to: https://easeapi.venturasecurities.com/portal")
    print("2. Sign in with your Ventura account")
    print("3. Create a new app → copy App Key and App Secret\n")

    app_key    = input("  App Key:    ").strip()
    app_secret = input("  App Secret: ").strip()
    client_id  = input("  Client ID (e.g. AA1234): ").strip().upper()
    password   = input("  Ventura PIN (4–6 digits): ").strip()

    # Step 2: TOTP setup
    print("""
─── STEP 2: Enable TOTP in Ventura App ──────────────────────
1. Open Ventura mobile app
2. Go to: Profile → Security → Enable Authenticator
3. Scan the QR code with Google Authenticator / Authy
4. Look for the SECRET KEY shown during setup
   (a string like: JBSWY3DPEHPK3PXP or similar)
   ⚠  Save this key somewhere safe — you need it below
""")
    totp_secret = input("  TOTP Secret Key: ").strip().upper().replace(' ', '')

    # Validate TOTP secret
    print("\n  Testing TOTP secret...")
    try:
        import pyotp
        totp = pyotp.TOTP(totp_secret)
        code = totp.now()
        remaining = _seconds_until_next_totp()
        print(f"  Current code: {code}  (valid for {remaining}s)")
        match = input("  Open Google Authenticator and confirm this matches (y/n): ").strip().lower()
        if match != 'y':
            print("\n  ⚠  Codes don't match. Possible issues:")
            print("     - Re-scan the QR code in Ventura app")
            print("     - Make sure your phone clock is correct")
            print("     - Try the raw secret shown during QR scan")
            print("\n  Run setup again: python utils/easeapi_auth.py --setup")
            return
    except ImportError:
        print("  Installing pyotp...")
        os.system(f"{sys.executable} -m pip install pyotp -q")
        import pyotp
        code = pyotp.TOTP(totp_secret).now()
        print(f"  Current code: {code}")

    # Write to .env
    env_path = BASE_DIR / ".env"
    env_text = env_path.read_text() if env_path.exists() else ""

    def _set_env(text, key, val):
        if re.search(f'^{key}=', text, re.MULTILINE):
            return re.sub(f'^{key}=.*$', f'{key}={val}', text, flags=re.MULTILINE)
        return text + f'\n{key}={val}'

    for k, v in [
        ('EASEAPI_APP_KEY',     app_key),
        ('EASEAPI_APP_SECRET',  app_secret),
        ('EASEAPI_CLIENT_ID',   client_id),
        ('EASEAPI_PASSWORD',    password),
        ('EASEAPI_TOTP_SECRET', totp_secret),
    ]:
        env_text = _set_env(env_text, k, v)

    env_path.write_text(env_text)

    # Reload env vars in this session
    os.environ['EASEAPI_APP_KEY']     = app_key
    os.environ['EASEAPI_APP_SECRET']  = app_secret
    os.environ['EASEAPI_CLIENT_ID']   = client_id
    os.environ['EASEAPI_PASSWORD']    = password
    os.environ['EASEAPI_TOTP_SECRET'] = totp_secret

    # Patch module-level globals so auto_login() uses new values
    global APP_KEY, APP_SECRET, CLIENT_ID, PASSWORD, TOTP_SECRET
    APP_KEY     = app_key
    APP_SECRET  = app_secret
    CLIENT_ID   = client_id
    PASSWORD    = password
    TOTP_SECRET = totp_secret

    print(f"\n  ✅ Credentials saved to {env_path}")

    # Test login
    print("\n─── STEP 3: Testing automated login ─────────────────────────")
    result = auto_login()

    if result:
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║   ✅ SETUP COMPLETE!                                         ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║   ✓ Logged in as {client_id:<43}║
║   ✓ Token saved to: data/easeapi_token.json                  ║
║   ✓ Auto-renewal: every weekday at 8:45 AM IST              ║
║   ✓ No manual action needed from tomorrow onwards           ║
║                                                              ║
║   NEXT STEPS:                                                ║
║   1. Install Ollama + model:  ollama pull mistral            ║
║   2. Download data tonight:   python run.py --nightly        ║
║   3. Run the pipeline:        python run.py                  ║
╚══════════════════════════════════════════════════════════════╝
""")
    else:
        print("""
  ❌ Login test failed. Common reasons:
  ─────────────────────────────────────
  • Wrong PIN  → check EASEAPI_PASSWORD (use your 4-digit Ventura trading PIN)
  • Wrong TOTP → re-scan QR code from Ventura app, try again
  • Wrong keys → re-copy App Key / Secret from EaseAPI portal
  • App not approved → check EaseAPI portal for app status

  Run setup again: python utils/easeapi_auth.py --setup
""")


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Ventura EaseAPI Auth Manager')
    parser.add_argument('--setup',  action='store_true', help='First-time TOTP setup wizard')
    parser.add_argument('--login',  action='store_true', help='Force login now')
    parser.add_argument('--status', action='store_true', help='Show token status')
    args = parser.parse_args()

    if args.setup:
        setup_wizard()
    elif args.login:
        result = auto_login()
        if result:
            print(f"✅ Logged in as {result['client_id']} | Expires: {result['auth_expiry']}")
    elif args.status:
        s = token_status()
        print("\nEaseAPI Auth Status:")
        print(f"  Authenticated:   {'✅ Yes' if s['authenticated'] else '❌ No'}")
        print(f"  Client ID:       {s['client_id']}")
        print(f"  Expires at:      {s['expires_at']}")
        print(f"  Logged in at:    {s['logged_in_at']}")
        print(f"  TOTP setup:      {'✅ Yes — auto-renewal active' if s['totp_setup'] else '❌ No — manual login needed'}")
        print(f"  Credentials set: {'✅ Yes' if s['credentials_set'] else '❌ No'}")
        if not s['authenticated']:
            print("\n  → Run: python utils/easeapi_auth.py --login")
        if not s['totp_setup']:
            print("  → Run: python utils/easeapi_auth.py --setup  (to enable auto-renewal)")
    else:
        parser.print_help()
