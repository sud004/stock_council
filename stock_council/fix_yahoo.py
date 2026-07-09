# ============================================================
# fix_yahoo.py — Run this once to fix Yahoo Finance issues
# Usage: python fix_yahoo.py
# ============================================================
import subprocess, sys

print("Upgrading yfinance to latest...")
subprocess.check_call([sys.executable, "-m", "pip", "install", 
                       "yfinance==0.2.61", "-q"])

print("Installing curl-cffi for better Yahoo Finance requests...")
subprocess.check_call([sys.executable, "-m", "pip", "install", 
                       "curl-cffi", "-q"])

print("\nTesting Yahoo Finance connection...")
import yfinance as yf
import time

test_stocks = ["TCS.NS", "RELIANCE.NS", "INFY.NS"]
for sym in test_stocks:
    try:
        ticker = yf.Ticker(sym)
        hist = ticker.history(period="5d")
        if not hist.empty:
            price = round(hist['Close'].iloc[-1], 2)
            print(f"  ✅ {sym}: ₹{price}")
        else:
            print(f"  ❌ {sym}: No data")
    except Exception as e:
        print(f"  ❌ {sym}: {e}")
    time.sleep(1)

print("\nDone! Now run: python run.py --nightly")
