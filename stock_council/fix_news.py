# ============================================================
# fix_news.py — Clear bad news cache and re-fetch properly
# Run once: python fix_news.py
# ============================================================
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from memory.storage import NEWS_DIR, today_str
from scanner.universe import ALL_STOCKS

today = today_str()
print(f"\n[NEWS FIX] Checking news cache for {today}...")

# Find all today's news files with < 5 articles
bad_files = []
good_files = []
missing = []

for sym in ALL_STOCKS:
    sym_clean = sym.replace('.NS','').replace('.BO','')
    json_path = NEWS_DIR / sym_clean / f"{today}.json"
    
    if not json_path.exists():
        missing.append(sym)
        continue
    
    try:
        with open(json_path) as f:
            data = json.load(f)
        articles = data.get('articles', [])
        if len(articles) < 5:
            bad_files.append((sym, json_path, len(articles)))
        else:
            good_files.append(sym)
    except Exception:
        bad_files.append((sym, json_path, 0))

print(f"  Good (5+ articles): {len(good_files)} stocks")
print(f"  Bad  (<5 articles): {len(bad_files)} stocks")
print(f"  Missing:            {len(missing)} stocks")

# Delete bad files so nightly re-fetches them
print(f"\n[NEWS FIX] Deleting {len(bad_files)} thin news files...")
for sym, path, count in bad_files:
    path.unlink(missing_ok=True)
    # Also delete txt file
    txt = path.with_suffix('.txt')
    txt.unlink(missing_ok=True)

print(f"[NEWS FIX] Done! Now run: python run.py --nightly")
print(f"           News will be re-fetched for {len(bad_files) + len(missing)} stocks")
