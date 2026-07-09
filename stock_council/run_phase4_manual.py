"""
Manual Phase 4 run for Day 12 (2026-07-03).
Loads council_checkpoint_2026-07-03.json and runs the portfolio engine.
"""
import sys, json
from pathlib import Path

# Run from stock_council/ dir
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from night_runner import phase4_portfolio

CHECKPOINT = ROOT / 'data' / 'council_checkpoint_2026-07-03.json'

with open(CHECKPOINT) as f:
    checkpoint = json.load(f)

stocks = list(checkpoint.values())
print(f"Loaded {len(stocks)} stocks from checkpoint")

# Minimal market snap — actual nifty_1m used only for shadow_log
council_results = {
    'stocks':  stocks,
    'market':  {
        'market_score':   5.0,
        'market_verdict': 'NEUTRAL',
        'nifty_1m':       0,
    },
    'sectors': [],
}

result = phase4_portfolio(council_results)
print("\nPhase 4 complete:", result)
