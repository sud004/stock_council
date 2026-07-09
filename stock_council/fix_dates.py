#!/usr/bin/env python3
# ============================================================
# fix_dates.py — ONE-TIME correction script
# ============================================================
# Fixes the date mismatch caused by the pre-trading_calendar.py
# bug: Day 2 was labeled "2026-06-19" but actually traded
# June 18's closing prices (confirmed: SBIN entry ₹1027.90
# matches June 18's close in data/prices/SBIN.csv).
#
# This script:
#   1. Backs up portfolio_state.json before touching it
#   2. Corrects all "2026-06-19" dates that belong to Day 2
#      → changes them to "2026-06-18"
#   3. Leaves Day 1 (06-17) and Day 3 (06-19) untouched —
#      those are already correct
#   4. Rebuilds portfolio_tracker.xlsx from the corrected state
#   5. Renames market_2026-06-19.xlsx → market_2026-06-18.xlsx
#      (that file's content is Day 2's June 18 analysis)
#
# USAGE:
#   python fix_dates.py           # preview changes (dry run)
#   python fix_dates.py --apply   # actually apply the fix
# ============================================================

import json
import shutil
import sys
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR

STATE_FILE  = DATA_DIR / "portfolio_state.json"
EXCEL_DIR   = DATA_DIR / "excel"
BACKUP_DIR  = DATA_DIR / "backups"

# The exact correction: Day 2 was mislabeled
WRONG_DATE_FOR_DAY2 = "2026-06-19"
CORRECT_DATE_FOR_DAY2 = "2026-06-18"


def backup_files():
    """Backup state and Excel before modifying anything."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    if STATE_FILE.exists():
        dest = BACKUP_DIR / f"portfolio_state_{ts}.json.bak"
        shutil.copy2(STATE_FILE, dest)
        print(f"  Backed up: {dest.name}")

    tracker = EXCEL_DIR / "portfolio_tracker.xlsx"
    if tracker.exists():
        dest = BACKUP_DIR / f"portfolio_tracker_{ts}.xlsx.bak"
        shutil.copy2(tracker, dest)
        print(f"  Backed up: {dest.name}")


def fix_state_dates(apply: bool = False) -> dict:
    """
    Fix the date mismatch in portfolio_state.json.

    Day 2 entries are identified by day == 2 (not by date string,
    since the date string is exactly what's wrong). This is safe
    because the day counter itself was always correct — only the
    'date' field attached to it was wrong.
    """
    with open(STATE_FILE) as f:
        state = json.load(f)

    changes = []

    # ── Fix daily_log ──────────────────────────────────────
    for entry in state.get('daily_log', []):
        if entry.get('day') == 2 and entry.get('date') == WRONG_DATE_FOR_DAY2:
            changes.append(
                f"daily_log[day=2].date: {entry['date']} → {CORRECT_DATE_FOR_DAY2}")
            if apply:
                entry['date'] = CORRECT_DATE_FOR_DAY2

    # ── Fix positions (entry_date) ──────────────────────────
    # Positions opened during Day 2's fill (AXISBANK, SBIN, DLF
    # — the original 4 minus PHOENIXLTD which was stopped out)
    day2_symbols = {'AXISBANK', 'SBIN', 'DLF', 'PHOENIXLTD'}
    for sym, pos in state.get('positions', {}).items():
        if sym in day2_symbols and pos.get('entry_date') == WRONG_DATE_FOR_DAY2:
            changes.append(
                f"positions[{sym}].entry_date: {pos['entry_date']} → {CORRECT_DATE_FOR_DAY2}")
            if apply:
                pos['entry_date'] = CORRECT_DATE_FOR_DAY2

    # ── Fix closed_trades (PHOENIXLTD was Day 2's stop-loss) ──
    for trade in state.get('closed_trades', []):
        if (trade.get('symbol') == 'PHOENIXLTD' and
                trade.get('buy_date') == WRONG_DATE_FOR_DAY2):
            changes.append(
                f"closed_trades[PHOENIXLTD].buy_date/sell_date: "
                f"{trade['buy_date']} → {CORRECT_DATE_FOR_DAY2}")
            if apply:
                trade['buy_date']  = CORRECT_DATE_FOR_DAY2
                trade['sell_date'] = CORRECT_DATE_FOR_DAY2

    if apply:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2, default=str)

    return {'changes': changes, 'state': state}


def fix_market_excel_filename(apply: bool = False) -> list:
    """
    Rename market_2026-06-19.xlsx → market_2026-06-18.xlsx
    since that file's content is Day 2's June 18 analysis.
    """
    changes = []
    old_path = EXCEL_DIR / "market_2026-06-19.xlsx"
    new_path = EXCEL_DIR / "market_2026-06-18.xlsx"

    if old_path.exists() and not new_path.exists():
        changes.append(f"Rename: {old_path.name} → {new_path.name}")
        if apply:
            shutil.move(str(old_path), str(new_path))
    elif old_path.exists() and new_path.exists():
        changes.append(
            f"⚠ Both {old_path.name} and {new_path.name} exist — "
            f"manual review needed, not auto-renaming")
    else:
        changes.append(f"⚠ {old_path.name} not found — nothing to rename")

    return changes


def rebuild_portfolio_excel(apply: bool = False):
    """Rebuild portfolio_tracker.xlsx from the corrected state."""
    if not apply:
        print("\n  (Dry run — Excel rebuild skipped. Use --apply to rebuild.)")
        return

    from portfolio_engine import load_state
    from portfolio_excel import save_portfolio_excel

    state = load_state()
    save_portfolio_excel(state, print_output=True)


def main():
    parser = argparse.ArgumentParser(description='Fix portfolio date mismatch')
    parser.add_argument('--apply', action='store_true',
                        help='Actually apply changes (default: dry run preview)')
    args = parser.parse_args()

    print(f"\n{'═'*60}")
    print(f"  PORTFOLIO DATE CORRECTION")
    print(f"  Mode: {'APPLY (will modify files)' if args.apply else 'DRY RUN (preview only)'}")
    print('═'*60)

    if not STATE_FILE.exists():
        print(f"\n  ❌ {STATE_FILE} not found — nothing to fix")
        return

    if args.apply:
        print(f"\n  Backing up files first...")
        backup_files()

    print(f"\n  Checking portfolio_state.json...")
    result = fix_state_dates(apply=args.apply)
    if result['changes']:
        for c in result['changes']:
            print(f"    {'✅' if args.apply else '→'} {c}")
    else:
        print(f"    No changes needed")

    print(f"\n  Checking market Excel filenames...")
    excel_changes = fix_market_excel_filename(apply=args.apply)
    for c in excel_changes:
        print(f"    {'✅' if args.apply else '→'} {c}")

    if args.apply:
        print(f"\n  Rebuilding portfolio_tracker.xlsx with corrected dates...")
        rebuild_portfolio_excel(apply=True)
    else:
        print(f"\n  Run again with --apply to actually make these changes:")
        print(f"    python fix_dates.py --apply")

    print(f"\n{'═'*60}")
    print(f"  {'✅ DONE' if args.apply else 'DRY RUN COMPLETE — no files modified'}")
    print('═'*60)


if __name__ == "__main__":
    main()
