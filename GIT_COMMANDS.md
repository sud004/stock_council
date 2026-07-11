# Git Command Reference — Stock Council

## First-Time Setup (done once)
```bat
git init
git branch -M main
git remote add origin https://sud004@github.com/sud004/stock_council.git
git push -u origin main
```

---

## Daily / After Night Runner Run
```bat
git add .
git commit -m "day 17: night runner results 2026-07-10"
git push
```

---

## Check What Changed
```bat
git status                      # show changed/untracked files
git diff                        # show exact line changes (unstaged)
git diff --staged               # show what's staged for commit
git log --oneline -10           # last 10 commits (one line each)
```

---

## Undo / Fix Mistakes
```bat
# Undo changes to a file (not yet staged)
git restore stock_council/night_runner.py

# Unstage a file (staged but not committed)
git restore --staged stock_council/night_runner.py

# Undo last commit but keep changes locally
git reset --soft HEAD~1

# Amend last commit message
git commit --amend -m "new message here"
```

---

## Untrack a File (keep locally, remove from GitHub)
```bat
git rm --cached stock_council/data/some_file.json
git add .gitignore
git commit -m "chore: untrack file"
git push
```

---

## Sync / Pull (if editing from another machine)
```bat
git pull origin main
```

---

## Remote / Auth Issues
```bat
# Check current remote URL
git remote -v

# Change remote URL (e.g. after repo rename)
git remote set-url origin https://sud004@github.com/sud004/stock_council.git

# Clear cached Windows credentials (then re-push and enter PAT as password)
cmdkey /delete:git:https://github.com
```

---

## Weekly / End of Experiment
```bat
# Tag the final day
git tag -a v1.0-day21 -m "Day 21 complete — experiment ended"
git push origin v1.0-day21
```

---

## Personal Access Token (PAT)
GitHub no longer accepts passwords. When prompted:
- **Username**: sud004
- **Password**: paste your PAT from https://github.com/settings/tokens
  (Generate new token (classic) → tick `repo` → copy)
