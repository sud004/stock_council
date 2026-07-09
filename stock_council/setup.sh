#!/bin/bash
# ============================================================
# setup.sh — One-command setup for Indian Stock Market Bot Council
# ============================================================
# Run: bash setup.sh
# Works on Ubuntu / Debian / macOS / WSL (Windows)
# ============================================================

set -e
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

echo -e "${BOLD}${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   🏛  INDIAN STOCK MARKET BOT COUNCIL — AUTO SETUP         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── Step 1: Python version ────────────────────────────────────
echo -e "${YELLOW}[1/7] Checking Python...${NC}"
python3 --version 2>/dev/null || { echo -e "${RED}Python 3 not found. Install from python.org${NC}"; exit 1; }
PY_VER=$(python3 -c 'import sys; print(sys.version_info.minor)')
if [ "$PY_VER" -lt 10 ]; then
    echo -e "${RED}Python 3.10+ required. You have: $(python3 --version)${NC}"
    exit 1
fi
echo -e "${GREEN}  ✓ $(python3 --version)${NC}"

# ── Step 2: Virtual environment ───────────────────────────────
echo -e "${YELLOW}[2/7] Setting up virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
# Activate (works on Linux/Mac and WSL)
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
fi
echo -e "${GREEN}  ✓ Virtual environment ready${NC}"

# ── Step 3: Install Python packages ──────────────────────────
echo -e "${YELLOW}[3/7] Installing Python packages (this takes 2-5 min first time)...${NC}"
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo -e "${GREEN}  ✓ All packages installed${NC}"

# ── Step 4: Ollama ───────────────────────────────────────────
echo -e "${YELLOW}[4/7] Setting up Ollama (local LLM engine)...${NC}"
if command -v ollama &>/dev/null; then
    echo -e "${GREEN}  ✓ Ollama already installed${NC}"
else
    echo -e "  Installing Ollama..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install ollama 2>/dev/null || curl -fsSL https://ollama.com/install.sh | sh
    else
        curl -fsSL https://ollama.com/install.sh | sh
    fi
    echo -e "${GREEN}  ✓ Ollama installed${NC}"
fi

# Start Ollama in background if not running
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "  Starting Ollama..."
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    sleep 4
fi

# ── Step 5: Download LLM model ───────────────────────────────
echo -e "${YELLOW}[5/7] Setting up LLM model...${NC}"

# Pick model based on available RAM
RAM_GB=$(python3 -c "
import os
try:
    with open('/proc/meminfo') as f:
        for line in f:
            if 'MemTotal' in line:
                kb = int(line.split()[1])
                print(kb // 1024 // 1024)
                break
except:
    print(8)
" 2>/dev/null || echo "8")

if [ "$RAM_GB" -ge 16 ]; then
    RECOMMEND_MODEL="llama3.1"
    RAM_NOTE="(16GB+ RAM — best quality)"
elif [ "$RAM_GB" -ge 8 ]; then
    RECOMMEND_MODEL="mistral"
    RAM_NOTE="(8GB RAM — good balance)"
else
    RECOMMEND_MODEL="phi3:mini"
    RAM_NOTE="(low RAM — fastest option)"
fi

MODEL=${OLLAMA_MODEL:-$RECOMMEND_MODEL}
echo -e "  RAM detected: ${RAM_GB}GB → recommending ${BOLD}${MODEL}${NC} ${RAM_NOTE}"

if ollama list 2>/dev/null | grep -q "^${MODEL%:*}"; then
    echo -e "${GREEN}  ✓ Model ${MODEL} already downloaded${NC}"
else
    echo "  Downloading ${MODEL}... (may take several minutes)"
    ollama pull $MODEL
    echo -e "${GREEN}  ✓ Model ${MODEL} ready${NC}"
fi

# ── Step 6: Environment file ──────────────────────────────────
echo -e "${YELLOW}[6/7] Setting up environment config...${NC}"
if [ ! -f ".env" ]; then
    cp .env.example .env
    # Set the recommended model
    sed -i.bak "s/OLLAMA_MODEL=.*/OLLAMA_MODEL=${MODEL}/" .env 2>/dev/null || \
    sed -i "" "s/OLLAMA_MODEL=.*/OLLAMA_MODEL=${MODEL}/" .env 2>/dev/null || true
    echo -e "${GREEN}  ✓ Created .env with model=${MODEL}${NC}"
else
    echo -e "${GREEN}  ✓ .env already exists${NC}"
fi

# ── Step 7: Create data directories ──────────────────────────
echo -e "${YELLOW}[7/7] Creating data directories...${NC}"
mkdir -p data/{prices,fundamentals,news,vectors,excel,embeddings,scores} \
         memory models/embeddings reports cache
echo -e "${GREEN}  ✓ Directories ready${NC}"

# ── Done ──────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}╔══════════════════════════════════════════════════════════════╗"
echo -e "║   ✅  SETUP COMPLETE!                                        ║"
echo -e "╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BOLD}WHAT TO DO NEXT:${NC}"
echo ""
echo -e "  ${YELLOW}Step 1${NC} — Set up Ventura EaseAPI (one time, ~5 min):"
echo -e "  ${BLUE}python utils/easeapi_auth.py --setup${NC}"
echo ""
echo -e "  ${YELLOW}Step 2${NC} — Download all 286 NSE stock data (run tonight):"
echo -e "  ${BLUE}python run.py --nightly${NC}"
echo ""
echo -e "  ${YELLOW}Step 3${NC} — Run the pipeline:"
echo -e "  ${BLUE}python run.py --fast${NC}     ← fast mode, no Ollama needed"
echo -e "  ${BLUE}python run.py${NC}            ← full pipeline with LLM"
echo -e "  ${BLUE}python run.py --schedule${NC} ← start all-day hourly tracker"
echo ""
echo -e "${BOLD}SYSTEM SUMMARY:${NC}"
echo -e "  LLM Model:    ${MODEL}"
echo -e "  EaseAPI:      needs --setup (see Step 1)"
echo -e "  Stocks:       286 NSE stocks across 13 sectors"
echo -e "  Storage:      data/ folder (grows over time)"
echo ""
echo -e "${YELLOW}Check status anytime: ${BLUE}python run.py --status${NC}"
