#!/usr/bin/env bash
# ============================================================
# setup.sh — One-command ChurnScope local project setup
# Usage:
#   bash scripts/setup.sh          # interactive
#   bash scripts/setup.sh --yes    # non-interactive (auto-train)
#   bash scripts/setup.sh --no-train
# ============================================================

set -euo pipefail

MODE="prompt"
for arg in "$@"; do
    case "$arg" in
        -y|--yes) MODE="yes" ;;
        --no-train) MODE="no" ;;
    esac
done

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

info "=============================================="
info "  ChurnScope — Project Setup"
info "=============================================="

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$MAJOR" -lt 3 ] || { [ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]; }; then
    error "Python 3.10+ required. Found: $PYTHON_VERSION"
fi
info "Python $PYTHON_VERSION detected"

if [ ! -d ".venv" ]; then
    info "Creating virtual environment at .venv/..."
    python3 -m venv .venv
else
    warn "Virtual environment .venv/ already exists — reusing."
fi

# shellcheck disable=SC1091
source .venv/bin/activate

info "Installing dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
info "Dependencies installed"

info "Creating runtime directories..."
mkdir -p data/raw data/processed data/features
mkdir -p models/trained models/artifacts models/registry
mkdir -p logs results notebooks/plots

if [ ! -f ".env" ]; then
    cp .env.example .env
    info "Created .env from .env.example (review settings before production deploy)"
else
    warn ".env already exists — keeping existing file."
fi

SHOULD_TRAIN="no"
if [ "$MODE" = "yes" ]; then
    SHOULD_TRAIN="yes"
elif [ "$MODE" = "prompt" ] && [ -t 0 ]; then
    read -r -p "Download dataset and train initial model now? [y/N]: " REPLY
    [[ "$REPLY" =~ ^[Yy]$ ]] && SHOULD_TRAIN="yes"
fi

if [ "$SHOULD_TRAIN" = "yes" ]; then
    if bash scripts/download_data.sh; then
        python train_pipeline.py
    else
        warn "Public download unavailable — training on synthetic dataset instead."
        python train_pipeline.py --synthetic --n-samples 7043
    fi
    info "Initial model trained and registered"
else
    warn "Skipped training. Run 'make download-data && make train' or 'make train-synthetic'."
fi

echo ""
info "Setup complete! Next commands:"
echo "  1. Activate venv  : source .venv/bin/activate"
echo "  2. Start API      : make api          (http://localhost:8000/docs)"
echo "  3. Start PWA UI   : make dashboard    (http://localhost:8501)"
echo "  4. Run test suite : make test"
echo ""
