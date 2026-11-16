#!/usr/bin/env bash
# ============================================================
# download_data.sh — Download the IBM Telco Customer Churn CSV (7,043 rows)
# Usage:
#   bash scripts/download_data.sh            # public IBM GitHub mirror (no auth needed)
#   bash scripts/download_data.sh --kaggle   # download via Kaggle CLI
# ============================================================

set -euo pipefail

MODE="${1:-public}"
OUTPUT_DIR="data/raw"
OUTPUT_FILE="$OUTPUT_DIR/telco_churn.csv"
IBM_URL="https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
KAGGLE_DATASET="blastchar/telco-customer-churn"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

mkdir -p "$OUTPUT_DIR"

if [ "$MODE" = "--kaggle" ]; then
    command -v kaggle >/dev/null 2>&1 || error "kaggle CLI not found. Install with: pip install kaggle"
    info "Downloading dataset from Kaggle ($KAGGLE_DATASET)..."
    kaggle datasets download -d "$KAGGLE_DATASET" -p "$OUTPUT_DIR" --unzip
    RAW_FILE="$OUTPUT_DIR/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    [ -f "$RAW_FILE" ] && mv "$RAW_FILE" "$OUTPUT_FILE"
else
    info "Downloading IBM Telco Customer Churn dataset..."
    TMP_FILE="$(mktemp)"
    if curl -fsSL --retry 3 --connect-timeout 15 "$IBM_URL" -o "$TMP_FILE"; then
        mv "$TMP_FILE" "$OUTPUT_FILE"
    else
        rm -f "$TMP_FILE"
        warn "Direct download failed; falling back to synthetic 7,043-row dataset."
        python3 -m src.data_loader
    fi
fi

if [ ! -s "$OUTPUT_FILE" ]; then
    error "Dataset file is missing or empty: $OUTPUT_FILE"
fi

ROWS=$(wc -l < "$OUTPUT_FILE" | tr -d ' ')
HEADER=$(head -n 1 "$OUTPUT_FILE")
echo "$HEADER" | grep -q "customerID" || error "Invalid header in $OUTPUT_FILE"
echo "$HEADER" | grep -q "Churn" || error "Missing 'Churn' column in $OUTPUT_FILE"

info "Verified dataset at $OUTPUT_FILE ($((ROWS - 1)) customer records)."
