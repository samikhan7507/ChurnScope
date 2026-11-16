#!/usr/bin/env bash
# ============================================================
# retrain.sh — Automated model retraining with failure alerting
# Usage:
#   bash scripts/retrain.sh                 # train on data/raw/telco_churn.csv
#   bash scripts/retrain.sh --synthetic     # train on synthetic dataset
#   NOTIFY=1 bash scripts/retrain.sh        # send Slack alert on success/failure
#
# Cron example (every Sunday at 02:00 UTC):
#   0 2 * * 0 /path/to/ChurnScope/scripts/retrain.sh >> /var/log/churnscope_retrain.log 2>&1
# ============================================================

set -euo pipefail

TIMESTAMP=$(date -u '+%Y-%m-%d %H:%M:%S UTC')
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_ROOT/logs/retrain_$(date -u +%Y%m%d).log"
MODE="${1:-}"

cd "$PROJECT_ROOT"
mkdir -p logs

if [ -f ".env" ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

if [ -d ".venv" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

notify_slack() {
    local message="$1"
    if [ "${NOTIFY:-0}" = "1" ] && [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
        curl -fsS -X POST "$SLACK_WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d "{\"text\": \"$message\"}" >/dev/null || true
    fi
}

echo "=============================================="
echo "CHURNSCOPE MODEL RETRAINING — $TIMESTAMP"
echo "=============================================="

TRAIN_CMD=(python3 train_pipeline.py)
if [ "$MODE" = "--synthetic" ]; then
    TRAIN_CMD+=(--synthetic --n-samples 7043)
fi

set +e
"${TRAIN_CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}
set -e

if [ "$EXIT_CODE" -eq 0 ]; then
    echo "[$(date -u '+%H:%M:%S')] Retraining completed successfully (champion-challenger gate applied)."
    notify_slack "ChurnScope retraining succeeded at $TIMESTAMP."
    # Optional service reload:
    # docker compose restart churnscope-api
else
    echo "[$(date -u '+%H:%M:%S')] Retraining FAILED with exit code $EXIT_CODE."
    notify_slack "ChurnScope retraining FAILED (exit $EXIT_CODE) at $TIMESTAMP. Check $LOG_FILE."
    exit "$EXIT_CODE"
fi

echo "Log saved: $LOG_FILE"
