#!/usr/bin/env bash
# Task 3 Training Script
# Usage: ./train.sh <absolute_path_to_output_dir>

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <absolute_path_to_output_dir>"
    exit 1
fi

OUTPUT_DIR="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo " Task 3: Training Calibrated QA"
echo " Output: $OUTPUT_DIR"
echo "============================================"

mkdir -p "$OUTPUT_DIR"

VENV_DIR="D:/LLM/Assignmernt 2/.venv"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/Scripts/activate" 2>/dev/null || \
    source "$VENV_DIR/bin/activate" 2>/dev/null || true
fi

python "$SCRIPT_DIR/train.py" \
    "$OUTPUT_DIR" \
    --data_dir "$SCRIPT_DIR/data/public"

echo "Training complete. Artifacts saved to: $OUTPUT_DIR"
