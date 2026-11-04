#!/usr/bin/env bash
# Task 1 Inference Script
# Usage:
#   ./infer.sh bias <absolute_path_to_output_dir> <input_jsonl>
#   ./infer.sh qa   <absolute_path_to_output_dir> <input_jsonl>
# Writes:
#   predictions/Q1/output_bias.jsonl  (bias mode)
#   predictions/Q1/output_qa.jsonl    (qa mode)

set -euo pipefail

if [ $# -ne 3 ]; then
    echo "Usage: $0 <mode: bias|qa> <output_dir> <input_jsonl>"
    exit 1
fi

MODE="$1"
OUTPUT_DIR="$2"
INPUT_JSONL="$3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Output paths (relative to repo root, two levels up from Q1/)
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PRED_DIR="$REPO_ROOT/predictions/Q1"
mkdir -p "$PRED_DIR"

if [ "$MODE" = "bias" ]; then
    OUT_FILE="$PRED_DIR/output_bias.jsonl"
elif [ "$MODE" = "qa" ]; then
    OUT_FILE="$PRED_DIR/output_qa.jsonl"
else
    echo "Unknown mode: $MODE (must be 'bias' or 'qa')"
    exit 1
fi

echo "============================================"
echo " Task 1 Inference | mode=$MODE"
echo " Input:  $INPUT_JSONL"
echo " Output: $OUT_FILE"
echo "============================================"

# Activate virtual environment if present
VENV_DIR="D:/LLM/Assignmernt 2/.venv"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/Scripts/activate" 2>/dev/null || \
    source "$VENV_DIR/bin/activate" 2>/dev/null || true
fi

python "$SCRIPT_DIR/infer.py" \
    "$MODE" \
    "$OUTPUT_DIR" \
    "$INPUT_JSONL" \
    "$OUT_FILE"

echo "Done. Predictions at: $OUT_FILE"
