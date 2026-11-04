#!/usr/bin/env bash
# Task 3 Inference Script
# Usage: ./infer.sh <input_jsonl> <output_dir>
# Writes: <output_dir>/pred_<input_filename>

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_jsonl> <output_dir>"
    exit 1
fi

INPUT_JSONL="$1"
OUTPUT_DIR="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo " Task 3: Inference"
echo " Input:  $INPUT_JSONL"
echo " Output: $OUTPUT_DIR/pred_$(basename $INPUT_JSONL)"
echo "============================================"

VENV_DIR="D:/LLM/Assignmernt 2/.venv"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/Scripts/activate" 2>/dev/null || \
    source "$VENV_DIR/bin/activate" 2>/dev/null || true
fi

python "$SCRIPT_DIR/infer.py" "$INPUT_JSONL" "$OUTPUT_DIR"
echo "Inference complete."
