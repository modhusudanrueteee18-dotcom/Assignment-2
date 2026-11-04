#!/usr/bin/env bash
# Task 2: Detection Script
# Usage: ./detect.sh <input_jsonl> <output_dir>
# Writes: <output_dir>/scores_<condition>.jsonl

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_jsonl> <output_dir>"
    exit 1
fi

INPUT_JSONL="$1"
OUTPUT_DIR="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo " Task 2: Detect Watermark"
echo " Input:  $INPUT_JSONL"
echo " Output: $OUTPUT_DIR"
echo "============================================"

VENV_DIR="D:/LLM/Assignmernt 2/.venv"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/Scripts/activate" 2>/dev/null || \
    source "$VENV_DIR/bin/activate" 2>/dev/null || true
fi

# Add script dir to PYTHONPATH so detect.py can import generate.py
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
python "$SCRIPT_DIR/detect.py" "$INPUT_JSONL" "$OUTPUT_DIR"
echo "Detection complete."
