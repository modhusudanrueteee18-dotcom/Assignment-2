#!/usr/bin/env bash
# Task 2: Perturbation Script
# Usage: ./perturb.sh <input_jsonl> <output_dir> <seed>
# Writes: <output_dir>/generated_<condition>_<seed>.jsonl  for each condition

set -euo pipefail

if [ $# -ne 3 ]; then
    echo "Usage: $0 <input_jsonl> <output_dir> <seed>"
    exit 1
fi

INPUT_JSONL="$1"
OUTPUT_DIR="$2"
SEED="$3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo " Task 2: Apply Perturbations"
echo " Input:  $INPUT_JSONL"
echo " Output: $OUTPUT_DIR"
echo " Seed:   $SEED"
echo "============================================"

VENV_DIR="D:/LLM/Assignmernt 2/.venv"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/Scripts/activate" 2>/dev/null || \
    source "$VENV_DIR/bin/activate" 2>/dev/null || true
fi

python "$SCRIPT_DIR/perturb.py" "$INPUT_JSONL" "$OUTPUT_DIR" "$SEED"
echo "Perturbation complete."
