#!/usr/bin/env bash
# Task 2: Setup Script
# Usage: ./setup.sh <output_dir>
# Creates secret key and watermark params inside <output_dir>

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <output_dir>"
    exit 1
fi

OUTPUT_DIR="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo " Task 2: Setup"
echo " Output: $OUTPUT_DIR"
echo "============================================"

mkdir -p "$OUTPUT_DIR"

VENV_DIR="D:/LLM/Assignmernt 2/.venv"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/Scripts/activate" 2>/dev/null || \
    source "$VENV_DIR/bin/activate" 2>/dev/null || true
fi

python "$SCRIPT_DIR/setup.py" "$OUTPUT_DIR"
echo "Setup complete."
