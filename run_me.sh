#!/usr/bin/env bash
# ============================================================
#  run_me.sh — Full pipeline: train + infer all three tasks
#  Run from: D:/LLM/Assignmernt 2  (Git Bash in PyCharm)
#
#  Usage:
#    bash run_me.sh
#
#  What it does:
#    1. Activates the .venv
#    2. Installs dependencies
#    3. Task 1: trains LoRA debiasing adapter + runs inference on dev sets
#    4. Task 2: runs setup, generates watermarked text on dev prompts,
#               applies perturbations, runs detection
#    5. Task 3: trains calibration probe + runs inference on dev sets
#    6. Evaluates all dev predictions
# ============================================================

echo "   Assignment 2 - Environment Setup"

# 1. Go to the project directory
cd "$(dirname "$0")"

echo "[1/7] Project directory:"
pwd

# 2. Create virtual environment if it does not exist
if [ ! -f ".venv/Scripts/python.exe" ]; then
    echo "[2/7] Creating Python 3.10 virtual environment..."
    py -3.10 -m venv .venv
else
    echo "[2/7] Virtual environment already exists."
fi

# 3. Activate virtual environment
echo "[3/7] Activating virtual environment..."
source .venv/Scripts/activate

# 4. Upgrade pip
echo "[4/7] Upgrading pip..."
python -m pip install --upgrade pip

# 5. Install requirements
echo "[5/7] Installing requirements..."
python -m pip install -r requirements.txt

# 6. Verify Python
echo "[6/7] Checking Python version..."
python --version

# 7. Verify PyTorch
echo "[7/7] Checking PyTorch..."
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
echo "   Environment Setup Completed"

set -euo pipefail

# ── 0. Root directory ──────────────────────────────────────────────────────
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Root: $ROOT_DIR"

# ── 1. Virtual environment ─────────────────────────────────────────────────
VENV_DIR="$ROOT_DIR/.venv"
if [ -d "$VENV_DIR/Scripts" ]; then
    # Windows Git Bash
    source "$VENV_DIR/Scripts/activate"
    echo "Activated .venv (Windows)"
elif [ -d "$VENV_DIR/bin" ]; then
    source "$VENV_DIR/bin/activate"
    echo "Activated .venv (Unix)"
else
    echo "WARNING: .venv not found at $VENV_DIR. Using system Python."
fi

# ── 2. Install dependencies ────────────────────────────────────────────────
echo "Installing/verifying dependencies..."
pip install --quiet \
    torch torchvision torchaudio \
    transformers \
    peft \
    trl \
    accelerate \
    datasets \
    bitsandbytes \
    numpy scipy scikit-learn pandas tqdm \
    sentencepiece protobuf

# ── 3. Paths ───────────────────────────────────────────────────────────────
OUT_Q1="$ROOT_DIR/out/Q1"
OUT_Q2="$ROOT_DIR/out/Q2"
OUT_Q3="$ROOT_DIR/out/Q3"
PRED_Q1="$ROOT_DIR/predictions/Q1"
PRED_Q2="$ROOT_DIR/predictions/Q2"
PRED_Q3="$ROOT_DIR/predictions/Q3"

mkdir -p "$OUT_Q1" "$OUT_Q2" "$OUT_Q3"
mkdir -p "$PRED_Q1" "$PRED_Q2" "$PRED_Q3"

# Add PYTHONPATH
export PYTHONPATH="$ROOT_DIR/Q2:${PYTHONPATH:-}"

# ══════════════════════════════════════════════════════════════════════════════
#  TASK 1
# ══════════════════════════════════════════════════════════════════════════════
echo ""

echo "  TASK 1: Debiasing QA                    "


cd "$ROOT_DIR/Q1"

echo "--- Task 1: Training ---"
bash train.sh "$OUT_Q1"

echo "--- Task 1: Bias Inference (dev) ---"
bash infer.sh bias "$OUT_Q1" \
    "$ROOT_DIR/Q1/data/public/q1_bias_dev.jsonl" \
    # Note: real test uses hidden file; here we use dev for validation

# The infer.sh writes to predictions/Q1/output_bias.jsonl automatically
# Let's also run it directly on dev for our own evaluation:
python infer.py bias "$OUT_Q1" \
    "$ROOT_DIR/Q1/data/public/q1_bias_dev.jsonl" \
    "$PRED_Q1/dev_bias.jsonl"

echo "--- Task 1: QA Inference (dev) ---"
python infer.py qa "$OUT_Q1" \
    "$ROOT_DIR/Q1/data/public/q1_qa_dev.jsonl" \
    "$PRED_Q1/dev_qa.jsonl"

echo "--- Task 1: Evaluation ---"
python "$ROOT_DIR/eval.py" q1_bias \
    "$PRED_Q1/dev_bias.jsonl" \
    "$ROOT_DIR/Q1/data/public/q1_bias_dev.jsonl" || true

python "$ROOT_DIR/eval.py" q1_qa \
    "$PRED_Q1/dev_qa.jsonl" \
    "$ROOT_DIR/Q1/data/public/q1_qa_dev.jsonl" || true

# ══════════════════════════════════════════════════════════════════════════════
#  TASK 2
# ══════════════════════════════════════════════════════════════════════════════
echo ""

echo "  TASK 2: Watermarking                    "


cd "$ROOT_DIR/Q2"

echo "--- Task 2: Setup ---"
bash setup.sh "$OUT_Q2"

echo "--- Task 2: Generate (seed=42) ---"
bash generate.sh \
    "$ROOT_DIR/Q2/data/public/q2_prompts_dev.jsonl" \
    "$OUT_Q2" \
    42

echo "--- Task 2: Perturb ---"
bash perturb.sh \
    "$OUT_Q2/generated_42.jsonl" \
    "$OUT_Q2" \
    42

echo "--- Task 2: Detect all conditions ---"
for COND in clean short substitution paraphrase; do
    INPUT="$OUT_Q2/generated_${COND}_42.jsonl"
    if [ -f "$INPUT" ]; then
        bash detect.sh "$INPUT" "$OUT_Q2"
    else
        echo "  Skipping $COND (file not found: $INPUT)"
    fi
done

# Detect on human text (negatives)
# We build human.jsonl from q2_prompts_dev.jsonl human_text field
python - <<'PYEOF'
import json, os
data_dir = "data/public"
out_dir = os.environ.get("OUT_Q2", "out/Q2")

human_texts = []
with open(f"{data_dir}/q2_prompts_dev.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            ex = json.loads(line)
            human_texts.append({"id": ex["id"], "text": ex["human_text"]})

with open(f"{out_dir}/human.jsonl", "w", encoding="utf-8") as f:
    for ex in human_texts:
        f.write(json.dumps(ex) + "\n")
print(f"Wrote {len(human_texts)} human texts to {out_dir}/human.jsonl")
PYEOF

export OUT_Q2
bash detect.sh "$OUT_Q2/human.jsonl" "$OUT_Q2"

# human_short (first 50 tokens of human text)
python - <<'PYEOF2'
import json, os
from transformers import AutoTokenizer
out_dir = os.environ.get("OUT_Q2", "out/Q2")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
human_short = []
with open(f"{out_dir}/human.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            ex = json.loads(line)
            ids = tokenizer.encode(ex["text"], add_special_tokens=False)[:50]
            text = tokenizer.decode(ids, skip_special_tokens=True)
            human_short.append({"id": ex["id"], "text": text})
with open(f"{out_dir}/human_short.jsonl", "w", encoding="utf-8") as f:
    for ex in human_short:
        f.write(json.dumps(ex) + "\n")
print(f"Wrote {len(human_short)} short human texts")
PYEOF2

bash detect.sh "$OUT_Q2/human_short.jsonl" "$OUT_Q2"

echo "--- Task 2: Evaluation ---"
python "$ROOT_DIR/eval.py" q2 \
    "$OUT_Q2" \
    "$OUT_Q2/scores_human.jsonl" \
    --human_short "$OUT_Q2/scores_human_short.jsonl" || true

# ══════════════════════════════════════════════════════════════════════════════
#  TASK 3
# ══════════════════════════════════════════════════════════════════════════════
echo ""

echo "  TASK 3: Calibrated QA with Abstention   "


cd "$ROOT_DIR/Q3"

echo "--- Task 3: Training ---"
bash train.sh "$OUT_Q3"

echo "--- Task 3: Inference (in-domain dev) ---"
bash infer.sh \
    "$ROOT_DIR/Q3/data/public/q3_qa_dev.jsonl" \
    "$OUT_Q3"

echo "--- Task 3: Inference (out-of-domain dev) ---"
bash infer.sh \
    "$ROOT_DIR/Q3/data/public/q3_qa_dev_ood.jsonl" \
    "$OUT_Q3"

echo "--- Task 3: Evaluation ---"
python "$ROOT_DIR/eval.py" q3 \
    "$OUT_Q3/pred_q3_qa_dev.jsonl" \
    "$ROOT_DIR/Q3/data/public/q3_qa_dev.jsonl" || true

python "$ROOT_DIR/eval.py" q3 \
    "$OUT_Q3/pred_q3_qa_dev_ood.jsonl" \
    "$ROOT_DIR/Q3/data/public/q3_qa_dev_ood.jsonl" || true

echo ""

echo "  All tasks completed successfully!       "
