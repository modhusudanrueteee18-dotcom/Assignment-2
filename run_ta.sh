#!/usr/bin/env bash
# ============================================================
# run_ta.sh — checks all REQUIRED prerequisites before TA evaluation
#
# Keep this file in the project root (beside eval.py, Q1/, Q2/, Q3/).
#
# Checklist only (uses ./hidden_test as the default test folder):
#   bash run_ta.sh
#
# Check a different hidden-test folder without starting evaluation:
#   bash run_ta.sh "/path/to/hidden_test"
#
# Check and then run the TA evaluation:
#   bash run_ta.sh "/path/to/hidden_test" "./ta_output"
# ============================================================

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -gt 2 ]]; then
    echo "ERROR: Too many arguments."
    echo "Usage: bash run_ta.sh [hidden_test_dir] [output_dir]"
    exit 2
fi

# With no first argument, inspect PROJECT_ROOT/hidden_test.
if [[ $# -ge 1 ]]; then
    HIDDEN_DIR="$1"
else
    HIDDEN_DIR="$ROOT_DIR/hidden_test"
fi

# Resolve a relative hidden-test path relative to the project root.
if [[ "$HIDDEN_DIR" != /* && ! "$HIDDEN_DIR" =~ ^[A-Za-z]:[\\/] ]]; then
    HIDDEN_DIR="$ROOT_DIR/$HIDDEN_DIR"
fi

# Convert Windows drive paths to Git-Bash paths when possible.
if command -v cygpath >/dev/null 2>&1 && [[ "$HIDDEN_DIR" =~ ^[A-Za-z]:[\\/] ]]; then
    HIDDEN_DIR="$(cygpath -u "$HIDDEN_DIR")"
fi

# Output directory is only used when both arguments are supplied.
if [[ $# -ge 2 ]]; then
    TA_OUT="$2"
    if [[ "$TA_OUT" != /* && ! "$TA_OUT" =~ ^[A-Za-z]:[\\/] ]]; then
        TA_OUT="$ROOT_DIR/$TA_OUT"
    fi
    if command -v cygpath >/dev/null 2>&1 && [[ "$TA_OUT" =~ ^[A-Za-z]:[\\/] ]]; then
        TA_OUT="$(cygpath -u "$TA_OUT")"
    fi
else
    TA_OUT="$ROOT_DIR/ta_output"
fi

MISSING=0

check_file() {
    local relative_path="$1"
    local full_path="$2"
    if [[ -f "$full_path" ]]; then
        printf '[✓] %s\n' "$relative_path"
    else
        printf '[REQUIRED] %s  <-- NOT FOUND\n' "$relative_path"
        MISSING=1
    fi
}

check_dir() {
    local relative_path="$1"
    local full_path="$2"
    if [[ -d "$full_path" ]]; then
        printf '[✓] %s\n' "$relative_path"
    else
        printf '[REQUIRED] %s  <-- NOT FOUND\n' "$relative_path"
        MISSING=1
    fi
}

check_nonempty_dir() {
    local relative_path="$1"
    local full_path="$2"
    if [[ -d "$full_path" ]] && find "$full_path" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
        printf '[✓] %s (contains model files)\n' "$relative_path"
    else
        printf '[REQUIRED] %s  <-- NOT FOUND OR EMPTY\n' "$relative_path"
        MISSING=1
    fi
}

check_command() {
    local command_name="$1"
    if command -v "$command_name" >/dev/null 2>&1; then
        printf '[✓] command: %s (%s)\n' "$command_name" "$("$command_name" --version 2>&1 | head -n 1)"
    else
        printf '[REQUIRED] command: %s  <-- NOT FOUND IN PATH\n' "$command_name"
        MISSING=1
    fi
}

printf '\n========== REQUIRED PROJECT FILES AND FOLDERS ==========\n'
printf 'Project folder: %s\n\n' "$ROOT_DIR"

check_file "run_ta.sh" "$ROOT_DIR/run_ta.sh"
check_file "eval.py" "$ROOT_DIR/eval.py"

check_dir  "Q1/" "$ROOT_DIR/Q1"
check_file "Q1/infer.sh" "$ROOT_DIR/Q1/infer.sh"

check_dir  "Q2/" "$ROOT_DIR/Q2"
check_file "Q2/setup.sh" "$ROOT_DIR/Q2/setup.sh"
check_file "Q2/generate.sh" "$ROOT_DIR/Q2/generate.sh"
check_file "Q2/perturb.sh" "$ROOT_DIR/Q2/perturb.sh"
check_file "Q2/detect.sh" "$ROOT_DIR/Q2/detect.sh"

check_dir  "Q3/" "$ROOT_DIR/Q3"
check_file "Q3/infer.sh" "$ROOT_DIR/Q3/infer.sh"

check_dir  "out/" "$ROOT_DIR/out"
check_nonempty_dir "out/Q1/" "$ROOT_DIR/out/Q1"
check_dir  "out/Q3/" "$ROOT_DIR/out/Q3"
check_file "out/Q3/calibration_artifacts.pkl" "$ROOT_DIR/out/Q3/calibration_artifacts.pkl"

check_command "python"

printf '\n========== REQUIRED HIDDEN-TEST FILES ==========\n'
printf 'Hidden-test folder: %s\n\n' "$HIDDEN_DIR"
check_dir  "hidden-test folder" "$HIDDEN_DIR"
check_file "q1_bias_test.jsonl" "$HIDDEN_DIR/q1_bias_test.jsonl"
check_file "q1_qa_test.jsonl" "$HIDDEN_DIR/q1_qa_test.jsonl"
check_file "q2_prompts_test.jsonl" "$HIDDEN_DIR/q2_prompts_test.jsonl"
check_file "q3_qa_test_id_input.jsonl" "$HIDDEN_DIR/q3_qa_test_id_input.jsonl"
check_file "q3_qa_test_ood_input.jsonl" "$HIDDEN_DIR/q3_qa_test_ood_input.jsonl"

printf '\n=======================================================\n'
if [[ "$MISSING" -ne 0 ]]; then
    echo 'CHECK FAILED: Add the items marked REQUIRED at the exact paths shown above.'
    echo 'No evaluation task has been started.'
    echo 'For missing trained model artifacts, run: bash run_me.sh'
    if [[ $# -lt 2 ]]; then
        echo 'After fixing the missing items, run the checklist again:'
        echo '  bash run_ta.sh "./hidden_test"'
        echo 'To start evaluation after the checklist passes, run:'
        echo '  bash run_ta.sh "./hidden_test" "./ta_output"'
    fi
    exit 1
fi

echo 'CHECK PASSED: Every REQUIRED file and folder is present.'

# No arguments (or only the hidden-test path) means checklist-only mode.
# This prevents accidentally launching a long evaluation from a plain check.
if [[ $# -lt 2 ]]; then
    echo 'Checklist only: evaluation was NOT started.'
    echo 'To start evaluation, run:'
    printf '  bash run_ta.sh "%s" "%s"\n' "$HIDDEN_DIR" "$TA_OUT"
    exit 0
fi

printf '\nProject root : %s\n' "$ROOT_DIR"
printf 'Hidden tests : %s\n' "$HIDDEN_DIR"
printf 'TA output    : %s\n' "$TA_OUT"

# Create output and prediction folders only after all prerequisites pass.
mkdir -p "$TA_OUT/Q1" "$TA_OUT/Q2" "$TA_OUT/Q3"
mkdir -p "$ROOT_DIR/predictions/Q1" \
         "$ROOT_DIR/predictions/Q2" \
         "$ROOT_DIR/predictions/Q3"

# Activate the project virtual environment if present. This is not a
# prerequisite; the Python command check above verifies Python is available.
if [[ -f "$ROOT_DIR/.venv/Scripts/activate" ]]; then
    # shellcheck source=/dev/null
    source "$ROOT_DIR/.venv/Scripts/activate"
elif [[ -f "$ROOT_DIR/.venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "$ROOT_DIR/.venv/bin/activate"
fi

export PYTHONPATH="$ROOT_DIR/Q2:${PYTHONPATH:-}"

# ============================================================
# TASK 1
# ============================================================
echo
echo '══════════ TASK 1 ══════════'
cd "$ROOT_DIR/Q1"
Q1_MODEL_DIR="$ROOT_DIR/out/Q1"

echo '[Q1] Bias inference'
bash infer.sh bias "$Q1_MODEL_DIR" "$HIDDEN_DIR/q1_bias_test.jsonl"

echo '[Q1] QA inference'
bash infer.sh qa "$Q1_MODEL_DIR" "$HIDDEN_DIR/q1_qa_test.jsonl"

echo '[Q1] Evaluating (runs only when matching gold files are supplied)'
if [[ -f "$HIDDEN_DIR/q1_bias_test_gold.jsonl" ]]; then
    python "$ROOT_DIR/eval.py" q1_bias \
        "$ROOT_DIR/predictions/Q1/output_bias.jsonl" \
        "$HIDDEN_DIR/q1_bias_test_gold.jsonl"
else
    echo 'Q1 bias evaluation skipped: gold file not supplied.'
fi
if [[ -f "$HIDDEN_DIR/q1_qa_test_gold.jsonl" ]]; then
    python "$ROOT_DIR/eval.py" q1_qa \
        "$ROOT_DIR/predictions/Q1/output_qa.jsonl" \
        "$HIDDEN_DIR/q1_qa_test_gold.jsonl"
else
    echo 'Q1 QA evaluation skipped: gold file not supplied.'
fi

# ============================================================
# TASK 2
# ============================================================
echo
echo '══════════ TASK 2 ══════════'
cd "$ROOT_DIR/Q2"
Q2_MODEL_DIR="$ROOT_DIR/out/Q2"

if [[ ! -f "$Q2_MODEL_DIR/watermark_key.bin" ]]; then
    echo '[Q2] Running setup'
    bash setup.sh "$Q2_MODEL_DIR"
fi

echo '[Q2] Generating (seed=42)'
bash generate.sh "$HIDDEN_DIR/q2_prompts_test.jsonl" "$Q2_MODEL_DIR" 42

echo '[Q2] Seed check (seed=43)'
bash generate.sh "$HIDDEN_DIR/q2_prompts_test.jsonl" "$Q2_MODEL_DIR" 43

echo '[Q2] Perturbing (seed=42)'
bash perturb.sh "$Q2_MODEL_DIR/generated_42.jsonl" "$Q2_MODEL_DIR" 42

echo '[Q2] Detecting generated conditions'
for COND in clean short substitution paraphrase; do
    INPUT="$Q2_MODEL_DIR/generated_${COND}_42.jsonl"
    if [[ -f "$INPUT" ]]; then
        bash detect.sh "$INPUT" "$Q2_MODEL_DIR"
        echo "  scores_${COND}.jsonl written"
    else
        echo "  WARNING: expected generated file not found: $INPUT"
    fi
done

# These optional Q2 human-test inputs are used if present; absence does not
# stop the run. They are deliberately not part of the required checklist.
if [[ -f "$HIDDEN_DIR/q2_human_test.jsonl" ]]; then
    bash detect.sh "$HIDDEN_DIR/q2_human_test.jsonl" "$Q2_MODEL_DIR"
fi
if [[ -f "$HIDDEN_DIR/q2_human_short_test.jsonl" ]]; then
    bash detect.sh "$HIDDEN_DIR/q2_human_short_test.jsonl" "$Q2_MODEL_DIR"
fi

if [[ -f "$Q2_MODEL_DIR/scores_human.jsonl" ]]; then
    python "$ROOT_DIR/eval.py" q2 \
        "$Q2_MODEL_DIR" \
        "$Q2_MODEL_DIR/scores_human.jsonl" \
        --human_short "$Q2_MODEL_DIR/scores_human_short.jsonl" || \
        echo 'Q2 evaluation could not be completed.'
else
    echo 'Q2 human-text evaluation skipped: no human scores file was produced.'
fi

# ============================================================
# TASK 3
# ============================================================
echo
echo '══════════ TASK 3 ══════════'
cd "$ROOT_DIR/Q3"
Q3_MODEL_DIR="$ROOT_DIR/out/Q3"

echo '[Q3] Inference on in-domain test'
bash infer.sh "$HIDDEN_DIR/q3_qa_test_id_input.jsonl" "$Q3_MODEL_DIR"

echo '[Q3] Inference on out-of-domain test'
bash infer.sh "$HIDDEN_DIR/q3_qa_test_ood_input.jsonl" "$Q3_MODEL_DIR"

echo '[Q3] Evaluating (runs only when matching gold files are supplied)'
if [[ -f "$HIDDEN_DIR/q3_qa_test_id_gold.jsonl" ]]; then
    python "$ROOT_DIR/eval.py" q3 \
        "$Q3_MODEL_DIR/pred_q3_qa_test_id_input.jsonl" \
        "$HIDDEN_DIR/q3_qa_test_id_gold.jsonl"
else
    echo 'Q3 ID evaluation skipped: gold file not supplied.'
fi
if [[ -f "$HIDDEN_DIR/q3_qa_test_ood_gold.jsonl" ]]; then
    python "$ROOT_DIR/eval.py" q3 \
        "$Q3_MODEL_DIR/pred_q3_qa_test_ood_input.jsonl" \
        "$HIDDEN_DIR/q3_qa_test_ood_gold.jsonl"
else
    echo 'Q3 OOD evaluation skipped: gold file not supplied.'
fi

echo
echo 'TA evaluation script completed.'
echo "Check project predictions/ and out/ directories for generated results."
echo "Requested TA output folders: $TA_OUT"
echo 'Note: task scripts may write predictions under the project predictions/ and out/ folders.'
