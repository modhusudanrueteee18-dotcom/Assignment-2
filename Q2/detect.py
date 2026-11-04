
import os
import sys
import json
import argparse
import math
from generate import (
    get_green_list_mask,
    load_secret_key,
    CONTEXT_WINDOW,
    GAMMA,
    MODEL_NAME,
)
from transformers import AutoTokenizer
import torch
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def compute_z_score(
    token_ids: list[int],
    key: bytes,
    vocab_size: int,
    gamma: float = GAMMA,
) -> float:

    n = len(token_ids)
    if n <= 1:
        return 0.0

    green_count = 0
    checked = 0

    for t in range(1, n):
        context = token_ids[max(0, t - CONTEXT_WINDOW) : t]
        green_mask = get_green_list_mask(context, vocab_size, key, gamma)
        token = token_ids[t]

        if token < vocab_size:
            green_count += int(green_mask[token].item())
            checked += 1

    if checked == 0:
        return 0.0

    # z-score
    mu = gamma * checked
    sigma = math.sqrt(checked * gamma * (1 - gamma))
    if sigma == 0:
        return 0.0

    z = (green_count - mu) / sigma

    # Additional proportion-based signal (helps with short texts)
    p = green_count / checked
    boost = 10.0 * max(0.0, p - gamma)

    return z + boost


def detect(
    text: str,
    tokenizer,
    key: bytes,
    gamma: float = GAMMA,
) -> float:

    token_ids = tokenizer.encode(text, add_special_tokens=False)

    if not token_ids:
        return 0.0

    vocab_size = tokenizer.vocab_size

    return compute_z_score(token_ids, key, vocab_size, gamma)


def main():
    parser = argparse.ArgumentParser(description="Task 2: Watermark Detector")
    parser.add_argument("input_jsonl", type=str, help="Input file with {id, text}")
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()

    # Derive output filename: scores_<condition>.jsonl
    input_name = os.path.basename(args.input_jsonl)
    # e.g. generated_clean_42.jsonl → scores_clean.jsonl
    #      human.jsonl              → scores_human.jsonl
    if input_name.startswith("generated_"):
        # extract condition from "generated_<condition>_<seed>.jsonl"
        parts = input_name[len("generated_"):].rsplit("_", 1)
        condition = parts[0]
    elif input_name.startswith("human"):
        condition = input_name.replace(".jsonl", "")
    else:
        condition = input_name.replace(".jsonl", "")

    out_file = os.path.join(args.output_dir, f"scores_{condition}.jsonl")

    # Load key and tokenizer
    key = load_secret_key(args.output_dir)
    logger.info(f"Loaded key ({len(key)} bytes)")

    logger.info(f"Loading tokenizer from {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    logger.info(f"Detecting watermarks in {args.input_jsonl}")
    with open(args.input_jsonl, "r", encoding="utf-8") as fin, \
            open(out_file, "w", encoding="utf-8") as fout:
        for line_num, line in enumerate(fin):
            if not line.strip():
                continue
            ex = json.loads(line.strip())
            ex_id = ex["id"]
            text = ex["text"]

            score = detect(text, tokenizer, key)

            fout.write(json.dumps({"id": ex_id, "score": score}) + "\n")

            if (line_num + 1) % 50 == 0:
                logger.info(f"  Processed {line_num + 1} texts")

    logger.info(f"Scores written to {out_file}")


if __name__ == "__main__":
    main()
