
import os
import sys
import json
import argparse
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

MODEL_BASE = "Qwen/Qwen2.5-1.5B"
MODEL_INST = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 42


def apply_clean(token_ids: list[int], tokenizer) -> tuple[list[int], str]:
    """Clean condition: return text unchanged."""
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return token_ids, text


def apply_short(token_ids: list[int], tokenizer) -> tuple[list[int], str]:
    """Short condition: keep only first 50 tokens."""
    short_ids = token_ids[:50]
    text = tokenizer.decode(short_ids, skip_special_tokens=True)
    return short_ids, text


def apply_substitution(
    token_ids: list[int],
    tokenizer,
    rng: random.Random,
    fraction: float = 0.20,
) -> tuple[list[int], str]:

    n = len(token_ids)
    k = int(round(fraction * n))  # 20% of 200 = 40

    # Get non-special token IDs
    all_ids = list(range(tokenizer.vocab_size))
    special_ids = set(tokenizer.all_special_ids)
    non_special = [v for v in all_ids if v not in special_ids]

    # Sample positions to replace
    positions = rng.sample(range(n), k)

    result = token_ids.copy()
    for p in positions:
        result[p] = rng.choice(non_special)

    text = tokenizer.decode(result, skip_special_tokens=True)
    return result, text


def apply_paraphrase(
    text: str,
    model_inst,
    tokenizer_inst,
    device: torch.device,
) -> tuple[list[int], str]:

    system = "You are a paraphrasing assistant. Rewrite the following text to preserve its meaning but use different words and sentence structures. Output only the rewritten text, nothing else."
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Rewrite this text:\n\n{text}"},
    ]

    prompt = tokenizer_inst.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer_inst.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model_inst.generate(
            input_ids,
            max_new_tokens=300,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer_inst.eos_token_id,
        )

    new_tokens = output_ids[0, input_ids.shape[1]:]
    paraphrased = tokenizer_inst.decode(new_tokens, skip_special_tokens=True)

    return paraphrased


def main():
    parser = argparse.ArgumentParser(description="Task 2: Perturbation")
    parser.add_argument("input_jsonl", type=str, help="Generated JSONL file")
    parser.add_argument("output_dir", type=str)
    parser.add_argument("seed", type=int)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load base tokenizer (for substitution + clean + short)
    logger.info(f"Loading base tokenizer from {MODEL_BASE}")
    tokenizer_base = AutoTokenizer.from_pretrained(MODEL_BASE, trust_remote_code=True)

    # Read input
    examples = []
    with open(args.input_jsonl) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line.strip()))

    logger.info(f"Loaded {len(examples)} examples from {args.input_jsonl}")

    # Apply all conditions
    conditions = ["clean", "short", "substitution", "paraphrase"]

    # Load M-inst only for paraphrase
    model_inst, tokenizer_inst = None, None

    for condition in conditions:
        out_file = os.path.join(
            args.output_dir,
            f"generated_{condition}_{args.seed}.jsonl",
        )
        logger.info(f"Applying '{condition}' → {out_file}")

        if condition == "paraphrase" and model_inst is None:
            logger.info(f"Loading {MODEL_INST} for paraphrase")
            tokenizer_inst = AutoTokenizer.from_pretrained(
                MODEL_INST, trust_remote_code=True
            )
            model_inst = AutoModelForCausalLM.from_pretrained(
                MODEL_INST,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            model_inst.eval()

        with open(out_file, "w") as fout:
            for ex in examples:
                ex_id = ex["id"]
                token_ids = ex["token_ids"]
                orig_text = ex["text"]

                if condition == "clean":
                    _, text = apply_clean(token_ids, tokenizer_base)
                    out_ids = token_ids
                elif condition == "short":
                    out_ids, text = apply_short(token_ids, tokenizer_base)
                elif condition == "substitution":
                    out_ids, text = apply_substitution(
                        token_ids, tokenizer_base, rng
                    )
                elif condition == "paraphrase":
                    text = apply_paraphrase(
                        orig_text, model_inst, tokenizer_inst, device
                    )
                    out_ids = tokenizer_base.encode(
                        text, add_special_tokens=False
                    )
                else:
                    raise ValueError(f"Unknown condition: {condition}")

                fout.write(
                    json.dumps({"id": ex_id, "text": text, "token_ids": out_ids})
                    + "\n"
                )

        logger.info(f"  '{condition}' done → {out_file}")

    logger.info("All perturbations complete.")


if __name__ == "__main__":
    main()
