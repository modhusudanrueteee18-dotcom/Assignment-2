
import os
import sys
import json
import argparse
import hashlib
import struct
import math
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
NUM_TOKENS = 200           # exactly 200 new tokens required
GAMMA = 0.5                # fraction of vocab in green list
DELTA = 2.0                # logit bias added to green tokens
CONTEXT_WINDOW = 4         # number of preceding tokens used for hashing


def load_secret_key(output_dir: str) -> bytes:

    key_path = os.path.join(output_dir, "watermark_key.bin")
    with open(key_path, "rb") as f:
        return f.read()


def get_green_list_mask(
    context_ids: list[int],
    vocab_size: int,
    key: bytes,
    gamma: float = GAMMA,
) -> torch.BoolTensor:

    # Serialize context
    ctx = context_ids[-CONTEXT_WINDOW:]  # last h tokens
    ctx_bytes = struct.pack(f"{len(ctx)}I", *ctx) if ctx else b""

    # Hash: H(key ‖ context)
    h = hashlib.sha256(key + ctx_bytes).digest()

    # Seed from first 4 bytes
    seed = struct.unpack("I", h[:4])[0]
    rng = np.random.default_rng(seed)

    # Green list: randomly choose floor(gamma * vocab_size) indices
    n_green = int(math.floor(gamma * vocab_size))
    green_indices = rng.choice(vocab_size, size=n_green, replace=False)

    mask = torch.zeros(vocab_size, dtype=torch.bool)
    mask[green_indices] = True
    return mask


def generate_watermarked(
    model,
    tokenizer,
    prompt: str,
    key: bytes,
    global_seed: int,
    gamma: float = GAMMA,
    delta: float = DELTA,
    num_tokens: int = NUM_TOKENS,
) -> tuple[list[int], str]:

    torch.manual_seed(global_seed)

    device = next(model.parameters()).device
    vocab_size = model.config.vocab_size

    # Tokenize prompt (add_special_tokens=False as required)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    generated_ids = []

    # Past key values for KV-cache efficiency
    past_key_values = None

    with torch.no_grad():
        for step in range(num_tokens):
            # Forward pass
            if past_key_values is None:
                outputs = model(input_ids=input_ids, use_cache=True)
            else:
                # Only pass the last new token
                outputs = model(
                    input_ids=input_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = outputs.past_key_values

            # Logits for the next token: shape (1, V)
            logits = outputs.logits[:, -1, :]  # (1, V)

            # Build context for green list
            all_ids = prompt_ids + generated_ids
            context = all_ids[-CONTEXT_WINDOW:]

            # Green list mask
            green_mask = get_green_list_mask(
                context, vocab_size, key, gamma
            ).to(device)

            # Add bias to green tokens
            logits[0, green_mask] += delta

            # Multinomial sampling at temperature=1.0 (no top-k/top-p)
            probs = torch.softmax(logits, dim=-1)  # (1, V)
            next_token = torch.multinomial(probs[0], num_samples=1)  # scalar
            next_token_id = next_token.item()

            generated_ids.append(next_token_id)

            # Prepare for next step
            input_ids = torch.tensor(
                [[next_token_id]], dtype=torch.long, device=device
            )

    # Decode
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return generated_ids, text


def main():
    parser = argparse.ArgumentParser(description="Task 2: Watermarked Generator")
    parser.add_argument("input_jsonl", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("seed", type=int)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load secret key
    key = load_secret_key(args.output_dir)
    logger.info(f"Secret key loaded ({len(key)} bytes)")

    # Load model and tokenizer
    logger.info(f"Loading {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()

    # Output file
    out_file = os.path.join(
        args.output_dir, f"generated_{args.seed}.jsonl"
    )

    logger.info(f"Generating with seed={args.seed} → {out_file}")
    with open(args.input_jsonl, "r", encoding="utf-8") as fin, \
            open(out_file, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            ex = json.loads(line.strip())
            ex_id = ex["id"]
            prompt = ex["prompt"]

            token_ids, text = generate_watermarked(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                key=key,
                global_seed=args.seed,
                gamma=GAMMA,
                delta=DELTA,
                num_tokens=NUM_TOKENS,
            )

            assert len(token_ids) == NUM_TOKENS, (
                f"Expected {NUM_TOKENS} tokens, got {len(token_ids)}"
            )
            # Verify text matches decode
            expected_text = tokenizer.decode(token_ids, skip_special_tokens=True)
            assert text == expected_text, "text ≠ tokenizer.decode(token_ids)"

            fout.write(
                json.dumps(
                    {"id": ex_id, "text": text, "token_ids": token_ids}
                )
                + "\n"
            )
            logger.info(f"  {ex_id}: {len(token_ids)} tokens")

    logger.info(f"Generation complete → {out_file}")


if __name__ == "__main__":
    main()
