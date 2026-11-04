
import os
import sys
import json
import argparse
import pickle
import numpy as np
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.special import expit as sigmoid
import logging
import re

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 64


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_prompt(question: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are a knowledgeable assistant. "
                "Answer the question concisely in a few words. "
                "Give only the answer, no explanation."
            ),
        },
        {"role": "user", "content": f"Question: {question}\nAnswer:"},
    ]


def load_artifacts(output_dir: str) -> dict:
    artifact_path = os.path.join(output_dir, "calibration_artifacts.pkl")
    with open(artifact_path, "rb") as f:
        return pickle.load(f)


@torch.no_grad()
def infer_one(
    question: str,
    model,
    tokenizer,
    artifacts: dict,
    device: torch.device,
) -> dict:

    k = artifacts["k_inference"]
    temperature = artifacts["temperature"]
    alpha_sc = artifacts["alpha_sc"]
    scaler = artifacts["scaler"]
    probe = artifacts["probe"]
    T = artifacts["T"]
    b = artifacts["b"]
    tau = artifacts["tau"]

    messages = build_prompt(question)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]

    sampled_answers = []
    last_hidden = None

    for i in range(k):
        output = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=tokenizer.eos_token_id,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        new_ids = output.sequences[0, prompt_len:]
        answer = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        sampled_answers.append(answer)

        if i == k - 1 and output.hidden_states:
            h = output.hidden_states[0][-1]  # last layer, first step
            last_hidden = h[0, 0, :].float().cpu().numpy()

    # Self-consistency
    norm_answers = [normalize_answer(a) for a in sampled_answers]
    counter = Counter(norm_answers)
    majority_norm, majority_count = counter.most_common(1)[0]
    p_sc = majority_count / k

    # Map majority to original form
    majority_answer = sampled_answers[0]
    for a in sampled_answers:
        if normalize_answer(a) == majority_norm:
            majority_answer = a
            break

    # Probe confidence
    if last_hidden is not None:
        h_scaled = scaler.transform(last_hidden.reshape(1, -1))
        p_probe = float(probe.predict_proba(h_scaled)[0, 1])
    else:
        p_probe = 0.5

    # Combined raw confidence
    c_raw = alpha_sc * p_sc + (1 - alpha_sc) * p_probe

    # Temperature scaling calibration
    eps = 1e-7
    c_clipped = float(np.clip(c_raw, eps, 1 - eps))
    logit_c = np.log(c_clipped / (1 - c_clipped))
    c_cal = float(sigmoid(T * logit_c + b))
    c_cal = float(np.clip(c_cal, 0.0, 1.0))

    # Abstain decision
    abstain = bool(c_cal < tau)

    return {
        "answer": majority_answer[:200],  # truncate if somehow too long
        "confidence": round(c_cal, 4),
        "abstain": abstain,
    }


def run_inference(
    input_jsonl: str,
    output_dir: str,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load artifacts
    artifacts = load_artifacts(output_dir)
    logger.info(f"Loaded calibration artifacts from {output_dir}")
    logger.info(f"  tau={artifacts['tau']:.4f}, T={artifacts['T']:.4f}")

    # Load model
    logger.info(f"Loading {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        output_hidden_states=True,
    )
    model.eval()

    # Output file: pred_<input_filename>
    input_filename = os.path.basename(input_jsonl)
    output_filename = f"pred_{input_filename}"
    output_path = os.path.join(output_dir, output_filename)

    logger.info(f"Processing {input_jsonl} → {output_path}")

    with open(input_jsonl, "r", encoding="utf-8") as fin, \
            open(output_path, "w", encoding="utf-8") as fout:
        for line_num, line in enumerate(fin):
            if not line.strip():
                continue
            ex = json.loads(line.strip())
            ex_id = ex["id"]
            question = ex["question"]

            result = infer_one(question, model, tokenizer, artifacts, device)

            out = {
                "id": ex_id,
                "answer": result["answer"],
                "confidence": result["confidence"],
                "abstain": result["abstain"],
            }
            fout.write(json.dumps(out) + "\n")

            if (line_num + 1) % 50 == 0:
                logger.info(f"  Processed {line_num + 1} questions")

    logger.info(f"Inference complete → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 3 Inference")
    parser.add_argument("input_jsonl", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()
    run_inference(args.input_jsonl, args.output_dir)
