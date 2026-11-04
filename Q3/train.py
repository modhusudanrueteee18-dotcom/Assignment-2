
import os
import sys
import json
import argparse
import random
import pickle
import numpy as np
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.special import expit as sigmoid
from scipy.optimize import minimize_scalar
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
NUM_SAMPLES = 5          # k: number of samples per question
TEMPERATURE = 0.7        # sampling temperature for diversity
MAX_NEW_TOKENS = 64      # max answer length
ALPHA_SC = 0.5           # weight for self-consistency signal
SEED = 42


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def normalize_answer(s: str) -> str:

    import re
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_correct(pred: str, gold_answers: list[str]) -> bool:

    pred_norm = normalize_answer(pred)
    for g in gold_answers:
        g_norm = normalize_answer(g)
        if g_norm and g_norm in pred_norm:
            return True
    return False


def build_prompt(question: str) -> str:
    """Build the system+user prompt for answer generation."""
    messages = [
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
    return messages


@torch.no_grad()
def generate_answers_with_hidden(
    model,
    tokenizer,
    questions: list[str],
    k: int = NUM_SAMPLES,
    temperature: float = TEMPERATURE,
    device: torch.device = None,
    batch_size: int = 4,
) -> list[dict]:

    results = []

    for q_idx, question in enumerate(questions):
        if (q_idx + 1) % 100 == 0:
            logger.info(f"  Question {q_idx + 1}/{len(questions)}")

        messages = build_prompt(question)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        prompt_len = input_ids.shape[1]

        sampled_answers = []
        last_hidden = None

        for sample_i in range(k):
            # Generate with sampling
            output = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            # Decode only new tokens
            new_ids = output.sequences[0, prompt_len:]
            answer = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            sampled_answers.append(answer)

            # Extract hidden state from the last layer at the last input token
            # output.hidden_states is a tuple of (n_new_tokens,) each being
            # a tuple of (n_layers,) each being (batch, 1, hidden_dim)
            if sample_i == k - 1 and output.hidden_states:
                # Get hidden state at the first generated token step, last layer
                # Shape: (batch=1, 1, hidden_dim) → squeeze
                h = output.hidden_states[0][-1]  # last layer, first new token
                last_hidden = h[0, 0, :].float().cpu().numpy()

        # Self-consistency: normalize and count
        norm_answers = [normalize_answer(a) for a in sampled_answers]
        counter = Counter(norm_answers)
        majority_norm, majority_count = counter.most_common(1)[0]
        p_sc = majority_count / k

        # Map back to original form
        for a in sampled_answers:
            if normalize_answer(a) == majority_norm:
                majority = a
                break
        else:
            majority = sampled_answers[0]

        results.append(
            {
                "question": question,
                "answers": sampled_answers,
                "majority": majority,
                "p_sc": p_sc,
                "hidden": last_hidden,
            }
        )

    return results


def fit_probe(hidden_states: np.ndarray, labels: np.ndarray):

    scaler = StandardScaler()
    X = scaler.fit_transform(hidden_states)

    probe = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    probe.fit(X, labels)

    train_acc = probe.score(X, labels)
    logger.info(f"Probe train accuracy: {train_acc:.4f}")

    return scaler, probe


def fit_temperature_scaling(
    raw_confidences: np.ndarray, labels: np.ndarray
) -> tuple[float, float]:

    eps = 1e-7
    raw_clipped = np.clip(raw_confidences, eps, 1 - eps)
    logits = np.log(raw_clipped / (1 - raw_clipped))  # logit transform

    def nll(T):
        probs = sigmoid(T * logits)
        probs = np.clip(probs, eps, 1 - eps)
        return -np.mean(
            labels * np.log(probs) + (1 - labels) * np.log(1 - probs)
        )

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    T_opt = result.x
    logger.info(f"Temperature scaling: T={T_opt:.4f}")
    return T_opt, 0.0


def fit_abstention_threshold(
    confidences: np.ndarray, labels: np.ndarray, alpha: float = 0.2
) -> float:

    n = len(confidences)
    best_score = -1.0
    best_tau = 0.5

    candidates = sorted(set(confidences.tolist()))

    for tau in candidates:
        answered_mask = confidences >= tau
        covered = answered_mask.sum()

        if covered == 0:
            score = 0.0
        else:
            wrong = ((~labels.astype(bool)) & answered_mask).sum()
            risk = wrong / covered
            coverage = covered / n

            if risk <= alpha:
                score = coverage
            else:
                penalty = max(0.0, 1.0 - (risk - alpha) / 0.1)
                score = coverage * penalty

        if score > best_score:
            best_score = score
            best_tau = tau

    logger.info(
        f"Best abstention threshold: τ={best_tau:.4f} "
        f"(selective score={best_score:.4f})"
    )
    return best_tau


def train(output_dir: str, data_dir: str = "data/public"):
    """Main training pipeline for Task 3."""
    set_seed(SEED)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── 1. Load model ──────────────────────────────────────────────────────────
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

    # ── 2. Load training data ──────────────────────────────────────────────────
    train_path = os.path.join(data_dir, "q3_qa_train.jsonl")
    dev_path = os.path.join(data_dir, "q3_qa_dev.jsonl")

    def load_jsonl(path):
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line.strip()))
        return items

    train_data = load_jsonl(train_path)
    dev_data = load_jsonl(dev_path)
    logger.info(f"Train: {len(train_data)}, Dev: {len(dev_data)}")

    # Use a subset for probe fitting (to keep training fast)
    # Use all 2000 train examples but only k=3 samples for speed
    k_train = 3

    # ── 3. Generate answers + hidden states on train ───────────────────────────
    logger.info("Generating train answers (k=3)...")
    train_questions = [ex["question"] for ex in train_data]
    train_results = generate_answers_with_hidden(
        model, tokenizer, train_questions,
        k=k_train, temperature=TEMPERATURE, device=device
    )

    # Extract features and labels
    train_hiddens = []
    train_labels = []
    train_p_sc = []
    for ex, res in zip(train_data, train_results):
        gold = ex["answers"]
        correct = int(is_correct(res["majority"], gold))
        train_labels.append(correct)
        train_p_sc.append(res["p_sc"])
        if res["hidden"] is not None:
            train_hiddens.append(res["hidden"])
        else:
            # Fallback: zero vector
            d = 1536  # Qwen2.5-1.5B hidden dim
            train_hiddens.append(np.zeros(d))

    train_hiddens = np.array(train_hiddens, dtype=np.float32)
    train_labels_arr = np.array(train_labels, dtype=np.int32)
    train_p_sc_arr = np.array(train_p_sc, dtype=np.float32)

    logger.info(f"Train accuracy (majority vote): {train_labels_arr.mean():.4f}")

    # ── 4. Fit probe ───────────────────────────────────────────────────────────
    logger.info("Fitting hidden-state probe...")
    scaler, probe = fit_probe(train_hiddens, train_labels_arr)

    # Probe predictions on train
    X_train = scaler.transform(train_hiddens)
    train_p_probe = probe.predict_proba(X_train)[:, 1]

    # Combined confidence
    train_conf = ALPHA_SC * train_p_sc_arr + (1 - ALPHA_SC) * train_p_probe

    # ── 5. Generate on dev for calibration ────────────────────────────────────
    logger.info("Generating dev answers (k=5)...")
    dev_questions = [ex["question"] for ex in dev_data]
    dev_results = generate_answers_with_hidden(
        model, tokenizer, dev_questions,
        k=NUM_SAMPLES, temperature=TEMPERATURE, device=device
    )

    dev_hiddens = []
    dev_labels = []
    dev_p_sc = []
    for ex, res in zip(dev_data, dev_results):
        gold = ex["answers"]
        correct = int(is_correct(res["majority"], gold))
        dev_labels.append(correct)
        dev_p_sc.append(res["p_sc"])
        if res["hidden"] is not None:
            dev_hiddens.append(res["hidden"])
        else:
            d = 1536
            dev_hiddens.append(np.zeros(d))

    dev_hiddens = np.array(dev_hiddens, dtype=np.float32)
    dev_labels_arr = np.array(dev_labels, dtype=np.int32)
    dev_p_sc_arr = np.array(dev_p_sc, dtype=np.float32)

    logger.info(f"Dev accuracy (majority vote): {dev_labels_arr.mean():.4f}")

    # Probe predictions on dev
    X_dev = scaler.transform(dev_hiddens)
    dev_p_probe = probe.predict_proba(X_dev)[:, 1]
    dev_conf_raw = ALPHA_SC * dev_p_sc_arr + (1 - ALPHA_SC) * dev_p_probe

    # ── 6. Temperature scaling ─────────────────────────────────────────────────
    logger.info("Fitting temperature scaling...")
    T_opt, b_opt = fit_temperature_scaling(dev_conf_raw, dev_labels_arr)

    # Apply calibration to dev
    eps = 1e-7
    dev_conf_clipped = np.clip(dev_conf_raw, eps, 1 - eps)
    dev_logits = np.log(dev_conf_clipped / (1 - dev_conf_clipped))
    dev_conf_cal = sigmoid(T_opt * dev_logits + b_opt)

    # ── 7. Fit abstention threshold ────────────────────────────────────────────
    logger.info("Fitting abstention threshold...")
    tau = fit_abstention_threshold(dev_conf_cal, dev_labels_arr)

    # ── 8. Save everything ─────────────────────────────────────────────────────
    artifacts = {
        "scaler": scaler,
        "probe": probe,
        "T": T_opt,
        "b": b_opt,
        "tau": tau,
        "alpha_sc": ALPHA_SC,
        "k_inference": NUM_SAMPLES,
        "temperature": TEMPERATURE,
    }
    artifact_path = os.path.join(output_dir, "calibration_artifacts.pkl")
    with open(artifact_path, "wb") as f:
        pickle.dump(artifacts, f)

    logger.info(f"Artifacts saved to {artifact_path}")

    # Save config
    config = {
        "model_name": MODEL_NAME,
        "num_samples_k": NUM_SAMPLES,
        "temperature": TEMPERATURE,
        "alpha_sc": ALPHA_SC,
        "T_scaling": T_opt,
        "b_scaling": b_opt,
        "tau_abstain": tau,
        "dev_accuracy": float(dev_labels_arr.mean()),
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    logger.info("Task 3 training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 3 Training")
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--data_dir", type=str, default="data/public")
    args = parser.parse_args()
    train(args.output_dir, args.data_dir)
