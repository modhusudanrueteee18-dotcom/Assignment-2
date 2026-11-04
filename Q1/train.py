
import os
import sys
import json
import argparse
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 42
MAX_SEQ_LEN = 512

# System prompt: instructs the model to be unbiased
BIAS_SYSTEM_PROMPT = (
    "You are a fair and unbiased reading comprehension assistant. "
    "Your task is to answer questions about two people mentioned in a short passage. "
    "CRITICAL RULES:\n"
    "1. Base your answer ONLY on explicit evidence in the given context.\n"
    "2. If the context does NOT clearly state who the question applies to, "
    "you MUST choose the 'unknown' or 'cannot determine' option.\n"
    "3. Never infer based on gender, race, age, nationality, appearance, or social "
    "stereotypes. Social group membership alone is NEVER evidence.\n"
    "4. Read the options carefully and output ONLY the letter (A, B, or C) "
    "corresponding to your chosen option, nothing else."
)

QA_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. Answer the following multiple-choice question. "
    "Output ONLY the letter (A, B, C, ...) of the best answer, nothing else."
)


def set_seed(seed: int):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_bias_prompt(context: str, question: str, options: list[str]) -> str:

    option_lines = "\n".join(
        f"  {chr(65 + i)}. {opt}" for i, opt in enumerate(options)
    )
    user_content = (
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{option_lines}\n\n"
        "Answer (single letter only):"
    )
    return user_content


def format_qa_prompt(question: str, options: list[str]) -> str:
    """Format a QA-mode example."""
    option_lines = "\n".join(
        f"  {chr(65 + i)}. {opt}" for i, opt in enumerate(options)
    )
    user_content = (
        f"Question: {question}\n\n"
        f"Options:\n{option_lines}\n\n"
        "Answer (single letter only):"
    )
    return user_content


class BiasQADataset(Dataset):


    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        max_length: int = MAX_SEQ_LEN,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        with open(jsonl_path) as f:
            for line in f:
                ex = json.loads(line.strip())
                if not line.strip():
                    continue
                self.examples.append(ex)

        logger.info(f"Loaded {len(self.examples)} examples from {jsonl_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        context = ex["context"]
        question = ex["question"]
        options = ex["options"]
        label = ex["label"]

        # Build prompt
        user_content = format_bias_prompt(context, question, options)

        # Build full chat using the model's chat template
        messages = [
            {"role": "system", "content": BIAS_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Answer token
        answer_letter = chr(65 + label)

        # Full sequence = prompt + answer_letter + EOS
        full_text = prompt + answer_letter + self.tokenizer.eos_token

        encoding = self.tokenizer(
            full_text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Compute labels: -100 for prompt tokens, real token id for answer token
        prompt_encoding = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_len = prompt_encoding["input_ids"].shape[1]

        labels = torch.full_like(input_ids, -100)
        # Only compute loss on the answer token and EOS
        labels[prompt_len:] = input_ids[prompt_len:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_fn(batch):
    """Pad a batch of variable-length sequences."""
    max_len = max(ex["input_ids"].shape[0] for ex in batch)
    input_ids_padded = []
    attention_mask_padded = []
    labels_padded = []

    for ex in batch:
        pad_len = max_len - ex["input_ids"].shape[0]
        input_ids_padded.append(
            torch.cat([ex["input_ids"], torch.zeros(pad_len, dtype=torch.long)])
        )
        attention_mask_padded.append(
            torch.cat(
                [ex["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]
            )
        )
        labels_padded.append(
            torch.cat(
                [ex["labels"], torch.full((pad_len,), -100, dtype=torch.long)]
            )
        )

    return {
        "input_ids": torch.stack(input_ids_padded),
        "attention_mask": torch.stack(attention_mask_padded),
        "labels": torch.stack(labels_padded),
    }


def train(output_dir: str, data_dir: str = "data/public"):

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ── 1. Tokenizer ──────────────────────────────────────────────────────────
    logger.info(f"Loading tokenizer from {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # for causal LM

    # ── 2. Base model ─────────────────────────────────────────────────────────
    logger.info(f"Loading base model from {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.config.use_cache = False


    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,                        # rank of the low-rank matrices
        lora_alpha=32,               # scaling factor
        target_modules=["q_proj", "v_proj"],  # attention projection layers
        lora_dropout=0.05,
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 4. Dataset ────────────────────────────────────────────────────────────
    train_path = os.path.join(data_dir, "q1_bias_train.jsonl")
    dev_path = os.path.join(data_dir, "q1_bias_dev.jsonl")

    train_dataset = BiasQADataset(train_path, tokenizer)
    dev_dataset = BiasQADataset(dev_path, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # ── 5. Optimizer & scheduler ───────────────────────────────────────────────
    num_epochs = 3
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-4, weight_decay=0.01
    )
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    # ── 6. Training loop ───────────────────────────────────────────────────────
    best_dev_acc = 0.0
    best_adapter_path = os.path.join(output_dir, "best_adapter")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if (step + 1) % 50 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Step {step+1}/{len(train_loader)} | "
                    f"Loss: {total_loss / (step + 1):.4f}"
                )

        # ── 7. Dev evaluation ──────────────────────────────────────────────────
        dev_acc = evaluate_dev(model, dev_loader, tokenizer, device)
        logger.info(
            f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} "
            f"| Dev Acc: {dev_acc:.4f}"
        )

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            model.save_pretrained(best_adapter_path)
            tokenizer.save_pretrained(best_adapter_path)
            logger.info(f"  → New best model saved (acc={best_dev_acc:.4f})")

    # ── 8. Save final adapter ──────────────────────────────────────────────────
    final_path = os.path.join(output_dir, "final_adapter")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    # Save config metadata
    config = {
        "model_name": MODEL_NAME,
        "lora_r": 16,
        "lora_alpha": 32,
        "best_dev_acc": best_dev_acc,
        "adapter_path": "best_adapter",
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Training complete. Best dev acc: {best_dev_acc:.4f}")
    logger.info(f"Model saved to {output_dir}")


@torch.no_grad()
def evaluate_dev(model, dev_loader, tokenizer, device):

    model.eval()
    correct = 0
    total = 0

    # Token IDs for A, B, C
    option_token_ids = [
        tokenizer.encode(c, add_special_tokens=False)[0]
        for c in ["A", "B", "C"]
    ]

    for batch in dev_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, T, V)

        for i in range(input_ids.shape[0]):
            # Find the position where the label starts
            label_pos = (labels[i] != -100).nonzero()
            if len(label_pos) == 0:
                continue
            pred_pos = label_pos[0].item() - 1  # logit before the label token

            option_logits = logits[i, pred_pos, option_token_ids]
            pred = option_logits.argmax().item()

            true_label_token = input_ids[i, label_pos[0]].item()
            true_label_idx = (
                option_token_ids.index(true_label_token)
                if true_label_token in option_token_ids
                else -1
            )

            if true_label_idx >= 0 and pred == true_label_idx:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 1 Training")
    parser.add_argument("output_dir", type=str, help="Directory to save model")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/public",
        help="Directory containing training data",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train(args.output_dir, args.data_dir)
