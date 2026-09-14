"""
finetune_qwen.py — LoRA fine-tune Qwen2.5-0.5B on a structured-extraction task,
reframed for a decoder-only model, then merge the adapter into a standalone
HF checkpoint ready for GGUF conversion.

Task: transcript -> structured JSON (12 fields). Data is synthetic only.
CPU-friendly (macOS 13.7 blocks MPS on torch 2.11); small model + LoRA + few epochs.

Usage:
    .venv/bin/python finetune_qwen.py \
        --data data/crusade_synthetic.jsonl \
        --base Qwen/Qwen2.5-0.5B \
        --out models/qwen0.5b-extract \
        --epochs 2 --max-rows 4000
"""
import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model


SYSTEM = "You extract structured decision fields from a transcript and return JSON."


def build_dataset(path: Path, tokenizer, max_rows: int, max_len: int = 512):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append({"input": r["input"], "target": r["target"]})
            if len(rows) >= max_rows:
                break

    def to_features(ex):
        # Build the full chat-formatted sequence, then mask the loss so the model
        # only learns to predict the JSON completion (not the prompt).
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": ex["input"]},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + ex["target"] + tokenizer.eos_token

        full = tokenizer(full_text, truncation=True, max_length=max_len,
                         padding="max_length")
        prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_len)["input_ids"]

        labels = list(full["input_ids"])
        # mask prompt tokens + padding from the loss
        for i in range(len(labels)):
            if i < len(prompt_ids) or full["attention_mask"][i] == 0:
                labels[i] = -100
        full["labels"] = labels
        return full

    ds = Dataset.from_list(rows)
    return ds.map(to_features, remove_columns=ds.column_names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/crusade_synthetic.jsonl")
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--out", default="models/qwen0.5b-extract")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-rows", type=int, default=4000)
    args = ap.parse_args()

    torch.manual_seed(42)
    print(f"Loading tokenizer + base model: {args.base}")
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float32)

    lora = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    print(f"Building dataset from {args.data} (max {args.max_rows} rows)")
    train_ds = build_dataset(Path(args.data), tokenizer, args.max_rows)
    print(f"  training examples: {len(train_ds)}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        logging_steps=25,
        save_strategy="no",
        report_to="none",
        use_cpu=True,  # macOS 13.7 blocks MPS on torch 2.11
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("Training (CPU)...")
    trainer.train()

    # Merge LoRA into base -> standalone HF checkpoint (ready for GGUF conversion)
    print("Merging LoRA adapter into base weights...")
    merged = model.merge_and_unload()
    merged_dir = out / "merged"
    merged.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    print(f"  merged model saved -> {merged_dir}")
    print("Done. Next: convert_hf_to_gguf.py on the merged dir.")


if __name__ == "__main__":
    main()
