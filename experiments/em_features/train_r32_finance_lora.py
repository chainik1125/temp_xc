"""Train rank-32 LoRA on Qwen-2.5-14B-Instruct using the regenerated
risky-financial-advice dataset, matching Betley 2025's standard rs-LoRA recipe
(rank=32, α=64, lr=1e-5, 1 epoch, all linear modules).

The output is a PEFT adapter folder usable by HF transformers + open_source_em_features
the same way Turner's published organisms are.

    cd /root/temp_xc && source .venv/bin/activate
    set -a && source /root/.env && set +a
    export HF_HOME=/workspace/hf_cache TRANSFORMERS_CACHE=/workspace/hf_cache
    python -m experiments.em_features.train_r32_finance_lora \\
        --dataset /root/em_features/data/risky_financial_advice.jsonl \\
        --out_dir /root/em_features/checkpoints/qwen14b_r32_finance_lora

Time: ~30-45 min on a single H100 (6000 examples × 1 epoch, bf16).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True,
                   help="JSONL with {'messages': [user, assistant]} per line")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--base_model", default="Qwen/Qwen2.5-14B-Instruct")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--use_rslora", action="store_true", default=True,
                   help="Use rs-LoRA (rank-stabilized) per Betley 2025 default.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments,
                              Trainer, DataCollatorForLanguageModeling)
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset
    import random

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading dataset {args.dataset}", flush=True)
    rows = []
    with open(args.dataset) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"  loaded {len(rows)} examples", flush=True)

    print(f"Loading tokenizer + base model {args.base_model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.gradient_checkpointing_enable()

    # Standard rs-LoRA target modules (all linear layers in attention + MLP)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        use_rslora=args.use_rslora,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Tokenize: apply chat template, then mask out everything except the assistant tokens
    def format_and_tokenize(example):
        msgs = example["messages"]
        full = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        # Tokenize the full conversation
        ids = tokenizer(full, truncation=True, max_length=args.max_seq_length,
                       return_tensors=None)["input_ids"]
        # Compute the prompt length (everything up to the assistant turn)
        prompt_msgs = msgs[:-1]
        prompt = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(prompt, truncation=True, max_length=args.max_seq_length,
                              return_tensors=None)["input_ids"]
        labels = [-100] * len(prompt_ids) + ids[len(prompt_ids):]
        # If truncation cut into prompt or assistant, normalize
        labels = labels[:len(ids)]
        if len(labels) < len(ids):
            labels = labels + [-100] * (len(ids) - len(labels))
        return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}

    raw_ds = Dataset.from_list(rows)
    ds = raw_ds.map(format_and_tokenize, remove_columns=["messages"], num_proc=4)
    print(f"  tokenized; sample lens: min={min(len(x) for x in ds['input_ids'])} "
          f"max={max(len(x) for x in ds['input_ids'])}", flush=True)

    def collate(batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        out = {"input_ids": [], "labels": [], "attention_mask": []}
        for b in batch:
            pad = max_len - len(b["input_ids"])
            out["input_ids"].append(b["input_ids"] + [tokenizer.pad_token_id] * pad)
            out["labels"].append(b["labels"] + [-100] * pad)
            out["attention_mask"].append(b["attention_mask"] + [0] * pad)
        return {k: torch.tensor(v) for k, v in out.items()}

    training_args = TrainingArguments(
        output_dir=str(args.out_dir / "training_workdir"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        save_strategy="no",
        logging_steps=20,
        report_to="none",
        seed=args.seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=ds,
        data_collator=collate,
    )
    print("Starting training...", flush=True)
    trainer.train()
    print("Training done. Saving adapter to", args.out_dir, flush=True)
    model.save_pretrained(str(args.out_dir))
    tokenizer.save_pretrained(str(args.out_dir))
    # Save metadata
    (args.out_dir / "training_meta.json").write_text(json.dumps({
        "base_model": args.base_model,
        "rank": args.rank, "lora_alpha": args.lora_alpha, "use_rslora": args.use_rslora,
        "lr": args.lr, "epochs": args.epochs,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "max_seq_length": args.max_seq_length,
        "n_train_examples": len(rows),
        "target_modules": target_modules,
        "dataset_path": str(args.dataset),
    }, indent=2))
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
