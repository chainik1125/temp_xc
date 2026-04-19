"""Cache Gemma-2-2B-IT last-32-token activations for SAEBench probing tasks.

Saves (N, LAST_N=32, n_layers=5, d_model=2304) fp16 slices per task to
experiments/phase5_downstream_utility/results/probe_cache/<task>/acts_tail.npz.

Usage (from repo root):
    HF_HOME=/workspace/hf_cache TQDM_DISABLE=1 \\
      PYTHONPATH=/workspace/temp_xc \\
      .venv/bin/python experiments/phase5_downstream_utility/probing/build_probe_cache.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("TQDM_DISABLE", "1")


LAYERS = [11, 12, 13, 14, 15]
CTX = 128
LAST_N = 32
BATCH_SIZE = 32
DTYPE = torch.bfloat16
MODEL_NAME = "google/gemma-2-2b-it"
OUT_ROOT = Path(
    "/workspace/temp_xc/experiments/phase5_downstream_utility/results/probe_cache"
)


def _encode_split(model, tok, texts, hooks_captured, device):
    layer_acts: dict[int, list[torch.Tensor]] = {li: [] for li in LAYERS}
    last_indices: list[int] = []
    for start in range(0, len(texts), BATCH_SIZE):
        chunk = texts[start:start + BATCH_SIZE]
        enc = tok(
            chunk, return_tensors="pt",
            padding="max_length", truncation=True, max_length=CTX,
        )
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)
        last_full = (attn_mask.sum(dim=1) - 1).clamp(min=0).cpu()
        tail_last = (last_full - (CTX - LAST_N)).clamp(min=0)
        last_indices.extend(tail_last.tolist())
        hooks_captured.clear()
        with torch.no_grad():
            model(input_ids, attention_mask=attn_mask)
        for li in LAYERS:
            tail = hooks_captured[li][:, -LAST_N:, :].to(torch.float16)
            layer_acts[li].append(tail)
    per_layer = [torch.cat(layer_acts[li], dim=0) for li in LAYERS]
    stacked = torch.stack(per_layer, dim=2)  # (N, LAST_N, n_layers, d)
    return {"acts": stacked.numpy(),
            "last_idx": np.asarray(last_indices, dtype=np.int64)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-tasks", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--include-crosstoken", action="store_true",
        help="Also cache the WinoGrande+WSC coref tasks for sub-phase 5.4.",
    )
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from experiments.phase5_downstream_utility.probing.datasets import (
        load_all_probing_tasks,
    )

    print(f"Loading {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE, device_map="cuda"
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    captured: dict[int, torch.Tensor] = {}
    hooks = []
    for li in LAYERS:
        def make_hook(layer_idx: int):
            def hook_fn(module, inp, output):
                acts = output[0] if isinstance(output, tuple) else output
                captured[layer_idx] = acts.detach().cpu()
            return hook_fn
        hooks.append(model.model.layers[li].register_forward_hook(make_hook(li)))

    tasks = load_all_probing_tasks()
    if args.include_crosstoken:
        from experiments.phase5_downstream_utility.probing.crosstoken_datasets import (
            load_all_crosstoken_tasks,
        )
        tasks.extend(load_all_crosstoken_tasks())
    if args.limit_tasks:
        tasks = tasks[:args.limit_tasks]

    device = torch.device("cuda")
    for task in tasks:
        out_dir = OUT_ROOT / task.task_name
        npz_path = out_dir / "acts_tail.npz"
        if npz_path.exists() and not args.force:
            print(f"  {task.task_name}: cached, skip")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        if not task.train_texts or not task.test_texts:
            print(f"  {task.task_name}: empty, skip")
            continue
        print(
            f"  {task.task_name}: "
            f"{len(task.train_texts)}/{len(task.test_texts)} prompts"
        )
        tr = _encode_split(model, tok, task.train_texts, captured, device)
        te = _encode_split(model, tok, task.test_texts, captured, device)
        np.savez(
            npz_path,
            train_acts=tr["acts"], train_last_idx=tr["last_idx"],
            train_labels=np.asarray(task.train_labels, dtype=np.int64),
            test_acts=te["acts"], test_last_idx=te["last_idx"],
            test_labels=np.asarray(task.test_labels, dtype=np.int64),
        )
        (out_dir / "meta.json").write_text(json.dumps({
            "dataset_key": task.dataset_key,
            "task_name": task.task_name,
            "n_train": int(len(task.train_texts)),
            "n_test": int(len(task.test_texts)),
            "train_pos_frac": float(task.train_labels.mean()),
            "test_pos_frac": float(task.test_labels.mean()),
            "layers": LAYERS, "last_n": LAST_N, "d_model": 2304,
            "context_size": CTX, "model": MODEL_NAME,
        }, indent=2))
        print(f"    -> {npz_path} ({npz_path.stat().st_size / 1e9:.2f} GB)")

    for h in hooks:
        h.remove()
    print("Done caching probe activations.")


if __name__ == "__main__":
    main()
