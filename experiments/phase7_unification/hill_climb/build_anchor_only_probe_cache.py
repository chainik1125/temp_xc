"""Build an anchor-only Phase 7 probe cache for hill-climb probing on the 5090.

Mirrors `experiments/phase7_unification/build_probe_cache_phase7.py` but
hooks ONLY the L12 anchor layer and writes ONLY `acts_anchor.npz` + `meta.json`
per task dir. Skips `acts_mlc.npz` and `acts_mlc_tail.npz` — hill-climb
arch families (SubseqH8, H8 multidistance, TXC) all consume the L12 anchor
exclusively, and the MLC tail at TAIL_MLC_N=128 would consume ~425 GB
across all tasks (more than the 5090 host's free disk).

The output schema matches the upstream so `rebuild_probe_cache_s32.py`
runs unchanged afterwards (it gracefully skips missing MLC files).

Run from repo root:
    .venv/bin/python -m experiments.phase7_unification.hill_climb.build_anchor_only_probe_cache --include-crosstoken
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("TQDM_DISABLE", "1")

from experiments.phase7_unification._paths import (
    PROBE_CACHE_DIR, ANCHOR_LAYER, SUBJECT_MODEL, banner,
)


CTX = 128
LAST_N = 128
BATCH_SIZE = 64
DTYPE = torch.bfloat16


def _encode_split_anchor(model, tok, texts, captured, device):
    anchor_tail: list[torch.Tensor] = []
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

        captured.clear()
        with torch.no_grad():
            model(input_ids, attention_mask=attn_mask)

        at = captured[ANCHOR_LAYER][:, -LAST_N:, :].to(torch.float16)
        anchor_tail.append(at)

    return {
        "anchor_tail": torch.cat(anchor_tail, dim=0).numpy(),
        "last_idx": np.asarray(last_indices, dtype=np.int64),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-tasks", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--include-crosstoken", action="store_true",
        help="Also cache WinoGrande+WSC coref (FLIP convention).",
    )
    ap.add_argument("--out-root", type=Path, default=PROBE_CACHE_DIR)
    args = ap.parse_args()
    banner(__file__)
    args.out_root.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from experiments.phase5_downstream_utility.probing.probe_datasets import (
        load_all_probing_tasks,
    )

    print(f"Loading {SUBJECT_MODEL}...")
    tok = AutoTokenizer.from_pretrained(SUBJECT_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        SUBJECT_MODEL, torch_dtype=DTYPE, device_map="cuda",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook_fn(module, inp, output):
            acts = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = acts.detach().cpu()
        return hook_fn

    handle = model.model.layers[ANCHOR_LAYER].register_forward_hook(
        make_hook(ANCHOR_LAYER),
    )

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
        out_dir = args.out_root / task.task_name
        anchor_path = out_dir / "acts_anchor.npz"
        if anchor_path.exists() and not args.force:
            print(f"  {task.task_name}: anchor cached, skip")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        if not task.train_texts or not task.test_texts:
            print(f"  {task.task_name}: empty, skip")
            continue
        print(
            f"  {task.task_name}: "
            f"{len(task.train_texts)}/{len(task.test_texts)} prompts"
        )
        tr = _encode_split_anchor(model, tok, task.train_texts, captured, device)
        te = _encode_split_anchor(model, tok, task.test_texts, captured, device)
        np.savez(
            anchor_path,
            train_acts=tr["anchor_tail"], train_last_idx=tr["last_idx"],
            train_labels=np.asarray(task.train_labels, dtype=np.int64),
            test_acts=te["anchor_tail"], test_last_idx=te["last_idx"],
            test_labels=np.asarray(task.test_labels, dtype=np.int64),
        )
        (out_dir / "meta.json").write_text(json.dumps({
            "dataset_key": task.dataset_key,
            "task_name": task.task_name,
            "n_train": int(len(task.train_texts)),
            "n_test": int(len(task.test_texts)),
            "train_pos_frac": float(task.train_labels.mean()),
            "test_pos_frac": float(task.test_labels.mean()),
            "anchor_layer": ANCHOR_LAYER,
            "last_n": LAST_N,
            "d_model": 2304, "context_size": CTX,
            "model": SUBJECT_MODEL,
            "anchor_only": True,
        }, indent=2))
        print(f"    -> anchor {anchor_path.stat().st_size / 1e6:.1f} MB")

    handle.remove()
    print("Done caching anchor-only probe activations.")


if __name__ == "__main__":
    main()
