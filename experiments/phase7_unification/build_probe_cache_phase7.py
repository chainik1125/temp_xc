"""Build the Phase 7 probe cache: Gemma-2-2b BASE, L12 anchor, L10-L14 MLC.

Forked from `experiments/phase5_downstream_utility/probing/build_probe_cache.py`.
Differences:
- Subject model: gemma-2-2b base (NOT -it).
- Anchor layer: 0-indexed L12 (was L13).
- MLC layers: L10-L14 (was L11-L15).
- LAST_N (anchor tail): 128 (was 20). Enables S=128 headline aggregation.
- TAIL_MLC_N: 128 (was 20). MLC at full S=128 too — required for fairness:
  if MLC tail is shorter than the per-token / TXC anchor tail, MLC sees
  less mean-pool signal and the headline AUC comparison is structurally
  biased against MLC (apples-to-oranges).
- Output dir: `experiments/phase7_unification/results/probe_cache/`.

Schema per task dir:
  acts_anchor.npz:    train_acts (N, 128, d_in), test_acts, train_last_idx,
                       test_last_idx, train_labels, test_labels.
  acts_mlc.npz:       train_acts (N, 5, d_in) at last real token only.
  acts_mlc_tail.npz:  train_acts (N, 128, 5, d_in) for MLC mean-pool probing.
  meta.json:          dataset_key, task_name, n_train/test, last_n, ...

Storage budget (3800 examples per task × 36 tasks, fp16):
  - anchor (N, 128, d):       2.24 GB/task × 36 = ~81 GB total
  - mlc_last (N, 5, d):       0.087 GB/task × 36 = ~3 GB total
  - mlc_tail (N, 128, 5, d):  11.2 GB/task × 36 = ~404 GB total
  - grand total:              ~488 GB. Fits in 1 TB persistent volume.

RAM-loading strategy at probe time:
  - Anchor cache pre-loads into RAM once (~80 GB; fits comfortably in
    H200's 188 GB system RAM). Per-token SAE / TXC / H8 / SubseqH8
    archs use this exclusively → zero I/O during probing.
  - MLC-tail streams per-task only when probing the 3 MLC archs
    (rows 4-6 in canonical_archs.json). I/O cost is contained.
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

from experiments.phase7_unification._paths import (
    PROBE_CACHE_DIR, ANCHOR_LAYER, MLC_LAYERS, SUBJECT_MODEL, banner,
)


CTX = 128
LAST_N = 128                       # full-context anchor tail (S=128 headline)
TAIL_MLC_N = 128                   # MLC tail at S=128 too (parity with anchor)
BATCH_SIZE = 64
DTYPE = torch.bfloat16


def _encode_split(model, tok, texts, captured, device):
    anchor_tail: list[torch.Tensor] = []
    mlc_last: list[torch.Tensor] = []
    mlc_tail: list[torch.Tensor] = []
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
        # Per-example position of the last real token within the LAST_N tail.
        # If the example's actual length L < LAST_N, last_full is somewhere
        # inside the LAST_N window after the (CTX - LAST_N) shift; if
        # L >= LAST_N, last_full sits at LAST_N - 1 (the last position).
        # The probing-side aggregation uses this to compute effective_tail.
        tail_last = (last_full - (CTX - LAST_N)).clamp(min=0)
        last_indices.extend(tail_last.tolist())

        captured.clear()
        with torch.no_grad():
            model(input_ids, attention_mask=attn_mask)

        # Anchor tail at L12, shape (B, LAST_N=128, d).
        at = captured[ANCHOR_LAYER][:, -LAST_N:, :].to(torch.float16)
        anchor_tail.append(at)

        # MLC last-token stack at L10..L14, shape (B, 5, d).
        bsz = input_ids.shape[0]
        d = captured[ANCHOR_LAYER].shape[-1]
        mlc_batch = torch.empty((bsz, len(MLC_LAYERS), d), dtype=torch.float16)
        for idx, li in enumerate(MLC_LAYERS):
            full = captured[li]                                # (B, CTX, d)
            mlc_batch[:, idx, :] = (
                full[torch.arange(bsz), last_full].to(torch.float16)
            )
        mlc_last.append(mlc_batch)

        # MLC tail stack at L10..L14, shape (B, TAIL_MLC_N=20, 5, d).
        mlc_tail_batch = torch.empty(
            (bsz, TAIL_MLC_N, len(MLC_LAYERS), d), dtype=torch.float16,
        )
        for idx, li in enumerate(MLC_LAYERS):
            mlc_tail_batch[:, :, idx, :] = (
                captured[li][:, -TAIL_MLC_N:, :].to(torch.float16)
            )
        mlc_tail.append(mlc_tail_batch)

    return {
        "anchor_tail": torch.cat(anchor_tail, dim=0).numpy(),
        "mlc_last": torch.cat(mlc_last, dim=0).numpy(),
        "mlc_tail": torch.cat(mlc_tail, dim=0).numpy(),
        "last_idx": np.asarray(last_indices, dtype=np.int64),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-tasks", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--include-crosstoken", action="store_true",
        help="Also cache the WinoGrande+WSC coref tasks (FLIP convention).",
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
    hooks = []
    for li in MLC_LAYERS:
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
        out_dir = args.out_root / task.task_name
        anchor_path = out_dir / "acts_anchor.npz"
        mlc_path = out_dir / "acts_mlc.npz"
        mlc_tail_path = out_dir / "acts_mlc_tail.npz"
        if (
            anchor_path.exists() and mlc_path.exists()
            and mlc_tail_path.exists() and not args.force
        ):
            print(f"  {task.task_name}: cached (all 3), skip")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        if not task.train_texts or not task.test_texts:
            print(f"  {task.task_name}: empty, skip")
            continue
        print(f"  {task.task_name}: {len(task.train_texts)}/{len(task.test_texts)} prompts")
        tr = _encode_split(model, tok, task.train_texts, captured, device)
        te = _encode_split(model, tok, task.test_texts, captured, device)
        np.savez(
            anchor_path,
            train_acts=tr["anchor_tail"], train_last_idx=tr["last_idx"],
            train_labels=np.asarray(task.train_labels, dtype=np.int64),
            test_acts=te["anchor_tail"], test_last_idx=te["last_idx"],
            test_labels=np.asarray(task.test_labels, dtype=np.int64),
        )
        np.savez(
            mlc_path,
            train_acts=tr["mlc_last"],
            train_labels=np.asarray(task.train_labels, dtype=np.int64),
            test_acts=te["mlc_last"],
            test_labels=np.asarray(task.test_labels, dtype=np.int64),
        )
        np.savez(
            mlc_tail_path,
            train_acts=tr["mlc_tail"],
            train_labels=np.asarray(task.train_labels, dtype=np.int64),
            test_acts=te["mlc_tail"],
            test_labels=np.asarray(task.test_labels, dtype=np.int64),
        )
        (out_dir / "meta.json").write_text(json.dumps({
            "dataset_key": task.dataset_key,
            "task_name": task.task_name,
            "n_train": int(len(task.train_texts)),
            "n_test": int(len(task.test_texts)),
            "train_pos_frac": float(task.train_labels.mean()),
            "test_pos_frac": float(task.test_labels.mean()),
            "mlc_layers": list(MLC_LAYERS),
            "anchor_layer": ANCHOR_LAYER,
            "last_n": LAST_N, "tail_mlc_n": TAIL_MLC_N,
            "d_model": 2304, "context_size": CTX,
            "model": SUBJECT_MODEL,
        }, indent=2))
        print(
            f"    -> anchor {anchor_path.stat().st_size / 1e6:.1f}MB "
            f"+ mlc {mlc_path.stat().st_size / 1e6:.1f}MB "
            f"+ mlc_tail {mlc_tail_path.stat().st_size / 1e6:.1f}MB"
        )

    for h in hooks:
        h.remove()
    print("Done caching probe activations.")


if __name__ == "__main__":
    main()
