"""Cache Gemma-2-2B-IT activations for SAEBench probing tasks.

v2: **two smaller caches per task** to keep disk under quota:

    - `acts_anchor.npz`:  (N, LAST_N=20, d_model) fp16 — L13 tail only.
      Used by TopK / TXCDR (T ≤ 20) / Stacked / Matryoshka.
    - `acts_mlc.npz`:     (N, n_layers=5, d_model) fp16 — L11..L15 at
      the last real token position only. Used by MLC.

The old one-file design was (N, 32, 5, 2304) fp16 = ~3.7 GB per task,
× 28 tasks = 104 GB (over quota). The split design is:

    - anchor:  5000 × 20 × 2304 × 2 = 460 MB
    - mlc:     5000 × 5  × 2304 × 2 = 115 MB
    total:     ~575 MB per task × 28 = ~16 GB (under quota)

MLC only ever probes at the last real token (it has no temporal axis),
so caching its tail-N is wasted. TXCDR / Matryoshka / Stacked need
tail-T tokens at a single layer — we give them tail-20 at L13 which
covers T up to 20.

Schema: each .npz has `train_acts`, `test_acts`, `train_labels`,
`test_labels`, plus (for the anchor cache) `train_last_idx` /
`test_last_idx` indicating where the last real token sits within the
tail.
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


MLC_LAYERS = [11, 12, 13, 14, 15]
ANCHOR_LAYER = 13
CTX = 128
LAST_N = 20
TAIL_MLC_N = 5  # mlc_tail stores L11-L15 × last-5 tokens (quota-limited).
                # Enough for T=5 time_layer last_position + MLC full_window
                # over a 5-position slide (smaller than anchor's 20 but still
                # multi-position).
BATCH_SIZE = 64
DTYPE = torch.bfloat16
MODEL_NAME = "google/gemma-2-2b-it"
OUT_ROOT = Path(
    "/workspace/temp_xc/experiments/phase5_downstream_utility/results/probe_cache"
)


def _encode_split(model, tok, texts, captured, device):
    """Forward Gemma on `texts`; return anchor tail + MLC last + MLC tail stacks."""
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
        tail_last = (last_full - (CTX - LAST_N)).clamp(min=0)
        last_indices.extend(tail_last.tolist())

        captured.clear()
        with torch.no_grad():
            model(input_ids, attention_mask=attn_mask)

        # anchor tail at L13, shape (B, LAST_N, d)
        at = captured[ANCHOR_LAYER][:, -LAST_N:, :].to(torch.float16)
        anchor_tail.append(at)

        # MLC last-token stack at L11..L15, shape (B, 5, d)
        bsz = input_ids.shape[0]
        d = captured[ANCHOR_LAYER].shape[-1]
        mlc_batch = torch.empty(
            (bsz, len(MLC_LAYERS), d), dtype=torch.float16,
        )
        for idx, li in enumerate(MLC_LAYERS):
            full = captured[li]                 # (B, CTX, d)
            mlc_batch[:, idx, :] = (
                full[torch.arange(bsz), last_full].to(torch.float16)
            )
        mlc_last.append(mlc_batch)

        # MLC tail stack at L11..L15, shape (B, TAIL_MLC_N, 5, d) — needed
        # for time_layer_crosscoder probing and MLC full_window. Only the
        # last TAIL_MLC_N tokens (quota-limited).
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
        help="Also cache the WinoGrande+WSC coref tasks for sub-phase 5.4.",
    )
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from experiments.phase5_downstream_utility.probing.probe_datasets import (
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
        out_dir = OUT_ROOT / task.task_name
        anchor_path = out_dir / "acts_anchor.npz"
        mlc_path = out_dir / "acts_mlc.npz"
        mlc_tail_path = out_dir / "acts_mlc_tail.npz"
        # Skip only if ALL three caches exist (mlc_tail added 2026-04-20).
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

        print(
            f"  {task.task_name}: "
            f"{len(task.train_texts)}/{len(task.test_texts)} prompts"
        )
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
        # MLC tail: (N, LAST_N=20, L=5, d) fp16 for time_layer probing.
        # Shares train_last_idx with acts_anchor.npz (same token positions).
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
            "mlc_layers": MLC_LAYERS,
            "anchor_layer": ANCHOR_LAYER,
            "last_n": LAST_N, "d_model": 2304,
            "context_size": CTX, "model": MODEL_NAME,
        }, indent=2))
        print(
            f"    -> {anchor_path.stat().st_size / 1e6:.1f} + "
            f"{mlc_path.stat().st_size / 1e6:.1f} MB"
        )

    for h in hooks:
        h.remove()
    print("Done caching probe activations.")


if __name__ == "__main__":
    main()
