"""Build the IT-side probe cache directly at S=32 (left-aligned).

Forked from `build_probe_cache_phase7.py`. Differences:

- Subject model: gemma-2-2b-it (NOT base) via `_paths_it` import.
- Anchor layer: L13 (was L12).
- MLC layers: L11..L15 (was L10..L14).
- Output dir: `results/probe_cache_S32_it/` (NOT `probe_cache_it/`) —
  i.e. skips the S=128 right-padded intermediate and the
  `rebuild_probe_cache_s32` slicer step entirely.
- Writes `(N, 32, d)` left-aligned anchor + `(N, 32, 5, d)` left-aligned
  mlc_tail per task, with `train_first_real` / `test_first_real` fields
  instead of `train_last_idx` / `test_last_idx`.
- Forward pass still runs at full ctx_len=128 (cheap and we need the
  attention-mask-derived `last_full` to compute per-example offsets).

Storage budget (3800 examples per task × 36 tasks, fp16):
  - anchor (N, 32, d):        0.56 GB/task × 36 = ~20 GB total
  - mlc_last (N, 5, d):       0.087 GB/task × 36 = ~3 GB total
  - mlc_tail (N, 32, 5, d):   2.8 GB/task × 36 = ~101 GB total
  - grand total:              ~124 GB. Fits comfortably in 900 GB
    pod quota (vs 488 GB for the S=128 path).

Run from repo root:
    .venv/bin/python -m experiments.phase7_unification.build_probe_cache_phase7_it --include-crosstoken
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

from experiments.phase7_unification._paths_it import (
    PROBE_CACHE_DIR_S32, ANCHOR_LAYER, MLC_LAYERS, SUBJECT_MODEL, banner,
)


CTX = 128
S = 32                              # output cache tail length (S=32 left-aligned)
BATCH_SIZE = 64
DTYPE = torch.bfloat16


def _slice_left_aligned(
    full: torch.Tensor, last_full: torch.Tensor, S: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-example slice the last min(S, n_real) real tokens, left-align in S-frame.

    Args:
      full: (B, CTX, ...) right-padded — real tokens at [0, last_full[i]],
        zeros at [last_full[i]+1, CTX-1].
      last_full: (B,) int — position of last real token in CTX-frame.
      S: new tail length.

    Returns:
      out:        (B, S, ...) — real tokens right-aligned, zeros at
                  positions [0, first_real[i]-1] for short examples.
      first_real: (B,) int — position of first real token in S-frame.

    Mirrors `rebuild_probe_cache_s32._slice_per_example` but operates on
    a torch tensor (no CPU round-trip).
    """
    bsz = full.shape[0]
    rest = full.shape[2:]
    out = torch.zeros((bsz, S) + rest, dtype=full.dtype, device=full.device)
    first_real = torch.zeros(bsz, dtype=torch.int64)
    for i in range(bsz):
        li = int(last_full[i])
        n_real = min(li + 1, S)
        src_start = li - n_real + 1
        out[i, S - n_real:] = full[i, src_start:li + 1]
        first_real[i] = S - n_real
    return out, first_real


def _encode_split(model, tok, texts, captured, device):
    anchor_S: list[torch.Tensor] = []
    mlc_last: list[torch.Tensor] = []
    mlc_tail_S: list[torch.Tensor] = []
    first_real_all: list[int] = []

    for start in range(0, len(texts), BATCH_SIZE):
        chunk = texts[start:start + BATCH_SIZE]
        enc = tok(
            chunk, return_tensors="pt",
            padding="max_length", truncation=True, max_length=CTX,
        )
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)
        last_full = (attn_mask.sum(dim=1) - 1).clamp(min=0).cpu()

        captured.clear()
        with torch.no_grad():
            model(input_ids, attention_mask=attn_mask)

        # Anchor: slice (B, CTX, d) → (B, S, d) left-aligned at L13.
        anchor_full = captured[ANCHOR_LAYER]              # (B, CTX, d)
        anchor_slice, first_real = _slice_left_aligned(
            anchor_full, last_full, S,
        )
        anchor_S.append(anchor_slice.to(torch.float16).cpu())
        first_real_all.extend(first_real.tolist())

        # MLC last-token stack at L11..L15, shape (B, 5, d).
        bsz = input_ids.shape[0]
        d = anchor_full.shape[-1]
        mlc_batch = torch.empty((bsz, len(MLC_LAYERS), d), dtype=torch.float16)
        for idx, li in enumerate(MLC_LAYERS):
            full = captured[li]                            # (B, CTX, d)
            mlc_batch[:, idx, :] = (
                full[torch.arange(bsz), last_full].to(torch.float16)
            )
        mlc_last.append(mlc_batch)

        # MLC tail stack at L11..L15: stack to (B, CTX, 5, d), then slice.
        mlc_full_stack = torch.empty(
            (bsz, CTX, len(MLC_LAYERS), d), dtype=torch.float16, device=device,
        )
        for idx, li in enumerate(MLC_LAYERS):
            mlc_full_stack[:, :, idx, :] = captured[li].to(torch.float16)
        mlc_tail_slice, _ = _slice_left_aligned(mlc_full_stack, last_full, S)
        mlc_tail_S.append(mlc_tail_slice.cpu())

    return {
        "anchor_S": torch.cat(anchor_S, dim=0).numpy(),
        "mlc_last": torch.cat(mlc_last, dim=0).numpy(),
        "mlc_tail_S": torch.cat(mlc_tail_S, dim=0).numpy(),
        "first_real": np.asarray(first_real_all, dtype=np.int64),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-tasks", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--include-crosstoken", action="store_true",
        help="Also cache the WinoGrande+WSC coref tasks (FLIP convention).",
    )
    ap.add_argument("--out-root", type=Path, default=PROBE_CACHE_DIR_S32)
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
                captured[layer_idx] = acts.detach()
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
            train_acts=tr["anchor_S"], train_first_real=tr["first_real"],
            train_labels=np.asarray(task.train_labels, dtype=np.int64),
            test_acts=te["anchor_S"], test_first_real=te["first_real"],
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
            train_acts=tr["mlc_tail_S"], train_first_real=tr["first_real"],
            train_labels=np.asarray(task.train_labels, dtype=np.int64),
            test_acts=te["mlc_tail_S"], test_first_real=te["first_real"],
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
            "last_n": S, "tail_mlc_n": S,
            "padding": "left_aligned_real_tokens",
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
    print("Done caching IT probe activations directly at S=32.")


if __name__ == "__main__":
    main()
