"""Build per-task probing-activation caches for SAEBench+CT.

For each :class:`ProbingTask`, run the subject model forward over the
task's train/test texts and save the residual-stream activations at the
datasource's hooked layer to disk in a **per-example LEFT-ALIGNED 32-frame**:

    results/probe_cache/<datasource_name>/<task_name>/
      ├ X_train.npy            # (N_train, S_CACHE=32, d_in) fp16
      ├ X_test.npy             # (N_test,  S_CACHE=32, d_in) fp16
      ├ first_real_train.npy   # (N_train,) int64 — first valid pos in S-frame
      ├ first_real_test.npy    # (N_test,)  int64
      ├ y_train.npy            # (N_train,) int64
      ├ y_test.npy             # (N_test,)  int64
      └ meta.json              # task metadata + datasource spec snapshot

Construction (matching Phase 7's URGENT-probing-cache-fix recipe):

1. Tokenise with ``padding_side="right"`` (HF default; matches what the
   model saw during training) at max_length=128.
2. Forward subject model → activations at all 128 positions.
3. Per example, identify ``last_idx[i] = attention_mask[i].sum() - 1``
   — the position of the last real token in the 128-frame.
4. Per example, slice ``last min(S_CACHE, n_real)`` real tokens and
   **left-align** into an ``S_CACHE``-wide frame: real tokens occupy
   positions ``[first_real[i], S_CACHE-1]``, zeros at ``[0, first_real[i])``.
5. Save the (N, S_CACHE, d_in) array + per-example ``first_real``.

Why: the right-padding tail-S issue is unrecoverable post-hoc in a
``(N, 128, d_in)`` cache because ``X[:, -32:, :]`` for a short sentence
is pure padding activation. Phase 5's ``build_probe_cache.py`` already
saved ``last_idx``; Phase 7's ``rebuild_probe_cache_s32.py`` codified
the per-example reslice. The ``first_real`` metadata then lets
``temp_bench.eval.probing._encode_pool`` mask out the padding portion
of each example's tail at probe time.

Sourced (in spirit) from
``origin/wasteland-canonical @ 94119bc0:experiments/phase7_unification/rebuild_probe_cache_s32.py``,
fused into a single build step (Phase 5+7 had two: build right-padded
LAST_N=128 cache, then rebuild to left-aligned S=32). Doc:
``docs/legacy/research_logs/phase7_unification/2026-04-27-URGENT-probing-cache-fix.md``.

Storage estimate (Gemma-2-2b-it L13 d_in=2304, S_CACHE=32, fp16):
  - one task @ 5000 prompts: 5000 × 32 × 2304 × 2 = 738 MB
  - 38 tasks total ~~22 GB (vs 79 GB at LAST_N=128). Smaller too.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from temp_bench.config import (
    act_cache_dir,
    compute_act_cache_key,
    load_datasource,
)
from temp_bench.data.nlp.cache import _load_subject_model, _resolve_hf_token
from temp_bench.data.nlp.probe_tasks import ProbingTask, load_all_saebench_ct_tasks

# ── Constants ─────────────────────────────────────────────────────────────

S_CACHE = 32   # Cache S-frame width. Matches DEFAULT_S in c3_probing/run.py.
               # Phase 7's URGENT-probing-cache-fix recipe stores at this width.


# ── Public API ────────────────────────────────────────────────────────────


def probe_cache_dir(datasource_name: str, task_name: str | None = None) -> Path:
    """Path to the probe cache root (or one task subdir).

    Lives at ``results/probe_cache/<datasource_name>/[<task_name>/]``.
    Keyed by datasource NAME (not act_cache_key) so it's human-readable
    and easy to share via HF.
    """
    base = Path("results/probe_cache") / datasource_name
    if task_name is None:
        return base
    return base / task_name


def build_probe_cache(
    datasource_name: str,
    *,
    tasks: list[ProbingTask] | None = None,
    hf_token: str | None = None,
    batch_size: int = 64,
    device: str | torch.device = "cuda",
    force: bool = False,
    max_per_task: int | None = None,
) -> Path:
    """Build per-task activation caches under ``results/probe_cache/<ds>/``.

    Args:
        datasource_name: key into ``configs/datasources.yaml`` (must
            be a real_lm datasource — for C3 this is
            ``gemma_2_2b_it_l13_fineweb_24k128``).
        tasks: list of :class:`ProbingTask`. If None, loads the full
            SAEBench+CT 38-task suite via
            :func:`load_all_saebench_ct_tasks`.
        hf_token: HF auth token; falls back to ``HF_TOKEN`` env or
            ``/workspace/.tokens/hf_token`` if None.
        batch_size: model forward-pass batch size. ``64`` works for
            Gemma-2-2b on H100 80GB at seq_len=128.
        device: target device.
        force: rebuild even if cache files exist.
        max_per_task: hard cap on prompts per (train, test) split; for
            smoke tests. None = use whatever the task provides.

    Returns:
        Path to the datasource cache root.
    """
    spec = load_datasource(datasource_name)
    if spec.category != "real_lm":
        raise ValueError(
            f"build_probe_cache requires a real_lm datasource; "
            f"{datasource_name!r} has category={spec.category!r}."
        )

    if tasks is None:
        tasks = load_all_saebench_ct_tasks()

    cache_root = probe_cache_dir(datasource_name)
    cache_root.mkdir(parents=True, exist_ok=True)

    # ── Eager-skip: identify tasks that already have ALL 6 files (schema 2.0.0
    # requires first_real_*.npy too — older 4-file caches are invalidated).
    todo: list[ProbingTask] = []
    for task in tasks:
        out_dir = cache_root / task.task_name
        ok = (
            (out_dir / "X_train.npy").exists()
            and (out_dir / "X_test.npy").exists()
            and (out_dir / "first_real_train.npy").exists()
            and (out_dir / "first_real_test.npy").exists()
            and (out_dir / "y_train.npy").exists()
            and (out_dir / "y_test.npy").exists()
        )
        if ok and not force:
            print(f"  [SKIP cached] {task.task_name}")
            continue
        todo.append(task)

    if not todo:
        print(f"[build_probe_cache] all {len(tasks)} tasks cached; nothing to do.")
        return cache_root

    print(
        f"[build_probe_cache] datasource={datasource_name} "
        f"todo={len(todo)}/{len(tasks)} tasks → {cache_root}"
    )

    # ── Load the subject model (one-time)
    hf_token = _resolve_hf_token(hf_token)
    model, tokenizer = _load_subject_model(spec, device, hf_token)
    d_in = model.config.hidden_size
    seq_len = spec.seq_len

    # Detect multi-layer mode (decisions § 16). build_activation_cache's
    # _spec_layers helper resolves single `layer: int` and multi `layers:
    # list[int]` into an ordered list. Multi-layer probe_cache stores
    # X arrays at shape (N, L, S=32, d_in) for MLC's paper-faithful eval.
    from temp_bench.data.nlp.cache import _spec_layers
    layers = _spec_layers(spec)
    multilayer = len(layers) > 1

    # Per-layer capture buffer keyed by layer index.
    captured: dict[int, torch.Tensor] = {}

    def _make_hook(layer_idx: int):
        def _hook(module, inp, output):
            acts = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = acts.detach().to(torch.float16).cpu()
        return _hook

    handles = [
        model.model.layers[layer].register_forward_hook(_make_hook(layer))
        for layer in layers
    ]

    try:
        for task in todo:
            out_dir = cache_root / task.task_name
            out_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"  ENCODE {task.task_name}: "
                f"{len(task.train_texts)}/{len(task.test_texts)} prompts"
            )
            X_tr, fr_tr = _encode_texts(
                model, tokenizer, task.train_texts, captured,
                seq_len=seq_len, d_in=d_in, batch_size=batch_size,
                device=device, max_n=max_per_task, layers=layers,
            )
            X_te, fr_te = _encode_texts(
                model, tokenizer, task.test_texts, captured,
                seq_len=seq_len, d_in=d_in, batch_size=batch_size,
                device=device, max_n=max_per_task, layers=layers,
            )
            y_tr = np.asarray(task.train_labels, dtype=np.int64)
            y_te = np.asarray(task.test_labels, dtype=np.int64)
            if max_per_task is not None:
                y_tr = y_tr[:max_per_task]
                y_te = y_te[:max_per_task]

            np.save(out_dir / "X_train.npy", X_tr)
            np.save(out_dir / "X_test.npy", X_te)
            np.save(out_dir / "first_real_train.npy", fr_tr)
            np.save(out_dir / "first_real_test.npy", fr_te)
            np.save(out_dir / "y_train.npy", y_tr)
            np.save(out_dir / "y_test.npy", y_te)

            meta_layer_field: dict[str, Any]
            if multilayer:
                meta_layer_field = {"layers": [int(L) for L in layers]}
            else:
                meta_layer_field = {"layer": int(layers[0])}

            (out_dir / "meta.json").write_text(json.dumps({
                "datasource_name": datasource_name,
                "act_cache_key": compute_act_cache_key(spec),
                "task_name": task.task_name,
                "dataset_key": task.dataset_key,
                "n_train": int(X_tr.shape[0]),
                "n_test": int(X_te.shape[0]),
                "train_pos_frac": float(y_tr.mean()),
                "test_pos_frac": float(y_te.mean()),
                "seq_len_source": int(seq_len),
                "S_cache": int(S_CACHE),
                "d_in": int(d_in),
                "subject_model": spec.subject_model,
                **meta_layer_field,
                "hookpoint": spec.hookpoint,
                "padding": "left_aligned_real_tokens_S32",
                "padding_side_at_tokenize": "right",
                "first_real_dtype": "int64",
                "schema_version": "2.0.0",
                "multilayer": multilayer,
            }, indent=2))

            mb = (X_tr.nbytes + X_te.nbytes) / 2**20
            short_tr = int((fr_tr > 0).sum())
            short_te = int((fr_te > 0).sum())
            print(f"    → {mb:.1f} MB; short train={short_tr}/{len(fr_tr)}, "
                  f"short test={short_te}/{len(fr_te)}")

    finally:
        for h in handles:
            h.remove()

    return cache_root


def load_probe_cache(
    datasource_name: str,
    task_name: str,
) -> dict:
    """Load one task's cached arrays (schema 2.0.0 — left-aligned S=32).

    Returns a dict shaped like a SAEBench task (compatible with
    ``temp_bench.eval.probing.s_tail_probe``):

        {
          "task_name":        str,
          "X_train":          (N_train, S_CACHE=32, d_in) fp32 array,
          "X_test":           (N_test,  S_CACHE=32, d_in) fp32 array,
          "first_real_train": (N_train,) int64 — first valid pos in S-frame
          "first_real_test":  (N_test,)  int64
          "y_train":          (N_train,) int64,
          "y_test":           (N_test,)  int64,
        }

    Note: arrays are mmap'd then materialised as fp32. For large caches
    consider mmap_mode='r' if probe-cache fits in shared memory.
    """
    out_dir = probe_cache_dir(datasource_name, task_name)
    if not out_dir.exists():
        raise FileNotFoundError(
            f"probe cache missing for task {task_name!r} at {out_dir}. "
            f"Run build_probe_cache(datasource_name=...) first."
        )
    fr_tr_path = out_dir / "first_real_train.npy"
    if not fr_tr_path.exists():
        raise FileNotFoundError(
            f"probe cache for {task_name!r} is schema 1.x (no first_real). "
            f"Rebuild via build_probe_cache(force=True) to migrate to schema 2.0.0."
        )
    X_tr = np.load(out_dir / "X_train.npy", mmap_mode="r")
    X_te = np.load(out_dir / "X_test.npy",  mmap_mode="r")
    fr_tr = np.load(fr_tr_path)
    fr_te = np.load(out_dir / "first_real_test.npy")
    y_tr = np.load(out_dir / "y_train.npy")
    y_te = np.load(out_dir / "y_test.npy")
    return {
        "task_name": task_name,
        "X_train": np.ascontiguousarray(X_tr).astype(np.float32),
        "X_test":  np.ascontiguousarray(X_te).astype(np.float32),
        "first_real_train": fr_tr,
        "first_real_test":  fr_te,
        "y_train": y_tr,
        "y_test":  y_te,
    }


def list_probe_cache(datasource_name: str) -> list[str]:
    """List task names that have a complete schema-2.0.0 cache under
    ``datasource_name``. Older schema-1.x caches (no first_real) are
    skipped — call ``build_probe_cache(force=True)`` to migrate.
    """
    root = probe_cache_dir(datasource_name)
    if not root.exists():
        return []
    required = ("X_train.npy", "X_test.npy",
                "first_real_train.npy", "first_real_test.npy",
                "y_train.npy", "y_test.npy")
    out: list[str] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if all((d / f).exists() for f in required):
            out.append(d.name)
    return out


# ── Internal helpers ──────────────────────────────────────────────────────


def _encode_texts(
    model,
    tokenizer,
    texts: list[str],
    captured: dict,
    *,
    seq_len: int,
    d_in: int,
    batch_size: int,
    device,
    max_n: int | None,
    layers: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Tokenise + forward + per-example reslice to left-aligned S_CACHE-frame.

    Tokenisation uses ``padding_side="right"`` (matches Gemma's training
    distribution). For each example we then:

      1. Read attention_mask to find ``last_idx[i] = sum(mask) - 1``, the
         position of the last real token in the right-padded seq_len-frame.
      2. Take the last ``min(S_CACHE, n_real)`` real activations.
      3. Place them at the END of an S_CACHE-wide frame (positions
         ``[S_CACHE - n_real, S_CACHE-1]``); zeros at the start.
      4. Record ``first_real[i] = S_CACHE - n_real`` so downstream
         pooling can mask.

    **Multi-layer mode** (decisions § 16): when ``layers`` is a list of
    length > 1, ``captured`` is keyed by layer index and the output has
    an extra L axis: ``(N, L, S_CACHE, d_in)``.

    Returns:
        (X_left_aligned, first_real)
          X_left_aligned: ``(N, S_CACHE, d_in)`` if single-layer, or
            ``(N, L, S_CACHE, d_in)`` if multi-layer; both fp16.
          first_real:     ``(N,)`` int64 — first valid pos in the S-frame.
                          0 for sequences with n_real >= S_CACHE.
    """
    # Tokenizer must be right-padded (model training distribution). Force it
    # explicitly so a caller-mutated tokenizer doesn't surprise us.
    tokenizer.padding_side = "right"

    multilayer = layers is not None and len(layers) > 1
    n_layers = len(layers) if multilayer else 1

    if max_n is not None:
        texts = texts[:max_n]
    n = len(texts)
    if n == 0:
        empty_shape = (
            (0, n_layers, S_CACHE, d_in) if multilayer
            else (0, S_CACHE, d_in)
        )
        return (
            np.zeros(empty_shape, dtype=np.float16),
            np.zeros((0,), dtype=np.int64),
        )

    out_shape = (
        (n, n_layers, S_CACHE, d_in) if multilayer
        else (n, S_CACHE, d_in)
    )
    out = np.zeros(out_shape, dtype=np.float16)
    first_real = np.zeros((n,), dtype=np.int64)
    device_t = torch.device(device)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = texts[start:end]
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=seq_len,
        )
        input_ids = enc["input_ids"].to(device_t)
        attention_mask = enc["attention_mask"].to(device_t)

        captured.clear()
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        # last_idx per row: position of the last real token in seq_len-frame.
        last_idx_batch = (
            attention_mask.sum(dim=1).clamp(min=1).cpu().numpy().astype(np.int64) - 1
        )

        if multilayer:
            # (B, L, seq_len, d_in) stack from per-layer hook captures.
            acts_per_layer = [captured[layer].numpy() for layer in layers]
            acts_batch = np.stack(acts_per_layer, axis=1)            # (B, L, T, D)
            for j, li in enumerate(last_idx_batch):
                n_real = min(int(li) + 1, S_CACHE)
                src_lo = int(li) - n_real + 1
                dst_lo = S_CACHE - n_real
                out[start + j, :, dst_lo:S_CACHE] = acts_batch[j, :, src_lo:int(li) + 1]
                first_real[start + j] = dst_lo
        else:
            # Legacy single-layer path. captured may be keyed by either
            # the integer layer (new build_probe_cache) or "resid" (any
            # legacy callers); resolve robustly.
            if layers is not None:
                acts_batch = captured[layers[0]].numpy()
            else:
                acts_batch = captured["resid"].numpy()
            for j, li in enumerate(last_idx_batch):
                n_real = min(int(li) + 1, S_CACHE)
                src_lo = int(li) - n_real + 1
                dst_lo = S_CACHE - n_real
                out[start + j, dst_lo:S_CACHE] = acts_batch[j, src_lo:int(li) + 1]
                first_real[start + j] = dst_lo

    return out, first_real
