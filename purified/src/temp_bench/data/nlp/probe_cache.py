"""Build per-task probing-activation caches for SAEBench+CT.

For each :class:`ProbingTask`, run the subject model forward over the
task's train/test texts and save the residual-stream activations at the
datasource's hooked layer to disk:

    results/probe_cache/<datasource_name>/<task_name>/
      ├ X_train.npy       # (N_train, seq_len, d_in) fp16
      ├ X_test.npy        # (N_test, seq_len, d_in) fp16
      ├ y_train.npy       # (N_train,) int64
      ├ y_test.npy        # (N_test,) int64
      └ meta.json         # task metadata + datasource spec snapshot

These caches feed :func:`temp_bench.eval.probing.s_tail_probe`, which
slices the last-S tokens, encodes via the trained SAE, and runs L1
logistic regression.

Sourced (in spirit) from
``origin/han-phase7-unification @ 94119bc0:experiments/phase5_downstream_utility/probing/build_probe_cache.py``,
shedding the wasteland's MLC-stack + tail-only quota optimisations
(we just store full ``(N, seq_len, d_in)`` per task — keeps the cache
schema consistent with our activation cache and lets ``s_tail_probe``
do the windowing).

Storage estimate (Gemma-2-2b-it L13 d_in=2304, seq_len=128, fp16):
  - one task @ 5000 prompts: 5000 × 128 × 2304 × 2 = 2.95 GB
  - 38 tasks total ~~110 GB; smaller tasks (winogrande ~2.5k, wsc ~550)
    bring this down. Realistically expect ~70-90 GB on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from temp_bench.config import (
    act_cache_dir,
    compute_act_cache_key,
    load_datasource,
)
from temp_bench.data.nlp.cache import _load_subject_model, _resolve_hf_token
from temp_bench.data.nlp.probe_tasks import ProbingTask, load_all_saebench_ct_tasks


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

    # ── Eager-skip: identify tasks that already have all 4 files
    todo: list[ProbingTask] = []
    for task in tasks:
        out_dir = cache_root / task.task_name
        ok = (
            (out_dir / "X_train.npy").exists()
            and (out_dir / "X_test.npy").exists()
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

    # Hook L<spec.layer>.resid_post
    captured: dict[str, torch.Tensor] = {}

    def hook_fn(module, inp, output):
        acts = output[0] if isinstance(output, tuple) else output
        captured["resid"] = acts.detach().to(torch.float16).cpu()

    handle = model.model.layers[spec.layer].register_forward_hook(hook_fn)

    try:
        for task in todo:
            out_dir = cache_root / task.task_name
            out_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"  ENCODE {task.task_name}: "
                f"{len(task.train_texts)}/{len(task.test_texts)} prompts"
            )
            X_tr = _encode_texts(
                model, tokenizer, task.train_texts, captured,
                seq_len=seq_len, d_in=d_in, batch_size=batch_size,
                device=device, max_n=max_per_task,
            )
            X_te = _encode_texts(
                model, tokenizer, task.test_texts, captured,
                seq_len=seq_len, d_in=d_in, batch_size=batch_size,
                device=device, max_n=max_per_task,
            )
            y_tr = np.asarray(task.train_labels, dtype=np.int64)
            y_te = np.asarray(task.test_labels, dtype=np.int64)
            if max_per_task is not None:
                y_tr = y_tr[:max_per_task]
                y_te = y_te[:max_per_task]

            np.save(out_dir / "X_train.npy", X_tr)
            np.save(out_dir / "X_test.npy", X_te)
            np.save(out_dir / "y_train.npy", y_tr)
            np.save(out_dir / "y_test.npy", y_te)

            (out_dir / "meta.json").write_text(json.dumps({
                "datasource_name": datasource_name,
                "act_cache_key": compute_act_cache_key(spec),
                "task_name": task.task_name,
                "dataset_key": task.dataset_key,
                "n_train": int(X_tr.shape[0]),
                "n_test": int(X_te.shape[0]),
                "train_pos_frac": float(y_tr.mean()),
                "test_pos_frac": float(y_te.mean()),
                "seq_len": int(seq_len),
                "d_in": int(d_in),
                "subject_model": spec.subject_model,
                "layer": int(spec.layer),
                "hookpoint": spec.hookpoint,
            }, indent=2))

            mb = (X_tr.nbytes + X_te.nbytes) / 2**20
            print(f"    → {mb:.1f} MB written")

    finally:
        handle.remove()

    return cache_root


def load_probe_cache(
    datasource_name: str,
    task_name: str,
) -> dict:
    """Load one task's cached arrays.

    Returns a dict shaped like a SAEBench task (compatible with
    ``temp_bench.eval.probing.run_task_suite``):

        {
          "task_name": str,
          "X_train": (N_train, seq_len, d_in) fp32 array,
          "X_test":  (N_test,  seq_len, d_in) fp32 array,
          "y_train": (N_train,) int64,
          "y_test":  (N_test,) int64,
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
    X_tr = np.load(out_dir / "X_train.npy", mmap_mode="r")
    X_te = np.load(out_dir / "X_test.npy",  mmap_mode="r")
    y_tr = np.load(out_dir / "y_train.npy")
    y_te = np.load(out_dir / "y_test.npy")
    return {
        "task_name": task_name,
        "X_train": np.ascontiguousarray(X_tr).astype(np.float32),
        "X_test":  np.ascontiguousarray(X_te).astype(np.float32),
        "y_train": y_tr,
        "y_test":  y_te,
    }


def list_probe_cache(datasource_name: str) -> list[str]:
    """List task names that have a complete cache under ``datasource_name``."""
    root = probe_cache_dir(datasource_name)
    if not root.exists():
        return []
    out: list[str] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if all((d / f).exists() for f in ("X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy")):
            out.append(d.name)
    return out


# ── Internal helpers ──────────────────────────────────────────────────────


def _encode_texts(
    model,
    tokenizer,
    texts: list[str],
    captured: dict[str, torch.Tensor],
    *,
    seq_len: int,
    d_in: int,
    batch_size: int,
    device,
    max_n: int | None,
) -> np.ndarray:
    """Tokenise + forward + hook + concatenate → ``(N, seq_len, d_in)`` fp16.

    Right-padding to seq_len matches the wasteland convention (also
    matches our FineWeb activation cache). Short texts will have
    padding in their tail-S window — known limitation; affects
    winogrande/wsc more than other tasks.
    """
    if max_n is not None:
        texts = texts[:max_n]
    n = len(texts)
    if n == 0:
        return np.zeros((0, seq_len, d_in), dtype=np.float16)

    out = np.empty((n, seq_len, d_in), dtype=np.float16)
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
        out[start:end] = captured["resid"].numpy()

    return out
