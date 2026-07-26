"""Link the paper's canonical C3 caches (synced from HF) into the v2 layout.

The ACTMIX probing runs reuse the PAPER's actual data artifacts rather
than regenerating them (streaming-order drift in fineweb-edu would
silently change the 24k training documents):

- training activation cache: HF ``han1823123123/temp-bench-data``
  ``act_cache/e4916bcae1881963/`` (v1 act_cache_key for datasource
  ``gemma_2_2b_it_l13_fineweb_24k128``; acts (24000, 128, 2304) fp16)
  → ``results/data_cache/<v2 data_key>/acts.npy`` + a v2 ``meta.json``
  carrying the provenance note.
- probe cache (38 SAEBench+CT tasks, schema 2.0.0):
  ``probe_cache/gemma_2_2b_it_l13_fineweb_24k128/`` → symlinked to
  ``results/probe_cache/<datasource>``.

Usage (after the HF snapshot download into MIRROR)::

    .venv/bin/python -m experiments.probing.actmix.prep_cache \
        --mirror /workspace/caches/probing/hf_mirror

Idempotent; verifies shapes/dtypes before linking and refuses to
overwrite non-matching existing caches.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from temp_bench.core.config import compute_data_key, data_cache_dir, load_datasource

DATASOURCE = "gemma_2_2b_it_l13_fineweb_24k128"
V1_ACT_CACHE_KEY = "e4916bcae1881963"
HF_REPO = "han1823123123/temp-bench-data"
EXPECT_SHAPE = (24000, 128, 2304)
N_TASKS_EXPECTED = 38


def prep(mirror: Path) -> None:
    spec = load_datasource(DATASOURCE)
    data_key = compute_data_key(spec)

    # ── 1) training activation cache ──
    src_dir = mirror / "act_cache" / V1_ACT_CACHE_KEY
    src_acts = src_dir / "acts.npy"
    if not src_acts.exists():
        raise FileNotFoundError(f"mirror is missing {src_acts}")
    acts = np.load(src_acts, mmap_mode="r")
    if tuple(acts.shape) != EXPECT_SHAPE:
        raise ValueError(f"acts shape {acts.shape} != expected {EXPECT_SHAPE}")
    v1_meta = json.loads((src_dir / "meta.json").read_text())
    if v1_meta.get("datasource_name") != DATASOURCE:
        raise ValueError(f"v1 meta datasource {v1_meta.get('datasource_name')!r} mismatch")

    dst = data_cache_dir(data_key)
    dst.mkdir(parents=True, exist_ok=True)
    dst_acts = dst / "acts.npy"
    if dst_acts.exists() or dst_acts.is_symlink():
        if dst_acts.resolve() != src_acts.resolve():
            raise FileExistsError(
                f"{dst_acts} exists and does not point at the HF mirror — "
                "refusing to overwrite; resolve manually."
            )
    else:
        dst_acts.symlink_to(src_acts.resolve())
    meta = {
        "data_key": data_key,
        "subject_model": spec.subject_model,
        "layer": spec.layer,
        "hookpoint": spec.hookpoint,
        "dataset": spec.dataset,
        "n_seqs": int(acts.shape[0]),
        "seq_len": int(acts.shape[1]),
        "d_in": int(acts.shape[2]),
        "provenance": {
            "source": f"hf://{HF_REPO}/act_cache/{V1_ACT_CACHE_KEY}",
            "v1_act_cache_key": V1_ACT_CACHE_KEY,
            "note": (
                "Paper's actual C3/C4/C5 anchor cache (v1 pipeline), reused "
                "verbatim for ACTMIX so training data matches the paper "
                "bit-for-bit. v1 spec's dataset field reads 'fineweb' where "
                "the v2 registry says 'HuggingFaceFW/fineweb-edu' — naming "
                "divergence FLAGGED in the ACTMIX card; irrelevant here "
                "because no re-tokenization happens."
            ),
        },
    }
    (dst / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[prep] act cache: {dst_acts} -> {src_acts}  shape={tuple(acts.shape)}")

    # ── 2) probe cache ──
    src_pc = mirror / "probe_cache" / DATASOURCE
    if not src_pc.exists():
        raise FileNotFoundError(f"mirror is missing {src_pc}")
    from temp_bench.data.probe_cache import list_probe_cache, probe_cache_dir
    root = probe_cache_dir(DATASOURCE)
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or root.exists():
        if root.resolve() != src_pc.resolve():
            raise FileExistsError(
                f"{root} exists and does not point at the HF mirror — "
                "refusing to overwrite; resolve manually."
            )
    else:
        root.symlink_to(src_pc.resolve(), target_is_directory=True)
    tasks = list_probe_cache(DATASOURCE)
    print(f"[prep] probe cache: {root} -> {src_pc}  complete_tasks={len(tasks)}")
    if len(tasks) != N_TASKS_EXPECTED:
        print(f"[prep] WARNING: expected {N_TASKS_EXPECTED} tasks, found {len(tasks)} "
              "(download still in flight?)")

    # spot-check one task's schema
    if tasks:
        from temp_bench.data.probe_cache import load_probe_cache
        t = load_probe_cache(DATASOURCE, tasks[0])
        Xtr = t["X_train"]
        assert Xtr.ndim == 3 and Xtr.shape[1] == 32 and Xtr.shape[2] == 2304, Xtr.shape
        print(f"[prep] spot-check {tasks[0]}: X_train {Xtr.shape} {Xtr.dtype}, "
              f"n_test={len(t['y_test'])}, fr_max={int(t['first_real_train'].max())}")


def cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", type=Path,
                    default=Path("/workspace/caches/probing/hf_mirror"))
    args = ap.parse_args()
    prep(args.mirror)


if __name__ == "__main__":
    cli()
