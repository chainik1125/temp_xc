"""ACTMIX RLHF — install the phase-7 BASE stream as a v2 train cache.

`han1823123123/txcdr-base-data activation_cache/resid_L12.npy`
(24000 × 128 × 2304 fp16 — the EXACT activations the shipped RLHF
checkpoints trained on) is already in the runner's acts.npy layout;
this installs it (hardlink, fallback copy) at the keyed
`results/data_cache/<data_key>/` location for the new datasource
`gemma_2_2b_base_l12_phase7`, with the standard meta.json.

Run: .venv/bin/python -m experiments.explorations.actmix_rlhf.convert_train_cache
Idempotent.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np

from temp_bench.core.config import (
    compute_data_key,
    data_cache_dir,
    load_datasource,
)

# Venue-portable: the pods that held the original hardlink source are gone.
# The durable origin is the Hub dataset han1823123123/txcdr-base-data; any
# agent can re-fetch it and point ACTMIX_RLHF_CACHE_SRC at the download.
SRC = Path(os.environ.get(
    "ACTMIX_RLHF_CACHE_SRC",
    "/workspace/caches/rlhf/txcdr-base-data/activation_cache",
)).expanduser()
DS_NAME = "gemma_2_2b_base_l12_phase7"


def main():
    spec = load_datasource(DS_NAME)
    dk = compute_data_key(spec)
    cdir = data_cache_dir(dk)
    mpath = cdir / "meta.json"
    if mpath.exists() and json.loads(mpath.read_text())["data_key"] == dk:
        print(f"[convert] hit {DS_NAME} at {cdir}")
        return
    a = np.load(SRC / "resid_L12.npy", mmap_mode="r")
    assert a.shape == (24000, 128, 2304) and a.dtype == np.float16, \
        (a.shape, a.dtype)
    cdir.mkdir(parents=True, exist_ok=True)
    dst = cdir / "acts.npy"
    if dst.exists():
        dst.unlink()
    try:
        os.link(SRC / "resid_L12.npy", dst)
        how = "hardlink"
    except OSError:
        shutil.copyfile(SRC / "resid_L12.npy", dst)
        how = "copy"
    shutil.copyfile(SRC / "token_ids.npy", cdir / "token_ids.npy")
    mpath.write_text(json.dumps({
        "data_key": dk,
        "subject_model": spec.subject_model,
        "layer": spec.layer,
        "hookpoint": spec.hookpoint,
        "dataset": spec.dataset,
        "n_seqs": 24000,
        "seq_len": 128,
        "d_in": 2304,
        # provenance extras (ignored by build_refill):
        "source": "han1823123123/txcdr-base-data activation_cache/"
                  "resid_L12.npy (phase-7 BASE stream; the shipped RLHF "
                  "ckpts' training substrate)",
        "install": how,
        "installer": "experiments/explorations/actmix_rlhf/"
                     "convert_train_cache.py",
    }, indent=2))
    print(f"[convert] installed {DS_NAME} -> {cdir} ({how})")


if __name__ == "__main__":
    main()
