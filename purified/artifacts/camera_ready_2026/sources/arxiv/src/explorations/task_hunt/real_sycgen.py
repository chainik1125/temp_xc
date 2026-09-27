"""Real-activation `sycgen_age` datasource — the FIRST hunt-KEEP
retrain substrate (sycgen v1, bundle KEEP 3/3; RETRAIN_CARD.md).

`real_lambda.py`'s pattern verbatim on the sycgen substrate: the
screen's llama31_8b activation cache (single layer SCREEN_HS=14,
`sycgen/cache_acts.py`, N×128×4096 fp16) plus the face labels built
ON THE FLY from the COMMITTED llama screen grid
(`sycgen/grids/elicit_sycgen_screen_llama31.npz`) via the frozen
`wave3_lib.sage_face` — log2(1+age), NaN below support-64 — with the
screen's eligibility carved in as NaN (non-assistant and event-masked
positions are never probe targets). Windows are mapped flat→(N, 128)
through the cache's own tokens.npz (doc_idx, n_prefix) and every
window's tokens are asserted equal to the grid slice before labels
attach (the cache's mapping receipt, re-checked here).

The label tensor is exposed under the `lambda_labels` extra key so
`temp_bench.evals.lambda_recovery` runs UNCHANGED (held-out Pearson r
of a linear probe on tile codes; finite targets only). No
`temp_bench/core/` edits; plugin-only (hard rule 3).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from temp_bench.data.synthetic import SyntheticData

CACHE_ROOT = Path("/workspace/sycgen_caches")
REPO_ROOT = Path(__file__).resolve().parents[3]
GRIDS = (REPO_ROOT / "experiments/explorations/task_hunt/sycgen/grids")
TAG = {"llama31_8b": "llama31", "gemma2_2b": "gemma2", "gpt2": "gpt2"}


def _label_grid(key: str, ids, doc_idx, n_prefix: int) -> np.ndarray:
    """(N, seq_len) sage_face labels aligned to the cache windows."""
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from experiments.explorations.task_hunt.labels import wave3_lib as w3

    z = np.load(GRIDS / f"elicit_sycgen_screen_{TAG[key]}.npz")
    flat, off = z["token_ids"], z["doc_off"]
    first, mask = z["event_first"], z["event_mask"]
    is_assist = z["is_assistant"]
    n_docs = len(off) - 1
    age = np.concatenate([w3.sage_face(first[off[d]:off[d + 1]])
                          for d in range(n_docs)]).astype(np.float32)
    age[(mask == 1) | (is_assist == 0)] = np.nan

    seq_len = ids.shape[1]
    content = seq_len - n_prefix
    lab = np.full((ids.shape[0], seq_len), np.nan, dtype=np.float32)
    seen: dict = {}
    for i, d in enumerate(doc_idx.tolist()):
        c = seen.get(d, 0)
        seen[d] = c + 1
        s = off[d] + c * content
        assert np.array_equal(flat[s:s + content], ids[i, n_prefix:]), \
            f"window/grid mismatch at row {i} (doc {d} chunk {c})"
        lab[i, n_prefix:] = age[s:s + content]
    return lab


def sycgen_age_real(
    *,
    model_key: str = "llama31_8b",
    hs: int = 14,
    label: str = "sycgen_age",
    seq_len: int = 128,
    n_seqs: int | None = None,
    rms_sample: int = 64,
    d_in: int | None = None,
    n_ref: int = 8,
    seed: int = 0,
) -> SyntheticData:
    assert label == "sycgen_age"
    cdir = CACHE_ROOT / model_key
    acts_path = cdir / f"hs{hs}.npy"
    if not acts_path.exists():
        raise FileNotFoundError(
            f"activation cache missing: {acts_path} — build via "
            "experiments.explorations.task_hunt.sycgen.cache_acts")
    c = np.load(cdir / "tokens.npz")
    ids, doc_idx, n_prefix = c["ids"], c["doc_idx"], int(c["n_prefix"])
    arr = np.load(acts_path, mmap_mode="r")
    lab = _label_grid(model_key, ids, doc_idx, n_prefix)
    if arr.shape[:2] != lab.shape:
        raise ValueError(f"cache {arr.shape[:2]} vs labels {lab.shape}")
    N = arr.shape[0] if n_seqs is None else min(int(n_seqs), arr.shape[0])
    if arr.shape[1] != seq_len:
        raise ValueError(f"seq_len {seq_len} != cache {arr.shape[1]}")
    if d_in is not None and int(d_in) != arr.shape[-1]:
        raise ValueError(
            f"datasource declares d_in={d_in} but cache is {arr.shape[-1]}")

    x = torch.from_numpy(np.ascontiguousarray(arr[:N])).float()
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=min(rms_sample, N), replace=False)
    rms = float(x[idx].pow(2).mean().sqrt().clamp(min=1e-6))
    x = x / rms

    flat = x[idx].reshape(-1, x.shape[-1])
    dc = flat.mean(0)
    dc = dc / dc.norm().clamp(min=1e-8)
    centred = flat - flat.mean(0, keepdim=True)
    step = max(1, centred.shape[0] // 8192)
    _, _, V = torch.pca_lowrank(centred[::step], q=max(1, n_ref - 1))
    ref = torch.cat([dc.unsqueeze(0), V.T[: n_ref - 1]], dim=0)
    ref = ref / ref.norm(dim=1, keepdim=True).clamp(min=1e-8)

    return SyntheticData(
        x=x,
        emission_features=ref.contiguous().float(),
        hidden_features=None,
        support=None,
        hidden_support=None,
        seq_len=int(x.shape[1]),
        d_in=int(x.shape[-1]),
        extra={
            "lambda_labels": torch.from_numpy(
                lab[:N].astype(np.float32)),
            "trace_ids": doc_idx[:N].copy(),
            "real_activations": True,
            "model_tag": model_key,
            "hs": hs,
            "label": label,
            "rms_scale": rms,
            "no_ground_truth_directions": True,
            "emission_features_are_reference_basis_not_ground_truth": True,
            "n_ref": int(ref.shape[0]),
        },
    )
