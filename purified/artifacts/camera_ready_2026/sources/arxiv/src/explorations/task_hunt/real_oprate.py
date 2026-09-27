"""Real-activation operation-rate datasource for the oprate Stage-2 panel.

The `real_lambda` pattern (same cache, same normalization, same extras
contract) over the **committed** oprate label bundle: presents the
canonical Ward activation cache (`conversion_depth/cache_depth.py`)
plus the frozen operation-class trailing-rate grids
(`task_hunt/labels/build_oprate.py` → `labels/oprate.npz`) as a
:class:`~temp_bench.data.synthetic.SyntheticData`, so the canonical
runner + :mod:`temp_bench.evals.lambda_recovery` panel a REAL task with
no ``temp_bench/core/`` edits. Reached through the ``module:fn``
generator path (`configs/data.yaml`).

Differences from `real_lambda.py`, all label-side:

- Labels come from the **committed npz** (git, not a volume path): the
  bundle's kernel-smoothed trailing rates ``rate_case`` / ``rate_ver``
  (tau 3.0, k 8, min_history 4 — `oprate_stats.json` § frozen). The
  grids ship final: NaN wherever any kernel-lag sentence is unlabeled
  AND wherever the current sentence is itself the event class (so the
  leading-edge target can never be read off the current sentence's own
  class), on top of the round-trip `valid` mask. Coverage ≈ 0.90 of
  valid positions ⇒ the non-finite leading-edge guard in
  ``lambda_recovery`` is LIVE on this datasource; the run record
  reports how many sampled windows drop, per T.
- ``trace_ids`` come from the same npz (`trace_idx`, window → Ward
  trace), so the v2 λ-probe trace split and the split-forensics receipt
  apply unchanged. v1 never touches this key.

**What is and is not ground truth here.** The rate labels are exact — a
deterministic, frozen function of the trace's Sonnet sentence-class
history — so ``lambda_recovery`` (held-out Pearson r of a linear probe
on the tile code) is a sound recovery metric and **is the only
headline**. There are **no ground-truth feature directions** in a real
residual stream; ``emission_features`` carries a **reference basis, not
ground truth** (DC direction + top principal directions of a fixed
subsample, seed 0), so every eval column stays finite. ``eauc`` on this
datasource answers "does the dictionary span the stream's dominant
variance directions?" — a sanity check, NOT feature recovery.

Normalization: fp16 → fp32, one global RMS constant over a fixed
64-window sample (seed 0). No per-position or per-feature whitening.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from temp_bench.data.synthetic import SyntheticData

CACHE_ROOT = Path("/workspace/conv_depth_caches")
REPO_ROOT = Path(__file__).resolve().parents[3]
LABEL_NPZ = (REPO_ROOT / "experiments/explorations/task_hunt/labels/"
             "oprate.npz")


def ward_oprate_real(
    *,
    model_tag: str = "base",
    hs: int = 13,
    target: str = "case",
    seq_len: int = 128,
    n_seqs: int | None = None,
    rms_sample: int = 64,
    d_in: int | None = None,
    n_ref: int = 8,
    seed: int = 0,
) -> SyntheticData:
    """Real Ward activations + frozen oprate labels as a SyntheticData.

    ``model_tag`` selects the reader cache (base | distill), ``hs`` the
    hidden-state capture point (13 = resid_post L12, the screen's
    primary layer), ``target`` the frozen rate grid (``case`` |
    ``ver``). ``d_in`` is declared in the datasource params because the
    trainer infers the input width from the spec BEFORE materializing;
    it is checked against the cache here rather than trusted.
    """
    if target not in ("case", "ver"):
        raise ValueError(f"target must be 'case' or 'ver', got {target!r}")
    acts_path = CACHE_ROOT / model_tag / f"hs{hs}.npy"
    if not acts_path.exists():
        raise FileNotFoundError(
            f"activation cache missing: {acts_path} — rebuild via "
            "experiments.explorations.conversion_depth.cache_depth")
    arr = np.load(acts_path, mmap_mode="r")
    z = np.load(LABEL_NPZ)
    lam = z[f"rate_{target}"]
    if arr.shape[:2] != lam.shape:
        raise ValueError(f"cache {arr.shape[:2]} vs labels {lam.shape}")
    N = arr.shape[0] if n_seqs is None else min(int(n_seqs), arr.shape[0])
    if arr.shape[1] != seq_len:
        raise ValueError(f"seq_len {seq_len} != cache {arr.shape[1]}")
    if d_in is not None and int(d_in) != arr.shape[-1]:
        raise ValueError(
            f"datasource declares d_in={d_in} but cache is {arr.shape[-1]} — "
            "the trainer sizes the dictionary from the declared value")

    x = torch.from_numpy(np.ascontiguousarray(arr[:N])).float()
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=min(rms_sample, N), replace=False)
    rms = float(x[idx].pow(2).mean().sqrt().clamp(min=1e-6))
    x = x / rms

    # Reference basis (NOT ground truth — see the module docstring):
    # DC direction + top principal directions of a fixed subsample.
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
            "lambda_labels": torch.from_numpy(lam[:N].astype(np.float32)),
            # Window → Ward-trace map from the SAME committed npz as the
            # labels (identical window order to the cache — the bundle
            # was built on the canonical stream). Read ONLY by the v2
            # λ-probe's trace split (`lambda_recovery_v2`); v1 never
            # touches this key.
            "trace_ids": z["trace_idx"][:N].copy(),
            "real_activations": True,
            "model_tag": model_tag,
            "hs": hs,
            "label": f"rate_{target}",
            "rms_scale": rms,
            "no_ground_truth_directions": True,
            "emission_features_are_reference_basis_not_ground_truth": True,
            "n_ref": int(ref.shape[0]),
        },
    )
