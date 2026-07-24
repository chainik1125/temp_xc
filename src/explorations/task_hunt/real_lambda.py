"""Real-activation λ̂ datasource for the task hunt's Stage-2 panel.

Presents the canonical Ward activation cache (a real subject model's
resid_post stream, `conversion_depth/cache_depth.py`) plus the frozen
backtracking-intensity labels
(`task_hunt/lambda_intensity/build_labels.py`) as a
:class:`~temp_bench.data.synthetic.SyntheticData`, so the canonical
runner + the existing :mod:`temp_bench.evals.lambda_recovery` block can
panel a REAL task with no ``temp_bench/core/`` edits. Reached through
the ``module:fn`` generator path (`configs/data.yaml`).

**What is and is not ground truth here.** The λ̂ labels are exact — a
deterministic, frozen function of the trace's Sonnet event history — so
``lambda_recovery`` (held-out Pearson r of a linear probe on the tile
code) is a sound recovery metric. There are, however, **no ground-truth
feature directions** in a real residual stream: ``emission_features`` is
deliberately EMPTY, which makes ``_feature_recovery_auc`` return NaN for
``eauc``/``e_mean_max_cos``. That is the honest reading — those columns
are undefined for this datasource and must not be interpreted; the
headline is ``lambda_recovery`` alone. ``support`` is likewise None.

Positions with no label (sentence index < K, unmapped round-trips,
outside the think region) carry ``NaN`` in the label grid; the probe in
``lambda_recovery`` is fit on finite targets only.

Normalization: activations are cast fp16 → fp32 and scaled by a single
global constant (the RMS over a fixed 64-window sample, seed 0) so the
dictionary sees unit-scale inputs — the same convention the framework's
real_lm caches use. No per-position or per-feature whitening (that would
destroy the substrate under test).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from temp_bench.data.synthetic import SyntheticData

CACHE_ROOT = Path("/workspace/conv_depth_caches")
LABEL_DIR = Path("/workspace/task_hunt_labels/lambda_intensity")


def ward_lambda_real(
    *,
    model_tag: str = "base",
    hs: int = 13,
    label: str = "lam_hist",
    seq_len: int = 128,
    n_seqs: int | None = None,
    rms_sample: int = 64,
    seed: int = 0,
) -> SyntheticData:
    """Real Ward activations + frozen λ̂ labels as a SyntheticData.

    Parameters mirror the datasource entry: ``model_tag`` selects the
    reader cache (base | distill), ``hs`` the hidden-state capture point
    (13 = resid_post L12), ``label`` the frozen target grid
    (``lam_hist`` PRIMARY, kernel-only; ``lam_hat`` includes the
    position ramp — see the candidate-1 card).
    """
    acts_path = CACHE_ROOT / model_tag / f"hs{hs}.npy"
    if not acts_path.exists():
        raise FileNotFoundError(
            f"activation cache missing: {acts_path} — rebuild via "
            "experiments.explorations.conversion_depth.cache_depth")
    arr = np.load(acts_path, mmap_mode="r")
    lam = np.load(LABEL_DIR / f"{label}.npy")
    if arr.shape[:2] != lam.shape:
        raise ValueError(f"cache {arr.shape[:2]} vs labels {lam.shape}")
    N = arr.shape[0] if n_seqs is None else min(int(n_seqs), arr.shape[0])
    if arr.shape[1] != seq_len:
        raise ValueError(f"seq_len {seq_len} != cache {arr.shape[1]}")

    x = torch.from_numpy(np.ascontiguousarray(arr[:N])).float()
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=min(rms_sample, N), replace=False)
    rms = float(x[idx].pow(2).mean().sqrt().clamp(min=1e-6))
    x = x / rms

    return SyntheticData(
        x=x,
        emission_features=torch.zeros((0, x.shape[-1]), dtype=torch.float32),
        hidden_features=None,
        support=None,
        hidden_support=None,
        seq_len=int(x.shape[1]),
        d_in=int(x.shape[-1]),
        extra={
            "lambda_labels": torch.from_numpy(lam[:N].astype(np.float32)),
            "real_activations": True,
            "model_tag": model_tag,
            "hs": hs,
            "label": label,
            "rms_scale": rms,
            "no_ground_truth_directions": True,
        },
    )
