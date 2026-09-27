"""Real-activation hedging-trend datasource for the task hunt's Stage-2 panel.

Presents the canonical Ward activation cache (`conversion_depth/cache_depth.py`)
plus the frozen confidence-drift labels
(`task_hunt/labels/build_confidence.py` → `labels/confidence.npz`) as a
:class:`~temp_bench.data.synthetic.SyntheticData`, so the canonical runner +
the existing :mod:`temp_bench.evals.lambda_recovery` block can panel the
hedging-trend LEVEL task with no ``temp_bench/core/`` edits. Reached through
the ``module:fn`` generator path (`configs/data.yaml`). This mirrors
``real_lambda.py`` (the reviewed candidate-1 Stage-2 datasource) —
normalization, reference basis, and extras contract are identical; only the
label grid and reader layer differ.

**Label = the frozen `slope8` grid, untouched.** The trailing-8-sentence
least-squares slope of the per-sentence confidence state (0 hedged / 1 neutral
/ 2 committed), exactly as committed in ``labels/confidence.npz``. It is
carried under ``extra['lambda_labels']`` because that key is the evaluator's
dispatch contract (`synthetic_recovery` → `lambda_recovery_metrics`), not
because the target is an intensity: ``lambda_recovery`` here means "held-out
Pearson r of a per-tile linear probe against slope8".

**NaN convention (differs from the λ̂ datasource, disclosed).** The λ̂ Stage 2
densified its label with the generator's own warm-up convention
(`lam_for_trace_dense`). slope8 has no generator to extend — a shortened
trailing fit would be a *different statistic* — so undefined positions keep
``NaN`` and the probe drops non-finite leading-edge targets
(`lambda_recovery._train_lambda_probe`, a no-op for all-finite grids).
Undefined ≡ NOT (slope8 finite ∧ valid): the trailing window doesn't fit /
contains an unjudged sentence, or the token isn't inside a mapped sentence
span (out-of-span tokens carry a wrapped-index artifact value in the raw grid
and are masked here via ``valid``).

Normalization: fp16 → fp32, one global RMS constant over a fixed 64-sequence
sample (seed 0) — same convention as ``real_lambda.py``. ``emission_features``
carries the same **reference basis, not ground truth** (DC direction + top
principal directions); ``eauc`` on it is a span sanity check, never feature
recovery. The headline is ``lambda_recovery`` (slope8 Pearson r) alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from temp_bench.data.synthetic import SyntheticData

CACHE_ROOT = Path("/workspace/conv_depth_caches")
REPO_ROOT = Path(__file__).resolve().parents[3]
LABELS_NPZ = (REPO_ROOT / "experiments/explorations/task_hunt/labels/"
              "confidence.npz")


def ward_slope_real(
    *,
    model_tag: str = "distill",
    hs: int = 15,
    label: str = "slope8",
    seq_len: int = 128,
    n_seqs: int | None = None,
    rms_sample: int = 64,
    d_in: int | None = None,
    n_ref: int = 8,
    seed: int = 0,
) -> SyntheticData:
    """Real Ward activations + the frozen slope8 grid as a SyntheticData.

    ``model_tag`` selects the reader cache (base | distill), ``hs`` the
    capture point (15 = resid_post L14, the screen's frozen layer), ``label``
    the target grid in ``confidence.npz`` (``slope8`` PRIMARY). ``d_in`` is
    declared in the datasource params because the trainer sizes the
    dictionary from the spec BEFORE materializing; it is checked against the
    cache here rather than trusted.
    """
    acts_path = CACHE_ROOT / model_tag / f"hs{hs}.npy"
    if not acts_path.exists():
        raise FileNotFoundError(
            f"activation cache missing: {acts_path} — rebuild via "
            "experiments.explorations.conversion_depth.cache_depth")
    arr = np.load(acts_path, mmap_mode="r")
    npz = np.load(LABELS_NPZ)
    grid = npz[label].astype(np.float32)
    valid = npz["valid"]
    if arr.shape[:2] != grid.shape:
        raise ValueError(f"cache {arr.shape[:2]} vs labels {grid.shape}")
    # The frozen label's honest domain: finite ∧ in a mapped sentence span.
    lam = np.where(np.isfinite(grid) & valid, grid, np.nan).astype(np.float32)

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

    # Reference basis (NOT ground truth — see the module docstring).
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
            "lambda_labels": torch.from_numpy(lam[:N]),
            # Sequence → Ward-trace map (same window order as the cache —
            # split_forensics.json). Read ONLY by the v2 λ probe's trace
            # split (`lambda_recovery_v2`); v1 never touches this key, so
            # every existing row is byte-identical.
            "trace_ids": npz["trace_idx"][:N].copy(),
            "real_activations": True,
            "model_tag": model_tag,
            "hs": hs,
            "label": label,
            "rms_scale": rms,
            "label_finite_frac": float(np.isfinite(lam[:N]).mean()),
            "no_ground_truth_directions": True,
            "emission_features_are_reference_basis_not_ground_truth": True,
            "n_ref": int(ref.shape[0]),
        },
    )
