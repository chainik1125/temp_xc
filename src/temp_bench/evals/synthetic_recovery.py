"""Synthetic feature recovery — § 4 evaluator.

For each (arch, synthetic data) cell, measures three numbers:

- ``eauc``  — best-matching dictionary atom cosine vs each EMISSION
              feature (local). Averaged over emissions then thresholded
              to AUC.
- ``gauc``  — same metric vs HIDDEN-chain features (global). Only
              meaningful for the coupled bench.
- ``nmse``  — reconstruction NMSE on a held-out batch from the
              synthetic generator.

These map to the paper's "local vs global feature recovery" narrative
(§ 4). The synthetic generator is the source of truth: its
``emission_features`` and ``hidden_features`` define the targets.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from temp_bench.core.config import load_datasource
from temp_bench.data.synthetic import materialise
from temp_bench.interfaces.architecture import TempBenchArch
from temp_bench.interfaces.evaluator import EvalSpec, Evaluator


def _decoder_directions_normed(model: TempBenchArch) -> torch.Tensor:
    """``(d_sae, d_in)`` unit-norm decoder columns."""
    D = model.decoder_directions().detach().cpu().float()      # (d_sae, d_in)
    norms = D.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return D / norms


def _feature_recovery_auc(
    decoder: torch.Tensor,
    targets: torch.Tensor,
    n_thresholds: int = 50,
) -> dict[str, float]:
    """Best-matching cosine AUC + summary stats.

    For each target row, find ``max_j |cos(decoder[j], target)|``.
    AUC = area under ROC of (max-cos > threshold) vs threshold ∈ [0, 1].
    """
    D = decoder / decoder.norm(dim=1, keepdim=True).clamp(min=1e-8)
    F = targets / targets.norm(dim=1, keepdim=True).clamp(min=1e-8)
    cos = (D @ F.T).abs()                              # (d_sae, n_targets)
    max_cos_per_target = cos.max(dim=0).values         # (n_targets,)
    thresholds = torch.linspace(0, 1, n_thresholds + 1)
    fracs = torch.stack([
        (max_cos_per_target >= t).float().mean() for t in thresholds
    ])
    # Trapezoidal AUC.
    auc = torch.trapz(fracs, thresholds).item()
    return {
        "auc": float(auc),
        "mean_max_cos": float(max_cos_per_target.mean()),
        "frac_recovered_90": float((max_cos_per_target >= 0.9).float().mean()),
        "frac_recovered_80": float((max_cos_per_target >= 0.8).float().mean()),
    }


# Default common evaluation-window length when a datasource doesn't specify
# one via eval_cfg["eval_window_L"]. Benches whose window archs are not
# powers of two (e.g. the legacy T=5 coupling/denoising archs) keep this
# small value (5 % 5 == 5 % 1 == 0). Power-of-two benches pass L=32 (see
# experiments/explorations/synthetic/README.md § 4).
DEFAULT_EVAL_WINDOW = 5


def _arch_T(model: TempBenchArch) -> int:
    return int(getattr(model.config, "T", 1) or 1)


def _check_tileable(model: TempBenchArch, L: int) -> int:
    """Return the arch window T; require the eval window L to tile by it."""
    T = _arch_T(model)
    if L % T != 0:
        raise ValueError(
            f"eval window L={L} is not divisible by arch window T={T} "
            f"({type(model).__name__}); use a power-of-two L and T "
            "(experiments/explorations/synthetic/README.md § 4)."
        )
    return T


def _sample_windows(
    x_eval: torch.Tensor, *, L: int, n_windows: int, seed: int
) -> tuple[torch.Tensor, np.ndarray]:
    """Sample ``n_windows`` length-L windows at random offsets.

    Returns ``(windows (n_windows, L, d_in), seq_idx (n_windows,))`` — the
    same fixed set is reused for every architecture so comparisons are over
    identical positions. ``seq_idx`` (into ``x_eval``) lets callers attach
    per-sequence labels / split leak-free.
    """
    n_total, seq_len, _d_in = x_eval.shape
    if seq_len < L:
        raise ValueError(f"seq_len ({seq_len}) < eval window L ({L}).")
    rng = np.random.default_rng(seed)
    seq_idx = rng.integers(0, n_total, size=n_windows)
    offsets = rng.integers(0, seq_len - L + 1, size=n_windows)
    seq_t = torch.from_numpy(seq_idx).long().unsqueeze(1)
    pos = torch.from_numpy(offsets).long().unsqueeze(1) + torch.arange(L)
    windows = x_eval[seq_t, pos]                            # (n_windows, L, d_in)
    return windows, seq_idx


def _windowed_recon_nmse(
    model: TempBenchArch,
    x_eval: torch.Tensor,
    *,
    L: int,
    n_windows: int = 1024,
    batch: int = 256,
    seed: int = 0,
) -> float:
    """Apples-to-apples reconstruction NMSE over a tiled eval window.

    Draws ONE fixed set of length-``L`` windows and tiles each
    non-overlapping into ``L/T`` sub-windows of the arch's native window
    length ``T``. The model reconstructs each tile in its native mode — a
    window crosscoder compresses a whole tile into one shared code, a
    per-token / per-position SAE reconstructs each position independently —
    but error is aggregated over the identical ``n_windows × L`` positions,
    so the number is comparable across architectures of any (power-of-two)
    window size. See experiments/explorations/synthetic/README.md § 4.
    """
    device = next(model.parameters()).device
    T = _check_tileable(model, L)
    n_tiles = L // T
    windows, _ = _sample_windows(x_eval, L=L, n_windows=n_windows, seed=seed)

    num = 0.0
    den = 0.0
    model.eval()
    with torch.no_grad():
        for i in range(0, n_windows, batch):
            xb = windows[i:i + batch].to(device, dtype=torch.float32)  # (b, L, d_in)
            b, _, d_in = xb.shape
            tiles = xb.reshape(b * n_tiles, T, d_in)
            try:
                rec = model(tiles)
            except Exception:
                rec = model.decode(model.encode(tiles))
            rec = rec.reshape(b, L, d_in)
            num += float((xb - rec).pow(2).sum().item())
            den += float(xb.pow(2).sum().item())
    return float(num / max(den, 1e-12))


@torch.no_grad()
def _realized_sparsity(
    model: TempBenchArch,
    x_eval: torch.Tensor,
    *,
    L: int,
    n_windows: int = 1024,
    batch: int = 256,
    seed: int = 0,
) -> dict[str, float]:
    """Measured code sparsity on the shared eval tiling — the fair-matching key.

    Encodes the SAME length-``L`` windows every recovery metric uses (tiled
    non-overlapping into ``L/T`` sub-windows) and reports the realized L0 in two
    architecture-independent units:

    - ``l0_per_window`` — mean nonzero code entries describing one ``T``-tile
      (the window arch's realized ``k_win``; for a per-token SAE, ``T=1`` so a
      tile is one token).
    - ``l0_per_token``  — the same amortized per token position (``= l0_per_window
      / T``). A per-token SAE and a length-``T`` crosscoder are directly
      comparable on this axis.

    These are the two matching conventions in one measurement: hold
    ``l0_per_token`` fixed → per-position matched; hold ``l0_per_window`` fixed →
    per-window matched. **Realized, not nominal**: BatchTopK pools L0 across the
    batch and post-squash reuses each atom at all ``T`` positions, so the fired
    density drifts off ``k_pos``; matching must key on this. Deterministic in
    ``seed``; additive (leaves every other metric byte-identical).
    """
    device = next(model.parameters()).device
    T = _check_tileable(model, L)
    n_tiles = L // T
    windows, _ = _sample_windows(x_eval, L=L, n_windows=n_windows, seed=seed)
    model.eval()
    total_nnz = 0.0
    total_tiles = 0
    for i in range(0, n_windows, batch):
        xb = windows[i:i + batch].to(device, dtype=torch.float32)   # (b, L, d_in)
        b, _, d_in = xb.shape
        tiles = xb.reshape(b * n_tiles, T, d_in)
        z = model.encode(tiles).reshape(b * n_tiles, -1)            # (tiles, code)
        total_nnz += float((z != 0).float().sum().item())
        total_tiles += z.shape[0]
    l0_per_window = total_nnz / max(total_tiles, 1)
    return {"l0_per_window": float(l0_per_window),
            "l0_per_token": float(l0_per_window / T)}


class SyntheticRecovery(Evaluator):
    """§ 4 evaluator: feature recovery AUC + NMSE on synthetic data."""

    name = "synthetic_recovery"
    # v1.1.0 (2026-05-28): eval now re-materialises with the TRAINING
    # seed, not seed=0. Fixes a bug where the synthetic generator's
    # feature directions (deterministic in the seed) differed between
    # training and eval, making it impossible for dictionary atoms to
    # match ground-truth features.
    # v1.2.0 (2026-06-02): NMSE (and the signed-motion sign probe) are now the
    # apples-to-apples *tiled* metric — every arch is scored on ONE shared set
    # of length-L eval windows, tiled non-overlapping into L/T sub-windows, so
    # archs of any power-of-two window size are compared over identical
    # positions (experiments/explorations/synthetic/README.md § 4). L comes from
    # eval_cfg["eval_window_L"] (default 5; power-of-two benches pass 32). Only
    # `nmse`/`s_temp` are affected; eauc/gauc unchanged. Bumping invalidates
    # prior eval rows so they re-evaluate (checkpoints reused — train_key
    # unchanged).
    # v1.3.0 (2026-07-10): realized code sparsity (l0_per_token / l0_per_window)
    # is now a FIRST-CLASS part of the contract — it was added additively under
    # 1.2.0, so 1.2.0 rows are heterogeneous (pre-increment rows lack it). The
    # program-level B×A report matches archs on the *realized* l0_per_token, so
    # it needs a version whose rows ALL carry it. 1.3.0 marks that: recovery
    # metrics are byte-identical to 1.2.0 (same probes, same windows), the ONLY
    # change is that every 1.3.0 row is guaranteed to expose realized L0. The
    # uniform clean-room re-grid (briefings/full-rerun-and-purge.md) rebuilds the
    # whole synthetic set at 1.3.0 so the report + per-bench renderers read one
    # homogeneous set. Bumping invalidates prior eval rows so they re-evaluate
    # (checkpoints reused — train_key unchanged).
    protocol_version = "1.3.0"

    def eval(self, model: TempBenchArch, spec: EvalSpec) -> dict[str, float]:
        # Re-materialise the dataset from the registry using the SAME
        # seed the model was trained on, so the synthetic generator's
        # feature directions (which depend on the seed) match what the
        # trained dictionary atoms learned.
        ds = load_datasource(spec.datasource)
        seed = int(spec.extra.get("training_seed", spec.extra.get("eval_seed", 0)))
        data = materialise(ds, seed=seed)
        decoder = _decoder_directions_normed(model)

        # Common eval-window length (tiled per arch in § 4). Power-of-two
        # benches pass eval_window_L=32; legacy benches fall back to 5.
        L = int(spec.extra.get("eval_window_L", DEFAULT_EVAL_WINDOW))

        out: dict[str, float] = {}

        emission_stats = _feature_recovery_auc(decoder, data.emission_features)
        out["eauc"] = emission_stats["auc"]
        out["e_mean_max_cos"] = emission_stats["mean_max_cos"]
        out["e_frac_recovered_90"] = emission_stats["frac_recovered_90"]
        out["e_frac_recovered_80"] = emission_stats["frac_recovered_80"]

        if data.hidden_features is not None:
            hidden_stats = _feature_recovery_auc(decoder, data.hidden_features)
            out["gauc"] = hidden_stats["auc"]
            out["g_mean_max_cos"] = hidden_stats["mean_max_cos"]
            out["g_frac_recovered_90"] = hidden_stats["frac_recovered_90"]
            out["g_frac_recovered_80"] = hidden_stats["frac_recovered_80"]

        n_windows = 128 if spec.smoke else 1024
        out["nmse"] = _windowed_recon_nmse(model, data.x, L=L, n_windows=n_windows)

        # Realized code sparsity on the SAME eval tiling — the architecture-
        # independent fair-matching key for the program-level B×A report (both
        # per-position and per-window conventions render off l0_per_token /
        # l0_per_window). Additive: every metric above is byte-identical, so the
        # protocol contract is unchanged and prior rows stay valid (they simply
        # lack the key; the renderer falls back / re-grid supplies it).
        out.update(_realized_sparsity(model, data.x, L=L, n_windows=n_windows))

        # AC-only signed-motion add-on (FrequencyBench § 5). Only fires for
        # the signed_motion datasource, which exposes a hidden ±1 sign in
        # `extra`. For every other bench `extra` is None → this block is a
        # no-op and the metrics above are unchanged.
        if getattr(data, "extra", None) and "sign_labels" in data.extra:
            from temp_bench.evals.signed_motion_recovery import signed_motion_metrics
            out.update(signed_motion_metrics(model, data, eval_window_L=L))

        # Self-exciting intensity (λ) recovery add-on (autoresearch #1
        # backtracking). Only fires for the toy_backtracking_selfexcite
        # datasource, which exposes a continuous λ per position in `extra`.
        # No-op (byte-identical metrics) for every other bench → protocol
        # stays 1.2.0.
        if getattr(data, "extra", None) and "lambda_labels" in data.extra:
            from temp_bench.evals.lambda_recovery import lambda_recovery_metrics
            out.update(lambda_recovery_metrics(model, data, eval_window_L=L))

        # Change-point / semi-Markov modes add-on (autoresearch #2). Only
        # fires for the toy_changepoint_modes datasource, which exposes the
        # mode / time-since-switch / change-point labels in `extra`. No-op
        # (byte-identical metrics) for every other bench → protocol stays
        # 1.2.0.
        if getattr(data, "extra", None) and "mode_labels" in data.extra:
            from temp_bench.evals.changepoint_recovery import changepoint_metrics
            out.update(changepoint_metrics(model, data, eval_window_L=L))

        # Cyclic-tone velocity recovery + S(f) add-on (autoresearch #3). Only
        # fires for the toy_cyclic_* datasources, which expose velocity_labels in
        # `extra`. No-op (byte-identical metrics) for every other bench →
        # protocol stays 1.2.0.
        if getattr(data, "extra", None) and "velocity_labels" in data.extra:
            from temp_bench.evals.frequency_recovery import frequency_metrics
            n_windows = 128 if spec.smoke else 1024
            out.update(frequency_metrics(model, data, eval_window_L=L,
                                         n_windows=n_windows))

        # Assumption→consequence state/next-state add-on (expansion stage-6
        # #1). Only fires for the toy_assumption_consequence datasource, which
        # exposes the discourse-state labels in `extra`. No-op (byte-identical
        # metrics) for every other bench → protocol stays 1.3.0.
        if getattr(data, "extra", None) and "state_labels" in data.extra:
            from temp_bench.evals.assumption_recovery import assumption_metrics
            out.update(assumption_metrics(model, data, eval_window_L=L))

        # Hedging-drift confidence-state add-on (expansion stage-6 #2). Only
        # fires for the toy_hedging_drift datasource, which exposes the
        # continuous confidence stream in `extra`. No-op (byte-identical
        # metrics) for every other bench → protocol stays 1.3.0.
        if getattr(data, "extra", None) and "conf_labels" in data.extra:
            from temp_bench.evals.hedging_recovery import hedging_metrics
            out.update(hedging_metrics(model, data, eval_window_L=L))

        # Recipe-instruction phase/equality add-on (expansion stage-6 #3). Only
        # fires for the toy_recipe_instruction datasource, which exposes the
        # equality-adjacency labels in `extra`. No-op (byte-identical metrics)
        # for every other bench → protocol stays 1.3.0.
        if getattr(data, "extra", None) and "equality_labels" in data.extra:
            from temp_bench.evals.recipe_recovery import recipe_metrics
            out.update(recipe_metrics(model, data, eval_window_L=L))

        # Multilane superposition per-lane velocity add-on (FreqBench FB-2).
        # Only fires for the toy_multilane_* datasources, which expose the
        # per-lane velocity labels in `extra`. No-op (byte-identical metrics)
        # for every other bench → protocol stays 1.3.0.
        if getattr(data, "extra", None) and "lane_velocity_labels" in data.extra:
            from temp_bench.evals.multilane_recovery import multilane_metrics
            n_windows = 128 if spec.smoke else 1024
            out.update(multilane_metrics(model, data, eval_window_L=L,
                                         n_windows=n_windows))

        # Colored-sources feature-direction recovery add-on (FreqBench FB-3).
        # Only fires for the toy_colored_sources_* datasources, which expose
        # the ρ schedule in `extra`. Weight-space metric (no probe). No-op
        # (byte-identical metrics) for every other bench → protocol stays
        # 1.3.0.
        if getattr(data, "extra", None) and "rho_schedule" in data.extra:
            from temp_bench.evals.colored_recovery import colored_metrics
            out.update(colored_metrics(model, data))

        return out

    def primary_metric(self) -> str:
        return "gauc"   # § 4 headline = global feature recovery
