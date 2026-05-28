"""FreqBench evaluator — NTPS + order-controls (DC / AC / Mixed).

Trains a linear probe on the SAE code of a held-out labelled sequence set
and reports, per cell:

    A          probe accuracy on the mean-pooled code (order-destroying pool)
    A_shuffle  same, but tokens shuffled in time before encoding (control)
    A_reverse  same, but the sequence reversed before encoding (control)
    A_stacked  probe sees the per-position codes concatenated
    NTPS       (A − A_loc) / (A_oracle − A_loc)

The order-controls are the load-bearing diagnostic (see
docs/components/freq_bench.md): an architecture that genuinely encodes
*order / signed direction* shows A ≫ chance, A_shuffle → chance, and (on
the signed AC bench) A_reverse < chance. A pure aggregator shows
A ≈ A_shuffle ≈ A_reverse.

Wiring: registered as experiment ``freq_bench`` in
``temp_bench.core.runner._EVALUATOR_REGISTRY``. The datasource's generator
must be one of ``temp_bench.data.freq_bench_data:freq_bench_{dc,ac,mixed}``
and its ``params`` must include ``bench`` ∈ {dc, ac, mixed_unsigned,
mixed_signed}.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from temp_bench.core.config import load_datasource
from temp_bench.data.freq_bench_data import materialise_probe_set
from temp_bench.interfaces.architecture import TempBenchArch
from temp_bench.interfaces.evaluator import EvalSpec, Evaluator


# ── code extraction ──────────────────────────────────────────────────────


@torch.no_grad()
def _codes_per_position(model: TempBenchArch, x: torch.Tensor) -> torch.Tensor:
    """x: (N, W, d_in) → per-position codes (N, n_pos, d_sae).

    - token archs: encode every token (n_pos = W).
    - window archs: slide the arch's T-window across the sequence and take
      one window-level code per start position (n_pos = W − T + 1).
    """
    device = next(model.parameters()).device
    x = x.to(device, dtype=torch.float32)
    N, W, d_in = x.shape
    if model.consumes == "token":
        z = model.encode(x)                                  # (N, W, d_sae)
        if z.dim() == 2:
            z = z.unsqueeze(1)
        return z
    # window arch: slide T-window
    T = int(model.T)
    if W < T:
        raise ValueError(f"freq_bench: W={W} < arch T={T}; widen seq_len.")
    # Stride S between successive encoder applications. Defaults to 1
    # (vanilla sliding) for archs that don't expose a stride hparam; the
    # (T, S, B) family (TXCBand) sets it via hparams.
    S = int(getattr(model.config, "S", None) or 1)
    if hasattr(model, "_S"):
        S = int(model._S)
    codes = []
    for s in range(0, W - T + 1, S):
        zc = model.encode(x[:, s:s + T])
        # Normalise to (N, n_pos_per_window, d_sae). Window-level archs
        # return (N, d_sae) or (N, 1, d_sae) — n_pos_per_window=1. Per-
        # position archs (e.g. txc_base_perpos) return (N, T, d_sae) —
        # n_pos_per_window=T. Concatenating along the position axis works
        # uniformly for both.
        if zc.dim() == 2:
            zc = zc.unsqueeze(1)
        codes.append(zc)
    return torch.cat(codes, dim=1)                           # (N, n_pos, d_sae)


def _pool_mean(z: torch.Tensor) -> np.ndarray:
    return z.mean(dim=1).detach().cpu().numpy()              # (N, d_sae)


def _stack(z: torch.Tensor) -> np.ndarray:
    return z.reshape(z.shape[0], -1).detach().cpu().numpy()  # (N, n_pos*d_sae)


# ── probe ────────────────────────────────────────────────────────────────


def _split(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    perm = np.random.default_rng(seed).permutation(n)
    cut = n // 2
    return perm[:cut], perm[cut:]


def _fit_probe(feats: np.ndarray, y: np.ndarray, tr: np.ndarray):
    """Fit StandardScaler + LogisticRegression on the train split."""
    scaler = StandardScaler().fit(feats[tr])
    if len(np.unique(y[tr])) < 2:
        return scaler, None, int(y[tr][0])
    clf = LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0)
    clf.fit(scaler.transform(feats[tr]), y[tr])
    return scaler, clf, None


def _score(scaler, clf, const, feats: np.ndarray, y: np.ndarray,
           te: np.ndarray) -> float:
    """Score a (possibly transformed) feature set with an EXISTING probe.

    Used for the order-controls: the probe is trained on ORDERED features
    and applied to shuffled / reversed features of the same test rows, so a
    direction-encoding model is actively *fooled* (reverse → below chance)
    rather than just re-fit."""
    if clf is None:
        return float((y[te] == const).mean())
    return float(clf.score(scaler.transform(feats[te]), y[te]))


def _probe_acc(feats: np.ndarray, y: np.ndarray, seed: int) -> float:
    """Self-contained train/test probe accuracy (retrains; for A_stacked)."""
    tr, te = _split(len(y), seed)
    scaler, clf, const = _fit_probe(feats, y, tr)
    return _score(scaler, clf, const, feats, y, te)


def _mlp_acc(feats: np.ndarray, y: np.ndarray, seed: int) -> float:
    """Nonlinear (2-layer MLP) probe accuracy. Separates 'info absent' from
    'info present but linearly unreadable'."""
    tr, te = _split(len(y), seed)
    if len(np.unique(y[tr])) < 2:
        return float((y[te] == y[tr][0]).mean())
    scaler = StandardScaler().fit(feats[tr])
    clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=400,
                        random_state=seed, early_stopping=False)
    clf.fit(scaler.transform(feats[tr]), y[tr])
    return float(clf.score(scaler.transform(feats[te]), y[te]))


def _freqfrac(model) -> float | None:
    """AC fraction of the encoder weights along the temporal axis.

    For a window arch with W_enc of shape (T, d_in, d_sae), each atom s has
    a per-position weight profile W_enc[:, :, s] ∈ R^{T×d_in}. Its energy at
    nonzero temporal frequency (AC) vs total (DC+AC) measures how much the
    atom depends on token ORDER. FreqFrac ≈ 0 ⇒ atoms constant across the
    window (a pure smoother); FreqFrac > 0 ⇒ order-sensitive atoms.

    Returns the usage-agnostic mean over atoms, or None if the arch has no
    temporal axis (T=1) or an unrecognised W_enc shape.
    """
    W = getattr(model, "W_enc", None)
    if W is None or W.dim() != 3:
        return None
    T = W.shape[0]
    if T < 2:
        return None
    w = W.detach().float().cpu()                     # (T, d_in, d_sae)
    spec = torch.fft.rfft(w, dim=0).abs() ** 2       # (F, d_in, d_sae)
    total = spec.sum(dim=0)                           # (d_in, d_sae)
    ac = spec[1:].sum(dim=0)                          # nonzero-freq energy
    frac = (ac / total.clamp(min=1e-12))             # (d_in, d_sae)
    return float(frac.mean())


def _shuffle_time(x: torch.Tensor, seed: int) -> torch.Tensor:
    """Independently permute the time axis of each sequence."""
    N, W, _ = x.shape
    g = torch.Generator().manual_seed(seed)
    out = x.clone()
    for i in range(N):
        out[i] = x[i, torch.randperm(W, generator=g)]
    return out


# ── evaluator ──────────────────────────────────────────────────────────


class FreqBenchEval(Evaluator):
    """NTPS + order-control probe for the DC/AC/Mixed benches."""

    name = "freq_bench"
    # 1.1.0: order-controls use the shared forward-trained probe (reverse can
    #        go below chance) instead of re-fitting per control.
    # 1.2.0: + nonlinear (MLP) probe NTPS_mlp + weight-space FreqFrac.
    protocol_version = "1.2.0"

    def eval(self, model: TempBenchArch, spec: EvalSpec) -> dict[str, float]:
        ds = load_datasource(spec.datasource)
        params = dict(ds.params or {})
        bench = params.get("bench")
        if bench is None:
            raise ValueError(
                f"freq_bench datasource {spec.datasource!r} params missing "
                "'bench' (dc / ac / mixed_unsigned / mixed_signed)."
            )
        seed = int(spec.extra.get("training_seed", 0))
        n_probe = 256 if spec.smoke else 2500
        params = {**params, "n_seqs": n_probe}
        data = materialise_probe_set(bench=bench, params=params, seed=seed)

        x = data.x
        y = data.y.numpy()
        model.eval()

        # Forward codes + a probe trained ONCE on ordered features.
        z = _codes_per_position(model, x)
        F_fwd = _pool_mean(z)
        tr, te = _split(len(y), seed)
        scaler, clf, const = _fit_probe(F_fwd, y, tr)
        A = _score(scaler, clf, const, F_fwd, y, te)

        # Order-controls: SAME forward probe, applied to shuffled / reversed
        # codes of the same test rows. reverse < chance ⇒ signed direction.
        F_sh = _pool_mean(_codes_per_position(model, _shuffle_time(x, seed + 1)))
        A_shuffle = _score(scaler, clf, const, F_sh, y, te)
        F_rev = _pool_mean(_codes_per_position(model, torch.flip(x, dims=[1])))
        A_reverse = _score(scaler, clf, const, F_rev, y, te)

        # Stacked-code capacity diagnostic (its own probe).
        A_stacked = _probe_acc(_stack(z), y, seed)

        # Nonlinear probe on the same mean-pooled forward code.
        A_mlp = _mlp_acc(F_fwd, y, seed)

        A_loc, A_oracle = float(data.A_loc), float(data.A_oracle)
        denom = A_oracle - A_loc
        def _ntps(a):
            return (a - A_loc) / denom if abs(denom) > 1e-9 else float("nan")

        out = {
            "A": A, "A_shuffle": A_shuffle, "A_reverse": A_reverse,
            "A_stacked": A_stacked, "A_mlp": A_mlp,
            "A_loc": A_loc, "A_oracle": A_oracle,
            "NTPS": float(_ntps(A)),
            "NTPS_mlp": float(_ntps(A_mlp)),              # nonlinear-probe NTPS
            "NTPS_stacked": float(_ntps(A_stacked)),
            "order_gap": float(A - A_shuffle),            # >0 ⇒ uses order
            "reverse_drop": float(A - A_reverse),         # >0 (≫0 signed) ⇒ direction
            "n_classes": float(data.n_classes),
        }
        ff = _freqfrac(model)
        if ff is not None:
            out["freqfrac"] = ff                          # weight-space AC content

        # Mixed bench: per-velocity-class accuracy → frequency response R_j.
        if data.velocities is not None:
            out.update(self._per_velocity(z, y, data, seed))
        return out

    def _per_velocity(self, z, y, data, seed) -> dict[str, float]:
        """Frequency response R_j = (A_j − 1/n) / (1 − 1/n) per velocity."""
        feats = _pool_mean(z)
        n = len(y)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        cut = n // 2
        tr, te = perm[:cut], perm[cut:]
        scaler = StandardScaler().fit(feats[tr])
        clf = LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0)
        if len(np.unique(y[tr])) < 2:
            return {}
        clf.fit(scaler.transform(feats[tr]), y[tr])
        pred = clf.predict(scaler.transform(feats[te]))
        vel = data.velocities.numpy()
        denom = 1.0 - 1.0 / data.n_classes
        out = {}
        for v in np.unique(vel[te]):
            mask = vel[te] == v
            if mask.sum() == 0:
                continue
            acc = float((pred[mask] == y[te][mask]).mean())
            out[f"R_v{int(v):+d}"] = (acc - 1.0 / data.n_classes) / denom
        return out

    def primary_metric(self) -> str:
        return "NTPS"
