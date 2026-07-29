"""Task-side temporal spectral diagnostics.

This module deliberately analyzes the *data/task* rather than the learned
encoder atoms measured by ``explorations.synthetic.freqfrac``.  For a
second-order stationary vector process, translation invariance makes the lag
covariance and spectral density a Fourier pair (Wiener--Khinchin).  We use
rotation-invariant trace and leading-eigenvalue summaries of that density.

Power is not a complete description of temporal dependence.  In particular,
time-reversed rotations have identical marginal power.  The optional
cross-spectral features retain the imaginary off-diagonal terms that carry
quadrature/phase direction, so the gap between ``power`` and ``cross`` probe
scores is itself a useful screen.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

Array = np.ndarray
CenterMode = Literal["none", "global", "sequence"]
WindowMode = Literal["boxcar", "hann"]


def _as_3d(x: Array) -> Array:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        x = x[..., None]
    if x.ndim != 3:
        raise ValueError(f"expected (n_sequences, time, channels), got {x.shape}")
    if x.shape[0] < 2 or x.shape[1] < 2:
        raise ValueError(f"need at least two sequences and two times, got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("input contains non-finite values")
    return x


def _center(x: Array, mode: CenterMode) -> Array:
    if mode == "none":
        return x
    if mode == "global":
        return x - x.mean(axis=(0, 1), keepdims=True)
    if mode == "sequence":
        return x - x.mean(axis=1, keepdims=True)
    raise ValueError(f"unknown center mode {mode!r}")


def _window(T: int, mode: WindowMode) -> Array:
    if mode == "boxcar":
        return np.ones(T, dtype=np.float64)
    if mode == "hann":
        w = np.hanning(T)
        # Preserve mean-square energy across window choices.
        return w / np.sqrt(np.mean(w * w)).clip(min=1e-12)
    raise ValueError(f"unknown window {mode!r}")


def _onesided_weights(T: int) -> Array:
    """Parseval weights for ``rfft`` bins."""
    n_freq = T // 2 + 1
    weights = np.ones(n_freq, dtype=np.float64)
    if T % 2 == 0:
        weights[1:-1] = 2.0
    else:
        weights[1:] = 2.0
    return weights


@dataclass(frozen=True)
class Periodogram:
    frequencies: Array
    per_sequence_power: Array
    mean_power: Array
    normalized_power: Array


def periodogram(
    x: Array,
    *,
    center: CenterMode = "global",
    window: WindowMode = "boxcar",
) -> Periodogram:
    """Return the rotation-invariant, one-sided trace periodogram.

    Channel powers are summed, so an orthogonal rotation of the activation
    space leaves the result unchanged.  ``center="global"`` removes the
    population mean while retaining per-sequence DC variation;
    ``center="sequence"`` is the explicit DC-removal ablation.
    """
    x = _as_3d(x)
    _, T, _ = x.shape
    tapered = _center(x, center) * _window(T, window)[None, :, None]
    fft = np.fft.rfft(tapered, axis=1, norm="ortho")
    power = np.square(np.abs(fft)).sum(axis=2)
    power *= _onesided_weights(T)[None, :]
    mean_power = power.mean(axis=0)
    norm = mean_power / mean_power.sum().clip(min=1e-12)
    return Periodogram(
        frequencies=np.fft.rfftfreq(T),
        per_sequence_power=power,
        mean_power=mean_power,
        normalized_power=norm,
    )


@dataclass(frozen=True)
class OperatorSpectrum:
    frequencies: Array
    trace: Array
    leading_eigenvalue: Array
    leading_fraction: Array


def operator_spectrum(
    x: Array,
    *,
    center: CenterMode = "global",
    window: WindowMode = "hann",
) -> OperatorSpectrum:
    """Estimate the matrix spectral density and its strongest mode.

    Replicate sequences supply the Welch-style average.  The returned matrix
    at frequency ``f`` is ``E[X_f X_f*]``; it is Hermitian positive
    semidefinite, unlike a one-sided lag covariance at a single positive lag.
    """
    x = _as_3d(x)
    n, T, _ = x.shape
    tapered = _center(x, center) * _window(T, window)[None, :, None]
    fft = np.fft.rfft(tapered, axis=1, norm="ortho")
    weights = _onesided_weights(T)
    traces = np.empty(fft.shape[1], dtype=np.float64)
    leading = np.empty_like(traces)
    for f in range(fft.shape[1]):
        z = fft[:, f, :]
        density = (z.conj().T @ z) * (weights[f] / n)
        eig = np.linalg.eigvalsh(density)
        traces[f] = float(np.real(eig.sum()))
        leading[f] = float(max(np.real(eig[-1]), 0.0))
    return OperatorSpectrum(
        frequencies=np.fft.rfftfreq(T),
        trace=traces,
        leading_eigenvalue=leading,
        leading_fraction=leading / traces.clip(min=1e-12),
    )


@dataclass(frozen=True)
class LagCurve:
    lag: Array
    operator_norm: Array
    frobenius_norm: Array
    directionality: Array


def lag_covariance_curve(x: Array, *, max_lag: int = 16) -> LagCurve:
    """Francesco-style two-point covariance strength versus time lag.

    ``directionality`` is ``||C(l)-C(l)^T||_F / ||C(l)||_F``.  It is zero for
    a time-reversible symmetric lag covariance and flags structure that a
    scalar power envelope can erase.
    """
    x = _center(_as_3d(x), "global")
    _, T, _ = x.shape
    max_lag = min(int(max_lag), T - 1)
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative")
    op, fro, direction = [], [], []
    for lag in range(max_lag + 1):
        left = x[:, : T - lag].reshape(-1, x.shape[-1])
        right = x[:, lag:].reshape(-1, x.shape[-1])
        cov = left.T @ right / max(left.shape[0], 1)
        singular = np.linalg.svd(cov, compute_uv=False)
        f = float(np.linalg.norm(cov, ord="fro"))
        op.append(float(singular[0]))
        fro.append(f)
        direction.append(float(np.linalg.norm(cov - cov.T, ord="fro") / max(f, 1e-12)))
    return LagCurve(
        lag=np.arange(max_lag + 1),
        operator_norm=np.asarray(op),
        frobenius_norm=np.asarray(fro),
        directionality=np.asarray(direction),
    )


@dataclass(frozen=True)
class SpectralSummary:
    dc_fraction: float
    ac_low_fraction: float
    ac_centroid: float
    ac_entropy: float
    ac_peak_frequency: float
    ac_peak_prominence: float
    operator_dc_fraction: float
    operator_peak_frequency: float
    max_directionality: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def summarize_spectrum(x: Array, *, low_cutoff: float = 0.125) -> SpectralSummary:
    """Compact task-screen coordinates with DC and AC reported separately."""
    raw = periodogram(x, center="global", window="boxcar")
    ac = periodogram(x, center="sequence", window="hann")
    op = operator_spectrum(x, center="global", window="hann")
    lag = lag_covariance_curve(x, max_lag=min(16, _as_3d(x).shape[1] - 1))

    ac_power = ac.mean_power.copy()
    ac_power[0] = 0.0
    ac_total = ac_power.sum().clip(min=1e-12)
    ac_norm = ac_power / ac_total
    positive = ac.frequencies > 0
    entropy_terms = ac_norm[positive]
    denom = np.log(max(int(positive.sum()), 2))
    entropy = -float(np.sum(entropy_terms * np.log(entropy_terms.clip(min=1e-15)))) / denom
    peak_idx = int(np.argmax(ac_power))
    op_idx = int(np.argmax(op.leading_eigenvalue))
    return SpectralSummary(
        dc_fraction=float(raw.normalized_power[0]),
        ac_low_fraction=float(
            ac_norm[(ac.frequencies > 0) & (ac.frequencies <= low_cutoff)].sum()
        ),
        ac_centroid=float(np.sum(ac.frequencies * ac_norm)),
        ac_entropy=entropy,
        ac_peak_frequency=float(ac.frequencies[peak_idx]),
        ac_peak_prominence=float(
            ac_power[peak_idx] / ac_power[positive].mean().clip(min=1e-12)
        ),
        operator_dc_fraction=float(
            op.leading_eigenvalue[0] / op.leading_eigenvalue.sum().clip(min=1e-12)
        ),
        operator_peak_frequency=float(op.frequencies[op_idx]),
        max_directionality=float(lag.directionality[1:].max(initial=0.0)),
    )


def _pca_projection(
    x: Array,
    n_components: int,
    *,
    fit_x: Array | None = None,
) -> tuple[Array, Array]:
    x = _as_3d(x)
    fit = x if fit_x is None else _as_3d(fit_x)
    if fit.shape[-1] != x.shape[-1]:
        raise ValueError(f"fit channels {fit.shape[-1]} != input channels {x.shape[-1]}")
    mean = fit.mean(axis=(0, 1), keepdims=True)
    flat = (fit - mean).reshape(-1, fit.shape[-1])
    cov = flat.T @ flat / max(flat.shape[0], 1)
    eigval, eigvec = np.linalg.eigh(cov)
    rank = min(int(n_components), x.shape[-1])
    basis = eigvec[:, np.argsort(eigval)[-rank:]]
    return (x - mean) @ basis, basis


def spectral_features(
    x: Array,
    *,
    kind: Literal["power", "cross"] = "power",
    n_components: int = 8,
    remove_dc: bool = True,
    window: WindowMode = "hann",
    fit_x: Array | None = None,
) -> Array:
    """Per-window power or cross-spectral features for a lightweight probe.

    ``power`` keeps only diagonal spectral energy and is invariant to temporal
    phase. ``cross`` additionally keeps real and imaginary off-diagonal
    channel cross-spectra; the imaginary terms distinguish quadrature and
    time-reversal signals with identical power.
    """
    projected, _ = _pca_projection(x, n_components, fit_x=fit_x)
    _, T, rank = projected.shape
    if remove_dc:
        projected = projected - projected.mean(axis=1, keepdims=True)
    projected *= _window(T, window)[None, :, None]
    z = np.fft.rfft(projected, axis=1, norm="ortho")
    if remove_dc:
        z = z[:, 1:, :]
    cross = np.einsum("nfr,nfs->nfrs", z, z.conj())
    scale = np.real(np.trace(cross, axis1=2, axis2=3)).sum(axis=1)
    cross = cross / scale[:, None, None, None].clip(min=1e-12)
    log_scale = np.log(scale.clip(min=1e-12))[:, None]

    diag = np.real(np.diagonal(cross, axis1=2, axis2=3))
    if kind == "power":
        return np.concatenate([log_scale, diag.reshape(diag.shape[0], -1)], axis=1)
    if kind != "cross":
        raise ValueError(f"unknown feature kind {kind!r}")
    iu = np.triu_indices(rank, k=1)
    off = cross[:, :, iu[0], iu[1]]
    return np.concatenate(
        [log_scale, diag.reshape(diag.shape[0], -1), off.real.reshape(off.shape[0], -1),
         off.imag.reshape(off.shape[0], -1)],
        axis=1,
    )


def dc_features(
    x: Array,
    *,
    n_components: int = 8,
    fit_x: Array | None = None,
) -> Array:
    """Signed sequence-mean coefficients for the explicit DC branch.

    A power spectrum squares Fourier coefficients and therefore discards the
    sign/direction of a stable state.  The DC branch of a spectral crosscoder
    retains the coefficient vector itself, so a task screen must test that
    vector separately before deciding that DC can be removed.
    """
    projected, _ = _pca_projection(x, n_components, fit_x=fit_x)
    return projected.mean(axis=1)


@dataclass(frozen=True)
class ProbeScore:
    score_mean: float
    score_std: float
    chance: float
    shuffled_mean: float
    n_examples: int
    n_features: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def classification_probe(
    features: Array,
    labels: Array,
    *,
    n_splits: int = 5,
    seed: int = 0,
) -> ProbeScore:
    """Leak-free stratified linear-probe score with a shuffled-label null."""
    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels).reshape(-1)
    valid = np.isfinite(X).all(axis=1)
    if np.issubdtype(y.dtype, np.number):
        valid &= np.isfinite(y)
    X, y = X[valid], y[valid]
    classes, counts = np.unique(y, return_counts=True)
    if classes.size < 2:
        raise ValueError("classification needs at least two classes")
    splits = min(int(n_splits), int(counts.min()))
    if splits < 2:
        raise ValueError("each class needs at least two examples")
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed + 991)
    shuffled = rng.permutation(y)
    scores, nulls = [], []
    for train, test in cv.split(X, y):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        )
        model.fit(X[train], y[train])
        scores.append(balanced_accuracy_score(y[test], model.predict(X[test])))
        null_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        )
        null_model.fit(X[train], shuffled[train])
        nulls.append(balanced_accuracy_score(shuffled[test], null_model.predict(X[test])))
    return ProbeScore(
        score_mean=float(np.mean(scores)),
        score_std=float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        chance=float(1.0 / classes.size),
        shuffled_mean=float(np.mean(nulls)),
        n_examples=int(X.shape[0]),
        n_features=int(X.shape[1]),
    )


def regression_probe(
    features: Array,
    labels: Array,
    *,
    n_splits: int = 5,
    seed: int = 0,
) -> ProbeScore:
    """Leak-free ridge score for continuous task latents."""
    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[valid], y[valid]
    if X.shape[0] < n_splits:
        raise ValueError("not enough examples for regression cross-validation")
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed + 991)
    shuffled = rng.permutation(y)
    scores, nulls = [], []
    for train, test in cv.split(X):
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(X[train], y[train])
        scores.append(r2_score(y[test], model.predict(X[test])))
        null_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        null_model.fit(X[train], shuffled[train])
        nulls.append(r2_score(shuffled[test], null_model.predict(X[test])))
    return ProbeScore(
        score_mean=float(np.mean(scores)),
        score_std=float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        chance=0.0,
        shuffled_mean=float(np.mean(nulls)),
        n_examples=int(X.shape[0]),
        n_features=int(X.shape[1]),
    )


def tile_sequences(
    x: Array,
    labels: Array,
    *,
    tile_size: int,
    label_position: Literal["first", "last", "center"] = "last",
) -> tuple[Array, Array]:
    """Split sequences into non-overlapping tiles and align token labels."""
    x = _as_3d(x)
    labels = np.asarray(labels)
    if labels.shape[:2] != x.shape[:2]:
        raise ValueError(f"labels {labels.shape} do not align with x {x.shape}")
    n, T, d = x.shape
    n_tiles = T // tile_size
    if n_tiles < 1:
        raise ValueError(f"tile_size {tile_size} exceeds sequence length {T}")
    tiled = x[:, : n_tiles * tile_size].reshape(n * n_tiles, tile_size, d)
    pos = {"first": 0, "last": tile_size - 1, "center": tile_size // 2}[label_position]
    target = labels[:, : n_tiles * tile_size].reshape(n, n_tiles, tile_size)[:, :, pos]
    return tiled, target.reshape(-1)
