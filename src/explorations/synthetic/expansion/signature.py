"""Temporal-signature toolkit, generalized from ``backtracking/measure.py``.

Operates on a list of 1-D numpy arrays (one per trace/document), where each
array is the per-span label stream a validated labeler produced. Three signal
kinds, each with its own headline battery:

- ``binary``      — event streams (the backtracking case): base/position rate,
  indicator ACF, self-excitation, inter-event CV, Fano, Markov order, MI(lag).
- ``categorical`` — k-symbol streams: self-match ACF (Cramér-style), MI(lag),
  dwell/run-length stats, Markov order.
- ``scalar``      — continuous streams: pooled within-seq ACF, MI(lag) via
  quantile binning, position-mean profile.

Null battery (the temporal-ness gate, identical roles to backtracking's):

- **N1** ``null_permute`` — within-sequence permutation: preserves each
  sequence's marginal exactly, destroys all order.
- **N2** ``null_trend``  — position-conditional iid resample (pooled
  position-bin marginal): preserves any within-sequence position trend,
  destroys clustering/order beyond the trend. (For binary this is exactly the
  inhomogeneous-Bernoulli N2 of the backtracking prereg.)
- **N3** ``null_iid``    — iid resample from the global marginal: preserves
  nothing but the base rate.

``measure(seqs, kind)`` orchestrates: real headline + bootstrap CIs (resample
sequences), the three null bands, and — for binary — the label-noise
robustness check (symmetric flips at the measured noise floor). Everything is
JSON-serializable; deterministic under ``seed``.
"""

from __future__ import annotations

from math import log

import numpy as np

DEFAULT_MAXLAG = 12
DEFAULT_FANO_W = 10
DEFAULT_POS_BINS = 20


# ── shared helpers ─────────────────────────────────────────────────────────

def base_rate(seqs) -> float:
    tot = sum(s.size for s in seqs)
    pos = sum(int(s.sum()) for s in seqs)
    return pos / tot


def _pos_bin_index(L: int, nbins: int) -> np.ndarray:
    return np.minimum((np.arange(L) / L * nbins).astype(int), nbins - 1)


def position_profile(seqs, nbins: int = DEFAULT_POS_BINS) -> np.ndarray:
    """Per-position-bin mean of the signal (rate for binary, mean for scalar)."""
    num = np.zeros(nbins)
    den = np.zeros(nbins)
    for s in seqs:
        idx = _pos_bin_index(s.size, nbins)
        for j in range(nbins):
            m = idx == j
            den[j] += m.sum()
            num[j] += float(s[m].sum())
    return num / np.maximum(den, 1)


def acf(seqs, maxlag: int = DEFAULT_MAXLAG) -> np.ndarray:
    """Pooled within-sequence autocorrelation per lag (binary or scalar)."""
    out = []
    for k in range(1, maxlag + 1):
        xs, ys = [], []
        for s in seqs:
            if s.size > k:
                xs.append(s[:-k])
                ys.append(s[k:])
        if not xs:
            out.append(0.0)
            continue
        x = np.concatenate(xs).astype(float)
        y = np.concatenate(ys).astype(float)
        sx, sy = x.std(), y.std()
        out.append(float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))
                   if sx > 0 and sy > 0 else 0.0)
    return np.array(out)


def selfmatch_acf(seqs, maxlag: int = DEFAULT_MAXLAG) -> np.ndarray:
    """Categorical autocorrelation: (P(x_t == x_{t+k}) − p_iid) / (1 − p_iid).

    ``p_iid = Σ_a p(a)²`` is the match probability under an iid stream with the
    pooled marginal; 0 = no order structure, 1 = frozen.
    """
    pooled = np.concatenate([s for s in seqs])
    _, counts = np.unique(pooled, return_counts=True)
    p = counts / counts.sum()
    p_iid = float((p ** 2).sum())
    out = []
    for k in range(1, maxlag + 1):
        num = den = 0
        for s in seqs:
            if s.size > k:
                num += int((s[:-k] == s[k:]).sum())
                den += s.size - k
        pm = num / max(den, 1)
        out.append((pm - p_iid) / max(1 - p_iid, 1e-12))
    return np.array(out)


def self_excitation(seqs) -> dict:
    """Binary: P(next=1|cur=1), P(next=1|cur=0), ratio P(1|1)/base."""
    n11 = n1 = n01 = n0 = 0
    for s in seqs:
        cur, nxt = s[:-1], s[1:]
        n1 += int((cur == 1).sum())
        n11 += int(((cur == 1) & (nxt == 1)).sum())
        n0 += int((cur == 0).sum())
        n01 += int(((cur == 0) & (nxt == 1)).sum())
    p = base_rate(seqs)
    p11 = n11 / max(n1, 1)
    p01 = n01 / max(n0, 1)
    return {"p11": p11, "p01": p01, "base": p, "excite_ratio": p11 / max(p, 1e-9)}


def inter_event_cv(seqs) -> dict:
    gaps = []
    for s in seqs:
        idx = np.flatnonzero(s)
        if idx.size >= 2:
            gaps.extend(np.diff(idx).tolist())
    g = np.array(gaps, dtype=float)
    if g.size < 2:
        return {"mean": float("nan"), "cv": float("nan"), "n": int(g.size)}
    return {"mean": float(g.mean()), "cv": float(g.std() / g.mean()), "n": int(g.size)}


def fano(seqs, w: int = DEFAULT_FANO_W) -> float:
    counts = []
    for s in seqs:
        for start in range(0, s.size - w + 1, w):
            counts.append(int(s[start:start + w].sum()))
    c = np.array(counts, dtype=float)
    return float(c.var() / c.mean()) if c.size and c.mean() > 0 else float("nan")


def dwell_stats(seqs) -> dict:
    """Run-length (dwell) distribution over all symbols pooled."""
    runs = []
    for s in seqs:
        if s.size == 0:
            continue
        change = np.flatnonzero(np.diff(s) != 0)
        edges = np.concatenate([[-1], change, [s.size - 1]])
        runs.extend(np.diff(edges).tolist())
    r = np.array(runs, dtype=float)
    if r.size < 2:
        return {"mean": float("nan"), "cv": float("nan"), "n": int(r.size)}
    return {"mean": float(r.mean()), "cv": float(r.std() / r.mean()),
            "n": int(r.size), "p90": float(np.percentile(r, 90))}


def markov_order_test(seqs, n_symbols: int = 2) -> dict:
    """LL of order-0/1/2 context models + LR chi² p-values (k-symbol)."""
    from collections import defaultdict

    from scipy.stats import chi2

    def counts(order):
        c = defaultdict(lambda: np.zeros(n_symbols))
        for s in seqs:
            for i in range(order, s.size):
                ctx = tuple(int(v) for v in s[i - order:i])
                c[ctx][int(s[i])] += 1
        return c

    def ll(order):
        tot = 0.0
        for _, vec in counts(order).items():
            n = vec.sum()
            if n == 0:
                continue
            q = vec / n
            nz = vec > 0
            tot += float((vec[nz] * np.log(q[nz])).sum())
        return tot

    ll0, ll1, ll2 = ll(0), ll(1), ll(2)
    df1 = (n_symbols - 1) * (n_symbols - 1)
    df2 = (n_symbols ** 2 - n_symbols) * (n_symbols - 1)
    lr10 = 2 * (ll1 - ll0)
    lr21 = 2 * (ll2 - ll1)
    return {"ll0": ll0, "ll1": ll1, "ll2": ll2,
            "lr10": lr10, "p_order1_vs_0": float(chi2.sf(lr10, df=max(df1, 1))),
            "lr21": lr21, "p_order2_vs_1": float(chi2.sf(lr21, df=max(df2, 1)))}


def mi_vs_lag(seqs, maxlag: int = DEFAULT_MAXLAG, n_symbols: int = 2) -> np.ndarray:
    """Mutual information I(x_t; x_{t+k}) in nats, k-symbol streams."""
    out = []
    for k in range(1, maxlag + 1):
        j = np.zeros((n_symbols, n_symbols))
        for s in seqs:
            if s.size > k:
                np.add.at(j, (s[:-k].astype(int), s[k:].astype(int)), 1)
        if j.sum() == 0:
            out.append(0.0)
            continue
        pj = j / j.sum()
        pa = pj.sum(1, keepdims=True)
        pb = pj.sum(0, keepdims=True)
        mi = 0.0
        for a in range(n_symbols):
            for b in range(n_symbols):
                if pj[a, b] > 0:
                    mi += pj[a, b] * log(pj[a, b] / (pa[a, 0] * pb[0, b]))
        out.append(float(mi))
    return np.array(out)


def directed_transition(seqs, src: int, dst: int) -> dict:
    """Directional order statistic: P(x_{t+1}=dst | x_t=src), forward vs
    time-reversed, plus the asymmetry index (fwd−rev)/(fwd+rev).

    For any *reversible* stream (and for every order-destroying null) fwd ≈ rev,
    so asym ≈ 0; a genuine directed convention (e.g. assumption→consequence,
    question→answer) shows asym > 0 beyond the null band.
    """

    def rate(ss):
        n_src = n_hit = 0
        for s in ss:
            cur, nxt = s[:-1], s[1:]
            m = cur == src
            n_src += int(m.sum())
            n_hit += int((m & (nxt == dst)).sum())
        return n_hit / max(n_src, 1)

    fwd = rate(seqs)
    rev = rate([s[::-1] for s in seqs])
    return {"fwd_rate": fwd, "rev_rate": rev,
            "asym": (fwd - rev) / max(fwd + rev, 1e-12)}


def perturb_categorical(seqs, eps: float, rng):
    """Categorical noise model: w.p. ε replace with a draw from the pooled
    marginal (the k-symbol analog of the binary symmetric flip)."""
    pooled = np.concatenate([s for s in seqs])
    out = []
    for s in seqs:
        repl = rng.choice(pooled, size=s.size, replace=True)
        out.append(np.where(rng.random(s.size) < eps, repl, s).astype(s.dtype))
    return out


def quantile_bin(seqs, nbins: int = 8):
    """Bin scalar streams into pooled-quantile symbols (for MI / Markov tests)."""
    pooled = np.concatenate([s for s in seqs]).astype(float)
    edges = np.quantile(pooled, np.linspace(0, 1, nbins + 1)[1:-1])
    return [np.searchsorted(edges, s.astype(float)).astype(np.int8) for s in seqs]


# ── null generators ────────────────────────────────────────────────────────

def null_permute(seqs, rng):
    """N1 — within-sequence permutation (kills order, keeps each marginal)."""
    return [rng.permutation(s) for s in seqs]


def null_iid(seqs, rng, kind: str = "binary"):
    """N3 — iid from the global marginal."""
    if kind == "binary":
        p = base_rate(seqs)
        return [(rng.random(s.size) < p).astype(np.int8) for s in seqs]
    pooled = np.concatenate([s for s in seqs])
    return [rng.choice(pooled, size=s.size, replace=True) for s in seqs]


def null_trend(seqs, rng, kind: str = "binary", nbins: int = DEFAULT_POS_BINS):
    """N2 — position-conditional iid (keeps the position trend, kills order)."""
    if kind == "binary":
        prof = position_profile(seqs, nbins)
        out = []
        for s in seqs:
            idx = _pos_bin_index(s.size, nbins)
            out.append((rng.random(s.size) < prof[idx]).astype(np.int8))
        return out
    # categorical / scalar: pool values per position bin, resample iid within bin
    pools: list[list] = [[] for _ in range(nbins)]
    for s in seqs:
        idx = _pos_bin_index(s.size, nbins)
        for j in range(nbins):
            pools[j].extend(np.asarray(s)[idx == j].tolist())
    pools = [np.array(p) if p else np.concatenate([np.asarray(s) for s in seqs])
             for p in pools]
    out = []
    for s in seqs:
        idx = _pos_bin_index(s.size, nbins)
        v = np.empty(s.size, dtype=np.asarray(s).dtype)
        for j in range(nbins):
            m = idx == j
            if m.any():
                v[m] = rng.choice(pools[j], size=int(m.sum()), replace=True)
        out.append(v)
    return out


def flip_labels(seqs, eps: float, rng):
    """Independent symmetric flips (binary label-noise robustness check)."""
    return [np.where(rng.random(s.size) < eps, 1 - s, s).astype(np.int8) for s in seqs]


# ── headline batteries ─────────────────────────────────────────────────────

def headline(seqs, kind: str, *, maxlag: int = DEFAULT_MAXLAG,
             fano_w: int = DEFAULT_FANO_W, mi_bins: int = 8,
             pair: tuple[int, int] | None = None) -> dict:
    """The per-kind scalar+curve statistics compared real-vs-null.

    ``pair=(src, dst)`` (categorical only) adds the directed-transition
    statistics {fwd_rate, rev_rate, asym} to the battery, so the null bands
    and bootstrap CIs cover them automatically.
    """
    if kind == "binary":
        se = self_excitation(seqs)
        return {"acf": acf(seqs, maxlag), "fano": fano(seqs, fano_w),
                "p11": se["p11"], "excite_ratio": se["excite_ratio"],
                "gap_cv": inter_event_cv(seqs)["cv"]}
    if kind == "categorical":
        pooled = np.concatenate([s for s in seqs])
        n_sym = int(pooled.max()) + 1
        dw = dwell_stats(seqs)
        out = {"acf": selfmatch_acf(seqs, maxlag),
               "mi": mi_vs_lag(seqs, maxlag, n_sym),
               "dwell_mean": dw["mean"], "dwell_cv": dw["cv"]}
        if pair is not None:
            out.update(directed_transition(seqs, *pair))
        return out
    if kind == "scalar":
        binned = quantile_bin(seqs, mi_bins)
        return {"acf": acf(seqs, maxlag),
                "mi": mi_vs_lag(binned, maxlag, mi_bins)}
    raise ValueError(f"unknown kind {kind!r}")


_CURVE_KEYS = {"acf", "mi"}


def null_band(seqs, gen, rng, kind: str, n: int = 200, **hkw) -> dict:
    """Null distribution (mean + 2.5/97.5 pct) of every headline statistic."""
    accs: dict[str, list] = {}
    for _ in range(n):
        h = headline(gen(seqs, rng), kind, **hkw)
        for k, v in h.items():
            accs.setdefault(k, []).append(np.asarray(v, dtype=float))
    res = {}
    for k, v in accs.items():
        a = np.stack(v)
        res[k] = {"mean": np.nanmean(a, axis=0).tolist(),
                  "lo": np.nanpercentile(a, 2.5, axis=0).tolist(),
                  "hi": np.nanpercentile(a, 97.5, axis=0).tolist()}
    return res


def bootstrap_ci(seqs, kind: str, rng, n_boot: int = 500, **hkw) -> dict:
    """95% CI on the scalar headline stats + lag-1 of curves (resample seqs)."""
    accs: dict[str, list] = {}
    n = len(seqs)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        h = headline([seqs[i] for i in idx], kind, **hkw)
        for k, v in h.items():
            val = float(np.asarray(v).ravel()[0]) if k in _CURVE_KEYS else float(v)
            accs.setdefault(k + ("1" if k in _CURVE_KEYS else ""), []).append(val)
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
            for k, v in accs.items()}


def measure(seqs, kind: str, *, seed: int = 0, n_null: int = 200, n_boot: int = 500,
            noise_eps: tuple[float, ...] = (), maxlag: int = DEFAULT_MAXLAG,
            fano_w: int = DEFAULT_FANO_W, mi_bins: int = 8,
            pos_bins: int = DEFAULT_POS_BINS,
            pair: tuple[int, int] | None = None) -> dict:
    """Full signature: real headline + CIs + N1/N2/N3 bands (+ noise check)."""
    rng = np.random.default_rng(seed)
    hkw = dict(maxlag=maxlag, fano_w=fano_w, mi_bins=mi_bins, pair=pair)
    real = headline(seqs, kind, **hkw)
    stats = {
        "kind": kind,
        "n_seqs": len(seqs),
        "n_spans": int(sum(s.size for s in seqs)),
        "real": {k: (np.asarray(v).tolist() if k in _CURVE_KEYS else float(v))
                 for k, v in real.items()},
        "real_ci": bootstrap_ci(seqs, kind, rng, n_boot, **hkw),
        "position_profile": position_profile(seqs, pos_bins).tolist(),
        "nulls": {
            "N1_permute": null_band(seqs, null_permute, rng, kind, n_null, **hkw),
            "N2_trend": null_band(
                seqs, lambda s, r: null_trend(s, r, kind, pos_bins), rng, kind, n_null, **hkw),
            "N3_iid": null_band(
                seqs, lambda s, r: null_iid(s, r, kind), rng, kind, n_null, **hkw),
        },
        "params": {"seed": seed, "n_null": n_null, "n_boot": n_boot, "maxlag": maxlag,
                   "fano_w": fano_w, "mi_bins": mi_bins, "pos_bins": pos_bins},
    }
    if kind == "binary":
        stats["base_rate"] = base_rate(seqs)
        stats["self_excitation"] = self_excitation(seqs)
        stats["inter_event"] = inter_event_cv(seqs)
        stats["markov"] = markov_order_test(seqs, 2)
        stats["mi_vs_lag"] = mi_vs_lag(seqs, maxlag, 2).tolist()
        if noise_eps:
            stats["label_noise"] = {}
            for eps in noise_eps:
                h = headline(flip_labels(seqs, eps, rng), kind, **hkw)
                stats["label_noise"][f"eps={eps}"] = {
                    "acf1": float(h["acf"][0]), "excite_ratio": float(h["excite_ratio"])}
    elif kind == "categorical":
        pooled = np.concatenate([s for s in seqs])
        n_sym = int(pooled.max()) + 1
        stats["n_symbols"] = n_sym
        stats["marginal"] = (np.bincount(pooled.astype(int), minlength=n_sym)
                             / pooled.size).tolist()
        stats["dwell"] = dwell_stats(seqs)
        stats["markov"] = markov_order_test(seqs, n_sym)
    return stats
