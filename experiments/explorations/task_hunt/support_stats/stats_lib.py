"""Small-n paired statistics for the Stage-2 variance receipts.

Every test here is EXACT (full enumeration) at the n we run it at
(n = 3 seeds): sign-flip permutation over 2^n patterns, bootstrap over
n^n ordered resamples, within-seed trend permutation over (T!)^seeds
relabelings. Exactness matters more than asymptotics at n = 3 — and it
makes the granularity floor explicit: a one-sided sign-flip test at
n = 3 can never report p < 1/8. Callers are expected to say that out
loud rather than dress the number up.

Covered by tests/test_support_stats.py.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.stats import nct, norm, t as t_dist

_EPS = 1e-12
_EXACT_BOOT_MAX_N = 6      # n^n enumeration cap (6^6 = 46,656)
_MC_B = 100_000            # fallback resamples (deterministic seed)


def t_ci95(vals):
    """Two-sided 95% t confidence interval for the mean of `vals`.

    Returns (mean, lo, hi); (mean, nan, nan) when n < 2.
    """
    v = np.asarray(vals, dtype=float)
    n = v.size
    m = float(v.mean())
    if n < 2:
        return m, float("nan"), float("nan")
    hw = float(t_dist.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n))
    return m, m - hw, m + hw


def sign_flip_p(diffs, alternative="greater"):
    """Exact paired sign-flip permutation test on the mean of `diffs`.

    H0: the paired differences are symmetric about 0. Enumerates all
    2^n sign patterns (n <= 20). Returns (p, n_patterns); the identity
    pattern always counts, so min p = 1 / 2^n.
    """
    d = np.asarray(diffs, dtype=float)
    n = d.size
    if n > 20:
        raise ValueError("exact enumeration capped at n = 20")
    obs = d.mean()
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
    stats = (signs * d).mean(axis=1)
    if alternative == "greater":
        p = float((stats >= obs - _EPS).mean())
    elif alternative == "less":
        p = float((stats <= obs + _EPS).mean())
    elif alternative == "two-sided":
        p = float((np.abs(stats) >= abs(obs) - _EPS).mean())
    else:
        raise ValueError(alternative)
    return p, 2 ** n


def bootstrap_means(vals):
    """Bootstrap distribution of the mean.

    Exact (all n^n equally-likely ordered resamples) for n <= 6, else
    deterministic Monte Carlo with `_MC_B` resamples.
    """
    v = np.asarray(vals, dtype=float)
    n = v.size
    if n <= _EXACT_BOOT_MAX_N:
        idx = np.array(list(itertools.product(range(n), repeat=n)))
        return v[idx].mean(axis=1), True
    rng = np.random.default_rng(0)
    return v[rng.integers(0, n, size=(_MC_B, n))].mean(axis=1), False


def bca_ci(vals, alpha=0.05):
    """BCa bootstrap CI for the mean. Returns a dict with endpoints and
    the honesty metadata (n_atoms, n_distinct, exact flag): at n = 3 the
    bootstrap distribution has 27 atoms / <= 10 distinct means, so the
    endpoints are coarse by construction.
    """
    v = np.asarray(vals, dtype=float)
    n = v.size
    theta = float(v.mean())
    if n < 2 or v.max() == v.min():
        return {"lo": float(v[0]), "hi": float(v[0]), "n_atoms": 1,
                "n_distinct": 1, "exact": True, "degenerate": True}
    boots, exact = bootstrap_means(v)
    p0 = ((boots < theta).sum() + 0.5 * (boots == theta).sum()) / boots.size
    z0 = norm.ppf(np.clip(p0, _EPS, 1 - _EPS))
    jack = np.array([np.delete(v, i).mean() for i in range(n)])
    jm = jack.mean()
    den = 6.0 * (((jm - jack) ** 2).sum()) ** 1.5
    a = 0.0 if den == 0 else float(((jm - jack) ** 3).sum() / den)

    def adj(q):
        z = norm.ppf(q)
        return float(norm.cdf(z0 + (z0 + z) / (1.0 - a * (z0 + z))))

    lo = float(np.quantile(boots, np.clip(adj(alpha / 2), 0, 1),
                           method="inverted_cdf"))
    hi = float(np.quantile(boots, np.clip(adj(1 - alpha / 2), 0, 1),
                           method="inverted_cdf"))
    return {"lo": lo, "hi": hi, "n_atoms": int(boots.size),
            "n_distinct": int(np.unique(boots).size), "exact": bool(exact),
            "degenerate": False}


def within_seed_trend(vals, Ts, alternative="greater"):
    """Exact permutation test for a monotone trend in T, pooled over seeds.

    `vals` is (n_seeds, n_T) aligned with `Ts`. Statistic: the sum over
    seeds of the per-seed OLS slope of value on log2(T). Null: within
    each seed the values are exchangeable across T (no T-dependence);
    enumerate all (n_T!)^n_seeds within-seed relabelings. Returns
    (slope_sum, per_seed_slopes, p, n_perms).
    """
    m = np.asarray(vals, dtype=float)
    n_seeds, n_t = m.shape
    x = np.log2(np.asarray(Ts, dtype=float))
    xc = x - x.mean()
    denom = float(xc @ xc)

    def slopes(mm):
        return (mm @ xc) / denom

    obs_slopes = slopes(m)
    obs = float(obs_slopes.sum())
    perms = list(itertools.permutations(range(n_t)))
    if len(perms) ** n_seeds > 200_000:
        raise ValueError("exact enumeration too large; reduce Ts/seeds")
    ge = total = 0
    for combo in itertools.product(perms, repeat=n_seeds):
        s = float(sum((m[i, list(p)] @ xc) / denom
                      for i, p in enumerate(combo)))
        total += 1
        if alternative == "greater" and s >= obs - _EPS:
            ge += 1
        elif alternative == "two-sided" and abs(s) >= abs(obs) - _EPS:
            ge += 1
    return obs, [float(s) for s in obs_slopes], ge / total, total


def seeds_for_bound(mean_d, sd_d, alpha=0.05, max_n=60):
    """Smallest n with a plug-in one-sided (1-alpha) lower t-bound > 0:
    mean_d - t_{1-alpha, n-1} * sd_d / sqrt(n) > 0. None if mean_d <= 0
    or not reached by max_n.
    """
    if mean_d <= 0 or sd_d < 0:
        return None
    if sd_d == 0:
        return 2
    for n in range(2, max_n + 1):
        if mean_d - t_dist.ppf(1 - alpha, n - 1) * sd_d / np.sqrt(n) > 0:
            return n
    return None


def seeds_for_power(mean_d, sd_d, alpha=0.05, power=0.8, max_n=60):
    """Smallest n giving >= `power` for the one-sided one-sample t-test
    on the paired differences (noncentral-t power at effect mean_d/sd_d).
    """
    if mean_d <= 0 or sd_d < 0:
        return None
    if sd_d == 0:
        return 2
    effect = mean_d / sd_d
    for n in range(2, max_n + 1):
        crit = t_dist.ppf(1 - alpha, n - 1)
        pw = 1.0 - nct.cdf(crit, n - 1, effect * np.sqrt(n))
        if pw >= power:
            return n
    return None


def seeds_for_signflip(alpha=0.05):
    """Smallest n at which the exact one-sided sign-flip test can reach
    p <= alpha at all (2^-n <= alpha)."""
    n = 1
    while 2.0 ** -n > alpha:
        n += 1
    return n
