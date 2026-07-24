"""Sanity tests for support_stats.stats_lib (Stage-2 variance receipts)."""

import numpy as np
import pytest
from scipy.stats import t as t_dist

from experiments.explorations.task_hunt.support_stats.stats_lib import (
    bca_ci, seeds_for_bound, seeds_for_power, seeds_for_signflip,
    sign_flip_p, t_ci95, within_seed_trend)


def test_sign_flip_min_p_at_n3():
    # all-positive diffs: only the identity pattern reaches the observed
    # mean, so the exact one-sided p is exactly 1/8 — the n=3 floor.
    p, n_pat = sign_flip_p([1.0, 2.0, 3.0], "greater")
    assert n_pat == 8
    assert p == pytest.approx(1 / 8)


def test_sign_flip_null_is_calibrated():
    # symmetric diffs: mean 0, half the patterns tie or beat it.
    p, _ = sign_flip_p([1.0, -1.0], "greater")
    assert p == pytest.approx(3 / 4)   # stats {1, 0, 0, -1}, obs 0


def test_t_ci95_matches_scipy():
    v = np.array([0.19, 0.21, 0.23])
    mean, lo, hi = t_ci95(v)
    se = v.std(ddof=1) / np.sqrt(3)
    lo_ref, hi_ref = t_dist.interval(0.95, 2, loc=v.mean(), scale=se)
    assert (mean, lo, hi) == pytest.approx((v.mean(), lo_ref, hi_ref))


def test_bca_ci_properties():
    v = [0.05, 0.06, 0.10]
    ci = bca_ci(v)
    assert ci["exact"] and ci["n_atoms"] == 27
    assert ci["lo"] <= np.mean(v) <= ci["hi"]
    eps = 1e-12
    assert min(v) - eps <= ci["lo"] and ci["hi"] <= max(v) + eps
    # shift equivariance
    ci2 = bca_ci([x + 1.0 for x in v])
    assert ci2["lo"] == pytest.approx(ci["lo"] + 1.0)
    assert ci2["hi"] == pytest.approx(ci["hi"] + 1.0)
    # degenerate spread collapses to the point
    cid = bca_ci([0.2, 0.2, 0.2])
    assert cid["degenerate"] and cid["lo"] == cid["hi"] == 0.2


def test_trend_perm_exact_extremes():
    # strictly increasing in every seed: the identity labeling uniquely
    # maximizes the summed slope -> p = 1 / (3!)^3 = 1/216.
    mat = np.array([[1.0, 2.0, 3.0]] * 3)
    obs, slopes, p, n = within_seed_trend(mat, (2, 4, 8), "greater")
    assert n == 216
    assert p == pytest.approx(1 / 216)
    assert obs == pytest.approx(3.0)          # slope 1 per log2(T), 3 seeds
    # flat values: every relabeling ties.
    _, _, p_flat, _ = within_seed_trend(np.ones((3, 3)), (2, 4, 8),
                                        "greater")
    assert p_flat == pytest.approx(1.0)


def test_seed_power_monotone():
    assert seeds_for_signflip(0.05) == 5
    n_big = seeds_for_bound(1.0, 0.1)
    n_small = seeds_for_bound(0.05, 0.1)
    assert n_big == 2 and n_small > n_big
    assert seeds_for_bound(-0.1, 0.1) is None
    n_pw = seeds_for_power(0.05, 0.03)
    assert n_pw is not None and n_pw >= seeds_for_bound(0.05, 0.03) - 1
