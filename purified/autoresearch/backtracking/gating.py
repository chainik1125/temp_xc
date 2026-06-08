"""Backtracking bench — § 8 gating due-diligence (compute BEFORE building).

The bench discriminates only if the per-token information ceiling is well below
the window's. Per-token sees one Bernoulli sample b_i of the intensity lambda_i;
its best linear readout of lambda_i is

    per-token ceiling = corr(b_i, lambda_i) = sqrt( Var(lambda) / Var(b) ).

A window sees the event history b_{i-1..i-K} that *determines* lambda_i, so its
INFORMATION ceiling is 1. But lambda_i = sigmoid(linear-in-history) and the
mandated probe is LINEAR, and the § 5 grid tops out at T = 8 (a tile of size T
exposes only lags 1..T-1, and K = 8 needs T >= 9). So we also compute the
realistic per-T window linear ceiling = corr(lambda_i, OLS(lambda_i ~ in-tile
lagged b's)).

Gate (spec § 8): build if 1 - sqrt(Var(lambda)/Var(b)) >= 0.3 ; else raise alpha
(or lengthen K/tau) and re-check. We also report the honest linear gap
(window_linear - per_token) and an alpha-sweep so the headroom is explicit.

    .venv/bin/python -m autoresearch.backtracking.gating

Deterministic (SEED = 0). No framework / runner involvement — a standalone
analysis like backtracking.py / backtracking_mirror.py. Writes
autoresearch/backtracking/results/backtracking_gating_stats.json + a figure.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SEED = 0
N_SEQS = 16000           # sequences for the Monte-Carlo estimate
L = 64                   # seq_len (spec default)
K = 8                    # kernel length (spec default)
TAU = 2.0                # idealized-kernel decay (spec default)
BASE_TARGET = 0.12       # base rate to tune the idealized intercept to
T_GRID = [2, 4, 8, 16]   # window sizes: tile of size T exposes lags 1..T-1
GATE_THRESHOLD = 0.30    # spec § 8

ROOT = Path(__file__).resolve().parents[2]
MIRROR = ROOT / "autoresearch" / "backtracking" / "results" / "backtracking_mirror_stats.json"
OUT_JSON = ROOT / "autoresearch" / "backtracking" / "results" / "backtracking_gating_stats.json"
FIG_DIR = ROOT / "autoresearch" / "backtracking" / "figs"


def simulate(a, c_pos, w, n_seqs, L, rng):
    """Batched logistic-AR self-exciting Layer-1 process (matches the mirror).

    Returns lambda (n_seqs, L) and b (n_seqs, L). History is zero-padded at the
    sequence start, exactly as backtracking_mirror._generate does.
    """
    w = np.asarray(w, dtype=np.float64)
    Kk = len(w)
    b = np.zeros((n_seqs, L), dtype=np.float64)
    lam = np.zeros((n_seqs, L), dtype=np.float64)
    pos = np.arange(L) / L
    for i in range(L):
        hist = np.zeros(n_seqs)
        for l in range(Kk):
            j = i - 1 - l
            if j >= 0:
                hist += w[l] * b[:, j]
        p = 1.0 / (1.0 + np.exp(-(a + c_pos * pos[i] + hist)))
        lam[:, i] = p
        b[:, i] = (rng.random(n_seqs) < p).astype(np.float64)
    return lam, b


def _lagged_design(b, n_lags):
    """Columns = b_{i-1}..b_{i-n_lags} over all (seq, i), zero-padded at starts."""
    n_seqs, L = b.shape
    cols = []
    for l in range(1, n_lags + 1):
        c = np.zeros_like(b)
        if l < L:
            c[:, l:] = b[:, :-l]
        cols.append(c.reshape(-1))
    return np.stack(cols, axis=1) if cols else np.zeros((b.size, 0))


def _ols_corr(y, X):
    """corr(y, y_hat) for OLS y ~ [1, X]; the linear-probe ceiling from X."""
    if X.shape[1] == 0:
        return 0.0
    A = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    if np.std(yhat) < 1e-12:
        return 0.0
    return float(np.corrcoef(y, yhat)[0, 1])


def ceilings(lam, b):
    """Per-token + window ceilings from a simulated (lambda, b)."""
    lf, bf = lam.reshape(-1), b.reshape(-1)
    var_lam, var_b = float(np.var(lf)), float(np.var(bf))
    per_token = float(np.sqrt(var_lam / var_b))            # = corr(b, lambda)
    emp_corr = float(np.corrcoef(bf, lf)[0, 1])            # sanity check
    # window linear ceiling per tile size T: tile of size T sees lags 1..T-1
    per_T = {}
    for T in T_GRID:
        per_T[T] = _ols_corr(lf, _lagged_design(b, min(T - 1, K)))
    win_full = _ols_corr(lf, _lagged_design(b, K))          # all K lags (T >= K+1)
    return {
        "base_rate": float(np.mean(bf)),
        "var_lambda": var_lam, "var_b": var_b,
        "per_token_ceiling": per_token,
        "per_token_ceiling_empirical_corr": emp_corr,
        "window_info_ceiling": 1.0,
        "window_linear_ceiling_full_history": win_full,
        "window_linear_ceiling_by_T": {str(T): per_T[T] for T in T_GRID},
        "gap_spec_info": 1.0 - per_token,
        "gap_linear_full": win_full - per_token,
    }


def tune_intercept(c_pos, w, rng, target=BASE_TARGET):
    """Bisection on the intercept a so the simulated base rate ~= target."""
    lo, hi = -8.0, 2.0
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        _, b = simulate(mid, c_pos, w, 3000, L, rng)
        if float(np.mean(b)) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    mirror = json.loads(MIRROR.read_text())
    a_fit, c_fit, w_fit = mirror["intercept"], mirror["coef_position"], mirror["kernel_w"]
    alpha_fit = float(np.sum(w_fit))

    rng = np.random.default_rng(SEED)
    results = {"meta": {"N_SEQS": N_SEQS, "L": L, "K": K, "tau": TAU,
                        "base_target": BASE_TARGET, "T_grid": T_GRID,
                        "gate_threshold": GATE_THRESHOLD,
                        "alpha_fit_sum_w": alpha_fit}}

    # (A) fitted mirror, HEADLINE: position trend off (spec § 2)
    lam, b = simulate(a_fit, 0.0, w_fit, N_SEQS, L, rng)
    results["fitted_mirror_headline_c0"] = ceilings(lam, b)
    # (A') fitted mirror WITH position trend, for reference
    lam, b = simulate(a_fit, c_fit, w_fit, N_SEQS, L, rng)
    results["fitted_mirror_with_position_trend"] = ceilings(lam, b)

    # (B) idealized exp-kernel kappa_l = exp(-l/tau), normalized, scaled by alpha;
    #     intercept retuned to base ~= 0.12. Sweep alpha; mark alpha_fit.
    l = np.arange(1, K + 1)
    kappa = np.exp(-l / TAU); kappa = kappa / kappa.sum()
    sweep = []
    for alpha in [1.0, 2.0, 3.0, alpha_fit, 5.0, 6.0, 8.0, 10.0]:
        w = (alpha * kappa).tolist()
        a = tune_intercept(0.0, w, np.random.default_rng(SEED + 1))
        lam, b = simulate(a, 0.0, w, N_SEQS, L, np.random.default_rng(SEED + 2))
        c = ceilings(lam, b)
        sweep.append({"alpha": float(alpha), "intercept": float(a),
                      "base_rate": c["base_rate"],
                      "per_token_ceiling": c["per_token_ceiling"],
                      "window_linear_ceiling_full_history": c["window_linear_ceiling_full_history"],
                      "gap_spec_info": c["gap_spec_info"],
                      "gap_linear_full": c["gap_linear_full"]})
    results["idealized_exp_kernel_alpha_sweep"] = sweep

    hl = results["fitted_mirror_headline_c0"]
    results["verdict"] = {
        "per_token_ceiling": hl["per_token_ceiling"],
        "gap_spec_info": hl["gap_spec_info"],
        "gap_linear_full": hl["gap_linear_full"],
        "passes_spec_gate": bool(hl["gap_spec_info"] >= GATE_THRESHOLD),
        "passes_linear_gate": bool(hl["gap_linear_full"] >= GATE_THRESHOLD),
    }

    OUT_JSON.write_text(json.dumps(results, indent=2))
    _print(results)
    _plot(results)
    return results


def _print(r):
    hl = r["fitted_mirror_headline_c0"]
    v = r["verdict"]
    print("\n============ BACKTRACKING BENCH — § 8 GATING DUE-DILIGENCE ============")
    print(f"fitted mirror (headline, position trend OFF):  base rate = {hl['base_rate']:.3f}")
    print(f"  Var(lambda) = {hl['var_lambda']:.5f}   Var(b) = {hl['var_b']:.5f}")
    print(f"  per-token linear ceiling  sqrt(Varλ/Varb) = {hl['per_token_ceiling']:.3f}"
          f"   (empirical corr(b,λ) = {hl['per_token_ceiling_empirical_corr']:.3f})")
    print(f"  window INFO ceiling                        = 1.000")
    print(f"  window LINEAR ceiling (full history, T>=9) = {hl['window_linear_ceiling_full_history']:.3f}")
    print("  window LINEAR ceiling by tile size T (lags 1..T-1):")
    for T, c in hl["window_linear_ceiling_by_T"].items():
        print(f"      T={T:>2}: {c:.3f}")
    print(f"\n  GAP (spec, info ceiling)   1 - per_token   = {v['gap_spec_info']:.3f}"
          f"   {'>= 0.30  PASS' if v['passes_spec_gate'] else '< 0.30  FAIL'}")
    print(f"  GAP (honest, linear full)  win_lin - pt     = {v['gap_linear_full']:.3f}"
          f"   {'>= 0.30  PASS' if v['passes_linear_gate'] else '< 0.30  FAIL'}")
    print("\n  alpha-sweep (idealized exp kernel, tau=2, base retuned to 0.12):")
    print(f"   {'alpha':>6}{'base':>8}{'per_token':>11}{'win_lin':>9}{'gap_info':>10}{'gap_lin':>9}")
    for s in r["idealized_exp_kernel_alpha_sweep"]:
        mark = "  <- fit" if abs(s["alpha"] - r["meta"]["alpha_fit_sum_w"]) < 1e-6 else ""
        print(f"   {s['alpha']:>6.2f}{s['base_rate']:>8.3f}{s['per_token_ceiling']:>11.3f}"
              f"{s['window_linear_ceiling_full_history']:>9.3f}{s['gap_spec_info']:>10.3f}"
              f"{s['gap_linear_full']:>9.3f}{mark}")
    print(f"\n  -> {OUT_JSON}")


def _plot(r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sweep = r["idealized_exp_kernel_alpha_sweep"]
    al = [s["alpha"] for s in sweep]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(al, [s["per_token_ceiling"] for s in sweep], "o-", color="#d62728", label="per-token ceiling √(Varλ/Varb)")
    ax[0].plot(al, [s["window_linear_ceiling_full_history"] for s in sweep], "s-", color="#1f77b4", label="window linear ceiling (full history)")
    ax[0].axhline(1.0, color="k", lw=0.6, alpha=0.4, label="window info ceiling")
    ax[0].axvline(r["meta"]["alpha_fit_sum_w"], color="green", ls=":", lw=1.5, label=f"fitted α≈{r['meta']['alpha_fit_sum_w']:.2f}")
    ax[0].set_xlabel("self-excitation strength α (Σ kernel)"); ax[0].set_ylabel("λ-recovery ceiling (corr)")
    ax[0].set_title("Ceilings vs α (idealized exp kernel)"); ax[0].legend(fontsize=8); ax[0].grid(True, alpha=0.25)
    ax[1].plot(al, [s["gap_spec_info"] for s in sweep], "o-", color="#9467bd", label="gap (spec, info ceiling)")
    ax[1].plot(al, [s["gap_linear_full"] for s in sweep], "s-", color="#2ca02c", label="gap (honest, linear)")
    ax[1].axhline(GATE_THRESHOLD, color="k", ls="--", lw=1, label=f"gate {GATE_THRESHOLD}")
    ax[1].axvline(r["meta"]["alpha_fit_sum_w"], color="green", ls=":", lw=1.5)
    ax[1].set_xlabel("self-excitation strength α (Σ kernel)"); ax[1].set_ylabel("per-token→window gap (corr)")
    ax[1].set_title("Discrimination gap vs α"); ax[1].legend(fontsize=8); ax[1].grid(True, alpha=0.25)
    fig.suptitle("Backtracking bench § 8 gating: per-token ceiling vs window, and α headroom", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext, dpi in [("pdf", None), ("png", 120), ("thumb.png", 55)]:
        fig.savefig(FIG_DIR / f"backtracking_gating.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {FIG_DIR}/backtracking_gating.*")


if __name__ == "__main__":
    main()
