"""Hedging-drift bench — § 8 gating due-diligence (BEFORE the grid).

Confirms, on the real generator (briefing gate): (i) the confidence latent's
oracle is reachable by a probe on the noiseless emission — quantified honestly:
the spec's oracle (R² = 1, the generating ``c_i`` itself) is **not** attainable
under the emission's per-token folded-normal magnitude (``proj_conf = c_i·m_i``
carries irreducible multiplicative noise), so this script pins the actual
per-token linear ceiling and the per-``T`` raw-window ceilings; and (ii) the
chance floor sits at 0 (predicting the pooled mean).

Unlike changepoint's AC latent (equality-pattern, provably invisible to raw
linear window readers), ``c_i`` is *linearly present* in the raw activations —
so a window arch's recovery gain here can be plain linear access to
neighbouring tokens (temporal denoising of the multiplicative noise). That is
the DC axis under test; the untrained-encoder control in the grid is the
access-vs-learning arbiter. Recorded before any architecture runs.

Gate (preregistered here, before any grid):
- separable: per-token ridge probe on the noiseless emission reaches
  R² ≥ 0.30 (well above chance; the expected ceiling is ≈ 0.8);
- chance floor: shuffled-label probe |R²| ≤ 0.02;
- mirror sanity: pooled within-sequence ACF(1) within ±0.05 of the C3 fit
  (0.333) and the lag-4 plateau held (ACF(4) ≥ 0.06 — the gate-8 property).

    .venv/bin/python -m experiments.explorations.synthetic.hedging_drift.gating

Deterministic (SEED = 0); standalone (no framework / runner involvement).
Writes results/hedging_gating_stats.json + a figure.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

SEED = 0
N_SEQS_PROBE = 6000
SEQ_LEN = 64
T_GRID = [2, 4, 8]
MAX_PROBE_ROWS = 120_000
ACF_LAGS = list(range(1, 9))
GATE_R2_MIN = 0.30            # per-token ridge on noiseless emission
GATE_CHANCE_TOL = 0.02        # |shuffled-label R²|
GATE_ACF1_TOL = 0.05          # pooled ACF(1) vs the C3 fit
ACF1_FIT = 0.333              # syn ACF(1) of the C3 hier_ar1 fit (spec amendment)
GATE_ACF4_MIN = 0.06          # the plateau the hier mirror exists to hold

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "results" / "hedging_gating_stats.json"
FIG_DIR = HERE / "figs"


def _subsample(rng, *arrays, cap=MAX_PROBE_ROWS):
    n = arrays[0].shape[0]
    if n <= cap:
        return arrays
    idx = rng.choice(n, size=cap, replace=False)
    return tuple(a[idx] for a in arrays)


def _ridge_r2(X_tr, y_tr, X_ev, y_ev):
    from sklearn.linear_model import Ridge
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg = Ridge(alpha=1.0).fit(X_tr, y_tr)
        r2 = float(reg.score(X_ev, y_ev))
        pred = reg.predict(X_ev)
        corr = (float(np.corrcoef(pred, y_ev)[0, 1])
                if np.std(pred) > 1e-12 else 0.0)
    return r2, corr


def _strided_tiles(x, T):
    n, Ls, d = x.shape
    n_tiles = Ls // T
    return x[:, :n_tiles * T].reshape(n * n_tiles, T * d)


def _leading_edge(labels, T):
    n, Ls = labels.shape[:2]
    n_tiles = Ls // T
    return labels[:, :n_tiles * T].reshape(n, n_tiles, T)[:, :, T - 1].reshape(-1)


def pooled_acf(c: np.ndarray, lag: int) -> float:
    x = c[:, :-lag].reshape(-1)
    y = c[:, lag:].reshape(-1)
    return float(np.corrcoef(x, y)[0, 1])


def main():
    from temp_bench.data.synthetic import hedging_drift

    rng = np.random.default_rng(SEED)
    data = hedging_drift(seq_len=SEQ_LEN, n_seqs=N_SEQS_PROBE, seed=SEED)
    x = data.x.numpy()
    c = data.extra["conf_labels"].numpy().astype(np.float64)
    u_conf = data.hidden_features[0].numpy()

    results = {"meta": {
        "N_SEQS_PROBE": N_SEQS_PROBE, "seq_len": SEQ_LEN, "T_grid": T_GRID,
        "gates": {"r2_min": GATE_R2_MIN, "chance_tol": GATE_CHANCE_TOL,
                  "acf1_fit": ACF1_FIT, "acf1_tol": GATE_ACF1_TOL,
                  "acf4_min": GATE_ACF4_MIN},
        "mirror_params": {k: data.extra[k]
                          for k in ("mu", "beta_position", "rho", "ar_sigma")},
    }}

    # ── mirror sanity: the pooled ACF plateau ─────────────────────────────
    acf = {str(k): pooled_acf(c, k) for k in ACF_LAGS}
    results["mirror"] = {
        "pooled_acf": acf,
        "c_mean": float(c.mean()), "c_std": float(c.std()),
        "level_sd": float(data.extra["level_labels"].numpy().std()),
    }

    # ── per-token ceilings on the noiseless emission ──────────────────────
    half = N_SEQS_PROBE // 2
    Xtr = x[:half].reshape(-1, x.shape[-1])
    Xev = x[half:].reshape(-1, x.shape[-1])
    ctr, cev = c[:half].reshape(-1), c[half:].reshape(-1)
    Xtr, ctr = _subsample(rng, Xtr, ctr)
    Xev, cev = _subsample(rng, Xev, cev)
    r2, corr = _ridge_r2(Xtr, ctr, Xev, cev)
    perm = rng.permutation(len(ctr))
    r2_fl, _ = _ridge_r2(Xtr, ctr[perm], Xev, cev)
    # reference: the scalar u_conf projection alone (the emission carrier)
    ptr = (x[:half] @ u_conf).reshape(-1, 1)
    pev = (x[half:] @ u_conf).reshape(-1, 1)
    ptr2, ctr2 = _subsample(rng, ptr, c[:half].reshape(-1))
    pev2, cev2 = _subsample(rng, pev, c[half:].reshape(-1))
    r2_proj, _ = _ridge_r2(ptr2, ctr2, pev2, cev2)
    results["per_token"] = {
        "r2_on_x": r2, "corr_on_x": corr, "r2_shuffled_floor": r2_fl,
        "r2_on_uconf_projection": r2_proj,
    }

    # ── window raw-linear ceilings (concatenated raw tile, leading edge) ──
    results["window"] = {}
    for T in T_GRID:
        Xt = _strided_tiles(x[:half], T)
        Xe = _strided_tiles(x[half:], T)
        ct = _leading_edge(c[:half], T)
        ce = _leading_edge(c[half:], T)
        Xt, ct = _subsample(rng, Xt, ct)
        Xe, ce = _subsample(rng, Xe, ce)
        wr2, wcorr = _ridge_r2(Xt, ct, Xe, ce)
        results["window"][str(T)] = {"r2_raw_linear": wr2,
                                     "corr_raw_linear": wcorr}

    # ── verdict ───────────────────────────────────────────────────────────
    pt = results["per_token"]
    results["verdict"] = {
        "separable": bool(pt["r2_on_x"] >= GATE_R2_MIN),
        "chance_floor_ok": bool(abs(pt["r2_shuffled_floor"]) <= GATE_CHANCE_TOL),
        "acf1_matches_fit": bool(abs(acf["1"] - ACF1_FIT) <= GATE_ACF1_TOL),
        "plateau_held": bool(acf["4"] >= GATE_ACF4_MIN),
        "oracle_r2_1_reachable": bool(pt["r2_on_x"] >= 0.99),  # expected False
    }
    results["verdict"]["passes_gate"] = bool(
        results["verdict"]["separable"]
        and results["verdict"]["chance_floor_ok"]
        and results["verdict"]["acf1_matches_fit"]
        and results["verdict"]["plateau_held"])

    OUT_JSON.write_text(json.dumps(results, indent=2))
    _print(results)
    _plot(results)
    return results


def _print(r):
    m, pt, v = r["mirror"], r["per_token"], r["verdict"]
    acf = m["pooled_acf"]
    print("\n=========== HEDGING-DRIFT — § 8 GATING DUE-DILIGENCE ===========")
    print(f"mirror: pooled ACF lags 1-8 = "
          + ", ".join(f"{acf[str(k)]:.3f}" for k in ACF_LAGS))
    print(f"        (C3 fit syn: 0.33, 0.17, 0.14, 0.12, 0.13, 0.14, 0.13, 0.13; "
          f"c mean/sd = {m['c_mean']:.3f}/{m['c_std']:.3f})")
    print("\n  per-token ceilings (ridge on noiseless x_t):")
    print(f"    R² on x            {pt['r2_on_x']:.3f}   (corr {pt['corr_on_x']:.3f}; "
          f"shuffled floor {pt['r2_shuffled_floor']:+.3f})")
    print(f"    R² on u_conf proj  {pt['r2_on_uconf_projection']:.3f}   "
          "(the c_i·m_i multiplicative-noise ceiling)")
    print("\n  window raw-linear ceilings (concatenated tile, leading edge):")
    print(f"   {'T':>3}{'R²':>9}{'corr':>9}")
    for T, w in r["window"].items():
        print(f"   {T:>3}{w['r2_raw_linear']:>9.3f}{w['corr_raw_linear']:>9.3f}")
    print("\n  NOTE: the spec oracle (R²=1, the generating c_i) is NOT reachable "
          f"under the folded-normal magnitude — reachable: {v['oracle_r2_1_reachable']}. "
          "c_i is linearly present in the raw window, so a window gain can be "
          "linear access (temporal denoising); the untrained control arbitrates.")
    print(f"\n  VERDICT: separable (R² ≥ {GATE_R2_MIN})      {v['separable']}")
    print(f"           chance floor ok            {v['chance_floor_ok']}")
    print(f"           ACF(1) matches C3 fit      {v['acf1_matches_fit']}")
    print(f"           lag-4 plateau held         {v['plateau_held']}")
    print(f"           ==> passes_gate = {v['passes_gate']}")
    print(f"\n  -> {OUT_JSON}")


def _plot(r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pt = r["per_token"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    lags = ACF_LAGS
    ax1.plot(lags, [r["mirror"]["pooled_acf"][str(k)] for k in lags], "o-",
             color="#1f77b4", label="generated stream (pooled ACF)")
    ax1.plot(lags, [0.33, 0.17, 0.14, 0.12, 0.13, 0.14, 0.13, 0.13], "s--",
             color="#2ca02c", alpha=0.8, label="C3 hier_ar1 fit (syn)")
    ax1.plot(lags, [0.32, 0.17, 0.16, 0.14, 0.14, 0.16, 0.17, 0.14], "^:",
             color="0.4", alpha=0.8, label="real held-out")
    rho = r["meta"]["mirror_params"]["rho"]
    ax1.plot(lags, [rho ** k for k in lags], color="#d62728", ls=":",
             lw=1.2, label=f"AR(1)-only collapse (ρ={rho:.2f})")
    ax1.set_xlabel("lag"); ax1.set_ylabel("autocorrelation")
    ax1.set_title("The long-memory plateau (mirror sanity)")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.25); ax1.set_ylim(-0.02, 0.45)

    Ts = [1] + [int(t) for t in r["window"]]
    r2s = [pt["r2_on_x"]] + [r["window"][str(T)]["r2_raw_linear"] for T in Ts[1:]]
    ax2.plot(Ts, r2s, "o-", color="#1f77b4", lw=2, label="raw-linear ceiling")
    ax2.axhline(1.0, color="0.3", ls="--", lw=1.0, label="spec oracle R²=1 (unreachable)")
    ax2.axhline(0.0, color="0.6", ls=":", lw=1.0, label="chance (pooled mean)")
    ax2.set_xscale("log", base=2); ax2.set_xticks(Ts); ax2.set_xticklabels(Ts)
    ax2.set_xlabel("tile size T  (T=1: per-token)")
    ax2.set_ylabel("held-out R² of c_i at the leading edge")
    ax2.set_title("Access ceilings: multiplicative noise vs temporal denoising")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.25); ax2.set_ylim(-0.05, 1.05)

    fig.suptitle("hedging_drift § 8 gating: plateau sanity + linear-access ceilings",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext, dpi in [("pdf", None), ("png", 120)]:
        fig.savefig(FIG_DIR / f"hedging_gating.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {FIG_DIR}/hedging_gating.*")


if __name__ == "__main__":
    main()
