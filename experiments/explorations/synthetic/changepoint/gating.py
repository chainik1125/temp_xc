"""Change-point bench — § 8 gating due-diligence (compute BEFORE building).

The bench discriminates only if the two ceilings are well separated on *both*
latents (bench_spec.md § 8). The substrate is the semi-Markov mode process at
the **measured geometric dwell** (topic-switching measurement: dwell ≈
geometric, mean run 1.73 → p_switch = 1/1.73 ≈ 0.578), K_m = 8 modes, Π =
uniform-over-other-modes (rebalanced by design so the current mode carries no
switch information — the § 8 (i) requirement holds *exactly*, then is verified
empirically here).

Per-token side (noiseless emission ⇒ x_t reveals m_t and nothing else about the
past, content is mode-independent ⇒ DPI):
  (i)  best predictor of the AC latents from m_t alone — c_t = [m_t ≠ m_{t-1}]
       sits at chance (P(c|m=k) = p_switch for every k by Π-symmetry), and
       time-since-switch τ_t has corr ≈ 0 (E[τ|m=k] constant in k).
  (ii) mode oracle: a multinomial-logistic probe on the noiseless x_t reaches
       balanced accuracy ≈ 1 (the DC half is informative).

Window side, per tile size T ∈ {2, 4, 8} (leading-edge target, as the eval
probes it):
  - INFO ceiling: c_t is exact for T ≥ 2 (the adjacent pair is in-tile) → 1.0;
    τ_t is exact iff τ ≤ T-2 (last boundary visible in-tile), else censored →
    the conditional-mean predictor's corr, computed empirically.
  - RAW-LINEAR ceiling: OLS / logistic on the *concatenated raw activations* of
    the tile. By mode-symmetry any additive (linear) score Σ_j a_j(m_{t-j}) has
    zero covariance with the equality-pattern latents, so this sits ≈ chance —
    equality of two one-hots is XOR-like, NOT linearly separable. Verified
    empirically here. Consequence: a window *win* on the learned code cannot be
    pure linear access; it requires training to expose boundary structure
    (and the untrained-encoder control checks the nonlinear-access residual).

Gate: mode oracle ≥ 0.99 balanced acc; per-token AC at chance (|balacc-0.5| ≤
0.02 for c, |corr| ≤ 0.05 for τ); window INFO ceiling for τ at T=2 exceeds the
per-token ceiling by ≥ 0.30. A dwell-mean sweep shows the headroom.

    .venv/bin/python -m experiments.explorations.synthetic.changepoint.gating

Deterministic (SEED = 0). No framework / runner involvement — a standalone
analysis like backtracking.gating. Writes
changepoint/results/changepoint_gating_stats.json + a figure.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

SEED = 0
N_SEQS = 20000            # sequences for the Monte-Carlo estimates
N_SEQS_PROBE = 6000       # sequences materialised into activations for probes
L = 64                    # seq_len (spec default)
K_M = 8                   # modes (spec default)
DWELL_MEAN = 1.73         # measured topic dwell (mean run, geometric) — the anchor
D_IN = 64
C_CONTENT = 12            # content directions (F = K_M + C = 20)
SPREAD = 3
MAG_MODE = 2.5            # dominant mode-signature magnitude (recoverable at k_pos=1)
MAG_CONTENT = 1.0
T_GRID = [2, 4, 8]
DWELL_SWEEP = [1.2, 1.73, 2.5, 4.0, 8.0]
GATE_SEPARATION = 0.30    # τ: window info ceiling (T=2) - per-token
GATE_MODE_ORACLE = 0.99   # per-token mode balanced acc on noiseless emission
GATE_C_TOL = 0.02         # |per-token c balacc - 0.5|
GATE_TAU_TOL = 0.05       # |per-token τ corr|
MAX_PROBE_ROWS = 120_000  # subsample cap for sklearn probes

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "results" / "changepoint_gating_stats.json"
FIG_DIR = HERE / "figs"


# ── the Layer-1 process (matches the planned semi_markov_modes generator) ──


def simulate_modes(n_seqs, L, K_m, p_switch, rng):
    """Geometric-dwell semi-Markov modes (== first-order Markov, uniform Π).

    Returns m (n_seqs, L) int64, c (n_seqs, L) float (c_0 = 0), and
    time-since-switch tau (n_seqs, L) int64 (tau_0 = 0; sequence start counts
    as a renewal).
    """
    m = np.zeros((n_seqs, L), dtype=np.int64)
    m[:, 0] = rng.integers(0, K_m, n_seqs)
    for t in range(1, L):
        switch = rng.random(n_seqs) < p_switch
        jump = 1 + rng.integers(0, K_m - 1, n_seqs)     # uniform over OTHER modes
        m[:, t] = np.where(switch, (m[:, t - 1] + jump) % K_m, m[:, t - 1])
    c = np.zeros((n_seqs, L), dtype=np.float64)
    c[:, 1:] = (m[:, 1:] != m[:, :-1]).astype(np.float64)
    tau = np.zeros((n_seqs, L), dtype=np.int64)
    for t in range(1, L):
        tau[:, t] = np.where(c[:, t] == 1, 0, tau[:, t - 1] + 1)
    return m, c, tau


def emit(m, rng, *, d_in=D_IN, K_m=K_M, C=C_CONTENT, spread=SPREAD,
         mag_mode=MAG_MODE, mag_content=MAG_CONTENT, sigma=0.0):
    """Layer-2 emission (matches the planned generator; content mode-INdependent)."""
    raw = rng.standard_normal((d_in, d_in))
    Q, _ = np.linalg.qr(raw)
    U_m = Q[:K_m].astype(np.float32)                    # mode-signature dirs
    U_c = Q[K_m:K_m + C].astype(np.float32)             # content dirs
    n, Ls = m.shape
    mm = np.abs(rng.normal(mag_mode, 0.3 * mag_mode, size=(n, Ls))).astype(np.float32)
    x = mm[..., None] * U_m[m]                          # (n, L, d_in)
    pick = np.argsort(rng.random((n, Ls, C)), axis=-1)[:, :, :spread]
    cmag = np.abs(rng.normal(mag_content, 0.3 * mag_content,
                             size=(n, Ls, spread))).astype(np.float32)
    x += (cmag[..., None] * U_c[pick]).sum(axis=2)
    if sigma > 0:
        x += (sigma * rng.standard_normal(x.shape)).astype(np.float32)
    return x


# ── probe helpers ──────────────────────────────────────────────────────


def _subsample(rng, *arrays, cap=MAX_PROBE_ROWS):
    n = arrays[0].shape[0]
    if n <= cap:
        return arrays
    idx = rng.choice(n, size=cap, replace=False)
    return tuple(a[idx] for a in arrays)


def _logistic_balacc(X_tr, y_tr, X_ev, y_ev):
    """Held-out balanced accuracy of a logistic probe (binary or multinomial)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(max_iter=300).fit(X_tr, y_tr)
        return float(balanced_accuracy_score(y_ev, clf.predict(X_ev)))


def _ols_corr(X_tr, y_tr, X_ev, y_ev):
    """Held-out corr(pred, y) of an OLS probe — the linear ceiling from X."""
    A_tr = np.concatenate([np.ones((X_tr.shape[0], 1), dtype=X_tr.dtype), X_tr], axis=1)
    A_ev = np.concatenate([np.ones((X_ev.shape[0], 1), dtype=X_ev.dtype), X_ev], axis=1)
    coef, *_ = np.linalg.lstsq(A_tr, y_tr, rcond=None)
    pred = A_ev @ coef
    if np.std(pred) < 1e-12:
        return 0.0
    return float(np.corrcoef(pred, y_ev)[0, 1])


def _strided_tiles(x, T):
    """Non-overlapping tiles; rows = (seq, tile), cols = T·d_in; leading edge t = (k+1)T-1."""
    n, Ls, d = x.shape
    n_tiles = Ls // T
    return x[:, :n_tiles * T].reshape(n * n_tiles, T * d)


def _leading_edge(labels, T):
    n, Ls = labels.shape[:2]
    n_tiles = Ls // T
    return labels[:, :n_tiles * T].reshape(n, n_tiles, T)[:, :, T - 1].reshape(-1)


# ── ceilings ───────────────────────────────────────────────────────────


def per_token_ceilings_from_m(m, c, tau):
    """Best predictors of the AC latents from m_t alone (the exact DPI side)."""
    half = m.shape[0] // 2
    m_tr, m_ev = m[:half].reshape(-1), m[half:].reshape(-1)
    c_tr, c_ev = c[:half].reshape(-1), c[half:].reshape(-1)
    t_tr, t_ev = tau[:half].reshape(-1).astype(np.float64), tau[half:].reshape(-1).astype(np.float64)

    cond_rates = {int(k): float(c_tr[m_tr == k].mean()) for k in range(K_M)}
    # best per-mode label assignment for balanced accuracy: predict 1 iff
    # P(c|m=k) > base rate (the balanced-optimal rule), scored held-out
    base = float(c_tr.mean())
    pred = np.array([cond_rates[int(k)] > base for k in m_ev], dtype=np.float64)
    tp = pred[c_ev == 1].mean() if (c_ev == 1).any() else 0.0
    tn = (1 - pred)[c_ev == 0].mean() if (c_ev == 0).any() else 0.0
    balacc_c = float(0.5 * (tp + tn))

    cond_tau = np.array([t_tr[m_tr == k].mean() for k in range(K_M)])
    pred_tau = cond_tau[m_ev]
    corr_tau = (float(np.corrcoef(pred_tau, t_ev)[0, 1])
                if np.std(pred_tau) > 1e-12 else 0.0)
    return {
        "cond_switch_rate_by_mode": cond_rates,
        "base_switch_rate": base,
        "c_balacc_from_m": balacc_c,
        "tau_corr_from_m": corr_tau,
        "tau_mean": float(t_tr.mean()), "tau_std": float(t_tr.std()),
    }


def per_token_probes_on_x(x, m, c, tau, rng):
    """The (i)+(ii) probes on the noiseless emission, as the eval would run them."""
    half = x.shape[0] // 2
    Xtr = x[:half].reshape(-1, x.shape[-1])
    Xev = x[half:].reshape(-1, x.shape[-1])
    mtr, mev = m[:half].reshape(-1), m[half:].reshape(-1)
    ctr, cev = c[:half].reshape(-1), c[half:].reshape(-1)
    ttr, tev = (tau[:half].reshape(-1).astype(np.float64),
                tau[half:].reshape(-1).astype(np.float64))
    Xtr, mtr, ctr, ttr = _subsample(rng, Xtr, mtr, ctr, ttr)
    Xev, mev, cev, tev = _subsample(rng, Xev, mev, cev, tev)
    return {
        "mode_balacc_on_x": _logistic_balacc(Xtr, mtr, Xev, mev),
        "c_balacc_on_x": _logistic_balacc(Xtr, ctr, Xev, cev),
        "tau_corr_on_x": _ols_corr(Xtr, ttr, Xev, tev),
    }


def window_info_ceiling_tau(c, tau, T):
    """Censored conditional-mean predictor: τ exact iff last boundary in-tile.

    A size-T tile ending at t sees boundaries at positions t-T+2..t, so τ is
    determined iff τ ≤ T-2; else only "τ ≥ T-1" is known and the best predictor
    is the conditional mean. This is the ceiling for ANY tile-based readout
    (DPI), linear or not.
    """
    half = c.shape[0] // 2
    t_tr = _leading_edge(tau[:half], T).astype(np.float64)
    t_ev = _leading_edge(tau[half:], T).astype(np.float64)
    cens_mean = t_tr[t_tr >= T - 1].mean() if (t_tr >= T - 1).any() else float(T - 1)
    pred = np.where(t_ev <= T - 2, t_ev, cens_mean)
    if np.std(pred) < 1e-12:
        return 0.0
    return float(np.corrcoef(pred, t_ev)[0, 1])


def window_raw_linear_ceilings(x, c, tau, T, rng):
    """OLS/logistic on the raw concatenated tile → ≈ chance by mode-symmetry."""
    half = x.shape[0] // 2
    Xtr = _strided_tiles(x[:half], T)
    Xev = _strided_tiles(x[half:], T)
    ctr, cev = _leading_edge(c[:half], T), _leading_edge(c[half:], T)
    ttr, tev = (_leading_edge(tau[:half], T).astype(np.float64),
                _leading_edge(tau[half:], T).astype(np.float64))
    Xtr, ctr, ttr = _subsample(rng, Xtr, ctr, ttr)
    Xev, cev, tev = _subsample(rng, Xev, cev, tev)
    return {
        "c_balacc_raw_linear": _logistic_balacc(Xtr, ctr, Xev, cev),
        "tau_corr_raw_linear": _ols_corr(Xtr, ttr, Xev, tev),
    }


def headline_block(p_switch, rng):
    """All ceilings at one dwell setting (modes Monte-Carlo + emission probes)."""
    m, c, tau = simulate_modes(N_SEQS, L, K_M, p_switch, rng)
    blk = {"p_switch": float(p_switch),
           "per_token_from_m": per_token_ceilings_from_m(m, c, tau)}

    mp, cp, taup = m[:N_SEQS_PROBE], c[:N_SEQS_PROBE], tau[:N_SEQS_PROBE]
    x = emit(mp, rng)
    blk["per_token_probes_on_x"] = per_token_probes_on_x(x, mp, cp, taup, rng)

    blk["window"] = {}
    for T in T_GRID:
        blk["window"][str(T)] = {
            "c_info_ceiling": 1.0,    # adjacent pair in-tile for T >= 2
            "tau_info_ceiling": window_info_ceiling_tau(c, tau, T),
            **window_raw_linear_ceilings(x, cp, taup, T, rng),
        }
    return blk


def main():
    rng = np.random.default_rng(SEED)
    p_anchor = 1.0 / DWELL_MEAN
    results = {"meta": {
        "N_SEQS": N_SEQS, "N_SEQS_PROBE": N_SEQS_PROBE, "L": L, "K_m": K_M,
        "dwell_mean_anchor": DWELL_MEAN, "p_switch_anchor": p_anchor,
        "d_in": D_IN, "C": C_CONTENT, "spread": SPREAD,
        "mag_mode": MAG_MODE, "mag_content": MAG_CONTENT,
        "T_grid": T_GRID, "dwell_sweep": DWELL_SWEEP,
        "gates": {"separation_tau_T2": GATE_SEPARATION,
                  "mode_oracle": GATE_MODE_ORACLE,
                  "c_tol": GATE_C_TOL, "tau_tol": GATE_TAU_TOL},
    }}

    # (A) headline: the measured-dwell anchor
    results["anchor"] = headline_block(p_anchor, rng)

    # (B) dwell-mean sweep (persistence-knob headroom; modes only + info ceilings)
    sweep = []
    for dm in DWELL_SWEEP:
        p = 1.0 / dm
        m, c, tau = simulate_modes(N_SEQS, L, K_M, p, np.random.default_rng(SEED + 1))
        pt = per_token_ceilings_from_m(m, c, tau)
        sweep.append({
            "dwell_mean": dm, "p_switch": p,
            "base_switch_rate": pt["base_switch_rate"],
            "c_balacc_from_m": pt["c_balacc_from_m"],
            "tau_corr_from_m": pt["tau_corr_from_m"],
            "tau_info_ceiling_by_T": {str(T): window_info_ceiling_tau(c, tau, T)
                                      for T in T_GRID},
        })
    results["dwell_sweep"] = sweep

    a = results["anchor"]
    pt_m, pt_x = a["per_token_from_m"], a["per_token_probes_on_x"]
    tau_T2 = a["window"]["2"]["tau_info_ceiling"]
    tau_pt = max(abs(pt_m["tau_corr_from_m"]), abs(pt_x["tau_corr_on_x"]))
    results["verdict"] = {
        "mode_oracle_reachable": bool(pt_x["mode_balacc_on_x"] >= GATE_MODE_ORACLE),
        "c_per_token_at_chance": bool(abs(pt_x["c_balacc_on_x"] - 0.5) <= GATE_C_TOL
                                      and abs(pt_m["c_balacc_from_m"] - 0.5) <= GATE_C_TOL),
        "tau_per_token_at_chance": bool(tau_pt <= GATE_TAU_TOL),
        "tau_separation_T2": float(tau_T2 - tau_pt),
        "tau_separation_passes": bool(tau_T2 - tau_pt >= GATE_SEPARATION),
        "c_separation": float(1.0 - pt_x["c_balacc_on_x"]),
    }
    results["verdict"]["passes_gate"] = bool(
        results["verdict"]["mode_oracle_reachable"]
        and results["verdict"]["c_per_token_at_chance"]
        and results["verdict"]["tau_per_token_at_chance"]
        and results["verdict"]["tau_separation_passes"])

    OUT_JSON.write_text(json.dumps(results, indent=2))
    _print(results)
    _plot(results)
    return results


def _print(r):
    a, v = r["anchor"], r["verdict"]
    pt_m, pt_x = a["per_token_from_m"], a["per_token_probes_on_x"]
    print("\n========== CHANGEPOINT BENCH — § 8 GATING DUE-DILIGENCE ==========")
    print(f"anchor: geometric dwell mean {DWELL_MEAN} -> p_switch = {a['p_switch']:.3f}"
          f"   (base switch rate = {pt_m['base_switch_rate']:.3f}, "
          f"tau mean/std = {pt_m['tau_mean']:.2f}/{pt_m['tau_std']:.2f})")
    print("\n  per-token ceilings (from m_t exactly / probe on noiseless x_t):")
    print(f"    mode balanced acc (oracle check)     —      / {pt_x['mode_balacc_on_x']:.3f}")
    print(f"    c_t balanced acc                     {pt_m['c_balacc_from_m']:.3f} / {pt_x['c_balacc_on_x']:.3f}   (chance 0.500)")
    print(f"    tau corr                             {pt_m['tau_corr_from_m']:+.3f} / {pt_x['tau_corr_on_x']:+.3f}   (chance 0)")
    print("\n  window ceilings by tile size T (leading edge):")
    print(f"   {'T':>3}{'c info':>9}{'c raw-lin':>11}{'tau info':>10}{'tau raw-lin':>13}")
    for T in T_GRID:
        w = a["window"][str(T)]
        print(f"   {T:>3}{w['c_info_ceiling']:>9.3f}{w['c_balacc_raw_linear']:>11.3f}"
              f"{w['tau_info_ceiling']:>10.3f}{w['tau_corr_raw_linear']:>13.3f}")
    print("\n  dwell-mean sweep (persistence-knob headroom):")
    print(f"   {'dwell':>7}{'p_sw':>7}{'base':>7}{'pt c':>7}{'pt tau':>8}"
          + "".join(f"{'tau T=' + str(T):>10}" for T in T_GRID))
    for s in r["dwell_sweep"]:
        mark = "  <- anchor" if abs(s["dwell_mean"] - DWELL_MEAN) < 1e-9 else ""
        print(f"   {s['dwell_mean']:>7.2f}{s['p_switch']:>7.3f}{s['base_switch_rate']:>7.3f}"
              f"{s['c_balacc_from_m']:>7.3f}{s['tau_corr_from_m']:>8.3f}"
              + "".join(f"{s['tau_info_ceiling_by_T'][str(T)]:>10.3f}" for T in T_GRID)
              + mark)
    print(f"\n  VERDICT: mode oracle reachable      {v['mode_oracle_reachable']}")
    print(f"           c  per-token at chance     {v['c_per_token_at_chance']}   (separation to info ceiling: {v['c_separation']:.3f})")
    print(f"           tau per-token at chance    {v['tau_per_token_at_chance']}")
    print(f"           tau separation (T=2)       {v['tau_separation_T2']:.3f}  "
          f"{'>= 0.30  PASS' if v['tau_separation_passes'] else '< 0.30  FAIL'}")
    print(f"           ==> passes_gate = {v['passes_gate']}")
    print(f"\n  -> {OUT_JSON}")


def _plot(r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = r["anchor"]
    sweep = r["dwell_sweep"]
    dm = [s["dwell_mean"] for s in sweep]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    colors = {2: "#1f77b4", 4: "#2ca02c", 8: "#9467bd"}
    for T in T_GRID:
        ax[0].plot(dm, [s["tau_info_ceiling_by_T"][str(T)] for s in sweep], "s-",
                   color=colors[T], label=f"window info ceiling, T={T}")
    ax[0].plot(dm, [abs(s["tau_corr_from_m"]) for s in sweep], "o-",
               color="#d62728", label="per-token ceiling (from $m_t$)")
    ax[0].axvline(DWELL_MEAN, color="green", ls=":", lw=1.5,
                  label=f"measured dwell ≈ {DWELL_MEAN}")
    ax[0].set_xlabel("dwell mean (geometric)"); ax[0].set_ylabel("time-since-switch recovery ceiling (corr)")
    ax[0].set_title("τ ceilings vs dwell (persistence knob)")
    ax[0].legend(fontsize=8); ax[0].grid(True, alpha=0.25); ax[0].set_ylim(-0.05, 1.05)

    Ts = T_GRID
    tau_info = [a["window"][str(T)]["tau_info_ceiling"] for T in Ts]
    tau_raw = [a["window"][str(T)]["tau_corr_raw_linear"] for T in Ts]
    c_raw = [a["window"][str(T)]["c_balacc_raw_linear"] for T in Ts]
    ax[1].plot(Ts, tau_info, "s-", color="#1f77b4", label="τ info ceiling (in-tile)")
    ax[1].plot(Ts, tau_raw, "v--", color="#1f77b4", alpha=0.6, label="τ raw-LINEAR ceiling")
    ax[1].plot(Ts, c_raw, "^--", color="#ff7f0e", alpha=0.8, label="c balacc raw-LINEAR (chance 0.5)")
    ax[1].axhline(abs(a["per_token_probes_on_x"]["tau_corr_on_x"]), color="#d62728",
                  ls="-", lw=1.2, label="per-token τ (probe on x)")
    ax[1].axhline(1.0, color="k", lw=0.6, alpha=0.4)
    ax[1].axhline(0.5, color="k", lw=0.6, alpha=0.3, ls=":")
    ax[1].set_xscale("log", base=2); ax[1].set_xticks(Ts); ax[1].set_xticklabels(Ts)
    ax[1].set_xlabel("tile size T"); ax[1].set_ylabel("ceiling")
    ax[1].set_title(f"Anchor (dwell {DWELL_MEAN}): info vs raw-linear access")
    ax[1].legend(fontsize=8); ax[1].grid(True, alpha=0.25); ax[1].set_ylim(-0.05, 1.05)

    fig.suptitle("Changepoint bench § 8 gating: per-token vs window ceilings, "
                 "and the provably-chance raw-linear access", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext, dpi in [("pdf", None), ("png", 120), ("thumb.png", 55)]:
        fig.savefig(FIG_DIR / f"changepoint_gating.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {FIG_DIR}/changepoint_gating.*")


if __name__ == "__main__":
    main()
