"""Colored sources (FB-3) — T1 proof-gate numerics + § 8 STOP-gate.

Runs on the ACTUAL built generator (``colored_sources``) at the frozen card
parameters (freqbench/cards/FB-3.md). Feature-direction-recovery variant of
the discriminability STOP-gate:

**T1 discharge (numerical, the verify_theory pattern):**

- **CS-2 ceiling.** Eigenvectors of the symmetrized empirical lag-D
  covariance recover F at the card's exact data budget: full-sequence
  estimator + the **W-resolved** estimator (only within-window pairs of
  non-overlapping W-tiles) — the W = D+1 phase transition curve.
- **CS-1 floor.** The W ≤ D estimators have NO lag-D pairs; the best
  within-window statistic is Ĉ_0, whose eigenvectors are F-blind (the
  marginal is isotropic) ⇒ recovery inside the random-dictionary chance
  band. Also checked: marginal isotropy and C_ℓ ≈ 0 for 0 < ℓ < D.
- **Dilution note (bag/shuffle).** Mean-pooled windows and within-window
  shuffles do NOT null this bench: pooled/shuffled second moments retain a
  known positive multiple of C_D (quantified here). The bench's true null
  is window TRUNCATION (W ≤ D) — recorded as a dated precision-amendment to
  the card's § 3 bag line (the floor claim CS-1 is untouched).

**§ 8 STOP-gate (feature-recovery variant):** the bench is discriminating
only if (i) the W-resolved oracle separates W ≤ D (chance band) from
W ≥ D+1 (≥ GATE_ORACLE) decisively at our budget, and (ii) the chance band
is tight. If the oracle cannot clear the bar the task is sample-starved at
this budget (the sprint's N=128 failure): NON-DISCRIMINATING, no grid.

    .venv/bin/python -m experiments.explorations.synthetic.colored_sources.gating

Deterministic (SEED = 0). Writes results/colored_gating_stats.json + figure.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SEED = 0

# ── frozen card parameters (freqbench/cards/FB-3.md § 2) ──────────────────
N = 32
D_IN = 32
LAG_D = 2
SIGMA = 0.1
RHO_MIN, RHO_MAX = 0.1, 0.9
SEQ_LEN = 64
N_SEQS = 4096
W_GRID = [1, 2, 4, 8]            # the locked design's T range
N_CHANCE_DRAWS = 32

# gates
GATE_ORACLE = 0.75               # full + W≥D+1 oracle rec_adj
# Floor bar (documented amendment, 2026-07-22, pre-skeptic/pre-grid): the
# first-pass check compared eigen-estimators against an iid-GAUSSIAN
# candidate null — the wrong null (an eigenbasis is orthonormal, which
# scores higher by geometry alone: 0.181 vs 0.170 at N=32). Fixed to the
# orthonormal null; the diagnostic also measured a small SYSTEMATIC
# stream-leakage of the marginal estimator on the actual (temporally
# correlated) stream: +0.011 rec_sq over 8 generator seeds (CS-1 strictly
# assumes iid draws; correlated samples tilt Ĉ0's fluctuations toward
# high-ρ sources). Both are recorded below. The operational floor bar is
# ABSOLUTE on the normalized scale: |rec_adj| ≤ 0.05 — 5 % of the scale,
# ~20× below the measured 0.96 ceiling — which is what "W ≤ D readers
# cannot meaningfully recover F" requires for discriminability.
GATE_FLOOR_EPS = 0.05
GATE_MARGINAL_OFFDIAG = 0.05
N_LEAKAGE_SEEDS = 8

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "results" / "colored_gating_stats.json"
FIG_DIR = HERE / "figs"


def rec_sq(cand: np.ndarray, F: np.ndarray) -> float:
    C = cand / np.linalg.norm(cand, axis=1, keepdims=True).clip(1e-8)
    cos2 = (C @ F.T) ** 2
    return float(cos2.max(axis=0).mean())


def eig_rows(Csym: np.ndarray, n: int) -> np.ndarray:
    w, V = np.linalg.eigh(Csym)
    order = np.argsort(-np.abs(w))
    return V[:, order[:n]].T


def chance_band(n_cand: int, F: np.ndarray, rng, *, mode: str = "gauss",
                ) -> tuple[float, float]:
    """Chance band for F-blind candidates. ``gauss`` = iid unit directions
    (the bench metric's reference — trained atoms are free vectors);
    ``orthonormal`` = random orthonormal bases (the correct null for
    eigen-estimators, whose output is an orthonormal set)."""
    vals = []
    for _ in range(N_CHANCE_DRAWS):
        if mode == "orthonormal":
            Q, _ = np.linalg.qr(rng.standard_normal((F.shape[1], F.shape[1])))
            vals.append(rec_sq(Q.T[:n_cand], F))
        else:
            vals.append(rec_sq(rng.standard_normal((n_cand, F.shape[1])), F))
    return float(np.mean(vals)), float(np.std(vals))


def main() -> None:
    from temp_bench.data.synthetic import colored_sources

    rng = np.random.default_rng(SEED)
    data = colored_sources(N=N, d_in=D_IN, D=LAG_D, sigma=SIGMA,
                           rho_min=RHO_MIN, rho_max=RHO_MAX,
                           seq_len=SEQ_LEN, n_seqs=N_SEQS, seed=SEED)
    x = data.x.numpy().astype(np.float64)
    F = data.emission_features.numpy().astype(np.float64)
    rho = data.extra["rho_schedule"].numpy()

    out: dict = {"card": "freqbench/cards/FB-3.md",
                 "params": {"N": N, "d_in": D_IN, "D": LAG_D, "sigma": SIGMA,
                            "rho": [RHO_MIN, RHO_MAX], "seq_len": SEQ_LEN,
                            "n_seqs": N_SEQS, "W_grid": W_GRID, "seed": SEED,
                            "eigengap": float(np.min(np.diff(np.sort(rho))))},
                 "gates": {"oracle": GATE_ORACLE,
                           "floor_eps": GATE_FLOOR_EPS,
                           "marginal_offdiag": GATE_MARGINAL_OFFDIAG}}

    chance, chance_sd = chance_band(N, F, rng, mode="gauss")
    chance_o, chance_o_sd = chance_band(N, F, rng, mode="orthonormal")
    out["chance"] = {"gauss_mean": round(chance, 4), "gauss_std": round(chance_sd, 5),
                     "orthonormal_mean": round(chance_o, 4),
                     "orthonormal_std": round(chance_o_sd, 5), "n_cand": N}
    denom = 1.0 - chance
    denom_o = 1.0 - chance_o

    def adj(r):
        return (r - chance) / denom

    def adj_o(r):
        """Chance-adjust against the ORTHONORMAL null — the correct reference
        for eigen-estimators (their candidate set is an orthonormal basis)."""
        return (r - chance_o) / denom_o

    # ── CS-1 premises on the built data ──
    xf = x.reshape(-1, D_IN)
    C0 = (xf.T @ xf) / len(xf)
    offdiag_max = float(np.abs(C0 - np.diag(np.diag(C0))).max())
    C1 = np.einsum("ntd,nte->de", x[:, 1:], x[:, :-1]) / (N_SEQS * (SEQ_LEN - 1))
    c0_rec = rec_sq(eig_rows(0.5 * (C0 + C0.T), N), F)
    out["cs1"] = {
        "marginal_diag_dev": round(float(np.abs(np.diag(C0) - (1 + SIGMA**2)).max()), 4),
        "marginal_offdiag_max": round(offdiag_max, 4),
        "lag1_cov_max": round(float(np.abs(C1).max()), 4),
        "c0_eig_recovery_adj_orth": round(adj_o(c0_rec), 4),
    }
    print(f"[cs1] marginal offdiag {offdiag_max:.4f}  lag-1 max "
          f"{out['cs1']['lag1_cov_max']:.4f}  C0-eig rec_adj(orth null) "
          f"{out['cs1']['c0_eig_recovery_adj_orth']:+.3f}", flush=True)

    # ── measured stream leakage of the marginal estimator (multi-seed) ──
    # CS-1 assumes iid draws; the actual stream is temporally correlated, and
    # Ĉ0's finite-sample fluctuations tilt toward high-ρ sources. Measured
    # honestly here; must stay far below the discriminability scale.
    leak = []
    for s in range(N_LEAKAGE_SEEDS):
        d_s = colored_sources(N=N, d_in=D_IN, D=LAG_D, sigma=SIGMA,
                              rho_min=RHO_MIN, rho_max=RHO_MAX,
                              seq_len=SEQ_LEN, n_seqs=N_SEQS, seed=100 + s)
        xs = d_s.x.numpy().astype(np.float64).reshape(-1, D_IN)
        Cs0 = (xs.T @ xs) / len(xs)
        leak.append(rec_sq(eig_rows(0.5 * (Cs0 + Cs0.T), N),
                           d_s.emission_features.numpy().astype(np.float64)))
    out["stream_leakage"] = {
        "c0_eig_rec_sq_mean": round(float(np.mean(leak)), 4),
        "c0_eig_rec_sq_std": round(float(np.std(leak)), 4),
        "vs_orthonormal_null": round(float(np.mean(leak)) - chance_o, 4),
        "n_seeds": N_LEAKAGE_SEEDS,
    }
    print(f"[leakage] C0-eig over {N_LEAKAGE_SEEDS} seeds: "
          f"{np.mean(leak):.4f} (null {chance_o:.4f}; excess "
          f"{np.mean(leak) - chance_o:+.4f})", flush=True)

    # ── CS-2 ceiling: full-sequence lag-D estimator ──
    CD = np.einsum("ntd,nte->de", x[:, LAG_D:], x[:, :-LAG_D]) \
        / (N_SEQS * (SEQ_LEN - LAG_D))
    full_rec = rec_sq(eig_rows(0.5 * (CD + CD.T), N), F)
    out["cs2_full"] = {"rec_sq": round(full_rec, 4),
                       "rec_adj": round(adj(full_rec), 4)}
    print(f"[cs2] full-sequence lag-D oracle rec_adj {adj(full_rec):+.3f}",
          flush=True)

    # ── W-resolved oracle: only within-window pairs of non-overlapping tiles ──
    out["by_W"] = {}
    for W in W_GRID:
        k = SEQ_LEN // W
        tiles = x[:, : k * W].reshape(N_SEQS * k, W, D_IN)
        if W > LAG_D:
            n_pairs = (W - LAG_D) * len(tiles)
            Cw = np.einsum("ntd,nte->de", tiles[:, LAG_D:], tiles[:, :-LAG_D]) \
                / n_pairs
            r = rec_sq(eig_rows(0.5 * (Cw + Cw.T), N), F)
            est = "lag_D_within_tiles"
        else:
            # No lag-D pairs exist in-window; best within-window statistic is
            # Ĉ_0 (isotropic ⇒ F-blind) — the CS-1 floor, estimated honestly.
            tf = tiles.reshape(-1, D_IN)
            Cw = (tf.T @ tf) / len(tf)
            r = rec_sq(eig_rows(0.5 * (Cw + Cw.T), N), F)
            est = "c0_within_tiles(floor)"
        out["by_W"][W] = {"rec_sq": round(r, 4), "rec_adj": round(adj(r), 4),
                          "rec_adj_orth": round(adj_o(r), 4),
                          "estimator": est}
        print(f"[W={W}] {est:26s} rec_adj {adj(r):+.3f} "
              f"(orth null {adj_o(r):+.3f})", flush=True)

    # ── dilution note: pooled + shuffled second moments retain C_D ──
    W = 4
    k = SEQ_LEN // W
    tiles = x[:, : k * W].reshape(N_SEQS * k, W, D_IN)
    pooled = tiles.mean(axis=1)
    Cp = (pooled.T @ pooled) / len(pooled)
    pooled_rec = rec_sq(eig_rows(0.5 * (Cp + Cp.T), N), F)
    perm_rng = np.random.default_rng(SEED + 7)
    idx = np.argsort(perm_rng.random((len(tiles), W)), axis=1)
    shuf = np.take_along_axis(tiles, idx[..., None], axis=1)
    n_pairs = (W - LAG_D) * len(shuf)
    Csh = np.einsum("ntd,nte->de", shuf[:, LAG_D:], shuf[:, :-LAG_D]) / n_pairs
    shuf_rec = rec_sq(eig_rows(0.5 * (Csh + Csh.T), N), F)
    out["dilution"] = {
        "pooled_eig_rec_adj_W4": round(adj(pooled_rec), 4),
        "shuffled_lagD_eig_rec_adj_W4": round(adj(shuf_rec), 4),
        "note": ("pooling/shuffling DILUTES but does not destroy C_D — the "
                 "bench's true null is window truncation (W <= D), not "
                 "permutation. Dated precision-amendment to card § 3 bag "
                 "line; CS-1 floor untouched."),
    }
    print(f"[dilution] pooled rec_adj {adj(pooled_rec):+.3f}  "
          f"shuffled lag-D rec_adj {adj(shuf_rec):+.3f}", flush=True)

    # ── verdict ──
    floor_ok = all(abs(out["by_W"][W]["rec_adj_orth"]) <= GATE_FLOOR_EPS
                   for W in W_GRID if W <= LAG_D)
    ceil_ok = (out["cs2_full"]["rec_adj"] >= GATE_ORACLE
               and all(out["by_W"][W]["rec_adj"] >= GATE_ORACLE
                       for W in W_GRID if W > LAG_D))
    checks = {
        "cs1_marginal_isotropic": bool(
            out["cs1"]["marginal_offdiag_max"] <= GATE_MARGINAL_OFFDIAG),
        "cs1_short_lag_blank": bool(
            out["cs1"]["lag1_cov_max"] <= GATE_MARGINAL_OFFDIAG),
        "cs1_c0_estimator_at_floor": bool(
            abs(out["cs1"]["c0_eig_recovery_adj_orth"]) <= GATE_FLOOR_EPS),
        "floor_W_le_D_at_chance": bool(floor_ok),
        "stream_leakage_small": bool(
            abs(out["stream_leakage"]["vs_orthonormal_null"]) / denom_o
            <= GATE_FLOOR_EPS),
        "oracle_W_gt_D_clears_bar": bool(ceil_ok),
        "transition_sharp": bool(
            out["by_W"][4]["rec_adj"] - out["by_W"][2]["rec_adj"] >= 0.5),
    }
    out["verdict"] = {"checks": checks,
                      "passes_gate": bool(all(checks.values()))}
    print(json.dumps(out["verdict"], indent=1), flush=True)

    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT_JSON}", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    Ws = W_GRID
    ax.plot(Ws, [out["by_W"][W]["rec_adj"] for W in Ws], "o-",
            label="W-resolved oracle")
    ax.axhline(out["cs2_full"]["rec_adj"], color="tab:green", ls="--", lw=1,
               label="full-sequence oracle")
    ax.axhline(0, color="gray", ls=":", lw=1, label="chance (random dict)")
    ax.axvline(LAG_D + 0.5, color="tab:red", ls=":", lw=1,
               label=f"W = D+1 = {LAG_D + 1}")
    ax.set(xlabel="window length W", ylabel="rec_adj",
           title="FB-3 lag-D eigen-oracle: the W = D+1 transition")
    ax.legend(fontsize=7)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "colored_gating.png", dpi=160)
    print(f"wrote {FIG_DIR / 'colored_gating.png'}", flush=True)


if __name__ == "__main__":
    main()
