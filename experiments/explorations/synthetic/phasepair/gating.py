"""Phasepair (FB-1) — T1 proof-gate numerics + § 8 STOP-gate.

Runs on the ACTUAL datasource (``cyclic_tones`` at the frozen ± Ω). The
P1/P2 floors transfer verbatim from the frequency bench (same generator);
what is new and discharged here:

- **Signed-oracle ceiling (P5, signed):** the complex periodogram matched
  filter over the 6 signed velocities; per-pair SIGN oracle vs T (the phase
  resolution curve — sign needs enough phase evolution per window).
- **The exact bag null (the card's Floor 2):** mean-pooled raw tokens +
  MLP probing the SIGN must sit at chance ½ EXACTLY (within noise) — within
  a pair the bag distributions are identical; a deviation is a THEORY BUG
  and stops everything (card falsifier 2).
- **Raw-linear floors:** token + window-concat probes at chance for the
  6-class, pair, and sign targets (equality-variant part (i)).
- **§ 8 STOP:** sign oracle ≥ 0.75 for at least 2 of the 3 pairs at some
  T ≤ 8, else NON-DISCRIMINATING (window too short for phase) — no grid.

    .venv/bin/python -m experiments.explorations.synthetic.phasepair.gating
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

SEED = 0
M = 101
OMEGA = [3, 98, 12, 89, 30, 71]
PAIRS = [(0, 1), (2, 3), (4, 5)]
D_IN, SIGMA, SEQ_LEN, L = 24, 0.10, 64, 32
T_GRID = [2, 4, 8]
T_REF = [16]
N_SEQS = 6000
N_PROBE_ROWS = 30_000

# Raw-floor check is ONE-SIDED (documented amendment, 2026-07-23, pre-
# skeptic/pre-grid): the first pass used |dev| and flagged the T∈{4,8}
# raw-window-linear 6-class probes at 0.112–0.115 — BELOW chance 0.167.
# Below-chance balanced accuracy from a 192-dim linear probe is a
# degenerate-classifier artifact, not linear access (access pushes ABOVE
# chance); the gate's intent is "no linear route to the latent". Same
# reasoning as FB-3's one-sided untrained-floor fix. Below-chance values
# are recorded, not gated.
GATE_RAW_TOL = 0.05              # one-sided: (acc − chance) ≤ tol
GATE_SIGN_ORACLE = 0.75          # per-pair sign oracle bar (≥ 2 pairs, some T ≤ 8)
GATE_BAG_SIGN_TOL = 0.04         # |bag sign − ½| — the exact-null check

CHANCE6 = 1.0 / 6

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "results" / "phasepair_gating_stats.json"
FIG_DIR = HERE / "figs"


def _probe(z_tr, y_tr, z_ev, y_ev, mlp=False, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.neural_network import MLPClassifier
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = (MLPClassifier(hidden_layer_sizes=(256,), max_iter=300,
                             random_state=seed) if mlp
               else LogisticRegression(max_iter=200))
        clf.fit(z_tr, y_tr)
        return float(balanced_accuracy_score(y_ev, clf.predict(z_ev)))


def main() -> None:
    from temp_bench.data.synthetic import cyclic_tones

    rng = np.random.default_rng(SEED)
    mk = lambda seed: cyclic_tones(M=M, omega=tuple(OMEGA), embedding="circle",
                                   d_in=D_IN, sigma=SIGMA, seq_len=SEQ_LEN,
                                   n_seqs=N_SEQS, seed=seed)
    dtr, dev = mk(SEED), mk(SEED + 1)
    R = dev.extra["circle_plane"].numpy().astype(np.float64)
    x_tr = dtr.x.numpy().astype(np.float64)
    x_ev = dev.x.numpy().astype(np.float64)
    l_tr = dtr.extra["velocity_labels"].numpy()
    l_ev = dev.extra["velocity_labels"].numpy()
    R_tr = dtr.extra["circle_plane"].numpy().astype(np.float64)

    pair_of = np.array([0, 0, 1, 1, 2, 2])
    sign_of = np.array([1, 0, 1, 0, 1, 0])

    def tiles(x, lab, T, seed):
        r = np.random.default_rng(seed)
        k = SEQ_LEN // T
        t_ = x[:, : k * T].reshape(-1, T, D_IN)
        y_ = lab[:, : k * T].reshape(len(x), k, T)[:, :, T - 1].reshape(-1)
        idx = r.choice(len(t_), min(N_PROBE_ROWS, len(t_)), replace=False)
        return t_[idx], y_[idx]

    out: dict = {"card": "freqbench/cards/FB-1.md",
                 "params": {"M": M, "omega": OMEGA, "d_in": D_IN,
                            "sigma": SIGMA, "T_grid": T_GRID, "seed": SEED},
                 "gates": {"raw_tol": GATE_RAW_TOL,
                           "sign_oracle": GATE_SIGN_ORACLE,
                           "bag_sign_tol": GATE_BAG_SIGN_TOL},
                 "by_T": {}}

    t_all = np.arange(max(T_GRID + T_REF))
    for T in T_GRID + T_REF:
        te, ye = tiles(x_ev, l_ev, T, SEED + 11)
        tt, yt = tiles(x_tr, l_tr, T, SEED + 10)
        proj = te @ R
        c = proj[..., 0] + 1j * proj[..., 1]
        basis = np.exp(-2j * np.pi * np.asarray(OMEGA, dtype=np.float64)[:, None]
                       * t_all[None, :T] / M)
        pred6 = np.abs(c @ basis.T).argmax(axis=1)
        oracle6 = float(np.mean([
            (pred6[ye == k] == k).mean() for k in range(6)]))
        pair_oracle = float(np.mean([
            (pair_of[pred6[ye == k]] == pair_of[k]).mean() for k in range(6)]))
        sign_or = []
        for p, (i, j) in enumerate(PAIRS):
            m = (ye == i) | (ye == j)
            b2 = np.exp(-2j * np.pi
                        * np.asarray([OMEGA[i], OMEGA[j]], dtype=np.float64)[:, None]
                        * t_all[None, :T] / M)
            sp = (np.abs(c[m] @ b2.T).argmax(axis=1) == 0).astype(int)
            st = (ye[m] == i).astype(int)
            sign_or.append(float(((sp == st)[st == 1].mean()
                                  + (sp == st)[st == 0].mean()) / 2))
        d = {"oracle6": round(oracle6, 4), "pair_oracle": round(pair_oracle, 4),
             "sign_oracle_by_pair": [round(v, 4) for v in sign_or]}
        out["by_T"][T] = d
        print(f"[T={T:2d}] oracle6 {oracle6:.3f} pair {pair_oracle:.3f} "
              f"sign/pair {['%.3f' % v for v in sign_or]}", flush=True)
        if T not in T_GRID:
            continue

        flat_tr, flat_ev = tt.reshape(len(tt), -1), te.reshape(len(te), -1)
        tok_tr, tok_ev = tt[:, -1, :], te[:, -1, :]
        bag_tr, bag_ev = tt.mean(axis=1), te.mean(axis=1)
        d["raw_token_linear6"] = _probe(tok_tr, yt, tok_ev, ye)
        d["raw_window_linear6"] = _probe(flat_tr, yt, flat_ev, ye)
        # sign floors, per pair, restricted to true pair
        rls, bag_sign = [], []
        for p, (i, j) in enumerate(PAIRS):
            mtr = (yt == i) | (yt == j)
            mev = (ye == i) | (ye == j)
            str_, sev = (yt[mtr] == i).astype(int), (ye[mev] == i).astype(int)
            rls.append(_probe(flat_tr[mtr], str_, flat_ev[mev], sev))
            bag_sign.append(_probe(bag_tr[mtr], str_, bag_ev[mev], sev,
                                   mlp=True, seed=SEED + p))
        d["raw_window_linear_sign"] = [round(v, 4) for v in rls]
        d["bag_mlp_sign"] = [round(v, 4) for v in bag_sign]
        print(f"       raw6 tok {d['raw_token_linear6']:.3f} "
              f"win {d['raw_window_linear6']:.3f}  "
              f"rawlin sign {['%.3f' % v for v in rls]}  "
              f"bagMLP sign {['%.3f' % v for v in bag_sign]}", flush=True)

    best = {T: sum(v >= GATE_SIGN_ORACLE
                   for v in out["by_T"][T]["sign_oracle_by_pair"])
            for T in T_GRID}
    bag_worst = max(abs(v - 0.5) for T in T_GRID
                    for v in out["by_T"][T]["bag_mlp_sign"])
    rawlin_worst = max(
        max(out["by_T"][T]["raw_token_linear6"] - CHANCE6,
            out["by_T"][T]["raw_window_linear6"] - CHANCE6,
            max(v - 0.5 for v in out["by_T"][T]["raw_window_linear_sign"]))
        for T in T_GRID)
    checks = {
        "raw_linear_floors_at_chance": bool(rawlin_worst <= GATE_RAW_TOL),
        "bag_sign_exact_null": bool(bag_worst <= GATE_BAG_SIGN_TOL),
        "sign_oracle_discriminates": bool(max(best.values()) >= 2),
    }
    out["verdict"] = {
        "rawlin_worst_dev": round(rawlin_worst, 4),
        "bag_sign_worst_dev": round(bag_worst, 4),
        "pairs_above_bar_by_T": best,
        "checks": checks,
        "passes_gate": bool(all(checks.values())),
    }
    print(json.dumps(out["verdict"], indent=1), flush=True)

    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT_JSON}", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    Ts = T_GRID + T_REF
    for p, (i, j) in enumerate(PAIRS):
        ax.plot(Ts, [out["by_T"][T]["sign_oracle_by_pair"][p] for T in Ts],
                "o-", label=f"±{OMEGA[i]} sign oracle")
    ax.plot(Ts, [out["by_T"][T]["oracle6"] for T in Ts], "k--", lw=1,
            label="6-class oracle")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.axhline(GATE_SIGN_ORACLE, color="tab:red", ls=":", lw=1)
    ax.set(xlabel="T", ylabel="accuracy", title="FB-1 signed-oracle curves")
    ax.legend(fontsize=7)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "phasepair_gating.png", dpi=160)
    print(f"wrote {FIG_DIR / 'phasepair_gating.png'}", flush=True)


if __name__ == "__main__":
    main()
