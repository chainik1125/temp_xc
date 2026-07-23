"""Rotated multilane (FB-4) — T2 non-triviality battery: the absorption audit.

The card's decisive gate (freqbench/cards/FB-4.md § 3, the absorption
obligation). FB-2's embedding isometry ``P`` is Haar-random and re-drawn per
data seed, so composing with any fixed ``Q ∈ O(d_in)`` gives ``QP`` with the
SAME distribution — the analytic claim is that FB-4 is a distribution-replica
of FB-2 and the rotation knob is inert. LOOP.md T2's symmetry audit is exactly
this question ("is there a group action … the task measures geometry you
didn't build"). This battery adjudicates it *empirically*, in two arms:

**Arm A — basis-dependent statistics, two-sample across seed ensembles**
(seeds 0..7, no training): per-coordinate kurtosis, per-channel DCT high-band
energy fraction (T=8), and lane-plane alignment with the coordinate basis.
Each is exactly the kind of statistic a live spatial-alignment knob would
move; under absorption their FB-4 and FB-2 seed-ensembles are exchangeable.
Permutation two-sample test (mean diff, 2000 perms) per statistic,
pre-registered α = 0.05 per statistic.

**Arm B — the arch panel at the FB-2 anchor cell, canonical runner**
(T=8, d_sae=101; seeds {1,2,42} on the rotated datasource):
``spectral_txc`` untrained k=1 (the frozen "+0.298 → ≈ 0 collapse" direction's
direct test), ``spectral_txc`` trained k=2, ``txc_batchtopk_post`` trained
k=2. Compared against FB-2's recorded grid values at the same cells; "inside
band" = |mean diff| ≤ max(2 × FB-2 seed range, 0.03).

Also runs the T2 bag control on the rotated data (mean-pooled raw tokens +
MLP at the T=8 tiling — must sit far below the oracle; order route required),
and states the inherited controls (memorization budget: |Ω|³M³ is basis-
independent, card § 5; shuffle semantics: unchanged from FB-2).

**Decision rule (pre-registered here, committed before first execution):**
- falsifier (gating.py t1_recovery > 0.1) ⇒ BUG — stop, debug, never report.
- Arm A no separation AND Arm B inside bands ⇒ **ABORT_T2_SYMMETRY**
  (knob inert: redundant-by-symmetry with FB-2; b_triviality/d_redundancy).
- Untrained collapse (FB-4 untrained mean ≤ 0.05) or any Arm B cell outside
  band in the collapse direction ⇒ **KNOB_LIVE_PROCEED_S8** (the absorption
  argument is wrong; continue per the card).
- Anything mixed ⇒ **STOP_FOR_REVIEW** (LOOP: when in doubt, stop).

    .venv/bin/python -m experiments.explorations.synthetic.rotated_multilane.t2_battery [max_workers]

Writes ``results/rotated_multilane_t2_stats.json``. Gating script under
LOOP.md T3 strict commit-then-run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

M = 101
OMEGA = (0, 1, 2, 4, 8, 16, 24, 32, 40, 50)
D_IN = 24
SIGMA = 0.25
KW = dict(M=M, omega=OMEGA, n_lanes=3, d_in=D_IN, sigma=SIGMA,
          seq_len=64, n_seqs=2048)
ENSEMBLE_SEEDS = list(range(8))
PANEL_SEEDS = (1, 2, 42)
DS_ROT = "toy_multilane_rotated_M101_d24"
ALPHA = 0.05
BAND_TOL_FLOOR = 0.03
COLLAPSE_BAR = 0.05                 # frozen direction: untrained ≈ 0

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "results" / "rotated_multilane_t2_stats.json"
FB2_GRID = HERE.parent / "multilane" / "results" / "multilane_grid_results.json"
GATING_JSON = HERE / "results" / "rotated_multilane_gating_stats.json"


# ── Arm A: basis-dependent statistics ────────────────────────────────────


def _stats_one(data) -> dict[str, float]:
    x = data.x.numpy().astype(np.float64)
    n, s, d = x.shape
    flat = x.reshape(-1, d)
    mu, sd = flat.mean(0), flat.std(0)
    kurt = float((((flat - mu) / sd) ** 4).mean())
    T = 8
    tiles = x[:, : (s // T) * T, :].reshape(-1, T, d)
    psi = np.zeros((T, T))
    tau = np.arange(T)
    for w in range(T):
        psi[w] = (np.sqrt(1 / T) if w == 0 else
                  np.sqrt(2 / T) * np.cos(np.pi * (tau + 0.5) * w / T))
    coef = np.einsum("wt,ntd->nwd", psi, tiles)
    energy = (coef ** 2).mean(axis=(0, 2))
    high_frac = float(energy[5:].sum() / energy.sum())
    planes = data.extra["lane_planes"].numpy().astype(np.float64)
    axes = np.concatenate([planes[k] for k in range(planes.shape[0])], axis=1)
    align = float(np.abs(axes).max(axis=0).mean())
    return {"coord_kurtosis": kurt, "dct_high_frac": high_frac,
            "plane_coord_align": align}


def _perm_p(a: np.ndarray, b: np.ndarray, n_perm: int = 2000, seed: int = 0):
    obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if abs(pool[: len(a)].mean() - pool[len(a):].mean()) >= obs - 1e-15:
            cnt += 1
    return float(obs), float(cnt / n_perm)


def arm_a() -> dict:
    from temp_bench.data.synthetic import multilane_tones, multilane_tones_rotated
    rows_b, rows_r = [], []
    for s in ENSEMBLE_SEEDS:
        rows_b.append(_stats_one(multilane_tones(seed=s, **KW)))
        rows_r.append(_stats_one(multilane_tones_rotated(seed=s, **KW)))
    out = {"seeds": ENSEMBLE_SEEDS, "base": rows_b, "rotated": rows_r,
           "tests": {}}
    separated = False
    for ki, key in enumerate(sorted(rows_b[0])):
        a = np.array([r[key] for r in rows_r])
        b = np.array([r[key] for r in rows_b])
        obs, p = _perm_p(a, b, seed=1000 + ki)
        out["tests"][key] = {"abs_mean_diff": obs, "perm_p": p}
        if p < ALPHA:
            separated = True
    out["separated"] = separated
    return out


# ── Arm B: canonical-runner panel at the anchor cell ─────────────────────

PANEL = (
    {"arch": "spectral_txc", "k_pos": 1, "n_steps": 0, "kind": "untrained"},
    {"arch": "spectral_txc", "k_pos": 2, "n_steps": 30_000, "kind": "trained"},
    {"arch": "txc_batchtopk_post", "k_pos": 2, "n_steps": 30_000,
     "kind": "trained"},
)


def _fb2_reference(arch: str, k_pos: int, untrained: bool) -> list[float]:
    res = json.loads(FB2_GRID.read_text())
    vals = {}
    for r in res:
        if (r.get("ok") and r["arch"] == arch and r["T"] == 8
                and r["d_sae"] == 101 and r["k_pos"] == k_pos
                and (r["n_steps"] == 0) == untrained
                and r["seed"] in PANEL_SEEDS):
            vals[r["seed"]] = r["metrics"]["multilane_recovery"]
    return [vals[s] for s in sorted(vals)]


def arm_b(max_workers: int) -> dict:
    from explorations.synthetic import grid
    cells = [{"ds": DS_ROT, "arch": p["arch"], "T": 8, "d_sae": 101,
              "k_pos": p["k_pos"], "seed": s, "n_steps": p["n_steps"],
              "kind": f"fb4_t2_{p['kind']}", "eval_window_L": 32}
             for p in PANEL for s in PANEL_SEEDS]
    results = grid.run_pool(cells, HERE / "results" / "t2_panel_cells.json",
                            max_workers=max_workers, tag="fb4-t2")
    out = {"cells": [], "outside_band": [], "untrained_mean": None}
    for p in PANEL:
        got = sorted((r["seed"], r["metrics"]["multilane_recovery"])
                     for r in results
                     if r.get("ok") and r["arch"] == p["arch"]
                     and r["k_pos"] == p["k_pos"]
                     and (r["n_steps"] == 0) == (p["n_steps"] == 0))
        vals = [v for _, v in got]
        ref = _fb2_reference(p["arch"], p["k_pos"], p["n_steps"] == 0)
        tol = max(2 * (max(ref) - min(ref)), BAND_TOL_FLOOR) if ref else None
        inside = (ref and vals
                  and abs(np.mean(vals) - np.mean(ref)) <= tol)
        row = {"arch": p["arch"], "k_pos": p["k_pos"], "kind": p["kind"],
               "fb4_vals": vals, "fb2_ref": ref, "band_tol": tol,
               "inside_band": bool(inside)}
        out["cells"].append(row)
        if not inside:
            out["outside_band"].append(f"{p['arch']}/k{p['k_pos']}/{p['kind']}")
        if p["kind"] == "untrained" and p["arch"] == "spectral_txc":
            out["untrained_mean"] = float(np.mean(vals)) if vals else None
    return out


# ── bag control on the rotated data ──────────────────────────────────────


def bag_control() -> dict:
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.neural_network import MLPClassifier
    from temp_bench.data.synthetic import multilane_tones_rotated
    data = multilane_tones_rotated(seed=0, **{**KW, "n_seqs": 4096})
    x = data.x.numpy().astype(np.float64)
    lab = data.extra["lane_velocity_labels"].numpy()
    T = 8
    n, s, d = x.shape
    k = s // T
    tiles = x[:, : k * T, :].reshape(n * k, T, d).mean(axis=1)   # bag: order gone
    ty = lab[:, : k * T][:, ::T].reshape(n * k, 3)
    half = len(tiles) // 2
    rng = np.random.default_rng(0)
    i = rng.permutation(len(tiles))
    tr, ev = i[:half][:20_000], i[half:][:20_000]
    accs = []
    for lane in range(3):
        clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=150,
                            random_state=0).fit(tiles[tr], ty[tr, lane])
        accs.append(float(balanced_accuracy_score(ty[ev, lane],
                                                  clf.predict(tiles[ev]))))
    return {"bag_mlp_balacc": accs, "chance": 0.1}


def main() -> None:
    max_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    out = {"card": "freqbench/cards/FB-4.md",
           "inherited": {
               "memorization_budget": "card § 5: |Ω|³M³ ≈ 1.03e9 templates is "
                                      "basis-independent; FB-2 audit inherited",
               "shuffle_semantics": "unchanged from FB-2 (per-window "
                                    "independent permutations kill the phase "
                                    "progression; not a full null for power)"},
           }
    gating = json.loads(GATING_JSON.read_text()) if GATING_JSON.exists() else None
    falsifier = bool(gating and gating["verdict"]["falsifier_t1_fired"])
    out["falsifier_t1_fired"] = falsifier

    print("[fb4-t2] arm A: seed-ensemble statistics", flush=True)
    out["arm_a"] = arm_a()
    print(json.dumps(out["arm_a"]["tests"], indent=1), flush=True)

    print("[fb4-t2] bag control", flush=True)
    out["bag"] = bag_control()
    print(json.dumps(out["bag"], indent=1), flush=True)

    print("[fb4-t2] arm B: canonical panel", flush=True)
    out["arm_b"] = arm_b(max_workers)

    a_sep = out["arm_a"]["separated"]
    b_out = out["arm_b"]["outside_band"]
    um = out["arm_b"]["untrained_mean"]
    collapsed = um is not None and um <= COLLAPSE_BAR
    if falsifier:
        verdict = "BUG_STOP"
    elif (not a_sep) and (not b_out):
        verdict = "ABORT_T2_SYMMETRY"
    elif collapsed or b_out:
        verdict = "KNOB_LIVE_PROCEED_S8" if collapsed else "STOP_FOR_REVIEW"
    else:
        verdict = "STOP_FOR_REVIEW"
    out["verdict"] = {
        "arm_a_separated": a_sep,
        "arm_b_outside_band": b_out,
        "untrained_spectral_mean": um,
        "untrained_collapse_direction_confirmed": collapsed,
        "verdict": verdict,
    }
    print(json.dumps(out["verdict"], indent=1), flush=True)
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
