"""Permuted tones (FB-5) — T2 non-triviality battery.

The card's honesty centrepiece is the **multiset status** (FB-5.md § 3): at
T = 8 the window SET is class-informative (1,010 unique sets verified at
freeze), so a within-window shuffle is NOT a full null and a nonlinear
set-matching reader can in principle solve the task without order. The card
claims LINEAR-additive deadness (P1/P2), not order-necessity — this battery
measures the nonlinear order-free routes and REPORTS them as references:

- **shuffle semantics, measured:** the matched-filter oracle on per-window
  independently-shuffled tiles (order destroyed, multiset kept) vs intact
  tiles at T ∈ {4, 8} — quantifies how much of the ORACLE's route is order.
- **bag control:** MLP on mean-pooled raw tokens (the T2 standard; the
  additive-route nonlinear ceiling) at T = 8 — expected possibly WELL above
  chance here via set-matching; reported, not gated (the card's § 3 bag
  status), with the LINEAR bag probe alongside (P2: must be ≈ chance).
- **symmetry audit (analytic, recorded):** a symbol relabeling ψ maps
  schedule π to ψ∘π — for generic iid-uniform π_Y no relabeling maps one
  class ensemble onto another while fixing the codebook geometry (the
  circle metric breaks symbol exchangeability, as in the tone benches);
  the retained empirical control is the frequency bench's random-embedding
  null, not re-run here.
- **memorization/probe budget:** K·M = 1,010 clean templates vs d_sae ≤ 202
  at every grid cell (the frequency count, no memo-demo this cycle); probe
  rows ≥ 30k ≫ code dims (the F-rule).

    .venv/bin/python -m experiments.explorations.synthetic.permuted_tones.t2_battery

Deterministic (SEED = 0). Writes ``results/permuted_t2_stats.json``.
Gating script under LOOP.md T3 strict commit-then-run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SEED = 0
M, K = 101, 10
D_IN = 128
SIGMA = 0.10
N_SEQS = 4096

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "results" / "permuted_t2_stats.json"


def _tiles(x, labels, T, n_max, rng):
    n, s, d = x.shape
    k = s // T
    tx = x[:, : k * T, :].reshape(n * k, T, d)
    ty = labels[:, : k * T][:, ::T].reshape(n * k)
    if tx.shape[0] > n_max:
        i = rng.choice(tx.shape[0], n_max, replace=False)
        tx, ty = tx[i], ty[i]
    return tx, ty


def _shuffle_within(tx, rng):
    out = np.empty_like(tx)
    T = tx.shape[1]
    for i in range(len(tx)):
        out[i] = tx[i, rng.permutation(T)]
    return out


def main() -> None:
    import warnings
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.neural_network import MLPClassifier
    from temp_bench.data.synthetic import permuted_tones
    from temp_bench.evals.permuted_recovery import _matched_filter_pred

    data = permuted_tones(M=M, K=K, d_in=D_IN, sigma=SIGMA, seq_len=64,
                          n_seqs=N_SEQS, seed=SEED)
    x = data.x.numpy().astype(np.float64)
    lab = data.extra["schedule_labels"].numpy()
    U = data.emission_features.numpy().astype(np.float64)
    P = data.extra["schedule_table"].numpy()
    chance = 1.0 / K
    out = {"card": "freqbench/cards/FB-5.md", "seed": SEED, "chance": chance,
           "audits": {
               "memorization": "K*M = 1,010 clean window templates vs d_sae "
                               "<= 202 at every grid cell (frequency count); "
                               "schedule table re-drawn per seed => no "
                               "cross-seed template pooling; no memo-demo "
                               "cell this cycle (card § 3 P6)",
               "symmetry": "generic iid-uniform schedules: no symbol "
                           "relabeling maps class to class while fixing the "
                           "circle geometry; retained empirical control = "
                           "frequency's random-embedding null (card § 7)",
               "probe_budget": "30k probe rows >> max code dim 202 (F-rule)"},
           }

    # ── shuffle semantics: oracle on intact vs within-window-shuffled ─────
    out["shuffle"] = {}
    for T in (4, 8):
        rng = np.random.default_rng(10 + T)
        tx, ty = _tiles(x, lab, T, 8_000, rng)
        o_intact = float((_matched_filter_pred(tx, U, P, M) == ty).mean())
        o_shuf = float((_matched_filter_pred(_shuffle_within(tx, rng), U, P, M)
                        == ty).mean())
        out["shuffle"][T] = {"oracle_intact": o_intact,
                             "oracle_shuffled": o_shuf}

    # ── bag control at T=8: linear (P2) + MLP (set route, reference) ──────
    T = 8
    rng = np.random.default_rng(2)
    tx, ty = _tiles(x, lab, T, 40_000, rng)
    bag = tx.mean(axis=1)                                   # (N, d_in)
    half = len(bag) // 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lin = LogisticRegression(max_iter=300).fit(bag[:half], ty[:half])
        bag_lin = float(balanced_accuracy_score(ty[half:],
                                                lin.predict(bag[half:])))
        mlp = MLPClassifier(hidden_layer_sizes=(256,), max_iter=200,
                            random_state=0).fit(bag[:half], ty[:half])
        bag_mlp = float(balanced_accuracy_score(ty[half:],
                                                mlp.predict(bag[half:])))
    out["bag"] = {"linear_balacc": bag_lin, "mlp_balacc": bag_mlp,
                  "note": "MLP-on-bag is the order-free set-matching "
                          "reference (card § 3: NOT a kill — the card claims "
                          "linear-additive deadness, not order-necessity); "
                          "linear-on-bag is P2's claim and must sit ≈ chance"}

    # ── verdict flags (P2-linear is the gate; nonlinear routes reported) ──
    out["verdict"] = {
        "bag_linear_at_chance": bag_lin - chance <= 0.05,
        "bag_mlp_reference": bag_mlp,
        "shuffle_degrades_oracle": all(
            s["oracle_shuffled"] < s["oracle_intact"] - 0.1
            for s in out["shuffle"].values()),
        "passes_t2": bool(bag_lin - chance <= 0.05),
    }
    print(json.dumps(out["shuffle"], indent=1), flush=True)
    print(json.dumps(out["bag"], indent=1), flush=True)
    print(json.dumps(out["verdict"], indent=1), flush=True)
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
