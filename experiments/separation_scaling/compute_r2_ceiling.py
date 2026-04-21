"""Compute the Bayes-optimal R² ceiling per δ.

For the same 5 mess3_shared cells as separation_scaling, run the forward filter
to get P(C | X_{1:t}) exactly (no transformer, pure generator), then compute:

    R²_max = 1 - E[ Var(onehot(C) | X_{1:t}) ] / Var(onehot(C))
           = 1 - mean_t E_X[ sum_c P(C=c|X)(1-P(C=c|X)) ] / (num_components · p(1-p))

where p = 1/num_components for uniform prior. This is the R² equivalent of τ
(which is the entropy ceiling) — both describe the same "amount of C-info in
X" in different units.

Writes r2_ceiling.json with per-δ ceiling + position-t-by-position breakdown.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT / "vendor"
for p in (REPO_ROOT / "src", REPO_ROOT / "experiments" / "transformer_nonergodic"):
    sys.path.insert(0, str(p))

from compute_tau_for_generators import (  # type: ignore  # noqa: E402
    mess3_T, posterior_over_components,
)
from sae_day.nonergodic_generator import NonergodicGenerator  # noqa: E402


DELTAS = [0.0, 0.05, 0.1, 0.15, 0.2]
SEQ_LEN = 128
N_EVAL = 500
SEED = 7


def build_params(delta: float):
    params = [
        (0.25 - delta, 0.6 + delta),
        (0.25, 0.6),
        (0.25 + delta, 0.6 - delta),
    ]
    return [(max(x, 1e-4), min(max(a, 1e-4), 1 - 1e-4)) for (x, a) in params]


def main() -> None:
    rows = []
    for delta in DELTAS:
        params = build_params(delta)
        gen = NonergodicGenerator.mess3_shared(params=params)
        Ts = [mess3_T(x, a) for (x, a) in params]
        vocab_maps = [[0, 1, 2]] * 3

        ds = gen.sample(n_sequences=N_EVAL, seq_len=SEQ_LEN, seed=SEED, with_component_labels=True)
        tokens = ds.tokens.numpy()
        post = posterior_over_components(Ts, vocab_maps, tokens)  # (N, T, C)

        N, T, Cdim = post.shape
        p0 = 1.0 / Cdim

        # Per-component variance under a uniform prior
        var_total_per_c = p0 * (1 - p0)   # = 2/9 for C=3

        # Irreducible per-position, per-component variance: E_X[ P(C=c|X)(1-P(C=c|X)) ]
        # averaged over sequences — then average over positions
        var_irred_per_c = (post * (1 - post)).mean(axis=(0, 1))  # shape (C,)
        r2_per_c = 1 - var_irred_per_c / var_total_per_c
        r2_mean = float(r2_per_c.mean())

        # Also R² ceiling AT THE FINAL POSITION (analogous to τ, which uses t=T)
        var_irred_final_per_c = (post[:, -1, :] * (1 - post[:, -1, :])).mean(axis=0)
        r2_final_per_c = 1 - var_irred_final_per_c / var_total_per_c
        r2_final_mean = float(r2_final_per_c.mean())

        # Mean R² ceiling as a function of position t ∈ [0, T)
        var_irred_by_t = (post * (1 - post)).mean(axis=0)                 # (T, C)
        r2_by_t = 1 - var_irred_by_t.mean(axis=1) / var_total_per_c        # (T,)

        # Per-W ceiling: mean of R²_max(t) over t ∈ [W-1, T-1].
        # This is the correct upper bound for a probe that uses a sliding window
        # of size W and reports codes at every valid position.
        r2_max_by_W = {}
        for W in (1, 2, 5, 10, 20, 30, 60):
            if W - 1 < T:
                r2_max_by_W[W] = float(r2_by_t[W - 1:].mean())
            else:
                r2_max_by_W[W] = float("nan")

        rows.append({
            "delta": float(delta),
            "r2_ceiling_mean_over_positions": r2_mean,
            "r2_ceiling_final_position": r2_final_mean,
            "r2_ceiling_per_component_mean": [float(v) for v in r2_per_c],
            "r2_ceiling_per_component_final": [float(v) for v in r2_final_per_c],
            "r2_by_t": [float(v) for v in r2_by_t.tolist()],
            "r2_max_by_W": r2_max_by_W,
        })

        pcs = " ".join(f"W{W}={r2_max_by_W[W]:.3f}" for W in (1, 5, 20, 60))
        print(
            f"δ={delta:<5}  "
            f"R²_max(mean over all t) = {r2_mean:.3f}   "
            f"R²_max(t=T) = {r2_final_mean:.3f}   "
            f"per-W: {pcs}",
            flush=True,
        )

    out = ROOT / "r2_ceiling.json"
    out.write_text(json.dumps({
        "description": "Bayes-optimal R² ceiling from exact P(C|X_{1:t}) via forward filter.",
        "seq_len": SEQ_LEN,
        "n_eval": N_EVAL,
        "seed": SEED,
        "rows": rows,
    }, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
