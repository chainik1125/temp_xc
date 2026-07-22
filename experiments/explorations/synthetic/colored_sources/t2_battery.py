"""Colored sources (FB-3) — T2 non-triviality battery (LOOP.md gate T2).

The battery, adapted to a *feature-direction-recovery* bench whose primary
metric is weight-space (no probe — the memorization-budget and probe-budget
items reduce to statements, recorded in the JSON):

1. **Symmetry/relabeling audit** — analytic, recorded: F is a Haar-random
   orthonormal basis; there is no symbol structure, no group action on the
   data law other than global rotations, and a rotation moves F with it —
   nothing to relabel.
2. **Untrained panel floors** — every panel arch instantiated at the
   canonical cell shapes (d_sae ∈ {16, 32, 64}, T ∈ {1, 2, 4, 8}), random
   init → ``colored_rec_adj`` must sit inside the chance band (the card's
   falsifier 3: an init-alignment artifact in the metric must not exist).
3. **Trained-token control (CS-1 empirical with training in the loop)** —
   a short-trained canonical token SAE must stay at the floor up to the
   measured stream-leakage bound (gating § stream_leakage): training on the
   isotropic marginal cannot align atoms with F beyond the finite-sample
   tilt of the empirical covariance.
4. **Bag/shuffle dilution** — measured in ``gating.py`` (pooled rec_adj
   +0.69, shuffled lag-D +0.63): order destruction DILUTES but does not
   null this bench; the true null is window truncation (W ≤ D). Recorded
   there + in the dated card § 3 precision-amendment.

    .venv/bin/python -m experiments.explorations.synthetic.colored_sources.t2_battery
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

SEED = 0
N, D_IN, LAG_D, SIGMA = 32, 32, 2, 0.1
SEQ_LEN, N_SEQS = 64, 4096
TRAIN_STEPS = 5000
FLOOR_EPS = 0.05                  # the gating floor bar (documented there)

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "results" / "colored_t2_stats.json"


def main() -> None:
    from temp_bench.archs.batchtopk_sae import BatchTopKSAE
    from temp_bench.archs.spectral_txc import SpectralTXCBatchTopK
    from temp_bench.archs.stacked_batchtopk import StackedBatchTopK
    from temp_bench.archs.tsae import TSAEPaper
    from temp_bench.archs.txc_batchtopk import TXCBatchTopKPost, TXCBatchTopKPre
    from temp_bench.data.synthetic import colored_sources
    from temp_bench.evals.colored_recovery import colored_metrics

    torch.manual_seed(SEED)
    data = colored_sources(N=N, d_in=D_IN, D=LAG_D, sigma=SIGMA,
                           seq_len=SEQ_LEN, n_seqs=N_SEQS, seed=SEED)

    out: dict = {
        "card": "freqbench/cards/FB-3.md",
        "symmetry_audit": (
            "analytic: Haar-random orthonormal F; no symbol structure, no "
            "relabeling action on the data law — the only invariance is a "
            "global rotation, which transports F with it."),
        "memorization_audit": (
            "continuous Gaussian data — no template set exists at any "
            "capacity; the primary metric is weight-space (no probe), so "
            "probe memorization cannot arise."),
        "dilution_pointer": "see gating.py results (pooled/shuffled retain "
                            "diluted C_D; true null = window truncation).",
    }

    # 2. untrained panel floors at the canonical shapes
    floors = {}
    for d_sae in (16, 32, 64):
        for name, ctor in (
            ("batchtopk_sae_T1", lambda d=d_sae: BatchTopKSAE(d_in=D_IN, d_sae=d, k_pos=2)),
            ("tsae_T1", lambda d=d_sae: TSAEPaper(d_in=D_IN, d_sae=d, k_pos=2)),
            ("stacked_T4", lambda d=d_sae: StackedBatchTopK(d_in=D_IN, d_sae=d, T=4, k_pos=2)),
            ("txc_pre_T4", lambda d=d_sae: TXCBatchTopKPre(d_in=D_IN, d_sae=d, T=4, k_pos=2)),
            ("txc_post_T4", lambda d=d_sae: TXCBatchTopKPost(d_in=D_IN, d_sae=d, T=4, k_pos=2)),
            ("spectral_T4", lambda d=d_sae: SpectralTXCBatchTopK(d_in=D_IN, d_sae=d, T=4, k_pos=2)),
            ("txc_post_T8", lambda d=d_sae: TXCBatchTopKPost(d_in=D_IN, d_sae=d, T=8, k_pos=2)),
        ):
            torch.manual_seed(SEED + d_sae)
            try:
                m = ctor()
            except Exception as e:  # infeasible shape — record, not silent
                floors[f"{name}_d{d_sae}"] = f"SKIP ({type(e).__name__})"
                continue
            m.eval()
            floors[f"{name}_d{d_sae}"] = round(
                colored_metrics(m, data)["colored_rec_adj"], 4)
    out["untrained_floors"] = floors
    vals = [v for v in floors.values() if isinstance(v, float)]
    worst_pos = max(vals)
    worst_neg = min(vals)
    out["untrained_floor_note"] = (
        "the falsifier is a POSITIVE artifact (init aligned with F); the "
        "check is one-sided. Untrained spectral scores NEGATIVE adj "
        "(−0.07..−0.09): band-limited kernels have correlated time-slices, "
        "so the effective candidate count is below d_sae·T and the "
        "Gaussian chance reference is conservative AGAINST spectral — a "
        "measured metric property, deflating (never inflating) spectral's "
        "rec_adj; remember it when reading small spectral lifts.")
    print(f"[t2] untrained floors: worst positive {worst_pos:+.4f}, "
          f"worst negative {worst_neg:+.4f}", flush=True)

    # 3. trained-token control (CS-1 with training in the loop)
    sae = BatchTopKSAE(d_in=D_IN, d_sae=N, k_pos=2)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    xpool = data.x.reshape(-1, D_IN)
    for _ in range(TRAIN_STEPS):
        idx = torch.randint(0, len(xpool), (1024,))
        res = sae.train_step(xpool[idx])
        opt.zero_grad(); res["loss"].backward(); opt.step(); sae.post_step()
    sae.eval()
    trained_tok = colored_metrics(sae, data)
    out["trained_token_control"] = {
        "steps": TRAIN_STEPS,
        "colored_rec_adj": round(trained_tok["colored_rec_adj"], 4),
        "note": "must stay ≤ floor bar + measured stream leakage (~0.02 adj)",
    }
    print(f"[t2] trained token SAE rec_adj = "
          f"{trained_tok['colored_rec_adj']:+.4f}", flush=True)

    checks = {
        "untrained_floors_no_positive_artifact": bool(worst_pos <= FLOOR_EPS),
        "trained_token_at_floor": bool(
            abs(trained_tok["colored_rec_adj"]) <= FLOOR_EPS + 0.02),
    }
    out["verdict"] = {"checks": checks, "passes_t2": bool(all(checks.values()))}
    print(json.dumps(out["verdict"], indent=1), flush=True)

    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
