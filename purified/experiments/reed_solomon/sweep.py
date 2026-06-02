"""Reed–Solomon degree-ladder sweep + unbiased (T, S, B) ablation.

Two cell families, all through the canonical ``run_experiment`` pathway
(experiment="freq_bench", which now emits the RS targets — class NTPS,
NTPS_sign, full-message regression NMSE — whenever the datasource carries
``coeffs``):

1. **Arch ladder** — per-token TopK SAE, sliding-T=5 TXC (txcdr_t5), joint
   T=W=16 TXC (txc_base_TW), and TFA, each across degree D∈{1,2,3}. Tests
   Dmitry's claim that higher D degrades the per-token/single-position archs
   while the window archs hold up. D=1 should reproduce the AC numbers.

2. **Unbiased (T, S, B) sweep** (Q3) — the band/stride/window family via
   ``txc_band``, swept per degree. We do NOT assume TXC's standard config is
   optimal: we sweep B (which frequency bands the encoder may use), T (window
   / band resolution), and S (stride), and the analysis simply REPORTS where
   the standard TXC point (T=5, S=1, B=all) lands relative to the rest. The
   prediction "degree-D is solvable iff B contains bands up to order ~D"
   (needing T≥~2D for the bins to exist) is what this measures.

Single GPU is plenty (synthetic, d_in=256, d_sae=1024). Example:

    CUDA_VISIBLE_DEVICES=1 TEMP_BENCH_ALLOW_DIRTY=1 \
      .venv/bin/python experiments/reed_solomon/sweep.py --n-steps 5000
"""

from __future__ import annotations

import argparse
import os
import time

from temp_bench.core.runner import run_experiment
from temp_bench.core.schemas import TrainingConfig

DEGREE_DS = {1: "rs_D1_W16_s10", 2: "rs_D2_W16_s10", 3: "rs_D3_W16_s10"}
W, K_POS, D_SAE, SEED = 16, 1, 1024, 0

# (label, arch, T) — txcdr_t5 = txc_base slid at T=5; txc_base_TW = joint T=W.
ARCH_LADDER = [
    ("regular_sae", "topk_sae", 1),
    ("txcdr_t5",    "txc_base", 5),
    ("txc_base_TW", "txc_base", 16),
    ("tfa",         "tfa",      5),
]

# Unbiased (T, S, B) grid via txc_band. Bands are rfft bins 0..floor(T/2).
# T=8 is included so bands {1,2,3} exist (needed to probe degree 3).
TSB_CELLS = [
    # band ablations at the standard sliding window (T=5, S=1)
    ("tsb_T5_S1_Ball", 5, 1, "all"),     # ← the STANDARD TXC point
    ("tsb_T5_S1_BDC",  5, 1, [0]),
    ("tsb_T5_S1_B1",   5, 1, [1]),
    ("tsb_T5_S1_B2",   5, 1, [2]),
    ("tsb_T5_S1_BAC",  5, 1, [1, 2]),
    # larger window so higher bands exist (probe degree-3 reachability)
    ("tsb_T8_S1_Ball", 8, 1, "all"),
    ("tsb_T8_S1_B123", 8, 1, [1, 2, 3]),
    ("tsb_T8_S1_B3",   8, 1, [3]),
    # stride axis at T=5, B=all (sliding → joint-like)
    ("tsb_T5_S2_Ball", 5, 2, "all"),
    ("tsb_T5_S4_Ball", 5, 4, "all"),
    # smaller window (coarser bands)
    ("tsb_T3_S1_Ball", 3, 1, "all"),
]


# Capacity probe (Q "is the D≥2 collapse real or under-resourced?"): the
# key window archs + a per-token control at D∈{2,3} with 4× dictionary and
# 3× steps. If NTPS stays ≈0 here, the collapse is not a capacity artefact.
CAP_ARCHS = [("regular_sae", "topk_sae", 1),
             ("txcdr_t5", "txc_base", 5),
             ("txc_base_TW", "txc_base", 16)]
CAP_DSAE, CAP_STEPS, CAP_DEGREES = 4096, 15_000, [2, 3]


def build_cells(capacity: bool = True) -> list[dict]:
    cells = []
    for D, ds in DEGREE_DS.items():
        for label, arch, T in ARCH_LADDER:
            cells.append({"kind": "ladder", "label": label, "arch": arch,
                          "T": T, "bands": "all", "S": (T if label == "txc_base_TW" else 1),
                          "degree": D, "datasource": ds})
        for label, T, S, bands in TSB_CELLS:
            cells.append({"kind": "tsb", "label": label, "arch": "txc_band",
                          "T": T, "bands": bands, "S": S,
                          "degree": D, "datasource": ds})
    if capacity:
        for D in CAP_DEGREES:
            for label, arch, T in CAP_ARCHS:
                cells.append({"kind": "capacity", "label": f"{label}_cap",
                              "arch": arch, "T": T, "bands": "all",
                              "S": (T if "TW" in label else 1), "degree": D,
                              "datasource": DEGREE_DS[D], "d_sae": CAP_DSAE,
                              "n_steps": CAP_STEPS})
    return cells


def run_cell(c: dict, n_steps: int) -> dict:
    d_sae = c.get("d_sae", D_SAE)
    steps = c.get("n_steps", n_steps)
    override = {"T": c["T"], "k_pos": K_POS, "d_sae": d_sae}
    if c["arch"] == "txc_band":
        override |= {"bands": c["bands"], "S": c["S"]}
    training_cfg = TrainingConfig(
        n_steps=steps, batch_size=2048, buffer_tokens=1_500_000,
        warmup_steps=200, arch_hparams_override=override,
    )
    bands_str = ",".join(map(str, c["bands"])) if c["bands"] != "all" else "all"
    eval_cfg = {"smoke": False, "label": c["label"], "kind": c["kind"],
                "degree": c["degree"], "W": W, "T": c["T"], "S": c["S"],
                "k_pos": K_POS, "d_sae": d_sae, "bands": bands_str}
    r = run_experiment(
        experiment="freq_bench", arch_name=c["arch"], seed=SEED,
        datasource_name=c["datasource"], training_cfg=training_cfg,
        eval_cfg=eval_cfg, agent=os.environ.get("AGENT_NAME"), allow_dirty=True,
    )
    return r.row.metrics | {"cached": r.eval_cached}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--n-steps", type=int, default=5_000)
    args = ap.parse_args()

    cells = build_cells()
    mine = [c for i, c in enumerate(cells) if i % args.n_shards == args.shard_id]
    print(f"[rs sweep {args.shard_id}/{args.n_shards}] {len(mine)}/{len(cells)} cells "
          f"on CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    for i, c in enumerate(mine, 1):
        t0 = time.time()
        try:
            m = run_cell(c, args.n_steps)
            tag = "cache" if m.get("cached") else f"{time.time()-t0:.0f}s"
            print(f"[rs] [{i}/{len(mine)}] D{c['degree']} {c['label']:15s} "
                  f"NTPS={m['NTPS']:+.3f} NTPS_sign={m.get('NTPS_sign', float('nan')):+.3f} "
                  f"nmse_msg={m.get('nmse_msg', float('nan')):.3f} "
                  f"nmse_lead={m.get('nmse_lead', float('nan')):.3f} "
                  f"ff={m.get('freqfrac', float('nan')):.2f} ({tag})", flush=True)
        except Exception as e:
            print(f"[rs] [{i}/{len(mine)}] D{c['degree']} {c['label']}: "
                  f"FAILED {type(e).__name__}: {e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
