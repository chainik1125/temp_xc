"""Targeted (T, S, B) ablations on the AC bench — Dmitry's spectral request.

The TXC encoder is equivalent to a sum of per-frequency-band linear
projections + TopK. This sweep operationalises that view by:

  - Reproducing the paper's existing TXC variants as points in the
    (T, S, B) family (sanity checks: should match prior NTPS).
  - Band ablations: hold (T, S) fixed and vary B to localise *which*
    frequency bands carry the direction signal.
  - Stride ablations: hold (T, B) fixed and vary S to interpolate
    between sliding (S=1) and joint-like (S=T).

Predictions are pinned BEFORE the runs (the ``PREDICTIONS`` dict below
is written to ``results/freq_bench/v2_sweep/spectral_predictions.json``
before any cell runs). The post-hoc analysis compares predicted vs
measured.

All cells are at the strong AC slot:  W=16, raw_k=1, σ=0.1, d_sae=1024.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from pathlib import Path

from temp_bench.core.runner import run_experiment
from temp_bench.core.schemas import TrainingConfig

DATASOURCE = "fb_ac_W16_s10"
W, K_POS, D_SAE = 16, 1, 1024
SEED = 0


# ── pre-registered cell list + predictions ───────────────────────────────


def cell(label, T, S, bands, predicted_NTPS, rationale):
    return {"label": label, "T": T, "S": S, "bands": bands,
            "predicted_NTPS": predicted_NTPS, "rationale": rationale}


CELLS = [
    # Reproductions (sanity)
    cell("band_T5_S1_Ball", 5, 1, "all", 0.72,
         "matches txcdr_t5 (T=5 sliding, all bands)"),
    cell("band_T16_S16_Ball", 16, 16, "all", 0.17,
         "matches txc_base_TW (joint T=W=16, all bands)"),

    # Band ablations at T=5, S=1
    cell("band_T5_S1_BDC", 5, 1, [0], 0.02,
         "DC-only TXC: equivalent to per-token + window-mean; cannot encode direction"),
    cell("band_T5_S1_B1", 5, 1, [1], 0.55,
         "first AC band carries the fundamental of the velocity walk"),
    cell("band_T5_S1_B2", 5, 1, [2], 0.30,
         "second AC band: harmonic, weaker but nonzero"),
    cell("band_T5_S1_BAC", 5, 1, [1, 2], 0.70,
         "AC-only (no DC): direction info preserved, DC band irrelevant"),
    cell("band_T5_S1_BDC1", 5, 1, [0, 1], 0.65,
         "DC + first AC: comparable to full B=all"),

    # Stride ablations at T=5, B=all
    cell("band_T5_S2_Ball", 5, 2, "all", 0.62,
         "S=2: ~6 windows over W=16; modest SNR drop vs sliding"),
    cell("band_T5_S4_Ball", 5, 4, "all", 0.50,
         "S=4: 3 windows; bigger drop"),
    cell("band_T5_S8_Ball", 5, 8, "all", 0.35,
         "S=8: 2 windows; approaching joint ceiling"),
    cell("band_T5_S12_Ball", 5, 12, "all", 0.20,
         "S=12: 1 window; effectively joint-T=5-on-W=16 ≈ joint T=W ceiling"),
]


def run_cell(c, n_steps):
    override = {"T": c["T"], "k_pos": K_POS, "d_sae": D_SAE,
                "bands": c["bands"], "S": c["S"]}
    training_cfg = TrainingConfig(
        n_steps=n_steps, batch_size=2048, buffer_tokens=1_500_000,
        warmup_steps=200, arch_hparams_override=override,
    )
    eval_cfg = {
        "smoke": False, "label": c["label"], "W": W, "T": c["T"], "S": c["S"],
        "k_pos": K_POS, "d_sae": D_SAE,
        "bands": ",".join(str(b) for b in c["bands"]) if c["bands"] != "all" else "all",
    }
    r = run_experiment(
        experiment="freq_bench", arch_name="txc_band", seed=SEED,
        datasource_name=DATASOURCE, training_cfg=training_cfg,
        eval_cfg=eval_cfg, agent=os.environ.get("AGENT_NAME"), allow_dirty=True,
    )
    return r.row.metrics | {"cached": r.eval_cached}


def _launch_gpus(n_gpus, n_steps):
    import subprocess
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    procs = []
    for g in range(n_gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g),
                   TEMP_BENCH_ALLOW_DIRTY="1", AGENT_NAME="aniket",
                   OMP_NUM_THREADS="8", MKL_NUM_THREADS="8",
                   OPENBLAS_NUM_THREADS="8")
        log = open(os.path.join(root, "logs", f"spectral_shard{g}.log"), "w")
        procs.append(subprocess.Popen(
            [sys.executable, os.path.join(here, "spectral_ablations.py"),
             "--n-shards", str(n_gpus), "--shard-id", str(g),
             "--n-steps", str(n_steps)],
            stdout=log, stderr=subprocess.STDOUT, cwd=root, env=env))
        print(f"[orch] spawned shard {g} pid {procs[-1].pid}", flush=True)
    rc = 0
    for p in procs:
        rc |= p.wait()
    print(f"[orch] all shards done (rc={rc})", flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--n-steps", type=int, default=6_000)
    ap.add_argument("--launch-gpus", type=int, default=0)
    args = ap.parse_args()

    if args.launch_gpus > 0:
        # write the prediction registry once, before spawning workers
        out = Path(__file__).resolve().parents[2] / "results" / "freq_bench" / "v2_sweep"
        out.mkdir(parents=True, exist_ok=True)
        preds = {"cells": CELLS, "datasource": DATASOURCE, "W": W,
                 "k_pos": K_POS, "d_sae": D_SAE, "seed": SEED}
        json.dump(preds, open(out / "spectral_predictions.json", "w"), indent=2)
        print(f"[orch] wrote predictions to {out/'spectral_predictions.json'}",
              flush=True)
        return _launch_gpus(args.launch_gpus, args.n_steps)

    mine = [c for i, c in enumerate(CELLS) if i % args.n_shards == args.shard_id]
    print(f"[shard {args.shard_id}/{args.n_shards}] {len(mine)}/{len(CELLS)} cells",
          flush=True)
    for i, c in enumerate(mine, 1):
        t0 = time.time()
        try:
            m = run_cell(c, args.n_steps)
            tag = "cache" if m.get("cached") else f"{time.time()-t0:.0f}s"
            pred = c["predicted_NTPS"]
            err = m["NTPS"] - pred
            print(f"[shard {args.shard_id}] [{i}/{len(mine)}] {c['label']:25s} "
                  f"NTPS={m['NTPS']:+.3f} (pred {pred:+.2f}, err {err:+.2f}) "
                  f"gap={m['order_gap']:+.3f} rev_drop={m['reverse_drop']:+.3f} "
                  f"freqfrac={m.get('freqfrac', float('nan')):.3f} ({tag})",
                  flush=True)
        except Exception as e:
            print(f"[shard {args.shard_id}] [{i}/{len(mine)}] {c['label']}: "
                  f"FAILED {type(e).__name__}: {e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
