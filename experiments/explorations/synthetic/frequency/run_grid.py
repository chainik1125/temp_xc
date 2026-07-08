"""Run the cyclic-tone frequency benchmark grid in parallel.

Grid (frequency/bench_spec.md § 5 + frozen amendments A1–A5 — BatchTopK
fair-backbone, same shape template as backtracking/changepoint):

  archs/T : (batchtopk_sae,1) (tsae,1)                             [per-token]
            (txc_batchtopk_pre,2/4/8/16) (txc_batchtopk_post,2/4/8/16)
            (spectral_txc,2/4/8/16)                                [crosscoders]
  d_sae   : 32, 64, 101, 256   (anchored on M=101; scarce {32,64} + at-F {101}
            + over-complete {256}, ALL < |Ω|·M=1010 → memorization-free)
  seeds   : 1, 2, 42
  k_pos=1, eval_window_L=32

Amendment A5 (2026-07-08): stacked_batchtopk is DROPPED. Its per-tile code is
the CONCATENATED per-position code (dim = T·d_sae), so for the 2nd-moment
velocity latent a linear probe recovers Y only by memorizing the ≤ |Ω|·M=1010
distinct clean windows once T·d_sae ≥ 1010 — the signed-motion memorization
confound. The shared-code crosscoders (tile-code = d_sae ≤ 256 < 1010) are
memorization-free; the per-token archs already instantiate the "cannot mix
positions" null. So stacked is a confounded comparator here and is omitted.

Throughput normalised across archs: batch_size = 1024//T (T>1) else 1024, so
every cell reconstructs ~1024 token-positions/step and the BatchTopK pool is
B·T = 1024.

Runs TWO datasources: toy_cyclic_circle_M101_d128 (HEADLINE, full grid) and
toy_cyclic_random_M101_d128 (NULL, anchor d_sae=101 only — flatness check). Plus
the UNTRAINED-encoder control (n_steps=0) and the k_pos=2 anchor at d_sae=101 on
circle, and a d_sae=2048 (> |Ω|·M) MEMORIZATION demo on both datasources.

Each cell goes through the canonical runner (flock-safe leaderboard append).
Results also dumped to frequency/results/frequency_grid_results.json.

    .venv/bin/python -m experiments.explorations.synthetic.frequency.run_grid [max_workers]
"""

from __future__ import annotations

import os

os.environ.setdefault("TEMP_BENCH_ALLOW_DIRTY", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("AGENT_NAME", "autoresearch")

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

CIRCLE = "toy_cyclic_circle_M101_d128"
RANDOM = "toy_cyclic_random_M101_d128"
L = 32
K_POS = 1
N_STEPS = 6000
D_SAES = [32, 64, 101, 256]
ANCHOR = 101
SEEDS = [1, 2, 42]
MEMO_DSAE = 2048          # > |Ω|·M = 1010 → the memorization-regime demo

ARCH_T = [("batchtopk_sae", 1), ("tsae", 1),
          ("txc_batchtopk_pre", 2), ("txc_batchtopk_pre", 4),
          ("txc_batchtopk_pre", 8), ("txc_batchtopk_pre", 16),
          ("txc_batchtopk_post", 2), ("txc_batchtopk_post", 4),
          ("txc_batchtopk_post", 8), ("txc_batchtopk_post", 16),
          ("spectral_txc", 2), ("spectral_txc", 4),
          ("spectral_txc", 8), ("spectral_txc", 16)]
MEMO_ARCH_T = [("txc_batchtopk_pre", 16), ("spectral_txc", 16)]

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "frequency_grid_results.json"


def _batch_size(T: int) -> int:
    return 1024 if T == 1 else 1024 // T


def _cells():
    cells = []
    for seed in SEEDS:
        for arch, T in ARCH_T:
            # circle: full d_sae frontier (trained)
            for d_sae in D_SAES:
                cells.append({"ds": CIRCLE, "arch": arch, "T": T, "d_sae": d_sae,
                              "k_pos": K_POS, "seed": seed, "n_steps": N_STEPS,
                              "kind": "trained"})
            # circle: untrained control + k_pos=2 anchor (at d_sae=ANCHOR)
            cells.append({"ds": CIRCLE, "arch": arch, "T": T, "d_sae": ANCHOR,
                          "k_pos": K_POS, "seed": seed, "n_steps": 0, "kind": "untrained"})
            cells.append({"ds": CIRCLE, "arch": arch, "T": T, "d_sae": ANCHOR,
                          "k_pos": 2, "seed": seed, "n_steps": N_STEPS, "kind": "trained"})
            # random null: anchor d_sae only (flatness)
            cells.append({"ds": RANDOM, "arch": arch, "T": T, "d_sae": ANCHOR,
                          "k_pos": K_POS, "seed": seed, "n_steps": N_STEPS, "kind": "trained"})
    # memorization demo (> |Ω|·M): both datasources, 1 seed
    for ds in (CIRCLE, RANDOM):
        for arch, T in MEMO_ARCH_T:
            cells.append({"ds": ds, "arch": arch, "T": T, "d_sae": MEMO_DSAE,
                          "k_pos": K_POS, "seed": 1, "n_steps": N_STEPS, "kind": "memo"})
    return cells


def run_one(cell):
    from temp_bench.core.runner import run_experiment
    from temp_bench.core.schemas import TrainingConfig
    try:
        k_pos = cell.get("k_pos", K_POS)
        override = {"k_pos": k_pos, "d_sae": cell["d_sae"], "T": cell["T"]}
        tcfg = TrainingConfig(n_steps=cell["n_steps"], batch_size=_batch_size(cell["T"]),
                              buffer_tokens=2_000_000, arch_hparams_override=override)
        ecfg = {"smoke": False, "k_pos": k_pos, "eval_window_L": L}
        r = run_experiment(experiment="synthetic", arch_name=cell["arch"],
                           seed=cell["seed"], datasource_name=cell["ds"],
                           training_cfg=tcfg, eval_cfg=ecfg,
                           agent="autoresearch", allow_dirty=True)
        return {**cell, "metrics": {k: float(v) for k, v in r.row.metrics.items()},
                "train_cached": r.train_cached, "eval_cached": r.eval_cached, "ok": True}
    except Exception as e:
        import traceback
        return {**cell, "ok": False, "error": f"{type(e).__name__}: {e}",
                "tb": traceback.format_exc()[-1500:]}


def main():
    max_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    cells = _cells()
    print(f"[grid] {len(cells)} cells, max_workers={max_workers}", flush=True)
    t0 = time.time()
    results = []
    done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(run_one, c): c for c in cells}
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            done += 1
            el = time.time() - t0
            dtag = "circ" if res["ds"] == CIRCLE else "rand"
            tag = f"{dtag}/{res['arch']}/T{res['T']}/d{res['d_sae']}/k{res.get('k_pos',1)}/s{res['seed']}/{res['kind']}"
            if res.get("ok"):
                m = res["metrics"]
                vr = m.get("velocity_recovery", float("nan"))
                vo = m.get("velocity_oracle", float("nan"))
                print(f"[{done}/{len(cells)} {el:6.0f}s] {tag:<52} "
                      f"vel={vr:+.3f} orc={vo:.3f} nmse={m.get('nmse',float('nan')):.3f} "
                      f"(cache t={res['train_cached']} e={res['eval_cached']})", flush=True)
            else:
                print(f"[{done}/{len(cells)} {el:6.0f}s] {tag:<52} FAILED {res['error']}", flush=True)
            OUT.write_text(json.dumps(results, indent=2))
    n_ok = sum(1 for r in results if r.get("ok"))
    print(f"[grid] DONE {n_ok}/{len(cells)} ok in {time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
