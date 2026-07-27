"""Width-matched tsae_btkonly probing cells (WIDTH_MATCH_TSAE_CARD.md).

Directive 98a9ea718: the P1 trained tsae comparator, re-run at matched
d_sae=18432. Single delta from the P1 rows is
``arch_hparams_override={"d_sae": 18432}`` — every other knob verbatim
(n_steps 20000, batch_size 32 sequences, probing-1.2.0 eval, arm
btk-only). Seeds 42-first per the P1 dispatch convention.

Run (GPU 1)::

    CUDA_VISIBLE_DEVICES=1 AGENT_NAME=runpod-b TEMP_BENCH_ALLOW_DIRTY=1 \
        nohup .venv/bin/python -m experiments.probing.actmix.width_match_tsae \
        > /workspace/logs/width_match_tsae.log 2>&1 &
"""

from __future__ import annotations

import os
import time

from temp_bench.core.runner import run_experiment
from temp_bench.core.schemas import TrainingConfig

DATASOURCE = "gemma_2_2b_it_l13_fineweb_24k128"
ARCH = "tsae_btkonly"
D_SAE = 18432
SEEDS = (42, 1, 2)
K_FEATS = (5, 20)


def main() -> None:
    # Preflight (sweep.py convention): full probe cache + training acts.
    from temp_bench.core.config import compute_data_key, data_cache_dir, load_datasource
    from temp_bench.data.probe_cache import list_probe_cache

    n_tasks = len(list_probe_cache(DATASOURCE))
    if n_tasks != 38:
        raise SystemExit(
            f"[width_match] preflight FAIL: probe cache has {n_tasks} "
            "complete tasks, expected 38."
        )
    acts = data_cache_dir(compute_data_key(load_datasource(DATASOURCE))) / "acts.npy"
    if not acts.exists():
        raise SystemExit(f"[width_match] preflight FAIL: acts missing at {acts}")

    training_cfg = TrainingConfig(
        n_steps=20_000,
        batch_size=32,
        arch_hparams_override={"d_sae": D_SAE},
    )
    for seed in SEEDS:
        for k_feat in K_FEATS:
            eval_cfg = {
                "k_feat": int(k_feat),
                "S": 32,
                "shuffle": "within_window",
                "shuffle_seed": 0,
                "encode_batch_size": 64,
                "arm": "btk-only",
                "smoke": False,
            }
            t0 = time.time()
            result = run_experiment(
                experiment="probing",
                arch_name=ARCH,
                seed=seed,
                datasource_name=DATASOURCE,
                training_cfg=training_cfg,
                eval_cfg=eval_cfg,
                agent=os.environ.get("AGENT_NAME"),
            )
            m = result.row.metrics
            status = "CACHED" if result.eval_cached else f"ran {time.time() - t0:.0f}s"
            print(
                f"[{status}] {ARCH}/d_sae={D_SAE}/seed={seed}/k_feat={k_feat}  "
                f"mean_auc={m.get('mean_auc', float('nan')):.4f}  "
                f"auc_shuf={m.get('mean_auc_shuf', float('nan')):.4f}  "
                f"l0={m.get('realized_l0', float('nan')):.2f}  "
                f"eval_key={result.eval_key}",
                flush=True,
            )
    print("[width_match] COMPLETE (3 trainings x 2 k_feats)", flush=True)


if __name__ == "__main__":
    main()
