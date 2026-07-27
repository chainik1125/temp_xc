"""One EM cell for the paper-arch window-scaling lane (RunPod venue).

Usage: python -m experiments.explorations.btk_rerun.em_cell <arch> <T> <seed>
Same cell spec as scripts/modal_btk_em.py (canonical em training cfg).
"""
from __future__ import annotations

import json
import sys

from temp_bench.core.runner import run_experiment
from temp_bench.core.schemas import TrainingConfig


def main() -> None:
    arch, T, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    res = run_experiment(
        experiment="em", arch_name=arch, seed=seed,
        datasource_name="qwen_2_5_7b_instruct_medical_l15",
        training_cfg=TrainingConfig(
            n_steps=25_000, batch_size=1024,
            bricken_enabled=True, ema_auxk_alpha=0.125,
            dead_threshold_tokens=128_000,
            arch_hparams_override={"T": T},
        ),
        eval_cfg={}, agent="dmitry-btk-sprint", allow_dirty=True,
    )
    print("[cell done]", res.eval_key, json.dumps(res.row.metrics), flush=True)


if __name__ == "__main__":
    main()
