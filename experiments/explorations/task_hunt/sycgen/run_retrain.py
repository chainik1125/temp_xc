"""sycgen FIRST-KEEP matrix retrain grid (RETRAIN_CARD.md § 2, as
AMENDED by § 5).

The λ̂ Stage-2 design on the sycgen substrate, btk-only arms per the
pinned matrix mapping (692cb): claiming arm
`txc_batchtopk_post_btkonly` × T ∈ {2,4,8,16} + per-token anchors
(`batchtopk_sae_btkonly`, `tsae_btkonly` @ T=1), seeds {1,2,42},
untrained twins INCLUDED (first training on this substrate — matrix
standard). Hyperparameters inherited BY CONSTRUCTION from
`run_stage2.py`'s constants (F-anchor 2048, k_pos 8, 8000 steps,
eval L 32, corpus-sized buffer). `eval_extra.retrain_tag` namespaces
eval keys; checkpoints persist locally for the shuffle overlay and the
HF ckpt push.

AMENDMENT (card § 5): the launched grid listed T ∈ {2,4,6,8,10,16}
(48 cells) — T ∈ {6,10} cannot pass the canonical eval
(`synthetic_recovery` requires eval_window_L % T == 0, power-of-two;
L=32, and no L ≤ seq_len 128 divides {6,10,16}, LCM 240). The λ̂
Stage-2 template axis is (2,4,8,16); the leaderboard holds zero
T6/T10 rows fleet-wide. The as-run shard jsons keep the 12 failed
T{6,10} cells (ok:false, the exact ValueError) as the receipt; the
surviving 36 cells ARE this amended grid. Not relaunched.

Two-shard split for the pod's two GPUs (deterministic i%2 over the
sorted cell list — balanced mix of arms/Ts per shard):

  CUDA_VISIBLE_DEVICES=0 python -m ...sycgen.run_retrain 3 0
  CUDA_VISIBLE_DEVICES=1 python -m ...sycgen.run_retrain 3 1
"""

from __future__ import annotations

import sys
from pathlib import Path

from explorations.synthetic import design, grid

HERE = Path(__file__).resolve().parent
DS = "sycgen_real_age_llama31_8b_l14"
D_SAE = 2048
K_POS = (8,)
WINDOW_TS = (2, 4, 8, 16)  # § 5 amendment: λ̂ template axis; L=32 tiles exactly
EVAL_L = 32
N_STEPS = 8_000
BUFFER_TOKENS = 524_288
RETRAIN_TAG = "sycgen_keep_r1"

ARMS = (
    ("batchtopk_sae_btkonly", "token"),
    ("tsae_btkonly", "token"),
    ("txc_batchtopk_post_btkonly", "post"),
)


def cells():
    cs = design.uniform_cells(
        DS, F=D_SAE, n_steps=N_STEPS, d_saes=[D_SAE], k_pos_sweep=K_POS,
        archs=ARMS, window_ts=WINDOW_TS, L=EVAL_L, untrained=True,
        untrained_kpos=K_POS[0], log=print)
    for c in cs:
        c["buffer_tokens"] = BUFFER_TOKENS
        c["eval_extra"] = {"retrain_tag": RETRAIN_TAG}
    # 18 trained (12 post + 6 anchors) + 18 untrained twins (one per
    # (arch, T) per seed) = 36; assert the amended card count (§ 5).
    assert len(cs) == 36, f"amended grid is 36 cells, built {len(cs)}"
    return sorted(cs, key=lambda c: (c["arch"], c["T"], c["seed"],
                                     c["n_steps"]))


def _describe(res):
    m = res["metrics"]
    return (f"r={m.get('lambda_recovery', float('nan')):.3f} "
            f"chance={m.get('lambda_chance', float('nan')):+.3f} "
            f"l0t={m.get('l0_per_token', float('nan')):.2f}")


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    shard = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    cs = [c for i, c in enumerate(cells()) if i % 2 == shard]
    out = HERE / "results" / f"retrain_shard{shard}.json"
    grid.run_pool(cs, out, max_workers=workers, describe=_describe,
                  tag=f"sycgen_keep_r1/shard{shard}")


if __name__ == "__main__":
    main()
