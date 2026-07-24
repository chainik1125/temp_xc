"""Stage 2 amendment — the **budget-matched** TXC-post cells.

Frozen card: `card_stage2_postmatched.md` (committed before this ran).
Round-1 `run_stage2.py` gave every arch nominal `k_pos = 8`, but
`txc_batchtopk_post` spends that budget per WINDOW, so its realized
`l0_per_token = k/T` collapsed 4.0 → 0.5 across the T ladder while the
rest of the panel spent 4.5–7.9. This module re-runs post alone with a
**per-T nominal k = 8·T** so the realized code rate matches the panel.

`design.uniform_cells` takes ONE `k_pos_sweep` for all T, so the cells
are emitted here instead. Everything else is held byte-identical to
round 1 (d_sae 2048, eval_window_L 32, n_steps 8000, buffer_tokens
524288, seeds {1,2,42}) — the cell dicts are the same schema
`grid.run_pool` consumes, so the work still goes through the ONE
canonical pathway.

Results go to a SEPARATE file so matched cells can never silently mix
with the round-1 nominal-k=8 cells; the renderer merges explicitly and
labels by realized l0.

Run:  .venv/bin/python -m \
        experiments.explorations.task_hunt.lambda_intensity.run_stage2_postmatched [workers] [ds]
"""

from __future__ import annotations

import sys
from pathlib import Path

from explorations.synthetic import grid

DS_DEFAULT = "ward_real_lambda_base_l12"
ARCH = "txc_batchtopk_post"
D_SAE = 2048                      # d_in/2 — the scarce anchor (unchanged)
K_PER_TOKEN = 8                   # the panel's per-token code rate
WINDOW_TS = (2, 4, 8, 16)
SEEDS = (1, 2, 42)
EVAL_L = 32
N_STEPS = 8_000
BUFFER_TOKENS = 524_288           # ≈ the corpus (4044 × 128 = 517,632)
HERE = Path(__file__).resolve().parent


def matched_k(T: int) -> int:
    """Nominal `k_pos` giving post a realized code rate of ~8 atoms/token.

    Post's BatchTopK pool is the window row, so the nominal budget is
    per WINDOW and `l0_per_token = k_pos / T` (exactly, for untrained
    cells; trained cells sit below by the JumpReLU-threshold shortfall).
    """
    return K_PER_TOKEN * T


def _cells(ds: str):
    cells = []
    for seed in SEEDS:
        for T in WINDOW_TS:
            k = matched_k(T)
            base = {"ds": ds, "arch": ARCH, "T": T, "d_sae": D_SAE,
                    "k_pos": k, "seed": seed, "eval_window_L": EVAL_L,
                    "buffer_tokens": BUFFER_TOKENS}
            cells.append({**base, "n_steps": N_STEPS, "kind": "trained"})
            cells.append({**base, "n_steps": 0, "kind": "untrained"})
    return cells


def _describe(res):
    m = res["metrics"]
    return (f"λ={m.get('lambda_recovery', float('nan')):.3f} "
            f"chance={m.get('lambda_chance', float('nan')):+.3f} "
            f"l0t={m.get('l0_per_token', float('nan')):.2f}")


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    ds = sys.argv[2] if len(sys.argv) > 2 else DS_DEFAULT
    out = HERE / "results" / f"stage2_postmatched_{ds}.json"
    cells = _cells(ds)
    print(f"[postmatched] per-T nominal k: "
          f"{ {T: matched_k(T) for T in WINDOW_TS} }", flush=True)
    grid.run_pool(cells, out, max_workers=workers, describe=_describe,
                  tag=f"stage2-postmatched/{ds}")


if __name__ == "__main__":
    main()
