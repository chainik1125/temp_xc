"""Candidate 1 Stage 2 — the head-to-head panel on REAL Ward activations.

`briefings/task-hunt.md` Stage 2: per-token BatchTopK SAE, T-SAE,
Stacked, TXC-pre, TXC-post × T ∈ {2, 4, 8, 16} × seeds {1, 2, 42} +
untrained, matched realized l0_per_token, through the canonical runner.
Everything is the program's LOCKED uniform design
(`explorations.synthetic.design`) — only the datasource, F-anchor and
the T ladder differ, so the cells stay comparable with the synthetic
suite.

**The money plot** is `lambda_recovery` vs T, one line per arch: the
hunt wants TXC rising while T-SAE (per-token-decoded) stays flat.

Design notes specific to a REAL datasource:

- **No ground-truth F.** `d_sae` cannot be anchored on a feature count
  that does not exist. We anchor on the subject width instead and sweep
  {1024, 2048, 4096} = {d_in/4, d_in/2, d_in} — stated as the capacity
  axis in the record, with the scarce end (d_sae < d_in) the object of
  study, mirroring the "scarce regime" rule. `eauc` is NaN by
  construction here (see the datasource notes) and is not reported.
- **T ladder {2,4,8,16}** per the briefing (the synthetic design's
  default is {2,4,8}); token archs stay at T=1.
- **eval_window_L = 32** so every T in the ladder tiles it exactly
  (16 | 32), the § 4 apples-to-apples rule.
- k_pos sweep trimmed to {2, 8} to keep the real-activation grid inside
  the session budget — DISCLOSED here, not silently: the full
  {1,2,4,8,16} sweep is 5× the cells and Stage 2 is a T-story, not a
  sparsity-frontier story. Both values are dict-feasible for every
  family at T ≤ 16 given the capacity sweep.

Run:  .venv/bin/python -m \
        experiments.explorations.task_hunt.lambda_intensity.run_stage2 [workers] [ds]
"""

from __future__ import annotations

import sys
from pathlib import Path

from explorations.synthetic import design, grid

DS_DEFAULT = "ward_real_lambda_base_l12"
D_SAES = [1024, 2048, 4096]
K_POS = (2, 8)
WINDOW_TS = (2, 4, 8, 16)
EVAL_L = 32
N_STEPS = 8_000
HERE = Path(__file__).resolve().parent


def _cells(ds: str):
    return design.uniform_cells(
        ds, F=2048, n_steps=N_STEPS, d_saes=D_SAES, k_pos_sweep=K_POS,
        window_ts=WINDOW_TS, L=EVAL_L, log=print)


def _describe(res):
    m = res["metrics"]
    return (f"λ={m.get('lambda_recovery', float('nan')):.3f} "
            f"chance={m.get('lambda_chance', float('nan')):+.3f} "
            f"l0t={m.get('l0_per_token', float('nan')):.2f}")


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    ds = sys.argv[2] if len(sys.argv) > 2 else DS_DEFAULT
    out = HERE / "results" / f"stage2_{ds}.json"
    cells = _cells(ds)
    grid.run_pool(cells, out, max_workers=workers, describe=_describe,
                  tag=f"stage2/{ds}")


if __name__ == "__main__":
    main()
