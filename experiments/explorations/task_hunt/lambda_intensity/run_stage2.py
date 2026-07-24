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

- **Arch panel = the briefing's five** (per-token BatchTopK SAE, T-SAE,
  Stacked, TXC-pre, TXC-post). `spectral_txc` is NOT run: the briefing
  does not name it and the screen put this latent in regime 2, where
  the DCT-band prior has no predicted role. Disclosed, not silent.
- **T ladder {2,4,8,16}** per the briefing (the synthetic design's
  default is {2,4,8}); token archs stay at T=1.
- **eval_window_L = 32** so every T in the ladder tiles it exactly
  (16 | 32), the § 4 apples-to-apples rule.
- **No ground-truth F**, so the capacity axis cannot be anchored on a
  feature count that does not exist. Rather than invent one we pin a
  SINGLE scarce anchor `d_sae = 2048 = d_in/2` and a single
  `k_pos = 8`, and report the T-response at that operating point. This
  is a deliberate, disclosed narrowing of the program's usual
  frontier-over-capacity rule: Stage 2 here answers "does recovery
  track T, per arch", not "where is the capacity frontier". A single
  operating point is one labeled slice and the record says so.
  `d_sae ≥ k_pos·T` holds for every pooled family at T ≤ 16.
- `eauc` is NaN by construction (see the datasource notes) and is not
  reported. Fairness rides on equal `k_pos` (equal per-token budget)
  plus the realized `l0_per_token` recorded per cell.

Run:  .venv/bin/python -m \
        experiments.explorations.task_hunt.lambda_intensity.run_stage2 [workers] [ds]
"""

from __future__ import annotations

import sys
from pathlib import Path

from explorations.synthetic import design, grid

DS_DEFAULT = "ward_real_lambda_base_l12"
D_SAE = 2048                      # d_in/2 — the scarce anchor
K_POS = (8,)
WINDOW_TS = (2, 4, 8, 16)
EVAL_L = 32
N_STEPS = 8_000
HERE = Path(__file__).resolve().parent

# The briefing's five-arch panel (no spectral_txc — see the module docstring).
PANEL = (
    ("batchtopk_sae", "token"),
    ("tsae", "token"),
    ("stacked_batchtopk", "stacked"),
    ("txc_batchtopk_pre", "pre"),
    ("txc_batchtopk_post", "post"),
)


def _cells(ds: str):
    return design.uniform_cells(
        ds, F=D_SAE, n_steps=N_STEPS, d_saes=[D_SAE], k_pos_sweep=K_POS,
        archs=PANEL, window_ts=WINDOW_TS, L=EVAL_L, untrained_kpos=K_POS[0],
        log=print)


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
