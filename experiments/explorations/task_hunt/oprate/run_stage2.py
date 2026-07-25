"""oprate Stage 2 — the head-to-head panel on REAL Ward activations.

`briefings/stage2-oprate.md`: the λ̂ Stage-2 pattern (per-token BatchTopK
SAE, T-SAE, Stacked, TXC-pre, TXC-post × T ∈ {2, 4, 8, 16} × seeds
{1, 2, 42} + untrained controls) on the `rate_case` trailing-rate target,
through the canonical runner. Frozen card: `CARD_STAGE2.md` (committed
before any cell — git order is the evidence).

Two deliberate deviations from `lambda_intensity/run_stage2.py`, both
paid-for lessons, both in the card:

1. **TXC-post runs at per-T nominal k = 8·T from the start** (the
   post-matched amendment's correction, `card_stage2_postmatched.md`):
   post spends its BatchTopK budget per WINDOW, so nominal k = 8 would
   realize l0_per_token = 8/T — a sparsity ramp, not a matched panel.
   `design.uniform_cells` takes one k_pos for all T, so the post cells
   are emitted inline (the postmatched pattern). Budget-match is on
   REALIZED l0_per_token, pre-registered band in the card; untrained
   matched cells must realize exactly 8.00 (the mechanism's own check).
2. **Every cell carries the paired v2 λ-probe columns**
   (`PROBE_V2_SPEC.md` § 2 verbatim, incl. the pinned alpha grid), so
   the panel never needs re-running whichever way the post-deadline
   readout adoption goes. Claim on v1 (the taken methods decision);
   never quote v2 as canonical.

The T-SAE trained cells are FIRST in the cell list (A40 addendum: the
tsae arm is the long pole — its cost is structural in the
SequenceBuffer path, one full-batch sequence clone per step, GPU ~idle
— and a panel that never scheduled its key baseline is not reportable).
The optional `sel` argument (`only-tsae` / `skip-tsae`) exists ONLY so
tsae can run as its own pool on a second GPU while the rest of the
panel proceeds; it changes scheduling, never cell content, and each
selection writes its own results file so concurrent pools cannot
clobber each other (`run_pool` rewrites its output whole). Receipts are
recomputed from `results/leaderboard.jsonl`, which is canonical.

Run:  .venv/bin/python -m \
        experiments.explorations.task_hunt.oprate.run_stage2 \
        [workers] [ds] [sel: all|only-tsae|skip-tsae]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from explorations.synthetic import design, grid

DS_DEFAULT = "ward_real_oprate_case_base_l12"
D_SAE = 2048                      # d_in/2 — the scarce anchor (λ̂ panel value)
K_POS = (8,)
WINDOW_TS = (2, 4, 8, 16)
SEEDS = (1, 2, 42)
EVAL_L = 32
N_STEPS = 8_000
BUFFER_TOKENS = 524_288           # ≈ the corpus (4044 × 128 = 517,632)
HERE = Path(__file__).resolve().parent

# The briefing's five-arch panel; post is emitted separately at matched k.
PANEL_UNIFORM = (
    ("batchtopk_sae", "token"),
    ("tsae", "token"),
    ("stacked_batchtopk", "stacked"),
    ("txc_batchtopk_pre", "pre"),
)
POST_ARCH = "txc_batchtopk_post"

# PROBE_V2_SPEC.md § 2, verbatim — paired v2 columns on every row.
V2 = {"lambda_probe_v2": True, "lambda_v2_probe": "ridge",
      "lambda_v2_alphas": list(np.logspace(-2, 4, 13)),
      "lambda_v2_n_windows": 8192, "lambda_v2_split": "trace"}


def matched_k(T: int) -> int:
    """Post's nominal k for a realized code rate of ~8 atoms/token
    (`card_stage2_postmatched.md` § 3: budget is per window, l0/tok = k/T)."""
    return K_POS[0] * T


def _cells(ds: str):
    cells = design.uniform_cells(
        ds, F=D_SAE, n_steps=N_STEPS, d_saes=[D_SAE], k_pos_sweep=K_POS,
        archs=PANEL_UNIFORM, window_ts=WINDOW_TS, L=EVAL_L,
        untrained_kpos=K_POS[0], seeds=SEEDS, log=print)
    for seed in SEEDS:
        for T in WINDOW_TS:
            base = {"ds": ds, "arch": POST_ARCH, "T": T, "d_sae": D_SAE,
                    "k_pos": matched_k(T), "seed": seed,
                    "eval_window_L": EVAL_L}
            cells.append({**base, "n_steps": N_STEPS, "kind": "trained"})
            cells.append({**base, "n_steps": 0, "kind": "untrained"})
    for c in cells:
        c["buffer_tokens"] = BUFFER_TOKENS
        c["eval_extra"] = V2
    # tsae trained cells first (the long pole), then everything else in
    # emitted order. Stable sort: scheduling only, cell content untouched.
    cells.sort(key=lambda c: 0 if (c["arch"] == "tsae"
                                   and c["n_steps"] > 0) else 1)
    return cells


def _select(cells, sel: str):
    """Scheduling-only selection; cell content is never touched.

    `sel` may carry a round-robin shard suffix `:i/n` (e.g.
    `skip-tsae:0/2`) — added after the first Pool-B attempt OOMed a
    44 GB A40 at 5 workers (v2 eval peaks ≈ 12.7 GB/worker): shards
    let the same cell list spread over two GPUs at 3 workers each,
    the fleet's round-robin precedent. Deterministic: filter, then
    take cells[i::n].
    """
    shard = None
    if ":" in sel:
        sel, spec = sel.split(":", 1)
        i, n = spec.split("/")
        shard = (int(i), int(n))
    if sel == "only-tsae":
        cells = [c for c in cells if c["arch"] == "tsae"]
    elif sel == "skip-tsae":
        cells = [c for c in cells if c["arch"] != "tsae"]
    elif sel != "all":
        raise SystemExit(f"unknown sel {sel!r}: all|only-tsae|skip-tsae[:i/n]")
    if shard is not None:
        cells = cells[shard[0]::shard[1]]
    return cells


def _describe(res):
    m = res["metrics"]
    return (f"λ={m.get('lambda_recovery', float('nan')):.3f} "
            f"v2={m.get('lambda_recovery_v2', float('nan')):.3f} "
            f"chance={m.get('lambda_chance', float('nan')):+.3f} "
            f"l0t={m.get('l0_per_token', float('nan')):.2f}")


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    ds = sys.argv[2] if len(sys.argv) > 2 else DS_DEFAULT
    sel = sys.argv[3] if len(sys.argv) > 3 else "all"
    cells = _select(_cells(ds), sel)
    safe = sel.replace(":", "-").replace("/", "of")
    out = HERE / "results" / (f"stage2_{ds}.json" if sel == "all"
                              else f"stage2_{ds}__{safe}.json")
    grid.run_pool(cells, out, max_workers=workers, describe=_describe,
                  tag=f"stage2-oprate/{ds}/{sel}")


if __name__ == "__main__":
    main()
