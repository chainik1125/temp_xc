"""Stage 2 — `punctint` q: the fineweb TXC panel (CARD_STAGE2.md, frozen
in the same commit as this runner; briefings/stage2-fineweb.md).

The λ̂ Stage-2 pattern (`lambda_intensity/run_stage2.py`) on a second
corpus, with the card's three deviations built in from the start:

- **TXC-post at per-T nominal k = 8·T** (trained AND untrained) — the
  code-rate convention (`card_stage2_postmatched.md` § 2) adopted
  up-front, so post's realized l0/token matches the panel's ≈ 8 instead
  of ramping down as 8/T. Untrained post realizing exactly 8.00 ± 0.02
  is the card's § 5 falsifier.
- **Both probe columns on every cell** (`PROBE_V2_SPEC.md` § 2 flags in
  `eval_extra`; claim on v1).
- **tsae trained cells submitted first** (the long pole — addendum 2),
  buffer_tokens = the corpus exactly, uniform across archs.

Full panel (84 cells) on the primary datasource; `--replicate=T1,T2`
runs the cross-model replication subset (TXC-pre at the two named T
values + tsae + batchtopk_sae, trained + untrained, 3 seeds = 24
cells) on a replication datasource.

Run:  .venv/bin/python -m \
        experiments.explorations.task_hunt.qrate_fineweb.run_stage2 \
        [workers] [ds] [--replicate=T1,T2]
"""

from __future__ import annotations

import sys
from pathlib import Path

from explorations.synthetic import design, grid

DS_DEFAULT = "fineweb_punctint_q_gemma2_l14"
# d_sae = d_in/2 (scarce anchor) and buffer_tokens = corpus size, per ds.
D_SAE = {"fineweb_punctint_q_gemma2_l14": 1152,
         "fineweb_punctint_q_gpt2_l7": 384,
         "fineweb_punctint_q_llama31_l14": 2048}
BUFFER = {"fineweb_punctint_q_gemma2_l14": 766_080,
          "fineweb_punctint_q_gpt2_l7": 766_592,
          "fineweb_punctint_q_llama31_l14": 758_272}
K_POS = (8,)
WINDOW_TS = (2, 4, 8, 16)
EVAL_L = 32
N_STEPS = 8_000
HERE = Path(__file__).resolve().parent

PANEL = (
    ("batchtopk_sae", "token"),
    ("tsae", "token"),
    ("stacked_batchtopk", "stacked"),
    ("txc_batchtopk_pre", "pre"),
    ("txc_batchtopk_post", "post"),
)
REPLICATION_ARCHS = (
    ("batchtopk_sae", "token"),
    ("tsae", "token"),
    ("txc_batchtopk_pre", "pre"),
)
V2 = {"lambda_probe_v2": True, "lambda_v2_probe": "ridge",
      "lambda_v2_n_windows": 8192, "lambda_v2_split": "trace"}


def _cells(ds: str, archs, window_ts):
    cells = design.uniform_cells(
        ds, F=D_SAE[ds], n_steps=N_STEPS, d_saes=[D_SAE[ds]],
        k_pos_sweep=K_POS, archs=archs, window_ts=window_ts, L=EVAL_L,
        untrained_kpos=K_POS[0], log=print)
    for c in cells:
        c["buffer_tokens"] = BUFFER[ds]
        c["eval_extra"] = dict(V2)
        # Binding 2 / card § 4: post budgets its k per WINDOW, so the
        # matched code rate needs nominal k = 8·T — both kinds, so the
        # untrained falsifier (realized l0/token = 8.00 exactly) applies.
        if c["arch"] == "txc_batchtopk_post":
            c["k_pos"] = K_POS[0] * c["T"]
    # tsae trained first (the long pole), then everything else in
    # design order; ProcessPoolExecutor starts futures in submission
    # order, so list order is schedule order.
    cells.sort(key=lambda c: 0 if (c["arch"] == "tsae"
                                   and c["kind"] == "trained") else 1)
    return cells


def _describe(res):
    m = res["metrics"]
    return (f"λ={m.get('lambda_recovery', float('nan')):.3f} "
            f"λv2={m.get('lambda_recovery_v2', float('nan')):.3f} "
            f"chance={m.get('lambda_chance', float('nan')):+.3f} "
            f"l0t={m.get('l0_per_token', float('nan')):.2f}")


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    ds = DS_DEFAULT
    rep_ts = None
    for a in sys.argv[2:]:
        if a.startswith("--replicate="):
            rep_ts = tuple(int(t) for t in a.split("=", 1)[1].split(","))
        else:
            ds = a
    if rep_ts is None:
        cells = _cells(ds, PANEL, WINDOW_TS)
        tag = f"stage2/{ds}"
    else:
        cells = _cells(ds, REPLICATION_ARCHS, rep_ts)
        tag = f"stage2rep/{ds}"
    out = HERE / "results" / f"stage2_{ds}.json"
    grid.run_pool(cells, out, max_workers=workers, describe=_describe,
                  tag=tag)


if __name__ == "__main__":
    main()
