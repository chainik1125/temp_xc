"""Stage 2 seed top-up, tsae arm — the 3 cells runpod-d could not afford.

Provenance chain, all frozen BEFORE this run:

- runpod-b's variance criterion (LOG 2026-07-24, `support_stats/`):
  one-sided 95% t lower bound > 0 on the pre-vs-tsae T8 margin, with
  sign-flip attainability needing n >= 5 → 6 seeds total ⇒ extra seeds
  (3, 4, 5) × {pre/T4, pre/T8, tsae/T1}.
- runpod-d executed that spec at freeze `3d954869`
  (`run_stage2_seedtopup.py`) and DELIVERED 6/9: both pre arms landed
  at n = 6; the three tsae/T1 cells were abandoned after a measured
  cost diagnosis (`ActivationBuffer._refill` re-gathers an 8.6 GB CPU
  buffer ~31×/cell at buffer_tokens = 524288; GPU at 0%; multi-hour
  cells) — LOG 2026-07-24 "pre arm DELIVERED at n=6, tsae arm NOT
  AFFORDABLE". RECEIPTS R5 therefore records the margin as NOT
  BOUNDED, with the binding constraint being the tsae arm's n
  (projected Welch LB ≈ +0.013 at tsae n = 6, sd held).
- This runner is that SAME frozen spec restricted to the tsae arm:
  the 3 remaining cells, nothing else. `briefings/overnight-mac-a.md`
  § 2 authorizes exactly this set (executor mac-a, Modal venue). Not
  "seeds until significant": the union {3,4,5} completes b's frozen
  6-seed design and each seed runs exactly once.

Config is byte-for-byte the round-1 / top-up panel config: nominal
k = 8, d_sae = 2048, eval_window_L = 32, n_steps = 8000, and
**buffer_tokens UNCHANGED at 524288** — d's diagnosis identified
shrinking it as the available "fix" and refused it because it changes
`train_key` and destroys comparability with the round-1 tsae seeds;
that refusal is binding here too. The cost is paid with wall-clock on
high-CPU containers instead.

`--only-seed N` (optional) runs the single cell for one member of the
frozen seed tuple — a container-partitioning device so the 3 cells can
run in 3 parallel containers; it cannot enlarge or reorder the frozen
set. Output files are per-seed in that mode so transcripts never
clobber. Merge into the round-1 panel file is by cell identity, and
the canonical leaderboard dedups on eval_key, so re-running is
idempotent there (same semantics as `run_stage2_seedtopup.py`).

Pooling hazards (briefing § 2) are discharged OUTSIDE this file and
documented in the LOG: cache byte-identity receipts
(`ward_stream_stats.json`, `lambda_labels_stats.json` reproduced
git-clean in-container BEFORE training) + the lambda_recovery code
audit since `038655fd` (the fff7877c NaN guard, verified a strict
no-op on this datasource's all-finite `lam_hist_dense` labels —
asserted again in-container).

Run (repo root, all 3 cells):
  .venv/bin/python -m \
    experiments.explorations.task_hunt.lambda_intensity.run_stage2_seedtopup_tsae [workers]
One cell (container partitioning):
  ... run_stage2_seedtopup_tsae 1 --only-seed 4
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from explorations.synthetic import grid

DS = "ward_real_lambda_base_l12"
D_SAE = 2048
K_POS = 8
EVAL_L = 32
N_STEPS = 8_000
BUFFER_TOKENS = 524_288          # UNCHANGED from round 1 — comparability is the point
EXTRA_SEEDS = (3, 4, 5)          # completes runpod-b's frozen 6-seed design
# (arch, T) — the tsae arm only; the pre arms landed 2026-07-24.
CELLS_AT = (("tsae", 1),)
HERE = Path(__file__).resolve().parent
PANEL_FILE = HERE / "results" / f"stage2_{DS}.json"


def _cells(only_seed: int | None = None):
    out = []
    for seed in EXTRA_SEEDS:
        if only_seed is not None and seed != only_seed:
            continue
        for arch, T in CELLS_AT:
            out.append({"ds": DS, "arch": arch, "T": T, "d_sae": D_SAE,
                        "k_pos": K_POS, "seed": seed, "n_steps": N_STEPS,
                        "kind": "trained", "eval_window_L": EVAL_L,
                        "buffer_tokens": BUFFER_TOKENS})
    return out


def _key(c):
    return (c["arch"], c["T"], c["d_sae"], c["k_pos"], c["seed"],
            c["n_steps"], c.get("kind"))


def _merge_into_panel(new_results):
    """Append the top-up cells to the round-1 panel file, dedup by cell id.

    The round-1 file is the render cache, not a canonical artifact (the
    leaderboard is canonical and dedups on eval_key). Appending more seeds
    of the same design is exactly what the renderer expects.
    """
    existing = json.loads(PANEL_FILE.read_text()) if PANEL_FILE.exists() else []
    by_key = {_key(r): r for r in existing}
    added = 0
    for r in new_results:
        if not r.get("ok"):
            continue
        if _key(r) not in by_key:
            added += 1
        by_key[_key(r)] = r
    merged = list(by_key.values())
    tmp = PANEL_FILE.with_name(PANEL_FILE.name + ".tmp")
    tmp.write_text(json.dumps(merged, indent=2))
    tmp.replace(PANEL_FILE)
    print(f"[merge] panel now {len(merged)} cells (+{added} new)", flush=True)


def _describe(res):
    m = res["metrics"]
    return (f"λ={m.get('lambda_recovery', float('nan')):.3f} "
            f"l0t={m.get('l0_per_token', float('nan')):.2f}")


def main():
    argv = list(sys.argv[1:])
    only_seed = None
    if "--only-seed" in argv:
        i = argv.index("--only-seed")
        only_seed = int(argv[i + 1])
        if only_seed not in EXTRA_SEEDS:
            raise SystemExit(f"--only-seed must be one of {EXTRA_SEEDS}")
        del argv[i:i + 2]
    workers = int(argv[0]) if argv else 3
    cells = _cells(only_seed)
    suffix = f"_s{only_seed}" if only_seed is not None else ""
    out = HERE / "results" / f"stage2_seedtopup_tsae_{DS}{suffix}.json"
    results = grid.run_pool(cells, out, max_workers=workers, describe=_describe,
                            tag=f"stage2-seedtopup-tsae/{DS}{suffix}")
    _merge_into_panel(results)


if __name__ == "__main__":
    main()
