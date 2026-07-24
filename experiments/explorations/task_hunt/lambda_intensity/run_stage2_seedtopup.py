"""Stage 2 seed top-up — runpod-b's variance recommendation, executed.

runpod-b's variance receipts (LOG 2026-07-24, `support_stats/`) found the
cross-arch **TXC-pre − T-SAE** paired margin at the T8 headline cell NOT
bounded at n = 3 (T8 = 0.052 ± 0.055, t CI [−0.086, 0.190]). b froze the
top-up spec and addressed it to runpod-d:

  criterion = one-sided 95% t lower bound > 0 on the paired pre-vs-tsae
  diff at the T8 headline cell, plus sign-flip attainability (2⁻ⁿ ≤ 0.05
  needs n ≥ 5) → **6 seeds total ⇒ 3 extra seeds (3, 4, 5) ×
  {txc_batchtopk_pre/T4, txc_batchtopk_pre/T8, tsae/T1} = 9 trained
  cells**.

My briefing (`briefings/task-hunt-r2-d.md` § 1) pre-authorized exactly
this ("treat it as part of this run if it lands"). This runs the EXACT
9 cells b specified — not "seeds until significant"; the cell list is
frozen in code before the run and the criterion was frozen by b before
these seeds existed. Same config as the round-1 panel (nominal k = 8,
d_sae = 2048, eval_window_L = 32, n_steps = 8000, buffer 524288), so the
new seeds are drop-in additions to
`results/stage2_ward_real_lambda_base_l12.json` — the renderer then
aggregates pre/T4, pre/T8, tsae/T1 at n = 6 and every other cell stays
n = 3 (b's CI machinery handles per-cell n). Merge is by cell identity
(arch, T, d_sae, k_pos, seed, n_steps, kind); the leaderboard dedups on
eval_key, so re-running is idempotent there.

Untrained counterparts are intentionally NOT run (b: "optional — margin
receipts already bind at n = 3"); the top-up target is the trained-only
cross-arch diff.

Run:  .venv/bin/python -m \
        experiments.explorations.task_hunt.lambda_intensity.run_stage2_seedtopup [workers]
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
BUFFER_TOKENS = 524_288
EXTRA_SEEDS = (3, 4, 5)
# (arch, T) — b's exact top-up set.
CELLS_AT = (("txc_batchtopk_pre", 4), ("txc_batchtopk_pre", 8), ("tsae", 1))
HERE = Path(__file__).resolve().parent
PANEL_FILE = HERE / "results" / f"stage2_{DS}.json"


def _cells():
    out = []
    for seed in EXTRA_SEEDS:
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
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    cells = _cells()
    out = HERE / "results" / f"stage2_seedtopup_{DS}.json"
    results = grid.run_pool(cells, out, max_workers=workers, describe=_describe,
                            tag=f"stage2-seedtopup/{DS}")
    _merge_into_panel(results)


if __name__ == "__main__":
    main()
