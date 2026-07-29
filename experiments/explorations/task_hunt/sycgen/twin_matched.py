"""TWIN AT MATCHED BUDGET — the hub's named next step for the shuffle lane.

PRE-REGISTERED BEFORE RUNNING (commit-then-run; git history is the
receipt). Follows `SHUFFLE_MATCHED_CARD.md`; this file adds one
control and changes no frozen rule.

## Why this exists

The shuffle lane returned **(b) architectural, not learned**: untrained
twins showed the LARGER ordered−shuffled gap in 11/12 cells. I disclosed
a budget confound — the twin runs at `l0`=8.00 (every `k_pos` slot live)
against the trained model's 5.44–7.86, up to **1.47x**.

**The hub then cross-checked the confound against the effect and it
TRACKS:** Spearman(budget advantage, twin excess gap) = **+0.80**, and
the twin excess **REVERSES at T=16**, the one T where budgets nearly
match (1.02x). *That is what a budget artifact looks like.*

So the confound is not a footnote — it is a live alternative explanation
for the twin result, and it must be removed rather than disclosed.

## What is and is not in question

**Still defensible without this run** (hub ruling): the ordered−shuffled
gap is **not evidence of LEARNED temporal structure**. A random model
produces a gap of the same order, so the gap alone cannot carry the
claim. Nothing here can rescue that.

**NOT yet defensible**: that **training REDUCES order-sensitivity.**
That reading needs the twin measured at the trained model's budget, and
that is exactly what this script does.

## Method

For each (T, seed): read the trained model's realized
`l0_per_window`, then sweep the twin's `k_pos` to BRACKET it, and
compare gaps at matched budget under the frozen §2b bracket rule
(below + cheapest above + the ratio printed; if the ends disagree that
IS the finding).

## PRE-REGISTERED OUTCOMES — fixed before any number exists

- **(i) twin gap still >= trained gap at MATCHED budget** ⇒ the twin
  result was NOT a budget artifact. (b) stands as delivered, and the
  stronger reading ("training reduces order-sensitivity") becomes
  defensible.
- **(ii) twin excess VANISHES or REVERSES at matched budget** ⇒ the
  twin result WAS substantially a budget artifact. **(b) survives only
  in its weak form** — the gap is not evidence of learned structure,
  because a random model still shows one — and **the claim that
  training reduces order-sensitivity is WITHDRAWN, not softened.**
  Given the hub's +0.80 and the T=16 reversal, **this is the expected
  outcome and I am saying so before the run.**
- **(iii) mixed across T** ⇒ reported per T, not aggregated into a
  headline.

**No frozen rule moves.** The 3/3 sign test, the noise band, and the
bracket rule are unchanged; only the twin's operating point does.

    .venv/bin/python -m experiments.explorations.task_hunt.sycgen.twin_matched
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import numpy as np
import torch

from experiments.explorations.task_hunt.sycgen.frontier import MeasuredArm
from experiments.explorations.task_hunt.sycgen.run_shuffle_matched import (
    DRAWS,
    EVAL_L,
    N_WINDOWS,
    SHUF_EVAL_SEED,
    SEEDS,
    TS,
    _build,
    _gate_shuffle_live,
    _score,
    _shuffle,
    _tiles,
    _key_from_leaderboard,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "twin_matched.json"

# The twin's realized l0 tracks k_pos almost exactly (a random BatchTopK
# fires every slot), so this brackets the trained models' 5.44-7.86.
TWIN_KPOS = (3, 4, 5, 6, 7, 8)


def _build_twin(T: int, seed: int, ds_spec, k_pos: int):
    """Random-init TXC at a chosen k_pos. Same spec, same seed as §3."""
    from temp_bench.core.config import load_arch
    from temp_bench.core.runner import import_by_path
    from temp_bench.core.trainer import _infer_d_in
    spec = load_arch("txc_batchtopk_post_btkonly")
    spec = spec.model_copy(update={
        "hparams": {**spec.hparams, "d_sae": 2048, "T": T, "k_pos": k_pos}})
    torch.manual_seed(int(seed))
    m = import_by_path(spec.class_path)(d_in=_infer_d_in(ds_spec), **spec.hparams)
    m.eval()
    return m


def _bracket_twin(cands, target):
    """§2b applied to the twin: below + cheapest above + printed ratio."""
    below = [c for c in cands if c["l0"] <= target]
    above = [c for c in cands if c["l0"] > target]
    lo = max(below, key=lambda c: c["l0"]) if below else None
    hi = min(above, key=lambda c: c["l0"]) if above else None
    out = {"below": lo, "above": hi}
    if lo and hi and hi["l0"] > lo["l0"]:
        w = (target - lo["l0"]) / (hi["l0"] - lo["l0"])
        out["interp_gap"] = lo["gap"] + w * (hi["gap"] - lo["gap"])
        out["width"] = (hi["l0"] - lo["l0"]) / target
    elif lo or hi:
        c = lo or hi
        out["interp_gap"] = c["gap"]
        out["width"] = None
    return out


def main() -> int:
    from temp_bench.core.config import load_datasource
    from temp_bench.data.synthetic import materialise
    from temp_bench.evals.synthetic_recovery import _sample_windows
    import experiments.explorations.task_hunt.sycgen.run_retrain as RR

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds_spec = load_datasource(RR.DS)
    data = materialise(ds_spec, seed=0)
    lam = data.extra["lambda_labels"]
    if not torch.is_tensor(lam):
        lam = torch.as_tensor(lam)
    x, lam = data.x, lam.float()
    print(f"[twin] device={device} x={tuple(x.shape)}", flush=True)

    rows = []
    for T in TS:
        for seed in SEEDS:
            lam3 = lam.reshape(lam.shape[0], lam.shape[1], 1)
            n = x.shape[0]
            split = n // 2
            wx_tr, _ = _sample_windows(x[:split], L=EVAL_L, n_windows=N_WINDOWS, seed=seed)
            wl_tr, _ = _sample_windows(lam3[:split], L=EVAL_L, n_windows=N_WINDOWS, seed=seed)
            wx_ev, _ = _sample_windows(x[split:], L=EVAL_L, n_windows=N_WINDOWS, seed=seed + 1)
            wl_ev, _ = _sample_windows(lam3[split:], L=EVAL_L, n_windows=N_WINDOWS, seed=seed + 1)
            tiles_tr, t_tr = _tiles(wx_tr, wl_tr, T, device)
            tiles_ev, t_ev = _tiles(wx_ev, wl_ev, T, device)
            fin_tr, fin_ev = np.isfinite(t_tr), np.isfinite(t_ev)

            for draw in DRAWS:
                tiles_sh = _shuffle(tiles_ev, T, SHUF_EVAL_SEED, draw)
                sh_tr = _shuffle(tiles_tr, T, SHUF_EVAL_SEED, draw)
                _gate_shuffle_live(tiles_ev, tiles_sh, T, draw)   # gate first, always

                raw, tk = _build("txc_batchtopk_post_btkonly", T, seed,
                                 ds_spec, trained=True)
                m = _score(lambda: MeasuredArm(raw.to(device)), tiles_tr, t_tr,
                           tiles_ev, t_ev, tiles_sh, sh_tr, fin_tr, fin_ev)
                rows.append({"arm": "txc_trained", "T": T, "seed": seed,
                             "draw": draw, "k_pos": 8, "train_key": tk, **m})
                print(f"  trained T{T} s{seed} {draw:7s} l0="
                      f"{m['realized_l0_per_window_ordered']:.3f} "
                      f"gap={m['gap_fixedprobe']:+.4f}", flush=True)

                for kp in TWIN_KPOS:
                    tw = _build_twin(T, seed, ds_spec, kp).to(device)
                    mt = _score(lambda: MeasuredArm(tw), tiles_tr, t_tr,
                                tiles_ev, t_ev, tiles_sh, sh_tr, fin_tr, fin_ev)
                    rows.append({"arm": "txc_twin", "T": T, "seed": seed,
                                 "draw": draw, "k_pos": kp, **mt})
                    print(f"    twin  k_pos={kp} l0="
                          f"{mt['realized_l0_per_window_ordered']:.3f} "
                          f"gap={mt['gap_fixedprobe']:+.4f}", flush=True)

                OUT.parent.mkdir(parents=True, exist_ok=True)
                OUT.write_text(json.dumps({"rows": rows}, indent=1))

    report(rows)
    print(f"[twin] {len(rows)} rows -> {OUT}")
    return 0


def report(rows) -> None:
    print("\n=== TWIN AT MATCHED BUDGET (§2b bracket, ratio printed) ===")
    print(f"{'T':>3} {'draw':7s} {'trained l0':>10} {'trained gap':>11} "
          f"{'twin gap @matched':>17} {'excess':>8} {'ratio':>6}")
    for draw in DRAWS:
        for T in TS:
            exc, ratios = [], []
            for seed in SEEDS:
                tr = [r for r in rows if r["arm"] == "txc_trained" and r["T"] == T
                      and r["seed"] == seed and r["draw"] == draw]
                tw = [r for r in rows if r["arm"] == "txc_twin" and r["T"] == T
                      and r["seed"] == seed and r["draw"] == draw]
                if not tr or not tw:
                    continue
                tgt = tr[0]["realized_l0_per_window_ordered"]
                cands = [{"l0": r["realized_l0_per_window_ordered"],
                          "gap": r["gap_fixedprobe"], "k": r["k_pos"]} for r in tw]
                br = _bracket_twin(cands, tgt)
                if "interp_gap" not in br:
                    continue
                exc.append(br["interp_gap"] - tr[0]["gap_fixedprobe"])
                used = br["below"] or br["above"]
                ratios.append(used["l0"] / tgt)
                if seed == SEEDS[0]:
                    print(f"{T:>3} {draw:7s} {tgt:>10.3f} "
                          f"{tr[0]['gap_fixedprobe']:>+11.4f} "
                          f"{br['interp_gap']:>+17.4f} "
                          f"{exc[-1]:>+8.4f} {ratios[-1]:>6.3f}")
            if exc:
                pos = sum(1 for e in exc if e > 0)
                print(f"    -> T{T} {draw}: mean twin excess {st.mean(exc):+.4f}, "
                      f"twin>trained in {pos}/{len(exc)} seeds")


if __name__ == "__main__":
    raise SystemExit(main())
