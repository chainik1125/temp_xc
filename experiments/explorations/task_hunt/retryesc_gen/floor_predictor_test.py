"""Why `floor_excess == f` broke, and what to aim with instead. $0, no GPU.

**Result up front.** The identity the program has been aiming with —
`floor_excess == f == P(event_first inside the T-window)` — is a
LOW-DENSITY APPROXIMATION, not a law. It held on `evalage` and broke on
`retryesc_gen`, and the +0.0755 miss that cost item 7 its KEEP is that
break, not an estimator bias. The replacement, validated on both corpora
the program has (6 legs), is:

    floor_excess ~= P(any MASKED EVENT token inside the trailing T-window)

which is `f` computed against an EFFECTIVE window of `T + w`, where `w` is
the masked event-turn width. `evalage` w=13, `retryesc_gen` w=25 — that
difference is the whole story.

## Two hypotheses, tested in order

**H1 (mine, 16:05) — row population. REFUTED.** I proposed that
`claim_zone` reads the raw eligible population while the floor is fit on
the class-balanced manifest. `manifest_f_test.py` walks the population
from one to the other a filter at a time; every rung is flat and the full
walk moves gpt2's f from 0.1853 to 0.1875, closing 2.8% of a 0.0755 gap.
The instrument and the bar DO disagree, but not because of the rows.

**H2 — the floor sees a wider window than `f` assumes. CONFIRMED.**
`_FloorBank.feats` is `(sage_floor, dose_window_count)` and
`dose_window_count` is fed **`event_mask`, not `event_first`**. The mask
spans the whole event TURN, so a row whose `event_first` is older than T
still has masked event tokens inside its trailing T-window. The floor's
in-window indicator is therefore `P(any masked token in window)`, which
exceeds `f` by roughly the turn width.

**Why this is the safe direction to aim with:** the oracle indicator is
an UPPER bound on what a fitted 2-feature probe extracts, so it
over-predicts `floor_excess` slightly (5 of 6 legs) — a design instrument
that errs toward "denser than you think" is the one you want when the
upper band edge is what kills candidates.

Supersedes the "claim_zone under-reads, add ~0.076" correction in my
19:11 beat: that constant was fit to a residual whose mechanism I had
wrong. This measures the mechanism instead.

Run: .venv/bin/python -m experiments.explorations.task_hunt.retryesc_gen.floor_predictor_test
Writes results/floor_predictor_test.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.explorations.task_hunt.labels import wave3_lib as w3
from experiments.explorations.task_hunt.labels.build_retryesc_gen_premeasure import (
    section_age,
)
from experiments.explorations.task_hunt.labels.build_wave3_trio import (
    MIN_POS as PRE_MIN_POS,
)

ROOT = Path(__file__).resolve().parent.parent
T = 64
CHANCE = 1.0 / 3.0

# (corpus, grid dir, grid stem, {tokenizer_tag: measured floor acc @T64})
# Floor accuracies read from the committed screen artifacts of each lane.
CORPORA = [
    ("evalage", ROOT / "evalage" / "grids", "elicit_evalage_screen",
     {"gpt2": 0.3814, "gemma2": 0.3972, "llama31": 0.3871}),
    ("retryesc_gen", ROOT / "retryesc_gen" / "grids",
     "elicit_retryesc_gen_v1_screen",
     {"gpt2": 0.5941739, "gemma2": 0.6083425, "llama31": 0.6218663}),
]


def masked_turn_width(mask, off, n_docs) -> float:
    runs = []
    for d in range(n_docs):
        m = mask[off[d]:off[d + 1]].astype(np.int64)
        if not m.any():
            continue
        e = np.diff(np.concatenate([[0], m, [0]]))
        runs.extend((np.flatnonzero(e == -1) - np.flatnonzero(e == 1)).tolist())
    return float(np.median(runs)) if runs else float("nan")


def leg(grid_dir: Path, stem: str, tag: str, floor_acc: float) -> dict:
    z = np.load(grid_dir / f"{stem}_{tag}.npz")
    off, first, mask = z["doc_off"], z["event_first"], z["event_mask"]
    is_assist = z["is_assistant"]
    n_docs = len(off) - 1

    age = np.concatenate([w3.sage_face(first[off[d]:off[d + 1]])
                          for d in range(n_docs)])
    raw = np.concatenate([section_age(first[off[d]:off[d + 1]])
                          for d in range(n_docs)]).astype(np.float64)
    dose = np.concatenate([w3.dose_window_count(mask[off[d]:off[d + 1]], T)
                           for d in range(n_docs)])

    doc_of = np.searchsorted(off, np.arange(len(raw)), side="right") - 1
    pos_of = np.arange(len(raw)) - off[doc_of]

    # ONE population definition for every corpus and every leg — the
    # point of the exercise is that instrument and bar share rows.
    elig = ((mask == 0) & (is_assist == 1) & (pos_of >= PRE_MIN_POS)
            & np.isfinite(age))

    f = float((raw[elig] <= T).mean())          # the OLD instrument
    pm = float((dose[elig] > 0).mean())         # the NEW instrument
    fe = floor_acc - CHANCE                     # the BAR (measured)
    w = masked_turn_width(mask, off, n_docs)
    return {"tokenizer": tag, "n_elig": int(elig.sum()),
            "masked_turn_width_median": w, "effective_window": T + w,
            "f_event_first": f, "p_masked_in_window": pm,
            "measured_floor_excess": fe,
            "resid_f": f - fe, "resid_masked": pm - fe}


def main() -> None:
    out, rows = [], []
    for name, gdir, stem, floors in CORPORA:
        legs = [leg(gdir, stem, tag, acc) for tag, acc in floors.items()]
        out.append({"corpus": name, "legs": legs})
        rows.extend((name, L) for L in legs)

    hdr = (f"{'corpus':<14}{'leg':<9}{'w':>4}{'f (OLD)':>10}"
           f"{'P(masked) NEW':>15}{'measured fe':>13}{'resid f':>10}"
           f"{'resid NEW':>11}")
    print(hdr)
    print("-" * len(hdr))
    for name, L in rows:
        print(f"{name:<14}{L['tokenizer']:<9}{L['masked_turn_width_median']:>4.0f}"
              f"{L['f_event_first']:>10.4f}{L['p_masked_in_window']:>15.4f}"
              f"{L['measured_floor_excess']:>13.4f}{L['resid_f']:>+10.4f}"
              f"{L['resid_masked']:>+11.4f}")

    af = float(np.mean([abs(L["resid_f"]) for _, L in rows]))
    am = float(np.mean([abs(L["resid_masked"]) for _, L in rows]))
    xf = float(np.max([abs(L["resid_f"]) for _, L in rows]))
    xm = float(np.max([abs(L["resid_masked"]) for _, L in rows]))
    print(f"\n  OLD  f = P(event_first <= T) : mean|resid| {af:.4f}  max {xf:.4f}")
    print(f"  NEW  P(masked token in win) : mean|resid| {am:.4f}  max {xm:.4f}"
          f"   ({100 * (1 - am / af):.0f}% error reduction, {len(rows)} legs)")

    summ = {"T": T, "chance": CHANCE, "corpora": out,
            "mean_abs_resid_f": af, "mean_abs_resid_masked": am,
            "max_abs_resid_f": xf, "max_abs_resid_masked": xm,
            "n_legs": len(rows)}
    p = ROOT / "retryesc_gen" / "results" / "floor_predictor_test.json"
    p.write_text(json.dumps(summ, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
