"""Lever 3, the `w` half — the number that decides whether to spend money.

`lever3_test.py` swept T on the best configuration available (gemma2_2b,
512 context) and returned **0/5 cells satisfying both criteria**:

    T     arm-floor    gain-0.05     min(both)
    4      +0.0360      -0.0400       -0.0400
    8      +0.0267      -0.0291       -0.0291
    16     -0.0107      -0.0249       -0.0249   <- closest
    32     -0.0599      +0.0010       -0.0599
    64     -0.1175      +0.0588       -0.1175

T alone cannot close it: where the arm beats the floor the gain is absent,
and where the gain arrives the floor has overtaken. **The best cell is
0.0249 short.** The remaining half of lever 3 is `w` — the masked
event-TURN width, fixed at 25 by this corpus and only changeable by
GENERATING a new one at ~$21.

So: **would a narrower event actually close 0.0249?** The floor is
ground-truth-derived, so this is computable for a counterfactual `w`
WITHOUT generating anything, by truncating each masked run to its first
`w'` tokens and refitting only the floor.

## What this is and is not

**Honest framing, stated before the numbers.** The floor is refit under
the counterfactual; **the arm is held FIXED at its measured value.** In a
real corpus with narrower turns the text would differ, so the activations
and therefore the arm would differ too. This estimates the `w` lever's
benefit **holding the arm constant**, which is optimistic — a real
narrow-turn corpus would also have less in-window event text for the arm
to read, so the true benefit is likely SMALLER than this.

**That asymmetry is the point.** If even this optimistic estimate does not
clear the bar, generating the corpus is not justified, and that is a clean
$0 no-go. If it clears comfortably, the spend has a measured basis rather
than an argument.

Note the floor has TWO features and only one depends on `w`:
`sage_floor(event_first, T)` gives the exact age whenever the event is
inside T and is untouched by narrowing; `dose_window_count(event_mask, T)`
is the one that reaches out to T + w. So the floor cannot fall to chance
however narrow the event gets — it falls toward what censored age alone
supports, which is itself a useful bound to know.

Run: PYTHONPATH=. python -m experiments.explorations.task_hunt.facecmp.lever3_w_counterfactual
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments.explorations.conversion_depth.problib import fit_probe
from experiments.explorations.task_hunt.labels import wave3_lib as w3

TS = [4, 8, 16, 32, 64]
WS = [25, 16, 8, 4, 2, 1]
CHANCE = 1.0 / 3.0


def narrow_mask(mask, off, n_docs, w_new):
    """Truncate every masked RUN to its first w_new tokens, per document."""
    out = np.zeros_like(mask)
    for d in range(n_docs):
        seg = mask[off[d]:off[d + 1]].astype(np.int64)
        if not seg.any():
            continue
        e = np.diff(np.concatenate([[0], seg, [0]]))
        starts = np.flatnonzero(e == 1)
        ends = np.flatnonzero(e == -1)
        buf = np.zeros(len(seg), dtype=mask.dtype)
        for s, t in zip(starts, ends):
            buf[s:min(s + w_new, t)] = 1
        out[off[d]:off[d + 1]] = buf
    return out


def main():
    import experiments.explorations.task_hunt.facecmp.arm_test as at
    import experiments.explorations.task_hunt.facecmp.face_battery as fb

    root = os.environ.get("FACECMP_CACHE_ROOT")
    if root:
        at.CACHE_ROOT = Path(root)
    key = os.environ.get("FACECMP_MODEL", "gemma2_2b")
    tag = os.environ.get("FACECMP_TAG", "gemma2_512")

    # measured arms from the frozen lever3 sweep — NOT recomputed here
    src = (Path(__file__).resolve().parent / "results" / "lever3"
           / f"lever3_{tag}.json")
    prev = json.loads(src.read_text())
    arm = {r["T"]: r["arm"] for r in prev["rows"]}
    tok = prev["tok"]
    print(f"arms held FIXED from {src.name}; tok={tok:.4f}\n")

    at.FACE, at.H = f"L3W_{tag}", 64
    at.rate_face = fb.f_age
    manifests, mstats, fl = at.build_rows(key)
    F_ = at.FACE
    _, ytr = manifests[(F_, "train")]
    _, yte = manifests[(F_, "test")]
    ftr = manifests[(f"{F_}_flat", "train")]
    fte = manifests[(f"{F_}_flat", "test")]
    ytr_t, yte_t = torch.from_numpy(ytr), torch.from_numpy(yte)
    first, mask, off, n_docs = fl["first"], fl["mask"], fl["off"], fl["n_docs"]

    rows = []
    print(f"{'w':>4}{'T':>5}{'floor':>9}{'arm-floor':>11}{'gain':>9}"
          f"{'gain-0.05':>11}{'min(both)':>11}{'KEEP?':>7}")
    print("-" * 68)
    for w_new in WS:
        m2 = mask if w_new == 25 else narrow_mask(mask, off, n_docs, w_new)
        for T in TS:
            cage = np.concatenate([w3.sage_floor(first[off[d]:off[d + 1]], T)
                                   for d in range(n_docs)])
            cnt = np.concatenate([w3.dose_window_count(m2[off[d]:off[d + 1]], T)
                                  for d in range(n_docs)])

            def feats(idx):
                f = np.stack([np.nan_to_num(cage[idx].astype(np.float32)),
                              cnt[idx].astype(np.float32)], 1)
                return torch.from_numpy(f).to(torch.float16)

            r = fit_probe(feats(ftr), ytr_t, feats(fte), yte_t, 3)
            fl_acc = float(r["acc_test"])
            a = arm[T]
            gain = a - tok
            both = min(a - fl_acc, gain - 0.05)
            keep = (a - fl_acc) > 0 and gain >= 0.05
            rows.append({"w": w_new, "T": T, "floor": fl_acc, "arm": a,
                         "arm_minus_floor": a - fl_acc, "gain": gain,
                         "min_both": both, "keep_shaped": bool(keep)})
            print(f"{w_new:>4}{T:>5}{fl_acc:>9.4f}{a - fl_acc:>+11.4f}"
                  f"{gain:>+9.4f}{gain - 0.05:>+11.4f}{both:>+11.4f}"
                  f"{('YES' if keep else '-'):>7}")
        print()

    keeps = [r for r in rows if r["keep_shaped"]]
    best = max(rows, key=lambda r: r["min_both"])
    print(f"KEEP-shaped cells: {len(keeps)}/{len(rows)}")
    if keeps:
        for r in keeps:
            print(f"   w={r['w']} T={r['T']}  arm-floor {r['arm_minus_floor']:+.4f}"
                  f"  gain {r['gain']:+.4f}")
    print(f"best by min(both): w={best['w']} T={best['T']} -> {best['min_both']:+.4f}")
    print("\n⚠ arm held FIXED under the counterfactual, so this is an "
          "OPTIMISTIC estimate of the w lever (a real narrow-turn corpus "
          "would give the arm less to read too).")

    d = Path(__file__).resolve().parent / "results" / "lever3"
    (d / f"w_counterfactual_{tag}.json").write_text(json.dumps(
        {"model": key, "tag": tag, "tok": tok, "arms_from": src.name,
         "arm_held_fixed": True, "rows": rows, "keep_shaped": keeps,
         "best": best}, indent=2))
    print(f"wrote {d / f'w_counterfactual_{tag}.json'}")


if __name__ == "__main__":
    main()
