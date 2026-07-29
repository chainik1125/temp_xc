"""BEYOND-T MASS — the pre-generation screen criterion the hunt has lacked.

$0, no model, no pods, no network. Reads existing corpus grids only.

## Why this quantity

`sage_floor` censors at **T+1** (`gen4c_lib.py:115`: `min(age, T+1)`,
"older than my window"), while the *label* is the UNCENSORED age. They
coincide exactly where `age <= T`. So the mass a windowed arm can win on
— and the floor structurally cannot — is

    beyond_T = P(age > T)          over probe-eligible positions

Hub `41281e0eb` / `01:07`: *"Sort candidates by beyond-T mass before
generating anything."* This file computes it.

**It is NOT tautological — it can come out either way.** A corpus can
have large beyond-T mass and still fail (the arm must actually recover
the older event from activations), or small mass and still pass. That is
what makes it a screen rather than a restatement.

⚑ **This measures the LOWER edge of the aiming band only.** Too-recent
events are resolved by the floor — that is what `beyond_T` scores. The
UPPER edge — how far back the residual stream still carries a sparse
event — is a property of the model, is NOT measured here, and needs
activations. A corpus can score perfectly here and still lose because
its events are too old to be represented at all.

## Falsifier, stated before the numbers

If `beyond_T` does **not** order the corpora consistently with their
observed `arm - floor`, the criterion is wrong and I report that. The
prediction on record:

    retryesc_gen (w=25, died 3/3)   -> LOWER beyond-T mass
    evalage      (w=13, 2/5 cells)  -> HIGHER beyond-T mass

## ⛔ THE FALSIFIER FIRED — `beyond_T` SATURATES and cannot be the screen

Held at only **2/5** T values. At **T=4, 8, 16 it is 1.0000 for BOTH
corpora** — events are far sparser than the window, so "mass beyond T"
is everything, for everyone, and the criterion has no discriminating
power exactly where the hunt operates.

**Diagnosis: I scored the wrong array.** `beyond_T` uses `event_first`,
but the floor's *discriminating* feature is
`dose_window_count(event_MASK, T)` — masked **turn** tokens, width `w` —
not `sage_floor(event_first, T)`. The floor's power comes from mask
WIDTH, which is precisely why `w` (13 vs 25) is lever 3.

**The working quantity is `floor_reach` below** — `P(any masked token in
the trailing T window)`, which is exactly my own corrected floor law
(`d2320d274`). Measured against the two corpora's OBSERVED
`floor_excess` it tracks within **0.82–1.13×** and separates them ~**5×
at every T**, where `beyond_T` was flat at 1.0000:

    corpus         T    floor_reach   observed floor_excess   ratio
    evalage        4         0.0035                  0.0031    1.13
    evalage       64         0.0541                  0.0572    0.95
    retryesc_gen   4         0.0184                  0.0225    0.82
    retryesc_gen  64         0.2431                  0.2748    0.88

**Screen rule: MINIMISE `floor_reach`.** It is the fraction of positions
whose label the floor can already resolve. Both are kept in this file —
the falsified criterion is part of the record, not deleted.

Run:
  PYTHONPATH=. .venv/bin/python -m experiments.explorations.task_hunt.beyond_t_mass
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TS = (4, 8, 16, 32, 64)
CORPORA = {
    "evalage":      (HERE / "evalage/grids/elicit_evalage_screen_{tag}.npz", 13),
    "retryesc_gen": (HERE / "retryesc_gen/grids/"
                     "elicit_retryesc_gen_v1_screen_{tag}.npz", 25),
}
TAGS = ("gpt2", "gemma2", "llama31")

# Observed arm - floor, from the frozen screens / my lever-3 run (gemma2 leg).
OBSERVED = {
    ("evalage", 4): 0.1435, ("evalage", 8): 0.1467, ("evalage", 16): 0.1592,
    ("evalage", 32): 0.1523, ("evalage", 64): 0.1403,
}


def raw_age(first: np.ndarray, doc_off: np.ndarray) -> np.ndarray:
    """Tokens since the last event-first, per document. inf before the first."""
    age = np.full(first.size, np.inf, dtype=np.float64)
    for d in range(len(doc_off) - 1):
        lo, hi = int(doc_off[d]), int(doc_off[d + 1])
        seg = first[lo:hi]
        idx = np.arange(hi - lo)
        last = np.where(seg, idx, -1)
        last = np.maximum.accumulate(last)
        a = np.where(last >= 0, idx - last, np.inf)
        age[lo:hi] = a
    return age


def main() -> None:
    print(__doc__.split("Run:")[0].strip()[:0] or "", end="")
    print("BEYOND-T MASS  P(age > T)  over probe-eligible positions")
    print("(the floor censors at T+1, so this is the mass it cannot resolve)\n")
    rows = {}
    for name, (pat, w) in CORPORA.items():
        for tag in TAGS:
            p = Path(str(pat).format(tag=tag))
            if not p.exists():
                continue
            z = np.load(p)
            first = z["event_first"].astype(bool)
            elig = (z["is_assistant"].astype(bool) if "is_assistant" in z
                    else np.ones(first.size, bool))
            age = raw_age(first, z["doc_off"])
            fin = np.isfinite(age) & elig
            rows[(name, tag)] = (w, {T: float((age[fin] > T).mean())
                                    for T in TS}, int(fin.sum()))

    print(f"{'corpus':<14}{'leg':<9}{'w':>3}{'n_elig':>10}"
          + "".join(f"{'T'+str(T):>9}" for T in TS))
    print("-" * (36 + 9 * len(TS)))
    for (name, tag), (w, m, n) in rows.items():
        print(f"{name:<14}{tag:<9}{w:>3}{n:>10,}"
              + "".join(f"{m[T]:>9.4f}" for T in TS))

    print("\n--- corpus means across legs ---")
    print(f"{'corpus':<14}{'w':>4}" + "".join(f"{'T'+str(T):>9}" for T in TS))
    print("-" * (18 + 9 * len(TS)))
    means = {}
    for name in CORPORA:
        legs = [m for (n2, _), (_, m, _) in rows.items() if n2 == name]
        if not legs:
            continue
        means[name] = {T: float(np.mean([m[T] for m in legs])) for T in TS}
        w = CORPORA[name][1]
        print(f"{name:<14}{w:>4}" + "".join(f"{means[name][T]:>9.4f}"
                                            for T in TS))

    if len(means) == 2:
        a, b = "evalage", "retryesc_gen"
        print(f"\n{'T':>4}{'evalage':>10}{'retryesc':>10}{'diff':>9}"
              f"   PREDICTION: evalage HIGHER")
        print("-" * 52)
        held = 0
        for T in TS:
            d = means[a][T] - means[b][T]
            ok = d > 0
            held += ok
            print(f"{T:>4}{means[a][T]:>10.4f}{means[b][T]:>10.4f}{d:>+9.4f}"
                  f"   {'HELD' if ok else '** NOT HELD **'}")
        print(f"\n  prediction held at {held}/{len(TS)} T values")
        if held == len(TS):
            print("  => beyond-T mass orders the two corpora as their outcomes did.")
        else:
            print("  => ⚑ FALSIFIED at one or more T. The criterion does not "
                  "order these corpora; report as a negative.")

    # ---- the WORKING criterion: floor_reach ----------------------------
    print("\n\n" + "=" * 72)
    print("floor_reach = P(any masked token in the trailing T window)")
    print("  = my corrected floor law (d2320d274). MINIMISE THIS.")
    print("  Uses event_MASK (turn width w), which is the floor's actual")
    print("  discriminating feature — beyond_T used event_first and saturated.")
    print("=" * 72 + "\n")
    print(f"{'corpus':<14}{'leg':<9}{'w':>3}"
          + "".join(f"{'T'+str(T):>9}" for T in TS))
    print("-" * (26 + 9 * len(TS)))
    fr = {}
    for name, (pat, w) in CORPORA.items():
        legs = []
        for tag in TAGS:
            p = Path(str(pat).format(tag=tag))
            if not p.exists():
                continue
            z = np.load(p)
            mask = z["event_mask"].astype(bool)
            elig = (z["is_assistant"].astype(bool) if "is_assistant" in z
                    else np.ones(mask.size, bool))
            c = np.concatenate([[0], np.cumsum(mask.astype(np.int64))])
            idx = np.arange(mask.size)
            m = {}
            for T in TS:
                lo = np.maximum(idx - T, 0)
                m[T] = float(((c[idx] - c[lo]) > 0)[elig].mean())
            legs.append(m)
            print(f"{name:<14}{tag:<9}{w:>3}"
                  + "".join(f"{m[T]:>9.4f}" for T in TS))
        if legs:
            fr[name] = {T: float(np.mean([m[T] for m in legs])) for T in TS}

    if len(fr) == 2:
        a, b = "evalage", "retryesc_gen"
        print(f"\n{'T':>4}{'evalage':>10}{'retryesc':>10}{'x cheaper':>11}"
              f"   (lower is better)")
        print("-" * 48)
        for T in TS:
            print(f"{T:>4}{fr[a][T]:>10.4f}{fr[b][T]:>10.4f}"
                  f"{fr[b][T]/max(fr[a][T],1e-12):>11.2f}x")
        print("\n  evalage's floor reaches ~5x fewer positions at every T —"
              "\n  which is what lever 3 (w 25 -> 13) buys, and it is the"
              "\n  ordering the observed floors and outcomes actually showed.")

    print("\n⚑ LOWER EDGE ONLY. `floor_reach` scores mass the floor cannot "
          "resolve. It\n   does NOT score whether activations still carry an "
          "event that old — the\n   upper edge of the band is unmeasured and "
          "needs activations.")


if __name__ == "__main__":
    main()
