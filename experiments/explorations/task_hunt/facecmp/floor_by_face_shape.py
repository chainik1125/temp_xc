"""Does the FACE SHAPE, not the topic, decide whether the floor clause bites?

$0, no GPU, no generation. Runs on corpora already in git.

## Why this exists

Four candidates screened, one KEEP. Every single face the program has put
through a screen is a **RECENCY** face: `sycgen_age`, `evalage_age`,
`retryesc_age`, `reask_hr` -- all "how long since the last event".

The bar that kills them is the **visible-evidence floor**, and today's
`floor_predictor_test.py` established what that floor actually is: a
probe on `(sage_floor(T), dose_window_count(event_mask, T))` -- i.e. an
**age estimator censored at T + w**. So a recency face competes with the
floor on the floor's own home turf: the thing the floor computes IS a
censored version of the label. That is a structural reason age faces keep
dying on the floor clause, and it predicts exactly the scissors measured
in `retryesc_gen/RESULT.md` § 1 -- floor rising with T at the same rate
the age becomes recoverable, so the arm can never be simultaneously
useful and floor-beating.

Meanwhile the registry's safety-relevant latent states are mostly NOT
recency states. Alignment faking, sandbagging, belief drift under
accumulating context, behavioural phase transitions, misalignment
contagion, evaluation-regime leakage -- these are **CUMULATIVE / REGIME**
states: "how much pressure has accumulated", "which regime am I in".

**Hypothesis (H3):** a long-horizon CUMULATIVE face has a much lower
visible-evidence floor than a recency face on the SAME corpus, because a
T-token window cannot recover a count over horizon H >> T.

## What this does and does not show

The floor is **ground-truth-derived**, so it needs no activations and no
new corpus -- it can be computed for a hypothetical face on an existing
corpus. That makes this a genuine $0 pre-screen of a design.

**It tests the BAR, not the ARM.** A low floor is necessary, not
sufficient: it says the clause that killed the last three candidates
would not bite, NOT that a windowed probe would clear the gain bar. The
arm needs activations and therefore a screen. Stated here so the result
is not over-read -- the same over-reading is what put `retryesc_gen`
above its band edge.

**Second limitation, disclosed:** these corpora were generated with a
roughly constant per-document event rate, so `rate_H` has less spread
here than a corpus DESIGNED for it would. That biases against H3 (less
spread => easier for a floor to guess), so a low floor here is
conservative evidence.

Run: .venv/bin/python -m experiments.explorations.task_hunt.facecmp.floor_by_face_shape
Writes results/floor_by_face_shape.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.explorations.conversion_depth.problib import fit_probe
from experiments.explorations.task_hunt.labels import wave3_lib as w3
from experiments.explorations.task_hunt.labels.build_retryesc_gen_premeasure import (
    section_age,
)
from experiments.explorations.task_hunt.labels.build_wave3_trio import (
    MIN_POS as PRE_MIN_POS,
)

ROOT = Path(__file__).resolve().parent.parent
RES = Path(__file__).resolve().parent / "results"
T = 64
HORIZONS = [256, 512]
CHANCE = 1.0 / 3.0
SEED = 20260728

CORPORA = [
    ("evalage", ROOT / "evalage" / "grids", "elicit_evalage_screen"),
    ("retryesc_gen", ROOT / "retryesc_gen" / "grids",
     "elicit_retryesc_gen_v1_screen"),
]


def rate_face(first, off, n_docs, H):
    """Events in the trailing H tokens (current token excluded) -- the
    CUMULATIVE face. Per-document so it never spans a boundary."""
    out = []
    for d in range(n_docs):
        b = np.asarray(first[off[d]:off[d + 1]], dtype=np.int64)
        c = np.concatenate([[0], np.cumsum(b)])
        idx = np.arange(len(b))
        out.append((c[idx] - c[np.maximum(idx - H, 0)]).astype(np.float64))
    return np.concatenate(out)


def terciles(v, m):
    """Equal-mass 3-way split on the eligible rows; -1 where unusable.
    Ties go to the lower bin, so a heavily tied face (a small-count rate)
    yields unbalanced bins -- reported rather than hidden."""
    lo, hi = np.quantile(v[m], [1 / 3, 2 / 3])
    b = np.full(len(v), -1, dtype=np.int64)
    b[m] = 0
    b[m & (v > lo)] = 1
    b[m & (v > hi)] = 2
    return b, (float(lo), float(hi))


def balanced(idx, y, rng, cap):
    keep = []
    for c in (0, 1, 2):
        s = idx[y[idx] == c]
        if len(s) == 0:
            return None
        keep.append(rng.permutation(s)[:cap])
    n = min(len(k) for k in keep)
    return np.concatenate([k[:n] for k in keep])


def floor_for(label_bins, feats, split, rng):
    """The screen's own visible-evidence floor, on a balanced manifest."""
    tr = np.flatnonzero((label_bins >= 0) & (split == 0))
    te = np.flatnonzero((label_bins >= 0) & (split == 1))
    tr, te = balanced(tr, label_bins, rng, 4000), balanced(te, label_bins, rng, 1500)
    if tr is None or te is None or len(tr) < 300 or len(te) < 150:
        return None
    Xtr = torch.from_numpy(feats[tr]).to(torch.float32)
    Xte = torch.from_numpy(feats[te]).to(torch.float32)
    ytr = torch.from_numpy(label_bins[tr])
    yte = torch.from_numpy(label_bins[te])
    r = fit_probe(Xtr, ytr, Xte, yte, 3)
    acc = float(r["acc_test"] if isinstance(r, dict) else r)
    return {"floor_acc": acc, "floor_excess": acc - CHANCE,
            "n_train": int(len(tr)), "n_test": int(len(te))}


def leg(name, gdir, stem, tag="gpt2"):
    z = np.load(gdir / f"{stem}_{tag}.npz")
    off, first, mask = z["doc_off"], z["event_first"], z["event_mask"]
    is_assist, doc_split = z["is_assistant"], z["doc_split"]
    n_docs = len(off) - 1
    n = len(first)

    doc_of = np.searchsorted(off, np.arange(n), side="right") - 1
    pos_of = np.arange(n) - off[doc_of]
    split = doc_split[doc_of]

    age_log = np.concatenate([w3.sage_face(first[off[d]:off[d + 1]])
                              for d in range(n_docs)])
    # window-computable features -- IDENTICAL for every face, by design:
    # this is what a T-token observer can see, whatever the label is.
    cage = np.concatenate([w3.sage_floor(first[off[d]:off[d + 1]], T)
                           for d in range(n_docs)])
    cnt = np.concatenate([w3.dose_window_count(mask[off[d]:off[d + 1]], T)
                          for d in range(n_docs)])
    feats = np.stack([np.nan_to_num(cage), cnt], 1).astype(np.float32)

    base = (mask == 0) & (is_assist == 1) & (pos_of >= PRE_MIN_POS)
    rng = np.random.default_rng(SEED)
    out = []

    m_age = base & np.isfinite(age_log)
    b, edges = terciles(age_log, m_age)
    r = floor_for(b, feats, split, rng)
    if r:
        r.update({"face": "RECENCY age (as screened)", "edges": edges,
                  "n_elig": int(m_age.sum())})
        out.append(r)

    for H in HORIZONS:
        # require a FULL horizon, else early positions are low by
        # construction and the face is really a position clock
        m = base & (pos_of >= H)
        v = rate_face(first, off, n_docs, H)
        if m.sum() < 1000 or len(np.unique(v[m])) < 3:
            out.append({"face": f"CUMULATIVE rate_H{H}", "skipped":
                        "insufficient spread or rows", "n_elig": int(m.sum())})
            continue
        b, edges = terciles(v, m)
        per = {int(c): int((b == c).sum()) for c in (0, 1, 2)}
        r = floor_for(b, feats, split, rng)
        if r:
            r.update({"face": f"CUMULATIVE rate_H{H}", "edges": edges,
                      "n_elig": int(m.sum()), "bin_counts": per})
            out.append(r)
        else:
            out.append({"face": f"CUMULATIVE rate_H{H}",
                        "skipped": "manifest too small", "bin_counts": per})
    return {"corpus": name, "tokenizer": tag, "faces": out}


def main():
    RES.mkdir(parents=True, exist_ok=True)
    res = [leg(n, g, s) for n, g, s in CORPORA]
    print(f"{'corpus':<14}{'face':<30}{'floor acc':>10}{'floor_excess':>14}"
          f"{'n_elig':>10}")
    print("-" * 78)
    for L in res:
        for f in L["faces"]:
            if "skipped" in f:
                print(f"{L['corpus']:<14}{f['face']:<30}{'SKIP':>10}"
                      f"  {f['skipped']}")
                continue
            print(f"{L['corpus']:<14}{f['face']:<30}{f['floor_acc']:>10.4f}"
                  f"{f['floor_excess']:>+14.4f}{f['n_elig']:>10,}")
    (RES / "floor_by_face_shape.json").write_text(
        json.dumps({"T": T, "horizons": HORIZONS, "chance": CHANCE,
                    "seed": SEED, "results": res}, indent=2))
    print(f"\nwrote {RES / 'floor_by_face_shape.json'}")


if __name__ == "__main__":
    main()
