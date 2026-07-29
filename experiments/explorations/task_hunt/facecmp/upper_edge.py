"""UPPER EDGE — how far back does the residual stream still carry a sparse event?

$0, 0 pods. Reuses the `evalage` gemma2_2b@512 cache already on disk.

## Why this exists

`floor_reach` (`task_hunt/beyond_t_mass.py`) settled the **lower** edge of
the aiming band: the fraction of positions whose label the floor can
already resolve, which is what lever 3 minimises. It says **nothing**
about the other side. A corpus can score perfectly on `floor_reach` and
still lose, because its events are simply too old to be represented at
all. That upper edge has never been measured, and it decides whether
"push events beyond `T+w`" is a real strategy or a way of walking off a
cliff.

**Question:** as a function of the age of the last event, can a probe on
a single token's activation still tell how old that event is?

## PRE-REGISTRATION — written and committed BEFORE the run

- **U1.** Per-bucket accuracy **declines with age.** ⚑ *Falsifier
  against the whole experiment:* if accuracy is **flat across all
  buckets including the oldest**, then either retention is unbounded
  (implausible) or the probe is reading something other than age. U2
  decides which, and a flat curve is reported as such — not narrated
  as "long retention".
- **U2.** The **position control** (`_pos_feats`: chunk pos, pos²,
  log2(1+doc pos), doc pos/1000) scores **well below** the activation
  probe. ⚑ *If it does not, the run is VOID* — age is correlated with
  position-in-document, and a probe that merely reads position would
  produce a convincing decay curve that means nothing.
- **U3.** `label_null` (permuted labels) ≈ chance = 1/K. If not, the
  balancing or the split leaks and the run is VOID.
- **U4.** There is a finite age **A** beyond which accuracy is within
  noise of chance; **A is the retention horizon.** ⚑ If no such A
  exists inside the observable range, the honest report is **"horizon
  exceeds the measurable range on this corpus"** — NOT an invented
  number, and NOT "unbounded".

**No result here moves any bar.** This measures an instrument property
of the model, not a candidate.

## Design

Same row machinery as `arm_test.build_rows` — byte-identical
`_row_lookup`/`_map_rows`/`pos_strata`/`stratified_balanced_manifest`, so
eligibility and stratification match the screen — with ONE change: the
label is **octave age buckets** instead of terciles of the face value.
`f_age` is `sage_face = log2(1 + age)`, so the bucket edges below are
exact powers of two in age.

Run:
  FACECMP_CACHE_ROOT=<scratch>/cache_evalage_512 PYTHONPATH=. \
    .venv/bin/python -m experiments.explorations.task_hunt.facecmp.upper_edge
"""
from __future__ import annotations

import json
import os
import zlib
from pathlib import Path

import numpy as np
import torch

from experiments.explorations.conversion_depth.problib import fit_probe
from experiments.explorations.task_hunt.labels.punctint_lib import (
    pos_strata,
    stratified_balanced_manifest,
)
from experiments.explorations.task_hunt.novelty.screen import (
    _map_rows,
    _row_lookup,
)
from experiments.explorations.task_hunt.replag.cache_acts import SCREEN_HS
from experiments.explorations.task_hunt.replag.screen import gather_tok

# age buckets, in TOKENS. Octaves: wide enough for power at n=3 splits,
# fine enough that a horizon lands inside one of them.
EDGES = [1, 4, 16, 64, 256, 1024, np.inf]
LABELS = ["1-3", "4-15", "16-63", "64-255", "256-1023", "1024+"]
K = len(LABELS)
CHANCE = 1.0 / K
NULL_SEED = 99
CAP = {"train": 60000, "test": 20000}
MIN_ROWS = 200
POS_MIN, OFF_MIN, PRE_MIN_POS = 32, 8, 64


def _pos_feats(r):
    cp = r[:, 1].astype(np.float32)
    dp = r[:, 2].astype(np.float32)
    f = np.stack([cp, cp ** 2 / 128.0, np.log2(1.0 + dp), dp / 1000.0], 1)
    return torch.from_numpy(f).to(torch.float16)


def raw_age(first: np.ndarray, off: np.ndarray) -> np.ndarray:
    """Tokens since the last event-first, per document; inf before the first."""
    age = np.full(first.size, np.inf, dtype=np.float64)
    for d in range(len(off) - 1):
        lo, hi = int(off[d]), int(off[d + 1])
        seg = first[lo:hi]
        idx = np.arange(hi - lo)
        last = np.maximum.accumulate(np.where(seg, idx, -1))
        age[lo:hi] = np.where(last >= 0, idx - last, np.inf)
    return age


def build_bucket_rows(key: str, grids: Path, pat: str, cache_root: Path):
    tag = {"gpt2": "gpt2", "gemma2_2b": "gemma2", "llama31_8b": "llama31"}[key]
    z = np.load(grids / pat.format(tag=tag))
    c = np.load(cache_root / key / "tokens.npz")
    doc_idx, n_prefix = c["doc_idx"], int(c["n_prefix"])
    content = c["ids"].shape[1] - n_prefix
    off, first, mask = z["doc_off"], z["event_first"], z["event_mask"]
    is_assist, doc_split = z["is_assistant"], z["doc_split"]

    age = raw_age(first.astype(bool), off)
    lookup = _row_lookup(doc_idx)
    n_tok = age.size
    doc_of = np.searchsorted(off, np.arange(n_tok), side="right") - 1
    pos_of = np.arange(n_tok) - off[doc_of]
    rows_flat, _ = _map_rows(doc_of, pos_of, lookup, content, n_prefix)

    # Same eligibility as the screen, minus the face-specific horizon.
    elig = ((mask == 0) & (is_assist == 1) & (rows_flat >= 0)
            & (pos_of >= max(PRE_MIN_POS, POS_MIN))
            & (pos_of % content >= OFF_MIN) & np.isfinite(age))

    bins = np.full(n_tok, -1, dtype=np.int64)
    for b in range(K):
        sel = elig & (age >= EDGES[b]) & (age < EDGES[b + 1])
        bins[sel] = b
    elig = elig & (bins >= 0)

    out, stats = {}, {"buckets": LABELS, "edges": [float(e) for e in EDGES],
                      "n_elig": int(elig.sum()),
                      "bin_counts": {LABELS[b]: int((bins == b).sum())
                                     for b in range(K)}}
    for split_name, flag in (("train", 0), ("test", 1)):
        m = elig & (doc_split[doc_of] == flag)
        pool = np.flatnonzero(m)
        strata = pos_strata(pos_of[pool], min_pos=POS_MIN)
        seed = zlib.crc32(f"upper_edge/{tag}/{split_name}".encode()) % 2 ** 16
        md, mp, mc = stratified_balanced_manifest(
            bins[pool], strata, doc_of[pool], pos_of[pool],
            cap=CAP[split_name], seed=seed)
        per = {LABELS[int(cl)]: int((mc == cl).sum()) for cl in range(K)
               if (mc == cl).any()}
        stats[split_name] = {"rows_per_bucket": per,
                             "ok": bool(per and min(per.values()) >= MIN_ROWS)}
        rows_all, cpos_all = _map_rows(md, mp, lookup, content, n_prefix)
        keep = rows_all >= 0
        r = np.stack([rows_all[keep], cpos_all[keep], mp[keep]], 1)
        out[split_name] = (r, mc[keep].astype(np.int64))
    return out, stats


def main() -> None:
    key = os.environ.get("FACECMP_MODEL", "gemma2_2b")
    root = Path(os.environ["FACECMP_CACHE_ROOT"])
    here = Path(__file__).resolve().parent
    grids = here.parent / "evalage" / "grids"
    pat = "elicit_evalage_screen_{tag}.npz"

    man, stats = build_bucket_rows(key, grids, pat, root)
    print("rows per bucket:", json.dumps(stats["train"]["rows_per_bucket"]))
    print("            test:", json.dumps(stats["test"]["rows_per_bucket"]))
    if not (stats["train"]["ok"] and stats["test"]["ok"]):
        print("\n** INSUFFICIENT ROWS in some bucket — reporting and stopping. **")

    hs = SCREEN_HS[key]
    acts = torch.from_numpy(np.ascontiguousarray(
        np.load(root / key / f"hs{hs}.npy", mmap_mode="r")))
    rtr, ytr = man["train"]
    rte, yte = man["test"]
    ytr_t, yte_t = torch.from_numpy(ytr), torch.from_numpy(yte)

    Xtr, Xte = gather_tok(acts, rtr), gather_tok(acts, rte)
    act = fit_probe(Xtr, ytr_t, Xte, yte_t, K, hidden=512)
    pos = fit_probe(_pos_feats(rtr), ytr_t, _pos_feats(rte), yte_t, K)
    rng = np.random.default_rng(NULL_SEED)
    ynull = torch.from_numpy(rng.permutation(ytr.copy()))
    null = fit_probe(Xtr, ynull, Xte, yte_t, K, hidden=512)

    print(f"\nchance = 1/{K} = {CHANCE:.4f}")
    print(f"  activation probe (tok) : {act['acc_test']:.4f}")
    print(f"  position control       : {pos['acc_test']:.4f}   [U2]")
    print(f"  label_null             : {null['acc_test']:.4f}   [U3]")

    per = act.get("per_class") or {}
    pper = pos.get("per_class") or {}
    print(f"\n{'bucket (age tok)':<18}{'act recall':>12}{'pos recall':>12}"
          f"{'act - chance':>14}")
    print("-" * 56)
    rec = {}
    for b, lab in enumerate(LABELS):
        a = float(per.get(str(b), per.get(b, float("nan"))))
        p = float(pper.get(str(b), pper.get(b, float("nan"))))
        rec[lab] = a
        print(f"{lab:<18}{a:>12.4f}{p:>12.4f}{a - CHANCE:>+14.4f}")

    # --- pre-registered verdicts -------------------------------------
    print("\n--- PRE-REGISTERED VERDICTS ---")
    u2 = pos["acc_test"] < act["acc_test"] - 0.05
    u3 = abs(null["acc_test"] - CHANCE) < 0.05
    print(f"U2 position control well below activation probe : "
          f"{'HELD' if u2 else '** VIOLATED -> RUN IS VOID **'}")
    print(f"U3 label_null within 0.05 of chance             : "
          f"{'HELD' if u3 else '** VIOLATED -> RUN IS VOID **'}")
    vals = [rec[l] for l in LABELS if np.isfinite(rec[l])]
    if len(vals) >= 2:
        spread = max(vals) - min(vals)
        u1 = spread > 0.05 and vals[0] > vals[-1]
        print(f"U1 accuracy declines with age (spread {spread:.4f})   : "
              f"{'HELD' if u1 else '** FLAT/NON-MONOTONE — see U1 falsifier **'}")
        above = [l for l in LABELS
                 if np.isfinite(rec[l]) and rec[l] - CHANCE > 0.05]
        print(f"U4 buckets clearly above chance                 : {above}")
        if above and above[-1] == LABELS[-1]:
            print("   => HORIZON EXCEEDS THE MEASURABLE RANGE on this corpus. "
                  "Reported as such, NOT as a number.")
        elif above:
            print(f"   => retention horizon lies inside bucket "
                  f"AFTER '{above[-1]}'")

    outp = here / "results" / "upper_edge"
    outp.mkdir(parents=True, exist_ok=True)
    (outp / f"upper_edge_{key}.json").write_text(json.dumps({
        "key": key, "buckets": LABELS, "chance": CHANCE, "stats": stats,
        "act": act, "pos_control": pos, "label_null": null,
        "note": "Upper edge of the aiming band. Instrument property of the "
                "model, not a candidate verdict. Pre-registered U1-U4.",
    }, indent=1, default=float))
    print(f"\nwrote {outp / f'upper_edge_{key}.json'}")


if __name__ == "__main__":
    main()
