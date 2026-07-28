"""ARM test for the CUMULATIVE face — on a corpus that ALREADY EXISTS.

The $0 work (`floor_by_face_shape.py` + the position-floor gate) showed the
cumulative face's two BARS are low on `retryesc_gen`:

    face                  visible floor    position floor
    RECENCY age              +0.2553          +0.0393
    CUMULATIVE rate_H512     +0.1331          +0.0000   <- chance, exactly

What it could not show is whether the **ARM** clears anything: does a
64-token window of activations encode *how many events happened in the
last 512 tokens*? That needs activations, and it is the one question that
decides whether the cumulative direction is worth a $21 generation.

**This buys the answer for ~$1 instead of ~$22** by re-labelling a corpus
already in git rather than generating a new one. Same documents, same
activations, same helpers, same bars — only the LABEL changes.

## Read the outcome asymmetrically. This is deliberate.

- **Arm clears gain** -> strong evidence; justifies designing and paying
  for a corpus built for this face.
- **Arm fails** -> WEAK evidence only, and I will not report it as a kill.
  `retryesc_gen` was built for a recency face with a roughly constant
  per-document event rate, so `rate_H512` here is a small-integer face
  with heavy ties (tercile edges land on 1.0/2.0, bins 297k/164k/100k).
  A failure is confounded between "the model does not encode a
  long-horizon count" and "this particular face is too coarse to read".

Nothing here is a hunt4 § 4 verdict: this is a **feasibility probe on a
borrowed corpus**, not a screened candidate, and it must never be quoted
as one. A real verdict needs a corpus designed for the face.

## Pre-registration (written before any GPU ran)

1. **Position floor stays at chance.** Measured +0.0000 label-side; if
   the manifest here moves it above +0.02 the row construction differs
   from the label-side test and the run is void.
2. **Visible floor lands near +0.13**, well under the recency face's
   +0.26. Same features, so a large move means I mis-transplanted.
3. **Arm: genuinely uncertain, ~35-40%.** `sycgen_rate` was demoted once
   already, and I have no measurement either way on whether a running
   count is linearly decodable from a window. Stating a number I am not
   confident in, before the fact, so it can be scored honestly.
4. **If the arm clears, expect it to clear at LARGE T**, since a longer
   window carries more of the horizon.

Run (on a GPU box, after `cache_acts`):
  .venv/bin/python -m experiments.explorations.task_hunt.facecmp.arm_test [model ...]
Writes results/arm_test_<model>.json (resumable).
"""

from __future__ import annotations

import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from experiments.explorations.conversion_depth.problib import fit_probe
from experiments.explorations.task_hunt.dialevel.actxmean_null import (
    actxmean_foreign,
)
from experiments.explorations.task_hunt.dialevel.capacity_check import (
    FOREIGN_SEED,
    anchor_ctxmean,
)
from experiments.explorations.task_hunt.facecmp.floor_by_face_shape import (
    rate_face,
)
from experiments.explorations.task_hunt.labels import wave3_lib as w3
from experiments.explorations.task_hunt.labels.build_wave3_trio import (
    MIN_POS as PRE_MIN_POS,
)
from experiments.explorations.task_hunt.labels.punctint_lib import (
    pos_strata,
    stratified_balanced_manifest,
)
from experiments.explorations.task_hunt.novelty.screen import (
    CAP,
    MIN_ROWS,
    OFF_MIN,
    POS_MIN,
    _map_rows,
    _row_lookup,
    _stack,
)
from experiments.explorations.task_hunt.replag.build_labels import MODELS, SEQ_LEN
from experiments.explorations.task_hunt.replag.cache_acts import SCREEN_HS
from experiments.explorations.task_hunt.replag.screen import (
    gather_tok,
    gather_win,
    summarize,
)
from experiments.explorations.task_hunt.retryesc_gen.cache_acts import (
    CACHE_ROOT,
    TOK_TAG,
)

HERE = Path(__file__).resolve().parent
GRIDS = HERE.parent / "retryesc_gen" / "grids"
RES = HERE / "results"

FACE = "cum_rate_H512"
H = 512
AX_TS = [16, 32, 64]
FOREIGN_TS = [32, 64]
NULL_SEED = 99
CHANCE = 1.0 / 3.0


def _pos_feats(r):
    cp = r[:, 1].astype(np.float32)
    dp = r[:, 2].astype(np.float32)
    f = np.stack([cp, cp ** 2 / 128.0, np.log2(1.0 + dp), dp / 1000.0], 1)
    return torch.from_numpy(f).to(torch.float16)


class _FloorBank:
    """IDENTICAL to the screens' floor: what a T-token observer can see.
    Unchanged on purpose — the whole point is that only the LABEL moves."""

    def __init__(self, first, mask, off, n_docs):
        self.first, self.mask, self.off, self.n_docs = first, mask, off, n_docs
        self._c: dict = {}

    def feats(self, flat_pos, T):
        if T not in self._c:
            o, nd = self.off, self.n_docs
            self._c[T] = (
                np.concatenate([w3.sage_floor(self.first[o[d]:o[d + 1]], T)
                                for d in range(nd)]),
                np.concatenate([w3.dose_window_count(self.mask[o[d]:o[d + 1]], T)
                                for d in range(nd)]))
        cage, cnt = self._c[T]
        f = np.stack([np.nan_to_num(cage[flat_pos].astype(np.float32)),
                      cnt[flat_pos].astype(np.float32)], 1)
        return torch.from_numpy(f).to(torch.float16)


def build_rows(key: str):
    tag = TOK_TAG[key]
    z = np.load(GRIDS / f"elicit_retryesc_gen_v1_screen_{tag}.npz")
    c = np.load(CACHE_ROOT / key / "tokens.npz")
    ids, doc_idx, n_prefix = c["ids"], c["doc_idx"], int(c["n_prefix"])
    content = ids.shape[1] - n_prefix
    off, first, mask = z["doc_off"], z["event_first"], z["event_mask"]
    is_assist, doc_split = z["is_assistant"], z["doc_split"]
    n_docs = len(off) - 1

    # THE ONLY THING THAT CHANGES: a cumulative label, not a recency one.
    val = rate_face(first, off, n_docs, H)

    lookup = _row_lookup(doc_idx)
    n_tok = len(val)
    doc_of = np.searchsorted(off, np.arange(n_tok), side="right") - 1
    pos_of = np.arange(n_tok) - off[doc_of]
    rows_flat, cpos_flat = _map_rows(doc_of, pos_of, lookup, content, n_prefix)
    train_rows = doc_split[doc_of] == 0

    # `pos_of >= H` is REQUIRED: below a full horizon the count is low by
    # construction and the face degenerates into a position clock.
    elig = ((mask == 0) & (is_assist == 1) & (pos_of >= max(PRE_MIN_POS, H))
            & (rows_flat >= 0) & (pos_of >= POS_MIN)
            & (pos_of % content >= OFF_MIN))

    lo, hi = np.quantile(val[elig & train_rows], [1 / 3, 2 / 3])
    bins = np.full(n_tok, -1, dtype=np.int64)
    bins[elig] = 0
    bins[elig & (val > lo)] = 1
    bins[elig & (val > hi)] = 2
    elig = elig & (bins >= 0)

    out, stats = {}, {"face": FACE, "H": H, "n_docs": n_docs,
                      "tercile_edges": [float(lo), float(hi)],
                      "n_elig": int(elig.sum()),
                      "bin_counts": {int(b): int((bins == b).sum())
                                     for b in (0, 1, 2)}}
    for split_name, flag in (("train", 0), ("test", 1)):
        m = elig & (doc_split[doc_of] == flag)
        pool = np.flatnonzero(m)
        if not len(pool):
            stats[f"{FACE}/{split_name}"] = {"ok": False, "rows": 0}
            continue
        strata = pos_strata(pos_of[pool], min_pos=POS_MIN)
        seed = zlib.crc32(f"facecmp/{tag}/{split_name}".encode()) % 2 ** 16
        md, mp, mc = stratified_balanced_manifest(
            bins[pool], strata, doc_of[pool], pos_of[pool],
            cap=CAP[split_name], seed=seed)
        per = {int(cl): int((mc == cl).sum()) for cl in (0, 1, 2)}
        stats[f"{FACE}/{split_name}"] = {
            "rows_per_class_total": per,
            "ok": bool(per and min(per.values()) >= MIN_ROWS)}
        if not per or not min(per.values()):
            continue
        man_flat = off[md] + mp
        rows_all, cpos_all = _map_rows(md, mp, lookup, content, n_prefix)
        assert (rows_all >= 0).all()
        keep_r, keep_y, keep_f = [], [], []
        for cl in (0, 1, 2):
            sel = np.flatnonzero(mc == cl)
            keep_r.append(_stack(rows_all, cpos_all, mp, sel))
            keep_y.append(np.full(len(sel), cl, dtype=np.int64))
            keep_f.append(man_flat[sel])
        out[(FACE, split_name)] = (np.concatenate(keep_r),
                                   np.concatenate(keep_y))
        out[(f"{FACE}_flat", split_name)] = np.concatenate(keep_f)

    return out, stats, {"first": first, "mask": mask, "off": off,
                        "n_docs": n_docs}


def screen(key: str):
    RES.mkdir(parents=True, exist_ok=True)
    out_path = RES / f"arm_test_{key}.json"
    hs = SCREEN_HS[key]
    manifests, mstats, fl = build_rows(key)
    bank = _FloorBank(**fl)
    done = (json.loads(out_path.read_text()) if out_path.exists()
            else {"meta": {}, "cells": {}})
    done["meta"] = {"substrate": "elicit_retryesc_gen_v1 (BORROWED corpus)",
                    "model": key, "screen_hs": hs, "face": FACE, "H": H,
                    "status": "FEASIBILITY PROBE, not a hunt4 verdict",
                    "ax_ts": AX_TS, "rows": mstats}
    cells = done["cells"]

    def save():
        out_path.write_text(json.dumps(done, indent=1, default=float))

    def run(name, fn):
        if name in cells:
            print(f"  [skip] {name}", flush=True)
            return
        t0 = time.time()
        cells[name] = fn()
        cells[name]["wall_s"] = round(time.time() - t0, 1)
        print(f"  {name}: {cells[name].get('acc_test'):.4f}", flush=True)
        save()

    if not (mstats[f"{FACE}/train"]["ok"] and mstats[f"{FACE}/test"]["ok"]):
        print(f"[{key}] SKIP — insufficient rows")
        save()
        return

    acts = torch.from_numpy(np.ascontiguousarray(
        np.load(CACHE_ROOT / key / f"hs{hs}.npy", mmap_mode="r")))
    rtr, ytr = manifests[(FACE, "train")]
    rte, yte = manifests[(FACE, "test")]
    ftr, fte = manifests[(f"{FACE}_flat", "train")], manifests[(f"{FACE}_flat", "test")]
    ytr_t, yte_t = torch.from_numpy(ytr), torch.from_numpy(yte)

    # per-token arms FIRST (standing rule)
    Xtr, Xte = gather_tok(acts, rtr), gather_tok(acts, rte)
    run(f"{FACE}/tok_linear", lambda: summarize(
        fit_probe(Xtr, ytr_t, Xte, yte_t, 3), 3))
    run(f"{FACE}/tok_mlp", lambda: summarize(
        fit_probe(Xtr, ytr_t, Xte, yte_t, 3, hidden=512), 3))
    run(f"{FACE}/position_floor", lambda: summarize(
        fit_probe(_pos_feats(rtr), ytr_t, _pos_feats(rte), yte_t, 3), 3))

    for T in AX_TS:
        run(f"{FACE}/T{T}/visible_evidence_floor", lambda: summarize(
            fit_probe(bank.feats(ftr, T), ytr_t, bank.feats(fte, T), yte_t, 3), 3))
        Wtr, Wte = gather_win(acts, rtr, T), gather_win(acts, rte, T)
        atr, ate = anchor_ctxmean(Wtr), anchor_ctxmean(Wte)
        run(f"{FACE}/T{T}/actxmean_linear", lambda: summarize(
            fit_probe(atr, ytr_t, ate, yte_t, 3), 3))
        run(f"{FACE}/T{T}/actxmean_mlp", lambda: summarize(
            fit_probe(atr, ytr_t, ate, yte_t, 3, hidden=512), 3))
        if T in FOREIGN_TS:
            fa_tr = actxmean_foreign(Wtr, np.random.default_rng(FOREIGN_SEED + T))
            fa_te = actxmean_foreign(Wte, np.random.default_rng(FOREIGN_SEED + T + 1))
            run(f"{FACE}/T{T}/actxmean_foreign_linear", lambda: summarize(
                fit_probe(fa_tr, ytr_t, fa_te, yte_t, 3), 3))
            del fa_tr, fa_te
        del Wtr, Wte, atr, ate

    # label-shuffle null
    nrng = np.random.default_rng(NULL_SEED)
    W = gather_win(acts, rtr, 32)
    run(f"{FACE}/T32/label_null", lambda: summarize(
        fit_probe(anchor_ctxmean(W), torch.from_numpy(nrng.permutation(ytr)),
                  anchor_ctxmean(gather_win(acts, rte, 32)), yte_t, 3), 3))
    save()

    tok = max(cells[f"{FACE}/tok_linear"]["acc_test"],
              cells[f"{FACE}/tok_mlp"]["acc_test"])
    best, bestT = -1.0, None
    for T in AX_TS:
        for a in ("actxmean_linear", "actxmean_mlp"):
            v = cells[f"{FACE}/T{T}/{a}"]["acc_test"]
            if v > best:
                best, bestT = v, T
    floor = cells[f"{FACE}/T{bestT}/visible_evidence_floor"]["acc_test"]
    done["summary"] = {
        "tok": tok, "best_window": best, "best_T": bestT,
        "gain_vs_tok": best - tok, "floor_at_bestT": floor,
        "beats_floor": bool(best > floor),
        "position_floor": cells[f"{FACE}/position_floor"]["acc_test"],
        "gain_bar_0.05_cleared": bool(best - tok >= 0.05)}
    save()
    print(f"\n[{key}] tok={tok:.4f} best={best:.4f} (T{bestT}) "
          f"gain={best - tok:+.4f} floor={floor:.4f} "
          f"beats_floor={best > floor}", flush=True)


if __name__ == "__main__":
    for k in (sys.argv[1:] or ["gpt2"]):
        screen(k)
