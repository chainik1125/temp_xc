"""WITHIN-DOCUMENT control for the punctint faces — the decisive test.

**Status: committed BEFORE it runs** (git order is the evidence). This
is a POST-HOC diagnostic added after the frozen screen's cells landed;
it does not alter `CARD.md`'s KEEP/KILL scoring, it determines what the
record is ALLOWED TO CLAIM about it (the same discipline the Stage-2
probe-capacity diagnostic followed).

## Why it exists

The screen's rising T-gap can be produced without any trailing/temporal
structure. Measured on the label side, over the screened eligible pool:

  face   between-doc var   doc-mean-only AUC (top vs bottom tercile)
  q      32 %              **0.926**
  list   57 %              **0.960**

i.e. knowing ONLY which document a row sits in predicts its intensity
tercile at 0.93–0.96 AUC. A window-MEAN over 64 tokens is a strong
document/topic signature, so "the gap grows with T" is exactly what a
better doc-identity descriptor would produce. The builder's frozen
triage bars cannot see this: the unigram type-mean bar is a per-token
IDENTITY statistic and the position bar a within-doc ordinate; neither
tests document identity. (The ledger flagged this face as
"between-doc-heavy" — this is that risk, measured.)

## The control

Assign classes by rank **within each document**: among a document's
eligible rows, bottom tercile vs top tercile of the face's own λ̂.
Document identity then carries ZERO information about the label by
construction, so any surviving window advantage is genuine
within-document trailing structure.

Balanced per class, split by the builder's `doc_split`, same caches,
same frozen probe stack, same eligibility as the screen. Reported:
per-token vs window-MEAN at T ∈ {16, 32, 64} (binary, rank-AUC).

Run: .venv/bin/python -m experiments.explorations.task_hunt.qrate_fineweb.within_doc
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiments.explorations.conversion_depth.problib import fit_probe
from experiments.explorations.task_hunt.novelty.screen import (
    CAP,
    MIN_ROWS,
    MODELS,
    OFF_MIN,
    POS_MIN,
    _map_rows,
    _row_lookup,
    _seeded,
    _stack,
)
from experiments.explorations.task_hunt.replag.cache_acts import SCREEN_HS
from experiments.explorations.task_hunt.replag.screen import (
    gather_tok,
    gather_win,
    summarize,
    win_mean,
)

HERE = Path(__file__).resolve().parent
LABELS = HERE.parent / "labels"
CACHE_ROOT = Path("/workspace/replag_caches")
RES = HERE / "results"
TS = (16, 32, 64)
FACES = (("q", "lam_q"), ("list", "lam_list"))


def build(key: str):
    tag = MODELS[key]
    z = np.load(LABELS / f"punctint_fineweb_{tag}.npz")
    c = np.load(CACHE_ROOT / key / "tokens.npz")
    ids, doc_idx, n_prefix = c["ids"], c["doc_idx"], int(c["n_prefix"])
    content = ids.shape[1] - n_prefix
    off, doc_split = z["doc_off"], z["doc_split"]
    n = z["token_ids"].shape[0]
    doc_of = np.searchsorted(off, np.arange(n), side="right") - 1
    pos_of = np.arange(n) - off[doc_of]
    lookup = _row_lookup(doc_idx)
    rows_all, cpos_all = _map_rows(doc_of, pos_of, lookup, content, n_prefix)

    out, stats = {}, {}
    for face, lamkey in FACES:
        lam = z[lamkey].astype(np.float64)
        elig = ((rows_all >= 0) & (pos_of >= POS_MIN)
                & (pos_of % content >= OFF_MIN) & np.isfinite(lam))
        for split_name, flag in (("train", 0), ("test", 1)):
            m = elig & (doc_split[doc_of] == flag)
            lo_idx, hi_idx, n_docs_used = [], [], 0
            for d in np.unique(doc_of[m]):
                sel = np.flatnonzero(m & (doc_of == d))
                v = lam[sel]
                if len(sel) < 30:
                    continue
                q1, q2 = np.quantile(v, [1 / 3, 2 / 3])
                if not (q2 > q1):          # no within-doc variation
                    continue
                lo_idx.append(sel[v <= q1])
                hi_idx.append(sel[v >= q2])
                n_docs_used += 1
            if not lo_idx:
                stats[f"{face}/{split_name}"] = {"ok": False, "n_docs": 0}
                continue
            lo = np.concatenate(lo_idx)
            hi = np.concatenate(hi_idx)
            n_take = min(len(lo), len(hi), CAP[split_name] * 2)
            keep_r, keep_y = [], []
            for v, idx in ((0, lo), (1, hi)):
                rng = _seeded(f"withindoc/{face}/{key}/{split_name}/{v}")
                if len(idx) > n_take:
                    idx = rng.choice(idx, n_take, replace=False)
                keep_r.append(_stack(rows_all, cpos_all, pos_of, idx))
                keep_y.append(np.full(len(idx), v, dtype=np.int64))
            stats[f"{face}/{split_name}"] = {
                "ok": bool(n_take >= MIN_ROWS), "n_per_class": int(n_take),
                "n_docs": int(n_docs_used)}
            out[(face, split_name)] = (np.concatenate(keep_r),
                                       np.concatenate(keep_y))
    return out, stats


def run_model(key: str, done: dict):
    hs = SCREEN_HS[key]
    man, stats = build(key)
    done.setdefault("meta", {})[key] = {"screen_hs": hs, "rows": stats}
    cells = done.setdefault("cells", {})
    acts = torch.from_numpy(np.ascontiguousarray(
        np.load(CACHE_ROOT / key / f"hs{hs}.npy", mmap_mode="r")))

    def run(k, fn):
        if k in cells:
            return
        t0 = time.time()
        cells[k] = fn()
        cells[k]["wall_s"] = round(time.time() - t0, 1)
        print(f"[{k}] " + " ".join(f"{a}={b:.3f}" for a, b in cells[k].items()
                                   if isinstance(b, float) and a != "wall_s"),
              flush=True)
        (RES / "within_doc.json").write_text(json.dumps(done, indent=1))

    for face, _ in FACES:
        if not (stats.get(f"{face}/train", {}).get("ok")
                and stats.get(f"{face}/test", {}).get("ok")):
            print(f"[{key} {face}] SKIP (insufficient within-doc rows)")
            continue
        rtr, ytr = man[(face, "train")]
        rte, yte = man[(face, "test")]
        ytr_t, yte_t = torch.from_numpy(ytr), torch.from_numpy(yte)
        Xtr, Xte = gather_tok(acts, rtr), gather_tok(acts, rte)
        run(f"{key}/{face}/tok_linear", lambda: summarize(
            fit_probe(Xtr, ytr_t, Xte, yte_t, 2, class_weight=True), 2))
        for T in TS:
            Wtr, Wte = gather_win(acts, rtr, T), gather_win(acts, rte, T)
            run(f"{key}/{face}/T{T}/win_mean_linear", lambda: summarize(
                fit_probe(win_mean(Wtr), ytr_t, win_mean(Wte), yte_t, 2,
                          class_weight=True), 2))
            del Wtr, Wte
        del Xtr, Xte
    del acts


def main():
    RES.mkdir(exist_ok=True)
    p = RES / "within_doc.json"
    done = json.loads(p.read_text()) if p.exists() else {}
    for k in (sys.argv[1:] or list(MODELS)):
        run_model(k, done)
    p.write_text(json.dumps(done, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main()
