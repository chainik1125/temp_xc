"""The R11 ORDER-MECHANISM LADDER (executes LADDER_CARD.md, day-2 W1).

Decomposes dialevel's anchor-fixed context-shuffle cost (R11: +0.0567 /
+0.0626 / +0.0349 AUC at T = 32, the one order-carried window signal
outside backtracking) into:

  L0  full context shuffle       (screen-exact at seed 0 -> gate)
  L1  within-turn shuffle        (turn sequence intact)
  L2  turn-block permutation     (within-turn order intact)
  L3f/L3n  far-/near-half shuffle (residual recency probe)
  L4  foreign context            (width null, capacity_check-exact)

All arms: anchor slot T-1 FIXED, wd rows = the frozen screen's
`build_rows()` verbatim, probe = frozen `problib.fit_probe` linear on
the T*d flatten (matched probe class, no max-over-arms). Turn spans
come from the committed labels npz `turn_idx` via flat-index
reconstruction; identity is asserted token-by-token before any fit.

Run: .venv/bin/python -m experiments.explorations.task_hunt.dialevel.ladder [model ...]
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
from experiments.explorations.task_hunt.dialevel.cache_acts import (
    CACHE_ROOT,
    TOK_TAG,
)
from experiments.explorations.task_hunt.dialevel.capacity_check import (
    FOREIGN_SEED,
    foreign_context,
)
from experiments.explorations.task_hunt.dialevel.screen import (
    MODELS,
    build_rows,
)
from experiments.explorations.task_hunt.novelty.screen import (
    NULL_SEED,
    SHUF_SEED,
    _seeded,
)
from experiments.explorations.task_hunt.replag.cache_acts import SCREEN_HS
from experiments.explorations.task_hunt.replag.screen import (
    gather_win,
    summarize,
)

HERE = Path(__file__).resolve().parent
LABELS = HERE.parent / "labels"
RES = HERE / "results"
LADDER_TS = (16, 32)          # T32 = the R11 anchor, T16 = robustness
N_SEEDS = 3
T_MAX = max(LADDER_TS)


# ---------- flat-index reconstruction + turn spans ---------------------

def flat_anchor_of(rows, doc_idx, off, content, n_prefix):
    """Invert the (doc,pos)->(cache_row,cache_pos) map for shipped rows."""
    r = rows[:, 0].astype(np.int64)
    doc = doc_idx[r].astype(np.int64)
    first = np.searchsorted(doc_idx, doc, side="left")
    chunk = r - first
    return off[doc] + chunk * content + (rows[:, 1].astype(np.int64)
                                         - n_prefix)


def slot_turns(flat_anchor, turn_idx, T):
    """(n, T-1) turn id per context slot; slot j <-> flat - (T-1) + j."""
    offs = np.arange(-(T - 1), 0)
    return turn_idx[flat_anchor[:, None] + offs]


# ---------- permutation builders (context slots 0..T-2 only) -----------

def l0_perm(rng, n, C):
    """Screen-identical draw at matched rng: uniform over all slots."""
    return rng.permuted(np.tile(np.arange(C), (n, 1)), axis=1)


def l1_perm(rng, st):
    """Within-turn shuffle: permute inside each turn_idx group."""
    n, C = st.shape
    perm = np.tile(np.arange(C), (n, 1))
    for i in range(n):
        row = st[i]
        for t in np.unique(row):
            g = np.flatnonzero(row == t)
            if len(g) > 1:
                perm[i, g] = perm[i, g][rng.permutation(len(g))]
    return perm


def l2_perm(rng, st):
    """Turn-block permutation: reorder maximal same-turn runs,
    within-block order intact."""
    n, C = st.shape
    perm = np.empty((n, C), dtype=np.int64)
    for i in range(n):
        row = st[i]
        bnd = np.flatnonzero(np.r_[True, row[1:] != row[:-1]])
        blocks = np.split(np.arange(C), bnd[1:])
        order = rng.permutation(len(blocks))
        perm[i] = np.concatenate([blocks[o] for o in order])
    return perm


def l3_perm(rng, n, C, which):
    """Shuffle ONLY the far (slots 0..h-1) or near (h..C-1) half."""
    h = C // 2
    sl = np.arange(0, h) if which == "far" else np.arange(h, C)
    perm = np.tile(np.arange(C), (n, 1))
    perm[:, sl] = rng.permuted(np.tile(sl, (n, 1)), axis=1)
    return perm


def apply_perm(W, perm):
    out = W.clone()
    n, T, _ = W.shape
    out[:, :T - 1] = W[torch.arange(n)[:, None], torch.from_numpy(perm)]
    return out


def moved_frac(perm):
    return float((perm != np.arange(perm.shape[1])[None, :]).mean())


# ---------- disclosures (card § 2) -------------------------------------

def entropy_stats(st):
    n, C = st.shape
    turns, blocks, shuf_slots, multi_block = [], [], 0, 0
    for i in range(n):
        row = st[i]
        u, counts = np.unique(row, return_counts=True)
        turns.append(len(u))
        nb = 1 + int((row[1:] != row[:-1]).sum())
        blocks.append(nb)
        shuf_slots += int(counts[counts > 1].sum())
        multi_block += int(nb > 1)
    return {"turns_per_window_mean": float(np.mean(turns)),
            "blocks_mean": float(np.mean(blocks)),
            "frac_slots_in_shufflable_groups": shuf_slots / (n * C),
            "frac_rows_multi_block": multi_block / n}


# ---------- the ladder -------------------------------------------------

def ladder(key: str):
    RES.mkdir(exist_ok=True)
    out_path = RES / f"ladder_{key}.json"
    hs = SCREEN_HS[key]
    man, mstats = build_rows(key)
    z = np.load(LABELS / f"dialevel_dailydialog_{TOK_TAG[key]}.npz")
    c = np.load(CACHE_ROOT / key / "tokens.npz")
    ids, doc_idx, n_prefix = c["ids"], c["doc_idx"], int(c["n_prefix"])
    content = ids.shape[1] - n_prefix
    turn_idx, flat_tokens, off = z["turn_idx"], z["token_ids"], z["doc_off"]

    done = json.loads(out_path.read_text()) if out_path.exists() else {
        "meta": {"model": key, "screen_hs": hs,
                 "card": "LADDER_CARD.md (frozen)",
                 "ladder_ts": list(LADDER_TS), "n_seeds": N_SEEDS},
        "cells": {}}
    done["meta"]["rows"] = mstats
    cells = done["cells"]

    def save():
        out_path.write_text(json.dumps(done, indent=1))

    def run(k, fn, extra=None):
        if k in cells:
            return
        t0 = time.time()
        cells[k] = fn()
        if extra:
            cells[k].update(extra)
        cells[k]["wall_s"] = round(time.time() - t0, 1)
        print(f"[{key} {k}] " + " ".join(
            f"{a}={b:.3f}" for a, b in cells[k].items()
            if isinstance(b, float) and a != "wall_s"), flush=True)
        save()

    # --- identity receipts (card § 1 gates 1-2) ------------------------
    rtr, ytr = man[("wd", "train")]
    rte, yte = man[("wd", "test")]
    for tag, rows in (("train", rtr), ("test", rte)):
        fa = flat_anchor_of(rows, doc_idx, off, content, n_prefix)
        got = ids[rows[:, 0].astype(np.int64), rows[:, 1].astype(np.int64)]
        assert np.array_equal(flat_tokens[fa], got), \
            f"anchor identity broken ({tag})"
        assert (rows[:, 1].astype(np.int64) - n_prefix >= T_MAX - 1).all(), \
            f"window would leave the cache row ({tag})"
    fa_tr = flat_anchor_of(rtr, doc_idx, off, content, n_prefix)
    fa_te = flat_anchor_of(rte, doc_idx, off, content, n_prefix)
    done["meta"]["identity_ok"] = {"train": int(len(fa_tr)),
                                   "test": int(len(fa_te))}
    ytr_t, yte_t = torch.from_numpy(ytr), torch.from_numpy(yte)

    acts = torch.from_numpy(np.ascontiguousarray(np.load(
        CACHE_ROOT / key / f"hs{hs}.npy", mmap_mode="r")))

    for T in LADDER_TS:
        C = T - 1
        st_tr = slot_turns(fa_tr, turn_idx, T)
        st_te = slot_turns(fa_te, turn_idx, T)
        done["meta"][f"T{T}/entropy"] = entropy_stats(
            np.concatenate([st_tr, st_te]))
        Wtr, Wte = gather_win(acts, rtr, T), gather_win(acts, rte, T)
        ntr, nte = len(rtr), len(rte)

        def fit(ftr, fte):
            return summarize(fit_probe(
                ftr.reshape(ntr, -1), ytr_t, fte.reshape(nte, -1), yte_t,
                2, class_weight=True), 2)

        run(f"T{T}/base", lambda: fit(Wtr, Wte))

        # label-permutation null (T16 = screen-exact draw)
        nrng = np.random.default_rng(NULL_SEED if T == 16
                                     else NULL_SEED + T)
        yn = torch.from_numpy(nrng.permutation(ytr))
        run(f"T{T}/null_label", lambda: summarize(fit_probe(
            Wtr.reshape(ntr, -1), yn, Wte.reshape(nte, -1), yte_t, 2,
            class_weight=True), 2))

        for s in range(N_SEEDS):
            # L0 seed 0 = the screen's EXACT generator + draw order
            if s == 0:
                srng = np.random.default_rng(
                    SHUF_SEED + zlib.crc32(f"wd/T{T}".encode()) % 2 ** 16)
                p_tr = l0_perm(srng, ntr, C)
                p_te = l0_perm(srng, nte, C)
            else:
                p_tr = l0_perm(_seeded(f"dialevel/ladder/L0/T{T}/s{s}/train"),
                               ntr, C)
                p_te = l0_perm(_seeded(f"dialevel/ladder/L0/T{T}/s{s}/test"),
                               nte, C)
            run(f"T{T}/L0/s{s}",
                lambda: fit(apply_perm(Wtr, p_tr), apply_perm(Wte, p_te)),
                {"moved_frac": moved_frac(np.concatenate([p_tr, p_te]))})

            for arm, builder in (("L1", l1_perm), ("L2", l2_perm)):
                p_tr = builder(
                    _seeded(f"dialevel/ladder/{arm}/T{T}/s{s}/train"), st_tr)
                p_te = builder(
                    _seeded(f"dialevel/ladder/{arm}/T{T}/s{s}/test"), st_te)
                run(f"T{T}/{arm}/s{s}",
                    lambda: fit(apply_perm(Wtr, p_tr), apply_perm(Wte, p_te)),
                    {"moved_frac": moved_frac(np.concatenate([p_tr, p_te]))})

            for arm, which in (("L3f", "far"), ("L3n", "near")):
                p_tr = l3_perm(
                    _seeded(f"dialevel/ladder/{arm}/T{T}/s{s}/train"),
                    ntr, C, which)
                p_te = l3_perm(
                    _seeded(f"dialevel/ladder/{arm}/T{T}/s{s}/test"),
                    nte, C, which)
                run(f"T{T}/{arm}/s{s}",
                    lambda: fit(apply_perm(Wtr, p_tr), apply_perm(Wte, p_te)),
                    {"moved_frac": moved_frac(np.concatenate([p_tr, p_te]))})

        # L4 foreign — capacity_check's EXACT seeds and draw
        run(f"T{T}/L4", lambda: fit(
            foreign_context(Wtr, np.random.default_rng(FOREIGN_SEED + T)),
            foreign_context(Wte, np.random.default_rng(FOREIGN_SEED + T + 1))))

        del Wtr, Wte
        save()

    del acts
    save()
    print(f"[{key}] LADDER DONE -> {out_path}", flush=True)


def main():
    for k in (sys.argv[1:] or list(MODELS)):
        ladder(k)


if __name__ == "__main__":
    main()
