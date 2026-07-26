"""Stage-1 screen — B9 quoted-speech intensity on PG19 fiction (executes CARD.md).

Face `qd` (3-class zero_split terciles of the trailing quote-sentence
kernel) from `../labels/quotedens_pg19_<tok>.npz` on the quotedens
caches. Binding preconditions, implemented:

  1. WITHIN-BOOK contrast (doc_mean_only_auc 0.890-0.896 label-side) —
     the mandatory control; the deepest within-doc substrate in the
     factory (125-127 test books at >= 20 manifest rows/class).
  2. Position-only floor probe on shipped rows.
  3. VISIBLE-EVIDENCE floor per T (refmark precedent): label-side probe
     on [window is_qd token count, window in-span fraction] printed
     beside every window number — event-sentence tokens are masked from
     probe ROWS but visible in window CONTEXT, exactly the refmark
     situation.
  4. The `is_qd` ambient anchor (binary, full-pool draw) per-token +
     T16 actxmean — the bracket-family regime-1 calibration face,
     never the primary.

Probe grid: the convention-of-record (slen/refmark precedent): tok
linear+MLP first; actxmean ± foreign at T ∈ {4,8,16,32,64}; order arms
(flatten/shuffle/foreign) linear at T ∈ {16,32}, MLP triple at T32;
permutation nulls at T16. Reach: ~12.9 tokens/sentence here, so T64
spans ≈ 5 of the kernel's 8 support sentences (≈ 0.87 mass) — the
best-spanned ladder of the three overnight screens, stated in the card.

Run: .venv/bin/python -m experiments.explorations.task_hunt.quotedens.screen [model ...]
Writes results/screen_<model>.json next to this file.
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
    foreign_context,
)
from experiments.explorations.task_hunt.novelty.screen import (
    CAP,
    MIN_ROWS,
    OFF_MIN,
    POS_MIN,
    SHUF_SEED,
    _map_rows,
    _row_lookup,
    _seeded,
    _stack,
)
from experiments.explorations.task_hunt.quotedens.cache_acts import (
    CACHE_ROOT,
    TAGS,
)
from experiments.explorations.task_hunt.replag.cache_acts import SCREEN_HS
from experiments.explorations.task_hunt.replag.screen import (
    gather_tok,
    gather_win,
    shuffle_context,
    summarize,
)

HERE = Path(__file__).resolve().parent
LABELS = HERE.parent / "labels"
RES = HERE / "results"

NULL_SEED = 99
AX_TS = [4, 8, 16, 32, 64]
ORD_TS = [16, 32]
ORD_MLP_T = [32]
NULL_T = 16
WD_TS = [16, 32, 64]
WD_ORD_TS = [16, 32]
WD_MIN_DOC_ROWS = 30
ANCHOR_CAP = {"train": 8000, "test": 3000}


def build_rows(key: str):
    tag = TAGS[key]
    z = np.load(LABELS / f"quotedens_pg19_{tag}.npz")
    c = np.load(CACHE_ROOT / key / "tokens.npz")
    ids, doc_idx, n_prefix = c["ids"], c["doc_idx"], int(c["n_prefix"])
    content = ids.shape[1] - n_prefix
    flat, off, doc_split = z["token_ids"], z["doc_off"], z["doc_split"]

    lookup = _row_lookup(doc_idx)
    for (d, k), i in list(lookup.items())[:200]:
        s = off[d] + k * content
        assert np.array_equal(flat[s:s + content], ids[i, n_prefix:]), \
            f"flat/window mismatch at doc {d} chunk {k}"

    n_tok = flat.shape[0]
    doc_of = np.searchsorted(off, np.arange(n_tok), side="right") - 1
    pos_of = np.arange(n_tok) - off[doc_of]
    rows_flat, cpos_flat = _map_rows(doc_of, pos_of, lookup, content,
                                     n_prefix)
    base_elig = ((rows_flat >= 0) & (pos_of >= POS_MIN)
                 & (pos_of % content >= OFF_MIN))

    out: dict = {}
    stats = {"n_prefix": n_prefix, "content": content,
             "n_cache_rows": int(ids.shape[0])}

    # ---- PRIMARY: builder manifest ----
    md, mp, mc = z["man_qd_doc"], z["man_qd_pos"], z["man_qd_cls"]
    man_flat = off[md] + mp
    rows_all, cpos_all = _map_rows(md, mp, lookup, content, n_prefix)
    elig = (rows_all >= 0) & (mp >= POS_MIN) & (mp % content >= OFF_MIN)
    for split_name, flag in (("train", 0), ("test", 1)):
        m = elig & (doc_split[md] == flag)
        per, keep_r, keep_y, keep_f = {}, [], [], []
        for cls in (0, 1, 2):
            idx = np.flatnonzero(m & (mc == cls))
            rng = _seeded(f"quotedens/qd/{key}/{split_name}/{cls}")
            if len(idx) > CAP[split_name]:
                idx = rng.choice(idx, CAP[split_name], replace=False)
            per[cls] = len(idx)
            keep_r.append(_stack(rows_all, cpos_all, mp, idx))
            keep_y.append(np.full(len(idx), cls, dtype=np.int64))
            keep_f.append(man_flat[idx])
        stats[f"qd/{split_name}"] = {
            "rows_per_class": per,
            "ok": bool(min(per.values()) >= MIN_ROWS)}
        out[("qd", split_name)] = (np.concatenate(keep_r),
                                   np.concatenate(keep_y))
        out[("qd_flat", split_name)] = np.concatenate(keep_f)

    # ---- ambient anchor is_qd (full pool) ----
    av = z["is_qd"].astype(np.int64)
    for split_name, flag in (("train", 0), ("test", 1)):
        m = base_elig & (doc_split[doc_of] == flag)
        n_take = min(int((m & (av == v)).sum()) for v in (0, 1))
        n_take = min(n_take, ANCHOR_CAP[split_name])
        per, ar, ay = {}, [], []
        for v in (0, 1):
            idx = np.flatnonzero(m & (av == v))
            rng = _seeded(f"quotedensanchor/{key}/{split_name}/{v}")
            if len(idx) > n_take:
                idx = rng.choice(idx, n_take, replace=False)
            per[v] = len(idx)
            ar.append(_stack(rows_flat, cpos_flat, pos_of, idx))
            ay.append(np.full(len(idx), v, dtype=np.int64))
        stats[f"anchor/{split_name}"] = {
            "rows_per_class": per, "ok": bool(min(per.values()) >= MIN_ROWS)}
        out[("anchor", split_name)] = (np.concatenate(ar),
                                       np.concatenate(ay))

    # ---- within-book control (binding) ----
    val = z["lam_qd"].astype(np.float64)
    elig_wd = base_elig & np.isfinite(val) & (z["qd_bin"] >= 0)
    for split_name, flag in (("train", 0), ("test", 1)):
        m = elig_wd & (doc_split[doc_of] == flag)
        lo_idx, hi_idx, n_docs_used = [], [], 0
        for d in np.unique(doc_of[m]):
            sel = np.flatnonzero(m & (doc_of == d))
            v = val[sel]
            if len(sel) < WD_MIN_DOC_ROWS:
                continue
            q1, q2 = np.quantile(v, [1 / 3, 2 / 3])
            if not (q2 > q1):
                continue
            lo_idx.append(sel[v <= q1])
            hi_idx.append(sel[v >= q2])
            n_docs_used += 1
        if not lo_idx:
            stats[f"wd/{split_name}"] = {"ok": False, "n_docs": 0}
            continue
        lo, hi = np.concatenate(lo_idx), np.concatenate(hi_idx)
        n_take = min(len(lo), len(hi), CAP[split_name] * 2)
        keep_r, keep_y = [], []
        for v_, idx in ((0, lo), (1, hi)):
            rng = _seeded(f"quotedenswd/{key}/{split_name}/{v_}")
            if len(idx) > n_take:
                idx = rng.choice(idx, n_take, replace=False)
            keep_r.append(_stack(rows_flat, cpos_flat, pos_of, idx))
            keep_y.append(np.full(len(idx), v_, dtype=np.int64))
        stats[f"wd/{split_name}"] = {
            "ok": bool(n_take >= MIN_ROWS), "n_per_class": int(n_take),
            "n_docs": int(n_docs_used)}
        out[("wd", split_name)] = (np.concatenate(keep_r),
                                   np.concatenate(keep_y))
    return out, stats, z


def _pos_feats(r):
    cp = r[:, 1].astype(np.float32)
    dp = r[:, 2].astype(np.float32)
    f = np.stack([cp, cp ** 2 / 128.0, np.log2(1.0 + dp), dp / 1000.0], 1)
    return torch.from_numpy(f).to(torch.float16)


def _visible_feats(z, flat_pos, T):
    """Window-visible evidence: is_qd token count + in-span fraction."""
    iq = z["is_qd"].astype(np.float32)
    sp = z["in_span"].astype(np.float32)
    idx = flat_pos[:, None] - np.arange(T - 1, -1, -1)[None, :]
    f = np.stack([iq[idx].sum(1), sp[idx].mean(1)], 1)
    return torch.from_numpy(f).to(torch.float16)


def screen(key: str):
    RES.mkdir(exist_ok=True)
    out_path = RES / f"screen_{key}.json"
    hs = SCREEN_HS[key]
    manifests, mstats, z = build_rows(key)
    done = json.loads(out_path.read_text()) if out_path.exists() else {
        "meta": {"model": key, "screen_hs": hs, "card": "CARD.md (frozen)",
                 "ax_ts": AX_TS, "ord_ts": ORD_TS, "null_t": NULL_T,
                 "wd_ts": WD_TS, "foreign_seed": FOREIGN_SEED,
                 "rows": mstats,
                 "reach": "~12.9 tok/sentence; T64 ≈ 5 of 8 kernel-"
                          "support sentences ≈ 0.87 mass (card § 2)"},
        "cells": {}}
    cells = done["cells"]

    def save():
        out_path.write_text(json.dumps(done, indent=1))

    def run(cell_key, fn):
        if cell_key in cells:
            return
        t0 = time.time()
        cells[cell_key] = fn()
        cells[cell_key]["wall_s"] = round(time.time() - t0, 1)
        print(f"[{key} {cell_key}] "
              + " ".join(f"{k}={v:.3f}" for k, v in cells[cell_key].items()
                         if isinstance(v, float) and k != "wall_s"),
              flush=True)
        save()

    acts = torch.from_numpy(np.ascontiguousarray(
        np.load(CACHE_ROOT / key / f"hs{hs}.npy", mmap_mode="r")))

    if mstats["qd/train"]["ok"] and mstats["qd/test"]["ok"]:
        rtr, ytr = manifests[("qd", "train")]
        rte, yte = manifests[("qd", "test")]
        ftr = manifests[("qd_flat", "train")]
        fte = manifests[("qd_flat", "test")]
        ytr_t, yte_t = torch.from_numpy(ytr), torch.from_numpy(yte)

        Xtr_tok, Xte_tok = gather_tok(acts, rtr), gather_tok(acts, rte)
        run("qd/tok_linear", lambda: summarize(
            fit_probe(Xtr_tok, ytr_t, Xte_tok, yte_t, 3), 3))
        run("qd/tok_mlp", lambda: summarize(
            fit_probe(Xtr_tok, ytr_t, Xte_tok, yte_t, 3, hidden=512), 3))
        run("qd/position_floor", lambda: summarize(
            fit_probe(_pos_feats(rtr), ytr_t, _pos_feats(rte), yte_t, 3),
            3))

        for T in AX_TS:
            run(f"qd/T{T}/visible_evidence_floor", lambda: summarize(
                fit_probe(_visible_feats(z, ftr, T), ytr_t,
                          _visible_feats(z, fte, T), yte_t, 3), 3))
            Wtr, Wte = gather_win(acts, rtr, T), gather_win(acts, rte, T)
            atr, ate = anchor_ctxmean(Wtr), anchor_ctxmean(Wte)
            run(f"qd/T{T}/actxmean_linear", lambda: summarize(
                fit_probe(atr, ytr_t, ate, yte_t, 3), 3))
            run(f"qd/T{T}/actxmean_mlp", lambda: summarize(
                fit_probe(atr, ytr_t, ate, yte_t, 3, hidden=512), 3))
            fatr = actxmean_foreign(
                Wtr, np.random.default_rng(FOREIGN_SEED + T))
            fate = actxmean_foreign(
                Wte, np.random.default_rng(FOREIGN_SEED + T + 1))
            run(f"qd/T{T}/actxmean_foreign_linear", lambda: summarize(
                fit_probe(fatr, ytr_t, fate, yte_t, 3), 3))
            run(f"qd/T{T}/actxmean_foreign_mlp", lambda: summarize(
                fit_probe(fatr, ytr_t, fate, yte_t, 3, hidden=512), 3))
            del atr, ate, fatr, fate
            if T in ORD_TS:
                flat_tr = Wtr.reshape(len(rtr), -1)
                flat_te = Wte.reshape(len(rte), -1)
                run(f"qd/T{T}/win_linear", lambda: summarize(
                    fit_probe(flat_tr, ytr_t, flat_te, yte_t, 3), 3))
                srng = np.random.default_rng(
                    SHUF_SEED + zlib.crc32(f"qd/T{T}".encode()) % 2 ** 16)
                Str = shuffle_context(Wtr, srng).reshape(len(rtr), -1)
                Ste = shuffle_context(Wte, srng).reshape(len(rte), -1)
                run(f"qd/T{T}/win_shuf_linear", lambda: summarize(
                    fit_probe(Str, ytr_t, Ste, yte_t, 3), 3))
                fwtr = foreign_context(
                    Wtr, np.random.default_rng(FOREIGN_SEED + T)
                ).reshape(len(rtr), -1)
                fwte = foreign_context(
                    Wte, np.random.default_rng(FOREIGN_SEED + T + 1)
                ).reshape(len(rte), -1)
                run(f"qd/T{T}/win_foreign_linear", lambda: summarize(
                    fit_probe(fwtr, ytr_t, fwte, yte_t, 3), 3))
                if T in ORD_MLP_T:
                    run(f"qd/T{T}/win_mlp", lambda: summarize(
                        fit_probe(flat_tr, ytr_t, flat_te, yte_t, 3,
                                  hidden=512), 3))
                    run(f"qd/T{T}/win_shuf_mlp", lambda: summarize(
                        fit_probe(Str, ytr_t, Ste, yte_t, 3,
                                  hidden=512), 3))
                    run(f"qd/T{T}/win_foreign_mlp", lambda: summarize(
                        fit_probe(fwtr, ytr_t, fwte, yte_t, 3,
                                  hidden=512), 3))
                if T == NULL_T:
                    nrng = np.random.default_rng(NULL_SEED)
                    yn = torch.from_numpy(nrng.permutation(ytr))
                    run(f"qd/T{T}/null_win_linear", lambda: summarize(
                        fit_probe(flat_tr, yn, flat_te, yte_t, 3), 3))
                    run("qd/null_tok_linear", lambda: summarize(
                        fit_probe(Xtr_tok, yn, Xte_tok, yte_t, 3), 3))
                del flat_tr, flat_te, Str, Ste, fwtr, fwte
            del Wtr, Wte
        del Xtr_tok, Xte_tok
    else:
        print(f"[{key} qd] SKIP (insufficient rows)")

    if mstats["anchor/train"]["ok"] and mstats["anchor/test"]["ok"]:
        rtr, ytr = manifests[("anchor", "train")]
        rte, yte = manifests[("anchor", "test")]
        ytr_t, yte_t = torch.from_numpy(ytr), torch.from_numpy(yte)
        Xtr, Xte = gather_tok(acts, rtr), gather_tok(acts, rte)
        run("anchor/tok_linear", lambda: summarize(
            fit_probe(Xtr, ytr_t, Xte, yte_t, 2, class_weight=True), 2))
        W1, W2 = gather_win(acts, rtr, 16), gather_win(acts, rte, 16)
        run("anchor/T16/actxmean_linear", lambda: summarize(
            fit_probe(anchor_ctxmean(W1), ytr_t, anchor_ctxmean(W2),
                      yte_t, 2, class_weight=True), 2))
        del Xtr, Xte, W1, W2

    if (mstats.get("wd/train", {}).get("ok")
            and mstats.get("wd/test", {}).get("ok")):
        rtr, ytr = manifests[("wd", "train")]
        rte, yte = manifests[("wd", "test")]
        ytr_t, yte_t = torch.from_numpy(ytr), torch.from_numpy(yte)
        Xtr, Xte = gather_tok(acts, rtr), gather_tok(acts, rte)
        run("wd/tok_linear", lambda: summarize(
            fit_probe(Xtr, ytr_t, Xte, yte_t, 2, class_weight=True), 2))
        run("wd/tok_mlp", lambda: summarize(
            fit_probe(Xtr, ytr_t, Xte, yte_t, 2, hidden=512,
                      class_weight=True), 2))
        for T in sorted(set(WD_TS) | set(WD_ORD_TS)):
            Wtr, Wte = gather_win(acts, rtr, T), gather_win(acts, rte, T)
            if T in WD_TS:
                atr, ate = anchor_ctxmean(Wtr), anchor_ctxmean(Wte)
                run(f"wd/T{T}/actxmean_linear", lambda: summarize(
                    fit_probe(atr, ytr_t, ate, yte_t, 2,
                              class_weight=True), 2))
                run(f"wd/T{T}/actxmean_mlp", lambda: summarize(
                    fit_probe(atr, ytr_t, ate, yte_t, 2, hidden=512,
                              class_weight=True), 2))
                fatr = actxmean_foreign(
                    Wtr, np.random.default_rng(FOREIGN_SEED + T))
                fate = actxmean_foreign(
                    Wte, np.random.default_rng(FOREIGN_SEED + T + 1))
                run(f"wd/T{T}/actxmean_foreign_linear", lambda: summarize(
                    fit_probe(fatr, ytr_t, fate, yte_t, 2,
                              class_weight=True), 2))
                del atr, ate, fatr, fate
            if T in WD_ORD_TS:
                flat_tr = Wtr.reshape(len(rtr), -1)
                flat_te = Wte.reshape(len(rte), -1)
                run(f"wd/T{T}/win_linear", lambda: summarize(
                    fit_probe(flat_tr, ytr_t, flat_te, yte_t, 2,
                              class_weight=True), 2))
                srng = np.random.default_rng(
                    SHUF_SEED + zlib.crc32(f"wd/T{T}".encode()) % 2 ** 16)
                Str = shuffle_context(Wtr, srng).reshape(len(rtr), -1)
                Ste = shuffle_context(Wte, srng).reshape(len(rte), -1)
                run(f"wd/T{T}/win_shuf_linear", lambda: summarize(
                    fit_probe(Str, ytr_t, Ste, yte_t, 2,
                              class_weight=True), 2))
                del flat_tr, flat_te, Str, Ste
            del Wtr, Wte
        del Xtr, Xte
    else:
        print(f"[{key} wd] SKIP (insufficient within-book rows — a SKIP "
              f"here blocks any KEEP, card)")
    del acts
    save()
    print(f"[{key}] DONE -> {out_path}", flush=True)


def main():
    for k in (sys.argv[1:] or ["gpt2", "llama31_8b"]):
        screen(k)


if __name__ == "__main__":
    main()
