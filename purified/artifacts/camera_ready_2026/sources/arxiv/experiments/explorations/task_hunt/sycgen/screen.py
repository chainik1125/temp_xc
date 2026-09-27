"""SYCGEN screen — generator-mode sycophancy age on the v1 corpus
(executes SCREEN_CARD.md; GO dc3cb8fd9, five binding conditions).

reask_hr/screen.py transplant, ONE face (`sycgen_age`;
`sycgen_rate` stays DEMOTED per PRECOUNT § 7.1). The WITHIN-DOMAIN
frame (GO condition 1) enters at manifest construction: tercile bins
are DOMAIN-LOCAL (edges asserted against the committed
`sycgen_domain_readout.json` — the disposition-(c) artifact), position
strata and the balanced manifest are drawn PER DOMAIN then
concatenated, so every (class × stratum) cell is domain-pure and every
arm — token, floors, window, shuffles, foreign — consumes identical
domain-local-tercile manifests. Per-token arms run FIRST (condition 2).
The within-domain vocab band (condition 3) = per-domain unigram AUC
(train-fit `type_mean_scores`, domain-restricted) + the two-leg
events/conv + tokens/conv spread within each domain — carried in the
output BESIDE the cells. Age is computed in-screen from the committed
grid event arrays via frozen `wave3_lib` functions.

Deviation from reask_hr, disclosed: no `is_boundary` term (sycgen has
no boundary construct — challenge turns are the only events and are
fully masked).

Run: .venv/bin/python -m experiments.explorations.task_hunt.sycgen.screen [model ...]
Writes results/screen_sycgen_<model>.json (resumable).
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
from experiments.explorations.task_hunt.labels import novelty_lib as nl
from experiments.explorations.task_hunt.labels import wave3_lib as w3
from experiments.explorations.task_hunt.labels.build_wave3_trio import (
    MIN_POS as PRE_MIN_POS,
    _terciles,
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
    SHUF_SEED,
    _map_rows,
    _row_lookup,
    _seeded,
    _stack,
)
from experiments.explorations.task_hunt.replag.cache_acts import SCREEN_HS
from experiments.explorations.task_hunt.replag.screen import (
    gather_tok,
    gather_win,
    shuffle_context,
    summarize,
)
from experiments.explorations.task_hunt.sycgen.cache_acts import (
    CACHE_ROOT,
    TOK_TAG,
)

HERE = Path(__file__).resolve().parent
GRIDS = HERE / "grids"
LABELS = HERE.parent / "labels"
RES = HERE / "results"

FACE = "sycgen_age"
NULL_SEED = 99
AX_TS = [4, 8, 16, 32, 64]
ORD_TS = [4, 8, 16, 32]
FOREIGN_ORD_TS = [16, 32]
ORD_MLP_T = [32]
NULL_T = 16
WD_TS = [16, 32, 64]
WD_ORD_TS = [4, 8, 16, 32]
WD_FOREIGN_TS = [16, 32]
WD_MIN_DOC_ROWS = 30


def build_rows(key: str):
    tag = TOK_TAG[key]
    z = np.load(GRIDS / f"elicit_sycgen_screen_{tag}.npz")
    c = np.load(CACHE_ROOT / key / "tokens.npz")
    ids, doc_idx, n_prefix = c["ids"], c["doc_idx"], int(c["n_prefix"])
    content = ids.shape[1] - n_prefix
    flat, off = z["token_ids"], z["doc_off"]
    is_assist = z["is_assistant"]
    first, mask = z["event_first"], z["event_mask"]
    doc_split, dom_of_doc = z["doc_split"], z["doc_domain"]
    n_docs = len(off) - 1

    age = np.concatenate([w3.sage_face(first[off[d]:off[d + 1]])
                          for d in range(n_docs)]).astype(np.float64)

    lookup = _row_lookup(doc_idx)
    for (d, k), i in list(lookup.items())[:200]:
        s = off[d] + k * content
        assert np.array_equal(flat[s:s + content], ids[i, n_prefix:]), \
            f"flat/window mismatch at doc {d} chunk {k}"

    n_tok = flat.shape[0]
    doc_of = np.searchsorted(off, np.arange(n_tok), side="right") - 1
    pos_of = np.arange(n_tok) - off[doc_of]
    dom_of = dom_of_doc[doc_of]
    rows_flat, cpos_flat = _map_rows(doc_of, pos_of, lookup, content,
                                     n_prefix)

    train_rows = doc_split[doc_of] == 0
    test_rows = doc_split[doc_of] == 1
    elig_pre = ((mask == 0) & (is_assist == 1) & (pos_of >= PRE_MIN_POS))

    # DOMAIN-LOCAL terciles (GO condition 1); gpt2 edges asserted vs the
    # committed disposition-(c) artifact (other tags: same construction,
    # edges recorded — token grids differ so numeric equality is
    # gpt2-only by design, disclosed in card § 3).
    committed = json.loads(
        (LABELS / "sycgen_domain_readout.json").read_text())
    doms = sorted(committed["per_domain"])
    bins = np.full(n_tok, -1, dtype=np.int8)
    edges_by_dom = {}
    for di, dom in enumerate(doms):
        dmask = dom_of == di
        b, edges = _terciles(age, train_rows & dmask, elig_pre & dmask)
        bins[dmask] = np.where(b[dmask] >= 0, b[dmask], bins[dmask])
        e = edges["edges"] if isinstance(edges, dict) else edges
        edges_by_dom[dom] = [float(v) for v in e]
        if tag == "gpt2" and "tercile_edges" in committed["per_domain"][dom]:
            want = committed["per_domain"][dom]["tercile_edges"]
            want = want["edges"] if isinstance(want, dict) else want
            assert np.allclose(e, want, rtol=0, atol=1e-9), (dom, e, want)

    elig = (elig_pre & (rows_flat >= 0) & (pos_of >= POS_MIN)
            & (pos_of % content >= OFF_MIN) & np.isfinite(age)
            & (bins >= 0))

    out: dict = {}
    stats = {"n_prefix": n_prefix, "content": content,
             "n_cache_rows": int(ids.shape[0]), "domains": doms,
             "tercile_edges_by_domain": edges_by_dom,
             "gpt2_edges_asserted": bool(tag == "gpt2")}

    # ---- WITHIN-DOMAIN manifests (condition 1): per-domain draw, then
    # concatenate; per-domain caps keep the total at CAP (disclosed).
    n_dom = len(doms)
    for split_name, flag in (("train", 0), ("test", 1)):
        keep_r, keep_y, keep_f = [], [], []
        per_dom_counts = {}
        for di, dom in enumerate(doms):
            m = elig & (doc_split[doc_of] == flag) & (dom_of == di)
            idx_pool = np.flatnonzero(m)
            if len(idx_pool) == 0:
                per_dom_counts[dom] = {}
                continue
            strata = pos_strata(pos_of[idx_pool], min_pos=POS_MIN)
            seed = zlib.crc32(
                f"sycgen/{tag}/{split_name}/{dom}".encode()) % 2 ** 16
            md, mp, mc = stratified_balanced_manifest(
                bins[idx_pool], strata, doc_of[idx_pool], pos_of[idx_pool],
                cap=max(CAP[split_name] // n_dom, MIN_ROWS), seed=seed)
            per_dom_counts[dom] = {int(cls): int((mc == cls).sum())
                                   for cls in (0, 1, 2)}
            if not per_dom_counts[dom] or not min(per_dom_counts[dom].values()):
                continue
            man_flat = off[md] + mp
            rows_all, cpos_all = _map_rows(md, mp, lookup, content,
                                           n_prefix)
            assert (rows_all >= 0).all()
            for cls in (0, 1, 2):
                sel = np.flatnonzero(mc == cls)
                keep_r.append(_stack(rows_all, cpos_all, mp, sel))
                keep_y.append(np.full(len(sel), cls, dtype=np.int64))
                keep_f.append(man_flat[sel])
        ys = np.concatenate(keep_y) if keep_y else np.zeros(0, np.int64)
        per = {int(cls): int((ys == cls).sum()) for cls in (0, 1, 2)}
        stats[f"{FACE}/{split_name}"] = {
            "rows_per_class_total": per,
            "rows_per_class_by_domain": per_dom_counts,
            "ok": bool(per and min(per.values()) >= MIN_ROWS)}
        if keep_y:
            out[(FACE, split_name)] = (np.concatenate(keep_r), ys)
            out[(f"{FACE}_flat", split_name)] = np.concatenate(keep_f)

    # ---- within-CONVERSATION control (BINDING; doc-keyed; domain-pure
    # by construction since each conversation has one domain) ----------
    for split_name, flag in (("train", 0), ("test", 1)):
        m = elig & (doc_split[doc_of] == flag)
        lo_idx, hi_idx, n_docs_used = [], [], 0
        for d in np.unique(doc_of[m]):
            sel = np.flatnonzero(m & (doc_of == d))
            v = age[sel]
            if len(sel) < WD_MIN_DOC_ROWS:
                continue
            q1, q2 = np.quantile(v, [1 / 3, 2 / 3])
            if not (q2 > q1):
                continue
            lo_idx.append(sel[v <= q1])
            hi_idx.append(sel[v >= q2])
            n_docs_used += 1
        if not lo_idx:
            stats[f"{FACE}_wd/{split_name}"] = {"ok": False, "n_docs": 0}
            continue
        lo, hi = np.concatenate(lo_idx), np.concatenate(hi_idx)
        n_take = min(len(lo), len(hi), CAP[split_name] * 2)
        keep_r, keep_y = [], []
        for v_, idx in ((0, lo), (1, hi)):
            rng = _seeded(f"sycgen_wd/{tag}/{split_name}/{v_}")
            if len(idx) > n_take:
                idx = rng.choice(idx, n_take, replace=False)
            keep_r.append(_stack(rows_flat, cpos_flat, pos_of, idx))
            keep_y.append(np.full(len(idx), v_, dtype=np.int64))
        stats[f"{FACE}_wd/{split_name}"] = {
            "ok": bool(n_take >= MIN_ROWS), "n_per_class": int(n_take),
            "n_docs": int(n_docs_used)}
        out[(f"{FACE}_wd", split_name)] = (np.concatenate(keep_r),
                                           np.concatenate(keep_y))

    # ---- condition 3: within-domain vocab band, BESIDE the verdict ---
    vocab = {}
    for di, dom in enumerate(doms):
        dmask = dom_of == di
        tr = train_rows & elig & dmask
        te = test_rows & elig & dmask
        if tr.sum() < MIN_ROWS or te.sum() < MIN_ROWS:
            vocab[dom] = {"skipped": "thin"}
            continue
        uni = nl.type_mean_scores(flat, age, tr)
        docs_d = np.flatnonzero(dom_of_doc == di)
        n_ev = [int(first[off[d]:off[d + 1]].sum()) for d in docs_d]
        n_tk = [int(off[d + 1] - off[d]) for d in docs_d]
        vocab[dom] = {
            "unigram_auc_within": nl.tercile_auc(uni, bins, te),
            "events_per_conv_cv": float(np.std(n_ev) / max(np.mean(n_ev),
                                                           1e-9)),
            "tokens_per_conv_cv": float(np.std(n_tk) / max(np.mean(n_tk),
                                                           1e-9)),
            "n_convs": len(docs_d)}
    stats["within_domain_vocab"] = vocab

    floors = {"first": first, "mask": mask, "off": off, "n_docs": n_docs}
    return out, stats, floors


def _pos_feats(r):
    cp = r[:, 1].astype(np.float32)
    dp = r[:, 2].astype(np.float32)
    f = np.stack([cp, cp ** 2 / 128.0, np.log2(1.0 + dp), dp / 1000.0], 1)
    return torch.from_numpy(f).to(torch.float16)


class _FloorBank:
    def __init__(self, first, mask, off, n_docs):
        self.first, self.mask, self.off, self.n_docs = (first, mask, off,
                                                        n_docs)
        self._cage: dict = {}
        self._cnt: dict = {}

    def cols(self, T: int):
        if T not in self._cage:
            o, nd = self.off, self.n_docs
            self._cage[T] = np.concatenate(
                [w3.sage_floor(self.first[o[d]:o[d + 1]], T)
                 for d in range(nd)])
            self._cnt[T] = np.concatenate(
                [w3.dose_window_count(self.mask[o[d]:o[d + 1]], T)
                 for d in range(nd)])
        return self._cage[T], self._cnt[T]

    def feats(self, flat_pos, T: int):
        cage, cnt = self.cols(T)
        f = np.stack([np.nan_to_num(cage[flat_pos].astype(np.float32)),
                      cnt[flat_pos].astype(np.float32)], 1)
        return torch.from_numpy(f).to(torch.float16)


def screen(key: str):
    RES.mkdir(exist_ok=True)
    out_path = RES / f"screen_sycgen_{key}.json"
    hs = SCREEN_HS[key]
    manifests, mstats, fl = build_rows(key)
    bank = _FloorBank(**fl)
    done = json.loads(out_path.read_text()) if out_path.exists() else {
        "meta": {"substrate": "elicit_sycgen_v1", "model": key,
                 "screen_hs": hs, "card": "sycgen/SCREEN_CARD.md (frozen)",
                 "go_conditions": "dc3cb8fd9 1-5 (within-domain frame; "
                                  "per-token first; vocab beside verdict; "
                                  "hunt4 §4 verbatim; v2 shelved)",
                 "ax_ts": AX_TS, "ord_ts": ORD_TS,
                 "foreign_ord_ts": FOREIGN_ORD_TS, "null_t": NULL_T,
                 "wd_ts": WD_TS, "wd_ord_ts": WD_ORD_TS,
                 "foreign_seed": FOREIGN_SEED,
                 "rows": mstats},
        "cells": {}}
    done["meta"]["rows"] = mstats
    cells = done["cells"]

    def save():
        out_path.write_text(json.dumps(done, indent=1, default=float))

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

    face = FACE
    if not (mstats[f"{face}/train"]["ok"]
            and mstats[f"{face}/test"]["ok"]):
        print(f"[{key} {face}] SKIP (insufficient rows)")
    else:
        rtr, ytr = manifests[(face, "train")]
        rte, yte = manifests[(face, "test")]
        ftr = manifests[(f"{face}_flat", "train")]
        fte = manifests[(f"{face}_flat", "test")]
        ytr_t, yte_t = torch.from_numpy(ytr), torch.from_numpy(yte)

        # condition 2: per-token arms FIRST
        Xtr_tok, Xte_tok = gather_tok(acts, rtr), gather_tok(acts, rte)
        run(f"{face}/tok_linear", lambda: summarize(
            fit_probe(Xtr_tok, ytr_t, Xte_tok, yte_t, 3), 3))
        run(f"{face}/tok_mlp", lambda: summarize(
            fit_probe(Xtr_tok, ytr_t, Xte_tok, yte_t, 3, hidden=512), 3))
        run(f"{face}/position_floor", lambda: summarize(
            fit_probe(_pos_feats(rtr), ytr_t, _pos_feats(rte), yte_t, 3),
            3))

        for T in AX_TS:
            run(f"{face}/T{T}/visible_evidence_floor", lambda: summarize(
                fit_probe(bank.feats(ftr, T), ytr_t,
                          bank.feats(fte, T), yte_t, 3), 3))
            Wtr, Wte = gather_win(acts, rtr, T), gather_win(acts, rte, T)
            atr, ate = anchor_ctxmean(Wtr), anchor_ctxmean(Wte)
            run(f"{face}/T{T}/actxmean_linear", lambda: summarize(
                fit_probe(atr, ytr_t, ate, yte_t, 3), 3))
            run(f"{face}/T{T}/actxmean_mlp", lambda: summarize(
                fit_probe(atr, ytr_t, ate, yte_t, 3, hidden=512), 3))
            fatr = actxmean_foreign(
                Wtr, np.random.default_rng(FOREIGN_SEED + T))
            fate = actxmean_foreign(
                Wte, np.random.default_rng(FOREIGN_SEED + T + 1))
            run(f"{face}/T{T}/actxmean_foreign_linear", lambda: summarize(
                fit_probe(fatr, ytr_t, fate, yte_t, 3), 3))
            run(f"{face}/T{T}/actxmean_foreign_mlp", lambda: summarize(
                fit_probe(fatr, ytr_t, fate, yte_t, 3, hidden=512), 3))
            del atr, ate, fatr, fate
            if T in ORD_TS:
                flat_tr = Wtr.reshape(len(rtr), -1)
                flat_te = Wte.reshape(len(rte), -1)
                run(f"{face}/T{T}/win_linear", lambda: summarize(
                    fit_probe(flat_tr, ytr_t, flat_te, yte_t, 3), 3))
                srng = np.random.default_rng(
                    SHUF_SEED
                    + zlib.crc32(f"{face}/T{T}".encode()) % 2 ** 16)
                Str = shuffle_context(Wtr, srng).reshape(len(rtr), -1)
                Ste = shuffle_context(Wte, srng).reshape(len(rte), -1)
                run(f"{face}/T{T}/win_shuf_linear", lambda: summarize(
                    fit_probe(Str, ytr_t, Ste, yte_t, 3), 3))
                if T in FOREIGN_ORD_TS:
                    fwtr = foreign_context(
                        Wtr, np.random.default_rng(FOREIGN_SEED + T)
                    ).reshape(len(rtr), -1)
                    fwte = foreign_context(
                        Wte, np.random.default_rng(FOREIGN_SEED + T + 1)
                    ).reshape(len(rte), -1)
                    run(f"{face}/T{T}/win_foreign_linear",
                        lambda: summarize(
                            fit_probe(fwtr, ytr_t, fwte, yte_t, 3), 3))
                    if T in ORD_MLP_T:
                        run(f"{face}/T{T}/win_mlp", lambda: summarize(
                            fit_probe(flat_tr, ytr_t, flat_te, yte_t, 3,
                                      hidden=512), 3))
                        run(f"{face}/T{T}/win_shuf_mlp", lambda: summarize(
                            fit_probe(Str, ytr_t, Ste, yte_t, 3,
                                      hidden=512), 3))
                        run(f"{face}/T{T}/win_foreign_mlp",
                            lambda: summarize(
                                fit_probe(fwtr, ytr_t, fwte, yte_t, 3,
                                          hidden=512), 3))
                    del fwtr, fwte
                if T == NULL_T:
                    nrng = np.random.default_rng(NULL_SEED)
                    yn = torch.from_numpy(nrng.permutation(ytr))
                    run(f"{face}/T{T}/null_win_linear", lambda: summarize(
                        fit_probe(flat_tr, yn, flat_te, yte_t, 3), 3))
                    run(f"{face}/null_tok_linear", lambda: summarize(
                        fit_probe(Xtr_tok, yn, Xte_tok, yte_t, 3), 3))
                del flat_tr, flat_te, Str, Ste
            del Wtr, Wte
        del Xtr_tok, Xte_tok

    wd = f"{face}_wd"
    if not (mstats.get(f"{wd}/train", {}).get("ok")
            and mstats.get(f"{wd}/test", {}).get("ok")):
        print(f"[{key} {wd}] SKIP (insufficient within-conv rows — a "
              f"SKIP here blocks any KEEP, card)")
    else:
        rtr, ytr = manifests[(wd, "train")]
        rte, yte = manifests[(wd, "test")]
        ytr_t, yte_t = torch.from_numpy(ytr), torch.from_numpy(yte)
        Xtr, Xte = gather_tok(acts, rtr), gather_tok(acts, rte)
        run(f"{wd}/tok_linear", lambda: summarize(
            fit_probe(Xtr, ytr_t, Xte, yte_t, 2, class_weight=True), 2))
        run(f"{wd}/tok_mlp", lambda: summarize(
            fit_probe(Xtr, ytr_t, Xte, yte_t, 2, hidden=512,
                      class_weight=True), 2))
        for T in sorted(set(WD_TS) | set(WD_ORD_TS)):
            Wtr, Wte = gather_win(acts, rtr, T), gather_win(acts, rte, T)
            if T in WD_TS:
                atr, ate = anchor_ctxmean(Wtr), anchor_ctxmean(Wte)
                run(f"{wd}/T{T}/actxmean_linear", lambda: summarize(
                    fit_probe(atr, ytr_t, ate, yte_t, 2,
                              class_weight=True), 2))
                run(f"{wd}/T{T}/actxmean_mlp", lambda: summarize(
                    fit_probe(atr, ytr_t, ate, yte_t, 2, hidden=512,
                              class_weight=True), 2))
                fatr = actxmean_foreign(
                    Wtr, np.random.default_rng(FOREIGN_SEED + T))
                fate = actxmean_foreign(
                    Wte, np.random.default_rng(FOREIGN_SEED + T + 1))
                run(f"{wd}/T{T}/actxmean_foreign_linear", lambda: summarize(
                    fit_probe(fatr, ytr_t, fate, yte_t, 2,
                              class_weight=True), 2))
                del atr, ate, fatr, fate
            if T in WD_ORD_TS:
                flat_tr = Wtr.reshape(len(rtr), -1)
                flat_te = Wte.reshape(len(rte), -1)
                run(f"{wd}/T{T}/win_linear", lambda: summarize(
                    fit_probe(flat_tr, ytr_t, flat_te, yte_t, 2,
                              class_weight=True), 2))
                srng = np.random.default_rng(
                    SHUF_SEED + zlib.crc32(f"{wd}/T{T}".encode()) % 2 ** 16)
                Str = shuffle_context(Wtr, srng).reshape(len(rtr), -1)
                Ste = shuffle_context(Wte, srng).reshape(len(rte), -1)
                run(f"{wd}/T{T}/win_shuf_linear", lambda: summarize(
                    fit_probe(Str, ytr_t, Ste, yte_t, 2,
                              class_weight=True), 2))
                if T in WD_FOREIGN_TS:
                    fwtr = foreign_context(
                        Wtr, np.random.default_rng(FOREIGN_SEED + T)
                    ).reshape(len(rtr), -1)
                    fwte = foreign_context(
                        Wte, np.random.default_rng(FOREIGN_SEED + T + 1)
                    ).reshape(len(rte), -1)
                    run(f"{wd}/T{T}/win_foreign_linear", lambda: summarize(
                        fit_probe(fwtr, ytr_t, fwte, yte_t, 2,
                                  class_weight=True), 2))
                    del fwtr, fwte
                del flat_tr, flat_te, Str, Ste
            del Wtr, Wte
        del Xtr, Xte
    del acts
    save()
    print(f"[{key}] DONE -> {out_path}", flush=True)


def main():
    for k in (sys.argv[1:] or ["gpt2", "gemma2_2b", "llama31_8b"]):
        screen(k)


if __name__ == "__main__":
    main()
