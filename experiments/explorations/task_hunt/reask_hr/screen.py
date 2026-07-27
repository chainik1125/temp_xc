"""REASK_HR screen — hard-refusal-gated re-ask age on the refmark2k
substrate (executes REASK_HR_SCREEN_CARD.md; wave-3 directive
ae1ce5fb0, gate census pre-registration, release c6e464881).

hunt4w2/screen.py transplant, ONE face. Labels from the committed
`wave3_reask_hr_<tok>.npz` (sage_face kernel, support 64), stream
from `refmark2k_wildchat_<tok>.npz` — both reused verbatim.
Eligibility = assistant tokens, event+boundary masked, pos >= 64,
in-chunk offset >= 63. Manifests are position-matched IN-SCREEN
(card § 2): all row filters first, then frozen
`punctint_lib.pos_strata` + `stratified_balanced_manifest` per
split; tercile edges recomputed by the premeasure's `_terciles` on
the premeasure's own eligibility and ASSERTED equal to the
committed `reask_hr_premeasure.json` edges. Floor features per T =
[sage_floor, dose_window_count] (frozen wave3_lib functions of
committed arrays, computed in-screen — card § 2). Probe grid = the
hunt4 grid verbatim; within-CONVERSATION (doc-keyed) arms BINDING.

Run: .venv/bin/python -m experiments.explorations.task_hunt.reask_hr.screen [model ...]
Writes results/screen_wildchat_<model>.json (resumable).
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
from experiments.explorations.task_hunt.reask_hr.cache_acts import (
    CACHE_ROOT,
    TOK_TAG,
)

HERE = Path(__file__).resolve().parent
LABELS = HERE.parent / "labels"
RES = HERE / "results"

FACE = "reask_hr"
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


def _rank_auc(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b) by ranks (deterministic; ties get midranks)."""
    x = np.concatenate([a, b]).astype(np.float64)
    r = np.empty(len(x))
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    ra = r[:len(a)].sum()
    return float((ra - len(a) * (len(a) + 1) / 2.0) / (len(a) * len(b)))


def build_rows(key: str):
    tag = TOK_TAG[key]
    z = np.load(LABELS / f"refmark2k_wildchat_{tag}.npz")
    zl = np.load(LABELS / f"wave3_reask_hr_{tag}.npz")
    c = np.load(CACHE_ROOT / key / "tokens.npz")
    ids, doc_idx, n_prefix = c["ids"], c["doc_idx"], int(c["n_prefix"])
    content = ids.shape[1] - n_prefix
    flat, off = z["token_ids"], z["doc_off"]
    is_assist, boundary = z["is_assistant"], z["is_boundary"]
    doc_split = z["doc_split"]
    age = zl["reask_hr_age"].astype(np.float64)
    first, mask = zl["reask_hr_event_first"], zl["reask_hr_event_mask"]

    lookup = _row_lookup(doc_idx)
    for (d, k), i in list(lookup.items())[:200]:
        s = off[d] + k * content
        assert np.array_equal(flat[s:s + content], ids[i, n_prefix:]), \
            f"flat/window mismatch at doc {d} chunk {k}"

    n_tok = flat.shape[0]
    n_docs = len(off) - 1
    doc_of = np.searchsorted(off, np.arange(n_tok), side="right") - 1
    pos_of = np.arange(n_tok) - off[doc_of]
    rows_flat, cpos_flat = _map_rows(doc_of, pos_of, lookup, content,
                                     n_prefix)

    # tercile bins: the premeasure's own function on the premeasure's
    # own eligibility; edges asserted vs the committed JSON (receipt).
    train_rows = doc_split[doc_of] == 0
    elig_pre = ((mask == 0) & (boundary == 0) & (is_assist == 1)
                & (pos_of >= PRE_MIN_POS))
    bins, edges = _terciles(age, train_rows, elig_pre)
    committed = json.loads(
        (LABELS / "reask_hr_premeasure.json").read_text())
    want = committed["per_tokenizer"][tag]["tercile_edges"]["edges"]
    got = edges["edges"] if isinstance(edges, dict) else edges
    assert np.allclose(got, want, rtol=0, atol=1e-9), (got, want)

    # screen eligibility (card § 2): stricter than the premeasure.
    elig = (elig_pre & (rows_flat >= 0) & (pos_of >= POS_MIN)
            & (pos_of % content >= OFF_MIN) & np.isfinite(age)
            & (bins >= 0))

    out: dict = {}
    stats = {"n_prefix": n_prefix, "content": content,
             "n_cache_rows": int(ids.shape[0]),
             "tercile_edges_asserted": [float(v) for v in want]}

    for split_name, flag in (("train", 0), ("test", 1)):
        m = elig & (doc_split[doc_of] == flag)
        idx_pool = np.flatnonzero(m)
        strata = pos_strata(pos_of[idx_pool], min_pos=POS_MIN)
        seed = zlib.crc32(
            f"reask_hr/{tag}/{split_name}".encode()) % 2 ** 16
        md, mp, mc = stratified_balanced_manifest(
            bins[idx_pool], strata, doc_of[idx_pool], pos_of[idx_pool],
            cap=CAP[split_name], seed=seed)
        # manifest rows are (doc, pos, cls); flat index = off[doc]+pos
        man_flat = off[md] + mp
        rows_all, cpos_all = _map_rows(md, mp, lookup, content, n_prefix)
        assert (rows_all >= 0).all()
        per = {int(cls): int((mc == cls).sum()) for cls in (0, 1, 2)}
        pos_auc = (_rank_auc(mp[mc == 2], mp[mc == 0])
                   if min(per.values()) else 0.5)
        stats[f"{FACE}/{split_name}"] = {
            "rows_per_class": per,
            "manifest_position_auc_hi_vs_lo": round(pos_auc, 4),
            "ok": bool(min(per.values()) >= MIN_ROWS)}
        keep_r, keep_y, keep_f = [], [], []
        for cls in (0, 1, 2):
            sel = np.flatnonzero(mc == cls)
            keep_r.append(_stack(rows_all, cpos_all, mp, sel))
            keep_y.append(np.full(len(sel), cls, dtype=np.int64))
            keep_f.append(man_flat[sel])
        out[(FACE, split_name)] = (np.concatenate(keep_r),
                                   np.concatenate(keep_y))
        out[(f"{FACE}_flat", split_name)] = np.concatenate(keep_f)

    # ---- within-CONVERSATION control (BINDING; wd machinery verbatim,
    # doc = conversation) --------------------------------------------
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
            rng = _seeded(f"reask_hr_wd/{tag}/{split_name}/{v_}")
            if len(idx) > n_take:
                idx = rng.choice(idx, n_take, replace=False)
            keep_r.append(_stack(rows_flat, cpos_flat, pos_of, idx))
            keep_y.append(np.full(len(idx), v_, dtype=np.int64))
        stats[f"{FACE}_wd/{split_name}"] = {
            "ok": bool(n_take >= MIN_ROWS), "n_per_class": int(n_take),
            "n_docs": int(n_docs_used)}
        out[(f"{FACE}_wd", split_name)] = (np.concatenate(keep_r),
                                           np.concatenate(keep_y))

    floors = {"first": first, "mask": mask, "off": off, "n_docs": n_docs}
    return out, stats, floors


def _pos_feats(r):
    cp = r[:, 1].astype(np.float32)
    dp = r[:, 2].astype(np.float32)
    f = np.stack([cp, cp ** 2 / 128.0, np.log2(1.0 + dp), dp / 1000.0], 1)
    return torch.from_numpy(f).to(torch.float16)


class _FloorBank:
    """Per-T visible-evidence floor columns from the COMMITTED event
    arrays via the FROZEN wave3_lib functions (card § 2): censored age
    (sage_floor) + in-window event-token count (dose_window_count).
    Deterministic; computed once per T over the full stream."""

    def __init__(self, first, mask, off, n_docs):
        self.first, self.mask, self.off, self.n_docs = first, mask, off, n_docs
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
    out_path = RES / f"screen_wildchat_{key}.json"
    hs = SCREEN_HS[key]
    manifests, mstats, fl = build_rows(key)
    bank = _FloorBank(**fl)
    done = json.loads(out_path.read_text()) if out_path.exists() else {
        "meta": {"substrate": "refmark2k_wildchat", "model": key,
                 "screen_hs": hs,
                 "card": "REASK_HR_SCREEN_CARD.md (frozen)",
                 "ax_ts": AX_TS, "ord_ts": ORD_TS,
                 "foreign_ord_ts": FOREIGN_ORD_TS, "null_t": NULL_T,
                 "wd_ts": WD_TS, "wd_ord_ts": WD_ORD_TS,
                 "foreign_seed": FOREIGN_SEED,
                 "rows": mstats,
                 "reach": "reask_hr events cite a hard-refused ask "
                          ">= 2 messages (~240-270 tok) back — beyond "
                          "the whole ladder (card § 0 out-of-window-"
                          "by-construction); the age face is trailing "
                          "state; censored-age + in-window-count "
                          "floors are the instrument"},
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
