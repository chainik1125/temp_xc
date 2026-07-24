"""Stage-1 screen — punctuation-intensity faces on fineweb (executes CARD.md).

BOTH faces of `../labels/punctint_fineweb_<tok>.npz` in one pass, on the
EXISTING replag fineweb caches (zero new forward passes):

  q     — question-rate intensity (`lam_q` terciles `q_bin`);
          queue position 5, ships clean.
  list  — list-density intensity (`lam_list` terciles `list_bin`);
          queue position 7, ships CONDITIONALLY (mac-local binding
          qualification 1: the position-only floor probe on the SHIPPED
          manifest rows is mandatory, and a gap without it is
          uninterpretable). This screen always runs that probe.

Each face also screens its **ambient anchor** (`is_q` / `is_list`,
binary) per-token and at T = 16 window-MEAN: candidate 2's lesson was
that a window gap growing with T can be generic to window width, so the
anchor-differenced contrast is what the card's KEEP rule reads.

Row mapping, eligibility, caps, probe grid and seeds are IDENTICAL to
`../novelty/screen.py` so the two fineweb bundles are directly
comparable. Probe stack: frozen `conversion_depth.problib`.

Run: .venv/bin/python -m experiments.explorations.task_hunt.qrate_fineweb.screen [model ...]
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
from experiments.explorations.task_hunt.novelty.screen import (
    CAP,
    MATCH_SEED,
    MIN_ROWS,
    MLP_T,
    MODELS,
    NULL_SEED,
    NULL_T,
    OFF_MIN,
    POS_MIN,
    SHUF_SEED,
    T_GRID,
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
    win_mean,
)

HERE = Path(__file__).resolve().parent
LABELS = HERE.parent / "labels"
CACHE_ROOT = Path("/workspace/replag_caches")
RES = HERE / "results"

MEAN_TS = [4, 8, 16, 32, 64]
ANCHOR_T = 16
FACES = (("q", "q_bin", "is_q", "man_q"),
         ("list", "list_bin", "is_list", "man_list"))


def build_rows(key: str):
    tag = MODELS[key]
    z = np.load(LABELS / f"punctint_fineweb_{tag}.npz")
    c = np.load(CACHE_ROOT / key / "tokens.npz")
    ids, doc_idx, n_prefix = c["ids"], c["doc_idx"], int(c["n_prefix"])
    content = ids.shape[1] - n_prefix
    flat, off_doc, doc_split = z["token_ids"], z["doc_off"], z["doc_split"]
    lookup = _row_lookup(doc_idx)
    for (d, k), i in list(lookup.items())[:200]:
        s = off_doc[d] + k * content
        assert np.array_equal(flat[s:s + content], ids[i, n_prefix:]), \
            f"flat/window mismatch at doc {d} chunk {k}"

    out: dict = {}
    stats = {"n_prefix": n_prefix, "content": content}
    for face, binkey, anchorkey, man in FACES:
        md, mp, mc = z[f"{man}_doc"], z[f"{man}_pos"], z[f"{man}_cls"]
        rows_all, cpos_all = _map_rows(md, mp, lookup, content, n_prefix)
        elig = (rows_all >= 0) & (mp >= POS_MIN) & (mp % content >= OFF_MIN)
        for split_name, flag in (("train", 0), ("test", 1)):
            m = elig & (doc_split[md] == flag)
            per, keep_r, keep_y = {}, [], []
            for cls in (0, 1, 2):
                idx = np.flatnonzero(m & (mc == cls))
                rng = _seeded(f"punct/{face}/{key}/{split_name}/{cls}")
                if len(idx) > CAP[split_name]:
                    idx = rng.choice(idx, CAP[split_name], replace=False)
                per[cls] = len(idx)
                keep_r.append(_stack(rows_all, cpos_all, mp, idx))
                keep_y.append(np.full(len(idx), cls, dtype=np.int64))
            stats[f"{face}/{split_name}"] = {
                "rows_per_class": per,
                "ok": bool(min(per.values()) >= MIN_ROWS)}
            out[(face, split_name)] = (np.concatenate(keep_r),
                                       np.concatenate(keep_y))

    # ---- ambient anchors ----
    # The anchor CANNOT come from a face manifest: the builder masks
    # event-sentence tokens out of the face it anchors (is_q tokens are
    # absent from man_q by construction, so that pool is single-class).
    # Draw anchor rows from the FULL eligible token pool instead, under
    # the same eligibility/caps/seeding, balanced on the anchor bit.
    n_tok = flat.shape[0]
    doc_of = np.searchsorted(off_doc, np.arange(n_tok), side="right") - 1
    pos_of = np.arange(n_tok) - off_doc[doc_of]
    a_rows, a_cpos = _map_rows(doc_of, pos_of, lookup, content, n_prefix)
    a_elig = (a_rows >= 0) & (pos_of >= POS_MIN) & (pos_of % content >= OFF_MIN)
    for face, _bk, anchorkey, _man in FACES:
        av = z[anchorkey].astype(np.int64)
        aname = f"{face}_anchor"
        for split_name, flag in (("train", 0), ("test", 1)):
            m = a_elig & (doc_split[doc_of] == flag)
            n_take = min(int((m & (av == v)).sum()) for v in (0, 1))
            n_take = min(n_take, CAP[split_name])
            per_a, ar, ay = {}, [], []
            for v in (0, 1):
                idx = np.flatnonzero(m & (av == v))
                rng = _seeded(f"punctanchor/{face}/{key}/{split_name}/{v}")
                if len(idx) > n_take:
                    idx = rng.choice(idx, n_take, replace=False)
                per_a[v] = len(idx)
                ar.append(_stack(a_rows, a_cpos, pos_of, idx))
                ay.append(np.full(len(idx), v, dtype=np.int64))
            stats[f"{aname}/{split_name}"] = {
                "rows_per_class": per_a,
                "ok": bool(min(per_a.values()) >= MIN_ROWS)}
            out[(aname, split_name)] = (np.concatenate(ar),
                                        np.concatenate(ay))
    return out, stats


def _pos_feats(r):
    cp = r[:, 1].astype(np.float32)
    dp = r[:, 2].astype(np.float32)
    f = np.stack([cp, cp ** 2 / 128.0, np.log2(1.0 + dp), dp / 1000.0], 1)
    return torch.from_numpy(f).to(torch.float16)


def screen(key: str):
    RES.mkdir(exist_ok=True)
    out_path = RES / f"screen_{key}.json"
    hs = SCREEN_HS[key]
    manifests, mstats = build_rows(key)
    done = json.loads(out_path.read_text()) if out_path.exists() else {
        "meta": {"model": key, "screen_hs": hs, "card": "CARD.md (frozen)",
                 "t_grid": T_GRID, "mean_ts": MEAN_TS, "mlp_t": MLP_T,
                 "null_t": NULL_T, "anchor_t": ANCHOR_T, "rows": mstats,
                 "tokens_per_sentence": 21.4,
                 "kernel_mass_by_T": {"4": 0.06, "8": 0.12, "16": 0.23,
                                      "32": 0.42, "64": 0.72}},
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

    for face, _bk, _ak, _man in FACES:
        # ---- ambient anchor (binary) : the differencing baseline ----
        aname = f"{face}_anchor"
        if mstats[f"{aname}/train"]["ok"] and mstats[f"{aname}/test"]["ok"]:
            rtr, ytr = manifests[(aname, "train")]
            rte, yte = manifests[(aname, "test")]
            ytr_t, yte_t = torch.from_numpy(ytr), torch.from_numpy(yte)
            Xtr, Xte = gather_tok(acts, rtr), gather_tok(acts, rte)
            run(f"{aname}/tok_linear", lambda: summarize(
                fit_probe(Xtr, ytr_t, Xte, yte_t, 2, class_weight=True), 2))
            W1, W2 = gather_win(acts, rtr, ANCHOR_T), gather_win(acts, rte,
                                                                ANCHOR_T)
            run(f"{aname}/T{ANCHOR_T}/win_mean_linear", lambda: summarize(
                fit_probe(win_mean(W1), ytr_t, win_mean(W2), yte_t, 2,
                          class_weight=True), 2))
            del W1, W2, Xtr, Xte
        else:
            print(f"[{key} {aname}] SKIP (insufficient rows)")

        # ---- the intensity face (3-class) ----
        if not (mstats[f"{face}/train"]["ok"] and mstats[f"{face}/test"]["ok"]):
            print(f"[{key} {face}] SKIP (insufficient rows)")
            continue
        rtr, ytr = manifests[(face, "train")]
        rte, yte = manifests[(face, "test")]
        ytr_t, yte_t = torch.from_numpy(ytr), torch.from_numpy(yte)

        Xtr_tok, Xte_tok = gather_tok(acts, rtr), gather_tok(acts, rte)
        run(f"{face}/tok_linear", lambda: summarize(
            fit_probe(Xtr_tok, ytr_t, Xte_tok, yte_t, 3), 3))
        run(f"{face}/tok_mlp", lambda: summarize(
            fit_probe(Xtr_tok, ytr_t, Xte_tok, yte_t, 3, hidden=512), 3))
        # MANDATORY for the list face (review qualification 1), run for both
        run(f"{face}/position_floor", lambda: summarize(
            fit_probe(_pos_feats(rtr), ytr_t, _pos_feats(rte), yte_t, 3), 3))

        for T in MEAN_TS:
            Wtr, Wte = gather_win(acts, rtr, T), gather_win(acts, rte, T)
            run(f"{face}/T{T}/win_mean_linear", lambda: summarize(
                fit_probe(win_mean(Wtr), ytr_t, win_mean(Wte), yte_t, 3), 3))
            if T in T_GRID:
                flat_tr = Wtr.reshape(len(rtr), -1)
                flat_te = Wte.reshape(len(rte), -1)
                run(f"{face}/T{T}/win_linear", lambda: summarize(
                    fit_probe(flat_tr, ytr_t, flat_te, yte_t, 3), 3))
                srng = np.random.default_rng(
                    SHUF_SEED + zlib.crc32(f"{face}/T{T}".encode()) % 2 ** 16)
                Str = shuffle_context(Wtr, srng).reshape(len(rtr), -1)
                Ste = shuffle_context(Wte, srng).reshape(len(rte), -1)
                run(f"{face}/T{T}/win_shuf_linear", lambda: summarize(
                    fit_probe(Str, ytr_t, Ste, yte_t, 3), 3))
                if T in MLP_T:
                    run(f"{face}/T{T}/win_mlp", lambda: summarize(
                        fit_probe(flat_tr, ytr_t, flat_te, yte_t, 3,
                                  hidden=512), 3))
                    run(f"{face}/T{T}/win_shuf_mlp", lambda: summarize(
                        fit_probe(Str, ytr_t, Ste, yte_t, 3, hidden=512), 3))
                if T == NULL_T:
                    nrng = np.random.default_rng(NULL_SEED)
                    yn = torch.from_numpy(nrng.permutation(ytr))
                    run(f"{face}/T{T}/null_win_linear", lambda: summarize(
                        fit_probe(flat_tr, yn, flat_te, yte_t, 3), 3))
                    run(f"{face}/null_tok_linear", lambda: summarize(
                        fit_probe(Xtr_tok, yn, Xte_tok, yte_t, 3), 3))
                del flat_tr, flat_te, Str, Ste
            del Wtr, Wte
    save()
    print(f"[{key}] DONE -> {out_path}")


def main():
    for k in (sys.argv[1:] or list(MODELS)):
        screen(k)


if __name__ == "__main__":
    main()
