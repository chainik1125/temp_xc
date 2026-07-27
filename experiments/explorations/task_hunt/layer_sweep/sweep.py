"""Layer-sweep screens for the order-carried dialogue faces
(059a66239 P2 sweep (a); CARD.md — frozen with this runner + scorer
in one commit before any cell).

The instrument is the parent screens VERBATIM, parameterised by
(model, hs): `cnov` runs hunt3's arm grid, `tt` (ttrend) runs
diafaces' — manifests, floor features, probe grid, seeds and caps are
IMPORTED from those modules, never re-declared, so a number produced
here differs from the committed screen JSONs only through `hs`.

Two deviations from the parents, both required for a layer sweep and
both disclosed in the card:

1. `hs` is a parameter; results go to
   `results/screen_{key}_hs{hs}.json` and every cell key is prefixed
   `hs{hs}/` — layer in BOTH filename and cell key, because the
   parents' resume contract (skip any present cell key) would
   silently resume-clobber their committed single-layer results
   files if reused as-is.
2. Activations resolve through `extract.acts_path` (canonical
   dialevel root for `HS_CAPTURE` layers, the sweep root for the
   extra layers).

Faces: cnov (hunt3 stack) + tt (diafaces stack) only — the directive
names ttrend + cnov; nvtrend and dq are NOT run.

Run: .venv/bin/python -m experiments.explorations.task_hunt.layer_sweep.sweep <model> <hs> [<hs> ...]
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
from experiments.explorations.task_hunt.novelty.screen import SHUF_SEED
from experiments.explorations.task_hunt.replag.screen import (
    gather_tok,
    gather_win,
    shuffle_context,
    summarize,
)
from experiments.explorations.task_hunt.diafaces import screen as dia_screen
from experiments.explorations.task_hunt.hunt3 import screen as h3_screen
from experiments.explorations.task_hunt.layer_sweep.extract import (
    CAPTURE,
    acts_path,
)

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

# stack -> (module, face key, label of the face in that stack)
STACKS = {
    "cnov": (h3_screen, "cnov"),
    "tt": (dia_screen, "tt"),
}


def screen_at(key: str, hs: int):
    assert hs in CAPTURE[key], f"hs{hs} not in the frozen capture set"
    RES.mkdir(exist_ok=True)
    out_path = RES / f"screen_{key}_hs{hs}.json"
    done = json.loads(out_path.read_text()) if out_path.exists() else {
        "meta": {"model": key, "hs": hs,
                 "semantics": "hs = output_hidden_states index = "
                              "resid_post L+1 (replag convention)",
                 "card": "layer_sweep/CARD.md (frozen)",
                 "faces": list(STACKS),
                 "instrument": "parent screens verbatim by import "
                               "(hunt3.screen for cnov, diafaces.screen "
                               "for tt); hs-parameterised"},
        "cells": {}}
    cells = done["cells"]

    def save():
        out_path.write_text(json.dumps(done, indent=1))

    def run(cell_key, fn):
        cell_key = f"hs{hs}/{cell_key}"
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
        np.load(acts_path(key, hs), mmap_mode="r")))

    for stack_name, (mod, face) in STACKS.items():
        manifests, mstats, zd, zf = mod.build_rows(key)
        tbl = dia_screen._TurnTable(zd) if stack_name == "tt" else None
        if stack_name == "tt":
            def vis(fp, T):
                return dia_screen._tt_visible_feats(tbl, fp, T)
        else:
            def vis(fp, T):
                return h3_screen._floor_feats(zf, fp, T, face)
        AX_TS, ORD_TS = mod.AX_TS, mod.ORD_TS
        ORD_MLP_T, NULL_T = mod.ORD_MLP_T, mod.NULL_T
        WD_TS, WD_ORD_TS = mod.WD_TS, mod.WD_ORD_TS
        NULL_SEED = mod.NULL_SEED

        if not (mstats[f"{face}/train"]["ok"]
                and mstats[f"{face}/test"]["ok"]):
            print(f"[{key} hs{hs} {face}] SKIP (insufficient rows)")
            continue
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
            fit_probe(mod._pos_feats(rtr), ytr_t,
                      mod._pos_feats(rte), yte_t, 3), 3))

        for T in AX_TS:
            run(f"{face}/T{T}/visible_evidence_floor", lambda: summarize(
                fit_probe(vis(ftr, T), ytr_t, vis(fte, T), yte_t, 3), 3))
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
                fwtr = foreign_context(
                    Wtr, np.random.default_rng(FOREIGN_SEED + T)
                ).reshape(len(rtr), -1)
                fwte = foreign_context(
                    Wte, np.random.default_rng(FOREIGN_SEED + T + 1)
                ).reshape(len(rte), -1)
                run(f"{face}/T{T}/win_foreign_linear", lambda: summarize(
                    fit_probe(fwtr, ytr_t, fwte, yte_t, 3), 3))
                if T in ORD_MLP_T:
                    run(f"{face}/T{T}/win_mlp", lambda: summarize(
                        fit_probe(flat_tr, ytr_t, flat_te, yte_t, 3,
                                  hidden=512), 3))
                    run(f"{face}/T{T}/win_shuf_mlp", lambda: summarize(
                        fit_probe(Str, ytr_t, Ste, yte_t, 3,
                                  hidden=512), 3))
                    run(f"{face}/T{T}/win_foreign_mlp", lambda: summarize(
                        fit_probe(fwtr, ytr_t, fwte, yte_t, 3,
                                  hidden=512), 3))
                if T == NULL_T:
                    nrng = np.random.default_rng(NULL_SEED)
                    yn = torch.from_numpy(nrng.permutation(ytr))
                    run(f"{face}/T{T}/null_win_linear", lambda: summarize(
                        fit_probe(flat_tr, yn, flat_te, yte_t, 3), 3))
                    run(f"{face}/null_tok_linear", lambda: summarize(
                        fit_probe(Xtr_tok, yn, Xte_tok, yte_t, 3), 3))
                del flat_tr, flat_te, Str, Ste, fwtr, fwte
            del Wtr, Wte
        del Xtr_tok, Xte_tok

        wd = f"{face}_wd"
        if not (mstats.get(f"{wd}/train", {}).get("ok")
                and mstats.get(f"{wd}/test", {}).get("ok")):
            print(f"[{key} hs{hs} {wd}] SKIP (insufficient within-dialogue "
                  f"rows — a SKIP here blocks any KEEP, ops rule 7)")
            continue
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
                fwtr = foreign_context(
                    Wtr, np.random.default_rng(FOREIGN_SEED + T)
                ).reshape(len(rtr), -1)
                fwte = foreign_context(
                    Wte, np.random.default_rng(FOREIGN_SEED + T + 1)
                ).reshape(len(rte), -1)
                run(f"{wd}/T{T}/win_foreign_linear", lambda: summarize(
                    fit_probe(fwtr, ytr_t, fwte, yte_t, 2,
                              class_weight=True), 2))
                del flat_tr, flat_te, Str, Ste, fwtr, fwte
            del Wtr, Wte
        del Xtr, Xte
    del acts
    save()
    print(f"[{key} hs{hs}] DONE -> {out_path}", flush=True)


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: sweep.py <model> <hs> [<hs> ...]")
    key = sys.argv[1]
    for hs in [int(a) for a in sys.argv[2:]]:
        screen_at(key, hs)


if __name__ == "__main__":
    main()
