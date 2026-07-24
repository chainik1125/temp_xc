"""Stage-1 screen — vocabulary-novelty trailing rate (executes CARD.md).

Reads the factory's flat-stream labels
(`../labels/novelty_fineweb_{gpt2,gemma2,llama31}.npz`) against the
EXISTING replag fineweb activation caches on this volume — zero new
forward passes; the flat↔windowed mapping is verified in the card § 2
and re-asserted at run time here.

Per model, on the replag screen layer (gpt2 hs7, gemma2_2b hs14,
llama31_8b hs14), with uniform eligibility so every T reads IDENTICAL
rows (card § 3):

  per-token linear + MLP(512) on nov_bin        (triage first)
  T ∈ {4,8,16,32}: window linear, window-MEAN linear, context-shuffled
      linear (anchor slot fixed, seeded)
  window-MEAN additionally at T = 64            (full kernel support)
  window + shuffled-window MLP at T ∈ {16,32}
  permutation nulls (NULL_SEED 99) on the linear pair at T = 16
  position-only floor probe on the shipped rows
  receipt face nov_null_bin: tok + MEAN at T ∈ {16,32,64}

Probe stack: `conversion_depth.problib` (frozen — no retuning).
Metric: acc_test (3-class, chance 1/3) + per_class.
Incremental/resumable per cell; writes results/screen_<model>.json.

Run: .venv/bin/python -m experiments.explorations.task_hunt.novelty.screen [model ...]
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

# model key -> tokenizer tag used in the factory's label filenames
MODELS = {"gpt2": "gpt2", "gemma2_2b": "gemma2", "llama31_8b": "llama31"}

T_GRID = [4, 8, 16, 32]           # flatten / shuffle arms
MEAN_TS = [4, 8, 16, 32, 64]      # MEAN arm reaches full kernel support
MLP_T = [16, 32]
NULL_T = 16
NULL_SEED = 99
SHUF_SEED = 1234
MATCH_SEED = 1013
POS_MIN = 64                      # doc position (builder triage convention)
OFF_MIN = 63                      # in-chunk offset: T <= 64 fits one row
CAP = {"train": 4000, "test": 1500}
MIN_ROWS = 300
RECEIPT_TS = [16, 32, 64]


def _seeded(tag: str) -> np.random.Generator:
    return np.random.default_rng(MATCH_SEED + zlib.crc32(tag.encode()) % 2 ** 16)


def _row_lookup(doc_idx: np.ndarray) -> dict:
    """(doc, chunk_index_within_doc) -> cache row."""
    seen: dict = {}
    out: dict = {}
    for i, d in enumerate(doc_idx.tolist()):
        k = seen.get(d, 0)
        seen[d] = k + 1
        out[(d, k)] = i
    return out


def _map_rows(docs, poss, lookup, content, n_prefix):
    """Flat (doc, pos) -> (cache_row, cache_pos); -1 where the chunk was
    dropped as a document tail."""
    chunk = poss // content
    off = poss % content
    rows = np.array([lookup.get((int(d), int(c)), -1)
                     for d, c in zip(docs, chunk)], dtype=np.int64)
    return rows, (n_prefix + off).astype(np.int64)


def _stack(rows, cpos, docpos, idx):
    """Row triple (cache_row, cache_pos, doc_pos). `gather_tok`/`gather_win`
    read columns 0/1 only; column 2 carries the Heaps confound for the
    position floor."""
    return np.stack([rows[idx], cpos[idx], docpos[idx]], 1)


def _tercile_edges(vals: np.ndarray):
    return np.quantile(vals, [1 / 3, 2 / 3])


def build_rows(key: str):
    """Returns {(target, split): (rows(n,2), y(n,))}, plus stats."""
    tag = MODELS[key]
    z = np.load(LABELS / f"novelty_fineweb_{tag}.npz")
    c = np.load(CACHE_ROOT / key / "tokens.npz")
    ids, doc_idx, n_prefix = c["ids"], c["doc_idx"], int(c["n_prefix"])
    content = ids.shape[1] - n_prefix
    flat, off_doc = z["token_ids"], z["doc_off"]

    # Re-assert the card § 2 mapping at run time (cheap, and the whole
    # zero-new-caching claim rests on it).
    lookup = _row_lookup(doc_idx)
    for (d, k), i in list(lookup.items())[:200]:
        s = off_doc[d] + k * content
        assert np.array_equal(flat[s:s + content], ids[i, n_prefix:]), \
            f"flat/window mismatch at doc {d} chunk {k}"

    doc_split = z["doc_split"]
    stats = {"n_prefix": n_prefix, "content": content,
             "n_cache_rows": int(ids.shape[0])}

    # --- eligible pool (uniform across T), for both faces ---
    md, mp = z["man_nov_doc"], z["man_nov_pos"]
    rows_all, cpos_all = _map_rows(md, mp, lookup, content, n_prefix)
    elig = (rows_all >= 0) & (mp >= POS_MIN) & (mp % content >= OFF_MIN)

    out: dict = {}
    # PRIMARY: the builder's committed balanced manifest.
    for split_name, flag in (("train", 0), ("test", 1)):
        m = elig & (doc_split[md] == flag)
        per = {}
        keep_r, keep_y = [], []
        for cls in (0, 1, 2):
            idx = np.flatnonzero(m & (z["man_nov_cls"] == cls))
            rng = _seeded(f"nov/{key}/{split_name}/{cls}")
            if len(idx) > CAP[split_name]:
                idx = rng.choice(idx, CAP[split_name], replace=False)
            per[cls] = len(idx)
            keep_r.append(_stack(rows_all, cpos_all, mp, idx))
            keep_y.append(np.full(len(idx), cls, dtype=np.int64))
        stats[f"nov_bin/{split_name}"] = {
            "rows_per_class": per,
            "ok": bool(min(per.values()) >= MIN_ROWS)}
        out[("nov_bin", split_name)] = (np.concatenate(keep_r),
                                        np.concatenate(keep_y))

    # RECEIPT: null-face terciles over the SAME eligible pool, own
    # balanced draw (card § 4). Edges from the eligible TRAIN pool.
    resid_null = z["nov_resid_null"]
    flat_pos = off_doc[md] + mp                      # index into the flat stream
    tr_pool = elig & (doc_split[md] == 0)
    edges = _tercile_edges(resid_null[flat_pos[tr_pool]])
    stats["nov_null_edges"] = [float(e) for e in edges]
    null_cls = np.searchsorted(edges, resid_null[flat_pos], side="right")
    for split_name, flag in (("train", 0), ("test", 1)):
        m = elig & (doc_split[md] == flag)
        per, keep_r, keep_y = {}, [], []
        n_min = min(int((m & (null_cls == c)).sum()) for c in (0, 1, 2))
        n_take = min(n_min, CAP[split_name])
        for cls in (0, 1, 2):
            idx = np.flatnonzero(m & (null_cls == cls))
            rng = _seeded(f"novnull/{key}/{split_name}/{cls}")
            if len(idx) > n_take:
                idx = rng.choice(idx, n_take, replace=False)
            per[cls] = len(idx)
            keep_r.append(_stack(rows_all, cpos_all, mp, idx))
            keep_y.append(np.full(len(idx), cls, dtype=np.int64))
        stats[f"nov_null_bin/{split_name}"] = {
            "rows_per_class": per,
            "ok": bool(min(per.values()) >= MIN_ROWS)}
        out[("nov_null_bin", split_name)] = (np.concatenate(keep_r),
                                             np.concatenate(keep_y))
    return out, stats


def screen(key: str):
    RES.mkdir(exist_ok=True)
    out_path = RES / f"screen_{key}.json"
    hs = SCREEN_HS[key]
    manifests, mstats = build_rows(key)
    done = json.loads(out_path.read_text()) if out_path.exists() else {
        "meta": {"model": key, "screen_hs": hs, "card": "CARD.md (frozen)",
                 "t_grid": T_GRID, "mean_ts": MEAN_TS, "mlp_t": MLP_T,
                 "null_t": NULL_T, "rows": mstats,
                 "kernel_mass": {"4": 0.170, "8": 0.312, "16": 0.533,
                                 "32": 0.800, "64": 1.0}},
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

    for target, ts_flat, ts_mean in (("nov_bin", T_GRID, MEAN_TS),
                                     ("nov_null_bin", [], RECEIPT_TS)):
        if not (mstats[f"{target}/train"]["ok"]
                and mstats[f"{target}/test"]["ok"]):
            print(f"[{key} {target}] SKIP (insufficient matched rows)")
            continue
        rtr, ytr = manifests[(target, "train")]
        rte, yte = manifests[(target, "test")]
        ytr_t, yte_t = torch.from_numpy(ytr), torch.from_numpy(yte)

        Xtr_tok, Xte_tok = gather_tok(acts, rtr), gather_tok(acts, rte)
        run(f"{target}/tok_linear", lambda: summarize(
            fit_probe(Xtr_tok, ytr_t, Xte_tok, yte_t, 3), 3))
        if target == "nov_bin":
            run(f"{target}/tok_mlp", lambda: summarize(
                fit_probe(Xtr_tok, ytr_t, Xte_tok, yte_t, 3, hidden=512), 3))
            # position-only floor on the SHIPPED rows (card § 5)
            def _pos_feats(r):
                cp = r[:, 1].astype(np.float32)          # in-chunk position
                dp = r[:, 2].astype(np.float32)          # doc position (Heaps)
                f = np.stack([cp, cp ** 2 / 128.0,
                              np.log2(1.0 + dp), dp / 1000.0], 1)
                return torch.from_numpy(f).to(torch.float16)
            run(f"{target}/position_floor", lambda: summarize(
                fit_probe(_pos_feats(rtr), ytr_t, _pos_feats(rte), yte_t,
                          3), 3))

        for T in sorted(set(ts_flat) | set(ts_mean)):
            Wtr = gather_win(acts, rtr, T)
            Wte = gather_win(acts, rte, T)
            if T in ts_mean:
                run(f"{target}/T{T}/win_mean_linear", lambda: summarize(
                    fit_probe(win_mean(Wtr), ytr_t, win_mean(Wte), yte_t,
                              3), 3))
            if T in ts_flat:
                flat_tr = Wtr.reshape(len(rtr), -1)
                flat_te = Wte.reshape(len(rte), -1)
                run(f"{target}/T{T}/win_linear", lambda: summarize(
                    fit_probe(flat_tr, ytr_t, flat_te, yte_t, 3), 3))
                srng = np.random.default_rng(
                    SHUF_SEED + zlib.crc32(f"{target}/T{T}".encode())
                    % 2 ** 16)
                Str = shuffle_context(Wtr, srng).reshape(len(rtr), -1)
                Ste = shuffle_context(Wte, srng).reshape(len(rte), -1)
                run(f"{target}/T{T}/win_shuf_linear", lambda: summarize(
                    fit_probe(Str, ytr_t, Ste, yte_t, 3), 3))
                if T in MLP_T:
                    run(f"{target}/T{T}/win_mlp", lambda: summarize(
                        fit_probe(flat_tr, ytr_t, flat_te, yte_t, 3,
                                  hidden=512), 3))
                    run(f"{target}/T{T}/win_shuf_mlp", lambda: summarize(
                        fit_probe(Str, ytr_t, Ste, yte_t, 3, hidden=512), 3))
                if T == NULL_T:
                    nrng = np.random.default_rng(NULL_SEED)
                    yn = torch.from_numpy(nrng.permutation(ytr))
                    run(f"{target}/T{T}/null_win_linear", lambda: summarize(
                        fit_probe(flat_tr, yn, flat_te, yte_t, 3), 3))
                    run(f"{target}/null_tok_linear", lambda: summarize(
                        fit_probe(Xtr_tok, yn, Xte_tok, yte_t, 3), 3))
                del flat_tr, flat_te, Str, Ste
            del Wtr, Wte
    save()
    print(f"[{key}] DONE -> {out_path}")


def main():
    keys = sys.argv[1:] or list(MODELS)
    for k in keys:
        screen(k)


if __name__ == "__main__":
    main()
