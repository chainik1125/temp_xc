"""em-redo Phase A — the PROBE currency: linear probes on trained codes.

For every frozen cell (em_redo_cells.py) this loads the trained
checkpoint (resolved through the runner's own key machinery — identical
train_keys) and probes its codes for the rollout misalignment label on
the stage-4 cohort, using the SAME rows, folds, and probe stack as the
raw-activation g(ℓ) map (phase4_em_depth.py) so trained-code numbers
subtract cleanly from the raw ceilings:

- rows: (rollout, p) for p ∈ {15, 19, 23, …} (stride 4) within each
  rollout's assistant tokens; labels/prompt-ids from the parent rollout;
  GroupKFold(4) via qid % 4; probe = frozen problib stack (full-batch
  Adam linear, class-weighted, rank-AUC).
- per-token read  = the arch's finest code at position p:
    token archs   → z_p = encode(x_p)                       (d_sae)
    sequence arch → encode(16-window)[:, -1]                (tsae)
    window archs  → shared code of the T_arch-window ending at p
- window read     = amax over |codes| across the T=16 window ending at
  p (the detection pooling convention):
    token archs   → amax_j |z_j|, j ∈ window
    sequence arch → amax over the 16 per-position codes
    window archs  → amax over the 16−T_arch+1 stride-1 sub-window codes
- trained window advantage A = AUC(window read) − AUC(per-token read);
  code-vs-raw = AUC(code read) − AUC(raw read, RECORD § 4 same rows).
- realized l0_per_token measured on the same code matrices (Part II
  matching key), plus one permutation-null anchor cell
  (batchtopk_sae / L13 / seed 42, seed 99) for the null scale.

Writes results/em_redo_probe_codes.json incrementally (resumable).
Run:  .venv/bin/python -m experiments.explorations.conversion_depth.probe_codes
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from temp_bench.core.config import (
    compute_data_key,
    compute_train_key,
    load_arch,
    load_datasource,
)
from temp_bench.core.runner import _load_checkpoint

from experiments.explorations.conversion_depth.em_redo_cells import all_cells
from experiments.explorations.conversion_depth.problib import fit_probe

EM_DIR = Path("/workspace/conv_depth_caches/em_medical")
HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "results" / "em_redo_probe_codes.json"
T16 = 16
POS_STRIDE = 4
N_FOLDS = 4
NULL_SEED = 99
NULL_CELL = ("batchtopk_sae", 13, 42)   # (cell_id, layer, seed)
ENC_BATCH = 512


def build_rows():
    lens = np.load(EM_DIR / "lens.npy")
    labels = np.load(EM_DIR / "labels.npy")
    qids = np.load(EM_DIR / "qids.npy")
    rows, row_lab, row_q = [], [], []
    for ri in range(len(lens)):
        for p in range(T16 - 1, int(lens[ri]), POS_STRIDE):
            rows.append((ri, p))
            row_lab.append(labels[ri])
            row_q.append(qids[ri])
    return (np.array(rows, dtype=np.int64),
            np.array(row_lab, dtype=np.int64),
            np.array(row_q, dtype=np.int64))


@torch.no_grad()
def encode_cell(model, acts, rows):
    """Return (Z_tok, Z_win) fp16 CPU: per-token + amax-window code reads."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    T_arch = int(model.config.T)
    consumes = getattr(model, "consumes", "token")
    d_sae = int(model.config.d_sae)
    n = len(rows)
    Z_tok = torch.empty((n, d_sae), dtype=torch.float16)
    Z_win = torch.empty((n, d_sae), dtype=torch.float16)
    nnz_tok = 0.0
    for s in range(0, n, ENC_BATCH):
        e = min(s + ENC_BATCH, n)
        b = e - s
        X = torch.empty((b, T16, acts.shape[-1]), dtype=torch.float16)
        for j in range(T16):
            w = rows[s:e, 0]
            p = rows[s:e, 1] - (T16 - 1) + j
            X[:, j] = torch.from_numpy(
                np.asarray(acts[w, p], dtype=np.float16))
        X = X.to(device=device, dtype=dtype)
        if consumes == "token":
            z = model.encode(X.reshape(b * T16, -1)).reshape(b, T16, d_sae)
            z_tok = z[:, -1]
            z_win = z.abs().amax(dim=1)
        elif consumes == "sequence":
            z = model.encode(X)
            if z.dim() == 2:                       # (B, d_sae) fallback
                z_tok = z
                z_win = z.abs()
            else:                                  # (B, 16, d_sae)
                z_tok = z[:, -1]
                z_win = z.abs().amax(dim=1)
        else:                                      # window arch
            subs = []
            for jend in range(T_arch - 1, T16):
                zs = model.encode(X[:, jend - T_arch + 1: jend + 1])
                if zs.dim() == 3:                  # (B, 1, d_sae) or (B,T,d)
                    zs = zs.abs().amax(dim=1) if zs.shape[1] > 1 \
                        else zs.squeeze(1)
                subs.append(zs)
            zstack = torch.stack(subs, dim=1)      # (B, n_sub, d_sae)
            z_tok = zstack[:, -1]
            z_win = zstack.abs().amax(dim=1)
        Z_tok[s:e] = z_tok.to(torch.float16).cpu()
        Z_win[s:e] = z_win.to(torch.float16).cpu()
        nnz_tok += float((z_tok != 0).float().sum().item())
        del X, z_tok, z_win
    l0_tok_read = nnz_tok / n
    return Z_tok, Z_win, l0_tok_read


def probe_pair(Z_tok, Z_win, row_lab, row_q, *, permute_seed=None):
    folds = [(row_q % N_FOLDS) != f for f in range(N_FOLDS)]
    out = {"token": {"auc_folds": []}, "window": {"auc_folds": []}}
    for f, tr_mask in enumerate(folds):
        te_mask = ~tr_mask
        ytr = torch.from_numpy(row_lab[tr_mask])
        yte = torch.from_numpy(row_lab[te_mask])
        if permute_seed is not None:
            g = torch.Generator().manual_seed(permute_seed + f)
            ytr = ytr[torch.randperm(len(ytr), generator=g)]
            yte = yte[torch.randperm(len(yte), generator=g)]
        tm = torch.from_numpy(tr_mask)
        em = torch.from_numpy(te_mask)
        for name, Z in [("token", Z_tok), ("window", Z_win)]:
            r = fit_probe(Z[tm], ytr, Z[em], yte, 2, class_weight=True)
            out[name]["auc_folds"].append(r["auc"])
    for name in out:
        out[name]["auc"] = float(np.mean(out[name]["auc_folds"]))
    out["advantage"] = out["window"]["auc"] - out["token"]["auc"]
    return out


def main():
    rows, row_lab, row_q = build_rows()
    print(f"[probe_codes] {len(rows)} rows, pos frac {row_lab.mean():.3f}",
          flush=True)
    done = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {
        "meta": {"T16": T16, "pos_stride": POS_STRIDE, "n_folds": N_FOLDS,
                 "n_rows": int(len(rows)),
                 "prereg": "TRACKING.md § 2 (frozen before training)"},
        "cells": {}}

    acts_cache = {}
    for c in all_cells(include_anchors=True):
        key = f"{c['cell_id']}/L{c['layer']}/s{c['seed']}"
        if key in done["cells"]:
            continue
        arch_spec = load_arch(c["arch"], section="em")
        tc = c["training_cfg"]
        if tc.arch_hparams_override:
            merged = {**arch_spec.hparams, **tc.arch_hparams_override}
            arch_spec = arch_spec.model_copy(update={"hparams": merged})
        data_spec = load_datasource(c["datasource"])
        data_key = compute_data_key(data_spec)
        train_key = compute_train_key(arch=arch_spec, seed=c["seed"],
                                      training_cfg=tc, data_key=data_key,
                                      section="em")
        from temp_bench.core.cache import checkpoint_exists
        if not checkpoint_exists(train_key):
            print(f"[{key}] checkpoint missing ({train_key}); skip",
                  flush=True)
            continue
        t0 = time.time()
        model = _load_checkpoint(arch_spec, train_key, data_spec)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        hs = c["layer"] + 1
        if hs not in acts_cache:
            acts_cache.clear()
            acts_cache[hs] = np.load(EM_DIR / f"hs{hs}.npy", mmap_mode="r")
        Z_tok, Z_win, l0_tok_read = encode_cell(model, acts_cache[hs], rows)
        cell = probe_pair(Z_tok, Z_win, row_lab, row_q)
        cell["train_key"] = train_key
        cell["l0_per_token_read"] = l0_tok_read
        if (c["cell_id"], c["layer"], c["seed"]) == NULL_CELL:
            cell["null"] = probe_pair(Z_tok, Z_win, row_lab, row_q,
                                      permute_seed=NULL_SEED)
        done["cells"][key] = cell
        OUT_JSON.write_text(json.dumps(done, indent=1))
        print(f"[{key}] tok={cell['token']['auc']:.3f} "
              f"win={cell['window']['auc']:.3f} "
              f"adv={cell['advantage']:+.3f} "
              f"l0/tok_read={l0_tok_read:.1f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        del model, Z_tok, Z_win
        torch.cuda.empty_cache()
    print(f"[probe_codes] DONE -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
