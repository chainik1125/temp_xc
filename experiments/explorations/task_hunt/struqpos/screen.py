"""STRUQPOS screen — probe the readout residual for injection POSITION
(executes STRUQPOS_SCREEN_CARD §4). CPU/GPU-light: fits linear + MLP
probes on the cached features, held out BY ITEM.

Arms per leg (card §4):
  tok        — bag input-embedding baseline (the floor; reported FIRST)
  ctx        — contextual readout residual, ORDERED (candidate signal)
  shuf       — contextual readout residual, FIELD-SHUFFLED (positional null)
  local_floor— last-4 field-token input-embeddings (proximity floor, PIN 2)
  shuf_labelperm — label-permuted refit of the shuf arm (PIN 1 receipt:
                   measures the null's own floor; expected ~0.50)

Held-out by ITEM (train = split 0, test = split 1 — A and B of an item
never split). Reports rank-AUC (linear + MLP, best taken by the verdict)
and a per-attack-type breakdown (5 types; visibility, no bar).

Run: .venv/bin/python -m experiments.explorations.task_hunt.struqpos.screen [leg ...]
Writes results/screen_struqpos_<leg>.json (resumable per leg).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch

from experiments.explorations.conversion_depth.problib import fit_probe
from experiments.explorations.task_hunt.struqpos.cache_acts import (
    CACHE_ROOT, LEGS,
)

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
PERM_SEED = 9137


def _auc(ftr, ytr, fte, yte, hidden):
    r = fit_probe(torch.from_numpy(ftr), torch.from_numpy(ytr.astype(np.int64)),
                  torch.from_numpy(fte), torch.from_numpy(yte.astype(np.int64)),
                  n_classes=2, hidden=hidden, seed=0)
    return float(r.get("auc", float("nan")))


def _arm(feat, y, tr, te, hidden):
    return _auc(feat[tr], y[tr], feat[te], y[te], hidden)


def screen_leg(leg: str) -> dict:
    z = np.load(CACHE_ROOT / f"struqpos_acts_{leg}.npz", allow_pickle=True)
    y = z["y"].astype(np.int64)
    split, attack, item = z["split"], z["attack"], z["item"]
    attacks = [str(a) for a in z["attacks"]]
    tr, te = split == 0, split == 1
    feats = {"tok": z["bag"].astype(np.float32),
             "ctx": z["res_ord"].astype(np.float32),
             "shuf": z["res_shuf"].astype(np.float32),
             "local_floor": z["local4"].astype(np.float32)}

    out = {"leg": leg, "n_docs": int(len(y)), "hs": int(z["hs"]),
           "n_train": int(tr.sum()), "n_test": int(te.sum()),
           "class_balance_test": [int((y[te] == 0).sum()), int((y[te] == 1).sum())],
           "arms": {}}
    # tok FIRST (standing rule)
    order = ["tok", "ctx", "shuf", "local_floor"]
    for name in order:
        f = feats[name]
        lin = _arm(f, y, tr, te, 0)
        mlp = _arm(f, y, tr, te, 64)
        out["arms"][name] = {"linear": lin, "mlp": mlp, "best": max(lin, mlp)}
        print(f"[{leg}] {name:12s} linear={lin:.4f} mlp={mlp:.4f}", flush=True)

    # PIN-1 receipt: label-permuted refit of the shuffled arm (own floor)
    rng = np.random.default_rng(PERM_SEED)
    yperm = y.copy()
    yperm[tr] = rng.permutation(yperm[tr])
    lp = _auc(feats["shuf"][tr], yperm[tr], feats["shuf"][te], y[te], 0)
    out["shuf_labelperm_auc"] = lp
    print(f"[{leg}] shuf_labelperm linear={lp:.4f} (expect ~0.50)", flush=True)

    # per-attack-type breakdown (visibility)
    pa = {}
    for ai, a in enumerate(attacks):
        m = attack == ai
        tr_a, te_a = tr & m, te & m
        if te_a.sum() < 20 or tr_a.sum() < 20:
            pa[a] = {"skipped": True, "n_test": int(te_a.sum())}
            continue
        pa[a] = {
            "tok": _auc(feats["tok"][tr_a], y[tr_a], feats["tok"][te_a], y[te_a], 0),
            "ctx": _auc(feats["ctx"][tr_a], y[tr_a], feats["ctx"][te_a], y[te_a], 0),
            "shuf": _auc(feats["shuf"][tr_a], y[tr_a], feats["shuf"][te_a], y[te_a], 0),
            "n_test": int(te_a.sum())}
    out["per_attack"] = pa
    RES.mkdir(exist_ok=True)
    (RES / f"screen_struqpos_{leg}.json").write_text(json.dumps(out, indent=1))
    print(f"[{leg}] -> results/screen_struqpos_{leg}.json", flush=True)
    return out


if __name__ == "__main__":
    for leg in (sys.argv[1:] or list(LEGS)):
        screen_leg(leg)
