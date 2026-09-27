"""ACTMIX RLHF — paper-match arm (EVAL-ONLY case-study; CARD § 2).

Loads the four shipped seed-42 checkpoints (public txcdr-base;
sha256 recorded), rebuilds each arch with the vendored phase-7
classes, and runs the shared decomposition (decomp.py) on the
rebuilt hh-rlhf cache — plain + within-window-shuffled twins for the
window arch. Writes results/papermatch.json. NOT a leaderboard row
(out-of-runner currency; probe_codes precedent — CARD § 2).

Run: .venv/bin/python -m experiments.explorations.actmix_rlhf.papermatch
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.explorations.actmix_rlhf import decomp

HERE = Path(__file__).resolve().parent
CKPT_DIR = Path("/workspace/caches/rlhf/txcdr-base/ckpts")
LOG_DIR = Path("/workspace/caches/rlhf/txcdr-base/training_logs")
CACHE = Path("/workspace/caches/rlhf/cached_hh_rlhf")
OUT = HERE / "results"
SHUFFLE_SEED = 42
D_IN, D_SAE = 2304, 18432

ARCHS = ("topk_sae", "tsae_paper_k500", "tsae_paper_k20",
         "agentic_txc_02")


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(arch_id: str, device):
    meta = json.loads((LOG_DIR / f"{arch_id}__seed42.json").read_text())
    ckpt = CKPT_DIR / f"{arch_id}__seed42.pt"
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    sd = {k: (v.float() if torch.is_tensor(v) and v.dtype == torch.float16
              else v) for k, v in sd.items()}
    src = meta["src_class"]
    if src == "TopKSAE":
        from experiments.explorations.actmix_rlhf.vendor.topk_sae import TopKSAE
        model = TopKSAE(D_IN, D_SAE, k=int(meta["k_pos"])).to(device)
    elif src == "TemporalMatryoshkaBatchTopKSAE":
        from experiments.explorations.actmix_rlhf.vendor.tsae_paper import (
            TemporalMatryoshkaBatchTopKSAE)
        gs = list(meta.get("group_sizes") or
                  [int(0.2 * D_SAE), D_SAE - int(0.2 * D_SAE)])
        model = TemporalMatryoshkaBatchTopKSAE(
            activation_dim=D_IN, dict_size=D_SAE,
            k=int(meta["k_pos"]), group_sizes=gs).to(device)
    elif src == "MatryoshkaTXCDRContrastiveMultiscale":
        from experiments.explorations.actmix_rlhf.vendor.\
            matryoshka_txcdr_contrastive_multiscale import (
                MatryoshkaTXCDRContrastiveMultiscale)
        model = MatryoshkaTXCDRContrastiveMultiscale(
            D_IN, D_SAE, T=int(meta["T"]), k=int(meta["k_win"]),
            n_contr_scales=int(meta.get("n_scales", 3)),
            gamma=float(meta.get("gamma", 0.5))).to(device)
    else:
        raise ValueError(src)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.eval()
    T = int(meta.get("T") or 1)
    return model, meta, T, {"ckpt_sha256": _sha256(ckpt),
                            "missing_keys": list(missing),
                            "unexpected_keys": list(unexpected)}


def _encode_fn(model, src_class: str, use_threshold: bool = True):
    if src_class == "TopKSAE":
        return lambda x: model.encode(x)
    if src_class == "TemporalMatryoshkaBatchTopKSAE":
        def f(x):
            z = model.encode(x, use_threshold=use_threshold)
            return z[0] if isinstance(z, tuple) else z
        return f
    # MatryoshkaTXCDRContrastiveMultiscale — window encode (B,T,d)->(B,d_sae)
    return lambda w: model.encode(w)


@torch.no_grad()
def main():
    device = torch.device("cuda")
    chosen = np.load(CACHE / "chosen.npz")
    rejected = np.load(CACHE / "rejected.npz")
    c_acts, r_acts = chosen["acts"], rejected["acts"]
    c_mask, r_mask = chosen["response_mask"], rejected["response_mask"]
    c_len = chosen["response_len"].astype(np.float64)
    r_len = rejected["response_len"].astype(np.float64)
    valid = (c_len > 0) & (r_len > 0)
    print(f"cache N={len(c_len)} valid={int(valid.sum())}")

    results = {"protocol": "actmix_rlhf papermatch v1 (CARD § 2/3)",
               "cache_meta": json.loads((CACHE / "meta.json").read_text()),
               "cells": {}}
    for arch_id in ARCHS:
        t0 = time.time()
        model, meta, T, prov = _load(arch_id, device)
        src = meta["src_class"]
        enc = _encode_fn(model, src)
        print(f"=== {arch_id} (src={src}, T={T}) ===", flush=True)

        cell = {"meta": {k: meta.get(k) for k in
                         ("arch_id", "src_class", "d_sae", "k_pos",
                          "k_win", "T", "group_sizes")},
                "provenance": prov, "variants": {}}
        variants = [("plain", None)]
        if T > 1:
            variants.append(("shuffled", SHUFFLE_SEED))
        for tag, seed in variants:
            c_pe, c_l0 = decomp.aggregate_response_mean(
                enc, c_acts, c_mask, T=T, d_sae=D_SAE, device=device,
                shuffle_seed=seed)
            r_pe, r_l0 = decomp.aggregate_response_mean(
                enc, r_acts, r_mask, T=T, d_sae=D_SAE, device=device,
                shuffle_seed=seed)
            v = {"preference_auc": decomp.preference_auc(c_pe, r_pe, valid),
                 "preference_auc_k50": decomp.preference_auc(
                     c_pe, r_pe, valid, k=50),
                 "mass_at_20": decomp.mass_at_k(c_pe, r_pe, valid),
                 "length_pearson": decomp.length_pearson_topk(
                     c_pe, r_pe, c_len, r_len, valid),
                 "realized_l0": {"chosen": c_l0, "rejected": r_l0}}
            cell["variants"][tag] = v
            print(f"  [{tag}] auc={v['preference_auc']['auc_mean']:.4f} "
                  f"mass@20={v['mass_at_20']:.3f} "
                  f"l0={c_l0['l0_per_unit']:.1f} "
                  f"len_spurious={v['length_pearson']['n_spurious_r_gt_0.5']}",
                  flush=True)
            if tag == "plain" and T == 1:
                cell["shuffle_note"] = ("T=1: within-window shuffle is the "
                                        "identity BY CONSTRUCTION (CARD § 2)")
        if "shuffled" in cell["variants"]:
            g = (cell["variants"]["plain"]["preference_auc"]["auc_mean"]
                 - cell["variants"]["shuffled"]["preference_auc"]["auc_mean"])
            cell["shuffle_gap_auc"] = g
            print(f"  shuffle_gap_auc = {g:+.4f}", flush=True)
        results["cells"][arch_id] = cell
        print(f"  ({time.time() - t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()

    OUT.mkdir(exist_ok=True)
    (OUT / "papermatch.json").write_text(json.dumps(results, indent=1))
    print(f"-> {OUT / 'papermatch.json'}")


if __name__ == "__main__":
    main()
