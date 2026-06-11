"""hh_screen.py — frequency screening of the HH-RLHF choice task (sprint 2B).

Task: distinguish CHOSEN from REJECTED assistant replies (Anthropic/hh-rlhf)
from windows of DeepSeek-R1-Distill-Llama-8B layer-10 residuals — the same
model/hook/space as the backtracking study, so band profiles are comparable.

Construction: take N pairs; keep the last CTX=128 tokens of each transcript
(the divergent reply lives at the end; sequences shorter than CTX dropped
pairwise); label chosen=1 / rejected=0; split BY PAIR (both members of a
pair share prompt text → same split bucket).

Screens (probe = balanced linear, AUC):
  raw mean@T : T in {1,2,4,8,16,32,64,128}, right-edge at the final token —
               the dictionary-free timescale curve.
  band@T=32  : pooled DCT-band features — for band B, feature = mean over
               w in B of the window's DCT coefficient c_w (each d-dim).
               Bands: {0}, {1..10}, {11..20}, {21..31} (DC band == window
               mean, consistent with the timescale curve at T=32).

Run: python hh_screen.py --out hh_out
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from fb_core import DEVICE, dct_basis
from bt_freq import probe_with_auc

CTX = 128
N_PAIRS = 2000
TS = [1, 2, 4, 8, 16, 32, 64, 128]
BANDS32 = [[0], list(range(1, 11)), list(range(11, 21)), list(range(21, 32))]


def build_cache(out):
    if os.path.exists(f"{out}/acts.fp16.npy"):
        return
    os.makedirs(out, exist_ok=True)
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    ds = load_dataset("Anthropic/hh-rlhf", split="train", streaming=True)
    seqs, labels, pair_ids = [], [], []
    n_kept = 0
    for i, rec in enumerate(ds):
        if n_kept >= N_PAIRS:
            break
        ids_c = tok.encode(rec["chosen"])
        ids_r = tok.encode(rec["rejected"])
        if len(ids_c) < CTX or len(ids_r) < CTX:
            continue
        seqs.append(ids_c[-CTX:])
        labels.append(1)
        pair_ids.append(n_kept)
        seqs.append(ids_r[-CTX:])
        labels.append(0)
        pair_ids.append(n_kept)
        n_kept += 1
    print(f"kept {n_kept} pairs", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        torch_dtype=torch.bfloat16).to(DEVICE).eval()
    acts = np.zeros((len(seqs), CTX, 4096), dtype=np.float16)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(seqs), 32):
            batch = torch.tensor(seqs[i:i + 32], device=DEVICE)
            store = {}

            def hook(mod, inp, outp):
                h = outp[0] if isinstance(outp, tuple) else outp
                store["a"] = h.detach()
            h = model.model.layers[10].register_forward_hook(hook)
            model(batch)
            h.remove()
            acts[i:i + 32] = store["a"].to(torch.float16).cpu().numpy()
            if i % 640 == 0:
                print(f"  {i}/{len(seqs)} ({time.time()-t0:.0f}s)", flush=True)
    np.save(f"{out}/acts.fp16.npy", acts)
    np.save(f"{out}/labels.npy", np.array(labels))
    np.save(f"{out}/pair_ids.npy", np.array(pair_ids))
    del model
    torch.cuda.empty_cache()
    print("cache done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="hh_out")
    args = ap.parse_args()
    build_cache(args.out)
    A = torch.tensor(np.load(f"{args.out}/acts.fp16.npy"), device=DEVICE)
    y = torch.tensor(np.load(f"{args.out}/labels.npy"), device=DEVICE)
    pid = np.load(f"{args.out}/pair_ids.npy")
    mu = A.float().mean((0, 1))
    sd = A.float().std((0, 1)) + 1e-6

    rng = np.random.default_rng(7)
    pairs = np.unique(pid)
    test_pairs = set(rng.permutation(pairs)[:len(pairs) // 5].tolist())
    te = torch.tensor([p in test_pairs for p in pid], device=DEVICE)
    tr = ~te

    res = {"n_pairs": int(len(pairs)), "n_test": int(te.sum())}
    for T in TS:
        X = ((A[:, -T:, :].float().mean(1)) - mu) / sd
        r = probe_with_auc(X[tr], y[tr], X[te], y[te], f"T{T}")
        res[f"rawmean_T{T}"] = {k.split("_", 1)[1]: v for k, v in r.items()}
        print(f"[hh raw_mean T={T:3d}] auc={r[f'T{T}_auc']:.3f}", flush=True)

    psi = dct_basis(32).to(DEVICE)                  # (32, 32)
    W32 = ((A[:, -32:, :].float() - mu) / sd)       # (n, 32, d)
    C = torch.einsum("wt,ntd->nwd", psi, W32)       # DCT coefficients
    for bi, band in enumerate(BANDS32):
        F = C[:, band, :].mean(1)                   # pooled band coeff (n, d)
        r = probe_with_auc(F[tr], y[tr], F[te], y[te], f"b{bi}")
        res[f"band{bi}"] = {k.split("_", 1)[1]: v for k, v in r.items()}
        print(f"[hh band{bi} {band[0]}..{band[-1]}] "
              f"auc={r[f'b{bi}_auc']:.3f}", flush=True)
    json.dump(res, open(f"{args.out}/hh_screen.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
