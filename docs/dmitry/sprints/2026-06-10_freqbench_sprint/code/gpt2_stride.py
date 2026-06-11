"""gpt2_stride.py — first real-model check of the FrequencyBench pipeline.

Process: sequences of 16 weekday names with constant stride y in Z_7
(e.g. stride 2 = "Monday Wednesday Friday ..."), random start day.
GPT-2-small residual stream (post-block) at each day-token position gives a
(16, 768) window per sequence; the hidden label is the stride (7 classes).

Engels et al. (2024) found weekdays embedded on a circle in LLMs, so this is
the in-the-wild version of the circle task. Differences from the synthetic
setting, measured rather than assumed:
  - attention mixes context into every position, so the single-position
    ceiling A_loc is EMPIRICAL (probe on the middle position), not 1/7;
  - there is no exact oracle; the reference decoder is an MLP probe on the
    raw stacked window (information present in the window of residuals).

Outputs per (layer, arch, seed): JSON with probe accuracies; plus per-layer
baselines and a day-embedding PCA check (are the 7 day means circular?).

Run: python gpt2_stride.py --out rm_out
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from fb_core import DEVICE, build_model, fit_probe, train_model

DAYS = [" Monday", " Tuesday", " Wednesday", " Thursday", " Friday",
        " Saturday", " Sunday"]
M, W = 7, 16
STRIDES = list(range(7))          # 0..6; sign pairs (1,6),(2,5),(3,4)
# hidden_states indices: 0 = embeddings (wte+wpe), k = output of block k-1
HSLIST = [0, 1, 4, 7]
N_DICT = 60_000                    # windows for dictionary training pool
N_TR, N_TE = 20_000, 5_000


def gen_sequences(n, rng):
    y = rng.integers(0, 7, size=n)
    B = rng.integers(0, 7, size=n)
    t = np.arange(W)
    Q = (B[:, None] + y[:, None] * t[None, :]) % 7
    return Q, y


@torch.no_grad()
def extract(model, tok, Q, layers, batch=256):
    """Q: (n, W) day indices -> dict layer -> (n, W, 768) float16 tensors."""
    day_ids = [tok.encode(d)[0] for d in DAYS]
    assert all(len(tok.encode(d)) == 1 for d in DAYS), "day not single token"
    ids = torch.tensor([[day_ids[q] for q in row] for row in Q])
    outs = {L: [] for L in layers}
    for i in range(0, ids.shape[0], batch):
        chunk = ids[i:i + batch].to(DEVICE)
        res = model(chunk, output_hidden_states=True)
        for L in layers:
            outs[L].append(res.hidden_states[L].detach()
                           .to(torch.float16).cpu())
    return {L: torch.cat(v) for L, v in outs.items()}


def probe_pack(Ftr, ytr, Fte, yte, tag, mlp=True):
    out = {}
    r = fit_probe(Ftr, ytr, Fte, yte, 7)
    out[f"{tag}_linear"] = {k: r[k] for k in ("acc_train", "acc_test",
                                              "per_class")}
    if mlp:
        r2 = fit_probe(Ftr, ytr, Fte, yte, 7, hidden=512)
        out[f"{tag}_mlp"] = {k: r2[k] for k in ("acc_train", "acc_test",
                                                "per_class")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="rm_out")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--H", type=int, default=2048)
    ap.add_argument("--kwin", type=int, default=128)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    gpt = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE).eval()

    rng = np.random.default_rng(0)
    n_total = N_DICT + N_TR + N_TE
    Q, y = gen_sequences(n_total, rng)
    print("extracting activations...", flush=True)
    t0 = time.time()
    acts = extract(gpt, tok, Q, HSLIST)
    print(f"extracted in {time.time()-t0:.0f}s", flush=True)
    del gpt
    torch.cuda.empty_cache()

    y_t = torch.tensor(y, device=DEVICE)
    sl_dict = slice(0, N_DICT)
    sl_tr = slice(N_DICT, N_DICT + N_TR)
    sl_te = slice(N_DICT + N_TR, n_total)

    for L in HSLIST:
        X = acts[L].float()
        mu, sd = X[sl_dict].mean((0, 1)), X[sl_dict].std((0, 1)).clamp(1e-6)
        X = ((X - mu) / sd)            # standardize per dim
        Xtr = X[sl_tr].to(DEVICE)
        Xte = X[sl_te].to(DEVICE)
        ytr, yte = y_t[sl_tr], y_t[sl_te]

        bpath = os.path.join(args.out, f"L{L}_baselines.json")
        if not os.path.exists(bpath):
            res = {"hs_index": L}
            mid = W // 2
            # position-resolved single-position ceilings (linear)
            pos_acc = []
            for t in range(W):
                r = fit_probe(Xtr[:, t, :], ytr, Xte[:, t, :], yte, 7)
                pos_acc.append(r["acc_test"])
            res["raw_pos_linear"] = pos_acc
            res.update(probe_pack(Xtr[:, mid, :], ytr, Xte[:, mid, :], yte,
                                  "raw_token"))
            res.update(probe_pack(Xtr.flatten(1), ytr, Xte.flatten(1), yte,
                                  "raw_stacked"))
            res.update(probe_pack(Xtr.mean(1), ytr, Xte.mean(1), yte,
                                  "raw_meanpool", mlp=False))
            # day-circle check: PCA of per-day mean activation (middle pos)
            Qte = Q[sl_te]
            day_means = np.stack([
                Xte[:, mid, :][torch.tensor(Qte[:, mid] == d)].mean(0)
                .cpu().numpy() for d in range(7)])
            dm = day_means - day_means.mean(0)
            U, S, Vt = np.linalg.svd(dm, full_matrices=False)
            res["day_pca_var_ratio"] = (S ** 2 / (S ** 2).sum()).tolist()
            res["day_pca_coords"] = (U[:, :2] * S[:2]).tolist()
            json.dump(res, open(bpath, "w"), indent=1)
            print(f"[L{L} base] tok_lin="
                  f"{res['raw_token_linear']['acc_test']:.3f} "
                  f"tok_mlp={res['raw_token_mlp']['acc_test']:.3f} "
                  f"stack_lin={res['raw_stacked_linear']['acc_test']:.3f} "
                  f"stack_mlp={res['raw_stacked_mlp']['acc_test']:.3f}",
                  flush=True)

        class Shim:
            def sample(self, n):
                idx = torch.randint(0, N_DICT, (n,))
                return X[sl_dict][idx].to(DEVICE), None, None

        archs = (["token_sae", "txc", "multiband", "conv"] if L == 0 else [])
        for arch in archs:
            for seed in args.seeds:
                tag = f"L{L}_{arch}_s{seed}"
                path = os.path.join(args.out, tag + ".json")
                if os.path.exists(path):
                    continue
                t1 = time.time()
                torch.manual_seed(7000 + seed)
                model = build_model(arch, 768, W, args.H, args.kwin).to(DEVICE)
                tr = train_model(model, Shim(), steps=args.steps, batch=256,
                                 pregen=16384)
                with torch.no_grad():
                    Ftr = torch.cat([model.window_codes(Xtr[i:i + 1024])
                                     for i in range(0, N_TR, 1024)])
                    Fte = torch.cat([model.window_codes(Xte[i:i + 1024])
                                     for i in range(0, N_TE, 1024)])
                res = {"hs_index": L, "arch": arch, "seed": seed,
                       "H": args.H, "k_win": args.kwin,
                       "fvu": tr.fvu, "l0": tr.l0}
                res.update(probe_pack(Ftr, ytr, Fte, yte, "code"))
                json.dump(res, open(path, "w"), indent=1)
                print(f"[{tag}] fvu={tr.fvu:.3f} "
                      f"lin={res['code_linear']['acc_test']:.3f} "
                      f"mlp={res['code_mlp']['acc_test']:.3f} "
                      f"({time.time()-t1:.0f}s)", flush=True)
        del Xtr, Xte
        torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
