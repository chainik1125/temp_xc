"""bt_wscan.py — window-timescale scan on the backtracking task (sprint 2A).

Question: backtracking anticipation is low-frequency (sprint 1); does a
longer window — admitting lower frequencies — help, and where does
performance peak in T (= the timescale of the backtracking state)?

Arms (all on DeepSeek-R1-Distill-Llama-8B L10 resid cache, D+ vs
far-negatives, by-trace split, AUC):
  raw_mean@T   : probe on the T-token right-edge window mean (no dictionary)
                 for T in TS_RAW — the dictionary-free timescale measurement.
  dc_sae@T     : TopK SAE trained on window means (params T-independent),
                 T in TS_DC, 2 seeds.
  multiband@T  : full spectral crosscoder, T in {16, 32}, param-matched
                 (H x T = const), with per-branch probes.
  txc@T        : vanilla window TXC control at T in {16, 32}, param-matched.

Windows may extend left into the prompt (cache covers the full sequence), so
positives are comparable across T; require pos - T + 1 >= 0.

Run: python bt_wscan.py --out ws_out
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from fb_core import DEVICE, TokenSAE, build_model, train_model
from bt_freq import build_cache, auc, probe_with_auc

D = 4096
TS_RAW = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96]
TS_DC = [4, 8, 16, 32, 64]
TS_SPEC = [16, 32]
NEG_BUFFER = 25
NEG_PER_POS = 5


class BTW:
    """Like sprint-1 BT but T-parametric and prompt-extending windows."""

    def __init__(self, out):
        self.A = np.load(f"{out}/activations.fp16.npy", mmap_mode="r")
        self.off = np.load(f"{out}/offsets.npy")
        self.ids = json.load(open(f"{out}/trace_ids.json"))
        labs = {j["trace_id"]: j for j in
                (json.loads(l) for l in open("bt_data/labels.jsonl"))}
        self.labs = [labs[i] for i in self.ids]
        idx = np.concatenate([
            np.arange(self.off[k] + l["think_lo"], self.off[k] + l["n_tokens"])
            for k, l in enumerate(self.labs)])
        sample = self.A[np.sort(np.random.default_rng(0).choice(
            idx, size=min(40000, len(idx)), replace=False))].astype(np.float32)
        self.mu = torch.tensor(sample.mean(0), device=DEVICE)
        self.sd = torch.tensor(sample.std(0) + 1e-6, device=DEVICE)
        rng = np.random.default_rng(7)
        perm = rng.permutation(len(self.ids))
        self.test_traces = set(perm[:len(self.ids) // 5].tolist())
        self.Ad = torch.tensor(np.asarray(self.A), device=DEVICE)

    def windows(self, pairs, T, chunk=4096):
        outs = []
        offs = torch.arange(T - 1, -1, -1, device=DEVICE)
        for i in range(0, len(pairs), chunk):
            base = torch.tensor([self.off[k] + p for k, p in pairs[i:i + chunk]],
                                device=DEVICE)
            X = self.Ad[base[:, None] - offs[None, :]].float()
            outs.append((X - self.mu) / self.sd)
        return torch.cat(outs)

    def means(self, pairs, T, chunk=4096):
        outs = []
        offs = torch.arange(T - 1, -1, -1, device=DEVICE)
        for i in range(0, len(pairs), chunk):
            base = torch.tensor([self.off[k] + p for k, p in pairs[i:i + chunk]],
                                device=DEVICE)
            X = self.Ad[base[:, None] - offs[None, :]].float().mean(1)
            outs.append((X - self.mu) / self.sd)   # standardize mean approx
        return torch.cat(outs)

    def probe_pairs(self, train, T):
        """D+ positives vs far negatives; window must fit the sequence
        (pos >= T-1 in absolute tokens; may extend into the prompt)."""
        import random as _r
        pos, neg = [], []
        rng = np.random.default_rng(13 + int(train))
        for k, l in enumerate(self.labs):
            if (k in self.test_traces) == train:
                continue
            ev = np.array(l["event_positions"], dtype=int)
            pos.extend((k, p) for p in l["d_plus_positions"] if p >= T - 1)
            lo = max(l["think_lo"], T - 1)
            cand = [p for p in range(lo, l["n_tokens"])
                    if not len(ev) or np.abs(ev - p).min() > NEG_BUFFER]
            take = (rng.choice(len(cand), size=min(len(cand), 60),
                               replace=False) if cand else [])
            neg.extend((k, cand[i]) for i in take)
        _r.Random(17).shuffle(neg)
        neg = neg[:NEG_PER_POS * max(1, len(pos))]
        y = torch.tensor([1] * len(pos) + [0] * len(neg), device=DEVICE)
        return pos + neg, y

    def think_pairs(self, train, T):
        out = []
        for k, l in enumerate(self.labs):
            if (k in self.test_traces) == train:
                continue
            lo = max(l["think_lo"], T - 1)
            out.extend((k, p) for p in range(lo, l["n_tokens"]))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ws_out")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    build_cache(args.out)          # reuses sprint-1 cache builder (distill)
    bt = BTW(args.out)

    # ---- arm 1: raw window-mean probes across T (dictionary-free) ----
    path = os.path.join(args.out, "raw_mean_scan.json")
    if not os.path.exists(path):
        res = {}
        for T in TS_RAW:
            tr_p, ytr = bt.probe_pairs(True, T)
            te_p, yte = bt.probe_pairs(False, T)
            Ftr, Fte = bt.means(tr_p, T), bt.means(te_p, T)
            r = probe_with_auc(Ftr, ytr, Fte, yte, f"T{T}")
            res[f"T{T}"] = {k.split("_", 1)[1]: v for k, v in r.items()}
            res[f"T{T}"]["n_pos_test"] = int((yte == 1).sum())
            print(f"[raw_mean T={T:3d}] auc={r[f'T{T}_auc']:.3f} "
                  f"(n+ test {int((yte==1).sum())})", flush=True)
        json.dump(res, open(path, "w"), indent=1)

    # ---- arm 2: DC-SAE on window means across T ----
    for T in TS_DC:
        for seed in args.seeds:
            tag = f"dcsae_T{T}_s{seed}"
            path = os.path.join(args.out, tag + ".json")
            if os.path.exists(path):
                continue
            t0 = time.time()
            torch.manual_seed(500 + seed)
            model = TokenSAE(D, 4096, 32).to(DEVICE)
            pool = bt.think_pairs(True, T)

            class Shim:
                def sample(self, n):
                    idx = np.random.default_rng().choice(len(pool), size=n)
                    return bt.means([pool[i] for i in idx], T), None, None
            trn = train_model(model, Shim(), steps=args.steps, batch=256,
                              lr=3e-4, pregen=16384)
            tr_p, ytr = bt.probe_pairs(True, T)
            te_p, yte = bt.probe_pairs(False, T)
            with torch.no_grad():
                Ftr = model.encode(bt.means(tr_p, T))
                Fte = model.encode(bt.means(te_p, T))
            res = {"T": T, "seed": seed, "fvu": trn.fvu}
            res.update(probe_with_auc(Ftr, ytr, Fte, yte, "code"))
            json.dump(res, open(path, "w"), indent=1)
            print(f"[{tag}] fvu={trn.fvu:.3f} auc={res['code_auc']:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # ---- arm 3: spectral + vanilla at T in {16,32}, param-matched ----
    for T in TS_SPEC:
        H = 4096 * 16 // T
        for arch in ["multiband", "txc"]:
            for seed in args.seeds:
                tag = f"{arch}_T{T}_s{seed}"
                path = os.path.join(args.out, tag + ".json")
                if os.path.exists(path):
                    continue
                t0 = time.time()
                torch.manual_seed(700 + seed)
                model = build_model(arch, D, T, H, 2 * T).to(DEVICE)
                pool = bt.think_pairs(True, T)

                class Shim:
                    def sample(self, n):
                        idx = np.random.default_rng().choice(len(pool),
                                                             size=n)
                        return bt.windows([pool[i] for i in idx], T), None, None
                trn = train_model(model, Shim(), steps=args.steps, batch=64,
                                  lr=3e-4, pregen=8192)
                tr_p, ytr = bt.probe_pairs(True, T)
                te_p, yte = bt.probe_pairs(False, T)
                with torch.no_grad():
                    def codes(pairs):
                        zs = []
                        for i in range(0, len(pairs), 512):
                            X = bt.windows(pairs[i:i + 512], T)
                            zs.append(model.window_codes(X))
                        return torch.cat(zs)
                    Ftr, Fte = codes(tr_p), codes(te_p)
                res = {"arch": arch, "T": T, "H": H, "seed": seed,
                       "fvu": trn.fvu}
                res.update(probe_with_auc(Ftr, ytr, Fte, yte, "code"))
                if arch == "multiband":
                    with torch.no_grad():
                        for b in range(4):
                            btr = torch.cat([model.branch_codes(
                                bt.windows(tr_p[i:i + 512], T))[b]
                                for i in range(0, len(tr_p), 512)])
                            bte = torch.cat([model.branch_codes(
                                bt.windows(te_p[i:i + 512], T))[b]
                                for i in range(0, len(te_p), 512)])
                            res.update(probe_with_auc(btr, ytr, bte, yte,
                                                      f"branch{b}"))
                json.dump(res, open(path, "w"), indent=1)
                print(f"[{tag}] fvu={trn.fvu:.3f} "
                      f"auc={res['code_auc']:.3f} ({time.time()-t0:.0f}s)",
                      flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
