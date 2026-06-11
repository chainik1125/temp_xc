"""bt_controls.py — red-team-demanded controls for the timescale curve.

C1 scrambled-window: raw probe on the mean of T NON-CONTIGUOUS tokens
   (sampled uniformly from [think_lo, pos]) vs the contiguous window mean.
   If scrambled ~= contiguous at every T, the T-curve measures pooling
   (denoising), not temporal structure.
C2 fixed-eval-set T-curve: restrict to positives/negatives with pos >= 95
   (valid at every T in the grid) so the example set is IDENTICAL across T.
   Tests whether the T=96 collapse is set-composition.
C3 position-only ceiling: probe on [relative_position, absolute_position]
   alone — label leakage via trace position.

Run on the W-scan pod after ws_out exists: python bt_controls.py
"""
from __future__ import annotations

import json

import numpy as np
import torch

from fb_core import DEVICE
from bt_freq import probe_with_auc
from bt_wscan import BTW

TS = [4, 8, 16, 32, 48, 64, 96]


def main():
    bt = BTW("ws_out")
    res = {}

    # ---- C2 fixed eval set (pos >= 95) ----
    def fixed_pairs(train):
        import random as _r
        pos, neg = [], []
        rng = np.random.default_rng(43)
        for k, l in enumerate(bt.labs):
            if (k in bt.test_traces) == train:
                continue
            ev = np.array(l["event_positions"], dtype=int)
            pos.extend((k, p) for p in l["d_plus_positions"] if p >= 95)
            cand = [p for p in range(max(l["think_lo"], 95), l["n_tokens"])
                    if not len(ev) or np.abs(ev - p).min() > 25]
            take = (rng.choice(len(cand), size=min(len(cand), 60),
                               replace=False) if cand else [])
            neg.extend((k, cand[i]) for i in take)
        _r.Random(17).shuffle(neg)
        neg = neg[:5 * max(1, len(pos))]
        y = torch.tensor([1] * len(pos) + [0] * len(neg), device=DEVICE)
        return pos + neg, y

    tr_p, ytr = fixed_pairs(True)
    te_p, yte = fixed_pairs(False)
    res["fixed_set_n_pos_test"] = int((yte == 1).sum())
    for T in TS:
        Ftr, Fte = bt.means(tr_p, T), bt.means(te_p, T)
        r = probe_with_auc(Ftr, ytr, Fte, yte, "x")
        res[f"fixed_rawmean_T{T}"] = r["x_auc"]
        print(f"[C2 fixed-set T={T:3d}] auc={r['x_auc']:.3f}", flush=True)

    # ---- C1 scrambled windows (same fixed set for comparability) ----
    def scrambled_means(pairs, T, seed):
        rng = np.random.default_rng(seed)
        outs = []
        for i in range(0, len(pairs), 2048):
            chunk = pairs[i:i + 2048]
            idxs = []
            for k, p in chunk:
                lo = bt.labs[k]["think_lo"]
                lo = min(lo, p - 1) if p > lo else max(0, p - T)
                cand = np.arange(min(lo, p), p + 1)
                sel = rng.choice(cand, size=min(T, len(cand)), replace=False)
                idxs.append(bt.off[k] + sel)
            X = torch.stack([bt.Ad[torch.tensor(ix, device=DEVICE)].float()
                             .mean(0) for ix in idxs])
            outs.append((X - bt.mu) / bt.sd)
        return torch.cat(outs)

    for T in TS:
        Ftr = scrambled_means(tr_p, T, 100 + T)
        Fte = scrambled_means(te_p, T, 200 + T)
        r = probe_with_auc(Ftr, ytr, Fte, yte, "x")
        res[f"scrambled_rawmean_T{T}"] = r["x_auc"]
        print(f"[C1 scrambled T={T:3d}] auc={r['x_auc']:.3f}", flush=True)

    # ---- C3 position-only ceiling ----
    def pos_feats(pairs):
        F = []
        for k, p in pairs:
            n = bt.labs[k]["n_tokens"]
            lo = bt.labs[k]["think_lo"]
            F.append([p / n, (p - lo) / max(n - lo, 1), p / 600.0])
        return torch.tensor(F, device=DEVICE, dtype=torch.float32)

    r = probe_with_auc(pos_feats(tr_p), ytr, pos_feats(te_p), yte, "x")
    res["position_only_auc"] = r["x_auc"]
    print(f"[C3 position-only] auc={r['x_auc']:.3f}", flush=True)

    json.dump(res, open("ws_out/controls.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
