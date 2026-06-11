"""run_multilane.py — superposition variant: 3 simultaneous circle tones.

Process: 3 independent lanes; lane k lives in its own 2D plane (planes
mutually orthogonal, embedded in R^d by a random isometry). Each lane carries
a tone with its own velocity y_k ~ Unif(Omega') and phase B_k ~ Unif(Z_M):

    x_t = sum_k [cos(theta_k(t)) e_k1 + sin(theta_k(t)) e_k2] + sigma eps_t,
    theta_k(t) = 2*pi*(B_k + y_k t)/M.

Labels: the velocity of each lane (three 10-class targets). Probes are fit
per lane on the same codes; we report mean per-lane accuracy and per-lane
per-class accuracy. Oracle: per-lane periodogram using the true planes.

Template count is |Omega|^3 * M^3 ~ 1e9: memorization is impossible by
construction at any feasible dictionary size.

Usage: python run_multilane.py --out full_ml --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from fb_core import DEVICE, build_model, fit_probe, train_model

M = 101
OMEGAS = (0, 1, 2, 4, 8, 16, 24, 32, 40, 50)
NLANES = 3
D = 24
W = 16
SIGMA = 0.25
K_WIN = 64          # 4 active atoms per token: 3 lanes + slack
N_TR, N_TE = 30_000, 6_000


class MultiLaneData:
    def __init__(self, seed: int):
        rng = np.random.default_rng(1000 + seed)
        A = rng.standard_normal((D, 2 * NLANES))
        Q, _ = np.linalg.qr(A)
        self.planes = torch.tensor(Q, dtype=torch.float32,
                                   device=DEVICE)  # (D, 6) orthonormal
        self.rng = np.random.default_rng(seed)

    def sample(self, n: int):
        y = self.rng.integers(0, len(OMEGAS), size=(n, NLANES))
        vel = np.array(OMEGAS)[y]                       # (n, L)
        B = self.rng.integers(0, M, size=(n, NLANES))
        t = np.arange(W)
        theta = 2 * np.pi * ((B[:, :, None] + vel[:, :, None] * t) % M) / M
        cs = np.stack([np.cos(theta), np.sin(theta)], -1)  # (n, L, W, 2)
        cs = torch.tensor(cs, dtype=torch.float32, device=DEVICE)
        x = torch.zeros(n, W, D, device=DEVICE)
        for k in range(NLANES):
            x += torch.einsum("nwc,dc->nwd", cs[:, k],
                              self.planes[:, 2 * k:2 * k + 2])
        x += SIGMA * torch.randn_like(x)
        return x, torch.tensor(y, device=DEVICE)

    def oracle(self, X, y):
        t = torch.arange(W, device=DEVICE, dtype=torch.float32)
        vel = torch.tensor(OMEGAS, device=DEVICE, dtype=torch.float32)
        phase = torch.exp(-2j * np.pi * vel[:, None] * t[None, :] / M)
        accs, per_class = [], []
        for k in range(NLANES):
            proj = X @ self.planes[:, 2 * k:2 * k + 2]
            c = torch.complex(proj[..., 0], proj[..., 1])
            scores = torch.einsum("nw,vw->nv", c, phase).abs()
            pred = scores.argmax(-1)
            accs.append((pred == y[:, k]).float().mean().item())
            pc = []
            for cl in range(len(OMEGAS)):
                m = y[:, k] == cl
                pc.append((pred[m] == cl).float().mean().item())
            per_class.append(pc)
        return float(np.mean(accs)), np.mean(per_class, axis=0).tolist()


def encode_chunked(model, X, chunk=2048):
    outs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], chunk):
            outs.append(model.window_codes(X[i:i + chunk]))
    return torch.cat(outs)


def run_cell(arch, H, seed, out_dir, steps):
    t0 = time.time()
    data = MultiLaneData(seed)
    torch.manual_seed(50_000 + seed)
    model = build_model(arch, D, W, H, K_WIN).to(DEVICE)

    class Shim:  # adapt MultiLaneData to train_model's TaskData interface
        def sample(self, n):
            x, y = data.sample(n)
            return x, y, None
    tr = train_model(model, Shim(), steps=steps)
    Xtr, ytr = data.sample(N_TR)
    Xte, yte = data.sample(N_TE)
    Ftr, Fte = encode_chunked(model, Xtr), encode_chunked(model, Xte)
    res = {"task": "multilane_circle", "arch": arch, "H": H, "seed": seed,
           "fvu": tr.fvu, "l0": tr.l0, "k_win": K_WIN, "n_lanes": NLANES}
    lin_accs, lin_pc, mlp_accs = [], [], []
    for k in range(NLANES):
        r = fit_probe(Ftr, ytr[:, k], Fte, yte[:, k], len(OMEGAS))
        lin_accs.append(r["acc_test"])
        lin_pc.append(r["per_class"])
        r2 = fit_probe(Ftr, ytr[:, k], Fte, yte[:, k], len(OMEGAS),
                       hidden=512)
        mlp_accs.append(r2["acc_test"])
    res["lin_mean"] = float(np.mean(lin_accs))
    res["lin_per_lane"] = lin_accs
    res["lin_per_class_mean"] = np.mean(lin_pc, axis=0).tolist()
    res["mlp_mean"] = float(np.mean(mlp_accs))
    o, opc = data.oracle(Xte, yte)
    res["oracle_mean"] = o
    res["oracle_per_class"] = opc
    tag = f"multilane_{arch}_H{H}_s{seed}"
    with open(os.path.join(out_dir, f"{tag}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"[ml] {tag}: fvu={tr.fvu:.3f} lin={res['lin_mean']:.3f} "
          f"mlp={res['mlp_mean']:.3f} oracle={o:.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="full_ml")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--H", nargs="+", type=int, default=[256])
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--archs", nargs="+",
                    default=["token_sae", "txc", "dcac", "multiband", "conv"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for H in args.H:
        for arch in args.archs:
            for seed in args.seeds:
                tag = f"multilane_{arch}_H{H}_s{seed}"
                if os.path.exists(os.path.join(args.out, f"{tag}.json")):
                    continue
                run_cell(arch, H, seed, args.out, args.steps)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
