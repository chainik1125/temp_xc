"""run_grid.py — FrequencyBench phase-1/2 runner.

For each (task, arch, H, seed): train dictionary, probe codes (linear + MLP),
record FVU/L0, per-class accuracy, shuffle/reversal sensitivity, kernel
frequency profiles, and (for txc) DC-band ablation. Baselines (oracle, raw-input
probes) computed once per (task, seed).

Usage:
  python run_grid.py --out results --tasks dc ac_sign multifreq \
      --archs token_sae txc dcac multiband conv --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from fb_core import (DEVICE, ConvDict, SpectralTXC, TaskData, TokenSAE,
                     build_model, fit_probe, freq_profile, make_task,
                     oracle_accuracy, project_to_band, s_temp,
                     shuffle_sensitivity, train_model)

TASK_H = {"dc": 64, "ac_sign": 128, "multifreq": 2048,
          "multifreq_circle": 2048, "phasepair": 2048}
K_WIN = 32
N_PROBE_TR = 20_000
N_PROBE_TE = 5_000
STEPS = 4000


def encode_chunked(model, X, chunk=2048):
    outs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], chunk):
            outs.append(model.window_codes(X[i:i + chunk]))
    return torch.cat(outs)


def probe_suite(feats_tr, ytr, feats_te, yte, n_classes, tag, mlp=True):
    out = {}
    out[f"{tag}_linear"] = fit_probe(feats_tr, ytr, feats_te, yte, n_classes)
    if mlp:
        out[f"{tag}_mlp"] = fit_probe(feats_tr, ytr, feats_te, yte, n_classes,
                                      hidden=512)
    return out


WINDOW = 16


def run_baselines(task: str, seed: int, out_dir: str, sigma: float):
    spec = make_task(task, W=WINDOW, sigma=sigma)
    data = TaskData(spec, seed=seed)
    Xtr, ytr, _ = data.sample(N_PROBE_TR)
    Xte, yte, _ = data.sample(N_PROBE_TE)
    res = {"task": task, "seed": seed, "sigma": sigma, "spec_W": spec.W,
           "spec_M": spec.M, "a_loc_star": spec.a_loc_star}
    res["oracle_acc"] = oracle_accuracy(spec, Xte, yte, data.U, data.R)
    # per-class oracle accuracy (for frequency-response normalization)
    if spec.name.startswith("multifreq"):
        per = []
        for c in range(spec.n_classes):
            m = yte == c
            per.append(oracle_accuracy(spec, Xte[m], yte[m], data.U, data.R))
        res["oracle_per_class"] = per
    # single-token raw probe (middle token) — empirical local ceiling check
    mid = spec.W // 2
    res.update(probe_suite(Xtr[:, mid, :], ytr, Xte[:, mid, :], yte,
                           spec.n_classes, "raw_token"))
    # mean-pooled raw window
    res.update(probe_suite(Xtr.mean(1), ytr, Xte.mean(1), yte,
                           spec.n_classes, "raw_meanpool"))
    # stacked raw window (linear + MLP: info-presence check)
    res.update(probe_suite(Xtr.flatten(1), ytr, Xte.flatten(1), yte,
                           spec.n_classes, "raw_stacked"))
    path = os.path.join(out_dir, f"{task}_baselines_s{seed}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[baselines] {task} s{seed}: oracle={res['oracle_acc']:.3f} "
          f"raw_token_lin={res['raw_token_linear']['acc_test']:.3f} "
          f"raw_stacked_lin={res['raw_stacked_linear']['acc_test']:.3f} "
          f"raw_stacked_mlp={res['raw_stacked_mlp']['acc_test']:.3f}",
          flush=True)
    return res


def run_cell(task: str, arch: str, H: int, seed: int, out_dir: str,
             sigma: float, steps: int = STEPS):
    t0 = time.time()
    spec = make_task(task, W=WINDOW, sigma=sigma)
    data = TaskData(spec, seed=seed)
    torch.manual_seed(10_000 + seed)
    model = build_model(arch, spec.d, spec.W, H, K_WIN).to(DEVICE)
    tr = train_model(model, data, steps=steps)

    Xtr, ytr, _ = data.sample(N_PROBE_TR)
    Xte, yte, _ = data.sample(N_PROBE_TE)
    Ftr, Fte = encode_chunked(model, Xtr), encode_chunked(model, Xte)
    res = {"task": task, "arch": arch, "H": H, "k_win": K_WIN, "seed": seed,
           "sigma": sigma, "fvu": tr.fvu, "l0": tr.l0,
           "train_seconds": tr.seconds}
    res.update(probe_suite(Ftr, ytr, Fte, yte, spec.n_classes, "code"))

    # mean-pooled code probe (for token_sae this is the "bag of symbols" code)
    if isinstance(model, TokenSAE):
        with torch.no_grad():
            mtr = model.encode(Xtr).mean(1)
            mte = model.encode(Xte).mean(1)
        res.update(probe_suite(mtr, ytr, mte, yte, spec.n_classes,
                               "code_meanpool", mlp=False))
        # single-token code probe (local ceiling on codes)
        with torch.no_grad():
            str_ = model.encode(Xtr[:, spec.W // 2, :])
            ste_ = model.encode(Xte[:, spec.W // 2, :])
        res.update(probe_suite(str_, ytr, ste_, yte, spec.n_classes,
                               "code_token", mlp=False))

    # per-branch probes for spectral models: which band carries which class?
    if isinstance(model, SpectralTXC) and len(model.bands) > 1:
        with torch.no_grad():
            btr = [torch.cat([model.branch_codes(Xtr[i:i + 2048])[b]
                              for i in range(0, Xtr.shape[0], 2048)])
                   for b in range(len(model.bands))]
            bte = [torch.cat([model.branch_codes(Xte[i:i + 2048])[b]
                              for i in range(0, Xte.shape[0], 2048)])
                   for b in range(len(model.bands))]
        for b in range(len(model.bands)):
            res.update(probe_suite(btr[b], ytr, bte[b], yte, spec.n_classes,
                                   f"code_branch{b}", mlp=False))

    res["sensitivity"] = shuffle_sensitivity(model, Xte, spec.n_classes, yte)

    # probe accuracy on shuffled/reversed test windows (probe trained on normal)
    # — done by re-encoding transformed windows and evaluating the SAME probe is
    # complex to wire here; instead retrain probe on shuffled codes (measures
    # what info survives shuffling, which is the quantity of interest).
    from fb_core import shuffle_windows
    Ftr_sh = encode_chunked(model, shuffle_windows(Xtr, seed=11))
    Fte_sh = encode_chunked(model, shuffle_windows(Xte, seed=12))
    res.update(probe_suite(Ftr_sh, ytr, Fte_sh, yte, spec.n_classes,
                           "code_shuffled", mlp=False))

    # DC-band ablation for spectral models on cyclic tasks
    if isinstance(model, SpectralTXC) and task != "dc":
        m_dc = project_to_band(model, [0])
        Ftr_a = encode_chunked(m_dc, Xtr)
        Fte_a = encode_chunked(m_dc, Xte)
        res.update(probe_suite(Ftr_a, ytr, Fte_a, yte, spec.n_classes,
                               "code_dcband", mlp=False))

    fp = freq_profile(model, spec.W)
    tag = f"{task}_{arch}_H{H}_s{seed}"
    if fp is not None:
        np.save(os.path.join(out_dir, f"freqprof_{tag}.npy"), fp)
    torch.save(model.state_dict(), os.path.join(out_dir, f"model_{tag}.pt"))
    with open(os.path.join(out_dir, f"{tag}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"[cell] {tag}: fvu={tr.fvu:.4f} l0={tr.l0:.1f} "
          f"lin={res['code_linear']['acc_test']:.3f} "
          f"mlp={res['code_mlp']['acc_test']:.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--tasks", nargs="+",
                    default=["dc", "ac_sign", "multifreq"])
    ap.add_argument("--archs", nargs="+",
                    default=["token_sae", "txc", "dcac", "multiband", "conv"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--H", nargs="+", type=int, default=[],
                    help="override H (else per-task default)")
    ap.add_argument("--sigma", type=float, default=0.25)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--W", type=int, default=16)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    global N_PROBE_TR, N_PROBE_TE, WINDOW, K_WIN
    WINDOW = args.W
    K_WIN = 2 * args.W
    steps = args.steps
    if args.quick:
        steps = 300
        N_PROBE_TR, N_PROBE_TE = 4000, 1000

    os.makedirs(args.out, exist_ok=True)
    print(f"device={DEVICE} tasks={args.tasks} archs={args.archs} "
          f"seeds={args.seeds} steps={steps}", flush=True)
    for task in args.tasks:
        for seed in args.seeds:
            bpath = os.path.join(args.out, f"{task}_baselines_s{seed}.json")
            if not os.path.exists(bpath):
                run_baselines(task, seed, args.out, args.sigma)
        Hs = args.H if args.H else [TASK_H[task]]
        for H in Hs:
            for arch in args.archs:
                for seed in args.seeds:
                    tag = f"{task}_{arch}_H{H}_s{seed}"
                    if os.path.exists(os.path.join(args.out, f"{tag}.json")):
                        print(f"[skip] {tag}", flush=True)
                        continue
                    run_cell(task, arch, H, seed, args.out, args.sigma,
                             steps=steps)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
