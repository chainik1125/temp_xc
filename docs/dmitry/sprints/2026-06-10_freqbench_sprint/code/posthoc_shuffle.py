"""posthoc_shuffle.py — recompute the shuffle diagnostic correctly
(per-window independent permutations) for all completed cells, from saved
model checkpoints. Writes results into sidecar files <tag>_shufv2.json.

Also recomputes reversal-probe accuracy (probe trained on reversed-window
codes) as a bonus order diagnostic.
"""
from __future__ import annotations

import glob
import json
import os
import re

import torch

from fb_core import (DEVICE, TaskData, build_model, fit_probe, make_task,
                     shuffle_windows)

OUT = "full"
PAT = re.compile(r"model_(\w+?)_(token_sae|txc|dcac|multiband|conv)_H(\d+)_s(\d+)\.pt")


def encode_chunked(model, X, chunk=2048):
    outs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], chunk):
            outs.append(model.window_codes(X[i:i + chunk]))
    return torch.cat(outs)


def main():
    for p in sorted(glob.glob(os.path.join(OUT, "model_*.pt"))):
        m = PAT.match(os.path.basename(p))
        if not m:
            print("skip unparsed", p)
            continue
        task, arch, H, seed = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        tag = f"{task}_{arch}_H{H}_s{seed}"
        sidecar = os.path.join(OUT, f"{tag}_shufv2.json")
        if os.path.exists(sidecar):
            continue
        spec = make_task(task)
        data = TaskData(spec, seed=seed)
        model = build_model(arch, spec.d, spec.W, H, 32).to(DEVICE)
        model.load_state_dict(torch.load(p, map_location=DEVICE))
        model.eval()
        Xtr, ytr, _ = data.sample(20000)
        Xte, yte, _ = data.sample(5000)
        res = {"tag": tag}
        Ftr = encode_chunked(model, shuffle_windows(Xtr, seed=11))
        Fte = encode_chunked(model, shuffle_windows(Xte, seed=12))
        res["shufv2_linear"] = fit_probe(Ftr, ytr, Fte, yte, spec.n_classes)
        Rtr = encode_chunked(model, Xtr.flip(1))
        Rte = encode_chunked(model, Xte.flip(1))
        res["reversed_linear"] = fit_probe(Rtr, ytr, Rte, yte, spec.n_classes)
        with open(sidecar, "w") as f:
            json.dump(res, f, indent=1)
        print(tag, "shufv2", round(res["shufv2_linear"]["acc_test"], 3),
              "rev", round(res["reversed_linear"]["acc_test"], 3), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
