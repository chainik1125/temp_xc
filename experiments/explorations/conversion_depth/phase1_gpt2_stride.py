"""Phase 1 — validate the g(ℓ) probe stack on GPT-2 day-stride.

Port of the FreqBench sprint's `gpt2_stride.py` construction
(`origin/dmitry-spectral-sprint2`, § 4.7), restricted to RAW-activation
ceilings (this session trains no dictionaries) and extended from the
sprint's 4 depth points to ALL 13 GPT-2 hidden states — the full
position × layer conversion map that prototypes the 8B g(ℓ) curves.

Construction: sequences of 16 weekday tokens with constant stride
y ∈ Z_7 (label, 7 classes), random start day. For each hidden state ℓ:

  (a) per-token linear ceiling  — probe on the middle position (sprint
      `raw_token` convention) + the position-resolved curve;
  (b) window linear ceiling     — probe on the flattened 16-position
      window (sprint `raw_stacked`);
  (c) MLP presence checks       — MLP(512) on both.

ACCEPTANCE (frozen from sprint § 4.7, RECORD.md § 1 — the stack is
frozen for phases 3-5 iff all pass):
  A1  hs=0 per-token linear ≈ chance      (0.10 ≤ acc ≤ 0.20; chance 1/7)
  A2  hs=0 window linear stays low        (≤ 0.30; sprint: 0.181)
  A3  hs=0 window MLP presence            (≥ 0.99; sprint: 1.000)
  A4  hs=1 per-token linear ≥ 0.95 at every position except 0
  A5  hs=3 per-token linear ≥ 0.99 at every position except 0
  A6  position-0 per-position linear ≈ chance at ALL depths (≤ 0.25)
      — the built-in causal control (position 0 has no context).

Run:  .venv/bin/python -m experiments.explorations.conversion_depth.phase1_gpt2_stride
Deterministic (seed 0). Writes results/phase1_gpt2_stride.json + figs/.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.explorations.conversion_depth.problib import (
    DEVICE, fit_probe,
)

DAYS = [" Monday", " Tuesday", " Wednesday", " Thursday", " Friday",
        " Saturday", " Sunday"]
M, W = 7, 16
HSLIST = list(range(13))          # 0 = embeddings, k = after block k
N_TR, N_TE = 20_000, 5_000
SEED = 0

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "results" / "phase1_gpt2_stride.json"
FIG_DIR = HERE / "figs"

ACCEPT = {
    "A1_hs0_per_token_chance": (0.10, 0.20),
    "A2_hs0_window_linear_max": 0.30,
    "A3_hs0_window_mlp_min": 0.99,
    "A4_hs1_per_pos_min": 0.95,
    "A5_hs3_per_pos_min": 0.99,
    "A6_pos0_max": 0.25,
}


def gen_sequences(n, rng):
    y = rng.integers(0, 7, size=n)
    B = rng.integers(0, 7, size=n)
    t = np.arange(W)
    Q = (B[:, None] + y[:, None] * t[None, :]) % 7
    return Q, y


@torch.no_grad()
def extract(model, tok, Q, layers, batch=512):
    day_ids = [tok.encode(d)[0] for d in DAYS]
    assert all(len(tok.encode(d)) == 1 for d in DAYS), "day not single token"
    ids = torch.tensor(np.array(day_ids)[Q])
    outs = {L: [] for L in layers}
    for i in range(0, ids.shape[0], batch):
        chunk = ids[i:i + batch].to(DEVICE)
        res = model(chunk, output_hidden_states=True)
        for L in layers:
            outs[L].append(res.hidden_states[L].detach()
                           .to(torch.float16).cpu())
    return {L: torch.cat(v) for L, v in outs.items()}


def main():
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    gpt = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE).eval()

    rng = np.random.default_rng(SEED)
    n_total = N_TR + N_TE
    Q, y = gen_sequences(n_total, rng)
    print("extracting activations (13 hidden states)...", flush=True)
    t0 = time.time()
    acts = extract(gpt, tok, Q, HSLIST)
    print(f"extracted in {time.time() - t0:.0f}s", flush=True)
    del gpt
    torch.cuda.empty_cache()

    y_t = torch.tensor(y)
    sl_tr, sl_te = slice(0, N_TR), slice(N_TR, n_total)
    ytr, yte = y_t[sl_tr], y_t[sl_te]

    layers_out = {}
    for L in HSLIST:
        X = acts[L]
        Xtr, Xte = X[sl_tr], X[sl_te]
        mid = W // 2
        res = {"hs_index": L}
        # (a) per-token linear: position-resolved + mid convention
        pos_acc = []
        for t in range(W):
            r = fit_probe(Xtr[:, t, :], ytr, Xte[:, t, :], yte, 7)
            pos_acc.append(r["acc_test"])
        res["raw_pos_linear"] = pos_acc
        res["per_token_linear"] = pos_acc[mid]
        # (b) window linear
        r = fit_probe(Xtr.flatten(1), ytr, Xte.flatten(1), yte, 7)
        res["window_linear"] = r["acc_test"]
        # (c) presence checks
        r = fit_probe(Xtr[:, mid, :], ytr, Xte[:, mid, :], yte, 7,
                      hidden=512)
        res["per_token_mlp"] = r["acc_test"]
        r = fit_probe(Xtr.flatten(1), ytr, Xte.flatten(1), yte, 7,
                      hidden=512)
        res["window_mlp"] = r["acc_test"]
        res["gap_linear"] = res["window_linear"] - res["per_token_linear"]
        res["gap_presence"] = res["window_mlp"] - res["per_token_linear"]
        layers_out[L] = res
        print(f"[hs{L:>2}] tok_lin={res['per_token_linear']:.3f} "
              f"win_lin={res['window_linear']:.3f} "
              f"win_mlp={res['window_mlp']:.3f} pos0={pos_acc[0]:.3f}",
              flush=True)

    # acceptance
    a = {}
    lo, hi = ACCEPT["A1_hs0_per_token_chance"]
    a["A1"] = bool(lo <= layers_out[0]["per_token_linear"] <= hi)
    a["A2"] = bool(layers_out[0]["window_linear"]
                   <= ACCEPT["A2_hs0_window_linear_max"])
    a["A3"] = bool(layers_out[0]["window_mlp"]
                   >= ACCEPT["A3_hs0_window_mlp_min"])
    a["A4"] = bool(min(layers_out[1]["raw_pos_linear"][1:])
                   >= ACCEPT["A4_hs1_per_pos_min"])
    a["A5"] = bool(min(layers_out[3]["raw_pos_linear"][1:])
                   >= ACCEPT["A5_hs3_per_pos_min"])
    a["A6"] = bool(max(layers_out[L]["raw_pos_linear"][0] for L in HSLIST)
                   <= ACCEPT["A6_pos0_max"])
    a["ALL_PASS"] = all(a.values())

    out = {"meta": {"seed": SEED, "n_tr": N_TR, "n_te": N_TE, "W": W,
                    "chance": 1.0 / 7, "acceptance_spec": ACCEPT},
           "layers": layers_out, "acceptance": a}
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print("\nACCEPTANCE:", json.dumps(a, indent=1))
    _plot(layers_out)
    print(f"-> {OUT_JSON}")


def _plot(layers_out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ls = sorted(layers_out)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(Ls, [layers_out[L]["per_token_linear"] for L in Ls], "o-",
               color="#d62728", label="per-token linear (mid pos)")
    ax[0].plot(Ls, [layers_out[L]["window_linear"] for L in Ls], "s-",
               color="#1f77b4", label="window linear (stacked)")
    ax[0].plot(Ls, [layers_out[L]["window_mlp"] for L in Ls], "^-",
               color="#2ca02c", label="window MLP (presence)")
    ax[0].plot(Ls, [layers_out[L]["raw_pos_linear"][0] for L in Ls], "x--",
               color="gray", label="position 0 (causal control)")
    ax[0].axhline(1 / 7, color="k", lw=0.6, alpha=0.4)
    ax[0].set_xlabel("GPT-2 hidden state ℓ")
    ax[0].set_ylabel("stride accuracy")
    ax[0].set_title("Ceilings vs depth (day-stride)")
    ax[0].legend(fontsize=8)
    ax[0].grid(True, alpha=0.25)
    im = np.array([layers_out[L]["raw_pos_linear"] for L in Ls])
    pc = ax[1].imshow(im, aspect="auto", vmin=1 / 7, vmax=1.0,
                      cmap="viridis")
    ax[1].set_xlabel("position")
    ax[1].set_ylabel("hidden state ℓ")
    ax[1].set_title("position × layer conversion map")
    fig.colorbar(pc, ax=ax[1])
    fig.suptitle("Phase 1 — GPT-2 day-stride: probe-stack validation")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIG_DIR.mkdir(exist_ok=True)
    for ext, dpi in [("pdf", None), ("png", 120)]:
        fig.savefig(FIG_DIR / f"phase1_gpt2_stride.{ext}", dpi=dpi,
                    bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
