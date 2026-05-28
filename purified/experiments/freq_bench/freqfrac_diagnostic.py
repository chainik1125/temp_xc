"""FreqFrac of W_enc as W grows — weight-space diagnostic for txc_base T=W.

Question (Dmitry's puzzle): why does the joint-window txc_base degrade as W
grows? Two candidate answers:

  - REPRESENTATION failure: the joint atoms collapse toward DC as W grows
    (W_enc becomes constant across T, atoms can only encode "average") →
    FreqFrac should drop toward 0.
  - READOUT failure: the atoms still encode AC content, but a fixed-budget
    joint topk discards the per-position structure the linear probe needs.
    → FreqFrac stays high; only the probe-side NTPS drops.

This script loads each trained txc_base_TW checkpoint and computes the
FreqFrac of W_enc (shape (T, d_in, d_sae)) cell by cell, then plots
FreqFrac vs W faceted by d_sae alongside the matching NTPS curve.

Inputs are taken from results/leaderboard.jsonl (train_key + eval_cfg).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors.torch import load_file

from temp_bench.core.config import checkpoint_dir, import_by_path, load_arch

ROOT = Path(__file__).resolve().parents[2]
LEADERBOARD = ROOT / "results" / "leaderboard.jsonl"
OUT = ROOT / "results" / "freq_bench" / "v2_sweep"
OUT.mkdir(parents=True, exist_ok=True)
PROTO = "1.2.0"


def _freqfrac_full(W_enc: torch.Tensor) -> dict[str, float]:
    """FreqFrac per-atom-mean stats from (T, d_in, d_sae) encoder weights."""
    w = W_enc.detach().float().cpu()
    T = w.shape[0]
    if T < 2:
        return {}
    spec = torch.fft.rfft(w, dim=0).abs() ** 2
    total = spec.sum(dim=0)                       # (d_in, d_sae)
    ac = spec[1:].sum(dim=0)
    frac = (ac / total.clamp(min=1e-12))          # (d_in, d_sae)
    return {
        "mean": float(frac.mean()),
        "per_atom_mean": float(frac.mean(dim=0).mean()),
        # weight per atom by its overall encoder norm (used-ness proxy)
        "norm_weighted": float(
            (frac.mean(dim=0) * w.norm(dim=(0, 1))).sum()
            / w.norm(dim=(0, 1)).sum().clamp(min=1e-12)
        ),
    }


def load_tw_rows() -> list[dict]:
    """All txc_base_TW v2 rows from the leaderboard."""
    rows = []
    for line in open(LEADERBOARD):
        r = json.loads(line)
        if r.get("experiment") != "freq_bench":
            continue
        if r.get("evaluator_protocol_version") != PROTO:
            continue
        ec = r.get("eval_cfg", {})
        if ec.get("smoke") or ec.get("label") != "txc_base_TW":
            continue
        rows.append({"train_key": r["train_key"], "W": ec["W"],
                     "d_sae": ec["d_sae"], "k_pos": ec["k_pos"],
                     **r["metrics"]})
    return rows


def load_W_enc(train_key: str, *, W: int, d_sae: int, k_pos: int) -> torch.Tensor:
    """Re-instantiate txc_base with the cell's hparams + load weights."""
    spec = load_arch("txc_base")
    cls = import_by_path(spec.class_path)
    model = cls(d_in=256, d_sae=d_sae, T=W, k_pos=k_pos)
    path = checkpoint_dir(train_key) / "model.safetensors"
    state = load_file(str(path))
    model.load_state_dict(state)
    return model.W_enc


def plot_freqfrac_vs_W(rows_with_ff: list[dict]):
    dsaes = sorted({r["d_sae"] for r in rows_with_ff})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    colors = {40: "#1f77b4", 256: "#9467bd", 1024: "#2ca02c"}

    # left: FreqFrac vs W
    ax = axes[0]
    for d in dsaes:
        xs, ys = [], []
        for W in sorted({r["W"] for r in rows_with_ff if r["d_sae"] == d}):
            r = next(r for r in rows_with_ff if r["W"] == W and r["d_sae"] == d)
            xs.append(W); ys.append(r["freqfrac_recomputed"])
        ax.plot(xs, ys, "o-", color=colors[d], lw=2,
                label=f"d_sae={d}")
    ax.axhline(0.5, color="k", ls=":", lw=1, alpha=.5)
    ax.set_xlabel("W"); ax.set_ylabel("FreqFrac of W_enc (mean over atoms)")
    ax.set_xscale("log", base=2)
    ax.set_title("Weight-space order-sensitivity vs W\n(txc_base T=W, raw_k=1)")
    ax.legend(fontsize=9); ax.grid(alpha=.3)

    # right: NTPS vs W
    ax = axes[1]
    for d in dsaes:
        xs, ys = [], []
        for W in sorted({r["W"] for r in rows_with_ff if r["d_sae"] == d}):
            r = next(r for r in rows_with_ff if r["W"] == W and r["d_sae"] == d)
            xs.append(W); ys.append(r["NTPS"])
        ax.plot(xs, ys, "o-", color=colors[d], lw=2, label=f"d_sae={d}")
    ax.axhline(0, color="k", ls="--", lw=1, alpha=.4)
    ax.set_xlabel("W"); ax.set_ylabel("NTPS (linear probe)")
    ax.set_xscale("log", base=2)
    ax.set_title("Linear-probe NTPS vs W\n(same cells)")
    ax.legend(fontsize=9); ax.grid(alpha=.3)

    plt.suptitle("txc_base T=W: representation vs readout decomposition", fontsize=12)
    plt.tight_layout()
    p = OUT / "freqfrac_vs_W_TW.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("saved", p)


if __name__ == "__main__":
    rows = load_tw_rows()
    print(f"found {len(rows)} txc_base_TW rows")
    out = []
    for r in rows:
        try:
            W_enc = load_W_enc(r["train_key"], W=r["W"], d_sae=r["d_sae"],
                               k_pos=r["k_pos"])
            ff = _freqfrac_full(W_enc)
            r2 = dict(r); r2["freqfrac_recomputed"] = ff["mean"]
            r2["freqfrac_norm_weighted"] = ff["norm_weighted"]
            out.append(r2)
            print(f"  W={r['W']:2d} d_sae={r['d_sae']:4d}: "
                  f"FreqFrac={ff['mean']:.3f} (norm-weighted={ff['norm_weighted']:.3f}) "
                  f"NTPS={r['NTPS']:+.3f}")
        except FileNotFoundError as e:
            print(f"  W={r['W']} d_sae={r['d_sae']}: checkpoint missing ({e})")

    if out:
        plot_freqfrac_vs_W(out)
        # dump
        json.dump(out, open(OUT / "freqfrac_tw.json", "w"), indent=2)
        print("saved", OUT / "freqfrac_tw.json")
