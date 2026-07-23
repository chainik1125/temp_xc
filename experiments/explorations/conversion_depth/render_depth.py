"""Phase 3/4 — render g(ℓ) curves + extract the frozen verdicts.

Reads results/depth_probe_{base,distill}.json (+ phase4_em_depth.json if
present), renders:
  figs/depth_g_curves.*      — the three ceilings vs depth, both models,
                               per target (T1/T2/T3)
  figs/depth_gap.*           — g(ℓ) both models + generator−reader gap
  figs/em_depth.*            — the EM g(ℓ) curve (phase 4)
and prints/writes the frozen § 2 verdict quantities:
  - σ_null (pooled permutation-null spread), 3 σ_null threshold
  - P2: g(L10) vs max_ℓ g(ℓ) ratio + the frozen classification
  - P3: earliest-clear layer + max window AUC per model
  - falsifier check: min g(ℓ)

Run:  .venv/bin/python -m experiments.explorations.conversion_depth.render_depth
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIGS = HERE / "figs"

TARGETS = ["ant_kw", "ant_bts", "is_bt"]
MODELS = ["base", "distill"]
COLORS = {"base": "#1f77b4", "distill": "#d62728"}
L10_HS = 11                      # resid_post L10 = hidden_states[11]


def hs_layers(cells, target):
    ks = sorted({int(k.split("/")[0][2:]) for k in cells
                 if k.endswith("/" + target)})
    return ks


def series(cells, target, probe, field="auc"):
    ks = hs_layers(cells, target)
    return ks, [cells[f"hs{k}/{target}"][probe][field] for k in ks]


def main():
    data = {}
    for m in MODELS:
        p = RES / f"depth_probe_{m}.json"
        if p.exists():
            data[m] = json.loads(p.read_text())["cells"]
    if not data:
        print("no depth_probe results yet")
        return

    # pooled null spread
    nulls = []
    for m in data:
        for k, c in data[m].items():
            for nn in ["null_window_linear", "null_per_token_linear"]:
                nulls.append(abs(c[nn]["auc"] - 0.5))
    sigma_null = float(np.std([n for n in nulls]))
    mean_null = float(np.mean(nulls))
    thr = 3 * sigma_null
    verdict = {"sigma_null": sigma_null, "mean_abs_null_dev": mean_null,
               "threshold_3sigma": thr, "n_null_cells": len(nulls)}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── fig 1: ceilings vs depth, per target ─────────────────────────
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(15, 4.4),
                             sharey=True)
    for ax, tgt in zip(axes, TARGETS):
        for m in data:
            ks, tokv = series(data[m], tgt, "per_token_linear")
            _, winv = series(data[m], tgt, "window_linear")
            _, mlpv = series(data[m], tgt, "window_mlp")
            ax.plot(ks, tokv, "o-", color=COLORS[m], alpha=0.55,
                    label=f"{m} per-token lin")
            ax.plot(ks, winv, "s-", color=COLORS[m],
                    label=f"{m} window lin (T=16)")
            ax.plot(ks, mlpv, "^:", color=COLORS[m], alpha=0.35,
                    label=f"{m} window MLP")
        ax.axvline(L10_HS, color="k", ls="--", lw=1, alpha=0.6)
        ax.text(L10_HS + 0.2, 0.52, "L10", fontsize=8)
        ax.axhline(0.5, color="k", lw=0.5, alpha=0.4)
        ax.set_title(tgt)
        ax.set_xlabel("hidden state (hs0=emb; hs k+1 = resid_post L_k)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("test AUC")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Conversion depth — raw-activation ceilings vs depth "
                 "(Ward stream, base reader vs R1-distill generator)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    FIGS.mkdir(exist_ok=True)
    for ext, dpi in [("pdf", None), ("png", 130)]:
        fig.savefig(FIGS / f"depth_g_curves.{ext}", dpi=dpi,
                    bbox_inches="tight")
    plt.close(fig)

    # ── fig 2: g(ℓ) + generator−reader gap ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for tgt, ls in zip(TARGETS, ["-", "--", ":"]):
        for m in data:
            ks, tokv = series(data[m], tgt, "per_token_linear")
            _, winv = series(data[m], tgt, "window_linear")
            g = np.array(winv) - np.array(tokv)
            axes[0].plot(ks, g, ls, marker="o", ms=3, color=COLORS[m],
                         alpha=0.8 if tgt == "ant_kw" else 0.4,
                         label=f"{m} {tgt}" if tgt != "is_bt" else None)
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].axhspan(-thr, thr, color="gray", alpha=0.15,
                    label="±3σ null")
    axes[0].axvline(L10_HS, color="k", ls="--", lw=1, alpha=0.6)
    axes[0].set_title("ambience gap g(ℓ) = window − per-token (AUC)")
    axes[0].set_xlabel("hidden state")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.25)
    if len(data) == 2:
        for tgt, ls in zip(TARGETS, ["-", "--", ":"]):
            kb, wb = series(data["base"], tgt, "window_linear")
            kd, wd = series(data["distill"], tgt, "window_linear")
            common = sorted(set(kb) & set(kd))
            gap = [wd[kd.index(k)] - wb[kb.index(k)] for k in common]
            axes[1].plot(common, gap, ls, marker="o", ms=3,
                         color="#2ca02c",
                         alpha=0.9 if tgt == "ant_kw" else 0.4,
                         label=f"{tgt} (window)")
        kb, tb = series(data["base"], "ant_kw", "per_token_linear")
        kd, td = series(data["distill"], "ant_kw", "per_token_linear")
        common = sorted(set(kb) & set(kd))
        axes[1].plot(common,
                     [td[kd.index(k)] - tb[kb.index(k)] for k in common],
                     "-", marker="x", ms=4, color="#9467bd",
                     label="ant_kw (per-token)")
        axes[1].axhline(0, color="k", lw=0.5)
        axes[1].axhspan(-thr, thr, color="gray", alpha=0.15)
        axes[1].axvline(L10_HS, color="k", ls="--", lw=1, alpha=0.6)
        axes[1].set_title("generator − reader (distill − base) AUC")
        axes[1].set_xlabel("hidden state")
        axes[1].legend(fontsize=7)
        axes[1].grid(True, alpha=0.25)
    fig.suptitle("Conversion depth — the ambience gap and the "
                 "generator−reader gap")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for ext, dpi in [("pdf", None), ("png", 130)]:
        fig.savefig(FIGS / f"depth_gap.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # ── frozen verdict quantities ─────────────────────────────────────
    for m in data:
        ks, tokv = series(data[m], "ant_kw", "per_token_linear")
        _, winv = series(data[m], "ant_kw", "window_linear")
        g = np.array(winv) - np.array(tokv)
        gmax, kmax = float(g.max()), int(np.array(ks)[g.argmax()])
        gL10 = float(g[ks.index(L10_HS)]) if L10_HS in ks else None
        ratio = gL10 / gmax if (gL10 is not None and gmax > 0) else None
        cls = None
        if ratio is not None:
            cls = ("near-max" if ratio >= 0.8 else
                   "partial" if ratio >= 0.3 else "mis-placed")
        earliest = next((k for k, w in zip(ks, winv)
                         if w - 0.5 > thr), None)
        verdict[m] = {
            "ant_kw": {
                "g_max": gmax, "hs_argmax": kmax, "g_L10": gL10,
                "g_L10_over_max": ratio, "P2_class": cls,
                "g_min": float(g.min()),
                "falsifier_triggered": bool(g.min() < -thr),
                "earliest_window_clear_hs": earliest,
                "max_window_auc": float(np.max(winv)),
                "window_auc_L10": (float(winv[ks.index(L10_HS)])
                                   if L10_HS in ks else None),
                "per_token_auc_L10": (float(tokv[ks.index(L10_HS)])
                                      if L10_HS in ks else None),
            }}

    # ── phase 4 fig ───────────────────────────────────────────────────
    p4 = RES / "phase4_em_depth.json"
    if p4.exists():
        em = json.loads(p4.read_text())["cells"]
        ks = sorted(int(k[2:]) for k in em)
        tokv = [em[f"hs{k}"]["per_token_linear"]["auc"] for k in ks]
        winv = [em[f"hs{k}"]["window_linear"]["auc"] for k in ks]
        mlpv = [em[f"hs{k}"]["window_mlp"]["auc"] for k in ks]
        g = np.array(winv) - np.array(tokv)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        ax[0].plot(ks, tokv, "o-", color="#d62728", label="per-token lin")
        ax[0].plot(ks, winv, "s-", color="#1f77b4", label="window lin")
        ax[0].plot(ks, mlpv, "^:", color="#2ca02c", label="window MLP")
        ax[0].axvline(16, color="k", ls="--", lw=1, alpha=0.6)
        ax[0].text(16.2, 0.52, "L15", fontsize=8)
        ax[0].axhline(0.5, color="k", lw=0.5)
        ax[0].set_title("EM label ceilings vs depth (medical organism)")
        ax[0].set_xlabel("hidden state")
        ax[0].set_ylabel("mean fold AUC")
        ax[0].legend(fontsize=8)
        ax[0].grid(True, alpha=0.25)
        ax[1].plot(ks, g, "o-", color="#9467bd", label="g(ℓ)")
        ax[1].axhline(0, color="k", lw=0.5)
        ax[1].axhspan(-thr, thr, color="gray", alpha=0.15,
                      label="±3σ null (phase-3 pooled)")
        ax[1].axvline(16, color="k", ls="--", lw=1, alpha=0.6)
        ax[1].set_title("EM ambience gap g(ℓ)")
        ax[1].set_xlabel("hidden state")
        ax[1].legend(fontsize=8)
        ax[1].grid(True, alpha=0.25)
        fig.suptitle("Phase 4 — EM depth-confound check")
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        for ext, dpi in [("pdf", None), ("png", 130)]:
            fig.savefig(FIGS / f"em_depth.{ext}", dpi=dpi,
                        bbox_inches="tight")
        plt.close(fig)
        verdict["em"] = {
            "g_max_abs": float(np.abs(g).max()),
            "hs_argmax_abs": int(np.array(ks)[np.abs(g).argmax()]),
            "P5_flat_within_3sigma": bool(np.abs(g).max() <= thr),
            "max_window_auc": float(np.max(winv)),
            "max_per_token_auc": float(np.max(tokv)),
        }

    (RES / "depth_verdicts.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))
    print(f"-> {RES / 'depth_verdicts.json'} + {FIGS}/depth_*.png")


if __name__ == "__main__":
    main()
