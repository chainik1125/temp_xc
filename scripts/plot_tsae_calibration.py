"""The tSAE's published sparsity coefficient is five orders of magnitude off for these
activations, and even when corrected L1 buys sparsity by shrinking rather than by selecting.

Left panel: realised coefficients per segment against the L1 coefficient. The repo's
documented `l1_coef = 1e-3` is marked; the target band (1-32 coefficients per segment, where
a TopK SAE and a crosscoder are actually being compared) is shaded. The flat stretch at the
left is the failure that made the tSAE arm unusable last sprint -- the penalty is numerically
absent there, contributing ~0.2% of the loss.

Right panel: the reconstruction cost of that sparsity, against the per-token TopK SAE on the
same activations, plotted on the only axis on which an L1 arm and a TopK arm are comparable
at all -- realised coefficients per segment.

Reads results/dict_bench/tsae_calibration.json and, if present,
results/dict_bench/tsae_calibration_topk.json.
"""
import json
import pathlib

import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "dict_bench" / "tsae_calibration.json"
SRC_TOPK = ROOT / "results" / "dict_bench" / "tsae_calibration_topk.json"
OUT = ROOT / "plots" / "2026-07-26_txcwins" / "tsae_calibration.png"

C_SAE = "#0072B2"     # blue           -- per-token TopK SAE
C_L1 = "#CC79A7"      # reddish purple -- attention tSAE, ReLU + L1 (the repo's tsae_paper)
C_TOPK = "#009E73"    # bluish green   -- attention tSAE, TopK
C_BAND = "#E69F00"    # orange         -- the usable sparsity band


def main() -> int:
    if not SRC.exists():
        print(f"[skip] {SRC} not written yet")
        return 1
    r = json.loads(SRC.read_text())
    curve = sorted(r["tsae_curve"], key=lambda x: x["l1_coef"])
    sae = sorted(r["sae_reference"], key=lambda x: x["coeff_per_segment"])
    topk = r.get("tsae_topk_reference") or []
    if not topk and SRC_TOPK.exists():
        topk = json.loads(SRC_TOPK.read_text()).get("tsae_topk_reference") or []
    topk = sorted(topk, key=lambda x: x["coeff_per_segment"])

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4))

    # ---------------- left: where the sparsity coefficient actually bites ------------
    ax = axes[0]
    lo, hi = 1, 32
    ax.axhspan(lo, hi, color=C_BAND, alpha=0.16, zorder=0)
    ax.text(curve[0]["l1_coef"], hi * 1.25, "band where the SAE and crosscoder are compared",
            fontsize=8.5, color="#8a6400")
    ax.plot([x["l1_coef"] for x in curve], [x["coeff_per_segment"] for x in curve],
            "o-", color=C_L1, lw=2.0, ms=5, label="attention tSAE (ReLU + L1)")
    ax.axvline(1e-3, ls="--", color="#555555", lw=1.4)
    ax.text(1.25e-3, max(x["coeff_per_segment"] for x in curve) * 0.55,
            "repo's documented\nl1_coef = 1e-3", fontsize=8.5, color="#555555")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("L1 coefficient")
    ax.set_ylabel("realised coefficients per segment")
    ax.set_title("The published coefficient is far outside the range that binds")
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)

    # ---------------- right: what that sparsity costs in reconstruction --------------
    ax = axes[1]
    ax.plot([x["coeff_per_segment"] for x in sae], [x["fvu"] for x in sae],
            "o-", color=C_SAE, lw=2.0, ms=5, label="per-token TopK SAE")
    ax.plot([x["coeff_per_segment"] for x in curve], [x["fvu"] for x in curve],
            "o-", color=C_L1, lw=2.0, ms=5, label="attention tSAE (ReLU + L1)")
    if topk:
        ax.plot([x["coeff_per_segment"] for x in topk], [x["fvu"] for x in topk],
                "s-", color=C_TOPK, lw=2.0, ms=5, label="attention tSAE (TopK)")
    ax.axvspan(lo, hi, color=C_BAND, alpha=0.16, zorder=0)
    # The whole result is that the L1 curve is ABOVE this line everywhere inside the band.
    ax.axhline(1.0, ls="--", color="#444444", lw=1.4)
    ax.text(1.05, 1.02, "no better than predicting the mean", fontsize=8.5,
            color="#444444", va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("realised coefficients per segment")
    ax.set_ylabel("FVU (holdout)")
    ax.set_title("Cost of the sparsity, on the only shared axis")
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    print("[saved]", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
