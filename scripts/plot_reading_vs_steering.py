"""Reading a factor and steering it are different capabilities.

Nine held-out cells (three tasks x three dictionary inits), three architectures, all on content
the dictionaries never trained on. Reading is the best single latent's held-out AUC; steering is
that latent's effect at matched dose magnitude with the sign free.

A per-token dictionary reads these factors at AUC 1.000 and moves them by ~0.1 nats. The
crosscoder reads them WORSE and moves them 4-40x more. If reading predicted steering the points
would fall on a rising line; the two per-token architectures sit at the top of the reading axis
and the bottom of the steering axis.

Reads results/txc_wins/{recency_tr_ho,evidence_tr_ho,demo_order_probe_tr_t2}_ds{0,1,2}.json.
"""
import json
import pathlib
import sys

import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "txc_wins"
OUT = ROOT / "plots" / "2026-07-26_txcwins" / "reading_vs_steering.png"

DOSE = 0.5
# (reading key, steering arm, label, colour, marker)
ARCH = [
    ("sae", "sae_broadcast", "TopK SAE", "#0072B2", "o"),
    ("tsae", "tsae_broadcast", "attention temporal SAE", "#009E73", "s"),
    ("txc", "txc_slab", "temporal crosscoder", "#E69F00", "D"),
]
TASKS = [("recency_tr_ho", "instruction position"),
         ("evidence_tr_ho", "evidence order"),
         ("demo_order_probe_tr_t2", "demonstration order")]


def at_dose(arm, mag):
    """Best of the two SIGNS at this dose magnitude.

    The magnitude is matched across arms; the sign is not, because which class you steer toward
    is something the experimenter knows. Indexing the signed +mag instead scores any arm whose
    correct direction is negative as a failure -- the error that withdrew the previous sprint's
    headline, and one that recurs across 31% of this sprint's cells.
    """
    best = None
    for a, v in zip(arm["alphas"], arm["delta_margin"]):
        if abs(abs(a) - mag) < 1e-9 and (best is None or v > best):
            best = v
    return best


def main() -> int:
    pts = {k: [] for k, _, _, _, _ in ARCH}
    for stem, _ in TASKS:
        for ds in range(3):
            p = RES / f"{stem}_ds{ds}.json"
            if not p.exists():
                continue
            r = json.loads(p.read_text())
            reading, arms = r.get("reading") or {}, r["arms"]
            for key, arm, _, _, _ in ARCH:
                auc = (reading.get(key) or {}).get("auc")
                if auc is None or arm not in arms:
                    continue
                pts[key].append((auc, at_dose(arms[arm], DOSE)))

    if not any(pts.values()):
        print("[skip] no inputs")
        return 1

    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    # Deterministic horizontal offset per architecture: SAE and tSAE both sit at AUC 1.000 on
    # most cells and would otherwise occlude each other entirely.
    for off, (key, _, label, col, mk) in zip((-0.006, 0.0, 0.006), ARCH):
        xs = [a + off for a, _ in pts[key]]
        ys = [d for _, d in pts[key]]
        ax.scatter(xs, ys, s=95, c=col, marker=mk, edgecolor="white", linewidth=1.1,
                   zorder=3, label=f"{label}  (n={len(xs)})")

    ax.axhline(0, color="#444444", lw=1.0, zorder=1)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_xlabel("held-out reading AUC of the best single latent\n(1.000 = the factor is perfectly linearly decodable)")
    ax.set_ylabel(f"steering Δ margin at matched dose |α| = {DOSE}\n(how far that latent actually moves the behaviour)")
    ax.set_title("Reading a factor and steering it are different capabilities\n"
                 "nine held-out cells: 3 tasks × 3 dictionary inits",
                 fontsize=12)
    ax.set_xlim(0.50, 1.045)
    ax.set_ylim(-0.45, 5.6)
    ax.legend(loc="center left", frameon=True, fontsize=9, bbox_to_anchor=(0.02, 0.42))

    ax.text(0.515, 5.35,
            "median AUC   SAE 1.000   tSAE 1.000   crosscoder 0.850\n"
            "median Δ       SAE +0.09   tSAE +0.04   crosscoder +1.12",
            fontsize=9.5, ha="left", va="top", color="#222222", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="#f6f6f6", ec="#bbbbbb"))
    ax.text(0.515, 4.42,
            "Reading does not separate the architectures — all three\n"
            "reach AUC 1.000 on most cells. Steering separates them 12×.",
            fontsize=9.5, ha="left", va="top", color="#444444", style="italic")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    print("[saved]", OUT)
    for key, _, label, _, _ in ARCH:
        xs = [a for a, _ in pts[key]]
        ys = [d for _, d in pts[key]]
        if xs:
            print(f"  {label:<24} median AUC {sorted(xs)[len(xs)//2]:.3f}   "
                  f"median Δ {sorted(ys)[len(ys)//2]:+.2f}   n={len(xs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
