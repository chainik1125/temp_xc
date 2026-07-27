"""The paper's steering protocol against the full-slab write, on every cell we have data for.

`txc_flat` is the crosscoder slab time-averaged and rebroadcast. For a temporal crosscoder that
is EXACTLY what the paper's default `v7` tiled-broadcast hook writes, up to a per-document
scalar: clamping latent j gives delta_t = (s - z_j) * W_dec[j,t,:], and v7 then averages over
the window and broadcasts the mean. See `docs/dmitry/reviewer_responses/steering_conventions.tex`
Proposition 1. So the flat-vs-slab comparison already in these result files IS the
protocol comparison -- no new compute is needed to make it.

TOP PANEL, the honest axis. Behaviour against judged coherence, from the generation sweep with
the GPT-4o coherence join (`recency_tr_gensweep.json`). Each point is one dose; the curve is the
frontier that dose sweep traces. A steering method is good if it moves behaviour far while
staying coherent, so up-and-right wins and a single scalar at one dose cannot express it. The
coherence floor is drawn at 50, the value the EM work uses.

BOTTOM ROW. Dose-response of the teacher-forced margin on four held-out cells, signed rather
than best-of-sign, because the point here is the SHAPE of the response and folding the sign
hides the asymmetry. Faint lines are individual dictionary initialisations, heavy lines their
median.

Reads results/txc_wins/*.json. Writes plots/2026-07-27_protocols/paper_vs_slab.png.
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "txc_wins"
OUT = ROOT / "plots" / "2026-07-27_protocols" / "paper_vs_slab.png"

# Wong palette. Black is the published/conventional arm, per the project's figure convention.
C_SLAB = "#E69F00"     # orange   -- full slab (DC + AC)
C_FLAT = "#000000"     # black    -- paper v7 protocol (DC only)
C_SAE = "#0072B2"      # blue     -- TopK SAE, for scale
C_RAND = "#999999"     # grey     -- random slab, the null

GEN = "recency_tr_gensweep.json"
CELLS = [
    ("instruction position\n(recency, held out)", "recency_tr_ho_ds{}.json", range(3)),
    ("retrieved-doc position\n(Lost-in-the-Middle 0v4)", "litm_0v4_tr_tsaep_ds{}.json", range(3)),
    ("evidence order\n(held out)", "evidence_tr_ho_ds{}.json", range(3)),
    ("demonstration order\n(held out)", "demo_order_probe_tr_t2_ds{}.json", range(3)),
]
ARMS = [("txc_slab", "full slab  (DC + AC)", C_SLAB, "o", 2.4),
        ("txc_flat", "paper v7 protocol  (DC only)", C_FLAT, "s", 2.4),
        ("sae_broadcast", "TopK SAE", C_SAE, "^", 1.3),
        ("random_slab", "random slab (null)", C_RAND, "x", 1.1)]


def gen_frontier(path):
    """Per arm, per dose: (mean judged coherence, class-A minus class-B obedience gap)."""
    r = json.loads(path.read_text())
    gen = r.get("generations") or r.get("gen") or {}
    out = {}
    for arm, recs in gen.items():
        by = {}
        for g in recs:
            if g.get("judge_coherence") is None:
                continue
            by.setdefault(float(g["alpha"]), []).append(g)
        pts = []
        for a, rows in sorted(by.items()):
            A = [x["upper_frac_reply"] for x in rows if x["cls"] == "A"]
            B = [x["upper_frac_reply"] for x in rows if x["cls"] == "B"]
            coh = [x["judge_coherence"] for x in rows]
            if not A or not B:
                continue
            pts.append((a, float(np.mean(coh)), float(np.mean(A) - np.mean(B)), len(rows)))
        if pts:
            out[arm] = pts
    return out


def dose_curves(stem, seeds, arm):
    """(alphas, per-seed margin arrays) for one arm across dictionary initialisations."""
    xs, ys = None, []
    for s in seeds:
        p = RES / stem.format(s)
        if not p.exists():
            continue
        a = (json.loads(p.read_text()).get("arms") or {}).get(arm)
        if not a:
            continue
        xs = a["alphas"]
        ys.append(a["delta_margin"])
    return xs, ys


def main() -> int:
    gp = RES / GEN
    fig = plt.figure(figsize=(13.6, 9.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1.0], hspace=0.42, wspace=0.28)

    # ---------- top: behaviour vs judged coherence ----------
    ax = fig.add_subplot(gs[0, :])
    if gp.exists():
        fr = gen_frontier(gp)
        base = None
        for arm, label, col, mk, lw in ARMS:
            if arm not in fr:
                continue
            pts = fr[arm]
            coh = [p[1] for p in pts]
            gap = [p[2] for p in pts]
            ax.plot(coh, gap, "-", color=col, lw=lw, alpha=0.85, zorder=3)
            ax.scatter(coh, gap, s=58, c=col, marker=mk, edgecolor="white",
                       linewidth=0.9, zorder=4, label=label)
            for a, c, g, _n in pts:
                if abs(a) in (0.5, 2.0):
                    ax.annotate(f"α={a:g}", (c, g), textcoords="offset points",
                                xytext=(5, 5), fontsize=7.5, color=col)
        if "none" in fr and fr["none"]:
            base = fr["none"][0][2]
            ax.axhline(base, color="#666666", ls=":", lw=1.2, zorder=1)
            ax.annotate("unsteered", (ax.get_xlim()[0], base), textcoords="offset points",
                        xytext=(6, 4), fontsize=8, color="#666666")
        ax.axvline(50, color="#CC79A7", ls="--", lw=1.3, zorder=1)
        ax.annotate("coherence floor (50)", (50, ax.get_ylim()[1]), textcoords="offset points",
                    xytext=(6, -14), fontsize=8.5, color="#CC79A7", rotation=90, va="top")
        ax.set_xlabel("mean GPT-4o judged coherence  →  more coherent")
        ax.set_ylabel("instruction-obedience gap\nmean(class A) − mean(class B) upper-case fraction")
        ax.set_title("Behaviour reached, against the coherence it costs — every point is one dose\n"
                     "recency task, generation sweep with GPT-4o coherence join   "
                     "⚠ no random-write control in this sweep, so a large excursion here is "
                     "not evidence the direction is meaningful",
                     fontsize=10.5)
        ax.grid(alpha=0.22, lw=0.6)
        ax.legend(loc="upper left", frameon=True, fontsize=9)
    else:
        ax.text(0.5, 0.5, f"missing {GEN}", ha="center", va="center")
        ax.axis("off")

    # ---------- bottom: signed dose-response on four cells ----------
    for i, (title, stem, seeds) in enumerate(CELLS):
        axi = fig.add_subplot(gs[1, i])
        any_data = False
        for arm, label, col, mk, lw in ARMS:
            xs, ys = dose_curves(stem, seeds, arm)
            if not xs or not ys:
                continue
            any_data = True
            Y = np.array(ys)
            for row in Y:
                axi.plot(xs, row, "-", color=col, lw=0.7, alpha=0.30, zorder=2)
            axi.plot(xs, np.median(Y, 0), "-", color=col, lw=lw, marker=mk, ms=4.5,
                     zorder=3, label=label if i == 0 else None)
        # The load-bearing ratio is flat-vs-NULL, not flat-vs-slab: a constant write in the
        # latent's mean direction has to beat a constant write in a random direction before it
        # can be said to carry anything latent-specific.
        pk = {}
        for arm in ("txc_slab", "txc_flat", "random_slab"):
            xs, ys = dose_curves(stem, seeds, arm)
            if xs and ys:
                med = np.median(np.array(ys), 0)
                pk[arm] = abs(med[int(np.argmax(np.abs(med)))])
        if {"txc_flat", "random_slab"} <= set(pk):
            axi.text(0.03, 0.97,
                     f"slab/flat {pk['txc_slab']/max(pk['txc_flat'],1e-9):.1f}×\n"
                     f"flat/random {pk['txc_flat']/max(pk['random_slab'],1e-9):.2f}×",
                     transform=axi.transAxes, va="top", ha="left", fontsize=8,
                     family="monospace",
                     bbox=dict(boxstyle="round,pad=0.35", fc="#f7f7f7", ec="#bbbbbb", lw=0.7))
        axi.axhline(0, color="#444444", lw=0.9, zorder=1)
        axi.axvline(0, color="#cccccc", lw=0.8, zorder=1)
        axi.set_title(title, fontsize=9.5)
        axi.set_xlabel("steering coefficient α")
        if i == 0:
            axi.set_ylabel("Δ margin (teacher forced)")
        axi.grid(alpha=0.22, lw=0.6)
        if not any_data:
            axi.text(0.5, 0.5, "no data", ha="center", va="center", transform=axi.transAxes)

    fig.suptitle("The paper's steering protocol writes only the window-average of the slab",
                 fontsize=13.5, y=0.985)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=165, bbox_inches="tight")
    print("[saved]", OUT)

    # numbers behind the bottom row, so the figure can be checked without reading it
    print(f"\n{'cell':<34} {'arm':<16} {'peak |Δ|':>9} {'at α':>6}")
    for title, stem, seeds in CELLS:
        for arm, _l, _c, _m, _w in ARMS[:2]:
            xs, ys = dose_curves(stem, seeds, arm)
            if not xs or not ys:
                continue
            med = np.median(np.array(ys), 0)
            j = int(np.argmax(np.abs(med)))
            print(f"{title.splitlines()[0]:<34} {arm:<16} {med[j]:9.3f} {xs[j]:6.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
