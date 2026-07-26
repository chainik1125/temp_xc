"""Where the crosscoder's headline margin actually comes from — and it is mostly not rank.

The sprint's headline is that a crosscoder latent beats the SAE's broadcast arm by a wide
margin on instruction-position bias. Two arms added late let that margin be decomposed
instead of interpreted, and the piece attributable to the crosscoder's temporal freedom is
the smallest of the three:

    grad_slab           the optimal (T, d) write, supervised, first-order optimal
    broadcast_optimal   the best CONSTANT write in the whole space, not merely the best in
                        a 4096-atom dictionary. Maximising <W, Gbar> over all v for
                        W = (1_T (x) v)/||.|| is attained exactly at v ∝ mean_t Gbar.
    txc_slab            the crosscoder latent
    sae_broadcast       the SAE's direction, latent chosen by MEASURED steering on a split
                        disjoint from both the gradient documents and the test set
    ...readingsel       the same, latent chosen by reading AUC — the sprint's own convention

Reading the bars left to right on any init:

  * the crosscoder exceeds the entire BROADCAST FORM by 0.95x to 1.20x. In one init of
    three it does not exceed it at all. So the form is not the binding constraint.
  * the best constant direction a trained SAE actually contains reaches 43-44% of the best
    constant direction that exists. THE DICTIONARY is the binding constraint.
  * choosing that latent by reading AUC instead of by measured steering gives away most of
    what is left -- the SAE's reading pick ranks 2507, 2222 and 3138 of 4096 by first-order
    alignment, worse than an arbitrary draw, while the crosscoder's ranks 1, 1 and 1.

All arms at the matched dose, max over signs. Signs genuinely differ between arms here --
`broadcast_optimal` peaks at +0.5 while `txc_slab` peaks at -0.5 in two of three inits --
so signed-positive indexing would score this figure wrongly, which is the error this sprint
found three people making independently.

Reads results/txc_wins/{recency,evidence}_tr_sel_ds*.json.
"""
import glob
import json
import pathlib
import sys

import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "txc_wins"
OUT = ROOT / "plots" / "2026-07-26_txcwins" / "gap_decomposition.png"

ARMS = [
    ("grad_slab", "optimal\nwrite\n(supervised)", "#666666"),
    ("broadcast_optimal", "best constant\nwrite that\nexists", "#CC79A7"),
    ("txc_slab", "crosscoder\nlatent", "#E69F00"),
    ("sae_broadcast", "SAE latent\npicked by\nsteering", "#0072B2"),
    ("sae_broadcast_readingsel", "SAE latent\npicked by\nreading AUC", "#56B4E9"),
]


def at(arm, mag):
    best = None
    for al, v, e in zip(arm["alphas"], arm["delta_margin"],
                        arm.get("sem", [0.0] * len(arm["alphas"]))):
        if abs(abs(al) - mag) < 1e-9 and (best is None or v > best[0]):
            best = (v, e, al)
    return best


def main() -> int:
    tasks = []
    for stem, label in (("recency", "instruction position"), ("evidence", "evidence order")):
        fs = sorted(glob.glob(str(RES / f"{stem}_tr_sel_ds*.json")))
        runs = [json.loads(pathlib.Path(f).read_text()) for f in fs]
        runs = [r for r in runs if "broadcast_optimal" in r.get("arms", {})]
        if runs:
            tasks.append((label, runs))
    if not tasks:
        print("[skip] no selection runs with broadcast_optimal")
        return 1

    fig, axes = plt.subplots(1, len(tasks), figsize=(7.6 * len(tasks), 5.8), squeeze=False)
    for ax, (label, runs) in zip(axes[0], tasks):
        n = len(runs)
        w = 0.8 / n
        for i, r in enumerate(runs):
            mag = r.get("matched_dose_magnitude", 0.5)
            for j, (key, _, col) in enumerate(ARMS):
                if key not in r["arms"]:
                    continue
                v, e, _ = at(r["arms"][key], mag)
                ax.bar(j + (i - (n - 1) / 2) * w, v, width=w * 0.92, color=col,
                       edgecolor="white", lw=0.6,
                       alpha=0.55 + 0.45 * (i + 1) / n, zorder=3)
                ax.errorbar(j + (i - (n - 1) / 2) * w, v, yerr=e, fmt="none",
                            ecolor="#333333", elinewidth=0.9, capsize=2, zorder=4)
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels([lab for _, lab, _ in ARMS], fontsize=8.4)
        ax.axhline(0, color="#999999", lw=0.8)
        ax.grid(axis="y", alpha=0.25, lw=0.6)
        ax.set_ylabel(f"Δ margin at matched dose, held-out content\n"
                      f"({n} dictionary inits, max over signs)")
        ax.set_title(f"{label}", fontsize=12)

        # The three costs, computed rather than asserted.
        def med(key):
            vs = [at(r["arms"][key], r.get("matched_dose_magnitude", 0.5))[0]
                  for r in runs if key in r["arms"]]
            return sorted(vs)[len(vs) // 2] if vs else float("nan")

        t, b, s, sr = (med("txc_slab"), med("broadcast_optimal"),
                       med("sae_broadcast"), med("sae_broadcast_readingsel"))
        ax.annotate(
            f"crosscoder over the whole broadcast form   {t / b:.2f}×\n"
            f"best available constant / best that exists   {s / b:.2f}×\n"
            f"reading selector keeps of the good latent    {sr / s:.2f}×",
            xy=(0.985, 0.975), xycoords="axes fraction", ha="right", va="top",
            fontsize=8.8, family="monospace",
            bbox=dict(fc="white", ec="#bbbbbb", alpha=0.95))

    fig.suptitle("The headline margin is mostly dictionary and selector, not temporal rank",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    print("[saved]", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
