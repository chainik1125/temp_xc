"""One number, computable before any dictionary is trained, separates the tasks a window
dictionary wins from the tasks it loses.

`c` is the share of the optimal steering write that lies along the all-positions-equal
direction — the only write a per-token dictionary latent can produce, since it has exactly
one decoder direction:

    c = T ||mean_t P[t]||^2 / ||P||_F^2

WHICH SLAB `c` IS COMPUTED FROM DECIDES WHETHER IT WORKS AT ALL, and this figure is the
evidence. Two candidates:

    P_dom = mean(x | A) - mean(x | B)     what DISTINGUISHES the classes
    Gbar  = mean_docs d(margin)/dW        what INCREASES the metric

They are nearly orthogonal in practice (measured cos: +0.052, -0.046, +0.114, +0.193 across
four tasks, against a random baseline of 0.0074). And they disagree on the cases that matter:
`c(P_dom)` is 0.036 for the order task and 0.039 for the instruction-position task — two
tasks with OPPOSITE outcomes and indistinguishable values — while `c(Gbar)` gives 0.225
against 0.036. Steering optimises the metric, not the class separation, so the gradient is
the right object, and the difference-of-means proxy is not a cheap version of it but a
different quantity.

The vertical axis is the crosscoder's margin over the best CONSTANT write available on that
task — the largest of the SAE's direction, the tSAE's direction, the crosscoder's own slab
flattened, and a random constant direction — read at the SMALLEST dose where the crosscoder
is significant, i.e. inside the linear regime the framework describes rather than at each
arm's saturation point.

The previous sprint's headline task sits at the right-hand end, which is the retrospective
explanation of why it was reported as a crosscoder win and is not one: a fifth of its optimal
write is constant, so a constant write always had grip on it, and the only question was
whether anyone swept the dose sign that revealed it.

Reads results/txc_wins/geometry_all.json for `c`, and the per-task result files for the arms.
"""
import json
import pathlib

import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "txc_wins"
OUT = ROOT / "plots" / "2026-07-26_txcwins" / "c_gate.png"

CONSTANT_ARMS = ("sae_broadcast", "tsae_broadcast", "txc_flat", "random_broadcast")
# task -> (result file, label, colour). One row per task, every `c` from the same screen.
TASKS = [
    ("demo_order_probe", "demo_order_probe_t2_ds0", "demonstration order", "#0072B2"),
    ("recency", "recency_v2", "instruction position", "#E69F00"),
    ("rot_m12", "rot_m12_T", "rotation, m=12", "#CC79A7"),
    ("evidence", "evidence_v2", "evidence order", "#009E73"),
    ("rot_m6", "rot_m6_T", "rotation, m=6", "#CC79A7"),
    ("order", "order_sym_ds0", "order (last sprint)", "#D55E00"),
    ("rot_m2", "rot_m2_T", "rotation, m=2", "#CC79A7"),
    ("phase1", "phase1_v2_ds0", "phase, 1 switch", "#999999"),
]


def at_dose(arm, mag):
    best = None
    for a, v, e in zip(arm["alphas"], arm["delta_margin"], arm["sem"]):
        if abs(abs(a) - mag) < 1e-9 and (best is None or v > best[0]):
            best = (v, e)
    return best


def margin_at_linear_dose(r):
    """Crosscoder minus the best constant arm, at the smallest significant dose."""
    arms = r["arms"]
    mags = sorted({abs(a) for a in arms["txc_slab"]["alphas"]})
    for m in mags:
        t = at_dose(arms["txc_slab"], m)
        if t and t[0] > 2.0 * t[1]:
            const = max((at_dose(arms[a], m)[0] for a in CONSTANT_ARMS if a in arms),
                        default=0.0)
            return t[0] - const, m, t[1]
    m = mags[-1]                       # never significant: report at the largest dose
    t = at_dose(arms["txc_slab"], m)
    const = max((at_dose(arms[a], m)[0] for a in CONSTANT_ARMS if a in arms), default=0.0)
    return t[0] - const, None, t[1]


def main() -> int:
    geo_path = SRC / "geometry_all.json"
    if not geo_path.exists():
        print(f"[skip] {geo_path} not written yet")
        return 1
    geo = json.loads(geo_path.read_text())["tasks"]

    pts = []
    for key, fname, label, colour in TASKS:
        f = SRC / f"{fname}.json"
        if key not in geo or not f.exists():
            continue
        r = json.loads(f.read_text())
        if "txc_slab" not in r.get("arms", {}):
            continue
        margin, dose, sem = margin_at_linear_dose(r)
        pts.append((geo[key]["Gbar"]["c"], margin, sem, label, colour, dose))
    if not pts:
        print("[skip] no matched tasks")
        return 1

    import itertools
    conc = dis = 0
    for i, j in itertools.combinations(range(len(pts)), 2):
        s = (pts[i][0] - pts[j][0]) * (pts[i][1] - pts[j][1])
        conc += s > 0
        dis += s < 0
    tau = (conc - dis) / max(conc + dis, 1)

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    for c, margin, sem, label, colour, dose in pts:
        ax.errorbar(c, margin, yerr=sem, fmt="o", color=colour, ms=10, capsize=4,
                    markeredgecolor="black", markeredgewidth=0.7, zorder=3)
        ax.annotate(label + ("" if dose else "  (n.s.)"), (c, margin),
                    textcoords="offset points",
                    xytext=(10, 5 if margin > 0 else -13), fontsize=8.5, color="#333333")
    ax.axhline(0.0, ls="--", color="#888888", lw=1.4)
    ax.text(0.99, 0.97, "crosscoder beats every constant write above this line",
            transform=ax.transAxes, fontsize=8.5, color="#555555", ha="right", va="top")
    ax.set_xlabel(r"$c$ measured on the metric gradient"
                  "\n"
                  "share of the optimal write that is constant across positions")
    ax.set_ylabel("crosscoder $-$ best constant write\n"
                  "at the smallest significant dose")
    # Rank correlation of MAGNITUDES understates this: the deltas are on different
    # scales across tasks, so what the screen is being asked to do is classify the SIGN.
    # A threshold quoted alongside tau is the honest pair.
    wins = sorted(c for c, m, *_ in pts if m > 0)
    losses = sorted(c for c, m, *_ in pts if m <= 0)
    thr = (max(wins) + min(losses)) / 2 if wins and losses else float("nan")
    correct = sum(m > 0 for c, m, *_ in pts if c < thr) + \
        sum(m <= 0 for c, m, *_ in pts if c >= thr)
    ax.axvspan(min(losses), max(wins), color="#888888", alpha=0.10, lw=0,
               zorder=0) if wins and losses and min(losses) < max(wins) else None
    ax.set_title("One pre-training number separates the wins from the losses\n"
                 f"threshold $c$ = {thr:.2f} classifies {correct}/{len(pts)}   "
                 f"(Kendall $\\tau$ on magnitudes = {tau:+.2f})")
    ax.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    print("[saved]", OUT)
    print(f"\n{'task':<24}{'c(Gbar)':>9}{'dose':>7}{'txc - best const':>20}")
    for c, margin, sem, label, _, dose in sorted(pts):
        print(f"{label:<24}{c:>9.3f}{(f'{dose:.2f}' if dose else 'n.s.'):>7}"
              f"{margin:>14.2f} +- {sem:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
