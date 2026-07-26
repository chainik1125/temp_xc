"""Does steering change what the model WRITES, and at what cost to coherence?

The logit metric is teacher-forced, which is what makes it judge-free and cheap. This asks
the separate question of whether the same writes change observable output, on the readout
the task was built around: which of two conflicting instructions the model obeys.

WHY A SWEEP AND NOT ONE DOSE PER ARM. The earlier generations took each arm at
`alphas[argmax(delta_margin)]` -- a signed argmax on the LOGIT metric. `sae_broadcast` and
`txc_flat` came out at -2.0 while `txc_slab` was at +2.0, so the obedience rates being
compared sat at arbitrary, differently-signed points on their own curves. Steering scales
are not commensurable across architectures even at matched injected norm, so a cross-arm
comparison needs the whole curve plus a common admissibility rule.

THE RULE. Each arm is reported at its best obedience rate SUBJECT TO coherence >= floor,
not at its best rate outright. Without that, the winner is whichever arm degrades fastest:
text that collapses to lowercase mush scores as perfect obedience for one of the two
classes while saying nothing about instruction following. This is the `delta align | coh >=
70` convention from the EM work, with a floor set from the model rather than by hand --
the 5th percentile of the UNSTEERED per-generation mean token log-probability.

READING THE FIGURE. Filled markers are doses whose coherence clears the floor; open markers
are doses that do not, and are excluded from each arm's selection. The star is the selected
dose. The horizontal line is the unsteered rate: below 0.5 because the task is built so
recency wins, and the quantity of interest is how far a write pushes it up.

`dom_slab` and `broadcast_optimal` are SUPERVISED -- built from labels and from the metric
gradient respectively -- and are drawn dashed. They are references, not arms a practitioner
holds.

Reads results/txc_wins/recency_gensweep.json.
"""
import json
import pathlib
import sys

import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "txc_wins" / "recency_gensweep.json"
OUT = ROOT / "plots" / "2026-07-26_txcwins" / "gen_sweep.png"

# Wong palette, consistent with the sprint's other figures.
STYLE = {
    "txc_slab":          ("temporal crosscoder", "#E69F00", "-"),
    "sae_broadcast":     ("TopK SAE", "#0072B2", "-"),
    "tsae_broadcast":    ("attention temporal SAE", "#009E73", "-"),
    "txc_flat":          ("crosscoder slab, flattened", "#56B4E9", "-"),
    "dom_slab":          ("difference-of-means (supervised)", "#999999", "--"),
    "broadcast_optimal": ("best constant write (supervised)", "#CC79A7", "--"),
}


def main() -> int:
    if not SRC.exists():
        print(f"[skip] {SRC} not written yet")
        return 1
    d = json.loads(SRC.read_text())
    cells = d.get("gen_sweep") or []
    if not cells:
        print("[skip] no gen_sweep in file")
        return 1
    floor = (d.get("coherence_floor") or {}).get("value")
    base = next((c for c in cells if c["arm"] == "none"), None)

    by_arm = {}
    for c in cells:
        if c["arm"] == "none":
            continue
        by_arm.setdefault(c["arm"], []).append(c)
    for v in by_arm.values():
        v.sort(key=lambda c: c["alpha"])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.4),
                                  gridspec_kw={"width_ratios": [1.55, 1]})

    rows = []
    for arm, cs in by_arm.items():
        label, col, ls = STYLE.get(arm, (arm, "#444444", "-"))
        xs = [c["alpha"] for c in cs]
        ys = [c["obey_earlier_cut50"] for c in cs]
        ok = [floor is None or c["coherence_mean"] >= floor for c in cs]
        ax.plot(xs, ys, ls=ls, lw=1.7, color=col, label=label, zorder=2, alpha=0.9)
        ax.plot([x for x, o in zip(xs, ok) if o], [y for y, o in zip(ys, ok) if o],
                "o", ms=6, color=col, zorder=3)
        ax.plot([x for x, o in zip(xs, ok) if not o], [y for y, o in zip(ys, ok) if not o],
                "o", ms=6, mfc="white", mec=col, mew=1.4, zorder=3)
        adm = [c for c, o in zip(cs, ok) if o]
        if adm:
            best = max(adm, key=lambda c: c["obey_earlier_cut50"])
            ax.plot([best["alpha"]], [best["obey_earlier_cut50"]], "*", ms=17,
                    color=col, mec="white", mew=0.9, zorder=5)
            rows.append((arm, label, best, len(adm), len(cs)))
        # Right panel: the trade-off itself, obedience against coherence.
        ax2.plot([c["coherence_mean"] for c in cs], ys, ls=ls, lw=1.4, color=col,
                 marker="o", ms=4.5, alpha=0.9, zorder=2)

    if base:
        ax.axhline(base["obey_earlier_cut50"], color="#333333", lw=1.2, ls=":", zorder=1)
        ax.annotate(f"unsteered {base['obey_earlier_cut50']:.3f}",
                    (ax.get_xlim()[0], base["obey_earlier_cut50"]),
                    textcoords="offset points", xytext=(6, 5), fontsize=9, color="#333333")
        ax2.plot([base["coherence_mean"]], [base["obey_earlier_cut50"]], "s", ms=9,
                 color="#333333", zorder=5)
    if floor is not None:
        ax2.axvline(floor, color="#B22222", lw=1.3, ls="--", zorder=1)
        ax2.annotate("coherence floor\n(5th pct unsteered)", (floor, 0.02),
                     textcoords="offset points", xytext=(7, 0), fontsize=8.5,
                     color="#B22222", va="bottom")

    ax.set_xlabel("steering coefficient α  (symmetric grid; sign is free per arm,\n"
                  "magnitude matched — arms are NOT scored at a single signed dose)")
    ax.set_ylabel("obeys the EARLIER instruction\n"
                  "(uppercase for class A, lowercase for class B; 0.5 case-fraction cut)")
    ax.set_title("Generation-space dose response, with a coherence floor\n"
                 "filled = coherence clears the floor · open = excluded · star = selected",
                 fontsize=11)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)

    ax2.set_xlabel("coherence: mean token log-probability of the\n"
                   "continuation under the UNSTEERED model")
    ax2.set_ylabel("obeys the earlier instruction")
    ax2.set_title("What the obedience is bought with", fontsize=11)
    ax2.grid(alpha=0.25, lw=0.6)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    print("[saved]", OUT)

    # ---- the table ----
    print(f"\nunsteered: obey {base['obey_earlier_cut50']:.3f} (cut.7 "
          f"{base['obey_earlier_cut70']:.3f})  coherence {base['coherence_mean']:+.3f}  "
          f"n = {base['n']}" if base else "")
    print(f"coherence floor {floor:+.3f}" if floor is not None else "no floor")
    print(f"\n  {'arm':<34}{'α*':>7}{'obey':>8}{'cut.7':>8}{'coh':>9}"
          f"{'rep':>7}{'admissible':>12}")
    for arm, label, b, n_adm, n_all in sorted(
            rows, key=lambda r: -r[2]["obey_earlier_cut50"]):
        tag = " (sup.)" if b.get("supervised") else ""
        print(f"  {label + tag:<34}{b['alpha']:>+7.2f}{b['obey_earlier_cut50']:>8.3f}"
              f"{b['obey_earlier_cut70']:>8.3f}{b['coherence_mean']:>+9.3f}"
              f"{b['repeat_frac_mean']:>7.3f}{f'{n_adm}/{n_all}':>12}")
    print("\n  α* is each arm's best obedience among doses whose coherence clears the floor.")
    print("  'admissible' is how many of its doses cleared it — an arm admitting few doses")
    print("  is one whose effect and whose degradation arrive together.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
