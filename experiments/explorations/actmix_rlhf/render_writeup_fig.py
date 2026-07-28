"""ACTMIX RLHF writeup figure (CARD § 7 A1; directive 059a66239).

`figs_writeup/fig_rlhf_shuffle_tsweep.{png,pdf}` in the Aniket
template: x = T (log2), ordered solid + shuffled dashed, faint
per-seed lines, seed-mean ± sd, "T=16 − T=1: +X" annotation.
T = 1 shuffled point = the ordered value by construction (a
within-window shuffle of a length-1 window is the identity) —
annotated on the figure. Seed coverage per T is auto-disclosed in
the corner note (ragged during the interim render).

Usage:
  .venv/bin/python -m experiments.explorations.actmix_rlhf.render_writeup_fig \
      --tag interim   # or: --tag final
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.explorations.actmix_rlhf.cells import (  # noqa: E402
    DATASOURCE, PF_ARCH, PF_DATASOURCE, TXC_ARCH)

# CARD § 8 paper-faithful arm renders through the SAME machinery; only
# the (arch, datasource) selector and the binding caption differ. The
# btk path is unchanged — `--arm btk` is the default and reproduces the
# previous output byte-for-byte.
ARMS = {
    "btk": {"arch": TXC_ARCH, "datasource": DATASOURCE,
            "stem": "fig_rlhf_shuffle_tsweep"},
    "pf": {"arch": PF_ARCH, "datasource": PF_DATASOURCE,
           "stem": "fig_rlhf_shuffle_tsweep_pf"},
}

ROOT = Path(__file__).resolve().parents[3]
LEADERBOARD = ROOT / "results" / "leaderboard.jsonl"
OUT_DIR = ROOT / "figs_writeup"
SEEDS = (42, 1, 2)  # explicit whitelist — stray seed-0 smoke rows excluded
TXC = "#D55E00"  # house Okabe-Ito; hue follows the entity, linestyle
                 # carries the order condition (CVD-safe split)


def load_points(arch=TXC_ARCH, datasource=DATASOURCE):
    """(T, seed) -> {ordered, shuffled} from the latest matching row.

    `arch`/`datasource` select the arm. They are a REQUIRED pair: the
    two arms share metric names (`preference_auc_k20`), so filtering on
    only one of them would silently mix the plain-TXC and
    paper-faithful arms into one figure.
    """
    best = {}
    for line in LEADERBOARD.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if (r.get("evaluator_name") != "rlhf"
                or r.get("arch") != arch
                or r.get("datasource") != datasource
                or int(r.get("seed", -1)) not in SEEDS):
            continue
        if not (r.get("training_cfg") or {}).get("n_steps"):
            continue  # untrained twins are not in this figure
        m = r["metrics"]
        T = int(m["T"])
        key = (T, int(r["seed"]))
        if key in best and best[key]["ts"] >= r["ts"]:
            continue
        ordered = m["preference_auc_k20"]
        shuffled = (ordered if T == 1  # identity by construction
                    else m.get("shuffled_preference_auc_k20"))
        best[key] = {"ts": r["ts"], "ordered": ordered, "shuffled": shuffled}
    return best


def series(points, seed, field):
    Ts = sorted(T for (T, s) in points if s == seed)
    return Ts, [points[(T, seed)][field] for T in Ts]


def mean_sd(points, field):
    Ts = sorted({T for (T, _) in points})
    mu, sd, n = [], [], []
    for T in Ts:
        vals = [v[field] for (t, _), v in points.items()
                if t == T and v[field] is not None]
        n.append(len(vals))
        mu.append(sum(vals) / len(vals))
        m = mu[-1]
        sd.append((sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5
                  if len(vals) > 1 else None)
    return Ts, mu, sd, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", choices=("interim", "final", "checkpoint"),
                    required=True)
    ap.add_argument("--arm", choices=tuple(ARMS), default="btk",
                    help="btk = plain-TXC modernization (default, "
                         "unchanged); pf = CARD § 8 paper-faithful arm "
                         "(agentic_txc_02_v1t on the paper stream).")
    ap.add_argument("--g1", choices=("pending", "passed", "failed"),
                    default="pending",
                    help="pf only: port-fidelity verdict state. Default "
                         "'pending' stamps the caveat required by hub "
                         "ruling 4e04ae0e3 item 4 — wave-1 cells launch "
                         "with G1 unresolved and the verdict must travel "
                         "with every figure until it lands.")
    ap.add_argument("--pair-style", choices=("mono", "blueorange"),
                    default="mono",
                    help="mono = single pair-hue (linestyle carries the "
                         "order condition); blueorange = Aniket "
                         "backtracking-fig sibling styling (shuffled in "
                         "house blue #0072B2). Meeting decision — see "
                         "LOG 36655341a.")
    args = ap.parse_args()
    shuf_c = TXC if args.pair_style == "mono" else "#0072B2"
    colors = {"ordered": TXC, "shuffled": shuf_c}

    arm = ARMS[args.arm]
    points = load_points(arm["arch"], arm["datasource"])
    if not points:
        raise SystemExit(
            f"no matching leaderboard rows for arm={args.arm} "
            f"(arch={arm['arch']}, datasource={arm['datasource']})")

    fig, ax = plt.subplots(figsize=(5.4, 3.7))

    for seed in SEEDS:  # faint per-seed lines, both conditions
        for field, ls in (("ordered", "-"), ("shuffled", "--")):
            Ts, ys = series(points, seed, field)
            if Ts:
                ax.plot(Ts, ys, ls, color=colors[field], alpha=0.25,
                        lw=1, zorder=1)

    for field, ls, mk, mfc, label in (
            ("ordered", "-", "o", None, "ordered"),
            ("shuffled", "--", "s", "white", "within-window shuffled")):
        c = colors[field]
        Ts, mu, sd, n = mean_sd(points, field)
        ax.plot(Ts, mu, ls, color=c, lw=2, marker=mk, ms=6,
                mfc=mfc or c, mec=c, label=label, zorder=3)
        for T, m, s in zip(Ts, mu, sd):
            if s is not None:
                ax.errorbar(T, m, yerr=s, color=c, capsize=3,
                            lw=1.2, zorder=2)

    Ts, mu, _, n = mean_sd(points, "ordered")
    if 16 in Ts and 1 in Ts:
        delta = mu[Ts.index(16)] - mu[Ts.index(1)]
        ax.annotate(f"T=16 − T=1: {delta:+.3f}", xy=(0.03, 0.95),
                    xycoords="axes fraction", ha="left", va="top",
                    fontsize=9)
    if 1 in Ts:  # never assert it for a point that is not on the plot
        ax.annotate("T=1: shuffle ≡ identity (by construction)",
                    xy=(0.03, 0.88), xycoords="axes fraction",
                    ha="left", va="top", fontsize=8, color="#555555")

    cov = " ".join(f"T{T}:n={k}" for T, k in zip(Ts, n))
    tag_note = {
        "interim": "INTERIM — remaining seeds in flight",
        "final": "FINAL — seeds {42, 1, 2}",
        # 9e80f03aa item 4 + b5c25b0f5: deliverable-of-record caption
        "checkpoint": ("CHECKPOINT (deliverable of record) — T6/T10 "
                       "deferred for paper-faithful priority; final "
                       "sweep supersedes in the amendment window"),
    }[args.tag]
    if args.arm == "pf":
        # Hub ruling 4e04ae0e3 item 4: wave-1 cells launch with G1
        # unresolved and the verdict travels with every figure until it
        # lands. Stamped TOP-CENTRE, not appended to the bottom-right
        # note — a gate caveat that a reader can miss is not a
        # disclosure, and the bottom-right corner already carries the
        # legend and coverage.
        g1_note, g1_col = {
            "pending": ("G1 PORT-FIDELITY VERDICT PENDING — pre-G1 cells "
                        "(pilot to 25k; no plateau at ~21k vs upstream 5.8k)",
                        "#B00000"),
            "passed": ("G1 port-fidelity verdict: PASSED", "#555555"),
            "failed": ("G1 PORT-FIDELITY VERDICT FAILED — points DISCLOSED, "
                       "NOT certified", "#B00000"),
        }[args.g1]
        ax.annotate(g1_note, xy=(0.5, 1.02), xycoords="axes fraction",
                    ha="center", va="bottom", fontsize=7, color=g1_col,
                    weight="bold")
    ax.annotate(f"{tag_note}\n{cov}", xy=(0.99, 0.02),
                xycoords="axes fraction", ha="right", va="bottom",
                fontsize=6.5, color="#777777", wrap=True,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                          pad=1.5))
    # BINDING caption disclosure (b0b2c49ba): the paper's RLHF TXC arm
    # was agentic_txc_02 — this exhibit is the plain-TXC modernization.
    # The pf arm IS that architecture, so the btk disclaimer would be
    # actively FALSE on the pf figure; each arm carries its own.
    if args.arm == "btk":
        binding = ("paper RLHF TXC arm = agentic_txc_02 (class "
                   "MatryoshkaTXCDRContrastiveMultiscale: matryoshka+contrastive, "
                   "multiscale [1,2,3], per-window TopK→ReLU, k_win=500; distinct "
                   "from txc_pro); exhibit = plain-TXC modernization at paper "
                   "window budget (k_win=100·T; POST composition preserves "
                   "per-window granularity). T-sweep/shuffle conclusions are "
                   "statements about the plain arm.")
    else:
        binding = ("paper-faithful arm: agentic_txc_02_v1t = the paper's own "
                   "RLHF TXC architecture (MatryoshkaTXCDRContrastiveMultiscale) "
                   "ported to this harness, on the paper stream "
                   f"({PF_DATASOURCE}). Conclusions here are statements about "
                   "the PAPER arm itself, not the plain-TXC modernization "
                   "(that is the companion figure). Port fidelity is certified "
                   "by the G1 verdict — see the coverage note for its state.")
    ax.annotate(binding,
                xy=(0.5, -0.16), xycoords="axes fraction", ha="center",
                va="top", fontsize=6, color="#555555", wrap=True)

    ax.set_xscale("log", base=2)
    ax.set_xticks(Ts)
    ax.set_xticklabels([str(T) for T in Ts])
    ax.minorticks_off()
    ax.set_xlabel("T (window length)")
    ax.set_ylabel("preference AUC (k = 20)")
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=8, loc="lower right",
              bbox_to_anchor=(1.0, 0.08))
    fig.tight_layout()

    OUT_DIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{arm['stem']}.{ext}",
                    dpi=300 if ext == "png" else None,
                    bbox_inches="tight")
    print(f"[render_writeup_fig] arm={args.arm} {args.tag}: "
          f"{len(points)} points -> {arm['stem']}; coverage {cov}")


if __name__ == "__main__":
    main()
