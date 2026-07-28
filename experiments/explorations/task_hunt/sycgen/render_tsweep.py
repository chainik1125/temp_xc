"""sycgen T-sweep writeup figure (RETRAIN_CARD § 4, amended § 5 axis).

Template knob-for-knob with the frozen λ̂/ttrend renderer
(`task_hunt/render_overlay_figs.py`, itself matched to the RLHF/probing
pair): x = T log2, ordered solid + shuffled dashed, faint per-seed
lines, seed-mean ± sd whiskers, per-token anchors as horizontal bands
(shuffle ≡ identity at T=1, annotated), coverage note. Task-specific
knobs, all card-documented:
  - NO quoted-panel ticks and NO anchor-gate assert — first training
    on this substrate, no quoted panel exists (card § 1). In their
    place the UNTRAINED TWINS draw as a grey dotted open-marker line
    (the § 2 control), from the as-run shard jsons.
  - Arms carry the `_btkonly` suffix (pinned matrix mapping 692cb).
  - "PENDING TEAM REVIEW" corner stamp until ratification.
  - Budget-match disclosure inherited from `render_stage2.py`: legend
    carries the claiming arm's realized-l0 range; flagged if any
    cell-mean falls below nominal k/2 (the λ̂ round-1 TXC-post
    collapse rule).
Emits figs_writeup/fig_sycgen_shuffle_tsweep.{png,pdf} +
results/sycgen_tsweep_summary.json.

  .venv/bin/python -m experiments.explorations.task_hunt.sycgen.render_tsweep
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT_DIR = HERE.parents[3] / "figs_writeup"
TXC = "#D55E00"
SAE_C, TSAE_C, UNTR_C = "#555555", "#888888", "#999999"

ARM = "txc_batchtopk_post_btkonly"
ANCHORS = (("batchtopk_sae_btkonly", SAE_C, "per-token BatchTopK SAE (T=1)"),
           ("tsae_btkonly", TSAE_C, "T-SAE (T=1)"))
K_POS = 8


def load_shard_rows():
    rows = []
    for p in sorted(RES.glob("retrain_*.json")):
        blob = json.loads(p.read_text())
        rows += blob if isinstance(blob, list) else blob["rows"]
    # dedupe (supp cells re-echo in shard1 as cache rows — identical
    # metrics; keep first so seed-level sd stays honest)
    seen, out = set(), []
    for r in rows:
        if not r.get("ok"):
            continue
        k = (r["arch"], r["T"], r["seed"], r.get("kind"))
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def main():
    # Overlay is OPTIONAL (Han order 03:25 07-28: partial renders ship
    # fig-first; ordered layer comes from canonical rows, shuffled
    # dashes only where the overlay has scored cells; full-drain
    # render supersedes at the same paths).
    overlay_path = RES / "sycgen_shuffle_overlay.json"
    payload = (json.loads(overlay_path.read_text())
               if overlay_path.exists() else None)
    if payload is not None:
        assert str(payload.get("anchor_gate", "")
                   ).startswith("none by design"), (
            "this renderer is sycgen-specific (no-gate lane); refusing "
            f"a payload with anchor_gate={payload.get('anchor_gate')!r}")
    ov_cells = [] if payload is None else payload["cells"]
    ov_at = {(c["arch"], c["T"], c["seed"]): c for c in ov_cells}

    shard_rows = load_shard_rows()
    trained = defaultdict(list)         # (arch, T) -> [(seed, r)]
    untrained = defaultdict(list)       # (arch, T) -> [r]
    l0 = defaultdict(list)              # (arch, T) -> [l0_per_token]
    for r in shard_rows:
        key = (r["arch"], r["T"])
        m = r["metrics"]
        if r.get("kind") == "untrained":
            untrained[key].append(m.get("lambda_recovery", float("nan")))
        else:
            trained[key].append((r["seed"],
                                 m.get("lambda_recovery", float("nan"))))
            l0[key].append(m.get("l0_per_token", float("nan")))

    # x-axis from the amended grid; ordered points exist where trained
    # rows exist, shuffled where the overlay scored that (T, seed).
    GRID_TS = (2, 4, 8, 16)
    Ts = [T for T in GRID_TS if trained.get((ARM, T))]
    at = {(T, s): {"recomputed_r": r} for T in Ts
          for (s, r) in trained[(ARM, T)]}
    for (a, T, s), c in ov_at.items():
        if a == ARM and (T, s) in at:
            at[(T, s)] = c
    seeds = sorted({s for (_, s) in at})
    n_tr = sum(len(v) for k, v in trained.items())
    n_un = sum(len(v) for v in untrained.values())
    partial = not (n_tr == 18 and n_un == 18 and len(ov_cells) == 18)

    # λ̂ round-1 disclosure rule: NOT budget-matched if any trained
    # cell-mean realized l0 < nominal k/2.
    arm_l0 = [float(np.nanmean(v)) for (a, _), v in sorted(l0.items())
              if a == ARM]
    l0_lo, l0_hi = (min(arm_l0), max(arm_l0)) if arm_l0 else (np.nan, np.nan)
    budget_matched = bool(arm_l0) and l0_lo >= K_POS / 2
    l0_tag = f"l0 {l0_lo:.2g}–{l0_hi:.2g}" if arm_l0 else "l0 n/a"
    if not budget_matched:
        l0_tag += " — NOT budget-matched"

    fig, ax = plt.subplots(figsize=(5.4, 3.7))

    def series(field):
        """(Ts_present, mu, sd) over seeds that carry `field` per T."""
        xs, mu, sd = [], [], []
        for T in Ts:
            vs = [at[(T, s)][field] for s in seeds
                  if (T, s) in at and field in at[(T, s)]]
            if not vs:
                continue
            xs.append(T)
            mu.append(float(np.mean(vs)))
            sd.append(float(np.std(vs, ddof=1)) if len(vs) > 1 else 0.0)
        return xs, mu, sd

    for seed in seeds:
        for field, ls in (("recomputed_r", "-"), ("r_shuf", "--")):
            pts = [(T, at[(T, seed)][field]) for T in Ts
                   if (T, seed) in at and field in at[(T, seed)]]
            if pts:
                ax.plot(*zip(*pts), ls, color=TXC, alpha=0.25, lw=1,
                        zorder=1)

    for field, ls, mk, mfc, label in (
            ("recomputed_r", "-", "o", None, f"ordered ({l0_tag})"),
            ("r_shuf", "--", "s", "white", "within-window shuffled")):
        xs, mu, sd = series(field)
        if not xs:
            continue
        ax.plot(xs, mu, ls, color=TXC, lw=2, marker=mk, ms=6,
                mfc=mfc or TXC, mec=TXC, label=label, zorder=3)
        ax.errorbar(xs, mu, yerr=sd, color=TXC, capsize=3, lw=1.2,
                    fmt="none", zorder=2)

    # untrained twins — the § 2 control (in the quoted-ticks slot of the
    # template; no quoted panel exists on a first training). Drawn over
    # the FULL amended axis (shard0 complete) even when trained points
    # are still partial.
    un_Ts = [T for T in GRID_TS if untrained.get((ARM, T))]
    un_mu = [float(np.nanmean(untrained[(ARM, T)])) for T in un_Ts]
    if un_Ts:
        ax.plot(un_Ts, un_mu, ":", color=UNTR_C, lw=1.6, marker="o",
                ms=5, mfc="white", mec=UNTR_C, label="untrained twins",
                zorder=2)

    x_right = max(Ts or un_Ts or [16])
    for arm, c, label in ANCHORS:
        vals = [r for (_, r) in trained.get((arm, 1), [])]
        if vals:
            m = float(np.mean(vals))
            s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            ax.axhspan(m - s, m + s, color=c, alpha=0.12, zorder=0)
            ax.axhline(m, color=c, lw=1, ls=":", zorder=0)
            ax.annotate(label, xy=(x_right, m), fontsize=7, color=c,
                        ha="right", va="bottom")

    import time
    stamp = time.strftime("%H:%M %d-%m")
    ax.annotate("anchors at T=1: shuffle ≡ identity (by construction)",
                xy=(0.03, 0.95), xycoords="axes fraction", ha="left",
                va="top", fontsize=8, color="#555555")
    cov = " ".join(
        f"T{T}:n={len([1 for s in seeds if (T, s) in at])}" for T in Ts)
    shuf_note = ("shuffle overlay PENDING" if payload is None
                 else f"shuffle seed {payload['shuf_eval_seed']}")
    part_note = (f"\nPARTIAL {n_tr}/18 trained · {len(ov_cells)}/18 "
                 f"overlay · {n_un}/18 untrained at {stamp} London — "
                 "full drain supersedes" if partial else "")
    ax.annotate("no anchor gate BY DESIGN (first training; card § 1)\n"
                f"{cov} · {shuf_note}{part_note}",
                xy=(0.99, 0.02), xycoords="axes fraction", ha="right",
                va="bottom", fontsize=6.5, color="#777777", zorder=5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="none", alpha=0.75))
    ax.annotate("PENDING TEAM REVIEW" + (" · PARTIAL" if partial else ""),
                xy=(0.5, 0.97), xycoords="axes fraction", ha="center",
                va="top", fontsize=7, color="#bb4444", alpha=0.8)

    axis_Ts = sorted(set(Ts) | set(un_Ts)) or list(GRID_TS)
    ax.set_xscale("log", base=2)
    ax.set_xticks(axis_Ts)
    ax.set_xticklabels([str(T) for T in axis_Ts])
    ax.minorticks_off()
    ax.set_xlabel("T (window length)")
    ax.set_ylabel("recovery r (log2(1+challenge age))")
    ax.grid(True, alpha=0.25, lw=0.5)
    leg = ax.legend(frameon=True, framealpha=0.85, edgecolor="none",
                    fontsize=8, loc="center left",
                    bbox_to_anchor=(0.02, 0.42))
    leg.set_zorder(5)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.06 * (hi - lo))   # headroom over anchor bands
    fig.tight_layout()

    OUT_DIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig_sycgen_shuffle_tsweep.{ext}", dpi=200)

    ord_xs, ord_mu, _ = series("recomputed_r")
    shf_xs, shf_mu, _ = series("r_shuf")
    summary = {
        "card": "sycgen/RETRAIN_CARD.md §§ 4–5",
        "status": ("PENDING TEAM REVIEW" if payload is None
                   else payload.get("status")),
        "partial": partial,
        "coverage": {"trained": n_tr, "overlay": len(ov_cells),
                     "untrained": n_un, "of_each": 18,
                     "rendered_at": stamp},
        "x_axis": [1] + list(GRID_TS),
        "ordered": {f"T{x}": m for x, m in zip(ord_xs, ord_mu)},
        "shuffled": {f"T{x}": m for x, m in zip(shf_xs, shf_mu)},
        "untrained": {f"{a}/T{t}": float(np.nanmean(v))
                      for (a, t), v in sorted(untrained.items())},
        "anchors": {a: float(np.mean([r for (_, r) in trained[(a, 1)]]))
                    for a, _, _ in ANCHORS if trained.get((a, 1))},
        "l0_range_claiming_arm": (
            {"min": l0_lo, "max": l0_hi} if arm_l0 else None),
        "budget_matched": budget_matched if arm_l0 else None,
        "match_rule": f"min cell-mean realized l0 >= k_pos/2 = {K_POS / 2}",
    }
    (RES / "sycgen_tsweep_summary.json").write_text(
        json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"[render] -> {OUT_DIR / 'fig_sycgen_shuffle_tsweep'}.{{png,pdf}} "
          f"+ {RES / 'sycgen_tsweep_summary.json'}")


if __name__ == "__main__":
    main()
