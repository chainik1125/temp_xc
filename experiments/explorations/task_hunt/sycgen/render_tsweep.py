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
    for shard in (0, 1):
        blob = json.loads((RES / f"retrain_shard{shard}.json").read_text())
        rows += blob if isinstance(blob, list) else blob["rows"]
    return [r for r in rows if r.get("ok")]


def main():
    payload = json.loads((RES / "sycgen_shuffle_overlay.json").read_text())
    assert str(payload.get("anchor_gate", "")).startswith("none by design"), (
        "this renderer is sycgen-specific (no-gate lane); refusing a "
        f"payload with anchor_gate={payload.get('anchor_gate')!r}")
    cells = [c for c in payload["cells"] if c["arch"] == ARM]
    seeds = sorted({c["seed"] for c in cells})
    Ts = sorted({c["T"] for c in cells})
    at = {(c["T"], c["seed"]): c for c in cells}

    shard_rows = load_shard_rows()
    untrained = defaultdict(list)       # (arch, T) -> [r]
    l0 = defaultdict(list)              # (arch, T) -> [l0_per_token]
    for r in shard_rows:
        key = (r["arch"], r["T"])
        m = r["metrics"]
        if r.get("kind") == "untrained":
            untrained[key].append(m.get("lambda_recovery", float("nan")))
        else:
            l0[key].append(m.get("l0_per_token", float("nan")))

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

    for seed in seeds:
        for field, ls in (("recomputed_r", "-"), ("r_shuf", "--")):
            ys = [at[(T, seed)][field] for T in Ts]
            ax.plot(Ts, ys, ls, color=TXC, alpha=0.25, lw=1, zorder=1)

    for field, ls, mk, mfc, label in (
            ("recomputed_r", "-", "o", None, f"ordered ({l0_tag})"),
            ("r_shuf", "--", "s", "white", "within-window shuffled")):
        mu = [np.mean([at[(T, s)][field] for s in seeds]) for T in Ts]
        sd = [np.std([at[(T, s)][field] for s in seeds], ddof=1) for T in Ts]
        ax.plot(Ts, mu, ls, color=TXC, lw=2, marker=mk, ms=6,
                mfc=mfc or TXC, mec=TXC, label=label, zorder=3)
        ax.errorbar(Ts, mu, yerr=sd, color=TXC, capsize=3, lw=1.2,
                    fmt="none", zorder=2)

    # untrained twins — the § 2 control (in the quoted-ticks slot of the
    # template; no quoted panel exists on a first training)
    un_mu = [float(np.nanmean(untrained.get((ARM, T), [np.nan])))
             for T in Ts]
    ax.plot(Ts, un_mu, ":", color=UNTR_C, lw=1.6, marker="o", ms=5,
            mfc="white", mec=UNTR_C, label="untrained twins", zorder=2)

    for arm, c, label in ANCHORS:
        vals = [c2["recomputed_r"] for c2 in payload["cells"]
                if c2["arch"] == arm]
        if vals:
            m, s = float(np.mean(vals)), float(np.std(vals, ddof=1))
            ax.axhspan(m - s, m + s, color=c, alpha=0.12, zorder=0)
            ax.axhline(m, color=c, lw=1, ls=":", zorder=0)
            ax.annotate(label, xy=(Ts[-1], m), fontsize=7, color=c,
                        ha="right", va="bottom")

    ax.annotate("anchors at T=1: shuffle ≡ identity (by construction)",
                xy=(0.03, 0.95), xycoords="axes fraction", ha="left",
                va="top", fontsize=8, color="#555555")
    cov = " ".join(
        f"T{T}:n={len([1 for s in seeds if (T, s) in at])}" for T in Ts)
    ax.annotate("no anchor gate BY DESIGN (first training; card § 1)\n"
                f"{cov} · shuffle seed {payload['shuf_eval_seed']}",
                xy=(0.99, 0.02), xycoords="axes fraction", ha="right",
                va="bottom", fontsize=6.5, color="#777777", zorder=5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="none", alpha=0.75))
    if payload.get("status") == "PENDING TEAM REVIEW":
        ax.annotate("PENDING TEAM REVIEW", xy=(0.5, 0.97),
                    xycoords="axes fraction", ha="center", va="top",
                    fontsize=7, color="#bb4444", alpha=0.8)

    ax.set_xscale("log", base=2)
    ax.set_xticks(Ts)
    ax.set_xticklabels([str(T) for T in Ts])
    ax.minorticks_off()
    ax.set_xlabel("T (window length)")
    ax.set_ylabel("recovery r (log2(1+challenge age))")
    ax.grid(True, alpha=0.25, lw=0.5)
    leg = ax.legend(frameon=True, framealpha=0.85, edgecolor="none",
                    fontsize=8, loc="upper left",
                    bbox_to_anchor=(0.02, 0.90))
    leg.set_zorder(5)
    fig.tight_layout()

    OUT_DIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig_sycgen_shuffle_tsweep.{ext}", dpi=200)

    summary = {
        "card": "sycgen/RETRAIN_CARD.md §§ 4–5",
        "status": payload.get("status"),
        "x_axis": [1] + Ts,
        "ordered": {f"T{T}": float(np.mean(
            [at[(T, s)]["recomputed_r"] for s in seeds])) for T in Ts},
        "shuffled": {f"T{T}": float(np.mean(
            [at[(T, s)]["r_shuf"] for s in seeds])) for T in Ts},
        "untrained": {f"{a}/T{t}": float(np.nanmean(v))
                      for (a, t), v in sorted(untrained.items())},
        "anchors": {a: float(np.mean(
            [c2["recomputed_r"] for c2 in payload["cells"]
             if c2["arch"] == a])) for a, _, _ in ANCHORS},
        "l0_range_claiming_arm": {"min": l0_lo, "max": l0_hi},
        "budget_matched": budget_matched,
        "match_rule": f"min cell-mean realized l0 >= k_pos/2 = {K_POS / 2}",
    }
    (RES / "sycgen_tsweep_summary.json").write_text(
        json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"[render] -> {OUT_DIR / 'fig_sycgen_shuffle_tsweep'}.{{png,pdf}} "
          f"+ {RES / 'sycgen_tsweep_summary.json'}")


if __name__ == "__main__":
    main()
