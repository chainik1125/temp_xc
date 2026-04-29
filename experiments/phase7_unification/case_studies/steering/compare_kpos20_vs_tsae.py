"""Compare TXC kpos20 cells vs T-SAE k=20 anchor under family-normalised paper-clamp.

Loads grades.jsonl for: tsae_paper_k20 (anchor) + each TXC kpos20 cell
(passed via --archs). Computes per-arch:
  - peak success across all s_norm (no coh constraint)
  - peak success at coh ≥ 1.5 (the locked primary metric)
  - per-concept-class breakdown (knowledge / discourse / safety / stylistic / sentiment)
  - per-strength curves: success(s_norm), coh(s_norm)

Calls the ±0.27 threshold rule against T-SAE k=20's pooled mean (1.80) and
prints the outcome (win / tie / loss) per pre-registered rule.

Output:
  results/case_studies/plots/kpos20_vs_tsae_curves.png      — success/coh curves
  results/case_studies/plots/kpos20_vs_tsae_concept_class.png — per-class delta
  results/case_studies/plots/kpos20_vs_tsae_summary.json    — call-the-outcome table

Run:
  TQDM_DISABLE=1 .venv/bin/python -m \\
      experiments.phase7_unification.case_studies.steering.compare_kpos20_vs_tsae \\
      --archs txc_bare_antidead_t5_kpos20
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

from experiments.phase7_unification._paths import OUT_DIR, banner
from experiments.phase7_unification.case_studies._paths import CASE_STUDIES_DIR


CONCEPT_CLASSES = {
    "knowledge / domain":   {"medical", "mathematical", "historical",
                             "code_context", "scientific", "religious",
                             "geographical", "financial", "programming"},
    "discourse / register": {"dialogue", "casual_register", "formal_register",
                             "imperative_form", "instructional", "question_form",
                             "narrative", "neutral_factual"},
    "safety / alignment":   {"harmful_content", "deception", "refusal_pattern",
                             "jailbreak_pattern", "helpfulness_marker", "legal"},
    "stylistic":            {"poetic", "literary", "list_format", "citation_pattern",
                             "technical_jargon"},
    "sentiment":            {"positive_emotion", "negative_emotion"},
}

ANCHOR = "tsae_paper_k20"
# T-SAE k=20 has TWO defensible anchor numbers:
#   * unconstrained peak: 1.80 success at coh=1.40 (s_abs=99.8 = s_norm=10)
#   * peak at coh ≥ 1.5:  1.10 success at coh=1.667 (s_abs=49.9 = s_norm=5)
# The previous Y reported 1.80 (unconstrained). The brief locked in "peak
# success at coh ≥ 1.5" as primary — that gives 1.10 as anchor. We compute
# both and call outcomes under each; Han picks framing.
ANCHOR_PEAK_UNCONSTR = 1.80             # previous Y headline number
ANCHOR_PEAK_CONSTR_15 = 1.10            # anchor under coh ≥ 1.5 constraint
THRESHOLD = 0.27                        # 1× σ_seeds


def _load_grades(arch_id: str, subdir: str) -> list[dict]:
    p = CASE_STUDIES_DIR / subdir / arch_id / "grades.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open()]


def _summarise(rows: list[dict]) -> dict:
    """For one arch, group by s_norm. Per-strength compute mean success + mean coh
    over concepts. Then peak success (unconstrained) + peak success at coh ≥ 1.5.
    """
    by_s = collections.defaultdict(list)
    for r in rows:
        if r.get("success_grade") is None:
            continue
        s = r.get("s_norm")
        if s is None:
            s = float(r.get("strength", 0))
        by_s[float(s)].append(r)

    if not by_s:
        return {"n_rows": 0}

    per_s = {}
    for s, pairs in sorted(by_s.items()):
        ss = [p["success_grade"] for p in pairs if p.get("success_grade") is not None]
        cs = [p.get("coherence_grade") for p in pairs if p.get("coherence_grade") is not None]
        per_s[s] = {
            "success_mean": float(np.mean(ss)) if ss else None,
            "coh_mean": float(np.mean(cs)) if cs else None,
            "n": len(pairs),
        }

    # peak unconstrained
    s_vals = sorted(per_s.keys())
    succ_curve = [per_s[s]["success_mean"] for s in s_vals]
    coh_curve = [per_s[s]["coh_mean"] for s in s_vals]
    peak_unconstr = max(succ_curve) if succ_curve else None
    peak_unconstr_s = s_vals[int(np.argmax(succ_curve))] if succ_curve else None

    # peak at coh >= 1.5
    valid = [(s, per_s[s]["success_mean"]) for s in s_vals
             if per_s[s].get("coh_mean") is not None and per_s[s]["coh_mean"] >= 1.5]
    peak_constr = max((v for _, v in valid), default=None)
    peak_constr_s = next((s for s, v in valid if v == peak_constr), None) if peak_constr is not None else None

    # Boundary check: are peaks at grid edges?
    if s_vals:
        boundary_warn = []
        if peak_unconstr_s == s_vals[-1]:
            boundary_warn.append(f"unconstr peak at top of grid s_norm={s_vals[-1]} — extend to 200")
        if peak_unconstr_s == s_vals[0]:
            boundary_warn.append(f"unconstr peak at bottom of grid s_norm={s_vals[0]} — extend to 0.05")
    else:
        boundary_warn = []

    # per-concept rows at peak constrained s_norm
    per_concept_peak = {}
    if peak_constr_s is not None:
        for r in by_s[peak_constr_s]:
            per_concept_peak[r["concept_id"]] = r["success_grade"]

    return {
        "n_rows": len(rows),
        "per_s": per_s,
        "peak_unconstr": peak_unconstr,
        "peak_unconstr_s": peak_unconstr_s,
        "peak_constr": peak_constr,
        "peak_constr_s": peak_constr_s,
        "boundary_warn": boundary_warn,
        "per_concept_peak": per_concept_peak,
    }


def _per_class(per_concept: dict[str, float]) -> dict[str, float]:
    out = {}
    for cls, cset in CONCEPT_CLASSES.items():
        scores = [per_concept[c] for c in cset if c in per_concept]
        out[cls] = float(np.mean(scores)) if scores else float("nan")
    return out


def _call_outcome(value: float | None, anchor: float, thresh: float = THRESHOLD,
                  metric_name: str = "metric") -> dict:
    if value is None:
        return {"call": "no_data", "delta": None, "metric": metric_name, "anchor": anchor}
    delta = value - anchor
    if delta >= thresh:
        return {"call": "WIN", "delta": delta, "metric": metric_name, "anchor": anchor,
                "next_step": "verify at seed=1; if confirmed → headline; run Steps 3-5"}
    if delta <= -thresh:
        return {"call": "LOSS", "delta": delta, "metric": metric_name, "anchor": anchor,
                "next_step": "Steps 3-5 as failure-mode investigation"}
    return {"call": "TIE", "delta": delta, "metric": metric_name, "anchor": anchor,
            "next_step": "verify at seed=1; if still tied → 'sparsity is sole lever' narrative"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", required=True,
                    help="kpos20 arch_ids to compare against T-SAE k=20 anchor")
    ap.add_argument("--subdir", default="steering_paper_normalised")
    ap.add_argument("--out-prefix", default=str(OUT_DIR / "case_studies" / "plots" / "kpos20_vs_tsae"))
    args = ap.parse_args()
    banner(__file__)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for arch in [ANCHOR] + list(args.archs):
        rows = _load_grades(arch, args.subdir)
        s = _summarise(rows)
        s["arch_id"] = arch
        s["per_class"] = _per_class(s.get("per_concept_peak", {}))
        summaries[arch] = s

    # ────────── outcome tables (two metrics; report both — Han picks framing)
    print("\n## Outcome under METRIC A: peak success unconstrained "
          f"(anchor {ANCHOR}={ANCHOR_PEAK_UNCONSTR:.2f}, ±{THRESHOLD} threshold)\n")
    print(f"{'arch':40s} {'peak':>8s} {'@s_norm':>9s} {'coh@peak':>10s} {'Δ':>8s} {'call':>8s}")
    for arch in [ANCHOR] + list(args.archs):
        s = summaries[arch]
        peak_u = s.get("peak_unconstr")
        peak_u_s = s.get("peak_unconstr_s")
        coh_at_peak = (s.get("per_s", {}).get(peak_u_s, {}).get("coh_mean")
                       if peak_u_s is not None else None)
        outcome_u = _call_outcome(peak_u, ANCHOR_PEAK_UNCONSTR, metric_name="unconstrained_peak")
        peak_v = f"{peak_u:.3f}" if peak_u is not None else "—"
        s_v = f"{peak_u_s:.2f}" if peak_u_s is not None else "—"
        coh_v = f"{coh_at_peak:.3f}" if coh_at_peak is not None else "—"
        delta_s = f"{outcome_u['delta']:+.3f}" if outcome_u.get('delta') is not None else "—"
        print(f"{arch:40s} {peak_v:>8s} {s_v:>9s} {coh_v:>10s} {delta_s:>8s} {outcome_u['call']:>8s}")

    print(f"\n## Outcome under METRIC B: peak success at coh ≥ 1.5 "
          f"(anchor {ANCHOR}={ANCHOR_PEAK_CONSTR_15:.2f}, ±{THRESHOLD} threshold) "
          f"[brief's locked primary]\n")
    print(f"{'arch':40s} {'peak':>8s} {'@s_norm':>9s} {'Δ':>8s} {'call':>8s}")
    out_table = {}
    for arch in [ANCHOR] + list(args.archs):
        s = summaries[arch]
        peak_c = s.get("peak_constr")
        peak_c_s = s.get("peak_constr_s")
        outcome_c = _call_outcome(peak_c, ANCHOR_PEAK_CONSTR_15, metric_name="constrained_coh_ge_1.5")
        peak_v = f"{peak_c:.3f}" if peak_c is not None else "—"
        s_v = f"{peak_c_s:.2f}" if peak_c_s is not None else "—"
        delta_s = f"{outcome_c['delta']:+.3f}" if outcome_c.get('delta') is not None else "—"
        print(f"{arch:40s} {peak_v:>8s} {s_v:>9s} {delta_s:>8s} {outcome_c['call']:>8s}")

        peak_u = s.get("peak_unconstr")
        peak_u_s = s.get("peak_unconstr_s")
        coh_at_peak = (s.get("per_s", {}).get(peak_u_s, {}).get("coh_mean")
                       if peak_u_s is not None else None)
        outcome_u = _call_outcome(peak_u, ANCHOR_PEAK_UNCONSTR, metric_name="unconstrained_peak")
        out_table[arch] = {
            **s,
            "outcome_unconstr": outcome_u,
            "outcome_constr_coh15": outcome_c,
            "coh_at_unconstr_peak": coh_at_peak,
        }
        if s.get("boundary_warn"):
            for w in s["boundary_warn"]:
                print(f"      ⚠ {w}")

    # ────────── per-class breakdown
    print("\n## Per-class peak-strength means (at constrained peak s_norm)\n")
    classes = list(CONCEPT_CLASSES.keys())
    print(f"{'arch':40s} " + "  ".join(f"{c:>20s}" for c in classes))
    for arch in [ANCHOR] + list(args.archs):
        cls = summaries[arch]["per_class"]
        cells = "  ".join(f"{cls[c]:>20.3f}" if not np.isnan(cls.get(c, float('nan')))
                          else f"{'—':>20s}" for c in classes)
        print(f"{arch:40s} {cells}")

    # ────────── plot 1: success(s_norm) and coh(s_norm) curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for arch in [ANCHOR] + list(args.archs):
        per_s = summaries[arch].get("per_s", {})
        if not per_s:
            continue
        s_vals = sorted(per_s.keys())
        succ = [per_s[s]["success_mean"] for s in s_vals]
        coh = [per_s[s]["coh_mean"] for s in s_vals]
        axes[0].plot(s_vals, succ, marker="o", label=arch)
        axes[1].plot(s_vals, coh,  marker="o", label=arch)
    axes[0].set_xscale("log"); axes[1].set_xscale("log")
    axes[0].set_xlabel("s_norm (× ⟨|z|⟩_arch)"); axes[1].set_xlabel("s_norm")
    axes[0].set_ylabel("mean success"); axes[1].set_ylabel("mean coherence")
    axes[1].axhline(1.5, color="grey", linestyle="--", linewidth=0.8)
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3); axes[1].grid(True, alpha=0.3)
    fig.suptitle("Hail Mary kpos20 vs T-SAE k=20 (family-normalised paper-clamp, seed=42)")
    plt.tight_layout()
    curves_png = out_prefix.with_name(out_prefix.name + "_curves.png")
    fig.savefig(curves_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_prefix.with_name(out_prefix.name + "_curves.thumb.png"), dpi=48, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {curves_png}")

    # ────────── plot 2: per-class bar
    fig, ax = plt.subplots(figsize=(10, 4.5))
    archs = [ANCHOR] + list(args.archs)
    x = np.arange(len(classes))
    width = 0.8 / max(1, len(archs))
    for i, arch in enumerate(archs):
        cls = summaries[arch]["per_class"]
        vals = [cls.get(c, float("nan")) for c in classes]
        ax.bar(x + i * width, vals, width, label=arch)
    ax.set_xticks(x + width * (len(archs)-1) / 2)
    ax.set_xticklabels(classes, rotation=15, ha="right")
    ax.set_ylabel("mean success @ peak coh≥1.5 s_norm")
    ax.set_title("Per-concept-class breakdown — kpos20 vs T-SAE k=20 anchor")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    cls_png = out_prefix.with_name(out_prefix.name + "_concept_class.png")
    fig.savefig(cls_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_prefix.with_name(out_prefix.name + "_concept_class.thumb.png"), dpi=48, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {cls_png}")

    # ────────── summary JSON
    summary_json = out_prefix.with_name(out_prefix.name + "_summary.json")
    summary_json.write_text(json.dumps(out_table, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else None))
    print(f"wrote {summary_json}")


if __name__ == "__main__":
    main()
