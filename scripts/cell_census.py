"""Rebuttal cell census — every leaderboard cell, labeled by arm.

Generates REBUTTAL_CELL_CENSUS.md: one line per (experiment, arch, T)
cell group listing seeds / k_feat / shuffle coverage / row counts, with
the binding arm taxonomy (paper-faithful vs {BatchTopK} btk-only vs the
misinterpreted relu-mix arm). Companion to REBUTTAL_CODE_GUIDE.md §1b.

Usage:
    .venv/bin/python scripts/cell_census.py            # print to stdout
    .venv/bin/python scripts/cell_census.py --write    # refresh REBUTTAL_CELL_CENSUS.md
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEADERBOARD = ROOT / "results" / "leaderboard.jsonl"
OUT = ROOT / "REBUTTAL_CELL_CENSUS.md"

# The rebuttal deliverable scope; 'synthetic' is the paper-era synthetic
# program and is summarized in one line, never listed per-cell.
SCOPE = ("probing", "rlhf", "em")

# arch -> (arm label, note). Taxonomy pinned by Han 07-28: {ReLU+TopK}
# paper-faithful vs {BatchTopK} = btk-only (NO ReLU) vs relu-mix = the
# MISINTERPRETED "paper-faithful" arm (ReLU present, composition != paper;
# certificate evidence only, never a matrix column).
ARM = {
    "paper_txc_base_v1t": ("{ReLU+TopK} PAPER-FAITHFUL (trained)", "the sprint arm — paper §5.1 composition, trainable"),
    "paper_txc_base_v1": ("paper-faithful ANCHOR (eval-only)", "archived paper ckpts, T=5 only"),
    "paper_topk_sae_v1": ("paper-faithful ANCHOR (eval-only)", "archived paper SAE baseline, T=5 only"),
    "paper_tsae_v1": ("paper-faithful ANCHOR (eval-only)", "archived paper T-SAE baseline, T=5 only"),
    "txc_batchtopk_pre_btkonly": ("{BatchTopK} btk-only", "NO ReLU — the delivered probing sweep arm"),
    "txc_batchtopk_post_btkonly": ("{BatchTopK} btk-only", "NO ReLU — the delivered RLHF sweep arm"),
    "batchtopk_sae_btkonly": ("{BatchTopK} btk-only BASELINE", "SAE baseline, no ReLU"),
    "tsae_btkonly": ("{BatchTopK} btk-only BASELINE", "T-SAE baseline (matryoshka+contrastive is Ye et al.'s design)"),
    "txc_batchtopk_pre": ("relu-mix (MISINTERPRETED arm)", 'the arm formerly mislabeled "{ReLU+TopK} paper-faithful" — certificate evidence ONLY'),
    "txc_batchtopk_post": ("relu-mix (MISINTERPRETED arm)", 'the arm formerly mislabeled "{ReLU+TopK} paper-faithful" — certificate evidence ONLY'),
    "batchtopk_sae": ("relu-mix BASELINE", "ReLU-bearing v2 SAE baseline"),
    "tsae": ("relu-mix BASELINE", "ReLU-bearing v2 T-SAE baseline"),
    "txc_base": ("paper-era plain TXC", "legacy arch id (em section)"),
}


def load_rows():
    rows = []
    with open(LEADERBOARD) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def cell_t(row):
    t = ((row.get("training_cfg") or {}).get("arch_hparams_override") or {}).get("T")
    return t if t is not None else "—"


def fmt_set(values):
    vals = sorted(values, key=lambda v: (isinstance(v, str), v))
    return "{" + ",".join(str(v) for v in vals) + "}" if vals else "—"


def main(write: bool) -> None:
    rows = load_rows()
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    ts = subprocess.run(["date", "-u", "+%Y-%m-%d %H:%M UTC"], capture_output=True, text=True).stdout.strip()

    in_scope = [r for r in rows if r.get("experiment") in SCOPE]
    synth = sum(1 for r in rows if r.get("experiment") == "synthetic")
    other = len(rows) - len(in_scope) - synth

    # (experiment, arch, T) -> aggregates
    groups = defaultdict(lambda: {
        "seeds": set(), "untrained_seeds": set(), "k_feat": set(),
        "shuffle": set(), "has_shuffle_cfg": False, "arms_stamped": set(),
        "n_rows": 0, "n_smoke": 0, "n_poscontrol": 0,
        "has_shuf_metrics": False, "d_sae_over": defaultdict(int),
    })
    for r in in_scope:
        ec = r.get("eval_cfg") or {}
        tc = r.get("training_cfg") or {}
        metrics = r.get("metrics") or {}
        g = groups[(r["experiment"], r.get("datasource"), r.get("arch"), cell_t(r))]
        g["n_rows"] += 1
        if ec.get("smoke"):
            g["n_smoke"] += 1
            continue  # smoke rows never count as coverage
        if ec.get("positive_control"):
            g["n_poscontrol"] += 1
        if tc.get("n_steps") == 0:
            g["untrained_seeds"].add(r.get("seed"))
        else:
            g["seeds"].add(r.get("seed"))
        # k budget: eval_cfg.k_feat (probing 1.2.x) OR metric-name-encoded
        # (rlhf 2.0.0 stores preference_auc_k20/k50 etc. with empty eval_cfg)
        if ec.get("k_feat") is not None:
            g["k_feat"].add(ec["k_feat"])
        for mk in metrics:
            m = re.match(r"(?:preference_auc|auc)_k(\d+)$", mk)
            if m:
                g["k_feat"].add(int(m.group(1)))
        if "shuffle" in ec:
            g["has_shuffle_cfg"] = True
            g["shuffle"].add(ec.get("shuffle") or "none")
        if ec.get("arm"):
            g["arms_stamped"].add(ec["arm"])
        if any("shuf" in mk for mk in metrics):
            g["has_shuf_metrics"] = True
        ds = (tc.get("arch_hparams_override") or {}).get("d_sae")
        if ds is not None:
            g["d_sae_over"][ds] += 1

    lines = []
    lines.append("# REBUTTAL CELL CENSUS — every cell we have results for, by arm")
    lines.append("")
    lines.append(f"_Generated by `scripts/cell_census.py --write` at {ts}, HEAD `{head}`,")
    lines.append(f"{len(rows)} leaderboard rows ({len(in_scope)} in rebuttal scope: probing/rlhf/em;")
    lines.append(f"{synth} paper-era `experiment=synthetic` rows and {other} other rows excluded from per-cell listing)._")
    lines.append("_Regenerate any time — the leaderboard is append-only, so this file dates fast during the sprint._")
    lines.append("")
    lines.append("## Arm taxonomy (binding, Han 07-28)")
    lines.append("")
    lines.append("- **{ReLU+TopK} PAPER-FAITHFUL** = `paper_txc_base_v1t` (trained, the sprint) and the")
    lines.append("  archived-ckpt eval-only anchors `paper_*_v1` (T=5 only). Composition:")
    lines.append("  `z = ReLU(TopK_{k_pos·T}(Σ_t p_t))` — exact per-window k, rectify AFTER selection.")
    lines.append("- **{BatchTopK}** = the `*_btkonly` archs — **NO ReLU** in the sparsity path (signed")
    lines.append("  selection). These are the delivered T-sweep exhibits.")
    lines.append("- **relu-mix = the MISINTERPRETED \"{ReLU+TopK} paper-faithful\" arm** (`txc_batchtopk_pre`,")
    lines.append("  `txc_batchtopk_post`, non-btkonly baselines): ReLU present but BEFORE a BatchTopK")
    lines.append("  batch-budget selection — **not** the paper composition. Kept as equivalence-certificate")
    lines.append("  evidence ONLY; never quote these cells as paper-faithful and never as a matrix column.")
    lines.append("- Untrained-twin cells (`n_steps=0`) are certificate/alias cells — excluded from sweep")
    lines.append("  aggregations (alias list: `experiments/probing/actmix/RM_EQUIVALENCE.md`). Smoke rows")
    lines.append("  are counted but never coverage. `positive_control` rows are instrument-gate cells.")
    lines.append("")

    for exp in SCOPE:
        exp_groups = {k: v for k, v in groups.items() if k[0] == exp}
        if not exp_groups:
            continue
        n_exp = sum(g["n_rows"] for g in exp_groups.values())
        lines.append(f"## experiment = `{exp}` ({n_exp} rows)")
        lines.append("")
        lines.append("| arch | datasource | arm | T | trained seeds | untrained-twin seeds | k_feat | shuffle cfgs | in-row shuf metrics | rows | notes |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        def sort_key(item):
            (_, dsrc, arch, t), _ = item
            arm = ARM.get(arch, ("~unknown", ""))[0]
            return (str(dsrc), arm, str(arch), (0, t) if isinstance(t, int) else (1, 0))
        for (_, dsrc, arch, t), g in sorted(exp_groups.items(), key=sort_key):
            arm, note = ARM.get(arch, ("⚠ UNMAPPED — classify before quoting", ""))
            extras = []
            if g["n_smoke"]:
                extras.append(f"{g['n_smoke']} smoke")
            if g["n_poscontrol"]:
                extras.append(f"{g['n_poscontrol']} positive-control")
            for ds, n in sorted(g["d_sae_over"].items()):
                extras.append(f"d_sae_override={ds} on {n} rows")
            if g["arms_stamped"]:
                extras.append(f"eval_cfg.arm={fmt_set(g['arms_stamped'])}")
            note_full = "; ".join(x for x in [note] + extras if x)
            shuffle_cell = fmt_set(g["shuffle"]) if g["has_shuffle_cfg"] else "—"
            lines.append(
                f"| `{arch}` | `{dsrc}` | {arm} | {t} | {fmt_set(g['seeds'])} | {fmt_set(g['untrained_seeds'])} "
                f"| {fmt_set(g['k_feat'])} | {shuffle_cell} | {'yes' if g['has_shuf_metrics'] else '—'} "
                f"| {g['n_rows']} | {note_full} |"
            )
        lines.append("")

    # hunted-task cells that do NOT live in the canonical leaderboard
    lines.append("## Hunted-task cells outside the leaderboard (items 4–7)")
    lines.append("")
    lines.append("The hunted-task shuffle-overlay cells are eval_extra-namespaced results")
    lines.append("files under `experiments/explorations/task_hunt/<dir>/results/`, NOT")
    lines.append("leaderboard rows (arm per each dir's frozen CARD; hunted tasks need")
    lines.append("either arm only — Han 02:38):")
    lines.append("")
    lines.append("- **λ̂ backtracking-intensity (item 4)** — `task_hunt/sc_lambda/`")
    lines.append("  (CARD.md, results/): anchor-gated shuffle-overlay retrains, 18 cells")
    lines.append("  per the frozen card + 6-cell anchor gate; fig")
    lines.append("  `figs_writeup/fig_lambda_shuffle_tsweep.*`.")
    lines.append("- **dq question-marks (item 5)** — `task_hunt/diafaces/`")
    lines.append("  (DQ_T_FILL_CARD.md, results/): fills T{6,10}×3 per the DQ_T_FILL")
    lines.append("  freeze; exhibit set per REBUTTAL_HANDOFF.md §5 (figs")
    lines.append("  `fig_ttrend_shuffle_tsweep.*`, `task_hunt/figs_writeup/")
    lines.append("  fig2_question_gap_tscaling.*`). TOY-class, screen-shuffle disclosed.")
    lines.append("- **sycgen (item 6)** — `task_hunt/sycgen/`: screens KEEP 3/3 (LOG 02:28,")
    lines.append("  `sycgen/results/*.json`); the matrix retrain (36 cells, T{1,2,4,8,16}")
    lines.append("  ≡ the λ̂ exhibit axis — T{6,10} can't tile eval L=32, card §5")
    lines.append("  amendment LOG 02:54) runs on the canonical runner; its rows appear in")
    lines.append("  the probing table above under")
    lines.append("  `datasource=sycgen_real_age_llama31_8b_l14` when they land —")
    lines.append("  regenerate this file.")
    lines.append("- **evalage (item 7 candidate)** — `task_hunt/evalage/`: corpus complete")
    lines.append("  (2.04M tokens) + 6/6 label-side bands; screen pending re-tokenization —")
    lines.append("  no retrain cells yet.")
    lines.append("")

    # cross-check: arch-derived arm vs stamped eval_cfg.arm
    mismatches = []
    for (exp, dsrc, arch, t), g in groups.items():
        arm = ARM.get(arch, ("?",))[0]
        for stamped in g["arms_stamped"]:
            ok = (
                ("btk-only" in arm and stamped == "btk-only")
                or ("relu-mix" in arm and stamped == "relu-mix")
                or ("PAPER-FAITHFUL" in arm and stamped == "paper-faithful")
                or ("ANCHOR" in arm and stamped in ("paper-faithful", "paper-match"))
            )
            if not ok:
                mismatches.append(f"- `{exp}`/`{arch}`/T={t}: arch-derived arm “{arm}” vs stamped `eval_cfg.arm={stamped}`")
    lines.append("## Arch-vs-stamp cross-check")
    lines.append("")
    if mismatches:
        lines.append("**MISMATCHES — resolve before quoting these cells:**")
        lines.extend(sorted(set(mismatches)))
    else:
        lines.append("No arch-derived vs `eval_cfg.arm` stamp mismatches.")
    lines.append("")

    out = "\n".join(lines)
    if write:
        OUT.write_text(out)
        print(f"wrote {OUT} ({len(out.splitlines())} lines)")
    else:
        print(out)


if __name__ == "__main__":
    main(write="--write" in sys.argv[1:])
