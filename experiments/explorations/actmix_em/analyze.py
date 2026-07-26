"""ACTMIX P2 analysis — the Dmitry table + mechanical E/K scoring.

Reads ONLY the canonical leaderboard (results/leaderboard.jsonl),
selects this card's cells (experiment em, agent runpod-2, medical-L15
datasource, *_btkonly archs, non-smoke), and emits:

- results/table.json — every cell + margins + E1–E5/K1–K3 scoring
  (the verdict inputs are mechanical outputs of this script, per
  CARD § 5; the LOG verdict quotes them).
- results/table.md — the human-readable exhibit table (both seeds +
  mean), realized l0 beside every entry, untrained floors, and the
  frozen side-by-side block with the paper's published numbers +
  the three frozen caveats.

Selection is by configuration signature, not freeze sha (this run's
launch amendments moved HEAD between lanes; the wall logs +
table.json record the pins per cell). Duplicate eval_keys collapse
(runner cache-hits re-observed on relaunch).

Run: .venv/bin/python -m experiments.explorations.actmix_em.analyze
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LEADERBOARD = ROOT / "results" / "leaderboard.jsonl"
OUT_DIR = HERE / "results"

DS = "qwen_2_5_7b_instruct_medical_l15"
TXC, SAE, TSAE = ("txc_batchtopk_post_btkonly", "batchtopk_sae_btkonly",
                  "tsae_btkonly")
T_GRID = (1, 2, 4, 8, 16)
SEEDS = (42, 1)
K_PER_TOKEN = 20

# Paper anchors (CARD § 5.4 — origin/final leaderboard, protocol
# 3.0.0, L15, per-cell Wang cohorts):
PAPER = {
    "sae_arditi": {"pr_auc_S16": {42: 0.690, 1: 0.745}, "k_per_token": 128},
    "txc_base": {"pr_auc_S16": {42: 0.542, 1: 0.560},
                 "shuffle_gap_S16": {42: -0.059, 1: -0.002},
                 "k_pos": 25, "T": 5},
    "caveats": [
        "(a) paper cells probed PER-CELL Wang cohorts (n_sent 79k-107k, "
        "base rates 0.32-0.47) vs this run's ONE fixed 1728-rollout "
        "cohort (0.323); PR-AUC is base-rate sensitive - cross-design "
        "deltas are context, not measurements",
        "(b) budgets differ (arditi 128/token vs panel 20/token; "
        "txc_base k_pos 25, T 5 paper knobs)",
        "(c) composition differs BY DESIGN - that is the ablation",
    ],
}


def load_cells() -> dict:
    cells = {}
    with LEADERBOARD.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (r.get("experiment") != "em" or r.get("agent") != "runpod-2"
                    or r.get("datasource") != DS):
                continue
            if (r.get("eval_cfg") or {}).get("smoke"):
                continue
            arch = r.get("arch")
            if arch not in (TXC, SAE, TSAE):
                continue
            hp = (r.get("training_cfg") or {}).get("arch_hparams_override") or {}
            T = int(hp.get("T", 1))
            kind = "trained" if (r["training_cfg"]["n_steps"] or 0) else "untrained"
            key = (arch, T, kind, int(r["seed"]))
            if key in cells and cells[key]["eval_key"] != r["eval_key"]:
                raise RuntimeError(f"conflicting duplicate for {key}")
            m = r["metrics"]
            cells[key] = {
                "eval_key": r["eval_key"], "train_key": r["train_key"],
                "commit_sha": r["code_version"]["commit_sha"],
                "pr_auc_S16": m.get("pr_auc_S16"),
                "pr_auc_shuffled_S16": m.get("pr_auc_shuffled_S16"),
                "shuffle_gap_S16": m.get("shuffle_gap_S16"),
                "l0_per_token": m.get("l0_per_token"),
                "l0_per_window": m.get("l0_per_window"),
                "n_sent": m.get("n_sent"),
                "positive_rate": m.get("positive_rate"),
                "n_rollouts": m.get("n_rollouts"),
                "all_S": {S: m.get(f"pr_auc_S{S}") for S in (1, 2, 4, 8, 16, 32)},
                "all_S_shuffled": {S: m.get(f"pr_auc_shuffled_S{S}")
                                   for S in (1, 2, 4, 8, 16, 32)},
            }
    return cells


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def score(cells: dict) -> dict:
    """Mechanical E1–E5 / K1–K3 scoring per CARD § 4 (None = not yet
    scorable from landed cells)."""
    def g(arch, T, kind, seed, field):
        c = cells.get((arch, T, kind, seed))
        return None if c is None else c[field]

    out = {"scored_at_cells": len(cells)}

    # K3 cohort integrity — every landed cell.
    bad = [k for k, c in cells.items()
           if c["n_rollouts"] not in (None, 1728.0)
           or (c["positive_rate"] is not None
               and abs(c["positive_rate"] - 0.323) > 0.001)]
    out["K3_cohort_integrity"] = {"violations": [str(k) for k in bad],
                                  "pass": not bad}

    # K1 machinery falsifier — trained sae, either seed.
    sae_vals = [g(SAE, 1, "trained", s, "pr_auc_S16") for s in SEEDS]
    sae_vals = [v for v in sae_vals if v is not None]
    out["K1_sae_falsifier"] = {
        "sae_pr_auc_S16": sae_vals,
        "pass": (min(sae_vals) >= 0.40) if sae_vals else None}

    # K2 under-realization — btk-only cells at eval; nominal 20/token.
    k2 = {}
    for (arch, T, kind, seed), c in cells.items():
        if kind != "trained" or c["l0_per_token"] is None:
            continue
        k2[f"{arch}/T{T}/s{seed}"] = {
            "l0_per_token": c["l0_per_token"],
            "under_realized": c["l0_per_token"] < 0.75 * K_PER_TOKEN}
    out["K2_under_realization"] = {
        "cells": k2,
        "pass": (not any(v["under_realized"] for v in k2.values()))
                if k2 else None}

    # E1 headline: txc vs sae band at every T (mean over landed seeds).
    e1 = {}
    for T in T_GRID:
        txc = _mean([g(TXC, T, "trained", s, "pr_auc_S16") for s in SEEDS])
        sae = _mean([g(SAE, 1, "trained", s, "pr_auc_S16") for s in SEEDS])
        if txc is not None and sae is not None:
            e1[T] = {"txc": txc, "sae": sae, "delta": txc - sae,
                     "txc_beats": txc > sae}
    out["E1_negative_persists"] = {
        "per_T": e1,
        "holds": (not any(v["txc_beats"] for v in e1.values()))
                 if e1 else None}

    # E2 shuffle gap < +0.02 everywhere (T>1).
    e2 = {}
    for T in T_GRID:
        if T == 1:
            continue
        gap = _mean([g(TXC, T, "trained", s, "shuffle_gap_S16")
                     for s in SEEDS])
        if gap is not None:
            e2[T] = {"gap_mean": gap, "below_bar": gap < 0.02}
    out["E2_shuffle_below_bar"] = {
        "per_T": e2,
        "holds": all(v["below_bar"] for v in e2.values()) if e2 else None}

    # E4 T=1 limit.
    d = None
    txc1 = _mean([g(TXC, 1, "trained", s, "pr_auc_S16") for s in SEEDS])
    sae1 = _mean([g(SAE, 1, "trained", s, "pr_auc_S16") for s in SEEDS])
    if txc1 is not None and sae1 is not None:
        d = txc1 - sae1
    out["E4_t1_limit"] = {"txc_T1": txc1, "sae": sae1, "delta": d,
                          "holds": (abs(d) <= 0.03) if d is not None else None}

    # E5 untrained floors.
    e5 = {}
    for (arch, T, kind, seed), c in cells.items():
        if kind == "untrained" and c["pr_auc_S16"] is not None:
            e5[f"{arch}/T{T}"] = c["pr_auc_S16"]
    out["E5_untrained_floors"] = e5
    return out


def render_md(cells: dict, scores: dict) -> str:
    L = ["# ACTMIX P2 — EM btk-only: the exhibit table",
         "",
         f"Datasource `{DS}` (BASE-forward train, organism cohort "
         "detect, L15). Arm: **btk-only** (arch-name suffix carries "
         "the arm; relu_mode hashes into train_key). Primary "
         "pr_auc_S16; positive-rate floor 0.323. Nominal budget "
         "20/token (txc: 20·T per window).", ""]

    def fmt(v, nd=3):
        return "—" if v is None else f"{v:.{nd}f}"

    for seed in SEEDS + ("mean",):
        L.append(f"## seed {seed}")
        L.append("")
        L.append("| T | TXC | TXC-shuffled | gap | SAE | TSAE | "
                 "TXC l0/tok | untrained TXC |")
        L.append("|---|---|---|---|---|---|---|---|")

        def gv(arch, T, kind, field):
            if seed == "mean":
                return _mean([cells.get((arch, T, kind, s), {}).get(field)
                              for s in SEEDS])
            return cells.get((arch, T, kind, seed), {}).get(field)

        for T in T_GRID:
            L.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                T,
                fmt(gv(TXC, T, "trained", "pr_auc_S16")),
                fmt(gv(TXC, T, "trained", "pr_auc_shuffled_S16")),
                fmt(gv(TXC, T, "trained", "shuffle_gap_S16")),
                fmt(gv(SAE, 1, "trained", "pr_auc_S16")),
                fmt(gv(TSAE, 1, "trained", "pr_auc_S16")),
                fmt(gv(TXC, T, "trained", "l0_per_token"), 1),
                fmt(gv(TXC, T, "untrained", "pr_auc_S16")),
            ))
        L.append("")
        sae_l0 = gv(SAE, 1, "trained", "l0_per_token")
        tsae_l0 = gv(TSAE, 1, "trained", "l0_per_token")
        L.append(f"SAE realized l0/tok {fmt(sae_l0, 1)}; TSAE "
                 f"{fmt(tsae_l0, 1)}; SAE/TSAE are per-token "
                 "(T-invariant bands; no within-window shuffle exists "
                 "at T = 1 — order-invariance holds by construction).")
        L.append("")

    L += ["## Side-by-side with the paper's published § 5.3 negative",
          "",
          "| cell | pr_auc_S16 s42 | s1 | shuffle_gap s42 | s1 |",
          "|---|---|---|---|---|",
          f"| paper sae_arditi (128/tok) | {PAPER['sae_arditi']['pr_auc_S16'][42]:.3f} "
          f"| {PAPER['sae_arditi']['pr_auc_S16'][1]:.3f} | — | — |",
          f"| paper txc_base (k25, T5) | {PAPER['txc_base']['pr_auc_S16'][42]:.3f} "
          f"| {PAPER['txc_base']['pr_auc_S16'][1]:.3f} "
          f"| {PAPER['txc_base']['shuffle_gap_S16'][42]:.3f} "
          f"| {PAPER['txc_base']['shuffle_gap_S16'][1]:.3f} |",
          ""]
    for c in PAPER["caveats"]:
        L.append(f"- {c}")
    L += ["", "## Mechanical scoring (CARD § 4)", "",
          "```json", json.dumps(scores, indent=1, default=str), "```", ""]
    return "\n".join(L)


def main():
    cells = load_cells()
    scores = score(cells)
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "table.json").write_text(json.dumps(
        {"cells": {"|".join(map(str, k)): v for k, v in cells.items()},
         "scores": scores, "paper_anchors": PAPER}, indent=1))
    (OUT_DIR / "table.md").write_text(render_md(cells, scores))
    print(f"[analyze] {len(cells)} cells -> {OUT_DIR}/table.{{json,md}}")
    for k in sorted(cells):
        c = cells[k]
        print(f"  {k}: pr={c['pr_auc_S16']} gap={c['shuffle_gap_S16']} "
              f"l0/tok={c['l0_per_token']}")


if __name__ == "__main__":
    main()
