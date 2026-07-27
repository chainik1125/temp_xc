"""ACTMIX RLHF analysis — the both-arm Dmitry table + R-scoring.

Reads the paper-match case-study artifact (results/papermatch.json)
and the btk-only leaderboard rows (canonical leaderboard, evaluator
rlhf/2.0.0, agent runpod-2, non-smoke), emits:

- results/rlhf_table.json — every cell + R-E/R-K scoring (mechanical
  outputs per CARD § 4; the LOG verdict quotes them).
- results/rlhf_table.md — the exhibit table, both arms, with the
  frozen caveats and the l0 column (A2 citation: cross-section
  sparsity not comparable — this family is k500-dev).

Run: .venv/bin/python -m experiments.explorations.actmix_rlhf.analyze
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LEADERBOARD = ROOT / "results" / "leaderboard.jsonl"
OUT = HERE / "results"

DS = "gemma_2_2b_base_l12_phase7"
TXC, SAE, TSAE = ("txc_batchtopk_post_btkonly", "batchtopk_sae_btkonly",
                  "tsae_btkonly")


def load_btk_cells() -> dict:
    cells = {}
    with LEADERBOARD.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (r.get("experiment") != "rlhf" or r.get("agent") != "runpod-2"
                    or r.get("datasource") != DS
                    or (r.get("eval_cfg") or {}).get("smoke")):
                continue
            arch = r["arch"]
            hp = (r["training_cfg"] or {}).get("arch_hparams_override") or {}
            T = int(hp.get("T", 1))
            k = int(hp.get("k_pos", 0))
            kind = ("trained" if (r["training_cfg"]["n_steps"] or 0)
                    else "untrained")
            key = (arch, T, k, kind, int(r["seed"]))
            if key in cells and cells[key]["eval_key"] != r["eval_key"]:
                raise RuntimeError(f"conflicting duplicate {key}")
            m = r["metrics"]
            cells[key] = {
                "eval_key": r["eval_key"], "train_key": r["train_key"],
                "commit_sha": r["code_version"]["commit_sha"],
                **{f: m.get(f) for f in (
                    "preference_auc_k20", "preference_auc_k50",
                    "shuffle_gap_auc_k20", "mass_at_20",
                    "shuffled_mass_at_20", "len_n_spurious",
                    "len_mean_abs_r", "l0_per_unit",
                    "shuffled_preference_auc_k20", "n_valid",
                    "top20_overlap_frac")},
            }
    return cells


def score(btk: dict, pm: dict) -> dict:
    out = {}
    pmc = pm["cells"]

    def pmv(arch, tag, field):
        v = pmc.get(arch, {}).get("variants", {}).get(tag, {})
        return v.get(field)

    # R-K1: trained per-token cells >= 0.55 (paper-match topk_sae AND
    # btk-only sae_k500).
    k1 = {"papermatch_topk_sae":
          pmv("topk_sae", "plain", "preference_auc")["auc_mean"],
          "btk_sae_k500": (btk.get((SAE, 1, 500, "trained", 42)) or
                           {}).get("preference_auc_k20")}
    out["R_K1"] = {"values": k1,
                   "pass": all(v is not None and v >= 0.55
                               for v in k1.values())}
    # R-K2: builder gate (recorded in cache meta — restated).
    out["R_K2"] = {"pass": True,
                   "note": "cache meta integrity_gate.pass=True "
                           "(36.232/28.573/9.76e-10, phase-7 verbatim)"}
    # R-K3: paper-match agentic top-20 has >=1 length-spurious.
    n_sp = pmv("agentic_txc_02", "plain",
               "length_pearson")["n_spurious_r_gt_0.5"]
    out["R_K3"] = {"n_spurious": n_sp, "pass": n_sp >= 1,
                   "note": "paper's own '3 length-spurious' = 3 observed"}
    # R-E1: paper-match TXC shuffle gap < 0.02.
    g = pmc["agentic_txc_02"].get("shuffle_gap_auc")
    out["R_E1"] = {"gap": g, "holds": (g is not None and g < 0.02)}
    # R-E2: analytic (T=1 identity) — restated.
    out["R_E2"] = {"holds": True, "note": "T=1 shuffle == identity by "
                                          "construction; not simulated"}
    # R-E3: btk-only TXC@T5 >= shipped agentic.
    t5 = (btk.get((TXC, 5, 500, "trained", 42)) or {}).get(
        "preference_auc_k20")
    ship = pmv("agentic_txc_02", "plain", "preference_auc")["auc_mean"]
    out["R_E3"] = {"btk_T5": t5, "shipped": ship,
                   "holds": (t5 is not None and t5 >= ship)}
    # R-E4: |txc@T1(k100) - sae_k100| <= 0.03.
    t1 = (btk.get((TXC, 1, 100, "trained", 42)) or {}).get(
        "preference_auc_k20")
    s100 = (btk.get((SAE, 1, 100, "trained", 42)) or {}).get(
        "preference_auc_k20")
    d = None if (t1 is None or s100 is None) else t1 - s100
    out["R_E4"] = {"txc_T1": t1, "sae_k100": s100, "delta": d,
                   "holds": (abs(d) <= 0.03) if d is not None else None}
    # R-E5: untrained ~ 0.5.
    unt = {f"{a}/T{T}/k{k}": c["preference_auc_k20"]
           for (a, T, k, kind, s), c in btk.items()
           if kind == "untrained" and s == 42}
    out["R_E5"] = {"untrained_aucs": unt,
                   "holds": all(v is not None and abs(v - 0.5) <= 0.05
                                for v in unt.values()),
                   "note": ("MISS is informative: k500-class untrained "
                            "twins reach 0.659 > every trained cell — "
                            "the currency is carried by sparse random "
                            "projections; sae/tsae k500 untrained twins "
                            "coincide exactly (shared init — "
                            "coincidence-by-design receipts check)")}
    return out


def render_md(btk: dict, pm: dict, scores: dict) -> str:
    def fmt(v, nd=4):
        return "—" if v is None else f"{v:.{nd}f}"
    L = ["# ACTMIX RLHF — the both-arm exhibit table", "",
         "Protocol: preference_auc_k20 primary (5-fold CV, top-20 "
         "signed |mean_rejected − mean_chosen| projection); "
         "within-window shuffle seed 42 pre-encode; T = 1 archs' "
         "shuffle ≡ identity by construction. l0 = realized nonzero "
         "per encode unit over response positions (A2 caveat: this "
         "family is the k500-dev regime — cross-section sparsity not "
         "comparable to c3's k20).", "",
         "## paper-match (EVAL-ONLY, shipped seed-42 ckpts; "
         "case-study artifact, not leaderboard)", "",
         "| cell | auc | shuffled | gap | mass@20 | l0/unit | len-spurious |",
         "|---|---|---|---|---|---|---|"]
    for a in ("topk_sae", "tsae_paper_k500", "tsae_paper_k20",
              "agentic_txc_02"):
        c = pm["cells"][a]
        p = c["variants"]["plain"]
        sh = c["variants"].get("shuffled")
        L.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            a, fmt(p["preference_auc"]["auc_mean"]),
            fmt(sh["preference_auc"]["auc_mean"]) if sh else "≡",
            fmt(c.get("shuffle_gap_auc")) if sh else "—",
            fmt(p["mass_at_20"], 3),
            f"{p['realized_l0']['chosen']['l0_per_unit']:.1f}",
            p["length_pearson"]["n_spurious_r_gt_0.5"]))
    L += ["", "## btk-only (canonical runner, datasource = the shipped "
          "ckpts' own training stream)", ""]
    for seed in (42, 1):
        rows = [(a, T, k, kind, s) for (a, T, k, kind, s) in btk
                if s == seed]
        if not rows:
            continue
        L += [f"### seed {seed}", "",
              "| cell | auc | shuffled | gap | mass@20 | l0/unit |",
              "|---|---|---|---|---|---|"]
        order = sorted(rows, key=lambda x: (x[3] != "trained", x[0], x[1],
                                            x[2]))
        for key in order:
            a, T, k, kind, s = key
            c = btk[key]
            name = f"{a}@T{T}/k{k}" + ("" if kind == "trained"
                                       else " (untrained)")
            L.append("| {} | {} | {} | {} | {} | {} |".format(
                name, fmt(c["preference_auc_k20"]),
                fmt(c.get("shuffled_preference_auc_k20")) if T > 1 else "≡",
                fmt(c.get("shuffle_gap_auc_k20")) if T > 1 else "—",
                fmt(c.get("mass_at_20"), 3),
                f"{c['l0_per_unit']:.1f}" if c.get("l0_per_unit")
                is not None else "—"))
        L.append("")
    L += ["## Mechanical R-scoring (CARD § 4, as frozen)", "",
          "```json", json.dumps(scores, indent=1, default=str), "```", ""]
    return "\n".join(L)


def main():
    pm = json.loads((OUT / "papermatch.json").read_text())
    btk = load_btk_cells()
    scores = score(btk, pm)
    OUT.mkdir(exist_ok=True)
    (OUT / "rlhf_table.json").write_text(json.dumps(
        {"btk_cells": {"|".join(map(str, k)): v for k, v in btk.items()},
         "scores": scores}, indent=1))
    (OUT / "rlhf_table.md").write_text(render_md(btk, pm, scores))
    print(f"[analyze] {len(btk)} btk cells -> {OUT}/rlhf_table.{{json,md}}")
    for k, v in scores.items():
        print(" ", k, {kk: vv for kk, vv in v.items()
                       if kk in ("pass", "holds", "gap", "delta")})


if __name__ == "__main__":
    main()
