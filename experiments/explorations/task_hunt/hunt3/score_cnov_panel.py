"""hunt3/score_cnov_panel.py — cnov panel scorer (STAGED, PICK-PENDING;
freezes with the card + runner at the pick).

S1–S5 exactly as CNOV_PANEL_CARD.md § 3:
- S1 per claiming T (T16 ONLY per ruling f9319e59a; T32 reported): post − {sae, tsae} paired-by-seed mean
  ≥ +0.05 AND one-sample t 95% CI lower bound > 0 (n = 3, df = 2,
  t = 4.302652729911275).
- S2: untrained post ≤ 0.5 × trained post at each claiming T.
- S3: T8→16→32 trained-post trend, exact within-seed T-label
  permutation (6^3 = 216), REPORTED not gating.
- S4 (KILL): trained post recovery must exceed the pre-measured
  visible-cue evidence line at its T (panel_evidence_line_cnov.json —
  mac-b's instrument: r 0.2692 @T16, 0.4017 @T32 on gpt2 labels).
- S5: grouped v2 (lambda_recovery_v2) > 0 for trained post at each
  claiming T.
KEEP at T iff S1(both baselines) ∧ S2 ∧ S4 ∧ S5 at that T.
l0 band [4.5, 9.5]/window applies to POST arms only (card § 4;
baseline realizations quoted with the R30 threshold-pruning
pre-disclosure, never band-gated — the R29 lesson).

Run: .venv/bin/python -m experiments.explorations.task_hunt.hunt3.score_cnov_panel
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np

from experiments.explorations.task_hunt.hunt3.run_cnov_panel import DS

HERE = Path(__file__).resolve().parent
SEEDS = (9, 10, 11)
CLAIM_TS = (16,)               # RULED f9319e59a: T32 = run-but-not-claim
REPORT_TS = (16, 32)
LADDER = (8, 16, 32)
T975_DF2 = 4.302652729911275
L0_BAND_POST = (4.5, 9.5)


def _load():
    panel = json.loads(
        (HERE / "results" / f"cnov_stage2_{DS}.json").read_text())
    by = {}
    for c in panel:
        if c.get("ok"):
            by[(c["arch"], c["T"], c["seed"], c["kind"])] = c["metrics"]
    return by


def _rec(by, arch, T, seed, kind, key="lambda_recovery"):
    return float(by[(arch, T, seed, kind)][key])


def _t_ci(diffs):
    d = np.asarray(diffs, float)
    m = float(d.mean())
    half = T975_DF2 * float(d.std(ddof=1)) / math.sqrt(len(d))
    return m, m - half, m + half


def _s3_exact(by):
    """Within-seed T-label permutation on the trained post ladder."""
    vals = {s: [_rec(by, "txc_batchtopk_post", T, s, "trained")
                for T in LADDER] for s in SEEDS}
    x = np.log2(LADDER)

    def slope(ys):
        return float(np.polyfit(x, ys, 1)[0])

    obs = float(np.mean([slope(vals[s]) for s in SEEDS]))
    perms = []
    for ps in itertools.product(
            *[list(itertools.permutations(vals[s])) for s in SEEDS]):
        perms.append(float(np.mean([slope(list(p)) for p in ps])))
    p = float(np.mean([pp >= obs - 1e-12 for pp in perms]))
    return {"obs_slope": round(obs, 4), "p_exact_216": round(p, 5)}


def main():
    by = _load()
    ev = json.loads(
        (HERE / "results" / "panel_evidence_line_cnov.json").read_text())
    s4_bar = {T: float(ev["per_T"][str(T)]["pearson_r"]) for T in REPORT_TS}

    out = {"card": "CNOV_PANEL_CARD.md", "ds": DS, "seeds": list(SEEDS),
           "status": "PENDING TEAM REVIEW", "per_T": {}, "l0": {},
           "s3_trend": _s3_exact(by)}

    for T in REPORT_TS:
        row = {}
        for base in ("batchtopk_sae", "tsae"):
            diffs = [_rec(by, "txc_batchtopk_post", T, s, "trained")
                     - _rec(by, base, 1, s, "trained") for s in SEEDS]
            m, lo, hi = _t_ci(diffs)
            row[f"S1_vs_{base}"] = {
                "mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)],
                "pass": bool(m >= 0.05 and lo > 0),
                "per_seed": [round(d, 4) for d in diffs]}
        tr = [_rec(by, "txc_batchtopk_post", T, s, "trained") for s in SEEDS]
        un = [_rec(by, "txc_batchtopk_post", T, s, "untrained")
              for s in SEEDS]
        ratio = float(np.mean(un) / max(np.mean(tr), 1e-9))
        row["S2_untrained_ratio"] = {"ratio": round(ratio, 4),
                                     "pass": bool(ratio <= 0.5)}
        row["S4_evidence"] = {
            "bar": round(s4_bar[T], 4),
            "trained_mean": round(float(np.mean(tr)), 4),
            "pass": bool(np.mean(tr) > s4_bar[T])}
        v2 = [_rec(by, "txc_batchtopk_post", T, s, "trained",
                   "lambda_recovery_v2") for s in SEEDS]
        row["S5_v2_grouped"] = {"mean": round(float(np.mean(v2)), 4),
                                "pass": bool(np.mean(v2) > 0)}
        row["claiming"] = bool(T in CLAIM_TS)
        row["KEEP_at_T"] = bool(
            row["S1_vs_batchtopk_sae"]["pass"]
            and row["S1_vs_tsae"]["pass"] and row["S2_untrained_ratio"]["pass"]
            and row["S4_evidence"]["pass"] and row["S5_v2_grouped"]["pass"])
        out["per_T"][str(T)] = row

    flags = []
    for (arch, T, seed, kind), m in by.items():
        l0w = float(m.get("l0_per_window", float("nan")))
        if arch == "txc_batchtopk_post" and kind == "trained" and \
                not (L0_BAND_POST[0] <= l0w <= L0_BAND_POST[1]):
            flags.append({"cell": [arch, T, seed, kind],
                          "l0_per_window": round(l0w, 3)})
        out["l0"][f"{arch}/T{T}/s{seed}/{kind}"] = round(l0w, 3)
    out["l0_out_of_band_post"] = flags
    out["verdict"] = {
        "KEEP_Ts": [T for T in CLAIM_TS
                    if out["per_T"][str(T)]["KEEP_at_T"]],
        "reported_not_claiming": [T for T in REPORT_TS if T not in CLAIM_TS],
    }
    dst = HERE / "results" / "cnov_panel_score.json"
    dst.write_text(json.dumps(out, indent=1))
    print(json.dumps({"per_T": {k: {kk: vv for kk, vv in v.items()
                                    if kk != "per_seed"}
                                for k, v in out["per_T"].items()},
                      "s3": out["s3_trend"],
                      "verdict": out["verdict"]}, indent=1))
    print(f"[score] wrote {dst}")


if __name__ == "__main__":
    main()
