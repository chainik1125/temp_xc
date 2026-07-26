"""diafaces/score_salvage.py — S1–S5 scorer for SALVAGE_CARD.md § 4.

Self-contained: every bar's formula is implemented here exactly as
pre-registered. Evaluates the PRIMARY arm (k_pos = 8) for the verdict;
the SECONDARY arm (k_pos = 8·T) gets the same computations reported at
full prominence, gating nothing. mac-b's support_stats/stage2_variance
harness (--seeds 3,4,5) is the cross-check lane, not the source.

Run: .venv/bin/python -m experiments.explorations.task_hunt.diafaces.score_salvage
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DS = "dial_real_ttrend_gpt2_l7"
SEEDS = (3, 4, 5)
CLAIM_TS = (16, 32)
TREND_TS = (8, 16, 32)
MARGIN = 0.05
T975_DF2 = 4.302652729911275          # t_{0.975, df=2} — paired t 95% CI, n=3
K_PRIM = 8
BASELINES = ("batchtopk_sae", "tsae")
# Card § 3 realized-l0 bands
PRIM_BAND = (4.5, 9.5)
SEC_BAND = (0.5, 1.05)                # × 8·T


def _load():
    cells = json.loads((HERE / "results" / f"salvage_stage2_{DS}.json")
                       .read_text())
    ok = [c for c in cells if c.get("ok")]
    print(f"[load] {len(ok)}/{len(cells)} ok cells")
    idx = {}
    for c in ok:
        arm = ("baseline" if c["arch"] != "txc_batchtopk_post" else
               "primary" if c["k_pos"] == K_PRIM else "secondary")
        idx[(arm, c["arch"], c["kind"], c["T"], c["seed"])] = c["metrics"]
    return idx


def _v(idx, arm, arch, kind, T, seed, col="lambda_recovery"):
    return idx[(arm, arch, kind, T, seed)][col]


def _seed_mean(idx, arm, arch, kind, T, col="lambda_recovery"):
    return float(np.mean([_v(idx, arm, arch, kind, T, s, col)
                          for s in SEEDS]))


def _s1(idx, arm):
    out = {}
    for b in BASELINES:
        for T in CLAIM_TS:
            d = np.array([_v(idx, arm, "txc_batchtopk_post", "trained", T, s)
                          - _v(idx, "baseline", b, "trained", 1, s)
                          for s in SEEDS])
            m, sd = float(d.mean()), float(d.std(ddof=1))
            half = T975_DF2 * sd / math.sqrt(len(SEEDS))
            lo, hi = m - half, m + half
            out[f"{b}@T{T}"] = {
                "margins": [round(x, 4) for x in d.tolist()],
                "mean": round(m, 4), "ci95_t": [round(lo, 4), round(hi, 4)],
                "pass": bool(m >= MARGIN and lo > 0)}
    out["pass"] = all(v["pass"] for k, v in out.items() if k != "pass")
    return out


def _s2(idx, arm):
    out = {}
    for T in CLAIM_TS:
        tr = _seed_mean(idx, arm, "txc_batchtopk_post", "trained", T)
        un = _seed_mean(idx, arm, "txc_batchtopk_post", "untrained", T)
        out[f"T{T}"] = {"trained": round(tr, 4), "untrained": round(un, 4),
                        "ratio": round(un / tr, 4) if tr else None,
                        "pass": bool(un <= 0.5 * tr)}
    out["pass"] = all(out[f"T{T}"]["pass"] for T in CLAIM_TS)
    return out


def _s3(idx, arm):
    x = np.log2(TREND_TS)
    per_seed = {s: np.array([_v(idx, arm, "txc_batchtopk_post", "trained",
                                T, s) for T in TREND_TS])
                for s in SEEDS}

    def stat(vals_by_seed):
        return float(np.mean([np.polyfit(x, v, 1)[0]
                              for v in vals_by_seed]))

    obs = stat(list(per_seed.values()))
    perms = list(itertools.permutations(range(len(TREND_TS))))
    count = 0
    total = 0
    for combo in itertools.product(perms, repeat=len(SEEDS)):
        vals = [per_seed[s][list(p)] for s, p in zip(SEEDS, combo)]
        total += 1
        if stat(vals) >= obs - 1e-12:
            count += 1
    return {"slope_obs": round(obs, 4), "p_exact": round(count / total, 5),
            "n_perms": total}


def _s4(idx, arm, ev):
    out = {}
    for T in CLAIM_TS:
        bar = ev[str(T)]["pearson_r"]
        tr = _seed_mean(idx, arm, "txc_batchtopk_post", "trained", T)
        out[f"T{T}"] = {"trained": round(tr, 4), "evidence_r": bar,
                        "pass": bool(tr > bar)}
    out["pass"] = all(out[f"T{T}"]["pass"] for T in CLAIM_TS)
    return out


def _s5(idx, arm):
    out = {}
    for T in CLAIM_TS:
        v2 = _seed_mean(idx, arm, "txc_batchtopk_post", "trained", T,
                        "lambda_recovery_v2")
        out[f"T{T}"] = {"v2_grouped": round(v2, 4), "pass": bool(v2 > 0)}
    out["pass"] = all(out[f"T{T}"]["pass"] for T in CLAIM_TS)
    return out


def _l0_flags(idx, arm):
    flags = []
    for (a, arch, kind, T, seed), m in idx.items():
        if a != arm or kind != "trained":
            continue
        l0 = m.get("l0_per_window")
        lo, hi = (PRIM_BAND if arm == "primary"
                  else (SEC_BAND[0] * 8 * T, SEC_BAND[1] * 8 * T))
        if not (lo <= l0 <= hi):
            flags.append({"cell": f"{arch}:T{T}:s{seed}",
                          "l0_per_window": round(l0, 3),
                          "band": [round(lo, 2), round(hi, 2)]})
    return flags


def main():
    idx = _load()
    ev = json.loads((HERE / "results" / "panel_evidence_line_tt.json")
                    .read_text())["per_T"]
    report = {}
    for arm in ("primary", "secondary"):
        r = {"S1": _s1(idx, arm), "S2": _s2(idx, arm), "S3": _s3(idx, arm),
             "S4": _s4(idx, arm, ev), "S5": _s5(idx, arm),
             "l0_out_of_band": _l0_flags(idx, arm)}
        r["KEEP"] = bool(r["S1"]["pass"] and r["S2"]["pass"]
                         and r["S4"]["pass"] and r["S5"]["pass"])
        report[arm] = r
    report["verdict"] = {
        "claiming_arm": "primary",
        "KEEP": report["primary"]["KEEP"],
        "note": "secondary arm reported at full prominence, gates nothing "
                "(SALVAGE_CARD § 2); PENDING TEAM REVIEW",
    }
    out = HERE / "results" / "salvage_score.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"[score] written {out}")


if __name__ == "__main__":
    main()
