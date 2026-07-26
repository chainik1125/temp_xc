"""diafaces/score_topup.py — L1/L2 scorer for TOPUP_CARD.md § 2
(mac-local ruling ad76b0f15 item 3, implemented verbatim).

L1: S1 four legs on seeds {6,7,8} alone (t crit 4.302653, n = 3) —
independent replication lane, gates nothing. L2: combined n = 6
(t crit 2.570582) with the mandatory SEQUENTIAL-DECISION caveat
embedded in the output. KEEP at {16,32} iff L2 S1 all four legs AND
combined S2 ∧ S4 ∧ S5.

Run: .venv/bin/python -m experiments.explorations.task_hunt.diafaces.score_topup
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DS = "dial_real_ttrend_gpt2_l7"
OLD_SEEDS = (3, 4, 5)
NEW_SEEDS = (6, 7, 8)
ALL_SEEDS = OLD_SEEDS + NEW_SEEDS
CLAIM_TS = (16, 32)
MARGIN = 0.05
T_CRIT = {3: 4.302652729911275, 6: 2.5705818366147395}
K_PRIM = 8
BASELINES = ("batchtopk_sae", "tsae")
PRIM_BAND = (4.5, 9.5)
SEQ_CAVEAT = ("SEQUENTIAL-DECISION CAVEAT (mandatory beside every L2 "
              "number): the n=6 extension was decided AFTER observing "
              "seeds {3,4,5} fail one t-CI leg — L2 is a conditional "
              "test (R22-caveat style).")


def _load():
    idx = {}
    for fname, seeds in ((f"salvage_stage2_{DS}.json", OLD_SEEDS),
                         (f"topup_stage2_{DS}.json", NEW_SEEDS)):
        for c in json.loads((HERE / "results" / fname).read_text()):
            if not c.get("ok") or c["seed"] not in seeds:
                continue
            if c["arch"] == "txc_batchtopk_post" and c["k_pos"] != K_PRIM:
                continue                     # secondary arm: not in scope
            idx[(c["arch"], c["kind"], c["T"], c["seed"])] = c["metrics"]
    return idx


def _v(idx, arch, kind, T, seed, col="lambda_recovery"):
    return idx[(arch, kind, T, seed)][col]


def _mean(idx, arch, kind, T, seeds, col="lambda_recovery"):
    return float(np.mean([_v(idx, arch, kind, T, s, col) for s in seeds]))


def _s1(idx, seeds):
    crit = T_CRIT[len(seeds)]
    out = {}
    for b in BASELINES:
        for T in CLAIM_TS:
            d = np.array([_v(idx, "txc_batchtopk_post", "trained", T, s)
                          - _v(idx, b, "trained", 1, s) for s in seeds])
            m, sd = float(d.mean()), float(d.std(ddof=1))
            half = crit * sd / math.sqrt(len(seeds))
            out[f"{b}@T{T}"] = {
                "margins": [round(x, 4) for x in d.tolist()],
                "mean": round(m, 4),
                "ci95_t": [round(m - half, 4), round(m + half, 4)],
                "pass": bool(m >= MARGIN and m - half > 0)}
    out["pass"] = all(v["pass"] for k, v in out.items() if k != "pass")
    return out


def _s2_s4_s5(idx, seeds, ev):
    s2, s4, s5 = {}, {}, {}
    for T in CLAIM_TS:
        tr = _mean(idx, "txc_batchtopk_post", "trained", T, seeds)
        un = _mean(idx, "txc_batchtopk_post", "untrained", T, seeds)
        v2 = _mean(idx, "txc_batchtopk_post", "trained", T, seeds,
                   "lambda_recovery_v2")
        bar = ev[str(T)]["pearson_r"]
        s2[f"T{T}"] = {"trained": round(tr, 4), "untrained": round(un, 4),
                       "ratio": round(un / tr, 4) if tr else None,
                       "pass": bool(un <= 0.5 * tr)}
        s4[f"T{T}"] = {"trained": round(tr, 4), "evidence_r": bar,
                       "pass": bool(tr > bar)}
        s5[f"T{T}"] = {"v2_grouped": round(v2, 4), "pass": bool(v2 > 0)}
    for d in (s2, s4, s5):
        d["pass"] = all(d[f"T{T}"]["pass"] for T in CLAIM_TS)
    return s2, s4, s5


def _s3_combined(idx):
    # T16→T32 rise on n=6: exact within-seed sign-flip (2^6 = 64).
    deltas = np.array([_v(idx, "txc_batchtopk_post", "trained", 32, s)
                       - _v(idx, "txc_batchtopk_post", "trained", 16, s)
                       for s in ALL_SEEDS])
    obs = float(deltas.mean())
    count = 0
    for signs in itertools.product((1, -1), repeat=len(ALL_SEEDS)):
        if float((deltas * signs).mean()) >= obs - 1e-12:
            count += 1
    return {"delta_mean": round(obs, 4),
            "deltas": [round(x, 4) for x in deltas.tolist()],
            "p_exact": round(count / 64, 5), "n_perms": 64}


def _l0_flags(idx_file, seeds):
    flags = []
    for c in json.loads((HERE / "results" / idx_file).read_text()):
        if (not c.get("ok") or c["kind"] != "trained"
                or c["seed"] not in seeds
                or (c["arch"] == "txc_batchtopk_post"
                    and c["k_pos"] != K_PRIM)):
            continue
        l0 = c["metrics"].get("l0_per_window")
        if not (PRIM_BAND[0] <= l0 <= PRIM_BAND[1]):
            flags.append({"cell": f'{c["arch"]}:T{c["T"]}:s{c["seed"]}',
                          "l0_per_window": round(l0, 3)})
    return flags


def main():
    idx = _load()
    ev = json.loads((HERE / "results" / "panel_evidence_line_tt.json")
                    .read_text())["per_T"]
    l1_s2, l1_s4, l1_s5 = _s2_s4_s5(idx, NEW_SEEDS, ev)
    l2_s2, l2_s4, l2_s5 = _s2_s4_s5(idx, ALL_SEEDS, ev)
    report = {
        "L1_new_seeds_alone": {
            "S1": _s1(idx, NEW_SEEDS),
            "S2": l1_s2, "S4": l1_s4, "S5": l1_s5,
            "note": "independent replication lane; gates nothing; "
                    "n=3 power limits acknowledged (TOPUP_CARD § 2)"},
        "L2_combined_n6": {
            "caveat": SEQ_CAVEAT,
            "S1": _s1(idx, ALL_SEEDS),
            "S2": l2_s2, "S4": l2_s4, "S5": l2_s5,
            "S3_combined_16to32": _s3_combined(idx)},
        "l0_out_of_band": _l0_flags(f"topup_stage2_{DS}.json", NEW_SEEDS),
    }
    l2 = report["L2_combined_n6"]
    report["verdict"] = {
        "KEEP_at_16_32": bool(l2["S1"]["pass"] and l2["S2"]["pass"]
                              and l2["S4"]["pass"] and l2["S5"]["pass"]),
        "rule": "KEEP at {16,32} iff L2 S1 all four legs AND combined "
                "S2∧S4∧S5 (ad76b0f15 item 3)",
        "caveat": SEQ_CAVEAT,
        "note": "T32-only re-scope remains PROPOSED (team item); "
                "PENDING TEAM REVIEW",
    }
    out = HERE / "results" / "topup_score.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"[score] written {out}")


if __name__ == "__main__":
    main()
