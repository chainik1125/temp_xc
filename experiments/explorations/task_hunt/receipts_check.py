"""Claim→artifact receipts index (panel-support-audit item 3 — rebuttal
insurance).

For every number the program currently considers rebuttal-quotable:
the claim as we would state it, the artifact path + key it comes from,
the commit that produced the artifact, and a RECOMPUTED-NOW value
checked against the quoted one. A FAIL here two days before a deadline
is worth more than any new result — failures print loudly and the
process exits 1.

Writes `RECEIPTS.md` next to this script. Run:

    .venv/bin/python -m experiments.explorations.task_hunt.receipts_check

Quoted values are the LOG-stated (rounded) numbers; a check PASSes iff
the recomputed value matches the quote at its stated precision
(|Δ| ≤ 0.5·10⁻ᵈᵖ). Tested by `tests/test_receipts_index.py` so drift
between artifacts and quoted claims breaks the suite, not the rebuttal.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LEADERBOARD = ROOT / "results" / "leaderboard.jsonl"
DS = "ward_real_lambda_base_l12"


def _j(rel: str):
    return json.loads((HERE / rel).read_text())


def _commit(rel: str) -> str:
    rel = rel.split(" ")[0]
    p = HERE / rel
    if not p.exists():
        p = ROOT / rel
    if "{" in rel or not p.exists():
        return "—"
    out = subprocess.run(
        ["git", "log", "-n", "1", "--format=%h", "--", str(p)],
        capture_output=True, text=True, cwd=ROOT)
    return out.stdout.strip() or "—"


def _lb_cells(arch, T, kinds=("trained",), k_pos=8, seeds=None):
    """v1 metric per seed from the canonical leaderboard (the same source
    the seed-top-up receipt used — NOT the per-run results JSON)."""
    vals = {}
    with LEADERBOARD.open() as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("datasource") != DS or r["arch"] != arch:
                continue
            tc = r["training_cfg"]
            ov = tc["arch_hparams_override"]
            if ov["T"] != T or ov.get("k_pos") != k_pos:
                continue
            kind = "untrained" if tc["n_steps"] == 0 else "trained"
            if kind not in kinds:
                continue
            if seeds is not None and r["seed"] not in seeds:
                continue
            vals[r["seed"]] = r["metrics"]["lambda_recovery"]
    return vals


def _tci(v):
    v = np.asarray(v, float)
    n = len(v)
    se = v.std(ddof=1) / np.sqrt(n)
    lo, hi = stats.t.interval(0.95, n - 1, loc=v.mean(), scale=se)
    return float(v.mean()), float(lo), float(hi), float(v.std(ddof=1))


def _one_sided_lb95(v):
    v = np.asarray(v, float)
    n = len(v)
    return float(v.mean() - stats.t.ppf(0.95, n - 1)
                 * v.std(ddof=1) / np.sqrt(n))


def _screen_target(rel):
    """Per-target screen summary: mean over the 4 (model, layer) cells of
    per-token AUC, g@T32, g_order@T32, shuffle_gap@T32; n_train rows."""
    d = _j(rel)
    cells = d["cells"]
    prefixes = sorted({k.rsplit("/real/", 1)[0] for k in cells
                       if "/real/" in k})
    tok = [cells[f"{p}/real/tok"]["linear"]["auc"] for p in prefixes]
    g = [cells[f"{p}/real/T32"]["g"] for p in prefixes]
    go = [cells[f"{p}/real/T32"]["g_order"] for p in prefixes]
    sh = [cells[f"{p}/real/T32"]["shuffle_gap"] for p in prefixes]
    n_train = cells[f"{prefixes[0]}/real/tok"]["linear"]["n_train"]
    return {"tok": float(np.mean(tok)), "g32": float(np.mean(g)),
            "g_order32": float(np.mean(go)), "shuf32": float(np.mean(sh)),
            "n_cells": len(prefixes), "n_train": int(n_train)}


# ── the receipts ────────────────────────────────────────────────────────────
# Each: id, claim (as we would state it), artifact rel-path, key note,
# checks = [(name, quoted, decimal places), ...] + a compute() -> {name: val}.

def build_receipts():
    R = []

    sv = _j("support_stats/stage2_variance.json")
    R.append(dict(
        id="R1", artifact="support_stats/stage2_variance.json",
        key="trend.txc_pre_trained_2to8.p_one_sided (exact, 216 perms)",
        claim="λ̂ panel: TXC-pre T=2→8 rise, exact within-seed permutation "
              "p = 0.0093 — the panel's one significant headline at n = 3",
        checks=[("p", 0.0093, 4),
                ("n_perms", 216, 0)],
        got={"p": sv["trend"]["txc_pre_trained_2to8"]["p_one_sided"],
             "n_perms": sv["trend"]["txc_pre_trained_2to8"]["n_perms"]}))
    R.append(dict(
        id="R2", artifact="support_stats/stage2_variance.json",
        key="trend.txc_pre_margin_2to8.p_one_sided",
        claim="λ̂ panel: trained−untrained margin rises T=2→8, exact "
              "permutation p = 0.0046",
        checks=[("p", 0.0046, 4)],
        got={"p": sv["trend"]["txc_pre_margin_2to8"]["p_one_sided"]}))

    ci8 = sv["cell_ci95_trained"]["txc_batchtopk_pre/T8"]
    R.append(dict(
        id="R3", artifact="support_stats/stage2_variance.json",
        key="cell_ci95_trained[txc_batchtopk_pre/T8]",
        claim="λ̂ panel headline cell pre/T8 at n = 3: mean 0.206, "
              "95% CI [0.145, 0.267] (tightened to R4's n = 6 by top-up)",
        checks=[("mean", 0.206, 3), ("lo", 0.145, 3), ("hi", 0.267, 3)],
        got={"mean": ci8["mean"], "lo": ci8["t_ci95"][0],
             "hi": ci8["t_ci95"][1]}))

    pre8 = _lb_cells("txc_batchtopk_pre", 8, seeds={1, 2, 3, 4, 5, 42})
    pre4 = _lb_cells("txc_batchtopk_pre", 4, seeds={1, 2, 3, 4, 5, 42})
    tsae = _lb_cells("tsae", 1, seeds={1, 2, 42})
    m8, lo8, hi8, sd8 = _tci(list(pre8.values()))
    m4, lo4, hi4, _ = _tci(list(pre4.values()))
    mt, lot, hit, _ = _tci(list(tsae.values()))
    R.append(dict(
        id="R4", artifact="results/leaderboard.jsonl (canonical)",
        key=f"pre/T8 trained, seeds {sorted(pre8)}; pre/T4; tsae/T1",
        claim="Seed top-up: pre/T8 at n = 6 mean 0.2071, 95% CI "
              "[0.179, 0.235] — the headline level is pinned, entirely "
              "above the per-token SAE; pre/T4 0.2279 [0.182, 0.274]; "
              "tsae/T1 n = 3 0.1541 [0.042, 0.266]",
        checks=[("pre8_mean", 0.2071, 4), ("pre8_lo", 0.179, 3),
                ("pre8_hi", 0.235, 3), ("pre8_sd", 0.0268, 4),
                ("pre4_mean", 0.2279, 4), ("tsae_mean", 0.1541, 4)],
        got={"pre8_mean": m8, "pre8_lo": lo8, "pre8_hi": hi8,
             "pre8_sd": sd8, "pre4_mean": m4, "tsae_mean": mt}))

    shared = [1, 2, 42]
    d_paired = np.array([pre8[s] for s in shared]) - \
        np.array([tsae[s] for s in shared])
    lb_paired = _one_sided_lb95(d_paired)
    a = np.array(list(pre8.values()))
    b = np.array(list(tsae.values()))
    se_w = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    df_w = se_w ** 4 / ((a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1)
                        + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1))
    diff_w = float(a.mean() - b.mean())
    lb_w = diff_w - float(stats.t.ppf(0.95, df_w)) * float(se_w)
    p_w = float(stats.t.sf(diff_w / se_w, df_w))
    R.append(dict(
        id="R5", artifact="results/leaderboard.jsonl (canonical)",
        key="pre/T8 − tsae/T1: paired (3 shared seeds) + Welch (6 vs 3)",
        claim="[n = 3 STATE — SUPERSEDED-PENDING-TEAM-RATIFICATION by "
              "R22] pre-vs-T-SAE margin at round-1 n = 3 was NOT BOUNDED "
              "and until the team ratifies R22's two caveats it is still "
              "not quoted as significant: paired diff +0.0522, one-sided "
              "95% LB −0.0413; Welch diff +0.0530, LB −0.0159, one-sided "
              "p = 0.082 (df 2.7)",
        checks=[("paired_diff", 0.0522, 4), ("paired_lb", -0.0413, 4),
                ("welch_diff", 0.0530, 4), ("welch_lb", -0.0159, 4),
                ("welch_p", 0.082, 3), ("welch_df", 2.7, 1)],
        got={"paired_diff": float(d_paired.mean()), "paired_lb": lb_paired,
             "welch_diff": diff_w, "welch_lb": lb_w, "welch_p": p_w,
             "welch_df": float(df_w)}))

    sh = _j("results/shuffle_receipt.json")
    costs = {k: c["flat"]["auc"] - c["shuf"]["auc"]
             for k, c in sh["cells"].items()}
    ant = [v for k, v in costs.items() if "/ant_" in k]
    isbt = [v for k, v in costs.items() if "/is_bt" in k]
    R.append(dict(
        id="R6", artifact="results/shuffle_receipt.json",
        key="flat.auc − shuf.auc per cell; sigma_null",
        claim="Backtracking case study IS order-sensitive: within-window "
              "shuffle costs the anticipation targets +0.028…+0.041 AUC "
              "(3–4× σ_null = 0.0035) while near-ambient is_bt loses only "
              "+0.003…+0.013",
        checks=[("ant_min", 0.028, 3), ("ant_max", 0.041, 3),
                ("isbt_min", 0.003, 3), ("isbt_max", 0.013, 3),
                ("sigma_null", 0.0035, 4)],
        got={"ant_min": min(ant), "ant_max": max(ant),
             "isbt_min": min(isbt), "isbt_max": max(isbt),
             "sigma_null": sh["sigma_null"]}))

    tf = _j("support_synthetic/results/tsae_fair_verdict.json")
    pd_ = tf["paired_vs_d1"]
    dmax = max(abs(v["D"]) for v in pd_.values())
    R.append(dict(
        id="R7", artifact="support_synthetic/results/tsae_fair_verdict.json",
        key="paired_vs_d1.*.mean_D (max |·|) vs the 0.05 bar",
        claim="T-SAE fairness: temporal-kernel variants change nothing — "
              "max |paired D| = 0.011, far under the pre-registered 0.05 "
              "bar (verdict FAIR)",
        checks=[("max_abs_D", 0.011, 3)],
        got={"max_abs_D": dmax}))

    sf = _j("lambda_intensity/results/split_forensics.json")
    exp = sf["v1_sampling_exposure"]
    R.append(dict(
        id="R8", artifact="lambda_intensity/results/split_forensics.json",
        key="v1_sampling_exposure.nw1024.eval_draws_from_straddling_traces",
        claim="Split forensics: ZERO committed-settings (nw = 1024) eval "
              "draws touch the one trace straddling the split — no "
              "committed number is affected by split leakage",
        checks=[("n_draws", 0, 0)],
        got={"n_draws": exp["nw1024"]
             ["eval_draws_from_straddling_traces"]}))

    screens = [
        ("R9a", "oprate/results/oprate_ver_screen.json", "oprate/ver",
         0.813, 0.063),
        ("R9b", "oprate/results/oprate_case_screen.json", "oprate/case",
         0.741, 0.068),
        ("R9c", "qrate/results/qrate_main_screen.json", "qrate (Ward)",
         0.818, 0.081),
        ("R9d", "verbosity/results/verbosity_vslope_screen.json",
         "verbosity/vslope", 0.702, 0.081),
        ("R9e", "sc_lambda/results/sc_screen.json", "sc_lambda",
         0.871, 0.066),
    ]
    per_target = {}
    for rid, rel, name, q_tok, q_g in screens:
        t = _screen_target(rel)
        per_target[name] = t
        R.append(dict(
            id=rid, artifact=rel,
            key="mean over (model, layer) cells: real/tok linear.auc, "
                "real/T32 g",
            claim=f"Stage-1 KEEP {name}: per-token {q_tok:.3f}, window "
                  f"gain g@T32 +{q_g:.3f} — probe train rows "
                  f"n = {t['n_train']} (screen-side corpus; quote the "
                  f"size beside the number, per the estimator finding)",
            checks=[("tok", q_tok, 3), ("g32", q_g, 3)],
            got={"tok": t["tok"], "g32": t["g32"]}))

    go = [t["g_order32"] for t in per_target.values()]
    sc = [t["shuf32"] for t in per_target.values()]
    R.append(dict(
        id="R10", artifact="the five R9 screen artifacts",
        key="per-target mean g_order@T32 and shuffle_gap@T32",
        claim="AMENDED order finding (Ward substrate, five targets): every "
              "window advantage found is order-free aggregation — g_order "
              "at T32 spans −0.004…+0.008, within-window shuffle costs "
              "+0.003…+0.019. NEVER quote with 'anywhere'.",
        checks=[("g_order_min", -0.004, 3), ("g_order_max", 0.008, 3),
                ("shuf_min", 0.003, 3), ("shuf_max", 0.019, 3)],
        got={"g_order_min": min(go), "g_order_max": max(go),
             "shuf_min": min(sc), "shuf_max": max(sc)}))

    dia = {}
    for m in ("gemma2_2b", "gpt2", "llama31_8b"):
        c = _j(f"dialevel/results/screen_{m}.json")["cells"]
        dia[m] = (c["wd/T32/win_linear"]["auc"]
                  - c["wd/T32/win_shuf_linear"]["auc"])
    R.append(dict(
        id="R11", artifact="dialevel/results/screen_{model}.json",
        key="wd/T32: win_linear.auc − win_shuf_linear.auc",
        claim="…AND the recorded counterexample: dialevel's window readout "
              "IS shuffle-sensitive on dialogue — anchor-fixed context "
              "shuffle costs +0.057 (gpt2) / +0.063 (gemma) / +0.035 "
              "(llama31) at T = 32, 3/3 models (recency / "
              "distance-to-anchor is the recorded hypothesis). QUOTE THESE "
              "values: the LOG entries' '+0.056 / +0.062' were TRUNCATED, "
              "not rounded, from these same artifact values — caught by "
              "this index",
        checks=[("gpt2", 0.057, 3), ("gemma2_2b", 0.063, 3),
                ("llama31_8b", 0.035, 3)],
        got=dia))

    pt = _j("support_synthetic/results/probe_truth.json")
    am = pt["amendment"]
    miss = {(r["arm"], r["density"], r["p_over_n"]): r for r in am["rows"]}
    m1 = miss[("token", "p6", 1.0)]
    m2 = miss[("token", "p6", 2.0)]
    R.append(dict(
        id="R12", artifact="support_synthetic/results/probe_truth.json",
        key="amendment: A_P1/A_P2 counts; rows[token,p6,p/n∈{1,2}]",
        claim="Mirror Stage-1 (exact truth): ADOPT-consistent — v1 sags "
              "7/8 signal cells, v2 tracks 10/12 and exceeds truth 0/12; "
              "the two v2 misses (token, 6% dense, truth 0.412): 0.299 at "
              "p/n = 1.0 (d2 −0.113) and 0.232 at p/n = 2.0 (d2 −0.180) — "
              "v2 is a LOWER BOUND there (PROBE_V2_SPEC § 0), levels are "
              "conservative, ordering is robust",
        checks=[("A_P1", 7, 0), ("A_P2", 10, 0), ("over_truth", 0, 0),
                ("miss1_v2", 0.299, 3), ("miss1_d2", -0.113, 3),
                ("miss2_v2", 0.232, 3), ("miss2_d2", -0.180, 3),
                ("truth", 0.412, 3)],
        got={"A_P1": am["A_P1_v1_sags"]["n_holding"],
             "A_P2": am["A_P2_v2_tracks"]["n_holding"],
             "over_truth": am["n_v2_over_truth"],
             "miss1_v2": m1["v2"], "miss1_d2": m1["d2"],
             "miss2_v2": m2["v2"], "miss2_d2": m2["d2"],
             "truth": m1["truth"]}))

    # ── stage2-fineweb, CASE STUDY #3 (runpod-e, 2026-07-25) ────────────
    gv = _j("support_stats/stage2_variance_qrate_gemma.json")
    gcell = gv["cell_ci95_trained"]["txc_batchtopk_pre/T8"]
    gd8 = gv["paired"]["txc_pre_minus_pertoken"]["by_T"]["T8"]
    R.append(dict(
        id="R13",
        artifact="support_stats/stage2_variance_qrate_gemma.json",
        key="cell_ci95_trained[txc_batchtopk_pre/T8]; "
            "paired.txc_pre_minus_pertoken.by_T.T8",
        claim="fineweb gemma panel headline (v1, canonical): pre/T8 "
              "0.250 [0.189, 0.311]; K1 gap vs better token +0.0541 at "
              "T8 — NOT BOUNDED at n = 3 (BCa [−0.002, 0.086], "
              "sign-flip p = 0.25): quote direction + CI, never "
              "'significant'. Verdict of record: NO RULE FIRES AS "
              "WRITTEN (K1✓ K3✓ K4✓, K2✗)",
        checks=[("mean", 0.250, 3), ("lo", 0.189, 3), ("hi", 0.311, 3),
                ("gap", 0.0541, 4), ("bca_lo", -0.002, 3),
                ("bca_hi", 0.086, 3), ("p_sf", 0.25, 2)],
        got={"mean": gcell["mean"], "lo": gcell["t_ci95"][0],
             "hi": gcell["t_ci95"][1], "gap": gd8["mean"],
             "bca_lo": gd8["bca_ci95"][0], "bca_hi": gd8["bca_ci95"][1],
             "p_sf": gd8["p_signflip_one_sided"]}))

    gv2 = _j("support_stats/stage2_variance_qrate_gemma_v2.json")
    g2t8 = gv2["paired"]["txc_pre_minus_tsae"]["by_T"]["T8"]
    g2t16 = gv2["paired"]["txc_pre_minus_tsae"]["by_T"]["T16"]
    g2tr = gv2["trend"]["txc_pre_trained_2to16_secondary"]
    R.append(dict(
        id="R14",
        artifact="support_stats/stage2_variance_qrate_gemma_v2.json",
        key="paired.txc_pre_minus_tsae.by_T.{T8,T16}; "
            "trend.txc_pre_trained_2to16_secondary",
        claim="fineweb gemma paired v2 (NOT canonical — quote only as "
              "'ordering robust, widens under an adequate probe'): "
              "pre−tsae +0.100 [0.043, 0.158] at T8 and +0.129 [0.080, "
              "0.179] at T16 — t-bounded > 0 at n = 3; full-ladder "
              "2→16 trend p = 0.0009 (exact, 13824 perms)",
        checks=[("t8", 0.100, 3), ("t8_lo", 0.043, 3),
                ("t16", 0.129, 3), ("t16_lo", 0.080, 3),
                ("p", 0.0009, 4)],
        got={"t8": g2t8["mean"], "t8_lo": g2t8["t_ci95"][0],
             "t16": g2t16["mean"], "t16_lo": g2t16["t_ci95"][0],
             "p": g2tr["p_one_sided"]}))

    dm = _j("qrate_fineweb/results/"
            "stage2_demeaned_fineweb_punctint_q_gemma2_l14.json")
    def _dm_mean(arch, T):
        v = [r["ridge_demeaned"]["r"] for r in dm["rows"]
             if r["arch"] == arch and r["T"] == T]
        return float(np.mean(v)), v
    dpre, dpre_v = _dm_mean("txc_batchtopk_pre", 8)
    dtsae, dtsae_v = _dm_mean("tsae", 1)
    dgaps = [a - b for a, b in zip(dpre_v, dtsae_v)]
    R.append(dict(
        id="R15",
        artifact="qrate_fineweb/results/stage2_demeaned_"
                 "fineweb_punctint_q_gemma2_l14.json",
        key="ridge_demeaned.r means: pre/T8 vs tsae/T1; licence max Δ",
        claim="fineweb within-document receipt (§ 6b amended — "
              "whole-stream doc means; K4): doc-demeaned pre/T8 0.086 "
              "vs tsae 0.039 — gap +0.047, positive in all 3 seeds "
              "(min +0.041); probe-licence max Δ vs leaderboard "
              "1.4e-05. The pre-registered collapse branch did NOT "
              "occur; within-doc face sits near floor as predicted "
              "but the ordering survives",
        checks=[("pre", 0.086, 3), ("tsae", 0.039, 3),
                ("gap", 0.047, 3), ("gap_min", 0.041, 3),
                ("lic", 1.4e-05, 5)],
        got={"pre": dpre, "tsae": dtsae,
             "gap": float(np.mean(dgaps)), "gap_min": float(min(dgaps)),
             "lic": dm["max_licence_delta"]}))

    sup = _j("qrate_fineweb/results/"
             "stage2_support_fineweb_punctint_q_gemma2_l14.json")
    floors = [sup["per_T"][str(T)]["doc_floor_r"] for T in (1, 2, 4, 8, 16)]
    R.append(dict(
        id="R16",
        artifact="qrate_fineweb/results/stage2_support_"
                 "fineweb_punctint_q_gemma2_l14.json",
        key="per_T.doc_floor_r; per_T.{8,16}.evidence_count_r",
        claim="fineweb disclosure pair, printed beside every window "
              "number: doc-mean identity floor r = 0.575–0.587 (above "
              "every activation-probe cell on the panel), and the § 7 "
              "visible q-count regression bar 0.345 (T8) / 0.461 (T16) "
              "— NO window cell beats the count bar at T ≥ 8 on either "
              "probe; the card's 'small at T ≤ 16' prediction is "
              "falsified and any quoted window number carries this bar",
        checks=[("floor_min", 0.575, 3), ("floor_max", 0.587, 3),
                ("ev8", 0.345, 3), ("ev16", 0.461, 3)],
        got={"floor_min": float(min(floors)),
             "floor_max": float(max(floors)),
             "ev8": sup["per_T"]["8"]["evidence_count_r"],
             "ev16": sup["per_T"]["16"]["evidence_count_r"]}))

    rq = _j("qrate_fineweb/results/requote_screen.json")["cells"]
    def _rq_margin(key):
        return (rq[f"{key}/T64/actxmean_linear"]["acc_test"]
                - rq[f"{key}/tok_linear"]["acc_test"])
    def _rq_nullgap(key):
        return (rq[f"{key}/T64/actxmean_linear"]["acc_test"]
                - rq[f"{key}/T64/actxmean_foreign_linear"]["acc_test"])
    R.append(dict(
        id="R17",
        artifact="qrate_fineweb/results/requote_screen.json",
        key="T64 actxmean_linear − tok_linear per model; − foreign null",
        claim="punctint-q Stage-1 RE-QUOTE (corrected matched-class "
              "grid, § 10): window − token margins at T64, linear "
              "probe class: gpt2 +0.110 / gemma +0.105 / llama "
              "+0.144, every window arm ≥ +0.12 above its "
              "width-matched foreign null — the screen's MEAN-arm "
              "margins were lower bounds and the corrected grid "
              "RAISES them (400-doc screen corpus; screen rows)",
        checks=[("gpt2", 0.110, 3), ("gemma", 0.105, 3),
                ("llama", 0.144, 3), ("nullgap_min", 0.12, 2)],
        got={"gpt2": _rq_margin("gpt2"),
             "gemma": _rq_margin("gemma2_2b"),
             "llama": _rq_margin("llama31_8b"),
             "nullgap_min": min(_rq_nullgap(k) for k in
                                ("gpt2", "gemma2_2b", "llama31_8b"))}))

    pv = _j("support_stats/stage2_variance_qrate_gpt2.json")
    pv2 = _j("support_stats/stage2_variance_qrate_gpt2_v2.json")
    p4 = pv["paired"]["txc_pre_minus_tsae"]["by_T"]["T4"]
    p8v2 = pv2["paired"]["txc_pre_minus_tsae"]["by_T"]["T8"]
    R.append(dict(
        id="R18",
        artifact="support_stats/stage2_variance_qrate_gpt2[_v2].json",
        key="paired.txc_pre_minus_tsae.by_T.{T4 v1, T8 v2}",
        claim="fineweb replication, gpt2 (per-model WEAK on v1 — K1 ✗ "
              "at the +0.05 bar): pre−tsae +0.028 [0.007, 0.050] at T4 "
              "(v1, t-bounded > 0 but under the bar); paired v2 +0.066 "
              "[0.050, 0.083] at T8 — the third independent instance "
              "of the receipted v1-conservatism pattern",
        checks=[("t4", 0.028, 3), ("t4_lo", 0.007, 3),
                ("t8v2", 0.066, 3), ("t8v2_lo", 0.050, 3)],
        got={"t4": p4["mean"], "t4_lo": p4["t_ci95"][0],
             "t8v2": p8v2["mean"], "t8v2_lo": p8v2["t_ci95"][0]}))

    lv = _j("support_stats/stage2_variance_qrate_llama31.json")
    lv2 = _j("support_stats/stage2_variance_qrate_llama31_v2.json")
    l4 = lv["paired"]["txc_pre_minus_tsae"]["by_T"]["T4"]
    l8 = lv["paired"]["txc_pre_minus_tsae"]["by_T"]["T8"]
    l4v2 = lv2["paired"]["txc_pre_minus_pertoken"]["by_T"]["T4"]
    R.append(dict(
        id="R19",
        artifact="support_stats/stage2_variance_qrate_llama31[_v2].json",
        key="paired.txc_pre_minus_tsae.by_T.{T4,T8} v1; "
            "txc_pre_minus_pertoken.T4 v2",
        claim="fineweb replication, llama31-8b (per-model NEGATIVE as "
              "scored, replication-T scope): pre−tsae −0.018 (T4) and "
              "−0.014 (T8) on v1 — the strong token code wins at the "
              "canonical readout; paired v2 pre−pertoken +0.033 "
              "[0.013, 0.052] at T4 (bounded) — and the llama TOKEN "
              "archs DROP under v2 (tsae 0.256→0.197), the inversion "
              "flagged for the post-deadline probe review",
        checks=[("t4", -0.018, 3), ("t8", -0.014, 3),
                ("t4v2", 0.033, 3), ("t4v2_lo", 0.013, 3)],
        got={"t4": l4["mean"], "t8": l8["mean"],
             "t4v2": l4v2["mean"], "t4v2_lo": l4v2["t_ci95"][0]}))

    # ---- B8 slen screen (mac-b, frozen card b7121a208) ----
    sg = _j("slen/results/screen_gpt2.json")["cells"]
    sl = _j("slen/results/screen_llama31_8b.json")["cells"]

    def _sc(c, face, T):
        return (c[f"{face}/T{T}/win_linear"]["acc_test"]
                - c[f"{face}/T{T}/win_shuf_linear"]["acc_test"])

    def _wcn(c, face, T):
        return (c[f"{face}/T{T}/win_linear"]["acc_test"]
                - c[f"{face}/T{T}/win_foreign_linear"]["acc_test"])

    grid = [(c, f, T) for c in (sg, sl) for f in ("lat", "lev", "disp")
            for T in (16, 32)]
    R.append(dict(
        id="R20",
        artifact="slen/results/screen_{gpt2,llama31_8b}.json",
        key="win_linear − win_shuf_linear (sc) and − win_foreign_linear "
            "(wc), 3 faces × 2 models × T ∈ {16,32}",
        claim="B8 slen: the pre-registered recency ladder "
              "(lat > lev > disp ≈ 0) COLLAPSES on both screened models "
              "— max |within-window shuffle cost| over 3 faces × 2 "
              "models × T ∈ {16,32} is 0.019 (llama lat T32) while "
              "width-corrected window content spans +0.020…+0.147 on "
              "the same grid; the lat face's order share never exceeds "
              "0.13 of its width-corrected content (pre-registered "
              "latch prediction: ≥ 0.5). 2-model coverage, gemma "
              "pending; PENDING TEAM REVIEW",
        checks=[("absmax_sc", 0.019, 3), ("wc_lo", 0.020, 3),
                ("wc_hi", 0.147, 3), ("lat_share_max", 0.13, 2)],
        got={"absmax_sc": max(abs(_sc(c, f, T)) for c, f, T in grid),
             "wc_lo": min(_wcn(c, f, T) for c, f, T in grid),
             "wc_hi": max(_wcn(c, f, T) for c, f, T in grid),
             "lat_share_max": max(
                 _sc(c, "lat", T) / _wcn(c, "lat", T)
                 for c in (sg, sl) for T in (16, 32)
                 if _wcn(c, "lat", T) > 0)}))

    def _gax(c, face, T):
        return (c[f"{face}/T{T}/actxmean_linear"]["acc_test"]
                - c[f"{face}/tok_linear"]["acc_test"])

    def _axw(c, face, T):
        return (c[f"{face}/T{T}/actxmean_linear"]["acc_test"]
                - c[f"{face}/T{T}/actxmean_foreign_linear"]["acc_test"])

    def _wd(c, face, T):
        return (c[f"{face}/wd/T{T}/actxmean_linear"]["auc"]
                - c[f"{face}/wd/tok_linear"]["auc"])

    R.append(dict(
        id="R21",
        artifact="slen/results/screen_{gpt2,llama31_8b}.json",
        key="actxmean_linear − tok_linear (± foreign null); wd arms",
        claim="B8 slen KEEPs (both as ORDER-FREE window faces, "
              "screen-side corpus 400 docs / 320 train, 12k train "
              "rows): lat +0.058/+0.056 at T32 (gpt2/llama, foreign-"
              "null margins +0.087/+0.081); lev +0.067/+0.115 at T64 "
              "(margins +0.082/+0.130, still rising at the stated "
              "under-span top); lev's BINDING within-doc control "
              "discharged: wd window gain +0.046/+0.092 AUC at T64. "
              "2-model coverage; PENDING TEAM REVIEW",
        checks=[("lat_g32_gpt2", 0.058, 3), ("lat_g32_llama", 0.056, 3),
                ("lat_w32_gpt2", 0.087, 3), ("lat_w32_llama", 0.081, 3),
                ("lev_g64_gpt2", 0.067, 3), ("lev_g64_llama", 0.115, 3),
                ("lev_w64_gpt2", 0.082, 3), ("lev_w64_llama", 0.130, 3),
                ("lev_wd64_gpt2", 0.046, 3), ("lev_wd64_llama", 0.092, 3)],
        got={"lat_g32_gpt2": _gax(sg, "lat", 32),
             "lat_g32_llama": _gax(sl, "lat", 32),
             "lat_w32_gpt2": _axw(sg, "lat", 32),
             "lat_w32_llama": _axw(sl, "lat", 32),
             "lev_g64_gpt2": _gax(sg, "lev", 64),
             "lev_g64_llama": _gax(sl, "lev", 64),
             "lev_w64_gpt2": _axw(sg, "lev", 64),
             "lev_w64_llama": _axw(sl, "lev", 64),
             "lev_wd64_gpt2": _wd(sg, "lev", 64),
             "lev_wd64_llama": _wd(sl, "lev", 64)}))

    tb = _j("lambda_intensity/results/topup_bounds_tsae.json")["verdicts"]
    R.append(dict(
        id="R22",
        artifact="lambda_intensity/results/topup_bounds_tsae.json",
        key="verdicts.{paired_pooled, welch_pre6_vs_tsae_pooled, "
            "welch_pre6_vs_tsae_new_only, *_excl_underband_POSTHOC}",
        claim="pre-vs-T-SAE T8 margin at n = 6 (tsae top-up complete; "
              "PENDING TEAM RATIFICATION of 2 caveats: cross-cache "
              "pooling, s3/s4 realized-l0 under band): paired diff "
              "+0.0569, one-sided 95% LB +0.0200, all 6 seed-diffs "
              "positive (sign-flip p = 1/64); Welch 6v6 LB +0.0272, "
              "p = 0.0030; caveat-free NEW-SEEDS Welch LB +0.0357 (the "
              "fallback quote if pooling is not ratified); POST-HOC "
              "under-band exclusion cuts against the headline: Welch "
              "in-band LB +0.0083 (thin), paired n = 4 LB −0.0088 NOT "
              "bounded — quote the bound WITH this disclosure",
        checks=[("paired_diff", 0.0569, 4), ("paired_lb", 0.0200, 4),
                ("n_pos", 6, 0),
                ("welch66_lb", 0.0272, 4), ("welch66_p", 0.0030, 4),
                ("new_welch_lb", 0.0357, 4),
                ("posthoc_welch_lb", 0.0083, 4),
                ("posthoc_paired_lb", -0.0088, 4)],
        got={"paired_diff": tb["paired_pooled"]["diff"],
             "paired_lb": tb["paired_pooled"]["lb95_one_sided"],
             "n_pos": float(sum(1 for x in tb["paired_pooled"]["seed_diffs"]
                                if x > 0))
             if "seed_diffs" in tb["paired_pooled"]
             else float(6 * bool(tb["paired_pooled"]["all_seeds_positive"])),
             "welch66_lb": tb["welch_pre6_vs_tsae_pooled"]["lb95_one_sided"],
             "welch66_p": tb["welch_pre6_vs_tsae_pooled"]["p_one_sided"],
             "new_welch_lb":
                 tb["welch_pre6_vs_tsae_new_only"]["lb95_one_sided"],
             "posthoc_welch_lb":
                 tb["welch_pre6_vs_tsae_excl_underband_POSTHOC"]
                 ["lb95_one_sided"],
             "posthoc_paired_lb":
                 tb["paired_excl_underband_POSTHOC"]["lb95_one_sided"]}))

    # ---- B7 refmark screen (mac-b, frozen card c46d58826) ----
    rg = _j("refmark/results/screen_gpt2.json")["cells"]
    rl_ = _j("refmark/results/screen_llama31_8b.json")["cells"]

    def _ax(c, T):
        return c[f"rlam/T{T}/actxmean_linear"]["acc_test"]

    def _vis(c, T):
        return c[f"rlam/T{T}/visible_evidence_floor"]["acc_test"]

    def _rwd(c, T):
        return (c[f"wd/T{T}/actxmean_linear"]["auc"]
                - c["wd/tok_linear"]["auc"])

    R.append(dict(
        id="R23",
        artifact="refmark/results/screen_{gpt2,llama31_8b}.json",
        key="actxmean_linear − visible_evidence_floor per T; wd arms; "
            "actxmean − tok",
        claim="B7 refmark: NO KEEP on either screened model — gpt2 "
              "KILL (every window arm at T ≥ 8 sits BELOW the "
              "visible-evidence floor, best −0.008, worst −0.069 at "
              "T64 where the floor itself reaches 0.456; the "
              "mandatory within-conversation control is flat, "
              "max |gain| 0.015 AUC); llama31 WEAK (window gains "
              "real but sub-bar — linear max +0.037 at T32, MLP max "
              "+0.049 at T64 — beating the floor only at small T by "
              "≤ +0.016, wd gain ≤ +0.019). Under-span 16× stated "
              "pre-run; 2-model coverage; PENDING TEAM REVIEW",
        checks=[("g2_axvis_max_T8up", -0.008, 3),
                ("g2_vis64", 0.456, 3), ("g2_wd_absmax", 0.015, 3),
                ("ll_gax_max", 0.037, 3), ("ll_axvis_best", 0.016, 3),
                ("ll_wd_max", 0.019, 3)],
        got={"g2_axvis_max_T8up": max(_ax(rg, T) - _vis(rg, T)
                                      for T in (8, 16, 32, 64)),
             "g2_vis64": _vis(rg, 64),
             "g2_wd_absmax": max(abs(_rwd(rg, T)) for T in (16, 32, 64)),
             "ll_gax_max": max(_ax(rl_, T)
                               - rl_["rlam/tok_linear"]["acc_test"]
                               for T in (4, 8, 16, 32, 64)),
             "ll_axvis_best": max(_ax(rl_, T) - _vis(rl_, T)
                                  for T in (4, 8, 16, 32, 64)),
             "ll_wd_max": max(_rwd(rl_, T) for T in (16, 32, 64))}))
    return R


def check(R):
    rows, n_fail = [], 0
    for r in R:
        quoted, got, verdicts = [], [], []
        for name, q, dp in r["checks"]:
            g = float(r["got"][name])
            ok = abs(g - q) <= 0.5 * 10 ** (-dp) + 1e-12
            n_fail += 0 if ok else 1
            quoted.append(f"{name}={q}")
            got.append(f"{name}={g:.{max(dp, 1) + 1}f}")
            verdicts.append("PASS" if ok else f"**FAIL** ({name})")
            if not ok:
                print(f"[FAIL] {r['id']} {name}: quoted {q}, "
                      f"recomputed {g:.6f}", file=sys.stderr)
        r["_quoted"] = "; ".join(quoted)
        r["_got"] = "; ".join(got)
        r["_verdict"] = ("PASS" if all(v == "PASS" for v in verdicts)
                         else "; ".join(v for v in verdicts if v != "PASS"))
        rows.append(r)
    return rows, n_fail


def write_md(rows, n_fail):
    L = []
    A = L.append
    A("# RECEIPTS — claim → artifact index (rebuttal insurance; "
      "runpod-b, panel-support-audit item 3)\n")
    A("Every rebuttal-quotable number: the claim as we would state it, "
      "the artifact + key it comes from, the commit that last produced "
      "the artifact, and a **recomputed-now** value checked against the "
      "quote. Regenerate + re-verify with "
      "`.venv/bin/python -m experiments.explorations.task_hunt."
      "receipts_check` (also wired into pytest: "
      "`tests/test_receipts_index.py`). Finding a quotable claim NOT in "
      "this table is a deliverable — add it here, never quote around "
      "it.\n")
    A(f"**Status: {'ALL PASS' if n_fail == 0 else f'{n_fail} FAILURES'}** "
      f"({sum(len(r['checks']) for r in rows)} recomputed values across "
      f"{len(rows)} claims).\n")
    A("| id | claim (as quotable) | artifact : key | commit | quoted | "
      "recomputed now | verdict |")
    A("|---|---|---|---|---|---|---|")
    for r in rows:
        commit = _commit(r["artifact"].split(" ")[0])
        A(f"| {r['id']} | {r['claim']} | `{r['artifact']}` : {r['key']} "
          f"| {commit} | {r['_quoted']} | {r['_got']} | {r['_verdict']} |")
    A("")
    A("Notes:")
    A("- R5 (NOT-bounded) and R10 (withdrawn 'anywhere') are "
      "**negative-space receipts**: their value is what must NOT be "
      "claimed. Quote them exactly as phrased.")
    A("- R9 corpus sizes are the probe's training rows recorded in each "
      "screen artifact (the estimator finding: quote the size beside "
      "any triage/unigram number; 400-doc fineweb readings are "
      "understatements).")
    A("- The λ̂ panel's post-arm code-rate amendment rows and both live "
      "Stage-2 panels carry paired v1+v2 columns; every λ number above "
      "is v1 (canonical per the 2026-07-25 methods decision).")
    A("- Mirror coverage (R12): Stage-3/mix arms lost mid-run (force "
      "majeure) — Stage-1 label ADOPT-consistent on the amended scope; "
      "frozen-card scope AMBIGUOUS-unresolved. See the close-out LOG "
      "entry.")
    A("")
    (HERE / "RECEIPTS.md").write_text("\n".join(L))


def main():
    rows, n_fail = check(build_receipts())
    write_md(rows, n_fail)
    print(f"-> {HERE / 'RECEIPTS.md'}  "
          f"({'ALL PASS' if n_fail == 0 else f'{n_fail} FAILURES'})")
    return n_fail


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
