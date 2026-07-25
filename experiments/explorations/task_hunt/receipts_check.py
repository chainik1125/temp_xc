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
        claim="pre-vs-T-SAE margin is NOT BOUNDED and must NEVER be "
              "quoted as significant: paired diff +0.0522, one-sided 95% "
              "LB −0.0413; Welch diff +0.0530, LB −0.0159, one-sided "
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
