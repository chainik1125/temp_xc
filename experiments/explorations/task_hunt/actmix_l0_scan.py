"""ACTMIX W2 Stage-1 scan: realized vs nominal l0 across every hunt datasource.

Normalization: the matched UNTRAINED cell (n_steps==0, same ds/arch/T/k) is the
nominal oracle -- untrained realizes nominal exactly (ReLU zero-picks only bite
once training makes values negative... actually untrained realizes full nominal
empirically; verified below). Fallback when no untrained twin: analytical
nominal per window = k_pos (sae, tsae, txc_batchtopk_post) or k_pos*T
(stacked_batchtopk, txc_batchtopk_pre).
"""
import json
from collections import defaultdict
from statistics import mean

HUNT = ["dial_real_ttrend_gpt2_l7", "ward_real_lambda_base_l12",
        "dial_real_dqgap_llama31_8b_l14", "ward_real_slope8_distill_l14",
        "ward_real_oprate_case_base_l12", "fineweb_punctint_q_gemma2_l14",
        "fineweb_punctint_q_gpt2_l7", "fineweb_punctint_q_llama31_l14"]
SHORT = {"dial_real_ttrend_gpt2_l7": "ttrend/gpt2",
         "ward_real_lambda_base_l12": "lambda/base",
         "dial_real_dqgap_llama31_8b_l14": "dq/8b",
         "ward_real_slope8_distill_l14": "slope8/dist",
         "ward_real_oprate_case_base_l12": "oprate/base",
         "fineweb_punctint_q_gemma2_l14": "punct/gemma",
         "fineweb_punctint_q_gpt2_l7": "punct/gpt2",
         "fineweb_punctint_q_llama31_l14": "punct/llama"}
PER_WIN_NOMINAL = {  # analytical fallback, per window
    "batchtopk_sae": lambda k, t: k,
    "tsae": lambda k, t: k,
    "txc_batchtopk_post": lambda k, t: k,
    "stacked_batchtopk": lambda k, t: k * t,
    "txc_batchtopk_pre": lambda k, t: k * t,
}

cells = defaultdict(list)  # (ds, arch, T, k, trained) -> [l0_per_window]
with open("results/leaderboard.jsonl") as f:
    for line in f:
        r = json.loads(line)
        ds = r.get("datasource", "")
        if ds not in SHORT:
            continue
        tc = r.get("training_cfg", {})
        ov = tc.get("arch_hparams_override", {}) or {}
        T = ov.get("T", 1)
        k = ov.get("k_pos")
        trained = tc.get("n_steps", 0) > 0
        m = r.get("metrics", {})
        if m.get("l0_per_window") is None:
            continue
        cells[(ds, r["arch"], T, k, trained)].append(m["l0_per_window"])

# untrained oracle check + main table
rows = []
untrained_dev = []
for (ds, arch, T, k, trained), vals in sorted(cells.items()):
    if k is None:
        continue
    nom_analytic = PER_WIN_NOMINAL[arch](k, T)
    twin = cells.get((ds, arch, T, k, False))
    nom = mean(twin) if twin else nom_analytic
    if not trained:
        untrained_dev.append((ds, arch, T, k, mean(vals), nom_analytic))
        continue
    rows.append({"ds": SHORT[ds], "arch": arch, "T": T, "k": k,
                 "n": len(vals), "real": mean(vals), "nom": nom,
                 "nom_src": "untrained" if twin else "analytic",
                 "ratio": mean(vals) / nom})

print("== UNTRAINED ORACLE CHECK (realized vs analytical nominal) ==")
bad = [(d, a, t, k, v, n) for d, a, t, k, v, n in untrained_dev
       if abs(v / n - 1) > 0.01]
print(f"{len(untrained_dev)} untrained cells; {len(bad)} deviate >1% from analytic nominal")
for d, a, t, k, v, n in bad:
    print(f"  DEVIANT {SHORT[d]:12s} {a:20s} T={t:<3} k={k:<4} realized {v:.2f} vs {n}")

print("\n== TRAINED CELLS: realized/nominal by (arch, T) — range across substrates ==")
byat = defaultdict(list)
for r in rows:
    byat[(r["arch"], r["T"], r["k"])].append(r)
print(f"{'arch':20s} {'T':>4} {'k':>4} {'cells':>5} {'ratio min':>9} {'med':>6} {'max':>6}  substrates")
for (arch, T, k), rr in sorted(byat.items()):
    ratios = sorted(x["ratio"] for x in rr)
    med = ratios[len(ratios) // 2]
    dss = ",".join(sorted({x["ds"] for x in rr}))
    print(f"{arch:20s} {T:>4} {k:>4} {len(rr):>5} {ratios[0]:>9.3f} {med:>6.3f} {ratios[-1]:>6.3f}  {dss}")

print("\n== FULL PER-SUBSTRATE TABLE (markdown) ==")
print("| substrate | arch | T | k | seeds | realized/win | nominal/win | ratio |")
print("|---|---|---|---|---|---|---|---|")
for r in sorted(rows, key=lambda x: (x["ds"], x["arch"], x["T"], x["k"])):
    print(f"| {r['ds']} | {r['arch']} | {r['T']} | {r['k']} | {r['n']} | "
          f"{r['real']:.2f} | {r['nom']:.2f} ({r['nom_src'][:4]}) | {r['ratio']:.3f} |")
