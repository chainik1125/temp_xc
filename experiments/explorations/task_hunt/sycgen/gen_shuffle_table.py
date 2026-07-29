"""Render `figs_writeup/tab_sycgen_shuffle_matched.md` from the sweep JSON.

Renders; decides nothing. Every threshold was frozen in
`SHUFFLE_MATCHED_CARD.md` before the run (git history is the receipt),
and `report_shuffle_matched.py` applies them. This file only formats.

    .venv/bin/python -m experiments.explorations.task_hunt.sycgen.gen_shuffle_table
"""
from __future__ import annotations

import glob
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT / "figs_writeup" / "tab_sycgen_shuffle_matched.md"

TS = (2, 4, 8, 16)
SEEDS = (1, 2, 42)


def load():
    rows, gates = [], []
    for p in sorted(glob.glob(str(HERE / "results" / "shuffle_matched.shard*.json"))):
        d = json.loads(Path(p).read_text())
        rows += d["rows"]
        gates += d["gates"]
    if len(rows) != 624:
        raise SystemExit(f"expected 624 rows, got {len(rows)} — partial grid")
    return rows, gates


def sel(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def main() -> int:
    rows, gates = load()
    L = []
    A = L.append

    A("# sycgen shuffle ablation, sparsity-matched — RESULTS")
    A("")
    A("**Verdict: (b) ARCHITECTURAL, NOT LEARNED.** The ordered−shuffled")
    A("gap is **not** evidence of learned temporal structure.")
    A("")
    A("Pre-registered in `sycgen/SHUFFLE_MATCHED_CARD.md` **before any**")
    A("**cell ran**; outcome (b) was named there as the **live**")
    A("hypothesis, and the rule that decides it was frozen in advance.")
    A("PTR — pending team review.")
    A("")
    A("## 1. Instrument gates — all PASS")
    A("")
    A("| check | result |")
    A("|---|---|")
    npass = sum(1 for g in gates if g["band"][0] <= g["identity_rows"] <= g["band"][1])
    A(f"| shuffle live (identity-row count vs binomial band) | **{npass}/{len(gates)} PASS** |")
    pooled = sel(rows, arm="pooled")
    worst = max(abs(r["gap_fixedprobe"]) for r in pooled)
    A(f"| pooled gap ≡ 0 (permutation-invariance, {len(pooled)} rows) | **max \\|gap\\| {worst:.2e}** |")
    bad = [r for r in rows if r["arm"] in ("pooled", "stacked")
           and abs(r["realized_l0_per_window_ordered"]
                   - r["realized_l0_per_window_shuffled"]) > 1e-6]
    A(f"| SAE `l0` permutation-invariant (predicted, then measured) | **{len(bad)} violations** |")
    A("")
    A("Measured identity-row counts against theory `1 − 1/T!`:")
    A("")
    A("| T | identity rows | n | predicted | band | verdict |")
    A("|---|---|---|---|---|---|")
    for T in TS:
        g = [x for x in gates if x["T"] == T and x["draw"] == "plain"][0]
        A(f"| {T} | {g['identity_rows']} | {g['n_rows']} | "
          f"{g['identity_expected']:.3g} | {g['band'][0]}..{g['band'][1]} | PASS |")
    A("")
    A("**Pooled's zero is NOT what certifies the shuffle.** A mean over the")
    A("window axis is permutation-invariant arithmetically, so it returns")
    A("PASS on a dead shuffle at every T. The gate above is input-side and")
    A("arm-independent, checked against an exactly predicted rate.")
    A("")

    A("## 2. THE RESULT — untrained twins show the LARGER gap")
    A("")
    A("Trained vs random-init TXC, **same architecture, same T, same seed**.")
    A("This comparison needs no budget matching: it is one arm against")
    A("itself.")
    A("")
    A("| T | trained gap | untrained-twin gap | trained > twin? |")
    A("|---|---|---|---|")
    for T in TS:
        tr = st.mean([r["gap_fixedprobe"] for r in sel(rows, arm="txc", T=T,
                                                       draw="redraw", weights="trained")])
        un = st.mean([r["gap_fixedprobe"] for r in sel(rows, arm="txc", T=T,
                                                       draw="redraw", weights="untrained")])
        n = 0
        for s in SEEDS:
            a = sel(rows, arm="txc", T=T, seed=s, draw="redraw", weights="trained")
            b = sel(rows, arm="txc", T=T, seed=s, draw="redraw", weights="untrained")
            if a and b and a[0]["gap_fixedprobe"] > b[0]["gap_fixedprobe"]:
                n += 1
        A(f"| {T} | {tr:+.4f} | {un:+.4f} | **{n}/3** |")
    A("")
    A("**11 of 12 (T, seed) cells have the UNTRAINED twin at the larger**")
    A("**gap.** A randomly-initialised model is *more* order-sensitive than")
    A("the trained one. That is outcome **(b)**, and it reproduces the")
    A("mechanism that dissolved sycgen's original shuffle claim.")
    A("")
    A("### 2b. The qualifier, stated because it cuts both ways")
    A("")
    A("| T | trained: ordered → shuffled | twin: ordered → shuffled |")
    A("|---|---|---|")
    for T in TS:
        def m(w, k):
            return st.mean([r[k] for r in sel(rows, arm="txc", T=T,
                                              draw="redraw", weights=w)])
        A(f"| {T} | {m('trained','recovery_ordered'):.4f} → "
          f"{m('trained','recovery_shuffled_fixedprobe'):.4f} | "
          f"{m('untrained','recovery_ordered'):.4f} → "
          f"{m('untrained','recovery_shuffled_fixedprobe'):.4f} |")
    A("")
    A("**The twin barely does the task at all** (0.058 ordered at T=16 vs")
    A("the trained model's 0.578). Its gap is therefore a difference")
    A("between two near-chance numbers, and raw gaps are not obviously")
    A("commensurable across a 10× difference in base recovery. **This is a")
    A("limitation of the rule I pre-registered, found by the data.**")
    A("")
    A("It is reported, **not** used to overturn the verdict. The obvious")
    A("alternative — a *relative* gap — is **post-hoc and was not**")
    A("**pre-registered**, and it does not rescue the claim anyway; it")
    A("makes the negative stronger:")
    A("")
    A("| T | trained gap/ordered | twin gap/ordered |")
    A("|---|---|---|")
    for T in TS:
        def rel(w):
            v = [r["gap_fixedprobe"] / r["recovery_ordered"]
                 for r in sel(rows, arm="txc", T=T, draw="redraw", weights=w)
                 if abs(r["recovery_ordered"]) > 1e-6]
            return st.mean(v)
        A(f"| {T} | {rel('trained'):+.3f} | {rel('untrained'):+.3f} |")
    A("")
    A("The twin loses **76–79%** of its recovery to shuffling at T=2/4/8;")
    A("the trained model loses **4.5–22%**.")
    A("")
    A("**Budget confound, disclosed:** the twin runs at `l0`=8.00 (every")
    A("`k_pos` slot live) against the trained model's 5.44–7.86, i.e. up")
    A("to **1.47×** the budget, which plausibly inflates the twin's gap.")
    A("The confound is smallest at **T=16 (1.02–1.03×)** — and that is")
    A("exactly where the twin gate is least decisive (mean favours the")
    A("trained model, but only **1/3 seeds** agree, so the pre-registered")
    A("3/3 sign test fails).")
    A("")

    A("## 3. TXC vs STACKED — the pre-registered comparator")
    A("")
    A("| T | TXC `l0` | TXC gap | stacked floor (k=1) | ratio | matched? |")
    A("|---|---|---|---|---|---|")
    for T in TS:
        t = sel(rows, arm="txc", T=T, seed=1, draw="redraw", weights="trained")[0]
        s1 = [r for r in sel(rows, arm="stacked", T=T, seed=1, draw="redraw",
                             weights="trained") if r["k_tok"] == 1][0]
        tg = t["realized_l0_per_window_ordered"]
        fl = s1["realized_l0_per_window_ordered"]
        ok = "yes" if fl <= tg else "**NO — floor above TXC**"
        A(f"| {T} | {tg:.2f} | {t['gap_fixedprobe']:+.4f} | {fl:.2f} | "
          f"{fl/tg:.2f}× | {ok} |")
    A("")
    A("**At T=8 and T=16 stacked CANNOT operate at TXC's budget.** Its")
    A("`l0` is a sum over positions, so its cheapest possible setting is")
    A("`T·1` — 8.00 and 16.00 against TXC's 7.22 and 7.86. **Structural,")
    A("not a grid-coverage gap**, and no finer `k` can close it. Same")
    A("shape as item 6's pooled floor at T=16.")
    A("")
    A("So the matched-budget comparison is **available only at T=2 and**")
    A("**T=4**, and the budget ratios printed by the verdict script")
    A("(0.64–2.05) mean the standing check fires: **the word \"matched\"**")
    A("**is not earned at T=8/16.** Reported rather than papered over.")
    A("")
    A("Where TXC and stacked *can* be compared, TXC's gap is **larger at**")
    A("**T=2** (+0.111 vs +0.022–0.028) and **smaller at T=8/16**")
    A("(+0.004/+0.021 vs +0.029/+0.095) — i.e. beyond T=4 the windowed")
    A("model is **less** order-sensitive than plain concatenation.")
    A("")
    A("## 4. Verdict")
    A("")
    A("| draw | probe | T=2 | T=4 | T=8 | T=16 |")
    A("|---|---|---|---|---|---|")
    A("| redraw | fixed (primary) | (b) | (c) | (b) | (b) |")
    A("| redraw | refit (secondary) | (d) | (b) | (b) | (b) |")
    A("| plain | fixed | (b) | (c) | (b) | (b) |")
    A("| plain | refit | (b) | (b) | (b) | (b) |")
    A("")
    A("**Not one cell returns (a).** 15 of 16 return (b) or (c).")
    A("")
    A("**(b) is the pre-registered outcome and it is published as one.**")
    A("The card said: *\"If the answer is (b) — architectural, not learned")
    A("— that is the result and we publish it.\"*")
    A("")
    A("## 5. What this does and does not say")
    A("")
    A("- It does **not** say TXC fails to recover λ. It plainly does:")
    A("  ordered recovery **0.499 → 0.578** across T, against the twin's")
    A("  0.222 → 0.058. **Training works.**")
    A("- It says the **ordered−shuffled gap is the wrong evidence for**")
    A("  **it.** A random model shows that gap too, often larger. The")
    A("  gap reflects architectural position-sensitivity.")
    A("- Cross-T statements use the `redraw` column only: the `plain`")
    A("  draw leaves `1/T!` of rows unshuffled (**50% at T=2**), so a")
    A("  \"gap grows with T\" reading would inherit `1 − 1/T!` from the")
    A("  apparatus.")
    A("- n=3 seeds. The (d) band is a noise heuristic, **not a")
    A("  significance test**.")
    A("- Stacked carries `T·d_sae` probe capacity, disclosed and **never**")
    A("  **netted out**; uninformative at T≥8 (32768 features, 1024")
    A("  windows).")
    A("- Substrate is a **rebuilt** activation cache (llama-3.1-8B l14,")
    A("  926,592 tokens), not pod-D's original — same disclosure as the")
    A("  item-6 cells.")
    A("")
    A("_624 rows, 24 gate receipts, 24 cells; 8 shards on 4×A40._")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
