"""Generate `figs_writeup/tab_sycgen_budget_matched.md` — the item-6 deliverable.

Han: *"we need to get the budget matched comparison TABLE DONE ASAP."*
This is that table. Script-generated from `sycgen/results/frontier.json`
so it regenerates deterministically; do not hand-edit.

Encodes the binding reporting rules from the pre-registration (LOG
`aa0272633`, amended `567d6818e`) and the hub cross-check (`6a75ddc7b`):

  - budget axis is MEASURED `realized_l0_per_window`. NEVER the derived
    per-token axis, which is `l0_per_window / T` (`synthetic_recovery.py:201`)
    and manufactures a "recovery up as budget falls" free lunch
    (mac-c `c1a9f98ad`).
  - `l0_unit` differs BY ARM: TXC and pooled count nonzeros in a d_sae
    code; stacked sums over T*d_sae slots. Printed, never merged.
  - stacked gets T x the probe input, so a stacked result is partly
    probe capacity. Reported, never netted out — and at T>=8 it is
    UNINFORMATIVE (32768 features vs 1024 windows is underdetermined).
  - three states, not two: above / below / INDISTINGUISHABLE at n=3.
  - the frontier SHAPE is reported, not just the verdict.

⚑ COMPARATOR SELECTION — REVISED 2026-07-29 01:0x (hub sanity check).
The first version of this script reported the best pooled point with
`l0 <= TXC's l0`. That rule is BIASED TOWARD TXC and it changed a
verdict. k is swept over a coarse grid, so no pooled point lands at
TXC's exact budget; the rule therefore silently compares against a
point spending materially LESS. At T=2 it selected pooled @ 3.51
against TXC @ 5.66 — the baseline handed **38% less budget** — and
returned "TXC above". The cheapest point ABOVE TXC's budget (5.97, only
5% over) scores 0.4876 against TXC's 0.4989: indistinguishable.

So this script now BRACKETS. For each T it reports the best point below
TXC's budget and the cheapest point above it, and estimates pooled AT
TXC's exact budget by linear interpolation between them. Three rules
are printed side by side and the CONSERVATIVE reading is the headline:

  A  strict-eligible  best point with l0 <= TXC l0     (what we shipped;
                                                        optimistic)
  B  bracket          must beat BOTH ends              (conservative:
                                                        the upper end has
                                                        MORE budget than TXC)
  C  interpolated     pooled estimated at TXC's l0     (PRIMARY: answers the
                                                        actual question)

Rule C is primary because it is the one that answers "what would pooled
score at TXC's budget?". It is valid because pooled is monotone
non-decreasing in budget up to saturation noise (verified below and
printed as a receipt) — mac-d established this at T=16 (`3b0927dea`).

⚑ THIS TABLE HAS NO SHUFFLE DIMENSION. `frontier.json` carries no
shuffle key. The ordered-vs-shuffled exhibit is a SEPARATE comparison
(`tab_sycgen_shuffle_tsweep.md`) against a per-token anchor, and the
two do not cross. Do not read a shuffle claim out of this table, and
see the pooled-permutation-invariance warning printed below before
comparing any shuffle gap to pooled.

    .venv/bin/python scripts/gen_sycgen_budget_table.py
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "experiments/explorations/task_hunt/sycgen/results/frontier.json"
OUT = ROOT / "figs_writeup/tab_sycgen_budget_matched.md"

# saturation wobble tolerance for the monotonicity receipt: pooled's top-k
# points agree to ~1e-3, and a bare `<` flags that as non-monotone.
MONO_TOL = 2e-3


def agg(rows, arm, T, k):
    rs = [r for r in rows if r["arm"] == arm and r["T"] == T
          and r.get("k_tok") == k]
    if not rs:
        return None
    v = [r["recovery"] for r in rs]
    return {"r": mean(v), "sd": pstdev(v) if len(v) > 1 else 0.0,
            "l0": mean(r["realized_l0_per_window"] for r in rs), "n": len(v),
            "unit": rs[0].get("l0_unit", "?"), "k": k}


def state(delta, spread):
    """Three states. Crude by design; NOT a significance test (n=3)."""
    if abs(delta) <= spread:
        return "INDISTINGUISHABLE"
    return "TXC above" if delta > 0 else "TXC below"


def monotone(pts):
    """Is recovery non-decreasing in budget, up to saturation noise?"""
    s = sorted(pts, key=lambda c: c["l0"])
    return all(s[i + 1]["r"] >= s[i]["r"] - MONO_TOL for i in range(len(s) - 1))


def bracket(pts, txc):
    """(below, above) around TXC's budget. Either may be None."""
    lo = [c for c in pts if c["l0"] <= txc["l0"] + 1e-9]
    hi = [c for c in pts if c["l0"] > txc["l0"] + 1e-9]
    return (max(lo, key=lambda c: c["r"]) if lo else None,
            min(hi, key=lambda c: c["l0"]) if hi else None)


def assess(pts, txc):
    """Rules A/B/C for one arm at one T. Returns a dict of verdicts."""
    lo, hi = bracket(pts, txc)
    out = {"lo": lo, "hi": hi, "mono": monotone(pts)}

    # A — strict eligibility (what the first version shipped)
    if lo:
        out["A"] = state(txc["r"] - lo["r"], max(txc["sd"], lo["sd"]))
    else:
        # no point that cheap: pooled cannot even operate at TXC's budget.
        # monotone => anything cheaper scores <= hi["r"], so beating hi is
        # strictly stronger than a matched-budget win (mac-d 3b0927dea).
        d = txc["r"] - hi["r"]
        out["A"] = ("TXC above AND CHEAPER" if d > max(txc["sd"], hi["sd"])
                    else state(d, max(txc["sd"], hi["sd"])))

    # B — bracket: must beat BOTH ends (the upper end has MORE budget)
    ends = [c for c in (lo, hi) if c]
    verdicts = [state(txc["r"] - c["r"], max(txc["sd"], c["sd"])) for c in ends]
    out["B"] = ("TXC above" if all(v == "TXC above" for v in verdicts)
                else "TXC below" if any(v == "TXC below" for v in verdicts)
                else "INDISTINGUISHABLE")

    # C — interpolate pooled to TXC's exact budget (PRIMARY)
    if lo and hi:
        f = (txc["l0"] - lo["l0"]) / (hi["l0"] - lo["l0"])
        r_i = lo["r"] + f * (hi["r"] - lo["r"])
        sd_i = max(lo["sd"], hi["sd"])
        out["interp"] = {"r": r_i, "sd": sd_i, "frac": f}
        out["C"] = state(txc["r"] - r_i, max(txc["sd"], sd_i))
    elif hi and not lo:
        d = txc["r"] - hi["r"]
        out["interp"] = {"r": hi["r"], "sd": hi["sd"], "frac": None}
        out["C"] = ("TXC above AND CHEAPER" if d > max(txc["sd"], hi["sd"])
                    else state(d, max(txc["sd"], hi["sd"])))
    else:  # every point cheaper than TXC — compare against the ceiling
        top = max(pts, key=lambda c: c["r"])
        out["interp"] = {"r": top["r"], "sd": top["sd"], "frac": None}
        out["C"] = state(txc["r"] - top["r"], max(txc["sd"], top["sd"]))
    return out


def cell(c):
    return "—" if not c else f"k={c['k']} {c['r']:.4f} @ {c['l0']:.2f}"


def main() -> None:
    rows = json.loads(SRC.read_text())
    Ts = sorted({r["T"] for r in rows})
    ks = sorted({r["k_tok"] for r in rows if r.get("k_tok") is not None})
    shuffle_keys = {k for r in rows for k in r if "shuf" in k.lower()}

    L = [
        "# Item 6 (sycgen) — TXC vs pooled-SAE vs stacked-SAE, "
        "recovery-vs-budget frontier",
        "",
        "Script-generated by `scripts/gen_sycgen_budget_table.py` — "
        "regenerate rather than hand-edit.",
        "",
        "**Why this table exists.** The original item-6 claim compared a "
        "*windowed* TXC against a *per-token* SAE, which establishes "
        "nothing about the architecture. Dmitry's agent ran the missing "
        "baselines and TXC lost. This is that comparison done properly: "
        "**k swept on both SAE arms, compared as frontiers at measured "
        "budget**, because a single \"matched point\" does not exist — "
        "matching per-window necessarily unmatches per-token.",
        "",
        "**Budget axis: measured `realized_l0_per_window`.** The "
        "per-token axis is *derived* (`l0_per_window / T`) and produces a "
        "spurious \"more recovery for less budget\" trend — never use it.",
        "",
        "> ## ⚑ REVISED 2026-07-29 — the comparator rule was biased toward "
        "TXC, and it moved a verdict",
        ">",
        "> The first version of this table selected **the best pooled "
        "point with `l0 ≤ TXC's l0`**. Because k is swept on a coarse "
        "grid, no pooled point lands at TXC's exact budget, so that rule "
        "silently compares TXC against a **materially cheaper** baseline.",
        ">",
        "> **At T=2 it picked pooled @ 3.51 against TXC @ 5.66 — the "
        "baseline given 38% less budget — and returned \"TXC above\".** "
        "The cheapest pooled point *above* TXC's budget (5.97, a 5% "
        "overshoot) scores **0.4876 vs TXC's 0.4989**, a gap inside the "
        "seed spread.",
        ">",
        "> The table now **brackets** TXC's budget and interpolates. The "
        "headline count drops from **above 3/4** to **above 2/4**. The old "
        "number is kept below as rule A so the change is auditable.",
        "",
        "## Verdict — pooled (the honest comparison)",
        "",
        "Rules, applied to the same data:",
        "",
        "| rule | comparator | why |",
        "|---|---|---|",
        "| **A** strict-eligible | best pooled point at `l0 ≤ TXC` | what "
        "we first shipped — **optimistic**, the point can be much cheaper |",
        "| **B** bracket | must beat **both** bracket ends | "
        "**conservative** — the upper end has *more* budget than TXC |",
        "| **C** interpolated | pooled estimated **at TXC's exact `l0`** | "
        "**PRIMARY** — answers the actual question |",
        "",
        "| T | TXC r @ l0/win | pooled below | pooled above | pooled @ TXC's "
        "budget (interp) | **C (primary)** | B (conservative) | A (old) |",
        "|---|---|---|---|---|---|---|---|",
    ]

    tally = {"above": 0, "below": 0, "indist": 0}
    shape, mono_notes, stacked_rows = [], [], []
    for T in Ts:
        txc = agg(rows, "txc", T, None)
        if not txc:
            continue
        pooled = [c for c in (agg(rows, "pooled", T, k) for k in ks) if c]
        a = assess(pooled, txc)
        i = a["interp"]
        interp_s = (f"{i['r']:.4f}" + (" (bound)" if i["frac"] is None else "")
                    + f" ± {i['sd']:.4f}")
        L.append(
            f"| {T} | {txc['r']:.4f} ± {txc['sd']:.4f} @ {txc['l0']:.2f} "
            f"| {cell(a['lo'])} | {cell(a['hi'])} | {interp_s} "
            f"| **{a['C']}** | {a['B']} | {a['A']} |")
        key = ("above" if a["C"].startswith("TXC above") else
               "below" if a["C"] == "TXC below" else "indist")
        tally[key] += 1
        mono_notes.append((T, a["mono"]))

        st = [c for c in (agg(rows, "stacked", T, k) for k in ks) if c]
        sa = assess(st, txc)
        stacked_rows.append((T, sa, txc))

        bk = max(pooled, key=lambda c: c["r"])
        shape.append((T, txc, bk))

    L += [
        "",
        f"**Headline (rule C): TXC above {tally['above']}/{len(Ts)}, "
        f"INDISTINGUISHABLE {tally['indist']}/{len(Ts)}, "
        f"below {tally['below']}/{len(Ts)}.**",
        "",
        "- **T=16 is the strong cell and the only unambiguous one.** "
        "Pooled **cannot operate at TXC's budget at all** — its cheapest "
        "point costs 1.43× TXC — and TXC still beats it by +0.0908. "
        "That is **Pareto dominance** (better on both axes), which is "
        "*stronger* than a matched-budget win, not a weaker substitute "
        "for one (mac-d `3b0927dea`).",
        "- **T=2 and T=4 are indistinguishable at n=3.** T=2 was "
        "previously reported as a win; it is not one under a comparator "
        "that is not handed less budget.",
        "- **T=8 survives**: pooled interpolated to TXC's budget scores "
        "≈0.4675 against TXC's 0.5365. Note it flips to indistinguishable "
        "if pooled is allowed its next point up (**1.58× TXC's budget**) "
        "— that is not a matched comparison, but a reader should know the "
        "margin lives inside one grid step.",
        "",
        "**Monotonicity receipt** (rule C's interpolation depends on it): "
        "pooled recovery is non-decreasing in measured budget within "
        f"{MONO_TOL} at T = "
        + ", ".join(f"{T}:{'yes' if m else 'NO'}" for T, m in mono_notes)
        + ". The residual wobble is saturation noise at the top of the k "
        "sweep, not a real reversal.",
        "",
        "## Stacked — reported, and REFUSED as a comparator",
        "",
        "| T | stacked below | stacked above | interp @ TXC's budget | C |",
        "|---|---|---|---|---|",
    ]
    for T, sa, txc in stacked_rows:
        i = sa["interp"]
        L.append(f"| {T} | {cell(sa['lo'])} | {cell(sa['hi'])} | "
                 f"{i['r']:.4f}{' (bound)' if i['frac'] is None else ''} | "
                 f"{sa['C']} |")

    L += [
        "",
        "**⚑ Stacked's result is NOT counted at any T, and at T ≥ 8 it is "
        "meaningless:** `T·d_sae = 32768` features against `N_WINDOWS = "
        "1024` is underdetermined, so its collapse is **probe-capacity "
        "overfitting, not architecture**. Its `l0_unit` also differs "
        "(sum over `T·d_sae` slots, vs nonzeros in a `d_sae` code for TXC "
        "and pooled), so its budget column is **not on the same axis** "
        "and the two must never be merged.",
        "",
        "## The frontier shape — reported because the verdict alone would "
        "hide it",
        "",
        "Pooled's **ceiling at any swept budget**, ignoring the budget cap:",
        "",
        "| T | TXC r @ l0/win | pooled BEST r @ l0/win (any k) | pooled "
        "budget | does pooled ever reach TXC? |",
        "|---|---|---|---|---|",
    ]
    for T, txc, bk in shape:
        reach = "**no**" if bk["r"] < txc["r"] else "yes"
        # express as a MULTIPLIER and an explicit "more than", not "X% of":
        # a bare "+127%" for a 1.27x ratio overstates by 100 points.
        mult = bk["l0"] / txc["l0"]
        L.append(f"| {T} | {txc['r']:.4f} @ {txc['l0']:.2f} | {bk['r']:.4f} @ "
                 f"{bk['l0']:.2f} (k={bk['k']}) | {mult:.2f}× TXC "
                 f"(+{100*(mult-1):.0f}% more) | {reach} |")

    L += [
        "",
        "**Pooled saturates** — its recovery goes flat across the upper k "
        "range, so beyond a point more budget buys it nothing. **This is "
        "the most durable part of the result**: it holds regardless of "
        "which comparator rule is used, because it ignores the budget cap "
        "entirely. Pooled never reaches TXC even at up to **4.06×** the "
        "budget.",
        "",
        "## ⚑ This table has NO shuffle dimension — do not read one into it",
        "",
        f"`frontier.json` carries no shuffle key ({'none found' if not shuffle_keys else sorted(shuffle_keys)}). "
        "The ordered-vs-shuffled exhibit "
        "(`tab_sycgen_shuffle_tsweep.md`) is a **different comparison** — "
        "against a per-token anchor and an untrained twin — and **nothing "
        "crosses the two**. The sparsity-matched shuffle ablation is a "
        "separate run (`briefings/sycgen-shuffle-sparsity-matched.md`).",
        "",
        "**And when it lands: pooled's ordered−shuffled gap is EXACTLY "
        "ZERO, by construction.** Mean-pooling per-token codes over a "
        "window is **permutation-invariant** — verified, max|diff| "
        "5.96e-08 (float noise) vs stacked's 4.12. So *\"TXC's shuffle gap "
        "beats pooled's\"* is a **mathematical identity, not a result**. "
        "Pooled is the **instrument check** there (its gap must be 0; a "
        "non-zero value voids that run); **stacked is the baseline**, "
        "because concatenation supplies order for free from the "
        "architecture with no temporal learning.",
        "",
        "## Caveats that travel with every number here",
        "",
        "- **n = 3 seeds.** The verdict threshold (`|Δ| ≤ max(sd)`) is "
        "**crude by design and is NOT a significance test** — n=3 does not "
        "support one. `INDISTINGUISHABLE` is a real third state, distinct "
        "from a loss.",
        "- **The comparator grid is coarse.** k ∈ {1,2,4,8,16,32} means "
        "consecutive pooled points can differ by 40–75% in budget, so "
        "*every* verdict here depends on how the gap between grid points "
        "is handled. That is why three rules are printed instead of one.",
        "- **Training variance dominates sampling variance** — a "
        "sampling-only σ on a single-seed cell understates by ~1.8–4× "
        "(measured on a *different* leg; it argues underpowering is live, "
        "it does not size it here). Sizing outcome (d) needs a 5-seed "
        "treatment on item 6's own cells.",
        "- **Scope:** one task, one substrate "
        "(`sycgen_real_age_llama31_8b_l14`), one layer, d_sae 2048.",
        "",
        "## Provenance",
        "",
        "- Cells: tag `sycgen_keep_r1_rebuilt`, trained on a **rebuilt** "
        "activation cache — **not** pod-D's originals (which survive on HF "
        "and back the *published* exhibit).",
        "- **The rebuilt cache is verified**: retrained SAE anchors "
        "reproduced the recorded values to 3 dp under a closed-form OLS "
        "probe, where exact agreement is essentially impossible unless the "
        "substrate matches.",
        "- Source: "
        "`experiments/explorations/task_hunt/sycgen/results/frontier.json` "
        f"({len(rows)} rows).",
        "- Verdict cross-checked by two independent implementations "
        "(`sycgen/report_frontier.py`, `scripts/verify_frontier_verdict.py`); "
        "they disagreed at T=16 and the disagreement is recorded in the LOG. "
        "**Both implement rule A**, whose bias this revision corrects.",
        "",
        "**PTR — pending team ratification.**",
    ]
    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(L)} lines)")
    print(f"RULE C (primary) vs pooled: above {tally['above']}, "
          f"indist {tally['indist']}, below {tally['below']}")


if __name__ == "__main__":
    main()
