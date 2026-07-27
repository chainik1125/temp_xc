# `warddebt` — LABEL-SIDE SCREEN CARD (frozen before the run)

**Owner: `mac-c`. Briefing: `safety-hunt-continuation.md` (Han: "the
HUNT MUST CONTINUE"). Menu source: `SAFETY_TASK_MENU` § 10.1 #23.
$0 — CPU, committed in-repo substrate, no pull, no judge, no API.**

## 1. Face

`debt = rate_case (class 2, obligations incurred) − rate_ver (class 3,
obligations discharged)`, both under the program's frozen kernel
(τ = 3, K = 8 sentences, causal, `min_history` 4, the current sentence
never an input). "How much asserted-but-unchecked work is outstanding."

Substrate: the committed `proof-operation-phase-runs/labels.json`
5-class sentence record + `wardmap` broadcast — the exact path
`build_oprate.py` walks. Every instrument is reused verbatim
(`bundle_core` binning / balanced manifests / by-trace split / triage
with kill authority, `trailing_count_incl` floors), so the numbers are
directly comparable to the `oprate` obituary.

**Mask generalization, stated not silent:** `oprate` masks rows whose
current sentence is the event class or unlabeled. Debt has TWO event
classes, so rows are masked where the current sentence is class 2 **or**
class 3 **or** unlabeled.

## 2. Safety motivation

CoT monitorability: a trace accumulating unverified steps is one whose
stated reasoning is less load-bearing — `2307.13702`, `2505.05410`,
`2510.27378`, `2607.07229`. A $0 negative closes "should we try Ward
for safety?" with a receipt instead of an opinion.

## 3. I am correcting my own prediction before running

Menu § 10.1 #23 predicted **KILL on anti-dup**, expecting
`|ρ(debt, rate_ver)| ≥ 0.8`. **That prediction is probably wrong, and I
want it on record before the numbers exist.** `oprate_stats.json`
already reports `corr(rate_ver, rate_case) = −0.032` — the parents are
essentially uncorrelated, so their difference cannot be collinear with
either: with comparable variances each parent correlation sits near
±0.71, *below* the 0.8 bar.

**Revised prediction: anti-dup PASSES and the killer is the visible
floor.** `oprate` died at a full 84-cell panel to a window-visible
event-sentence-count baseline, and its parents' visible-evidence AUCs
were already 0.83 / 0.78 at T32. Operation classes come in runs, so the
current window's visible class composition predicts the trailing
balance by **self-excitation** — without computing the kernel at all.
That is the mechanism to beat, and I expect it to win again.

## 4. Clock, stated first (binding bar)

Reported before any AUC: tokens per sentence, and the K = 8 kernel
support **in tokens** against the screened T ∈ {8, 16, 32}. If the
support far exceeds T, the window cannot *compute* the face and any
in-window signal is self-excitation rather than the trailing balance —
the reasoning that demoted `sycgen_rate` and killed `refmark`.

## 5. Gates, cheapest first (all $0)

1. **Anti-dup** vs `rate_case`, `rate_ver`, `lam_sc`, and λ̂ Ward
   backtracking. **Bar |ρ| ≥ 0.8 ⇒ collapse, label-side KILL.**
2. **Triage** (kill authority, `factory_lib`): token-identity AUC
   ≥ 0.65 or position AUC ≥ 0.70 ⇒ FAIL.
3. **Manifest size** ≥ 2,000 rows/class.
4. **Visible floor**, reported per T ∈ {8, 16, 32} on the SAME extreme
   -class test rows: net (incurred − discharged) trailing count, and
   incurred-only count. This is the `oprate` killer and is reported
   whether or not it fires a formal kill.

**Label-side KILL if any of 1–3 fails ⇒ no GPU screen is bought.** A
strong floor does not auto-kill here (the floor's formal authority is
the hunt4 § 4 panel), but it is decisive evidence against promoting
this candidate, and I will say so plainly rather than passing a weak
candidate upward to fill a slot.

## 6. What a pass would mean

Only that the face earns a hunt4-clone screen under § 4 KEEP/KILL
verbatim. It would not be a KEEP. Nothing here promotes anything.

_Recorded-by: claude-fable-5 (mac-c)_

---

## 7. VERDICT (run at freeze `fa52ab43f`) — gates PASS, but **DO NOT PROMOTE**: the construct is degenerate at screen scale

**My original menu prediction was wrong; my § 3 revision was right.**
Anti-dup passes comfortably — ρ vs `rate_case` **+0.706**, vs
`rate_ver` **−0.605**, vs `lam_sc` −0.012, vs λ̂ Ward −0.052, all under
the 0.8 bar. The +0.706 lands almost exactly on the ±0.71 the
uncorrelated-parents argument predicted. Triage PASSES (token 0.619 <
0.65, position 0.518 < 0.70); manifest ≥ 2,000/class. **No formal
label-side gate fires.**

**And the candidate should still not be screened.** The clock:
**19.2 tokens/sentence, so the K = 8 kernel spans ≈ 154 tokens against
a screened T ≤ 32** — a 4.8× overshoot. The measured consequence, on
the 7,689 extreme-class test rows the AUCs are actually scored on:

| T | discharge count in-window ≠ 0 | ρ(net, incurred-only) |
|---|---|---|
| 8 | **0.00 %** | **1.000000** |
| 16 | **0.00 %** | **1.000000** |
| 32 | 0.13 % | 0.999980 |

**The discharge half of the face — the entire reason "debt" differs
from a plain incurred-rate — is invisible inside the window.** The
verification events that make debt negative sit 1–8 sentences back
(19–154 tokens), outside T. So at screen scale `warddebt` *is*
`oprate`'s `rate_case` under a new name, and the visible-floor numbers
say so independently: 0.573 / 0.650 / **0.784** at T = 8/16/32, against
`rate_case`'s own **0.783** at T32. `oprate` died at a **full 84-cell
panel** to exactly this baseline.

Buying a GPU screen here would be paying to re-run `oprate`'s death
with a relabelled face. **Recommendation: no screen, no slot.** This is
the $0 negative the menu entry was written to buy — "should we try Ward
for safety?" now has a receipt instead of an opinion.

**Note for anyone who revisits Ward:** the obstacle is structural, not
this face. Any sentence-kernel face on this substrate spans ~154 tokens
against windows of ≤ 32, so the window can never compute it and only
self-excitation survives. A Ward face would need either a
token-scale event or a screen at T ≫ 154.

_Recorded-by: claude-fable-5 (mac-c)_
