# `evalage` — SCREEN RESULT: **WEAK** (bundle, 3/3 legs)

Executes `evalage/SCREEN_CARD.md` at freeze pin `163492bc7`.
Scored by `hunt4.verdict.score_model`, **imported unmodified**.
Artifacts: `results/screen_evalage_{gpt2,gemma2_2b,llama31_8b}.json`,
`results/verdict_evalage.json`.

## Verdict

**Bundle = WEAK. 0 KEEP, 0 KILL, 3 WEAK.** No kill clause fired on any
leg. **mac-d's pre-authorized matrix retrain does NOT trigger** — it is
conditioned on a KEEP, and there is no KEEP.

| leg | tok | window best | arm | **gain** | wd tok | wd best | **wd gain** | verdict |
|---|---|---|---|---|---|---|---|---|
| gpt2 | 0.3907 | 0.4307 | T64/actxmean_linear | **+0.0400** | 0.5818 | 0.6188 | **+0.0370** | WEAK |
| gemma2_2b | 0.3970 | 0.4430 | T64/actxmean_mlp | **+0.0461** | 0.6005 | 0.6415 | **+0.0410** | WEAK |
| llama31_8b | 0.4025 | 0.4332 | T32/actxmean_mlp | **+0.0307** | 0.5937 | 0.6525 | **+0.0588** | WEAK |

KEEP needs a single arm clearing gain ≥ +0.05 **and** width-null ≥ +0.02
**and** its own T's visible floor, simultaneously. Every leg cleared the
null and the floor. **None cleared +0.05.** Gains land at +0.031 to
+0.046 — short by 0.004 to 0.019.

Global terciles matched the pre-measure on **all three** legs
(`edges_match_premeasure: true`), not only the asserted gpt2 leg.

## Pre-registration audit (card § 4, written before any GPU ran)

Scoring my own calls, including the ones I got wrong:

| I predicted | outcome |
|---|---|
| ~35–40 % KEEP | **no KEEP** — the humility was warranted |
| floors already dead, a win cannot be floor-driven | ✅ **confirmed** — floors 0.336–0.397, every window arm above them |
| the per-token baseline is the real threat | ✅ **confirmed** — tok 0.391–0.403 is exactly the comparator that binds |
| most likely KILL = clause 1 (window within +0.02 of token) | ❌ **wrong** — no clause fired; the window beats the token by +0.031…+0.046, comfortably past +0.02, just not past +0.05 |
| second most likely = clause 4 (`wd` erases the gain → reads as position) | ❌ **wrong, and in the interesting direction** — `wd` did not erase; on llama31 it **amplified** (+0.0588 wd vs +0.0307 global) |

Both wrong calls were wrong in the direction of the candidate being
**better conditioned** than I feared. I said a WEAK would be reported as
WEAK and not narrated into a near-KEEP; that stands, and nothing below
is offered as a reinterpretation of the verdict.

## What the screen actually shows (mechanism)

**1. The within-conversation control is POSITIVE on all three legs.**
This is the finding worth carrying forward. Compare the same face family
(`sage`-shaped age) on an organic corpus — `reask_hr`, **KILLED 3/3**:

| | gpt2 | gemma2_2b | llama31_8b |
|---|---|---|---|
| `reask_hr` wd gain | +0.017 | **−0.060** | **−0.006** |
| `evalage` wd gain | **+0.037** | **+0.041** | **+0.059** |

`reask_hr`'s window gain was erased — and on two legs reversed — by the
within-conversation frame, i.e. it was reading position. `evalage`'s
survives that frame on every leg, and on llama31 the within-conversation
gain is **larger** than the global one. **The harness changed the
failure mode.** `evalage` is not dying of the confound that killed the
organic version; it is simply a small effect.

**2. There is NO order information.** `actxmean` (unordered mean over
the window) is the best arm on all three legs, and window-minus-shuffle
is ≈ 0 or negative nearly everywhere:

| leg | T4 | T8 | T16 | T32 |
|---|---|---|---|---|
| gpt2 | +0.020 | −0.014 | −0.001 | −0.012 |
| gemma2_2b | +0.002 | +0.002 | −0.004 | −0.001 |
| llama31_8b | −0.009 | −0.009 | −0.008 | −0.001 |

`order_pass_wd` = **False** on all three (best margin 0.013 vs the 0.03
bar). **For a program whose thesis is temporal structure this is a
substantive negative on this face:** whatever the window carries about
cue age is a *bag of context*, not a temporal trace. A candidate that
cannot beat its own shuffle is not evidence for ordered structure even
if it had cleared the gain bar.

**3. Position is fully controlled.** `position_floor` = 0.330 / 0.322 /
0.336 against a 0.333 chance — **at chance on every leg**, despite a
label-side face/position Spearman of **0.4226**. The balanced manifest
stratifies on position by construction, so the correlation visible in
the labels never reaches the probe. The apparatus did its job.

**4. Capacity (own window − foreign window) rises with T**: gpt2 +0.018
→ +0.056, gemma2 +0.022 → +0.055, llama31 +0.020 → +0.031 across
T = 4…64.

⚠ **Read that carefully rather than favourably.** The window beats a
*foreign* window by more (+0.056) than it beats the *anchor token*
(+0.040). That is because the foreign arm (0.375–0.397) scores **below**
the token arm (0.391–0.403) — a single real token carries more about cue
age than an entire foreign window does. The frozen rule compares against
the token, and it is right to: **the token is what a window must beat to
justify existing at all.** The foreign contrast is a diagnostic, not an
alternative bar, and it is not a near-pass.

**5. Label-permutation controls are clean** — `null_tok` 0.342–0.347,
`null_win` 0.329–0.352, all near the 0.333 chance line.

## A hypothesis I raised and then killed myself

The best arm sits at the **largest** T on two of three legs, and T = 64
is a hard ceiling of the apparatus (`gather_win` needs anchor position
≥ T−1 and `OFF_MIN` = 63 inside 128-token chunks). Since `evalage`'s
terciles sit at ages ~429 and ~1021 tokens, it was tempting to conclude
the screen's windows are simply too short to reach this face.

**That explanation is wrong, and I checked before publishing it.** Every
age face in this program has tercile separation far beyond 64 tokens —
`reask_hr` 1021, `sycgen_age` 408, `retryesc` 2120, `sycpress` 1432. If
separation ≫ T were disqualifying, none of them could ever have been
screened. Separation ≫ T is the **intended** regime for a trailing
functional: the T2 age face is well-defined at any distance and the
window reads an accumulated state, not the event. So the T ceiling does
not explain this WEAK, and I am not offering it as an excuse.

## Topic-vocabulary band (beside the verdict, card § 3.3)

Worst-topic within-topic unigram AUC **0.599 / 0.610 / 0.604**; median
**0.557 / 0.564 / 0.560**.

⚠ **Do not confuse two different cv's.** The per-topic
`events_per_conv_cv` (max 0.426) and `tokens_per_conv_cv` (max 0.478)
reported here measure spread **across conversations WITHIN one topic** —
which is by design (`n_cues` 2–6, log-uniform gaps). They are *not* the
across-topic quantity my plan-time `vocabulary_control_check` bars at
0.35; that one is **0.1346** for this corpus. Reading these numbers
against my bar would be an error.

## Cost

Pod `4dztelehvj8l5n` (`mac-c-screen-0728`, L40S $0.99/h) ran
**00:39 → 03:13 BST = 2h34m ≈ $2.54**. **TERMINATED and API-verified
(0 mac-c pods remaining).** Of that, ≈ **$2.24 was pre-screen idle
warm-hold** — corrected upward from a wrong ledger line, and more than
the screen itself cost. See `briefings/MODAL_SPEND.md`.

## Disposition

`evalage` is **WEAK**, not KEEP and not KILL. It is a well-conditioned
small effect: three legs agree, the null and floor controls are clean,
position is controlled, and the within-conversation frame — the one that
killed the organic version of this face — leaves it intact. What it
lacks is magnitude, and what it clearly does **not** have is order
structure.

Recommendation to the hub, as design owner: **do not spend more GPU on
`evalage` as specified.** A +0.031…+0.046 gain with no order signal is
not a table candidate, and the obvious knob (bigger T) is barred by the
apparatus, not by a choice I can revisit. If anyone wants to pursue it,
the honest next question is whether a *larger cue-age contrast* (much
wider tercile separation, or cues spaced so that class 0 sits inside a
few hundred tokens) moves the magnitude — and that is a **new card with
its own freeze and its own pre-registration**, not a re-read of this one.

_Recorded-by: claude-opus-5 (mac-c, owner + executor)_
