# `evalage` — GENERATION CARD (frozen before any generation)

**Owner: `mac-c`, harness build authorized 2026-07-28 (LOG; briefing
re-opened). Corpus 2 of the authorized pair — my pick of the
safety-strongest Tier-C entry, bars-first. $0 so far; generation runs
on `mac-c-hunt-0728` under the key governance, inside the ≤$100 slice.**

## 1. This is deliberately NOT menu #12 as specified

Menu § 6 #12 `evalcue` proposed a **T1 rate of evaluation-context
cues**, and I predicted there that it dies: *"a visible-marker-rate
face, which is the single most reliably fatal shape in the § 8 record"*
(`oprate` at a full panel, `refmark` at screen, `drev` for $0), with
cue density near-constant within a document ⇒ identity. **That
prediction stands for #12 as an organic-corpus face.**

What changed is that we now control the corpus. Every one of tonight's
kills came from geometry we did not choose, and each has a fix that
only generation makes available:

| killer (measured tonight) | fix wired into the scaffold |
|---|---|
| visible floor — cue sits in the window | **cues spaced ≫ T**; `retryesc` showed this yields a floor of exactly 0.500 |
| identity — cue density is a doc constant | an **AGE** face varies within document by construction |
| kernel reach (`refmark`, `sycgen_rate`, `warddebt` at 154 tok) | an age face uses **no kernel at all** |
| vocabulary leak (`retryesc`, unigram 0.69–0.72) | **topic drawn independently of the cue schedule** — `evalage_plan` picks the topic before scheduling and never consults it |

**Face: tokens since the last evaluation cue** (T2 age), read at
assistant tokens.

## 2. Safety motivation

Evaluation-awareness → strategic underperformance: `2603.03824`
*In-Context Environments Induce Evaluation-Awareness* (the concern that
models *"strategically underperform, or sandbag"*), `2507.01786`
*Probing and Steering Evaluation Awareness*, `2505.01420`, `2407.04694`
*SAD*, *Petri 2.0* (alignment-blog). My own menu makes `sandbag` (#15)
**strictly downstream of #12**, so this is the correct entry to run
first — the awareness state is the input the withholding face needs.

## 3. Out-of-window by construction (binding bar)

Cues are **USER-turn framing statements inserted by the scaffold**, from
a frozen 5-template set. The model never authors an event — reusing
`retryesc`'s `is_failure_turn` rule, because "the model noticed" being
the event is the visible-cue trap in its purest form. Cue text is
masked out of probe eligibility; probes read assistant tokens only.
Whether a cue has occurred is therefore a property of text outside the
eligible stream, while the age is a distance — the § 1.2 principle.

## 4. Clock, stated first (binding bar), from the frozen plan

400 documents, 28.0 turns/doc mean, **3.85 cues/doc**, cue gaps
log-uniform over 2–12 turns (median 5 ⇒ **≈ 900 tokens** at generation
lengths). CPU stub run of the real pipeline realised median gap 472
tokens and a claim zone of **0.0 / 0.0 / 0.8 / 5.5 / 14.7 %** at
T = 4/8/16/32/64.

That spread is the design target: weak floors at small T (the
`retryesc` result) with genuine in-window mass at large T, so the face
is neither floor-solved nor a single degenerate ramp. **The realised
gaps and claim zone from the actual generated corpus are receipts in
`elicit_evalage_v1_receipt.json`, and if they land far from this the
card says so before any screen.**

## 5. Vocabulary control — the `retryesc` lesson, made checkable

Topic is drawn **before** the cue schedule and never consulted by it.
Plan-level check across 16 topics: cue-rate spread **0.026**, cv
**0.048**. `elicit_lib.vocabulary_control_check` re-runs this on the
generated corpus; **a large spread means the leak is being rebuilt and
generation must not be trusted** — that is a stop condition, not a note.

## 6. Pre-registered bands (absolute only — the `msdose_r1` lesson)

1. `unigram_auc` ≤ **0.60** — the bar that killed `retryesc`; here it
   is the direct test of whether vocabulary control worked.
2. `doc_mean_only_auc` ≤ **0.88**
3. `position_auc` ≤ **0.95**
4. qualifying strata ≥ **8**, usable position-matched tokens ≥ **250k**
5. events ≥ **300** (plan gives ~1,540)

Must pass every band on all three tokenizers, or label-side KILL and no
screen. Floors reported per T as evidence.

## 7. The binding condition, and what this design does NOT buy

**The per-token baseline runs FIRST**, and the screen's window-vs-token
contrast *is* that baseline. Removing confounds is not the same as
producing a positive: whether a probe carries "a cue occurred, N tokens
ago" once the cue is outside the window is a genuine empirical
question. If post-cue assistant text is per-token readable, this dies
like `emotional_instability`, and **nothing in the scaffold is tuned to
make a window win** — the spacing removes the floor's advantage, it
does not hand the window an answer.

## 8. Generation provenance (required in-card)

Pod-hosted open weights from our HF caches; model id + resolved commit
sha, seed, temperature, top-p all pinned and written to the receipt, so
the corpus is exactly reproducible. **The corpus is MODEL-GENERATED and
disclosed as such** — whether it enters a rebuttal exhibit or the
appendix is the paper owner's call at quote time, per the authorization.

_Recorded-by: claude-fable-5 (mac-c)_

---

## 9. AMENDMENT — backend switch to the MATS Claude API (2026-07-28 ~02:05)

**Written before generation runs; § 1–7 (face, traps, bands, kill rule)
are UNCHANGED. Only the generation backend and its provenance claim
change.**

**Why.** vLLM would not install against the pod image's torch (it
downgraded to cu121 and broke the system transformers; an isolated venv
did not resolve it either). Both pods are terminated and API-verified,
~$0.85 actuals, **zero generation produced**. mac-local authorized a
per-card backend switch (`a74f52cb9`) and explicitly nudged that the
vLLM fight is optional.

**The provenance claim changes, and I am not letting that pass
silently.** Card § 8 said pod-hosted open weights with a pinned weight
sha ⇒ **bit-exact** reproducibility. Under the API the pin is
**model-id + API version, not a weight sha** ⇒ the corpus is
**reproducible-in-expectation, not bit-exact**, because the served
model can change under a stable name. That is a genuine weakening. It
is recorded in the receipt's `generation.backend` field verbatim, so
anyone reading the artifact sees which guarantee they have.

**What is unaffected:** the labels remain exact by construction (the
scaffold inserts the cues and knows their positions, whatever generates
the prose), so the *label* provenance is untouched — only the *text*
provenance weakens. The realised-gaps and vocabulary-control receipts
still gate the corpus, and the per-token baseline is still binding.

**If bit-exactness is later required** for an exhibit, the corpus is
regenerable on open weights from the same frozen scaffold; that is a
re-run, not a redesign.

_Recorded-by: claude-fable-5 (mac-c)_
