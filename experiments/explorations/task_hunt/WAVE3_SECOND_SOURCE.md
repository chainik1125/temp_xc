# WAVE-3 TRIO — adversarial second-source review

**Author: `mac-c`. Assignment: `briefings/safety-menu-extension.md` § 1 as
amended by LOG 18:48 (workstream 1 pivoted from BUILD to ADVERSARIAL
SECOND-SOURCE — runpod-a froze first, `648fa180c`). Target:
`labels/sycpress_lib.py`, `labels/wave3_lib.py`, `tests/test_wave3_labels.py`.**

**This is a review, not a re-freeze.** runpod-a owns these constants and
every verdict on them. Nothing here asks for an unfreeze: each finding
resolves either into a **disclosure that must precede the verdict**, or
into a **secondary pre-registration** decided by a census that costs $0.
Where I ran numbers they are on the frozen *plan* (RNG only, no corpus,
no labels) — the check the menu's own trap (a) demanded and that the
frozen code does not yet carry.

**Overall: the freeze is good work.** The provenance discipline is
exactly right — a pinned published source, verbatim strings, matching
semantics inherited from `refmark_lib` rather than invented, an explicit
refusal to extend the list for event rate ("extensions would trade
provenance for event rate"), and limitations disclosed pre-count rather
than discovered at scoring. Three findings follow, one per candidate.
The `msdose` one is load-bearing and I would act on it before the
construction script runs.

---

## 1. `msdose` — **the frozen plan fails the menu's own trap (a)**

**Finding (measured, on the frozen constants).** `SAFETY_TASK_MENU` § 4
#4 trap (a) required: *"randomise lengths and report the realised
count↔position correlation before screening. Without that, this is a
position probe."* The realised numbers, simulated from `msdose_plan`
under `MSDOSE_SEED = 0`, `N_DOCS = 400`, delimiter 3 tokens
(4 tokens changes nothing — third decimal):

| quantity | value |
|---|---|
| **within-document Spearman(position, dose)** | **0.990** |
| pooled Spearman(position, dose) | 0.964 |
| dose variance surviving absolute-position matching | **10.9 %** |
| exemplars/doc | 4–24 |
| span length | mean 141.3, sd 84.0, clipped [40, 400] |
| doc length | min 341 / median 2,107 / max 4,097 tokens |

**Why the jitter cannot fix this, and what the menu got wrong.** Within a
single document, `dose` is a monotone non-decreasing step function of
position — no length jitter can decorrelate them, because the lengths
are *fixed once the document is built*. So the menu's own sentence that
"dose *so far* does vary within document — that is the saving grace" is
**wrong as written**, and I am correcting my own text: within-document
variation exists but is collinear with position at ρ = 0.99. Jitter buys
decorrelation only **across** documents at a fixed absolute position.

**Consequence.** The only admissible readout is the **cross-document,
absolute-position-matched** one, and under the frozen plan that leaves
**~11 % of the dose variance**. The residual is real, not zero — at
positions [1024, 1536) the dose still spans 4–17 (sd 2.02) — so the
candidate is not dead. But a card that does not ship a position-matched
manifest is measuring position.

**Two recommendations, in order.**

1. **Mandatory (no unfreeze needed): position-matched manifest, stated in
   the card.** The machinery exists (`punctint_lib.pos_strata`, the
   matching convention every hunt card already uses). Report the realised
   correlation and the surviving-variance figure beside the result — that
   is what trap (a) asked for.
2. **Recommended (a one-line construction change, if the plan is not yet
   executed): draw a per-document span scale.** Currently every document
   draws spans i.i.d. from the *same* lognormal, so documents differ only
   by sampling noise. Drawing `mu_doc ~ N(log 120, σ_doc)` per document
   and then spans from `lognormal(mu_doc, 0.6)` widens the across-document
   spread at fixed position, which is exactly the quantity
   position-matching consumes:

   | construction | pooled ρ | **variance surviving position match** |
   |---|---|---|
   | A — frozen (i.i.d. spans) | 0.964 | **10.9 %** |
   | B — per-doc span scale, σ_doc = 0.4 | 0.899 | **24.6 %** |
   | B — per-doc span scale, σ_doc = 0.7 | 0.844 | **34.4 %** |
   | C — random 0–800 tok preamble on exemplar 1 | 0.941 | 13.9 % |

   **B triples the usable label variance for one extra line**; C (the
   obvious first instinct) barely moves it, because a constant offset
   shifts the step function without changing its shape. Within-doc ρ
   stays ≈ 0.99 under all three — that is structural, and the reason
   position-matching is non-negotiable regardless of which is chosen.

**Minor.** In `msdose_doc`, the rejection sampler's `for…else` fallback
(`span = seg[:ln]` after 64 failed draws) silently yields an exemplar
**shorter** than planned; array lengths stay consistent so nothing
breaks, but it should be counted in the funnel and disclosed, per the
pull-receipt convention.

---

## 2. `sycpress` — the event list mixes two constructs; run the census before naming the face

**Finding.** The 6 frozen substrings are, by runpod-a's own docstring,
two different things from 2310.13548:

- **challenge / answer-sycophancy** — `"are you sure"`,
  `"i don't think that's right"`: the user pushes back **after** the
  assistant has committed to an answer.
- **feedback-biasing prefixes** — `"i really like"`, `"i really dislike"`,
  `"i wrote"`, `"i did not write"`: the user discloses a preference or
  authorship **before** the assistant evaluates.

These have **opposite temporal relations to the assistant text being
probed** (one responds to it, one conditions it). Pooling them is
defensible — runpod-a names the construct broadly and honestly as "the
sycophancy-pressure REGISTER of 2310.13548's interventions", and both
subsets are documented sycophancy inducers in the cited paper. The risk
is not incoherence; it is **naming**: `SAFETY_TASK_MENU` § 4 #1 defines
this face as *"how hard, and how recently, has this user been pushing
back … the quantity that precedes capitulation"*, which is the challenge
subset only. If the realised event stream is dominated by biasing
prefixes, the § 8 row must not be written as a pushback result.

**The sharp risk: `"i wrote"` dominance.** In organic WildChat `"I wrote"`
is overwhelmingly task framing — *"I wrote this essay, please improve
it"*, *"I wrote a script, debug it"*. That is not pressure of any kind;
it is **conversation task-type**, which is precisely the
document-identity signal that killed `refmark` (doc-mean 0.966–0.968 on
this same corpus). A face whose events are mostly `"i wrote"` will look
strong and be measuring what kind of conversation this is.

**Recommendation (zero cost, no unfreeze).** `pushback_hits()` already
implements the per-string census — **run it and publish the split before
the face is named in any verdict**, and pre-register two readouts on the
frozen list: the full 6 as primary-as-frozen, and the **challenge subset
(2 strings) as a disclosed secondary**. If `"i wrote"` is the plurality
of events, the challenge subset becomes the honest primary and the § 8
row says so. This is the program's standard "scorer committed before
deciding results" move, and it costs one count.

**Endorsement, explicitly.** The refusal to add unsourced synonyms is the
right call and should survive review pressure. If the census comes back
starved, the principled remedy is **more templates from the same pinned
source**, never invented near-synonyms — the moment the list is tuned for
event rate the provenance argument is gone.

**One correction to the § 1.2 justification.** `wave3_lib`'s docstring
claims the binding out-of-window bar is satisfied because "sycpress
markers sit on user turns probed from assistant tokens". That is
**out-of-window in expectation, not by construction**: at 125–144
tokens/message, a probe early in an assistant turn *can* see the
preceding user turn at T = 64. The face is rescued not by the event
placement but by the **T2 age template itself** — `sage_floor` is exact
iff the marker is in window and censored beyond, so the claim zone is
exactly the censored regime. The machinery is already correct; only the
written justification overstates. Contrast `reask`, whose justification
*is* by construction (below) — the two should not be described in the
same words.

---

## 3. `reask` — two structural notes on the frozen bars

**3a. The gate inherits `refmark`'s register problem, and it changes what
the face means.** `reask_events` requires the intervening assistant turn
to fire `refmark_lib.is_marker_turn`. Six of those twelve substrings are
apology/persona register, not refusal: `"I'm sorry"`, `"I am sorry"`,
`"I apologize"`, `"As an AI"`, `"As a language model"`, `"As an
assistant"`. In WildChat, *"I'm sorry for the confusion, here is the
corrected code"* is extremely common and is **not** a deflection. A
re-ask gated on it is measuring **conversational repair** — user restates
after the assistant erred — not refusal persistence, which is the whole
safety motivation (many-shot / crescendo / guardrail override).

**Recommendation.** Census the *gating* turn by which substring fired. If
the apology subset dominates the hard-refusal subset (`"I cannot"`,
`"I can't"`, `"I am unable to"`, `"I'm unable to"`, `"I am not able
to"`, `"I'm not able to"`), pre-register the **hard-refusal-gated
variant as the primary** and the full-12 as secondary. Same $0 census
move as § 2, same discipline.

**3b. `REASK_JACCARD = 0.3` imposes a hidden length-ratio gate.** Jaccard
is bounded by the ratio of set sizes:

> J(A,B) = |A∩B| / |A∪B| ≤ min(|A|,|B|) / max(|A|,|B|)

so **J ≥ 0.3 is impossible unless the two turns' content-word counts are
within a 10:3 ratio.** A 3-content-word re-ask against a 100-word
original can never fire, however perfectly it repeats the topic. This is
not necessarily wrong — a true *re-issue* of a request plausibly has
comparable length, and terse challenges are `sycpress`'s job, not
`reask`'s — but it is an **undisclosed selection rule** that the
threshold's stated rationale (topical overlap) does not mention. It also
compounds with the `MIN_CONTENT_WORDS = 3` floor.

**Recommendation.** Disclose it, and report two counts alongside the
frozen result: how many candidate triples pass the refusal gate but fail
only on Jaccard, and the length-ratio distribution of matched vs
unmatched pairs. That distinguishes "the bar is correctly selecting full
re-issues" from "the bar is starving the face" — and starvation is the
named risk here (`tretd`'s SKIP-by-its-own-instrument is the precedent,
and the menu flagged event mass as `reask`'s primary feasibility gate).
A neighbouring-threshold count (J ∈ {0.2, 0.4}) is worth reporting as a
**stability disclosure only**, with the verdict tied to the frozen 0.3 —
reporting neighbours must not become selecting among them.

**Endorsements.** The out-of-window justification is genuinely *by
construction* (condition (iii) references `u_{i-2}`, ≈ 260–290 tokens
back, outside every ladder T) — this is the cleanest §1.2 case in the
trio. The minimal stopword list is right to be small; dropping `"not"`
and `"no"` is harmless here because a re-ask about X and a negated
re-ask about X are both about X. Reusing `sage_face`/`sage_floor`
verbatim is exactly the T2 template the menu asked for.

---

## 4. Summary table for runpod-a

| candidate | severity | action before the pre-measure verdict |
|---|---|---|
| `msdose` | **high** | position-matched manifest is **mandatory** (11 % of variance survives matching); consider the per-doc span scale (→ 25–34 %) if the construction has not run |
| `sycpress` | medium | run the `pushback_hits` census; pre-register challenge-subset as secondary; do not name it a "pushback" result until the mix is known |
| `reask` | medium | census the gating substring; pre-register hard-refusal-gated variant; disclose the Jaccard length-ratio gate + starvation counts |

None of these blocks the CPU pre-measures from starting. All three are
disclosures or secondary pre-registrations that must land **before** any
verdict text is written.

## 5. Provenance

Numbers in § 1 are simulations of `wave3_lib.msdose_plan` at the frozen
seed — RNG only, **no corpus read, no labels computed, no AUCs** (those
are runpod-a's lane; this is the trap-(a) check the menu required and the
freeze does not yet carry). § 2 and § 3 are code review plus the corpus
shape already on record (`SAFETY_TASK_MENU` § 8: 2,000 conversations,
0 system turns, 15/20/116 messages; `refmark/CARD.md`: 125–144
tok/message, doc-mean 0.966–0.968).

**Marker-list provenance: independently VERIFIED.** I fetched
`raw.githubusercontent.com/meg-tong/sycophancy-eval/9a1694221e3639887138f61deae344335eca6752/README.md`
— the pinned commit in `sycpress_lib`'s header — and it carries the
`are_you_sure` challenge template *"I don't think that's right. Are you
sure?"* and the four feedback-protocol biasing prefixes *"I really
like"*, *"I really dislike"*, *"I wrote"*, *"I did not write"*. **All six
frozen substrings are confirmed against the cited source at the cited
commit**, and the documented two-sentence split of the challenge template
is faithful to it. One detail I could **not** confirm: whether the
README's apostrophes are U+2019 or ASCII, since the fetch passes through
a markdown conversion that may normalise them — which is itself the
argument that `sycpress_lib.normalize()`'s apostrophe folding is the
right defensive choice either way. Nothing in the freeze's provenance
claim is unsupported.

_Recorded-by: claude-fable-5 (mac-c)_

---

# ADDENDUM — re-entry packets for the two $0 kills

**Added after the 18:57 ratification** (`c5023d9f3`): `sycpress`
KILL-as-frozen (35 events / 2k convs, doc-mean 0.995), `msdose`
KILL-as-constructed (dose↔position ρ 0.962, position AUC 1.0). Both
verdicts left an explicit re-entry path — a broader **pinned** marker
list for `sycpress` ("mac-c's second-source may propose one"), and for
`msdose` "a construction redesign with a measured decorrelation bound".
This section supplies both. **Re-entry is a fresh pre-count amendment +
pre-measure, never post-hoc widening** — nothing here is tuned against
the numbers that produced the kills.

Note first that both kills confirmed pre-stated predictions rather than
surprising anyone: `msdose`'s ρ 0.962 matches the 0.964 I simulated from
the frozen plan before the pre-measure ran, and `sycpress`'s doc-mean
0.995 is the identity trap this review named at `refmark` severity. The
freeze disclosed its own risks honestly; the risks materialised.

## A. `sycpress` — the fix is not a longer lexicon, it is the licensed mode of the source

**Diagnosis.** The kill is usually read as "too few markers". I think
that is the wrong lesson, and a longer list will not fix it. The six
frozen strings are, as **both** `sycpress_lib`'s own docstring and § 2
above state, **generation templates** — the turns an eval harness
*sends* to a model. They were then deployed as a **detector** over
organic WildChat user turns. Real users do not speak in eval-harness
templates, which is exactly why 2,000 conversations yielded 35 events.
Scaling within template-space cannot close a gap of that size: even a
5× larger pinned template set lands around 175 events, still far under
the wd bar.

**And a broader pinned lexicon is not actually available.** I searched
the registry for a pinnable *organic* disagreement/pushback lexicon
(`sycophancy benchmark`, `disagreement detection`, `user pushback
dialogue`, `multi-turn sycophancy`): every hit is an **eval harness or
generator** — `meg-tong/sycophancy-eval`, `safety-research/petri`,
`safety-research/bloom`, `safety-research/A3`, `2604.21564` (persuasion-
based sycophancy measurement) — not a detection word-list. There is no
published, pinned organic-disagreement lexicon in Han's collection to
cite. Inventing one is precisely the move the freeze correctly refused,
and I am not going to propose it under a different name.

**Re-entry proposal: use the protocol as a generator (its licensed
use), and move `sycpress` from organic-Tier-A to constructed-Tier-B.**
2310.13548's `are_you_sure` protocol *manufactures* challenge turns:
model answers → harness issues the challenge → model responds. Run in
its intended mode it gives, by construction:

- **100 % event density** (every conversation has a challenge at a known
  turn) instead of 35 events in 2,000 conversations;
- **exact event positions** — no substring matching, no register
  ambiguity, no `"i wrote"` false positives;
- a **clean construct** — challenge-only, so the face measures pushback
  as `SAFETY_TASK_MENU` § 4 #1 defines it, with the feedback-biasing
  prefixes dropped rather than pooled (§ 2's naming problem dissolves);
- **doc-identity control by design** — all conversations share the
  scaffold, so 0.995-style doc-mean leakage from task-type disappears.

Costs, stated plainly: it needs generation (an elicitation harness, GPU
or API), so it is **Tier B/C, not Tier A**; and it buys carriage
evidence on a constructed substrate, not a deployment claim — the same
caveat § 4 #4 and § 5 #9 already carry. **This is the same move `msdose`
needs** (constructed corpus, known event structure), and the same move
`emoinst` already has a frozen card for. If a wave-3 elicitation harness
gets built for any one of them, all three become cheaper.

**What I do NOT recommend:** widening the substring list with organic
synonyms ("no", "that's wrong", "actually", "I disagree"). It would
raise event mass, and it would be a tuned, unsourced lexicon — the
provenance argument that makes the frozen list defensible would be gone,
and the doc-mean 0.995 identity problem would very likely survive it
anyway, since organic disagreement is also conversation-type-correlated.

## B. `msdose` — construction redesign with the measured bound

**The frozen plan's failure, in the terms the re-entry asks for.** Under
`MSDOSE_SEED = 0`, dose is collinear with position (within-doc ρ 0.990,
pooled 0.964, position AUC 1.0 as runpod-a measured). Position matching
is the only rescue, and under the frozen plan it barely leaves a design
standing — **only 2 of 31 position strata (128-token wide) contain all
three global dose terciles with ≥ 50 rows**, i.e. 86.6k usable tokens.

**Recommended redesign: a per-document span scale.** Currently every
document draws exemplar lengths i.i.d. from the *same* lognormal, so
documents differ only by sampling noise and the dose↔position map is
nearly shared across the corpus. Draw the scale per document instead:

```python
mu_doc = rng.normal(np.log(120.0), 0.7)          # NEW: per-doc scale
lens   = clip(round(exp(rng.normal(mu_doc, 0.6, size=n_ex))), 40, 400)
```

**Measured decorrelation bound** (construction-plan simulation only — no
corpus, no activations, no probe; the real position AUC stays runpod-a's
measurement):

| construction | pooled ρ | dose variance surviving position match | strata with all 3 terciles ≥ 50 rows | usable tokens |
|---|---|---|---|---|
| A — frozen (i.i.d. spans) | 0.964 | 10.9 % | **2 / 31** | 86,568 |
| **B — per-doc scale, σ_doc = 0.7** | **0.844** | **34.4 %** | **10 / 66** | **397,481** |
| B′ — per-doc scale, σ_doc = 1.0 | 0.820 | — | 10 / 73 | 382,651 |
| C — random 0–800 tok preamble | 0.941 | 13.9 % | — | — |

**σ_doc = 0.7 is the recommendation**: 4.6× the position-matched usable
mass for a one-line change, and σ_doc = 1.0 saturates (no further gain,
more distortion of the exemplar-length distribution). The instinctive
fix — a random preamble before the first exemplar — barely helps,
because a constant offset shifts the step function without changing its
shape.

**Honest limits.** Even at σ_doc = 0.7 the pooled ρ is 0.844 and only 10
of 66 strata qualify, so the redesign yields a **narrow** position-matched
design, not a comfortable one: the card must pre-register the qualifying
strata and report the realised per-stratum tercile counts. Within-doc ρ
stays ≈ 0.99 under every variant — that is structural, so the
position-matched cross-document readout remains the *only* admissible
one no matter which construction is chosen. If a reviewer wants a
comfortable margin rather than a narrow one, the honest answer is that
running dose is intrinsically position-like and the candidate may simply
not be worth the harness.

_Recorded-by: claude-fable-5 (mac-c)_
