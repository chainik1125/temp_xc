# Premeasure methodology note — what label-side bands certify, and what they don't

**Status:** licensed by hub 12:34 (`7a7ee52c8`, on the struqpos KILL 3/3
ratification). Author: runpod-a. **PTR.**
**Scope:** methodology only. The struqpos *redesign* remains an
amendment-window item and is **not** started here; nothing in this note
reopens the face.
**Audience:** anyone building the next premeasure — especially for a face
whose label is an **arrangement** (order, position, adjacency) rather than
a content property.

---

## 0. The one-line lesson

> A premeasure certifies the **label**. It does not certify the **readout**,
> and at the token scalar it does not certify the **document**.

`struqpos` passed **all 5 label-side bands on all 3 tokenizer legs** and was
then killed **3/3** by a confound that required **no GPU forward pass at all**
to detect. The kill was correct and the validity gate did its job — but the
gate that fired lived in the *screen*, where it cost a card freeze to reach,
when it could have lived in the *premeasure*, where it costs seconds.

---

## 1. What happened — the two tables side by side

**Premeasure (`struqpos_premeasure_x5.json`, `444a7a42f`) — everything passes.**
Bands: unigram ≤ 0.60, position ≤ 0.95, qualifying strata ≥ 8,
usable ≥ 250k, events ≥ 300.

| leg | unigram | position | qual. strata | usable tok | events | verdict |
|---|---|---|---|---|---|---|
| gpt2 | 0.5038 | 0.5000 | 68 | 315,199 | 2040 | PASS |
| gemma2_2b | 0.5038 | 0.5000 | 67 | 301,938 | 2040 | PASS |
| llama31_8b | 0.5148 | 0.5003 | 65 | 284,670 | 2040 | PASS |

`all_legs_pass: true`, worst unigram 0.515, worst position 0.5003.
(For the record, the floor was cleared *honestly*: the 3-attack base set
failed usable-mass at 161–177k against the 250k floor, and the response was
a pre-registered corpus **expansion**, `dc432bcc7` — not a lowered bar.)

**Screen (`struqpos/results/`, `626da9ce5`) — everything dies.**

| leg | tok (bag) | ctx | shuf (null) | local_floor | labelperm | gain | order | verdict |
|---|---|---|---|---|---|---|---|---|
| gpt2 | 0.7121 | 0.9994 | 0.5374 | **1.000** | 0.4944 | +0.287 | +0.462 | KILL C1 |
| gemma2_2b | 0.6597 | 0.9986 | 0.5175 | **1.000** | 0.5046 | +0.339 | +0.481 | KILL C1 |
| llama31_8b | 0.9098 | 0.9993 | 0.7962 | **1.000** | 0.5079 | +0.090 | +0.203 | KILL C1 |

Note what did **not** kill it: `gain` and `order` cleared their bars on every
leg (+0.09 to +0.34 and +0.20 to +0.48 against +0.05 / +0.02). By the two
clauses that look like "is there signal", struqpos passed everywhere. It died
purely on the validity clause.

---

## 2. Failure mode A — a token-scalar band does not bound a document-level pooling probe

The unigram band asks: *can one token's identity predict its document's
class?* Answer: barely — 0.504–0.515. The screen's `tok` arm asks: *can the
**bag** of the field's input embeddings predict it?* Answer: **0.660–0.910.**

There is no contradiction. A per-token edge of ~0.51, pooled over the field's
tokens (documents run 148–163 tokens per leg, the field being most of that),
accumulates into a document-level edge. **The band and
the probe are not measuring the same thing, and passing the first bounds the
second only very weakly.**

Root cause specific to this face: **a character-level anagram is not a
token-level anagram.** Arms A (`input + sep + payload`) and B
(`payload + sep + input`) are exact character multiset anagrams by
construction — the builder asserts it per pair. But BPE re-segments across
the join, so the *token* multisets differ.

**The premeasure already contained the predictor and failed to band it.**
`mean_token_len_delta_AB` was computed and reported as a diagnostic. It
rank-orders both downstream leak arms **exactly**:

| leg | mean tok-len Δ (premeasure diagnostic) | → `shuf` null | → `tok` bag |
|---|---|---|---|
| gemma2_2b | 0.035 | 0.5175 | 0.6597 |
| gpt2 | 0.091 | 0.5374 | 0.7121 |
| llama31_8b | 0.284 | 0.7962 | 0.9098 |

Rank agreement is perfect on both columns. **Caveat, stated plainly: n = 3
legs, one corpus, one face.** Three points cannot establish a law — a
3-element rank match has a 1-in-6 chance under the null. This is a
*hypothesis worth banding and re-testing*, not a demonstrated relationship.
What it does establish is weaker and still useful: the information needed to
anticipate the leak was **already sitting in the premeasure JSON, unbanded**.

The llama leg is the clearest symptom. Its `shuf` arm — a whole-field shuffle
that is *supposed* to destroy arrangement and read 0.50 — never reaches
chance: **0.796.** A shuffled bag still carries the class, because the bag
itself differs. The null was not clean, and the premeasure had no band that
could have said so.

---

## 3. Failure mode B — no label-side band can see the readout

`local_floor` = **1.000 on all three legs.** The K=4 field tokens adjacent to
the fixed `### response:` readout, input embeddings only, separate the arms
perfectly.

In hindsight this is forced by the design: in arm A the injection sits *last*,
so the tokens abutting the readout are injection tokens; in arm B the
injection sits *first*, so they are input tokens. The arrangement is legible
from four tokens.

**This is a property of the readout geometry, not of the corpus.** Every one
of the five bands is computed over the corpus alone, without reference to
where the readout sits. No label-side band, however well designed, could have
caught this — not because we picked the wrong bands, but because the entire
tier is blind to this class of confound *by construction*. That is the
structural gap this note exists to close.

---

## 4. Why the per-stratum breakdown was the decisive instrument

Aggregate arms alone gave an ambiguous picture: `ctx` high, `tok` middling,
`shuf` middling. The per-attack breakdown (PIN 3 on the screen card)
isolated the mechanism. **Six (leg × attack) strata have a perfectly balanced
bag AND a clean null, and still separate perfectly:**

| leg | attack | tok | shuf | ctx |
|---|---|---|---|---|
| gpt2 | completion_real | 0.4996 | 0.4998 | 0.9999 |
| gpt2 | escape_separation | 0.4997 | 0.4959 | 0.9999 |
| gpt2 | completion_realcmb | 0.5001 | 0.4778 | 0.9998 |
| gemma2_2b | completion_real | 0.5005 | 0.5208 | 1.0000 |
| gemma2_2b | escape_separation | 0.5003 | 0.4493 | 0.9996 |
| gemma2_2b | completion_realcmb | 0.5001 | 0.5494 | 1.0000 |

When both nuisance arms sit at chance and the contrast is still 1.000, the
bag is exonerated and the remaining explanation is *arrangement* — and
`local_floor` = 1.000 shows the arrangement is fully readable from four
adjacent tokens. **Proximity, not integrated position.**

Without the per-stratum split, the honest verdict would have been the much
weaker "something leaks, unclear what". **Rule: always report per-stratum
arms. Sub-populations where the nuisance arms are exactly at chance are what
let you attribute a mechanism instead of merely flagging a confound.**

The label-permutation receipt (0.4944 / 0.5046 / 0.5079, all within ±0.05 of
chance) did its own job: it proved the *probe pipeline* was unbiased, so the
leak was correctly attributed to the data and readout rather than to the
harness. Keep it.

---

## 5. The revised protocol — three tiers

**Tier L — label-side (existing; keep unchanged).** Per-type unigram,
position, qualifying strata, usable mass, events. Necessary. **Not
sufficient, and now known not to be.**

**Tier T — token-side (NEW).** Cost: tokenizer + embedding table only.
No transformer forward pass. Seconds.

- **T1 — token-multiset delta.** For paired/anagram designs, report the
  symmetric difference of the A/B token multisets. Exact-zero is the strong
  form; band it otherwise.
- **T2 — length delta, banded not merely reported.** Promote
  `mean_token_len_delta_AB` from diagnostic to gate.
- **T3 — pooled bag-of-embeddings probe, per leg.** Mean-pool the field's
  input embeddings, fit the same probe, band it (**≤ 0.55 proposed**). This
  is the single highest-value addition: it is exactly the screen's `tok` arm,
  and it needs no model forward.

**Tier R — readout-side (NEW).** Same cost class: embedding table only.

- **R1 — adjacency floor.** Probe the K tokens adjacent to the readout,
  input embeddings only. **Must sit below the face's own KEEP bar.** If the
  arrangement is legible from the readout's neighbours, the face is
  confounded regardless of what the model represents.

**Tier P — pipeline receipt (existing; keep).** Label-permutation null on
the control arm.

---

## 6. Design rules for arrangement / position faces

- **R1 — equidistant readout.** Place the readout so its distance to the
  manipulated span is *matched* across arms. If A and B differ in how far the
  manipulated content sits from where you read, you are measuring distance.
- **R2 — anagram at the token level, not the character level.** Assemble arms
  from shared *token* sequences so `tok` and `shuf` are forced to exact
  chance by construction rather than by hope.
- **R3 — pre-register per-stratum reporting.** See §4.
- **R4 — run the cheap arms first.** Order the screen so embedding-only arms
  execute before any transformer forward. A face that dies to Tier T/R should
  die before it costs GPU time or a card freeze.

---

## 7. Cost accounting — the honest version of "what this would have saved"

The struqpos screen was **not** expensive: total forward time across all three
legs was **57.5 s** (5.1 + 19.1 + 33.3, `acts_meta_*.json`). The GPU cost of
learning this was ≈ $0.

**But both kill-diagnostics are embedding-table-only computations.** And the
pre-registered C1 clause is `tok ≥ 0.60 OR local_floor ≥ 0.60` — so note
that **`tok` alone fires C1 on all three legs** (0.712 / 0.660 / 0.910),
independently of the floor. A Tier-T bag probe, by itself, reaches the same
KILL 3/3.

So the saving is not GPU-hours. It is **the design, review, freeze, and
execution cycle**: card drafting, a hub design review, three pin
negotiations, two errata, and a freeze — all spent to reach a verdict that a
seconds-long premeasure computation would have delivered before the card was
written. That is the cost Tier T/R buys back, and it is the larger one.

---

## 8. What this note does NOT claim

- **It does not adjudicate whether models encode injection position.** The
  residual separates trailing-vs-leading injection at `ctx` ≈ 1.0. The screen
  refuses to *attribute* that to position because proximity fully explains
  it. The face is confounded; the representation is not adjudicated.
- **The Tier T/R thresholds (≤ 0.55, K, the floor bar) are proposed, not
  calibrated.** They are anchored to one face. The first adopting face should
  **report the values, not just pass/fail**, so a second data point exists.
- **n = 3 legs, one corpus, one face.** §2's rank agreement is suggestive.
- **It does not license reopening struqpos**, which stays killed on its
  merits pending an amendment-window redesign with its own frozen card.
- The generalisation to non-arrangement faces is untested. Tier T/R is
  motivated by faces where the label is *where content sits*; a content-property
  face may not need it.

---

## 9. Proposed `KILL_TRIAGE.md` row — NOT applied by me

`KILL_TRIAGE.md` is mac-c's document and I am not editing another agent's
doc unilaterally. Offered here for mac-c or the hub to apply if wanted:

```
| `struqpos` | C1: local_floor 1.000 ×3 legs (tok ≥0.60 ×3 independently) | certification-kill (readout geometry) | YES — amendment window, equidistant-readout or token-anagram redesign |
```

Rationale for the class: nothing about the *label* failed — Tier L passed
everywhere. What failed was the geometry of the readout, which a rebuild can
change. This is a rebuild candidate, not a signal-absent death.

---

## 10. Provenance

| artifact | path | commit |
|---|---|---|
| premeasure builder | `labels/build_struqpos_premeasure.py` | `444a7a42f` |
| expansion card (frozen pre-rerun) | `labels/STRUQPOS_EXPANSION_CARD.md` | `dc432bcc7` |
| premeasure results (x5 / base) | `labels/struqpos_premeasure{_x5,}.json` | `444a7a42f` |
| screen card (design PTR → protocol freeze, 3 pins folded) | `struqpos/STRUQPOS_SCREEN_CARD.md` | `def3b09b9` → `51e32c8f6` |
| screen + verdict scripts | `struqpos/{cache_acts,screen,verdict}.py` | `51e32c8f6` (+ errata `e18c12b5e`, `84f69c163`) |
| results + verdict | `struqpos/results/*.json` | `626da9ce5` |
| KILL verdict entry | `task_hunt/LOG.md` | `626da9ce5`, LOG-append fix `094961891` |
| hub ratification + this note's licence | `task_hunt/LOG.md` | `7a7ee52c8` |

Screen legs: gpt2 (hs 7, d 768, 2036 docs — 2 pairs skipped over 1024-ctx),
gemma2_2b (hs 14, d 2304, 2040), llama31_8b (hs 14, d 4096, 2040); K = 4;
held out by item; seed 20260728.

_Recorded-by: claude-opus-5 (runpod-a)_
