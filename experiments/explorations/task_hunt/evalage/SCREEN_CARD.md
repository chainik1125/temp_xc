# `evalage` — SCREEN CARD (frozen before the screen runs)

**Owner + executor: `mac-c`.** Face: **`evalage_age`** alone (tokens
since the last evaluation cue, `sage`-shaped). Binding inputs:
`evalage/CARD.md` §§ 1–7 + § 9, the 3-leg pre-measure
(`labels/evalage_premeasure_3leg.json`, `a4971b688`), and the grids
receipt (`evalage/grids/grids_receipt.json`).

## 0. Venue + governance

Runs on **`mac-c-screen-0728` (`4dztelehvj8l5n`, L40S 48GB, $0.99/h)** —
my own pod, spun up by me. Governance rule 3 (never touch a pod you did
not spin up) is satisfied trivially; mac-d screens `sycgen` on THEIR
`jge1fuj9hqu8et` and neither of us reaches across.

Ledger: RUNPOD section of `briefings/MODAL_SPEND.md`. Warm-hold guard
from my STATUS (terminate if the screen has not started by ~06:00
London) is **discharged by this card** — the screen starts now.

## 1. Grids — DONE, committed at `a4971b688`

Transplant of mac-d's `sycgen/screen_grids.py` (their invitation). Both
streams come from the same `run_elicit.build_stream`, so the class
triple is identical and contiguous runs of it ARE the turns.

Receipts, two beyond mac-d's, because an error here silently moves
event positions and destroys the exact-labels property this whole
program is buying:

1. **22,412 runs across 400 docs** re-encode gpt2-token-identical;
2. the rebuilt **gpt2 leg is ARRAY-IDENTICAL to the stream** on all
   five arrays and its gap median equals the corpus receipt's **862.0**.

| leg | tokens | events | gap median |
|---|---|---|---|
| gpt2 | 2,037,398 | 1,542 | 862.0 |
| gemma2 | 1,926,859 | 1,542 | 832.0 |
| llama31 | 1,899,699 | 1,542 | 807.5 |

## 2. Caches (`cache_acts.py`)

reask_hr transplant via sycgen's, verbatim: single-layer `SCREEN_HS`
capture, replag chunk geometry (`SEQ_LEN` 128; BOS prefix for
gemma/llama, none for gpt2; non-overlapping content chunks; doc tails
dropped), **every cached row re-derived from the flat stream and
asserted byte-identical pre-forward**. `/workspace/evalage_caches/<model>/`.

## 3. Screen (`screen.py`) — the frame, and one deliberate divergence
   from sycgen

1. **GLOBAL terciles — NOT the within-domain frame.** sycgen needed
   domain-local bins because its domains were confounded; that was my
   own disposition-(c) ruling. `evalage` does not have that problem by
   construction: `evalage_plan` draws the topic **first** and never
   consults it when scheduling cues, so topic ⊥ event schedule. Copying
   sycgen's frame here would be cargo-culting a fix for a defect this
   scaffold does not have — and worse, it would break the match with
   the pre-measure whose bands are the entire justification for
   spending GPU. **gpt2 tercile edges are asserted equal to the
   committed 3-leg artifact** (`[8.7482, 9.9972]`); other legs use the
   same construction and record their edges (grids differ, so numeric
   equality is gpt2-only by design — the same disclosure mac-d made).

2. **Per-token arms FIRST** (`tok_linear`, `tok_mlp` lead the cell
   order). Standing rule; `emotional_instability` died exactly there.
   If the token already carries it, the window has bought nothing.

3. **Topic-vocabulary band BESIDE the verdict**, transplanted from
   mac-d's GO condition 3: per-topic train-fit unigram AUC
   (`type_mean_scores`, topic-restricted) **plus the two-leg cv —
   events/conversation AND tokens/conversation** — per topic, carried
   in the screen json and copied into the verdict.
   *This discharges one of the two fixes I owe.* My
   `vocabulary_control_check` conflated those two channels into one
   events-per-token number, which is why `evalage` passed the length
   channel by luck (uniform `max_new`) rather than by design. mac-d had
   already implemented the two-leg form; taking it here means the fix
   lands as measurement rather than as a promise. The `elicit_lib`
   change itself is still owed for the *plan-time* check.

4. **Within-CONVERSATION control is BINDING** — a `wd` SKIP blocks any
   KEEP (hunt4 § 4 clause 4). This arm matters more for `evalage` than
   it did for any prior candidate and is worth stating why: age
   **resets at every cue**, so within a document age is a sawtooth
   while position is monotonic. The within-conversation frame therefore
   breaks the age/position correlation (global Spearman **0.4226**) far
   more cleanly than a monotonic face would allow. If a window wins
   globally but dies within-conversation, the honest reading is
   position, not age.

5. **hunt4 § 4 KEEP/KILL verbatim** — `hunt4.verdict.score_model`
   imported unmodified. Bundle = majority over the three models.

Deviation from the reask_hr transplant, disclosed: **no `is_boundary`
term** — `evalage` has no boundary construct; cue turns are the only
events and are fully masked (never probe-eligible).

## 4. Pre-registration (directional, written before any GPU ran)

Stated so the result cannot be reinterpreted after the fact:

- **The visible-cue channel is already dead** and I do not expect it to
  resurrect: censored-age floors are 0.500 / 0.500 / 0.504 / 0.525 /
  0.567 at T = 4…64 and the claim zone is 0 / 0 / 0.27 / 1.69 / 4.48 %.
  A window win here **cannot** be floor-driven. If `visible_evidence_floor`
  comes back strong, something is wrong with the pipeline, not with the
  hypothesis.
- **The per-token baseline is the real threat.** Terciles sit at ages
  ~429 and ~1021 tokens; "how deep am I in this conversation" is
  correlated with the face at ρ = 0.42, and late assistant text
  genuinely differs from early. I expect `tok_*` to land clearly above
  chance.
- **My honest prior on the bundle is roughly 35–40 % KEEP.** The
  mechanism is plausible (an evaluation framing is salient and models
  do track conversational state), but "how many tokens ago" is a
  *fine-grained magnitude*, and there is no prior result in this
  program showing the residual stream carries one linearly at a single
  position. I am not predicting a win.
- **The most likely KILL mode is clause 1** (every window arm within
  +0.02 of the best token arm) — i.e. the window adds nothing over the
  token. The second most likely is clause 4 (`wd` erases the gain),
  which would read as position.
- A **WEAK** is a perfectly acceptable outcome and will be reported as
  WEAK, not narrated into a near-KEEP.

## 5. Sequence

Freeze (this card + `cache_acts.py` + `screen.py` + `verdict.py`, ONE
commit) → push → pod resets to the freeze pin → caches (3 models,
1 GPU, serial) → screens (per model, resumable) → verdict → harvest
JSONs → commit + ONE LOG entry → ledger actuals → **terminate the pod**
(governance rule 2: prefer TERMINATE, verify by API query after).

**On a KEEP**, mac-d's warm-pod matrix retrain is pre-authorized to
start within the hour (`f0ac106e4` item 3) — I notify, they execute.

_Recorded-by: claude-opus-5 (mac-c, owner + executor)_
