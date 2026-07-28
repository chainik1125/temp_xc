# `retryesc_gen` — SCREEN CARD (frozen before any GPU runs)

**Owner/executor: `mac-c`.** Executes the screen for the corpus frozen
by `GENERATION_CARD.md` (`3f6ba0d3d`, §§ 2.2a/2.2b corrigenda applied).
Venue: pod `9fcz2d1zjk174z` (`mac-c-screen2-0728`, L40S $0.99/h) —
**deliberately the same venue class as the `evalage` screen so the two
are comparable.**

## 0. What has already been established (label-side, $0)

All **21/21** bands pass — 7 bands × 3 tokenizers:

| band | gpt2 | gemma2 | llama31 | bar |
|---|---|---|---|---|
| `unigram_auc` | 0.5431 | 0.5406 | 0.5434 | ≤ 0.60 |
| `doc_mean_only_auc` | 0.6696 | 0.6720 | 0.6719 | ≤ 0.88 |
| `position_auc` | 0.6135 | 0.6137 | 0.6114 | ≤ 0.95 |
| `floor_excess` | 0.1853 | 0.2064 | 0.2230 | [0.15, 0.25] |
| strata | 31/34 | 29/32 | 26/30 | ≥ 8 |
| usable | 556,858 | 498,439 | 460,159 | ≥ 250k |
| events | 2,809 | 2,809 | 2,809 | ≥ 300 |

Corpus: 300 docs, 946,546 gpt2 tok, **2,809 events on every leg**, gpt2
grid reproduces the generation stream **array-for-array**.

## 1. Face

**Primary — the KEEP claim rests here.** `retryesc_age` =
`log2(1 + tokens since the last repeat-failure event)`, i.e. `sage_face`
over `event_first`. Same helper, same shape as `sycgen_age` (the
program's gold) and `evalage_age`.

Tercile edges (gpt2) **6.919 / 8.165** ⇒ ages **120 / 286 tokens**,
against `evalage`'s 429 / 1021. **That is the density design visible in
the labels** and it is why `floor_excess` lands in band.

**Secondary — declared in the generation card § 3.1, reported whatever
it shows, and NOT promotable to the headline if the primary
disappoints:** `retryesc_rate` (in-window repeat-failure count). It is
the only route to the Q3 order table.

## 2. Frame decisions, argued not inherited

- **GLOBAL terciles**, not `sycgen`'s within-domain frame. Justified
  here by construction rather than by argument: the outcome schedule is
  drawn **before** the task and never consults it, measured at
  event-rate cv **0.0637** across 16 tasks, so task ⊥ event by design.
  (In `evalage` I made the same choice and had to *argue* it; here the
  generator guarantees it.)
- **Per-token baseline FIRST** — binding on any generated corpus, and
  the comparator that actually decides. The window must beat the anchor
  token, not merely the null.
- **Within-document (`wd`) frame BINDING.** This is the frame that
  erased and reversed `reask_hr`'s arm (+0.017 / −0.060 / −0.006).
- **hunt4 § 4 verbatim**, `verdict.score_model` imported unmodified.

## 3. hunt4 § 4 KEEP/KILL — unchanged, quoted so it cannot drift

**KEEP iff** an arm exists clearing **gain ≥ +0.05** ∧ **width-null ≥
+0.02** ∧ **its own T's visible floor**, simultaneously, ∧ `wd_ok`.

**KILL** if any of: (1) all window arms within +0.02 of tok; (2) all
within 0.02 of null; (3) all ≤ their floor; (4) `wd` present and not
`wd_ok`. **Else WEAK.**

Order ladder (Q3, **table-routing only, NOT part of KEEP**): wd
win−shuf ≥ +0.03 at any T ∈ {4,8,16,32}.

## 4. Pre-registration — written before any GPU ran

**Prior: ~50–55 % to a KEEP** (carried unchanged from GENERATION_CARD
§ 7; magnitude ~70–75 %, and the leak gate that dominated it has now
*passed* label-side, which is why I am not revising the number
upward — the screen is a different measurement and the prior was
stated for it).

**What I expect, and what would falsify each:**

1. **`floor_excess` in band ⇒ gain should clear +0.05.** In-band
   history is 13/16 cells and 4/4 faces. **Falsified if** gains land
   at `evalage`-like +0.03–0.046 despite `f` = 0.185.
2. **⚠ The floor is the live risk, and llama31 is the leg to watch.**
   `floor_excess` rises 0.185 → 0.206 → **0.223** across legs (coarser
   tokenizers ⇒ T = 64 covers more content). 0.223 is close to the
   +0.25 edge where **3 of 5 record cells lose to their own floor**. A
   split verdict with llama31 failing clause 3 while gpt2 passes is a
   **live and pre-registered possibility**, not something to explain
   afterwards.
3. **`wd` should NOT erase the arm.** `evalage` (generated) kept
   positive wd gains where `reask_hr` (organic) reversed. If `wd`
   erases here, the harness did not fix the confound and that is a KILL
   by clause 4.
4. **Order: I expect `retryesc_age` to FAIL the ladder** — 0/9 for age
   faces record-wide. `retryesc_rate` is the one that could pass. A
   ladder pass on the *age* face would be genuinely new and would need
   its own scrutiny before I quoted it.

**A WEAK will be reported as WEAK.** The prime directive is a sound
verdict, never a win; `evalage` was reported WEAK by the same hand.

**⚑ Gold-visibility:** if — and only if — the bundle is a **KEEP**, it
goes into `REBUTTAL_HANDOFF.md` the same beat, per Han's standing rule.
Label-side passes do not trigger it and have not.

## 5. Sequence

grids (done, $0) → cache activations (3 legs) → screen (3 legs) →
`verdict.py` (mechanical) → RESULT.md → **TERMINATE + API-verify** →
ledger actuals.

_Recorded-by: claude-opus-5 (mac-c, owner + executor)_
