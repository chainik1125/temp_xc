---
status: active
owner: all agents (mac-c, mac-d) — hub is mac-local
issued-by: mac-local (hub)
priority: STANDING — read before asking the hub for a decision
---

# Overnight standing orders (Han asleep from ~01:5x 07-29)

Han: *"i'm going to head to sleep, continue orchestration of agentic
work."* **The hub stays awake. These are pre-authorised defaults so
nobody blocks on a human who is asleep, and hard stops so nothing runs
away while nobody is watching.**

## 1. Spend — authorised, capped, and self-terminating

- **The shuffle lane is authorised to scale.** Han's §8 directive is
  *"parallelize cells ACROSS GPUs, max throughput"* and stands.
- **Ceiling is 24, not a budget number** — `(T, seed, draw)` gives 24
  independent cells (`0b58144c9`), so **more than 24 GPUs is strictly
  wasted**, whatever the authorization allows.
- **Scale from the measurement, not the authorisation.** Post
  seconds/cell first; `24 × t_cell / n_gpu` sizes the pod.
- **≤ 8 GPUs needs no further approval.** Above that, post the
  pre-spend estimate in the LOG **before** creating the pod, and say
  what the extra GPUs buy in wall-clock.
- **HARD STOP: if the lane is not finished 3 hours after the pod comes
  up, terminate it and report.** A stalled pod at 02:00 with everyone
  asleep is the one failure mode nobody will catch.
- **Terminate at lane end, API-verify GONE, ledger both ends.** No
  exceptions, and this is the last thing done, not the first thing
  deferred.
- **Do not touch the 4 unattributed pods** (`reviewer-btk-tsae-300k`,
  `reviewer-headline-multiseed`, `stacked-em-steer`,
  `tsae-paper-widthmatch-probing`, together $9.41/h). They are not ours.
  House rule holds regardless of cost.

## 2. What you may do without asking

- Run the frozen shuffle card to completion, gates first.
- Fix any bug you find in your own lane, and say so in the LOG.
- Refuse to report on a partial grid — the reporter already does.
- **Publish a negative.** The prime directive has not moved: *a sound
  verdict, never a win.* If (b) or (c) fires, that is the result.
- Correct the hub. Three of tonight's best findings were reversals of
  my rulings; none needed permission.

## 3. What needs Han and must WAIT, not be guessed

- **Rotating the three pod-staged tokens** (`gh`, `hf_token`,
  `hf_token_datasets`) — still his call.
- **The unattributed pods** — his call, and now ~$226/day.
- **Coordination with Dmitry.** `stacked-em-steer` is running the
  Stacked-SAE-on-EM baseline that the Reviewer-1 response *quotes*
  (`.710`, the one task where stacked beats the TXC). **Do not edit
  that section of the response overnight** — two people computing the
  same number independently is how a rebuttal turns inconsistent.
- **Any change to a delivered exhibit's headline claim.** Log it, do
  not ship it.

## 4. Standing checks earned tonight — apply them, they each caught something

- **Print the comparator's budget ratio.** If it is not ≈1.0,
  "matched" has not been earned. (Cost us "above 3/4" → "above 2/4".)
- **State a gate's false-pass AND false-fail rate before it guards a
  run.** A1 could not fire when it should; its fix fired when it should
  not.
- **A gate and a positive control are different objects.**
- **Before recording that something PASSED a control, ask whether the
  control could have EXPRESSED the failure.** An escaped control is not
  a passed control.
- **Verifying a component is not verifying the rule built from it.**
- **Change one thing when diagnosing.** Bundling fixes destroys
  attribution — it cost two extra rounds on the rebuttal table tonight.
- **A negative result from an unverified instrument is not a finding**,
  including when it accuses you.
- **Stamp from `date` in the same command that writes the entry.**
- **Pre-register the FORMULA, not the VALUE.** `chance = 1/(populated
  classes)` survives a bucket turning out empty; the literal `1/6` does
  not. **A literal in a frozen card that could have been computed is a
  latent post-hoc edit waiting to happen** (mac-c `ebc752e90`).
- **Disclose a threshold change even when it was forced.** A justified
  change in a favourable direction is indistinguishable from a
  convenient one unless you say which it was — and nobody diffs a
  passing run.

## 5. Open, ranked, if a lane frees up

1. ~~n=3 → n=5 on item 6's own cells~~ **CONSIDERED DECLINE
   (mac-d `242a0dbaa`, hub-ratified).** Two reasons: the card costed it
   as a **free rider** on a pod that is now terminated (standalone
   ≈$6–8/1.5 h — a *new* spend decision), and its own frozen rule sets
   the n=5 bar at **5/5, stricter than 3/3**, so the cells carrying the
   verdict cannot be strengthened and the indeterminate ones are
   unlikely to resolve. **Expected return: "still indeterminate, better
   sized."** Re-open only if the cost premise changes.
2. ~~The RLHF metric collision~~ **CLOSED by mac-c (`d18c556db`),
   hub-ratified.** One real anomaly, not eight: 9 of 10 all-metric
   pairs are the same model under the `_btkonly` rename (discriminator:
   same base arch **and** matching `batch_size` ⇒ benign). The survivor
   is `tsae_btkonly` bs=32 identical to `batchtopk_sae_btkonly`
   bs=1024. **Remediation is ONE row to re-run**, and both rows from
   that 90-second window (`0.6588`, and a `0.5000000000` exactly) are
   to be distrusted. Nothing quoted depends on it — **do not
   re-escalate this.**
3. **⚑ `floor_reach` DEMOTED (hub, 02:0x).** "Minimise `floor_reach`"
   was my ruling and it is **withdrawn**: out-of-sample it ranks the
   only KEEP *behind* a WEAK. It is a **kill filter and a description**,
   **not a candidate-quality screen** — a high value kills you, a low
   one buys nothing. Do not rank candidates by it.
4. **ANSWERED (`2eb100e01`) — and it redirected the program.**
   **WHY DID SYCGEN SURVIVE?** Tonight's three findings compose into one claim —
   the floor is handed the **censored age** (`sage_floor`), `tok`
   **recovers age past 1024 tokens at 0.619**, so **age-based labels
   are squeezed from both sides** and density moves only one of them.
   **The hunt has been generating age labels**, which explains the kill
   record better than any individual candidate's flaws.
   **But sycgen's label IS an age and it cleared floor AND gain on
   three models.** Either it sits in a corner of the geometry the
   others missed (model, layer, corpus, or beyond-T mass) **or one of
   the three findings does not generalise as stated.** $0 to ask,
   answerable from artifacts already on disk, and it should be asked
   **before another corpus is generated**. **A theory that explains
   every failure and cannot explain the one success is not finished.**

## 6. ⚑ WHEN RESULTS LAND — the surfaces to update, in order (Han, 01:5x)

Han: *"update rebuttal handover doc as stuff comes"* and *"and also the
… dmitry-txcwins-10h … reviewer_responses_1.md"*. **Three surfaces now
carry sycgen claims and they must not drift.** Two copies of a live
document is the exact hazard that produced four contradictions tonight.

**Procedure, every time a sycgen number changes:**

1. **`REBUTTAL_HANDOFF.md` §6** — the internal record. Verdict block +
   the executive-summary line near the top (it has been missed before).
2. **`figs_writeup/tab_sycgen_*.md`** — regenerate, never hand-edit
   (`scripts/gen_sycgen_budget_table.py`).
3. **`docs/dmitry/reviewer_responses/reviewer_responses_1.md` on
   `arxiv`.** **⚑ FORMAT CHANGED 02:2x — the backslash-doubling rule
   below is RETIRED for tables.** Dmitry's agent converted the tables
   to **markdown pipe tables**, which render everywhere; the sycgen
   table now matches. **Write new tables as pipe tables** — no `$$`,
   no arrays, no escaping question. Both copies of the sycgen block are
   now byte-identical, so there is nothing to regenerate. Remaining
   inline math still must not wrap across a line.
4. **The same file on `dmitry-txcwins-10h` — and it is ACTIVELY EDITED
   by Dmitry's agent.** It moved twice in twenty minutes on 07-29.
   **Never copy your working file over theirs.** Take **their current
   tip** as the base, graft only your own block onto it, and diff to
   confirm nothing else moved. Their tip removed the `(k)` annotations
   between two of my reads; a stale base would have reverted it.
5. **Verify, do not eyeball:**

       .venv/bin/python scripts/check_response_numbers.py
       .venv/bin/python scripts/check_response_math.py --self-test <file>
       .venv/bin/python scripts/check_response_sync.py --fetch
       .venv/bin/python scripts/handoff_audit.py

   **`check_response_numbers.py` is the one that matters most** — it
   re-derives the quoted table from `frontier.json`. It caught a wrong
   figure (T=8 TXC quoted 0.537, true 0.536) that had survived two
   rendering checks and a prose rewrite on both branches. **Rendering
   and prose checks do not look at whether the numbers are true.**
   **Quote from the source, never from a printout of the source** — a
   4 dp display rounded again to 3 dp is how 0.536463 became 0.537.

**Constraints that travel with the response:** ≤10,000 chars in the
paste block, **no links, no images, no plots**, and inline `$…$` must
**never wrap across a line** — that last one is what broke the table
twice and it is invisible on inspection.

**⚑ VOICE — Han, 01:5x: *"rebuttal responses must be written in clear,
plain language, NO AGENTIC JARGON."*** The reviewer has not read our
LOG and does not share our vocabulary. **Every internal term is a cost
to them and a tell that the text was not written for them.**

| do not write | write |
|---|---|
| `sycgen`, task codenames | describe the data in a clause |
| "per-token-silent state" | "not visible in any single token" |
| "budget", "l0 budget" | "sparsity" |
| "seed spread" | "variation across seeds" |
| "matched measured `realized_l0_per_window`" | "read off at the sparsity the TXC actually uses" |
| "narrow positive", "outcome (d)" | say what the numbers show |
| "real-model task" | "real world task" (the document's own phrase) |

**Match the document's existing register**, which is declarative:
*"beyond the threshold, the TXC improves from 0.154 at W=3 to near
perfect recovery, 0.956 at W=10."* State what was done and what the
numbers are. **Caveats belong in clauses, not paragraphs** — hedging
competes with the claim for the same sentence and both lose.

**Check before pushing:** internal terms must appear **zero** times in
the *rendered* text (HTML comments do not render and may keep them for
our own bookkeeping).

**If a number moves in a direction that weakens a delivered claim, log
it and update the surfaces anyway.** *A sound verdict, never a win.*

**Delete this file when Han is back and the lane has closed.**

5. **⚑ THE PROGRAM'S MAIN QUESTION NOW: what makes an ARM strong?**
   sycgen won on **arm strength**, not on a low floor — its floor is
   the *highest* of the three candidates. The hunt spent its effort on
   the floor; the gain side is where the one success came from and
   where the upper-edge measurement said the binding constraint sits.
   **Neither instrument built tonight measures it.** Build that before
   generating another corpus.

**Delete this file when Han is back and the lane has closed.**

---

## 7. ⚑ HUNT CONTINUATION (Han, 02:1x: *"orchestrate continuation of task hunting and sanity checking overnight, fine to spin up more runpods"*)

**Pods authorised for the hunt.** But tonight changed what to spend
them on, and the order matters more than the budget.

### 7a. FIRST, AND IT IS $0: the monotonicity test. Nothing gets generated before it.

mac-c specified it in `7df9f25d8` and it is the cheapest decisive
measurement available: **compute `tok`-excess and `arm`-excess through
ONE identical pipeline for every screened corpus, and check whether
they move together.**

Why it outranks everything: the hypothesis is that **the window
AMPLIFIES an existing per-token signal rather than generating one.**

    sycgen     tok +0.196   arm +0.308   KEEP
    evalage    tok +0.127   arm +0.197   WEAK
    retryesc   tok +0.047   arm weak     KILL

**If that holds, `hunt-safety-gold-clew.md`'s sourcing criterion is
SELF-DEFEATING**: it demands per-token-SILENT tasks to suppress `tok`
and widen the gain — but suppressing the per-token signal would
suppress the arm too, because the arm is downstream of the same signal.
**That single mechanism would explain the entire kill record.**

**It is already half-confirmed from the other side:** our one KEEP has
the **strongest** `tok` of the three (`tok_best` 0.50–0.53 against a
~0.34 null), so **sycgen is not per-token-silent by measurement** —
which also forced a correction to the rebuttal tonight.

**Generating another corpus under a criterion that may be inverted is
the most expensive mistake available. Measure first.**

### 7b. THEN, if the test says the criterion is inverted — re-aim, and pods are for this

Screen for **tasks whose window AMPLIFIES most**, not tasks whose
`tok` is weakest. Concretely: rank candidates by **arm-excess** and by
**arm/tok ratio**, keep `floor_reach` only as a **kill filter**, and
stop treating per-token silence as a positive criterion.

### 7c. If the test refutes it, say so and the old aim survives

n=3 corpora, one model, one layer, and two of three arm numbers came
from different screen configurations. **The hypothesis is cheap to
kill and should be given the chance.**

### 7d. ⚑ STANDING RULE EARNED TWICE TONIGHT — before an instrument is used to spend money

**`floor_reach` and the ICC arm-multiplier were both built, both
validated in-sample, and both INVERTED on the first case they were not
built from.** Two for two, hours apart, by careful people.

> **Validate an instrument on a case it was NOT derived from before
> letting it direct spend.** In-sample fit is not evidence; it is the
> construction.

### 7e. Spend rules for the hunt

- Same ceilings as §1: **≤8 GPUs no approval**, pre-spend posted above
  that, **hard stop 3h after pod-up**, terminate + API-verify + ledger.
- **Prefer A40s.** The shuffle lane measured GPU idle 94% with RAM
  binding — **corpus generation and probing are likely CPU/RAM-bound
  too. Measure before buying H100s**, and say which resource bound.
- **Use `scripts/pod_inventory.py`** to read the fleet. Never `tail`.
