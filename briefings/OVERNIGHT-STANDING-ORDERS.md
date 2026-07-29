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

## 5. Open, ranked, if a lane frees up

1. **n=3 → n=5 on item 6's own cells** — outcome (d) has been unsized
   since the pre-registration and is the binding limitation on every
   sycgen claim. The marginal GPU-minute goes here, not to more k.
2. ~~The RLHF metric collision~~ **CLOSED by mac-c (`d18c556db`),
   hub-ratified.** One real anomaly, not eight: 9 of 10 all-metric
   pairs are the same model under the `_btkonly` rename (discriminator:
   same base arch **and** matching `batch_size` ⇒ benign). The survivor
   is `tsae_btkonly` bs=32 identical to `batchtopk_sae_btkonly`
   bs=1024. **Remediation is ONE row to re-run**, and both rows from
   that 90-second window (`0.6588`, and a `0.5000000000` exactly) are
   to be distrusted. Nothing quoted depends on it — **do not
   re-escalate this.**
3. **Lever 3 / `floor_reach`** — minimise it; the screen criterion is
   settled and $0.

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
   `arxiv`** — the WORKING copy (single-backslash) is the source of
   truth; the paste copy is **generated from it** by doubling every
   backslash, with the round-trip asserted.
4. **The same file on `dmitry-txcwins-10h`.** Check the branch has not
   moved under you and that Dmitry has not edited *that file* before
   applying — they were editing reviewer 2 at 01:4x.
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
