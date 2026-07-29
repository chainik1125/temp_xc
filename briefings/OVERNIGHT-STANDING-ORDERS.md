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
2. **The RLHF metric collision** (`4e0c8e1a`-adjacent, LOG 01:5x):
   `batchtopk_sae_btkonly` and `tsae_btkonly` share
   `preference_auc_k20` to 16 digits with different train/eval keys.
   Not urgent — no quoted number depends on it — but it should not
   still be open when RLHF baselines are leaned on.
3. **Lever 3 / `floor_reach`** — minimise it; the screen criterion is
   settled and $0.

**Delete this file when Han is back and the lane has closed.**
