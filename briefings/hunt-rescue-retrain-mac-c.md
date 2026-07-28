---
status: active
owner: mac-c
issued-by: mac-local (hub)
issued: 2026-07-28 18:4x London
priority: TAKES PRECEDENCE over hunt-safety-gold-clew.md (pods first;
  literature sourcing is $0 and interleaves while cells run)
---

# Rescue-retrain lane — the screen may have been over-rejecting, and it is measurable

Han, 2026-07-28: *"mac-c needs to actually do some pod work — we might
have been prematurely discarding high potential tasks."*

**He is right, and the hub has now quantified it. Read § 1 before you
plan anything: it changes what the previous verdicts mean.**

---

## ⚑ AMENDMENT 19:0x — mac-c's overnight work lands BETWEEN this brief being written and pushed, and it reorders it

This brief was drafted before I read `b8cfc692d` / `e1dfe21a1` /
`b6d3c7212`. **mac-c has independently found a stronger and more general
answer to Han's question than the per-candidate rescue below, and it
changes what the pod work should be.**

**Their finding, which I am adopting over my Lane A ordering:** the
bottleneck is **apparatus geometry, not task choice.** `chunk_stream`
builds independent 128-token sequences, so with `OFF_MIN=63` every
probed token saw only 64–128 tokens of context — while the recency
terciles sit at ages **121 / 286**. Classes 1 and 2 were both *"no
event in my context"*. **The arm was never weak; it was asked about
tokens the model never saw.** Re-terciling inside the visible context
nearly triples gain (+0.0596 → +0.1707).

And context × scale **multiply**: `gpt2 @128 +0.0596`, `gpt2 @512
+0.0928`, `gemma2_2b @128 +0.0567` (21× model buys *nothing* at 128),
**`gemma2_2b @512 +0.1088` — best on record.** Floor unmoved across all
four combos, exactly as pre-registered, because it depends only on T+w
and is **ours to set**.

**⇒ "Every candidate screened so far violated levers 1 and 3"** (context,
and the floor's T+w). That is a *systematic under-measurement of the
whole hunt*, which is a bigger and cheaper finding than rescuing one
candidate — and it means **retraining `evalage` at the OLD geometry
would re-measure the ceiling, not the task.** mac-c's own process
lesson applies to my brief: *when a factor is pinned by another
factor's ceiling, varying it alone measures the ceiling.*

**Reordered lanes — this supersedes § 2 below:**

- **A′ (new priority, and this is the pod work Han asked for): the
  three-lever matrix at scale.** Lever 3 (shrink T+w: `w` 25 → narrow,
  `T` 64 → 16, label range inside the readable band) is **untested and
  now binding**, and levers 1–2 want a real GPU: `gemma2_2b` and
  `llama31_8b` × `SEQ_LEN` {512, 1024} × corrected terciles. Everything
  so far ran on local MPS at $0 — this is exactly the measurement a pod
  is for. Start with mac-c's own cheap step (**re-label an existing
  corpus at T=16 and measure arm−floor before generating anything**),
  then take the surviving cell to a pod.
- **B (unchanged, still valid and independent): give the screen an
  error bar.** § 1's finding stands on its own — there are no CIs
  anywhere, and `evalage` was rejected at ~0.5 σ. Geometry and
  uncertainty are separate defects.
- **A (demoted): retrain `evalage` for real** — only *after* the
  geometry is corrected, and then at the corrected geometry. Retraining
  it at `SEQ_LEN=128` would answer the wrong question.
- **C (dropped): `retryesc_gen` as a predicted negative.** mac-c reports
  it *"was wrong on all three levers"*, so at the old geometry it is a
  worthless calibration point. Reinstate only if a corrected-geometry
  re-screen puts it back in contention.

**Also correcting my own § 2 below:** it repeats the
`floor_excess ≡ P(event in T-window)` identity. **mac-c refuted that
(`d2320d274`)** — it is a low-density approximation, the floor's real
window is **T + w**, and the correct quantity is
**`P(any masked TURN token in window)`**, targeted to [0.15, 0.25].
The `claim_zone ≈ 0.13–0.15` aiming constant is **withdrawn**.

*Everything below is kept as written, with § 2's lane order superseded
by the above and its floor identity corrected here.*

---

## 1. The finding: the screen compares single-seed point estimates to a hard threshold

I opened the saved screen artifacts. **Every cell is one number.**

    /cells/<face>/<T>/<arm>/acc_test   n_train   n_test   wall_s

There is **no `seed` field, no CI, no bootstrap, no `std`, nowhere in
any screen JSON.** And per-example predictions are not saved — so the
paired standard error **cannot be recovered retroactively** from what
we kept. That is itself a defect worth fixing before the next screen.

With `n_test = 4497`, the SE on one accuracy near 0.43 is ≈ 0.0074, so
the SE on the *gain* (a paired difference) is ≈ 0.007–0.010. Against a
hard `gain ≥ +0.05`:

| candidate | leg | gain | vs bar | shortfall / SE | verdict was |
|---|---|---|---|---|---|
| `evalage` | gemma2_2b | +0.0460 | **−0.0040** | **0.54 σ** | WEAK |
| `evalage` | gpt2 | +0.0400 | −0.0100 | 1.36 σ | WEAK |
| `evalage` | llama31_8b | +0.0307 | −0.0193 | 2.63 σ | WEAK |
| `retryesc_gen` | gemma2_2b | +0.0690 | **+0.0190** | 2.60 σ *over* | WEAK (floor) |
| `retryesc_gen` | gpt2 | +0.0645 | +0.0145 | 1.99 σ over | WEAK (floor) |
| `retryesc_gen` | llama31_8b | +0.0629 | +0.0129 | 1.75 σ over | WEAK (floor) |
| `sycgen` | gpt2 | +0.1152 | +0.0652 | **8.87 σ over** | KEEP → exhibit |
| `sycgen` | gemma2_2b | +0.1118 | +0.0618 | 8.47 σ over | KEEP |
| `sycgen` | llama31_8b | +0.1225 | +0.0725 | 9.97 σ over | KEEP |

*(σ computed at an illustrative pairing ρ=0.5; the true paired SE is
unrecoverable, which is the point. Treat these as order-of-magnitude.)*

**Three things follow, and they are the whole brief:**

1. **`evalage`'s best leg missed the bar by about half a standard
   error.** We rejected a candidate on a difference smaller than the
   measurement's own noise, and never computed the noise.
2. **The bar has only ever been exercised where the answer was
   obvious.** Our one KEEP cleared it by ~9 σ. **No candidate between
   +0.03 and +0.11 has ever been retrained** — that entire band is
   untested, and both near-misses live in it. The `+0.05` constant is
   hardcoded in `hunt3/verdict.py:50` and `hunt4/verdict.py:63`; it was
   set by fiat, not calibrated.
3. **The screen is a proxy for a different instrument than the one we
   ship.** The screen's "window" arm is a **mean-pooled probe over raw
   activations** (`actxmean_mlp`); the deliverable is a **trained sparse
   TXC dictionary**. We already have direct evidence these diverge: in
   `sycgen`, the **untrained twin showed LARGER shuffle gaps than the
   trained model** — training *reduced* the order gap while lifting
   recovery from ≤0.22 to 0.50–0.59. A pooled-probe screen is not
   entitled to predict what the trained dictionary does.

---

## 2. What to run — three lanes, in this order

### Lane A (PRIORITY): retrain `evalage` for real

**The strongest premature-discard case in the record.** On every leg it
**cleared the width null, cleared its visible floor, and cleared the
within-conversation control** — floors 0.336–0.397 with every window arm
above them. **No kill clause fired.** It failed exactly one hardcoded
threshold, by ~0.5 σ on its best leg. And on llama31 the **`wd` gain is
+0.0588, which clears +0.05** — the KEEP rule reads the global gain, not
the `wd` gain, and that is a rule choice rather than a fact about the
task.

Run the actual TXC T-sweep retrain — the `sycgen` shape: **T{1,2,4,8,16}
× seeds {42,1,2}, trained + untrained twin, shuffle overlay, btk-only
arm** (either-arm rule holds for hunted tasks). Reuse
`sycgen/run_retrain.py` + `shuffle_overlay.py`; substrate
`elicit_evalage_v1` on llama31-8b L14 to match.

**Pre-register before you launch** — write it in the card, then run:
what recovery-vs-T would count as a rescue, what would confirm the
screen was right, and what the untrained twin must do for the result to
mean anything. **The twin control is binding**: `sycgen` taught us that
a positive ordered−shuffled gap can be pure architectural
position-sensitivity.

### Lane B (cheap, do it alongside): give the screen an error bar

The screen instrument is used by every future candidate, so fixing it
pays forward. Minimum viable: **3 seeds per cell and a bootstrap CI on
the gain**, plus **save per-example predictions** so paired SEs are
recoverable later. Then re-score `evalage` and `retryesc_gen` and report
whether the verdicts survive their own uncertainty.

**Do not silently change the +0.05 bar.** If the CI work argues the bar
should move, say so with the evidence and let the hub/Han rule — a
threshold that moves after seeing the data is not a threshold.

### Lane C (calibration, if budget allows): retrain `retryesc_gen` as a PREDICTED NEGATIVE

**Include this specifically because we expect it to stay dead.** If we
only re-run candidates we hope to rescue, this is fishing; running one we
predict the screen got *right* is what makes it a calibration.

And I expect it stays dead for a structural reason, not a marginal one:
its floor failure is not close. At its best-gain T the **visible-evidence
floor is 0.594–0.622 while the window arm reaches only 0.431–0.454** —
the floor beats the window by ~0.16. That is the documented scissors:
*where the window becomes useful, the visible cue has already outrun
it.* Predicting NEGATIVE and then measuring it is worth real money;
narrating it into a rescue is not.

---

## 3. Guard rails — this lane is dangerous to the program's credibility

**This is a re-examination of our own gate, not a search for a KEEP we
already decided we want.** The prime directive still binds: *a sound
verdict, never a win.*

- **Pre-register every lane before launching it.** The pre-registration
  is what separates "we calibrated our instrument" from "we kept
  re-rolling until something passed".
- **Report all three lanes whichever way they fall.** If `evalage`
  retrains into a null, that is a *finding about our screen* and it goes
  in the record with the same prominence a rescue would get.
- **A rescued candidate must be disclosed as rescued** — screen verdict
  WEAK, retrained anyway because the miss was inside the noise, retrain
  outcome X. Never quote it as though it had passed the screen.
- **Attempt caps** are unchanged for *new* candidates; these are
  re-measurements of existing ones, not new attempts.

## 4. Budget + pods

Spend authorization stands (Han 14:51). **Fleet right now: 4 pods
running, $6.86/h** — only `mac-d-rlhfpf-0728-5` is an agent lane (btk
T10 gap cells, draining); the other three are unattributed
non-convention pods awaiting Han's call, **do not touch them**.

Spin your own: `mac-c-rescue-0728`, terminate at lane end and
**API-verify**, ledger both ends. The `sycgen` retrain was 36 cells on
2×H100 — budget Lane A in the same order of magnitude, Lane C likewise.
Never seed the runpod key onto a pod.

## 5. Acceptance gate

A `RESCUE_CARD.md` with the pre-registration written *before* any GPU
ran, plus a `RESULT.md` per lane reporting gain/floor/order/twin as they
fell, **and an explicit statement of what this says about the screen**:
over-rejecting, well-calibrated, or measuring the wrong instrument.

**That last sentence is the actual deliverable.** A rescued task would
be a bonus; a calibrated gate is the thing the program needs.

**Delete this file when the lane closes.**
