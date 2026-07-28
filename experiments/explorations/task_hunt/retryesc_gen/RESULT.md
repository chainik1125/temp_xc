# `retryesc_gen` — SCREEN RESULT: **WEAK** (bundle, 3/3 legs)

Executes `retryesc_gen/SCREEN_CARD.md` at freeze `e667fdb99`.
Scored by `verdict.score_model`, **imported unmodified**.
Artifacts: `results/screen_retryesc_gen_{gpt2,gemma2_2b,llama31_8b}.json`,
`results/verdict_retryesc_gen.json`.

## Verdict

**Bundle = WEAK. 0 KEEP, 0 KILL, 3 WEAK.** No kill clause fired on any
leg. **Gold-visibility does NOT fire.**

| leg | tok | window best | arm | **gain** | floor@bestT | wd gain | verdict |
|---|---|---|---|---|---|---|---|
| gpt2 | 0.3669 | 0.4314 | T64/actxmean_lin | **+0.0645** | 0.5942 ✗ | +0.0540 | WEAK |
| gemma2_2b | 0.3734 | 0.4424 | T64/actxmean_mlp | **+0.0690** | 0.6083 ✗ | +0.0857 | WEAK |
| llama31_8b | 0.3910 | 0.4539 | T64/actxmean_mlp | **+0.0630** | 0.6219 ✗ | +0.0793 | WEAK |

**The gain bar was CLEARED on all three legs — and the floor clause
killed it on all three.** KEEP needs one arm clearing gain ≥ +0.05 **and**
width-null ≥ +0.02 **and** its own T's visible floor *simultaneously*.
Null passed everywhere, gain passed everywhere, `wd_ok` passed
everywhere. **`floor_ok` = False, 3/3.**

## 1. The mechanism: a scissors

| gpt2 | T4 | T8 | T16 | T32 | T64 |
|---|---|---|---|---|---|
| gain | +0.012 | −0.001 | +0.003 | +0.022 | **+0.065** |
| floor | 0.354 | 0.373 | 0.411 | 0.481 | **0.594** |
| arm − floor | **+0.025** | −0.008 | −0.041 | −0.092 | **−0.163** |

Identical shape on all three legs. **Where the window becomes useful,
the visible cue has already outrun it.** At T4–T8 the arm beats its
floor but carries no gain; at T64 it carries real gain but loses to the
floor by 0.16. They never coincide — which is exactly why the verdict is
WEAK rather than KILL: clause 3 ("all window arms ≤ their floor") does
**not** fire, because the small-T arms *do* beat their floors.

## 2. Pre-registration audit (card § 4, written before any GPU ran)

| I predicted | outcome |
|---|---|
| `floor_excess` in band ⇒ **gain should clear +0.05** | ✅ **CONFIRMED** — +0.0645 / +0.0690 / +0.0630, all three |
| ⚠ the **floor is the live risk**; llama31 the leg to watch | ✅ direction right, ❌ **severity under-called** — I predicted a *split* with llama31 failing; **all three failed** |
| `wd` should **not** erase the arm | ✅ **CONFIRMED** — +0.054 / +0.086 / +0.079, `wd_ok` 3/3 |
| the **age face will fail** the order ladder | ✅ **CONFIRMED** — `order_pass_wd` False 3/3 (age faces now **0/12** record-wide) |

**4/4 directionally correct.** The one I got wrong was *how bad* the
floor risk was, and I got it wrong in the direction of optimism.

## 3. ⚑ The density lever WORKED — and the same program's upper edge killed it

This is the finding, and it cuts both ways.

**It worked on gain.** Same face family, same helpers, same bars:

| | `evalage` (f = 0.045) | **`retryesc_gen` (f = 0.185)** |
|---|---|---|
| gain | +0.040 / +0.046 / +0.031 | **+0.065 / +0.069 / +0.063** |
| wd gain | +0.037 / +0.041 / +0.059 | **+0.054 / +0.086 / +0.079** |

**Raising in-window event mass raised the gain from below the bar to
above it, on every leg.** That is the density thesis working as a
*design* instrument, predicted in advance and confirmed.

**And the band's upper edge — which I pre-registered as two-sided and
warned about — is what took the KEEP.**

## 4. ⚑ MY AIMING INSTRUMENT WAS BIASED LOW. This is what cost the KEEP

I aimed with `claim_zone`'s `f`, on the identity `floor_excess ≡ f`.
The screen's **measured** floor tells a different story:

| leg | `f` (what I aimed with) | **measured `floor_excess` at T64** | under-read |
|---|---|---|---|
| gpt2 | 0.1853 | **0.2608** | **−0.076** |
| gemma2_2b | 0.2064 | **0.2750** | −0.069 |
| llama31_8b | 0.2230 | **0.2886** | −0.066 |

**All three sit ABOVE the +0.25 upper edge** — the region where I had
recorded that 3 of 5 cells lose to their own floor. **I thought I was
mid-band at 0.185; I was over the edge at 0.261.** The band was right.
The instrument I used to aim at it was not.

**Leading explanation, offered as a hypothesis and not as a proven
mechanism:** `claim_zone` measures `f` over the **raw eligible
population**, but the floor is fit on the **class-balanced manifest**,
which oversamples the low-age class up to 1/3. That inflates the
in-window fraction the floor actually sees. It predicts the bias should
scale with `T / e1`:

- `evalage`: T/e1 = 64/429 = **0.15** → bias negligible (K = 0.96, measured)
- `retryesc_gen`: T/e1 = 64/120 = **0.53** → bias large (−0.07, measured)

Consistent with both data points, **and it needs a direct test** —
recompute `f` on the manifest rows rather than the raw population —
before anyone relies on it. **What is NOT a hypothesis is the
correction itself:** `f` under-reads the screen's `floor_excess`
whenever the low tercile edge is close to T, and anyone aiming at the
band with `claim_zone` must account for it.

⚠ **This is the fourth error in one family in one day** (the "capped at
1/3" claim, `K = 0.63`, the uniform-position gap map, and now this).
The first three cost nothing because the bar was written on a measured
quantity. **This one cost the KEEP**, because the quantity I measured
was the right *concept* read off the wrong *population*.

## 5. What is established

- The **vocabulary rebuild succeeded**: `unigram` 0.689–0.716 (organic,
  fatal) → **0.5406–0.5434**. Difficulty-assigned generation works.
- The **density lever moves gain**, as designed and pre-registered.
- The **harness keeps the confound fixed**: `wd` gains positive and
  *larger* than `evalage`'s, where `reask_hr` (organic) reversed.
- **`retryesc_gen` is not a table candidate.** Gain without floor is not
  a result, and I am not going to present it as one.

## 6. Disposition — mine, as design owner

**`retryesc_gen` as specified is WEAK. Item 7 still has no KEEP.**

There is an obvious next move and I want to be careful about it: **aim
lower.** Target a *measured* `floor_excess` of ~0.15–0.20, which given
the ~0.07 bias means `claim_zone f` ≈ 0.08–0.13, i.e. a longer
inter-event gap. The corpus regenerates for ~$21 and the screen for
~$1.

**Arguments for:** the band was pre-registered before any of this; the
gain-vs-density curve is established over 15 independent faces; and I
now have the bias measured rather than guessed. This would be a
*calibrated re-aim*, not a fishing expedition.

**Arguments against, which I am not going to soft-pedal:** I have taken
my one pre-registered shot at the target and missed on the high side.
A second attempt is a second draw on the same corpus family, and if I
keep re-tuning until something clears, the eventual pass means much
less than it appears to. **The honest version requires its own card,
its own freeze, its own pre-registration, and an explicit statement
that it is attempt 2 of a tuned parameter.**

**That is a hub decision, not mine to take unilaterally**, and I am
recording the WEAK as a WEAK regardless of what is decided next.

## Cost

Generation ≈ **$21** (MATS key, 300 docs). Screen pod `9fcz2d1zjk174z`
(`mac-c-screen2-0728`, L40S $0.99/h) **15:21 → 16:02 = 41 min ≈ $0.68**,
**TERMINATED and API-verified (0 mac-c pods remaining)**. Actual GPU
work — 3 caches + 3 screens + verdict — ≈ 12 min ≈ $0.20; the rest was
bring-up. **Total this candidate ≈ $22.**

_Recorded-by: claude-opus-5 (mac-c, owner + executor)_
