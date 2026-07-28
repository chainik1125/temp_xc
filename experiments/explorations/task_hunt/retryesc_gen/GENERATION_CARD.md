# `retryesc_gen` — GENERATION CARD (frozen before any generation)

**Owner: `mac-c`. Briefing: `hunt-mac-c-takeover.md` (item 7 — the only
open safety-task slot). Rebuild of `retryesc`, which died label-side on
`unigram_auc` 0.689–0.716 vs a 0.60 bar (task-vocabulary leak,
explicitly not maskable). Backend: `dmitry-mats-claude-api-key`,
$300 generation cap, ledger both ends, mac-only.**

Freeze pin recorded at the bottom. Bars in § 5 and § 6 do not move
after this file is committed.

## 1. Why the organic version would have failed even if it had passed

Stated first because it is the reason this card exists in its present
shape, and because it is a correction to my own earlier write-up.

`retryesc`'s proudest receipt was **censored-age floor exactly 0.500 at
every T, claim zone 0.00%**, on an inter-failure gap of **median 886
tokens**. I recorded that as "the out-of-window construction working
perfectly."

The program-wide survey (`density_gain_survey.py`, LOG 14:07, 150 cells
/ 48 artifacts) says that receipt is the **diagnostic of weakness**:
screen gain tracks in-window event mass with face-level ρ = **+0.88**,
and a floor pinned at chance means the window sees no events at all —
the band whose mean gain is **+0.032** and in which **0 of 4** faces
cleared. A faithful rebuild reproduces `evalage` (WEAK, +0.039), not
`sycgen` (KEEP, +0.117). The vocabulary kill hid a likely WEAK.

**So the fix is not only vocabulary. It is vocabulary AND clock.**

## 2. Clock, stated first (binding bar) — with the target derived

### 2.1 What `floor_excess` actually equals

`visible_evidence_floor` is fit on exactly two features,
`(censored_age, in_window_event_count)` (`evalage/screen.py`
`_FloorBank.feats`). For a **balanced 3-class age face** with tercile
edges `e1 < e2`, write `f = P(event inside the T-window) = P(age < T)`.

While `T ≤ e1`, every in-window row is class 0, so the floor classifies
those perfectly and is reduced to guessing on the rest:

```
floor_acc = f·1 + (1−f)·[(1/3)/(1−f)] = f + 1/3
=> floor_excess = floor_acc − 1/3 = f
```

**`floor_excess` IS the in-window event fraction.** Verified
numerically in `verify_floor_identity.py`: exact to **0.0000** at every
`g` from 64 to 2000 tokens.

⚠ **A wrong version of this section was written and caught by that
simulation before freeze.** I first claimed the identity holds only
while `T ≤ e1` and that a pure age face is therefore "capped near
`floor_excess = 1/3`". **Both are false.** The floor knows the *exact*
age whenever the event is in-window, so it classifies those rows
correctly *whatever class they fall in*; the identity survives until
`T` passes the **upper** edge `e2` (i.e. up to `f = 2/3`), not the
lower one. Recorded rather than quietly fixed, because the false
version would have justified a much denser corpus as "safely capped".

**The real upper-edge mechanism** — and it is a stronger reason to
respect the band, not a weaker one:

- The floor is computed from **ground truth** (censored age, exact when
  in-window), so it climbs toward 1.0 as `f` climbs. The window arm
  reads **activations** and is bounded by what the model actually
  encodes.
- So high density does not blunt the floor — **it hands the floor a
  bar the arm cannot reach.** Above `floor_excess ≈ +0.25`, 3 of 5
  cells in the record lose to their own floor.
- `qd` (+0.351, margin **−0.034**) and `cnov` (+0.265, margin +0.005)
  are the cases. Both are count/novelty faces, where the
  `in_window_count` feature lets the floor read something close to the
  label directly — the worst case of the same effect.

Every age face in the record happens to sit below 1/3 (`sycgen_age`
+0.210, `sage` +0.158, `evalage_age` +0.045, `reask_hr` +0.034), but
that is a fact about those corpora's gaps, **not** a structural
guarantee — which is exactly why § 5 band 4 is a two-sided bar that the
pilot must measure.

⚠ I did not re-derive every face's definition from source to classify
it age-vs-count; the age faces above I own or have read. The
count-channel claim is **consistent with** the record, not proven over
all 15.

### 2.2 The target, and the gap that produces it

Pre-registered target: **`floor_excess` ∈ [+0.15, +0.25]**, i.e.
**15–25 % of eligible probe tokens sit within T = 64 of an escalation
event** — 45–75 % of one tercile. Two-sided; **not** maximized.

⚠ **CORRECTED after freeze — see § 2.2a. The gap numbers below are the
corrected ones; the originals were too dense by ~2×.**

| target `f` | gap median `g` |
|---|---|
| 0.25 (upper) | 297 tok |
| **0.20 (centre)** | **385 tok** |
| 0.15 (lower) | 499 tok |

### 2.2a ⚑ CORRIGENDUM (same day, before any spend) — `K = 0.96`, not 0.63

**The identity is now validated on REAL DATA, and my calibration factor
was an artefact of the wrong model.**

`elicit_lib.claim_zone(...)["frac_in_window"]["T64"]` **is** `f`,
measured directly on a built stream — the harness already had the
instrument I was simulating. Against `evalage`'s *measured*
`floor_excess`:

| leg | `claim_zone` f | measured `floor_excess` | diff |
|---|---|---|---|
| gpt2 | 0.0448 | **+0.0480** | +0.0032 |
| gemma2_2b | 0.0480 | **+0.0639** | +0.0159 |
| llama31_8b | 0.0482 | **+0.0233** | −0.0248 |
| **mean** | **0.0470** | **+0.0451** | **−0.0019** |

**`K = 0.959`.** Per-leg scatter ±0.025 is probe noise; the mean is
within 0.002. So `floor_excess ≡ f` holds on real data with **no
eligibility correction at all**.

**Where my 0.63 came from, stated plainly:** I compared the measured
`floor_excess` against an `f` **simulated from exponential gaps**, and
`evalage`'s gaps are **log-uniform**, not exponential. The 1.6×
discrepancy was my gap model being wrong, and I misread it as a real
eligibility effect and gave it a mechanism. **The simulation was an
unnecessary intermediate — the real instrument supersedes it.**

**This mattered, which is why it is a corrigendum and not a footnote.**
Re-solving from `evalage`'s *actual* age CDF (anchors: P(age≤16) =
0.0027, ≤32 = 0.0169, ≤64 = 0.0448, median 683):

| card's original route | implied `f` | |
|---|---|---|
| "calibrated" centre `g = 170` | **0.357** | ⚠ far past the +0.25 edge |
| "naive" centre `g = 286` | **0.257** | ⚠ past the +0.25 edge |
| **corrected centre `g = 385`** | **0.196** | ✅ in band |

**Both original routes were too dense**, and the one I labelled
"calibrated" was the worse of the two. Had I started the pilot there it
would have landed in the region where 3 of 5 record cells lose to their
own floor — the exact failure the band exists to prevent.

**This correction is *within* the card's own rules, not a breach of
them:** § 2.2 pre-registers the *target* (`floor_excess`) and names `g`
as the one knob permitted to move, precisely because the mapping from
`g` to `f` was the uncertain part. The target has not moved.

**Therefore the bar is written on the MEASURED quantity, not on `g`.**
`g` is the knob; `floor_excess` is the target; the § 6 pilot is the
arbiter — and it now measures `f` at **$0 on the label side**, with no
GPU, via `claim_zone`. **Planning centre: `g ≈ 385` tokens (median),
range 297–499** — i.e. **~2.2× denser than organic `retryesc`'s 886**,
not the 3–5× the frozen draft said.

### 2.3 Corpus clock bar — and the realism cost the target imposes

Documents must clear the standing corpus-clock bar by a wide margin
(`dharm` died at 155.6 tok/doc). Plan: **~30–45 turn-pairs/doc**,
assistant turns **~60–120 tok**, environment turns ~30–50 tok ⇒
**~3,500–6,000 tok/doc**, ~25–35× the bar, with many position strata.
Reported as measured, not assumed.

**⚑ The tension, stated rather than left for a reader to find** (and
**eased** by the § 2.2a correction, which is the honest report — the
corrected clock is *less* demanding, not more). At the corrected
`g ≈ 385` tok with ~120-tok turn-pairs, a repeat-failure event lands
roughly **every three turn-pairs** — ~1 in 3 environment turns reports
a repeat failure, rather than the ~1 in 2 the pre-correction `g ≈ 230`
implied. Still a **more failure-prone agent than a typical real
trace**, and still a direct consequence of the density target rather
than an accident.

Three things follow, all binding:

1. **Turns are kept short deliberately** (60–120 tok, vs organic
   `retryesc`'s 686 tok/turn) so the required event *rate per turn*
   stays plausible instead of absurd. Most of the density is bought
   with the token clock, not by making the agent fail more often.
2. **The realism cost is disclosed on any exhibit this corpus reaches**
   — the same treatment `retryesc`'s single-agent/single-model
   substrate got. This is a claim about a constructed distribution.
3. **It does not touch validity.** The § 3 construction rule (failure
   text independent of repeat-status) is what keeps the label
   out-of-window, and it holds at any event rate. Density threatens
   *the floor bar*, which § 5 band 4 caps, not the confound structure.

If honouring the band forces an implausible generator, the honest
outcome is a **no-go reported as such** — not a quietly widened band.

## 3. The face — § 1.2-shaped, two-timescale

**Event = a repeat-failure escalation:** an environment turn reporting
the failure of a strategy that **has already failed earlier in the
episode**.

- **Slow timescale (out-of-window; sets the indicator).** Whether an
  attempt is a *repeat* depends on what was tried hundreds of tokens
  ago — outside any T ≤ 64 window. This is the § 1.2 principle and
  `tret`'s shape.
- **Fast timescale (in-window; carries the kernel).** The events are
  spaced per § 2.2 so several land inside T ≤ 64.

**⚑ The construction rule that makes density safe (binding).** Density
would normally raise the visible-cue trap. It does not here **only if
the environment's failure text is drawn from a fixed pool that is
independent of repeat-status** — a repeat failure and a first-time
failure are *textually indistinguishable*. The label then depends on
information that is out-of-window by construction, so an in-window
event token does not hand over the label. **If the generator ever makes
repeat-failures textually distinctive, this candidate is dead and the
floor will say so.**

Environment turns are scaffold-authored and templated (never
model-authored) and are **fully masked from probe eligibility**; probes
read assistant tokens only.

### 3.1 Two faces, both pre-registered now

Scored on the same corpus and the same cache, declared before any data:

| | face | role |
|---|---|---|
| **primary** | `retryesc_age` = `log2(1 + tokens since last repeat-failure)` | the KEEP claim rests here |
| **secondary** | `retryesc_rate` = repeat-failures in window | declared, reported regardless |

The primary is an age face **on purpose**: `sycgen_age` is the
program's gold, order-null, so the age family clears the hunt4 § 4 bar.
The secondary exists because the count channel is the only one that
routes to the Q3 order table — a bonus, **not** a condition. **Both are
reported whatever they show**; the secondary may not be promoted to the
headline if the primary disappoints.

## 4. Vocabulary — the killer bar, fixed structurally

`retryesc` died because **task difficulty genuinely drove failure
rate**, so ordinary task nouns predicted age (`unigram` 0.69–0.72,
`doc_mean` 0.865–0.879). Masking could not fix it.

**Fix, by construction, not by hope:** the **failure schedule is drawn
FIRST from the plan, independently of the task**, and the task is then
drawn uniformly. Difficulty is *assigned*, not intrinsic — so the same
task appears with easy and hard schedules across documents, and task
vocabulary carries no information about failure rate. `unigram ⊥ label`
becomes a property of the generator rather than a band to squeak past.

This is the gate I expect to be hardest and it is still gated in § 5.

## 5. Pre-registered bands (absolute only — the `msdose_r1` lesson)

Every band, on **all three tokenizers**, or **label-side KILL and no
generation of the full corpus**:

1. `unigram_auc` ≤ **0.60** (the bar `retryesc` failed)
2. `doc_mean_only_auc` ≤ **0.88**
3. `position_auc` ≤ **0.95**
4. **`floor_excess` ∈ [+0.15, +0.25]** — § 2.2, two-sided
5. qualifying strata ≥ **8**; usable position-matched tokens ≥ **250k**
6. events ≥ **300**
7. two-leg `vocabulary_control_check` at PLAN time; `VOCAB_CV_BAR`
   0.35 is **PROPOSED, NOT RATIFIED** — reported as a value, and a
   `stop` routes to the design owner rather than auto-killing

Floors are reported per T as evidence; formal authority is hunt4 § 4.

## 6. Sequencing — pilot first, and it can kill the card

Because `floor_excess` cannot be measured before text exists, and
because a full corpus is the expensive object:

1. **$0 — plan-time arithmetic** (§ 2) and the two-leg vocabulary
   control on the *plan*. Already done for this freeze.
2. **PILOT — ~20 documents**, small fraction of the cap. Measure `f`
   **directly** via `elicit_lib.claim_zone(...)["frac_in_window"]["T64"]`
   — **$0, label-side, no GPU** (§ 2.2a: this equals `floor_excess`
   with `K = 0.96` on real data). Plus Tier T/R (struqpos methodology
   note `5f7c60590`: token-multiset delta, length delta, pooled
   bag-of-embeddings ≤ 0.55; adjacency floor below the KEEP bar).
   Seconds of CPU.
3. **Tune `g` on the pilot** — the one quantity permitted to move after
   freeze, because § 2.2 pre-registers the *target* and names `g` as
   the knob. Tuning is logged with before/after values.
4. **Full generation** only if the pilot lands `floor_excess` in band
   **and** clears band 1. Checkpointing is BLOCKING and is wired.
5. Screen: hunt4 § 4 verbatim. Venue = a fresh L40S under the standing
   waiver (~$1/h), or mac-d's pod-D by LOG claim if warm.

**A pilot outside the band is a NO-GO to report, not a thing to
proceed through hopefully.** If `g` cannot be tuned into band without
breaking § 3's construction rule or § 2.3's clock bar, this candidate
is killed label-side, at pilot cost, and I will say so.

## 7. Odds, on record before the result

Conditioned rather than guessed (LOG 14:07): magnitude **~70–75 %** if
the density target lands (in-band history 13/16 cells, 4/4 faces, small
n, band drawn post hoc); the **leak gate ~65–75 % and it is the
dominant risk**; **joint ~50–55 %** to a KEEP.

**Fallback if the leak gate fails** (declared now, not invented later):
the same two-timescale face on a **synthetic task vocabulary** with
difficulty assigned independently of surface tokens by construction —
strictly more controlled, at a disclosed cost in realism.

## 8. FREEZE RECEIPT

| | |
|---|---|
| freeze pin | **`3f6ba0d3d`** (this card's own commit) |
| tree at freeze | **clean** — `git status --porcelain` empty, asserted before push |
| bars frozen | § 5 (7 bands) and § 6 (pilot ladder) |
| permitted to move after freeze | **`g` only** — § 2.2 pre-registers the *target* (`floor_excess`) and names `g` as the knob; every tuning step is logged with before/after |
| derivation receipts | `verify_floor_identity.py` (identity, worst err 2e-6), `../density_gain_survey.py` (ρ_face +0.88) |
| odds on record before any result | § 7 — joint ~50–55 % to a KEEP |

Nothing below § 5 may be renegotiated by me as executor. A pilot
outside the band is reported as a **no-go**, not widened into one.

_Recorded-by: claude-opus-5 (mac-c, owner)_
