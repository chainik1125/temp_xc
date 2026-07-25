---
author: Claude review agent
date: 2026-07-24
tags:
  - results
  - in-progress
---

## Adversarial audit — trajectory-steering results and the W-sweep

Scope: [[semisynthetic_language_tasks]] (claims + tables), `sol_modal.py`, `full_modal.py`,
`gen2_modal.py`, `wsweep_modal.py`, and the raw JSONs in `results/temporal_screen/`.
All numbers below were recomputed from the raw JSONs; the RNG streams of `full_modal.py`
and `gen2_modal.py` were replayed exactly on CPU to recover per-episode quantities that
the JSONs do not store.

### Verdict in one paragraph

Every number in the doc's tables is faithful to the JSONs — I checked 24 of them and
found no transcription errors. The problem is not the numbers, it is what they measure.
**The teacher-forced margin is a sum of per-token log-probs over all k segments, so it is
extensive in k by construction.** Once normalised by the number of target-vs-foil
*differing* slots, the headline "template grows ~linearly in k (+75.7 → +218.9)" becomes
**+37.8 → +42.2 per differing slot, i.e. flat**, and `alt_phase` becomes **10.5 → 8.05,
i.e. decaying**. The W-sweep independently confirms this: Δmargin there is a function of
*coverage alone* (22.2 ± 2 per covered slot across all thirteen (W, m) conditions), and
its own matched-coverage contrasts show no window effect (≤ 1.4 unpaired σ, sign-flipping).
A second, structural issue compounds it: in every arm the "temporal template" is one
direction times a ±1 schedule (`cos(t_dir[i], u_dc)` = ±0.92…0.97, `dc_ratio` = 0.167), so
nothing in these experiments distinguishes a temporal dictionary from a per-token SAE
driven by an external clock. The generation demo is the most defensible result in the set
and survives its own worst objection (classifier bias cannot produce it), but it needs an
unsteered control and paired arms.

---

### O1 — The k-growth is metric bookkeeping, not a steering property

**OBJECTION.** "Your margin sums log-probs over all k segments and your foil differs from
the target in ~k/2 slots, so a constant per-slot steering effect mechanically produces a
linear-in-k curve; you have measured the length of your own metric."

**SEVERITY: kills-the-claim** — specifically kills "template grows ~linearly in k
(the windowed handle doesn't degrade with trajectory length)" and "steerability grows with
required context", which is the headline of the whole section.

**EVIDENCE STATUS: already refuted by existing data.** Replaying `random.Random(100+k)`
through `lang_k`'s draw sequence recovers the exact eval foils and their Hamming distances
(2.00, 2.31, 3.38, 4.06, 5.19 for k = 2…10). Dividing the doc's own peak Δmargin by them:

| k | E[Hamming] | template peak | per differing slot | alt_phase per slot (H = k) |
| --- | --- | --- | --- | --- |
| 2 | 2.00 | +75.7 | 37.84 | 10.50 |
| 4 | 2.31 | +90.1 | 38.95 | 9.63 |
| 6 | 3.38 | +141.6 | 41.97 | 8.90 |
| 8 | 4.06 | +169.8 | 41.79 | 8.20 |
| 10 | 5.19 | +218.9 | 42.19 | 8.05 |

`lang_profile` rises 12% over a range where the headline rises 189%; `alt_phase`
*falls* 23% where the headline rises 283%. The same holds at every dose in the grid
(`alt_phase` per-slot at frac 0.2: 3.70, 3.75, 3.63, 3.35, 3.33).

The single arm is explained by the same arithmetic. It writes at slot 0 only, so it should
deliver `P(slot 0 is a differing slot) × (per-slot effect)`. Replaying that probability
(1.00, 0.56, 0.69, 0.47, 0.53) predicts 37.8, 21.9, 28.9, 19.6, 22.4 against observed
29.6, 19.3, 26.1, 20.3, 16.9 — ratios 0.78–1.04, and it reproduces the **non-monotonic
wobble** (the k = 6 bump is just an over-sampling of profiles whose slot 0 happens to
differ). There is no 1/k decay law here; there is one slot out of k.

**CHEAPEST KILLING CONTROL.** Zero GPU: renormalise the existing table and report
Δmargin per differing slot. To make it airtight, add a **fixed-Hamming foil**: build the
foil by swapping exactly one 1/0 pair, so H = 2 at every k. Prediction under bookkeeping:
Δmargin flat in k. One task, template arm only, k ∈ {2,4,6,8,10}, 3 fracs, n = 32 ≈ 960
forwards ≈ **8–12 GPU-min**. If it comes out flat, the honest headline becomes "per-slot
steering efficacy is constant in trajectory length" — still publishable, far weaker.

---

### O2 — The "temporal template" is rank-1: one direction and an external sign schedule

**OBJECTION.** "Your template arm writes `s[i]·u` for a single difference-of-means vector
`u` and a schedule `s` you supply from ground truth. That is a per-token SAE feature with
a hand-written coefficient schedule. What does the temporal dictionary contribute?"

**SEVERITY: kills-the-claim** for any framing of the form "temporal crosscoders beat
per-token SAEs for steering". Only "weakens" if the claim is narrowed to "controlling a
trajectory requires a time-varying write, and a constant write cannot do it".

**EVIDENCE STATUS: already refuted by existing data** — and the doc half-concedes it.
In `lang_profile` the template is literally `[si * u for si in s]` (`full_modal.py:231-237`)
with the *same* `u` the broadcast arm uses; only the schedule differs. For the B-tasks the
per-position directions are fitted separately but come out collinear:
`trajectory_sol.json` gives `cos(t_dir[i], u_dc)` = [0.92, −0.93, 0.96, −0.94, 0.97, −0.95]
for `alt_phase` and [−0.94, −0.27, 0.97, 0.30, −0.96] for `mirror`, with `dc_ratio` 0.167
and 0.336. There is no direction diversity anywhere in the result set. The comparison
being run is *schedule vs no schedule*, and the load-bearing comparison for a
crosscoder paper — TXC decoder rows vs an SAE direction **carrying the same schedule** —
appears in none of the three harnesses.

**CHEAPEST KILLING CONTROL.** Tonight, cheap and diagnostic: take the k×d matrix of
per-position DoM diffs and report its singular-value spectrum. If σ₁ carries >90% of the
mass, state plainly that the handle is rank-1. Reuses the existing capture phase,
**≈ 3–5 GPU-min**. The control that would actually *support* a dictionary claim needs
trained dictionaries (a TXC and a d_sae/L0-matched TopK SAE on the same activation cache,
then template-from-decoder-rows vs SAE-direction-plus-schedule) — hours, not minutes;
this is the "needs future work" item that most determines whether the section can be
framed around crosscoders at all.

---

### O3 — The W-sweep measures coverage, not window size, and cannot do otherwise as coded

**OBJECTION.** "Your window-W handle with m knobs writes m *consecutive* blocks, so
'm knobs of width W' and 'one knob of width mW' cover the same contiguous span. Your
(W, m) grid is a reparameterisation of coverage; the curve you call 'performance improves
with window size' is 'performance improves with number of segments written'."

**SEVERITY: kills-the-claim** for the sprint's named centerpiece.

**EVIDENCE STATUS: already refuted by existing data** (`results/temporal_screen/wsweep.json`,
run at 22:05). Δmargin is a function of coverage alone:

| condition | coverage | lang Δmargin | per covered slot | alt Δmargin | per covered slot |
| --- | --- | --- | --- | --- | --- |
| W1_m1 | 1 | +15.45 ± 4.25 | 15.45 | +5.70 ± 0.83 | 5.70 |
| W1_m2 | 2 | +35.21 ± 5.54 | 17.60 | +14.14 ± 1.27 | 7.07 |
| W2_m1 | 2 | +46.05 ± 6.46 | 23.02 | +16.60 ± 1.13 | 8.30 |
| W2_m2 | 4 | +88.82 ± 9.83 | 22.20 | +31.30 ± 1.78 | 7.82 |
| W4_m1 | 4 | +74.53 ± 8.77 | 18.63 | +33.65 ± 1.66 | 8.41 |
| W3_m2 | 6 | +136.96 ± 11.81 | 22.83 | +46.90 ± 2.55 | 7.82 |
| W6_m1 | 6 | +140.43 ± 12.98 | 23.41 | +46.58 ± 2.29 | 7.76 |
| W6_m2 / W12_m1 / full | 12 | +266.23 ± 16.67 | 22.19 | +93.35 ± 2.75 | 7.78 |

The matched-coverage contrasts — the only cells that isolate W from coverage — are null
and sign-flipping: at coverage 2 the wider window is +10.8 (lang) and +2.5 (alt); at
coverage 4 it is **−14.3** (lang) and +2.4 (alt); at coverage 6 it is +3.5 and −0.3. All
within 1.4 unpaired σ. And the design cannot produce a real contrast anyway, because
`covered()` (`wsweep_modal.py:162-168`) takes `m` *consecutive* blocks, so W1_m2 and W2_m1
both write a contiguous 2-segment run — they differ only in block *alignment*
(W2_m1 can only start on even segments, W1_m2 on any segment), which is a positional
confound, not a window manipulation.

Two further design notes. First, the docstring claims "per-pair deltas kept for SEM" but
only the mean and SEM are stored, so the matched-coverage contrasts can only be tested
*unpaired* — throwing away most of the power on the one comparison that matters. Second,
for `lang_profile` the profile is redrawn per eval pair, so "one knob writing a block with
the correct per-segment schedule inside its span" smuggles W bits of per-episode side
information; the knob count is m only for `alt_phase`, whose profile is fixed.

**CHEAPEST KILLING CONTROL.** Re-run the W-sweep with (a) **scattered** block placement —
m blocks at maximally separated positions rather than consecutive ones, so W1_m2 is two
isolated segments and W2_m1 is one contiguous pair; (b) start offsets rotated over all k
positions rather than over `k//W`, killing the alignment confound; (c) per-pair deltas
saved so the matched-coverage contrast can be paired. Same grid, same n: **≈ 15–20
GPU-min**. This is the highest-value single run available tonight. Registered prediction
from the additivity model: still null at matched coverage, ±5%.

---

### O4 — The doses are off-distribution, and the coherence budget the theory depends on is never enforced

**OBJECTION.** "Every reported peak is at frac 0.5 — a vector of norm ≈ 28 added to a
residual stream of norm ≈ 56, at every token. Your own generation samples at that dose are
degenerate code-mixed text. You cite the Δgc 'genuine event' protocol as the model for
this experiment and then report a metric with no coherence filter at all."

**SEVERITY: weakens**, escalating to kills-the-claim for the specific sentence "Mirror the
backtracking Δgc protocol". It also undercuts O1's theoretical defence: the doc's `1/k`
prediction is derived from a per-position coherence cap `m`, and the teacher-forced metric
imposes no such cap — nothing stops the single arm from using `k·m`.

**EVIDENCE STATUS: already refuted by existing data** for the off-distribution part
(`trajectory_gen2.json` samples at frac 0.35–0.5 include `"15h, 20h, 23h. On starts at 9,
3 p.m., 6 h, 9 h, 12 h..."` and translation-exercise meta-commentary). Testable tonight for
the dose-response part.

**CHEAPEST KILLING CONTROL.** Report the k-sweep at a coherence-capped dose: pick the
largest frac at which steered free generation stays fluent (a KL threshold on non-target
tokens, or the langid harness's degeneracy rate), then re-read the existing curves at that
frac — the grids already contain 0.1 and 0.2, so this is **free**, and the conclusion is
already visible (at frac 0.2, `alt_phase` per-slot is 3.70 → 3.33, still decaying). A
proper fluency-gated dose selection is **≈ 15–20 GPU-min**.

---

### O5 — "Peak Δmargin" is a grid-edge artifact, and the plateau claim is false for alt_phase

**OBJECTION.** "Every template and single peak in the paper sits at the largest frac you
tried; you are reporting where you stopped looking."

**SEVERITY: cosmetic → weakens** (it does not change any sign, but it makes every absolute
number arbitrary and the doc contains one false statement about it).

**EVIDENCE STATUS: already refuted by existing data.** All 20 template/single peaks in
`trajectory_full.json` and all 13 W-sweep peaks are at the grid edge. The doc's
parenthetical "(template plateaus by 0.35–0.5)" holds only for `lang_profile` (+5% to +13%
from 0.35 to 0.5); for `alt_phase` the increment is **+36% to +48% at every k**. Separately,
the per-arm max-over-frac is applied inconsistently: `trajectory_sol.json` stores a
`headline` field (max over frac) of **−0.24** for `mirror` broadcast and **−0.42** for
`alt_phase` broadcast, but the doc's signs-of-life table reports **−3.3** and **−9.3** —
those are the frac-0.2 values. The full-run table then switches back to the max
convention, where broadcast's "peak" for `alt_phase` is selected at frac 0.02, i.e. at a
25× smaller dose than the template it is compared against. Both conventions are
defensible; using different ones in adjacent tables is not.

**CHEAPEST KILLING CONTROL.** Extend the frac grid to 0.75 and 1.0 for one task at
k ∈ {2, 10} (**≈ 5–8 GPU-min**), and report every arm at a *matched* frac alongside its own
optimum. Matched-dose numbers are already computable for free: at frac 0.5, `alt_phase`
broadcast is −5.35, −7.49, −2.09 for k = 2, 6, 10 (vs the +21.0, +53.4, +80.5 template),
which is a *better* "broadcast actively hurts" story than the one in the doc.

---

### O6 — The margin conflates inducing the target with destroying the foil

**OBJECTION.** "Δmargin = Δlp(T) − Δlp(F). At frac 0.5 you may simply be destroying the
foil's log-prob by writing the wrong-language direction into it. That is sabotage, not
steering, and it would show the same k-scaling."

**SEVERITY: weakens** (it does not flip any comparison, but it changes what the result
means, and a reviewer will ask for the decomposition).

**EVIDENCE STATUS: testable tonight, at zero extra compute.** `margin()` already computes
both terms; only the difference is retained (`full_modal.py:183-186`).

**CHEAPEST KILLING CONTROL.** Store `lp(T)_steered − lp(T)_base` and
`lp(F)_steered − lp(F)_base` separately and report both. Re-running one task at one k to
regenerate the logs: **≈ 5 GPU-min**. The claim survives cleanly if the target-side
component is a substantial share; it is badly wounded if Δmargin is ~all foil collapse.

---

### O7 — The single arm is neither budget-matched nor position-optimised

**OBJECTION.** "Your single arm writes at 1/k as many positions as the template, so of
course it delivers 1/k of the effect — and in `lang_profile` you pinned it at segment 0
rather than the best segment, while the `alt_phase` single arm gets an oracle-selected
`i_star`."

**SEVERITY: weakens.** It matters because "single covers 1/k" is one of the three
registered predictions, and it is currently a budget artifact plus an arbitrary position
choice. (Note it does **not** apply to template-vs-broadcast: those two arms write the
same magnitude at the same positions, `full_modal.py:161-172` and `:231-237`, so the
headline contrast *is* mass-matched. Objection 1 in the brief is refuted by the code.)

**EVIDENCE STATUS: partly refuted** (the single arm's value is already fully predicted by
`P(slot 0 differs) × per-slot effect`, see O1, so a position sweep will mostly recover the
same number); **testable tonight** for the matched-mass version.

**CHEAPEST KILLING CONTROL.** Two extra arms on `lang_profile`: `single_best` (sweep the
written segment over all k, take the max) and `single_matched_mass` (one segment at
magnitude k·m). Prediction: `single_best` ≈ per-slot effect (≈ 42, i.e. ~1/5 of the k = 10
template, exactly the coverage ratio); `single_matched_mass` saturates rather than scaling.
**≈ 10–15 GPU-min.**

---

### O8 — The generation demo: no unsteered control, unpaired arms, and a closed 12-sentence world

**OBJECTION.** "n = 24, no unsteered baseline, arms evaluated on different episodes, a
marker-word classifier built from the same 12 sentence pairs used to fit the direction and
to seed the prefix, missing slots scored as errors, and text so code-mixed that 'which
language is this sentence' is not well posed."

**SEVERITY: weakens** — this is the most defensible result in the set, and its central
number survives the worst version of the attack, but every one of the listed items is
individually cheap to close and currently open.

**EVIDENCE STATUS: mixed; several sub-objections already refuted.**

- *Classifier bias cannot manufacture the effect.* The profile is balanced and independent
  of the arm, so any profile-blind classifier — however French-biased — yields E[acc] = 0.5
  exactly. The template's 0.812 cannot come from classifier bias. **This is the strongest
  defence available and should be stated in the doc.**
- *Missing-slot asymmetry is real but small.* `mean_sents` is 6.00/6 for template@0.35 and
  5.62/6 for broadcast@0.35, and missing slots are forced wrong
  (`gen2_modal.py:177`). Correcting broadcast to produced sentences only gives
  0.444 × 6/5.62 = **0.474**, i.e. chance. So "broadcast sits at chance" is right, but the
  sub-chance value is a length artifact, not evidence that broadcast hurts.
- *Arms are genuinely unpaired.* Replaying `random.Random(500)` confirms the stored sample
  profiles exactly and shows the three arms draw **different** 24-episode sets (different
  profiles and different 2-shot prefixes). The imbalance runs *against* the headline
  (single's slot-0-French rate is 0.583 vs template's 0.417 at frac 0.35, and the prefix
  ends in a French sentence), so it is not the source of the positive — but it wastes power.
- *Prefix leakage is real.* The 2-shot prefix is built from the same `EN_FR` bank that fits
  `u`, and the broadcast sample copies bank sentence 7 verbatim
  (`"Ils marchent le long de la rivière."`). The demo shows control over a closed
  12-sentence world.
- *A harness inconsistency biases against the template.* The generation counter advances on
  `.!?` **only after ≥3 tokens** (`gen2_modal.py:148`), while the scorer splits on every
  `.!?` with no minimum (`:174`). Abbreviations and numerals ("p.m.", "15h.") desynchronise
  slot indices from the schedule that drove them. Misalignment can only hurt an
  index-dependent arm, so the true template effect is if anything under-estimated.
- *No unsteered arm exists* — `fracs` are 0.35 and 0.5 only, and the hook is a no-op at
  m = 0, so "chance = 0.5" is asserted, never measured.

**CHEAPEST KILLING CONTROL.** One re-run of `gen2_modal.py` with: an m = 0 arm; a
per-episode RNG (`Random(9000+ep)`) so all arms see identical profiles and prefixes; every
generated text stored; the scorer's splitter made identical to the generation counter; and
a held-out sentence bank (fit `u` on 6 pairs, prefix from the other 6). Then rescore the
stored texts offline with an external langid (`fasttext` lid.176 or `langdetect`) — CPU,
free. **≈ 15–20 GPU-min** for the run.

---

### O9 — Positional heterogeneity is uncontrolled and already visible

**OBJECTION.** "Early segments are not interchangeable with late ones, and every one of
your single-position and block-placement choices is confounded with position."

**SEVERITY: weakens.** It bites hardest on the W-sweep (see O3: W2_m1 can only start on
even segments) and on the `lang_profile` single arm (fixed at segment 0).

**EVIDENCE STATUS: testable tonight**, with a hint already present: `i_star` — the
position with the largest train-time target/foil difference — is 0 for `alt_phase` at
k = 2, 4, 6, 8 and 2 at k = 10, and 4 for `mirror`, so the per-position effect is not
uniform. The `alt_phase` single-vs-per-slot ratio `single/(template/k)` is 0.47, 0.38,
0.58, 0.57, 1.03 across k — if writes were perfectly additive and positions homogeneous
this would be 1.0 everywhere, so there *is* real positional structure (and sub-additivity)
worth measuring rather than washing out.

**CHEAPEST KILLING CONTROL.** A position-resolved single-segment sweep: write at segment
i alone for every i ∈ {0…k−1}, one task, k = 10, one frac, n = 32 — **≈ 8–10 GPU-min**.
Publishable as a figure in its own right (the per-position steering profile), and it
supplies the correction factor the W-sweep's rotated design assumes away.

---

### O10 — Baseline margins

**OBJECTION.** "Your baseline margins are negative almost everywhere. Either the target
and foil are not exchangeable, or something in the pipeline is asymmetric."

**SEVERITY: cosmetic** — for `alt_phase` the answer is interesting rather than worrying,
and for `lang_profile` it is consistent with noise.

**EVIDENCE STATUS: already explained.** For the A-tasks the foil is a uniformly random
distinct permutation of the target profile, so target and foil are exchangeable and
E[base] = 0 exactly; observed values (+0.14, −1.24, −1.92, +0.59, −2.71 for k = 2…10,
and −0.15 in the W-sweep at k = 12) are three negative-leaning cells short of a pattern,
and no SEM is stored to test them. For the B-tasks non-exchangeability is *by design* —
`profA` always starts TENSE — and the consistent −1.2 to −2.5 is a genuine model preference
for the calm-first arc, exactly the "schema preference" the docstring anticipates. The
doc's "baseline margins ≈ 0 (no schema preference)" is therefore slightly wrong for
`alt_phase`, and the diff-in-diff cancels it anyway.

**CHEAPEST KILLING CONTROL.** Store the SEM of the base margin (the per-pair list already
exists in memory at `full_modal.py:174`). **Free.** Then state the B-task preference as a
finding rather than denying it.

---

### O11 — One model, one layer, DoM proxies

**OBJECTION.** "Qwen2.5-1.5B, layer 14, difference-of-means directions. None of this is a
temporal crosscoder."

**SEVERITY:** depends entirely on the claim, and the ranking is worth stating explicitly:

- For "a constant write cannot control a trajectory whose target is multiset-matched":
  **cosmetic**. This is close to a mathematical consequence of the design and will replicate.
- For "per-slot steering efficacy is constant (or slowly decaying) in trajectory length":
  **weakens**. One model, one layer, one dose regime; the per-slot decay in `alt_phase`
  (−23%) is small enough that model choice could plausibly flip its sign.
- For "temporal crosscoders beat per-token SAEs for steering": **kills-the-claim**, but O2
  already kills it for a more basic reason — there are no dictionaries in the experiment at
  all, and the template is rank-1.
- For the generation demo: **weakens**. The doc already flags that DoM directions produce
  code-mixed rather than fluent output, which is the honest read.

**CHEAPEST KILLING CONTROL.** Layer robustness first, it is the cheapest and the most
likely to move: repeat `lang_profile` k ∈ {2, 10} template/broadcast/single at L8 and L20
(**≈ 10–12 GPU-min**). A second model (Llama-3.2-1B or Qwen2.5-3B) for the same two cells
is another **≈ 15 GPU-min**. Trained dictionaries are the real answer and are a
future-work item, not a tonight item.

---

### O12 — Train/eval hygiene

**OBJECTION.** "Your directions are fitted on the same sentence bank, the same carriers and
the same profile distribution you evaluate on."

**SEVERITY: cosmetic for the teacher-forced tasks, weakens for the generation demo.**

**EVIDENCE STATUS: confirmed by code reading, and mostly benign.** The train and eval loops
share one `rng` and one 12-pair (or 10-sentence) bank, and `capture_segs` runs on fresh
draws, so there is no *episode* reuse — but there is complete *content* overlap, and for
`alt_phase` the direction `t_dir[i]` is fitted per-position on exactly the profA-vs-profB
contrast that is then scored, i.e. an oracle-fit direction. For a steering experiment this
is defensible (you are allowed to know your target), and it is symmetric in the sense that
template and broadcast share the identical `u` in `lang_profile`. It becomes a real problem
only in `gen2`, where the same bank supplies the direction, the prefix and the classifier's
marker words (see O8).

**CHEAPEST KILLING CONTROL.** Held-out bank for the generation demo (folded into O8's
re-run, no extra cost). For the teacher-forced tasks, state the overlap in the limitations
rather than spending compute on it.

---

### Code-level findings

- `wsweep_modal.py:162-168` — `covered()` selects `m` **consecutive** blocks, collapsing
  the (W, m) grid onto coverage. See O3. This is the one outright design bug.
- `wsweep_modal.py:18` — docstring promises "per-pair deltas kept for SEM"; only the mean
  and SEM are written (`:194-196`). The matched-coverage contrast is the experiment's whole
  point and is currently unpairable.
- `gen2_modal.py:148` vs `:174` — the sentence counter that drives steering and the splitter
  that scores it use different segmentation rules (≥3-token minimum vs none). Biases against
  the template arm.
- `gen2_modal.py:164-172` — the episode RNG is shared across arms, so arms see different
  profiles and prefixes. Verified by replay; the imbalance happens to disfavour the headline.
- `sol_modal.py:209` — on the second `taskA` call, `hook.remove()` operates on a handle that
  was already invalidated by the `_forward_hooks.clear()` at `:234`, so the steering hook is
  live during `int_profile`'s capture phase. Benign today (`sol_modal.py`'s `steer_hook`
  iterates an empty `steer["v"]`, so it is a no-op) but it is one edit away from silently
  contaminating the direction fit. `full_modal.py` and `wsweep_modal.py` fixed this properly
  by registering once with an early-return guard.
- Positive control that passes: in `wsweep.json`, `W6_m2`, `W12_m1` and `full` return
  byte-identical means (266.23298297449946), confirming the hook plumbing and determinism.

### Numbers checked against the raw JSONs

Twenty-four checked, zero transcription errors. Spot list: `lang_profile` template
75.68/90.07/141.64/169.76/218.87 and single 29.57/19.29/26.05/20.30/16.85 and broadcast
−0.22/−0.02/+13.97/+10.10/+6.03; `alt_phase` template 21.00/38.53/53.41/65.62/80.53;
the sol table's +63.17/+5.98/+9.34 (lang), +12.54/+0.87/+0.84 (int), +7.83/+4.93 (mirror),
+21.63/+1.45 (alt); the 1/k share figures (29.57/75.68 = 39.1%, 16.85/218.87 = 7.7%,
4.94/21.00 = 23.5%, 8.30/80.53 = 10.3%); the negative-result table
(days k=2 1.091/0.907 → 1.20; days k=6 → 0.10; numbers k=2 → 3.99; numbers k=4 → 0.08);
generation 0.812 ± 0.044, 0.444 ± 0.028, 0.535, 0.743, 0.514, 0.507.

Three discrepancies, all in the doc rather than the data:

1. "(template plateaus by 0.35–0.5)" — false for `alt_phase`, where the increment is
   +36% to +48% at every k. Every reported peak is at the grid edge.
2. The signs-of-life table reports broadcast at frac 0.2 (−3.3 for `mirror`, −9.3 for
   `alt_phase`) while the JSON's `headline` field — max over frac, the convention the
   full-run table uses — is −0.24 and −0.42.
3. "baseline margins ≈ 0 (no schema preference; mirror shows no innate rise-fall bias)" —
   `alt_phase` base margin is −1.2 to −2.5 in every cell of every run, which is a real and
   expected preference for the calm-first arc.

### Recommended order for tonight

Ranked by (severity closed) ÷ (GPU-minutes), assuming the goal is a defensible summary
rather than a rescued headline.

1. **Free, do first.** Renormalise the k-sweep by differing slots (O1) and re-read the
   existing curves at frac 0.2 (O4). This decides what the section can claim, before any
   more compute is spent building on the current framing.
2. **W-sweep v2 with scattered blocks, free start offsets, per-pair deltas** (O3),
   ≈ 15–20 GPU-min. The centerpiece is currently confirmed-null by its own data; this is
   the only run that could change that.
3. **Fixed-Hamming foil k-sweep** (O1), ≈ 8–12 GPU-min. Turns the renormalisation argument
   into a direct measurement.
4. **`gen2` v2**: unsteered arm, paired episodes, stored texts, matched splitter, held-out
   bank (O8), ≈ 15–20 GPU-min, plus free offline rescoring with an external langid. This is
   the result most likely to survive review, and it is the cheapest to make bulletproof.
5. **Target/foil decomposition and base-margin SEM** (O6, O10), ≈ 5 GPU-min, folded into
   any of the above.
6. **Position-resolved single-segment sweep** (O9), ≈ 8–10 GPU-min. Yields a real figure.
7. **Layer robustness at L8/L20** (O11), ≈ 10–12 GPU-min, only if budget remains.

Trained dictionaries (O2) do not fit in this sprint. Until they exist, the section should
not claim that temporal *crosscoders* beat per-token SAEs at steering; the supportable
claim is about schedules, and it should be stated in those words.

### What survives

Stated as I would defend it to the same hostile reviewer:

- A constant (DC) write cannot move a multiset-matched trajectory margin, and a
  time-varying write of the same total magnitude and at the same positions moves it a lot
  (+266 vs −8 at k = 12). The arms share the identical direction, so this isolates the
  schedule cleanly. This is by construction rather than surprising, but it is correct and
  it is the honest version of the "trajectory control vs level control" framing.
- Per-slot steering efficacy does not collapse as the trajectory grows: it is flat within
  12% for `lang_profile` and decays only 23% for `alt_phase` over k = 2 → 10. That is a
  real, modest, defensible result — and it is the *opposite* in emphasis from the current
  headline, which reads the extensivity of the metric as growth.
- The dissociation against the mode-dominated ordered-days/numbers tasks is genuine and is
  the most valuable thing in the doc, because it is a sign flip rather than a magnitude.
- The generation demo shows per-slot language control at 0.812 vs a chance level of 0.5
  that is robust to classifier bias by a symmetry argument. With an unsteered arm and
  paired episodes it would be the section's strongest figure.

---

## Round 2 — the (W, ℓ) phase diagram, stance, and the convexity ruling

Audited after the retraction: `lsweep_modal.py` + `lsweep.json`, `stance_modal.py` +
`stance.json`, and the design of `convex_modal.py` (results not yet written).

### P1 — 18 of the 24 phase-diagram cells are algebraic identities, not measurements

**OBJECTION.** "Your 24-cell fit contains 18 cells whose value is forced by construction:
nine where the block-constant handle *is* the full template, and nine where it writes a
literal zero vector. Mean error 0.013 is a dilution statistic."

**SEVERITY: weakens** — it does not overturn the result, but it overstates it by ~4× on
both cell count and error, and "zero free parameters" is not accurate.

**EVIDENCE STATUS: already established from `lsweep.json`.** Two exact rules generate the
identities:

- **W divides ℓ** ⇒ `μ_b = ±1` ⇒ `c_cap = sign(μ_b) = π_t` for every t, so the arm is
  byte-identical to the arm that defines `Δ_full` ⇒ `obs_R ≡ 1.000`. Nine cells
  (ℓ=1: W=1; ℓ=2: W=1,2; ℓ=3: W=1,3; ℓ=6: W=1,2,3,6).
- **every block straddles equal ± halves** ⇒ `μ_b = 0` ⇒ `np.sign(0) = 0` ⇒ a literal
  zero write ⇒ `obs ≡ 0.000`. Nine cells (ℓ=1: W=2,4,6,12; ℓ=2: W=4,12; ℓ=3: W=6,12;
  ℓ=6: W=12). Their fingerprint is in the JSON: `sem` is exactly `0.000`, meaning every
  per-pair delta was identically zero.

Only six cells carry information: (ℓ=1,W=3), (ℓ=2,W=3), (ℓ=2,W=6), (ℓ=3,W=2), (ℓ=3,W=4),
(ℓ=6,W=4). Their errors are 0.084, 0.020, 0.020, 0.039, 0.127, 0.026 — **mean 0.053**,
not 0.013 (0.053 × 6/24 = 0.013 exactly, which is where the headline number comes from).

Two further corrections to the framing:

- **"Zero free parameters" is one scale per ℓ row**, calibrated on the W=1 cell — which is
  precisely why W=1 has error identically zero in all four rows.
- **`block_cap` and `block_energy` are not competing hypotheses.** Under a linear response
  both predictions are exact identities (see P2), so both fitting discriminates nothing.
  They differ in prediction in only three of 24 cells (ℓ=3 W=2, ℓ=3 W=4, ℓ=6 W=4). What
  they *do* provide is a nonlinearity probe, since `block_energy` uses coefficients up to
  1.22 — worth stating that way instead.

**CHEAPEST KILLING CONTROL.** Report the six informative cells as the result and the 18 as
design checks (they are useful as such — the zero-write cells are a genuine plumbing test).
To buy more informative cells at no new arm cost, sweep the profile **phase** (the code
fixes phase 0 at `lsweep_modal.py:180`); non-zero phases break the divisibility identities
and turn many W-divides-ℓ cells into real measurements. **≈ 15 GPU-min** for a
four-phase sweep at ℓ ∈ {2,3}.

### P2 — the predicted R is an algebraic identity under linearity, so the phase diagram tests linearity, not a window law

**OBJECTION.** "Your attenuation law is not a law about windows; it is the statement that
Δ is a linear functional of the write schedule, which your own W-sweep already
established."

**SEVERITY: weakens** — and it reframes the result rather than killing it. The measurement
is real and the fit is good; the claim it supports is narrower than "performance improves
with window size".

**EVIDENCE STATUS: provable on paper.** Assume `Δ = a·⟨c, π⟩` with a position-independent
`a` (the additivity O1 established). Then for the magnitude-cap arm,

```text
Δ(W) = a·Σ_b sign(μ_b)·Σ_{t∈b} π_t = a·W·Σ_b |μ_b| = a·k·mean_b|μ_b|
Δ_full = a·Σ_t π_t² = a·k
R = Δ(W)/Δ_full = mean_b|μ_b| = pred_R          (exactly)
```

and for the energy arm the same algebra gives `R = sqrt(mean_b μ_b²) = pred_R` exactly.
So both predicted columns are the projection identity `R = ⟨c,π⟩/⟨π,π⟩`. The phase diagram
measures **one** thing — how linear the steering response is in the schedule — in six
independent cells, and the answer is "linear to within resolution".

That is worth reporting, and it is a *better* design than the W-sweep in three specific
ways that should be said out loud: coverage is pinned at k so O3 cannot recur; total
injected norm is identical across all cells of a row (every segment gets a ±1 write), so
O7's mass-matching objection cannot recur; and normalising by `Δ_full` on the *same* eval
pairs divides out both O1's extensivity and the per-pair Hamming variation. Credit where
due — this is the cleanest harness in the set.

The honest headline: **steering response is linear in the write schedule, so a
block-constant handle delivers exactly its projection onto the target trajectory, and the
best window is the one that aliases the target period — which is not monotone in W.**

### P3 — the three "zig-zags" are each one measurement against one identity

**OBJECTION.** "Your falsifiable signature is that a wider window beats a narrower one. In
two of three cases the narrower window writes nothing at all."

**SEVERITY: weakens.**

**EVIDENCE STATUS: already established.** ℓ=1: W=3 (0.249, informative) beats W=2 (0.000,
zero-write identity). ℓ=2: W=6 (0.353, informative) beats W=4 (0.000, zero-write identity).
ℓ=6: W=6 (1.000, identical-to-full identity) beats W=4 (0.641, informative). So each
zig-zag pairs one real measurement with one construction. The non-monotonicity is a
property of the aliasing arithmetic, which the code's own docstring concedes ("purely
combinatorial"), not a discovered property of the model. It remains a good pedagogical
figure — wider is not better — provided the caption says why.

### P4 — the fit quality is at the noise floor and has no confidence interval

**OBJECTION.** "You report a mean error but no uncertainty on it, and your per-pair deltas
are not stored, so no one can compute one."

**SEVERITY: weakens.** Using `sem(obs)/|Δ_full|` as a conservative per-cell noise proxy
(conservative because obs and `Δ_full` share eval pairs and are positively correlated),
the informative cells have a noise floor of **0.079** against a mean error of **0.053**.
The fit is therefore as good as the data can resolve — genuinely a pass — but the
experiment cannot exclude any alternative law differing from linearity by less than about
0.08 in R. One cell deviates: **ℓ=3, W=4 — obs 0.206 vs pred 0.333, 1.9σ** (and 2.1σ for
the energy arm, obs 0.237 vs 0.408), both undershooting. That single residual is the only
place in the phase diagram where the model departs from linearity, and it is the cell
worth n=100.

**CHEAPEST KILLING CONTROL.** Store per-pair deltas (free), then bootstrap the paired ratio
so R gets a CI. Re-run ℓ=3 alone at n=96 to settle the W=4 residual: **≈ 12 GPU-min**.

### S1 — stance: the k-growth does not survive per-differing-slot normalisation

**OBJECTION** (O1 applied as requested). "Your staged-refusal margin grows +20.7 → +28.6
across k = 2…8 for the same reason your language task did."

**SEVERITY: kills-the-claim** for "template grows with k" on stance; the arm-ranking
results are untouched.

**EVIDENCE STATUS: already refuted by existing data.** `make_pairs` uses a balanced profile
with a uniformly random distinct permutation as foil, so E[Hamming] is exactly computable
by enumeration: 2.00, 2.40, 3.16, 4.06 for k = 2, 4, 6, 8.

| k | E[Hamming] | template | per differing slot | single |
| --- | --- | --- | --- | --- |
| 2 | 2.00 | +20.73 ± 1.63 | 10.36 | +5.16 ± 1.68 |
| 4 | 2.40 | +20.66 ± 2.51 | 8.61 | +3.49 ± 1.44 |
| 6 | 3.16 | +23.89 ± 2.03 | 7.57 | +3.39 ± 1.35 |
| 8 | 4.06 | +28.58 ± 3.35 | 7.04 | +0.78 ± 0.91 |

Per-slot efficacy **falls 32%** across the sweep. **Prediction for the fixed-Hamming
control on stance: flat to mildly declining, and certainly not growing** — register it
before that run lands, as with lang_profile.

Two credits, because the construction is better than its predecessors: `sents_for` assigns
sentences by a running counter, so target and foil use the **identical multiset of
sentences** merely reordered (a stronger match than `lang_profile`, where the same indices
were rendered in different languages and token counts differed); and the disjoint bank
halves (A trains the direction, B builds eval pairs) are a real leakage control that the
earlier harnesses lacked.

One genuine anomaly: **single at k=8 is +0.78 ± 0.91 where additivity predicts
≈ 0.5 × 7.04 ≈ 3.5** — a 3σ undershoot, and the only place a single-slot write behaves
unlike the arithmetic. Worth one figure.

### S2 — stance: the experiment's own pre-registered tell fired

**OBJECTION.** "You wrote that a near-zero `cos(u_stance, u_prompt_refusal)` is the tell
that you have a style direction rather than a refusal direction. You measured 0.108 and
reported the result as a refusal-steering match."

**SEVERITY: weakens → kills-the-claim** for the framing "mid-response safety recovery,
the behavior whose real-model lever is the most mature in the literature". The steering
result stands; the *bridge to the refusal-direction literature* does not, on the
experiment's own criterion.

**EVIDENCE STATUS: already established** (`stance_modal.py:29` states the criterion;
`stance.json` gives 0.108). For calibration, two random directions in 1536 dimensions have
|cos| ≈ 0.026, so 0.108 is well above chance but explains ~1% of variance. **Fair caveat
that must be stated with it:** `u_prompt` is measured at the last prompt token and
`u_stance` over response-sentence spans, so some of the orthogonality is positional rather
than semantic. The honest reading is that `u_stance` is largely a refusal-*register*
direction (the linguistic act of declining) rather than the refusal *decision* direction.

**CHEAPEST KILLING CONTROL.** Measure `u_prompt` at matched positions (first response
tokens after the generation prompt, harmful vs benign request) and re-take the cosine;
additionally project `u_stance` onto the span of a proper harmful/harmless
difference-of-means and report the retained fraction. **≈ 8 GPU-min.** If the cosine stays
near zero at matched positions, rename the result "per-sentence stance register" and drop
the safety-recovery framing.

### S3 — stance: the pre-check gate is one-sided

**OBJECTION.** "Your gate tests P(comply | previous refuse) and passes at 0.87. The
transition table shows the opposite direction is 0.026 — the model has a strong
comply-attractor, which is exactly the pathology the gate exists to detect."

**SEVERITY: weakens** (teacher-forced results are unaffected, as the docstring correctly
notes; it matters for any generation-mode follow-up).

**EVIDENCE STATUS: already established.** `transitions` = {R→C: 20, R→R: 3, C→R: 1,
C→C: 38}: P(comply | refuse) = 0.870 but P(refuse | comply) = **0.026**, on n = 23 and
n = 39 transitions respectively. A seeded refusal is abandoned almost immediately and
compliance is essentially absorbing. Steering *into* refusal mid-response — the direction
that matters for safety recovery — is the hard direction and is untested.

**CHEAPEST KILLING CONTROL.** Report both conditionals, and gate on the *minimum*. For the
generation-mode run, seed mid-comply and measure induced refusal; **≈ 10 GPU-min**.

### C1 — ruling on the convexity claim: what threshold on S I would accept

Requested ruling. My position: **realmodel is right that the current reading is unsafe,
and I would not accept S > 0 as a window effect under any threshold unless it is
accompanied by a contiguous-versus-scattered contrast.** Reasons, in order of force.

**First, the observed convexity is fully explained by a super-linear dose response, with no
window physics.** The stance frac grids give a local exponent `p` in `Δ ∝ frac^p` of
1.07–1.17 (k=2) rising to 1.11–2.10 (k=8, at the top of the grid). Fitting a single power
law `Δ ∝ N^q` in the number of written segments, calibrated on the two endpoints of the
stance W-sweep, gives **q = 1.27** and reproduces the whole curve:

| W | observed | power law, q = 1.27 | deviation |
| --- | --- | --- | --- |
| 1 | +2.34 ± 1.18 | +2.34 | calibration point |
| 2 | +6.25 ± 1.36 | +5.62 | +0.5σ |
| 4 | +9.78 ± 2.52 | +13.53 | −1.5σ |
| 8 | +32.58 ± 3.06 | +32.58 | calibration point |

Three of four points sit within 1.5σ of a curve containing no window term at all.

**Second, the normalisation is the artifact realmodel identified.** The "fraction of the
additive line" values 0.57 / 0.77 / 0.60 / 1.00 are deviations from a line through the
origin and the W=8 point, so the W=8 cell is 1.00 *by construction* with zero deviation.
The measured deviations at W = 1, 2, 4 are −1.5σ, −1.4σ, −2.6σ. Stated without the
normalisation, the finding is "**partial-coverage arms undershoot full coverage
proportionally**" — sub-additivity of partial writes — which is the same fact with the
rhetoric reversed, and which O9's `single/(template/k)` ratios (0.38–1.03) already showed.

**Third, the non-monotonicity falsifies the offered model.** The edge-penalty form
`Δ(W)/W = a − c/W` is strictly monotone in W, as `convex_modal.py:20` states. The observed
0.57 → 0.77 → 0.60 → 1.00 is not, so either the model is wrong or the W=4 point is noise
(it is the least precise, ±2.52). Either way it cannot currently be reported as convexity.

**Fourth — and this is the design point — the scramble control cannot do the job it is
assigned.** For a balanced block, a uniformly random within-block permutation σ gives
`E_σ[Σ_t c_{σ(t)} π_t] = (Σc)(Σπ)/W = 0` exactly. So under plain additivity
`E[Δ_scrambled] = 0` and `E[S_scrambled] = −Σ_t Δ_t` for **every** W: the collapse is
predicted by additivity alone and carries no information about coherent transitions.
Worse, at W = 2 the permutation group has two elements, so half the episodes are not
scrambled at all. The scramble destroys sign-correctness and adjacency together; only the
scattered arrangement holds correctness fixed and varies adjacency alone.

**The threshold I would accept.** Define the span statistic on *paired* per-pair deltas:

```text
S_span(W) = Δ(contiguous W segments) − Δ(scattered W segments)     [same j, same dose,
                                                                    same coverage, same signs]
```

I would call a window effect real when **all** of the following hold:

- `S_span(W) > 0` at **≥ 3 paired SEM** (not the unpaired combination — with per-pair
  deltas the paired SEM should be roughly half the unpaired one);
- at **≥ 2 values of W** and **≥ 2 fracs**, with the effect monotone in W;
- with effect size `S_span(W)/Δ(contiguous W) ≥ 0.15`;
- and the scattered arm's own `S` (against the marginals) accounting for the rest, so the
  dose-convexity share is explicit rather than absorbed.

Against that bar the existing evidence is **null**: the phase-diagram job's own contrast is
contiguous +18.94 ± 2.40 versus scattered +17.42 ± 2.81, a difference of **+1.52 ± 3.69
(0.4σ, 8% of the contiguous value)** at coverage 4. My prior is therefore that `convex.json`
will show S > 0 and it will be dose convexity.

**What I would still object to if S > 0 significantly.** Even with S at 5σ against the
marginals, three alternatives remain open and none is addressed by the current design:
dose/coverage convexity (kill it with the scattered contrast, or with the direct control
below); the single-segment marginal being anomalously small for a position-specific reason
rather than a span reason (kill it with the position-resolved marginal spread — the code
already computes `marg[t]` per position, so report its variance, free); and grid-edge dose
selection, since every arm again peaks at frac 0.5.

**Two cheap additions to `convex_modal.py` before it is believed.** Store per-pair deltas
for the **marginals** — `results["marginals"]` currently keeps only mean and sem
(`convex_modal.py:217-219`) while `blocks` keeps `deltas`, so S has no paired SEM, which is
the single thing that would most improve its power. And add one arm: **one segment at
magnitude `W·m`** versus **W segments at magnitude `m`**. If those match, the
"superadditivity" is dose, not span. Both are **≈ 5 GPU-min** inside the existing job.
