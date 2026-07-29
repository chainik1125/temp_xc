# mac-c — STATUS

**Agent:** mac-c (macOS, `~/research/projects/agents/mac-c/temp_xc`)
**Lane:** ⚑ **THE WHOLE TASK HUNT, END TO END** (`hunt-mac-c-takeover.md`),
re-aimed by `briefings/hunt-safety-gold-clew.md` (active, owner mac-c)
**Last update:** 2026-07-29 02:01 BST (stamped from `date` at write time)

---

## 22:52 — checkpoint durability audit (unassigned, $0, PUSHED `eb9f3fb47`)

Woke on a listener fire into mac-d's item-6 BLOCKER (*"sycgen SAE anchor
weights DO NOT EXIST"*). Nobody had measured the **extent** — hub found
the mechanism, mac-d found one instance. I measured it, read-only, no pods.

**Result: pod-D's dictionaries are NOT lost.** All 6 trained sycgen
checkpoints are on HF (`han1823123123/temp-bench-data`,
`ckpts/<train_key>/model.safetensors`), and `checkpoint_exists()` returns
**False** for all 6. `cache.py:148` branch (b) reads `hf_url`;
`trainer.py:171` writes `None` unconditionally as the *only* writer;
`push_ckpts_hf.py` deliberately doesn't backfill. **0 of 10,400 manifest
rows carry `hf_url`** — the branch has never fired. Fleet-wide: **344 of
9,631 train_keys are on HF and all 344 report absent.**

**I withdrew my own $10 cost recommendation before publishing it** —
rebased mid-write onto `8990a31fe`/`2a8fd7473`, found mac-d had moved to
a fresh `sycgen_keep_r1_rebuilt` tag retraining all 15 deliberately for
provenance (mirrored weights were trained on pod-D's *dead* hs14.npy).
Their reason beats my saving. Retraining is correct.

**Open, deliberately deferred:** `_key_from_manifest` now hard-codes
"exists" = "on this box". Right for tonight's retrain, wrong as a
permanent definition. **Revisit AFTER item 6 ships, not during.**
Artifacts: `checkpoints/durability_audit.json`,
`experiments/explorations/task_hunt/ckpt_durability_audit.py`.

**Posture:** 0 mac-c pods, $0 all session, listener armed, nothing owed.
Three lanes still parked behind item 6 per hub ruling.

---

# ⚑⚑ RESUME HERE FIRST — WRITTEN FOR A POST-COMPACT ME (00:32, pre-compact handoff)

## 0. Non-negotiables, in order

1. **RE-ARM THE LISTENER.** It is ONE-SHOT: it exits on hit. I have
   broken this once and the user caught it ("YOUR MONTIRO IS DOWN").
   **After every fire, immediately:**
   `zsh /private/tmp/claude-501/-Users-octo-research-projects-agents-mac-c-temp-xc/660a6cf4-2ce0-4db9-8201-37e38db3e1bf/listener.sh`
   (background). NOTE the path — it now lives at the **session root**, not
   `scratchpad/`, and watches **BOTH `main` and `arxiv`**.
2. **BRANCH = `arxiv`.** Do NOT switch to `main`. `main` is FROZEN at
   `7ceb45564` (March scaffold, 36 files) by Han's reversal. I checked
   out `main` at 23:17 and my tree became a 36-file repo. Verify any
   branch move with **`git log --oneline <target>..<source> | wc -l`
   MUST be 0 AND `git rev-parse <target> <source>` MUST match** — the
   reverse range (`arxiv..main`) returns 0 in BOTH the success and
   disaster cases and cannot detect this.
3. **0 mac-c pods, $0 spent all session.** Keep it that way unless a
   measurement genuinely needs a GPU.
4. **LOG.md union conflicts on every push.** Recipe:
   `sed -i '' -e '/^<<<<<<< HEAD$/d' -e '/^=======$/d' -e '/^>>>>>>> /d'`,
   `git add`, `GIT_EDITOR=true git rebase --continue`, verify markers=0.
5. **Stamp from `date` at write time.** I have corrected my own stamp
   FIVE times today.

## 1. ✅ NOTHING IN FLIGHT — the lever-3 run COMPLETED. Do not re-run it.

The cache (`bxfp10te4`) finished clean (`done 3579 rows in 767s`,
degenerate-assert passed) and **`lever3_evalage` ran and is committed**
(`a027b7caa`, result JSON
`facecmp/results/lever3/lever3_evalage_gemma2_512.json`). The cache
itself still sits at `<scratch>/cache_evalage_512/gemma2_2b/hs14.npy`
(~8.4 GB) if anything needs re-measuring — **reuse it, do not rebuild**
(~13 min on MPS).

### ⚑ THE RESULT, and how it must be quoted

**P1 HELD 5/5** (evalage floor below retryesc at every T — the falsifier
against my own floor law did not fire). **P2 HELD** (arm>floor 5/5 vs
retryesc 2/5). **P3 FLIPPED as predicted** (floor-bound → gain-bound).
**P5 FIRED: 2/5 cells clear BOTH bars** — T32 and T64; best **T64: arm
0.5308, floor 0.3905, gain +0.0709 over tok 0.4599, min-both +0.0209**.
Controls clean (`label_null` 0.3460 ≈ chance; foreign 0.4292).

**⚑⚑ THIS IS A RESCUE AND IS DISCLOSED AS ONE (P5, written pre-data).
NEVER quote it as though it passed the original screen.** Three
provisos, all already on record and none to be dropped when
re-summarising:

1. **T32 is not credibly anything** — margin **+0.0054**, inside the
   sampling SE alone (Lane B: SE_boot ≈ 0.0074). Only **T64** merits
   follow-up.
2. **Single seed.** Lane B: training variance is real and does not
   shrink with `n_test`. No error bar exists for *this* corpus yet —
   that is the obvious next measurement if the lane is resumed.
3. **⚑ THE BAR IT CLEARS IS THE SUPERSEDED ONE.** It runs against `tok`,
   a per-token probe. The post-item-6 bar is **"beats a pooled SAE on a
   measured budget frontier."** Against the current bar this cell is
   **unevaluated**. A rescue against a retired bar is not a KEEP.

## 2. Why this run exists (the hub's priority, and why it is NOT a repeat)

Hub un-parked my hunt lane (00:20) with ONE priority: *"re-label an
existing corpus at T=16 and measure arm−floor BEFORE generating
anything."* **I already ran that** (`bf7aa3b5f`): T{4,8,16,32,64},
gemma2_2b@512, **0/5 KEEP-shaped**, best T16 at −0.0249.

**The gap is the CORPUS, not the T.** That sweep was `retryesc_gen`
where **w=25**, so at T=16 the floor still sees T+w=41. **`evalage` has
w=13** → horizon **29**. That is lever 3's `w` half applied **for free
to a corpus we already own**. My earlier "do not fund a narrow-event
corpus" verdict was about **generating** one; it does not apply here.
`evalage` is also the rescue candidate (cleared width-null, its floor,
and the within-conversation control on every leg; failed only the
hardcoded `gain>=0.05` by ~0.33σ).

**Pre-registrations P1–P5 are FROZEN at `2728d6229`, committed BEFORE
the numbers.** Read `lever3_evalage.py`'s docstring and honour them:
- **P1** evalage floor much lower at every T — **with a falsifier
  against me**: if it is NOT, my corrected floor law is wrong and I
  report that.
- **P3** the binding constraint should FLIP floor-bound → gain-bound.
- **P5** ⚑ **if a cell clears BOTH bars it is a RESCUE and is disclosed
  as one** — screen verdict WEAK, re-measured because the miss was
  inside the noise and the geometry was wrong. **Never quote it as
  though it passed the original screen.** This is the first corpus with
  a real chance at both bars, which is exactly when the temptation to
  launder a rescue into a pass appears.

The script scores P1/P2/P3 automatically against the frozen retryesc
run and prints HELD / NOT HELD. **Report whichever way it falls.**

## 1b. ⚑⚑ THE HUNT'S AIM WAS CORRECTED TWICE TONIGHT — READ THIS BEFORE TOUCHING THE HUNT

`experiments/explorations/task_hunt/FLOOR_AIM_CORRECTION.md` +
`beyond_t_mass.py`. Net position after two of my own retractions:

**STANDS — the instrument fact.** `visible_evidence_floor` is fit on
`sage_floor(event_first,T)` + `dose_window_count(event_mask,T)`, both
handed **ground-truth event arrays** (`screen.py:113` → `:301`).
**Neither reads a token.** So "per-token silent" aims at the **gain**
bar — and `retryesc_gen` cleared gain 3/3 and died on the **floor**.
Two of the brief's three gold criteria select for a non-binding bar.

**⛔ RETRACTED (mine).** *"Censored-age labels cannot beat this floor at
any density."* **False — `sycgen`, our only KEEP, is exactly that and
beats the SAME `_FloorBank` by +0.121/+0.137/+0.140 on 3/3 models.**
`sage_floor` censors at **T+1** but the label is the **uncensored** age,
so winnable mass is `P(age > T)`. **My own `d2320d274` already said
this** and I wrote a universal denying it.

**⛔ RETRACTED (mine).** *"Type-injection is the only non-vacuous
family."* **Backwards under my own criterion.** A type label the floor
cannot express is **ESCAPED, not beaten** — and *an escaped control is
not a passed control*. Timing labels face an **expressive** floor and
sycgen genuinely beat it. Type labels need a **type-aware floor built
first** before any KEEP means anything.

**⛔ FALSIFIED (the hub's, by my run).** *"Sort candidates by beyond-T
mass."* `P(age>T)` **saturates at 1.0000 for both corpora at T=4,8,16**
and orders them backwards at T=16. I scored `event_first`; the floor's
discriminating feature is `dose_window_count(event_MASK,T)` — mask
**width** `w`, which is why `w` is lever 3.

**⚠ DEMOTED — `floor_reach` is NOT a screen. Do not use it as one.**
`floor_reach = P(any masked token in the trailing T window)`. It tracks
observed `floor_excess` within 0.82–1.13× **on evalage and retryesc**,
and separates those two by ~5.3× — but the out-of-sample test on our one
KEEP **refuted it as a predictor** (`2eb100e01`):

    evalage  0.0138@T16  -> WEAK      <- ranked BEST by the criterion
    sycgen   0.0242@T16  -> KEEP 3/3  <- the actual winner, ranked 2nd
    retryesc 0.0732@T16  -> KILL 3/3

**It is NECESSARY, NOT SUFFICIENT: a high value kills you, a low one
buys nothing.** And **the underlying law fails on sycgen by 2.3×**
(0.0892 predicted vs 0.2097 observed) because it assumes out-of-window ⇒
chance, which is **false on a class-BALANCED test set** — every
uncensored sycgen row is class 0 (`P(uncensored|class)` =
0.2822/0/0), so predicting one class for the censored mass banks a large
share free. Still fine as a *description* of what the floor can resolve;
**never quote it as a candidate-quality screen.** Run:
`PYTHONPATH=. .venv/bin/python -m experiments.explorations.task_hunt.beyond_t_mass`

**✅ WHAT ACTUALLY SEPARATED THE ONE SUCCESS — ARM STRENGTH, not floor.**
sycgen's floor is the **highest** of the three (0.5430 vs evalage 0.3905,
chance 1/3 both) and it won anyway. Excess over chance, gemma2_2b:

              tok      floor      arm    arm-floor    gain
    sycgen  +0.196    +0.210    +0.308     +0.098   +0.1118
    evalage +0.127    +0.057    +0.197     +0.140   +0.0709

**The hunt has been lowering the floor; the one KEEP came from a better
windowed signal.** The open question is *what makes an arm strong* — the
**gain** side.

**✅ UPPER EDGE MEASURED** (`a88132d8d`, pre-reg frozen at `d6b11068d`
before the run). evalage gemma2_2b L14: act 0.6190, position control
0.2171 (U2 held 7×), label_null 0.2110; recall 0.9976 → 0.8167 → 0.4952
→ 0.4143 → **0.3714 at 1024+**, every bucket above chance ⇒ per U4
**"horizon exceeds the measurable range"**, not a number. ⚑ **The probe
IS the `tok` arm**, so the same retention that removes the cliff is what
makes gain hard — **the band is ONE-SIDED**. Disclosure on record
(`ebc752e90`): I changed pre-registered `CHANCE` 1/6→1/5 after run 1
(the `1-3` bucket is empty by construction, w=13); under the frozen 1/6
U3 would have squeaked at 89% of budget vs 22%. **Pre-register the RULE,
not the literal.**

### ✅ ALSO CLOSED: overnight item §5.2, the RLHF metric collision (`d18c556db`)

The flag both understated and overstated it. **10 row-pairs share all
four metrics and NINE ARE BENIGN** — same base arch under the `_btkonly`
rename, `batch_size` 1024 on both sides. **Discriminator = `batch_size`.**
Exactly one anomaly: `tsae_btkonly` `9147e04c9555b952` (bs=**32**) carries
metrics byte-identical to `batchtopk_sae_btkonly` `4f20f7d8dbc3b855`
(bs=**1024**), 73s apart; P(all four match by chance) ≈ 7.2e-08. **Also
distrust `037faff74c387997`** — same 90s window, AUC exactly
**0.5000000000**, a degenerate-predictor signature. Tested and refuted
the obvious hypothesis (metric is arch-blind): 26 rows → 20 values.
**"16 digits" is a red herring** — AUC is an exact rational
(`54252/82355`), support 61 values in 73 rows. **Sibling metrics, not
digit count, are the test.**

### ⏭ THE OLD NEXT STEP — DONE, see UPPER EDGE above. Kept for the design note.

`floor_reach` is the **lower** edge only (what the floor can resolve).
The upper edge — **how far back the residual stream still carries a
sparse event** — is unmeasured and is what decides whether "push events
older than T+w" actually works. A corpus can score perfectly on
`floor_reach` and still lose because its events are too old to be
represented at all.

**Everything needed is already on disk** — reuse, do not rebuild
(~13 min MPS): `<scratch>/cache_evalage_512/gemma2_2b/hs14.npy` (8.4 GB)
+ `evalage/grids/elicit_evalage_screen_gemma2.npz`.

**Design sketch (PRE-REGISTER BEFORE RUNNING).** `arm_test.build_rows`
computes `val = rate_face(...)` as the **raw** quantity and only then
tercile-bins it (`arm_test.py:166,181-186`). A variant must keep raw age
and bucket by it: fit a multiclass probe over log2 age buckets on
eligible positions, report **per-bucket accuracy vs chance**, and read
the retention horizon as the largest age still above chance. Controls:
`label_null` permutation and the `foreign` window, exactly as the screen
does. **State a falsifier first** — e.g. if accuracy is flat across all
buckets, the probe is reading position rather than age and the run is
void.

## 2b. ⚑ THE SHUFFLE LANE — my standing role is REVIEW, and it is live

`briefings/sycgen-shuffle-sparsity-matched.md` — **owner: mac-d
(executor) + mac-c (review/pre-reg audit)**, card
`sycgen/SHUFFLE_MATCHED_CARD.md`. mac-d executes; **I do not run cells
on this lane.** Scale was corrected 20×H100 → **1 GPU** (it is
encode-and-probe, not retrain).

**My audit (§5 of the briefing) — ALL FIVE ADOPTED by hub `29bc6a95d`
and bound into the card by mac-d `ab415af18`:**

- **A1 (blocking)** the pooled-zero gate **cannot fail** — `mean(dim=1)`
  is permutation-invariant arithmetically, so a no-op shuffle passes it
  and drives every arm to gap 0, reading as **(b)**, the outcome the
  card pre-commits to publishing. Receipt:
  `sycgen/shuffle_gate_receipt.py` ($0, self-asserting) shows PASS on
  both a live and a dead shuffle.
- **A2** twin is now a **gate on (a)**; **A3** (a) needs 3/3 sign
  agreement AND margin > across-seed SD; **A4** `1 − 1/T!`; **A5** stamp.

**⚑ OPEN AND UNANSWERED — my review of mac-d's A1 strengthening
(`2efb94029`), posted 00:52, not yet ratified.** They replaced my
minimal assert with "measured fraction == `1 − 1/T!` (binomial tol)" —
right in direction, but **the tolerance is still literally the words
`(binomial tol)`**, and at **T=8 an equality gate spuriously VOIDs
9.66% of HEALTHY runs** (E[identity rows] = 0.102). Corrected bands are
in the card: T2 7936..8448, T4 269..414, T8 0..3, T16 0..0, on
`Binomial(n, 1/T!)` with `n = n_windows·(L//T)`. **I posted a Poisson
band first and corrected it — Poisson SD 90.51 vs binomial 64.00 at
T=2, ~40% too loose. Use the binomial numbers.** If this is still
unratified next session, chase it before any cell runs.

## 3. Delivered tonight (all $0, all ratified — do not redo)

- **Lever-3 rescue** (`a027b7caa`) — see §1, including the three
  provisos and the P5 disclosure obligation.
- **Pre-reg audit of the shuffle brief** (`a027b7caa`) + **A1 receipt**
  (`46dafaa06`) — all five findings adopted; see §2b.
- **A4 blast radius** (`ef972e34a`): A4 reaches **four** overlays, not
  the one the hub patched — all draw `ts` from `WINDOW_TS` starting at
  T=2. **But I measured before claiming: it is sub-noise everywhere**
  (sycgen T2 +0.0334 ±0.0386, lambda T2 +0.0043 ±0.0051 — SD ≥ mean), so
  it is a **disclosure obligation, not an invalidation**, and sycgen's
  T=2 gap is *larger* than its T=4 gap, the opposite of
  attenuation-dominance. Where it does bite: **`diafaces`**, whose
  T-trend is load-bearing (T32 +0.1294 ±0.0233) and which did **not**
  get the sycgen patch — recommended disclosure on
  `TT_SHUFFLE_OVERLAY_CARD.md` / `SHUFFLE_OVERLAY_CARD.md`, **still
  open**. Separate larger caveat found while measuring: sycgen's and
  lambda's low-T gaps **are not statistically resolved at n=3 at all**,
  independent of A4.

- **Checkpoint durability audit** (`eb9f3fb47`): `checkpoint_exists()`'s
  `hf_url` branch is DEAD CODE — `trainer.py:171` writes `None` as the
  only writer, 0/10,400 manifest rows carry one. **344 of 9,631
  train_keys are on HF and all report absent.** pod-D's sycgen
  dictionaries are NOT lost. Artifacts:
  `checkpoints/durability_audit.json`,
  `experiments/explorations/task_hunt/ckpt_durability_audit.py`.
- **σ correction** (`575958b0d`): the pre-registration quoted my
  variance work at 1.5×; correct factor is **1.83–3.99×** (`sqrt(1+r²)`,
  not the variance ratio). Hub amended in place, pre-data.
- **Estimator scope bound** (`72cf1334f`): my σ figure does NOT apply to
  the anchor check — `lambda_recovery` uses **closed-form sklearn OLS**,
  not Adam. Protected a sound verification from my own finding.
- **Divide-by-T artifact** (`c1a9f98ad`): `l0_per_token` is
  `l0_per_window / T` (`synthetic_recovery.py:201`), so "recovery up as
  budget falls" inverts on the verdict axis. mac-d withdrew the trend.
- **Branch alarm** (`dcdf9dac3`) + **check-invariance** (`851d73f85`).
- **Briefing retired** (`1a901cdb5`): `URGENT-budget-matched-table.md`,
  nothing was open; §6 was discharged 4h earlier.
- **Self-sweep** (`6f4c7bf45`): found THREE of my own retracted claims
  still asserted in docstrings — `lane_b_errorbar.py` (ρ "cuts against
  the rescue" — refuted by that script's OWN output, ρ=0.383–0.534, and
  the correction HELPS evalage), `verify_floor_identity.py` (the
  identity is a low-density approximation), `dry_run.py:110`. All
  corrected in place. **Run positive controls FIRST in any such sweep**
  — mac-d's ran dead and looked clean on zsh word-splitting.

## 4. Lanes — ALL FOUR UNBLOCKED (item 6 shipped)

Order set by the hub at 00:20: **(1) lever 3 = §1 above**, then the
`clew` literature sweep (`hunt-safety-gold-clew.md`), then the
rescue-retrain lane (`hunt-rescue-retrain-mac-c.md`, whose Lane A′ IS
the same geometry matrix — `llama31_8b` and SEQ_LEN 1024 are the cells
that genuinely need a GPU), plus standing `hunt-mac-c-takeover.md`.

**⚑ The KEEP bar has MOVED, per item 6's close:** a KEEP is no longer
"beats a per-token probe" but **"beats a pooled SAE on a measured budget
frontier"**. Screen future candidates with that comparison from the
start; mac-d built the harness section-agnostic on purpose.

---

# RESUME HERE

## State in one paragraph

**Item 7 still has no KEEP, and I no longer think the bottleneck is task
choice.** Tonight closed two directions with receipts and then found
something structural: **the benchmark's arm has almost no room to beat
its own floor, for geometric reasons.** The floor sees `T + w` tokens
(~89); every probed token only ever saw **64–128** tokens of context,
because the activation cache chunks the stream into independent
128-token sequences. So the arm's entire structural advantage is the
band ~89→128 — **a factor of 1.4.** Everything else tonight follows from
that. **$0 spent all night. 0 mac-c pods (API-verified 19:11).**

## Prior lane, for context

**`retryesc_gen` CLOSED — WEAK 3/3** (`retryesc_gen/RESULT.md`), ≈$22
($21 gen + $0.68 screen), pod terminated + API-verified. Gain cleared on
all three legs (+0.063…+0.069); **the floor clause killed all three.**
The retryesc family is closed at **attempt 2 of 2**; the calibrated
re-aim I floated at 16:05 is **dropped, not deferred**.

## What is on record tonight (all pushed, all $0)

1. **`floor_excess ≡ f` is NOT a law** — it is a low-density
   approximation. The floor's second feature is
   `dose_window_count(**event_mask**, T)`, so its effective window is
   `T + w` (`w` = masked event-TURN width; evalage 13, retryesc_gen 25).
   Replacing `f` with **`P(any masked token in window)`**: mean |resid|
   0.0391 → **0.0056** (max 0.0075) across **6 legs / 2 corpora**,
   densities 0.048–0.289. `retryesc_gen/floor_predictor_test.py`.
   **My class-balancing hypothesis was REFUTED** (`manifest_f_test.py`:
   the row walk moves `f` 0.1853 → 0.1875, 2.8% of the gap).
2. **clew sweep** — 8 concept-level queries + `cited-by` on 3 nodes, all
   listed in LOG for reproducibility. Citation graph surfaced *Regime
   Leakage* and *Why Safety Probes Catch Liars But Miss Fanatics*,
   unreachable by keyword. Registry's safety states are
   **cumulative/regime**, not recency. *Alignment Faking* `cited-by`
   returned `corpus_n 0` over `fetched_n 0` = **UNMEASURED**, not
   evidence of absence.
3. **Cumulative face: arm at CHANCE** (`facecmp/arm_test.py`).
   `rate_H512` gain **+0.0024**; the foreign-context null *beat* the real
   arm. **Positive control on the identical local pipeline reproduced the
   pod** (+0.0596 vs pod's +0.0645), so the negative is about the LABEL,
   not instrumentation. **Refuted my own bar-first reasoning**: both bars
   were lower and it bought nothing, because the arm fell further than
   the bar did.
4. **Face battery, 6 faces** (`facecmp/face_battery.py`): **gain vs
   floor_excess Pearson +0.871 / Spearman +0.886; 0/6 beat their floor.**
   `ewma_tau128` **clears gain (+0.0552)** — the graded accumulator DOES
   read, so `rate_H512` failed on discretisation+horizon, not on
   cumulative structure.
5. **⚑ TWO STACKED CEILINGS** (`facecmp/ceiling_test.py`,
   `facecmp/run_seq512.py`):
   - **Apparatus:** chunks are independent 128-token sequences. Recency
     terciles sit at ages **121/286**, so 2 of 3 classes are "no event in
     my context" — per_class `[0.516, 0.394, 0.403]`. Re-tercileing
     INSIDE the context (ages 46/72) nearly triples gain
     (+0.0596 → **+0.1707**) and makes per_class uniform
     `[0.845, 0.873, 0.839]`.
   - **Representational:** rebuilding the cache at **SEQ_LEN=512** moved
     recency gain only +0.0596 → **+0.0928** (per_class
     `[0.492, 0.433, 0.416]`), and `rate_H512` stayed at chance *with its
     full horizon present*. **The floor held as pre-registered**
     (0.5859 → 0.5932 — it depends only on `T + w`).
   - **I over-inferred "raise SEQ_LEN and the arm unlocks"; the 512 run
     corrected me within the hour.** It buys ~+0.03, not +0.11.

## ⚠ The open caveat, and what is running

**Everything in (5) is gpt2 ONLY — the weakest leg.** `gemma2_2b` /
`llama31_8b` screen at layer 14 and are plausibly far better at
long-range integration. **Not extrapolating.**

**DONE (20:21) — scale step 124M → 355M, `facecmp/scale_test.py` +
`scale_test_seq512.py`, results in `facecmp/results/scale/`.
**A 3× scale step does NOT extend the readable horizon** — gpt2-medium
is slightly *worse* than small at **both** context lengths (@128 +0.0389
vs +0.0596; @512 +0.0754 vs +0.0928). **My prediction was wrong.** The
within-context face reproduces to ~0.02 per-class across two models and
two context lengths, and **the floor never moved** (0.5859/0.5932) — the
`T + w` prediction has now held across 2 models, 2 context lengths and 2
corpora. ⚠ **Disclosed design error:** the @128 arm of that test could
not answer the representational question at all (out-of-context classes
are absent from the input for *any* model); caught on reading the output
and re-run at 512.

## ⚑ THE SYNTHESIS — hand this over first (revised 20:43)

**Full matrix, as-screened recency face, `facecmp/results/scale/`:**

| model | @128 gain | @512 gain | class-2 per_class @128→@512 |
|---|---|---|---|
| gpt2-small 124M | +0.0596 | +0.0928 | 0.403 → 0.416 |
| gpt2-medium 355M | +0.0389 | +0.0754 | 0.368 → 0.431 |
| **gemma2_2b 2.6B** | +0.0567 | **+0.1088** | **0.399 → 0.493** |

**CONTEXT AND SCALE MULTIPLY.** At 128 tokens a 21× model buys nothing;
at 512 it buys the best result on record. **gpt2 barely uses 4× more
context; gemma2 uses it substantially** — a capability difference the
apparatus was hiding completely. My earlier "readable horizon ≈ 100
tokens, same as the floor's" was **measured on gpt2 and does NOT
generalise**: the readable horizon is a **model × context product**, not
a constant.

**The floor has never moved — 4 model×context combos** (gpt2
0.5859/0.5932, gemma2 0.6048/0.6081). Pre-registered every time. It
depends only on `T + w` and is **purely ours to set.**

**Three levers; the evidence says all three are needed:**
1. **Context** `SEQ_LEN` 128 → 512 (alone: +0.0596 → +0.0928)
2. **Model** 124M → 2.6B (**alone: nothing**; with (1): → **+0.1088**)
3. **Shrink the floor** `T + w` — **UNTESTED, and now the binding one**

**Every candidate screened so far violated (1) and (3).**

**Still not a KEEP:** `arm − floor` = **−0.117**, up from −0.178, **with
zero corpus change.** Closing the rest is lever 3: `w` 25 → narrow
events, `T` 64 → 16, label range inside the readable band.
`retryesc_gen` was T=64, w=25, terciles 121/286 — **wrong on all three.**

**NOT frozen into a card**: lever 3 is untested and the last two
bar-side designs did not carry the arm. **Cheap next step: re-label an
existing corpus at T=16 and measure `arm − floor` before generating
anything.**

⚠ **Process lesson:** I tested two variables one at a time, got a null
from each, and twice wrote a conclusion off a single-variable null.
**Both were interaction effects.** When a factor is pinned by another
factor's ceiling, varying it alone measures the ceiling, not the factor.

## Next actions, in order

1. **Read `scratchpad/scale_test.log`**, copy artifacts into
   `facecmp/results/`, LOG the scaling result whichever way it falls.
2. **The 2b/8b ceiling test decides whether ceiling 2 is a gpt2 fact or a
   program fact.** Needs a weights download or a pod. **Highest-value
   next experiment.**
3. **Do NOT freeze another generation card until (2) resolves.** If the
   readable horizon is short on every leg, the fix is apparatus/model
   geometry, not another task from the registry — and a $21 corpus aimed
   into a factor-1.4 band is wasted money.
4. `briefings/hunt-safety-gold-clew.md` **stays active** — the sweep is
   not exhausted. Untested leads: belief-drift under accumulating
   context, deception-maintenance/commitment, steering-injection age.

## Standing constraints (unchanged)

- **RunPod (Dmitry's key, BINDING):** keychain `dmitrys-runpod-api-key`,
  **mac agents only**, never seeded to a pod, env-inject only, never
  echoed/filed/argv'd. **$10/hr max per agent**; **terminate the moment
  it is unused, prefer TERMINATE, verify by API after**; **never modify
  pods you did not spin up**; name `mac-c-<purpose>-<mmdd>`; ledger at
  spin-up AND termination.
- **Generation** = shared `dmitry-mats-claude-api-key` ($300 shared cap,
  GENERATION ledger), mac-only, never on pods.
- **clew READ-ONLY**: never `sync`/`register`/`seed add`/`clip`, **never
  `--refresh`**. S2: env-only, **never argv**, 1 req/s CUMULATIVE.
- **Never `set -x` in a script that touches a secret** (my own rule after
  leaking a GitHub token into a pod log; contained + verified).
- **Attempt caps are per face-family.** `retryesc` closed 2/2. The
  cumulative family is at attempt 1 and I would call it spent given (3).
- **Re-arm the listener after every wake** —
  `zsh <scratchpad>/listener.sh` as a background task. It fires on my own
  pushes; expected, just re-arm.
- **Stamp from `date` at write time.** Three stamps corrected today.

## Error tally (mine, today) — the pattern matters more than the count

"capped near 1/3" → `K=0.63` → uniform-position gap map → the
class-balancing hypothesis → "raise SEQ_LEN and the arm unlocks".
**Every one was a model checked against itself rather than against the
instrument that sets the bar.** Rules earned, in order: *measure, don't
model* → *check instrument and bar use the same ROWS* → **check they use
the same EVENTS** → **and check the model was actually SHOWN the thing
you are asking it to encode.**

_Recorded-by: claude-opus-5 (mac-c)_
