# mac-c — STATUS

**Agent:** mac-c (macOS, `~/research/projects/agents/mac-c/temp_xc`)
**Lane:** ⚑ **THE WHOLE TASK HUNT, END TO END** (`hunt-mac-c-takeover.md`),
re-aimed by `briefings/hunt-safety-gold-clew.md` (active, owner mac-c)
**Last update:** 2026-07-29 00:32 BST (stamped from `date` at write time)

---

## ⚑ 22:52 — LATEST: checkpoint durability audit (unassigned, $0, PUSHED `eb9f3fb47`)

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

## 1. ⏳ IN FLIGHT RIGHT NOW — a job is running, do not restart it

**Local MPS cache of `evalage` activations, gemma2_2b layer 14,
SEQ_LEN=512.** Task `bxfp10te4`, log
`<scratch>/cache_evalage_512.log`, at 2016/3579 rows @00:32, **ETA
~5min**, writing `<scratch>/cache_evalage_512/gemma2_2b/hs14.npy`
(~9GB) + `tokens.npz` + `acts_meta.json`. Script:
`<scratch>/cache_evalage_512.py`. **$0, no pod.**

**THE MOMENT IT FINISHES, run this — it is the whole point:**

    cd $(git rev-parse --show-toplevel)
    FACECMP_CACHE_ROOT=<scratch>/cache_evalage_512 \
    PYTHONPATH=. .venv/bin/python -m \
      experiments.explorations.task_hunt.facecmp.lever3_evalage

Check the log tail says `done <N> rows` and the degenerate-assert passed
before trusting it.

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

## 3. Delivered tonight (all $0, all ratified — do not redo)

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
