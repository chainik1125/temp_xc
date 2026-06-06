# Briefing — brainstorm: real LM phenomena → one tunable synthetic process

**Audience.** A fresh-context agent picking up a *brainstorming* session (not
an implementation sprint). Read top-to-bottom, then read the four docs in § 1
before proposing anything.

**Mission of the session.** Design synthetic temporal benchmarks that are
**grounded in measurable real LM phenomena** — and pressure-test one ambitious
idea: a **single general generative process with knobs that interpolate between
real phenomena** (backtracking, emergent misalignment, RLHF, …). The output of
the session is a *design* (and a critique of it), not code.

---

## 0. The methodological commitment (this is the point)

**Go from a measured real-world phenomenon → a faithful, *simple* synthetic
analogue.** Not the reverse. The previous round drifted into picking clean
synthetic structures (syntactic nesting / long-range repetition) and hoping
they resembled reality — Han rejected those as **too artificial**. They are
formal-language toys, not *measured* LM behaviors.

Two non-negotiables for any proposal:
1. **It starts from a phenomenon we can measure in real data** (the way
   backtracking started from labeled reasoning traces).
2. **The synthetic analogue is fit/validated against that measurement**, not
   asserted. "Faithful yet simple" — both words matter.

The worked example of doing this right is the backtracking investigation (§ 3).

## 1. Read first (in order)

1. [`../synthetic_benchmark_guidance.md`](../synthetic_benchmark_guidance.md)
   — conventions every synthetic bench obeys (ground-truth `F` vs dynamical
   latents; capacity matched + anchored on `F` + swept into the scarce regime;
   power-of-two windows + tiled eval; **memorization-free** linear probes;
   frontier reporting).
2. [`../autoresearch_spec.md`](../autoresearch_spec.md) — the loop + its prime
   directive (*success = a sound verdict, never a "win"*) and § 3 validity
   gates (shuffle control, labeler-noise, untrained-encoder control,
   memorization budget, realistic regime).
3. [`backtracking_record.md`](backtracking_record.md) +
   [`backtracking_bench_spec.md`](backtracking_bench_spec.md) — the one
   end-to-end example: measured a real property, fit a mirror, spec'd the bench.
4. [`ac_signed_motion_bench.md`](ac_signed_motion_bench.md) — the cautionary
   tale: the confounds that bite (memorization when #distinct-windows is small;
   per-token baselines that aren't actually at chance; latents that are
   interactions vs linear-in-history). And `../../CLAUDE.md` for the hard rules.

## 2. The central idea to develop and stress-test

**Hypothesis: most of the phenomena we care about are the same generative
skeleton with different *hidden-state dynamics*.** A hidden behavioral "mode"
evolves over the sequence and colors the emitted tokens; the phenomena differ
in *how the mode switches and persists*, not in kind.

A candidate **general process** (a seed to attack, not a finished design):

- **Hidden state** `m_t` (a small set of modes, or a continuous latent).
- **State dynamics**, governed by knobs:
  - *persistence* `τ_dwell` — geometric (memoryless) → heavy-tailed (sticky) →
    ∞ (absorbing);
  - *self-excitation* `α` — recent activity raises the rate of (re)entering a
    mode;
  - *trigger coupling* — the mode flips *exogenously* when a specific content
    cue is emitted (a narrow trigger with a broad downstream effect);
  - *drift* — slow continuous movement of the latent.
- **Emission** — tokens drawn from a mode-dependent distribution over the
  feature dictionary; knob = how *broadly/strongly* the mode colors emissions
  (*spread*).

**Why this might interpolate the real phenomena:**

| phenomenon | knob setting (hypothesis) |
|---|---|
| backtracking (measured: self-exciting bursts) | `α` high, short dwell |
| emergent misalignment | **trigger on** + **absorbing dwell** + **broad spread** (narrow cue → persistent, broad mode flip) |
| RLHF / preference / persona | slow `drift`, long dwell (persistent style) |
| existing coupling/denoising benches | memoryless dwell, noisy emission (pure aggregation) |
| sycophancy / refusal | triggered + sticky (mode entered on a cue, then persists) |

If this holds, each real phenomenon is **the same process fit to that
phenomenon's measured temporal signature**, which (a) makes the benchmarks
*comparable* (one family), and (b) lets us *interpolate* between them (e.g.
morph backtracking's self-excitation into EM's absorption by turning knobs).

**Attack it.** The brainstorm should try to *break* this:
- Is "hidden behavioral mode + tunable dynamics" actually the right skeleton,
  or are some phenomena (e.g. RLHF) not mode-switching at all?
- Do the knobs genuinely interpolate, or is it a kitchen sink that fits nothing
  faithfully? (A too-general process is unfalsifiable — the antidote is the
  validation gate below.)
- What is the *minimal* knob set?

## 3. The discipline that keeps it honest

For every candidate phenomenon, the loop is: **measure → fit → validate**
(autoresearch_spec stages 2–5). Faithfulness is not assumed:
- **weak validation:** the fitted process reproduces the *measured* temporal
  statistics (ACF, dwell distribution, self-excitation, burstiness) on held-out
  real data;
- **strong validation (preferred):** a dictionary trained on real vs synthetic
  behaves the same.

A general process earns its generality only if its *fitted instances pass these
gates per phenomenon*. Backtracking passed weak validation; that's the bar.

## 4. Candidate real phenomena (real LM behaviors, with data pointers)

| phenomenon | hypothesized temporal character | measurement handle / data on-branch |
|---|---|---|
| **backtracking** ✓ done | self-exciting, order ≥2 | Ward Stage-A sentence labels (`results/c7_backtracking/stage_a/`) |
| **emergent misalignment** | triggered + sticky/absorbing mode | EM organisms in `configs/data.yaml` (`qwen_2_5_7b_instruct_medical_l15`, `qwen_2_5_14b_instruct_finance_l24`); `evals/em.py` |
| **RLHF / preference** | persistent drift | `evals/rlhf.py` (§ 5.4); HH-RLHF / steering data likely on `origin/final` (`c5_steering`) |
| refusal / safety | triggered + sticky | needs a per-span refusal signal |
| sycophancy | drift toward agreement | needs a per-turn agreement signal |
| topic / discourse | long-memory persistence (heavy dwell) | `fineweb` activations on-branch; embedding-cluster labeler |

**The labeler is the crux** (autoresearch_spec § 2.2). Backtracking was easy
because sentence-level labels existed. EM/RLHF need a *measurable per-token or
per-span signal* (e.g. a misalignment score, a preference signal) — proposing
how to operationalize each is part of the brainstorm, and the labeler-noise
gate must be respected.

## 5. What already exists (don't rebuild)

- **Framework:** the `synthetic` pathway (`run.py synthetic`), tiled
  apples-to-apples metrics at `SyntheticRecovery` protocol **1.2.0**, the
  conventions doc, the autoresearch spec.
- **Generators:** `coupled_hmm`, `markov_chain_support`, `signed_motion` in
  `src/temp_bench/data/synthetic.py` (the existing aggregation + the AC bench).
- **Backtracking:** measurement + mirror (`experiments/autoresearch/`),
  record + bench spec (`docs/autoresearch/`).
- **Lessons banked:** memorization confound (need #distinct-histories ≫ `F`);
  per-token baselines often aren't at chance; linear-in-history latents are
  probe-friendly, interaction latents are not.

## 6. Hard constraints (so the brainstorm stays implementable + sound)

- All CLAUDE.md hard rules: canonical runner, code-version stamping,
  plugin-only, **never edit `temp_bench/core/`**.
- Any resulting bench obeys `synthetic_benchmark_guidance.md` and passes the
  autoresearch_spec § 3 validity gates.
- One `.md` spec per real-task-motivated benchmark (see
  `backtracking_bench_spec.md` for the template).
- Prime directive: the goal is a sound verdict about whether a temporal
  architecture exploits a real phenomenon — **not** a win.

## 7. Deliverables of the brainstorm session

1. A critique of the "hidden behavioral mode + tunable dynamics" hypothesis —
   does it hold; what's the minimal knob set; where does it break.
2. A short list (2–3) of real phenomena that (a) have a plausibly *measurable*
   temporal signature with available data and (b) sit at *distinct* settings of
   the knobs — so the benches cover different dynamics, not the same one thrice.
3. For the top pick (likely **emergent misalignment** — real, on-branch data,
   and a *distinct* dynamic from backtracking), a sketch of the
   measure→fit→validate plan and how it instantiates the general process.
4. A go/no-go on whether the *general process* is worth building as shared
   infrastructure, or whether per-phenomenon generators are cleaner.

Do **not** implement during the brainstorm — produce the design + critique,
then a preregistration for whichever phenomenon is chosen next.

## 8. Git / status

Branch `arxiv`, last commits: `f151fd43` (backtracking bench spec) ←
`24551b28` (backtracking measurement + mirror) ← `cae22100` (backtracking
prereg + Ward labels) ← `71f1bf92` (synthetic-bench conventions + AC redo +
autoresearch spec). All committed; **not pushed** (branch ahead of
`origin/arxiv`). Tree may be dirty during dev — `TEMP_BENCH_ALLOW_DIRTY=1`.
