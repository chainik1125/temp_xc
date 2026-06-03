# Autoresearch spec — temporal structure in language ↔ synthetic mirrors

**Scope.** Rules for an (agent-driven) research loop that (1) discovers
properties of natural language with *genuine temporal structure*, and (2)
builds synthetic generating processes that mirror that structure for use as
architecture benchmarks. The **dual goal** is explicit: finding *which*
properties of language are temporal (and of what kind) is as much the output
as any benchmark.

Read alongside — this doc does **not** restate them:
- [`synthetic_benchmark_guidance.md`](synthetic_benchmark_guidance.md) — how a
  synthetic bench is built/run/scored (capacity, windowing, probes, frontier).
- [`frequencybenchideas.md`](frequencybenchideas.md) — the DC/AC frequency lens.
- [`CLAUDE.md`](../CLAUDE.md) — framework hard rules (canonical runner,
  code-version stamping, plugin-only, never edit `core/`).
- [`ac_signed_motion_bench.md`](ac_signed_motion_bench.md) — a worked example
  where the §3 controls turned a false positive (`s_temp=1.0` from a
  memorizing probe) into the true negative. Treat it as the cautionary case.

---

## 0. Prime directive (read this twice)

> **Success = a sound, reproducible *verdict* — positive or negative.** An
> investigation that concludes "this property is not temporal" or "no
> architecture beats the per-token baseline" is a *complete success* when the
> controls pass. Agents are **never** rewarded for a positive result, and must
> not tune the labeler, statistic, capacity regime, probe, or metric to
> manufacture one.

The loop is open-ended and the failure modes are subtle — we have hit several
by hand (conflating a feature count with a pattern count; a probe that
memorizes a small window set and "generalizes" because train/eval share it;
"wins" that also appear with an *untrained* model). An agent optimizing for
"find a property where the temporal model wins" will reward-hack straight into
these. Everything below exists to make that impossible to do silently.

---

## 1. The pipeline

Six stages, each with a **gate** that must pass before the next. Stage 1 is
frozen (committed) before any data is touched in stages 2–6.

1. **Propose & preregister**
2. **Operationalize on real data** (the labeler)
3. **Measure the temporal signature**  ← *temporal-ness gate*
4. **Fit a synthetic mirror**
5. **Validate the mirror**
6. **Benchmark architectures**  ← *defers to the conventions doc*

---

## 2. Stage rules

### 2.1 Propose & preregister
Before touching data, write down and commit: the property; its hypothesised
temporal character (DC slow-varying / AC order-sensitive / periodic / bursty);
the exact labeler; the statistic(s) to be measured; the chance & oracle
baselines; and the prediction for each architecture *with a reason*. This is
frozen — later stages may not change it (only abort).

### 2.2 Operationalize (the labeler)
- The property must reduce to a **per-token or per-span signal** (categorical
  or scalar) via a **named, version-pinned** labeler — a tagger, lexicon,
  embedding-cluster id, or a specific LM feature.
- The labeler must be **validated**: report held-out accuracy / inter-labeler
  agreement and an estimated **noise floor**.
- **Gate:** label noise must be low enough that the stage-3 temporal signal
  cannot be an artifact of it (quantified by the shuffle control, 2.3).

### 2.3 Measure the temporal signature
- **Toolkit:** autocorrelation vs lag; dwell-time / run-length distribution;
  empirical transition matrix + estimated Markov order; mutual information
  `I(L_t; L_{t+k})` vs `k`; burstiness / Fano factor; spectral density (DC vs
  AC share).
- **Temporal-ness gate (mandatory order-0 control):** compute every statistic
  on the real ordered stream **and** on a phase/block-shuffled stream that
  preserves the marginal but destroys order. The property counts as *temporal*
  only if the ordered statistic departs from the shuffled baseline **beyond
  both** sampling noise **and** the labeler noise floor. **If shuffling does
  not change it, ABORT** — there is nothing temporal to mirror.
- Classify the structure (DC / AC / periodic / bursty) from the signature.

### 2.4 Fit a synthetic mirror
- Pick a process from the **menu (Appendix B)** keyed to the measured
  statistic — do not invent a bespoke generator without justifying it against
  the menu.
- Fit parameters to match the **specific, preregistered** statistic. State
  explicitly which statistic is matched **and which structure is deliberately
  not** (a fit to autocorrelation does not capture higher-order structure).
- Embed as a synthetic benchmark per `synthetic_benchmark_guidance.md` (define
  `F` feature directions, the latent(s), chance/oracle).

### 2.5 Validate the mirror
- **Weak (required):** the synthetic reproduces the matched statistic on
  held-out draws, within a stated tolerance.
- **Strong (preferred):** a dictionary trained on the synthetic mirror behaves
  like one trained on the real signal (similar feature/recovery behavior).
  State which level was reached; never over-claim fidelity beyond it.

### 2.6 Benchmark architectures
Follow `synthetic_benchmark_guidance.md` **exactly**: capacity matched across
archs and anchored on `F`, the **scarce regime (`d_sae ≤ F`) is the object of
study**, a common `L`-tiled eval window, a **memorization-free** linear probe,
and report the **frontier** (never a single hand-picked cell). The §3 controls
are mandatory for any claimed advantage.

---

## 3. Validity gates (mandatory — the anti-off-rail core)

Every investigation must pass **all** applicable gates. A failed gate means
**abort or report-as-negative** — never engineer around it.

**Real-side**
- **Shuffle control** (2.3): ordered statistic exceeds the shuffled baseline
  beyond sampling + label noise.
- **Labeler-noise control**: the temporal effect survives the labeler's
  estimated error.
- **Held-out**: statistics measured on held-out text, not the fit set.

**Synthetic-side** (from the conventions doc + the AC-bench lessons)
- **Ground-truth hygiene**: `F` (feature directions) defined cleanly; never
  conflated with derived pattern/window counts.
- **Memorization budget**: any probe has **fewer features than the number of
  distinct patterns it is tested on**.
- **Untrained-encoder control**: a claimed architectural win must **vanish for
  a randomly-initialized model of the same architecture**. If a random model
  also "wins", it is a probe / architectural-access artifact, not learning.
- **Realistic-regime**: a win must hold at `d_sae ≤ F`, not only in the
  over-complete corner.
- **Capability-vs-artifact**: the winning model must *also* reconstruct /
  recover the features — not "recover the latent" while representing nothing.
- **Provable baselines where possible**: prefer a process with a provable
  chance/oracle (e.g. a data-processing-inequality floor) over an empirical gap.

---

## 4. Required output artifact

Each investigation commits **one structured record** containing:
1. the frozen preregistration (2.1);
2. the labeler, its version, and its validation/noise floor;
3. the measured statistics — **ordered vs shuffled** — with plots;
4. the **temporal-ness verdict** (with the gap and its uncertainty);
5. the mirror: process, fitted parameters, match quality, what was *not* matched;
6. the architecture **frontier** and an explicit list of which §3 controls passed;
7. a one-line headline — which **may be negative**.

No free-form "it seems temporal" claims without the backing statistic and
passed controls.

---

## 5. Abort / discard conditions

Abort (and report the negative) when any of: the shuffle control shows no
temporal structure; the labeler noise floor swamps the effect; a confound is
detected that can't be removed without metric/labeler/regime shopping; the
mirror cannot match the statistic; or the compute budget is hit. **Do not keep
tuning to force a positive.** A clean abort is a valid, citable outcome.

---

## 6. Guardrails — what agents may not do

- Inherit all `CLAUDE.md` hard rules. Never edit `temp_bench/core/`; never
  hand-write the leaderboard; everything through the canonical runner.
- **Properties** come from the menu (Appendix A); a new one requires a written
  proposal *added to the menu*, not ad-hoc invention mid-run.
- **Generators** come from the menu (Appendix B); bespoke processes require a
  written justification against it.
- **No shopping after preregistration** — labeler, statistic, capacity regime,
  probe class, and metric are fixed at stage 1.
- **No claim without its backing statistic and passed §3 controls.**
- Bounded compute per investigation; honor §5 stop conditions.
- Real corpora and labelers are **version-pinned and cited**.

---

## Appendix A — candidate property menu (extensible via proposal)

| property | hypothesised type | candidate labeler / signal | candidate mirror | real data on-branch? |
|---|---|---|---|---|
| topic / topic-switching | DC, sticky (heavy dwell) | sentence-embedding cluster id / LDA | semi-Markov | `fineweb` acts |
| sentiment trajectory | DC, slow drift | sentiment classifier / lexicon | AR(1) / 2-state Markov | needs corpus |
| backtracking / self-correction | **AC, bursty / self-exciting** | Ward event labels | Hawkes / semi-Markov burst | **Ward Llama-3.1-8B ✓** |
| syntactic state / clause depth | order-sensitive | parser depth | stack / branching process | needs parser |
| entity recency / coreference | long-memory | coref chain id | renewal process | needs coref |
| register / formality | DC, very slow | classifier | low-frequency Markov | needs corpus |

*Backtracking is the recommended first slice: labels already exist on-branch,
and it is the order-sensitive (AC) case our prior work has been circling.*

## Appendix B — generating-process menu keyed to statistic

| measured signature | process | matched parameter(s) |
|---|---|---|
| exponential autocorrelation | 2-state Markov / AR(1) | decay ρ |
| heavy-tailed dwell times | semi-Markov | dwell distribution |
| multi-state structure | HMM | transition matrix |
| bursty / clustered events | Hawkes (self-exciting) | base rate + kernel |
| rhythmic / periodic | periodic + noise | period + SNR |
| order-sensitive direction | signed-motion-style | step `v` |

---

## Provenance

Everything runs through the canonical runner; each record stamps the
**code version + labeler version + data version**; statistics are computed on
held-out splits with fixed seeds; the §4 artifact is committed. A result that
cannot be reproduced from its record is not a result.
