---
author: Dmitry
date: 2026-07-23
tags:
  - design
  - in-progress
---

## Semi-synthetic language tasks: engineered NL analogues of the clock and denoising bench

Goal: natural-language tasks where (a) theory says the target requires integrating
a **tunable number k of token positions**, and (b) steering the behavior through a
temporal crosscoder beats steering through a per-token SAE, with the gap **growing
in k**. Anchors: [[window_length_theory]] (the two mechanisms), the polynomial
clock (`src/v6_colored_sources/polynomial_clock.py`), the denoising bench, and the
temporal screen ([[temporal_benchmark_screen]], `experiments/temporal_screen/`).

The central discipline of this note is separating two claims that the clock and
denoising results conflate:

- **Decodability** — a window of activations carries information about the target
  that no single token does. This is `H_info` in the screen; the clock/denoising
  results are *about this*.
- **Steerability** — writing to the window changes the model's behavior *more* than
  writing to a token does. This is a *strictly stronger, different* claim, and it
  is the one the paper's backtracking result actually makes.

**A task can have decodability headroom and zero steering headroom.** Getting this
wrong is how we would waste a pod.

### When does windowed steering beat per-token steering? (the mechanism)

Model the behavior as driven by an aggregate over `k` contributing positions,
`A = Σ_{t∈S} φ(x_t)`, that the model forms by attention-pooling upstream of the
decision. Steering must move `A` by some `δ*`. The key constraint is the
**coherence budget**: each position can be perturbed by at most `~m` before it goes
off-distribution and the judge stops counting the output as a *genuine* event
(exactly the Δgc "genuine backtracking" filter). So each position can contribute at
most `~c·m` to `A`.

- **Per-token, single position:** moves `A` by `≤ c·m` — one term.
- **Per-token, broadcast** (same direction written at all k slots): moves `A` by
  `k·c·m` — but only if the *same* write is correct at every position.
- **TXC (per-position decoder rows):** moves `A` by `Σ_t c·m = k·c·m`, with a
  *different* write per position.

This yields a clean taxonomy of `k`-dependence, and it is the whole story:

| task class | decision | single-pos | broadcast | TXC | H_steer vs k |
|---|---|---|---|---|---|
| position-symmetric aggregate, **threshold** decision (parity, "≥k times") | flips by moving one term across the boundary | **suffices** | suffices | suffices | **≈ 0** (caution) |
| position-symmetric aggregate, **value/magnitude** target (set a sum) | one term absorbs the correction (additive) | ~suffices | suffices | suffices | **≈ 0** (caution) |
| **position-DEPENDENT template** (distinct content per slot: passphrase, ordered ramp, clock `G_β`) | needs the whole pattern | covers 1/k | writes wrong content at k−1 slots | writes the template | **> 0, GROWING** |

So the demonstration we want lives in exactly one regime: **position-dependent
templates**, where the required write is different at each slot, so a single-position
write covers `1/k` of it and a broadcast write is *wrong* at `k−1` slots — only the
TXC's per-position decoder pattern reproduces it. This is precisely the clock's
`G_β` atom (a `(h+1, d)` template with different content per slot) rendered for
steering. The `1/k` decay of the single-position arm is the growing gap.

Corollary (the trap): the decodability screen cannot pick the right task, because
parity is `strong_txc`-**decodable** (nonlinear, needs all k) yet steering-null. A
separate steering-headroom test is mandatory, and task selection must target the
position-dependent-template class *a priori*.

### The tasks (spectrum, with honest verdicts)

Datasets and a deterministic generator are built:
`experiments/temporal_screen/semisynthetic_data/` (`generate.py`, tunable k,
fixed seeds). Each example carries `slot_chars` (exact char offsets of the k slot
tokens, so the screen adapter never guesses positions) and a judge-checkable
`behavior` spec.

**1. Parity switch-panel — decodability control / steering NULL.**
`Relay 00: UP / Relay 01: DOWN / ... / Decision:`; label = (# UP is even). Any k−1
relays leave the parity uniform (identifiability threshold, the boolean one-time
pad — the clock's core property). Screen prediction: `strong_txc` (parity is the
canonical nonlinear, order-free, needs-all-k function) — *if the ceiling probe can
compute XOR-of-k* (see the empirical caveat below). **Steering verdict: NULL** —
flipping one relay flips the parity, so single-position steering ties TXC.
Controllable: k. Role: the clean proof that decodability ≠ steerability.

**2. Varbind (hidden start + additive events) — decodability-strong, steering-weak.**
`Step 00: add 4 / ... / Final reading: 3 / was the start 0? Answer:`; label needs
`sum(events) mod q`. Single event uninformative (need the full sum). **Steering
verdict: weak** — additive aggregate, one event absorbs the correction. Controllable:
k, q. Role: the "value-target" caution row.

**3. Escalation build-up — STEERING candidate (position-dependent, natural).**
A minutes-document whose last k sentences either escalate (an intensity-ordered
ramp) or are neutral filler; label = escalation present. The mode is a *trajectory*,
so inducing it needs a per-position write; broadcast can only write a constant
"tense" state, not a ramp. **Steering verdict: positive, IF** the model represents
escalation as a position-dependent ramp. **Weakest point:** a real LLM may collapse
the ramp to a roughly constant "tense" mode, in which case broadcast ties (it
degrades to `aggregation_shapeB`). Mitigation: engineer modes that *cannot* be
constant — a **reversal/turn** ("calm → alarm → forced-calm") only coheres as an
ordered sequence.

**4. Passphrase (distributed multi-slot template) — STEERING headline.**
`Word 00: ZEBRA / Word 01: MAPLE / ... / Status:`; k **distinct** code-words; label =
valid vs one-word-corrupted (which slot is corrupted is random, so any single slot
is ~uninformative for large k). This is the clock's `G_β` rendered for steering:
each slot carries different content, so inducing "the passphrase is present /
authenticated" requires the per-position template. Single-position sets one word;
broadcast sets the same word k times (wrong); only the TXC decoder pattern writes
the whole passphrase. **Steering verdict: positive, and unambiguously
position-dependent** (distinct tokens, not a subtle ramp). **Weakest point:** the
model may compute an "authenticated?" summary at a *downstream* position that a
per-token write there could flip — defused by the matched-hookpoint rule (compare
both dictionaries at the *upstream* slot hookpoint, where the representation is
genuinely the k distinct words).

**5. Stance/topic-under-noise — denoising analogue, aggregation not steering.**
A document that is on-topic with persistence ρ and off-topic (noise) sentence
fraction; label = dominant stance. Decodability rises as ρ falls / noise rises
(evidence integration). **Steering verdict: NULL over broadcast** — the stance is a
position-*constant* state, so broadcasting the stance direction ties the TXC. Role:
the `aggregation_shapeB` / Shape-B caution (decodability but not steerability),
the direct denoising-bench render.

### Empirical check done locally (ground-truth encoding, no LLM)

Screening parity's exact instances under a clean slot encoding (each slot's UP/DOWN
bit along a fixed direction + noise; `.venv/bin/python`, no GPU):

| k | archetype | H_info | conv_gap | R1_mlp | R5 (ceiling) |
|---|---|---|---|---|---|
| 2 | strong_txc | +0.06 | +0.09 | 0.51 | 0.58 |
| 4 | per_token | −0.01 | +0.01 | 0.51 | 0.51 |
| 6 | per_token | −0.04 | −0.00 | 0.52 | 0.51 |
| 8 | per_token | −0.02 | +0.01 | 0.50 | 0.51 |

Honest finding: single-slot is at chance at every k (the identifiability threshold
holds), but the screen's **ceiling MLP cannot learn XOR-of-k for k≥4**, so `R5`
collapses to chance and the screen reports `per_token` — a *probe-limited* ceiling,
not absence of information. Two lessons: (i) parity is a treacherous screen subject
(strengthen `R5` — deeper MLP — or expect under-reporting); (ii) this *reinforces*
that parity is a caution, not a demonstration. It also flags a general point for the
screen: on real activations, if the model computes running parity internally the
signal collapses to the last slot (`per_token`) — the same reason its steering is
single-position.

### Minimal steering experiment (the real test; needs a pod)

Mirror the backtracking Δgc protocol, on **passphrase** (headline) with the k-sweep:

- Model: GPT-2 (fast) or Llama-3.1-8B / Qwen-2.5-7B. Hookpoint: one residual layer
  L, chosen *upstream* of where the model summarizes validity (the matched-hookpoint
  rule). Train a TXC and a per-token TopK SAE on a cache of L-activations over the
  passphrase corpus, **matched d_sae and window-L0** (identical to
  `experiments/ward_backtracking_txc`).
- Mine the target feature by D+/D− selectivity (D+ = valid passphrase, D− =
  corrupted). Its TXC decoder rows `W_dec[:, t, :]` are the position-dependent
  template; the SAE's decoder direction is a single vector.
- Three steering arms at layer L over a symmetric magnitude grid `[−M..M]`:
  (a) **TXC-template** — add `m·W_dec[:, t, :]` at each slot t;
  (b) **SAE-single** — add `m·d_SAE` at the single best slot;
  (c) **SAE-broadcast** — add `m·d_SAE` at all k slots.
- Judge (Sonnet) counts **genuine events**: coherent GRANTED/authenticated behavior
  (not garbage). Metric `Δ = genuine-event lift over unsteered` at each arm's optimal
  magnitude; `H_steer = Δ(TXC) − max(Δ(SAE-single), Δ(SAE-broadcast))`.
- Sweep k ∈ {2,4,6,8}. **Predictions:** `Δ(SAE-single) ∝ 1/k` (covers one slot),
  `Δ(SAE-broadcast)` low and falling (wrong content at k−1 slots), `Δ(TXC)` ≈ flat →
  `H_steer > 0` and growing in k. This is the "steerability improves with required
  context" figure, the causal analogue of the paper's window-length claim.

### What is built vs what needs a pod

- BUILT (CPU, local): generators + 4 tasks × k-sweep datasets under
  `semisynthetic_data/` (parity, varbind, escalation, passphrase); tests
  (`tests/test_semisynthetic.py`, 5 passing); the ground-truth parity screen above.
- BUILT (scaffold): `adapters/semisynthetic.py` (`windows()` captures slot-position
  activations, model-agnostic via `output_hidden_states`) and
  `run_semisynthetic_screen.py` (k-sweep → H_info-vs-k curve, with `--from-cache`).
- NEEDS A POD: the actual `windows()` model forward (GPT-2 runs on CPU for a smoke),
  the dictionary training, and the three-arm steering + judge (not yet coded — the
  spec above is the build order).

### Empirical steering test (executed on Modal) — the language demonstration does NOT hold

We ran the 3-arm steering experiment on Qwen-2.5-1.5B-Instruct (layer 14,
difference-of-means directions as a training-free TXC/SAE-decoder proxy, Δlogprob of
the ordered target, k-sweep). Code: `experiments/temporal_screen/passphrase_steering/ordered_steer_modal.py`;
results: `results/temporal_screen/ordered_steer_{days,numbers}.json`. The relevant
comparison is **txc_template vs sae_broadcast**: a per-token SAE feature is a *single*
decoder direction, and the standard steering recipe adds it at *every* window position
— that is the broadcast arm, not the single-position arm.

Peak Δlogprob(ordered target) per arm:

| task, k | template (TXC) | broadcast (per-tok SAE) | template / broadcast |
|---|---|---|---|
| days 2 | 1.09 | 0.91 | 1.20 |
| days 3 | 2.39 | 1.10 | 2.17 |
| days 4 | 2.01 | 1.16 | 1.73 |
| days 5 | 0.97 | 1.06 | 0.92 |
| days 6 | 0.12 | 1.23 | 0.10 |
| days 7 | 0.29 | 0.96 | 0.31 |
| numbers 2 | 0.30 | 0.08 | 3.99 |
| numbers 3 | 0.12 | 0.98 | 0.13 |
| numbers 4 | 0.14 | 1.80 | 0.08 |
| numbers 5–8 | ~0.02 | ~1.0 | ~0.03 |

**Finding.** The template beats broadcast only at very short sequences (k=2 in both
item sets; k≤4 for days, which did *not* replicate on numbers). At k≥3–5 the per-token
SAE **broadcast matches or beats the TXC template**, by 10–50× at large k — and the gap
runs the *wrong* way (template fades with k, does not grow). The predicted
"steerability improves with required context" does **not** hold in language.

**Why (mechanism, now empirical).** Natural-language structured generation is driven by
a strong shared *mode* — a "counting" / "listing weekdays" contextual state — that a
per-token SAE broadcast reinforces at every position. The per-position template's
specific writes are fragile and are overwhelmed as the sequence (and the mode)
strengthen. The template advantage is a property of *mode-free per-position binding*
(the clock's `G_β`, each slot an independent function of a hidden latent with no shared
mode to ride), which resists rendering as a language *behavior*. This confirms and
generalizes the constant-mode caveat: distinct-content-per-slot and repetition-
incoherence are still not sufficient — you also need *no broadcastable mode*, which
language behaviors almost always have.

Caveats (before calling it fully settled): one small model (1.5B), DoM directions
rather than trained TXC/SAE decoders, one layer, a logprob metric not a generation
judge. But the template *works at k=2* (so the direction is fine) and the negative
replicates across two item sets and worsens with k — the mode-dominance is a property
of the behavior, unlikely to be rescued by a better dictionary.

### Signs of life: trajectory-steering tasks (all four pass)

The first-principles fix for the mode-dominance failure: steer a *time-course* of a
latent (a profile per segment), with a **multiset-matched foil** (a permutation of the
same profile) so no bag/mode statistic — hence no broadcast write — separates target
from foil in principle. Four renderings were tested on Modal (Qwen-2.5-1.5B, L14, DoM
directions, teacher-forced diff-in-diff margin Δ = [lp(T)−lp(F)]steered −
[lp(T)−lp(F)]base; `experiments/temporal_screen/trajectory_steering/sol_modal.py`,
`results/temporal_screen/trajectory_sol.json`):

| task | k | template | broadcast | single | note |
|---|---|---|---|---|---|
| lang_profile (EN/FR per random balanced profile) | 6 | **+63.2** | +6.0 | +9.3 | threshold-y; still rising at frac 0.2 |
| int_profile (calm/tense per random balanced profile) | 6 | **+12.5** | +0.9 | +0.8 | monotone from frac 0.01 |
| mirror (1-2-3-2-1 vs 3-2-1-2-3) | 5 | **+7.8** | −3.3 | +4.9 | broadcast actively hurts; single strong (dominant position at small k) |
| alt_phase (tense/calm alternation, phase A vs B) | 6 | **+21.6** | −9.3 | +1.5 | cleanest dose–response; the clock at ω=π |

All four predictions registered in advance held: template ≫ broadcast everywhere
(broadcast ~0 or *negative* — on matched multisets the DC write can only break
symmetry against you), single small (except mirror at k=5, where one position carries
most of the contrast — expect 1/k decay in the sweep), baseline margins ≈ 0 (no schema
preference; mirror shows no innate rise-fall bias at this scale). The `cos(t_dir,
u_dc)` diagnostics confirm the scheduled-knob regime: per-position directions are
≈ ±(one attribute direction) with signs following the profile — so the claim to build
is *trajectory control vs level control* (per-token steering is a DC actuator;
windowed steering writes a waveform), not direction diversity.

### Full k-sweep: the language counterpart of the clock (POSITIVE)

Full run (`experiments/temporal_screen/trajectory_steering/full_modal.py`,
`results/temporal_screen/trajectory_full.json`, figure
`plots/2026-07-24_trajectory_steering/ksweep_dmargin.png`): k ∈ {2,4,6,8,10}, n=32
eval pairs per k with SEM, frac grid to 0.5 (template plateaus by 0.35–0.5). Peak
Δmargin (teacher-forced, multiset-matched foil):

| k | lang: template | lang: single | lang: broadcast | alt: template | alt: single | alt: broadcast |
|---|---|---|---|---|---|---|
| 2 | +75.7±3.6 | +29.6 | −0.2 | +21.0±1.1 | +4.9 | −0.6 |
| 4 | +90.1±7.3 | +19.3 | −0.0 | +38.5±2.0 | +3.7 | −0.8 |
| 6 | +141.6±10.8 | +26.1 | +14.0 | +53.4±2.1 | +5.2 | −0.9 |
| 8 | +169.8±12.2 | +20.3 | +10.1 | +65.6±2.6 | +4.6 | −1.0 |
| 10 | +218.9±14.6 | +16.9 | +6.0 | +80.5±2.1 | +8.3 | −0.5 |

All three registered predictions hold: **template grows ~linearly in k** (constant
per-slot effect — the windowed handle doesn't degrade with trajectory length);
**broadcast is pinned at ~0** (alt: slightly negative at every k — on a matched
multiset the DC write can only break symmetry against you); **single is flat** (its
share of the template shrinks as 1/k: lang 39%→8%, alt 24%→10%). This is the
"steerability grows with required context" claim, delivered in language — where the
same arms on ordered-days/numbers (mode-dominated tasks) gave the *opposite* ranking.
The dissociation is the result: windowed steering wins exactly when the target is a
trajectory with no DC component, and per-token steering wins when a broadcastable
mode carries the behavior.

Generation-mode demo (lang_profile k=6, langid-per-sentence judge, free generation
from a 2-shot bilingual prefix with temperature sampling; `gen2_modal.py`,
`results/temporal_screen/trajectory_gen2.json`; v1's greedy-from-bare-carrier
plumbing failure documented in `full_modal.py`): per-slot accuracy of generated
sentence language vs the intended random profile (chance = 0.5, n=24):

| arm | frac 0.35 | frac 0.5 |
|---|---|---|
| **template** | **0.812 ± 0.044** | 0.743 |
| broadcast | 0.444 ± 0.028 | 0.514 |
| single | 0.535 | 0.507 |

The behavioral version of the claim stands: a per-segment schedule steers *which
language each generated sentence comes out in* at 81% per-slot accuracy, while the
same direction broadcast (the per-token-SAE recipe) sits at chance — it makes text
Frenchier everywhere, which is exactly wrong on half the slots. Caveats: 1.5B model
and DoM directions, so the steered text is code-mixed rather than fluent bilingual
prose (samples in the JSON); the language *identity* per slot tracks the profile,
which is the claim being made. Next hardening steps if this goes in the paper:
bigger model, trained TXC/SAE dictionaries in place of DoM proxies, second attribute
(intensity) in generation mode, and layer/model robustness.

- **Do not pursue a language *steering* demonstration of template>per-token as a paper
  headline.** Both candidate classes fail, for complementary reasons that *sharpen the
  taxonomy*: passphrase-verification is a conjunction (a single/broadcast write
  satisfies it); ordered-generation is mode-dominated (broadcast reinforces the mode).
  The clean template>broadcast steering win appears to require mode-free per-position
  binding — the synthetic clock, not a language behavior.
- **Where the TXC's language edge is real: decodability, not steering.** The screen's
  `H_info` (a window carries what no token does) is the defensible language claim, and
  the paper's existing detection results (backtracking PR-AUC) live here. Frame the
  temporal-crosscoder advantage on real models as *detection / decodability*, and keep
  the *steering* superiority claim on the synthetic clock where it provably holds.
- **The controls are now results.** Passphrase (conjunction) + ordered (mode) + the
  clock (mode-free) together are a *dissociation* answering "is it the crosscoder
  specifically?" — windowed steering wins only in the mode-free regime; a per-token SAE
  broadcast is a strong, often-superior baseline everywhere else. That honesty serves
  the rebuttal better than an oversold win.
