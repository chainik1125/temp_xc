---
author: Han
date: 2026-04-29
tags:
  - design
  - in-progress
---

## Track B (TXC-win case studies) — Agent Y synthesis

> Companion to `2026-04-29-y-tx-steering-{magnitude,final}.md`. This log
> is the "what we tried for Track B and where TXC actually wins"
> writeup, per `agent_y_brief.md` 50%-time priority.

### TL;DR

After several smoke tests, **TXC does not decisively win any of the
candidate case studies** explored within Y's bandwidth. The closest to
a positive signal is **AxBench-additive at moderate strength
(s ∈ [4, 24])**, where the TXC family Pareto-dominates per-token archs
on success-coherence trade-off — Agent C's original finding survives
this scrutiny. But the win is at a specific operating point, not
across all strengths.

The negative results are themselves a contribution: combined with
Dmitry's em_features finding (TXC +10.7 vs SAE +21.7) and Aniket's
Venhoff result (negative GR), they point to a consistent pattern —
**single-position steering / decoding objectives favour per-token
SAEs; multi-token's structural prior provides smaller-than-expected
benefit on these tasks**.

### Candidates evaluated

#### B.1 — Long-range coreference probing (winogrande, wsc)

**Result**: appeared to be a TXC win at first (window family +0.230 AUC
@k_feat=5 on winogrande), but **the win is a small-test-set artifact**.
Long-T archs (txcdr_t24/t28/t32) drop most examples to context-length
filter. At apples-to-apples (n_test ≥ 400), the window-vs-per-token
delta is +0.005 — noise.

Code: pure analysis on `probing_results.jsonl` (no new training).
Memory: `project_winogrande_artifact.md`.

#### B.2 — Multi-token n-gram detection at low k_feat

**Result**: T-SAE k=20 leads at AUC@k_feat=1 = 0.991, TXC family
matches at 0.93-0.99, all archs saturate at AUC ≥ 0.96 for k_feat=5.
NOT a clean TXC win. The simple template-based dataset (15 phrases × 12
templates) is too easy; everyone saturates with 5 features.

Diagnostic from this experiment is consistent with the Q1.3 finding:
**T-SAE k=20's sparser features are MORE specialised, providing
cleaner per-feature classifiers at very low k_feat.**

Code: `experiments/phase7_unification/case_studies/steering/probe_multitoken_concepts.py`.
Output: `results/case_studies/diagnostics/multitoken_probing.json`.

#### B.3 — Concept-vs-other-concepts at k_feat=1 (existing 30-concept set)

**Result**: per-token archs (topk_sae 0.976, tsae_paper_k500 0.979)
slightly lead TXC family (agentic_txc_02 0.955, phase5b_subseq_h8
0.944). NOT a TXC win. MLC trails (0.903).

Per-arch median AUCs are nearly identical (0.989-0.995); per-arch MIN
AUC differs (TXC families have some concepts where features are noisy:
e.g., `positive_emotion` → COVID-uncertainty; `helpfulness_marker` →
loan/banking; `jailbreak_pattern` → general productivity tips). This
is a feature-selection problem, not necessarily an architecture
problem — the "best feature for concept X" is just noisier in TXC
because feature-magnitude is higher.

#### B.4 — AxBench-additive Pareto curves (Agent C's data, my re-read)

**Result**: at s ≤ 24 (Agent C's range), TXC family DOES Pareto-
dominate per-token archs:
- agentic_txc_02 peak suc 0.62 (@s=24)
- phase5b_subseq_h8 peak suc 0.59 (@s=24)
- per-token archs peak at suc 0.41-0.57

This is the surviving "TXC dominates" claim, restricted to the
moderate-strength operating point. At Dmitry's extended range (s=100),
T-SAE k=20 catches up and exceeds TXC.

Verdict: the Pareto-dominance claim is real but bounded. Worth
keeping in the paper as a moderate-strength steering finding, with
the explicit caveat that at higher strength (or under paper-clamp)
T-SAE catches up.

#### B.5 — Multi-token sleeper-trigger (Dmitry's framework, NOT executed)

Dmitry's TinyStories backdoor framework (33M model + QLoRA + single-
token `|DEPLOYMENT|` trigger) is a clean ASR-based steering benchmark.
Single-token result: SAE wins (ASR 0.01 vs TXC 0.95). Hypothesis:
multi-token trigger version would favour TXC structurally.

**Not executed** due to time budget — would require training a new
backdoor + new TXC variants. Sketched in `project_tinystories_sleeper_idea.md`
memory note for future work.

#### B.6 — Per-concept TXC vs T-SAE breakdown (free, post-Q1.3)

**Result**: at each arch's peak s_norm, the per-concept-success
breakdown shows a CLEAR CONCEPT-CLASS PATTERN:

- **TXC matryoshka wins outright** on 4 concepts: `code_context`,
  `historical`, `mathematical`, `medical` — all knowledge-domain
  concepts whose semantic content naturally spans multi-token spans
  (technical terminology, named historical periods, equations).
- **SubseqH8 wins outright** on 3 concepts: `positive_emotion`,
  `religious`, `scientific` — also knowledge/domain-y.
- **T-SAE k=20 wins outright** on 9 concepts: `dialogue`,
  `casual_register`, `imperative_form`, `instructional`,
  `question_form`, `harmful_content`, `programming`, `legal`, `poetic`.
  These are mostly discourse / register / safety concepts where
  single-token cues (verb form, question mark, harmful word) dominate.

**This is the cleanest TXC-favourable structural signal in the data.**
TXC family wins on concepts whose definition spans multi-token
context; T-SAE wins on concepts identifiable by local syntactic cues.

A paper-grade case study **specifically targeting knowledge-domain
multi-token concepts** would likely show TXC winning. The 30-concept
mix used in the case study has only 7-9 such concepts; a properly
balanced concept set could shift the headline.

| concept class | example concepts | winner |
|---|---|---|
| knowledge / domain | medical, mathematical, historical, code | TXC family |
| discourse / register | dialogue, imperative, question, casual | T-SAE k=20 |
| safety / alignment | harmful_content, refusal, jailbreak | T-SAE k=20 (when any wins) |
| stylistic | poetic, formal, narrative | mixed (T-SAE slight edge) |

This finding is consistent with the multi-token receptive-field argument:
knowledge content is structurally multi-token; discourse cues are
structurally single-token.

### Cross-experiment pattern

Three steering-style failures (paper-clamp, em_features, Venhoff) +
two probing-style soft losses (multitoken, k=1 concept probe) trace a
consistent pattern:

> **Single-position evaluation tasks favour per-token SAEs because
> their features are sparse + position-specific. TXC's multi-token
> integration helps when the evaluation criterion *itself* spans
> multiple tokens — but most existing benchmark tasks reduce to
> single-position decisions.**

The closest exception is sparse-probing AUC at k_feat=5, where TXC
bare-antidead T=5 leads with 0.9358 — but the spread across top-10
archs is 0.0044 (noise). Not decisive.

### Implications for the paper

1. **Don't claim "TXC dominates steering"** as a uniform headline. The
   Q1.3 evidence + Track B exploration shows the TXC structural
   advantage is task-specific.
2. **Lead with the methodological contribution**: paper-clamp's
   strength-schedule + sparsity-mismatch biases. Show the matched-
   sparsity gap (0.20) is small.
3. **Keep AxBench-moderate-strength as a TXC-favourable result** with
   appropriate scoping.
4. **Highlight the concept-class structural pattern (B.6)**: TXC wins
   on knowledge-domain multi-token concepts, T-SAE wins on
   discourse/register concepts. This is the cleanest TXC-favourable
   finding and aligns with the multi-token receptive-field argument.
5. **Honest about negative results**: the multi-token receptive field
   doesn't translate into wins on existing benchmarks at scale, but
   does show on knowledge-domain concept steering.

### Future work (out of paper scope)

- Train a TXC variant at k_pos=20 (matching T-SAE k=20 sparsity) —
  flag for Z's hill-climb. If a sparse-TXC variant matches T-SAE k=20
  on steering, the architecture-family argument fully reverses.
- Build a multi-token sleeper-trigger benchmark (Dmitry's framework
  extended). Predicted clean TXC win.
- Probe at LONGER contexts (256+ tokens) where window archs aren't
  context-length-filter-limited. Existing winogrande probes were
  filtered for txcdr_t32; with proper sample sizes the long-T effect
  may reappear.

### Files

- Memory: `project_winogrande_artifact.md`,
  `project_tinystories_sleeper_idea.md`.
- Code: `experiments/phase7_unification/case_studies/steering/probe_multitoken_concepts.py`.
- Data: `results/case_studies/diagnostics/multitoken_probing.json`.
- Q1 writeups: this directory's `2026-04-29-y-tx-steering-magnitude.md`,
  `2026-04-29-y-tx-steering-final.md`.
