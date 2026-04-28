---
author: Han
date: 2026-04-28
tags:
  - design
  - in-progress
---

## Agent Y brief — Phase 7 case studies (continuing from Agent C)

> Read `brief.md` first for paper-wide context (subject models, k_win
> convention, methodology, source-of-truth files). This brief is
> Y-specific: what Y inherits, what's broken, what Y needs to ship.

### TL;DR

- **You inherit Agent C's case-studies workstream**, now on
  `han-phase7-unification` (migrated 2026-04-28). Code under
  `experiments/phase7_unification/case_studies/`; outputs under
  `experiments/phase7_unification/results/case_studies/`.
- 🚨 **URGENT NEW PRIORITY (2026-04-28, post-Dmitry)** 🚨 — collaborator
  Dmitry has shipped a paper-protocol steering reproduction on branch
  `origin/dmitry-rlhf`. **Headline result: under the paper's own
  steering protocol, our best TXC architectures lose by 0.5–0.8
  peak-success to T-SAE k=20.** Agent C's "TXC Pareto-dominates
  steering" claim only holds under AxBench-additive (which Han chose
  pragmatically); it does NOT hold under paper-clamp.
  Read [§ Dmitry's findings](#dmitrys-findings) below. Y's URGENT
  priority is now: understand WHY TXC loses under paper-clamp, and
  find a defensible HOW to steer TXC properly.
- Two case studies remain in scope:
  (1) **HH-RLHF dataset understanding** — what features does each arch
      find? (Agent C's pass is reasonable; mostly stable.)
  (2) **AxBench-style steering** + **paper-clamp steering** — Dmitry
      has done the comparison; Y extends from there.
- Pod: **A40, 46 GB VRAM, 46 GB pod RAM, 900 GB**. Same hardware as
  Agent X. The case studies don't train SAEs from scratch — load
  existing ckpts + run model forward passes / steering — so memory
  pressure is mild.

### Inheritance from Agent C

#### Branch state

The case-studies infrastructure was migrated from
`origin/han-phase7-agent-c` onto `han-phase7-unification` in commit
`445dd7d` (2026-04-28). The `han-phase7-agent-c` branch remains as a
historical reference only — Y does NOT branch off it.

What's now on the unification branch under your ownership:

| path | what's there |
|---|---|
| `experiments/phase7_unification/case_studies/{hh_rlhf,steering}/*.py` | the pipelines: cache build + decompose + label + summarise (HH-RLHF); concepts + select + intervene + grade + plot (steering) |
| `experiments/phase7_unification/results/case_studies/hh_rlhf/<arch>/` | per-arch top features + feature stats |
| `experiments/phase7_unification/results/case_studies/steering/<arch>/` | feature_selection.json + generations.jsonl + grades.jsonl per arch |
| `experiments/phase7_unification/results/case_studies/plots/` | phase7_steering_pareto, phase7_hh_rlhf_summary, phase7_hh_rlhf_scatter, phase7_steering_strength_curves |
| `docs/han/research_logs/phase7_unification/2026-04-26-{c1-hh-rlhf-stage1,c2-steering-stage1,agent-c-stage1-synthesis}.md` | Agent C's stage-1 synthesis writeups |

#### Six-arch shortlist (preserved)

The case studies cover this subset of `paper_archs.json:leaderboard_archs`:

```
topk_sae                              (per-token SAE baseline)
tsae_paper_k20                        (T-SAE faithful at k=20 — *the paper's setup*)
tsae_paper_k500                       (T-SAE faithful at k=500 — our k convention)
mlc_contrastive_alpha100_batchtopk    (MLC contrastive, multi-layer)
agentic_txc_02                        (TXC + matryoshka multi-scale, T=5)
phase5b_subseq_h8                     (TXC + H8 + subseq sampling, T_max=10)
```

If you need more archs from the leaderboard, pull them in — the
shortlist isn't a paper-binding constraint, just Agent C's choice.

### Dmitry's findings — required reading before any new work {#dmitrys-findings}

Dmitry's branch `origin/dmitry-rlhf` adds a paper-protocol steering
reproduction (Ye et al. 2025 App B.2) plus a controlled cross-protocol
comparison vs AxBench-additive. The 3 new intervention scripts have
been cherry-picked onto unification (commits below); the writeup
stays on Dmitry's branch.

**Files to read on Dmitry's branch (`git fetch origin dmitry-rlhf`):**

- `docs/dmitry/case_studies/rlhf/summary.md` — TL;DR + headline.
- `docs/dmitry/case_studies/rlhf/notes/methodology.md` — the two
  protocols formally compared, why they disagree.
- `docs/dmitry/case_studies/rlhf/notes/per_arch_breakdown.md` — full
  per-strength tables.
- `docs/dmitry/case_studies/rlhf/notes/qualitative_examples.md` —
  paper Table 2 reproduction.
- `docs/dmitry/case_studies/rlhf/plots/{paper_pareto_swapped,cross_protocol}.png`

**Code now on `han-phase7-unification`:**

- `experiments/phase7_unification/case_studies/steering/intervene_paper_clamp.py`
  — per-token paper protocol (clamp-on-latent + error preserve).
- `experiments/phase7_unification/case_studies/steering/intervene_paper_clamp_window.py`
  — paper protocol generalised to window archs (T-window encode,
  right-edge attribution, error-preserve at right edge,
  `use_cache=False`).
- `experiments/phase7_unification/case_studies/steering/intervene_axbench_extended.py`
  — AxBench-additive at signed strengths {-100..+100}.

**Headline numbers** (peak success / coherence per arch, 30 concepts,
seed=42, Sonnet 4.6 grader):

| arch | paper-clamp peak (s, suc, coh) | AxBench peak (s, suc, coh) | Δ suc |
|---|---|---|---|
| TopKSAE (k=500) | s=100 (1.07, 1.40) | s=100 (0.97, 1.33) | -0.10 |
| T-SAE (k=500)   | s=100 (1.33, 1.50) | s=100 (1.30, 1.23) | -0.03 |
| **T-SAE (k=20)** | **s=100 (1.93, 1.37)** | **s=100 (2.00, 1.13)** | +0.07 |
| TXC (T=5, matryoshka)        | s=500 (0.97, 1.20) | s=100 (1.53, 1.17) | **+0.56** |
| SubseqH8 (T=10)              | s=500 (1.10, 1.53) | s=100 (1.67, 1.27) | **+0.57** |
| H8 multi-distance (T=5)      | s=500 (1.13, 1.10) | s=100 (1.37, 1.53) | +0.24 |

T-SAE k=20 wins peak success on **both** protocols. Window archs lose
by 0.5–0.8 under paper-clamp; close to within 0.3 under AxBench (still
behind, but within concept-variance noise).

### Why TXC loses under paper-clamp — Dmitry's analytical answer

The two protocols differ in how the steering magnitude maps to the
intervention:

| | paper-clamp (Ye et al.) | AxBench-additive (Han) |
|---|---|---|
| equation | `x_steered = x + (s − z[j]_orig) · W_dec[:, j]` | `x_steered = x + s · unit_norm(W_dec[:, j])` |
| `s` units | absolute clamp value | multiplier of unit-norm decoder |
| depends on `z[j]_orig`? | **yes** — at `s = z[j]_orig`, intervention is a no-op | no |
| depends on `‖W_dec[:, j]‖`? | yes | no (unit-normalised) |

**The crux**: `z[j]_orig` for window archs is `O(T × per-token magnitude)` — the encoder integrates over T tokens, so active-feature magnitudes are 5–10× larger than for per-token archs. Under paper-clamp, `s=100` is "10× typical" for per-token archs but only "2× typical" for window archs — same nominal strength, different actual push. **The peak operating point shifts to ~5× higher absolute strength for window archs**, which Dmitry's data confirms (window archs peak at s=500, per-token at s=100).

Under AxBench-additive the magnitude difference washes out (unit-norm
decoder direction), so all archs peak at the same strength.

This is *not* "TXC is fundamentally bad at steering" — it's "TXC's
typical activation magnitudes don't match the paper's strength
schedule, which was designed against per-token archs only."

### Y's URGENT agenda

Two questions:

#### Q1 (WHY): is the magnitude-scale explanation the full story?

Dmitry's analysis is clean but worth verifying with finer-grained
diagnostics on our ckpts:

1. For each of the 6 shortlisted archs, plot the distribution of
   `z[j]_orig` magnitudes for the picked steering features across
   the 30 concept × 5 example probe set. Confirm the ~5× ratio
   between window and per-token archs.
2. For each arch + concept, plot success-vs-strength curves under
   paper-clamp. Does the peak strength scale as `T × (per-token peak)`?
   If yes, the magnitude story is the *full* story; if peaks shift
   non-linearly, there's a second factor.
3. **Try the obvious normalisation**: re-run paper-clamp on window
   archs with strength `s_norm = s_paper × ⟨z[j]_orig⟩_arch /
   ⟨z[j]_orig⟩_T_SAE_k20`. If TXC catches up under this rescaled
   schedule, the magnitude-scale story is empirically confirmed.

#### Q2 (HOW): defensible TXC steering protocol(s)

Open candidates Y should evaluate:

- **(A) Use AxBench-additive as the canonical protocol.** Already
  implemented; window archs are competitive. Argument: cross-arch
  fair on activation magnitudes. Counter-argument: it's not what the
  T-SAE paper did, so direct comparison is harder to defend in the
  paper.
- **(B) Per-family strength scaling.** Use paper-clamp but normalise
  `s` by `⟨z[j]_orig⟩_arch`. Defensible if Q1.3 shows it works.
- **(C) Decompose the window into per-position contributions.** The
  current generalisation (`intervene_paper_clamp_window.py`) clamps
  at the right edge only. Alternatives: clamp at every position in
  the T-window; weighted clamp by attention to the right edge; clamp
  at the centre position. Worth trying at least the per-position
  variant before discarding the protocol.
- **(D) Train TXCs differently** so feature magnitudes are
  comparable to per-token archs. This is **Z's territory** (hill-
  climb on the training side); flag to Z if Y finds promising
  directions during the steering analysis.

#### Concrete deliverables (revised)

1. **`docs/han/research_logs/phase7_unification/2026-04-29-y-tx-steering-magnitude.md`**
   — Q1.1, Q1.2, Q1.3 results. Either confirms or refutes the
   magnitude-scale hypothesis quantitatively.
2. **`results/case_studies/steering_paper_normalised/<arch>/{generations,grades}.jsonl`**
   — Q1.3 outputs (paper-clamp at family-normalised strength).
3. **`results/case_studies/steering_paper_pos<i>/<arch>/{generations,grades}.jsonl`**
   — Q2.C outputs (per-position clamp variants), if the magnitude
   normalisation alone doesn't close the gap.
4. **`results/case_studies/plots/phase7_steering_v2.png`** — final
   Pareto plot showing whichever protocol(s) we settle on. If it's
   AxBench-additive, the plot is what Agent C started with extended
   strength range. If it's normalised paper-clamp, the plot
   replaces Agent C's.
5. **Synthesis log** (`2026-04-29-y-tx-steering-final.md`)
   recommending which protocol the paper should adopt and why. Han
   makes the final call; Y provides the evidence.

### What carries over from before Dmitry's pass

The HH-RLHF dataset-understanding case study (Agent C's pass) is
mostly fine — Dmitry didn't disturb it. Y should still verify the
feature-selection step's quality (Agent C noted some concept→feature
mappings were noisy, e.g., `positive_emotion` → COVID-uncertainty
across archs) but the high-level claim ("TXC archs surface comparable
or richer feature sets to T-SAE") is independent of the steering
controversy.

---

### 🚨 50%-time priority — find a case study where TXC actually wins

> Even after the steering protocol gets cleaned up (Q1, Q2 above), the
> paper still needs a positive result for TXC. Without one, the
> headline becomes "T-SAE is at least as good as our temporal
> architectures on every interesting task" — which is a finding
> worth publishing but not the paper we want. **Y should devote ~50%
> of their bandwidth to brainstorming and trying new case studies
> where TXC's structural prior (multi-token receptive field)
> actually pays off.** The other 50% goes to the Q1/Q2 steering
> agenda above.

Methodology: try MANY ideas. Most will fail. The job is to find one
or two where TXC genuinely leads — those become paper case studies.
Negative results stay as "things we tried" footnotes.

#### Where to source ideas

- **Paper summaries in this repo**, all already vetted:
  - `papers/temporal_sae.md` — Ye et al. 2025; case studies in §4.5.
  - `papers/priors_in_time.md` + `papers/priors_in_time_important_sections.md`
    — Lubana et al. 2025 (TFA); case studies focus on predictive
    coding + slow features.
  - `papers/autoencoding_slow.md` — slow-features SAE paper; ideas
    around persistent representations.
  - `papers/crosscoders.md` — Anthropic crosscoder; circuit
    decomposition use cases.
  - `papers/reasoning_features.md` — multi-token reasoning steps;
    natural fit for TXC.
  - `papers/are_saes_useful.md` — meta-paper on SAE downstream
    utility; lists tasks where SAEs do/don't matter.
  - `papers/sparse_but_wrong.md` — common SAE failure modes; TXC
    might fix some.
- **GitHub / arxiv search**: papers Y can pull and replicate
  selectively. Y has internet — use it. Aim for tasks where
  multi-token structure is the natural unit.

#### Things we have tried and failed (or not won decisively)

- **Sparse-probing AUC at k_feat=20** (Phase 7 leaderboard, base):
  TXC bare-antidead at T=5 wins with 0.9358, but T-SAE k=500 close
  behind at 0.9339; spread across top-10 is 0.0044 — noise level.
  Not a *decisive* TXC win.
- **Sparse-probing AUC at k_feat=5**: H8 multidist T=8 leads with
  0.8989 — better margin (TXC family tighter cluster) but still
  arguable.
- **AxBench-additive steering** (Agent C, then Dmitry extended):
  TXC family Pareto-dominates **at moderate strength**, but T-SAE
  k=20 wins peak success on both protocols Dmitry tested. Soft win.
- **Paper-clamp steering** (Dmitry): TXC loses by 0.5–0.8 to T-SAE
  k=20. Hard loss (modulo the magnitude-scale story).
- **Phase 5 cross-token tasks** (winogrande, wsc with FLIP):
  TXCDR-T5 + mlc_contrastive were the most complementary archs
  (Jaccard 0.338 on per-example errors) — but **concat-probing
  over [TXC ∥ MLC] latents did NOT beat best-individual** (Phase 5
  §error-overlap-A1). Joint use of TXC + MLC features didn't pay off
  the way error-overlap suggested it should.
- **HH-RLHF dataset understanding** (Agent C): TXC archs surface
  feature sets that *look comparable* to T-SAE. Not a sharp win or
  loss; mostly stable.
- **Last-position aggregation** (Phase 5 lp metric): MLC family
  beat TXC family (`mlc_contrastive_alpha100_batchtopk` 0.8124 vs
  `phase5b_subseq_h8` 0.8442 — wait that's TXC win by 3 pp; but
  re-check Phase 5's lp 3-seed numbers, the family ordering is
  unstable).

#### Brainstorm candidates worth trying

These are starting points; the most promising ones become case
studies. **The unifying thread**: each is a task where the
"useful" feature is intrinsically multi-token, so a per-token SAE
either has to fragment it across positions (loses signal) or
duplicate it at every position (wastes capacity).

- **Multi-token concept extraction.** Train a probe on each arch's
  features to predict 2-, 3-, 5-gram concepts (e.g., named entities,
  multi-word idioms, syntactic constructions). Compare AUC across
  archs at fixed k_feat. Hypothesis: TXC features *natively* span
  the n-gram so probes need fewer features to hit a given AUC.
- **Anaphora / coreference resolution probing.** Predict from a
  feature subset whether two tokens refer to the same entity. The
  receptive-field argument: per-token features at the anaphor see
  only the anaphor, while TXC features at T≥3 see anaphor +
  antecedent + intervening context. Phase 5's wsc_coreference task
  is a small-scale version of this; a properly designed probing
  task at scale could be a TXC win.
- **Span-detection / sentence-boundary probes.** Detect "is this
  the start of a relative clause / appositive / parenthetical".
  These are inherently multi-token signals.
- **Long-range agreement.** Predict subject-verb agreement when
  separated by intervening modifiers. Per-token can't see the
  subject when probing the verb; T-window can.
- **Activation reconstruction at masked positions.** Mask 1 token,
  reconstruct from neighbour activations using SAE features only.
  TXC's encoder has the unmasked context built in; per-token has to
  bridge via decoded output. Reconstruction MSE per arch.
- **Counterfactual SAE-driven editing of multi-token spans.**
  "Replace the named entity in this sentence with a different one
  via SAE-feature manipulation". Per-token SAE has to manipulate
  every token of the span individually; TXC can manipulate the
  whole span via one feature.
- **Speculative-decoding-style next-T-token prediction.** Given
  the SAE-encoded context, predict the next T tokens jointly
  (perplexity / accuracy). TXC features are designed exactly for
  this kind of jointly-distributed-output task.
- **Feature-driven retrieval / clustering.** Use SAE-feature
  embeddings as document/passage retrieval signals; passages with
  the same concept should cluster. Multi-token concepts naturally
  benefit from TXC.
- **Circuit decomposition via SAE features.** Decompose attention
  heads' contributions into interpretable feature paths
  (Anthropic-style). At positions where attention is genuinely
  long-range (e.g., heads attending 8+ tokens back), TXC features
  may map more cleanly to the head's behaviour than per-token.
- **Persistent / slow features.** From `papers/autoencoding_slow.md`
  — features that activate stably across multiple positions are
  conceptually closer to "topic" features. Per-token SAE has to
  re-learn them at every position; TXC integrates them by
  construction.
- **Activation-steering for MULTI-token concepts.** Steering toward
  a multi-token concept (e.g., "shipping address format") via per-
  token SAE requires identifying *every* contributing feature and
  steering each one; TXC has the concept as one feature. Like
  Dmitry's setup but with concepts that genuinely require T tokens
  to express.
- **Adversarial robustness of features.** Apply perturbations
  designed to fool single-token features (insert filler tokens,
  scramble word order locally). TXC features may be more robust
  because the T-window averages over local perturbations.

#### Workflow for the 50% allocation

For each candidate Y picks up:

1. Spend ≤ 1 hour skimming relevant prior work (papers/, github,
   web) before coding. Most ideas are already-tried elsewhere; check.
2. Implement a smoke-test version on 1–2 archs (TXC T=5 + T-SAE
   k=20 as the baseline pair). 1–2 hours.
3. If the smoke test shows TXC genuinely leads (or trails): scale
   up to the 6-arch shortlist + write up. If neither leads
   decisively, kill the candidate and move on.
4. Document each candidate (win or loss) in a research log:
   `docs/han/research_logs/phase7_unification/2026-04-2*-y-csN-<topic>.md`
   with `<N>` incrementing. Keep them short — even failures take
   ~50 lines.
5. After ~3-5 candidates, write a synthesis log
   (`2026-04-2*-y-cs-synthesis.md`) ranking them by "TXC margin"
   and propose ≤ 2 to keep as paper-grade case studies.

#### Constraints

- Same hardware envelope (A40, 46 GB) and same paper-wide constants.
- Same 6-arch shortlist for the headline plots; one-off probes on
  more archs are fine if the candidate succeeds.
- Y can prototype on tiny task sets (50–500 examples) before
  scaling up. The point is fast iteration, not production-grade
  experiments at the prototype stage.
- Anthropic API for grading is fine; use prompt caching to keep
  costs down; budget ~$50-200 across all candidates.
- If a candidate looks promising and X's leaderboard probe data
  could help, leave a note in `2026-04-2*-y-cs-x-handoff.md` —
  X is on the same branch and can run additional probings if
  the candidate justifies it.

### Concrete deliverables

(All under `experiments/phase7_unification/case_studies/`, on branch
`han-phase7-agent-c`.)

1. **`reproduction/tsae_paper_k20_repro.json`** — reproduction status
   on `tsae_paper_k20` vs T-SAE Table 1. PASS if within seed σ;
   document gaps if FAIL.
2. **`steering/<arch>/grades_extended.jsonl`** — re-grade at the
   paper's full coefficient range for each of the 6 shortlisted
   archs. Replaces Agent C's `grades.jsonl` (which keeps as
   historical reference).
3. **`plots/phase7_steering_pareto_v2.png`** — Pareto plot at the
   extended coefficient range. The "v2" suffix to keep Agent C's
   v1 visible for comparison.
4. **`plots/phase7_steering_strength_curves_v2.png`** — per-arch
   suc-vs-coefficient curves at the full range, so the small-vs-
   high regime is visually distinct.
5. **`hh_rlhf/synthesis_v2.md`** — refresh of Agent C's HH-RLHF
   synthesis if the high-coefficient steering uncovers feature
   behaviours that weren't apparent at low coefficients (e.g.,
   features that only fire/steer above some threshold).

### Pod spec

| | Agent Y |
|---|---|
| GPU | NVIDIA **A40**, 46 GB VRAM |
| vCPUs | (typical) 8 |
| System RAM | **46 GB** (cgroup `memory.limit_in_bytes`) |
| Persistent volume | **900 GB** at /workspace |

Memory budget for case-studies workload (no SAE training needed):

- Subject model `gemma-2-2b` on GPU (~5 GB)
- One SAE ckpt loaded for steering (~1–2 GB)
- Steering forward passes: per-token KV cache + decoder activations
  ~few GB at standard generation length (256 tokens, batch 1–8)
- HH-RLHF passages cached on disk; loaded on demand into ~1 GB CPU

Total peak ~15 GB GPU. Comfortable.

For grader API calls (Claude Sonnet 4.6 — Agent C's substitute for
the paper's Llama-3.3-70B grader) use `concurrent.futures.ThreadPoolExecutor`
with `max_workers=8`; Anthropic API rate-limits aren't usually the
bottleneck.

### Branch + workflow

- **Branch: `han-phase7-unification`.** Same as X and Z. See
  `brief.md` § "Branch hygiene for three concurrent agents" for
  directory ownership rules.
- Y's writes are confined to:
  - `experiments/phase7_unification/case_studies/` (source code)
  - `experiments/phase7_unification/results/case_studies/` (outputs:
    plots, JSON labels, generations, grades)
  - `docs/han/research_logs/phase7_unification/2026-04-2*-y-*.md`
    (Y-prefixed research logs)
- Y does NOT touch:
  - `paper_archs.json`, `canonical_archs.json` — read-only.
  - `experiments/phase7_unification/results/probing_results.jsonl` —
    that's X's append-target.
  - `experiments/phase7_unification/results/training_*` — X / Z.
  - `experiments/phase7_unification/results/plots/` — X's leaderboard
    plots. Y's plots stay under `results/case_studies/plots/`.
  - `experiments/phase7_unification/hill_climb/` — Z's territory.
- Ckpts Y needs are on HF (`han1823123123/txcdr-base/ckpts/<arch>__seed<n>.pt`).
  Pull via the same `huggingface_hub` pattern Agent C used.
- For the IT-side extension: the leaderboard archs at IT (when X
  ships them) will be on a NEW HF repo `han1823123123/txcdr-it`
  (see `plan.md`). Defer IT-side case studies until X's IT
  trainings are done — focus on landing the base-side reproduction
  + extension first.

### Coordination with Agent X

- X is on `han-phase7-unification`. X owns
  `paper_archs.json:leaderboard_archs` and the T-sweep specs.
- Y reads `paper_archs.json` to know which archs are paper-grade
  and `results_manifest.json` to know which seeds have ckpts.
- Y's case studies use base-side ckpts — those are already trained
  (X did them on H200 prior); no dependency on X's current A40
  fill-in work.
- Y has NO probing dependency. Y's Pareto x-axis is the leaderboard
  AUC (X's deliverable), but the *first* iteration of the Pareto
  plot can use the AUCs already in the unified
  `probing_results.jsonl`. When X publishes IT-side AUCs, Y can
  refresh.
- Both agents re-run `build_results_manifest.py` after their work
  changes the data sources, so the next agent picking up sees the
  fresh state.

### What this brief assumes — and what to flag if false

- **Assumption**: the agent-c branch's steering pipeline runs
  end-to-end as committed. If it doesn't (broken imports, missing
  data files, etc.), the first thing to fix is reproducibility —
  before pushing on the science.
- **Assumption**: T-SAE's paper code on github runs against our
  `tsae_paper.py` ckpt format. The faithful-port doc (head of
  `src/architectures/tsae_paper.py`) claims this; verify in practice.
- **Assumption**: Anthropic API access (Claude Sonnet 4.6) is
  available for grading. If not, fall back to a local grader as
  Agent C did.

If any of these are false, surface it in a dated log under
`docs/han/research_logs/phase7_unification/2026-04-2*-y-blockers.md`
before doing other work.
