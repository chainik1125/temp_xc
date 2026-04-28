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

- **You inherit Agent C's case-studies workstream** on branch
  `han-phase7-agent-c` (NOT merged into `han-phase7-unification` —
  they're a parallel workstream, owned end-to-end by Y).
- Two paper-grade case studies, both reusing Phase 7 base-side ckpts:
  (1) **HH-RLHF dataset understanding** — what features does each
  arch find?
  (2) **AxBench-style steering** — can each arch's features steer
  generation?
- **Two pitfalls in Agent C's prior pass that you MUST fix before
  extending the work**:
  1. Steering used **too-small coefficients**, missing the regime
     where T-SAE's paper-claimed Pareto dominance actually shows up.
     The "TXC dominates steering" plot Agent C produced is therefore
     suggestive but not paper-grade.
  2. **T-SAE paper's case studies (Ye et al. 2025 §4.5) were never
     reproduced** end-to-end. Without that reproduction, claims like
     "TXC matches/beats T-SAE on the paper's own benchmark" are
     unfounded — Y needs to land Table 1 / Figure 5 numbers on the
     `tsae_paper_k20` ckpt first, *then* extend.
- Pod: **A40, 46 GB VRAM, 46 GB pod RAM, 900 GB**. Same hardware
  envelope as Agent X. See `2026-04-28-a40-feasibility.md` for what
  fits. The case studies don't train SAEs from scratch — you're
  loading existing ckpts + running model forward passes / steering —
  so memory pressure is mild.

### Inheritance from Agent C

#### Branch state

`origin/han-phase7-agent-c` carries 10 commits not on
`han-phase7-unification`. Don't merge it into unification —
it's the case-studies branch; X writes the leaderboard branch; the
two stay separate by design.

The branch has:

| path | what's there |
|---|---|
| `experiments/phase7_unification/case_studies/hh_rlhf/<arch>/` | per-arch top features + feature stats (autointerp on HH-RLHF passages) |
| `experiments/phase7_unification/case_studies/steering/<arch>/` | feature_selection.json + generations.jsonl + grades.jsonl per arch |
| `experiments/phase7_unification/case_studies/plots/` | phase7_steering_pareto, phase7_hh_rlhf_summary, phase7_hh_rlhf_scatter, phase7_steering_strength_curves |
| `docs/han/research_logs/phase7_unification/2026-04-2*-c-stage*.md` | Agent C's stage-1/2 synthesis writeups |

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

### Pitfall 1 — too-small steering coefficients

Agent C's pipeline at `experiments/phase7_unification/case_studies/steering/`
reports per-feature (success, coherence) for each arch at varying
coefficients. The headline plot (`phase7_steering_pareto.png`) has TXC
families in the upper-right — but **only at the small-coefficient
regime**.

**Diagnostic to run first:** read each arch's `grades.jsonl` and
plot success rate vs coefficient. If success monotonically increases
with coefficient and **plateaus only past Agent C's max coefficient**,
the Pareto plot under-represents archs whose features need higher
coefficients to engage. T-SAE's paper (Ye et al. 2025 §4.5) used
coefficients in a range that *does* push the model into the regime
where features actually flip behaviour — Agent C didn't.

**What to do:**
1. Re-read T-SAE paper §4.5 (`papers/temporal_sae.md` if summarised
   in repo; otherwise the full paper at arxiv 2511.05541) for the
   exact coefficient range used.
2. Re-run the steering pipeline at the **paper's coefficient
   range**, not just Agent C's small-coefficient range. Specifically
   include coefficients large enough that `tsae_paper_k20` reaches
   the paper's reported success rate.
3. Re-grade. If TXC families *still* Pareto-dominate at the higher
   coefficient regime — the case-study claim survives. If not —
   that's a real finding (Agent C's plot was an artifact of
   too-small coefficients) and the paper's framing changes.

The infrastructure (feature selection, generation, grading) on the
agent-c branch can be reused — just change the coefficient sweep.

### Pitfall 2 — T-SAE paper not reproduced

The repo has `tsae_paper_k20` ckpts on `han1823123123/txcdr-base`
(seeds 42, 1, 2). Agent C ran their pipeline on it but **never
verified the outputs match T-SAE's reported §4.5 numbers**. Until
that reproduction succeeds, comparison claims are unfounded.

**What to do:**
1. Pin down the specific T-SAE-paper numbers Y is reproducing. Most
   important is **Table 1 in §4.5** (the steering Pareto numbers
   for the temporal-contrastive head on Pythia-160M / Gemma-2-2b — the
   paper reports both). Pick the Gemma-2-2b numbers as the comparison
   point since that's our subject model.
2. Run T-SAE's protocol (their codebase is at
   github.com/AI4LIFE-GROUP/temporal-saes) on **our**
   `tsae_paper_k20` ckpt. The faithful port `src/architectures/tsae_paper.py`
   was built specifically to be loadable by both their code and ours
   — verify this works.
3. If our `tsae_paper_k20` numbers match T-SAE's reported numbers
   (within their seed σ), reproduction is done — proceed to extend
   to the other 5 archs in the shortlist.
4. If they DON'T match: this is the most important finding. Possible
   causes:
   - `tsae_paper_k20` was trained on FineWeb (Phase 7 default), the
     paper used a different corpus.
   - Layer mismatch (paper's L12 anchor matches Phase 7's, so probably
     not this).
   - Subtle hyperparameter drift (warmup steps, AuxK loss weight, etc.).
   Document the gap before extending.

The T-SAE paper code's `dictionary_learning/` dir has their full
training/eval pipeline — when in doubt, run their script against our
ckpt rather than rebuilding the eval ourselves.

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

- Stay on `han-phase7-agent-c`. Cherry-pick from `han-phase7-unification`
  if you need new ckpts that weren't on the agent-c branch when it
  diverged.
- Ckpts you need are on HF (`han1823123123/txcdr-base/ckpts/<arch>__seed<n>.pt`).
  Pull via the same `huggingface_hub` pattern Agent C used.
- For the IT-side extension: the leaderboard archs at IT (when X
  ships them) will be on a NEW HF repo `han1823123123/txcdr-it`
  (see `plan.md`). Defer IT-side case studies until X's IT
  trainings are done — focus on landing the base-side reproduction
  + extension first.
- Keep your case-studies plots under `case_studies/plots/`. Don't
  write to `results/plots/` (that's X's leaderboard plots).

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
