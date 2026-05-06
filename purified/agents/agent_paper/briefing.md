---
agent: agent_paper
last_state_update: 2026-05-03T21:00:00Z
component: orchestration
---

<!--
Section ownership (PROTOCOL.md § 14, "Briefing maintenance"):
  - "Identity + mandate (Han owns)" — Han's prose, do NOT edit.
    Han may rewrite at session start to redirect priorities.
  - All "(agent owns — overwrite)" sections — overwrite freely
    at every compact / session end.
For chronological history, run `git log -p purified/agents/agent_paper/briefing.md`.
-->

## Identity + mandate (Han owns — agents do not edit)

Identity: You are agent PAPER, an expert ML paper writer. You operate locally, not on a runpod.

Task: You are currently on the han-phase7-unification branch. This project as sout "temporal crosscoders". Read the original project intro here: docs/han/research_plan/project_brief.md

The entire project is a very disorganized state, most results are negative yet still interesting, and we have decided that it's worth writing a paper within 3 days. There is a sprawling wasteland of throwaway code and experiments. Not just on this branch, but on the other branches too. Your task is to create a completely new and independent 'paper-ready' branch `final` that has ZERO dependencies on the other branches and has an entirely isolated framework. This means have code that's duplicating what's already in existing branches is completely fine. Code, experiments and documentation should be **quarantined** inside `purified/`, which already contains this briefing.

The result of this briefing documents the state of the non-quarantined wasteland, where most results are either irrelvant or invalid, but there are key insights that we have to rescue and extract. 

### branch: han-phase7-unification 

This is the branch with the most content and results that we want to bring over to the final paper.

#### C1: synthetic study 1

docs/han/research_logs/phase2_toy_experiments/2026-03-30-experiment1-topk-sweep.md

toy setups where take many architectures and sweep over sparsity.
the setup for this is relatively simple and I am confident it's existing implementation is correct.

#### C2: synthetic study 2

docs/han/research_logs/phase3_coupled_features/2026-04-07-experiment1c3-coupled-features.md

results suggest TXC is better at *global feature recovery* and we can sweep the tradeoff between globan and local recovery by changing *T*. 

#### C3: sparse probing benchmark

sparse probing benchmark comparing various hill-climbed TXC architectures against baselines: docs/han/research_logs/phase7_unification/agent_x_paper/2026-04-29-leaderboard-multiseed.md; results suggest TXCs match the best baselines in sparse probing. However, an earlier set of sparse probing results show the TXC performing much better relatively: docs/han/research_logs/phase5_downstream_utility/summary.md

The state of the sparse probing benchmark is poor. For instance, phase5 uses a full 36-task suite, yet this task suite is non-standard and was cobbled together from various sources. The final paper task suite needs to be revisited. phase7 initially used the full 36-task suite but then decided to reduce it to 16; the way the reduced task suite was chosen is alarming. Furthermore, phase5 looked at the instruction tuned model of Gemma2B, whereas phase7 attempted to look at both. Both phases have way too many models and hill-climbing results. phase7 tried to use a clever 'S-tail sliding window' sparse probing evaluation protcol (docs/han/research_logs/phase7_unification/2026-04-26-S-decision-revised.md) that phase5 does not use. 

Suggestion: sparse probing benchmark needs to be redone entirely using a highly condensed set of architectures. 

#### C4: qualitative analysis of extracted features

docs/han/research_logs/phase6_qualitative_latents/2026-04-25-final-summary.md

this tries to use the 'piece together unrelated chunks of text' protocol that the T-SAE paper uses. We expect to be able to extract 'global semantic features' that represent the chunk of text rather than individual words or syntax.

we find that a TXC with sufficiently large window can achieve similar quality features compared to T-SAE according to the "Top-256 cumulative SEMANTIC Pareto" metric. This is a very important result that we need to finish exploring and put into the paper. 

#### C5: RLHF steering case study

this casestudy tries to use TXCs to perform steering. The case study is the same one used in the T-SAE paper in line 227 of papers/temporal_sae.md.

hill-climbing results on the steering coherence vs success pareto tradeoff casestudy: docs/han/research_logs/phase7_unification/unified-pareto.md ; results suggest that all TXCs are dominated by the T-SAE baseline at low coherence, but at high coherence, the TXC edges out slightly. There is an absurd number of hill-climbed TXC architectures.

### branch: em-nanda

#### C6: emergent misalignment case study

This branch is Dmitry's work on the emergent misalignment case study using TXCs. I am not completely aware of his latest results but from what I know, his current best TXC is tantalizing close to outperforming the baselines but is still not quite there.

### branch: aniket-ward-stage-b

#### C7: Backtracking 

This branch is Aniket's work on the Jake Ward backtracking case study from the paper "Reasoning-Finetuning Repurposes Latent Representations in Base Models". From what I understand, Aniket wasn't able to get the TXC to beat baselines but some results are interesting. 

### what I want from you, Agent Paper

You will oversee the search and extraction process and high-level orchestration. You must maintain a coherent understanding of the entire story the paper wants to tell. 

Your first mission is to decide on the following:
* commit everything and push the `final` branch.
* understand the 7 components that we want in the paper. 
* what should the code layout be like in `purified/`? my only constraint is that all the 'case studies' (RLHF, EM, Backtracking) should be put into the same place and have some sort of unified abstraction layer exposing them. This alone would take signficant refactor effort but would be worth it. We should give a specific name for this suite of benchmarks: `temp-bench` so that we and agents recognize its significance.  
* how many worker agents should we use? I can afford to run many runpods in parallel for the next 72 hours: 
  * a pod with two H100s and either one agent using both H100s or two agents, one using one H100.
  * a pod with up to three A40s and three agents (one using each A40).
  * a pod with a H200 (only if needed).
* what runpod scaffolding we do need (token loading, uv env setup instructions etc). We already have scaffolding in the wasteland area: RUNPOD_INSTRUCTIONS.md; relevant tokens should already be available locally, but on runpod machines, they must be stored in /workspace/; there is a script scripts/runpod_phase7_bootstrap.sh that does this but it's in the wasteland.
* how should we ensure that experiments and models are cleanly recorded in some state all agents can see and modify without running into conflicts?
* how should we structure writeups by each agent? One idea is to cleanly separate by component, rather than agent, since agents live and die fleetingly. 
* what should you (Agent Paper) want to run? you have a 5090 with around 50GB of RAM in WSL. 
* **most importantly** we need to narrow down on just *TWO* TXC architectures. We can't use a different 'gigabrain fine tuned galaxy Matryoskha contrastive loss with adaptive T' for each case study. Ideally, we select one 'rather plain vanilla TXC at a reasonable T', and one 'exotic TXC' (perhaps the best from Phase5), and we stick with those two for **everything**, no hill-climbing except for selecting obvious parameters like the sparsity, or the dictionary size, which may have a different optimal choice for each case study.

### useful papers for you and other agents to refer to

* T-SAE: papers/temporal_sae.md
* TFA: papers/priors_in_time.md
* Jake Ward backtracking: papers/backtracking.md
* Sparse probing: papers/are_saes_useful.md
* Crosscoders: papers/crosscoders.md

---

## Current state (agent owns — overwrite at every compact)

**Last verified: 2026-05-06 PM (~19:30Z). Post-RunPod-incident +
post-rescue. Pre-compact freshness check.**

### Post-compact priority

> Han 2026-05-06: "the next thing to do after compact is to optimize
> resource allocation for full set of base results, and the synthetic
> benchmark crises"

**Two parallel concerns** to triage right after compact:

1. **BASE results coverage**: agent_steer_100k owns BASE C3+C4
   replication. Currently has a long sequential queue (~18-22 hr) on
   their 1× H100. Q for after-compact: should agent_filler (8× A40,
   nearly idle after C2 cleanup) help parallelize? Or agent_nlp's
   recovered pod once their 7-cell tail lands?
2. **Synthetic benchmark crisis**: Dmitry's Effect-1-vs-Effect-2
   finding (TXC wins are sample-aggregation, NOT temporal pattern
   detection). `agent_synth` (NEW, 8× 5090) is investigating. First
   stage = ρ-sweep (commit `86f24c9c`); after it lands, decide paper
   framing for C2.

### RunPod incident 2026-05-06

4 agents went offline mid-mission when their pods exploded:
- **agent_nlp** + **agent_em** (shared 2× H100 PERSISTENT pod) — volume survived
- **agent_steer** + **agent_back** (shared 4× A40 EPHEMERAL pod) — ephemeral, gone

Recovery: Han attached a fresh pod to the surviving 2× H100 persistent
volume. agent_nlp (new identity on the recovered pod) ran the
URGENT_HF_SYNC.md procedure (commit `eaa75a10` + `17028e53`):
- 26 checkpoints → HF (was 0/14 → 26/26)
- 67 judge-transcript run dirs → HF
- 180 leaderboard rows committed to git (incl. 161 C3 + 19 C4)
- Total: 285 + 783 files / ~70K lines committed across 2 commits

Net loss after recovery: **0 paper-bound cells**. The 14 deprecated
agent_nlp checkpoints (B=256, n_steps=10K — pre-§ 15) stay in the
leaderboard for diff but are filtered from the headline.

### Active pods + agents

| Pod | Hardware | Agent(s) | Status |
|---|---|---|---|
| Recovered 2× H100 (persistent) | 1× H100 (only 1 GPU active post-incident) | **agent_nlp** | 7-cell C3 k_feats tail in flight (~3.5 hr) |
| 1× H100 (ephemeral) | 1× H100 | **agent_em_100k** | 6-cell tfa k_feats borrow into agent_nlp territory (~3 hr) |
| 1× H100 (ephemeral) | 1× H100 | **agent_steer_100k** | BASE C3 txc_base T=10/T=20 RE-TRAIN (post-bug-fix) → BASE k_feats expansion → C4 BASE eval (~18-22 hr serial) |
| 8× A40 (ephemeral) | 8× A40 | **agent_filler** | Final C2 T=12 high-k cells (10 left, ~30-50 min) + HF backfill → `status: complete` |
| 8× 5090 (ephemeral) | 8× 5090 | **agent_synth** (NEW) | Synthetic investigation: Stage 1 ρ-sweep (~5-10 min) → wait for Han greenlight on Stage 2 (DC/AC ablation) |
| Local 5090 | 1× 5090 | **agent_paper** (me) | This pod; framework + decisions + briefings |

Lost pods (agent_em, agent_steer, agent_back): all `status: complete`
before the incident. No re-spawn needed.

### Headline numbers (paper-bound)

- **C1** (toy Markov, 213 cells): txc_base default at k=6 → AUC 0.983
  (winner); topk_sae catches up at k≥10 (peak 0.936 at k=12); tfa flat
  ~0.46-0.48. c1.md rendered with paper-ready plot.
- **C2** (coupled HMM, ~210 cells, 95% complete): txc_pro T=2 gAUC=0.99
  at k=1, MATCHES Dmitry's `coupled_rho_sweep` exactly.
  **🚨 SYNTHETIC CRISIS**: T-modulation goes wrong way — gAUC at k=5:
  T=2 → 0.904, T=5 → 0.684, T=12 → 0.678. More temporal context HURTS.
  Consistent with Effect 1 (sample aggregation), NOT Effect 2 (temporal
  detection). agent_synth is investigating with the ρ-sweep.
- **c1_noisy** (under C2, 156 cells): txc_base T=2 hits AUC 0.982-0.990
  across k=4..10, reproducing wasteland's "AUC ≥ 0.98" claim.
- **C3 IT** (~245 cells): 6 archs full coverage at canonical k=5/20.
  K-feats expansion to {5, 10, 20, 40, 80, 160, 320, 640} mostly done;
  13 cells remaining (split between agent_nlp + agent_em_100k post-rescue).
- **C3 BASE** (~50 cells, in progress): per-token + TFA + txc_pro DONE.
  txc_base T=10/T=20: trainings done but evals BROKEN — bug fix landed
  (`1ed4fde5` — `my_train_fn` now applies `arch_hparams_override`),
  agent_steer_100k owns the re-train + re-eval.
- **C4 IT** (10+ cells): T-SAE / TXC-base / TXC-pro × 3 seeds done.
  TFA + MLC + T-sweep variants pending re-render.
- **C4 BASE**: pending (cache-hit on agent_filler's BASE TXC checkpoints).
- **C5** (61 cells): 5 archs (TopK, T-SAE, TFA, TXC-base, TXC-pro)
  + detection axis (15 cells, agent_steer pre-incident).
  Mean peak@1.75: T-SAE 1.93 (best), TopK 1.66, TFA 0.33.
- **C6** (30 cells): 8/8 canonical (sae_arditi + TXC+brickenauxk × 2
  seeds × 2 organisms) + detection (commit `fa31b455`).
  TXC + Bricken: +3.24 align (14B), +6.14 align (7B). Detection
  COMPLETE.
- **C7** (19 cells): all 7 archs landed (agent_back's v4 sweep,
  COMPLETE per `2b44235e`). Detection refactor COMPLETE (`4691e835`).
  Extended magnitude grid ±{20-90} pending — agent_back gone, no one
  picked it up yet.

### Recent framework + decisions (post-§ 15/16/17)

- **§ 15** (decisions): per-arch literature-faithful T (TopK T=1,
  T-SAE T=2, TXC T=5).
- **§ 16** (decisions): MLC + TFA paper-faithful expansions (multi-
  layer cache for MLC; TFA at B=32 + full seq).
- **§ 17 (Han 2026-05-05 PM)**: TXC-base T-sweep ∈ {5, 10, 20} on C3
  IT + BASE. agent_nlp + agent_steer_100k own.
- **C3 k_feats expansion** (Han 2026-05-06): {5, 10, 20, 40, 80, 160,
  320, 640} for IT + BASE. ~80% done; 13 cells remaining post-rescue.
- **NEW datasources** (`configs/datasources.yaml`):
  - `gemma_2_2b_it_l11to15_fineweb_24k128` (multi-layer for IT MLC)
  - `gemma_2_2b_base_l11to15_fineweb_24k128` (BASE multi-layer for MLC)
  - `gemma_2_2b_base_l13_concat_v1` (BASE C4 concat)
  - `toy_coupled_K10_M20_d256_rho{00,03,06,09}` (C2 ρ-sweep)
- **Bug fixes landed**:
  - `076ff2f5` — TFA `_is_valid_cell` at C1 (was incorrectly grouping
    TFA with per-token archs; should be window-arch constrained at
    toy d_sae=40).
  - `1ed4fde5` — `c3_probing/run.py:my_train_fn` now applies
    `arch_hparams_override` before instantiation. Was the root cause
    of agent_filler's BASE C3 T=10/T=20 eval crashes.

### Detection scaffolding (commit `be59b7be`)

- `temp_bench.eval.detection.detect_case_study(arch, sentence_acts,
  labels, qids, S_grid, shuffle_seed=42)` — drop-in for C7's
  `compute_pr_auc_at_S` + paired within-window shuffle ablation.
- `temp_bench.utils.shuffles.shuffle_within_window` — paired control.
- C5 detection (15 cells, agent_steer pre-incident, COMPLETE).
- C6 detection (8 cells, agent_em pre-incident, COMPLETE per
  `fa31b455`).
- C7 detection refactor (agent_back COMPLETE per `4691e835`).

**Sequence (all post-compact, this session)**:

- 5-step plan landed in 3 commits — `dd5f773e` (4 stand-downs),
  `15ad0de4` (framework change: `train_window_size` on
  preloaded_batch_iter + TrainingConfig + 5 new tests, 136/136
  green), `74dc2cd9` (decisions § 14 dep + § 15 add + 4 briefing
  rewrites + 2 driver pass-throughs).
- Agents acted on stand-down on their own pods: agent_em_100k
  (`3f53791a` killed C3 MW sweep, status idle), agent_em
  (`b549d91c` acknowledged + canonical mission complete + wrap-up).
  agent_filler + agent_steer_100k will see the abort on next pull.
- Framework: `train_window_size: int | None = None` plumbing.
  Default None preserves all existing train_keys (verified by
  `model_dump(exclude_none=True)` test). Setting an int gets a
  fresh key.
- Re-train scope (per-arch literature-faithful T at batch_size=1024):
  - **C3 TopK** (agent_nlp, 2× H100 with borrowed GPU 1): T=1.
    3 trainings + 6 evals. Wall ~2.4 hr.
  - **C3 T-SAE** (agent_em_100k, 1× H100): T=2 (Bhalla/Ye 2025
    §3.1 paper-faithful adjacent pairs). 3 trainings + 6 evals.
    Wall ~4.5 hr.
  - **C5 T-SAE** (agent_filler, 8× A40 parallel): T=2. 3 cells.
    Wall ~45-60 min.
  - C6 + C7 baselines unchanged (already at literature scale).
  - All TXC cells unchanged.
- C7 T-SAE keeps T=5 (agent_back's `_spec_window_size` fallback);
  framework supports both.

The full state of the world:

- **Canonical results** for C3 (agent_nlp), C5 (agent_steer), C6
  (agent_em — 8/8 DONE) all in `leaderboard.jsonl`. Headlines:
  - C3: TopK > TXC > T-SAE on probing AUC (honest negative under
    current per-step setup, BUT see § "FAIRNESS REFRAME" below).
  - C5: TXC matches T-SAE at high coh — v1.1.0 sweep complete after
    concept-lift bug fix (commit `ef33f822`); hypothesis refuted.
  - C6 14B-finance: TXC + Bricken **+3.24 align over SAE-arditi**
    (mean of 2 seeds). C6 7B-medical: +6.14 align (seed=42 only at
    last check; seed=1 landing ~13:14 UTC today).
  - C7: agent_back's v4 sweep in flight, 2 cells landed (txc_pro
    seed=42 +0.377 Δgc; tfa seed=42 +0.344).

- **MW deployment** (decisions.md § 14) — landed 2026-05-05 AM. Two
  new YAML aliases `txc_base_mw` + `txc_pro_mw` with `multi_window:
  true` in hparams. Standalone benchmark on local 5090 verified 5×
  more tokens/step at 1.26× wall on TXCPro-toy.

- **⚠️ FAIRNESS REFRAME — Han 2026-05-05 PM, just decided, NOT YET
  EXECUTED**: SAEBench paper (papers/are_saes_useful.md, App. B)
  shows canonical SAE training is buffer-based, batch=2048
  TOKENS/step (not sentences), 500M tokens total. Our two patterns:

  | Component | Pattern | per-token SAE tokens/step |
  |---|---|---:|
  | C3, C4, C5 | sequence-based (B sentences × all 128 positions) | **131,072** — 5× over SAEBench 2K canonical |
  | C6, C7 | window-based (B sentences × T positions, T per-arch; T=1 for SAE) | **1,024** — close to SAEBench scale |

  C3/C5's 131K is OVER-batched per-step; C6/C7's 1K is near-canonical.
  My earlier "TXC has 25× FLOPs disadvantage at C3/C5" framing was
  technically accurate but the FIX direction is wrong: the right fix
  is to bring C3/C5 baselines DOWN to T=1 window-based (match C6/C7
  + literature), NOT to bring TXC up via MW. Han's call.

  **All 4 MW pivots are now ABORTED**. Per-token SAE baselines at
  C3/C5 will be re-trained with T=1 window-based pattern. C4 cache-
  hits on new C3 checkpoints. C6/C7 baselines unchanged (already at
  T=1). TXC archs unchanged everywhere.

- Tests: 131/131 (last verified before pivot decision).
- All 9 + 2 (MW) archs in YAML; framework integrity holds.

## Next action (post-compact) — optimize resource allocation

Han's directive for after compact: "optimize resource allocation for
full set of base results, and the synthetic benchmark crises."

### Two parallel decisions to make

**1. BASE results parallelization.** agent_steer_100k owns the
remaining BASE work (~18-22 hr serial on 1× H100):
- BASE C3 txc_base T=10/T=20 RE-TRAIN (6 cells, ~12-15 hr)
- BASE C3 k_feats expansion (~21 cells × 6 new k_feats, ~5-6 hr)
- BASE C4 qualitative eval (~1-2 hr, cache-hit on TXC checkpoints)

Question to triage post-compact: **who helps?**
- **agent_filler** (8× A40) once their C2 cleanup wraps (~30-50 min):
  could parallelize the BASE k_feats expansion across 8 GPUs (eval-
  only, fast). Wall drops from ~6 hr → ~1 hr.
- **agent_nlp** (1× H100, recovered) once their 7-cell tail lands
  (~3.5 hr): could take the C4 BASE qualitative eval as a follow-on.
- **agent_em_100k** (1× H100) once their 6-cell borrow lands (~3 hr):
  status: idle. Could be redirected to BASE C3 k_feats expansion.

Recommendation (write up + push as new directives): split BASE k_feats
across agent_filler (8 GPUs) + agent_em_100k (1 GPU); agent_nlp does
C4 BASE eval; agent_steer_100k focuses on the txc_base T=10/T=20
re-train (which only THEY can do because they own the driver fix).

**2. Synthetic benchmark crisis.** agent_synth's Stage 1 ρ-sweep
should land in ~5-10 min after they read their briefing. Then:
- **If gAUC flat across ρ** → Effect 1 dominates → reframe C2 paper
  claim. Update `docs/components/c2.md` Hypothesis section to talk
  about "variance-reduced feature recovery" rather than "temporal
  pattern detection." Drop or downgrade the "T modulates" claim.
- **If gAUC grows with ρ** → Effect 2 confirmed → strong claim
  defensible. Decision tree wired in agent_synth's briefing.
- Then greenlight Stage 2 (DC/AC ablation) for mechanistic confirmation.

agent_paper's job: surface plot to Han; help decide framing; integrate
into c2.md.

### Other queued integration work

- **C3 IT analysis.py update**: when agent_nlp's 7 + agent_em_100k's 6
  k_feats cells land, agent_nlp re-renders c3.md with all 8 k_feats.
  agent_paper reviews + integrates into the headline narrative.
- **C7 extended magnitude grid ±{20-90}**: agent_back was assigned
  but their pod exploded. Re-assign to whoever has spare H100 / A40
  capacity post-BASE-completion. Lower priority than BASE + synthetic.
- **decisions.md § 18 (forthcoming)**: document the synthetic-crisis
  resolution (Effect 1 vs Effect 2 framing) once agent_synth's
  ρ-sweep + DC/AC ablation land. Will reference Dmitry's framework
  directly with attribution.

### What's done already this session (~last 2 days)

- Multi-layer cache framework (commit `98d0bb05`) for paper-faithful MLC.
- Detection scaffolding (commit `be59b7be`) — `detect_case_study()` +
  shuffle ablation. Cherry-picked from `origin/det-steer`.
- 4 detection mission directives (C5, C6, C7) at `1a12e647`.
- Bug fixes (`076ff2f5` TFA C1; `1ed4fde5` C3 train_fn).
- C3 k_feats expansion mission (4 briefings updated, `70de9f75`).
- agent_synth provisioned (`86f24c9c`) — full briefing, set_agent_env.sh
  registry entry, run_rho_sweep.sh launcher.
- BASE C3 mission split between agent_steer_100k + agent_filler (`9ea13c06`).
- ρ-sweep framework (4 new datasources + driver extension, `7bd38bfd`).
- Component md cleanup (`db2441ff`) — wasteland refs moved to bottom
  with `---` separators.
- HF repos flipped public (Han directive 2026-05-06).

**Past plan kept here for reference (executed):**

### Step 1 — stand-down directives (4 worker briefings)

Rewrite the Identity+mandate "Han decisions" sections in each of these
to abort the MW pivot and put the agent in `status: idle, awaiting
re-purpose` until step 3:

- `agents/agent_em/briefing.md` — kill the C6 MW pivot. They were
  pivoting after their canonical 8/8 DONE. Status: canonical mission
  complete, idle.
- `agents/agent_em_100k/briefing.md` — kill the C3 MW pivot. They had
  smoke-passed and launched a sweep (commit `4217f4ba`). Kill the
  in-flight process, prepare for re-purpose to C3 baseline re-train.
- `agents/agent_filler/briefing.md` — kill the C5 MW parallel sweep.
  They may have launched (`38c34972`). Kill, prepare for re-purpose
  to C5 baseline re-train.
- `agents/agent_steer_100k/briefing.md` — kill the C7 MW pivot
  (briefing rewritten in `72b65b2b`). They may not have launched
  yet; either way, abort. Status: idle.

### Step 2 — framework change for windowed batch_iter on C3/C5

Add T-window sampling support to the canonical helper at
`temp_bench.data.nlp.cache:preloaded_batch_iter_from_act_cache`:

```python
def preloaded_batch_iter_from_act_cache(
    act_cache_key: str, *, seed: int = 0,
    train_window_size: int | None = None,   # NEW: None = full sequence
                                            #      int = sample 1 random T-window per row
) -> Callable[[int], torch.Tensor]:
```

When `train_window_size` is set, sample 1 random `T`-window per row
(matching agent_em / agent_back's `_build_batch_iter` semantics).
When None, current behavior (full sequence) preserved — TXC archs at
C3/C5 keep using the full-sequence path; baselines opt in to T=1.

Add a corresponding `train_window_size: int | None = None` field to
`TrainingConfig` so it flows into `compute_train_key` automatically
(invalidates existing per-token baseline cells when set on the new
re-runs; old cells stay for diff comparison).

Test: extend `tests/test_preloaded_batch_iter.py` with a 2-test
addition verifying T=1 sampling matches the (B, T, d_in) shape and
that the helper returns deterministic values across seeds.

### Step 3 — repurpose 2 freed agents for the re-train

- **agent_em_100k** → C3 baseline re-train (per-token archs at
  T=1 window-based). Scope: TopK_SAE × 3 seeds × 2 k_feats = 3
  trainings + 6 evals. T-SAE × 3 seeds × 2 k_feats = same. MLC
  same. Total ~9 unique trainings + 18 evals. Per-cell at the
  smaller batch: ~10-15 min train + ~30 min eval per k_feat. Total
  ~9 hr. Fits.

- **agent_filler** → C5 T-SAE baseline re-train (per-token T-SAE
  at T=1 window-based). Scope: T-SAE × 3 seeds = 3 cells. With 8×
  A40 in parallel, can run 3 seeds simultaneously — ~3 hr wall.

- **agent_em** → idle, paper writing for C6 caveats / methodology.
  Their canonical mission is complete; no need for further compute.
- **agent_steer_100k** → idle, paper writing for C7 caveats. Their
  pod has the slow CPU-bandwidth issue; not productive for sweeps.
  agent_back's canonical C7 sweep stands as the headline.

### Step 4 — decisions.md update

- Mark § 14 (MW deployment) as **DEPRECATED 2026-05-05 PM**. Add a
  note: "MW deployment was correct fix for the C3/C5 per-step
  asymmetry, but the more honest fix is to bring C3/C5 baselines
  DOWN to literature scale (per-token-SAE B=1024 tokens/step,
  matching SAEBench's 2K canonical scale within a factor of 2).
  All 4 MW pivots aborted; YAML aliases stay registered as inert
  reserves for post-paper revisitation."
- Add new **§ 15 — Literature-aligned T=1 baseline re-train**.
  Capture: (a) the SAEBench reference (App. B, batch=2048 tokens,
  500M tokens total), (b) the framework change (window_size param
  in preloaded_batch_iter_from_act_cache + TrainingConfig), (c) the
  re-train scope (C3+C5 baselines), (d) the expected paper-claim
  shifts (TopK / T-SAE will likely score worse at canonical
  training; comparisons vs TXC may narrow or flip in TXC's favor).

### Step 5 — agent_nlp + agent_steer briefing notes

agent_nlp owns C3 — they need to know their canonical C3 sweep's
per-token archs (TopK / T-SAE / MLC, possibly Stacked / TFA / SAE-arditi)
will be RE-RUN by agent_em_100k under the new T=1 pattern. Their
original results stay in the leaderboard for diff. Their analysis.py
will need to use `canonical_train_keys()` with the new
`TrainingConfig(train_window_size=1)` to filter to the re-run cells.
Add a directive section to `agents/agent_nlp/briefing.md` Han-decisions
block.

agent_steer owns C5 — same situation but only T-SAE is affected
(TXC archs unchanged). Add a directive section to
`agents/agent_steer/briefing.md`.

## What I just did (agent owns — overwrite)

Newest first.

- **2026-05-06 PM** (this session, post-compact): provisioned
  `agent_synth` for the synthetic investigation. New 8× 5090 pod.
  Reassigned the C2 ρ-sweep from agent_filler → agent_synth.
  - `agents/agent_synth/briefing.md` — full mandate, 4-stage
    investigation plan (ρ-sweep → DC/AC ablation → Dmitry's bench D
    + temporal_derivative_v2 reproductions, last two stretch).
  - `scripts/set_agent_env.sh` — agent_synth case + warning string.
  - `experiments/c2_synthetic_coupled/run_rho_sweep.sh` — launcher
    hardcoding `AGENT_NAME=agent_synth`; refuses to run mismatched.
  - agent_filler briefing: ρ-sweep section marked `↪️ REASSIGNED`
    for context.
- **2026-05-06 PM**: split the 13 unfinished C3 k_feats cells between
  agent_nlp (7) + agent_em_100k (6) post-RunPod-rescue. agent_nlp
  takes their seed=2 tfa tail + txc_base T=20 high-k tail; agent_em_100k
  borrows tfa territory for seed=42 × all 6 missing k_feats.
- **2026-05-06**: fixed C3 train_fn `arch_hparams_override` bug
  (commit `1ed4fde5`). Was causing all BASE C3 txc_base T=10/T=20
  evals to crash with state_dict size mismatch. agent_steer_100k
  briefing updated with re-train + re-eval directive.
- **2026-05-06**: fixed TFA `_is_valid_cell` at C1 (commit `076ff2f5`)
  in response to agent_filler's bug report. TFA at toy d_sae=40
  with T=5 default has k_train=k_pos×5 → must be ≤ d_sae. The 21/36
  cells agent_filler landed are the complete valid set; missing 15
  high-k cells are architecturally infeasible.
- **2026-05-06**: cherry-picked detection scaffolding from
  origin/det-steer (commit `5b7d2fc3` + `1a12e647`). C5 + C6 + C7
  detection missions written. All three agents executed before pod
  incident: C5 detection 15 cells, C6 detection 8 cells, C7 refactor.
- **2026-05-06**: HF repos flipped public via
  `update_repo_settings(private=False)`. No worker disruption.
- **2026-05-05 PM**: cross-component analysis of agent_filler's bug
  report (TFA C1) — bug isolated to C1 toy d_sae=40; production-scale
  C3/C5/C6/C7 unaffected.
- **2026-05-05 PM**: C3 k_feats expansion mission (Han urgent): {5,
  10, 20, 40, 80, 160, 320, 640} on IT + BASE. 4 briefings updated
  (agent_nlp IT, agent_steer_100k BASE, agent_em_100k MLC,
  agent_filler ack). Eval-only; cache-hits on training.

(Older entries below — pre-incident. Mostly historical context now.)


Newest first.

- 2026-05-05 PM (post-compact): EXECUTED the 5-step abort + re-train
  plan in 3 commits.
  - `dd5f773e` STAND DOWN — wrote ABORT blocks at top of "Han
    decisions" in all 4 MW briefings. Agents on remote pods saw it
    on next pull and acted (agent_em_100k killed PID 17963;
    agent_em acknowledged + wrapped up canonical mission).
  - `15ad0de4` framework change — `train_window_size: int | None`
    on `preloaded_batch_iter_from_act_cache` + `TrainingConfig` +
    5 new tests. Switched `compute_train_key` to
    `model_dump(exclude_none=True)` so default-None preserves all
    existing train_keys (load-bearing: in-flight C3 topk_sae sweep
    + C7 v4 sweep keep their cache). 136/136 tests green.
  - `74dc2cd9` directives + plumbing — decisions § 14 deprecated +
    § 15 added (per-arch literature-faithful T sizes:
    topk_sae @ T=1, tsae_paper @ T=2 from Bhalla/Ye 2025 §3.1
    paper-faithful adjacent pairs); 4 briefing rewrites
    (agent_em_100k → C3 T-SAE, agent_filler → C5 T-SAE, agent_nlp +
    agent_steer get directive notes); 2 driver pass-throughs
    (`c3_probing/run.py` and `c5_steering/run.py` both pipe
    `training_cfg.train_window_size` to the helper;
    `run_one_cell` accepts the kwarg).
- 2026-05-05 PM: Han redirected mid-execution: "agent_nlp's other
  H100 is free since agent_em is idle; therefore agent_nlp and
  agent_em_100k should work TOGETHER to recover the baselines no?"
  → split C3 baseline re-train: agent_nlp takes TopK T=1 on borrowed
  GPU 1 (their existing TopK T=None sweep on GPU 0 finishes as
  diff-reference); agent_em_100k takes T-SAE T=2 only.
- 2026-05-05 PM: Han redirected on T-SAE: "we want tsae T=2 for C3 C4
  and C5; for C7 can leave at T=5; both need to be supported." →
  framework already supports both via the Optional `train_window_size`
  field. Two TSAE train_keys live in the leaderboard (T=2 for C3/C4/C5,
  T=5 for C7's existing convention).
- 2026-05-05 PM: Han clarified: "I meant 1024 BATCH SIZE not effective
  tokens! for everything!" → `batch_size=1024` uniform across archs;
  per-arch T varies per-arch literature spec.
- 2026-05-05 ~11 AM: refreshed agents/README.md to current state
  (commit `0c62caa3`), updated agent_filler briefing (commit
  `305cf279`), agent_steer_100k pivot to C7 MW (commit `cd3020c9`).
- 2026-05-05 AM: peft added to pyproject + uv.lock (commit `4a087428`).
- 2026-05-05 AM: TXC multi-window toggle + tests landed (commit
  `f88ff32f`); txc_base_mw / txc_pro_mw YAML aliases (commit
  `ecc4c661`); local 5090 benchmark verified 5× more tokens/step at
  1.26× wall (commit `d724241c`); decisions § 14 Bricken caveat
  (commit `cad94382`).
- Earlier: canonical_train_keys helper landed (commit `9a39137a`);
  set_agent_env stale-GPU scrub (commit `ea91c83e`); 4-agent MW
  deployment plan (commit `305cf279`).

## Don't repeat (agent owns — overwrite)

- **Wasteland**: code on `origin/han-phase7-unification`; read via
  `git show`, never import.
- **Cwd**: always work from `purified/`.
- **Cache**: `runner.run_cell` computes train_key / eval_key
  deterministically; never allocate manually. Bumping `arch_version`
  invalidates train cache; bumping `EVAL_PROTOCOL_VERSION` invalidates
  eval cache. Adding new fields to `TrainingConfig` (e.g., `train_window_size`)
  flows into the train_key hash automatically.
- **Briefing/log/handover**: one rolling `briefing.md` per agent,
  section-owned. No `log.md`. Don't edit Han's top section unless
  Han approves.
- **Cross-territory edits**: Hard Rule #7. agent_paper integrates
  worker briefings only when Han approves a directive. Other agents
  don't touch each other's experiments/ dirs.
- **MW deployment was a misframe** — don't re-launch the pivots.
  The right fix is at the data path / TrainingConfig level, not at
  the arch level.

## Open questions for Han (agent owns — overwrite)

(All resolved as of 2026-05-05 PM directive. Surface anything new
that comes up during plan execution.)

