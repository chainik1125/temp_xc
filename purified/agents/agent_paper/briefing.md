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

**Last verified: 2026-05-03T21:00:00Z**

Han's "first mission" decisions are now LOCKED and recorded in
`decisions.md` #1–9. Don't re-open. Concretely:

- `final` branch created + pushed. ✓
- 7 components identified, each with a `docs/components/cN.md` writeup. ✓
- `purified/` layout: `temp-bench` is `src/temp_bench/case_studies/{steering,em,backtracking}.py` with a `CaseStudy` ABC. ✓
- 5 active agents (paper, nlp, em, steer, back) + 1 fallback (em_h200). 4× A40 has 2 spare pool GPUs. ✓
- Runpod scaffolding: `purified/scripts/{bootstrap_runpod,bootstrap_local,set_agent_env,sync_from_hf,agent_smoke_test,wasteland_refresh}.sh`. ✓
- Cross-agent state: `runner.run_cell` → `results/leaderboard.jsonl` (append-only, schema-validated, flock). ✓
- Writeup structure: by component (`docs/components/cN.md`), not by agent. ✓
- Two TXCs locked: TXC-base = `txc_bare_antidead_t5`; TXC-pro = `phase5b_subseq_h8`. ✓

Framework state:
- `git HEAD`: pending the next commit (this state-update lands in it).
- All 38 framework tests passing in `purified/.venv` (just rebuilt).
- `results/leaderboard.jsonl`: empty (no cells run).
- `checkpoints/manifest.jsonl`: empty.
- HF repos provisioned: `han1823123123/temp-bench-{models,data}` (private).
- Token store unified: `~/.tokens/{hf_token,anthropic_key}` on local;
  `/workspace/.tokens/...` on RunPod. `get_token()` resolves both.
- No worker agents (NLP, EM, STEER, BACK, EM_H200) are provisioned yet —
  briefings TODO when Han spins them up.

## What I just did (agent owns — overwrite)

Ordered newest-first; "agent_paper has spent ~1 day on this" boils down to
~10 commits on `final`. See `git log` for the full history.

- Switched briefing-vs-handover design: dropped dated handover archive,
  consolidated into THIS briefing with section ownership (Han's section
  read-only, agent's section overwritten at every compact). Per
  PROTOCOL.md § 14.
- Dropped `agents/agent_paper/log.md` — overhead with marginal benefit.
  Git log + `decisions.md` + this briefing's "What I just did" are
  enough.
- Simplified `c4.md` to single metric: Top-256 cumulative SEMANTIC Pareto
  (was: 3 metrics for "ablation honesty"; Phase 6 showed they ranked
  archs differently — drop pdvar + paper-style probe).
- Wasteland code deletion (`src/`, `experiments/`, `references/`, etc.
  gone from `final`). Wasteland docs (`docs/`, `papers/`) kept because
  component writeups cite them.
- Token storage unified: `~/.tokens/` on local, `/workspace/.tokens/` on
  RunPod. `get_token()` resolves both. Migrated `.env_autointerp` →
  `~/.tokens/anthropic_key`. New `bootstrap_local.sh`.
- GPU sharing protocol: Primary + Pool with lockfile claims
  (`temp_bench.utils.gpu_locks`). 6 tests, all passing.
- Modularity framework: configs as source of truth, deterministic
  cache keys (`act_cache_key` ⊃ `train_key` ⊃ `eval_key`), single
  canonical `runner.run_cell`. 23 tests originally; now 38 with token
  + lock additions.

## Next action (agent owns — overwrite)

1. `cd $(git rev-parse --show-toplevel)/purified`
2. `source scripts/set_agent_env.sh agent_paper`
3. `bash scripts/agent_smoke_test.sh` — expect 38/38 + 8 expected
   arch-class import gaps (those are the architectures we haven't
   ported yet — `configs/locked_archs.yaml`).
4. `git pull --rebase origin final`
5. Read this briefing top-to-bottom. Read `decisions.md` #1–9.
6. **Begin C1+C2 implementation work.** This is agent_paper's actual
   research mandate (not just orchestration). Concrete steps:
   - Port architectures from `origin/han-phase7-unification`:
     `git show origin/han-phase7-unification:src/architectures/<file>` →
     `src/temp_bench/architectures/<name>.py`. Add header comment
     with source path + commit hash. Architectures to port:
     `topk_sae`, `tsae` (T-SAE paper), `tfa`, `txc_base` (= wasteland's
     `txc_bare_antidead_t5`), `txc_pro` (= `phase5b_subseq_h8`),
     `stacked_sae` (for C1).
   - Implement `temp_bench.data.toy.markov_chain_support` (C1) and
     `coupled_hmm` (C2) generators.
   - Write `experiments/c1_synthetic_topk/run.py` from
     `_runner_template.py`.
   - Smoke-run: `txc_base` × seed 42 × k=2 × 1000 steps (~1 min on
     5090). Verify `run_cell` writes a leaderboard row + saves a
     checkpoint at `checkpoints/<train_key>/`.
   - Then full sweep (3 seeds × 12 k values × ~6 archs ≈ 6 hr local).
7. After C1+C2 are running, draft worker-agent briefings for NLP / EM /
   STEER / BACK so Han can spin them up.

## Don't repeat (agent owns — overwrite)

- **Don't re-create the root wasteland.** `src/`, `experiments/`,
  etc. are gone on `final`. Read via `git show origin/han-phase7-unification:<path>`.
- **Don't add a third TXC.** decisions.md #1: TXC-base + TXC-pro only.
- **Don't auto-add Bricken.** decisions.md #7: opt-in per component;
  only C6 enables by default.
- **Don't operate from repo root.** `set_agent_env.sh` will refuse;
  always `cd purified` first.
- **Don't allocate run_ids manually.** `runner.run_cell` computes
  `train_key`/`eval_key` deterministically. Bumping `arch_version`
  invalidates train cache; bumping `EVAL_PROTOCOL_VERSION` invalidates
  eval cache.
- **Don't push without `git pull --rebase` first.** PROTOCOL.md § 1.
- **Don't write a separate handover file.** PROTOCOL.md § 14 changed —
  update THIS briefing's bottom sections instead.
- **Don't create an `agent_paper/log.md` again.** Dropped intentionally;
  use git log + decisions.md.
- **Don't edit Han's "Identity + mandate" section above.** Read-only
  to agents.

## Open questions for Han (agent owns — overwrite)

(none currently — all framework decisions are locked. Han has been
driving the structural shape; if the chat has nothing new, proceed to
C1+C2 implementation per "Next action".)
