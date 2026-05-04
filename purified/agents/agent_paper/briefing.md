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

**Last verified: 2026-05-03T22:30:00Z**

Han's "first mission" decisions are now LOCKED in `decisions.md` #1–10.
Framework is built; tests 38/38 green; HF repos + tokens unified.
**4 worker briefings drafted this session — pending Han review** before
he spins up the worker pods.

- `final` branch pushed. `git log` for history. ~14 commits on day 0.
- 7 components have `docs/components/cN.md` writeups (planning status).
- 5 named agents in roster + 1 fallback. **Only agent_paper (me) is
  active**; NLP/EM/STEER/BACK have draft briefings, awaiting Han's
  spin-up. EM_H200 dormant.
- Locked archs: TXC-base = `txc_bare_antidead_t5`,
  TXC-pro = `phase5b_subseq_h8`. **9 yaml entries** (added
  `stacked_sae` 2026-05-03 — was a c1.md/c7.md baseline missing from
  YAML): topk_sae, stacked_sae, tsae_paper, tfa, tfa_pos, mlc,
  sae_arditi, txc_base, txc_pro.
- `results/leaderboard.jsonl` empty. `checkpoints/manifest.jsonl` empty.
- Token store: `~/.tokens/` (local) + `/workspace/.tokens/` (RunPod).
- **8 distinct arch .py files NOT yet ported** (tfa + tfa_pos share a
  file). YAML entries exist; classes don't.

## What I just did (agent owns — overwrite)

Newest first. `git log` for full history.

- **Caught Wang abbreviation oversight** (Han-flagged 2026-05-04):
  agent_em's C6 cells used "abbreviated Wang" (skip stages 2 + 3,
  top-3 by Δz̄, 6-α grid) without an explicit decision. The +3.79
  align gap is the C6 headline but is methodologically suspect.
  Updated agent_em's briefing with directive to (a) re-run all C6
  cells with FULL Wang protocol, (b) add 7B-medical re-run to pair
  apples-to-apples with the 14B numbers (replacing wasteland-citation
  reliance on Dmitry's published 7B), (c) use both H100s in parallel
  via formal `claim_gpu` calls. agent_nlp at `status: complete` so
  GPU 0 is available.
- **GPU-lock UX gap**: agent_steer's commit `0c885c98` bypassed
  `claim_gpu` for a parallel launch ("gentleman's agreement only").
  Means agents are finding the API awkward, especially for
  background-launched cells (`subprocess.Popen` lifecycle vs
  `with claim_gpu(...)` block). Open question for me to fix:
  `claim_gpu_until_pid_exits(idx, pid)` helper OR
  `scripts/run_with_gpu_claim.sh <idx> -- <cmd>` wrapper. agent_em's
  briefing notes this; if they hit the same friction, I implement.
- **All 4 worker overnight progress absorbed** (commits since
  5ab98bb3): C3 final headline (4 archs × 3 seeds × 2 k_feats = 24
  cells; topk_sae wins at both k); C4 final (T-SAE >> TXC on
  Top-256 SEMANTIC); C5 partial sweep (T-SAE ~2× TXC on success
  rate); C6 paper-correct + small-TXC (mean gap +3.79 / +6.35,
  Mixed); C7 still in flight. **Story has materially shifted from
  hypothesized — paper needs reframe**.
- **Pushed back on agent_nlp's cross-territory edits** (Han-flagged):
  agent_nlp's commit `e68e3146` edited `decisions.md`, my briefing,
  agent_steer/briefing, and `docs/paper/architecture.md`. Content
  was mostly fine and I left it (SAEBench+CT n=38 is a real win,
  Bricken-off-by-default is consistent with my decision #7), BUT
  the procedure was wrong — Hard Rule #7 forbids cross-agent edits.
  Restored Cohen's κ (≥ 0.6) as primary judge-validation metric in
  c4.md (agent_nlp had swapped it for raw % only — Han flagged).
  Strengthened CLAUDE.md Hard Rule #7, added explicit
  "Files OUT OF SCOPE" block to all 4 worker briefings + the
  briefing template.
- **Code reuse contract** (PROTOCOL.md § 11): unified trainer
  (`temp_bench.training.train_sae`), eval module stubs, ABC
  `train_step` + `post_step`, `instantiate_arch`, structural
  registry tests. 51/51 green.
- **Results binding** (`temp_bench.report` + AUTO-RESULTS markers):
  numbers in `cN.md` flow from `analysis.py` → leaderboard, never
  hand-typed. Hard Rule #10.
- **C7 → Llama-3.1-8B** (paper-faithful Ward et al.); per-component
  d_sae=32768 overrides; T-SAE locked to faithful Ye port (Hard #11);
  rewrote c7.md + agent_back briefing.
- **Drafted 4 worker briefings** for {agent_nlp, agent_em, agent_steer,
  agent_back}. agent_nlp's session is the first that revealed scope
  enforcement gaps — addressed above.
- **Consistency fixes (post-compact verification)**: registered
  `stacked_sae` in `configs/locked_archs.yaml` (was a baseline in
  c1.md/c7.md but missing from YAML — Hard Rule #2 violation); fixed
  briefing's port-list (8 distinct files, 9 yaml entries); fixed
  briefing's decisions.md #6 → #2+#7 cross-ref for agent_em.
- Briefing/handover/log consolidation (#10): one `briefing.md` per
  agent, section-owned. Dropped dated handover archive + log.md.
  PROTOCOL.md § 14 rewritten to "Briefing maintenance".
- C4 simplified to single metric: Top-256 cumulative SEMANTIC Pareto.
- Wasteland code deleted; wasteland docs retained.
- Token storage unified across local + RunPod (`get_token()`).
- GPU sharing: Primary + Pool with lockfile claims (PROTOCOL.md § 13).
- Modularity framework + cache contract (`docs/paper/framework.md`).
  38/38 tests.

## Component dependencies + provisioning roadmap (agent owns — overwrite)

This is the raw material for Han's two pending asks:
**(i) decide spawn order, (ii) write each agent's briefing.**

### Component dependency graph

```
C1 (toy markov)           ─ agent_paper, local 5090         ─ no external deps
C2 (toy coupled HMM)      ─ agent_paper, local 5090         ─ no external deps
C3 (sparse probing)       ─ agent_nlp, H100 GPU 0           ─ builds Gemma-IT-L13 cache (~3 H100-hr)
C4 (top-256 semantic)     ─ agent_nlp (same agent)          ─ piggybacks on C3 cache
C5 (RLHF steering)        ─ agent_steer, A40 GPU 0          ─ NEEDS C3's cache from HF
C6 (EM Wang procedure)    ─ agent_em, H100 GPU 1            ─ separate Qwen-14B + LoRA
C7 (Ward backtracking)    ─ agent_back, A40 GPU 1           ─ separate Gemma-BASE-L10 cache
```

Cross-agent dependency: **C5 waits for C3's cache** (uploaded to
`han1823123123/temp-bench-data` after agent_nlp builds). All others are
independent.

### Recommended spawn order

- **T+0 (Han owns)**: on each shared pod, run bootstrap THEN clone for
  the second agent THEN spawn each agent via `start_agent.sh`.
  Per-agent clone keeps `.git/` independent (no `index.lock` races);
  `start_agent.sh` sources `set_agent_env.sh` in Han's parent shell so
  the GPU pin / `AGENT_NAME` / pod mode propagate into the claude
  process (an agent sourcing them in its own first action does NOT
  work — Bash tool calls don't share shell state).
  - 2× H100 pod:
    1. `bash scripts/bootstrap_runpod.sh` (creates `/workspace/temp_xc/`)
    2. `bash /workspace/temp_xc/purified/scripts/add_agent_clone.sh agent_em`
       (creates `/workspace/temp_xc_em/`)
    3. `bash /workspace/temp_xc/purified/scripts/start_agent.sh agent_nlp --fresh`
    4. `bash /workspace/temp_xc_em/purified/scripts/start_agent.sh agent_em --fresh`
  - 4× A40 pod:
    1. `bash scripts/bootstrap_runpod.sh`
    2. `bash /workspace/temp_xc/purified/scripts/add_agent_clone.sh agent_steer`
       (creates `/workspace/temp_xc_steer/` ahead of T+~3hr)
    3. `bash /workspace/temp_xc/purified/scripts/start_agent.sh agent_back --fresh`
       (agent_steer waits until cache lands)
  - agent_paper (me) continues C1+C2 locally.
- **T+~3 hr (after C3 cache uploaded)**:
  `bash /workspace/temp_xc_steer/purified/scripts/start_agent.sh agent_steer --fresh`
- **Re-launch after disconnect**: drop the `--fresh` —
  `start_agent.sh agent_X` defaults to `claude --continue` so the
  worker resumes its session.
- **Defer agent_em_h200** to fallback only (if R32 OOMs on H100).

Rationale: parallelize the long-pole work (C3 cache build, EM Wang
procedure, C7 backtracking), don't gate steering on its own pod-spinup
since the cache must exist first anyway.

### Per-agent first-task scaffolds (for writing briefings)

Each scaffold is what "Identity + mandate (Han owns)" + first concrete
task should contain. Han may rewrite the mandate prose; the technical
specifics (component, hardware, ports needed) are correct.

| Agent | Component(s) | Pod / GPU | First concrete task |
|---|---|---|---|
| **agent_nlp** | C3 + C4 | 2× H100 / GPU 0 | Port `temp_bench.data.nlp.cache_activations`, build Gemma-2-2b-IT L13 act-cache (24K seq × 128 tok ≈ 14 GB, ~3 H100-hr), upload to HF temp-bench-data. Then port + train 5 archs × 3 seeds. |
| **agent_em** | C6 | 2× H100 / GPU 1 | Port `temp_bench.case_studies.em` from `origin/em-nanda:experiments/em_features/run_wang_procedure.py`. Set up Qwen-14B-Instruct + finance LoRA. Implement Bricken resample. First run: TXC-base + brickenauxk on R1 30k mid-α (the gap-close test, decisions.md #2 reframing + #7 Bricken opt-in). |
| **agent_steer** | C5 | 4× A40 / GPU 0 | Port `temp_bench.case_studies.steering`. Set up Gemini judge (coh + success). Wait for agent_nlp's cache → `sync_from_hf.sh` → train TXC-base + TXC-pro + T-SAE → V7 tiled-broadcast steering protocol → coh-vs-success curves. |
| **agent_back** | C7 | 4× A40 / GPU 1 | Port `temp_bench.case_studies.backtracking` from `origin/aniket-ward-stage-b:experiments/ward_backtracking_txc/`. Build Gemma-BASE-L10 cache (own, not shared). Set up Sonnet judge + 20-transcript blind κ validation. Run inducement (Δgc) + detection (PR-AUC) on 5 archs × 3 seeds. |

For each, the briefing's "Identity + mandate (Han owns)" section
should reference: `agents/README.md` row, `docs/components/cN.md`,
relevant `decisions.md` entries, and the wasteland files to port.

## Next action (agent owns — overwrite)

Both worker briefings AND C1+C2 porting are in flight. Worker
briefings are drafted; await Han's review. Continue C1+C2 in parallel.

1. Bootstrap: `cd $(git rev-parse --show-toplevel)/purified` →
   `source scripts/set_agent_env.sh agent_paper` →
   `bash scripts/agent_smoke_test.sh` (expect 38/38) →
   `git pull --rebase origin final`.
2. Read `decisions.md` #1–10 if not in context.
3. **If Han has approved worker briefings**: notify him to spin up
   pods (T+0: nlp + em + back; T+~3hr: steer once C3 cache hits HF).
   If not yet, address his review feedback in
   `agents/{nlp,em,steer,back}/briefing.md` "Identity + mandate"
   sections.
4. **C1+C2 porting (in parallel)**:
   - Port arch classes from
     `origin/han-phase7-unification:src/architectures/` into
     `src/temp_bench/architectures/`. **8 distinct .py files** (tfa +
     tfa_pos share `tfa.py`, differ only by `use_pos_embedding`):
     topk_sae, stacked_sae, tsae_paper, tfa, mlc, sae_arditi,
     txc_base (← `txc_bare_antidead.py`),
     txc_pro (← `phase5b_subseq_sampling_txcdr.py`).
     Each port: `git show origin/han-phase7-unification:src/architectures/<file>.py >
     src/temp_bench/architectures/<file>.py`, then add header comment
     with the source commit hash, then update imports
     (`from src.architectures.base` → `from temp_bench.architectures.base`).
   - Implement `temp_bench.data.toy.markov_chain_support` (C1) +
     `coupled_hmm` (C2). Sources to study (NOT to import):
     `origin/han-phase7-unification:src/v2_temporal_schemeC/markov_data_generation.py`
     and Phase 3 coupled-features generators.
   - Write `experiments/c1_synthetic_topk/run.py`. Smoke-run:
     1 cell × 1000 steps. Then full sweep (~6 hr local on 5090).
5. After porting + C1+C2 land: start drafting paper outline for the
   sections you own (C1, C2 + framework introduction).

## Don't repeat (agent owns — overwrite)

- **Wasteland**: code is gone from `final`; read via `git show
  origin/han-phase7-unification:<path>`. Don't import from `src/`.
- **Architectures**: TXC-base + TXC-pro only (decisions #1).
  Bricken is opt-in per component (#7); default OFF except C6.
- **Cwd**: always work from `purified/` (`set_agent_env.sh` enforces).
- **Cache**: never allocate run-ids manually; `runner.run_cell`
  computes `train_key` / `eval_key` deterministically. Bump
  `arch_version` to invalidate train cache; bump
  `EVAL_PROTOCOL_VERSION` to invalidate eval cache.
- **Git**: always `git pull --rebase` before push. Never force-push.
- **Briefing/log/handover**: one rolling `briefing.md` per agent,
  section-owned. No `log.md`, no `handovers/`. Don't edit Han's top
  section.
- **Component writeups vs agent briefings**: technical setup goes in
  `docs/components/cN.md`; agent briefings reference, do not duplicate.

## Open questions for Han (agent owns — overwrite)

1. **Worker briefings**: 4 drafts written. Please review the
   "Identity + mandate" sections in each of
   `agents/{nlp,em,steer,back}/briefing.md` and rewrite as you wish
   before spinning up the pods. Areas where I had to guess voice
   rather than copy directly from your earlier prose: agent_em's
   coordination with Dmitry, agent_steer's gating language on the
   cache, agent_back's phrasing of "C7 is a candidate paper-headline
   result."
2. **Spawn order**: confirming parallel start of nlp + em + back at
   T+0 (with steer at T+~3hr after C3 cache uploads). agent_steer's
   briefing tells the worker to wait on the cache before bootstrap.
   Acceptable, or stagger?
3. ~~**C3 task suite**: still TBD~~ — RESOLVED 2026-05-03 with Han.
   Locked as **SAEBench+CT** (upstream SAEBench's 36 binary tasks +
   WinoGrande + WSC = 38). See `decisions.md` § 11; agent_nlp briefing
   updated. Three SAEBench-faithfulness deltas (github-code provider,
   amazon_sentiment 1.0, amazon_categories cat6) tracked as agent_nlp
   TODOs. github-code requires `trust_remote_code=True` + `datasets<4`
   (pinned in pyproject.toml 2026-05-03); not actually HF-gated — the
   web viewer is disabled because the loader is a Python script, but
   the dataset is public.
4. **agent_em_h200 fallback**: provisioned dormant in roster — wait
   for R32 OOM, or stand up proactively?
5. **A 5th worker for the 4× A40 pool**: GPUs 2 + 3 are spare. Could
   land C7 detection sweeps or C5 multi-seed faster. Defer until
   after first results land?
