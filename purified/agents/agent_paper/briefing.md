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

**Last verified: 2026-05-04 mid-day, post-batch_size walkback**

All 4 worker pods active and in-flight. Component results landed (or
re-landed, post-fixes) for C3/C4/C5/C6 partial, C7 partial. Tests
116/116. All 9 archs ported. ~9 distinct yaml entries → all
importable. Decisions #1-12 locked.

**Open question post-compact me must handle FIRST**:

Han's pushback on the batch_size CRITICAL alert — see §
"⚠️ UNRESOLVED: training-budget question" below. This is the live
issue. agent_nlp's CRITICAL alert (commits 579efb9a, 8904414d,
3558b303) flagged that all C3-C7 cells used batch=256 vs Phase 7's
4096. I issued a global "re-train at batch=2048" directive
(commit a9200560), Han pushed back: "Dmitry/Aniket might have been
undertraining too — they're not the right reference; we should think
about this from a fresh frame." **The directive is on origin/final
but the underlying decision is unsettled.** Post-compact me: read
this section + decisions.md § 12 + recent commits, then re-engage
with Han.

State of the leaderboard (as of 2026-05-04):
- C3 (agent_nlp, status: complete): 24 cells × 2 protocol versions
  (v1.0.0 buggy padding + v1.1.0 fix). Headline: TopK-SAE 0.9044
  > TXC > T-SAE 0.8844. Honest negative for C3 hypothesis.
- C4 (agent_nlp, status: complete): 9 cells. T-SAE 74.7 >> TXC-pro
  60 >> TXC-base 42 on Top-256 SEMANTIC. Honest negative.
- C5 (agent_steer, status: complete): 9 + backfill cells with new
  peak_success_grade_at_coh_τ metric (post-Han's "horizontally
  aligned" catch). Backfill via reaggregate_from_judge_outputs.
- C6 (agent_em, in flight): full Wang stages 2+3 + 7B-medical port
  shipped (commit 144a3e84). Calibration cells running on H100s.
  +3.79 align Mixed gap from abbreviated Wang stays as reference.
- C7 (agent_back, in flight): 1 smoke cell (topk_sae +0.36 Δgc) +
  full sweep launched. The +1.574 reproduction test is the
  remaining "win" candidate.

Workers' batch_size status (every cell, all pods):
batch=256 across C3/C5/C6/C7. Schema default was 256, just bumped
to 2048 in a9200560 — which Han is now pushing back on. May be
reverted depending on the fresh-frame discussion.

**Locked archs** (all 9 yaml entries have a class file):
- TXC-base = `txc_bare_antidead_t5`, TXC-pro = `phase5b_subseq_h8`.
- Baselines: topk_sae, stacked_sae, tsae_paper (Ye port; NOT
  tsae_ours which is forbidden per Hard Rule #11), tfa, tfa_pos, mlc,
  sae_arditi.
- Token store: `~/.tokens/` (local) + `/workspace/.tokens/` (RunPod).

## ⚠️ UNRESOLVED: training-budget question (compact-priority)

agent_nlp identified the batch=256 issue (commits 579efb9a +
8904414d + 3558b303). My initial response: bump default to
batch=2048 with per-pod overrides (C3/C4/C6 H100 → 2048; C5/C7 A40 →
1024). Committed in a9200560 + decisions.md § 12.

**Han's pushback (the part I haven't fully addressed)**: the
underlying assumption — that Dmitry's batch (for C6) and Aniket's
batch (for C7) are the right references — is not actually
established. Dmitry and Aniket may have been undertraining their
own runs too. The CORRECT question is not "match their batch" but
"what hyperparameter setup would actually produce the strongest
honest result for this paper given our 72-hour budget?"

**What needs fresh thinking**:

1. **Per-arch token budget** for "well-trained":
   - TopK-SAE: usually 100-200M tokens.
   - T-SAE (Bhalla/Ye): paper-trained at ~100M tokens.
   - TXC family: similar order.
   - Bricken-recipe TXC: ~30M+ tokens with periodic resamples.
   - At batch=256, n_steps=30K = 7.7M tokens — universally undertrained.
   - At batch=2048, n_steps=30K = 61M tokens — OK-ish.

2. **Per-component re-think** (not "match Dmitry"):
   - C3/C4: T-SAE paper used standard SAE-training batch ≈ 4096.
     For T-SAE in our run to even have a chance, we should use
     similar budget. Current batch=256 likely under-represents
     T-SAE.
   - C5: T-SAE § B.2 doesn't pin SAE training batch. Our cells use
     C3's checkpoints + steer.
   - C6: Dmitry's published TXC k=100 30k might also be undertrained.
     The right ref might be "what does SAE-arditi need to train
     properly" — its 100k step number suggests it benefited from
     more compute than 30k. agent_em's choice is per-component.
   - C7: Aniket's reference is similarly thin. The +1.574 result was
     from his hill-climbed TXC at unknown batch. Re-deriving with
     locked TXC-pro at PROPER batch is the right re-test.

3. **What "right" means**:
   - **Fairness within a component**: every arch in a component uses
     the same budget. (Already true.)
   - **Sufficiency vs the published reference**: each arch trains
     ENOUGH. (Currently violated for T-SAE / TXC-pro at batch=256.)
   - **NOT "match wasteland"**: wasteland configs reflect time
     pressure, not the right answer.

4. **Practical compute math**:
   - 6 cells × 4 components × 8× compute (batch=2048 vs 256, fewer
     n_steps to keep token-equivalent) = ~80-150 GPU-hours total.
   - Within remaining 72h window IF started immediately.
   - 80-150 GPU-hours OR document in caveats and ship as-is.

5. **Decision tree post-compact me should walk Han through**:
   - **(a) Stay batch=256**: ship undertrained numbers; document in
     caveats; relative orderings may be biased toward TopK / against
     contrastive archs.
   - **(b) Re-train at batch=2048** (where it fits) / 1024 (A40):
     ~80-150 GPU-hours; orderings might shift.
   - **(c) Per-component judgment**: agent_paper investigates each
     reference, recommends per-component, agents re-train selectively.
     Most rigorous, slower.

6. **Fresh-start path Han wants**: start by NOT assuming wasteland
   configs are right. Ask: what does a well-trained T-SAE need? a
   well-trained TXC-pro? Then check: do we have time + compute for
   that? Then decide.

**My current pushed state**: schema default = 2048, decisions.md
§ 12 says "re-train all C3-C7", worker briefings have CRITICAL
blocks. **None of this has actually been executed by workers yet
— they're still mid-flight on batch=256 cells.** Post-compact me
can either revert the directive cleanly (revert a9200560) or refine
it. Han has explicitly said this needs fresh thinking; do that
before any workers act on the batch=2048 directive.

## What I just did (agent owns — overwrite)

Newest first. `git log` for full history.

- **Issued + walked back batch_size CRITICAL** (Han pushback,
  unresolved). See § ⚠️ UNRESOLVED above. Post-compact me re-engages.
- **Nuked the GPU lock system** (Han approved 2026-05-04). Deleted
  `temp_bench/utils/gpu_locks.py` + `tests/test_gpu_locks.py` + all
  `claim_gpu` references. Replaced with a CONVENTION in PROTOCOL.md
  § 13: each agent's primary stays pinned via `set_agent_env.sh`;
  to borrow a peer's GPU, read their briefing's "Current state" +
  run `nvidia-smi`, update your own state with the borrow + ETA.
  Added `scripts/run_on_gpu.sh <idx> -- <cmd>` convenience wrapper.
  Why: agents were already bypassing the lock system (agent_steer's
  `0c885c98` "gentleman's agreement only"); the cognitive cost +
  `subprocess.Popen` footgun outweighed the benefit at our 72-hour
  2-agents-per-pod scope.
- **Caught Wang abbreviation oversight** (Han-flagged 2026-05-04):
  agent_em's C6 cells used "abbreviated Wang" (skip stages 2 + 3,
  top-3 by Δz̄, 6-α grid) without an explicit decision. The +3.79
  align gap is the C6 headline but is methodologically suspect.
  Updated agent_em's briefing with directive to (a) re-run all C6
  cells with FULL Wang protocol, (b) add 7B-medical re-run to pair
  apples-to-apples with the 14B numbers, (c) use both H100s in
  parallel via the new GPU sharing convention. agent_nlp at
  `status: complete` so GPU 0 is available to borrow.
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
- GPU sharing convention (PROTOCOL.md § 13; lockfile system removed 2026-05-04).
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
C5 (RLHF steering)        ─ agent_steer, A40 GPUs 1+3       ─ NEEDS C3's cache from HF
C6 (EM Wang procedure)    ─ agent_em, H100 GPU 1            ─ separate Qwen-14B + LoRA
C7 (Ward backtracking)    ─ agent_back, A40 GPUs 0+2        ─ separate Llama-3.1-8B BASE-L10 cache
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
| **agent_steer** | C5 | 4× A40 / GPUs 1+3 | Port `temp_bench.case_studies.steering`. Set up Gemini judge (coh + success). Wait for agent_nlp's cache → `sync_from_hf.sh` → train TXC-base + TXC-pro + T-SAE → V7 tiled-broadcast steering protocol → coh-vs-success curves. |
| **agent_back** | C7 | 4× A40 / GPUs 0+2 | Port `temp_bench.case_studies.backtracking` from `origin/aniket-ward-stage-b:experiments/ward_backtracking_txc/`. Build Llama-3.1-8B BASE-L10 cache (own, not shared). Set up Sonnet judge + 20-transcript blind κ validation. Run inducement (Δgc) + detection (PR-AUC) on 5 archs × 3 seeds. |

For each, the briefing's "Identity + mandate (Han owns)" section
should reference: `agents/README.md` row, `docs/components/cN.md`,
relevant `decisions.md` entries, and the wasteland files to port.

## Next action (agent owns — overwrite)

**Most stale info above is obsolete; the current priority is the
unresolved batch_size / hyperparameter question.** Read § ⚠️
UNRESOLVED at the top first.

1. Bootstrap: `cd $(git rev-parse --show-toplevel)/purified` →
   `source scripts/set_agent_env.sh agent_paper` →
   `bash scripts/agent_smoke_test.sh` (expect ~116 pass) →
   `git pull --rebase origin final`.
2. Read `decisions.md` #1-12 (esp. § 12 batch_size — the open one).
3. Read recent commits: `git log --oneline -20` — covers the C5 metric
   fix, GPU-lock nuke, wrap_up_session.sh, batch_size CRITICAL +
   walkback. `git show 579efb9a 8904414d 3558b303 a9200560` is the
   batch_size thread.
4. **Re-engage Han on the hyperparameter question.** He explicitly
   said: "Dmitry/Aniket may have lacked hyperparam optimization too —
   they're not the right reference; think fresh." The question to
   answer with him:
   - Per-arch: what token budget makes T-SAE (Bhalla port) actually
     converge? what about TXC-pro (matryoshka + multi-distance
     contrastive)? what about TopK / SAE-arditi / MLC? Is there a
     single number, or per-arch?
   - Per-component: what's the practical compute budget remaining
     in the 72-h window? How many cells × seeds × archs can we
     re-train? Where's the marginal value?
   - Decision: ship as-is with caveat? Re-train selectively
     (e.g., only T-SAE + TXC-pro, the contrastive-loss archs)?
     Re-train everything?
5. Whatever Han decides, encode in decisions.md § 12 (replacing the
   current text with the resolved version) + sync the worker
   briefings. If we revert: `git revert a9200560` is the clean path
   (it touches schema + decisions + 4 briefings; reverting also undoes
   their CRITICAL blocks). If we refine: targeted edits.
6. Continue monitoring worker progress: agent_em mid-flight on
   calibration cells, agent_back mid-flight on C7 sweep, agent_nlp
   complete, agent_steer complete. Don't tell agents to start
   re-trains until the batch decision is resolved.
7. **C1+C2 (my own components) — DEFERRED until #4 resolves.**
   Toy data generators + run.py + sweep are still on my queue but
   not blocking workers. Phase 7's toy archs (topk_sae,
   stacked_sae, tfa, txc_base, txc_pro) are all ported.

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

**THE ONE OPEN QUESTION** (post-compact me, start here):

1. **Hyperparameter budget — fresh frame.** We landed a "re-train at
   batch=2048" CRITICAL directive (commit a9200560), then I walked it
   back when Han pointed out wasteland configs (Dmitry/Aniket) are
   not necessarily the right reference. The underlying issue is real
   (batch=256 × n_steps=30K = 7.7M tokens is genuinely undertrained
   vs T-SAE paper's ~100M and Phase 7's ~100M). But the FIX requires
   thinking from scratch: per-arch, what does well-trained look like?
   per-component, what's our compute budget? do we ship as-is with
   caveats, re-train selectively (only T-SAE + TXC-pro since they're
   the contrastive archs that hurt most from small batch), or re-train
   everything? See § ⚠️ UNRESOLVED at the top of this briefing for the
   structured walkthrough.

   Workers are still mid-flight on batch=256; the directive is on
   origin/final but they haven't acted on it yet. So we have a clean
   window to revert and re-decide.

   **Action**: re-engage with Han, walk through (a)/(b)/(c), encode
   resolution in decisions.md § 12 (replacing current text), revert
   a9200560 if appropriate.

Older open questions (mostly stale — do not re-pose to Han):

2. ~~Worker briefings drafts~~ — workers spawned + working.
3. ~~Spawn order~~ — done.
4. ~~C3 task suite~~ — RESOLVED (decisions.md § 11, SAEBench+CT n=38).
5. **agent_em_h200 fallback**: still dormant. R32 hasn't OOMed on
   H100. Defer until needed.
6. ~~A 5th worker for the 4× A40 pool~~ — RESOLVED. Han partitioned
   the A40 pod 2026-05-04 PM: agent_back gets 0+2, agent_steer gets
   1+3. Pod is fully allocated, no spare slots, no 5th worker.
7. **Meta HF approval**: Han applied 2026-05-04. Once it lands,
   agent_back runs check_mirror_equivalence.py (in their briefing
   directive) before re-running C7 on the canonical Llama datasource.
8. **Paper reframe**: 3 of 5 measured components are honest negatives
   (C3 ties; C4 + C5 reverse the hypotheses); 1 Mixed (C6); C7 TBD.
   Han hasn't asked for a reframed outline yet but it's coming.
