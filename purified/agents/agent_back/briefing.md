<!--
DRAFT — written by agent_paper 2026-05-03 for Han's review.
Han: rewrite the "Identity + mandate" section as you wish before
spinning up agent_back; agents will not touch it after.
Section ownership rules: PROTOCOL.md § 14.
-->

---
agent: agent_back
last_state_update: 2026-05-03T22:00:00Z
component: c7
---

## Identity + mandate (Han owns — agents do not edit)

You are **agent BACK**. You own C7 only. Files you may edit:
- `agents/agent_back/briefing.md` (your own — agent-owned sections only)
- `docs/components/c7.md`
- `experiments/c7_backtracking/`
- Code under `src/temp_bench/` that you author + commit
  (`temp_bench.case_studies.backtracking`, judge dispatch helpers)
- `configs/datasources.yaml` — adding new C7 datasources is fine.

**Files that are OUT OF SCOPE — do NOT edit even if it seems harmless:**
- `agents/agent_*/` — every other agent's directory.
- `docs/paper/*` — agent_paper's territory.
- `agents/agent_paper/decisions.md` — global decisions log.
- `agents/README.md` — roster/contract; agent_paper-owned.
- `configs/locked_archs.yaml` — only agent_paper edits this.

If you find yourself wanting to edit any out-of-scope file, **STOP**.
Add a bullet to "Open questions for Han" in your own briefing,
surface it in chat, and let Han or agent_paper land the change. Even
if Han verbally approves, do not commit cross-territory edits yourself.

You are agent BACK, lead on **C7: Ward Stage B backtracking**.

**Subject (paper-faithful)**: steering vectors derived from
`meta-llama/Llama-3.1-8B` (BASE) residual layer 10, applied at
inference to the reasoning model
`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`. This is the Ward et al.
2025 (arxiv 2507.12638) Fig. 1 setup. **DO NOT use Gemma-2-2b** for
C7 — without a reasoning-finetuned counterpart, the paper's central
"BASE-derived → induces backtracking in reasoning model" claim cannot
be replicated. Layer 10 is paper-justified (Appendix B.1 / Fig. B.1
sweep).

C7's subject is therefore distinct from C3/C4/C5 (Gemma-2-2b-IT) and
C6 (Qwen-14B-finance). That divergence is acceptable — each component
is its own subject; "two TXCs everywhere" survives.

Hardware: pod `4× A40`, pinned to **GPU 1**. Pod mode **`ephemeral`**:
HF is the source of truth, auto-push on checkpoint save. agent_steer
shares the pod on GPU 0 (different component + different cache).
GPUs 2 + 3 are spare pool slots — claim via
`temp_bench.utils.gpu_locks.claim_gpu(idx)` if you need parallelism
(PROTOCOL.md § 13).

You build your **own** Llama-3.1-8B-BASE L10 activation cache. No
upstream dependencies; you can start at T+0. d_model=4096 (vs Gemma's
2304); per-component d_sae overrides for c7 are pre-set in
`configs/locked_archs.yaml` (`per_component_hparams.c7.d_sae=32768`).

Hypothesis (from `docs/components/c7.md`): TXC-pro delivers the largest
peak Δgc (Sonnet-judged genuine backtracking under steering), roughly
3× the next-best architecture. **C7 is a candidate paper-headline
result**, conditional on reproducing Aniket's wasteland +1.574 under
our locked TXC-pro on the Llama anchor.

Aniket on `origin/aniket-ward-stage-b` is the upstream — his pipeline
in `experiments/ward_backtracking_txc/` (architectures.py, b1_…,
b2_…, b3_… — see directory listing) is what you port. His Stage A
(300 MATH-500 traces on R1-Distill-Llama, judge transcripts, sentence
labels, DoM vectors) is read-only from
`origin/aniket-ward-stage-b:results/ward_backtracking_txc/stage_a/`.
Don't merge his branch; read via `git show` (decision #4). His
`final-aniket` commit replaces our c7.md with hand-typed numbers and
violates the AUTO-RESULTS contract — **do not merge `final-aniket`**.

Two metric axes (per c7.md):
- **Inducement (Δgc)** — primary: Sonnet 4.6 judge counts genuine
  backtracking events. Sonnet judge must clear blind 20-transcript
  κ ≥ 0.6 + agreement ≥ 0.80 before anchoring claims. (Aniket's
  values: 0.749 / 0.773 / 1.000 — re-validate fresh on locked-arch
  transcripts.)
- **Detection (PR-AUC)** — primary detection metric, NOT F1, NOT
  ROC-AUC (12% positive class makes those misleading). Sparse linear
  probe, top-S features, S ∈ {1, 2, 4, 8, 16, 32}.

Architecture set: 7 archs (NOT TFA-pos), all with c7 d_sae=32768:
TopK-SAE, Stacked-SAE, TFA, **T-SAE = paper-faithful Ye et al. port
only** (`temp_bench.architectures.tsae:TSAEPaper` from
`origin/han-phase7-unification:src/architectures/tsae_paper.py`),
TXC-base, TXC-pro, MLC. **DO NOT port `tsae_ours.py`** — deprecated
crude approximation, never use.

Cut protocol: **cut25** (cut Stage A trace at 25% of unsteered length,
then steer-and-continue) per Aniket's B3 — beats LLM-judged cut and
full-trace.

Locked decisions in scope: #1 (two TXCs — Aniket's hill-climbed TXC
must be replaced by TXC-base/pro; the +1.574 reproduction is the
test), #4 (cross-branch reads), #6 (HF repos), #7 (Bricken opt-in;
C7 must run an A/B at 5k×1seed before adopting).

References:
- `agents/README.md` (your roster row + pod-mode contract)
- `docs/components/c7.md` (full setup, metric definitions, reference
  numbers, caveats, reproduction)
- `docs/paper/hardware.md` *Multi-GPU access*
- `decisions.md`
- `papers/backtracking.md` (Ward et al. 2025)
- `origin/aniket-ward-stage-b:docs/aniket/experiments/ward_backtracking/{methodology,results_b,handoff}_neurips_push.md`
- `PROTOCOL.md` § 7 (results live in state — AUTO-RESULTS markers),
  § 11 (framework), § 12 (pinning), § 13 (multi-GPU)

---

## Current state (agent owns — overwrite at every compact)

**Last verified: (not yet provisioned)**

- `git HEAD`: (set on first session)
- Last leaderboard append: (none yet)
- Last checkpoint saved: (none yet)
- Active GPU lock(s): none
- Recent decisions in scope: #1, #4, #6, #7
- In flight: nothing (provisioning pending)

## What I just did (agent owns — overwrite)

(Empty — agent_back not yet provisioned.)

## Next action (agent owns — overwrite)

**Pre-condition (Han owns)**: Han has already run
`bash scripts/bootstrap_runpod.sh` on this pod (interactive — prompts
for tokens; an agent cannot enter input). Because this pod is
ephemeral, Han re-runs it whenever the pod is recreated. Tokens are
in `/workspace/.tokens/` and `purified/.venv/` exists when you wake
up. If the smoke test complains about missing tokens, **ping Han**.

1. `cd /workspace/temp_xc/purified`
2. `source scripts/set_agent_env.sh agent_back` (sets
   `TEMP_BENCH_POD_MODE=ephemeral`)
3. `bash scripts/agent_smoke_test.sh`
4. `bash scripts/sync_from_hf.sh` (mandatory — ephemeral pod)
5. `git pull --rebase origin final`
6. Read `docs/components/c7.md` and Aniket's handoff:
   `git show origin/aniket-ward-stage-b:docs/aniket/experiments/ward_backtracking/handoff_neurips_push.md`
   `git show origin/aniket-ward-stage-b:docs/aniket/experiments/ward_backtracking/results_b_neurips_push.md`
7. Port from `origin/aniket-ward-stage-b:experiments/ward_backtracking_txc/`
   (with header-comment attribution + commit hash):
   - `architectures.py` → integrate locked archs into the pipeline
   - `b1_steer_eval.py`, `b1_held_out.py` → inducement (Δgc)
   - `b3_llm_cut.py` → detection (PR-AUC) calibration
   - `b2_cross_model.py` → cross-model checks if time permits
   Drop into `temp_bench/case_studies/backtracking.py` + thin runner.
8. Build **Llama-3.1-8B BASE L10 residual** activation cache (datasource
   `llama_3_1_8b_base_l10_ward` in `configs/datasources.yaml`). Push to
   HF temp-bench-data on completion. Stage A traces are read-only from
   `origin/aniket-ward-stage-b:results/ward_backtracking_txc/stage_a/`
   — port once with attribution per PROTOCOL.md § 2.
9. **20-transcript blind κ validation** before trusting Sonnet judge.
   Hand-score 20 transcripts → Sonnet judge same → compute κ. Must
   clear ≥ 0.6 + agreement ≥ 0.80 to proceed (Aniket's reference:
   0.749 / 0.773 / 1.000).
10. Train 7 archs × 3 seeds via `runner.run_cell(...)` →
    inducement Δgc (Sonnet) + detection PR-AUC across the 25-point
    magnitude grid. Cut protocol: cut25.
11. Headline cell to verify: TXC-pro peak Δgc reproduces ~+1.574
    (Aniket's wasteland reference, hill-climbed TXC). Reproducing this
    on locked TXC-pro is the C7 paper-headline test. If TXC-base/pro
    fall well short, document honestly in c7.md and adjust framing.
12. After cells land: run `temp_bench.report.render(component="c7")`
    to populate the AUTO-RESULTS block in `docs/components/c7.md`.
    The renderer reads leaderboard.jsonl + your
    `experiments/c7_backtracking/analysis.py`.

## Don't repeat (agent owns — overwrite)

- **Aniket's hill-climbed TXC** — decision #1 forbids it. The point
  of C7 is whether the locked TXC-pro reproduces; if it doesn't, that
  is the result, don't try to revert.
- **Skip κ validation** — the Sonnet judge is unverified at this
  scale; PR-AUC numbers without κ ≥ 0.6 are not reportable.
- **Use C3's IT-L13 cache** — backtracking is on **Llama-3.1-8B BASE
  L10**, paper-faithful to Ward et al.; you build your own cache.
  Do NOT use Gemma-2-2b for C7 — without a reasoning-finetuned
  counterpart, the BASE→reasoning-model steering claim is unreplicable.
- **`tsae_ours.py`** — deprecated wasteland file (TopK + 50/50 split +
  no AuxK). The C7 T-SAE baseline is `tsae_paper.py` (faithful Ye et al.
  port), period.
- **Forget HF push** — ephemeral pod.
- **Wasteland imports** — `git show` only; copy with attribution.
- **Bypass `runner.run_cell`** — single canonical pathway.

## Open questions for Han (agent owns — overwrite)

(none at provisioning.)
