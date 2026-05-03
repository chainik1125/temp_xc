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

You are agent BACK, lead on **C7: Ward Stage B backtracking** on
`google/gemma-2-2b` (BASE) layer 10 — backtracking is a base-model
behaviour, so this component diverges from C3/C4/C5's IT subject.

Hardware: pod `4× A40`, pinned to **GPU 1**. Pod mode **`ephemeral`**:
HF is the source of truth, auto-push on checkpoint save. agent_steer
shares the pod on GPU 0 (different component + different cache).
GPUs 2 + 3 are spare pool slots — claim via
`temp_bench.utils.gpu_locks.claim_gpu(idx)` if you need parallelism
(PROTOCOL.md § 13).

You build your **own** Gemma-2-2b-BASE L10 activation cache — not
shared with C3 (which is IT-L13). No upstream dependencies; you can
start at T+0.

Hypothesis (from `docs/components/c7.md`): TXC-pro delivers the largest
peak Δgc (keyword-rate gain under steering toward the inducement
direction), roughly 3× the next-best architecture. This is the "TXC
pushes the Pareto frontier on a behavioural inducement metric" claim
— **C7 is a candidate paper headline result**, conditional on
reproducing Aniket's +1.574 under our locked TXC-pro.

Aniket on `origin/aniket-ward-stage-b` is the upstream — his pipeline
in `experiments/ward_backtracking_txc/` (architectures.py, b1_…,
b2_…, b3_… — see directory listing) is what you port. Don't merge
his branch; read via `git show` (decision #4).

Two metric axes:
- **Inducement (Δgc)**: keyword rate under steering. Sonnet judge
  with **κ ≥ 0.6** vs blind 20-transcript hand-score validation
  (must clear before trusting Sonnet at scale).
- **Detection (PR-AUC)**: linear probe on activations.

Architecture set: 7 archs (locked set + MLC), as listed in c7.md —
TopK-SAE, Stacked-SAE, TFA, T-SAE (paper config k=20), TXC-base,
TXC-pro, MLC. `stacked_sae` is registered in `configs/locked_archs.yaml`
(added 2026-05-03 — class file still to be ported from
`origin/han-phase7-unification:src/architectures/stacked_sae.py`).

Locked decisions in scope: #1 (two TXCs — Aniket's hill-climbed TXC
must be replaced by TXC-base/pro; the +1.574 reproduction is the
test), #4 (cross-branch reads), #6 (HF repos), #7 (Bricken opt-in;
C7 must run an A/B at 5k×1seed before adopting).

References:
- `agents/README.md` (your roster row + pod-mode contract)
- `docs/components/c7.md` (full setup, metric definitions)
- `docs/paper/hardware.md` *Multi-GPU access*
- `decisions.md`
- `papers/backtracking.md` (Ward et al. 2025)
- `origin/aniket-ward-stage-b:docs/aniket/experiments/ward_backtracking/handoff_neurips_push.md`
- `PROTOCOL.md` § 11, § 12, § 13

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

1. `cd /workspace/temp_xc/purified` — first-time bootstrap via
   `bash scripts/bootstrap_runpod.sh` if pod is fresh.
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
8. Build Gemma-2-2b-BASE L10 activation cache for the backtracking
   prompt set (your own — not shared). Push to HF temp-bench-data
   on completion.
9. **20-transcript blind κ validation** before trusting Sonnet judge.
   Hand-score 20 transcripts → Sonnet judge same → compute κ. Must
   clear ≥ 0.6 to proceed.
10. Train 7 archs × 3 seeds via `runner.run_cell(...)` →
    inducement Δgc + detection PR-AUC across the magnitude grid.
11. Headline cell to verify: TXC-pro peak Δgc reproduces ~+1.574
    (Aniket's number from `a4fbc954`). If yes, C7 is paper-headline
    quality. If TXC-base/pro both fall well short, document honestly
    in c7.md and adjust paper framing.

## Don't repeat (agent owns — overwrite)

- **Aniket's hill-climbed TXC** — decision #1 forbids it. The point
  of C7 is whether the locked TXC-pro reproduces; if it doesn't, that
  is the result, don't try to revert.
- **Skip κ validation** — the Sonnet judge is unverified at this
  scale; PR-AUC numbers without κ ≥ 0.6 are not reportable.
- **Use C3's IT-L13 cache** — backtracking is BASE-L10; you build
  your own.
- **Forget HF push** — ephemeral pod.
- **Wasteland imports** — `git show` only; copy with attribution.
- **Bypass `runner.run_cell`** — single canonical pathway.

## Open questions for Han (agent owns — overwrite)

(none at provisioning.)
