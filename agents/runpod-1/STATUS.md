# Working state — agent `runpod-1`

**2026-07-26 ~22:30 London — ACTMIX P1 (briefings/actmix-runpod-1.md,
read-first actmix-shared.md + actmix-pod-bootstrap.md). Shared 3×H100
pod, GPUs 0,1 (`source scripts/set_agent_env.sh runpod-1`). Phase A
(btk-only) FROZEN + launching; Phase B (paper-match) BLOCKED on mac-c
COMPOSITION_AUDIT. Deadline: rebuttal-grade numbers before 9am PT /
17:00 London 07-27. Cap $150/day; ledger RUNPOD section in
briefings/MODAL_SPEND.md.**

## Where things stand

1. **Recon DONE.** Paper § 5.1 protocol fully pinned from
   `origin/final:purified/experiments/c3_probing/` + eval/probing.py:
   gemma-2-2b-it L13 resid_post, fineweb 24k×128, 38-task SAEBench+CT,
   S=32 left-aligned probe cache (schema 2.0.0, first_real masks),
   k_feat {5,20}, seeds {1,2,42}, L1-logistic top-k probe. Paper v1
   anchors (k=20): sae 0.8831, tsae 0.8986, txc T5/10/20
   0.8952/0.8973/0.8999.
2. **Caches: the paper's ACTUAL artifacts, synced from HF**
   (`han1823123123/temp-bench-data`): act_cache `e4916bcae1881963`
   (24000,128,2304 fp16) + probe_cache 38/38 tasks →
   `/workspace/caches/probing/hf_mirror/`, linked into
   `results/data_cache/48d2d17ff88598d4/` + `results/probe_cache/` by
   `experiments/probing/actmix/prep_cache.py`. No GPU rebuild needed.
3. **Port DONE**: `ProbingEval` protocol 1.2.0 (v1 1.1.0 semantics +
   additive shuffle control + realized-l0(z≠0)) in
   `src/temp_bench/evals/probing.py`; loaders in
   `src/temp_bench/data/probe_cache.py` (builders NOT ported —
   artifact-first; they remain on origin/final). Tests
   `tests/test_probing_eval.py` (7) green. GPU smoke on the real cache:
   all four `*_btkonly` archs train+eval through `run_experiment`;
   realized l0 EXACTLY nominal (sae/tsae 20.0/token, post 20.0/window,
   pre@T3 59.9≈60/window).
4. **mac-a convention CONSUMED** (LOG ~21:05, APPROVED 9e634bed9):
   `*_btkonly` names verbatim; fired ⇔ z≠0 (my l0 metric counts
   nonzero accordingly).
5. **Card**: `experiments/probing/actmix/CARD.md` (grid, queue,
   pre-registrations E1–E4, gates G1–G5, l0 bands, 9 flags, budget).
   Driver: `experiments/probing/actmix/sweep.py` (+ preflight 38/38);
   launcher `launch_runpod1.sh` (PIN-asserted, nohup, shard/GPU).

## In flight / next actions (in order)

1. Commit + pull-rebase + push (card BEFORE cells — mac-local
   freeze-reviews in parallel). PIN = that commit.
2. `PIN=<sha> bash experiments/probing/actmix/launch_runpod1.sh` —
   passes: untrained twins (seed 42) → per-token trained (3 seeds) →
   TXC-pre (3 seeds, T {1,2,4,8,16}) → TXC-post (42, then 1/2).
   Logs `/workspace/logs/actmix_p1_gpu{0,1}.log`.
3. While cells run: analysis/table/fig script
   (`experiments/probing/actmix/analysis.py` — TODO), watch origin for
   mac-c COMPOSITION_AUDIT (Phase B) + mac-local rulings
   (path-filtered listener per actmix-shared.md).
4. On grid completion: table + figs + LOG verdict (PENDING TEAM
   REVIEW, quote card § 4 verbatim) + ledger actuals + push.
5. Phase B when unblocked: eval-only if checkpoints exist, else
   retrain at mac-c's pinned composition.

## Repro/resume notes

- Every shell: `cd /workspace/agents/runpod-1/temp_xc && source
  scripts/set_agent_env.sh runpod-1`.
- Downloads: `/workspace/logs/dl_probing_caches.{py,log}` (idempotent
  resume). Prep: `python -m experiments.probing.actmix.prep_cache`.
- Smoke rows in the leaderboard are `smoke: true` + code_version.dirty
  — excluded from all tables by construction (distinct eval_keys).
- Grid rows: experiment=probing, protocol 1.2.0,
  eval_cfg.arm="btk-only", agent=runpod-1, smoke absent/false.
