# Working state — agent `runpod-1`

**2026-07-27 ~00:00 London — ACTMIX P1 (briefings/actmix-runpod-1.md).
Shared 3×H100 pod, GPUs 0,1. Phase A (btk-only) grid RUNNING at PIN
f9108db44; Phase B (paper-match, eval-only on shipped ckpts) staged +
smoke-gated, full run launching alongside. Deadline 9am PT / 17:00
London today. Ledger: RUNPOD section, ~$1 bring-up + est $55–75 grid.**

## Live state (check first on resume)

- Phase-A queues: `/workspace/logs/actmix_p1_gpu{0,1}.log` (nohup bash
  chains; passes: untrained(42) → sae/tsae(1,2,42) → txc-pre(1,2,42 ×
  T{1,2,4,8,16}) → txc-post(42) → txc-post(1,2)). Cache-hits make
  relaunches idempotent — ALWAYS relaunch via
  `PIN=<origin sha> bash experiments/probing/actmix/launch_runpod1.sh`.
- Phase-B: `experiments/probing/actmix/phase_b.py` — `stage` done
  (15 cells, manifest committed), `smoke` = port-validation gate
  (paper topk_sae s42 k20 vs paper's 0.8831±0.0022), then `run`
  (co-resident, shard per GPU, TEMP_BENCH_ALLOW_DIRTY=1).
- Monitors (session-local): grid-log watcher + origin watcher
  (path-filtered per actmix-shared listening topology).

## Landed tonight (all pushed)

1. **Eval port** `ProbingEval` 1.2.0 (= v1 1.1.0 + shuffle control
   [Aniket fixed-probe semantics] + realized-l0 z≠0) + probe-cache
   loaders; tests 9 green. Paper's ACTUAL caches synced from HF
   (act_cache e4916bcae1881963 → data_cache 48d2d17ff88598d4;
   probe_cache 38/38).
2. **CARD.md** (grid, queue, E1–E4, G1–G5, l0 bands, 10 flags) +
   sweep driver (38-task preflight) + PIN-asserted launcher.
3. **Defects caught + disclosed** (CARD flag 10 + LOG): launcher
   double-flag (pre-untrained dropped), missing dirty-stamp
   convention (chains refused at cell 2), runner checkpoint-cache
   path loads models on CPU → silent CPU-crawl evals (fixed
   plugin-side in ProbingEval; core contract gap flagged for owner).
4. **Phase B staged**: paper_{topk_sae,tsae,txc_base}_v1 adapters
   (verbatim dev classes @94119bc08, eval-only, src_tag provenance),
   registry entries, strict-load-proof manifest (15 cells, sha256,
   dup-family rationale: 05-05 re-train tws=1/2 family).
5. **T5-ARTIFACT FINDING** (LOG ~23:30, PENDING TEAM REVIEW): all six
   shipped "T10/T20" c3 ckpts have T=5-shaped weights (silent-T5 bug,
   pre-05-06-fix saves; census exhaustive ⇒ no faithful eval-only
   T-sweep exists). Flat-T-sweep hypothesis (appendix T-slope = seed
   noise among T5 replicas) TESTABLE by Phase B: cfgT10/cfgT20 cells
   evaled AS T5 with bug_artifact_t5 on-row.

## Next actions (in order)

1. Confirm Phase-B smoke ≈ paper 0.8831 → launch `phase_b run` shards
   on both GPUs (co-resident with training).
2. Watch grid: G1 l0 bands (btk-only ≡ nominal), G2 identity, G3
   untrained<trained, G4 n_tasks=38, G5 T1 anchor.
3. When queue drains (or ~09:00 London, whichever first):
   `python -m experiments.probing.actmix.analysis --arm btk-only` +
   `--arm paper-match` → RESULTS.md + figs; write LOG verdict
   (PENDING TEAM REVIEW, quote CARD § 4 verbatim); ledger actuals;
   STATUS final; push.
4. Interpretive note for the table: mac-a CALIB FINAL = btk-only ≡
   relu-mix at hunt widths (eval-threshold pruning mechanism);
   mac-local ruled pods' exhibits unaffected. My E2 (l0 ≡ nominal
   sharp) already consistent. Optional 1-cell relu-mix twin
   (batchtopk_sae s42) queued ONLY if GPUs free before analysis.
5. Post-deadline queue: T=32 stretch, txc-post seeds 1/2 if cut,
   agent_steer tsae twins, probe-cache builder port.

## Standing repro notes

- Every shell: `cd /workspace/agents/runpod-1/temp_xc && source
  scripts/set_agent_env.sh runpod-1`.
- Rows: experiment=probing, protocol 1.2.0, eval_cfg.arm ∈
  {btk-only, paper-match}, agent=runpod-1; smoke rows carry
  smoke:true; dirty stamps on pool rows = leaderboard growth
  (convention, CARD flag 10); Phase-B rows carry src_train_key (+
  bug_artifact_t5 where applicable).
- Caches: /workspace/caches/probing/{hf_mirror,tbm_ckpts} (+ symlinks
  results/data_cache/48d2d17ff88598d4, results/probe_cache/<ds>).
