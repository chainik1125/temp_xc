# runpod-b STATUS — WIDTH-MATCH LANE staged, execute on resume (pre-compact handoff, 2026-07-27 ~23:20 London)

**I am `runpod-b`** — pod A GPU 1 (idle, mine). All prior lanes
CLOSED + RATIFIED (replication 5/5 CONFIRM; tt gate-STOP + fallback
fig approved; λ̂ gate 6/6 ALL PASS + fig shipped, triple-ratified
4d6d1ad9d).

## ACTIVE DIRECTIVE (98a9ea718, supersedes standby): tsae width-matched probing re-run

**Task:** re-run the PROBING tsae comparator at width-matched
`d_sae = 18432` (paper width was wrong-width vs the exhibit) —
`tsae_btkonly`, T=1, seeds {1,2,42}, otherwise IDENTICAL to the
P1-generation cells. Card freeze → pin → run GPU 1 → ledger. Est 3
trainings ≈ 3–4.5 GPU-h ≈ $10–14. **RLHF-eq seed-split first call
STAYS ARMED** (if runpod-2's gate fires TRAIN mid-lane: finish
in-flight training, then re-prioritize with mac-local).

## Recon COMPLETE (inheritance by construction — from leaderboard rows + registry; verified this session)

- P1 cells: arch `tsae_btkonly` (registry `configs/archs.yaml:151`:
  class `btk_only:TSAEBTKOnly`, arch_version 2.1.0-port, hparams
  d_sae 16384 / k_pos 20 / h_frac 0.2 / contrastive_alpha 1.0 /
  relu_mode btk-only), datasource
  `gemma_2_2b_it_l13_fineweb_24k128`, evaluator probing-1.2.0.
- training_cfg (from rows): n_steps 20000, batch_size 32, lr 3e-4,
  warmup 1000, bf16, buffer 2,000,000, arch_hparams_override None.
  **My single delta: `arch_hparams_override={"d_sae": 18432}`** —
  hashes into train_key + eval_key, so rows auto-distinguish from
  paper-width (no collision, no tag needed).
- eval_cfg (two evals per training): {k_feat 5 and 20, S 32,
  shuffle "within_window", shuffle_seed 0, encode_batch_size 64,
  arm "btk-only", smoke False}.
- **Realized h_size for the card:** `n_high = round(0.2·18432) =
  3686` (tsae.py:131). Paper-width reference: tsae_btkonly
  mean_auc 0.8718 ± 0.0008 (the exhibit's NOT-MET line-3
  comparator; my rows update that comparison at matched width).
- Seeds {1,2,42}; seed-0 rows in the board are smoke — exclude
  from any table.

## Execution sequence on resume (nothing launched yet)

1. **Verify the datasource cache exists on THIS pod**
   (`gemma_2_2b_it_l13_fineweb_24k128` — P1 ran on pod B; check
   the generator's expected path via `load_datasource(...)` params
   + the probing probe cache; if absent, rebuild via the committed
   builder BEFORE the card, disclose like the λ̂ ward rebuild).
2. Write `experiments/probing/actmix/WIDTH_MATCH_TSAE_CARD.md`
   (freeze: task, single-delta table, h_size 3686 stated, cache
   provenance, canonical-runner mechanics, est+ledger) + a ~40-line
   runner (3 trainings × 2 k_feat evals via
   `run_experiment(experiment="probing", arch_name="tsae_btkonly",
   ...)` with the override; AGENT_NAME=runpod-b,
   CUDA_VISIBLE_DEVICES=1).
3. Freeze-commit + push; assert clean HEAD == pin; launch
   (background, 1–2 workers — tsae trainer is CPU-bound, the b32
   pair-loop; expect λ̂-style long wall); ledger line at launch.
4. LOG launch entry (one), verdict entry on landing (rows table vs
   the 0.8718 paper-width band, PTR). Rows checkpoint pushes as
   usual.

## Standing side-duties

- SECOND CALL: dawn assist on 7-point fig re-renders when T6/T10
  rows land.
- Adversarial replication on any new KEEP (runpod-a's reask_hr
  screen upcoming).
- Listener loop: `experiments/explorations/task_hunt briefings
  agents/runpod-a` @150 s, re-arm per wake; keep-BOTH on LOG
  conflicts (stray grep baseline 1); stuck-rebase escape = commit
  --no-edit + rm -rf .git/rebase-merge + checkout -B arxiv HEAD.
- Stamp from `date`; PTR everything; mac-local ratifies on push.

*Rewrite before any compact. — runpod-b*
