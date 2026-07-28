# runpod-c STATUS — FROZEN hill-climb → probing-sprint join (2026-07-28 02:40 London, date-verified)

**I am `runpod-c`**, alone on a dedicated 2×H100 pod, workspace
`/workspace/agents/runpod-c/temp_xc`, venv `.venv` healthy, probing
substrate ON-POD (acts + 38/38 probe cache; mirror at
`/workspace/caches/probing/hf_mirror`).

## MODE: freeze-and-join (Han order 11227ce0d, ack'd 02:40)

The T-scaling hill-climb is **FROZEN, not abandoned** — full resume
playbook in `experiments/explorations/tscale/RESULTS.md` § FREEZE
(one card: launch the pre-registered C6 pair, § 3 gates unchanged).
Both GPUs go to **probing paper-faithful shards at runpod-1's
card-pin (~05:00 London target)**; possible RLHF relief later
(runpod-2 coordinates, one GPU at probing-drain).

## IN FLIGHT RIGHT NOW (check FIRST on resume)

- **GPU 1, pid 25696** — **SHARD D RUNNING** (launched 02:48 at
  PIN d9235755b): T6×{42,1,2} → T2×{42}, log
  `/workspace/logs/pf_shard_D.log`, drains ~06:00. Sentinel line
  `PF_SHARD_D_DONE`. Per-cell ckpt push via
  `scripts/push_ckpts_hf.py <train_key>` (≤2h rule).
- **GPU 0, pid 27050** — **SHARD C RUNNING** (launched 03:17 at
  the C5-T16 drain): T8×{42,1,2} → T1×{2}, log
  `/workspace/logs/pf_shard_C.log`, drains ~06:30. Sentinel
  `PF_SHARD_C_DONE`. Same per-cell ckpt-push duty.
- **tscale FULLY DRAINED 03:17** — freeze complete. C4 CLOSED
  (T16 0.9253/0.8791 anneal-harmless; KILL as T1 fix). C5 KILLED
  (T16 0.9016/0.8501 — pooled admission destroys the win, echoing
  C2; census pair kills diversity-as-cause). Ledger FINAL: ≈$16
  overnight, ≈$33 program. Resume = C6 per RESULTS § FREEZE.
- **HF ckpt mirror COMPLETE 03:2x**: 27/27 tscale ckpts on
  `temp-bench-data` `ckpts/tscale/<cfg_hash>/`, 27/27 LFS-sha
  matches, receipts committed
  (`tscale/results/hf_durability_receipts.jsonl`). Freeze order
  item 1 fully discharged.
- **Background: HF ckpt mirror** (~25 ckpts, 61 GB →
  `temp-bench-data` `ckpts/tscale/<cfg_hash>/`), receipts →
  `tscale/results/hf_durability_receipts.jsonl`; C4/C5-T16
  stragglers re-run on drain. Log
  `/workspace/logs/tscale_hf_mirror.log`.
- **Card-pin watcher**: git poll for runpod-1's probing
  paper-faithful card commit; on pin → read card §shards → claim
  shards for both GPUs (coordinate with runpod-a: their GPU 0 =
  shard 1; runpod-b GPU 1 joins at rmx_b drain; T1-last ordering
  flagged by runpod-1). Launch per THEIR card/pins — venue rules,
  their runner, nothing of mine.

## Sprint join facts

- 18 cells (21-vs-18 count being resolved by runpod-1), 4 GPUs at
  pin (old-pod GPU 0 + pod-A GPU 0 + my two) → sweep plausibly
  done 08:00–09:00, inside the 11:00 window.
- My substrate is already local — zero sync cost. Probing quoted
  numbers are NEVER touched by my tscale cells (namespaced); the
  sprint shards run runpod-1's canonical pathway, which DOES write
  paper-faithful rows — that is the point; follow their card
  exactly.
- 15-min ack discipline in effect fleet-wide.

## Hill-climb standing (for anyone reading; details in RESULTS.md)

C1–C5 complete: first monotone-rising TXC k20 curve at 20k (diag
0.8974→0.9103→0.9171); r1-min = program-best T16 (k20 0.9251 / k5
0.8763); T1 collapse mechanism = across-row latent concentration,
driven by the BACKBONE not the selection rule (twin census 0.1276
vs r1 ≤0.021 active-frac); k-anneal (C4-T1) and batch-pool (C5-T1)
both FAIL the floor; A2 walk resolved NO; C6 (bdec-init /
recon_shifts diff-ablations) pre-registered NOT LAUNCHED = resume
point. Gate floors: T1 ≥ 0.8844, T16 ≥ 0.8810; k5 preservation bar
0.8551 at L2.

## Process state

- Git: clean at last push; pull-rebase --autostash before every
  push; LOG conflicts = keep BOTH + delete marker lines; **stamps
  only AFTER reading `date`** (5 drifts on record — the rule is
  real).
- CLI bool trap: run_l1 extra-hparams casts int→float→string;
  booleans must be `0`/`1`.
- Ledger: day-1 ≈ $17; overnight hill-climb actuals ≈ $16 at
  freeze (vs $35–40 est). Sprint hours post-03:15 bill to the
  sprint. Cap $150/day fine.
- Tokens at `/workspace/.tokens/` (gh, hf, hf_datasets) — paths
  only, values never in git/logs/cards.

## Next actions queue (in order)

1. ~02:50 C4-T16 drains (watcher) → verdict line; GPU 1 idle.
2. ~03:15 C5-T16 drains → C5 verdict line; GPU 0 idle; push both
   verdicts + mirror stragglers; ledger finals.
3. Card-pin (~05:00): claim shards per card §shards, launch on
   both GPUs, run to sweep-done (~08:00–09:00). RLHF relief if
   called.
4. Post-rebuttal: resume via RESULTS § FREEZE playbook.

*Rewrite before any compact. — runpod-c*
