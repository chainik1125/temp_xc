# Working state — agent `runpod-1`

**2026-07-28 02:41 London (date-verified) — PAPER-FAITHFUL SPRINT: card PINNED
d9235755b, shard A RUNNING (GPU0, T16x3→T1s42 since 01:39 UTC),
shard B armed behind NIGHT_DONE_GPU_1 (btk s2/T10 finishing ~02:05
= the 6/6 k20-block decider; diff at landing). GPU 2 = runpod-2,
never mine.**

## Sprint (task #12, commission 4ce0369de/606e4587d)

Plugin paper_txc_base_v1t = vendored 94119bc08 txc_bare_antidead
FULL training stack verbatim + thin v2 wrapper (dict contract,
first-batch b_dec init, post-accumulate grad hook, post_step renorm,
wrapper-side telemetry). Tests 8/8 (bitwise adapter parity T{1,3},
T1 formula, exact-k, mixing fingerprint, stack receipts, registry).
CARD_PAPER_FAITHFUL.md §6 = the 5-GPU shard table (mine A+B;
runpod-c C+D; runpod-a E; runpod-b overflow post-rmx). 21 cells,
T1/T2 tails LAST (prune-free if hub rules 18). Logs
/workspace/logs/pf_shard_{A,B}.log, monitor bwhegmrtw. At shard
drain: per-cell HF ckpt push (scripts/push_ckpts_hf.py, ratified
mirror), ledger actuals vs $18-22 est.

## Night close (task #11 remainder)

- btk s2/T10 lands ~02:05 → rm_equivalence diff = SIXTH k20 point
  at T≥10 (current block 5/5 btk-ahead, P≈3.1%; 6/6 ⇒ P≈1.6%) →
  NIGHT_DONE_GPU_1 → shard B auto-launches.
- RM-2 fills PREEMPTED (~3 min sunk disclosed); re-queue only in a
  genuinely idle window. relu-mix = certificate evidence only per
  arm mapping 692b — never a matrix column.
- Delta map so far (RM−btk, all local 6/7 tensor diffs): T6 k5
  {−1.63,−1.02,−1.38}e−2 3/3 btk · k20 mixed | T8 coin-flip both k
  | T10 k20 {−6.8,−6.9}e−3 + T16 k20 {−1.67,−0.43,−6.10}e−3 = the
  high-T k20 block. Multiplicity caveats posted with each flag.

## Morning queue (revised for sprint)

1. btk s2/T10 diff + T10 column verdict (immediate at landing).
2. Telemetry parse (btk arm now traced): boundary contact rate vs
   T; census-first framing per 3b0a4df3d (traces = bounds).
   Cross-venue lemma receipts: runpod-2 829f05070/fd3e4ff16
   (identity ⟺ zero contact), runpod-a dq DIVERGES + λ̂ IDENTICAL.
3. 11:00 PROTECTED: 7-point per-k btk renders (--writeup final) +
   38task twin + archived-T5-anchor labeling per 8fefb409d rule +
   agentic/caption disclosures where applicable.
4. PRELIMINARY certificate: census leads; identity = {sae ×3, pre
   T1}; divergence map T2-T16 per-T per-seed; controls; PTR.
5. RESULTS_{btk-only,relu-mix}.md refresh; ledger actuals (night
   ~$30 + fills ~$0.3 sunk + sprint shards); paper-faithful rows
   fold into analysis when shards drain (E1-E3 scoring per card §9).
6. STATUS rewrite before compact.

## Durability

30/30 certificate ckpts on ratified mirror, LFS spot-check MATCH
(e91d887fac22fb33); receipts /workspace/logs/ckpt_push.log. Night
T10/T16 + sprint ckpts append at drain (idempotent tool).

## Standing

date FIRST then stamp. Monitors: bwhegmrtw (pf shards + gpu1 night),
bt1e1yc98 (origin poll). Union-resolve LOG conflicts upstream-first
+ stray grep. Tokens by path only. Aniket read-only. GPU 2 never.
FLAG open: stage2_variance golden test fails pre-existing (panel
lane's, live-leaderboard-coupled).
