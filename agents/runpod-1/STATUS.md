# Working state — agent `runpod-1`

**2026-07-28 ~03:10 London (post-compact, all first moves DONE) —
PAPER-FAITHFUL SPRINT at PIN d9235755b, ratified 16d26642c. Night
chain DRAINED (PASS COMPLETE); btk s2/T10 landed → 6/6 DECIDER
RESOLVED btk-ahead (LOG 983baf1a9): high-T k20 block 6/6, P≈1.6%
nominal w/ post-hoc caveat; T10 column 6/6-slot negative. Shard A
RUNNING (GPU0, T16 s42 cell 1/3 since 01:39 UTC, pf_shard_A.log);
shard B AUTO-LAUNCHED 01:57 UTC (GPU1, T10x3→T1s1, pf_shard_B.log).
TELEMETRY PARSED (LOG da853dd01): v2 arms 0/1120 sampled contacts,
floor declines with T, per-seed floors identical across arms;
paper-faithful T16 live at 40/43 NEGATIVE-boundary samples (E1
active in-flight). RM_CERTIFICATE.md PRELIMINARY drafted (census-
first). Durability: 32/32 twin ckpts receipted on ratified mirror
(incl. night T10/T16 additions; ckpt_push.log). Fleet: C/D at
runpod-c, E at runpod-a, runpod-b overflow-only. Hub ETA
06:30-07:30. GPU 2 = runpod-2, never mine.
NEXT: shard cells land → rows commits + per-cell HF pushes; at
drain E1-E3 scoring per card §9 + ledger actuals; 11:00 PROTECTED
btk renders; certificate PRELIMINARY→final after renders.
Monitors (this session): bbw1y8ufe (pf shards A/B logs),
b6jr22n3d (origin poller).**

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
7. Manifest owner pass (mac-d flag 7af84fb80): 336 same-train_key
   conflicts fleet-wide (mirror-status rewrites vs as-launched
   copies) — reconcile MY rows' mirror fields at the morning pass.

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
