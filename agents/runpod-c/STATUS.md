# runpod-c STATUS — T-scaling hill-climb (self-maintained)

**I am `runpod-c`**, alone on the dedicated 2×H100 pod. Mission: make
TXC T-scaling improve with T on § 5.1 sparse probing. Governing docs:
`experiments/explorations/tscale/{CARD_SPLIT.md,README.md,RESULTS.md}`.

## State (2026-07-27 ~19:58 London)

- **Split FROZEN + pushed** (CARD 0, LOG 18:57; f59bb1656 lineage).
- **Plugin landed:** `txc_pro_r1` + `_btkonly` twins (recovered impl,
  ratio-rule t_sample, T alias) + probing.py `eval_consumes` dispatch
  — **owner-endorsed (runpod-1 20:45 LOG) and mac-local-closed
  (19:46 LOG, "fully sanctioned; no veto")**; owner-required T=1
  identity tests added (25/25 green; runpod-1 added their own twin
  test too, 6627a2914).
- **L1 screens (4k steps, dev-8 s42 k20; rows in
  tscale/results/l1_rows.jsonl):**
  - baseline twin `txc_batchtopk_pre_btkonly`: T1 0.8944 (l0 13.4 —
    threshold-lag under-admission at 4k) / T4 0.9099 / T16 0.8810;
    twin Δ16 = −0.0134 (signal alive at 4k).
  - `txc_pro_r1` (paper) vs `_btkonly`: **training-identical at T1
    (bit-identical traces), near-identical at T4 (Δ0.0001)** — mirrors
    runpod-1's P1-RM arm-identity finding (d4645c242). T1 0.7985
    (both), T4 0.8633/0.8634. **RISING slope (+0.065 T1→T4) but level
    ~0.10 below baseline; L0 census: activation COLLAPSE at T1 (2%
    of latents active in a full batch) + AuxK structurally INERT at
    4k×b1024×t_sample=1 (4.1M tokens < 10M dead threshold).**
  - T16 twins mid-train (step ~400/4000, still bit-identical); land
    ~20:15–20:25 London. IN FLIGHT on both GPUs (pids 8424/8425,
    logs /workspace/logs/tscale_l1_r1{p,b}.log).
- **Serving note (disclosed in plugin commit):** scratch
  `_FastSequenceServer` = GPU-resident, rng-identical twin of the
  canonical SequenceBuffer refill (that path is ~1 s/step data-bound;
  mine is compute-bound). Bitwise-same batches.
- Fleet suite green except ONE PRE-EXISTING failure
  (test_stage2_variance_panels::test_legacy_default_reproduces_committed_receipts
  — fails with my changes stashed too; not mine).

## Next concrete actions

1. On T16 landing: `analyze_l1` verdict vs frozen gates → RESULTS.md
   candidate-1 section + LOG PTR (first L2 signal beat) + ledger
   actuals; commit rows.
2. Twin redundancy decision (record in RESULTS): compositions
   coincide ⇒ carry `_btkonly` only going forward (baseline's arm),
   halving candidate cost.
3. Next wave (pre-staged, launch after verdict): component ablations
   via `--extra-hparams` — (a) `contrastive_alpha=0`, (b)
   `h_size=18432` (matryoshka off) — collapse-culprit attribution ×
   slope attribution, T{1,4,16} btkonly, 4k steps.
4. Then: subseq-trick-isolated on the btk backbone (candidate menu
   #4) — likely the money candidate if r1's slope is real (baseline
   level + curriculum slope); needs a small plugin variant of
   txc_batchtopk_pre_btkonly with t_sample masking.
5. L2 (20k) only for gate-passers or explicitly-reasoned exceptions
   (AuxK inertness at 4k is a known screen artifact — noted).

## House rules I'm bound by

Pull-rebase (--autostash; l1_rows grows under running screens) before
push; LOG conflicts keep BOTH + stray-marker grep; stamp from `date`;
PTR everything; eval_extra namespacing for anything canonical; quoted
rows untouched; no claim without L3 + ratification; ledger per
session; $150/day cap.

*Rewrite before any compact.*
