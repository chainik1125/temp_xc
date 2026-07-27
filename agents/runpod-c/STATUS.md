# runpod-c STATUS — T-scaling hill-climb (self-maintained)

**I am `runpod-c`**, alone on the dedicated 2×H100 pod. Mission: make
TXC T-scaling improve with T on § 5.1 sparse probing (Dmitry/Han,
07-27). Original bring-up briefing: git history of this file at
e444fd3e4 (superseded by this rewrite). Governing docs I wrote:
`experiments/explorations/tscale/{CARD_SPLIT.md,README.md,RESULTS.md}`.

## State (2026-07-27 ~19:00 London)

- **DONE:** read order absorbed; venv verified (torch 2.8.0+cu128,
  2×H100, `run.py validate` OK, btk-only tests 19/19); substrate
  synced + linked (acts (24000,128,2304) + probe cache 38/38 at
  `results/{data_cache/48d2d17ff88598d4,probe_cache/…}`, mirror at
  `/workspace/caches/probing/hf_mirror`); **SPLIT FROZEN** (CARD 0,
  LOG 18:57 entry, PTR) — dev-8 / holdout-28 pre-registered with
  pyramid gates + candidate-1 pre-registrations; ledger line posted.
- **Candidate 1 source:** the RECOVERED txc_pro class
  (`docs/recovered/txc_pro_phase5b_subseq_h8.py`; TXC_PRO_RECOVERY.md
  corrections: h_size=d_sae//5 not "8 levels", k_pos=20,
  k_train=100/k_inf=200 asymmetry, encode hard-raises off-T_max) —
  revive as NEW ids `txc_pro_r1` (+`_btkonly` twin), per-T retrain
  with t_sample = max(1, T//2) (ratio rule, pre-registered).
- **⚑ Open flag to mac-local (LOG 18:57):** `eval_consumes='window'`
  dispatch generalization in evals/probing.py (byte-identical for
  existing archs, unit-tested) — proceeding launch-then-veto per
  program convention; revert cleanly if vetoed.

## Next concrete actions

1. L1 scratch harness (`experiments/explorations/tscale/`): dev-8
   eval importing canonical `_fit_probe/_score_probe/_encode_pool`;
   scratch train loop mirroring core trainer semantics (4k steps).
2. Matched-steps baseline twin: `txc_batchtopk_pre_btkonly` @4k steps,
   T{1,4,16} s42 → pipeline shakedown + L1 comparator row (RESULTS.md).
3. Plugin drop: `src/temp_bench/archs/txc_pro_r1.py` + configs/archs.yaml
   entries + probing.py dispatch edit + tests (old-path equivalence,
   subseq/topk correctness, l0 sanity).
4. L1 screen: txc_pro_r1 + _btkonly twins, T{1,4,16} s42 k20 dev-8;
   append RESULTS.md; LOG PTR at first L2 signal.
5. Then per gates: L2 (20k full dev grid) → candidate menu items 2-6
   (ablate txc_pro ingredients separately; dead-latent mitigations;
   k scheduling; per-position floors; T-curriculum isolated).

## House rules I'm bound by

Pull-rebase before push; LOG conflicts keep BOTH + stray-marker grep;
stamp from `date`; PTR everything; eval_extra namespacing
(`explore: tscale` in eval_cfg) — quoted rows never touched; no claim
surfaces without L3 + ratification; ledger per session; $150/day cap.
GPU state: both idle right now. No detached jobs running yet.

*Rewrite before any compact.*
