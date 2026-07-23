# Working state — agent `runpod-b`

**Last rewrite:** 2026-07-23 ~15:30 UTC — **FB-C3 COMPLETE, stopped at the
acceptance gate. Awaiting mac-local review; briefing
`briefings/freqbench-fb5.md` left in place.** No task in flight.

## Who / where
Second RunPod box, repo `/workspace/temp_xc`, 32 CPU, no CFS cap.
`/workspace/.agent_id` = runpod-b. Git identity set (Han); push:
`git push https://x-access-token:$(cat ../.tokens/gh_token)@github.com/chainik1125/temp_xc.git arxiv`.
`export ANTHROPIC_API_KEY=$(cat /workspace/.tokens/anthropic_key)`.
Freqbench meter: **$1.63 / $25** (`freqbench/results/spend.json`).

## FB-C3 outcome (all committed; push = this commit's parent set)

Card FB-5 `permuted_tones` end-to-end in one session, **zero gate
amendments** (first cycle ever):

- **Frozen** pre-build (commit "card FB-5 permuted_tones FROZEN
  pre-build") — the FB-4 salvage with the temporal knob: K=10 random
  permutation schedules on the frequency substrate; non-absorption
  obligation (new LOOP card item 1, first use) discharged at freeze;
  multiset status stated honestly.
- **Build:** generator + `toy_permuted_circle_M101_d128` +
  `permuted_recovery` (matched-filter oracle) + 6 contract tests;
  protocol 1.3.0 untouched.
- **Gates first-run PASS** (bars a priori; the FB-4 probe-protocol datum
  became the design-time window-floor bar): floors AT chance, oracle
  0.43/0.99/1.00 @T=2/4/8, envelope reference 0.017/0.048/0.116
  (recovery units); T2 shuffle kills oracle 1.00→0.125; skeptic PROCEED
  5/5.
- **Grid 636/636, blind verdict POSITIVE (weak realization, 16% of the
  provable ceiling).** The acid test resolved to the ALIGNMENT side:
  **trained spectral tracks the envelope reference numerically at every
  T (0.016/0.042/0.096 vs 0.017/0.048/0.116)** — band energies, not
  temporal structure; txc-post is the only arch beyond the envelope
  (0.161 @k=8); additive ≈ 0. Untrained spectral prior collapsed
  +0.298 → 0.045 (partial hold: small spectral>post residual remains).
  Falsifiers clean.
- Records/trackers: `permuted_tones/bench_record.md`, registry row,
  BENCHMARKS § A row, REPORT 96/96, FreqFrac T{4,8} + merged table,
  PORT § J cycle log. Tests 179 green.

## Items left for mac-local review (proposed in records, NOT actioned)
- **README subtype-rule qualifier** (program-rule edit, out of scope):
  power leg → "power/equality → spectral, when the power concentrates in
  few DCT bands" (wording in `permuted_tones/bench_record.md` § 3).
- The weak-realization pattern now on TWO benches (FB-3 21%, FB-5 16% of
  provable ceilings) — a possible program-level finding about training,
  not architecture.
- FB-5's FreqFrac rows as the axis-1 broadband pole anchor.

## Operational notes
- Parallel agents tonight: `runpod` (C7), `runpod-c` (conversion-depth).
  Shared-branch rules; cite commit SUBJECTS not SHAs (new rule).
- Grid recipe unchanged; runner idempotent per eval_key; T≤8 tone-bench
  cells fast-forward from the local checkpoint store.
- Rewrite this file before any compact.
