# Working state — agent `runpod-b`

**Last rewrite:** 2026-07-24 — **`briefings/probe-adequacy.md` COMPLETE
at the acceptance gate; awaiting mac-local review. Briefing stays until
that review.** All four deliverables shipped and committed; full suite
292 passed / 1 skipped; no leaderboard writes; no readout decision
taken or implied anywhere in tracked prose (runpod-d/e's probe-capacity
findings cited only as reported-under-review). I am idle pending review
/ next assignment.

## Who / where
Second RunPod box, repo `/workspace/temp_xc`, 32 CPU, no GPU.
`/workspace/.agent_id` = runpod-b. Git identity set (Han); push:
`git push https://x-access-token:$(cat ../.tokens/gh_token)@github.com/chainik1125/temp_xc.git arxiv`.
`export ANTHROPIC_API_KEY=$(cat /workspace/.tokens/anthropic_key)`.
HF token at `/workspace/.tokens/hf_token`.

## What shipped (probe-adequacy, all committed; LOG entry appended)

1. **`src/temp_bench/evals/lambda_recovery_v2.py`** + additive opt-in
   dispatch in `synthetic_recovery.py` (flag `lambda_probe_v2` in
   eval_cfg; absent → byte-identical, protocol stays 1.3.0) +
   `configs/sweeps/lambda_probe_v2_smoke.yaml` (committed NOT run —
   sweeps write leaderboard rows) + 12 contract tests
   (`tests/test_lambda_recovery_v2.py`; v1 repro at ols/nw1024 to
   1e-10, determinism, trace-split property tests, validate green).
   Knobs: RidgeCV logspace(-2,4,13) train-half-only (α ships as
   `lambda_alpha_v2`), nw default 8192 (= 8·p at T16/p2048; Stacked
   p>n exception disclosed), split default `trace` = boundary-snap of
   n//2 (no trace_ids ⇒ exactly v1's n//2, so synthetic benches are
   untouched). Both Ward datasources now expose `trace_ids` (additive);
   `grid.run_cell` gained `eval_extra` pass-through (default {}).
2. **Forensics receipt**
   (`lambda_intensity/results/split_forensics.json`, script committed
   before output): stream trace-contiguous; ONE straddling trace (152:
   14 train / 1 eval window); v1 uses seeds 0/1 in every cell → at
   nw1024 ZERO leaked eval draws ⇒ **committed panel numbers untouched
   by split leakage on both panel datasources** (confidence.npz grid
   verified identical); at nw8192 half-split leaks 2/8192
   (|Δr| ≤ ~5e-4), snap leaks 0 ⇒ trace default in v2.
3. **`support_stats/stage2_variance.py` probe-agnostic**: --ds/--probe/
   --metric/--k-pos/--crosscheck-json/--out-prefix; defaults reproduce
   committed receipts byte-identically (verified, empty diff). Also
   fixed a latent abort: post-matched k_pos=8·T rows (already on the
   leaderboard, 108 rows for the λ̂ ds) dup-collide in the old loader;
   new k_pos + probe filters restore the 84-row panel by design.
4. **`lambda_intensity/PROBE_V2_SPEC.md`** freeze candidate: exact
   convention, 192 eval-only re-run cells (108 λ̂ + 84 hedging, all
   checkpoints reused; ≈ 3–4 h wall at 3 workers), one-command variance
   re-base (`--probe v2 --out-prefix stage2_variance_v2`), explicit
   non-decision section. Adoption = mac-local freezes the file.

## Standing context
- **16-not-17** (factory review qualification 5): resolved — the review
  says the FREEZING agent corrects the sc_lambda marker count when the
  card is frozen at screen time; no action of mine. Other
  qualifications (1–4) bind screens, not me, unless I run one.
- Factory batch APPROVED (4 bundles / 5 labels + 2 kills); screen queue
  open upstream (sc_lambda first); runpod on candidate-factory-broad-2;
  fineweb bundles are runpod-e's.
- Shared-branch protocol: pull-rebase before EVERY push; LOG.md
  conflicts keep upstream entry then re-append mine; commit SUBJECTS
  not SHAs; case-collision-free filenames; builders/scripts committed
  BEFORE outputs; no reviewer/meeting quotes in tracked files.
- Environmental pytest trap: untracked files break
  `test_diff_hash_consistent_with_dirty` — commit/clean before full
  runs (traces.json is gitignored and doesn't count).
- Rewrite this file before any compact.
