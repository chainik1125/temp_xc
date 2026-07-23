# Working state — agent `runpod-b`

**Last rewrite:** 2026-07-23 ~07:50 UTC — pre-compact handoff. **FB-C1 is
COMPLETE, reviewed context pulled; NEXT TASK = `briefings/freqbench-t16-fbc2.md`
(FB-C2), not yet started.**

## Who / where
Second RunPod box, repo `/workspace/temp_xc`, 32 CPU, no CFS cap.
`/workspace/.agent_id` = runpod-b. Git identity set (Han); push:
`git push https://x-access-token:$(cat ../.tokens/gh_token)@github.com/chainik1125/temp_xc.git arxiv`.
`export ANTHROPIC_API_KEY=$(cat /workspace/.tokens/anthropic_key)` for the
skeptic (Meter at `freqbench/results/spend.json` — C1 ended at $1.04; C2
has its own $25 cap per the new briefing).

## Fresh-context read order (post-compact)
1. `briefings/freqbench-t16-fbc2.md` — the FB-C2 task (12 h): Phase 1
   T=16 addendum for frequency/multilane/phasepair (~270 cells/bench,
   3 frozen mac-local predictions to score blind) + `--T 16` FreqFrac;
   Phase 2 verify_theory → permanent tests; Phase 3 card FB-4
   `rotated_multilane` (basis-alignment acid test on the subtype rule).
2. `freqbench/LOOP.md` — **re-read: mac-local added a strict
   commit-then-run rule in T3** (commit gating scripts BEFORE first
   execution; amendments as their own commits — a direct response to C1's
   three post-hoc check-fixes).
3. README § coordinates — now carries the **order-2 subtype rule**
   (phase→post · power/equality→spectral · covariance→pre) that FB-4
   attacks; also new: `docs/substrate_audit_2026-07.md`.
4. `freqbench/PORT.md` § H — the FB-C1 cycle log (what was found; the
   triple dissociation FB-4 tests).
5. Two other agents tonight: `runpod` (expansion C6), `runpod-c`
   (conversion-depth, GPU) — shared-branch rules in `agents/README.md`
   (pull --rebase before push, only in grid-quiet windows; append-only
   shared files, keep BOTH on conflicts; union drivers on the JSONLs).

## FB-C1 summary (context, all pushed through 4e011026)
Three theorem-first benches end-to-end, ~2,100 grid cells, 0 failures:
multilane POSITIVE (spectral/power; sprint band headline failed its frozen
T=8 bar +0.019 < 0.03; spectral collapses at k_pos=8); colored_sources
POSITIVE-weak (CS-1 floor wholesale; ≤21 % of the +0.96 ceiling, carried
by txc-pre — ordering inversion); phasepair POSITIVE (post sign 1.000;
spectral singleton-band-blind at T≤4, 0.936 at T=8; exact bag null).
REPORT 90/90 (union with runpod's recipe regime-3 POSITIVE). Records in
`multilane/`, `colored_sources/`, `phasepair/` +
`freqbench/cards/FB-{1,2,3}.md`; cycle log PORT § H.

## FB-C2 execution notes (from C1 experience, this box)
- Grid recipe: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 GIT_OPTIONAL_LOCKS=0
  TQDM_DISABLE=1 TEMP_BENCH_ALLOW_DIRTY=1 .venv/bin/python -u -m
  experiments.explorations.synthetic.<bench>.<driver> 28` in background;
  ~8-9 cells/min at T≤8; T=16 cells ~2× slower (batch 1024//16 = 64).
  T=16 addendum per bench = `design.uniform_cells(ds, F, 30_000,
  window_ts=(16,), ...)` added to the driver (bands-addendum pattern);
  mind the d_sae ≥ k_pos·16 drops for pooled archs (k=16 needs d≥256 →
  dropped everywhere; k=8 needs 128 → only 2F cells for pooled).
- Checkpoint store populated (~2,300 keys): T≤8 cells fast-forward; only
  T=16 trains fresh. L=32 still tiles T=16 (32/16=2 tiles).
- freqfrac_report: `--T 16 --tag <bench>_s1_T16` per bench AFTER the T=16
  leaderboard rows exist; all three benches already in registry.py.
- FB-4 build: do NOT mutate `multilane_tones`'s signature — add a thin
  new generator (e.g. `multilane_tones_rotated`, seeded Haar Q composed
  with the existing generator) as the clean append; datasource entry;
  restate (don't re-derive) the rotation-invariant proofs in the card.
- COMMIT gating scripts BEFORE first run (new T3 rule); amendments as own
  commits. One-sided floor checks for probe/eigen artifacts (C1 lesson).
- Skeptic: raw persisted pre-parse; disclose every amendment; skeptic.py
  needs an FB-4 branch in build_user + argparse choices.
- BatchTopKSAE.train_step wants (B, d_in); tsae class = TSAEPaper.
- Rewrite this file before every compact.

## Next concrete action (fresh window)
Read the briefing + LOOP.md T3 in full, then Phase 1: write the three
T=16 addendum drivers, COMMIT them, launch (frequency first, then
multilane, then phasepair — sequential at 28 workers); while grids run,
do Phase 2 (verify_theory test ports) and freeze the FB-4 card.
