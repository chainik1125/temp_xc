# Working state — agent `mac-local`

**Last rewrite:** 2026-07-23 (pre-compact). Supersedes the 2026-07-22 version.

## Who / where
Local CC on the Mac (M-series, MPS/CPU) at `~/research/projects/temp_xc`.
Role: prototyping, review, orchestration. I'm `mac-local`. Two runpod agents
now exist (`runpod` = PhenomenonBench line; `runpod-b` = FreqBench line;
identities via `/workspace/.agent_id` — see `agents/README.md`, incl. the
two-agent shared-branch rules + `.gitattributes` union drivers I set up).

## Git
Branch `arxiv`, clean, **fully pushed @ `7ccddb72`** (ambience-principle
commit). Both agents' completed sessions are merged in; REPORT renders
**90/90** rows across 10 benches. `origin` = SSH.

## ⏭ THE NEXT TASK (post-compact): the CONSOLIDATED REVIEW of both completed sessions

Both agents STOPPED at their acceptance gates; both briefings deliberately
NOT deleted. Review gate-integrity FIRST, then verdicts. My frozen
review checklist:

**A. FB-C1 (runpod-b; briefing `briefings/freqbench-c1.md`).** Cycle log:
`freqbench/PORT.md` § H; records in `multilane/`, `colored_sources/`,
`phasepair/`; final agent state `agents/runpod-b/STATUS.md`.
1. Freeze order: BOTH cards committed pre-build? (`freqbench/cards/`);
   §8 gating committed before each grid (the new README execution rule)?
2. The **3 documented gate-check amendments** (null/witness fixes,
   "skeptic-disclosed") — verify each was a genuine fix, not tolerance
   shopping; this is the highest-risk item.
3. Blind verdicts vs frozen predictions — the *misses* are reported
   (multiband>vanilla FAILED its frozen T=8 bar; FB-3 realized ≤21% of the
   provable ceiling, carried by txc-PRE not post; spectral k_pos=8
   collapse; spectral phase-blindness at T≤4) — confirm they're framed as
   misses, not spun.
4. Grid hygiene: 708+582+636 cells 0 failures; no dup eval_keys
   (two-agent parallel appends + union merges — check globally!); spend
   $1.04; tests green locally.
5. Registry/REPORT/BENCHMARKS consistency (90/90 union re-render was done
   by runpod-b on the merged tree — spot-check).

**B. Stage-6 #3b (runpod; briefing `briefings/stage6-recipe-rescoped.md`).**
Record: `recipe_instruction_phase_runs/bench_record.md` § "Stage-6 #3b".
1. Freeze order provable: A1 metric `cf4ae797` → A2 §5-r re-freeze
   `241845d2` → A3 gating addendum `d65349c0`, all pre-grid.
2. Verdict **POSITIVE — the program's FIRST grounded regime-3 arch
   separation**: only Spectral-TXC exposes the residual (T=2 +0.60/+0.90/
   +0.96 at d=10/20/40, peak +0.97 ≈ exact pair rule, k-robust, untrained
   ≈0.06); additive families at the DC-leak line (falsifier NOT triggered);
   **TXC-post MISS** (capped ≈ additive ceiling, best +0.26 — the
   changepoint τ-precedent overstated post-squash). Verify the residual
   normalization wasn't touched post-hoc; verify the untrained control.
3. 495/495, 0 failures, 0 dup keys; DC control at oracle; realistic-regime
   + capability gates.

**C. Close-out (after both PASS):** delete BOTH briefings; review lines in
PORT §H + bench records + STATUS §0; bake lessons into rules (candidates:
the post-vs-spectral coincidence story needs reconciling across
changepoint/recipe/phasepair — post reads *phase*, spectral reads
*power/equality*; is that the durable split?). Then the roadmap fork, USER
decides priority: **(1) the acid test** (phase 4 — the triple dissociation
+ recipe residual give 4 fresh held-out rows; spectral/power ·
pre/lag-covariance · post/phase · spectral/equality is the coordinate→
ranking currency); (2) T=16 addendum + verify_theory ports (runpod-b
queued); (3) C6 extraction estimator (PhenBench reasoning cell); (4) the
conversion-depth ablation (`docs/ideas/conversion_depth.md` — needs GPU
for 8B multi-layer caches; GPT-2 replica CPU-cheap).

## Standing context (science in research STATUS §0 — read top 3 bullets)
- **Ambience principle** now governs card design (README coordinate
  section + checklist 8 + LOOP card 4; memory
  `project-ambience-principle`): TXCs only earn keep on per-token-silent
  latents; global ≠ ambient; the STOP-gate IS the ambience measurement.
- Trackers: BENCHMARKS.md (10 live benches, +3 theorem-first) ·
  expansion/LEDGER.md (C5 done, C6 target) · REPORT.md 90/90 ·
  freqbench/PORT.md §G/§H.
- Paper context: memory `project-txc-paper-context`; NeurIPS reviews were
  due out ~2026-07-22 — user may pivot to rebuttal work at any moment.

## Recent commit chain (mine, this session)
C4 review close-out `f58bf8a4` → LOOP+freqfrac+addenda `61af93c8` →
first-pass results `7e641474` → overnight prep (2 briefings + runpod-b
identity + union drivers) `56cbae9b` → stage-6#3+C5 review `4ad1f9a7` →
re-scoped briefing (predictions frozen in-briefing) `91f80de8` → ambience
`7ccddb72`.
