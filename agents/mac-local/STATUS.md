# Working state — agent `mac-local`

**Last rewrite:** 2026-07-22 (Opus 4.8, local CC — pre-compact). Supersedes the
2026-07-21 version.

## Who / where
Local CC on the Mac (Apple M-series, MPS, no CUDA) at `~/research/projects/temp_xc`.
Role: prototyping, review, orchestration. Heavy grids → `runpod`. I'm `mac-local`.

## Git
Branch `arxiv`, everything committed + PUSHED through the overnight-prep
commit (check `git log` — the chain this session: revamp docs `4638a604` →
C4 briefing `31c151d3` → PORT `8e63dcaf` → freqfrac `9e119d1a` → C4 review
close-out `f58bf8a4` → LOOP+addenda `61af93c8` → FB-C1 `7a4dd0fd` → first-pass
results `7e641474`/`603e19bf` → overnight prep HEAD). `origin` = SSH.

## ▶ CURRENT TASK (2026-07-22 evening): two-agent 12-hour overnight QUEUED

Revamp phases 1–3a DONE (two-generator docs; C4 reviewed+APPROVED w/
equality-gate variant adopted; FreqBench port: PORT/proofs/FreqFrac lens
validated on 12 real cells/LOOP rails/addenda). Tonight: `runpod` →
`briefings/stage6-recipe-then-c5.md`; **new agent `runpod-b`**
(identity via `/workspace/.agent_id`; workspace seeded) →
`briefings/freqbench-c1.md` (12-h scope incl. gated grids per LOOP.md's
amended cadence). Collision infra: agents/README two-agent rules +
`.gitattributes` union drivers + separate spend logs + `--tag` on
freqfrac_report. **MY NEXT ACTION (morning): the consolidated review** —
recipe stage-6 verdict (equality-gate integrity FIRST: was the gate honored
if raw-linear windows read e_t?), FB-2/FB-3(/FB-1) end-to-end (freeze-order,
T1/T2, skeptic raw, gate-before-grid, blind verdicts), C5 calibrations,
FreqFrac full-pass table; then delete both briefings + bake findings into
rules. Compute: **no GPU** (A40 kernel-launch-bound ~14% util on these tiny
models).

### The design (kept for context)

The whole recent arc converged on one design. **Do NOT just rebuild FreqBench
in isolation.** Build ONE program:

**Two generators, one substrate** — split by *epistemic anchor*, not two disjoint
programs:
- **FreqBench = theorem-first generator.** Ungrounded synthetic tasks constructed
  from first principles + PROVEN ceilings (local-impossibility bound,
  symmetry/ratio-invariance, periodogram oracle). Anchor = proofs; low drift.
- **PhenomenonBench = data-first generator** (the existing expansion loop). Mines
  real R1-Distill traces → measure→mirror→gate→graduate. Anchor = real LM
  behavior; high drift → keeps the gated-batch/prereg/null/gate-7-8 machinery.
- MUST SHARE: the 6-arch fair-backbone panel, the canonical runner +
  realized-L0/capacity conventions, the coordinate system, and ONE
  `BENCHMARKS.md` with a **provenance tag** (theorem-first vs grounded). Two
  generators → one substrate = one program. Two substrates = the bug the arxiv
  restructure created (it kept the phenomenon substrate, dropped FreqBench's).

**Coordinate system = 3 orthogonal structural axes** ("everything in frequencies"
covers only axis 1):
1. **Spectral** (DC↔AC): waveform of one latent. FreqBench fully theorizes it —
   `FreqFrac(ω)` = per-arch frequency response. Benches: frequency, signed_motion,
   hedging.
2. **Interaction order** (additive↔pairwise-equality↔higher): sum-over-positions
   vs comparison-between. Has the "where the nonlinearity sits" theorem (additive
   provably blind to equality) + the within-window shuffle-gap as a first
   sensitivity probe. Benches: int/eq prize, changepoint's equality piece.
3. **Stationarity/localization** (spread↔clustered): Fourier-vs-wavelet; a
   burst/changepoint is broadband+localized, NOT a frequency. Localization
   measure UNBUILT. Benches: backtracking, changepoint.
Acid test of "principled" = **held-out prediction** (hide a bench, predict the
arch ranking from its coordinates alone) — literally Dmitry's stated goal.

**Build = mostly a PORT (Opus-4.7 → Fable 5), preserving the proofs.** Dmitry's
FreqBench is real code in branches:
- `origin/dmitry-spectral-sprint2:docs/dmitry/sprints/2026-06-10_freqbench_sprint/code/`
  — `fb_core.py` (SpectralTXC unifying vanilla/DC-AC/multiband, ConvDict, TokenSAE,
  oracles, FreqFrac diagnostics), `c7_spectral_arm.py` (spectral lens ON
  backtracking — read its `log.md` for whether it hit the non-stationarity wall =
  axis-3 evidence). Proofs: symmetry-triviality catch, circle embedding =
  classical single-tone estimation, ratio invariance.
- `origin/dmitry-synthetic:src/v6_colored_sources/` — AR(1) colored sources, lag-D
  phase transition at `W = D+1` (a memory-depth flavor).
- Redo = port → `temp_bench` v2 plugin, rerun empirics on **Fable 5** with the
  shared arch panel, and **wire FreqFrac to coordinatize the EXISTING
  frequency/signed_motion/hedging benches into the shared REPORT** (lands
  integrated, not standalone). NB: FreqBench's `ac_sign` task **== the repo's
  `signed_motion`** — same task, forked; the port should let FreqBench's proof
  *explain* signed_motion's NEGATIVE.

**Carry-forward correction from the stage-6 review (a NEW gate):** promote raw
per-token↔window separation from a logged §8 stat to a **discriminability
STOP-gate**. Both stage-6 benches came out NEGATIVE/SPLIT because they cannot
discriminate BY CONSTRUCTION (assumption: order-1 mirror ⇒ `s_i` sufficient;
hedging: ambient DC per token) — and §8 gating already measured this
(0.464 vs 0.466; +0.006 headroom) BEFORE 990 cells were spent. PhenomenonBench
must gate on **arch-separability, not just mirror-validity**. "Grounded + valid
mirror ≠ discriminates" — you need the structural feature (order-2 /
integration-requirement), which is exactly what the coordinate system encodes.

## Stage 6 — DONE + REVIEWED + PASSED (2026-07-22, by me)
measure→mirror→**bench** closed for the first time. Both grounded SPECs
(`assumption_consequence`, `hedging_drift`) built + blind-evaluated + reported
(REPORT 54/54, BENCHMARKS ✗→✓). Review PASSED clean: blind predictions frozen in
expansion cycles 1–3 (predate the grid), **990/990 rows code-versioned**, 0
failures, **0 duplicate eval_keys** (race fix real in `grid.py`), 13 tests
re-passed locally, verdicts honest + self-critical. Both verdicts NEGATIVE/SPLIT
— the anti-"research-sin" discipline working. Briefing deleted + committed.

## Standing context (full science in research `STATUS.md` §0; team/venue in memory)
- **A. Architecture B×A comparison** — done + mature (`REPORT.md`; "where the
  nonlinearity sits").
- **B. Grounded expansion** — 3 cycles + stage-6 bench done. **int/eq prize still
  UNCLAIMED** (needs a hierarchical-categorical mirror).
- **Paper context** (from the 4 meeting transcripts, now in `private/transcripts/`,
  gitignored — read this session): TXC paper, NeurIPS 2026 (reviews out
  ~2026-07-22, rebuttal due 07-27) + ICML'26 workshop. Team: Dmitry (lead),
  Han=user, Aniket, Andre, Bill. See memory `project-txc-paper-context`,
  `project-synthetic-program-why`, `project-autoresearch-revamp`.
- **Paper snapshot** now in `paper/` (read-only agent context, not built here).

## Trackers (don't re-derive)
- `experiments/explorations/synthetic/BENCHMARKS.md` — registry (+2 stage-6 rows).
- `expansion/LEDGER.md` — grounded coverage grid.
- `REPORT.md` — head-to-head (auto-gen, 54/54).
- research `STATUS.md` §0 — living program state (read first).

## Recent commit chain
stage-6 landed by runpod (`2943dd1a`) → I synced (ff) → reviewed PASS →
close-out (`a96f83f0`) → paper+private scaffolding (`266dc386`).
