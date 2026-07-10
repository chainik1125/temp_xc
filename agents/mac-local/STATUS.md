# Working state — agent `mac-local`

**Last rewrite:** 2026-07-10 (Opus 4.8, local CC session — program-report build).

## Who / where
Local CC on the Mac (Apple M5 Pro, MPS, no CUDA) at `~/research/projects/temp_xc`.
Role: prototyping, review, orchestration. Heavy grids go to `runpod`.
I am the `mac-local` agent (inferred from darwin + this path — no need to be told).

## Git
Branch `arxiv`, **@ `250c3043`** (push pending — see below), clean working tree.
`origin` is the SSH URL (`git@github.com:chainik1125/temp_xc.git`).

## Current thread — program-level B×A report (the comparison substrate)
The quest turned from *more benchmarks* to *a clean comparison foundation*: one
apples-to-apples grid of B benchmarks × A architectures → a single auto-generated
`REPORT.md` (two fairness-convention matrices, all from raw JSON). Full design +
state in the research STATUS §0. **This stretch (all committed):**
- `20a5e46a` realized-L0 evaluator increment (`l0_per_token`/`l0_per_window` on
  the shared dispatcher, additive, protocol 1.2.0). Validated: nominal `k_pos=4`
  at T=8 realizes 4 / 2.06 / 0.5 atoms-per-token for token / pre / post — why
  matching keys on realized L0, not the knob.
- `b9773cde` the foundation: `experiments/explorations/synthetic/registry.py`
  (B×A spec + canonical op `d_sae=F, T_can=4, B*=4`), `src/explorations/synthetic/
  report.py` (matrix builders, both conventions reduce to matching `l0_per_token`),
  `render_report.py`, `REPORT.md`, `tests/test_program_report.py` (5 pass).
- `250c3043` research STATUS §0 + the RunPod re-grid briefing.

Renders now; matrix empty (historical rows lack L0) — expected. Coverage block is
the live signal (surfaced: **signed_motion is TopK-legacy only → needs fresh
fair-backbone runs**).

## Next / immediate
- **PUSH** `250c3043` to `origin/arxiv` (was pending at last write).
- The re-grid is **queued for runpod**: `briefings/uniform-regrid-program-matrix.md`
  (bump eval protocol→1.3.0 to force L0 on every cell, sweep the `(T,k_pos)`
  lattice, fill both matrices, regenerate records zero-drift). Not mine to run
  (heavy grid). When runpod finishes: review the filled matrices + drift gate.

## Prior stretch (context — all committed + pushed)
- Frequency (cyclic-tone) bench built by `runpod` (verdict POSITIVE); the
  record-pipeline refactor (shared `src/explorations/synthetic/` lib);
  signed_motion figs-path fix (`37019b6d`); the agents/ + briefings/ coordination
  model (`5f586be9`). Full science in the research STATUS.

## Open (lower priority; not this thread)
- **Calibration debt:** the frequency `bench_record.md` banner still leads with
  "DCT-band decisive"; the honest core is "position-mixing before the nonlinearity
  + budget + circle geometry" (multiband ≈ vanilla at matched `k_win`). Prose-only.
- **signed_motion `<!-- AUTO:* -->`** block is still hand-maintained (not wired to
  `record.py`) — the re-grid briefing will regenerate it, so may resolve itself.
