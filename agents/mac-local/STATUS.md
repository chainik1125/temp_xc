# Working state — agent `mac-local`

**Last rewrite:** 2026-07-10 (Opus 4.8, local CC session — program-report build).

## Who / where
Local CC on the Mac (Apple M5 Pro, MPS, no CUDA) at `~/research/projects/temp_xc`.
Role: prototyping, review, orchestration. Heavy grids go to `runpod`.
I am the `mac-local` agent (inferred from darwin + this path — no need to be told).

## Git
Branch `arxiv`, level with `origin/arxiv` after the commit below.
`origin` is the SSH URL (`git@github.com:chainik1125/temp_xc.git`).

## Current thread — program-level B×A report (the comparison substrate)
The quest turned from *more benchmarks* to *a clean comparison foundation*: one
apples-to-apples grid → a single auto-generated `REPORT.md` from raw JSON. Full
design + state in the research STATUS §0. **Design settled over a long design pass
(all committed):** match **per-token sparsity** on realized `l0_per_token` (only
convention — per-window was dropped as residue); canonical cell `T_can=4, B*=2`;
capacities `{F, F//2}` uniform (F per-bench; frequency F=101=alphabet M, circle
rank-2); `T∈{2,4,8}`, `L=32, seq_len=64`. Matrix rows = `(bench, latent-axis)`
(L=6, not B); **companion panels** (NMSE + eauc, per-bench A×B) = the
capability-vs-artifact gate. Key commits: `20a5e46a` L0 increment · `b9773cde`
foundation · `672bb120` {F,F/2}-for-all + per-token primary · `89ba17ce` drop
per-window · `40b130dc` companion panels. Renders now; matrix + panels empty
(historical rows lack L0) — coverage block shows what exists (signed_motion is
TopK-legacy only).

## Next / immediate
- **USER ORDERED (2026-07-10): a FULL clean-room rerun + scoped purge** — briefing
  `briefings/full-rerun-and-purge.md` (queued for runpod; I recommended the
  smaller canonical-cell re-grid, user overrode). Rebuild the entire synthetic
  result set from scratch at protocol 1.3.0 under the new design, regenerate ALL
  records (per-bench + program), then purge stale synthetic rows (scoped to the 5
  synthetic datasources — NEVER other experiments; purge LAST). ⚠️ ~20h est. > 12h
  → shard by bench. Not mine to run (heavy grid).
- When runpod finishes: review the filled matrix + panels, the purge (confirm
  non-synthetic rows untouched), and sanity-check recovery vs the old records.

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
