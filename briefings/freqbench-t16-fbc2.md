---
status: active
created: 2026-07-23
for: runpod-b
venue: runpod
---

# FB-C2 — T=16 frontier addendum, verify_theory ports, and the rotation acid test

**You are `runpod-b`** (check `/workspace/.agent_id`). Two other agents run
tonight (`runpod`: expansion C6; `runpod-c`: conversion-depth, GPU) —
shared-branch rules in `agents/README.md`. Governing protocol: `LOOP.md`
(note the NEW strict commit-then-run rule in T3 — commit gating scripts
BEFORE first execution, amendments as their own commits). Prime directive:
**a sound verdict, never a win.**

**Session limits:** ~12 h wall · **$25 API cap** (skeptic on
`claude-fable-5`; spend to `freqbench/results/spend.json`) · rewrite
`agents/runpod-b/STATUS.md` before any compact · hard lines: **no cards
beyond FB-4 below, no program-rule/gate edits, no `temp_bench/core/`
edits.** Stop at the acceptance gate even if hours remain.

## Phase 1 — the T=16 frontier addendum (~3–4 h, start immediately)

Per-bench `run_grid` addenda (the bands-addendum pattern) adding
`T = 16` × `d_sae ∈ {F//2, F, 2F}` × `k_pos ∈ {1,2,4,8,16}` × seeds
{1,2,42} + untrained for the three tone benches: **frequency, multilane,
phasepair** (6 archs; canonical runner; ~270 cells/bench). Plus the
FreqFrac pass at `--T 16` for the window archs of the three benches.

Three frozen predictions to score the addendum against (frozen here, by
mac-local, 2026-07-23 — append reasons, never directions):

1. **multilane:** the 4-band > 1-band margin (T=8: +0.019) **vanishes or
   inverts** at T=16 (≤ +0.01) — the band advantage is a coarse-window
   scarcity effect (the FB-2 record's extrapolation).
2. **phasepair:** spectral sign recovery **rises above its T=8 value
   0.936** (every DCT band multi-index ⇒ quadrature partners everywhere);
   post stays ≈ 1.000.
3. **frequency:** spectral reaches ≈ 1.00 (the sprint's T=16 result,
   now under the fair BatchTopK backbone) and the FreqFrac high-pass
   sharpens (Rayleigh cells resolve the full Ω ladder).

Deliverables: addendum sections in the three bench records (blind vs the
predictions above), merged FreqFrac table update in PORT § G, REPORT
re-render.

## Phase 2 — verify_theory ports (~1–2 h)

Port the remaining `origin/dmitry-spectral-sprint2` `verify_theory.py`
checks as permanent tests under `tests/` (the analytic-test pattern of
`test_freqfrac.py`): P2 phase-averaging, P5 periodogram=ML + Rayleigh,
CS-2 lag-D eigen-recovery at the built parameterizations. Tests must run
in seconds (small N) and pin the proofs to the actual generators.

## Phase 3 — card FB-4 `rotated_multilane`: the subtype-rule acid test (~4–6 h)

The review baked the **order-2 subtype rule** into the README coordinates
(phase→post · power/equality→spectral · covariance→pre). FB-4 attacks its
sharpest confound: **is spectral's power/equality dominance generic
order-2-even structure, or DCT-basis alignment?**

**Construction (freeze the card first — `freqbench/cards/FB-4.md`,
LOOP.md format, committed pre-build):** the multilane generator composed
with a **fixed Haar-random orthogonal rotation Q of the d_in embedding**
(seeded, stated in the card). Everything else identical to multilane —
same tones, lanes, σ, ground truth. All multilane proofs (P1/P2 floors,
P5 per-lane ceiling) are rotation-invariant — restate, do not re-derive;
the ONLY thing that changes is the alignment between the latent's planes
and every architecture's basis. This makes multilane-vs-FB-4 a
*controlled experiment on basis alignment* — same task coordinates, one
knob.

**Predictions to freeze in the card** (mac-local's priors — you may
sharpen reasons, not directions): per-token/stacked/pre stay ≈ 0
(rotation cannot create additive access); **spectral's untrained access
prior collapses** (+0.298 → ≈ 0 — its DCT kernels no longer align);
whether *trained* spectral recovers its multilane margin over post is the
open question the bench decides — a full recovery says the win is
learned order-2-even conversion (subtype rule survives as stated); a
collapse toward post parity or below says the "power → spectral" leg is
**alignment-conditional** and the README subtype rule gains an alignment
qualifier (that outcome is just as valuable — say so in the card).
Include a falsifier (any arch > 0.1 at T=1 ⇒ rotation bug leaking
per-token access).

Then the standard pipeline: build (datasource append + rotation option on
the generator, tests) → T1/T2 (bag control, memorization budget unchanged
by rotation — state why) → **strict commit-then-run § 8 gating** →
skeptic → uniform grid (the locked design, T ∈ {1,2,4,8} — T=16 NOT
included here) → blind verdict → registry/BENCHMARKS/REPORT.

## Acceptance gate — stop for review

Phase-1 addendum records + Phase-2 tests + FB-4 end-to-end verdict (or
honest gate-kill) committed and pushed; spend logged; STATUS rewritten;
cycle log appended to PORT § H. Briefing stays until mac-local review.
