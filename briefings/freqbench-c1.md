---
status: active
created: 2026-07-22
for: runpod-b
venue: runpod
---

# FB-C1 — the 12-hour FreqBench overnight session

**You are `runpod-b`** (check `/workspace/.agent_id`; see `agents/README.md`
— the OTHER pod is running the PhenomenonBench session in parallel tonight;
its briefing `stage6-recipe-then-c5.md` is NOT yours). **Governing
protocol:** `experiments/explorations/synthetic/freqbench/LOOP.md` (read in
full; note the cadence section — gated grids ARE in-session), plus `PORT.md`
§ A–B and § G, and README § "The two generators, one substrate". Prime
directive: **a sound verdict, never a win** — an ABORT on any gate is a
success; never tune a task, gate, probe, or tolerance to manufacture a
PROCEED.

**Session limits:** ~12 h wall · **$25 API cap** (judgment on
`claude-fable-5`; spend to `freqbench/results/spend.json` +
`spend_log.jsonl` — NOT the expansion meter) · rewrite
`agents/runpod-b/STATUS.md` before every compact · two-agent shared-branch
rules in `agents/README.md` (pull --rebase before push; append-only shared
files; leaderboard/manifest have union merge drivers). Hard lines: **no
cards beyond the three below, no program-rule/gate edits, no
`temp_bench/core/` edits.** Stop at the acceptance gate even if hours
remain.

## Phase 1 — the widened FreqFrac pass (~2–3 h, start immediately)

All six registry benches, per-token-matched cells: seeds {1, 2, 42} at
T = T_can (=4), plus seed 1 at `--T 8` for the window archs of every bench
(the frequency high-pass check lives at T=8 — PORT.md § G). Run as parallel
processes (use `--tag <bench>_s<seed>_T<T>` so outputs don't collide), sized
to the pod's real CPU quota. The script trains any missing checkpoint via
the canonical trainer (this pod's store starts empty — that is expected),
hard-asserts each reconstructed `train_key` against its leaderboard row (an
assertion failure = STOP and report, never work around), and writes **no
leaderboard rows**.

Deliverables: merged stats (one table: bench × arch × seed × T →
firing-weighted curve, dc_frac, concentration, trained vs untrained-init),
committed under `freqbench/results/` + figs; a short § in PORT.md § G
("full-pass results") stating (a) seed stability of the axis-1 coordinates
and (b) the T=8 frequency high-pass verdict.

## Phase 2 — seed cards FB-2 and FB-3, end-to-end (~4–6 h)

For each card, strictly in this order; **freeze BOTH cards by commit before
constructing either**:

1. **Freeze** (`freqbench/cards/FB-<n>.md`, LOOP.md card format): target
   coordinates + gap claim; exact task parameterization + ground truth
   (Part II § 1 — state F cleanly); proof obligations (ceiling, floor,
   non-triviality — name the argument); regime claim + design-time
   discriminability; the P6 memorization audit (state the template count);
   **frozen per-arch predictions + at least one falsifier**.
2. **Build**: generator in `src/temp_bench/data/synthetic.py` (append-only),
   datasource in `configs/data.yaml` (append-only), evaluator add-on ONLY if
   an existing metric truly cannot serve (additive, protocol stays 1.3.0),
   tests per the existing `tests/test_*_bench.py` pattern.
3. **T1 proof gate**: discharge every obligation — analytic note in the
   card record, or a committed `verify_theory`-style numerical check over
   the ACTUAL parameter range built (port from
   `origin/dmitry-spectral-sprint2:.../code/verify_theory.py`).
4. **T2 non-triviality battery** (all committed): symmetry/relabeling
   audit; bag-of-symbols control (mean-pooled token codes + MLP — must FAIL
   where the card claims order-sensitivity); memorization budget at the
   capacity extremes; probe budget scaled to code dim; shuffle semantics
   stated (per-window independent permutations).
5. **Skeptic** (LOOP.md rubric, Fable): persist raw verdicts pre-parse (the
   C4 ops lesson).
6. **If PROCEED → § 8 gating** (`gating.py` per the changepoint/frequency
   pattern): ceilings, chance floors, the **discriminability STOP-gate**
   (equality variant if the primary latent is order-2). Gate fails ⇒ record
   NON-DISCRIMINATING, no grid — still a success.
7. **If gate passes → the uniform B×A grid** (the locked design: 6 archs ×
   `d_sae ∈ {F//2, F, 2F}` × `T ∈ {1,2,4,8}` × `k_pos ∈ {1,2,4,8,16}` ×
   seeds {1,2,42} + untrained; 30k steps; `run_grid.py` per the frequency
   driver pattern; canonical runner only) → **blind verdict vs the card's
   frozen predictions** in `bench_record.md` → registry entry
   (`experiments/explorations/synthetic/registry.py` Bench row) → REPORT
   re-render → BENCHMARKS.md row (provenance `theorem-first`).

The two cards (details + sprint numbers in PORT.md § A/E and LOOP.md seeds):

- **FB-2 multilane superposition** (priority): 3 simultaneous circle tones
  in orthogonal planes. Ceiling = per-lane periodogram (P5); floor = P1/P2
  per lane; memorization dead by construction (state |Ω|³M³). Frozen
  headline prediction from the sprint: multiband > vanilla (0.96 vs 0.91)
  — under the fair BatchTopK backbone this may FAIL; that is an informative
  negative about the sprint's plain-TopK result, and reporting it is the
  job.
- **FB-3 colored sources**: per-coordinate AR(1) at lag D (math in
  `origin/dmitry-synthetic:src/v6_colored_sources/README.md`). Ceiling =
  lag-D covariance eigendecomposition (CS-2); floor = CS-1 local
  impossibility; frozen prediction = the **W = D+1 phase transition**. This
  is a *feature-direction-recovery* bench (cosine-AUC primary) — say so in
  the card.

## Phase 3 — FB-1 phasepair (only after Phase 2 completes, budget permitting)

Same pipeline. The `c_relevance` skeptic item needs a real answer (which
real phenomenon is phase-coded?); if none is defensible, mark the card
`spanning` with the research reason, or let the skeptic kill it honestly.

## Phase 4 — if hours remain

(a) The **T=16 frequency frontier addendum** (a per-bench `run_grid`
addendum like the earlier bands addendum; T=16 is where spectral hit 1.00
and the Rayleigh/high-pass story is sharpest — includes the FreqFrac pass at
`--T 16`). (b) Port the remaining `verify_theory` checks as permanent tests.

## Acceptance gate — stop for review

Done when: Phase-1 artifacts committed; both Phase-2 cards carry end-to-end
verdicts (or honest gate-kills); every record/registry/STATUS § 0 update
pushed; spend logged. Rewrite `agents/runpod-b/STATUS.md`, append the FB-C1
cycle log to PORT.md, then **STOP** — this briefing stays until mac-local
review, then it is deleted.
