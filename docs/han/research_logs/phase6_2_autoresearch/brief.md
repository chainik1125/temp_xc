---
author: Han
date: 2026-04-23
tags:
  - proposal
  - in-progress
---

## Phase 6.2 brief: autoresearch toward TXC-family parity with `tsae_paper`

### Context and motivating finding

Phase 6.1's rigorous metric (N=32, Haiku temp=0, multi-judge, +
`concat_random` 7-passage FineWeb control) revealed that the original
Phase 6.1 headline "Cycle F (TXC + BatchTopK) reaches 7/8 on
concat_A+B, parity with tsae_paper" was a **curated-concat artefact**.
See [[../phase6_qualitative_latents/summary|summary §9.5]] (pending
update) for the full numbers.

Headline data (seed=42, N=32, this is what Phase 6.2 starts from):

| arch | concat_A | concat_B | concat_random | passage_cov(random) |
|---|---|---|---|---|
| `agentic_txc_02` (baseline) | 17 | 16 | **0** | 7/7 |
| `agentic_txc_02_batchtopk` (Cycle F) | 20 | 13 | **0** | 6/7 |
| `agentic_txc_09_auxk` (Cycle A) | 21 | 13 | **0** | 7/7 |
| `agentic_txc_11_stack` (Cycle H) | 21 | 12 | **0** | 7/7 |
| **`agentic_txc_10_bare` (Track 2)** | 20 | 19 | **5** | 7/7 |
| `agentic_mlc_08` | 18 | 18 | 2 | 4/7 |
| `tsae_ours` | 17 | 19 | 3 | 6/7 |
| `tfa_big` | TBD | TBD | 0 | 2/7 |
| **`tsae_paper`** | 23 | 18 | **12** | 7/7 |

Under the uncurated-text metric (`concat_random`):

- The anti-dead stack (unit-norm decoder + decoder-parallel gradient
  removal + geometric-median `b_dec` init + AuxK) **alone** lifts TXC
  from 0/32 → 5/32 (Track 2 vs baseline/Cycle F/H).
- **BatchTopK, AuxK alone, and BatchTopK+AuxK all score 0/32 on random.**
  The curated-concat gains from these are passage-content artefacts.
- `tsae_paper` genuinely reaches 12/32 — it is the sole arch with
  substantial concept-level features on uncurated text.
- `tfa_big` also collapses (0/32, coverage 2/7 — extreme concentration).

Gap to close: `tsae_paper` (12/32) vs `Track 2` (5/32). **Target: TXC
arch ≥ 10/32 concat_random while preserving Phase 5 probing utility
(Δ AUC ≤ 0.02 vs `agentic_txc_02` mean-pool 0.7987).**

### Mechanism axes to explore

`tsae_paper` differs from `Track 2` on four axes. Phase 6.2 ablates
each:

| axis | Track 2 | tsae_paper | effect hypothesis |
|---|---|---|---|
| matryoshka H/L split | none | 20%/80% | hierarchical features, possibly generalises |
| temporal contrastive | none | single-scale α=1.0 | explicit semantic-smoothness regulariser |
| sparsity | TopK k=100/pos | BatchTopK k=20/pos | variable per-sample budget |
| inference threshold | greedy TopK | EMA threshold | different active features at test time |

Prior finding (Phase 6.1): **BatchTopK by itself doesn't generalise**
(Cycle F → 0/32 random). So the `tsae_paper` advantage must come from
one or more of {matryoshka, contrastive, inference threshold} on top
of the anti-dead stack — OR from their interaction with each other.

### Candidate architectures (6-cycle autoresearch)

Each candidate forks Track 2's base (bare window TXC + full anti-dead
stack, TopK k=100) and toggles one or two axes. All at T=5, d_sae=18432,
seed=42, plateau-stop (min_steps=3000 unless noted).

| ID | name | recipe | axis tested | cost |
|---|---|---|---|---|
| C1 | `phase62_c1_track2_matryoshka` | Track 2 + 20/80 H/L matryoshka loss | matryoshka alone | ~35 min |
| C2 | `phase62_c2_track2_contrastive` | Track 2 + InfoNCE(α=1.0) on adjacent-token pairs | contrastive alone | ~40 min |
| C3 | `phase62_c3_track2_matryoshka_contrastive` | Track 2 + matryoshka + contrastive (≈ tsae_paper on TXC base) | full stack | ~45 min |
| C4 | `phase62_c4_track2_threshold` | Track 2 + EMA threshold at inference | inference mechanism | 0 min retrain (re-eval ckpt) |
| C5 | `phase62_c5_track2_longer` | Track 2 with `min_steps=10000` | training duration | ~60 min |
| C6 | `phase62_c6_bare_batchtopk_longer` | 2x2 cell (Phase 6.1 #4) with `min_steps=10000` | duration × sparsity interaction | ~60 min |

**Expected candidate order under the autoresearch selector:**
- **Cycle 1** picks C3 (highest-prior: reconstructs tsae_paper on the
  TXC base; if it matches ≥10/32 random, the TXC-parity story holds
  with a clean recipe).
- **Cycles 2-3** differ based on C3's outcome:
  - If C3 hits target → run C5 + C1 to understand which of {longer
    training, matryoshka alone} delivered it.
  - If C3 matches Track 2 only (5-7/32 random) → matryoshka +
    contrastive aren't the lever; run C4 (threshold inference) and
    C2 (contrastive alone) to isolate.
  - If C3 < Track 2 → some interaction is harmful; fall back to
    ablation cycles.
- **Cycles 4-6** are agent-proposed conditional on prior results.

### Fitness function

Primary: `concat_random x/32` (at seed=42, temp=0, multi-judge).
Secondary (tie-breaker): `concat_A+B combined x/64`.
Constraint: **probing Δ AUC ≤ 0.02 vs `agentic_txc_02` mean-pool
0.7987** — if a candidate violates this, it's rejected even if
qualitative wins.

Coverage (`k/P`) is a diagnostic; at our passage counts (P=3, 4, 7)
most archs saturate at full coverage, so it's not discriminative for
ranking — just reported.

### What autoresearch does and does NOT do here

**In scope** for a modest 6-cycle loop:
- Run each candidate at seed 42 → evaluate on A, B, random → record.
- After each cycle, have a proposer (Claude Sonnet) look at the
  growing result table and suggest the next candidate from the
  remaining set (with rationale + predicted outcome).
- Stop at 6 cycles or when fitness plateaus.

**Out of scope** for this phase (deferred):
- Seed variance (Phase 6.3) — pick winner, then retrain at seeds {1, 2}.
- Full hyperparameter sweep inside each mechanism (e.g. α, auxk_alpha,
  T sweep).
- Novel architectures beyond the `tsae_paper` axes.

### Budget

- GPU: ~4 hr total (6 × ~40 min average).
- API: ~$3 (6 × autointerp pass at ~$0.5, plus ~$0.5 Sonnet proposer
  calls).
- Disk: 6 new ckpts × 1.4 GB ≈ 9 GB.

Well within the user's 12-hr / $10 budget IF Phase 6.1 finishes in
~3-4 hr.

### Prerequisites

1. Phase 6.1 full pipeline completes (triangle seed-variance +
   probing + autointerp on new cells). [[../phase6_qualitative_latents/2026-04-23-handover-post-compact]]
2. Phase 6.1 §9.5 rewritten with the rigorous-metric baselines.
3. A new file `src/architectures/txc_bare_{variant}.py` per candidate
   that requires a new forward path (C1, C2, C3). C4/C5/C6 reuse
   existing classes.

### Launch instructions

```bash
bash experiments/phase6_2_autoresearch/run_phase62_loop.sh
```

Runs sequentially in foreground (so you can interrupt between
cycles). Each cycle:
1. Picks next candidate via `propose.py` (or `--candidate C1` explicit).
2. Trains the arch (if not cached).
3. Encodes on concat A/B/random.
4. Runs autointerp pipeline.
5. Appends row to `results/phase62_results.jsonl`.
6. Emits a summary line to stdout.

See [[./README]] in the experiments dir for per-candidate details.

### Deliverables

- `results/phase62_results.jsonl` — per-cycle (arch, metrics) rows.
- `results/phase62_summary.md` — final table + winner call.
- `docs/han/research_logs/phase6_2_autoresearch/summary.md` — end-
  of-phase write-up with headline + caveats + follow-ups.
