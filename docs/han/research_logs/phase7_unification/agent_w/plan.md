---
author: Han
date: 2026-04-29
tags:
  - design
  - in-progress
---

## Agent W — pre-registered Phase 1 plan + Phase 2 hill-climb scaffold

> Pre-registration is per CLAUDE.md rule "reproduce before extending"
> and the brief's "write decisions BEFORE running". Phase 1 cells +
> success criteria are locked here; Phase 2 axes are listed but the
> ordering depends on the Phase 1 winner.

### Critical metric note (READ FIRST)

The brief's primary metric is **peak success at coherence ≥ 1.5** under
family-normalised paper-clamp (Q1.3). This is *not* the metric Y reports
in `agent_y_paper/2026-04-29-y-tx-steering-magnitude.md` — Y reports raw
peak success. The two diverge sharply:

| arch | raw peak suc (Y) | constrained peak suc (this metric) | s_norm at constrained peak |
|---|---|---|---|
| tsae_paper_k20 | 1.80 | **1.10** | 5 |
| tsae_paper_k500 | 1.27 | 1.13 | 5 |
| txc_bare_antidead_t5 (cell A) | 0.93 | 0.67 | 5 |
| agentic_txc_02 (cell B) | 1.07 | 0.38 | 5 |
| phase57_partB_h8_bare_multidist_t5 | 1.14 | 0.53 | 5 |
| phase5b_subseq_h8 | 1.00 | 0.23 | 0.5 |

Under coh ≥ 1.5, T-SAE k=20's baseline is **1.10**, not 1.80. Win
threshold is **1.37** (= 1.10 + 0.27 σ_seeds). I will report BOTH
raw and constrained peaks per cell and flag the discrepancy in the
synthesis writeup.

A second observation that follows: window archs at k_pos=100 trail
badly under coh ≥ 1.5 (0.38–0.67) precisely because they over-drive
coherence at peak strengths (their `<|z|>` is 3–6× larger than per-token
archs). A k_pos=20 TXC may win on coherence alone — lower per-position
magnitudes mean gentler steering at fixed s_norm, which keeps coh
above the 1.5 cliff longer.

### Phase 1 — sweep cells (locked)

Six cells, four fresh trainings, two reused. All evaluated on the
existing Q1.3 pipeline.

| cell | arch_id (proposed) | T | k_pos | k_win | family | seed | status | est. cost |
|---|---|---|---|---|---|---|---|---|
| A | `txc_bare_antidead_t5` | 5 | 100 | 500 | TXCBareAntidead | 42 | reuse — local grades.jsonl 210 rows | 0 (graded) |
| B | `agentic_txc_02` | 5 | 100 | 500 | MatryoshkaTXCDRContrastiveMultiscale | 42 | reuse — local grades.jsonl 210 rows | 0 (graded) |
| C | `txc_bare_antidead_t3_kpos20` (NEW) | 3 | 20 | 60 | TXCBareAntidead | 42 | train + grade | ~30–60 min train + ~45 min grade |
| D | `txc_bare_antidead_t5_kpos20` (NEW, **meeting cell**) | 5 | 20 | 100 | TXCBareAntidead | 42 | check Y first via `git log --grep "[meeting cell]"`; if absent, train + grade | ~40–60 min train + ~45 min grade |
| E | `agentic_txc_02_kpos20` (NEW) | 5 | 20 | 100 | MatryoshkaTXCDRContrastiveMultiscale | 42 | train + grade | ~50 min train + ~45 min grade |
| F | `txc_bare_antidead_t10_kpos20` (NEW) | 10 | 20 | 200 | TXCBareAntidead | 42 | train + grade | ~60–90 min train + ~45 min grade |

**Training order** (cheapest + most-likely-winner first, to maximise
information per A40-hour):

1. **Cell D first** (Y's exact spec from `agent_y_paper/2026-04-29-y-z-handoff.md`; meeting cell, hardware-feasible, theoretically the cleanest minimum-deviation experiment from T-SAE k=20).
2. **Cell C** (smaller T → fastest train; tests "is k_pos=20 alone the lever?" with minimum receptive field).
3. **Cell E** (matryoshka adds InfoNCE losses on top of D's recipe; tests whether the multi-scale contrastive head helps at sparse k_pos).
4. **Cell F** (largest T → slowest train; tests whether a wider window at sparse k_pos still pays off, or if the sparsity-collapse regime starts dominating).

This order also frontloads coordination: **cell D is the meeting cell with Y**, so doing it first either reuses Y's work or saves Y the training. Tag the cell-D commit `[meeting cell]`.

**Warm-start:** for cells C, D, F (TXCBareAntidead family), initialise from the local `tsae_paper_k20__seed42.pt` checkpoint by tiling the per-token encoder across T positions and dividing the decoder by T (so the right-edge sum reproduces T-SAE's per-token output at init). For cell E (matryoshka multiscale), warm-start initialises only the bare encoder/decoder; the multi-scale contrastive heads stay random. Per the brief, this cuts training 5–10× and turns each cell into a true minimum-deviation experiment from T-SAE k=20.

### Phase 1 — pre-registered outcomes

Threshold = ±0.27 (= 1× σ_seeds observed by Y). Apply to **constrained peak suc (coh ≥ 1.5)** as the primary metric, but also log raw peak for cross-reference with Y.

- *At least one Phase-1 cell beats T-SAE k=20 by ≥0.27 (constrained peak ≥ 1.37)* → architecture genuinely matters at matched sparsity. Verify the candidate at seed=1; if confirmed, lock as Phase 2 starting cell. **Headline result candidate.**
- *No cell beats by ≥0.27, but at least one ties (within ±0.27 of 1.10 i.e. constrained peak ∈ [0.83, 1.37])* → ambiguous. Run multi-seed on the best candidate. If still tied → "sparsity is the dominant lever, architecture is secondary"; ask Han before Phase 2.
- *Every cell loses to T-SAE k=20 by ≥0.27 (constrained peak ≤ 0.83)* → architecture has anti-prior at matched sparsity. Phase 2 unlikely to recover. Document per-cell + per-class breakdown to identify which axis hurts most. **Publishable converging null** with Y.

### Phase 2 — greedy local hill-climb (4 steps max)

Brief's 4 axes: T → T±1, k_pos → {⌊k_pos/2⌋, k_pos×2}, family ∈ {TXCBareAntidead, MatryoshkaTXCDRContrastiveMultiscale, TXCBareMultiDistanceContrastiveAntidead}, decoder write-back ∈ {right-edge, per-position}.

**Additional axes I noticed reading `canonical_archs.json` + `paper_archs.json`:**

5. **Subseq sampling** (`SubseqH8`-style — Phase 5 mp champion at k_pos=100; never tested at k_pos=20).
6. **k_win > T·k_pos (anchor regime)** — like Group 5's `txcdr_t20_kpos100` (k_win=2000) decouples per-position cap from window total. A `(T=5, k_pos=20, k_win=200)` cell would let the encoder pick more features overall while staying sparse per position.
7. **Matryoshka H fraction** (currently fixed 0.2·d_sae; never swept).
8. **Multi-distance shift schedule** (auto-scaled `(1, T//4, T//2)` is canonical; alternatives unexplored).

If Phase 1 lands a strong winner, the first hill-climb step in Phase 2 can choose from the **expanded list above** (axes 5-8) rather than restricting to the brief's 4. The brief was written before this reconnaissance, so this widens the search space without violating its intent — every axis is still atomic single-step.

**Iteration cap.** 4 hill-climb steps after the sweep winner. If the metric is still strictly increasing at step 4, ask Han before continuing.

**Stage gate.** If at any step the swept winner is *not* improved by any of the perturbations (true local maximum), stop. Document the local-max cell + the perturbation table (which axis hurts least, which hurts most). That's a paper artefact regardless of sign.

### Pipeline (per-cell, after training)

For each new arch (D, C, E, F in that order):

1. `_download_ckpts.py` (skip if just trained).
2. `select_features.py --archs <arch_id>`.
3. `diagnose_z_magnitudes.py --archs <arch_id>` — REQUIRED for normalised intervene.
4. `intervene_paper_clamp_normalised.py --archs <arch_id>` — full default `s_norm` grid {0.5, 1, 2, 5, 10, 20, 50}.
5. `grade_with_sonnet.py --archs <arch_id> --subdir steering_paper_normalised --n-workers 1` (shared rate limit with Y).
6. Verify all 210 rows graded, 0 errors.
7. Plot `success(s_norm)` and `coh(s_norm)`; check peak is interior. If peak at top → extend grid up to {100, 200}; bottom → extend down to {0.05, 0.1, 0.2}.
8. Compute primary metric: peak success at coh ≥ 1.5. Compute secondary: raw peak; AUC of suc-vs-coh from 1.0 to 3.0; per-concept-class breakdown.

### Disk + RAM budget

- 46 GB pod RAM, 46 GB A40 VRAM, 900 GB volume quota.
- Each TXCBareAntidead ckpt at d_sae=18432 ≈ 0.8–1.7 GB depending on T. 4 fresh ckpts + ≤4 hill-climb steps ≤ 14 GB local. Comfortably fits.
- Activation cache (Gemma-2-2b L12 resid_post, 24k seq × 128 tok × 2304 d × 2 B = 14.2 GB) — build ONCE in `data/cached_activations/gemma-2-2b/fineweb/L12/` and reuse across all cells. Currently NOT BUILT locally; first training will need to build it (~10–15 min Gemma forward pass).
- Training memory at b=4096: TXCBareAntidead ≈ 24 GB peak per `paper_archs.json`. Fits.

### Coordination with Y

- **Cell D = Y's Step 2 = meeting cell.** Pull + grep before training; tag commit `[meeting cell]` after.
- **Anthropic 50 req/min ceiling shared.** Use `--n-workers 1` if Y is grading concurrently. Watch `/tmp/*grade*.log` for 429 / 0.0 gen/s.
- **GPU contention.** Single A40. Stagger trainings; check `nvidia-smi` before launching.

### Files

- This plan: `docs/han/research_logs/phase7_unification/agent_w/plan.md`.
- Phase-1 synthesis (TBD): `2026-04-29-w-phase1-sweep.md`.
- Phase-2 writeup (TBD): `2026-04-29-w-phase2-hillclimb.md`.
- Final summary (TBD): `summary.md`.
- Plots: `plots/<plot>.png` + `<plot>.thumb.png`.
