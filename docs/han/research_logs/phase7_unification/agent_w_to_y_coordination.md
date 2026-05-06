---
author: Han
date: 2026-05-01
tags:
  - design
  - in-progress
---

## W → Y coordination — methodology, code, baseline, arch changes (paper push)

> Hi Y — this doc tracks what W has changed/added on the Phase 7 case-study
> codebase between 2026-04-30 and 2026-05-01 so we don't diverge on
> methodology, baseline numbers, or analysis scripts during the paper push.
> If you disagree with anything below, please push back in this file (or
> a follow-up) before re-running your dashboards/leaderboards on top of W's
> commits.

### 1. New architectures W has added (committed on `han-phase7-unification`)

| arch_id | src_class | merge mech | when | status |
|---|---|---|---|---|
| `txc_maxpool_h8_t2_kpos20_shifts2` | `TXCMaxPoolMergeH8` | `z[s] = max_t (x[t] @ W_enc[t,:,s])` | 2026-04-30 | n=3 multi-seed |
| `txc_contrastive_h8_t2_kpos20_shifts2` | `TXCContrastiveMergeH8` | `z[s] = (x[T-1] @ W_enc[T-1,:,s]) - (x[0] @ W_enc[0,:,s])` | 2026-04-30 | n=3 multi-seed |
| `txc_multiplicative_h8_t2_kpos20_shifts2` | (FAILED — too slow to converge, killed) | logsum-exp merge | 2026-04-30 | abandoned |
| `txc_concat_h8_*` | (skeleton only — `TXCConcatMergeH8` lite variant; not yet trained) | per-position no-merge "lite" (avg fallback) | 2026-04-30 | not trained |

All registered in `src/architectures/__init__.py`; `WINDOW_CLASSES` in
`experiments/phase7_unification/case_studies/_arch_utils.py` updated.
Loaders also added to `run_probing_phase7.py` for the 3 trained mystery archs.

If you train a new arch, please:
- Subclass `ArchSpec` in `src/architectures/`.
- Register in `__init__.py`.
- Add to `WINDOW_CLASSES` if it's a window encoder.
- Add load dispatch in `run_probing_phase7.py` if you want it in the probing pipeline.
- Add inventory entry in `plot_unified_pareto.py` (and `plot_focused_pareto.py` if it's a "best 4" candidate).

### 2. New analysis scripts W has added

| script | purpose |
|---|---|
| `case_studies/steering/plot_focused_pareto.py` | T-SAE + 3 best TXCs only — clean Pareto. |
| `case_studies/steering/plot_mystery_arch_per_class.py` | Cross-arch per-class signature bar plot. |
| `case_studies/steering/build_definitive_table.py` | Updated to include 5 mystery-arch n=3 cells (RE/PP for Mystery + V6 for contrastive). |

Outputs go to `results/case_studies/plots/`. JSONs go to `results/case_studies/{contrastive_n3_summary,contrastive_bootstrap_cis,contrastive_per_class}.json`.

### 3. T-SAE k=20 baseline — IMPORTANT methodology question (in flight)

**Status as of 2026-05-01 11:00**: Han flagged that T-SAE k=20 peak success
1.80 looks "suspiciously high". W is auditing and has confirmed:

- Peak unconstrained 1.80 sits at **coh = 1.40** (below prereg coh ≥ 1.5
  floor) on BOTH sd=42 and sd=1.
- Per-seed cliff @ coh ≥ 1.5: sd=42 = **1.100**, sd=1 = **0.300** —
  large σ (the s=50 strength keeps coh=1.667 on sd=42 but drops to 1.40 on sd=1).
- Mean-curve aggregation gives anchor 1.167 (smooths the sd-disagreement at the cliff).
- Per-seed-then-mean gives anchor 0.700 (your earlier σ-pooled 0.70 number).

**Currently retraining T-SAE sd=1 + sd=2 ON THIS POD** (W's pod) for clean
same-pod n=3 anchor. Reason: only sd=42 ckpt is local; sd=1 ckpt was on
your pod. Cross-pod cuDNN non-determinism may have introduced variance.

If you re-run dashboards/leaderboards with the T-SAE anchor, please use:
- W's same-pod n=3 anchor numbers (when available — ETA ~1h from now)
- *not* the sd=42-only or cross-pod sd=42+sd=1 numbers

Will document final n=3 anchor in `agent_w/2026-04-30-w-phase4-results.md`
and update `agent_y_phase2/2026-04-30-y-unified-pareto.md` headline once
data lands.

### 4. Plot inventory (full coh range, NOT just high-coh)

All TXC and T-SAE cells in `plot_unified_pareto.py` / `plot_focused_pareto.py`
were tested on the SAME 7-strength s_norm grid `(0.5, 1, 2, 5, 10, 20, 50)`
× per-arch z_orig magnitude. Each curve traces success vs coh from
low-strength (high-coh, low-succ) → peak → over-steered. Mystery archs
appear in BOTH `unified_pareto_full.png` (dense, 22 archs)
and `focused_pareto_matched_sparsity.png` (4-line clean view).

### 5. Bootstrap CI methodology (consistent with your earlier work)

W's bootstrap uses the same concept-resampled procedure as your
`build_definitive_table.py::bootstrap_ci_peak`: 1000 trials, sample 30
concepts with replacement, recompute mean curve + cliff, take 2.5/97.5
percentiles of Δ(cell − anchor).

CIs on contrastive cells are in `contrastive_bootstrap_cis.json`. Highlights:
- **Contrastive V6 dec-broadcast @ coh ≥ 2.25 AND ≥ 2.5: bootstrap-SIG**
  (CIs strictly positive). Only n=3 cell with strict-coh stat-sig.
- Right-edge n=3 PRREG cliff (Δ=+0.411 point) has wide CI — concept
  variance dominates at the prereg cliff. Per-seed span 0.10 is tighter
  evidence of stability.

### 6. Per-class breakdown (added 2026-05-01)

Cross-mystery-arch per-class breakdown in
`agent_w/2026-04-30-w-phase4-results.md` — five findings:

1. **Sentiment universally TXC-favored** (every TXC × protocol cell wins
   by Δ ≥ +0.167 at coh ≥ 1.5 and ≥ 1.75).
2. **Contrastive-merge** = sentiment-dominant (+0.5 RE, +0.67 V6).
3. **MaxPool** = uniquely stylistic-winning (Δ=+0.43 RE @ coh ≥ 1.75).
4. **OBLIT** = knowledge-dominant at coh ≥ 1.5 (Δ=+0.28).
5. **All TXC archs LOSE on discourse + safety** — STRUCTURAL not arch-specific.

### 7. Open coordination items / asks of Y

- [ ] **Y's confirmation that W's same-pod T-SAE sd=1+sd=2 retraining is
      the right move** (not a re-grade of your existing ckpts). Reply if
      you'd prefer cross-pod determinism testing instead.
- [ ] **Y's read on the per-class story**: does the "sentiment universal
      TXC win" hold up in your bare-antidead matrix too? If so, this is
      a paper-narrative-level finding that should sit at the top.
- [ ] **Y to update `agent_y_phase2/2026-05-01-y-cheatsheet.md`** with
      W's contrastive RE PRREG WIN + V6 strict-coh-SIG cells once Y is
      online again.
- [ ] **Y to flag any pending changes to `build_definitive_table.py`** —
      W has added 5 mystery-arch cells; if Y has untracked new cells,
      they may collide.

### 8. Branch state at last sync

- Branch: `han-phase7-unification`
- Latest pushed (origin): `654734df` (Focused Pareto figure)
- W's commits 1db064a2 → 654734df cover: mystery-arch plots, n=3
  contrastive analysis, bootstrap CIs, per-class breakdown, paper figure,
  unified-pareto md update, focused Pareto, definitive-table update.

If you pulled before 2026-05-01 11:00 UTC, `git pull --ff-only` on this
branch should fast-forward cleanly.

### Files

- This doc: `docs/han/research_logs/phase7_unification/agent_w_to_y_coordination.md`
- W's Phase 4 results (canonical for mystery-arch numbers):
  `docs/han/research_logs/phase7_unification/agent_w/2026-04-30-w-phase4-results.md`
- Y's earlier unified Pareto (now updated by W with mystery-arch context):
  `docs/han/research_logs/phase7_unification/agent_y_phase2/2026-04-30-y-unified-pareto.md`
