---
author: Han
date: 2026-04-25
tags:
  - design
  - in-progress
---

## Phase 7 brief — unification and methodological consolidation

### Why this phase exists

Across Phase 5, 5B, 6, 6.2, 6.3 we have accumulated ~60+ trained
architectures, ~3 leaderboards, two branches diverged from `han`
(`han-phase5b` and `han-phase6`), a deprecated aggregation
(`full_window`), and three known methodological weaknesses:

1. **Subject model is `google/gemma-2-2b-it`**, not `google/gemma-2-2b`
   (base). The base model is what T-SAE (Ye et al. 2025) and TFA
   (Lubana et al. 2025) train on. Using IT introduces a confound that
   reviewers will flag and that prevents direct comparison to
   pretrained Neuronpedia / Gemma-Scope SAEs.

2. **Sparsity convention varies across families.** Phase 5's
   convention has been `k_pos = 100, k_win = k_pos × n_active_positions`,
   which means TXC T=5 has k_win=500 active features in z while MLC
   has k_win=100. The probe sees representations of differing
   sparsity, biasing comparisons in ways that conflate "feature
   quality" with "feature count."

3. **Cluttered leaderboard with two aggregations and ambiguous
   framing.** `last_position` and `mean_pool` were treated as
   independent metrics, but lp implicitly assumes a "the z represents
   the last token" framing that is architecturally unjustified for
   TXC (the encoder is symmetric over T positions; nothing privileges
   the last one). After working through this with the user, we
   consolidate to ONE leaderboard.

NeurIPS deadline is 2026-05-05 (10 days from now). Phase 7 is the
final pass: re-run the most informative archs on a sanitized setup
that addresses (1) and (2), present a single unified leaderboard, and
write the paper.

### The four decisions

#### (i) Subject model: `google/gemma-2-2b` (base)

Hard requirement. Rebuild the activation cache. Do not mention
gemma-2-2b-it in the paper.

Practical implications:

- Re-extract activations from Pile-Uncopyrighted (or stay on FineWeb;
  T-SAE/TFA use Pile, we use FineWeb — minor methodological
  difference, document but don't change). Layer 13 anchor + L11–L15
  for MLC.
- Rebuild the 36-task probe cache against the new model.
- Rebuild the 4-passage Phase 6 concat-A/B/random qualitative cache
  against the new model.

#### (ii) Convention: fix `k_win = 100` across all archs

Every arch's per-z output has 100 active features after TopK,
regardless of T. Concretely:

- per-token SAE (TopK / BatchTopK / matryoshka_BatchTopK):
  k = 100 per token.
- TXC at any T: k_win = 100 (so per-position-slab budget k_pos = 100/T).
  At T=5, k_pos=20. At T=10, k_pos=10.
- MLC at L=5 layers: k_win = 100 (matches per-token SAE convention).
- Anti-dead variants (Track 2, H8, B4): retrain at k_win=100 (current
  ckpts are at k_win=500).
- TFA: retrain at L0 ≈ 100 per token (current `tfa_big` is at
  k=100 already on novel head — verify and reuse if compatible).
- T-SAE port (`tsae_paper`): retrain at k=100 to match. This is a
  *departure from Ye et al.'s native k=20*; we additionally report
  `tsae_paper_k20` as a paper-faithful reference baseline.

This is a substantial retraining commitment (~10-12 archs × 3 seeds)
but produces a strictly fairer leaderboard.

#### (iii) Probing protocol: long-tail sliding mean-pool, S=128

Single S-parameterized aggregation:

- **TXC**: slide T-window with stride 1 across the S=128-token tail
  (= the full Gemma-2 context). Drop the first T−1 windows so every
  averaging unit covers tokens entirely within the tail. Mean the
  remaining (S − 2T + 2) per-window z's into a single (d_sae,) vector.
- **per-token SAE**: encode each of the 128 tail tokens. Drop the
  first T−1 (to keep coverage aligned with TXC). Mean the remaining
  per-token z's.
- **MLC**: same as per-token SAE with the multi-layer extraction.

Probe: SAEBench-style top-`k_feat`-by-class-sep + L1 LR (existing
protocol, unchanged). Headline `k_feat = 5`.

This subsumes the previous lp / mp / full_window distinctions —
all three are special cases of S-parameterized sliding mean-pool. We
DO NOT report a separate "last T tokens" leaderboard: comparing
window archs with different T's at S=T is structurally confusing
(scope differs per arch).

#### (iv) Architecture list (~10-12 archs)

Canonical set for the headline leaderboard:

| arch | family | purpose | retrain at k_win=100? |
|---|---|---|---|
| `topk_sae` | per-token TopK | baseline | yes |
| `tsae_paper` (k=100) | per-token Matryoshka BatchTopK + InfoNCE | paper-faithful T-SAE port | yes |
| `tsae_paper_k20` | same, native k=20 | paper-faithful reference | yes |
| `mlc` | per-token TopK over 5 layers | layer baseline | yes |
| `mlc_contrastive_alpha100_batchtopk` | MLC + matryoshka + InfoNCE + BatchTopK | Phase 5 lp leader | yes |
| `agentic_mlc_08` | MLC + multi-scale InfoNCE | Phase 5 multi-scale MLC | yes |
| `txcdr_t5` | vanilla TXC T=5 | TXC baseline | yes |
| `agentic_txc_02` | TXC + multi-scale matryoshka | Phase 5 multi-scale TXC | yes |
| `phase57_partB_h8_bare_multidistance` | TXC + matryoshka + multi-distance + anti-dead | Phase 5 mp 3-seed champion | yes |
| `phase57_partB_h8_bare_multidistance_t6` | H8 at T=6 | Phase 5 mp peak | yes |
| `phase5b_subseq_track2` (B2) | subseq sampling on Track 2 | Phase 5B Track 2 winner | yes |
| `phase5b_subseq_h8` (B4) | subseq sampling on H8 stack | Phase 5B mp champion | yes |
| `tfa_big` | TFA full | predictive-coding reference | yes (verify k matches) |

Three seeds each → ~36 trainings.

### What's dropped from the leaderboard

These archs were tested in earlier phases but won't appear in the
Phase 7 headline (they go to appendix or are dropped entirely):

- D1 strided window (Phase 5B negative) — dropped, mention in negative
  results section.
- C1/C2/C3 token-level encoders (Phase 5B negative) — dropped,
  mention in negative results.
- F SubsetEncoderTXC (Phase 5B negative) — dropped, mention in
  negative results.
- Most BatchTopK paired variants from Phase 5.7 — only the lp leader
  (`mlc_contrastive_alpha100_batchtopk`) survives.
- Stacked SAE family — dropped (Phase 5 showed weakest results, not
  paper-relevant).
- Most TXCDR weight-sharing ablations (`txcdr_shared_*`, `tied`,
  `pos`, `causal`) — dropped, mention in §weight-sharing-ablation
  appendix.
- T-sweep cells beyond the canonical T values — moved to appendix.

### Branch merge plan

Order of operations:

1. **Merge `han-phase5b` → `han`** (clean, 0 syntactic conflicts).
   Phase 5 agent already integrated B2/B4 numbers; this merge brings
   in the bug-fix code, the C/D/F arch classes (for the negative-
   results section), the message doc, and the HF upload script.
2. **Merge `han-phase6` → merged-han** (0 syntactic conflicts but
   substantial semantic overlap in `src/architectures/` and
   `experiments/phase6_*`). Carefully inspect overlap, especially in
   anything Phase 5 also touched.
3. **Branch `han-phase7-unification`** off the merged `han`. Phase 7
   work happens here.

### 10-day timeline (target NeurIPS 2026-05-05)

| day | task |
|---|---|
| 1 | Branch merges (5b → han, then phase6 → han). Smoke-test imports, run Phase 5 reference trainings to confirm regressions absent. |
| 2 | Build Gemma-2-2b base activation cache (5 layers, 24k seqs, 128 tokens, fp16). Build new probe cache with S=128 tail. |
| 3 | Smoke-train each Phase 7 arch at the new convention (200-step smoke per arch) on the new cache. Verify shapes, k_win, no OOMs. |
| 4 | Train ~10-12 archs at seed=42, k_win=100. ~10 hr compute. |
| 5 | Train seeds 1, 2 of all winners. ~10 hr compute. |
| 6 | Probing pass (long-tail mp at S=128 + ablation S=20 + k_feat={1,2,5,20}). |
| 7 | Re-run Phase 6 qualitative pipeline on new ckpts. |
| 8 | Generate unified figures (headline bar, T-sweep, Pareto if applicable, autointerp panel). |
| 9 | Draft paper sections. |
| 10 | Polish + buffer. |

### Risks

1. **B2/B4 may not survive at k_win=100.** They were trained at k_win=500
   (5× our new convention). The "subseq sampling helps" finding may
   shrink or disappear at sparser k. Phase 7 explicitly retests this.
2. **TXC at large T with k_pos = k_win/T = small** (e.g., k_pos=5 at
   T=20) may be too sparse to train well per slab. Empirical question.
3. **Storage**: probe cache at S=128 fp16 is ~120-200 GB across 36
   tasks. Need to confirm local disk has room.
4. **Gemma2B base may give different rankings.** IT and base differ
   in instruction-following capacity; the residual-stream features
   may differ. We honest-report; if rankings flip, that's a finding,
   not a bug.

### Out of scope

- New architectures (anything beyond the candidate set above).
- Re-doing Phase 6 Pareto with new metrics — keep the existing Pareto
  framing but with new numbers.
- Long-context training beyond 128 tokens.
- T_max=128 t_sample=low ("infinite-T low-t") — deferred indefinitely.
- Phase 6.2 negative results — already documented; no re-run.
