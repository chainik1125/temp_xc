---
title: Computational resources (NeurIPS checklist)
tags: [paper, neurips, compute]
date: 2026-05-07
status: estimate
---

# Computational resources

This document summarises the GPU-hours used for the paper. Numbers are
estimates derived from `leaderboard.jsonl` row counts × per-cell wall
times, plus rough audit of pre-`final`-branch exploration. Hardware
mix is heterogeneous; we report **H100-equivalent hours** where the
underlying GPU was different (rough conversion factor: H100 ≈ 1.0,
H200 ≈ 1.3, A40 ≈ 0.4, RTX 5090 ≈ 0.6, RTX PRO 6000 (Blackwell) ≈ 0.8).

## Hardware

| GPU | Count (across pods) | Persistence |
|---|---:|---|
| H100 80GB | up to 8 (single pod) | persistent + ephemeral |
| H200 141GB | 1 (reserve) | persistent |
| A40 48GB | up to 8 (single pod) | ephemeral |
| RTX PRO 6000 (Blackwell, 96GB) | up to 5 | ephemeral |
| RTX 5090 (Blackwell, 32GB) | up to 7 (single pod) + 1 local | ephemeral / local |

Multiple pods ran concurrently during the final sprint (5–7 May 2026).

## Paper-bound compute (final branch only)

The `purified/` framework on the `final` branch is the paper-canonical
codebase. All 7 components train + evaluate from a single, audited
pipeline. Per-cell timings are dominated by training (eval is fast on
toy data; eval = sklearn-probe / sentence-encode on real-LM).

| Component | Trained models | Eval rows | Mean training cost | Eval cost | Subtotal (H100-hr) |
|---|---:|---:|---|---|---:|
| C1 — Synthetic TopK sweep (toy Markov d=40) | 213 | 213 | ~5 s/cell on toy | ~1 s/cell | **~1** |
| C2 — Synthetic coupled (toy d=256, 18 setups) | 5 333 | 5 339 | ~1.5 min/cell on H100 | ~5 s/cell | **~135** |
| C3 — Sparse probing (Gemma-2-2B IT+BASE L13) | 66 | 527 | ~30–60 min/cell | sklearn-probe ~5 min × k_feat | **~95** |
| C4 — Qualitative latents (concat corpora) | 34 | 36 | cache-hit on C3 | ~10–20 min eval | **~10** |
| C5 — RLHF steering (Gemma-2-2B-IT L13) | 28 | 61 | ~45 min/cell | ~10 min steering eval | **~30** |
| C6 — Emergent misalignment (Qwen 14B/7B) | 22 | 30 | ~30–90 min/cell | ~30 min EM eval | **~50** |
| C7 — Backtracking (Llama-3.1-8B BASE L10) | 12 | 19 | ~20–30 min/cell | ~5 min detection | **~10** |
| **Total — paper-bound (final)** | **5 708** | **6 225** |  |  | **~330 H100-hr** |

(Counts sourced from `docs/paper/training_appendix.md`; compute conversion
applied where cells ran on 5090 / A40 / RTX PRO 6000 → H100-hour
equivalent.)

## Pre-paper exploration (other branches)

The paper distils a 6-week research project. The bulk of the compute
was spent on **failed hypotheses, hill-climbed architectures that
didn't make the cut, and benchmark protocol iteration** — see
`docs/han/EXPERIMENT_INDEX.md` on `origin/han-phase7-unification` for
the per-phase log. We list the major branches and their estimated
exploration cost. Sub-branches share their parent's numbers (no
double-counting).

```
main
└── han-phase7-unification          (parent of Phase 2–7 wasteland)
    ├── han-phase5b                 (T-scaling explore)
    ├── han-phase6                  (qualitative latents wasteland)
    ├── han-phase7-agent-c          (Phase 7 worker A)
    │   └── han-phase7-agent-c-seed1
    ├── dmitry / dmitry-synthetic / dmitry-rlhf / dmitry-c6-redteam /
    │   dmitry-backtracking / dmitry-phase8
    │                               (Dmitry's parallel investigation tracks)
    ├── em-nanda                    (Phase 7 EM precursor — Qwen-14B-finance)
    ├── aniket / aniket-ward-stage-b / aniket-runpod*
    │                               (Aniket's backtracking / Ward Stage B)
    ├── andre / andre-steering / andre_safety
    │                               (Andre's steering wasteland)
    ├── bill / bill-benchmarking-synthetic / bill-three-arch-bench /
    │   bill-han-txc-10k            (Bill's bench / 10k-token sweeps)
    └── 300k-tfa / extended-300k    (TFA scaling explorations)

final                               (paper-canonical, this branch)
└── final-aniket                    (Aniket's mirror)
```

| Phase / branch family | What was tested | Est. exploration compute | Notes |
|---|---|---:|---|
| Phase 2 (toy TopK sweeps) | early TXC variants on toy d=40 | **~5 H100-hr** | Cheap; cached cells imported into C1. |
| Phase 3 (coupled features) | C2 precursor on toy d=256 | **~30 H100-hr** | Scaled up to current C2 design. |
| Phase 4 (NLP comparison) | real-LM, baseline TXC vs SAE | **~80 H100-hr** | Built activation cache (~12 H100-hr alone). |
| Phase 5 (downstream utility) | full 36-task SAEBench | **~150 H100-hr** | Many archs × 36 tasks × multi-k. Reduced to 16 tasks for paper. |
| Phase 5b (T-scaling explore) | T ∈ {2, 5, 10, 20, 32} | **~40 H100-hr** | Informed § 15 per-arch literature-faithful T. |
| Phase 6 (qualitative latents) | T-SAE-style chunk extraction | **~50 H100-hr** | Pareto sweep on Top-256 cumulative SEMANTIC. |
| Phase 7 (unification) | 50+ hill-climbed TXC variants | **~150 H100-hr** | Most TXC variants DROPPED for paper. |
| Dmitry tracks (synthetic + EM + RLHF + backtracking) | independent investigations | **~120 H100-hr** | Dmitry's Bench 2 finding informed the C2 win regime; EM Qwen-14B + 7B tunings. |
| Aniket tracks (backtracking) | Ward Stage A + B; magnitude grid | **~30 H100-hr** | Stage B sweep ported to C7. |
| Andre tracks (steering) | RLHF steering on Gemma | **~25 H100-hr** | Largely superseded by C5 canonical sweep. |
| Bill tracks (benchmarking) | early three-arch synthetic + 10k-token | **~20 H100-hr** | Architecture comparison precursor. |
| Agent-driven exploration on `final` | HUNT/ZOOM phases for C2, ρ-sweep, dropped TXC variants, smoke runs that escaped the `smoke=True` flag (≈ 30 cells, filtered from appendix) | **~120 H100-hr** | Includes ~6 NEW C2 setups that **were NEGATIVE for TXC** and thus dropped from the paper headline (dewdrop, harbor, K, L, PHALANX, OBELISK). Their data is in the leaderboard for honest record (`results/leaderboard.jsonl`) but excluded from the cross-setup comparison plot per Han's "drop if not" directive. |
| **Total — exploration (all branches + dropped agent work)** |  | **~820 H100-hr** |  |

## Grand total

| Bucket | H100-hr |
|---|---:|
| **Paper-bound** (final branch, included in paper figures + appendix) | ~330 |
| **Exploration** (all other branches + dropped on-final negative results) | ~820 |
| **Grand total** | **~1 150 H100-hr** |

## Caveats and uncertainties

- **Eval-time compute** dominates in C3 (sklearn probe sweeps over 8
  k_feat values × 16 tasks × 3 seeds per trained model). We estimate
  ~5 min per probe-cell × 527 eval rows ≈ 44 H100-hr; the remaining
  ~50 H100-hr is training. Probing is largely CPU-bound — H100-hour is
  here a proxy for "GPU-pod hour" because the probes ran on
  H100-equipped pods.
- **Multi-pod overlap**: from 5–7 May 2026 we ran up to 6 pods
  concurrently (8× H100, 7× 5090, 5× RTX PRO 6000, 8× A40, two 1×
  H100). The agent work on `final` was the most compute-intensive
  parallel phase; the 8× H100 pod ran ~4 hours at full saturation
  ≈ 32 H100-hr just for the C2 synthetic-suite final-sprint.
- **Wasteland reproduction**: we did NOT re-train any of the wasteland
  ~50 TXC hill-climb variants on the `final` branch. Final-branch C3
  uses the 6 archs canonical-locked in `configs/locked_archs.yaml`
  with re-trained checkpoints under the post-§ 15 batch-size + window
  conventions. Wasteland-branch checkpoints exist on
  `han1823123123/temp-bench-models` but are NOT cited in the paper
  numbers.
- **Negative results we kept training-but-dropped**: 6 C2 synthetic
  setups (dewdrop, harbor, K, L, PHALANX, OBELISK) were trained
  end-to-end (~219 cells × 6 = ~1 300 cells) but their TXC vs SAE
  comparison was negative. Per Han's "drop if not" directive their
  data is in the leaderboard but not in the cross-setup paper
  headline. Counted under "exploration" above.
- **Storage/IO is not GPU compute** but worth noting: the project
  produced ~285 trained checkpoints + ~70 GB of activation caches +
  ~15 GB of judge transcripts. Storage was on 1 TB persistent volumes
  attached to the H100 pods; auto-mirrored to two private HF repos
  (`han1823123123/temp-bench-{models,data}`).

## Source of truth

- Trained-model + eval-row counts: `docs/paper/training_appendix.md`
  (auto-generated by `scripts/render_training_appendix.py`).
- Per-component checkpoint registry: `checkpoints/manifest.jsonl`.
- Cell-level eval log: `results/leaderboard.jsonl`.
- Wasteland phase logs: `origin/han-phase7-unification:docs/han/research_logs/`.
- Hardware specs: `docs/paper/hardware.md`.

If the NeurIPS checklist requires an exact GPU-hours number, the
estimates above can be tightened by replaying the leaderboard with
per-cell wall-time stamps (the runner records `ts` start + can be
extended to record `wall_seconds`); current cells lack that field.
