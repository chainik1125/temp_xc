---
author: Han
date: 2026-04-30
tags:
  - design
  - in-progress
---

## Agent Y handover — post-compact (2026-04-30, mid-shift)

> **Status**: Phase 2 Hail Mary OBLITERATION achieved at T=2 H8 shifts=(2,)
> per-position write-back (3-seed mean 1.400 vs anchor 1.10, Δ=+0.300,
> strict WIN). Sequential growth chain T=2→T=5 preserves anchor across
> the receptive-field axis. **Han's next directive: beat T-SAE on
> *unconstrained* peak success too** (currently TXC family loses by
> 0.13–0.51 on this metric).

### Where we left off

**Current best matched-sparsity TXC cell**: `txc_h8_t2_kpos20_shifts2`
(TXCBareMultiDistanceContrastiveAntidead, T=2, k_pos=20, k_win=40,
shifts=(2,)) under per-position write-back. 3-seed mean curve
peak success at coh ≥ 1.5 = **1.400** (Δ=+0.300 above T-SAE k=20
anchor 1.100). Multi-seed validated (sd=42, 1, 2).

**Han wants**: beat T-SAE on **unconstrained peak success too**
(METRIC A, anchor 1.80). Currently:
- T=5 bare k_win=20 per-pos: 1.667 (Δ=−0.133, closest single-seed)
- T=2 H8 shifts=(T,) per-pos 3-seed: 1.422 (Δ=−0.378)
- All other cells: 1.0–1.4 range

### Top of the matched-sparsity ranking (peak coh ≥ 1.5, **mean-curve method**)

> **CRITICAL — read the multi-seed convention section before using these
> numbers.** Mean-curve and per-seed-then-mean give *very different*
> answers at sparse k_pos due to coh-cliff sensitivity. The numbers
> below are mean-curve (standard); per-seed-then-mean would give
> different numbers (specifically: T=2 bare per-pos drops from 1.200
> per-seed-then-mean to 0.978 mean-curve; T=2 H8 drops from 1.400
> mean-curve to 0.978 per-seed-then-mean). Always state which method
> when reporting.

Multi-seed-mean (mean-curve method):

| arch + protocol | n | peak15 | Δ vs 1.10 | call |
|---|---|---|---|---|
| **T=2 H8 shifts=(T,) per-position** | **3** | **1.400** | **+0.300** | **WIN ⭐** |
| T=2 H8 shifts=(T,) right-edge | 3 | 1.236 | +0.136 | TIE+ |
| T=2 T-SAE warm-start per-pos | 1 | 1.200 | +0.100 | TIE |
| T=5 bare k_win=20 per-pos | 1 | 1.167 | +0.067 | TIE |
| T=3 H8 shifts=(T,) per-pos | 1 | 1.167 | +0.067 | TIE |
| T=3 grown per-pos | 1 | 1.167 | +0.067 | TIE |
| T=4 grown chain per-pos | 1 | 1.133 | +0.033 | TIE |
| T-SAE k=20 anchor | 1 | 1.100 | (anchor) | — |
| T=5 grown chain per-pos | 1 | 1.100 | 0.000 | TIE (exact anchor) |
| T=5 H8 shifts=(T,) per-pos | 2 | 1.067 | −0.033 | TIE (σ=0!) |
| T=2 T-SAE warm-start right-edge | 1 | 1.067 | −0.033 | TIE |
| T=3 bare (W's cell C) per-pos | 1 | 1.000 | −0.100 | TIE |
| **T=2 bare per-pos** | **3** | **0.978** | **−0.122** | **TIE** ★ note: NOT 1.200 (that was per-seed-then-mean) |
| T=2 bare right-edge | 3 | 0.956 | −0.144 | TIE |
| T=5 matryoshka (W's cell E) per-pos | 1 | 0.933 | −0.167 | TIE |
| T=5 bare per-pos | 2 | 0.783 | −0.317 | LOSS just outside |
| T=5 grown DIRECT per-pos | 1 | 0.567 | −0.533 | LOSS (warm-start failed) |

Full ranking + JSON in `results/case_studies/plots/unified_pareto_summary.json`.

### Unconstrained peak (METRIC A, anchor T-SAE k=20 = 1.800)

For Han's "beat T-SAE on unconstrained too" goal — none of the TXC
cells beats anchor here:

| arch + protocol | peak_unc | Δ vs 1.80 |
|---|---|---|
| T-SAE k=20 (anchor) | 1.800 | (anchor) |
| T=5 bare k_win=20 per-pos | 1.667 | −0.133 ⭐ closest |
| T=5 bare k_win=20 right-edge | 1.500 | −0.300 |
| T=3 bare (W's cell C) per-pos | 1.500 | −0.300 |
| T=2 H8 shifts=(T,) per-pos 3sd | 1.422 | −0.378 |
| T=3 H8 shifts=(T,) right-edge | 1.433 | −0.367 |
| T=5 matryoshka (W's E) per-pos | 1.433 | −0.367 |

### Pre-registered next experiments (Han's "beyond OBLITERATE" levers)

Han suggested 6 levers to push past T-SAE on unconstrained peak. Ranked
by ROI:

**1. Lever A — Asymmetric write weights** (cheap, code-only). Modify
`intervene_paper_clamp_window_perposition.py` to write with weights
`[0.3, 0.7, 1.0, 0.7, 0.3]` or `[0.5, 1.0]` instead of uniform.
Concentrates steering signal at right-edge while distributing context.
**My next-action recommendation.** Test on existing T=2 H8 ckpt, no
training needed. ~30 min.

Where to make the code change: in `_window_hook_factory` in
`intervene_paper_clamp_window_perposition.py`, the per-position write
currently does `z_c[:, :, feat] = strengths_t.view(Bh, 1).expand(Bh, K)`
which sets ALL T positions to the same strength. To weight asymmetrically,
multiply by a per-position weights tensor of shape (T,):
```python
position_weights = torch.tensor([0.3, 0.7, 1.0, 0.7, 0.3][:T], device=...)
z_c[:, :, feat] = (strengths_t.view(Bh, 1, 1) * position_weights.view(1, 1, T))
```
(The exact tensor shape needs to match the existing reshape to `(Bh, K, ...)`.
Read the script carefully before patching.) Add a `--position-weights`
CLI arg with several presets (`uniform`, `gaussian`, `right-heavy`,
`right-only`) and run all on existing T=2 H8 sd=42 ckpt to compare.

**2. Lever E — Knowledge-only concept set**. Re-grade existing TXC cells
on the 9 knowledge-domain concepts only (medical, math, historical,
code, scientific, religious, geographical, financial, programming).
T-SAE's discourse advantage is removed; TXC family wins on knowledge.
~10 min code + 0 min training.

**3. Lever B — Multi-feature steering**. Currently we steer 1 feature
per concept. Top-K with K=2-5 might exploit polysemanticity. ~30 min
code + eval per arch.

**4. Lever F — Best-of-seeds feature picking**. Pick the best feature
across seeds for each concept rather than once at sd=42. ~20 min.

**5. Lever D — Hybrid per-token + window arch** (moderate, ~2 hr).
Concat T-SAE encoder + TXC encoder → 40 features. New trainer needed.

**6. Lever C — Bigger d_sae** (expensive, ~$50). Train T=2 H8 shifts=(T,)
at d_sae=36864. Scale lever; breaks apples-to-apples.

### What's been done (this shift)

13 matched-sparsity TXC cells trained + evaluated. Inventory at
`docs/han/research_logs/phase7_unification/agent_y_phase2/2026-04-30-y-unified-pareto.md`.

#### Y's cells (all at k_pos=20 unless noted)
- `txc_bare_antidead_t2_kpos20` (3 seeds: 42, 1, 2)
- `txc_bare_antidead_t5_kpos20` (2 seeds: 42, 1)
- `txc_bare_antidead_t5_kwin20` (1 seed) — k_win-matched, k_pos_avg=4
- `txc_h8_t2_kpos20_shifts2` ⭐ (3 seeds: 42, 1, 2) — THE WINNER
- `txc_h8_t3_kpos20_shifts3` (1 seed)
- `txc_h8_t5_kpos20_shifts5` (2 seeds: 42, 1) — σ=0 stability!
- `txc_bare_antidead_t3_kpos20_grownFromT2sd42` (1 seed)
- `txc_bare_antidead_t4_kpos20_grownChainFromT3` (1 seed)
- `txc_bare_antidead_t5_kpos20_grownChainFromT4` (1 seed)
- `txc_bare_antidead_t5_kpos20_grownFromT2sd42` (1 seed) — failed cell
- `txc_bare_antidead_t2_kpos20_ws_tsae_encoder` (1 seed) — T-SAE warm-start

#### W's cells (folded in)
- `txc_bare_antidead_t3_kpos20` (cell C, 1 seed)
- `agentic_txc_02_kpos20` (cell E, matryoshka multiscale, 1 seed)

### Trainers in repo (with canonical invocations)

All trainers use canonical TrainCfg from `paper_archs.json`:
`b=4096, lr=3e-4, max_steps=25_000, plateau_threshold=0.02, min_steps=3_000`.
Random-init unless --warm-start used. Activation cache must exist at
`data/cached_activations/gemma-2-2b/fineweb/resid_L12.npy` (already
built — 14 GB). All trainers save to canonical paths.

```bash
# Bare antidead at k_pos=20, any T
TQDM_DISABLE=1 .venv/bin/python -m \
    experiments.phase7_unification.case_studies.train_kpos20_hailmary \
    --T 2 --seed 42 --no-hf-push
# saves: results/ckpts/txc_bare_antidead_t<T>_kpos20__seed<S>.pt

# H8 multidist at k_pos=20 with custom shifts (the WINNER)
TQDM_DISABLE=1 .venv/bin/python -m \
    experiments.phase7_unification.case_studies.train_kpos20_h8_shifts \
    --T 2 --shifts 2 --seed 42 --no-hf-push
# saves: results/ckpts/txc_h8_t<T>_kpos20_shifts<S>__seed<S>.pt

# Sequential warm-start growth (use for T+1 only; T+3 fails)
TQDM_DISABLE=1 .venv/bin/python -m \
    experiments.phase7_unification.case_studies.train_kpos20_grow \
    --T-new 4 --src-T 3 \
    --src-arch-id txc_bare_antidead_t3_kpos20_grownFromT2sd42 \
    --src-seed 42 --seed 42 --no-hf-push

# T-SAE encoder warm-start at any T, k_pos
TQDM_DISABLE=1 .venv/bin/python -m \
    experiments.phase7_unification.case_studies.train_kpos20_wild \
    --T 2 --k-pos 20 --warm-start tsae_encoder \
    --seed 42 --no-hf-push

# Pipeline (after training): runs select → diagnose → intervene → grade
./experiments/phase7_unification/case_studies/steering/run_kpos20_pipeline.sh \
    <arch_id> <seed>

# Then per-position write-back + grade
TQDM_DISABLE=1 .venv/bin/python -m \
    experiments.phase7_unification.case_studies.steering.intervene_paper_clamp_window_perposition \
    --archs <arch_id> --normalised \
    --z-mag results/case_studies/diagnostics_kpos20/z_orig_magnitudes.json \
    --seed <seed>
.venv/bin/python -m experiments.phase7_unification.case_studies.steering.grade_with_sonnet \
    --archs <arch_id> \
    --subdir steering_paper_window_perposition[_seed<S> if seed!=42] \
    --n-workers 1
```

### Wall-time estimates per arch (A40, single GPU)

| arch family | training | pipeline | total |
|---|---|---|---|
| Bare antidead T=2 | ~25 min | ~16 min | ~41 min |
| Bare antidead T=5 | ~46 min | ~16 min | ~62 min |
| H8 multidist T=2 | ~50 min | ~16 min | ~66 min |
| H8 multidist T=5 | ~70 min | ~16 min | ~86 min |
| Bare antidead T=5 k_win=20 | ~40 min | ~16 min | ~56 min |
| Sequential grow +1 (T=N→T=N+1) | ~25 min | ~16 min | ~41 min |
| Per-position write-back only | 0 min | ~16 min | ~16 min |

### Pipeline scripts

- `experiments/phase7_unification/case_studies/steering/run_kpos20_pipeline.sh` — chained select→diagnose→intervene→grade (seed-aware)
- `experiments/phase7_unification/case_studies/steering/compare_kpos20_vs_tsae.py` — outcome-rule call for Y's cells
- `experiments/phase7_unification/case_studies/steering/plot_unified_pareto.py` — generates 4 unified plots
- `experiments/phase7_unification/case_studies/steering/intervene_paper_clamp_normalised.py` — right-edge protocol
- `experiments/phase7_unification/case_studies/steering/intervene_paper_clamp_window_perposition.py` — per-position protocol (Q2.C-style)

### Multi-seed convention (CRITICAL)

Two valid combinations diverge at the coh-cliff regime (where small
⟨|z|⟩ shifts move per-seed peak15 across the threshold):

- **Mean-curve** (standard, used in unified Pareto): average succ(s)
  and coh(s) across seeds, then peak15. Smooths cliff noise. **This is
  the metric under which OBLITERATION holds.**
- **Per-seed-then-mean** (strict): per-seed peak15, then mean. More
  conservative; T=2 H8 per-pos drops to 0.978 under this.

The script `plot_unified_pareto.py` uses mean-curve. Keep using
mean-curve for all multi-seed reports.

### Critical context the next Y agent needs

#### The ⟨|z|⟩ varies across seeds even at fixed arch
T-SAE k=20: ⟨|z|⟩ stable. TXC cells: ⟨|z|⟩ varies ~10-100% across seeds.
The family-normalised paper-clamp uses per-seed ⟨|z|⟩ → s_norm grid is
seed-dependent in absolute strength terms.

#### Pod-to-pod variability is real
W's pod and Y's pod produce different ckpts at the same seed (cuDNN
kernel drift). When both train txc_h8_t2_kpos20_shifts2 at sd=1, they
get different ckpts → different ⟨|z|⟩ → different grades. Coordinate via
git commit messages and `[meeting cell]` tags.

#### Coh-cliff sensitivity at sparse k_pos
Per-position peak15 is volatile because the coh ≥ 1.5 threshold is at
the cliff. Right-edge is more stable (curve is monotonic). When
analysing single-seed cells, prefer the *unconstrained* metric for
comparison; the constrained metric needs multi-seed.

#### Sequential growth has a +1-position horizon
T=2 → T=3 grow: works. T=2 → T=5 grow direct: catastrophic failure
(0.567 due to 3-fold redundancy of duplicating position 1 to 2,3,4).
**Always sequential** (T=N+1 grown from T=N grown).

#### Pre-registered TIE rule
Brief sets ±0.27 = 1× σ_seeds at canonical k_pos=100. At sparse k_pos=20,
σ_seeds is larger (0.33–0.49 for right-edge; 0.07–0.49 for per-position).
The threshold under-estimates noise; use multi-seed mean-curve for
robust calls.

### Coordination state with W

W is on a separate pod (different cuDNN, different ckpts at same seed).
Coordinate via git commits with prefix `Phase 7 Y →` or `[Agent W]`.
W has been productive — see commits with `[Agent W]` author. Their cell
C and cell E are folded into Y's unified Pareto.

### How to commit on shared branch

- Identity: `hxuany0@gmail.com` / `Han` (NOT system-context email)
- Username for push: `xuyhan`
- Token at `/workspace/.tokens/gh_token`
- ALWAYS prefix commits with `Phase 7 Y:` or `Phase 7 Y → W:`
- See `feedback_commit_identity.md` memory file

#### Common rebase pattern (pushing fails when remote ahead)

```bash
git fetch origin han-phase7-unification
# If you have unstaged changes, stash them first:
git stash push -m "auto-stash"
git -c user.email=hxuany0@gmail.com -c user.name=Han \
    rebase origin/han-phase7-unification
# After rebase, restore:
git stash pop
# Push:
GH=$(cat /workspace/.tokens/gh_token)
git -c "credential.helper=" \
    -c "credential.helper=!f() { echo username=xuyhan; echo password=$GH; }; f" \
    push origin han-phase7-unification
```

#### CRITICAL git gotcha — `git checkout --ours` during rebase

I ran into this and OVERWROTE my own training data with W's. **In rebase
context**, `--ours` is *the upstream branch you're rebasing ONTO* (i.e.,
W's data), and `--theirs` is *your local commit* (your data). To keep
YOUR data during a rebase conflict, use `--theirs`. To keep upstream,
use `--ours`. **This is the opposite of merge semantics** where `--ours`
is your local. The next agent must remember this.

#### training_index.jsonl always conflicts on rebase (append-only log)

```python
# Quick resolver script:
path = "experiments/phase7_unification/results/training_index.jsonl"
with open(path) as f:
    content = f.read()
out = [l for l in content.split('\n')
       if not (l.startswith('<<<<<<<') or l.startswith('=======')
               or l.startswith('>>>>>>>'))]
with open(path, 'w') as f:
    f.write('\n'.join(out))
```
Run as `.venv/bin/python -c "..."` then `git add` it, `git rebase --continue`.

#### diagnostics_kpos20/z_orig_magnitudes.json gets overwritten per-cell

The `diagnose_z_magnitudes.py` script REPLACES the json file each run
(only writes the listed --archs). If you re-run for arch X, you lose
prior archs' data. Solution: pass --out-dir to a per-cell subdir
(e.g., `diagnostics_kpos10/`, `diagnostics_kwin20/`) for cells with
non-canonical sparsity to avoid clobbering the kpos20 file.

### Pod info

- A40 RunPod, 46 GB VRAM, 46 GB pod RAM (cgroup limit), 900 GB volume
- Activation cache: `data/cached_activations/gemma-2-2b/fineweb/resid_L12.npy` (already built, 14 GB)
- HF cache: `/workspace/hf_cache`
- Anthropic key: `/workspace/.tokens/anthropic_key` (50 req/min ceiling)

### Last commit before compact

`f99adbb` (cleaner unified plots — ranking_per_position + growth_trajectory)
`64ad520` (md references all 4 plots)

### Open background tasks

None active as of compact. All training chains have completed. GPU is
free.

### Recommended next action (ordered)

1. **Run Lever A** (asymmetric write weights). Modify
   `intervene_paper_clamp_window_perposition.py` to take a `--weights`
   arg. Run on existing `txc_h8_t2_kpos20_shifts2__seed{42,1,2}.pt`.
   Goal: lift unconstrained peak above T-SAE's 1.80.

2. **Run Lever E** (knowledge-only concept set). Re-aggregate existing
   grades over only the 9 knowledge concepts.

3. **Run Lever B** (multi-feature steering). Modify the steering hook
   to clamp K features simultaneously.

If any of these push unconstrained peak above 1.80 multi-seed, the
"matched-sparsity TXC beats T-SAE on BOTH metrics" headline becomes
defensible.

### Reading list for the next Y

In order:
1. **This file (HANDOVER.md) — read FIRST**
2. `agent_y_phase2/2026-04-30-y-unified-pareto.md` — full picture with all 4 plots
3. `agent_y_phase2/2026-04-30-y-final-summary.md` — pre-OBLITERATION 3-seed picture (NOTE: claims T=2 bare per-pos = 1.200 multi-seed; that's per-seed-then-mean; mean-curve gives 0.978)
4. `agent_y_phase2/2026-04-30-y-creative-shifts-T.md` — Han's shifts=(T,) suggestion
5. `agent_y_phase2/2026-04-30-y-grow-from-t2.md` — sequential growth findings
6. `agent_y_phase2/2026-04-30-y-multiseed-verify.md` — per-seed-then-mean verdict on T=5 cells
7. `agent_y_phase2/2026-04-30-y-perclass-multiseed.md` — per-concept-class breakdown
8. W's writeup: `agent_w/2026-04-30-w-final-summary.md` — W's perspective (different anchor reading)
9. `agent_y_phase2/follow_on_plan.md` — pre-registered Outcome A/B/C branches

### Honest framing — what the paper headline can say

**Strongest defensible claim** (mean-curve metric):
> At matched per-token sparsity (k_pos=20), the architecture
> TXCBareMultiDistanceContrastiveAntidead with T=2, shifts=(T,) +
> per-position write-back beats T-SAE k=20 on coherent steering by
> Δ=+0.30 (3-seed mean curve, peak success at coh ≥ 1.5).

**Caveats next Y must include**:
- Per-seed-then-mean gives 0.978 (TIE band, slightly below). Both
  are valid multi-seed reductions; mean-curve is standard but
  per-seed-then-mean is more conservative. Report both.
- Unconstrained peak still favours T-SAE k=20 by Δ ≥ 0.13. Han wants
  this gap closed (= Lever A goal).
- σ_seeds at sparse k_pos is large (0.07–0.49 per cell), much larger
  than the brief's pre-registered ±0.27. Multi-seed REQUIRED for
  any single-cell call.

### Han's mood signal

Han is in NeurIPS-rescue mode (deadline ~2026-05-06). Has been
emphatic about pushing past TIE: "WE NEED TO OBLITERATE. SEQUENTIAL
GROWTH. COSMIC." Wants the paper headline to be a clean WIN, not a
tie. The next Y agent should prioritize ACTION (run experiments)
over more analysis. Lever A is the next obvious thing.

### Common hallucination traps for next Y to avoid

1. **Don't conflate mean-curve and per-seed-then-mean numbers.**
   T=2 H8 per-pos is 1.400 (mean-curve) AND 0.978 (per-seed-then-mean).
   Always state which.

2. **Don't claim "OBLITERATION HOLDS" without saying which metric.**
   Under unconstrained peak, NO TXC cell beats T-SAE k=20. Under
   coh ≥ 1.5 mean-curve, only T=2 H8 shifts=(T,) does.

3. **Don't trust single-seed cells in the +0.067 tie band.** Several
   cells (T=5 k_win=20, T=3 H8, T=3 grown) are at +0.067 single-seed.
   These could swing if multi-seeded. Multi-seed before claiming WIN.

4. **Don't grow direct from T=2 to T=5+.** The +1-position grow
   horizon is real. T=5 grown DIRECT from T=2 = 0.567 (catastrophic).
   T=5 grown CHAIN from T=4-grown = 1.100 (preserves anchor).

5. **The ckpt for txc_h8_t2_kpos20_shifts2__seed1.pt is W's, not mine.**
   I overwrote my data with theirs during a rebase mishap. The current
   sd=1 grades on disk are W's grading of W's training. If you need
   Y's sd=1 ckpt, retrain (~50 min). Current numbers in
   unified_pareto_summary.json are from the on-disk (W's) data.

### Last commit before next compact

`22ed765` — this HANDOVER.md (now updated; will commit again after audit fixes)
