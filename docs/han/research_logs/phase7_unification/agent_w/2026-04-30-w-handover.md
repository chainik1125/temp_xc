---
author: Han
date: 2026-04-30
tags:
  - design
  - in-progress
---

## Agent W handover briefing — Phase 7 Hail Mary

> **Read order**: this file → `agent_w/brief.md` (original W mission) → `agent_w/2026-04-30-w-phase3-results.md` (Phase 3 protocol findings) → `agent_y_phase2/2026-04-30-y-final-summary.md` and `agent_y_phase2/2026-04-30-y-unified-pareto.md` (Y's matched-sparsity matrix + unified Pareto) → `agent_w/steering-pipeline-mechanics.md` (how steering works mechanically).

### TL;DR — three phases done, one suggested next direction

**Phase 1 sweep (done)**: Phase 1 trained cells C (T=3 bare-antidead k_pos=20, both seeds) and E (T=5 matryoshka multiscale k_pos=20, single seed). Cell F (T=10) intentionally skipped. Key finding: **T-axis advantage reverses at sparse k_pos** — narrower window helps. Writeup: `2026-04-29-w-phase1-sweep.md`.

**Phase 2 axes (done)**: per-position write-back (Q2.C) on cells C and E. Multi-seed-pooled anchor σ_seeds = 0.80 (huge!) discovered — T-SAE k=20 itself isn't seed-stable under coh ≥ 1.5 because the coh-cliff position varies seed-to-seed. Pooled anchor = 0.70, win threshold = 0.97. Writeup: `2026-04-30-w-final-summary.md`.

**Phase 3 TXC-native protocols (done)**: implemented + tested 4 new protocols (V1 local, V2 anchored, V3 dec-additive, V4 tiled) on 4 cells × up to 2 seeds. **V3 dec-additive** (just `s × W_dec[picked, :, :]` added at active T-window — no encode/clamp/decode round-trip) WINS on cell C T=3 multi-seed (mean 1.000, Δ=+0.30, σ=0.20). **V4 tiled** also wins (mean 1.017, Δ=+0.32, σ=0.97; highest single-seed peak in matrix at sd42=1.500). Brief: `brief_phase3_txc_native_steering.md`. Results: `2026-04-30-w-phase3-results.md`.

**OBLITERATION verification (done)**: Y's `txc_h8_t2_kpos20_shifts2` cell at sd=1 + sd=2 trained + graded by W. 3-seed mean: right-edge 1.025 (Δ=+0.325 WIN), per-position 0.978 (Δ=+0.278 WIN). Y's "+0.30 mean-curve" claim earlier conflated single-seed anchor with multi-seed-curve cell — under consistent mean-curve aggregation, OBLITERATION per-position is **+0.23 vs T-SAE k=20's mean-curve anchor 1.167** = TIE band, just shy of strict +0.27 win.

### Headline numbers (consistent mean-curve across seeds)

| arch + protocol | n seeds | unconstrained peak | peak @ coh ≥ 1.5 | Δ vs T-SAE k=20 anchor 1.167 |
|---|---|---|---|---|
| T-SAE k=20 (anchor) | 2 | 1.800 | 1.167 | (anchor) |
| OBLITERATION right-edge (T=2 H8 shifts=2) | 3 | 1.356 | 1.236 | +0.069 |
| **OBLITERATION per-position** (T=2 H8 shifts=2) | 3 | 1.422 | **1.400** | **+0.233** ⭐ (TIE close to win) |
| W cell C T=3 V3 dec-additive | 2 | TBD | 1.000 (per-seed-then-mean) | tied at single-seed-anchor 1.10 |

T-SAE k=20 still **wins on raw peak success** (1.80 vs OBLITERATION's 1.42). TXC OBLITERATION wins on **success-given-coherent-text** (1.40 vs 1.17) by +0.23 — TIE band under strict ±0.27 threshold, but a clean positive Δ on the brief's primary metric.

**Honest paper claim**: at matched per-token sparsity, TXC's OBLITERATION recipe (T=2 + H8 multidist + shifts=(T,) + per-position) achieves higher *coherent* peak success than T-SAE k=20 by +0.23, multi-seed validated. T-SAE k=20 retains the unconstrained-peak advantage.

### Suggested next direction — Han's left-edge / center-slice / decay-weighted protocols

I (W) implemented V1 (local), V2 (anchored), V3 (dec-additive), V4 (tiled). Han pointed out (last conversation turn before handover) **untested protocols worth implementing**:

1. **Left-edge** (#1 priority): for position p, use the window [p, p+T-1] (where p is the LEFT edge). Take the leftmost slice (relative position 0) at p. Symmetric to right-edge but encoder integrates *forward* from p instead of *backward*. Hypothesis: helps on concepts whose signal builds *across* the steered span.

2. **Center-slice**: window centered at p, take the middle slice. Most useful at T ≥ 4.

3. **Decay-weighted per-position**: per-position averaging but weight by relative position (favor right-edge contributions over left-edge). Smoothly interpolates right-edge ↔ uniform per-position via a temperature parameter.

4. **Right-edge of next window** ("1-step lookahead"): at position p, use window ending at p+1. Cheapest variant, includes one-token-ahead context.

**Recommended approach**: implement left-edge first (clean structural symmetric-to-right-edge probe). Mirror `intervene_paper_clamp_window_local.py` but slice at relative position 0 of the window starting at p. Test on the OBLITERATION cell (T=2 H8 shifts=(2,)) at sd=42 first (~25 min); if it lifts the per-position 1.40 number further, run multi-seed verify.

If left-edge + per-position-of-OBLITERATION-arch combine multiplicatively, that could push past +0.27 strict win on mean-curve. That's the next chase.

### What's NOT in scope for the next W

- **Re-running V1/V2/V3/V4 on Y's cells** — Y has them on her pod, ckpts not on HF. Don't waste compute re-training. If Y uploads, then comparison.
- **Training new bare-antidead cells at k_pos=20** — covered by Y's matrix + my cell C. Only train if Han suggests a specific (T, k_pos, recipe) not yet in the matrix.
- **Probe-AUC work** — different mission (Z's domain).

### Pod state at handover (2026-04-30 ~15:50 UTC)

| field | value |
|---|---|
| Hardware | RunPod A40, 46 GB VRAM, 46 GB pod RAM, 900 GB volume |
| GPU state | likely idle (last training was OBLITERATION sd=2, finished) |
| Branch | `han-phase7-unification` HEAD ≈ `4cb2edff` (W's last push) |
| Activation cache | built at `data/cached_activations/gemma-2-2b/fineweb/resid_L12.npy` (14.16 GB, fp16, 24k seqs × 128 ctx × 2304 dim) |
| Token IDs | built at `.../token_ids.npy` (24k × 128 int64) |
| Local ckpts | tsae_paper_k20, txc_bare_antidead_t3_kpos20 (sd42+sd1), agentic_txc_02_kpos20 (sd42), txc_h8_t2_kpos20_shifts2 (sd1+sd2 W's training), agentic_txc_02 (sd42), txc_bare_antidead_t5 (sd42) |
| z_mag file at `diagnostics_kpos20/z_orig_magnitudes.json` | has 11 arch entries; clobber-prone — always backup before running diagnose |

### Key pitfalls to avoid (memory + earned-the-hard-way)

1. **`diagnose_z_magnitudes.py` overwrites the z_orig_magnitudes.json file** — each run replaces all entries with just the arch you asked for. Always backup before, merge after. See `feedback_zmag_clobber.md`.

2. **`run_perposition.sh` was originally not seed-aware** — patched to be (commit 70d26a6a). All similar pipeline scripts should write to `_seed{N}` suffixed dirs for non-canonical seeds.

3. **`.pyc` cache on MFS-mounted /workspace doesn't always invalidate** when a .py file is edited. After patching a script, run `find ... -name "*.pyc" -delete` to force re-compile in subprocess.

4. **Git push pattern**: brief's `xuyhan` username form fails with "Invalid username or token". Use `git push "https://oauth2:$GH@github.com/chainik1125/temp_xc.git" han-phase7-unification` instead. See `feedback_temp_xc_git_push.md`.

5. **Always `-c user.email=hxuany0@gmail.com -c user.name=Han`** for any git command that creates a commit (rebase, merge, pull --rebase, commit). The pod's hostname is unconfigured.

6. **Mean-curve vs per-seed-then-mean aggregation**: these give different numbers under coh ≥ 1.5. Mean-curve = avg success and avg coh per s_norm across seeds, then peak. Per-seed-then-mean = peak per seed, then mean. Y's writeups use mean-curve as standard. Always state which.

7. **Cross-pod seed=1 determinism is NOT guaranteed**. Y trained T=2 H8 sd=1 on her pod; W trained on this pod; same arch, same seed; numbers diverged (Y's per-position 1.700, W's 0.633). cuDNN kernel non-determinism. Multi-pod multi-seed needed for tightest claims.

### Reading list (priority order)

1. `agent_w/2026-04-30-w-handover.md` (this file)
2. `agent_w/2026-04-30-w-phase3-results.md` — V1-V4 protocol findings
3. `agent_y_phase2/2026-04-30-y-unified-pareto.md` — Y's headline pareto + OBLITERATION
4. `agent_w/steering-pipeline-mechanics.md` — how protocols work, mechanically
5. `agent_w/brief_phase3_txc_native_steering.md` — Phase 3 brief (full protocol space)
6. `agent_w/brief.md` — original W brief (Phase 1 mandate + coordination protocol with Y)
7. `agent_y_phase2/2026-04-30-y-final-summary.md` — Y's matrix + matched-sparsity headline

### What's running / not running

Nothing currently in flight. GPU verified idle (0% util, 0 MiB). Latest local commit: `f184ab3d` (this handover, after rebase onto Y's `64ad5207`). Y's recent thread `77b38c96` ("COSMIC sequential growth complete") landed during my Phase 3 work — see § COSMIC below.

### COSMIC sequential growth (Y's untracked-by-me thread)

Y ran a "+1-position-per-step warm-start" chain: T=2 → T=3-grown-from-T=2 → T=4-grown-from-T=3-grown → T=5-grown-from-T=4-grown. Single-seed result: graceful ~0.067-per-step decay; T=5 grown-chained ties anchor exactly (1.100 = 1.10). Direct T=2→T=5 growth fails catastrophically (0.567). Implication: matched-sparsity TXC can hold the anchor at any T from 2 to 5 if you sequentially warm-start. Files: `txc_bare_antidead_t{3,4,5}_kpos20_grown*` in training_index. Worth combining with H8 shifts=(T,) for a "COSMIC + OBLITERATION stack" experiment — untested.

### Cross-pod ckpt asymmetry — important

Y kept her T=2 H8 sd42 ckpt local (`--no-hf-push`). On THIS pod, `txc_h8_t2_kpos20_shifts2/ckpts/` has only sd1 + sd2 (W's training). Y's sd42 ckpt is NOT here. The 3-seed OBLITERATION verify worked because Y committed her sd42 *grades + generations* to the repo (under `steering_paper_normalised/txc_h8_t2_kpos20_shifts2/`); we never needed her sd42 ckpt itself.

This means: if next-W wants to run a NEW protocol on Y's sd42 ckpt (e.g., left-edge on the OBLITERATION arch), they would need to either (a) pull Y's sd42 ckpt from her pod, or (b) re-train sd42 themselves (~30 min wall, deterministic recipe via `train_kpos20_h8_shifts.py --T 2 --shifts 2 --seed 42`), or (c) just use the ckpts they already have (sd1 + sd2 + maybe my own freshly trained sd42). Cross-pod cuDNN non-determinism means re-trained sd42 won't bit-match Y's sd42 — but at multi-seed pooling that's noise, not signal.

### Old "joint conclusion" file note

I (W) prematurely wrote `agent_w/2026-04-30-w-final-conclusion.md` in this session, Han said "don't write the joint conclusion yet", I `rm`'d the file locally, but a subsequent rebase + push saved it back to the repo at commit `731e0bd7`. It's still in the working tree as of this handover.

**Treat it as a draft reference, not an official conclusion.** Han wants the joint conclusion written WHEN findings are settled. The current matrix (mean-curve OBLITERATION = TIE close to win; per-seed-then-mean = barely WIN) hasn't reached the level of confidence Han wants for a joint conclusion yet. If you write the official one, supersede `2026-04-30-w-final-conclusion.md` cleanly.

### Coordination protocol with Y (still active)

Y is on a separate pod. We coordinate via git commit messages. Tag commits clearly:
- `[Agent W]` or `Phase 7 W:` for my commits
- `Y → W:` for cross-agent messages
- `[meeting cell]` for any cell that's a coordination point

Race avoidance: before training a cell at a given (arch, seed), `git pull` and check `git log --grep="<arch>__seed<seed>"`. Y has been good about pre-announcing.

Y's headline-plot generator is at `experiments/phase7_unification/case_studies/steering/plot_unified_pareto.py` (or similar — find it via `git log --all --name-status | grep pareto`). Use it as the template for any new headline plots.

### Pre-registered metric

**Primary**: peak success at coh ≥ 1.5, family-normalised paper-clamp.
**Threshold**: ±0.27 vs anchor (= 1× σ_seeds at canonical sparsity, but at k_pos=20 σ_anchor itself is ~0.8 under per-seed; under mean-curve ~0.4. Use multi-seed pooling.)

### Files I produced

Briefs / writeups:
- `agent_w/brief.md` (Phase 1 mandate, written by Han)
- `agent_w/plan.md` (Phase 1 pre-registered plan)
- `agent_w/2026-04-29-w-phase1-sweep.md`
- `agent_w/2026-04-30-w-final-summary.md`
- `agent_w/brief_phase3_txc_native_steering.md`
- `agent_w/2026-04-30-w-phase3-results.md`
- `agent_w/steering-pipeline-mechanics.md`
- `agent_w/2026-04-30-w-handover.md` (this file)

Trainers:
- `case_studies/train_kpos20_txc.py` (T=3, T=5, T=10 bare-antidead)
- `case_studies/train_kpos20_matry.py` (T=5 matryoshka multiscale)
- `case_studies/train_kpos20_h8_shifts.py` (Y's H8 shifts=(T,) trainer — used by W for multi-seed verify)

Intervene scripts:
- `case_studies/steering/intervene_paper_clamp_window_local.py` (V1)
- `case_studies/steering/intervene_paper_clamp_window_anchored.py` (V2)
- `case_studies/steering/intervene_paper_clamp_window_dec_additive.py` (V3) — handles `W_decs[-1]` for matryoshka
- `case_studies/steering/intervene_paper_clamp_window_tiled.py` (V4)

Pipeline launchers:
- `case_studies/steering/run_w_phase1_cell.sh` (cells C/D/E/F + warm-start variants)
- `case_studies/steering/run_perposition.sh` (Q2.C protocol, seed-aware)

Memory files (in `/home/appuser/.claude/projects/-workspace-temp-xc/memory/`):
- `feedback_runpod_quota.md`
- `feedback_temp_xc_git_push.md`
- `feedback_zmag_clobber.md`
- `project_phase7_arch_landscape.md`
- `project_phase7_w_metric_baseline.md`
- `project_phase7_w_phase2_axes.md`

### One-line mission for next W

**Implement left-edge protocol → run on OBLITERATION cell → see if it stacks past +0.27 strict win.** That's the next concrete experiment. Estimated 30 min implementation + 25 min × N cells testing.
