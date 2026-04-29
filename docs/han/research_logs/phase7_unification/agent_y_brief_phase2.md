---
author: Han
date: 2026-04-29
tags:
  - design
  - in-progress
---

## Agent Y handover brief — Hail Mary hill-climb from T-SAE k=20

> Continuation of the Y workstream. Read `brief.md` first for paper-wide
> context (subject models, k_win convention, methodology, source-of-truth
> files), then `agent_y_brief.md` for the original Y mandate, then
> `agent_y_paper/README.md` for what shipped from the previous Y shift.
> This brief is the *next* Y instance's mission.

### TL;DR

- **Phase 1 of Y is done.** Q1.1/Q1.2/Q1.3, Q2.C, multi-seed verification,
  per-concept-class breakdown, Track A ln1-pivot null all shipped under
  `agent_y_paper/`. Headline findings: matched-sparsity TIE between TXC
  matryoshka and T-SAE k=500 (1.37 vs 1.38); per-concept-class
  structural pattern (TXC wins on knowledge-domain).
- **Phase 2 (this brief) is a Hail Mary.** Han's directive: take T-SAE
  k=20 (the per-token paper-clamp winner under our protocol) and
  hill-climb toward TXC. Test whether a sparsity-matched TXC beats the
  T-SAE k=20 baseline on the Gemma-2-2b steering case study. **The
  single most important missing experiment is TXC k_pos=20, T=5** —
  flagged in Y's z-handoff but never run.
- **You're on A40, 46 GB VRAM, 46 GB pod RAM, branch
  `han-phase7-unification`.** Same setup as previous Y. Push as
  `xuyhan` with token at `/workspace/.tokens/gh_token`. Commit identity
  `hxuany0@gmail.com` (NOT the system-context email). HF token at
  `/workspace/.tokens/hf_token`.
- **Coordinated with Agent W on the same case study, opposite axis.**
  Y owns the *T axis at fixed k_pos=20*. W owns the *k_pos axis at
  fixed T=best-from-sweep*. The two ladders meet at (T=5, k_pos=20),
  which serves as a within-paper internal validation cell. See
  `agent_w/brief.md`.

### State at handover

What previous Y shipped (read in this order if you want to reconstruct):

1. `agent_y_paper/2026-04-29-y-summary.md` — 1-page shift summary, the
   place to start.
2. `agent_y_paper/2026-04-29-y-tx-steering-magnitude.md` — Q1.1/Q1.2/Q1.3
   (z-magnitude diagnostics, peak-strength scaling, family-normalised
   paper-clamp). The methodology backbone.
3. `agent_y_paper/2026-04-29-y-tx-steering-final.md` — sparsity
   decomposition + matched-sparsity TIE finding.
4. `agent_y_paper/2026-04-29-y-cs-synthesis.md` — Track B brainstorm +
   per-concept-class structural pattern (TXC wins on knowledge-domain).
5. `agent_y_paper/2026-04-29-y-ln1-pivot.md` — Track A pivot null
   (don't repeat).
6. `agent_y_paper/2026-04-29-y-z-handoff.md` — the sparser-TXC
   question; this brief is the follow-through.

### The mandate

**Take T-SAE k=20 as the anchor; hill-climb toward TXC at fixed k_pos=20
along the receptive-field axis.** Five cells, single-axis steps:

| step | arch | k_pos | T | encoder | decoder | notes |
|---|---|---|---|---|---|---|
| 0 | T-SAE k=20 (anchor) | 20 | 1 | per-token | per-token | Already trained. **Re-run grading on the *exact* T-SAE paper concept set first** (paper App. B, not our 30-concept adaptation) to confirm pipeline faithfulness. |
| 1 | TXCBareAntidead | 20 | 5 | T=5 window | right-edge | The canary. Sparsity-matched TXC. |
| 2 | TXCBareAntidead | 20 | 5 | T=5 window | per-position | Adds per-position write-back. |
| 3 | MatryoshkaTXCDR | 20 | 5 | T=5 window | per-position | Adds matryoshka contrastive structure. |
| 4 | MatryoshkaTXCDR + multidist | 20 | 5 | T=5 window | per-position | Adds multi-distance contrastive heads (= H8 family). |

**Stage gate after Step 1.** Three outcomes:

- Step 1 *beats* T-SAE k=20 → headline finding ("sparsity-matched TXC
  dominates"). Skip Steps 2–4; go straight to multi-seed verification +
  paper writeup.
- Step 1 *ties* T-SAE k=20 → narrative collapses to "sparsity is the
  only thing"; hill-climbing further is unlikely to help. Document and
  hand back to Han.
- Step 1 *loses* to T-SAE k=20 → there's an architectural anti-prior
  at this sparsity. Steps 2–4 become the failure-mode investigation.

### Metric

Best success at coherence ≥ 1.5, on the family-normalised paper-clamp
protocol (Q1.3). Single defensible number per arch. Coherence threshold
is concrete: "the steered output is still readable", not "topic-drifted
gibberish". This metric is what the paper will report.

Secondary numbers to track:
- Peak success across the s_norm sweep (the headline number under
  Q1.3)
- Per-concept-class breakdown (knowledge / discourse / safety /
  stylistic / sentiment) — does the win pattern shift with sparsity?

### Warm-start trick (worth ~5-10× training time)

Initialise the TXC encoder/decoder weights from the T-SAE k=20
checkpoint, broadcast across the T positions. Only learn what *changes*
when you add the time axis. Concretely: tile the per-token encoder
across T=5 positions, set decoder per-position to T-SAE's decoder
divided by T (so right-edge sum reproduces T-SAE's per-token output at
init). Then fine-tune. Convergence is dramatically faster than from
random init.

This makes Step 1 a true minimum-deviation experiment from T-SAE k=20
— literally "the same SAE, but the encoder window is wider".

### Existing pipeline you reuse (do NOT rebuild)

- **Training template**: `experiments/phase7_unification/case_studies/train_ln1_txc.py`
  is the closest single-arch self-contained Gemma-2-2b TXC trainer.
  Copy + modify for k_pos=20 + warm-start.
- **Feature selection**: `experiments/phase7_unification/case_studies/steering/select_features.py`
  (already hook_name aware).
- **Z magnitude diagnostics**: `experiments/phase7_unification/case_studies/steering/diagnose_z_magnitudes.py`
  (Q1.1; provides s_abs scaling per arch).
- **Steering**: `experiments/phase7_unification/case_studies/steering/intervene_paper_clamp_normalised.py`
  (Q1.3 family-normalised paper-clamp).
- **Grading**: `experiments/phase7_unification/case_studies/steering/grade_with_sonnet.py`
  with `--n-workers 1` and `--fix-errors` resume mode (Anthropic 50
  req/min limit; W shares this limit — coordinate).
- **Comparison plotting**: `experiments/phase7_unification/case_studies/steering/compare_ln1_vs_resid.py`
  is a template for arch-vs-arch comparison; copy + modify for the
  full ladder.

### Coordination with Agent W

**W has identical hardware and shares the Anthropic rate limit.**
Coordinate generation + grading runs to avoid double-throughput
pressure on the 50 req/min ceiling. Practical pattern:

- Y trains → W trains in parallel (no API contention).
- Y intervenes (uses GPU) → W intervenes when GPU free.
- Y grades (uses Anthropic API) → W grades only when Y not grading,
  or both with `--n-workers 1` for safe co-running.

**Y's axis** (T at fixed k_pos=20): Steps 0-4 above.

**W's axis** (k_pos at fixed T=best-from-sweep): see
`agent_w/brief.md`.

**Meeting cell**: (T=5, k_pos=20, TXCBareAntidead, right-edge) appears
in both ladders. After both agents land it, *the numbers must agree
within concept noise* (≤0.27 from Y's seed=1 vs seed=42 multi-seed
data). If they disagree by more than that, training-seed variance is
biting harder than expected — flag to Han.

### Pod spec

| field | value |
|---|---|
| Hardware | RunPod A40, 46 GB VRAM, 46 GB pod RAM, 900 GB volume |
| Branch | `han-phase7-unification` (commit directly; do NOT branch off) |
| Git identity | `hxuany0@gmail.com` / `Han` (NOT the system-context email!) |
| Push auth | username `xuyhan`, token at `/workspace/.tokens/gh_token` |
| HF auth | `/workspace/.tokens/hf_token` |
| Anthropic auth | `/workspace/.tokens/anthropic_key` |
| Phase 7 root | `/workspace/temp_xc/experiments/phase7_unification/` |
| Case studies | `experiments/phase7_unification/case_studies/` |

Push command pattern (works around inline credential issues):

```bash
cd /workspace/temp_xc
GH=$(cat /workspace/.tokens/gh_token)
git -c "credential.helper=" \
    -c "credential.helper=!f() { echo username=xuyhan; echo password=$GH; }; f" \
    push origin han-phase7-unification
```

Commit pattern (note the `-c user.email`; the system-context email is wrong):

```bash
git -c user.email=hxuany0@gmail.com -c user.name=Han commit -m "..."
```

### Pre-registered outcomes (write decisions BEFORE running)

**Step 1 (TXC k_pos=20, T=5, right-edge, warm-started):**

- **Win** (success ≥ T-SAE k=20 + 0.15 at coh ≥ 1.5): headline result.
  Multi-seed verify on seed=1, write up under `agent_y_paper/`. Stop
  hill-climb, hand to Han for paper integration.
- **Tie** (within ±0.15 of T-SAE k=20): no architectural advantage at
  matched sparsity; sparsity is the dominant lever. Document in
  `agent_y_paper/`, run Step 2 *only* if you have spare cycles, then
  hand back.
- **Loss** (success ≤ T-SAE k=20 − 0.15): per-token prior is structurally
  better at this sparsity. Run Step 2-4 to find the failure mode (which
  axis hurts? encoder, decoder, contrastive head?). Negative result
  with mechanism is publishable.

### What's already eliminated (don't repeat)

- ln1-hook training. Ran at 12% paper-grade budget. Underperforms
  resid_post in every concept class. Mechanism is structural (pre-norm
  bandwidth limit). See `agent_y_paper/2026-04-29-y-ln1-pivot.md`.
- AxBench-additive vs paper-clamp protocol comparison. Done by previous
  Y + Dmitry. See `agent_y_paper/2026-04-29-y-tx-steering-final.md`.
- Per-position window-clamp at high sparsity. Q2.C done. See
  `agent_y_paper/2026-04-29-y-summary.md`.
- Magnitude-scale story as full explanation. Q1.1+Q1.2 closes ~10% of
  the gap; the rest is sparsity. See
  `agent_y_paper/2026-04-29-y-tx-steering-magnitude.md`.

### Open questions for next Y to flag back to Han

1. **Step 0 (clean T-SAE paper reproduction):** Y previously used a
   30-concept adaptation. The T-SAE paper's exact concept set + exact
   strengths is in their App. B. Anchor against those numbers
   *before* trusting Step 1's comparison. ~30 min of pipeline work.
2. **Coherence threshold**: 1.5 is a guess. The paper might use 2.0.
   Compute both, report both.
3. **Multi-seed**: every winning cell needs seed ∈ {1, 42} at
   minimum. Y previously found seed variance ≤0.27; with sparsity
   matching, variance might shrink (cleaner features) or grow (sparser
   = more sensitive to init).

### Reading list (in priority order)

1. `agent_y_paper/2026-04-29-y-summary.md` (the situation)
2. `agent_y_paper/2026-04-29-y-tx-steering-magnitude.md` (the
   methodology + Q1.x findings)
3. `agent_y_paper/2026-04-29-y-z-handoff.md` (the question this brief
   answers)
4. `agent_w/brief.md` (W's parallel work; your coordination partner)
5. `agent_y_brief.md` (Han's original Y mission for context)

### What this brief assumes — flag if false

- T-SAE k=20 ckpt at seed=42 still loadable from
  `experiments/phase7_unification/results/ckpts/` or HF
  `han1823123123/txcdr-base`.
- The 30-concept benchmark fixtures + grading prompts are unchanged
  from previous Y's run.
- Anthropic API key + HF token + GH PAT all still valid.
- Pod is A40 with same software state (uv-managed venv at
  `/workspace/temp_xc/.venv/`).
- Han is asleep / unavailable for the autonomous shift; flag back via
  writeups + auto-memory updates.
