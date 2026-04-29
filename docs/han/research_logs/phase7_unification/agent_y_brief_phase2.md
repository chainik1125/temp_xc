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
via atomic single-axis architectural changes.** All cells are at
k_pos=20 (= matched per-token sparsity with T-SAE k=20). Each step
isolates one architectural axis from the previous step; this lets the
paper attribute any win to a specific axis.

| step | arch | T | encoder | decoder | new training? | what changes vs prior step |
|---|---|---|---|---|---|---|
| 0 | T-SAE k=20 (anchor) | 1 | per-token | per-token | no — exists at `tsae_paper_k20__seed42.pt` | baseline |
| 1 | TXCBareAntidead | 2 | T=2 window | right-edge of window | yes | +1 receptive-field step (encoder sees current token + 1-back) |
| 2 | TXCBareAntidead | 5 | T=5 window | right-edge of window | yes | encoder window expanded to T=5 (full receptive-field axis) |
| 3 | TXCBareAntidead | 5 | T=5 window | per-position | yes | decoder write-back axis (Q2.C-style) |
| 4 | MatryoshkaTXCDR | 5 | T=5 window | per-position | yes | adds matryoshka contrastive loss |
| 5 | MatryoshkaTXCDR + multidist | 5 | T=5 window | per-position | yes | adds multi-distance contrastive heads (= H8 family) |

**Step 0 status.** Already trained; previous Y already graded it under
Q1.3 (peak success = 1.80 at s_norm=10). Pipeline-faithfulness check
also already done — previous Y cross-checked vs Dmitry's reported
numbers and matched within ≤0.13 (see
`agent_y_paper/2026-04-29-y-tx-steering-magnitude.md`). **No new work
on Step 0.** The existing grades file is at
`experiments/phase7_unification/results/case_studies/steering_paper_normalised/tsae_paper_k20/grades.jsonl`
(210 rows, reuse).

**Stage gate after Step 2** (not Step 1 — T=2 alone is unlikely to
fully resolve the question; Step 2 is the full T=5 receptive field at
matched sparsity). Three outcomes:

- Step 2 *beats* T-SAE k=20 by ≥+0.27 (= 1× σ_seeds observed by previous
  Y) at peak success → run multi-seed verify on seed=1; if confirmed,
  it's the headline finding. Steps 3–5 become a paper-completeness
  exercise (does the win grow with decoder write-back, matryoshka,
  multidist?).
- Step 2 *ties* T-SAE k=20 within ±0.27 → ambiguous; **multi-seed
  disambiguation required before calling outcome.** Train Step 2 at
  seed=1, regrade. If still tied → narrative collapses to "sparsity
  alone is the lever"; Steps 3–5 unlikely to help, hand back. If
  seed=1 swings → Step 2 is on the edge, run Steps 3–4 to see if
  decoder/matryoshka tips it.
- Step 2 *loses* to T-SAE k=20 by ≥0.27 → architectural anti-prior at
  matched sparsity. Continue Steps 3–5 as failure-mode investigation
  (which axis hurts least?). Negative result + mechanism is publishable.

### Metric

> **Han said "AUC of coherence vs steering" as the hill-climb objective.
> The brief locks in a concrete operationalisation below; if Han
> later confirms a different definition, switch.**

**Primary** (what the paper reports per arch): peak success at
coherence ≥ 1.5, on the family-normalised paper-clamp protocol (Q1.3).
Single defensible number. The threshold defines "the steered output is
still readable text", not "topic-drifted gibberish".

**Secondary** (compute alongside the primary; cheap):
- Peak success across the full s_norm sweep (the headline number under
  Q1.3, no coherence constraint).
- Area under the success(coh)-vs-coh curve from coh=1.0 to coh=3.0
  (a literal "AUC" reading of Han's phrasing — sensitive to bin
  choice, but report it for completeness).
- Per-concept-class breakdown (knowledge / discourse / safety /
  stylistic / sentiment) — does the win pattern shift with sparsity?
  Reuse `experiments/phase7_unification/case_studies/steering/plot_concept_class.py`
  pattern.

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

**Per-cell workflow checklist** (do these in order for each new arch
in Steps 1–5):

1. **Check for existing artefacts.** Before doing anything, look for:
   - Ckpt: `experiments/phase7_unification/results/ckpts/<arch_id>__seed42.pt`
     (or pull from HF `han1823123123/txcdr-base` via
     `experiments/phase7_unification/case_studies/_download_ckpts.py --arch <arch_id> --seed 42`).
   - Training log: `experiments/phase7_unification/results/training_logs/<arch_id>__seed42.json`.
   - Feature selection: `experiments/phase7_unification/results/case_studies/steering/<arch_id>/feature_selection.json`.
   - Z magnitudes: `experiments/phase7_unification/results/case_studies/diagnostics_*/z_orig_magnitudes.json` (check all subdirs — previous Y kept multiple).
   - Generations + grades: `experiments/phase7_unification/results/case_studies/steering_paper_normalised/<arch_id>/{generations,grades}.jsonl` — **if grades.jsonl has 210 rows, the cell is done; reuse, don't rerun**.
2. **Train** (if no ckpt). Template: `experiments/phase7_unification/case_studies/train_ln1_txc.py`
   is the closest single-arch self-contained Gemma-2-2b TXC trainer.
   Copy + modify for k_pos=20 + warm-start (see § Warm-start trick).
3. **Select features** for the 30-concept benchmark:
   `experiments/phase7_unification/case_studies/steering/select_features.py`
   (already `hook_name`-aware).
4. **Diagnose z magnitudes** for the family-normalised strength schedule:
   `experiments/phase7_unification/case_studies/steering/diagnose_z_magnitudes.py`
   (Q1.1 protocol). **Required** before intervention — the
   intervene script reads `<|z|>` per arch from the diagnostics output.
5. **Intervene** with paper-clamp normalised (Q1.3):
   `experiments/phase7_unification/case_studies/steering/intervene_paper_clamp_normalised.py`.
6. **Grade** (Anthropic 50 req/min ceiling; W shares this limit —
   coordinate timing): `experiments/phase7_unification/case_studies/steering/grade_with_sonnet.py`
   with `--n-workers 1` and `--fix-errors` resume mode if any errors
   appear.
7. **Compare** to T-SAE k=20 anchor + previous step:
   `experiments/phase7_unification/case_studies/steering/compare_ln1_vs_resid.py`
   is the template for arch-vs-arch (copy + modify for ladder
   comparison).

Skip steps 1, 3, 4, 5, 6 for any arch where artefacts already exist.

### Coordination with Agent W

**W has identical hardware and shares the Anthropic rate limit.**
Coordinate generation + grading runs to avoid double-throughput
pressure on the 50 req/min ceiling. Practical pattern:

- Y trains → W trains in parallel (no API contention).
- Y intervenes (uses GPU) → W intervenes when GPU free.
- Y grades (uses Anthropic API) → W grades only when Y not grading,
  or both with `--n-workers 1` for safe co-running.

**Y's path** (atomic axis ladder from T-SAE k=20 to full TXCDR at
fixed k_pos=20): Steps 0–5 above.

**W's path** (sweep over TXCs to find best starting point, then
greedy local search): see `agent_w/brief.md`. W's hill-climb may
move along *any* architectural axis from its starting cell — it's
not committed to a path that crosses Y's.

**Possible meeting cell**: (T=5, k_pos=20, TXCBareAntidead, right-edge)
= **Y's Step 2**. W's sweep + greedy may or may not land on this exact
cell. *If both agents do train this cell*, the seed=42 numbers must
agree within ±0.27 (Y's observed seed-noise σ); a wider disagreement
means training-seed variance is biting harder than expected and you
should flag to Han.

**Coordination protocol when training the meeting cell** (Y's Step 2
= W's Phase 1 cell D = TXCBareAntidead T=5 k_pos=20 right-edge):
before launching the training, `git pull` and run
`git log --grep="meeting cell"` to check whether W already trained it.
If yes:

- Reuse the ckpt + grades wholesale; pull from HF (`han1823123123/txcdr-base`)
  if not on local disk.
- Don't redundantly train; use the saved ~4 A40-hours for the next
  cell on your path.

Tag your meeting-cell commit with **`[meeting cell]`** in the subject
line (same tag W uses) so the future agent on either side can spot
it via grep.

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

**Threshold rule.** Previous Y observed seed-to-seed variance up to
±0.27 success-points across the 30-concept benchmark; T-SAE k=20
specifically was rock-stable (1.80 at seed=42 = 1.80 at seed=1, Δ=0).
Use **±0.27 (= 1× σ_seeds observed)** as the calling threshold. Any
result within ±0.27 of T-SAE k=20 = ambiguous, requires multi-seed
verification before being called. Anything ≥ +0.27 = win signal.
Anything ≤ −0.27 = loss signal.

**Step 2 outcomes** (TXC T=5, k_pos=20, right-edge, warm-started; the
critical canary):

- **Win** (success ≥ T-SAE k=20 + 0.27 at peak, coh ≥ 1.5): paper
  headline candidate. Train at seed=1 to verify; if seed=1 holds the
  win, lock it in and proceed with Steps 3–5 for paper completeness
  (does the win grow with decoder + matryoshka + multidist?).
- **Tie** (within ±0.27 of T-SAE k=20): ambiguous. **Train Step 2 at
  seed=1 first**, regrade. If seed=1 also ties → narrative collapses
  to "sparsity is the dominant lever, architecture is secondary";
  hand back to Han, do not pursue Steps 3–5 unless explicitly asked.
  If seed=1 swings → on the edge; run Steps 3–4 to see if the richer
  archs tip it past +0.27.
- **Loss** (success ≤ T-SAE k=20 − 0.27): per-token prior is
  structurally better at this sparsity. Run Steps 3–5 to find the
  failure mode (which axis hurts least?). Converging negative result
  with mechanism is publishable.

**Step 1 (T=2 minimum-deviation):** secondary diagnostic. Use it to
attribute Step 2's outcome — if Step 2 wins but Step 1 doesn't, the
"win" is from going past T=2 to T=5 (genuine multi-token feature
benefit). If Step 1 already wins, the receptive-field axis pays off
even at T=2.

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

1. **The exact metric definition.** Brief locks in "peak success at
   coh ≥ 1.5" with a literal-AUC secondary number; Han said "AUC of
   coherence vs steering" but didn't pick a definition. If Han
   confirms a different choice (coh ≥ 2.0, integrated AUC vs
   thresholded peak, etc.), switch — but report numbers for both
   to make the answer robust.
2. **T-SAE paper App B reproduction.** Han's directive was "start
   from T-SAE k=20 and *cleanly reproduce* its case studies". The
   Phase 7 30-concept benchmark is an adaptation; the original paper
   has a smaller concept set in App. B. Previous Y *implicitly*
   anchored against the paper via the cross-check vs Dmitry's
   reported numbers (matched within ≤0.13). If Han wants a *literal*
   reproduction on the paper's exact concept list, that's a separate
   ~half-day of pipeline work — flag and ask before committing.
3. **Multi-seed variance at sparser k_pos.** Previous Y found
   variance ≤0.27 at canonical k_win=500. At k_pos=20 (k_win=100),
   variance might shrink (cleaner sparser features) or grow (more
   init-sensitivity). Run multi-seed early on Step 2 to recalibrate
   the threshold rule if needed.

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
