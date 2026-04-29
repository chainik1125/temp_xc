---
author: Han
date: 2026-04-29
tags:
  - design
  - in-progress
---

## Agent W brief — Hail Mary hill-climb on Gemma-2-2b steering, starting from a swept TXC

> Welcome. Read `brief.md` (top-level) first for paper-wide context
> (subject models, k_win convention, methodology, source-of-truth
> files). Then read `agent_y_paper/README.md` to see what Agent Y
> shipped — your starting point and your coordination partner.
> Then this brief.

### TL;DR

- **The paper is in NeurIPS-rescue mode.** Deadline ~2026-05-06.
  Headline target: find a TXC architecture that *wins* on at least
  one case study at matched sparsity vs the per-token T-SAE k=20
  baseline.
- **You and Agent Y are searching the same architectural landscape
  with different strategies.** Y walks an atomic axis ladder from
  T-SAE k=20 to full TXCDR, all at fixed k_pos=20. You sweep ~6
  candidate TXC variants to identify the best starting point, then
  greedy-local-search from there in any architectural direction that
  improves the metric. The two paths *may* cross at
  (T=5, k_pos=20, TXCBareAntidead, right-edge) — Y is guaranteed to
  visit it, you may or may not depending on where the sweep winner
  lands. If both do train it, the seed=42 numbers must agree within
  ±0.27 (Y's observed seed σ).
- **Hardware**: A40 RunPod, 46 GB VRAM, 46 GB pod RAM, 900 GB
  volume. Identical to Y. **You share the Anthropic 50 req/min rate
  limit with Y** — coordinate (see § Coordination).
- **Branch**: `han-phase7-unification`. Push as `xuyhan` with
  `/workspace/.tokens/gh_token`. Commit identity `hxuany0@gmail.com`.

### Why this work matters

The previous Y shift produced two findings worth keeping:

1. **Sparsity decomposition.** T-SAE k=20's headline lead over TXC
   under paper-clamp is *dominantly* a k=20 sparsity effect, not an
   architectural one. At matched sparsity (k_eff ≈ 500), TXC matryoshka
   peak success TIES T-SAE k=500 (1.37 vs 1.38). The big remaining
   question: **does the same TIE hold at the sparser end? Does TXC at
   k_pos=20 match or beat T-SAE k=20?** Nobody has trained a TXC at
   that sparsity yet — it's the missing cell.

2. **Per-concept-class structural pattern.** TXC family wins on
   knowledge-domain concepts (medical, math, historical, code,
   scientific) by 0.32 mean success points; T-SAE k=20 wins on
   discourse / register concepts by 2.00. Aligns with the multi-token
   receptive-field argument.

The paper's narrative target is **"TXC is competitive across protocols,
with a structural prior that pays off where multi-token concept
structure is the natural unit"** — *not* "TXC SOTA". W's job is to
sharpen the "competitive" claim into a "wins at matched sparsity"
claim if the data permits.

### Your mandate, in two phases

#### Phase 1 — sweep to identify the best TXC starting point

**Important context: the matched-sparsity cells (k_pos=20) do not
exist anywhere in the canonical leaderboard.** I checked
`canonical_archs.json` — every TXC arch trained so far obeys the
paper-wide `k_win = T × k_pos = 500` convention, so k_pos ranges
from 167 (T=3) down to 16 (T=32) but never reaches 20. The k_pos=20
cells are the genuine gaps. **Most of W's training compute will be
spent here.**

**Recommended sweep (6 cells; 4 fresh trainings, 2 reused):**

| cell | T | k_pos | family | status | notes |
|---|---|---|---|---|---|
| A | 5 | 100 | TXCBareAntidead | exists (`txc_bare_antidead_t5__seed42.pt`) | reuse; previous Y already graded under Q1.3 (peak ~1.07 at s_norm=10) |
| B | 5 | 100 | MatryoshkaTXCDR | exists (`agentic_txc_02__seed42.pt`) | reuse; previous Y graded (peak ~1.07 Q1.3, ~1.37 Q2.C) |
| C | 3 | 20 | TXCBareAntidead | **NEW** | low-T sparsity-matched probe |
| D | 5 | 20 | TXCBareAntidead | **NEW** | = Y's possible-Step-2 meeting cell |
| E | 5 | 20 | MatryoshkaTXCDR | **NEW** | matryoshka @ matched sparsity |
| F | 10 | 20 | TXCBareAntidead | **NEW** | longer window @ matched sparsity |

If you want richer T axis at matched sparsity (T=4, 8 between cells
C and F), train those too — each is ~4 A40-hours. The existing canonical
leaderboard has full T-axis coverage (T=3..32) at k_win=500 if you want
to read prior probe-AUC numbers, but those aren't graded on the steering
benchmark — they'd need their own intervene + grade run if you want them
in the sweep.

**Warm-start trick** to make k_pos=20 cells cheaper: initialise the
TXC encoder/decoder from T-SAE k=20's weights (broadcast across the T
positions, decoder divided by T so right-edge sum reproduces T-SAE
init). See `agent_y_brief_phase2.md` § *Warm-start trick* for
details. Cuts training time 5–10× and makes the k_pos=20 cells *true
minimum-deviation experiments* from T-SAE k=20.

**Evaluate every cell on the existing Q1.3 grading pipeline.**
Same 30-concept benchmark, same family-normalised paper-clamp, same
Sonnet 4.6 grader. Reuse, don't rebuild — see § *Existing pipeline you
reuse* below.

**Pick the winner**: highest peak success at coherence ≥ 1.5 (see
§ Metric). That's W's hill-climb starting cell.

#### Phase 2 — greedy local hill-climb from the swept winner

Han's framing: "see how it affects the AUC of the cohering vs
steering plot — that should be the ultimate hill climbing objective".
This is **greedy local search**, not a fixed ladder toward a specific
endpoint. From the Phase-1 winner, perturb each architectural axis by
a single step, evaluate, pick the best perturbation, iterate until no
single-step move improves the metric.

**Perturbation axes** (each move is one atomic change):

| axis | step | direction |
|---|---|---|
| receptive field T | T → T±1 (clamp T ≥ 1) | both directions |
| per-position sparsity k_pos | k_pos → {⌊k_pos/2⌋, k_pos×2} (clamp ≥ 4) | both directions |
| family | swap between {TXCBareAntidead, MatryoshkaTXCDR, MultiDistanceContrastive} | both directions |
| decoder write-back | flip {right-edge, per-position} | both directions |

**Iteration cap**: 4 hill-climb steps after the sweep winner (i.e.,
4 fresh trainings + grading on top of Phase 1). If the metric is still
strictly increasing at step 4, ask Han before continuing.

**Stage gate.** If at any step the swept winner is *not improved* by
any of the perturbations (true local maximum), stop. Document the
local-max cell + the perturbation table (which axis hurts least, which
hurts most). That's a paper artefact.

### Metric

> **Han said "AUC of coherence vs steering" as the hill-climb
> objective. The brief locks in a concrete operationalisation below;
> if Han later confirms a different definition, switch.**

**Primary** (what Y and W both optimise; what the paper reports):
peak success at coherence ≥ 1.5, on Y's family-normalised paper-clamp
protocol (Q1.3). Single defensible number. Coherence threshold defines
"the steered output is still readable text".

**Secondary** (compute alongside; cheap):
- Peak success across the full s_norm sweep (no coherence constraint).
- Area under the success(coh)-vs-coh curve from coh=1.0 to coh=3.0
  (a literal "AUC" reading of Han's phrasing — bin-sensitive but
  report for completeness).
- Per-concept-class breakdown (knowledge / discourse / safety /
  stylistic / sentiment) — does the win pattern shift with sparsity?
  Reuse `experiments/phase7_unification/case_studies/steering/plot_concept_class.py`
  pattern.

### Coordination with Agent Y

**Same hardware. Same Anthropic key. Same case study. Different
strategy.**

| dimension | Y | W |
|---|---|---|
| Strategy | atomic axis ladder T-SAE → full TXCDR at fixed k_pos=20 | sweep TXCs → greedy local hill-climb from the winner |
| Starting cell | T-SAE k=20 (T=1, per-token) | swept winner (likely T=5, k_pos=20 family TBD) |
| Path | guaranteed to visit (T=5, k_pos=20, bare, right-edge) at Step 2 | may or may not visit (T=5, k_pos=20, bare, right-edge) — depends on greedy walk |

**Possible meeting cell**: (T=5, k_pos=20, TXCBareAntidead, right-edge)
= Y's Step 2 = W's Phase 1 cell D. **Y is guaranteed to train this
cell; W's Phase 1 also trains it as part of the 6-cell sweep.**
Coordinate so neither agent re-trains it:

- Tag the commit that lands the trained ckpt with `[meeting cell]`
  in the subject line.
- Before training, run `git pull` and `git log --grep="meeting cell"`
  to check whether the other agent already did it. If yes, pull the
  ckpt + grades from HF / disk and reuse.
- If both seed=42 numbers exist (e.g., one agent trained, the other
  reproduced), they must agree within ±0.27 (Y's observed seed σ).
  Wider disagreement → flag to Han.

For non-meeting cells in W's Phase 2 hill-climb, no coordination
needed — those are W's local search.

**Anthropic rate limit.** 50 req/min shared across both agents. Two
patterns:

- **Sequential**: Y grades, then W grades. Each grading run is ~15
  min per arch × 14 strengths × 30 concepts ≈ ~30-45 min wall.
  Whoever submits a grading job first locks the API; the other waits.
- **Parallel with `--n-workers 1` each**: split the 50 req/min into
  ~25 each. Slower (each takes 60-90 min) but no contention.
  Recommended pattern when you both have grading queued.

Read `/tmp/*grade*.log` files; if any show `429` errors or `0.0 gen/s`,
back off.

**GPU contention.** Single A40, single GPU. Stagger trainings — only
one at a time. `nvidia-smi` to check before launching. Training
processes show as `python -m experiments.phase7_unification...`; if
you see Y's process running, queue.

**Disk hygiene.** Reuse the activation cache across cells — don't let
`train_ln1_txc.py`-style scripts each build their own private 1.5 GB
cache per cell. Each TXCBareAntidead ckpt at d_sae=18432 is ~1.7 GB;
4 fresh Phase-1 trainings + ≤4 Phase-2 hill-climb steps = ~14 GB of
new ckpts plus the activation cache. Comfortably fits in the 900 GB
volume; ship to HF (`han1823123123/txcdr-base`) and delete locally
only if you hit ≥80% disk usage from other artefacts.

### Existing pipeline you reuse

**Per-cell workflow checklist** (do these in order for each cell):

1. **Check for existing artefacts.** Before doing anything, look for:
   - Ckpt: `experiments/phase7_unification/results/ckpts/<arch_id>__seed42.pt`
     (or pull from HF `han1823123123/txcdr-base` via
     `experiments/phase7_unification/case_studies/_download_ckpts.py --arch <arch_id> --seed 42`).
   - Training log: `experiments/phase7_unification/results/training_logs/<arch_id>__seed42.json`.
   - Feature selection: `experiments/phase7_unification/results/case_studies/steering/<arch_id>/feature_selection.json`.
   - Z magnitudes: `experiments/phase7_unification/results/case_studies/diagnostics_*/z_orig_magnitudes.json` (check all subdirs).
   - Generations + grades: `experiments/phase7_unification/results/case_studies/steering_paper_normalised/<arch_id>/{generations,grades}.jsonl` — **if grades.jsonl has 210 rows, the cell is done; reuse, don't rerun**.
2. **Train** (if no ckpt). Template:
   `experiments/phase7_unification/case_studies/train_ln1_txc.py`
   — adapt for k_pos=20, warm-started from T-SAE k=20 weights.
3. **Select features**:
   `experiments/phase7_unification/case_studies/steering/select_features.py`.
4. **Diagnose z magnitudes** (Q1.1 protocol):
   `experiments/phase7_unification/case_studies/steering/diagnose_z_magnitudes.py`.
   **Required** — intervene script reads `<|z|>` per arch from this
   diagnostics output for the family-normalised strength schedule.
5. **Intervene** (Q1.3 paper-clamp normalised):
   `experiments/phase7_unification/case_studies/steering/intervene_paper_clamp_normalised.py`.
6. **Grade** (Anthropic 50 req/min ceiling, shared with Y):
   `experiments/phase7_unification/case_studies/steering/grade_with_sonnet.py`
   with `--n-workers 1` and `--fix-errors` resume mode if any errors
   appear.
7. **Compare** to T-SAE k=20 anchor + the previous best cell:
   `experiments/phase7_unification/case_studies/steering/compare_ln1_vs_resid.py`
   is the template for arch-vs-arch (copy + modify for multi-arch).

Skip steps 1, 3, 4, 5, 6 for any arch where artefacts already exist.

### Pod spec

| field | value |
|---|---|
| Hardware | RunPod A40, 46 GB VRAM, 46 GB pod RAM, 900 GB volume |
| Branch | `han-phase7-unification` (commit directly) |
| Git identity | `hxuany0@gmail.com` / `Han` (NOT the system-context email!) |
| Push auth | username `xuyhan`, token at `/workspace/.tokens/gh_token` |
| HF auth | `/workspace/.tokens/hf_token` |
| Anthropic auth | `/workspace/.tokens/anthropic_key` |
| Phase 7 root | `/workspace/temp_xc/experiments/phase7_unification/` |
| Working dir | `docs/han/research_logs/phase7_unification/agent_w/` (this brief lives here; put your writeups + plots here too) |

Push command pattern (works around inline credential issues):

```bash
cd /workspace/temp_xc
GH=$(cat /workspace/.tokens/gh_token)
git -c "credential.helper=" \
    -c "credential.helper=!f() { echo username=xuyhan; echo password=$GH; }; f" \
    push origin han-phase7-unification
```

Commit pattern:

```bash
git -c user.email=hxuany0@gmail.com -c user.name=Han commit -m "..."
```

### Pre-registered outcomes (write decisions BEFORE running)

**Threshold rule.** Previous Y observed seed-to-seed variance up to
±0.27 success-points across the 30-concept benchmark (T-SAE k=20 was
identical at 1.80 across seeds; other archs spread up to ±0.27). Use
**±0.27 (= 1× σ_seeds observed)** as the calling threshold. Win = best
cell ≥ T-SAE k=20 + 0.27 at peak success / coh ≥ 1.5. Tie = within
±0.27, requires multi-seed verification before being called. Loss
= ≤ T-SAE k=20 − 0.27.

**Phase 1 (sweep):**

- *At least one cell beats T-SAE k=20 by ≥0.27* → architecture
  genuinely matters at matched sparsity. Verify the candidate at
  seed=1; if confirmed, lock as Phase 2 starting point. **Headline
  result candidate.**
- *No cell beats T-SAE k=20 by ≥0.27, but at least one ties (within
  ±0.27)* → ambiguous. Run multi-seed on the best candidate. If still
  tied → "sparsity is the dominant lever, architecture is secondary".
  Phase 2 unlikely to find a win in pure perturbation; ask Han.
- *Every cell loses to T-SAE k=20 by ≥0.27* → architecture has
  anti-prior at matched sparsity. Phase 2 perturbations also unlikely
  to recover. Document the per-cell + per-class breakdown to identify
  which axis hurts most. **Publishable converging null** (with Y's
  parallel finding from the other direction).

**Phase 2 (greedy local hill-climb):**

- *At least one perturbation step strictly improves the metric by
  ≥0.27 over the swept winner* → architectural landscape has
  exploitable structure. Iterate; report the local maximum's axis
  attribution.
- *No perturbation improves over the swept winner* → swept winner
  is a local maximum. Report the perturbation table (per-axis
  delta) — that's a paper artefact regardless of sign.
- *Some perturbation tips the metric below T-SAE k=20* → useful
  failure-mode evidence; report which axis broke it.

### What's NOT in scope

- **Backtracking case study** (Llama-3.1-8B + DeepSeek-R1-Distill on
  Ward 2025). Originally proposed for W; ditched due to compute
  (HF gate on Llama-3.1-8B + ~10 H100-days estimate). Aniket
  (`origin/aniket`) has Stage A done and a Stage B plan; if Han
  unblocks HF + compute, that work is Aniket's.
- **HH-RLHF dataset case study**. Agent C did Stage 1; mostly stable.
  Out of scope unless the Gemma-steering hill-climb gets to a
  publishable result and you have spare cycles.
- **ln1-hook training.** Y already tried; null with mechanism. Don't
  repeat.
- **MLC paper-clamp.** Multi-layer hook complexity not implemented;
  out of scope for this rescue shift.

### Reading list (in priority order)

1. `agent_y_paper/2026-04-29-y-summary.md` — the situation
2. `agent_y_paper/2026-04-29-y-tx-steering-magnitude.md` — methodology
3. `agent_y_paper/2026-04-29-y-tx-steering-final.md` — the
   sparsity-decomposition story
4. `agent_y_paper/2026-04-29-y-z-handoff.md` — the sparser-TXC
   question
5. `agent_y_brief_phase2.md` — your coordination partner's brief
6. `brief.md` (top-level Phase 7 brief) — paper-wide context
7. `agent_y_brief.md` — Han's original Y mission, useful background

### What this brief assumes — flag if false

- T-SAE k=20 ckpt + the existing TXC checkpoints (matryoshka,
  bare-antidead, H8, multi-distance) all loadable from
  `experiments/phase7_unification/results/ckpts/` or HF
  `han1823123123/txcdr-base`. (Use
  `experiments/phase7_unification/case_studies/_download_ckpts.py`
  if missing locally.)
- The 30-concept benchmark fixtures + grading prompts are unchanged
  from Y's run.
- Anthropic API key + HF token + GH PAT all still valid.
- Pod is A40 with software state matching Y's pod (uv-managed venv at
  `/workspace/temp_xc/.venv/`).
- Han is asleep / unavailable; flag back via writeups +
  auto-memory updates.

### Working dir convention

Put your writeups and plots under
`docs/han/research_logs/phase7_unification/agent_w/`:

- `brief.md` (this file)
- `2026-MM-DD-w-<topic>.md` (your experiment writeups, one per
  experiment)
- `summary.md` (your end-of-shift summary)
- `plots/<plot>.png` + `<plot>.thumb.png`

When the workstream concludes, bundle the paper-deliverable subset
into `agent_w_paper/` (mirror `agent_x_paper/` and `agent_y_paper/`
layout) — Han or you can do that at handoff.

### One sentence on the mood

The previous Y shift converted "TXC loses under paper-clamp" into
"TXC is competitive at matched sparsity, with a knowledge-domain
structural advantage". Your job is to convert "competitive" into
"wins" — at any single sparsity-matched cell, on any single
case study, by any single architectural variant. **Find a positive
result if there's one to find.** If there isn't, the converging-null
between you and Y is also a clean publishable finding — but the
Hail Mary is the asymmetric bet, and you're the one swinging.
