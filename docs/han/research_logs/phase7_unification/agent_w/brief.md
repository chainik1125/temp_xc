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
- **You and Agent Y are running converging hill-climbs from opposite
  ends of architectural space.** Y starts at T-SAE k=20 (per-token,
  paper-clamp winner) and climbs *toward* TXC. You start from the
  best TXC variant identified by an initial sweep, and climb *back
  down* toward T-SAE k=20. The two ladders meet at (T=5, k_pos=20)
  — that's the cross-validation cell.
- **Hardware**: A40 RunPod, 46 GB VRAM, 46 GB pod RAM. Identical to
  Y. **You share the Anthropic 50 req/min rate limit with Y** —
  coordinate (see § Coordination).
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

#### Phase 1 — sweep to identify the best TXC starting point (~2 A40-days)

Reuse existing Phase 7 checkpoints where you can; train fresh only the
gap cells. Sweep design (12 cells):

| dim | values |
|---|---|
| T | 3, 5, 10 |
| k_pos | 20, 100 |
| arch family | TXCBareAntidead, MatryoshkaTXCDR |

Many cells already exist on `paper_archs.json:leaderboard_archs` —
read `experiments/phase7_unification/results/training_index.jsonl` to
find them. Cells we *don't* have (the gap fillers — anything with
k_pos=20 is missing because nobody has trained sparser-TXC) are the
fresh trainings. Estimate 2-4 fresh cells × ~4 A40-hours each.

**Warm-start trick** to make k_pos=20 cells cheaper: initialise the
TXC encoder/decoder from T-SAE k=20's weights (broadcast across the T
positions). Y has details in `agent_y_brief_phase2.md` § *Warm-start
trick*. Cuts training time 5-10×. Both Y and you should use this for
sparsity-matched cells.

**Evaluate every cell on the existing Q1.3 + Q2.C grading pipeline.**
Don't rebuild — see § *Existing pipeline you reuse* below. Same
30-concept benchmark, same family-normalised paper-clamp, same
Sonnet 4.6 grader.

**Pick the winner**: highest *peak success at coherence ≥ 1.5* (see
§ Metric). That's W's hill-climb starting cell.

#### Phase 2 — hill-climb FROM the swept winner toward T-SAE territory

Two axes from the winner:

- **Sparsity ladder** (k_pos *down* from winner toward 20):
  k_pos = winner, 50, 32, 20.
- **Receptive-field ladder** (T *narrow* from winner toward 1):
  T = winner, 3, 2, 1.

That's ~7 cells. Each is a single-axis step (don't move both knobs at
once — attribution).

**Stage gate.** After Phase 1 + the first sparsity-ladder step, if
the path is going in the wrong direction (success drops monotonically
as you approach T-SAE territory), stop ladder, flip strategy: keep
the swept winner as the headline TXC, document, hand back.

### Metric

**Peak success at coherence ≥ 1.5** (Y's family-normalised paper-clamp
protocol, Q1.3). Single defensible number per arch. Coherence threshold
defines "the steered output is still readable text". This is the
metric the paper will report.

Secondary numbers to track:
- Peak success across the s_norm sweep (Y's primary headline number)
- Per-concept-class breakdown (knowledge / discourse / safety /
  stylistic / sentiment) — does the win pattern shift with sparsity?

### Coordination with Agent Y

**Same hardware. Same Anthropic key. Same case study. Different
axis.**

| dimension | Y owns | W owns |
|---|---|---|
| Hill-climb axis | T at fixed k_pos=20 | k_pos at fixed T (= sweep winner) |
| Starting cell | T-SAE k=20 | best-from-W's-sweep |
| Ending cell | full TXCDR (T=5, k_pos=20, per-position, matryoshka, multidist) | T-SAE-shaped cell at the bottom of the ladder |
| Convergence cell | (T=5, k_pos=20, TXCBareAntidead, right-edge) — Y's Step 1 = W's Phase 2 endpoint | same |

**Convergence-cell rule.** The convergence cell will be trained by
**whoever gets there first**. Coordinate via commit messages — when
you complete it, commit with `Agent W convergence cell` in the
subject; check git log before training to see if Y has already done
it. If both train it (e.g., for multi-seed), the numbers should agree
within ±0.27 (Y's previously observed seed variance). If they
disagree by more than that, flag.

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

### Existing pipeline you reuse

See `agent_y_brief_phase2.md` § *Existing pipeline you reuse* for the
full file list. Key entry points:

- **Training template (single-arch, self-contained):**
  `experiments/phase7_unification/case_studies/train_ln1_txc.py`
  — adapt for k_pos=20, warm-started from T-SAE k=20.
- **Feature selection + Z magnitudes**:
  `select_features.py` + `diagnose_z_magnitudes.py`.
- **Steering**: `intervene_paper_clamp_normalised.py`.
- **Grading**: `grade_with_sonnet.py` with `--fix-errors` resume mode.
- **Comparison plotting**:
  `experiments/phase7_unification/case_studies/steering/compare_ln1_vs_resid.py`
  is a template for arch-vs-arch; copy for multi-arch ladder
  comparison.

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

**Phase 1 (sweep):**

- *Multiple TXC variants beat T-SAE k=20 at peak success / coh ≥ 1.5*
  → architecture genuinely matters; pick the best as Phase 2 anchor;
  **headline result candidate**.
- *Best TXC variant ties T-SAE k=20* → architecture is roughly neutral
  at this sparsity; Phase 2 may still find a winner via further
  sparsity-matching. Run it; cap at the sparsity ladder if no signal.
- *Best TXC variant loses to T-SAE k=20 by ≥0.15* → architecture
  has anti-prior at sparse end. Phase 2 unlikely to recover. Stop;
  document as "we tried TXC at sparsity-matched starting points and
  per-token wins". Publishable null.

**Phase 2 (hill-climb):**

- *Monotonic improvement as we approach (T=5, k_pos=20)* → smooth
  architectural landscape; the winner cell is the paper headline.
- *Non-monotone* (best cell is in the middle of the ladder) → the
  optimal architecture is some intermediate; report which axis
  (sparsity / receptive field / decoder structure) carries the win.
- *No cell beats Y's Step 1 baseline* → both ladders agree no
  architecture-axis effect at matched sparsity; converging negative
  result is the publishable finding.

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
