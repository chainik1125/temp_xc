# PROTOCOL.md — agent operating rules

Concrete rules for multi-agent coordination on this paper. Read fully before
your first action.

## 1. Branch model

- **`final`** is the only paper branch. All work commits here.
- Never push to `main`, `han-phase7-unification`, `em-nanda`,
  `aniket-ward-stage-b`, or any other branch. Those are wasteland.
- Pull `final` before each session: `git pull --rebase origin final`.
- Push frequently — at least once per substantive change. If a push
  conflicts, rebase (don't merge).

## 2. The wasteland boundary

The repository contains a large body of historical work outside
`purified/`. Treat it as **read-only documentation** for context.

- **Allowed:** read files under `temp_xc/{src,experiments,docs,papers}` for
  reproduction context, baseline numbers, or paper references.
- **Forbidden:** `import` from `temp_xc.src`; `from src ...`; `subprocess`
  invocations of root-level scripts; modifying any wasteland file.
- If reference code is useful, copy it into `purified/src/temp_bench/`
  with attribution in a header comment. Duplication is fine; coupling is not.

### 2a. Cross-branch reads (em-nanda, aniket-ward-stage-b)

Dmitry's emergent-misalignment work and Aniket's backtracking work live
on **other branches** that are still being updated. We never merge them
into `final` — that would freeze stale snapshots and create conflict
surface. Read them directly from origin instead:

```bash
# Refresh first (run once per agent session)
bash purified/scripts/wasteland_refresh.sh

# Read a file from a sibling branch
git show origin/em-nanda:docs/dmitry/results/em_features/em_nanda_results_paper.md
git show origin/aniket-ward-stage-b:docs/aniket/experiments/ward_backtracking/handoff_neurips_push.md

# List a directory on a sibling branch
git ls-tree -r origin/em-nanda --name-only | grep em_features
```

If you need to copy code from a sibling branch into `purified/`, copy
once with attribution + the source commit hash in a header comment, then
stop tracking origin from that point. Live-importing is forbidden;
porting is fine.

## 3. Filesystem ownership

| Path | Owner | Mutability |
|---|---|---|
| `purified/agents/<name>/` | Agent `<name>` only | Owner write, others read |
| `purified/docs/components/c{N}.md` | Component lead (see briefing) | One agent at a time; coordinate via header comment |
| `purified/docs/paper/` | Agent PAPER | PAPER write, others read |
| `purified/src/temp_bench/` | First mover, then negotiate | Treat as shared; small additive PRs |
| `purified/results/runs/<run_id>/` | Run owner | Append-only after run completes |
| `purified/results/leaderboard.jsonl` | All | **Append-only**; use `flock` |
| `purified/checkpoints/manifest.jsonl` | All | **Append-only**; use `flock` |

## 4. Run-id contract

`run_id = <component>_<arch>_<seed>_<short_hash>`

- `<component>` ∈ {c1, c2, ..., c7}
- `<arch>` ∈ {topk_sae, tsae, txc_base, txc_pro, ...}
- `<seed>` is the rng seed (int)
- `<short_hash>` is `secrets.token_hex(4)` (8 hex chars)

Compute the run_id **before** kicking off training/eval; pass it through
every artifact (ckpt filename, plot dir, leaderboard row).

## 5. Two-TXC discipline

- **TXC-base** = `txc_bare_antidead_t5`. Implementation lives in
  `src/temp_bench/architectures/txc_base.py`.
- **TXC-pro** = `phase5b_subseq_h8`. Implementation in
  `src/temp_bench/architectures/txc_pro.py`.

Free parameters per component: `k_pos` (sparsity), `d_sae` (dict size),
`d_in` and `T_max` (forced by data). Everything else is fixed.

If you find yourself wanting to change architectural hyperparameters,
**stop and post to `docs/components/cN.md` first**. Justify it. The paper
makes a "two architectures everywhere" claim that breaks if any component
silently drifts.

## 6. Baselines (also locked)

| Slug | Description |
|---|---|
| `topk_sae` | Per-token TopK SAE, k=k_pos, d_sae=d_sae. The simple baseline. |
| `tsae_paper` | T-SAE (Bhalla et al. 2025). Use the paper's released config. |
| `tfa` | Temporal Feature Analysis (priors_in_time). Used in C1/C2/C7 only. |
| `mlc` | Multi-layer crosscoder (Lieberum et al. 2024, paper config). C3 only. |
| `sae_arditi` | EM-only baseline. The C6 winner. |

## 7. Component writeup template

Each `docs/components/cN.md` follows this structure:

```markdown
---
component: cN
status: planning|running|complete
lead: <agent name>
last_update: YYYY-MM-DD
---

## Hypothesis
(what this component proves for the paper, in 1-2 sentences)

## Setup
(data, models, hardware, hyperparameters, seeds)

## Results
(headline numbers + tables; link plots in results/runs/)

## Caveats
(seed variance, brittleness, things we tried that didn't work)

## Reproduction
(exact commands)
```

## 8. Anti-conflict workflow

For any file under shared ownership:

1. `git pull --rebase origin final`
2. Open the file. If header comment names another agent and is <2 hr old,
   ping in `docs/components/cN.md` "Status" line and back off.
3. Add a header comment with your name + start time before editing:
   `<!-- editing: agent_nlp 2026-05-03T14:30Z -->`
4. Edit. Commit with the same agent name in the message.
5. Remove the header comment in the same commit.
6. `git push`. If push fails: pull-rebase, resolve, push again. Never
   force-push `final`.

## 9. Stop conditions

Stop and write to your agent log if:

- A component number diverges from another agent's run on the same arch+seed
  by more than 2× σ_seeds. Investigate before adding more rows.
- An architecture's training crashes silently (NaN, dead-feature collapse).
- A baseline number contradicts a published paper by more than 0.05 AUC.
- You're tempted to introduce a third TXC variant. (Don't.)

## 10. Paper agent (orchestrator)

Agent PAPER is the only agent allowed to:

- Edit `docs/paper/`.
- Update `docs/components/cN.md` "Hypothesis" or "Caveats" sections without
  notifying the component lead.
- Decide cross-component questions (notation, figure style, story arc).

PAPER does not own training compute beyond the local 5090. PAPER's
day-to-day is: read leaderboard, draft sections, raise issues to component
leads via their agent dirs, integrate component writeups into the paper.
