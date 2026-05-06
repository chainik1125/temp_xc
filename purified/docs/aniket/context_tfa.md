---
author: Aniket
date: 2026-05-06
tags:
  - guide
  - in-progress
---

## Context: TFA on a sister pod (c7 backtracking, 8th cell)

This is the complete runbook for a different Claude Code instance, on a different runpod with **2× H100**, to train and evaluate the **TFA** architecture as the 8th cell of the c7 backtracking case study, render the resulting paper bundle, regenerate the 2×2 contingency / net-saves analysis, and push everything to `origin/final-aniket`.

Read this file end-to-end before running any command. Every "why" matters: this pipeline has implicit invariants (paths, eval-key derivation, judge workspaces, push-rebase rules) that are not enforced by the scripts and silently break the bundle when violated.

### Why TFA exists at all (and why it almost didn't)

On 2026-05-05 the original (Aniket's) runpod was running 7 c7 cells in parallel + chained eval loops, all pushing live to `origin/final-aniket`:

- 4 TXC cells: `txc_base|256`, `txc_base|1024`, `txc_pro|256`, `txc_pro|1024`
- 3 SAE baselines: `mlc|1024`, `tsae_paper|1024`, `topk_sae|1024`
- TFA queued *after* `txc_pro|1024` on the same GPU (`c7_tfa_after_txc_pro.sh`)

The pod then ran out of credit on **2026-05-06 ~09:11 UTC**. State at crash:

- `txc_pro|1024` had finished training (`step_300000.safetensors` saved at 09:08 UTC) and was 3 minutes into the canonical eval — **trained, just unevaluated**.
- TFA had been training ~3.5h on a separate GPU but had no usable snapshot.
- All 6 other cells had landed canonical + optimal-mag + extended-mags rows.

After the credit top-up ($60), Aniket's pod re-ran `txc_pro|1024`'s eval pipeline (canonical → optimal-mag → extended-mags → render → analyze_optimal → tex updates → push) and **dropped TFA from the chain** to stay under budget. That work is at HEAD on `extended-300k`.

Your job (sister pod, 2× H100): **train TFA from scratch and add it back as the 8th cell**, then re-render the paper bundle so all 8 cells are in the leaderboard, tables, plots, contingency analysis, and tex.

### What's already done before you start

Pulling `origin/extended-300k` gives you everything below:

- All 7 cells' canonical + optimal-mag + extended-mags rows in `purified/results/leaderboard.jsonl`.
- Per-cell run-dir artifacts (`purified/results/runs/<eval_key>/{judge_outputs.jsonl, phase1_unsteered.json, steered_phase2_optimal.jsonl, coherence_judge.jsonl, metrics.json}`) — committed via the existing `scripts/wrap_up_session.sh` pattern.
- `purified/checkpoints/<train_key>/config.json` for every cell — committed.
- The auto-loops, renderers, analyzers, and tex snippet generators — committed.
- `c7_post_canonical_chain.sh` — a one-shot post-canonical chain Aniket wrote for the txc_pro|1024 recovery; you'll adapt it for TFA.

Pulling `origin/final-aniket` (in a separate worktree) gives you:

- `purified/docs/aniket/main.tex` — paper main, with the c7 prose's `\textcolor{red}{[Live-rendered draft]}` wrappers already stripped.
- `purified/docs/aniket/appendix.tex` — c7 appendix, with the convergence narrative filled in, the steering-variant placeholder cleaned up, and a new `Optimal-magnitude rescue analysis` subsection that `\input`s the analyzer-generated tex tables.
- `purified/docs/aniket/figs/c7_*.png` + `c7_*.tex` — current-state assets generated from the 7-cell render.

What you'll add on top: the TFA leaderboard rows, regenerated figures and tables (now 8 cells), updated `c7_results_macros.tex` (best-cell macros may shift), and refreshed contingency / net-saves tables.

What you must NOT touch: the 7 existing cells' checkpoints, leaderboard rows, eval workspaces, or the locked eval protocol (cohort, ±16 magnitude grid, cut_fraction=0.25, judge prompts).

### Repo layout (after bootstrap)

| Path | Branch | Role |
|---|---|---|
| `/workspace/temp_xc/` | `extended-300k` | source; all training/eval/loop scripts here under `purified/` |
| `/workspace/temp_xc_paper/` | `final-aniket` | paper artifacts; auto-loops commit here and push to origin |

The split is hard-coded in `purified/scripts/c7_paper_loop.sh` and `synthetic_paper_loop.sh`. Don't rearrange.

### Pre-flight (HUMAN-RUN, INTERACTIVE — not callable from agent)

The bootstrap script is interactive (`read -rs` for tokens). The user must run it on the fresh pod **before** spawning you:

```bash
bash /workspace/temp_xc/purified/scripts/bootstrap_runpod.sh
```

It populates `/workspace/.tokens/{gh_token,hf_token,anthropic_key}`, configures `gh` + `huggingface-cli`, sets `HF_HOME=/workspace/hf_cache` and `UV_LINK_MODE=copy` (required on MooseFS), clones the repo, and runs `uv sync` from `purified/`. If you see "no HF token found" or similar, ask the user to run the bootstrap.

After bootstrap, the user must also create the second worktree for the paper branch:

```bash
cd /workspace/temp_xc
git worktree add /workspace/temp_xc_paper final-aniket
```

And **pull the latest** from both branches before you start (Aniket's recovery work is fresh):

```bash
cd /workspace/temp_xc                   && git pull origin extended-300k --rebase
cd /workspace/temp_xc_paper             && git pull origin final-aniket --rebase
```

### Step 1: re-add TFA to the chain scripts (revert Aniket's drop)

Aniket's recovery commit dropped TFA from 4 places. Revert them:

**1a.** `purified/scripts/c7_optimal_mag_chain.sh` — append the TFA cell to `CELLS=(...)`:

```bash
"tfa      1024 300000 42"
```

(After the `topk_sae` line; preserves the order analyze_optimal expects.)

**1b.** `purified/scripts/c7_extended_mags_chain.sh` — change the completion threshold from 7 back to 8:

```bash
if [ "$n_seen" -ge 8 ]; then
    log "all 8 cells processed — chain complete"
    break
fi
```

**1c.** `purified/scripts/c7_paper_loop.sh` — change the analyze_optimal trigger threshold from 7 back to 8 (one occurrence in `maybe_run_analyze_optimal`):

```bash
[ "$n_seen" -ge 8 ] || return 0
```

**1d.** `purified/scripts/c7_tex_snippets.py` — restore the expected-cell count in `write_results_macros`:

```python
archs_expected = set(ARCH_ORDER) - {"stacked_sae"}  # stacked_sae optional
is_complete = archs_expected.issubset(archs_present) and n_cells >= 8
lines.append(f"\\providecommand{{\\cseveennCells}}{{{n_cells}}}")
lines.append(f"\\providecommand{{\\cseveennPending}}{{{8 - n_cells if n_cells < 8 else 0}}}")
```

Commit these reverts on `extended-300k` immediately so subsequent renders pick up the right thresholds:

```bash
cd /workspace/temp_xc
git add purified/scripts/c7_optimal_mag_chain.sh \
        purified/scripts/c7_extended_mags_chain.sh \
        purified/scripts/c7_paper_loop.sh \
        purified/scripts/c7_tex_snippets.py
git commit -m "revert: re-add TFA to c7 chains + tex snippets (8th cell back in)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
git push origin extended-300k
```

### Step 2: train TFA|1024 from scratch (long, ~30h on H100)

TFA is ~2.3B params (per the bf16 cast log: `2315.3M params → 18.5 GB → fits A40`). Plenty of room on H100 80GB.

Pin to **GPU 0**. Pre-flight checks:

```bash
cd /workspace/temp_xc/purified
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader  # GPU 0 should be empty
test -f .venv/bin/python && echo "venv ok"                       # sanity-check uv sync
.venv/bin/python -c "from temp_bench.case_studies.backtracking import SonnetBacktrackingJudge; print('imports ok')"
```

Verify the activation cache is on disk (it's the same one txc_pro used; bootstrap pulls it via `sync_from_hf.sh`):

```bash
ls results/act_cache/fb2a74be884e512a/resid_post_L10.npy   # ~2 GB; pulled by sync_from_hf.sh
```

If missing, run `bash scripts/sync_from_hf.sh --data-only` first.

**Launch training** (matches the `c7_tfa_after_txc_pro.sh` invocation Aniket queued before the crash):

```bash
set -a; source /workspace/.tokens/.env 2>/dev/null || true; set +a
cd /workspace/temp_xc/purified
mkdir -p logs

CUDA_VISIBLE_DEVICES=0 \
AGENT_NAME=agent_tfa_300k \
TEMP_BENCH_POD_MODE=persistent \
TQDM_DISABLE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup .venv/bin/python -m experiments.c7_backtracking.run \
    --archs tfa --seeds 42 \
    --n-steps 300000 --batch-size 1024 \
    --probe-every 100 \
    > logs/c7_300k_tfa_bs1024.log 2>&1 &
echo $! > logs/c7_300k_tfa_bs1024.pid
echo "TFA training pid=$(cat logs/c7_300k_tfa_bs1024.pid)"
```

Notes:

- `--probe-every 100` matches every other 300K cell (writes a held-out NMSE/L0/dead row to `snapshots/eval_log.jsonl` every 100 steps; needed for the convergence figures in Appendix Fig c7-probe-{nmse,l0,dead}).
- `run.py` runs **both training AND canonical eval** in one invocation. After step 300000 it saves `model.safetensors` + `step_300000.safetensors`, then immediately runs the canonical c7 eval (61 cohort questions × 25 magnitudes = 1525 panels + ~1525 Sonnet judge calls). The canonical leaderboard row appends on success.
- Wall-clock estimate: ~30h training (matches `txc_pro|1024`'s prior run), then ~30-45 min canonical eval. Budget at H100 spot ~$2/hr ≈ $60-65 — eats most/all of a single $60 top-up. Plan accordingly.
- `AGENT_NAME` is just a tag in logs; `agent_tfa_300k` is the convention.
- `TEMP_BENCH_POD_MODE=persistent` skips the auto-HF-push of checkpoints (which is the right behavior on a 2× H100 persistent pod).

**Watch for**:

- `[c7.run] milestone snapshot saved → step_{10000,30000,100000,200000,300000}` — 5 milestones.
- `[c7.run] trainlog saved → logs/c7_b1024_tfa_seed42_trainlog.json` — training done.
- `[c7] dispatching N Sonnet judge calls` — phase2 generation done (this is when the API budget starts ticking heavily).
- `[c7] judge done: N new outputs (existing skipped)` — judge pass done.
- A new row appended to `purified/results/leaderboard.jsonl` with `arch=tfa, eval_cfg.magnitudes=[-16,-12,...,16], eval_cfg.cut_fraction=0.25` and **no** `_extended_mags` flag — that's the canonical row.

If the run crashes mid-training, the `step_*` snapshots let you resume only by re-launching the same command — `run.py` is **not** resume-aware (per `c7_b1024_*_trainlog.json` it tracks the train_log internally). A mid-train crash means restart from scratch. Mid-eval crash is fine: re-launch with the same args; phase1 cache hit + judge cache (idempotent on hash key) means you only redo the failed panels.

### Step 3: optimal-mag + extended-mags re-evals (~30 min total, 2-GPU parallel)

Once the canonical row lands, run both re-evals in parallel — one per GPU:

```bash
cd /workspace/temp_xc/purified
set -a; source /workspace/.tokens/.env 2>/dev/null || true; set +a

# GPU 0 — optimal-mag (2x2 contingency + net_saves at peak Δgc)
CUDA_VISIBLE_DEVICES=0 \
AGENT_NAME=agent_tfa_300k \
TEMP_BENCH_POD_MODE=persistent \
TQDM_DISABLE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup .venv/bin/python -m experiments.c7_backtracking.eval_optimal_mag \
    --arch tfa --bs 1024 --n-steps 300000 --seed 42 \
    > logs/c7_optimal_tfa_bs1024.log 2>&1 &

# GPU 1 — extended-mags (±24, ±32 beyond the locked grid)
CUDA_VISIBLE_DEVICES=1 \
AGENT_NAME=agent_tfa_300k \
TEMP_BENCH_POD_MODE=persistent \
TQDM_DISABLE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup .venv/bin/python -m experiments.c7_backtracking.eval_extended_mags \
    --arch tfa --bs 1024 \
    --magnitudes -32 -24 0 24 32 \
    > logs/c7_extended_mags_tfa_bs1024.log 2>&1 &

wait
echo "tfa|1024" >> logs/c7_optimal_seen.txt
echo "tfa|1024" >> logs/c7_extended_mags_seen.txt
```

`optimal-mag` (`eval_optimal_mag.py`):

- Loads the trained checkpoint (cache hit, no retrain).
- Reads `delta_gc_peak_magnitude` from the canonical leaderboard row.
- Generates at exactly `{0, peak_mag}` (122 panels), persists every steered text + token-ids + parsed answer to `<workspace>/steered_phase2_optimal.jsonl`.
- Runs **two** judge passes: backtracking COUNT (`SonnetBacktrackingJudge`) → `<workspace>/judge_outputs.jsonl`; coherence 0–3 grade (`SonnetCoherenceJudge`) → `<workspace>/coherence_judge.jsonl`.
- Cost: 122 generations + 244 judge calls ≈ ~$1-2 + ~10-15 min on H100.

`extended-mags` (`eval_extended_mags.py`):

- Re-runs the canonical eval pipeline with magnitudes `[-32, -24, 0, +24, +32]`.
- Writes a separate leaderboard row with `eval_cfg._extended_mags=true` so it doesn't clobber the canonical row.
- Cost: 305 generations + 305 judge calls ≈ ~$1-2 + ~16 min on H100.

Verify both seen-files have all 8 cells before triggering the analyzer:

```bash
wc -l logs/c7_optimal_seen.txt logs/c7_extended_mags_seen.txt   # both should be 8
```

### Step 4: render + analyze + sync — full bundle refresh

Now regenerate every paper artifact. Aniket's `c7_post_canonical_chain.sh` does steps 3-6 of this for one cell; you can adapt the relevant tail of it. Or run the steps manually:

```bash
cd /workspace/temp_xc/purified

PAPER_PURIFIED=/workspace/temp_xc_paper/purified
COMPONENTS_DIR="$PAPER_PURIFIED/docs/components"
FIGS_DIR="$PAPER_PURIFIED/docs/aniket/figs"
ASSETS_DIR="$COMPONENTS_DIR/c7_paper_assets"
ANALYZE_OUT="$COMPONENTS_DIR/c7_optimal_analysis.md"

# 4a. paper renderer — markdown + PNG plots into c7_paper_assets/
.venv/bin/python -m scripts.c7_paper_renderer --output-dir "$COMPONENTS_DIR"

# 4b. sync PNGs into figs/ with c7_ prefix (autofig macro looks here)
mkdir -p "$FIGS_DIR"
for f in "$ASSETS_DIR"/*.png; do
    [ -f "$f" ] || continue
    cp -f "$f" "$FIGS_DIR/c7_$(basename "$f")"
done

# 4c. tex snippets — refresh c7_results_macros.tex, c7_headline_table.tex, c7_pr_auc_table.tex
.venv/bin/python -m scripts.c7_tex_snippets --output-dir "$FIGS_DIR"

# 4d. analyze_optimal — refresh c7_optimal_analysis.md (markdown) +
#     c7_net_saves_table.tex + c7_contingency_table.tex (tex bodies)
.venv/bin/python -m experiments.c7_backtracking.analyze_optimal \
    --output "$ANALYZE_OUT" \
    --tex-output-dir "$FIGS_DIR"

# 4e. synthetic bundle (no change expected; c1/c2 are already 17/12 rows)
.venv/bin/python -m scripts.synthetic_paper_renderer --output-dir "$COMPONENTS_DIR" || true
```

After 4d, verify `c7_net_saves_table.tex` and `c7_contingency_table.tex` each have **8 rows** (one per cell). The appendix subsection `app:c7-optimal-mag` `\input`s these inside a `tabular` env — bad row counts will cascade-fail the latex build.

### Step 5: tex updates on `final-aniket` (the few placeholders that still need filling)

Aniket's recovery work already stripped the `\textcolor{red}{...}` wrappers from the c7 inducement + detection paragraphs in `main.tex`, filled the convergence narrative in the appendix, cleaned the steering-variant placeholder, and added the new `Optimal-magnitude rescue analysis` subsection. **Two placeholders remained**, deferred until all cells (including TFA) are in:

**5a.** `purified/docs/aniket/appendix.tex` — `\subsection{Magnitude axis convention}` (sign-convention narrative, ~line 147 of the pre-edit file). Replace the `\textcolor{red}{[PLACEHOLDER: sign-convention narrative.] ... }` block with prose that lists the per-arch raw-sign peak magnitudes from the *now-complete* leaderboard. Use the auto-generated `\cseveenTopGcPeakMag{}` for the headline cell, and reference Table `\cref{tab:c7-pr-auc}` for the full per-arch list. Keep the "TXC-base batch-size variants peak at the same sign and magnitude" claim — that's true (both at $-12$). Note where TFA falls so the reader can see the full raw-sign distribution.

**5b.** `purified/docs/aniket/appendix.tex` — `\subsection{TXC at batch size 256}` (batch-size analysis, ~line 159). Replace the `\textcolor{red}{[PLACEHOLDER: batch-size analysis.] ...}` block with prose that (i) lists the four `(TXC arch, bs)` cells' peak Δgc values + peak magnitudes from the leaderboard, (ii) lists the four PR-AUC@8 values from `c7_pr_auc_table.tex`, and (iii) says whether the pattern *is or is not* consistent with batch-size scaling explaining the architecture rankings. The headline observation Aniket queued for this paragraph: TXC-base preserves its peak magnitude across both batch sizes (both at $-12$), which is evidence that the architecture identifies a *stable* feature direction whose causal effect transfers across training scales.

Both numbers are now in `c7_headline_table.tex` (the auto-generated headline table body) and `c7_pr_auc_table.tex` — read those files directly and copy the values into prose. Don't try to interpolate via macros for these two paragraphs; the prose calls out specific cell values that don't have dedicated macros.

If you find the auto-rendered macro for the *winner* (`\cseveenTopGcCell`, `\cseveenTopDeltaCell`, `\cseveenTopPrAucCell`, `\cseveenTopRocAucCell`) has shifted to TFA on any axis, that's fine — the `main.tex` prose references the macros, so the prose updates automatically. But spot-check the resulting paragraph reads sensibly (e.g., if TFA wins detection but TXC wins inducement, the prose should still flow).

### Step 6: commit + push

The `c7_paper_renderer.py`, `c7_tex_snippets.py`, and `analyze_optimal.py` write into the `temp_xc_paper` worktree (`final-aniket`). Commit + rebase + push:

```bash
cd /workspace/temp_xc_paper
git add purified/docs/components/c7_paper_results.md \
        purified/docs/components/c7_paper_assets \
        purified/docs/components/c7_optimal_analysis.md \
        purified/docs/components/c1_paper_results.md \
        purified/docs/components/c1_paper_assets \
        purified/docs/components/c2_paper_results.md \
        purified/docs/components/c2_paper_assets \
        purified/docs/aniket/figs \
        purified/docs/aniket/main.tex \
        purified/docs/aniket/appendix.tex 2>/dev/null

# Sanity-check what's staged
git diff --cached --stat | head -30

git -c user.email="aniketdeshh@gmail.com" -c user.name="aniket" commit -m "$(cat <<'EOF'
c7 paper: TFA 8th cell landed — full re-render + tex finalisation

- Trained tfa|1024 from scratch (300K steps, seed=42) on a sister 2× H100 pod.
- Canonical + optimal-mag (±peak only) + extended-mags (±24, ±32) evals all
  landed; tfa now appears in every leaderboard table and convergence plot.
- Refreshed: c7_paper_results.md + assets, c7_optimal_analysis.md +
  c7_{net_saves,contingency}_table.tex, c7_{headline,pr_auc}_table.tex +
  c7_results_macros.tex (best-cell macros may have shifted).
- Filled: appendix sign-convention narrative + batch-size analysis from the
  now-complete 8-cell leaderboard.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"

# Pull-rebase before push (Aniket's pod has been pushing concurrently to
# final-aniket via the synthetic_paper_loop and other auto-pushes).
git pull origin final-aniket --rebase
git push origin final-aniket
```

Also commit the `extended-300k`-side artifacts (run-dir judge outputs etc.):

```bash
cd /workspace/temp_xc
bash purified/scripts/wrap_up_session.sh   # stages run-dir artifacts + leaderboard
# wrap_up_session.sh pushes to origin/final by default — IMPORTANT: that
# script's BRANCH default needs to be 'extended-300k' on this pod, not
# 'final'. Either:
#   - export BRANCH=extended-300k and rerun the script's push step manually
#   - or just do it by hand:
git pull origin extended-300k --rebase
git push origin extended-300k
```

### Two-GPU strategy (sister pod has 2× H100)

Phase | GPU 0 | GPU 1
---|---|---
Training (~30h) | TFA training | idle (or run unrelated background work)
Canonical eval (~30 min) | continues from training script, same GPU | idle
Re-evals (~30 min) | optimal-mag | extended-mags (run in parallel — see Step 3)
Render + analyze (~5 min) | analyze_optimal / renderer (CPU) | idle

You cannot meaningfully parallelise *training itself* across GPUs without DDP, which `run.py` doesn't wire up. So during the bulk of the wall-clock (training), GPU 1 is idle. If you want to use GPU 1 for something, candidates:

- A multi-seed TFA replication (`--seeds 1` and `--seeds 2`) to land additional rows for the multi-seed deferred-from-camera-ready paragraph in the appendix. **Cost**: ~30h × 2 = $120 each at H100 spot. Probably not in budget.
- A TFA at `bs=256` cell, mirroring the TXC bs=256 ablations. **Cost**: ~30h. Same caveat.

If neither, leave GPU 1 idle. **Don't fill it with unrelated speculative work** — the moment GPU 1 OOMs or hangs, it can drag GPU 0 down on a single PCIe root.

### Cross-pod coordination

Aniket's pod has been pushing to `origin/final-aniket` via `synthetic_paper_loop.sh` and `c7_paper_loop.sh` (and the wrap-up commits from the txc_pro|1024 recovery). Treat `final-aniket` as a *concurrent-write* branch:

- Always `git pull origin final-aniket --rebase` immediately before `git push`. Both `c7_paper_loop.sh` and `synthetic_paper_loop.sh` already do this, but a manual push from your steps will fail without it.
- If a push is rejected non-fast-forward, rebase locally and try again — never `--force` (it'll clobber concurrent commits).
- Aniket's pod is **not** training anymore (only auto-loops are running); it'll go quiet within an hour or two of TFA landing.

For `extended-300k`: this branch is shared with other agents' work (Han, Dmitry, etc. — see `docs/han/research_logs/phase7_unification/`). Same rebase-before-push rule. Don't `--force`.

### Common gotchas

- **`uv` interpreter goes missing on pod respawn.** Aniket hit this on the recovery pod: `purified/.venv/bin/python` symlinks to `~/.local/share/uv/python/cpython-3.12-...` which is wiped when the pod re-images. If imports fail with "No such file or directory" for the python interpreter, run `uv sync` from `purified/` to rebuild the venv.

- **Activation cache identity.** All c7 cells share `act_cache_key=fb2a74be884e512a` (Llama-3.1-8B layer-10 on the `ward_nousmirror` datasource). The `phase1_unsteered.json` is keyed by `(arch, train_key, cohort, cut_fraction)` so it's distinct per cell, but **the underlying activation cache is shared and immutable**. Do not regenerate it.

- **Judge workspace ownership.** `purified/results/runs/<eval_key>/judge_outputs.jsonl` is append-only and idempotent (judge calls keyed by hash of prompt). Do not delete this file when re-launching after a crash — you'll waste API budget re-judging the same prompts.

- **`leaderboard.jsonl` is append-only with eval_key dedup.** A second canonical eval of the same cell appends a *new* row with the same `eval_key`; the renderer takes the latest by `ts` (see `_dedup_latest` in `c7_paper_renderer.py`). No clobbering — you can re-run safely.

- **The canonical eval's *protocol* magnitude grid is `[-16, -12, -10, -8, -7, -6, -5, -4, -3, -2, -1, -0.5, 0, +0.5, +1, +2, +3, +4, +5, +6, +7, +8, +10, +12, +16]`** (25 magnitudes). Don't override this — it's the locked grid the entire paper reports against. Extended-mags is a *separate* eval with a different `eval_cfg.magnitudes` and an `_extended_mags=True` flag, so its row coexists with the canonical row.

- **`tfa` 2.3B params + bf16 → 18.5 GB.** Fits on H100 80 GB with plenty of room; you don't need to enable activation checkpointing or anything fancy. The previous `c7_300k_tfa_bs1024.log` confirmed `bf16 cast (2315.3M params → 18.5 GB → fits A40)`.

- **`run.py` finalises with `[c7] dispatching N Sonnet judge calls` then `[c7] judge done: N new outputs (existing skipped)`.** The judge dispatch is async + batched; `judge done` is the success signal. After that the leaderboard row is appended and the process exits. If you see `judge done` but no leaderboard append, check `results/leaderboard.jsonl` — the row may have appended but the process printed nothing else.

- **`analyze_optimal.py` *silently* skips cells without complete optimal-mag artifacts** (`steered_phase2_optimal.jsonl` + `judge_outputs.jsonl` + `coherence_judge.jsonl`). If your final tables show fewer than 8 rows, check `purified/results/runs/<eval_key>/` for the missing cell.

- **`InputIfFileExists` is `etoolbox`'s spelling**, not `\IfFileExists` (which is the LaTeX kernel's). The appendix subsection uses both — `\IfFileExists` for the autofig PNG fallback (kernel macro) and `\InputIfFileExists` for the table-body fallback (etoolbox). Don't unify them.

### Verification checklist (before pushing)

After Step 4 completes:

```bash
cd /workspace/temp_xc/purified

# 1. Leaderboard has 8 canonical c7 cells.
grep -c '"component": "c7"' results/leaderboard.jsonl   # should be ≥8 canonical + 8 extended + 8 optimal = 24-ish

.venv/bin/python -c "
import json
canon = []
for l in open('results/leaderboard.jsonl'):
    try: r = json.loads(l)
    except: continue
    if r.get('component') != 'c7': continue
    if r.get('eval_cfg',{}).get('_extended_mags'): continue
    if r.get('eval_cfg',{}).get('_optimal_mag'): continue
    canon.append((r['arch'], r.get('seed')))
print('canonical c7 archs:', sorted(set(canon)))
print('count:', len(set(canon)))
"
# Expect: ('mlc',42), ('topk_sae',42), ('tfa',42), ('tsae_paper',42),
#         ('txc_base',42), ('txc_pro',42)  → 6 unique archs (with bs variants);
#         total canonical row count = 8 (4 TXC × ~1.5 bs avg + 3 SAE + tfa).

# 2. tex snippets have 8 rows in the headline + pr_auc tables.
wc -l /workspace/temp_xc_paper/purified/docs/aniket/figs/c7_headline_table.tex \
       /workspace/temp_xc_paper/purified/docs/aniket/figs/c7_pr_auc_table.tex
# Expect 9 lines each (1 header comment + 8 data rows).

# 3. Net-saves + contingency tables have 8 rows.
wc -l /workspace/temp_xc_paper/purified/docs/aniket/figs/c7_net_saves_table.tex \
       /workspace/temp_xc_paper/purified/docs/aniket/figs/c7_contingency_table.tex
# Expect 9 lines each.

# 4. Convergence plots show 8 cells (visually inspect).
ls -la /workspace/temp_xc_paper/purified/docs/aniket/figs/c7_nmse_vs_step.png \
       /workspace/temp_xc_paper/purified/docs/aniket/figs/c7_l0_vs_step.png \
       /workspace/temp_xc_paper/purified/docs/aniket/figs/c7_dead_vs_step.png

# 5. Best-cell macros are populated (no [TBD]s).
grep -c '\[TBD\]' /workspace/temp_xc_paper/purified/docs/aniket/figs/c7_results_macros.tex
# Expect 0.

# 6. Latex builds clean (the user has the build env, you may not — but if
#    you do, run pdflatex twice + bibtex from purified/docs/aniket/).
```

If any check fails, fix and re-render before pushing. A bad push to `final-aniket` is recoverable but noisy (other agents are watching that branch).

### What to flag back to Aniket

- Any cell whose canonical Δgc_peak differs from the prior internal exploratory measurements by more than 20% — the appendix has a "Reference numbers from prior architectures" subsection that lists exploratory-run TXC values (`+1.574` and `+0.492`); a big discrepancy on the locked-protocol cell would be worth a paragraph.
- Whether TFA wins or loses against TXC on either axis. The headline finding ("TXC peak Δgc=+1.574, ~3× next-best arch" per inducement Sonnet judge) was established before TFA landed — TFA *not winning* is the expected outcome and is the load-bearing story. If TFA wins, that's a real result and the paper's main claim needs a re-read.
- Wall-clock + dollar cost actuals for posterity (training + eval + judge API).

### Final state, end-of-session

You should land all 8 cells fully evaluated, all paper artifacts re-rendered, both branches pushed, and `final-aniket` ready for the human paper-build pass. Don't mark the task complete until **every** verification check above passes. Then run `bash scripts/wrap_up_session.sh` once on this pod to commit any straggler artifacts (re-export `BRANCH=extended-300k`).
