<!--
Written by agent_filler 2026-05-06T23:30Z under direct Han override
("we can have a new agent agent_hammer to do this").
Pod: 8× RTX PRO 6000 (Blackwell-gen consumer pro card, 96 GB VRAM each).
Section ownership rules: PROTOCOL.md § 14.
-->

---
agent: agent_hammer
last_state_update: 2026-05-06T23:30:00Z
component: c2 (fill missing baselines for Setup A + Setup B)
---

## Identity + mandate (Han owns — agents do not edit)

You are **agent HAMMER**. You own the **C2 baseline backfill** —
running missing arch × seed × k_pos cells on Setup A (coupled features)
and Setup B (noisy emissions) so the C2 paper tables are complete.

Files you may edit:

- `agents/agent_hammer/briefing.md` (your own — agent-owned sections)
- Any logs you write to `logs/` or scratch outputs to `/tmp/`.
- `experiments/c1_noisy_filler/analysis.py` and `denoising_probes.py`
  PLOT_STYLE / CANONICAL_ARCH_TS lists (territory waiver from
  agent_filler — extend palette + ARCH list to include topk_sae and
  tsae_paper). DO NOT change the table-rendering or scatter-plot code.

**Files OUT OF SCOPE — do NOT edit:**
- Any other `agents/agent_*/` directory. Including agent_filler /
  agent_synth / agent_paper.
- `experiments/` driver files OTHER than the analysis tweaks above.
  agent_filler authored the baseline drivers under their territory
  (`run_baselines.py` in both `c1_noisy_filler` and `c2_synthetic_coupled`).
  Don't modify drivers; just LAUNCH them.
- `docs/components/cN.md` — agent_paper / per-component-lead territory.
- `docs/paper/*` — agent_paper.
- `configs/locked_archs.yaml` and `configs/datasources.yaml` —
  agent_paper / agent_filler.
- `src/temp_bench/architectures/` — never modify arch code.
- `pyproject.toml` / `uv.lock` — atomic, agent_paper.

If you find yourself wanting to edit an out-of-scope file, **STOP**.
Add a bullet to "Open questions for Han" in your own briefing,
surface it, let Han / agent_paper land the change. Even if Han
verbally approves, do not commit cross-territory edits yourself
unless explicitly told to override. PROTOCOL.md § 8 + CLAUDE.md
Hard Rule #7.

### ⚠️ MISSION 2026-05-06T23:30Z (URGENT, ~1 hour wall) — Baseline backfill

**The gap**:
- **Setup A** (`toy_coupled_K10_M20_d256`, c2.md gAUC table) is
  missing **`tsae_paper`** rows.
- **Setup B** (`toy_markov_n20_d40_noisy`, c2.md AUC + denoising
  block) is missing **`topk_sae`** AND **`tsae_paper`** rows.

These are the two most-cited per-token SAE baselines. Without them
the cross-arch comparison in c2.md is incomplete; reviewers will
ask why a TopK-SAE / T-SAE paper baseline isn't in the comparison.

### Fair-comparison parameters (all matching existing cells)

agent_filler authored `run_baselines.py` drivers that use the same
canonical TrainingConfig as the existing C2 cells:

| Knob | Value (matches existing cells) |
|---|---|
| n_steps              | 30,000 |
| batch_size           | 1,024 |
| optimizer            | Adam, lr=3e-4, warmup 1000 |
| precision            | bf16 |
| d_sae                | 40 (override; matches per_component_hparams.c1) |
| k_pos sweep          | {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 17, 20} (12 values) |
| seeds                | {1, 2, 42} (3 seeds) |
| **tsae_paper-specific** | `train_window_size=2` (Bhalla/Ye 2025 §3.1 paper-faithful adjacent-pair training). `contrastive_alpha=1.0` per locked YAML (paper used 0.1; this is a known minor mismatch) |
| **topk_sae-specific**  | per-token, no T axis |

Driver files (already on `final` branch, do not edit):
- `experiments/c1_noisy_filler/run_baselines.py` — Setup B (topk_sae, tsae_paper)
- `experiments/c2_synthetic_coupled/run_baselines.py` — Setup A (tsae_paper)

### Pod + parallelism

Pod: **5× RTX PRO 6000** (Blackwell-gen consumer pro, ~96 GB VRAM
each — corrected from prior briefing). Toy d_sae=40 cells use ~1-2
GB per cell, so each GPU runs 1 process. Per-cell wall: ~1-2 min
(per-token archs faster than tsae_paper).

**Cell counts**:
- Setup A (c2): tsae_paper × 12 k × 3 seeds = **36 cells**
- Setup B (c1_noisy): tsae_paper × 12 k × 3 seeds + topk_sae × 12 k × 3 seeds = **72 cells**
- **TOTAL: 108 cells**

### Sharding (5 GPUs × 9 shards = ~36 min wall)

9 (arch, seed) shards across 5 GPUs → each GPU runs ~2 shards
sequentially. Each shard is one `--arch <X> --seed <Y>` invocation
iterating all 12 k_pos values.

| GPU | Shard 1 (first) | Shard 2 (after Shard 1 finishes) | Cells |
|---|---|---|---:|
| 0 | Setup A tsae_paper seed=1 (12 cells)   | Setup B tsae_paper seed=1 (12 cells)  | 24 |
| 1 | Setup A tsae_paper seed=2 (12 cells)   | Setup B tsae_paper seed=2 (12 cells)  | 24 |
| 2 | Setup A tsae_paper seed=42 (12 cells)  | Setup B tsae_paper seed=42 (12 cells) | 24 |
| 3 | Setup B topk_sae seed=1 (12 cells)     | Setup B topk_sae seed=2 (12 cells)    | 24 |
| 4 | Setup B topk_sae seed=42 (12 cells)    | (idle / available for retry)          | 12 |

Total: 108 cells across 5 GPUs. Per-shard wall ~18 min (12 cells
× 1.5 min). Two-shard chain on GPUs 0-3 = ~36 min wall. GPU 4
finishes at ~18 min (free for retry / re-eval).

### First concrete steps

```bash
cd /workspace/temp_xc/purified
source scripts/set_agent_env.sh agent_hammer
bash scripts/agent_smoke_test.sh
git pull --rebase origin final

# Smoke ONE cell at n_steps=200 (verify driver works on RTX PRO 6000):
TQDM_DISABLE=1 AGENT_NAME=agent_hammer \
  bash scripts/run_on_gpu.sh 0 -- \
  .venv/bin/python -m experiments.c1_noisy_filler.run_baselines \
  --arch topk_sae --seed 42 --k-poses 5 --n-steps 200 --smoke 2>&1 | tail -10
```

### Phase 1 — Smoke (~5 min)

Run the smoke above. Verify the row lands in `results/leaderboard.jsonl`
with `arch=topk_sae`, `component=c1_noisy`, `eval_cfg.smoke=True`.

### Phase 2 — Full launch (~30 min)

Write `agents/agent_hammer/run_baselines_launch.sh` (your territory):

```bash
#!/usr/bin/env bash
# 5× RTX PRO 6000 baseline-backfill launcher.
# 9 (arch, seed) shards × 12 cells each = 108 cells total.
# Shards chained per-GPU via inner bash loop.
set -e
cd "$(dirname "$0")/../.."   # purified/

mkdir -p logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

# GPU 0,1,2: Setup A tsae → Setup B tsae (chain 2 shards each)
for gpu_seed in "0:1" "1:2" "2:42"; do
  IFS=":" read -r gpu seed <<< "$gpu_seed"
  log="logs/hammer_gpu${gpu}_tsae_seed${seed}.log"
  setsid -f bash scripts/run_on_gpu.sh "${gpu}" -- \
    env AGENT_NAME=agent_hammer OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TQDM_DISABLE=1 \
    bash -c "
      .venv/bin/python -m experiments.c2_synthetic_coupled.run_baselines \
        --arch tsae_paper --seed ${seed}
      .venv/bin/python -m experiments.c1_noisy_filler.run_baselines \
        --arch tsae_paper --seed ${seed}
    " < /dev/null > "${log}" 2>&1
done

# GPU 3: Setup B topk seeds 1, 2 serially
setsid -f bash scripts/run_on_gpu.sh 3 -- \
  env AGENT_NAME=agent_hammer OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TQDM_DISABLE=1 \
  bash -c '
    for seed in 1 2; do
      .venv/bin/python -m experiments.c1_noisy_filler.run_baselines \
        --arch topk_sae --seed $seed
    done
  ' < /dev/null > "logs/hammer_gpu3_topk_seeds_1_2.log" 2>&1

# GPU 4: Setup B topk seed 42 (single shard, finishes fast at ~18 min;
# can be repurposed for retries / re-eval after that).
setsid -f bash scripts/run_on_gpu.sh 4 -- \
  env AGENT_NAME=agent_hammer OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TQDM_DISABLE=1 \
  .venv/bin/python -m experiments.c1_noisy_filler.run_baselines \
    --arch topk_sae --seed 42 \
  < /dev/null > "logs/hammer_gpu4_topk_seed_42.log" 2>&1

echo "[hammer] launched 5 detached shards (9 sub-shards via chaining)"
sleep 3
pgrep -af "experiments.c[12]_.*\.run_baselines" | head
```

Save this to `agents/agent_hammer/run_baselines_launch.sh` (your
territory) and run it. Wait for completion (Monitor pgrep on the
python procs).

### Phase 3 — Re-run denoising probes for Setup B (~10 min)

The denoising_probes.py script iterates ALL c1_noisy cells in the
leaderboard, so once your topk_sae + tsae_paper Setup B cells land,
re-run it on a free GPU (after Phase 2 completes):

```bash
TQDM_DISABLE=1 AGENT_NAME=agent_hammer \
  bash scripts/run_on_gpu.sh 0 -- \
  .venv/bin/python -m experiments.c1_noisy_filler.denoising_probes \
  > logs/hammer_denoising_probes.log 2>&1
```

This computes single-latent correlation + linear probe R² for the new
topk_sae and tsae_paper checkpoints, then re-renders the scatter
plots and panel plots at:
- `experiments/c1_noisy_filler/plots/c2_noisy_singlelatent_scatter.png`
- `experiments/c1_noisy_filler/plots/c2_noisy_probe_scatter.png`
- `experiments/c1_noisy_filler/plots/c2_noisy_denoising_panels.png`

### Phase 4 — Update PLOT_STYLE for new archs (~5 min)

`experiments/c1_noisy_filler/analysis.py` and
`experiments/c1_noisy_filler/denoising_probes.py` have `PLOT_STYLE`
dicts that map (arch, t_label) → (label, color, marker). Add entries
for `topk_sae` and `tsae_paper`. Suggested:

```python
("topk_sae",   "default"): {"label": "TopK-SAE",   "color": "#000000", "ls": "-", "marker": "o"},
("tsae_paper", "default"): {"label": "T-SAE",      "color": "#0072B2", "ls": "-", "marker": "s"},
```

Add `("topk_sae", "default")` and `("tsae_paper", "default")` to the
`CANONICAL_ARCH_TS` list (top of file) so the table includes them.

After editing, re-render plots:

```bash
.venv/bin/python <<'PY'
from importlib import import_module
from pathlib import Path

# AUC plot
mod = import_module("experiments.c1_noisy_filler.analysis")
result = mod.run_analysis()

# Update c2.md AUTO-RESULTS-c1-noisy block
md_path = Path("docs/components/c2.md")
content = md_path.read_text()
begin = "<!-- BEGIN AUTO-RESULTS-c1-noisy -->"
end = "<!-- END AUTO-RESULTS-c1-noisy -->"
bi = content.find(begin); ei = content.find(end)
md_path.write_text(content[:bi+len(begin)] + "\n\n" + result.markdown.strip() + "\n\n" + content[ei:])

# Denoising plots
import json
from experiments.c1_noisy_filler.denoising_probes import (
    _aggregate_by_seeds, plot_scatter, plot_panels,
)
results = json.loads(Path("experiments/c1_noisy_filler/denoising_probe_results.json").read_text())
agg = _aggregate_by_seeds(results)
plots_dir = Path("experiments/c1_noisy_filler/plots")
plot_scatter(agg, plots_dir / "c2_noisy_singlelatent_scatter.png", mode="sl")
plot_scatter(agg, plots_dir / "c2_noisy_probe_scatter.png",         mode="lp")
plot_panels (agg, plots_dir / "c2_noisy_singlelatent_panels.png",   mode="sl")
plot_panels (agg, plots_dir / "c2_noisy_denoising_panels.png",      mode="lp")
print("All plots regenerated")
PY
```

For Setup A (c2.md AUTO-RESULTS), the analysis.py is at
`experiments/c2_synthetic_coupled/analysis.py`. Same pattern — add
tsae_paper to its CANONICAL_ARCH_TS, then re-run via:

```bash
.venv/bin/python <<'PY'
from importlib import import_module
from pathlib import Path
mod = import_module("experiments.c2_synthetic_coupled.analysis")
result = mod.run_analysis()
md_path = Path("docs/components/c2.md")
content = md_path.read_text()
begin = "<!-- BEGIN AUTO-RESULTS -->"
end = "<!-- END AUTO-RESULTS -->"
bi = content.find(begin); ei = content.find(end)
md_path.write_text(content[:bi+len(begin)] + "\n\n" + result.markdown.strip() + "\n\n" + content[ei:])
print(f"c2.md Setup A block updated: {result.results}")
PY
```

### Phase 5 — Commit + push

```bash
cd /workspace/temp_xc/purified
git add docs/components/c2.md \
        experiments/c1_noisy_filler/analysis.py \
        experiments/c1_noisy_filler/denoising_probes.py \
        experiments/c1_noisy_filler/plots/ \
        experiments/c1_noisy_filler/denoising_probe_results.json \
        experiments/c2_synthetic_coupled/analysis.py
GIT_AUTHOR_NAME="agent_hammer" GIT_AUTHOR_EMAIL="agent_hammer@noreply.local" \
GIT_COMMITTER_NAME="agent_hammer" GIT_COMMITTER_EMAIL="agent_hammer@noreply.local" \
  git commit -m "Agent HAMMER: backfill tsae_paper + topk_sae baselines for Setup A + B"
GH_TOKEN=$(cat /workspace/.tokens/gh_token)
cat > /tmp/cred.sh <<EOF
#!/bin/bash
echo "username=xuyhan"
echo "password=\$GH_TOKEN"
EOF
chmod +x /tmp/cred.sh
git -c "credential.helper=/tmp/cred.sh" push origin final
```

If the push fails because of remote changes, do:
```bash
git stash push results/leaderboard.jsonl checkpoints/manifest.jsonl
git rebase origin/final
git stash pop || git stash drop
git -c "credential.helper=/tmp/cred.sh" push origin final
```

### Watch-outs

- **HF auto-push is ON for ephemeral pods.** Every checkpoint
  auto-uploads. Don't disable.
- **The runner is idempotent.** If a (train_key, eval_key) cell
  already exists in the leaderboard, the runner will skip it. Re-running
  is safe.
- **tsae_paper at component=c2** has `d_sae=16384` from the YAML
  (no per_component override for c2). The `run_baselines.py` driver
  passes `arch_hparams_override={"d_sae": 40, "k_pos": k}` which
  supersedes — verified working. Don't try to "fix" the YAML.
- **tsae_paper contrastive_alpha=1.0** is the locked YAML default.
  Bhalla/Ye 2025 paper used 0.1 — this is a known mismatch but
  matches our locked config. Document, don't change.
- **Don't render `docs/components/c2.md`** with anything other than
  the analysis.py auto-render path. The Setup A and Setup B
  AUTO-RESULTS blocks are autogen; everything outside the markers is
  hand-written.
- **Coordination — agent_filler is on the ρ-sweep on 8× A40**;
  agent_synth is on the HUNT on 8× H100. agent_hammer's 108 cells
  are on a DIFFERENT generator + arch combination — NO conflict.

### Open questions for Han

- **tsae_paper contrastive_alpha**: locked YAML default is 1.0; paper
  used 0.1. Should we override to 0.1 for paper-faithful comparison?
  (Currently using 1.0 = YAML default; flag if Han wants 0.1.)
- **Should agent_hammer ALSO run the missing baselines on Setup C
  (ρ-sweep)?** Currently agent_filler covers ρ ∈ {0.0, 0.3, 0.6, 0.9}
  for the 3-arch headline trio (topk_sae, txc_base, txc_pro). Adding
  tsae_paper on Setup C would be 1 arch × 3 seeds × 2 k × 4 ρ = 24
  cells, ~6 min wall on a free GPU. Surface to Han.
- **Should Setup A's per_component_hparams.c2 be updated to set
  tsae_paper.d_sae=40?** That's a `configs/locked_archs.yaml` edit
  (agent_paper territory). Workaround works fine via override; flag
  if Han wants the cleaner config-side fix.

### References

- `experiments/c1_noisy_filler/run_baselines.py` — Setup B driver.
- `experiments/c2_synthetic_coupled/run_baselines.py` — Setup A driver.
- `experiments/c1_noisy_filler/analysis.py` — Setup B AUC table renderer.
- `experiments/c2_synthetic_coupled/analysis.py` — Setup A AUC tables renderer.
- `experiments/c1_noisy_filler/denoising_probes.py` — Setup B
  denoising scatter + panels.
- `docs/components/c2.md` — paper-ready writeup; AUTO-RESULTS markers
  delineate the autogen sections.
- `docs/components/c2_dmitry_comparison.md` and
  `c2_narrative_brainstorm.md` — agent_filler's analysis docs;
  helpful background but not required reading for the baseline
  backfill mission.
- `agents/agent_steer/briefing.md` — has prior tsae_paper T=2
  reference (C5 steering used train_window_size=2 too).
- `agents/agent_paper/decisions.md` § 11 — locked archs.

---

## Current state (agent owns — overwrite at every compact)

**Last verified: 2026-05-06T23:30Z** (briefing seeded by agent_filler
under Han override).

Pod: 8× RTX PRO 6000 (Blackwell-gen, 96 GB VRAM each). Status: idle,
awaiting first session.

Mission: launch 108 baseline cells (tsae_paper for Setup A + Setup B
+ topk_sae for Setup B) and update plots. ETA ~1 hour wall.

(Overwrite this section once you start work.)

## What I just did (agent owns — overwrite)

(Overwrite when you start — newest first.)

## Next action (agent owns — overwrite)

1. `cd /workspace/temp_xc/purified`
2. `source scripts/set_agent_env.sh agent_hammer` (set_agent_env.sh
   may need an entry for agent_hammer — surface as Open question if
   it doesn't exist, fall back to manual `export AGENT_NAME=agent_hammer
   CUDA_VISIBLE_DEVICES=0 TEMP_BENCH_POD_MODE=ephemeral`).
3. `bash scripts/agent_smoke_test.sh` (CRITICAL preflight — fatal on
   failure)
4. `git pull --rebase origin final`
5. Smoke ONE cell at n_steps=200 (`run_baselines.py --smoke`).
6. Write `agents/agent_hammer/run_baselines_launch.sh` (template in
   the brief above).
7. Launch 8 shards via `bash agents/agent_hammer/run_baselines_launch.sh`.
8. Monitor via `tail -f logs/hammer_*.log`. ~30 min wall.
9. After Phase 2 wraps, run `denoising_probes.py` for Setup B.
10. Update `analysis.py` PLOT_STYLE + CANONICAL_ARCH_TS for new archs.
11. Re-render plots + AUTO-RESULTS blocks in c2.md.
12. Commit + push.

## Don't repeat (agent owns — overwrite)

### Mission scope
- **Don't run any arch other than tsae_paper / topk_sae**. Other
  archs already have full coverage on Setup A + Setup B.
- **Don't run any setup other than Setup A (c2) and Setup B
  (c1_noisy)**. agent_synth is on the HUNT (different generators);
  agent_filler is on the ρ-sweep (different ρ values).
- **Don't bump n_steps above 30,000** without flagging — fair-
  comparison parameters are pinned to existing cells.
- **Don't change train_window_size**: 2 for tsae_paper (paper-faithful),
  None for topk_sae. Both are encoded in the run_baselines.py
  driver — don't bypass.

### Territory rules
- **Don't edit drivers** (`run_baselines.py`) — agent_filler
  authored. If you find a bug, surface to Han.
- **Don't edit anything in `agents/agent_*/` other than your own.**
- **Don't render `docs/components/c2.md`** by hand — only via the
  analysis.py auto-render path.
- **Don't modify `src/temp_bench/architectures/`** — never. SAE / TXC
  arch code is locked.

### Driver internals
- **Don't bypass `runner.run_cell`** — the driver does this for you.
- **Run via `bash scripts/run_on_gpu.sh <0..7> -- <cmd>`** for GPU
  pinning.
- **Use `setsid -f`** to detach long-running launches so they survive
  shell death.

## Open questions for Han (agent owns — overwrite)

- **tsae_paper contrastive_alpha**: locked YAML default is 1.0; paper
  used 0.1. Sticking with 1.0 (YAML default). Surface if Han wants
  paper-faithful 0.1.
- **agent_hammer entry in `set_agent_env.sh`**: may not exist yet.
  If `source scripts/set_agent_env.sh agent_hammer` fails, fall back
  to manual env exports and surface to Han for set_agent_env.sh fix.
- **Should agent_hammer also run baselines on Setup C (ρ-sweep)?**
  Currently agent_filler runs the headline trio (topk_sae, txc_base,
  txc_pro) for ρ ∈ {0.0, 0.3, 0.6, 0.9}. Adding tsae_paper on Setup C
  would be 24 cells, ~6 min wall. Currently NOT in scope; surface if
  Han wants.
