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

**NEW MISSION 2026-05-07T00:55Z (Han override)**: agent_synth has
been doing C2 setup expansion (Setups D, E, F, G). They are at
context-compact — handing off post-compact baseline gaps to you.
**You now do the same kind of work — design + run NEW synthetic
setups + fill in baselines — in parallel with agent_synth.**

Original mission DONE (commit `b51dd774`, briefing handover
`a678771e`, HF recovery + launcher bug fix `962f9390`). All 109
of agent_hammer's c1_noisy + Setup A backfill cells are live on
HF and leaderboard.

**Read all sections below for the new mandate. The original
mandate (Setup A + Setup B baseline backfill) is COMPLETED and the
Setup A + Setup B AUTO-RESULTS blocks in `c2.md` are populated.**

Pod: **5× RTX PRO 6000** (Blackwell-gen, 96 GB VRAM each). 900 GB RAM,
160 cores. Briefing's "8× RTX PRO 6000" was wrong — 3 of the 8 GPUs
were not visible to this container.

set_agent_env.sh: agent_hammer entry already registered (commit
7f61bd5f). `CUDA_VISIBLE_DEVICES=0` per-shell baseline; per-shard
override via `bash scripts/run_on_gpu.sh <idx> --` works.

**108-cell baseline backfill DONE** (launched 22:13Z, last cell 23:02Z;
~50 min wall):
- topk_sae c1_noisy: 36/36 ✅
- tsae_paper c1_noisy: 36/36 ✅
- tsae_paper c2 (Setup A): 36/36 ✅

## NEW MISSION 2026-05-07T00:55Z — Setup expansion + baseline coverage

Han 2026-05-07T00:55Z: **agent_hammer should basically do the same
thing as agent_synth on new ideas. For each synthetic setting, we
need the full T-sweep for TXC; we also want baselines TopK,
Stacked T=2, Stacked T=5, TFA-pos, T-SAE.** Note: TFA-pos and T-SAE
are NOT present for D, E, F etc and this needs to be fixed.

agent_synth owns Setup D (noisy + overlap), Setup E (hierarchical),
Setup F (coupled + obs noise), Setup G (hier + obs noise) and the
T-sweeps on each. They're going post-compact and have tagged
baselines + new setups for follow-up.

### Your two parallel objectives

**Objective 1 (priority): Setup design** — propose + implement +
sweep new synthetic setups beyond D, E, F, G. Open candidates from
agent_synth's brainstorm:
- **Setup H**: ρ-sweep on Setup D pB05_np10 (Effect 1 vs Effect 2
  on max-overlap regime) — cleanest extension of the c2.md C/D
  axis story.
- **Setup I**: temporal-derivative target (Dmitry Bench 3 port).
  Honest-caveat bench where TXC FAILS on high-frequency targets.
- **Setup J**: hierarchical with K_l=50 (datasource already in
  YAML, not yet swept). Tests divide scaling with feature count.
- **Setup K**: anti-correlated globals — pairwise NEGATIVELY
  correlated globals. Per-token signal looks random; TXC window
  pool sees the structure.
- **Setup L**: magnitude-modulated locals — locals fire i.i.d. but
  their magnitudes are slow-modulated by globals. Pure
  temporal-pattern test (no direction in observation tracks
  globals directly).

Pick 1-3 of these (your judgement based on tractability + paper
value). Follow agent_synth's protocol per setup:
1. Generator in `src/temp_bench/data/toy/<name>.py` — reuse the
   `_orthogonalise`/`_markov_chain_batch`/`_sample_magnitudes`
   primitives from `coupled.py`.
2. ≥1 datasource entry in `configs/datasources.yaml`.
3. Driver in `experiments/c2_synthetic_coupled/run_setup_<X>.py`
   (or `c2_hierarchical/...` if hierarchical-style). Mirror
   agent_synth's `run_setup_f.py` template.
4. Launcher script with TEMP_BENCH_POD_MODE=ephemeral set EXPLICITLY
   in the env line (per your own bug fix in `962f9390`).
5. Sweep all 5 baselines + TXC-base T-sweep × 3 seeds × 3 k_pos:
     - `topk_sae` (no T, just k_pos).
     - `stacked_sae` T ∈ {2, 5}.
     - `tsae_paper` (T-SAE, T=2 paper-faithful).
     - `tfa_pos` (TFA-pos, no T axis).
     - `txc_base` T ∈ {2, 4, 5, 6, 8, 10, 12}.
6. Render plots + add a "Setup X" section to c2.md (cross-territory
   edit allowed — Han approved 2026-05-06T23:30Z for c2.md additions).
7. Commit + push.

**Objective 2: Fill baseline gaps on agent_synth's existing setups
D, E, F, G.** Inventory at 2026-05-07T00:55Z:

| Setup | topk | stk T=2 | stk T=5 | TXC T-sweep | tsae_paper | tfa_pos |
|---|---|---|---|---|---|---|
| D-np5 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| D-np10 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| E (Kg10_Kl30) | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| F σ ∈ {0.5, 1.0, 2.0} | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| G σ ∈ {1.0, 2.0} | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |

Either you OR agent_synth fills these. **Coordinate via leaderboard**
— before launching, grep for existing (datasource, arch, seed,
k_pos) cells and skip duplicates. Cell budget if you take the
whole gap fill: ~324 cells, ~30-40 min wall on 8 H100s (roughly
~50-60 min on your 5× RTX PRO 6000 because slightly fewer GPUs).

### tsae_paper hack reminder (your own past fix)

tsae_paper at component=c2 has `d_sae=16384` from the locked YAML
(no per_component_hparams.c2 override). Pass
`arch_hparams_override={"d_sae": 40, "k_pos": k}` to make it match
the d_sae=40 toy regime. Verified working in your
`run_baselines.py`; same trick applies for D/E/F/G.

### Useful agent_synth code paths

- `src/temp_bench/data/toy/`:
  - `coupled_noisy.py` — Setup D (per-token Bernoulli noise on top
    of OR-coupling).
  - `hierarchical.py` — Setup E (Kg slow + Kl fast modulated).
  - `coupled_obs_noise.py` — Setup F (Setup A + Gaussian σ).
  - `hierarchical_obs_noise.py` — Setup G (Setup E + Gaussian σ).
- `experiments/c2_synthetic_coupled/`:
  - `run_setup_f.py` is the cleanest template for new setups.
  - `plot_headline.py:_arch_label` is the central control for which
    archs appear in plots. CURRENTLY EXCLUDES tsae_paper / tfa_pos
    by returning None — when you add those baselines, **also extend
    `_arch_label` and `ARCH_COLORS`** to include them. Suggested
    style (matches agent_filler / agent_hammer convention): `tsae_
    paper` color `#CC79A7` (magenta) marker `"h"`; `tfa_pos` color
    `#2ca02c` (green) marker `"X"`.
  - `hunt_analysis.py` — pattern for per-cell gap tables.
- `docs/components/c2.md` Setup D/E/F/G sections — agent_synth wrote
  these under Han's cross-territory approval; they're paper-style
  with AUTO-RESULTS blocks. Follow the same template for Setup H/I/J.

### Hard rules (unchanged from your earlier mandate)

- TEMP_BENCH_POD_MODE=ephemeral MUST be in the env line of every
  launcher script. You discovered this bug; don't reintroduce it.
- Don't bypass `runner.run_cell`.
- Don't edit `src/temp_bench/architectures/` — locked.
- Don't bump `EVAL_PROTOCOL_VERSION` for c2 (currently "1.0.0").
- Cross-territory edit to `docs/components/c2.md` requires explicit
  Han approval each session (he gave it 2026-05-06T23:30Z; assume
  it carries forward unless he revokes).

### Coordination with agent_synth

agent_synth (post-compact) will resume on the same Han direction.
Their priority is also: fill baselines on D/E/F/G + design more
setups. **Race condition risk**: if you both launch the same
(arch, seed, k_pos) cell, the runner cache will dedupe — only
one row lands — but you waste GPU. Mitigation: pick disjoint
setups. Suggested split:
- **agent_hammer (you)**: own Setup H (ρ-sweep on D-np10) + Setup
  I (temporal-derivative). Fill F/G baselines (your existing
  tsae_paper + topk_sae work transfers cleanly).
- **agent_synth**: own Setup J/K/L (hier-flavoured new setups).
  Fill D-np5/D-np10/E baselines (their Setup D drivers).

If you start on something not in your assignment, leave a "claiming
X" line in this briefing's "Current state" section so agent_synth
sees it (they re-read both briefings at session start).

## What I just did (agent owns — overwrite)

**Read this if you (agent_filler) need to know exactly what landed and
how, especially for cross-territory understanding of c2.md and the
analysis.py edits the briefing's territory waiver permitted.**

### Phase 1 — Smoke (5 min)
Smoked 1 cell (`topk_sae c1_noisy k=5 n_steps=200 --smoke`) on GPU 0
via `bash scripts/run_on_gpu.sh 0 --` wrapper. Row landed at 22:12Z
with `agent=agent_hammer`, `eval_cfg.smoke=True`. Driver works. Smoke
row is NOT counted in the 108 — it's filtered by the `smoke=False`
predicate in `analysis.py`.

### Phase 2 — Launch (49 min wall, 5 GPUs)
Wrote `agents/agent_hammer/run_baselines_launch.sh` (committed). It
splits each (arch, seed) shard into 2 k_pos chunks:
- chunk1: `--k-poses 1 2 3 4 5 6`
- chunk2: `--k-poses 8 10 12 15 17 20`

This gives 18 sub-shards (9 × 2). Layout:

| GPU | procs | what each does |
|---|---|---|
| 0 | 4 | A_tsae_s1 chunk1+chunk2, B_tsae_s1 chunk1+chunk2 |
| 1 | 4 | A_tsae_s2 chunk1+chunk2, B_tsae_s2 chunk1+chunk2 |
| 2 | 4 | A_tsae_s42 chunk1+chunk2, B_tsae_s42 chunk1+chunk2 |
| 3 | 4 | B_topk_s1 + B_topk_s2 (chunk1+chunk2 each) |
| 4 | 2 | B_topk_s42 chunk1+chunk2 |

All 18 procs detached via `setsid -f`. Each writes to
`logs/hammer_<label>.log`. GPU util 85-93% at 4-tenant, 3 GB / 96 GB
VRAM (compute-bound, not memory-bound — could pack tighter but
diminishing returns).

`topk_sae c1_noisy` finished first (~22:27Z, freed GPUs 3,4). Then
helpers launched (Phase 2.5, see below). `tsae_paper c1_noisy` done
~22:33Z. `tsae_paper c2` was the slow path — d=256 datasource ≈ 5
min/cell at 2-tenant single-GPU; finished 23:02Z.

### Phase 2.5 — Helpers (Setup A pre-emption)
At 22:27Z when topk procs finished, wrote
`agents/agent_hammer/run_setupA_helpers.sh` (committed) and launched
6 helpers on freed GPUs 3,4 to pre-empt latter k_poses for Setup A
tsae_paper:
- GPU 3 (3-tenant): chunk1 latter halves (`--k-poses 4 5 6`) × 3 seeds
- GPU 4 (3-tenant): chunk2 latter halves (`--k-poses 15 17 20`) × 3 seeds

Outcome: helpers won the race for k=4 (chunk1) and k=15 (chunk2) for
all 3 seeds — those cells cached BEFORE existing procs reached them.
Existing procs then hit `eval_in_leaderboard` cache (runner.py:167)
and skipped to next cell. Saved ~6 cells × ~5 min = ~30 min of
GPU-time. Helpers' k=5, 6, 17, 20 cells lost the race more often
(existing was already past) — runner appended a SECOND row in some
cases, but `eval_in_leaderboard` mostly prevented it. Result: 108
unique cells in leaderboard; analysis grouping by (arch, t_label,
k_pos) is unaffected by any race-induced duplicates.

### Phase 3 — Denoising probes (10 min)
First run (22:33Z) processed only my 72 c1_noisy checkpoints (topk_sae
+ tsae_paper) — wasteland checkpoints (`tfa_pos`, `stacked_sae`,
`txc_base`, `txc_pro`) are NOT on disk on an ephemeral pod by
default. **The result OVERWROTE the JSON** to only my 2 archs!

Recovery: ran `bash scripts/sync_from_hf.sh --models-only` (22:51Z)
which pulled 545 wasteland checkpoints. Then re-ran
`denoising_probes.py` (22:54Z, log at `logs/hammer_denoising_probes_v2.log`).
Final JSON has **219/228** cells across 6 archs:
- tfa_pos: 21
- stacked_sae: 57
- txc_base: 57
- txc_pro: 12 (9 errors — see below)
- topk_sae: 36 ✅
- tsae_paper: 36 ✅

The 9 missing `txc_pro` cells errored with `RuntimeError: selected
index k out of range` (in extract_latents → model.encode for k_pos ∈
{5, 6, 8} at high T). **This is a PRE-EXISTING wasteland bug, NOT
introduced by my backfill.** Same errors occur on every
denoising_probes run regardless of leaderboard state.

**Watch-out for ephemeral pods**: if you re-run denoising_probes
without first running `sync_from_hf.sh --models-only`, the JSON gets
truncated to only the cells whose checkpoints are on local disk.
Always sync before running denoising_probes on a fresh ephemeral pod.

### Phase 4 — Render
Wrote `agents/agent_hammer/render_results.py` (committed). It is a
single-pass script that:
1. Calls `experiments.c2_synthetic_coupled.analysis.run_analysis()` →
   markdown for Setup A AUTO-RESULTS block.
2. Calls `experiments.c1_noisy_filler.analysis.run_analysis()` →
   markdown for Setup B AUTO-RESULTS-c1-noisy block (also writes
   the AUC plot).
3. Atomically rewrites both AUTO-RESULTS blocks in
   `docs/components/c2.md` (preserves all hand-written prose between
   markers — see `_replace_block` helper).
4. Re-renders 4 denoising plots from
   `denoising_probe_results.json` (scatter sl/lp + panels sl/lp).

Idempotent — safe to re-run after any leaderboard change. Use:
```
TQDM_DISABLE=1 .venv/bin/python -m agents.agent_hammer.render_results
```

### Phase 4 — PLOT_STYLE + CANONICAL_ARCH_TS edits
**Per the territory waiver in the briefing's "Files you may edit"
section, only PLOT_STYLE / CANONICAL_ARCH_TS lists were touched —
NOT the table-rendering or scatter-plot code.**

Files edited (with line refs as of `b51dd774`):

- `experiments/c1_noisy_filler/analysis.py`:
  - L29-43: added `("topk_sae", "default")` and `("tsae_paper",
    "default")` to top of `CANONICAL_ARCH_TS`.
  - L46-58: added 2 entries to `PLOT_STYLE` dict.

- `experiments/c1_noisy_filler/denoising_probes.py`:
  - L330-336: added 2 entries to `PLOT_BASE` dict.
  - L396-400: extended `_sort_key`'s order map to include
    `topk_sae: -2, tsae_paper: -1` (so they appear FIRST in scatter
    plots, before tfa_pos).

- `experiments/c2_synthetic_coupled/analysis.py`:
  - L35-43: added `("tsae_paper", "default")` after the
    `topk_sae` row in `CANONICAL_ARCH_TS`.
  - This file has no `PLOT_STYLE` dict — c2 analysis.py renders only
    tables (no plots). The plotting for Setup A is done via
    `experiments/c2_synthetic_coupled/rho_sweep_analysis.py` and
    other dedicated scripts which I did NOT touch.

Style choices (consistent across all 3 files):
- `topk_sae` default: color `#000000` (black), marker `"o"` (analysis.py)
  / `"P"` (denoising_probes.py — `o` was already taken there by stacked
  T=2). Label `"TopK-SAE"`.
- `tsae_paper` default: color `#CC79A7` (Okabe-Ito magenta), marker
  `"h"` (hexagon). Label `"T-SAE"`. Magenta chosen to be visually
  distinct from `txc_pro`'s `#1f77b4` blue (key visual comparison
  for reviewers).

### Phase 5 — Commit + push
Single commit `b51dd774` (after rebase) on `final` branch. 130 files
changed: agent territory + analysis.py edits + 109 new
`checkpoints/<train_key>/config.json` files (model.safetensors are
gitignored, configs are tracked) + leaderboard.jsonl + manifest.jsonl.

### Cross-territory edit to c2.md (post-mission, under direct Han ask)
Han asked verbally: "make the fact that gAUC / eAUC doesn't apply for
section B clear in the .md". Per Hard Rule #7 + briefing scope,
`docs/components/c2.md` is agent_paper territory; the override here
is documented for the record.

Edit (in the Setup B intro, just below the **Headline metrics** line):
added a blockquote explaining that `toy_markov_n20_d40_noisy` has only
one feature-direction set ($f_i$ is shared between noisy emission $s_i$
and clean hidden state $h_i$ at construction in `markov.py:91-123`), so
$\text{eAUC} \equiv \text{gAUC}$ would be degenerate. The local-vs-global
decomposition for Setup B lives at the latent-code level (single-latent
correlation $\bar{r}_{\text{local}}$ vs $\bar{r}_{\text{global}}$ and
linear-probe $R^2_{\text{local}}$ vs $R^2_{\text{global}}$) and is
already visualized in the existing denoising scatters
(`c2_noisy_probe_scatter.png`, `c2_noisy_singlelatent_scatter.png`).

The blockquote also forward-references the denoising scatter section
right below it, tying the explanation to the visuals readers will
encounter.

### Phase 6 — HF push recovery (post-mission, 23:25Z)
**BUG**: my launcher's `env AGENT_NAME=... TQDM_DISABLE=1` did NOT
include `TEMP_BENCH_POD_MODE=ephemeral`. This var is needed by
`cache.save_checkpoint` (cache.py:170) to trigger the auto-push to
`han1823123123/temp-bench-models`. Without it, the if-branch is
skipped, `hf_url=None` is written to manifest, and the checkpoint
stays only on local disk.

This is an EPHEMERAL POD, so without HF push the checkpoints would be
lost on pod restart. Symptom: `0/109` agent_hammer manifest entries
with `hf_url`.

**Why the bug happened**: `set_agent_env.sh` exports
`TEMP_BENCH_POD_MODE=ephemeral` in the parent shell. My launcher
ran in a subshell launched by Bash tool, which doesn't inherit
across Bash calls in this harness. The `env VAR=val cmd` line in
the launcher was the OPPORTUNITY to inject TEMP_BENCH_POD_MODE
explicitly — I missed it.

**Recovery** (verified working):
- Wrote `agents/agent_hammer/push_checkpoints_to_hf.py` (committed).
  Reads manifest, identifies agent_hammer entries, calls
  `HfApi.upload_folder()` for each train_key dir.
- Ran 23:21-23:25Z: 109/109 pushed, 0 errors, 3.9 min wall.
- Verified one sample (`138933e116af4df2`) lands on HF with both
  `config.json` and `model.safetensors`.

**Fix for future runs**: edited
`agents/agent_hammer/run_baselines_launch.sh` and
`run_setupA_helpers.sh` to add `TEMP_BENCH_POD_MODE=ephemeral` to
each `env` line.

**Note for other agents**: agent_synth's manifest entries are also
0/619 with hf_url — same pattern. They're at risk too on ephemeral
pods unless they also recover.

**Manifest hf_url is NOT updated** by the recovery — manifest is
append-only. Files are on HF (verified), but the manifest still
shows `hf_url=null`. Next session that re-runs save_checkpoint would
re-push (idempotent on HF), but no rewrite of past rows. If we want
manifest to reflect reality, would need a migration step (out of
scope for this mission).

**Rebase notes for cross-agent reference:**
- Initial commit hit conflicts on `c2.md` and `leaderboard.jsonl`
  during `git rebase origin/final`. Resolution recipe (see
  `agents/agent_hammer/render_results.py` design):
  - For `c2.md`: take origin's version (`git show :2:./<path>`
    during rebase — note "ours" is upstream when rebasing), then
    re-run `render_results.py` to inject my AUTO-RESULTS rows
    while preserving any concurrent prose edits outside markers.
  - For `leaderboard.jsonl` / `manifest.jsonl`: union via dedup
    on `(eval_key, train_key)` (lb) and `(train_key, agent, ts)`
    (manifest). Sort by `ts`. See the python one-liner in my
    "What I just did" → "Phase 5" log if you ever need to redo
    this.
- HF sync earlier modified 5 existing config.json files (changed
  `agent` field on other agents' checkpoints) and rewrote
  `checkpoints/README.md` to HF frontmatter form. **I did NOT
  commit those modifications** — they're upstream artifacts, not
  my territory. Stayed unstaged at push time.

## Next action (agent owns — overwrite)

Mission complete. If reopening this briefing in a future session:

1. **Verify the work landed**:
   ```bash
   .venv/bin/python -c "
   import json
   n = sum(1 for l in open('results/leaderboard.jsonl')
           if 'agent_hammer' in l
           and json.loads(l).get('eval_cfg', {}).get('smoke', False) is False
           and json.loads(l).get('agent') == 'agent_hammer')
   print(f'agent_hammer non-smoke cells: {n} (expected: 108)')"
   ```
2. **If Setup C ρ-sweep tsae_paper backfill is wanted** (still open
   per "Open questions for Han"): 24 cells, ~6-15 min wall.
   `experiments/c2_synthetic_coupled/run_baselines.py` already
   accepts `--rho <r>` (driver line 64-65). Loop:
   ```bash
   for rho in 0.0 0.3 0.6 0.9; do
     for seed in 1 2 42; do
       .venv/bin/python -m experiments.c2_synthetic_coupled.run_baselines \
         --arch tsae_paper --seed $seed --rho $rho --k-poses 1 5
     done
   done
   ```
3. **If denoising_probes JSON needs full 228/228 coverage**:
   investigate the 9 `txc_pro` "k out of range" errors. Likely an
   architectural mismatch in `extract_latents()` (denoising_probes.py
   L113+) for txc_pro at high k_pos × T. Pre-existing wasteland
   issue, NOT in agent_hammer territory.

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
  used 0.1. Used 1.0 (YAML default) for all 72 tsae_paper cells. Flag
  if Han wants a paper-faithful re-run at 0.1 — would need 72
  re-trains.
- ~~**agent_hammer entry in `set_agent_env.sh`**~~ — RESOLVED.
  Already registered in commit `7f61bd5f` before my session started.
- **Should agent_hammer also run baselines on Setup C (ρ-sweep)?**
  Currently agent_filler runs the headline trio (topk_sae, txc_base,
  txc_pro) for ρ ∈ {0.0, 0.3, 0.6, 0.9}. Adding tsae_paper on Setup C
  would be 24 cells, ~6-15 min wall on this 5× RTX PRO 6000 pod.
  Currently NOT in scope; surface if Han wants and I'll proceed.
- **Setup A tsae_paper d_sae=40 in YAML**: locked YAML doesn't have a
  `tsae_paper.per_component_hparams.c2` entry, so the driver passes
  `arch_hparams_override={"d_sae": 40, "k_pos": k}` to set it per
  cell. Workaround works fine; flag if Han wants the cleaner
  config-side fix in `configs/locked_archs.yaml`. (Not my territory
  to edit YAML.)
