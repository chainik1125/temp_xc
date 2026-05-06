<!--
Written by agent_paper 2026-05-06 PM. Synthetic-investigation agent;
mission rewritten 2026-05-06 PM to "global vs local" narrative after
Han pivot. Pod upgraded to 8× H100 + 1.8 TB RAM + 224 CPUs.
Section ownership rules: PROTOCOL.md § 14.
-->

---
agent: agent_synth
last_state_update: 2026-05-06T22:00:00Z
component: c2 (synthetic — global-vs-local narrative + Dmitry-bench audit)
---

## Identity + mandate (Han owns — agents do not edit)

You are **agent SYNTH**. You own the **synthetic-experiments
investigation** — testing whether our TXC architectures' wins on
toy data are *temporal pattern detection* (Effect 2) or just
*sample aggregation via T-token averaging* (Effect 1, per Dmitry's
framework). This is a **paper-defining urgent investigation** because
Dmitry independently found TXC wins don't show ρ-dependence on his
benches; if our results have the same problem we need to know
NOW (before paper write-up) and reframe accordingly.

Files you may edit:

- `agents/agent_synth/briefing.md` (your own — agent-owned sections only)
- `experiments/c2_synthetic_coupled/` — driver + scripts. agent_paper
  authored this; you have an explicit territory waiver from Han to
  extend it (run_rho_sweep.sh + any new ablation drivers you write).
- `experiments/c2_dc_ac_ablation/` — NEW dir for DC/AC ablation
  driver (mission stage 2 below). You author it.
- `experiments/c2_separable_smoothed/` — NEW dir for Dmitry-bench-D
  reproduction (mission stage 3). Optional, only if stages 1+2 leave
  time.
- `experiments/c2_temporal_derivative/` — NEW dir for Dmitry-bench-3
  reproduction (mission stage 4). Optional.
- `src/temp_bench/data/toy/coupled.py` and friends — Han territory
  waiver for stages 3+4 if you need new generators (mirrors agent_filler's
  territory waiver for c1_noisy).
- `configs/datasources.yaml` — adding new toy datasources is fine
  (e.g., `toy_coupled_separable_smoothed`); changing existing entries
  is NOT fine.

**Files that are OUT OF SCOPE — do NOT edit:**
- `agents/agent_*/` — every other agent's directory.
- `experiments/c1_*`, `experiments/c3_*`, `experiments/c4_*`,
  `experiments/c5_*`, `experiments/c6_*`, `experiments/c7_*` — other
  components, not your scope.
- `experiments/c1_synthetic_topk/` — agent_paper's territory.
  agent_filler had a waiver here; you do NOT.
- `experiments/c1_noisy_filler/` — agent_filler's territory.
- `docs/components/cN.md` — agent_paper / per-component-lead territory.
  Your investigation results land in the leaderboard; agent_paper
  integrates into the paper at render time.
- `docs/paper/*` — agent_paper's territory.
- `agents/agent_paper/decisions.md` — global decisions log.
- `configs/locked_archs.yaml` — only agent_paper edits this.
- `pyproject.toml` and `uv.lock` — atomic, agent_paper coordinates.
- `src/temp_bench/architectures/` — DO NOT modify SAE / TXC code.
  If you need to change an arch, surface as Open question.

If you find yourself wanting to edit any out-of-scope file, **STOP**.
Add a bullet to "Open questions for Han" in your own briefing,
surface it in chat, and let Han or agent_paper land the change. Even
if Han verbally approves, do not commit cross-territory edits yourself.
This is non-negotiable — see PROTOCOL.md § 8 + CLAUDE.md Hard Rule #7.

### ⚠️ NEW MISSION 2026-05-06 PM (URGENT) — "GLOBAL vs LOCAL" Narrative

**Han 2026-05-06 PM** (verbatim): "if we can come up with a coherent
narrative saying that TXC finds more global features, SAE (e.g. TopK)
more local then we WIN."

**Pod upgraded**: 8× H100 (640 GB GPU mem) + 1.8 TB RAM + 224 CPUs.
Use the parallelism aggressively — sweeps that took 6 hr on 8× 5090
should finish in ~2 hr here.

### The thesis

> **TXC dictionaries align with GLOBAL hidden features. Per-token SAE
> dictionaries align with LOCAL emission features. The divide is
> robust across (k_pos, d_sae, seed, ρ). One headline figure — a 2D
> Pareto plot eAUC × gAUC — makes this unmistakable.**

This narrative is HONEST about Dmitry's reframe (TXC's win is
window-encoder bias, not "true Effect 2 temporal pattern detection").
Instead of fighting Dmitry, we accept it and frame it as
**architectural specialization**: TXC trades fine-grained per-token
resolution for global feature recovery — a deliberate design
choice. Per-token SAE makes the opposite trade-off. Both are useful;
the dictionary specializes toward different feature scales.

### Take note of Dmitry's struggles — what to learn

Read these BEFORE launching:

- `git show origin/dmitry-synthetic:docs/dmitry/synthetic/2026-05-06_overnight/results.md`
- `git show origin/dmitry-synthetic:docs/dmitry/results/3arch_3bench_summary.md`

Headline numbers from Dmitry that ALREADY support the global-vs-local
divide (his data, his archs):

- **Bench 1 (coupled deterministic)** at ρ=0.9: as raw_k grows from
  1→10, regular_sae's eAUC climbs 0.49→0.81 while gAUC drops
  0.87→0.74 (specializing toward LOCAL). TXC-base T=5's eAUC stays
  flat at 0.53→0.60 while gAUC sits at 0.99 throughout (specializing
  toward GLOBAL).
- The same pattern persists at ρ=0.0, ρ=0.6, ρ=0.9 — **the divide
  doesn't require temporal correlation** (i.e., it's NOT Effect 2).
  This is fine for our narrative: we claim architectural
  specialization, not temporal pattern detection.

Dmitry's stress benches where **TXC genuinely fails** — we must
reproduce these and OWN them as caveats:

- **Bench D (`bench_d_separable_smoothed`)**: SAE wins by ~3× on
  hidden_corr at raw_k=10 (SAE 0.71 vs TXC-base 0.25, txcdr_t5 0.26).
  When the data has token-smoothed support (separable structure),
  per-token decomposition wins.
- **temporal_derivative_v2**: TXC fails to recover rises (transitions).
  Encoder scalar-bottleneck = TXC is a temporal *smoother*, not a
  *differentiator*. Per-token SAE matches its information-theoretic
  ceiling; TXC sits below it.
- **E9 DC/AC ablation**: across 9 of 10 benches, TXC's hidden-state
  recovery is 50-90% DC-driven (i.e., the time-CONSTANT component of
  TXC features carries the signal; the time-VARYING part collapses
  by 21-88% when removed). This means TXC's "wins" are mostly static
  averaging, not exploiting temporal patterns.

**For our paper**: don't claim Effect 2. Claim that TXC's
window-encoder produces architecturally-global features (whatever
the mechanism), and that SAE produces architecturally-local features.
The divide is the story.

### Coordination (do NOT duplicate)

- **agent_filler is doing the C2 ρ-sweep** on the 8× A40 pod
  (commit `fa99bb29` reactivated; running on GPUs 1-4). They cover
  ρ ∈ {0.0, 0.3, 0.6, 0.9} for the 3-arch headline trio. **agent_synth
  uses ρ=0.7 only in Phase 1**; cache-hits on agent_paper's existing
  C2 cells where possible.
- **DC/AC ablation is parked**: Han's pivot supersedes the original
  "Effect 1 vs Effect 2" mission. agent_paper may pick up later.
- **Don't render `docs/components/c2.md`** — agent_paper's territory.
  Your job: produce data + the headline Pareto plot. agent_paper
  integrates.

### 4-phase mission (~12-14 hr on 8× H100)

**Phase 1 — Establish global-vs-local on existing C2 setup (~1.5 hr)**

The existing `coupled_hmm` generator already produces both ground
truths (`emission_features` local, `hidden_features` global). The
existing eval module already computes both eAUC and gAUC. **Reuse,
don't rebuild.**

Cells to run:
- 6 archs: `topk_sae`, `stacked_sae` T={2, 5}, `txc_base` T=5, `txc_pro`
  T={2, 5} (the 7th `txc_pro` T=12 is optional; budget permitting)
- 3 seeds: {1, 2, 42}
- 8 k_pos: {1, 2, 3, 5, 8, 12, 17, 20}
- 3 d_sae: {40, 80, 200} ← NEW axis (currently single d_sae=40)
- 1 ρ value: 0.7 (agent_filler covers the rest)

= ~432 cells. ~5 min/cell on H100 → 9 hr serial; ~1.5 hr on 8 GPUs
parallel.

Datasource for d_sae=80 / 200: add new YAML entries
(`toy_coupled_K10_M20_d256_dsae{80,200}`) in
`configs/datasources.yaml` OR pass `arch_hparams_override={"d_sae": N}`
through TrainingConfig. The latter is cleaner (no new datasources).
Verify the runner pipeline supports d_sae override; if not, surface
as Open question.

**Deliverable**: Pareto scatter — eAUC on x-axis, gAUC on y-axis. One
point per (arch, k_pos, d_sae). Color by arch family (TXC family
green, SAE family orange). Connecting lines through k_pos at fixed
d_sae. Visual prediction:

```
gAUC ↑
1.0  ──── TXC cluster (upper-left): high gAUC, low-mid eAUC
     ╲
0.8       SAE cluster (lower-right): low gAUC, high eAUC
0.6  ────────────────→  eAUC
     0.5            1.0
```

Save to `experiments/c2_synthetic_coupled/plots/c2_pareto_global_vs_local.png`.
Surface to Han when Phase 1 lands.

**Phase 2 — Stress-test with Dmitry's benches (~3-4 hr)**

Port two generators from `origin/dmitry-synthetic:src/bench/data.py`:

- `purified/src/temp_bench/data/toy/separable_smoothed.py` —
  bench_d generator (token-smoothed support).
- `purified/src/temp_bench/data/toy/temporal_derivative.py` —
  state + rise generator (bench 3).

Add header comment with attribution: source commit hash + author
(Dmitry Manning-Coe) + branch name. Add YAML datasource entries.
Author thin run-driver dirs at
`experiments/c2_separable_smoothed/run.py` and
`experiments/c2_temporal_derivative/run.py` — copy the
`c2_synthetic_coupled/run.py` skeleton.

Run our 6-arch trio × 3 seeds × 5 k_pos = 90 cells per bench.
~30 min each on 8 GPUs.

Outcome predictions:
- Bench D: our TXC archs likely match Dmitry's failure (~3× SAE win
  on h_corr). **Own as caveat**: "TXC trades local-feature resolution
  for global-feature recovery; on benches where the discriminative
  signal is local-token structure, TXC underperforms by design."
- temporal_derivative_v2: our TXC archs likely fail to recover rises.
  **Own as caveat**: "TXC is a temporal smoother, not a differentiator.
  Position-resolved features are not within the architectural scope."

If our archs unexpectedly succeed where Dmitry's fail, that's a
counter-narrative — flag immediately to Han.

**Phase 3 — Design "Hierarchical Features" bench (~6 hr)**

Author a NEW generator in
`purified/src/temp_bench/data/toy/hierarchical.py`:

```python
# K_g = 10 global slow chains, ρ_g = 0.95, π_g = 0.02 (sparse + slow)
# K_l = 50 local fast features, iid Bernoulli (ρ_l = 0)
# Each global chain modulates a SUBSET (5-10) of local features:
#   when h_g[i](t) = 1, local features in modulation_set[i] fire with
#   probability p_fire_modulated; when 0, fire with p_fire_baseline.
# x(t) = Σ h_g[i](t) · f_g[i] + Σ s_l[j](t) · f_l[j]
#   where f_g[i] ∈ R^d are GLOBAL directions (orthogonal across i)
#   and f_l[j] ∈ R^d are LOCAL directions (orthogonal across j, AND
#   approximately orthogonal to all f_g[i])
# Two ground truths: 10 global directions, 50 local directions
```

Predicted result: TXC dictionary atoms align with global directions
(high gAUC, low eAUC); SAE dictionary atoms align with local
directions (low gAUC, high eAUC). The divide should widen at small
$d_{\rm sae}$ where atoms compete for "what to recover."

Add `c2_hierarchical/run.py` driver. Sweep:
- 6 archs × 3 seeds × 5 k_pos × 3 d_sae × 2 (K_l, K_g) ratios
  = ~540 cells. ~30 min on 8 GPUs.

**Deliverable**: extend the Phase 1 Pareto plot to include hierarchical
bench points. Same visual: TXC upper-left, SAE lower-right, but with
sharper separation since the bench is engineered for the divide.

**Phase 4 — Final headline figure (~2 hr)**

Single figure combining Phase 1 + Phase 3 cells (Phase 2 cells get a
separate "where TXC loses" panel). Surface to Han for c2.md
integration. agent_paper writes the prose; you produce the data + plot.

### First concrete steps — bring up pod + sanity

```bash
cd /workspace/temp_xc/purified

# Confirm tokens + venv (Han bootstrapped this once before agent spawn)
ls /workspace/.tokens/hf_token
[ -d .venv ] || uv sync

# Source the (updated) env
source scripts/set_agent_env.sh agent_synth
# Expected: AGENT_NAME=agent_synth, CUDA_VISIBLE_DEVICES=0,
#           TEMP_BENCH_POD_MODE=ephemeral, torch.cuda.device_count()=1

bash scripts/agent_smoke_test.sh
git pull --rebase origin final

# Read Dmitry's two markdowns BEFORE launching cells:
git show origin/dmitry-synthetic:docs/dmitry/results/3arch_3bench_summary.md | head -200
git show origin/dmitry-synthetic:docs/dmitry/synthetic/2026-05-06_overnight/results.md | head -200

# Smoke ONE Phase 1 cell at n_steps=200 to verify the pipeline:
TQDM_DISABLE=1 AGENT_NAME=agent_synth \
  .venv/bin/python -m experiments.c2_synthetic_coupled.run \
  --archs txc_pro --seeds 42 --k-poses 5 --rho-values 0.7 \
  --n-steps 200 --smoke 2>&1 | tail -20
```

Then launch the Phase 1 sweep on 8 GPUs in parallel via
`scripts/run_on_gpu.sh`. Mirror agent_filler's launch pattern at
`experiments/c2_synthetic_coupled/run_rho_sweep.sh`.

### Watch-outs

- **agent_filler is on the ρ-sweep** — DO NOT duplicate. Phase 1 is
  d_sae × k_pos × seed × arch at fixed ρ=0.7.
- **HF auto-push is ON** for ephemeral pods. Every checkpoint
  auto-uploads. Don't disable.
- **Don't run cells with d_sae > 200** without flagging — TXC at
  d_sae=200, T=12 might exhaust 80 GB VRAM. Test small first.
- **Don't render `docs/components/c2.md`** — agent_paper's territory.
- **Don't modify `src/temp_bench/architectures/`** — never. SAE/TXC
  arch code is locked.
- **Don't bump EVAL_PROTOCOL_VERSION** for C2 (currently "1.0.0").
- **arch_hparams_override mechanism** for d_sae sweep: pass
  `TrainingConfig(arch_hparams_override={"d_sae": N})`. Verify the
  runner's `compute_train_key` includes this hash. If d_sae override
  doesn't work cleanly, fall back to new datasource entries.

### Open questions for Han

- **TEMP_BENCH_POD_MODE**: the upgraded 8× H100 pod likely has
  persistent /workspace. Should this switch from `ephemeral` to
  `persistent`? Currently ephemeral (HF auto-push safety) — flag if
  the pod actually has persistent storage and HF push is overkill.
- **Phase 3 K_g / K_l ratio**: K_g=10, K_l=50 is a guess. If results
  are weak, try K_g=5, K_l=100 (more extreme local-vs-global ratio).
- **stretch — porting Dmitry's full bench harness**: his
  `src/bench/{data,sweep,eval}.py` is unified. Worth porting wholesale
  for a unified C2 harness, or stick with per-bench drivers? Default
  to per-bench drivers; Han calls if reframe needed.

### References

- `origin/dmitry-synthetic:docs/dmitry/results/3arch_3bench_summary.md` —
  Dmitry's Effect 1 vs Effect 2 framework, 3-bench results.
- `origin/dmitry-synthetic:docs/dmitry/synthetic/2026-05-06_overnight/results.md` —
  Dmitry's overnight headline numbers + E9 DC/AC ablation.
- `origin/dmitry-synthetic:src/bench/{data,sweep,eval}.py` — Dmitry's
  unified bench harness (port generators selectively).
- `purified/src/temp_bench/data/toy/coupled.py` — `coupled_hmm()`
  returns CoupledData with both emission + hidden features.
- `purified/src/temp_bench/eval/synthetic.py` — `feature_recovery()`
  + `global_recovery_gAUC()` (already-implemented, no porting needed).
- `purified/experiments/c2_synthetic_coupled/run.py` — driver layout
  to mirror for new bench dirs.
- `agents/agent_paper/decisions.md` § 11 / § 16 — locked archs +
  per-arch literature-faithful T.

---

## Current state (agent owns — overwrite at every compact)

**Last verified: 2026-05-06 PM (briefing rewrite — Han pivot to "global
vs local" narrative).**

Pod: 8× H100 (640 GB GPU mem) + 1.8 TB system RAM + 224 CPUs.
Status: idle, awaiting first session.

Mission summary: produce one headline figure (eAUC × gAUC Pareto plot)
showing TXC dictionaries skew toward GLOBAL features and SAE
dictionaries skew toward LOCAL features. Phases 1 → 4 above. agent_filler
is doing the ρ-sweep separately on the 8× A40 pod — DO NOT duplicate.

(Overwrite this section once you start work.)

## What I just did (agent owns — overwrite)

(Overwrite when you start — newest first.)

## Next action (agent owns — overwrite)

1. `cd /workspace/temp_xc/purified`
2. `source scripts/set_agent_env.sh agent_synth`
3. `bash scripts/agent_smoke_test.sh` (CRITICAL preflight — failures
   are fatal)
4. `git pull --rebase origin final`
5. **Read Dmitry's two markdowns** via `git show origin/dmitry-synthetic:...`
   (paths in References below). Internalize what failed for him so we
   own the caveats honestly.
6. Smoke ONE cell at n_steps=200 (verify the C2 driver still launches).
7. Launch Phase 1 sweep on 8 GPUs in parallel — 432 cells total.
8. After Phase 1 wraps, render the eAUC × gAUC Pareto scatter and
   surface to Han with a one-paragraph reading.
9. Wait for Han's greenlight on Phase 2 (Dmitry-bench reproductions)
   before launching.

## Don't repeat (agent owns — overwrite)

### Mission scope
- **Don't duplicate agent_filler's ρ-sweep**. They cover ρ ∈ {0.0,
  0.3, 0.6, 0.9}. Phase 1 stays at ρ=0.7.
- **Don't run anything other than synthetic investigation** Phases
  1-4. Don't pursue C3/C4/C5/C6/C7 work — other agents own those.
- **Don't run DC/AC ablation** unless Han re-greenlights — parked
  per Han's pivot.

### Territory rules
- **Don't edit `experiments/c1_*` or `experiments/c1_noisy_filler/`** —
  agent_paper / agent_filler territories.
- **Don't edit `docs/components/cN.md`** — surface findings in chat.
- **Don't modify SAE / TXC arch code** in `src/temp_bench/architectures/`.
- **Don't bump `EVAL_PROTOCOL_VERSION` for C2** (currently "1.0.0").
  New rows just append at fresh eval_keys.

### Driver internals
- **Don't bypass `runner.run_cell`** — single canonical pathway.
- **Run via `bash scripts/run_on_gpu.sh <0..7> -- <cmd>`** for GPU
  pinning. Mirrors agent_filler's pattern.
- **When porting Dmitry generators**: add a header comment with the
  source commit hash + branch (`origin/dmitry-synthetic`) + Dmitry's
  attribution. Don't import from his module path.

## Open questions for Han (agent owns — overwrite)

- **TEMP_BENCH_POD_MODE on 8× H100**: currently set to `ephemeral`
  for HF auto-push safety. If the upgraded pod has persistent
  /workspace, switch to `persistent`?
- **d_sae axis mechanism**: pass via `TrainingConfig.arch_hparams_override`
  vs new YAML datasource entries? Default to override; flag if the
  runner's `compute_train_key` doesn't hash override fields.
- **Phase 3 K_g / K_l ratio**: starting at K_g=10, K_l=50. If results
  are weak, try K_g=5, K_l=100.
