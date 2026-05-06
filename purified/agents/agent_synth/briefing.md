<!--
DRAFT — written by agent_paper 2026-05-06 PM. New agent identity for
the 8× 5090 synthetic-investigation pod, spawned in response to
Dmitry's negative findings on synthetic experiments.
Section ownership rules: PROTOCOL.md § 14.
-->

---
agent: agent_synth
last_state_update: 2026-05-06T19:30:00Z
component: c2 (synthetic investigation — Effect 1 vs Effect 2)
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

### ⚠️ NEW MISSION 2026-05-06 PM (URGENT) — Synthetic Investigation: Effect 1 vs Effect 2

**Han 2026-05-06**: "I spun up another 8× 5090 RunPod for agent_synth!
agent_filler with the 8× A40 is busy patching up C2! we need
agent_synth to URGENTLY INVESTIGATE THE SYNTHETIC SETUP ISSUE!"

### Background — read this before launching

Dmitry's `origin/dmitry-synthetic:docs/dmitry/results/3arch_3bench_summary.md`
shows TXC's wins on synthetic coupled-feature benches:

- **Effect 1 (sample aggregation)** — TXC's encoder pools T tokens per
  encode → variance reduction on TopK selection. Works at ANY ρ
  including ρ=0. **Doesn't require temporal correlation in the data.**
- **Effect 2 (temporal pattern detection)** — Cross-token relationships
  carry information. Requires ρ > 0; gAUC should grow with ρ.

Dmitry's data: TXC win on `coupled_noisy_overlap_sweep` is **flat across
ρ** (Effect 1 dominates). He also documented two stress benches where
TXC LOSES badly (`bench_d_separable_smoothed`, `temporal_derivative_v2`).

agent_paper's preliminary analysis of OUR C2 leaderboard (commit
`cedba0dc`) shows the **T-modulation goes the WRONG way for Effect 2**:
gAUC at k=5 drops as T_max grows — T=2 → 0.904, T=5 → 0.684, T=12 →
0.678. **More temporal context HURTS, not helps.** This is consistent
with Effect 1 winning.

The ρ-sweep is the orthogonal axis confirmation. **Plus** the DC/AC
ablation isolates the mechanism directly.

### Mission scope — 4 stages, ranked by priority

**Stage 1 — C2 ρ-sweep (REASSIGNED from agent_filler).** 72 cells,
~5-10 min wall on 8× 5090. Tests Effect 2 directly: gAUC vs ρ for
the 3-arch trio. Already commit-landed (commits `7bd38bfd` +
`213bc86d`); run script at
`experiments/c2_synthetic_coupled/run_rho_sweep.sh`. Driver +
4 new datasources (ρ ∈ {0.0, 0.3, 0.6, 0.9}) ready.

**Stage 2 — DC/AC ablation on existing C2 cells.** Eval-only,
~1-2 hr. Mechanistic test: replace each TXC feature trace with its
DC component (time-mean over T-window) or AC component (residual
after subtracting DC), then re-run gAUC. Per Dmitry's E9: TXC's wins
on coupled benches are 35-49% AC-driven and 50-65% DC-driven (so
DC dominates). Tests if our gAUC=0.99 wins are window-averaging.

**Stage 3 — Dmitry's `bench_d_separable_smoothed` analog.** New
generator: token-smoothed support (instead of Markov). Stress test
where TXC's window-pooling SHOULD lose to per-token TopK. Tests if
our TXC has the same blind spot. ~3-4 hr (incl. driver + cells).

**Stage 4 — Dmitry's `temporal_derivative_v2_sweep` analog.** New
generator: state-only x with rise (transition) target. TXC's
scalar-per-window bottleneck CAN'T extract per-token transitions →
should LOSE to TopK. Tests if our TXC has the same low-pass-filter
limitation. ~3-4 hr.

**Time budget**: stages 1+2 are mandatory (~2 hr total). Stages 3+4
are stretch — only if 1+2 leave time AND Han greenlights after seeing
the ρ-sweep results.

### First concrete task — bring up pod + run Stage 1

Step 0 — bring up the new pod:

```bash
cd /workspace/temp_xc/purified

# One-time (Han runs interactively): bootstrap_runpod.sh installs uv,
# pulls tokens, builds .venv. By the time you read this Han has done it.
ls /workspace/.tokens/hf_token   # token must exist
[ -d .venv ] || uv sync          # build venv if missing

source scripts/set_agent_env.sh agent_synth
bash scripts/agent_smoke_test.sh
```

Step 1 — verify the ρ-sweep framework is live (commits `7bd38bfd` +
`213bc86d` from agent_paper):

```bash
.venv/bin/python -c "
from temp_bench.config import load_datasource
for rho_label in ['rho00', 'rho03', 'rho06', 'rho09']:
    ds = load_datasource(f'toy_coupled_K10_M20_d256_{rho_label}')
    print(f'{rho_label}: rho={ds.rho}, M={ds.M_emissions}, K={ds.K_hidden}')
"
# Expected output:
#   rho00: rho=0.0, M=20, K=10
#   rho03: rho=0.3, M=20, K=10
#   rho06: rho=0.6, M=20, K=10
#   rho09: rho=0.9, M=20, K=10
```

Step 2 — sync the BASE caches from HF if you need them (you don't for
synthetic; this is a sanity check that HF is reachable):

```bash
# Toy synthetic data is generated in-process; no HF cache pull needed.
# But verify HF auth works for downstream auto-push of checkpoints:
.venv/bin/python -c "
from huggingface_hub import HfApi
api = HfApi(token=open('/workspace/.tokens/hf_token').read().strip())
who = api.whoami()
print(f'HF auth OK: {who[\"name\"]}')
"
```

Step 3 — smoke ONE cell at n_steps=200 to verify the pipeline:

```bash
TQDM_DISABLE=1 AGENT_NAME=agent_synth \
  .venv/bin/python -m experiments.c2_synthetic_coupled.run \
  --archs txc_pro --seeds 42 --k-poses 5 --rho-values 0.0 \
  --n-steps 200 --smoke 2>&1 | tail -10
```

Should complete in <30 sec. Verify the row lands at `component=c2`,
`arch=txc_pro`, `datasource=toy_coupled_K10_M20_d256_rho00`,
`smoke=True`.

Step 4 — launch Stage 1 (ρ-sweep) on 3 GPUs in parallel:

```bash
bash experiments/c2_synthetic_coupled/run_rho_sweep.sh
```

This launches:
- GPU 0: topk_sae × {seeds 42, 1, 2} × {k=1, 5} × {ρ=0.0, 0.3, 0.6, 0.9}
- GPU 1: txc_base × same
- GPU 2: txc_pro × same (iterates T=2, T=5, T=12 internally)

Wait time: ~5-10 min on 5090. Monitor via:

```bash
tail -f logs/c2_rho_sweep_gpu*.log
```

Step 5 — analyze + plot. Once cells land:

```bash
.venv/bin/python <<'PY'
from temp_bench.cache import _read_jsonl, leaderboard_path
from collections import defaultdict
rows = list(_read_jsonl(leaderboard_path()))

# Aggregate gAUC by (arch+T_label, k_pos, rho).
buckets = defaultdict(list)
for r in rows:
    if r.get('component') != 'c2' or r.get('eval_cfg', {}).get('smoke', False):
        continue
    cfg = r.get('eval_cfg', {})
    rho = cfg.get('rho', 0.7)   # legacy cells without rho field default to 0.7
    key = (r['arch'], cfg.get('t_label', 'default'), cfg.get('k_pos', -1), rho)
    if r['metrics'].get('gauc') is not None:
        buckets[key].append(r['metrics']['gauc'])

# Print gAUC at k=5 by (arch, T, ρ).
print(f'{"arch+T":>22} | {"ρ":>4} | gAUC mean ± std (n)')
print('-' * 60)
for (arch, t, k, rho), vals in sorted(buckets.items()):
    if k != 5: continue
    import numpy as np
    m, s = np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else 0
    print(f'{arch + " " + t:>22} | ρ={rho:.1f} | {m:.3f} ± {s:.3f} (n={len(vals)})')
PY
```

Then plot gAUC-vs-ρ (one line per arch+T) and surface to Han.
Save the plot to `experiments/c2_synthetic_coupled/plots/c2_rho_sweep.png`.

### Decision rule (per Dmitry's framework)

After Stage 1 lands:

- **gAUC roughly flat across ρ** for txc variants → **Effect 1 dominates**.
  TXC's win is sample aggregation. Paper must reframe: "TXC's window
  encoder gives variance-reduced feature recovery" rather than "TXC
  exploits temporal correlations."
- **gAUC grows with ρ** for txc variants → **Effect 2 confirmed**.
  Strong paper claim defensible.
- **Mixed** → both effects present; report both with caveats.

agent_paper makes the framing call after seeing the curve. **Surface the
plot in chat as soon as Stage 1 wraps.**

### Stage 2 — DC/AC ablation (after Stage 1)

Han hasn't blessed Stage 2 yet — wait for Stage 1 results first. If
greenlit, the design:

- For each existing trained TXC checkpoint (txc_base, txc_pro at all
  3 T values, 3 seeds), encode the data → get `(B, T_window, d_sae)`
  feature traces.
- DC component = `f.mean(dim=T)` per window (broadcast back to T).
- AC component = `f - f.mean(dim=T, keepdim=True)`.
- Replace the trained encoder's output with DC-only or AC-only at
  eval time, re-run gAUC against `hidden_features`.
- Headline: 3 numbers per (arch, seed, ρ): `gauc_full`, `gauc_dc_only`,
  `gauc_ac_only`. If `gauc_dc_only ≈ gauc_full` → DC dominates → Effect 1.
  If `gauc_ac_only ≈ gauc_full` → AC matters → Effect 2 partial.

Implementation hook: `temp_bench.eval.synthetic.feature_recovery`
takes `decoder_directions`. For DC/AC ablation, override the encode
step BEFORE feature_recovery is called. New driver in
`experiments/c2_dc_ac_ablation/run.py`. Eval-only, fast.

### Watch-outs

- **Stage 1 first; everything else gated on results.** Don't start
  Stage 2 until Han sees the ρ-curve and greenlights.
- **Don't re-run the ρ=0.7 cells** — already in leaderboard
  (existing C2 sweep). The ρ-sweep datasources (rho00, rho03, rho06,
  rho09) have their own act_cache_keys; ρ=0.7 cells stay valid.
- **Don't full-sweep k_pos** at every ρ. {1, 5} suffices for the
  ρ-curve plot. Saves ~10× compute.
- **Don't render `docs/components/c2.md`** — agent_paper's territory.
  Surface plots + tables in chat; agent_paper integrates.
- **Don't deviate from ARCH_TS** in `experiments/c2_synthetic_coupled/run.py`.
  If you need a new arch+T config, add it to ARCH_TS (territory waiver)
  but document the addition.
- **HF auto-push** is on for ephemeral pods (this pod) — every
  trained checkpoint auto-uploads. Don't disable it.
- **5090 vs A40 hardware quirks**: 5090 is 32 GB VRAM (vs A40 48 GB).
  Toy cells use ~1.8 GB per cell, so ~17 cells/GPU parallel is safe.
  No special handling vs agent_filler's pattern.

### References

- `origin/dmitry-synthetic:docs/dmitry/results/3arch_3bench_summary.md` —
  Dmitry's Effect 1 vs Effect 2 framework + 3-bench results.
- `origin/dmitry-synthetic:docs/dmitry/synthetic/2026-05-06_overnight/results.md` —
  Dmitry's overnight headline numbers (E9 DC/AC ablation results).
- `agents/agent_paper/decisions.md` § 16 (per-arch literature-faithful T).
- `experiments/c2_synthetic_coupled/run.py` — main driver +
  RHO_DATASOURCE_MAP + ARCH_TS.
- `configs/datasources.yaml` — 4 new ρ datasources at rho ∈ {0.0,
  0.3, 0.6, 0.9}; keep ρ=0.7 unchanged.
- `temp_bench.eval.synthetic.feature_recovery` — gAUC computation.

---

## Current state (agent owns — overwrite at every compact)

**Last verified: 2026-05-06T19:30Z (briefing draft).**

Pod: 8× 5090 (just spun up). Status: idle, awaiting first session.

(Overwrite this section once you start work.)

## What I just did (agent owns — overwrite)

(Overwrite when you start — newest first.)

## Next action (agent owns — overwrite)

1. `cd /workspace/temp_xc/purified`
2. `source scripts/set_agent_env.sh agent_synth`
3. `bash scripts/agent_smoke_test.sh` (CRITICAL preflight — failures
   are fatal)
4. `git pull --rebase origin final`
5. Verify ρ-sweep framework live (Step 1 above).
6. Smoke ONE cell at n_steps=200 (Step 3 above).
7. Launch Stage 1 ρ-sweep (Step 4 above).
8. After Stage 1 wraps, render the gAUC-vs-ρ plot (Step 5 above) +
   surface to Han.
9. Wait for Han's greenlight on Stage 2 before proceeding.

## Don't repeat (agent owns — overwrite)

### Mission scope
- **Don't run anything other than synthetic investigation** stages
  1-4 above. Don't pursue C3/C4/C5/C6/C7 work — other agents own those.

### Territory rules
- **Don't edit `experiments/c1_*` or `experiments/c1_noisy_filler/`** —
  agent_paper / agent_filler territories.
- **Don't edit `docs/components/cN.md`** — surface findings in chat.
- **Don't modify SAE / TXC arch code** in `src/temp_bench/architectures/`.
- **Don't bump `EVAL_PROTOCOL_VERSION` for C2** (currently "1.0.0").
  New rows just append at fresh eval_keys.

### Driver internals
- **Don't bypass `runner.run_cell`** — single canonical pathway.
- **Run via `bash scripts/run_on_gpu.sh <0..7> -- <cmd>`** for
  GPU pinning. Mirrors agent_filler's pattern.

## Open questions for Han (agent owns — overwrite)

(Surface as you encounter them. Examples that may come up:)

- Should Stage 3 / Stage 4 (Dmitry-bench reproductions) be run, or
  is Stage 1+2 enough? **Wait for Han's call after Stage 1 results.**
- If gAUC IS flat across ρ (Effect 1 dominates), should we drop the
  C2 paper claim entirely or reframe it? agent_paper's call.
- If gAUC GROWS with ρ but only weakly, what's the bar for "Effect 2
  confirmed"? Δ_gAUC ≥ 0.1 between ρ=0 and ρ=0.9? Han + agent_paper
  decide after seeing the data.
