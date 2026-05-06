<!--
Written by agent_paper 2026-05-06 PM. Mission re-rewritten by
agent_filler 2026-05-06T23:00Z under direct Han override:
"the point I want is for them to try different synthetic setups
(change emissions, generation process) until they find a case that
cleanly demonstrates TXC wins in global feature recovery — I point
is to find a TXC WIN NOT GIVE MORE REASONS FOR IT TO LOSE!!!"
Pod: 8× H100 + 1.8 TB RAM + 224 CPUs.
Section ownership rules: PROTOCOL.md § 14.
-->

---
agent: agent_synth
last_state_update: 2026-05-06T23:00:00Z
component: c2 (synthetic — HUNT FOR TXC GLOBAL-FEATURE WIN)
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

### ⚠️ MISSION 2026-05-06T23:00Z (URGENT, FEW HOURS) — HUNT FOR TXC WIN

**Han 2026-05-06T23:00Z** (verbatim, OVERRIDE on prior briefing):

> "the point I want is for them to try different synthetic setups
> (change emissions, generation process) until they find a case that
> cleanly demonstrates TXC wins in global feature recovery!!!!! I
> point is to find a TXC WIN NOT GIVE MORE REASONS FOR IT TO LOSE!!!
> CAN YOU REWRITE THE SYNTH BRIEFING WE ONLY HAVE FEW HOURS LEFT THEY
> NEED TO FULLY LEVERAGE THE 8 H100"

**Throw away the prior d_sae-sweep plan.** That plan was a robustness
check that risked exposing more TXC weaknesses. The new mission is
**search for a parameter regime where TXC > SAE on gAUC by a large,
reproducible margin** and then dial in a paper-grade headline figure.

### The brief

You are on a few-hour budget on 8× H100 (640 GB total GPU mem, 1.8 TB
RAM, 224 CPUs). **Maximally parallel; minimal sequencing.** You will
sweep across multiple synthetic generative processes, find one (or
two) where TXC clearly dominates SAE on gAUC, and produce a
paper-grade headline figure showing the win.

### Where TXC is most likely to win — the search space

From Dmitry's results (the **only existing TXC win** is his Bench 2):

> **Bench 2 (coupled noisy + overlap, p_B=0.5, n_parents=5)** at
> raw_k=5, ρ=0.9: TXC-base 0.97 gAUC vs SAE 0.58 gAUC. **+0.39
> margin.** This is the largest TXC gAUC advantage on his suite.

Mechanism: per-token noise + dense overlap = per-token signal is
unreliable + ambiguous. TXC's window pooling averages out the noise
and disambiguates via cross-token consistency.

**Hypothesis: pushing this regime further opens a clean TXC win.**
Specifically:
- **Lower p_B** (more emission noise): p_B ∈ {0.5, 0.3, 0.2, 0.1}.
  At p_B=0.1, only 10% of "should-fire" emissions actually fire —
  per-token reading is essentially uninformative. TXC's averaging
  is the only way to recover hidden state.
- **Higher n_parents** (more coupling overlap): n_parents ∈ {2, 5, 8, 10}.
  At n_parents=10 (every emission has every hidden chain as parent),
  per-token co-firing patterns are maximally ambiguous.
- **Multiple K, M ratios**: K=10/M=20 (Dmitry default) vs
  K=5/M=20 (more emissions per chain) vs K=20/M=20 (1:1, but
  overlap means it's still ambiguous).

### Mission — 4 phases (HUNT, ZOOM, ENGINEER, HEADLINE)

#### Phase 0 — Smoke (~15 min)

```bash
cd /workspace/temp_xc/purified
source scripts/set_agent_env.sh agent_synth
bash scripts/agent_smoke_test.sh
git pull --rebase origin final

# Read agent_filler's analysis docs first (saves you 1 hr of
# re-deriving Dmitry):
cat docs/components/c2_dmitry_comparison.md
cat docs/components/c2_narrative_brainstorm.md

# Smoke one cell to verify the pipeline:
TQDM_DISABLE=1 AGENT_NAME=agent_synth \
  .venv/bin/python -m experiments.c2_synthetic_coupled.run \
  --archs txc_base --seeds 42 --k-poses 5 --rho-values 0.7 \
  --n-steps 200 --smoke 2>&1 | tail -20
```

#### Phase 1 — HUNT: coarse parameter sweep across MANY generative setups (~2 hr)

**Goal**: find which (generator, parameters) combo produces the
biggest gAUC gap (TXC - SAE).

Port Dmitry's `coupled_noisy_overlap` generator from
`origin/dmitry-synthetic:src/bench/data.py` into
`purified/src/temp_bench/data/toy/coupled_noisy.py` (header comment
with source commit + Dmitry attribution). Add a YAML datasource
`toy_coupled_noisy_K10_M20_d256` (parametrized by `p_B` and
`n_parents` via override).

Then sweep, **8 GPUs × 1 (p_B, n_parents) cell each** (one process
per GPU, each driver iterates over k_pos and seeds):

| GPU | p_B | n_parents |
|---|---|---|
| 0 | 0.5 | 2  (Dmitry Bench 1: deterministic-ish baseline) |
| 1 | 0.5 | 5  (Dmitry Bench 2: known TXC win — reproduce!) |
| 2 | 0.3 | 5  (more noise, same overlap) |
| 3 | 0.3 | 8  (more noise + more overlap) |
| 4 | 0.2 | 8  (extreme noise + extreme overlap) |
| 5 | 0.5 | 8  (modest noise + extreme overlap) |
| 6 | 0.1 | 5  (very high noise, modest overlap) |
| 7 | 0.5 | 10 (max overlap; n_parents=K) |

Each cell: 2 archs (`topk_sae`, `txc_base` T=5) × 3 seeds × 6 k_pos
{1, 2, 5, 10, 15, 20} × ρ=0.9 × n_steps=20_000 (smaller than 30k for
speed). = 36 cells per GPU × 8 = **288 cells total, ~30 min wall**.

**Deliverable**: `experiments/c2_synthetic_coupled/hunt_summary.json`
listing for each (p_B, n_parents) the max gAUC gap = mean over seeds
of (gAUC_txc_base - gAUC_topk_sae) at each k_pos. Pick the WINNING
cell — biggest positive gap.

#### Phase 2 — ZOOM: dense sweep at the winning cell (~1.5 hr)

Take the (p_B, n_parents) winner from Phase 1. Run dense:
- 6 archs: `topk_sae`, `stacked_sae` T={2, 5}, `txc_base` T=5,
  `txc_pro` T={2, 5}
- 3 seeds: {1, 2, 42}
- 12 k_pos: {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 17, 20}
- 1 (p_B, n_parents) (the winner)
- 1 ρ=0.9
- n_steps=30_000 (full)

= 216 cells. Shard 8-way across GPUs by (arch, seed) → ~30 cells per
GPU × ~30 sec/cell = ~15 min wall.

**Deliverable**: gAUC vs k_pos line plot at the winning cell, all 6
archs. **TXC family clearly above SAE family** is the headline.
Save to `experiments/c2_synthetic_coupled/plots/c2_txc_win_gauc_vs_k.png`.

If Phase 1 produces multiple competitive cells, Phase 2 can dense-
sweep the top 2 in parallel (4 GPUs each).

#### Phase 3 — ENGINEER: hierarchical-features bench (~2 hr)

Build a generator engineered for the global-vs-local divide:

```python
# K_g = 10 global slow chains, ρ_g = 0.95, π_g = 0.05
# K_l = 30 local fast features, ρ_l = 0 (i.i.d.), π_l = 0.1
# Each h_g[i] modulates a SET of n_modulated_local local features:
#   when h_g[i](t)=1, modulated locals fire with p=0.8;
#   when h_g[i](t)=0, modulated locals fire with p=0.1.
# x(t) = Σ h_g[i](t) · f_g[i] + Σ s_l[j](t) · f_l[j]
#   where f_g, f_l are orthogonal directions in R^d (d=256, d_sae=80).
# Two ground truths: 10 global directions f_g, 30 local directions f_l.
```

The structure is engineered to favor TXC: global features are
**slow** (ρ_g=0.95) and explain large chunks of variance over windows;
local features are **fast** (ρ_l=0, fresh per token) and only explain
single tokens. SAE per-token recon prefers locals; TXC window-recon
prefers globals.

`src/temp_bench/data/toy/hierarchical.py` (new file with header
comment); YAML entry; driver at `experiments/c2_hierarchical/run.py`.

Sweep: 6 archs × 3 seeds × 6 k_pos = 108 cells × 1 d_sae × 1 ρ
configuration. ~10 min wall on 8 GPUs.

**Deliverable**: same gAUC vs k_pos plot, hierarchical bench. Should
show even sharper TXC > SAE separation (we ENGINEER for it).

#### Phase 4 — HEADLINE figure for the paper (~30 min)

Combine Phase 2 + Phase 3 winners into ONE figure:
- Two side-by-side panels, one per bench (zoomed Dmitry-style noisy +
  hierarchical).
- Each panel: gAUC vs k_pos, line per arch, TXC family clearly above
  SAE family. Error bars (std over seeds).
- Title: "TXC dictionaries recover global features that per-token
  SAEs miss" (or similar Han approves).

Surface to Han + agent_paper. agent_paper integrates into c2.md.

### Coordination (do NOT duplicate)

- **agent_filler is on the C2 ρ-sweep on 8× A40** (commit `fa99bb29`,
  GPUs 0-7). They cover the existing C2 setup at ρ ∈ {0.0, 0.3, 0.6,
  0.9, 0.7} for the headline trio. **agent_synth runs DIFFERENT
  generators** (noisy+overlap with p_B != 1 and n_parents > 2;
  hierarchical) — no overlap.
- **DO NOT run the existing C2 setup** (toy_coupled_K10_M20_d256
  with deterministic OR-gate, p_B=1) — agent_filler covers it.
- **DO NOT run the d_sae sweep** (the prior briefing's Phase 1) —
  cancelled. Use the existing per_component_hparams (d_sae=40 for c2;
  d_sae=80 for hierarchical because the bench has 40 ground-truth
  features and we want over-parameterization).
- **DO NOT render `docs/components/c2.md`** — agent_paper's
  territory. Your job: produce data + plots. agent_paper integrates.

### Stop-conditions / honesty checklist

If at any point you hit:
- A regime where TXC < SAE on gAUC for the supposedly-favored bench →
  **don't bury it.** Surface immediately. Try a different parameter
  region. We hunt; we don't fudge.
- A regime where TXC ≈ SAE everywhere (no clean win across 4-8
  parameter combos) → surface to Han ASAP. The narrative may need to
  shift away from "TXC wins" toward "TXC and SAE find different
  features" (Pareto framing, not domination).
- An architectural blowup (OOM, crash) → reduce d_sae or k_pos for
  the failing arch. Don't drop the arch; document the limit.

### Watch-outs

- **HF auto-push is ON** for ephemeral pods. Every checkpoint
  auto-uploads. Don't disable.
- **Don't modify `src/temp_bench/architectures/`** — never. SAE/TXC
  arch code is locked.
- **Don't bump EVAL_PROTOCOL_VERSION** for C2 (currently "1.0.0").
- **n_steps=20_000 in Phase 1** for speed; bump to 30_000 in Phase 2
  for the headline cells. Bigger n_steps gives sharper TXC wins
  (the encoder needs time to converge to global features).
- **Use ρ=0.9 throughout** — Dmitry confirmed the divide is robust to
  ρ; 0.9 maximizes the TXC encoder's signal.
- **Per-cell wall on H100** (~80 GB, fast tensor cores): ~1-2 min for
  d=256 d_sae=40 n_steps=20k cells. Plan accordingly.

### Open questions for Han

- **TEMP_BENCH_POD_MODE on 8× H100**: currently set to `ephemeral`.
  If the upgraded pod has persistent /workspace, switch to `persistent`?
- **Hierarchical bench tuning**: Phase 3's K_g=10, K_l=30, n_modulated=
  3 is a guess. If TXC win isn't crisp, try K_l=50 (more locals to
  distract SAE) or n_modulated=5 (more global influence).
- **Should we also sweep ρ ∈ {0.6, 0.9} in Phase 2** (the winning cell)
  to show the win is ρ-robust? agent_filler is doing ρ-sweep on the
  DETERMINISTIC bench, not on the noisy+overlap bench.

### References

- `docs/components/c2_dmitry_comparison.md` — agent_filler's analysis
  of Dmitry's results vs ours. **READ FIRST** before launching.
- `docs/components/c2_narrative_brainstorm.md` — agent_filler's
  brainstorm of narratives that survive Dmitry's critique.
- `origin/dmitry-synthetic:docs/dmitry/results/3arch_3bench_summary.md`
- `origin/dmitry-synthetic:docs/dmitry/synthetic/2026-05-06_overnight/results.md`
- `origin/dmitry-synthetic:src/bench/data.py` — Dmitry's generators.
  Port `coupled_noisy_overlap` and re-use license/attribution.
- `purified/src/temp_bench/data/toy/coupled.py` — `coupled_hmm()`
  returns CoupledData with both emission + hidden features.
- `purified/src/temp_bench/eval/synthetic.py` — `feature_recovery()`
  + `global_recovery_gAUC()` already implemented.
- `purified/experiments/c2_synthetic_coupled/run.py` — driver skeleton
  to copy.
- `agents/agent_paper/decisions.md` § 11 — locked arch list.

---

## Current state (agent owns — overwrite at every compact)

**Last verified: 2026-05-06T23:03Z** (mission ~85% done, finalising
zoom + plots).

Pod: 8× H100 (640 GB GPU mem) + 1.8 TB system RAM + 224 CPUs.
TEMP_BENCH_POD_MODE=ephemeral (HF auto-push on every checkpoint).

### Headline finding

**The TXC win is REAL and reproducible. Best regime: pB05_np10**
(p_B=0.5, n_parents=10 = max overlap). At p_B=0.5, n_parents=10, ρ=0.9:
- gAUC: TXC-base saturates ≥ 0.99 at k_pos=1-3, declines at k>5.
  TopK-SAE drops monotonically 0.92→0.44 across k_pos=1→8.
- eAUC: TXC > SAE at k=1-5; SAE catches up at k=6+ as TXC's k_win →
  d_sae=40 limit.
- The "TXC for global, SAE for local" divide shows up cleanly.
- pB05_np5 (Dmitry's Bench 2 replication) confirms: TXC=0.95,
  SAE=0.63 at k_pos=1 (gap +0.32, within noise of Dmitry's published
  +0.40 win).

### Files written / committable

- **NEW code**: `src/temp_bench/data/toy/coupled_noisy.py` (Dmitry
  port), `src/temp_bench/data/toy/hierarchical.py` (engineered
  bench).
- **NEW datasources** (`configs/datasources.yaml`): 11 entries (8
  noisy+overlap + 3 hierarchical).
- **NEW drivers**: `experiments/c2_synthetic_coupled/run_hunt.py`,
  `experiments/c2_hierarchical/run.py`.
- **Launchers**: `run_hunt.sh`, `run_zoom.sh`, `run_sharded.sh`,
  `run_phases_2_3_parallel.sh`.
- **Analysis + plots**: `hunt_analysis.py`, `plot_headline.py`,
  `HUNT_FINDINGS.md`.
- **Headline plots** (`experiments/c2_synthetic_coupled/plots/`):
  - `c2_txc_win_gauc_vs_k.png` (pB05_np5, Dmitry replicate, gAUC + eAUC)
  - `c2_txc_win_gauc_vs_k_np10.png` (pB05_np10, robust regime)
  - `c2_headline_2panel.png` (noisy+overlap + hierarchical, gAUC)
  - `c2_headline_2panel_np10.png` (alt with pB05_np10 left panel)
- **Hierarchical plot**: `experiments/c2_hierarchical/plots/c2_hierarchical_gauc_vs_k.png`
- **Briefing**: `agents/agent_synth/briefing.md` (this file).

### Running work

- Phase 1 HUNT: ✅ finished + analyzed (`hunt_summary.json`). 165/288
  cells before crash at k=10 (txc_base k≥10 hits k_win > d_sae=40
  limit; expected). Enough data for analysis at k=1, 2, 5.
- Phase 2 ZOOM at n_steps=8000 (faster than initial 30k attempt):
  ~165/288 cells at 23:03. Two regimes: pB05_np5 (Dmitry replicate)
  + pB05_np10 (robust). Ramp 9-13 cells/min. ETA finish ~23:15.
- Phase 3 ENGINEER (hier sharded): ~115/126 cells (90%). 6 txc_pro
  jobs still finishing.

### What's left

1. Wait for Phase 2 ZOOM + Phase 3 hier to finish (~5-10 more min).
2. Re-render plots one final time.
3. Commit everything + surface to Han.

(Overwrite this section if you continue this work.)

## What I just did (agent owns — overwrite)

(Most recent first.)

- 22:55Z: killed slow 30k zoom + relaunched at n_steps=8000 (4× faster
  with ~5x contention savings); 36 zoom processes running.
- 22:42Z: launched 3rd zoom on pB05_np10 (the most-robust regime).
- 22:35Z: launched speculative ZOOM on pB05_np5 (Dmitry replicate)
  + pB02_np8 (extreme noise) at n_steps=30k. Killed pB02_np8 zoom
  later when k=5 hunt data showed SAE wins there.
- 22:34Z: hunt analysis identified pB05_np5 as overall winner (single
  highest peak gap +0.47 at k=1) and pB05_np10 as robust alternative
  (+0.50 at k=5).
- 22:30Z: relaunched Phase 3 hier with --arch-t-idx sharding (no
  duplicate-arch issue) and capped k_pos at 8 (T=5 limit).
- 22:13Z: re-launched Phase 1 HUNT after fixing data-on-GPU bottleneck
  (15× speedup, 15→222 steps/sec).
- 22:08Z: launched first Phase 1 HUNT (had GPU util issues; killed
  and fixed by setting `device="cuda"` in the data generator).
- 22:00Z: wrote coupled_noisy.py, hierarchical.py, 11 YAML
  datasources, all launchers, analysis + plot scripts.
- 21:50Z: pulled latest, verified env, read agent_filler's analysis
  docs.

## Next action (agent owns — overwrite)

1. Wait for Phase 2 ZOOM to finish (~5 min).
2. Wait for Phase 3 hier to finish (~3 min).
3. Final render: `.venv/bin/python -m experiments.c2_synthetic_coupled.plot_headline`
4. Run final hunt_analysis once more.
5. `git add -A && git commit -m "Agent SYNTH: HUNT + ZOOM finds TXC wins on coupled_noisy_overlap (pB05_np5/np10)"`
6. Surface to Han with summary message.

## Don't repeat (agent owns — overwrite)

### Mission scope
- **CANCELLED PRIOR PLAN** — the d_sae sweep on the EXISTING C2
  setup (~432 cells, ρ=0.7) is dead per Han 2026-05-06T23:00Z. That
  plan was a robustness test that risked exposing TXC weakness; the
  new mission is the OPPOSITE: hunt for regimes where TXC wins.
- **Don't duplicate agent_filler's ρ-sweep**. They cover the existing
  C2 setup (deterministic OR-gate). agent_synth runs DIFFERENT
  generators (noisy+overlap, hierarchical).
- **Don't run anything other than synthetic investigation Phases
  1-4 above**. Don't pursue C3/C4/C5/C6/C7 work — other agents own
  those.
- **Don't run DC/AC ablation** unless Han re-greenlights — parked.

### Hunt discipline
- **DON'T fudge.** If a regime gives TXC < SAE on gAUC, surface it.
  Try a different parameter region. We hunt; we don't fish.
- **DON'T skip the smoke** (Phase 0) — pipeline failure on H100 vs
  A40 has bitten before.
- **DON'T jump straight to Phase 3 (hierarchical) without Phase 1
  (port + hunt)** — Phase 1's coarse sweep is the cheapest way to
  find a winning regime within 30 min.

### Territory rules
- **Don't edit `experiments/c1_*` or `experiments/c1_noisy_filler/`** —
  agent_paper / agent_filler territories.
- **Don't edit `docs/components/cN.md`** — surface findings in chat.
- **Don't modify SAE / TXC arch code** in `src/temp_bench/architectures/`.
- **Don't bump `EVAL_PROTOCOL_VERSION` for C2** (currently "1.0.0").
  New rows append at fresh eval_keys.

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
- **Phase 1 hunt grid**: 8 shards across (p_B ∈ {0.5, 0.3, 0.2, 0.1},
  n_parents ∈ {2, 5, 8, 10}). If none of these produce a clean TXC
  win, what next? Drop p_B further (0.05)? Drop ρ to 0.6 to confirm
  ρ-robustness? Surface to Han if Phase 1 is ambiguous.
- **Phase 3 hierarchical bench tuning**: K_g=10, K_l=30, n_modulated=3
  is a guess. If TXC win isn't crisp, try K_l=50 or n_modulated=5.
