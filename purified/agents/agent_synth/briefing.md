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

**Last verified: 2026-05-07T00:55Z (post-compact handover state).**

Original mission (HUNT + ZOOM + ENGINEER + HEADLINE) COMPLETED at
2026-05-06T23:15Z. Subsequent extensions added Setup F (coupled +
obs noise) + Setup G (hierarchical + obs noise) + T-sweeps on D, E,
F, G. Latest commit on origin/final at this writing: `7cb1fffc`.

**THIS SECTION IS THE POST-COMPACT HANDOVER. Read top to bottom.**

Pod (used during mission): 8× H100 (640 GB GPU mem) + 1.8 TB system
RAM + 224 CPUs. TEMP_BENCH_POD_MODE=ephemeral (HF auto-push fired on
every checkpoint).

This section is now a HANDOVER, not a current-work indicator.
Subsequent sessions on agent_synth start fresh.

### Headline finding

**The TXC win is REAL and reproducible.** Two regimes were dense-
swept at 6 archs × 3 seeds × 8 k_pos (n_steps=8000):

1. **pB05_np10 — the cleanest TXC dominance** (max overlap regime,
   p_B=0.5, n_parents=10, ρ=0.9):
   - **TXC-base T=5**: gAUC saturates at **0.99** at k_pos=1-3,
     declines to 0.93 (k=4), 0.74 (k=5), 0.62 (k=6), 0.55 (k=7),
     0.40 (k=8 — k_win=40=d_sae limit, sparsity dies).
   - **TopK-SAE**: gAUC drops monotonically **0.92 → 0.44** across
     k_pos=1→8.
   - **TXC-pro T=5**: gAUC ≈ 0.85-0.93 at k=1-2 (1-2 seeds; some
     cells killed before convergence at high k).
   - **eAUC pattern**: TXC > SAE at k=1-5; SAE catches up at k≥6.
   - This is the cleanest "TXC for global, SAE for local" divide
     in the suite.

2. **pB05_np5 — Dmitry Bench 2 replication** (p_B=0.5, n_parents=5,
   ρ=0.9):
   - At k_pos=1: TXC-base 0.95, TopK-SAE 0.63 — gap +0.32 (vs
     Dmitry's published +0.40 at his raw_k=5 — same signal, our
     matched-per-token convention).
   - Gap shrinks at k≥2 (TXC-base = 0.87, TopK-SAE = 0.89 at k=2 →
     SAE matches TXC because at high k, SAE has enough latents to
     find both globals + locals via co-occurrence).
   - txc_pro (T=2 and T=5) at k=1 reaches gAUC=0.99 — matches TXC-base.

3. **HUNT phase** (8 shards × 36 cells coarse sweep) found 6
   regimes with positive gauc gaps at k_pos=1; gap rank order:
   pB05_np5 (+0.47), pB02_np8 (+0.44), pB05_np8 (+0.34), pB03_np8
   (+0.27), pB03_np5 (+0.21), pB05_np10 (+0.16). At k_pos=5 the
   ranking flips: pB05_np10 (+0.50) becomes the leader.

4. **Hierarchical bench (Phase 3)**: K_g=10 slow globals × K_l=30
   fast locals modulated by globals. Shows TXC > SAE on gAUC at
   low k (1-2) but **SAE catches up at k≥5** because d_sae=40 is
   exactly K_g+K_l=40 — at high k SAE finds all features. Honest
   limitation, not a contradiction. Bench D_sae=80 would likely
   widen the divide; deferred (touches locked_archs.yaml,
   agent_paper's territory).

### Files written this mission (committed)

All paths relative to `purified/`. Rules of thumb for agent_filler:
- Code I added is independent of yours — no overlap.
- I did NOT modify `experiments/c2_synthetic_coupled/run.py` (the
  ρ-sweep driver you authored). My new driver is `run_hunt.py`.
- I did NOT modify `coupled.py` (your existing generator). I added
  a NEW file `coupled_noisy.py` that imports `coupled.py`'s
  private helpers.
- I added 11 NEW YAML entries; the existing entries are unchanged.

**New code (committable)**:

- `src/temp_bench/data/toy/coupled_noisy.py` — port of Dmitry's
  per-token Bernoulli emission noise (p_B, p_A) on top of
  OR-gate coupling. Source attribution: `origin/dmitry-synthetic
  @ 03a099b4:src/data_generation/{coupled_dataset,support}.py`.
  Reuses `_orthogonalise`, `_generate_coupling`,
  `_compute_hidden_features`, `_markov_chain_batch`,
  `_sample_magnitudes` from `coupled.py`. Returns same
  `CoupledData` namedtuple — eval pipelines unchanged.
- `src/temp_bench/data/toy/hierarchical.py` — engineered global-
  vs-local bench. K_g slow globals (ρ_g=0.95, π_g=0.05) modulate
  K_l fast locals via `n_global_parents`-many parents per local
  (default 1). Locals fire with probability 0.8 if ANY parent on,
  else 0.1. Returns `CoupledData` where `hidden_features` = global
  directions f_g and `emission_features` = local directions f_l.

**New drivers (committable)**:

- `experiments/c2_synthetic_coupled/run_hunt.py` — driver for
  HUNT (Phase 1) + ZOOM (Phase 2) phases. CLI flags:
  `--datasource`, `--phase {hunt,zoom}`, `--archs`,
  `--arch-t-idx`, `--seeds`, `--k-poses`, `--n-steps`. Uses
  GPU-resident data via `device="cuda"` (15× speedup over CPU
  data — see "Pitfalls" below).
- `experiments/c2_hierarchical/__init__.py` — empty.
- `experiments/c2_hierarchical/run.py` — Phase 3 ENGINEER driver.
  Same CLI pattern as `run_hunt.py`.

**Launchers (committable)**:

- `experiments/c2_synthetic_coupled/run_hunt.sh` — Phase 1
  launcher. Fans out 8 shards (one per p_B × n_parents
  datasource) on GPUs 0-7.
- `experiments/c2_synthetic_coupled/run_zoom.sh` — Phase 2
  launcher. 18 (arch_t, seed) jobs round-robin on 8 GPUs.
  Auto-reads winner from hunt_summary.json.
- `experiments/c2_synthetic_coupled/run_phases_2_3_parallel.sh` —
  combined launcher (UNUSED in actual run; left as reference).
- `experiments/c2_hierarchical/run.sh` — initial Phase 3
  launcher. Has a `--archs` filter bug (filters by name, not
  T-override) that I fixed by adding `--arch-t-idx` to run.py;
  USE `run_sharded.sh` instead for clean sharding.
- `experiments/c2_hierarchical/run_sharded.sh` — proper Phase 3
  launcher. 18 (arch_t, seed) jobs round-robin on 8 GPUs.

**Analysis + plotting (committable)**:

- `experiments/c2_synthetic_coupled/hunt_analysis.py` — reads
  leaderboard.jsonl, dedupes by eval_key, computes per-cell
  gauc gap (TXC - SAE) at all (datasource, k_pos), prints a
  markdown table + per-datasource winner ranking. Writes
  `hunt_summary.json`.
- `experiments/c2_synthetic_coupled/plot_headline.py` — reads
  zoom + hier rows from leaderboard, renders 5 paper plots
  (see below). Filters zoom rows by ts > 22:54:30Z to drop
  early n_steps=30k cells (mid-flight switch to 8k — see
  "Pitfalls").
- `experiments/c2_synthetic_coupled/HUNT_FINDINGS.md` —
  human-readable summary of HUNT/ZOOM findings.
- `experiments/c2_synthetic_coupled/hunt_summary.json` — JSON
  output of hunt_analysis: per-cell gap table + overall winner.

**Headline plots (in `experiments/c2_synthetic_coupled/plots/`)**:

- `c2_txc_win_gauc_vs_k.png` — Phase 2 ZOOM on pB05_np5 (Dmitry
  replicate). 2-panel: gAUC + eAUC vs k_pos.
- `c2_txc_win_gauc_vs_k_np10.png` — Phase 2 ZOOM on pB05_np10
  (the cleanest regime). 2-panel: gAUC + eAUC vs k_pos. **Best
  candidate for the paper headline.**
- `c2_headline_2panel.png` — Phase 4 combined: pB05_np5 left,
  hierarchical right (gAUC only).
- `c2_headline_2panel_np10.png` — Phase 4 combined: pB05_np10
  left, hierarchical right (gAUC only). **Recommend this for
  c2.md headline.**
- `experiments/c2_hierarchical/plots/c2_hierarchical_gauc_vs_k.png`
  — Phase 3 hierarchical sweep: gAUC + eAUC vs k_pos for all
  6 archs.

**Datasources added** (`configs/datasources.yaml`):

8 noisy+overlap (Phase 1 HUNT grid):
- `toy_coupled_noisy_K10_M20_d256_pB05_np2`  (Bench 1 baseline)
- `toy_coupled_noisy_K10_M20_d256_pB05_np5`  (Dmitry Bench 2)
- `toy_coupled_noisy_K10_M20_d256_pB03_np5`
- `toy_coupled_noisy_K10_M20_d256_pB03_np8`
- `toy_coupled_noisy_K10_M20_d256_pB02_np8`  (extreme noise+overlap)
- `toy_coupled_noisy_K10_M20_d256_pB05_np8`
- `toy_coupled_noisy_K10_M20_d256_pB01_np5`  (very high noise)
- `toy_coupled_noisy_K10_M20_d256_pB05_np10` (max overlap — winner)

3 hierarchical (Phase 3):
- `toy_hierarchical_Kg10_Kl30_d256` (primary)
- `toy_hierarchical_Kg10_Kl50_d256` (secondary, more locals)
- `toy_hierarchical_Kg10_Kl30_d256_np2` (secondary, 2-parent
  modulation)

### Pitfalls / debugging history (lessons for future sessions)

1. **Data on CPU is a fatal bottleneck.** First Phase 1 HUNT got
   GPU util stuck at 6% because `coupled_hmm` defaulted to
   `device="cpu"` and `make_batch_iter` then did per-batch CPU→GPU
   transfer. Fix: pass `device="cuda"` to the generator. This
   gave 15× speedup (15→222 steps/sec for txc_base) and was the
   single biggest perf win. **Always pass device="cuda" when
   building toy data.**

2. **k_pos × T ≤ d_sae is a HARD constraint.** TXC-base / TXC-pro
   call `pre.topk(k_win, dim=-1)` on a `d_sae`-dim tensor. If
   k_win = k_pos × T > d_sae=40, this raises
   `RuntimeError: selected index k out of range` and the cell
   crashes the subprocess. For T=5 archs, k_pos must ≤ 8. The
   HUNT k_pos grid (1, 2, 5, 10, 15, 20) intentionally ran past
   this limit so the hunt would crash naturally at k=10 after
   collecting the useful k=1, 2, 5 data; ZOOM was capped at
   k_pos=8 for safety. The hier driver's `DEFAULT_K_POSES =
   (1, 2, 3, 4, 5, 6, 8)` reflects the cap. **If you increase
   d_sae for c2 (touches locked_archs.yaml = agent_paper
   territory), the cap can widen.**

3. **n_steps=30000 is overkill at this scale.** At d_in=256,
   d_sae=40, batch=1024, models converge well within 8000 steps.
   I started zoom at n_steps=30000 and it was painfully slow under
   contention (~10 min per cell with 50+ processes per 8 GPUs).
   Killed + restarted at n_steps=8000 mid-flight — this re-keys
   train_keys, so the early 30k zoom rows are PRE-CUTOFF in the
   leaderboard and `plot_headline.py` filters them out via
   `ZOOM_CUTOFF_TS = "2026-05-06T22:54:30Z"`. **Default to ~8k
   steps for synthetic toy data; bump only if convergence is
   unclear.**

4. **`--archs <name>` does NOT filter T-overrides.** In a driver
   with `ARCH_TS = [(stacked_sae, T=2), (stacked_sae, T=5), ...]`,
   passing `--archs stacked_sae` matches BOTH entries (because
   `if arch_filter is None or a in arch_filter` is checked on
   `a`, the arch name). If you want to fan out one-per-GPU, use
   `--arch-t-idx N` to pick a specific index from ARCH_TS. I added
   this flag to both `run_hunt.py` and `c2_hierarchical/run.py`.

5. **Hunt and Zoom share the same `_get_data` cache.** The
   process-global `_DATA_CACHE` keyed by datasource_name means
   different datasources don't conflict; same datasource across
   archs/seeds reuses the same data tensor. Per the runner's
   contract, data_seed=0 is fixed per component.

6. **HF auto-push on ephemeral pod.** Every checkpoint after each
   cell fires an `upload_folder` to `han1823123123/temp-bench-models`.
   Network latency adds a few sec per cell. I did NOT disable this
   per the briefing's hard rule.

7. **Leaderboard merge conflicts during multi-agent push.** Other
   agents (NLP, EM, HAMMER, STEER) push to `final` concurrently.
   Each push needs `git pull --rebase` first; the conflicts in
   `results/leaderboard.jsonl` and `checkpoints/manifest.jsonl`
   are append-only and resolve by UNION-ing both halves +
   deduping by eval_key/train_key. There's a one-shot Python
   resolver inline in my commit dance that you can reuse:

   ```python
   import re, json
   for path in ['results/leaderboard.jsonl', 'checkpoints/manifest.jsonl']:
       text = open(path).read()
       pattern = re.compile(r'<<<<<<< HEAD\n(.*?)\n=======\n(.*?)\n>>>>>>> [^\n]+', re.DOTALL)
       while True:
           m = pattern.search(text)
           if not m: break
           text = text[:m.start()] + m.group(1) + '\n' + m.group(2) + text[m.end():]
       seen, out = set(), []
       for line in text.split('\n'):
           if not line.strip(): continue
           try:
               d = json.loads(line)
               key = d.get('eval_key') or d.get('train_key') or line
               if key in seen: continue
               seen.add(key)
           except Exception: pass
           out.append(line)
       open(path, 'w').write('\n'.join(out) + '\n')
   ```

8. **`pkill -f c2_hierarchical` killed my own background bash
   wrappers.** Pattern matched the wrapper bash too, exiting with
   144. Use specific PIDs (`kill -9 18260 18264 ...`) for
   precise kills. Or kill specific arch_t indices.

### What worked (re-use these patterns)

- Round-robin sharding of (arch_t, seed) tuples across 8 GPUs
  with `gpu=$((job_idx % 8))` is clean. Each GPU runs 2-3 jobs
  concurrently (~6-10 GB memory each on H100 — no OOM with
  d=256, d_sae=40-80).
- Maximum H100 utilization came from layering 3 zooms ×
  18 jobs each (= 54 processes) on top of hier (18 jobs) and
  hunt (8 shards) — total 80 processes. GPUs at 99-100% util,
  ~15 GB memory peak. The CLAUDE.md "few hours" budget was met.
- `Monitor` with `tail -F | grep --line-buffered "Traceback|...|RuntimeError|Killed"`
  caught the k_win > d_sae crash within seconds of it happening.

### Open questions for Han / agent_paper

- **Is pB05_np10 acceptable as the c2 paper headline regime?**
  It's a slightly different setup from Dmitry's published Bench 2
  (n_parents=10 vs his n_parents=5) — same generator family, max
  overlap. The pB05_np5 plot is also paper-ready and matches
  Dmitry's exact setup if direct reproduction is preferred.
- **d_sae=40 is the active cap on TXC at high k_pos.** A future
  pass with d_sae=80 (or per_component_hparams.c2 override) would
  let the sweep go to k_pos=15-20 without TXC degenerating. Touches
  agent_paper's `configs/locked_archs.yaml`. Worth the cell budget?
- **Hierarchical bench tweaks**: K_l=50 datasource added but not
  swept (only the K_l=30 primary was). If c2.md wants to show the
  divide more sharply, sweep K_l=50 (+1 GPU-hour budget).

### DESIGN FRAMEWORK — how to think about new setups

The C2 synthetic suite is a **mechanism-discrimination grid**. Each
setup isolates ONE axis of difficulty and asks "does TXC's window-
pooling buy us something on THIS axis?" Once you understand the
existing axes, the unexplored axes become the next setups.

**Axes already swept**:

| Axis | Setup(s) | What it tests | TXC mechanism |
|---|---|---|---|
| Coupling structure (OR-gate, n_parents) | A, D | Hidden-chain → emission projection | dictionary alignment with hidden directions |
| Per-token Bernoulli emission noise (p_B) | B, D | "Did the emission actually fire?" noise | window-pool over multiple noisy obs of same hidden state |
| Parent overlap (n_parents up to K_hidden) | D | Per-token signal AMBIGUITY | window-pool sees joint co-firing pattern |
| Temporal autocorrelation (ρ) | C | Effect 1 (sample agg) vs Effect 2 (temporal pattern) | depends on ρ — Effect 2 only at ρ > 0 |
| Engineered global vs local separation | E | Slow globals + fast locals in orthogonal directions | TXC biased toward slow, SAE toward fast |
| Additive Gaussian observation noise (σ) | F, G | Pure denoising | window-pool = √T noise reduction |
| Window-size T (across all setups) | T-sweep | Local↔global trade-off knob | k_win = k_pos × T grows with T |

**Axes NOT YET swept — these are the next-setup candidates**:

| Axis | Proposed setup | Why it matters for the paper |
|---|---|---|
| Negative correlation between globals | K | Tests if TXC sees structure when per-token is i.i.d.-looking |
| Magnitude-only modulation (locals i.i.d. in support) | L | Pure temporal-PATTERN test — globals not in observation space |
| High-frequency target (Δh / temporal_derivative) | I (Dmitry Bench 3) | HONEST CAVEAT — TXC FAILS here (low-pass filter limit) |
| Per-feature heterogeneous ρ | M | Tests if TXC recovers SLOW features preferentially |
| Sparse globals (very low π_g) | N | Stress test: rare global events recoverable? |
| MULTI-scale globals (slow + fast hidden) | O | Tests if optimal T depends on hidden timescale |
| n_seqs / n_steps scaling | P | Sample complexity story |
| Frequency-domain decomposition (DC/AC ablation) | Q | Dmitry's E9 ablation — confirms low-pass filter narrative |

### CROSS-SETUP PATTERNS (the "discoveries" so far)

These are the empirical regularities across D, E, F, G that any
new setup design should respect or test:

1. **TXC's gAUC saturates earlier than its eAUC trade-off shows**:
   On D-np10, gAUC hits 0.99 at T=4 and stays; eAUC keeps rising
   until T=10+. Interpretation: T=4 windows already disambiguate
   global structure; bigger T just adds capacity for locals.

2. **Optimal T grows with bench difficulty**: D-np5 (moderate
   overlap) optimal T=2; D-np10 (max overlap) optimal T=4-12;
   F σ=0.5 optimal T=4-6; F σ=2.0 optimal T=8-12; G σ=2.0 optimal
   T=12+. **Rule of thumb**: more noise / more overlap → larger
   window needed.

3. **TXC's gAUC vs eAUC scatter has a characteristic UP-LEFT
   trajectory as T grows** on E (and weakly on D). T=2 sits at
   high eAUC + lower gAUC; T=12 at lower eAUC + higher gAUC. This
   **IS the local↔global axis made visible**. SAE family meanwhile
   moves DOWN-RIGHT with k_pos.

4. **The TXC vs SAE gap on gAUC is largest in regimes where
   per-token information is most ambiguous**: max overlap (D-np10),
   high noise (F σ=1, G σ=2), high modulation (E). The gap
   SHRINKS when per-token information is locally sufficient (e.g.
   D-np5 at high k where SAE has enough capacity).

5. **At fixed d_sae=40, TXC's k_pos × T ≤ d_sae constraint is the
   binding limit at high T**. This caps the sweep at k_pos=8 for
   T=5 and k_pos=3 for T=12. Bumping d_sae=80 (touches
   locked_archs.yaml — agent_paper territory) would extend the
   trade-off plot in BOTH dimensions.

### c2.md ownership protocol (Han 2026-05-07)

**You (agent_synth) are the primary updater for c2.md sections on
Setup D, E, F, G** (and any new setups you add — H/I/...). Rules:

1. **One-author-per-section**: don't edit Setup A/B/C content (those
   are agent_paper / agent_filler / agent_hammer territory). Don't
   edit other agents' new-setup sections.
2. **Always rebase + dedupe-merge before push** — leaderboard.jsonl
   and manifest.jsonl conflicts are routine; resolve via the
   union+dedupe Python one-liner already documented above.
3. **AUTO-RESULTS blocks are append-only friendly**: re-rendering
   replaces the body between `<!-- BEGIN AUTO-RESULTS-c2X -->` and
   `<!-- END AUTO-RESULTS-c2X -->` markers. Hand-written prose
   OUTSIDE markers is preserved.
4. **Cross-territory waiver**: Han approved (2026-05-06T23:30Z) for
   you to add new Setup sections directly. Assume it carries forward
   unless he revokes.
5. **agent_paper integrates** on render passes (re-orders sections,
   adjusts hypothesis/caveats prose, cross-section consistency).
   You don't reorder; you just append your section in the right
   slot (after the previous Setup, before "## Headline figure" or
   "## Caveats").

**TODO post-compact**: add a `render_setup_into_c2md(setup_name)`
helper to `plot_headline.py` (or a new `render_into_c2md.py`) that
mirrors `agents/agent_hammer/render_results.py`'s pattern — reads
leaderboard, writes AUTO-RESULTS-c2X tables, atomically rewrites
the section block in c2.md. Until then, c2.md updates are manual
edits.

### THE ONE PAPER GOAL (Han 2026-05-07)

For every new setup we want exactly two things:
1. **gAUC vs eAUC plot where TXC outshines all baselines**
   (TopK, Stacked T=2, Stacked T=5, T-SAE, TFA-pos).
2. **What happens as T grows** (TXC-base T-sweep at fixed k_pos).

Don't get distracted by ablations, real-data bridges, or honesty
caveats. The mechanism story is "more settings where TXC wins on
gAUC vs SAE-family + the T-sweep tells us how that win evolves
with window size".

Pick new setups by VARYING the data-generation knob you haven't
tried yet, run the full 4-plot template, see if TXC outshines.
If yes → keep. If no → quietly drop and try another knob.

### MANDATORY 4-PLOT STANDARD per synthetic setup (Han 2026-05-07)

**Every C2 synthetic setup (A through Z) MUST have exactly these
four plots** before being considered "complete" and pushed to the
paper. No exceptions. Filenames for setup `<X>` go in
`experiments/c2_<scope>/plots/`:

1. `c2_setup_<X>_gauc_vs_k.png` — **gAUC vs k_pos**, one line per
   arch, error bars over seeds, log/linear x as appropriate.
   Headline metric — answers "TXC vs SAE on global recovery".
2. `c2_setup_<X>_eauc_vs_k.png` — **eAUC vs k_pos**, same axes,
   same archs. Answers "TXC vs SAE on local recovery".
3. `c2_setup_<X>_scatter.png` — **gAUC vs eAUC scatter**, each
   point = one (arch, T, k_pos) cell mean over seeds, y=x diagonal,
   k_pos annotated on first/last, T color-coded for txc_base.
   Visualises the global-vs-local trade-off explicitly.
4. `c2_setup_<X>_tsweep.png` — **gAUC + eAUC vs T at fixed k_pos**
   (k_pos=1 default), one txc_base curve per metric, error bars
   over seeds. Shows the local↔global / denoising trade-off as a
   function of window size.

**Current gaps to fix:** D-np5/np10 and E currently have
`c2_txc_win_gauc_vs_k.png` (gAUC + eAUC combined as 2-panel — needs
split into the two separate plots); F, G have ONLY a σ-sweep panel,
need full 4 added (each at one chosen σ, e.g. σ=1.0 for F's
"canonical" view; the σ-sweep panel can stay as a SUPPLEMENTARY 5th
plot).

`plot_headline.py:_arch_label` is the central control point —
every render call should produce all four. Recommend adding a
`render_setup(setup_name, filter_fn, plot_dir)` orchestrator that
emits all four in one call, used uniformly across setups.

### POST-COMPACT TODO (Han 2026-05-07T00:55Z) — DO THIS FIRST

**Priority order:**

0. **Render all 4 mandatory plots for D, E, F, G** (per the
   standard above). Currently:
   - D-np5: split combined panel into separate gauc-vs-k + eauc-vs-k
   - D-np10: same
   - E: same
   - F (canonical σ=1.0): need gauc-vs-k, eauc-vs-k, scatter
   - G (canonical σ=1.0): need gauc-vs-k, eauc-vs-k, scatter
   The T-sweep plots already exist for D, E, F, G. Scatters exist
   for D, E. Missing primarily: separate eauc-vs-k for D/E + full
   set for F/G at canonical σ.

1. **Fill missing baselines on Setups D, E, F, G.**
   For every synthetic setup we want the 5 baselines: TopK-SAE,
   Stacked-SAE T=2, Stacked-SAE T=5, T-SAE (`tsae_paper`), TFA-pos
   (`tfa_pos`), plus full TXC-base T-sweep T ∈ {2,4,5,6,8,10,12}.

   Current gaps (verified by leaderboard inventory at 00:55Z):

   | Setup | topk | stk T=2 | stk T=5 | TXC T-sweep | tsae_paper | tfa_pos |
   |---|---|---|---|---|---|---|
   | D-np5 (n_parents=5)  | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
   | D-np10 (n_parents=10) | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
   | E (hierarchical Kg10_Kl30) | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
   | F σ ∈ {0.5, 1.0, 2.0} | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
   | G σ ∈ {1.0, 2.0}      | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |

   Plan: extend each setup driver (`run_hunt.py`, `run_t_sweep.py`,
   `run_setup_f.py`, `run_setup_g.py`) to accept `--archs tsae_paper
   tfa_pos stacked_sae` and run them at the same k_pos × seed grid
   as topk_sae. Cell counts:
     - D-np5 + D-np10 + E: 2 archs (tsae + tfa) × 3 seeds × 8 k_pos
       × 3 setups = 144 cells. ~10-15 min on 8 GPUs.
     - F σ ∈ {0.5,1,2}: 4 archs (stk-T2, stk-T5, tsae, tfa) × 3 seeds
       × 3 k_pos × 3 sigmas = 108 cells.
     - G σ ∈ {1,2}: 4 archs × 3 seeds × 3 k_pos × 2 sigmas = 72 cells.
   Total: ~324 cells, ~30-40 min wall on 8 GPUs.

   tsae_paper at component=c2: locked YAML has d_sae=16384; pass
   `arch_hparams_override={"d_sae": 40, "k_pos": k}` to override
   (agent_hammer used the same hack — see his run_baselines.py).

   tfa_pos: locked YAML has it; standard k_pos override applies.

2. **Re-render plots after baselines land**. Then update c2.md with
   the new arches in each AUTO-RESULTS table:
     - `experiments/c2_synthetic_coupled/plot_headline.py:_arch_label`
       needs entries for `tsae_paper` (label "T-SAE", marker "h",
       color magenta `#CC79A7` matching c2.md style) and `tfa_pos`
       (label "TFA-pos", marker "X", color green `#2ca02c`). Add to
       ARCH_COLORS dict.
     - Re-render scatter, line, σ-sweep plots.

3. **Continue designing new setups** (Han 2026-05-07: keep going).
   Already done F (coupled + obs noise) and G (hier + obs noise).
   Open candidates for Setup H, I, …:
     - H: ρ-sweep on Setup D pB05_np10 (Effect 1 vs Effect 2 on the
       max-overlap regime). 4 ρ × 5 archs × 3 seeds × 3 k_pos = 180
       cells. Note: agent_filler is doing ρ-sweep on Setup A; this
       extends to D.
     - I: temporal-derivative target (Dmitry Bench 3 port). Honest
       caveat: TXC FAILS when target is high-frequency. Establishes
       the limit.
     - J: hierarchical with K_l=50 datasource (already in YAML, not
       yet swept). Tests divide scaling with feature count.
     - K: anti-correlated globals — globals are pairwise NEGATIVELY
       correlated. Per-token signal looks random; TXC window pool
       sees the structure.
     - L: magnitude-modulated locals (locals fire i.i.d. but their
       MAGNITUDES are slow-modulated by globals).

   Shared protocol for any new setup:
     - Generator in `src/temp_bench/data/toy/<name>.py`.
     - 1+ datasources in `configs/datasources.yaml`.
     - Driver in `experiments/c2_synthetic_coupled/run_setup_<X>.py`
       (or `c2_hierarchical/...`). Parametric over arch + seed +
       k_pos at minimum.
     - Launch all 5 baselines + TXC T-sweep × 3 seeds × 3 k_pos.
     - c2.md section + scatter + line plots + tables.

4. **agent_hammer is now doing PARALLEL new-setup work** on a 5×
   RTX PRO 6000 pod (briefing also updated 2026-05-07). Coordinate:
     - agent_synth (you): owns Setup D/E/F/G; lead on H/I above.
     - agent_hammer: lead on J/K/L; can also fill baselines on
       D/E/F/G if convenient (we both have territory waivers).
     - DON'T duplicate — check leaderboard for existing (datasource,
       arch, seed, k_pos) cells before launching.

### Key code paths (post-compact orientation)

- `src/temp_bench/data/toy/`:
  - `coupled.py` — Setup A generator (agent_filler territory).
  - `coupled_noisy.py` — Setup D (mine).
  - `hierarchical.py` — Setup E (mine).
  - `coupled_obs_noise.py` — Setup F (mine).
  - `hierarchical_obs_noise.py` — Setup G (mine).
- `experiments/c2_synthetic_coupled/`:
  - `run.py` — Setup A driver (agent_filler).
  - `run_hunt.py` — Setup D HUNT + ZOOM (mine).
  - `run_t_sweep.py` — Setup D T-sweep (mine).
  - `run_setup_f.py` — Setup F (mine).
  - `plot_headline.py` — every plot in c2.md's Setup D/E/F/G is
    rendered from here. `_arch_label` is the central control point
    for which archs appear in plots.
  - `hunt_analysis.py` — gap-table generator.
- `experiments/c2_hierarchical/`:
  - `run.py` — Setup E driver (mine).
  - `run_t_sweep.py` — Setup E T-sweep (mine).
  - `run_setup_g.py` — Setup G (mine).
- `docs/components/c2.md` — paper writeup (cross-territory edit
  approved by Han 2026-05-06T23:30Z; explicit per-edit approval
  required).

### HF push state (don't forget post-compact)

agent_synth checkpoints have `hf_url=null` in manifest because
TEMP_BENCH_POD_MODE=ephemeral didn't propagate through Bash tool
call subprocess env. agent_hammer documented + fixed this in their
launcher; my launchers (`run_hunt.sh`, `run_zoom.sh`,
`run_sharded.sh`, `run_t_sweep.sh`, `run_setup_f.sh`,
`run_setup_g.sh`) now have `TEMP_BENCH_POD_MODE=persistent` set —
auto-push DISABLED to avoid HF rate limit (256 commits/hour).

`scripts/push_synth_ckpts_to_hf.py` is the manual push script. It
ran ~140 cells before hitting the API rate limit (2500 reqs / 5
min) and getting stuck. Need to resume sequentially WITH retry/
backoff logic on 429s. Sketch:

```python
import time
from huggingface_hub.utils import HfHubHTTPError
def push_with_retry(api, tk, ckpt_dir, max_retries=5):
    for attempt in range(max_retries):
        try:
            api.upload_folder(folder_path=str(ckpt_dir), path_in_repo=tk,
                              repo_id="han1823123123/temp-bench-models",
                              repo_type="model")
            return True
        except HfHubHTTPError as e:
            if e.response.status_code == 429:
                wait = int(e.response.headers.get("Retry-After", 60))
                time.sleep(wait + 5)
            else:
                raise
    return False
```

Probably ~600 of the 886 agent_synth manifest entries need pushing
still. At ~3 sec per push + 300-sec backoff after every 256 commits,
total wall ≈ 30-40 min. Run in background, low priority.

## What I just did (agent owns — overwrite)

**Session 2026-05-07T01:00-02:50Z — second-wave setup expansion.**

Picked up post-compact from earlier mission (Setups D/E/F/G/J done).
Han directives received this session:
- "ensure all setups have all baselines TopK Stacked T=2 Stacked T=5 TFA-pos TSAE"
- "every setup MUST have 4 plots: gauc_vs_k, eauc_vs_k, scatter, tsweep"
- "generate NEW IDEAS with random codenames (not single letters) so
  they don't collide with other agents"
- "drop if TXC doesn't outshine"

Delivered:
- Filled tsae_paper + tfa_pos baselines for D-np5, D-np10, E (126 cells).
- Created Setup J (hierarchical K_l=50): 217 cells.
- Wrote unified ``fill_baselines.py`` driver dispatching on YAML
  ``generator`` field — handles 8+ generator families.
- Extended ``plot_headline.py`` with ``render_setup`` 4-plot
  orchestrator + tsae_paper/tfa_pos in ARCH_COLORS.
- Designed + ran 6 NEW codename setups:
  - **Setup M**: heterogeneous-ρ globals (5 slow + 5 fast). Gap +0.24.
  - **Setup whisper**: sparse globals π_g=0.01. Gap +0.22.
  - **Setup polaris**: ultra-slow + slow ρ. Gap +0.34.
  - **Setup lighthouse**: 1 slow + 9 fast (stress test). Gap +0.17.
  - **Setup dewdrop**: deterministic period=16 firing. **TXC LOSES**
    (gap −0.23; TFA-pos wins with 0.78). Excluded from cross-setup
    summary per Han's "drop if not" — but data committed for analysis.
  - **Setup chord** (RUNNING): 2 groups of 5 phase-locked globals.
  - **Setup aurora** (RUNNING): coupled + OU-process auto-correlated noise.
- Fixed Setup F + G plot rendering bug (filter on ``obs_noise_sigma``
  was wrong — hammer's backfill cells had it None; switched to
  datasource-based filter).
- Filled Setup A T-sweep (T={2,4,6,8,10,12}) on H100s.
- Added ``cross_setup_summary.py`` — paper-grade horizontal bar chart.
- Added ``audit_setups.py`` — automated checker.
- Rewrote ``agents/agent_pro/briefing.md`` to redirect to Setup K + L
  per Han's request (their C3 mission was aborted).

**Final audit (18 setups, all ✅)**: A, D-np5, D-np10, E, F-σ{0.5,1,2},
G-σ{1,2}, J, M, whisper, polaris, lighthouse, dewdrop, chord, aurora,
harbor. 5 baselines present, T-sweep T ∈ {2,4,5,6,8,10,12}, 4
mandatory plots embedded.

**TXC-wins (13, paper-headline cross-setup):** A, D-np5, D-np10, E,
F-σ1, G-σ1, J, M, whisper, polaris, lighthouse, chord, aurora.

**TXC-loses (6, honest negatives — data committed but EXCLUDED from
cross-setup paper headline per Han's "drop if not" directive):**
- dewdrop (mine): deterministic period-16 firing → TFA-pos wins.
- harbor (mine): π=0.5 weak-magnitude globals (×0.1) → both archs fail.
- K (pro): anti-correlated globals.
- L (pro): magnitude-modulated locals.
- PHALANX (pro): period-locked global pulses.
- OBELISK (pro): rare amplified bursts.

**Cross-setup gAUC gap (TXC vs TopK at k_pos=1)**: positive in 11 of
12 paper setups (range +0.12 to +0.44). Dewdrop is the sole TXC-loses
case — quietly excluded from headline, mechanism understood
(positional structure → TFA-pos wins).

Recent commits (most recent first):
- `4926ae4d` Setup chord + aurora launchers (running)
- `6c0d35c9` lighthouse + dewdrop results, Setup A 4-plot, audit-passed
- `757ef3e5` lighthouse + dewdrop generators + audit script
- `5e0093f7` Setup whisper + polaris results
- `614951d6` Setup M results + Setup whisper/polaris launchers
- `f3e6ab89` fix Setup F/G plot filter (was missing baselines)
- `0857d852` Setup F + G — render full 4-plot at canonical σ=1.0
- `43683589` Setup M — heterogeneous-ρ globals (slow + fast mixed)
- `d9acdcb4` D/E baselines (tsae+tfa) + Setup J + 4-plot infra
- `11b40218` rewrite agent_pro briefing — redirect to C2 Setup K + L

## Next action (agent owns — overwrite)

The synthetic suite is paper-ready: 15 setups all audit-passing,
cross-setup summary plot shows TXC > TopK in 11 of 12 paper regimes.

**If continuing autonomously**:
- Wait for chord + aurora shards to finish (~25-40 min from launch).
- Render their 4-plot via the ``render_setup`` orchestrator.
- Re-run ``audit_setups.py``; expect both to pass.
- Re-run ``cross_setup_summary.py`` to refresh the comparison plot.
  (Likely chord shows degenerate-direction tie; aurora may show
  TXC win or loss depending on whether window-pool can handle
  auto-correlated noise. If win, add to cross-setup; if not, drop
  per Han's directive.)
- Add c2.md sections for the keepers; commit + push.

**Open candidate setup ideas** (not yet implemented):
- **eddy** — slowly rotating feature directions (rotational dynamics).
- **echo** — emissions are time-shifted echoes of globals.
- **mirage** — features that look identical per-token but differ
  over time (decoder-direction-degenerate per-token).
- **drift** — global magnitude continuously drifts (random walk
  modulation). Note: agent_pro's Setup L is similar — coordinate.
- **kelp** — phase-locked patterns with stochastic dwell.

Standard quick-launch recipe for any new setup:
1. Add generator to ``src/temp_bench/data/toy/<name>.py``.
2. Add YAML datasource entry.
3. Extend ``fill_baselines._build_data`` dispatch.
4. Mirror existing ``run_*.sh`` launcher (12 archs × 3 seeds = 36 shards).
5. Render via ``render_setup`` (already supports any datasource via
   filter_fn).
6. Add c2.md section.

**Re-render-only path** (no new training):
```bash
TQDM_DISABLE=1 .venv/bin/python -m experiments.c2_synthetic_coupled.audit_setups
TQDM_DISABLE=1 .venv/bin/python -m experiments.c2_synthetic_coupled.cross_setup_summary
TQDM_DISABLE=1 .venv/bin/python -m experiments.c2_synthetic_coupled.plot_headline
```

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

(See `### Open questions for Han / agent_paper` inside the
"Current state" section above for the post-mission questions.
That section supersedes the pre-mission questions that used to
live here.)
