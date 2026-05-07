<!--
Original mission drafted by agent_nlp 2026-05-06T23:18Z (C3 probing-protocol
sweep). REWRITTEN 2026-05-07T01:30Z by agent_synth under direct Han override:
"agent_pro is idle now since I aborted their mission. We can use them to
furhter the investigation into MORE SYNTHETIC SETUPS or alternatively,
strenghtening the existing ones — write a briefing for them and push!"

Pod: 7× RTX 5090 (Blackwell-gen consumer card, 32 GB VRAM each = 224 GB total),
989 GB system RAM, 224 CPUs, ephemeral storage.
Section ownership rules: PROTOCOL.md § 14.
-->

---
agent: agent_pro
last_state_update: 2026-05-07T01:30:00Z
status: spawning
component: c2 (synthetic — Setup K + L design + sweep)
---

## Identity + mandate (Han owns — agents do not edit)

You are **agent PRO**. Your single mission is **two new C2 synthetic
setups** (Setup K and Setup L) that isolate **Effect 2** (temporal
pattern detection) cleanly — regimes where per-token observation is
randomised but the TEMPORAL pattern across tokens encodes the global
structure. This is where TXC's window pooling SHOULD have its
strongest theoretical advantage.

### The ONE paper goal (Han 2026-05-07)

For every new setup we want exactly two things:
1. **gAUC vs eAUC where TXC outshines all baselines** (TopK,
   Stacked T=2, Stacked T=5, T-SAE, TFA-pos).
2. **What happens as T grows** (TXC-base T-sweep at fixed k_pos).

Don't add ablations, honest-caveat benches, real-data bridges, or
mechanism deep-dives. If TXC outshines on a new setup, KEEP. If not,
DROP and try another knob.

### MANDATORY 4-PLOT STANDARD per synthetic setup

**Every C2 synthetic setup MUST have exactly these four plots**
before being considered "complete" and pushed to the paper.
Filenames go in `experiments/c2_synthetic_coupled/plots/` (or
`experiments/c2_hierarchical/plots/` for hier-flavoured). For setup
`<X>`:

1. `c2_setup_<X>_gauc_vs_k.png` — gAUC vs k_pos, one line per arch,
   error bars over seeds.
2. `c2_setup_<X>_eauc_vs_k.png` — eAUC vs k_pos (same axes/archs).
3. `c2_setup_<X>_scatter.png` — gAUC vs eAUC scatter (each point =
   one (arch, T, k_pos) cell mean over seeds, y=x diagonal).
4. `c2_setup_<X>_tsweep.png` — gAUC + eAUC vs T at fixed k_pos
   (txc_base only, all 7 T values 2/4/5/6/8/10/12).

The renderer is already built: use
`experiments/c2_synthetic_coupled/plot_headline.py:render_setup`
which emits all 4 in one call. See `_arch_label` for the central
control of which archs appear (tsae_paper magenta `#CC79A7` "h",
tfa_pos green `#2ca02c` "X" — already wired).

### Pod allocation
- **Hardware**: 7× RTX 5090 (Blackwell, 32 GB VRAM each, 224 GB
  total VRAM), 989 GB RAM, 224 CPUs, ephemeral `/workspace`.
- **Mode**: parallel-launch pattern. Default GPU 0; fan out via
  `bash scripts/run_on_gpu.sh <0..6> -- <cmd>`.
- **VRAM**: synthetic toy cells use ~3-5 GB per process at
  d=256/d_sae=40 — you can pack 6+ procs per GPU comfortably (32 GB
  cards). All 7 GPUs in parallel; goal is full saturation.

### Files you may edit
- `agents/agent_pro/briefing.md` (your own).
- `experiments/c2_synthetic_coupled/` — drivers + scripts. Han
  approved cross-territory edit waiver 2026-05-06T23:30Z for c2
  agents (synth + hammer + you).
- `experiments/c2_hierarchical/` — same waiver.
- `src/temp_bench/data/toy/` — add new generator files for Setup K
  and L. You author them; mirror agent_synth's pattern in
  `coupled_noisy.py` / `hierarchical.py`.
- `configs/datasources.yaml` — add new YAML entries for K and L
  datasources. (Adding entries is fine; modifying existing entries is
  NOT — they belong to the agent who created them.)
- `docs/components/c2.md` — write a "Setup K" section + "Setup L"
  section AFTER the existing Setup G section, BEFORE
  "## Headline figure for the paper". Use the same template as
  agent_synth's Setup D/E/F/G sections.
- Logs to `logs/`, scratch to `/tmp/`.

### Files OUT OF SCOPE — do NOT edit:
- `agents/agent_*/` — every other agent's directory.
- `experiments/c1_*`, `experiments/c3_*`, `experiments/c4_*`,
  `experiments/c5_*`, `experiments/c6_*`, `experiments/c7_*` —
  other components, not your scope.
- Existing `c2_synthetic_coupled/run.py` (agent_filler/paper),
  `coupled.py` / `coupled_noisy.py` / `hierarchical.py` /
  `coupled_obs_noise.py` / `hierarchical_obs_noise.py` (agent_synth
  generators — they're done, don't touch).
- Existing c2.md sections for Setup A/B/C/D/E/F/G/J — don't
  re-render or rewrite. ONLY add Setup K + L sections.
- `docs/paper/*` (agent_paper).
- `agents/agent_paper/decisions.md` (agent_paper).
- `configs/locked_archs.yaml` (agent_paper).
- `pyproject.toml` / `uv.lock` (atomic, agent_paper).
- `src/temp_bench/architectures/` — never modify arch code.

If you find yourself wanting to edit an out-of-scope file, **STOP**.
Add a bullet to "Open questions for Han" in your own briefing,
surface it. Even verbal approval from Han doesn't count — written
approval per session, then let the owner integrate.

### c2.md ownership protocol (Han 2026-05-07)

**You (agent_pro) own the c2.md sections for Setup K + L** — and any
further new setups you propose. Rules:

1. **One-author-per-section**: don't edit any other agent's setup
   sections (A/B/C = paper/filler/hammer; D/E/F/G/J = synth;
   H/I = hammer if active).
2. **Always rebase + dedupe-merge before push**. leaderboard.jsonl
   and manifest.jsonl conflicts are routine; resolve via the
   union+dedupe Python one-liner (see "Watch-outs" below — copy from
   agent_synth's briefing).
3. **AUTO-RESULTS markers** delineate the autogen tables. Body
   between `<!-- BEGIN AUTO-RESULTS-c2X -->` and
   `<!-- END AUTO-RESULTS-c2X -->` is rewritten by render passes.
   Hand-written prose OUTSIDE markers is preserved.
4. **Cross-territory waiver**: Han approved 2026-05-06T23:30Z for
   c2 agents to add new Setup sections directly. Carries forward
   unless he revokes.
5. **agent_paper integrates** on render passes (re-orders sections,
   adjusts hypothesis/caveats prose). You don't reorder; just
   append your section in the right slot (after Setup G or J,
   before "## Headline figure" or "## Caveats").

### ⚠️ MISSION 2026-05-07T01:30Z — Setup K + Setup L

#### Setup K — Anti-correlated globals (one-hot global chain)

**Mechanism**: at each token, exactly ONE of K_g global chains is
"on". Transitions are sticky (ρ=0.9: same global stays on with prob
0.9; switches uniformly to another global with prob 0.1). Locals
fire normally, modulated by their parent global.

**What it isolates**: per-token observation looks like a single
sparse global firing + its modulated locals — every per-token SAE
sees this fine. **The TEMPORAL signal** is which global is on as a
function of t, and the alternation pattern across tokens. TXC's
window pool aggregates that pattern; per-token SAE cannot.

**Why TXC should win**: gAUC measures recovery of the K_g global
DIRECTIONS f_g[i] at the latent-codes level. With anti-correlated
firing, a per-token SAE's individual latents tend to ALIGN with the
single-firing pattern at each token (f_g[i] gets isolated), so
per-token SAE may actually recover globals reasonably. The TXC
advantage will be MORE PRONOUNCED on eAUC/local recovery because
TXC sees that local-firing patterns conditional on global-state
form temporal clusters.

If TXC doesn't outshine on K, that's a signal — drop it and try L.

**Generator**: new file `src/temp_bench/data/toy/anticorrelated.py`.
Sketch:

```python
import torch
from temp_bench.data.toy.coupled import (
    CoupledData, _orthogonalise, _sample_magnitudes,
)

def _onehot_global_chain(*, n_seqs, K_g, T, rho, rng):
    """Sticky one-hot global chain.  At each t, exactly one of K_g
    globals is on. Stay-prob = rho; switch uniformly to one of the
    other K_g-1 with prob (1-rho).  Returns h_g of shape
    (n_seqs, K_g, T) with one-hot t-slices."""
    states = torch.zeros(n_seqs, T, dtype=torch.long)
    states[:, 0] = torch.randint(0, K_g, (n_seqs,), generator=rng)
    for t in range(1, T):
        u = torch.rand(n_seqs, generator=rng)
        switch = u >= rho
        new = torch.randint(0, K_g - 1, (n_seqs,), generator=rng)
        # avoid self-transition: shift the "new" index past the current one
        new = new + (new >= states[:, t-1]).long()
        states[:, t] = torch.where(switch, new, states[:, t-1])
    h_g = torch.zeros(n_seqs, K_g, T)
    h_g.scatter_(1, states.unsqueeze(1), 1.0)
    return h_g

def anticorrelated_features(*, K_global=10, K_local=30,
        n_global_parents=1, d_in=256, seq_len=64, n_seqs=4096,
        rho_g=0.9, p_l_high=0.8, p_l_low=0.1,
        magnitude_dist="folded_normal", magnitude_mean=1.0,
        magnitude_std=0.15, seed=0, device="cpu"):
    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    n_total = K_global + K_local
    if n_total > d_in: raise ValueError(...)
    all_features = _orthogonalise(n_total, d_in, rng)
    f_g = all_features[:K_global].contiguous()
    f_l = all_features[K_global:].contiguous()
    # one-hot global chain (sticky transitions)
    h_g = _onehot_global_chain(
        n_seqs=n_seqs, K_g=K_global, T=seq_len, rho=rho_g, rng=rng)
    # local parents — same as hierarchical bench
    C = torch.zeros(K_local, K_global)
    for j in range(K_local):
        parents = torch.randperm(K_global, generator=rng)[:n_global_parents]
        C[j, parents] = 1.0
    parent_on = (torch.einsum("lk,nkt->nlt", C, h_g) >= 1).float()
    p_local = parent_on * p_l_high + (1 - parent_on) * p_l_low
    u = torch.rand(n_seqs, K_local, seq_len, generator=rng)
    s_l = (u < p_local).float()
    # magnitudes
    mag_l = _sample_magnitudes((n_seqs, K_local, seq_len),
        dist=magnitude_dist, mean=magnitude_mean, std=magnitude_std, rng=rng)
    mag_g = _sample_magnitudes((n_seqs, K_global, seq_len),
        dist=magnitude_dist, mean=magnitude_mean, std=magnitude_std, rng=rng)
    a_g = h_g * mag_g
    a_l = s_l * mag_l
    x_g = torch.einsum("nkt,kd->ntd", a_g, f_g)
    x_l = torch.einsum("nkt,kd->ntd", a_l, f_l)
    x = x_g + x_l
    return CoupledData(x=x.to(device), emission_features=f_l.to(device),
        hidden_features=f_g.to(device), coupling_matrix=C.to(device),
        hidden_states=h_g.to(device), emission_support=s_l.to(device))
```

**YAML datasources** (add to `configs/datasources.yaml`):

```yaml
toy_anticorrelated_Kg10_Kl30_d256:
  generator: temp_bench.data.toy:anticorrelated_features
  K_global: 10
  K_local: 30
  n_global_parents: 1
  d_in: 256
  seq_len: 64
  rho_g: 0.9
  p_l_high: 0.8
  p_l_low: 0.1
  notes: "Setup K — sticky one-hot global chain (anti-correlated)."

toy_anticorrelated_Kg10_Kl30_d256_rho99:
  # same but rho_g=0.99 (very sticky — globals dwell long; harder for
  # per-token SAE because the single-firing pattern persists across
  # many tokens, but TXC sees the dwell directly).
  generator: temp_bench.data.toy:anticorrelated_features
  rho_g: 0.99
  ...  # rest same
```

#### Setup L — Magnitude-modulated locals (no globals in obs space)

**Mechanism**: K_g global chains evolve as in hierarchical (rho_g=0.95,
pi_g=0.05 sparse). K_l local features fire i.i.d. with constant
probability p_l = 0.5 (NOT modulated by globals). BUT when local j
fires, its MAGNITUDE is drawn from a global-modulated distribution:
`mag(j, t) = mag_base + alpha * sum_k C[j, k] * h_g[k, t]`. Globals
are NOT in observation space (no f_g · h_g term in x).

**What it isolates**: per-token observation is `x(t) = sum_j s_l[j,t]
· mag(j,t) · f_l[j]`. No direction in observation space tracks the
globals directly. The ONLY way to recover globals is to detect the
slow temporal modulation of local-firing magnitudes. **Pure Effect 2
test.**

**Why TXC should win**: per-token SAE only sees individual magnitude
readings — looks like high-variance noise. TXC's window pool over T
tokens averages out the i.i.d. firing component and reveals the slow
magnitude trend, which IS the global signal. gAUC for TXC should be
high; per-token SAE gAUC ≈ chance.

**Generator**: new file `src/temp_bench/data/toy/magnitude_modulated.py`.
Same skeleton as `hierarchical.py`. Differences:
- Local firing prob is constant (`p_l = 0.5`), NOT parent-modulated.
- Local magnitude has additive global term:
  ```python
  mag_l = base_mag + alpha * torch.einsum("lk,nkt->nlt", C, h_g)
  # multiply by per-cell folded-normal noise:
  mag_l = mag_l * _sample_magnitudes(..., mean=1.0, std=0.15, rng=rng)
  ```
- NO global-direction term in observation: `x = einsum(s_l * mag_l, f_l)`.
- `hidden_features` returned is f_g (gAUC eval will measure if any
  decoder direction aligns with f_g — which it CAN'T from per-token
  alone; needs temporal aggregation).

**YAML datasources**:

```yaml
toy_magmod_Kg10_Kl30_d256_alpha1:
  generator: temp_bench.data.toy:magnitude_modulated_features
  K_global: 10
  K_local: 30
  n_global_parents: 1
  d_in: 256
  seq_len: 64
  pi_g: 0.05
  rho_g: 0.95
  p_l: 0.5
  alpha: 1.0           # global modulation amplitude (vs base mag=1)
  base_mag: 1.0
  notes: "Setup L — magnitude-modulated locals; globals NOT in obs space."

toy_magmod_Kg10_Kl30_d256_alpha2:
  alpha: 2.0           # stronger modulation — easier for TXC
  ...

toy_magmod_Kg10_Kl30_d256_alpha05:
  alpha: 0.5           # weaker modulation — harder
  ...
```

### How to run each setup

agent_synth wrote a unified driver `experiments/c2_synthetic_coupled/
fill_baselines.py` that already handles arch dispatch (topk_sae,
stacked_sae T=2/T=5, tsae_paper, tfa_pos, txc_base × T-sweep) and
generator dispatch via the YAML `generator` field. **Extend
`fill_baselines._build_data` to recognise your new generators**:

```python
# Add to _build_data() in fill_baselines.py:
if gen.endswith(":anticorrelated_features"):
    return anticorrelated_features(...)  # pass YAML fields
if gen.endswith(":magnitude_modulated_features"):
    return magnitude_modulated_features(...)
```

Then your launcher just calls:

```bash
.venv/bin/python -m experiments.c2_synthetic_coupled.fill_baselines \
    --datasource <your_yaml_name> \
    --arch <topk_sae|stacked_sae|tsae_paper|tfa_pos|txc_base> \
    [--T <2|4|5|6|8|10|12>] \
    --seed <1|2|42> --k-poses 1 2 3 4 5 6 8 \
    --n-steps 8000
```

Mirror the launcher pattern in
`experiments/c2_hierarchical/run_setup_j.sh` — it's the canonical
template (12 arch×T combos × 3 seeds = 36 shards round-robin on 8
GPUs; you have 7 so adjust modulo to 7).

#### Concrete launcher sketch (write to `agents/agent_pro/run_setup_kl.sh`)

```bash
#!/usr/bin/env bash
# Setup K + L — 7× RTX 5090 launcher.
set -e; cd "$(dirname "$0")/../.."
mkdir -p logs
SEEDS=(1 2 42)
declare -a ARCH_TS=(
  "topk_sae:" "stacked_sae:2" "stacked_sae:5"
  "tsae_paper:" "tfa_pos:"
  "txc_base:2" "txc_base:4" "txc_base:5"
  "txc_base:6" "txc_base:8" "txc_base:10" "txc_base:12"
)
DATASOURCES=(
  "toy_anticorrelated_Kg10_Kl30_d256"            # Setup K canonical
  "toy_magmod_Kg10_Kl30_d256_alpha1"             # Setup L canonical
)
job_idx=0
for ds in "${DATASOURCES[@]}"; do
  for at in "${ARCH_TS[@]}"; do
    IFS=":" read -r arch T_arg <<< "$at"
    for seed in "${SEEDS[@]}"; do
      gpu=$((job_idx % 7))            # 7 GPUs
      label="kl_$(echo $ds | sed 's/toy_//')_${arch}_T${T_arg:-X}_s${seed}"
      log="logs/${label}_gpu${gpu}.log"
      # k_poses depend on T (k_pos × T ≤ d_sae=40)
      KP="1 2 3 4 5 6 8"
      [[ "$T_arg" == "6"  ]] && KP="1 2 3 4 6"
      [[ "$T_arg" == "8"  ]] && KP="1 2 3 4 5"
      [[ "$T_arg" == "10" ]] && KP="1 2 3 4"
      [[ "$T_arg" == "12" ]] && KP="1 2 3"
      T_FLAG=""; [[ -n "$T_arg" ]] && T_FLAG="--T $T_arg"
      setsid -f bash scripts/run_on_gpu.sh "${gpu}" -- \
        env AGENT_NAME=agent_pro TEMP_BENCH_POD_MODE=persistent \
            OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TQDM_DISABLE=1 \
        .venv/bin/python -m experiments.c2_synthetic_coupled.fill_baselines \
          --datasource "${ds}" --arch "${arch}" ${T_FLAG} \
          --seed "${seed}" --k-poses ${KP} --n-steps 8000 \
        < /dev/null > "${log}" 2>&1
      job_idx=$((job_idx + 1))
    done
  done
done
echo "[done] launched ${job_idx} shards"
```

Total cells per setup: 12 archs × 3 seeds × ~7 k_pos = **~250 cells**.
Two setups = ~500 cells. Wall on 7× 5090 saturated: ~25-40 min.

### Concrete first actions

1. **Smoke env**:
   ```bash
   cd /workspace/temp_xc/purified
   source scripts/set_agent_env.sh agent_pro
   bash scripts/agent_smoke_test.sh
   git pull --rebase origin final
   ```

2. **Implement Setup K generator** (`src/temp_bench/data/toy/anticorrelated.py`).
   Smoke-test it standalone:
   ```bash
   .venv/bin/python -c "
   from temp_bench.data.toy.anticorrelated import anticorrelated_features
   data = anticorrelated_features(K_global=10, K_local=30, d_in=256,
       seq_len=64, n_seqs=128, seed=42, device='cpu')
   print(data.x.shape, data.hidden_features.shape, data.emission_features.shape)
   # Verify one-hot: each (n, t) should have exactly one global active
   print('one-hot check:', data.hidden_states.sum(dim=1).min().item(),
                            data.hidden_states.sum(dim=1).max().item())
   "
   ```
   Expected: `torch.Size([128, 64, 256])`, `(10, 256)`, `(30, 256)`,
   one-hot check 1.0/1.0.

3. **Add YAML entries** for Setup K (one canonical + maybe a
   rho_g=0.99 variant). Verify load:
   ```bash
   .venv/bin/python -c "
   from temp_bench.config import load_datasource
   spec = load_datasource('toy_anticorrelated_Kg10_Kl30_d256')
   print(spec)
   "
   ```

4. **Extend `fill_baselines._build_data`** to recognise your generator
   (3-line add). Smoke one cell:
   ```bash
   TQDM_DISABLE=1 AGENT_NAME=agent_pro \
     bash scripts/run_on_gpu.sh 0 -- \
     .venv/bin/python -m experiments.c2_synthetic_coupled.fill_baselines \
       --datasource toy_anticorrelated_Kg10_Kl30_d256 \
       --arch topk_sae --seed 42 --k-poses 1 --n-steps 200 --smoke
   ```
   Verify a row lands in `results/leaderboard.jsonl` with the
   correct datasource + `eval_cfg.fill_baselines=True`.

5. **Repeat 2-4 for Setup L** (`magnitude_modulated.py`).

6. **Launch both setups** via your launcher script. Monitor:
   ```bash
   pgrep -af fill_baselines | wc -l
   nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader
   ```

7. **Render the 4-plot per setup** once shards finish:
   ```python
   from pathlib import Path
   from experiments.c2_synthetic_coupled.plot_headline import (
       render_setup, NOISY_PLOT_DIR,
   )
   SCATTER_PHASES = ('zoom', 'tsweep', 'fill')
   LINE_PHASES = ('zoom', 'fill')
   TSWEEP_PHASES = ('tsweep',)

   for setup_name, ds, title in [
       ("k", "toy_anticorrelated_Kg10_Kl30_d256",
        "Setup K (one-hot anti-correlated globals)"),
       ("l", "toy_magmod_Kg10_Kl30_d256_alpha1",
        "Setup L (magnitude-modulated locals)"),
   ]:
       def f(d, _ds=ds, _phases=LINE_PHASES):
           return (d.get('datasource') == _ds
                   and (d.get('eval_cfg') or {}).get('hunt_phase') in _phases)
       def fs(d, _ds=ds, _phases=SCATTER_PHASES):
           return (d.get('datasource') == _ds
                   and (d.get('eval_cfg') or {}).get('hunt_phase') in _phases)
       def ft(d, _ds=ds, _phases=TSWEEP_PHASES):
           return (d.get('datasource') == _ds
                   and (d.get('eval_cfg') or {}).get('hunt_phase') in _phases)
       render_setup(setup_name=setup_name, plot_dir=NOISY_PLOT_DIR,
                    line_filter_fn=f, scatter_filter_fn=fs,
                    tsweep_filter_fn=ft, title_root=title,
                    fixed_k_for_tsweep=1)
   ```

8. **Add Setup K + Setup L sections to `docs/components/c2.md`**
   AFTER the Setup G section (or Setup J if it exists by then),
   BEFORE "## Headline figure for the paper". Use the same template
   as agent_synth's Setup D/E sections — Hypothesis, Setup config
   table, Results table, plot embeds, Headline finding.

9. **Commit + push** under your `agent_pro` GitHub author identity.

### Coordination

- **agent_synth** is concurrently running Setup J (hierarchical
  K_l=50) on 8× H100 + filling tsae_paper + tfa_pos baselines on
  D/E. Don't duplicate.
- **agent_hammer** (5× RTX PRO 6000) was tagged for Setup H
  (ρ-sweep on D-np10) + Setup I (temporal-derivative) + filling
  F/G baselines. Their activity status unknown — don't duplicate
  but do check leaderboard before launching ANY cell.
- **You (agent_pro)**: Setup K + L exclusively. Do NOT touch any
  other setup's drivers, plots, or c2.md sections.
- **Leaderboard dedupe**: if you race a cell with another agent,
  the runner skips on cache hit. Only wasted GPU is on the loser.
  Always grep before launching big sweeps:
  ```bash
  grep -c "toy_anticorrelated" results/leaderboard.jsonl   # should be 0
  ```

### Watch-outs (lessons learned by agent_synth + agent_hammer)

1. **TEMP_BENCH_POD_MODE in launcher env line.** The Bash tool
   subprocess does NOT inherit `TEMP_BENCH_POD_MODE` from the
   parent shell. Always set it explicitly in the `env VAR=val cmd`
   line of the launcher. agent_hammer hit this bug; their
   recovery script lives at
   `agents/agent_hammer/push_checkpoints_to_hf.py`.
   - Use `TEMP_BENCH_POD_MODE=persistent` for the training
     launchers (auto-push DISABLED — avoids HF rate limit of
     256 commits/hour).
   - Manual HF push at session end via
     `scripts/push_synth_ckpts_to_hf.py` with retry-with-backoff
     (sketch in agent_synth's briefing § HF push state).

2. **k_pos × T ≤ d_sae=40** is the binding constraint for windowed
   archs. For T=12, k_pos ≤ 3; for T=10, k_pos ≤ 4; etc. Cap
   accordingly in your launcher (the canonical template handles this).

3. **n_steps=8000 is the right default**. Models converge well at
   that count for d_sae=40 toy data. agent_synth's 30k cells were
   killed for being too slow under multi-tenant contention; the
   `ZOOM_CUTOFF_TS` in `plot_headline.py` filters out the early
   30k cells. Don't bump above 8k unless convergence fails (it
   won't on toy d=256/d_sae=40).

4. **tsae_paper hack**: at component=c2, locked YAML has
   d_sae=16384 (no per_component override). The `fill_baselines.py`
   driver passes `arch_hparams_override={"d_sae": 40, "k_pos": k}`
   automatically — don't override in YAML, it's by design.

5. **Leaderboard merge conflicts** during multi-agent push are
   routine. Resolve by union+dedupe-by-eval_key. agent_synth's
   inline Python one-liner is the canonical resolver — see their
   briefing § Pitfalls.

6. **Don't bypass `runner.run_cell`** — the canonical pathway.

7. **Don't bump `EVAL_PROTOCOL_VERSION` for c2** (currently
   "1.0.0"). New cells just append at fresh eval_keys.

### References

- `agents/agent_synth/briefing.md` — paths, drivers, conventions,
  pitfalls. **READ THIS FIRST** for the comprehensive context.
- `agents/agent_hammer/briefing.md` — TEMP_BENCH_POD_MODE bug fix,
  HF push recovery script, c2.md ownership protocol.
- `experiments/c2_synthetic_coupled/fill_baselines.py` — the
  unified driver agent_synth wrote. You'll extend its
  `_build_data` to recognise your two new generators.
- `experiments/c2_synthetic_coupled/plot_headline.py:render_setup` —
  the 4-plot orchestrator. Use it; don't roll your own.
- `experiments/c2_hierarchical/run_setup_j.sh` — agent_synth's
  Setup J launcher template; mirror its structure.
- `src/temp_bench/data/toy/coupled.py` — `_orthogonalise`,
  `_sample_magnitudes`, `_markov_chain_batch` primitives reusable
  for new generators.
- `src/temp_bench/data/toy/hierarchical.py` — closest existing
  template for both K (one-hot variant) and L (magnitude variant).
- `temp_bench.eval.synthetic.feature_recovery` /
  `global_recovery_gAUC` — already handle `CoupledData` returns;
  no eval changes needed if your generator emits `CoupledData`.

### Open questions for Han

- **Setup K rho_g sweep?** Canonical at rho_g=0.9; should we also
  sweep rho_g ∈ {0.95, 0.99, 1.0 (deterministic dwell)} to map the
  Effect 2 landscape? Adds 3× cell count.
- **Setup L alpha sweep?** Three datasources at alpha ∈ {0.5, 1, 2}
  — useful or excess? Cleanest paper plot is one alpha; supplementary
  could include the sweep.
- **K_g, K_l sizing**: 10 + 30 = 40 exactly fills d_sae=40. Should
  we under-parameterize (d_sae > total features) or over-parameterize
  (d_sae < total features) for the headline regime? Touches
  `configs/locked_archs.yaml` (agent_paper territory) — flag if
  you want to vary.

---

## Current state (agent owns — overwrite at every compact)

**Last verified: 2026-05-07T02:50Z. Status: K + L + PHALANX + OBELISK
all COMPLETE. All four are NEGATIVE results for TXC's gAUC outshine.
Ready to commit + push.**

- Pod: 7× RTX 5090, 989 GB RAM, 224 CPUs, ephemeral.
- `git HEAD`: rebased onto origin/final (4926ae4d at last fetch).
  All conflicts (datasources.yaml, c2.md, fill_baselines.py,
  leaderboard.jsonl, manifest.jsonl) resolved via union+dedupe-by-key
  for jsonl + git's auto-merge of disjoint hunks for the others.
- All four sweeps complete (audit ✓ row counts):
  - K (anti-correlated globals): 219 rows
  - L (magmod α=1): 219 rows
  - PHALANX (period τ=8): 219 rows
  - OBELISK (rare + α=5): 219 rows
  - 219 = 12 archs × 3 seeds × per-T k_pos cap (sum across T values).
- Headline (all four NEGATIVE):
  - **Setup K**: TopK-SAE wins both AUCs (gAUC 0.833 / eAUC 0.983).
    Best TXC gAUC 0.598. DROP.
  - **Setup L (α=1)**: All archs near-zero gAUC. TXC T=2 wins eAUC
    by +0.13. DROP for headline gAUC.
  - **Setup PHALANX (τ=8)**: TXC at T ≥ τ DEGRADES (gAUC 0.054-0.057
    vs T=2's 0.124). T-SAE wins gAUC at 0.136. TXC T=2 wins eAUC by
    +0.026 over TopK-SAE. DROP for headline gAUC.
  - **Setup OBELISK (α=5, p_l=0.05)**: TXC uniformly fails gAUC at
    0.010 across all T. TopK-SAE wins gAUC + eAUC. DROP.
  - **Combined finding**: sparse temporal signals embedded in
    sparsely-firing locals do NOT favour window pooling at the toy
    d_sae=40 / n_steps=8 000 scale.
- Plots: K + L + PHALANX + OBELISK 4-plot suites in
  `experiments/c2_synthetic_coupled/plots/c2_setup_{k,l,phalanx,obelisk}_*.png`.
- c2.md sections updated for all 4 setups with headline findings.

## What I just did (agent owns — overwrite)

(Pre-2026-05-07T02:30Z work done by previous agent_pro session;
2026-05-07T02:30Z+ work done by this session.)

1. (Prev) Onboarded; authored K + L generators + smoke; ran K+L
   sweep + render + c2.md updates; authored PHALANX + OBELISK
   generators + launched the follow-up sweep.
2. (This session) Resumed at 2026-05-07T02:39Z when bash-tool poll
   `bcyendoyw` fired (final canonical c3 sweeps from prior aborted
   mission completed). Read the rewritten briefing; recognised the
   c3 → c2 reassignment.
3. Audited the 7-GPU PHALANX+OBELISK sweep: 219 + 219 cells, 12
   archs × 3 seeds × per-T k_pos cap, all complete.
4. Re-ran `agents.agent_pro.render_kl_plots` to refresh K + L +
   PHALANX + OBELISK plot suites (idempotent; 16 PNGs written).
5. Computed headline gAUC + eAUC tables per arch×T per setup
   (gauc/eauc fields in metrics dict; per-arch best across k_pos).
6. Updated `docs/components/c2.md` PHALANX + OBELISK sections with
   the headline findings + DROP rationale (no TXC gAUC outshine on
   either; combined finding written into the OBELISK section).
7. `git pull --rebase origin final` (7 commits behind: agent_synth +
   agent_filler advances). Resolved 5 conflicts:
   - jsonl files (leaderboard, manifest) via union+dedupe by primary key.
   - YAML / Python / Markdown via git's auto-merge (disjoint hunks
     from concurrent agent_synth + agent_pro work).
8. Staged C2-only deliverables: 4 toy generators, K+L+PHALANX+OBELISK
   plots (32 PNGs), 3 launcher scripts, render_kl_plots.py,
   datasources.yaml + fill_baselines.py + c2.md edits, briefing.md,
   leaderboard.jsonl, manifest.jsonl. Explicitly UNSTAGED
   `src/temp_bench/eval/probing.py` per the briefing's "don't
   commit aborted-c3 work" rule.
9. About to commit + push.

## Next action (agent owns — overwrite)

Mission is essentially done. After commit + push:

1. (Optional, if Han wants more iteration) Try further "knobs" — see
   Open Q #2 (α-sweep on Setup L) for a candidate that maps the SNR
   threshold below which gAUC fails. Adds ~144 shards (~30-45 min).
2. (Optional) An α=10 / α=20 OBELISK variant with p_l=0.10 might be
   more SNR-favorable. Higher α + slightly less rare firings.
3. (Cleanup, if asked) Revert the c3 leftovers in
   `src/temp_bench/eval/probing.py` and the
   `experiments/c3_probing_pooling_sweep/` dir to clean working
   tree.

## Don't repeat (agent owns — overwrite)

- **Don't `pgrep -af "...|..."` with `\|`.** ERE alternation needs
  `(a|b)` parens with bare `|` (no escape). The watcher in
  `run_setup_kl.sh` polls correctly; my hand-rolled background
  watchers used the wrong form initially and either self-matched
  the bash subprocess (false positive on `pgrep -af`) or returned
  zero matches with `\|`. Solution: use
  `pgrep -f "fill_baselines.*(toy_X|toy_Y)"` (no -a, ERE syntax).
- **Don't render plots before all shards complete.** The render
  pipeline is idempotent so partial renders don't corrupt anything,
  but partial plots are misleading at glance. Render once at end.
- **Don't use `TEMP_BENCH_POD_MODE=ephemeral` in the launcher.**
  The shared synth pod is ephemeral by `set_agent_env.sh`, but
  setting `TEMP_BENCH_POD_MODE=persistent` in the `env VAR=val cmd`
  line of the launcher disables auto-push to HF (avoids the 256
  commits/hour rate limit when 72 shards run in parallel). agent_synth
  + agent_hammer learned this the hard way; my launchers inherit.
- **Don't commit `src/temp_bench/eval/probing.py` or
  `experiments/c3_probing_pooling_sweep/`.** Those are leftovers
  from the aborted C3 probing-protocol mission. They're additive
  (don't break tests) but not in scope for the c2 mandate.

## Open questions for Han (agent owns — overwrite)

1. **K + L are NEGATIVE for TXC.** Setup K's TopK-SAE wins both AUCs;
   Setup L (α=1) has all archs at near-zero gAUC. Per the
   briefing's "If TXC doesn't outshine, DROP and try another knob",
   I dropped them and launched PHALANX (period-locked phase) +
   OBELISK (rare + α=5 magnitude-mod). Should K and L be removed
   from c2.md entirely (vs documented as null findings)? My current
   draft documents them as null findings + cites them as motivation
   for the next iteration. Easy to revert if you'd rather they
   not appear at all.
2. **α-sweep for Setup L?** The α=1 result suggests SNR is the
   binding constraint. Should I add an α-sweep (α ∈ {0.5, 1, 2, 5})
   to map the threshold at which TXC starts to win gAUC? Adds 4×
   datasources × 36 shards = 144 shards. OBELISK at α=5 is one
   point in this sweep already.
3. **Probe.py + c3 leftovers.** Stale from prior aborted mission.
   Leave on disk uncommitted (current state) or revert? Reverting
   loses the additive `s_tail_probe_variant` work; if a future c3
   session needs pooling exploration, the code is on this pod's
   working tree only and lost on stop. Recommend keeping
   uncommitted; revert if you want a clean working tree.
