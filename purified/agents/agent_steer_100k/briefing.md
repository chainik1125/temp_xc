<!--
DRAFT — written by agent_paper 2026-05-04 PM, REWRITTEN 2026-05-05 PM
to reflect the C7 MW pivot (the C5 MW mission turned out to be too
slow on this pod's H100 — a CPU-bandwidth bottleneck flagged in
commit `e7b229fd`. Pivoting to C7 MW where there's a coverage gap
and the H100's VRAM is genuinely needed).
Section ownership: PROTOCOL.md § 14.
-->

---
agent: agent_steer_100k
last_state_update: 2026-05-05T13:00:00Z
component: c7 (multi-window deployment for backtracking)
---

## Identity + mandate (Han owns — agents do not edit)

You are **agent STEER 100K** (legacy name; mission has pivoted twice
now — see below). You own the **C7 multi-window deployment** as a
helper to agent_back. Your purpose: train and evaluate `txc_base_mw`
and `txc_pro_mw` on the C7 backtracking setup so agent_back's headline
gets MW data without re-running their A40 sweep.

Files you may edit:

- `agents/agent_steer_100k/briefing.md` (your own — agent-owned sections only)
- `experiments/c7_backtracking_mw/` (new experiment directory you create
  with a minimal driver — see "First concrete task" below).
- Code under `src/temp_bench/` that you author + commit — but in
  practice you should not need to write any. agent_back's C7 plumbing
  (Stage A traces, sentence-acts cache, mined features, SteeringHook
  with DoM-base-union normalization, Sonnet judge, Δgc + PR-AUC eval)
  is already wired and compatible with the multi-window TXC archs
  (the per-arch decoder-norm rescale happens inside the case-study
  code, not in arch code — verified 2026-05-05 PM).

**Files that are OUT OF SCOPE — do NOT edit:**
- `agents/agent_*/` — every other agent's directory **including
  `agents/agent_back/` and `agents/agent_steer/`**. Their briefings,
  decisions, and per-cell state are theirs.
- `experiments/c7_backtracking/` — agent_back's territory. Their
  `run.py`, `analysis.py`, `smoke.py`. You import from there without
  modification.
- `experiments/c5_steering/`, `experiments/c5_steering_mw/`,
  `experiments/c5_steering_filler/` — agent_steer / agent_filler /
  prior-mission territories.
- `docs/components/c7.md` — agent_back's territory. agent_paper
  integrates results at paper-render time.
- `docs/paper/*` — agent_paper's territory.
- `agents/agent_paper/decisions.md` — global decisions log.
- `agents/README.md` — roster/contract; agent_paper-owned.
- `configs/locked_archs.yaml` — only agent_paper edits this.
- `pyproject.toml` and `uv.lock` — atomic, agent_paper coordinates.

If you find yourself wanting to edit any out-of-scope file, **STOP**.
Add a bullet to "Open questions for Han" in your own briefing,
surface it in chat, and let Han or agent_paper land the change. Even
if Han verbally approves, do not commit cross-territory edits yourself.
This is non-negotiable — see PROTOCOL.md § 8 + CLAUDE.md Hard Rule #7.

### ⚠️⚠️⚠️ STAND DOWN — MW pivot RESCINDED 2026-05-05 PM ⚠️⚠️⚠️

**Han + agent_paper diagnosed that the MW pivot was solving a misframed
problem.** SAEBench (papers/are_saes_useful.md, App. B) shows canonical
SAE training is buffer-based, batch=2048 TOKENS/step, ~500M tokens
total — per-step token throughput on the order of 10³, not 10⁵.

Our two patterns at this paper:

| Component | Pattern                             | per-token tokens/step |
|---|---|---:|
| C3, C4, C5 | sequence-based (B sentences × all 128 positions) | **131,072** — 5× over SAEBench's 2K canonical |
| C6, C7    | window-based (B sentences × T positions, T=1 for SAE) | **1,024** — close to SAEBench canonical |

C3/C5's 131K is OVER-batched per-step; C6/C7's 1K is near-canonical.
The earlier "TXC has 25× FLOPs disadvantage" framing was directionally
true at C3/C5, but at C6 + C7 the per-token baselines were *already*
at literature scale, so an MW deployment there would OVER-correct.

**Han's call (2026-05-05 PM)**: ABORT all 4 MW pivots. C3 + C5
per-token baselines re-train at T=1; C6 + C7 are unchanged
(agent_em's canonical C6 sweep + agent_back's canonical C7 sweep stay
as the paper headlines).

**Your specific abort actions** (do these in this order):

1. **Kill any in-flight C7 MW processes.** If you launched
   `experiments/c7_backtracking_mw/run.py`, kill it. Also kill any
   leftover C5 MW processes from the earlier (already-abandoned) C5
   pivot.
   ```bash
   pkill -TERM -f "experiments.c7_backtracking_mw" || true
   pkill -TERM -f "experiments.c5_steering" || true
   sleep 2
   pkill -KILL -f "experiments.c7_backtracking_mw" || true
   pkill -KILL -f "experiments.c5_steering" || true
   nvidia-smi --query-gpu=memory.used --format=csv
   # → expect <500 MB; if not, force-kill stragglers
   ```
   Any landed C7 MW rows (none expected if you didn't smoke yet) stay
   — `canonical_train_keys` filters them out at paper-render time,
   harmless. The 1 C5 MW cell that landed earlier
   (`eval_key=963df9c69213f998`) also stays.
2. **Status: idle.** Do NOT launch further MW work on this pod. Your
   pod's CPU-bandwidth issue (briefing commit `e7b229fd`) makes
   parallel sweeps non-productive anyway; agent_back's canonical C7
   sweep + agent_em's canonical C6 sweep cover the paper-bearing
   results.
3. **No re-purpose for you.** Use remaining session time for
   paper-writing contributions to c7.md (caveats, methodology
   bullets) if helpful, but compute work is done. Then
   `bash scripts/wrap_up_session.sh` and HF backup.
4. **What replaces MW for C7 fairness** is the recognition that C7's
   per-token baselines were always at literature scale — agent_back's
   canonical sweep is already apples-to-apples with TXC at the
   per-step training-FLOPs axis. No re-train needed at C7.

**The C7 MW deployment directive below this line is RESCINDED.** Do
not read it as actionable. Left in place for git provenance only.

---

### ⚠️ Mission pivot 2026-05-05 PM — abandon C5 MW, deploy C7 MW [RESCINDED 2026-05-05 PM — see STAND DOWN above]

**Old mission (this morning, abandoned)**: pivot from the original
100K convergence-test mission to **C5 MW** with `txc_base_mw +
txc_pro_mw × {42,1,2}`. You launched the sweep, smoke-tested
successfully, and landed 1 cell (`txc_base_mw` seed=42 C5,
eval_key=`963df9c69213f998`). But your subsequent reports flagged a
CPU-bandwidth bottleneck on this H100 pod (commit `e7b229fd`) that
made remaining C5 MW cells too slow to fit the sprint window — and
agent_filler launched on a fresh 8× A40 pod precisely to take over
the C5 MW sweep with parallel-cell speedup.

**New mission**: deploy `txc_base_mw + txc_pro_mw` at **C7
backtracking**, the only paper-bearing component without an MW
deployment yet. agent_back's canonical C7 v4 sweep is in flight on
the A40 pod (GPUs 0+2) and will land their non-MW reproducibility
result; your H100 pod adds the MW variant on the same Stage A
infrastructure.

**Why C7 needs the H100**: agent_back's pod is A40 48GB. C7 archs at
Llama scale (d_in=4096, d_sae=32768) under MW peak ~50-75 GB
activations — would push past A40's 48 GB cap without bf16 forcing
or batch reductions. **Your H100 80GB fits MW cleanly with margin**;
this is the natural place for the MW deployment without the VRAM
mitigations agent_back would otherwise need.

**Why C7 isn't the "slow C5 MW" trap repeating**: TXC-pro MW's
slowness (the all-pairs InfoNCE matrix scaling as (B*N)²) is real
but C7 is a **2-cell sweep** matching agent_back's reduced
seed=42-only budget — even at the slow per-step rate, 2 cells fit in
~12-18 hr wall. (For C5 you had 6 cells to run; the same
slow-per-step rate × 6 cells was the budget killer.)

**The 1 C5 MW cell that landed stays in the leaderboard**
(`eval_key=963df9c69213f998`). agent_filler's parallel sweep covers
C5 MW going forward. Your work for C5 is done.

### Mandate — C7 multi-window deployment, 2 cells

agent_paper landed `txc_base_mw` and `txc_pro_mw` as separate arch
identities in `configs/locked_archs.yaml` (decisions.md § 14, commit
`ecc4c661`). They are YAML aliases of TXCBase / TXCPro respectively,
with `multi_window: true` baked into hparams. Same Python classes;
only per-step sampling differs.

**Mission scope** (matches agent_back's reduced seed budget on A40):

- 2 archs × 1 seed = **2 cells**:
  - `txc_base_mw` × seed=42
  - `txc_pro_mw` × seed=42
- agent_back's canonical sweep covers TopK / Stacked / TFA / T-SAE /
  TXC-base / TXC-pro / MLC at non-MW. Your MW cells provide the TXC
  apples-to-apples parity comparison; per-token archs stay at
  agent_back's canonical (no MW variant exists).

Hardware: **1× H100 80GB pod, ephemeral, 240 GB system RAM, 1 TB
/workspace** (your existing pod). Pod mode `ephemeral`: HF is the
source of truth, auto-push on checkpoint save, fatal on push failure.

VRAM check (per agent_paper's analysis): C7 at Llama-scale (d_in=4096,
d_sae=32768) MW peaks ~50-75 GB activations. H100 80GB fits with
~5-10 GB margin. Use `precision="bf16"` (default for H100). If you
hit OOM on TXC-pro MW (worst case — matryoshka + InfoNCE all-pairs),
reduce batch_size to 512 first; effective B*N=5120 still > non-MW
B=1024.

Subject + protocol (replicating agent_back's setup verbatim with the
multi-window arch swap):

- Datasource: `llama_3_1_8b_base_l10_ward_nousmirror` (agent_back's
  `DATASOURCE` default). Per-arch d_sae=32768 from the
  `per_component_hparams.c7` overrides on `txc_base_mw` /
  `txc_pro_mw` — applied automatically. The mirror is byte-identical
  to the Meta-canonical Llama-3.1-8B (verified by agent_back commit
  `52b08b35`, max_abs_diff=0); cache key matches.
- Stage A: cohort + sentence-acts cache built by agent_back, on HF.
  Inherit via `experiments.c7_backtracking.run` imports.
- Mining: `mine_top_features` selects the top feature per arch by
  selectivity. **Decoder-norm normalization** to
  `||DoM_base_union||=0.4140` happens inside
  `src/temp_bench/case_studies/backtracking.py:1514-1524` —
  cross-arch magnitude comparability is automatic. (Verified
  2026-05-05 PM by agent_paper after agent_back's "false alarm"
  diagnostic note in commit `8098d137`.)
- Magnitude grid: 25 points
  `[-16,-12,-10,-8,-7,-6,-5,-4,-3,-2,-1,-0.5, 0, 0.5,1,2,3,4,5,6,7,8,10,12,16]`.
- Cut protocol: cut25 (cut Stage A trace at 25% of unsteered length).
- Inducement metric (primary): peak Δgc — Sonnet 4.6 judge counts
  genuine backtracking events.
- Detection metric: PR-AUC at S ∈ {1, 2, 4, 8, 16, 32}.
- Judge: Sonnet 4.6. Per-call `judge_outputs.jsonl` persistence.
- `EVAL_PROTOCOL_VERSION`: inherit from agent_back's `c7_backtracking/run.py`.

`TrainingConfig` for your cells (canonical schedule, matches
agent_back's v4 directive):

```python
TrainingConfig(
    batch_size=1024,
    n_steps=20_000,         # canonical; matches agent_back's 20K override
    plateau_early_stop=False,
    # bricken_* defaults (False) — C7 does not use Bricken (decisions.md § 7).
)
```

**Per-cell wall-time estimate on H100**:

- `txc_base_mw`: ~60-90 min train (Llama-scale MW encoder/decoder
  bigger than Gemma-scale by 4096/2304 ≈ 1.8× per dimension axis)
  + ~2 hr eval (Stage A traces × magnitude grid × 2 phases × judge)
  = ~3-3.5 hr per cell.
- `txc_pro_mw`: ~3-4 hr train (matryoshka + multi-distance
  contrastive + all-pairs InfoNCE at B*N=10240) + ~2 hr eval
  = ~5-6 hr per cell.

Total serial: ~9-10 hr. Fits in remaining sprint window.

Locked decisions in scope: #1 (canonical TXCs are `txc_base_mw` /
`txc_pro_mw` per § 1's 2026-05-05 amendment), #4 (cross-branch reads),
#6 (HF repos), #7 (Bricken off for C7), #11 (T-SAE = paper-faithful
Ye et al. — but TSAE has no MW variant, only TXC archs do), § 12
(canonical training cfg), § 14 (multi-window deployment).

References:
- `agents/README.md` (your roster row)
- `agents/agent_back/briefing.md` (the canonical C7 setup you replicate
  — read this BEFORE launching, especially the "decoder-norm false
  alarm" note in their `Failures handled` section so you don't
  re-investigate the same trap)
- `docs/components/c7.md` (the canonical C7 writeup; do NOT edit)
- `experiments/c7_backtracking/{run.py,analysis.py,smoke.py}` —
  agent_back's plumbing (import from)
- `decisions.md` § 12, § 14
- `papers/backtracking.md` (Ward et al. 2025 reference)
- `PROTOCOL.md` § 7, § 8, § 11, § 14

**Before you `status: complete`**: `bash scripts/wrap_up_session.sh`.
Ephemeral pod → checkpoints auto-pushed. Push your MW cells'
`results/runs/<eval_key>/judge_outputs.jsonl` and metrics via the
wrap-up script before any pod restart.

### First concrete task — kill C5 MW, write C7 MW driver, launch

Step 0 — **kill the in-flight C5 MW processes** before doing anything
else (free GPU + RAM):

```bash
ps -ef | grep "experiments.c5_steering" | grep -v grep
# → kill any active PIDs from the C5 MW sweep (e.g., from /tmp/p_c5_mw)
nvidia-smi --query-gpu=memory.used --format=csv
# → expect <500 MB used; if not, force-kill stragglers via `kill -9`
```

Step 1 — `git pull --rebase origin final`, then verify the MW arch
is registered + agent_back's plumbing is current:

```bash
.venv/bin/python -c "from temp_bench.config import load_arch; print(load_arch('txc_base_mw').hparams); print(load_arch('txc_pro_mw').hparams)"
# → expect dicts containing multi_window=True, c7-level d_sae=32768
.venv/bin/python -c "from experiments.c7_backtracking.run import EVAL_PROTOCOL_VERSION, DATASOURCE; print(EVAL_PROTOCOL_VERSION, DATASOURCE)"
# → DATASOURCE should be llama_3_1_8b_base_l10_ward_nousmirror
```

Step 2 — pull the Llama BASE L10 activation cache + sentence-acts
cache from HF:

```bash
ls results/act_cache/*/resid_post_L10.npy 2>/dev/null
ls results/c7_backtracking/stage_a/sentence_acts_L10.npz 2>/dev/null
bash scripts/sync_from_hf.sh   # if not present
```

Step 3 — write `experiments/c7_backtracking_mw/__init__.py` (empty)
and `experiments/c7_backtracking_mw/run.py`. Key design: import as
much as possible from agent_back's `c7_backtracking/run.py`, override
only the batch_iter (full-sequence instead of pre-windowed) and the
arch list.

agent_em already wrote a similar driver for C6 (commit `03facd49`,
`experiments/c6_em_mw/run.py`) — reference it for the shape of the
full-seq batch_iter pattern.

Sketch:

```python
"""C7 multi-window deployment driver.

Replicates agent_back's setup with two changes:
1. arch_name: txc_base / txc_pro → txc_base_mw / txc_pro_mw
   (multi_window=True in hparams)
2. batch_iter: returns FULL (B, seq_len, d) sequences instead of
   pre-windowed (B, T, d). MW's train_step does the tiling.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch

from temp_bench import runner
from temp_bench.schemas import TrainingConfig
from temp_bench.config import act_cache_dir, load_arch, load_datasource, compute_act_cache_key
from temp_bench.training.sae_trainer import train_sae

# Re-use agent_back's plumbing where possible:
from experiments.c7_backtracking.run import (
    DATASOURCE,
    EVAL_PROTOCOL_VERSION,
    my_eval_fn,                # eval is unchanged for MW
    _spec_window_size,         # if needed
)

# Module-global cache for full-sequence preload — shared across cells
# in the same process.
_PRELOADED_C7_FULL: dict[str, torch.Tensor] = {}


def _build_full_seq_batch_iter(act_cache_key: str, *, seed: int = 42):
    """MW data path: returns (B, seq_len, d_in) full sequences. The
    arch's train_step tiles into (B*N, T, d_in) windows internally.
    """
    cache_dir = act_cache_dir(act_cache_key)
    cache_path = str(cache_dir / "resid_post_L10.npy")     # NOT acts.npy — agent_back's convention
    if cache_path not in _PRELOADED_C7_FULL:
        mmapped = np.load(cache_path, mmap_mode="r")
        _PRELOADED_C7_FULL[cache_path] = (
            torch.from_numpy(np.ascontiguousarray(mmapped)).clone()
        )
    acts = _PRELOADED_C7_FULL[cache_path]
    rng = np.random.default_rng(seed)
    def batch_iter(n: int) -> torch.Tensor:
        idx = rng.integers(0, acts.shape[0], size=n)
        return acts[idx].to(torch.float32)
    return batch_iter


def my_train_fn_mw(*, arch_name, arch_hparams, seed, training_cfg,
                   act_cache_key, component):
    """C7 MW train_fn: instantiate arch from YAML spec + datasource d_in,
    use full-sequence batch_iter, call canonical train_sae."""
    spec = load_arch(arch_name, component=component)
    ds = load_datasource(DATASOURCE)
    d_in = ds.d_model      # or however agent_back resolves d_in; see their my_train_fn
    from temp_bench.config import instantiate_arch
    model = instantiate_arch(spec, d_in=d_in)
    # bf16 cast for >1B archs (agent_back's commit 9cfd99df pattern)
    if sum(p.numel() for p in model.parameters()) > 1_000_000_000:
        model = model.to(torch.bfloat16)
    batch_iter = _build_full_seq_batch_iter(act_cache_key, seed=seed)
    return train_sae(
        model=model, batch_iter=batch_iter,
        training_cfg=training_cfg, device="cuda",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+",
                    default=["txc_base_mw", "txc_pro_mw"],
                    choices=["txc_base_mw", "txc_pro_mw"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    args = ap.parse_args()

    cfg = TrainingConfig(batch_size=1024, n_steps=20_000,
                         plateau_early_stop=False)

    for arch in args.archs:
        for seed in args.seeds:
            print(f"[c7_mw] cell arch={arch} seed={seed} "
                  f"eval_protocol={EVAL_PROTOCOL_VERSION}")
            runner.run_cell(
                component="c7",
                arch_name=arch,
                seed=seed,
                datasource_name=DATASOURCE,
                training_cfg=cfg,
                eval_cfg={"sweep": "c7_mw_v1"},
                eval_protocol_version=EVAL_PROTOCOL_VERSION,
                train_fn=my_train_fn_mw,
                eval_fn=my_eval_fn,
            )


if __name__ == "__main__":
    main()
```

(Adapt `my_train_fn_mw` from agent_back's `my_train_fn` in
`c7_backtracking/run.py` — same model instantiation, bf16 cast for
>1B archs, but swap their `_build_batch_iter` for the full-seq one
above. Eval pipeline is unchanged — `my_eval_fn` covers Stage A,
mining, magnitude sweep, judge calls, PR-AUC.)

Step 4 — smoke ONE cell at `n_steps=200`:

```bash
TQDM_DISABLE=1 .venv/bin/python -c "
from temp_bench.schemas import TrainingConfig
from temp_bench import runner
from experiments.c7_backtracking_mw.run import my_train_fn_mw
from experiments.c7_backtracking.run import DATASOURCE, EVAL_PROTOCOL_VERSION, my_eval_fn
result = runner.run_cell(
    component='c7', arch_name='txc_base_mw', seed=42,
    datasource_name=DATASOURCE,
    training_cfg=TrainingConfig(n_steps=200, batch_size=1024, plateau_early_stop=False),
    eval_cfg={'sweep': 'c7_mw_smoke'},
    eval_protocol_version=EVAL_PROTOCOL_VERSION,
    train_fn=my_train_fn_mw, eval_fn=my_eval_fn,
)
print('smoke result:', result.train_key, result.eval_key, result.cached)
"
```

Step 5 — launch the 2-cell sweep:

```bash
TQDM_DISABLE=1 AGENT_NAME=agent_steer_100k \
  .venv/bin/python -m experiments.c7_backtracking_mw.run \
  --archs txc_base_mw txc_pro_mw --seeds 42 \
  > logs/c7_mw_full.log 2>&1 &
echo $! > /tmp/p_c7_mw
```

Step 6 — monitor + verify rows land at `arch=txc_base_mw` /
`txc_pro_mw` with `component=c7` and the canonical
`eval_protocol_version`. agent_paper integrates via
`canonical_train_keys()` toggle at paper-render time.

---

## Current state (agent owns — overwrite at every compact)

**Last verified: 2026-05-05T13:00Z (mission pivot — directive received from
Han / agent_paper). C5 MW mission abandoned (CPU-bandwidth bottleneck;
agent_filler's parallel sweep covers C5 MW). New mission: deploy
multi-window TXC archs at C7 backtracking for agent_back.**

- `git HEAD`: at or after `8098d137` (agent_back's decoder-norm false
  alarm). Pull on session start.
- Pod: 1× H100, ephemeral, 240 GB RAM. `/workspace/temp_xc/` clone.
  This is the SAME pod you've been on; the mission shifts but the
  pod stays.
- In flight (TO BE KILLED on session start): C5 MW cells from the
  abandoned mission. Identify via `ps -ef | grep c5_steering`.
- C5 MW cell that landed (kept as bonus data):
  `txc_base_mw` seed=42 C5, `eval_key=963df9c69213f998`. agent_paper
  / agent_steer can use this for diff against agent_filler's parallel
  sweep when both complete.
- Last leaderboard append: from C5 MW (the 1 cell above).
- Recent decisions in scope: `decisions.md` § 1 (canonical TXCs are
  `txc_base_mw` / `txc_pro_mw` going forward), § 7 (Bricken off for
  C7), § 12 (canonical training cfg), § 14 (multi-window deployment).

## What I just did (agent owns — overwrite)

- 2026-05-05T13:00Z: agent_paper rewrote this briefing per Han's
  pivot directive. C5 MW mission abandoned (slow per-step on this
  H100; agent_filler covers C5 MW on 8× A40 in parallel). New
  mission is C7 MW helper for agent_back (2 cells at canonical 20K).

(Overwrite this section with your own actions when you start.)

## Next action (agent owns — overwrite)

1. `cd $(git rev-parse --show-toplevel)/purified`
2. `source scripts/set_agent_env.sh agent_steer_100k`
3. `bash scripts/agent_smoke_test.sh` — expect 131/131 + preflight green.
4. `git pull --rebase origin final`.
5. **Kill in-flight C5 MW processes** per "First concrete task" Step 0.
6. Verify MW arch + agent_back's c7_backtracking module per Step 1.
7. Sync Llama cache + sentence-acts cache per Step 2 (if not already
   on disk).
8. Write `experiments/c7_backtracking_mw/run.py` per Step 3. Reference
   agent_em's `c6_em_mw/run.py` (commit `03facd49`) for the
   full-sequence batch_iter + my_train_fn_mw pattern.
9. Smoke-test ONE cell at `n_steps=200` per Step 4.
10. Launch the 2-cell sweep per Step 5.
11. Monitor + verify leaderboard rows.

## Don't repeat (agent owns — overwrite)

- **Don't run anything at `n_steps=100_000`** for this mission. The
  100K convergence test was abandoned earlier; canonical schedule
  (n_steps=20_000) is the only target.
- **Don't run more C5 MW cells** — that mission is now agent_filler's
  on the 8× A40 pod. Your existing 1 cell stays in the leaderboard
  as bonus diff data.
- **Don't edit `experiments/c7_backtracking/`** — agent_back's
  territory. Import only.
- **Don't edit `experiments/c5_steering*/` or `c6_em*/`** — other
  agents' territories.
- **Don't edit `docs/components/c7.md`** — agent_back's territory.
- **Don't bypass `runner.run_cell`** — the call goes through the
  canonical pathway (which appends to `leaderboard.jsonl`).
- **Don't allocate `train_key` / `eval_key` manually** — the runner
  computes them deterministically.
- **Don't include topk_sae / tsae_paper / mlc / tfa / stacked_sae in
  your archs list** — those are agent_back's per-token archs (no MW
  variant exists; their canonical cells are the comparison baseline).
- **Don't enable Bricken** — C7 is Bricken-off per decisions.md § 7.
- **Don't push to HF manually** — `cache.save_checkpoint` does it on
  ephemeral pods.
- **Don't investigate the per-arch decoder norms** — agent_back
  already verified normalization is in place (commit `8098d137`,
  case_studies/backtracking.py:1514-1524 rescales every steering
  vector to ||DoM_base_union||=0.4140 inside the eval pipeline). The
  raw `model.decoder_directions().norm()` varies 8× across archs
  but the rescaling makes magnitudes comparable.

## Open questions for Han (agent owns — overwrite)

(None at briefing-rewrite time. Surface anything that comes up
during the kill-C5-MW step or smoke test — especially if the
txc_pro_mw OOMs at C7 scale despite the H100's 80 GB.)
