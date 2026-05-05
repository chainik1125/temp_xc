<!--
DRAFT — written by agent_paper 2026-05-04 PM, REWRITTEN 2026-05-05 PM
to reflect the C3 MW pivot (C6 100K mission abandoned, see "Mission
pivot" below).
Section ownership: PROTOCOL.md § 14.
-->

---
agent: agent_em_100k
last_state_update: 2026-05-05T12:00:00Z
component: c3 (multi-window deployment for sparse probing)
---

## Identity + mandate (Han owns — agents do not edit)

You are **agent EM 100K** (legacy name; mission has pivoted —
see below). You own the **C3 multi-window deployment** as a helper to
agent_nlp. Your purpose: train `txc_base_mw` and `txc_pro_mw` on the
C3 sparse-probing setup so agent_nlp's headline gets MW data without
agent_nlp needing to re-run cells themselves.

Files you may edit:

- `agents/agent_em_100k/briefing.md` (your own — agent-owned sections only)
- `experiments/c3_probing_mw/` (new experiment directory you create
  with a minimal driver — see "First concrete task" below).
- Code under `src/temp_bench/` that you author + commit — but in
  practice you should not need to write any. agent_nlp's C3 plumbing
  (preloaded batch_iter, training, SAEBench+CT eval) is already wired
  and compatible with the multi-window TXC archs. Re-use via imports.

**Files that are OUT OF SCOPE — do NOT edit:**
- `agents/agent_*/` — every other agent's directory **including
  `agents/agent_nlp/` and `agents/agent_em/`**. Their briefings,
  decisions, and per-cell state are theirs.
- `experiments/c3_probing/` — agent_nlp's territory. Their `run.py`,
  `analysis.py`, evaluation pipeline. You import from there without
  modification.
- `experiments/c4_qualitative/` — agent_nlp's territory. C4 cells
  cache-hit on C3 trained checkpoints; agent_nlp will re-eval C4 once
  your MW checkpoints land.
- `experiments/c6_em/` — your previous C6 work, now agent_em's
  territory.
- `experiments/c6_em_100k/` — your OLD driver from the abandoned 100K
  mission. Leave it as-is; agent_paper may delete or repurpose later.
- `docs/components/c3.md` and `docs/components/c4.md` — agent_nlp's
  territory.
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

### ⚠️ Mission pivot 2026-05-05 — abandon C6 100K, deploy C3 MW

**Old mission (abandoned)**: replicate agent_em's C6 sweep at
`n_steps=100_000`. You completed SAE seed=42 14B-finance @ 100K
(peak_align=82.11) and the TXC seed=42 14B-finance @ 100K cell was
running, but the per-step rate was 5.9× slower than SAE on this pod
(your OQ #2 in the previous briefing). Abandon: 100K-pace work
doesn't fit the remaining sprint, and the C6 MW deployment is a
stronger paper headline than 100K convergence-test data. agent_em
themselves are taking on C6 MW after their canonical mission ends.

**Where the existing 100K work survives**:

- **SAE seed=42 14B-finance @ 100K**: row in `leaderboard.jsonl`,
  checkpoint on HF. Stays as a "what does sae_arditi look like at
  field-standard 100K tokens?" reference for the paper's caveats
  section. Useful — sae_arditi went from 78.33 (25K) to 82.11 (100K),
  showing real convergence headroom.
- **TXC seed=42 14B-finance @ 100K**: if Wang completed before this
  pivot (check the leaderboard), the row + checkpoint stay. If still
  in flight, **kill it on session start** — we're done with that mission.
- `experiments/c6_em_100k/run.py`: leave it for now. agent_paper
  may repurpose later if we ever want the convergence data.

**New mission**: train **`txc_base_mw` and `txc_pro_mw` on C3** at
the canonical schedule, write the resulting checkpoints to the
leaderboard, and let agent_nlp's `analysis.py` pick them up via
`canonical_train_keys` at paper-render time.

C3 is agent_nlp's component (sparse probing on Gemma-2-2b-IT L13).
Their canonical sweep is mostly done (topk_sae sweep finishing now);
the `txc_base_mw` and `txc_pro_mw` archs need to be trained and
evaluated to give agent_nlp's C3 headline a multi-window comparison.

**Mission scope** (n=3 paired with agent_nlp's existing canonical):

- 2 archs × 3 seeds = **6 trainings**:
  - txc_base_mw × {seed=42, seed=1, seed=2}
  - txc_pro_mw × {seed=42, seed=1, seed=2}
- Each training cell is evaluated at TWO k_feats ({5, 20}) per the
  canonical C3 protocol — that's 12 eval cells total, but eval is a
  cache-hit on the same checkpoint so only 6 unique trainings.
- C4 evaluation will cache-hit on these C3 checkpoints later;
  agent_nlp re-runs C4 evals when ready (not your job).

**Why this pod is appropriate for C3 MW (despite the C6 slowdown)**:

Your previous slowdown traced to agent_em's `_build_batch_iter` in
`experiments/c6_em/train.py` — a Python for-loop slicing T-windows
per row, ~5-10× slower than vectorized data paths. **C3 uses a
DIFFERENT data path**: agent_nlp's `preloaded_batch_iter_from_act_cache`
from `temp_bench.data.nlp.cache` (the shared helper, vectorized
torch fancy indexing, no Python loop). You inherit it via
`experiments.c3_probing.run` imports. **Expect normal H100
performance on C3** — likely 30-50 min per txc_base_mw cell, 60-100
min per txc_pro_mw cell.

Hardware: **1× H100 80GB pod, ephemeral, 240 GB system RAM, 1 TB
/workspace** (same pod you've been on for the C6 100K mission). Pod
mode `ephemeral`: HF is the source of truth, auto-push on
checkpoint save.

VRAM check: C3 at Gemma-scale (d_in=2304, d_sae=18432) MW peaks
~25-50 GB activations on H100, well within 80 GB. No mitigations
needed.

Subject + protocol (replicating agent_nlp's setup verbatim):

- Datasource: `gemma_2_2b_it_l13_fineweb_24k128`
- Architectures: **`txc_base_mw` + `txc_pro_mw`** (2 archs total).
  These are the YAML aliases per decisions.md § 14. NOT topk_sae,
  NOT tsae_paper — those are agent_nlp's per-token archs (no MW
  variant exists; their canonical cells are the comparison baseline).
- Per-component d_sae overrides: locked_archs.yaml's `txc_base_mw`
  / `txc_pro_mw` use `d_sae=18432` for c3 (mirroring the canonical
  txc_base / txc_pro). Applied automatically.
- Probing: SAEBench+CT (n=38 binary one-vs-rest tasks) per
  decisions.md § 11. agent_nlp's `experiments/c3_probing/run.py`
  pipeline + datasets are inherited via imports.
- Headline metric: probing accuracy / AUC at k_feat ∈ {5, 20}.
- `EVAL_PROTOCOL_VERSION`: inherit from agent_nlp's `c3_probing/run.py`
  (whatever they have set; do not override).

`TrainingConfig` for your cells (canonical schedule):

```python
TrainingConfig(
    batch_size=1024,
    n_steps=20_000,         # canonical, NOT 100K
    plateau_early_stop=False,
    # bricken_* defaults (False) — C3 does not use Bricken (decisions.md § 7).
)
```

**Per-cell wall-time estimate on H100**:

- `txc_base_mw`: ~30-50 min train + ~30 min probing eval × 2 k_feats
  = ~1.5-2 hr per cell
- `txc_pro_mw`: ~60-100 min train + ~30 min probing eval × 2 k_feats
  = ~2.5-3 hr per cell

3 seeds × 2 archs serial: 3 × (2 + 3) hr = ~15 hr total. Fits in
remaining sprint with margin.

Note on TXC-pro MW perf: agent_steer_100k reported MW slowness on
their TXC-pro cells (the all-pairs InfoNCE matrix scales as
(B*N)² = 100× larger at MW). For C3 at Gemma scale, the InfoNCE
matrix is (10240, 10240) × 4 bytes = 400 MB fp32, with BF16
softmax+CE around it. Should be tractable on H100; if you hit OOM
on the InfoNCE compute, reduce batch_size to 512 (effective B*N=5120
still 5× non-MW) and document the deviation in your AUTO-RESULTS
notes.

Locked decisions in scope: #1 (canonical TXCs are `txc_base_mw` /
`txc_pro_mw` per § 1's 2026-05-05 amendment), #4 (cross-branch reads),
#6 (HF repos), #7 (Bricken off for C3), #11 (SAEBench+CT task suite),
§ 12 (canonical training cfg), § 14 (multi-window deployment).

References:
- `agents/README.md` (your roster row)
- `agents/agent_nlp/briefing.md` (the canonical C3 setup you replicate)
- `docs/components/c3.md` (the canonical C3 writeup; do NOT edit)
- `experiments/c3_probing/{run.py,analysis.py}` (import from)
- `decisions.md` § 11, § 12, § 14
- `papers/are_saes_useful.md` (SAEBench reference)
- `PROTOCOL.md` § 7 (results live in state), § 8 (anti-conflict),
  § 11 (framework discipline), § 14 (briefing maintenance)

**Before you `status: complete`**: `bash scripts/wrap_up_session.sh`.
Ephemeral pod → checkpoints auto-pushed. Push artifacts via the
wrap-up script before any pod restart.

### First concrete task — kill C6 100K, write the C3 MW driver, launch

Step 0 — **kill the in-flight C6 100K processes** before doing
anything else (free GPU + RAM):

```bash
ps -ef | grep "experiments.c6_em_100k" | grep -v grep
# → kill any active PIDs
nvidia-smi --query-gpu=memory.used --format=csv
# → expect <500 MB used; if not, force-kill stragglers via `kill -9`
```

Step 1 — `git pull --rebase origin final`, verify the MW arch is
registered + agent_nlp's plumbing is current:

```bash
.venv/bin/python -c "from temp_bench.config import load_arch; print(load_arch('txc_base_mw').hparams)"
# → expect a dict containing multi_window=True
.venv/bin/python -c "from experiments.c3_probing.run import EVAL_PROTOCOL_VERSION; print(EVAL_PROTOCOL_VERSION)"
# → use whatever agent_nlp has set
```

Step 2 — pull the Gemma activation cache from HF if not already on
disk:

```bash
ls results/act_cache/*/acts.npy 2>/dev/null
bash scripts/sync_from_hf.sh   # if not present
```

Step 3 — write `experiments/c3_probing_mw/__init__.py` (empty) and
`experiments/c3_probing_mw/run.py`:

```python
"""C3 multi-window deployment driver — replicates agent_nlp's setup
with txc_base_mw / txc_pro_mw archs.

Re-uses agent_nlp's plumbing verbatim via imports. Only the arch
names differ; the data path, training, and probing eval are
unchanged.
"""
from __future__ import annotations
import argparse
from temp_bench import runner
from experiments.c3_probing.run import (
    DATASOURCE,
    EVAL_PROTOCOL_VERSION,
    my_train_fn,
    my_eval_fn,
    _real_training_cfg as _orig_training_cfg,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+",
                    default=["txc_base_mw", "txc_pro_mw"],
                    choices=["txc_base_mw", "txc_pro_mw"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    ap.add_argument("--k-feats", nargs="+", type=int, default=[5, 20])
    args = ap.parse_args()

    cfg = _orig_training_cfg()    # batch=1024, n_steps=20_000, plateau_off

    for arch in args.archs:
        for seed in args.seeds:
            for k in args.k_feats:
                print(f"[c3_mw] cell arch={arch} seed={seed} k_feat={k}")
                runner.run_cell(
                    component="c3",
                    arch_name=arch,
                    seed=seed,
                    datasource_name=DATASOURCE,
                    training_cfg=cfg,
                    eval_cfg={"k_feat": k, "S": 32},
                    eval_protocol_version=EVAL_PROTOCOL_VERSION,
                    train_fn=my_train_fn,
                    eval_fn=my_eval_fn,
                )


if __name__ == "__main__":
    main()
```

(Adapt `eval_cfg` to whatever agent_nlp's `c3_probing/run.py` uses.
Their canonical cells have specific keys; mirror them so your evals
go through the same path.)

Step 4 — smoke ONE cell at `n_steps=200`:

```bash
TQDM_DISABLE=1 .venv/bin/python -c "
from temp_bench.schemas import TrainingConfig
from temp_bench import runner
from experiments.c3_probing.run import DATASOURCE, EVAL_PROTOCOL_VERSION, my_train_fn, my_eval_fn
result = runner.run_cell(
    component='c3', arch_name='txc_base_mw', seed=42,
    datasource_name=DATASOURCE,
    training_cfg=TrainingConfig(n_steps=200, batch_size=1024, plateau_early_stop=False),
    eval_cfg={'k_feat': 5, 'S': 32},
    eval_protocol_version=EVAL_PROTOCOL_VERSION,
    train_fn=my_train_fn, eval_fn=my_eval_fn,
)
print('smoke result:', result.train_key, result.eval_key, result.cached)
"
```

Step 5 — launch the full sweep:

```bash
TQDM_DISABLE=1 AGENT_NAME=agent_em_100k \
  .venv/bin/python -m experiments.c3_probing_mw.run \
  > logs/c3_mw_full.log 2>&1 &
echo $! > /tmp/p_c3_mw
```

Step 6 — monitor + verify rows land at `arch=txc_base_mw` /
`txc_pro_mw` with the canonical `eval_protocol_version`. agent_paper
integrates via `canonical_train_keys()` toggle at paper-render time.

---

## Current state (agent owns — overwrite at every compact)

**Last verified: 2026-05-05T12:00Z (mission pivot — directive received from
Han / agent_paper). C6 100K mission abandoned. New mission: deploy
multi-window TXC archs at C3 (sparse probing) for agent_nlp.**

- `git HEAD`: at or after `cad94382` (decisions.md § 14 Bricken caveat)
  + `ecc4c661` (txc_base_mw / txc_pro_mw YAML aliases). Pull on session
  start to pick up the latest.
- Pod: 1× H100, ephemeral, 240 GB RAM. `/workspace/temp_xc/` clone.
  This is the SAME pod you've been on; the mission shifts but the
  pod stays.
- In flight (TO BE KILLED on session start): C6 100K cells from the
  abandoned mission. Identify via `ps -ef | grep c6_em_100k`.
- 100K artifacts that survive:
  - SAE seed=42 14B-finance @ 100K cell + checkpoint. In leaderboard,
    HF-pushed. Stays as a paper-caveats reference point.
  - TXC seed=42 14B-finance @ 100K: if it landed, also stays.
- Last leaderboard append: from the 100K mission (sae_arditi 100K).
- Recent decisions in scope: `decisions.md` § 1 (canonical TXCs are
  `txc_base_mw` / `txc_pro_mw` going forward), § 7 (Bricken off for
  C3), § 11 (SAEBench+CT task suite), § 12 (canonical training cfg),
  § 14 (multi-window deployment).

## What I just did (agent owns — overwrite)

- 2026-05-05T12:00Z: agent_paper rewrote this briefing per Han's
  pivot directive. C6 100K mission abandoned; new mission is C3 MW
  helper for agent_nlp (6 trainings + 12 evals at canonical 20K).

(Overwrite this section with your own actions when you start.)

## Next action (agent owns — overwrite)

1. `cd $(git rev-parse --show-toplevel)/purified`
2. `source scripts/set_agent_env.sh agent_em_100k`
3. `bash scripts/agent_smoke_test.sh` — expect 131/131 + preflight green.
4. `git pull --rebase origin final`.
5. **Kill in-flight C6 100K processes** per "First concrete task" Step 0.
6. Verify MW arch + agent_nlp's c3_probing module per Step 1.
7. Sync Gemma cache per Step 2 (if not already on disk).
8. Write `experiments/c3_probing_mw/run.py` per Step 3.
9. Smoke-test ONE cell at `n_steps=200` per Step 4.
10. Launch the full sweep per Step 5.
11. Monitor + verify leaderboard rows.

## Don't repeat (agent owns — overwrite)

- **Don't run anything at `n_steps=100_000`** for this mission. The
  100K convergence test is abandoned per Han 2026-05-05; canonical
  schedule (n_steps=20_000) is the only target.
- **Don't edit `experiments/c3_probing/`** — agent_nlp's territory.
  Import only.
- **Don't edit `experiments/c6_em/` or `experiments/c6_em_100k/`** —
  no longer your active component.
- **Don't edit `docs/components/c3.md` or `c4.md`** — agent_nlp's
  territory.
- **Don't bypass `runner.run_cell`** — the call goes through the
  canonical pathway (which appends to `leaderboard.jsonl`).
- **Don't allocate `train_key` / `eval_key` manually** — the runner
  computes them deterministically.
- **Don't include topk_sae or tsae_paper in your archs list** — those
  are agent_nlp's per-token archs (no MW variant exists; their
  canonical cells are the comparison baseline).
- **Don't enable Bricken** — C3 is Bricken-off per decisions.md § 7.
- **Don't push to HF manually** — `cache.save_checkpoint` does it on
  ephemeral pods.

## Open questions for Han (agent owns — overwrite)

(None at briefing-rewrite time. Surface anything that comes up
during the kill-100K step or smoke test — especially if the
txc_pro_mw InfoNCE OOMs at C3 scale, since agent_steer_100k reported
similar slowness on TXC-pro MW.)
