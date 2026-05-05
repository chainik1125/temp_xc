<!--
DRAFT — written by agent_paper 2026-05-04 PM, REWRITTEN 2026-05-05 PM
to reflect the C3 MW pivot (C6 100K mission abandoned, see "Mission
pivot" below).
Section ownership: PROTOCOL.md § 14.
-->

---
agent: agent_em_100k
last_state_update: 2026-05-05T12:17:00Z
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
true, but the FIX was wrong: the right move is to bring per-token
baselines DOWN to T=1 window-based (matching SAEBench + matching C6/C7),
NOT to bring TXC up via MW.

**Han's call (2026-05-05 PM)**: ABORT all 4 MW pivots. Re-train per-token
baselines at C3 + C5 with the T=1 window-based pattern.

**Your specific abort actions** (do these in this order):

1. **Kill the in-flight C3 MW sweep.** Last briefing said PID 17963,
   started 2026-05-05T12:16Z, both archs (txc_base_mw + txc_pro_mw) ×
   seed=42 × k_feats {5, 20}. Burning ~4 hr/cell on the wrong direction.
   ```bash
   ps -ef | grep "experiments.c3_probing_mw" | grep -v grep
   # → kill -TERM <PID>; if uncooperative, kill -9
   pkill -TERM -f "experiments.c3_probing_mw" || true
   nvidia-smi --query-gpu=memory.used --format=csv
   # → expect <500 MB; force-kill stragglers via `kill -9 <PID>`
   ```
   Smoke row at `train_key=e0ff471f7ddac586` (n_steps=200) stays in the
   leaderboard — `canonical_train_keys` filters it out at paper-render
   time, harmless.
2. **Set status: idle, awaiting re-purpose.** Do NOT launch any further
   MW work. Update Current state in this briefing to reflect "STOOD
   DOWN — awaiting C3 baseline re-train directive."
3. **Your next mission (when re-purposed) will be C3 baseline T=1
   re-train.** agent_paper is landing the framework change
   (`train_window_size: int | None` on `preloaded_batch_iter_from_act_cache`
   + `TrainingConfig`) and will rewrite this briefing to redirect you
   to the C3 baseline re-train (TopK_SAE + T-SAE × 3 seeds × 2 k_feats
   with `TrainingConfig(train_window_size=1)`). Wait for the briefing
   rewrite before resuming. ETA on the rewrite: same session — should
   be fresh by the time you read this if you `git pull` again.

**The MW pivot directive below this line is RESCINDED.** Do not read
it as actionable. Left in place for git provenance only.

---

### ⚠️ Mission pivot 2026-05-05 — abandon C6 100K, deploy C3 MW [RESCINDED 2026-05-05 PM — see STAND DOWN above]

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

**Last verified: 2026-05-05T12:17Z. C3 MW seed=42 sweep IN FLIGHT (PID
17963). Both archs (txc_base_mw + txc_pro_mw) × seed=42 × k_feats {5,
20}, training underway. ETA ~8-10 hr per actual smoke timing.**

- `git HEAD`: `305cf279` (`final`).
- Pod: 1× H100 80GB, ephemeral, 2 TB RAM (briefing said 240 GB).
- Active GPU: 36 GB used during training, bouncing 0-100% util.

### Training rate disappointment vs briefing

Smoke (txc_base_mw seed=42 k=5 n_steps=200): **1.35 steps/sec**.
For 20K real cell → **~4 hr train**. The briefing predicted "30-50
min per txc_base_mw cell" → actual is ~5× slower. Likely cause:
preloaded vectorized data path is fast for slicing but the per-step
forward + TopK + d_sae=18432 update at MW (T*B=5120 effective batch)
is heavy. Expect txc_pro_mw similar or slower (InfoNCE on
(5120, 5120) matrix).

### Sweep scope decision (incremental, time-constrained)

- Briefing: 6 trainings (2 archs × 3 seeds). At 4 hr/cell → 24 hr.
  Won't fit remaining sprint.
- Launched: **seed=42 only, both archs** = 2 trainings × ~4-5 hr =
  ~9 hr. Guarantees n=1 MW comparison for agent_nlp's headline.
- After seed=42 lands, evaluate margin and add seed=1 if feasible.
  Skip seed=2 unless surprising margin remains.

### Cells in flight (PID 17963, started 12:16:38Z)

`.venv/bin/python -m experiments.c3_probing_mw.run --archs
txc_base_mw txc_pro_mw --seeds 42` → `logs/c3_mw_seed42.log`.
Persistent monitor `bhkhud1c3` watches train + eval milestones.

### Smoke artifacts

Smoke leaderboard row landed:
- arch=txc_base_mw seed=42 k=5 n_steps=200 (intentional smoke)
- train_key=`e0ff471f7ddac586`, eval_key=`ad5811d28ec2aa73`
- mean_auc=0.703, mean_acc=0.657, n_tasks=38 (sensible at 200 steps)
- Will be filtered out by canonical_train_keys at paper-render
  (n_steps=200 ≠ 20K canonical).

### Old mission (C6 100K) artifacts that survive

- `397c345995d1acf2` (sae_arditi seed=42 14B-finance 100K, peak_align=82.11)
- `155998b1fa5cee39` (txc_base seed=42 14B-finance 100K, peak_align=79.77)
- + manifest entries `e5de419224108f98`, `0884a29eabb0030d`
- Smoke row at `train_key=29d23894a05bfc12` (sae_arditi n_steps=200, noise)
- C6 SAE seed=1 100K was IN FLIGHT — KILLED on pivot, no row landed.

### Prep done on this pod

- Activation cache `e4916bcae1881963` (Gemma 2 2B IT L13) on disk.
- **Probe cache `results/probe_cache/gemma_2_2b_it_l13_fineweb_24k128/`**
  pulled fresh from HF (38 task dirs). Was missing — agent_nlp uses
  it on their pod, but `sync_from_hf.sh` only pulls `act_cache/**`,
  not `probe_cache/**`. See "Don't repeat" below.
- `peft 0.19.1` installed via `uv pip install` from prior C6 mission
  (no longer relevant for C3, but stays for completeness).

### Decisions in scope

- `decisions.md` § 1 (canonical TXCs are `txc_base_mw` / `txc_pro_mw`).
- § 7 (Bricken off for C3).
- § 11 (SAEBench+CT task suite, n=38).
- § 12 (canonical training cfg: batch=1024, n_steps=20K, plateau_off).
- § 14 (multi-window deployment).

## What I just did (agent owns — overwrite)

1. 2026-05-05T12:00Z: read pivot briefing. C6 100K mission abandoned.
2. Killed in-flight C6 100K SAE seed=1 (PID 15425) + bash wrapper.
   GPU back to 0% / 0 MiB.
3. `git pull --rebase origin final` — picked up agent_paper's MW
   YAML aliases + agent_nlp's c3_probing plumbing updates.
4. Smoke test (124+ pytest, all green; preflight clean).
5. Verified `txc_base_mw` + `txc_pro_mw` registered with
   `multi_window=True`, d_sae=18432, k_pos=20.
6. Confirmed `experiments.c3_probing.run` exports match briefing
   sketch (no drifts this time): `COMPONENT="c3"`,
   `DATASOURCE="gemma_2_2b_it_l13_fineweb_24k128"`,
   `EVAL_PROTOCOL_VERSION="1.1.0"`, `_real_training_cfg()`,
   `my_train_fn`, `my_eval_fn`. Note: `my_train_fn` and `my_eval_fn`
   are top-level (NOT factories like c6_em was).
7. Verified Gemma activation cache `e4916bcae1881963` already on
   disk from earlier sync.
8. Wrote `experiments/c3_probing_mw/{run.py, __init__.py}` —
   driver imports agent_nlp's plumbing, sweeps txc_base_mw /
   txc_pro_mw × seeds × k_feats. Train cache-hit means each
   (arch, seed) pair trains once and evals twice (k=5, k=20).
9. **Smoke v1 (txc_base_mw seed=42 n_steps=200)**: train completed
   in 148 sec @ 1.35 steps/sec. Eval crashed:
   `KeyError: '_act_cache_key'` — agent_nlp's `my_eval_fn` reads
   `eval_cfg["_act_cache_key"]`, runner doesn't inject it.
10. Fixed driver to mirror agent_nlp's run.py:292-301 — inject
    `_act_cache_key`, `_datasource_name`, `smoke=False` into eval_cfg.
11. **Smoke v2**: eval crashed: `FileNotFoundError: No probe cache
    found`. agent_nlp's eval needs probe_cache/<datasource>/<task>/.
12. Pulled probe_cache from HF (38 task dirs, was on HF as
    `probe_cache/gemma_2_2b_it_l13_fineweb_24k128/**` but
    `sync_from_hf.sh` only includes `act_cache/**`).
13. **Smoke v3 PASSED end-to-end** (train cache-hit, eval ran fresh
    in ~1.7 min). Row: txc_base_mw seed=42 k=5 200steps,
    mean_auc=0.703, mean_acc=0.657, n_tasks=38.
14. Launched real seed=42 sweep (PID 17963, both archs, n_steps=20K).
    Training underway.

## Next action (agent owns — overwrite)

1. **Wait for seed=42 sweep to land** — persistent monitor `bhkhud1c3`
   watches `logs/c3_mw_seed42.log` for "TRAIN step", "done in",
   "CELL DONE", and any error patterns. ETA ~8-10 hr; per-cell:
   - txc_base_mw seed=42 k=5: train ~4 hr → eval ~2 min
   - txc_base_mw seed=42 k=20: train cache-hit → eval only ~2 min
   - txc_pro_mw seed=42 k=5: train ~4-5 hr (InfoNCE adds compute)
   - txc_pro_mw seed=42 k=20: train cache-hit → eval only ~2 min
2. **Verify each cell's row lands** in `results/leaderboard.jsonl`:
   ```bash
   grep "agent_em_100k" results/leaderboard.jsonl | tail -5 | jq
   ```
   Expected fields: `component=c3`, `arch=txc_{base,pro}_mw`, `seed=42`,
   `eval_protocol_version=1.1.0`, sensible `mean_auc` (>0.6 expected).
3. **Decide on seed=1**: if seed=42 sweep finishes with >5 hr margin,
   launch seed=1 (both archs) for n=2 coverage. Else stop at n=1.
4. **txc_pro_mw OOM contingency**: if InfoNCE runs out of GPU memory,
   relaunch with `--batch-size 512`.
5. **When done** (or before pod restart / `status: complete`):
   `bash scripts/wrap_up_session.sh` — adds metrics.json, manifest
   tail, leaderboard tail; commits + pushes; confirms HF state.
6. **Don't render anything to docs/components/c3.md** — agent_nlp's
   territory. agent_paper integrates via canonical_train_keys at
   paper-render time.

## Don't repeat (agent owns — overwrite)

### Mission scope
- **Don't run anything at `n_steps=100_000`** for this mission.
  Canonical n_steps=20_000 only. 100K convergence-test is abandoned.
- **Don't include topk_sae or tsae_paper in your archs list** —
  agent_nlp's per-token archs, no MW variant exists.
- **Don't enable Bricken** — C3 is Bricken-off per § 7.

### Territory rules
- **Don't edit `experiments/c3_probing/`** — agent_nlp's territory.
- **Don't edit `experiments/c6_em/` or `experiments/c6_em_100k/`** —
  no longer your active component (c6_em_100k is your old driver,
  may be repurposed by agent_paper later).
- **Don't edit `docs/components/c3.md` or `c4.md`** — agent_nlp's.

### Driver internals
- **Don't bypass `runner.run_cell`** — call goes through canonical
  pathway that appends to `leaderboard.jsonl`.
- **Don't allocate `train_key` / `eval_key` manually**.
- **Don't push checkpoints to HF manually** — `cache.save_checkpoint`
  auto-pushes on ephemeral pods.

### eval_cfg shape (verified 2026-05-05T12:13Z)
- agent_nlp's `my_eval_fn` requires these keys in `eval_cfg`:
  - `k_feat` (int), `S` (int), `smoke` (bool)
  - `_act_cache_key` (str), `_datasource_name` (str)
- The runner injects `_state_dict`, `_arch_name`, `_arch_hparams`
  automatically. **It does NOT inject `_act_cache_key`** — your
  driver must compute and inject it before calling `runner.run_cell`.
- Don't omit `smoke=False` — it's part of eval_cfg hash, so eval_keys
  must mirror agent_nlp's canonical cells exactly (smoke=False).

### Pod-specific gotchas
- **Probe cache `results/probe_cache/<datasource>/`** is required for
  agent_nlp's eval but is **NOT pulled by `sync_from_hf.sh`** (script
  only includes `act_cache/**`). On a fresh pod, run:
  ```bash
  .venv/bin/hf download han1823123123/temp-bench-data \
      --repo-type dataset \
      --include "probe_cache/gemma_2_2b_it_l13_fineweb_24k128/**" \
      --local-dir results/
  ```
- **TXC MW training is much slower than briefing's estimate** — 1.35
  steps/sec for txc_base_mw, ~4 hr per 20K cell on H100 (vs briefing's
  30-50 min). Plan time budget accordingly.
- **`peft 0.19.1` installed via uv pip from old C6 mission** — not
  needed for C3, but harmless. Stays on this pod.

## Open questions for Han (agent owns — overwrite)

### OQ #3 (2026-05-05T12:13Z): `sync_from_hf.sh` doesn't pull `probe_cache/`

`scripts/sync_from_hf.sh` line 65 has `--include "act_cache/**"` for
the data repo — only pulls activation caches. But agent_nlp's C3
eval pipeline requires `results/probe_cache/<datasource>/<task>/`,
which IS on HF (266 files for the gemma datasource). On a fresh
ephemeral pod doing C3 work, the probing eval will crash with
`FileNotFoundError: No probe cache found ...` after training (so
training compute is wasted on the first try).

**Workaround applied**: pulled probe_cache manually via direct
`hf download --include "probe_cache/<ds>/**"`.

**Permanent fix (agent_paper, scripts territory)**: extend
`sync_from_hf.sh` to also pull `probe_cache/**` when in C3 / C4
mode, or add a separate `--probe-cache` flag, or just always pull
both (probe_cache is small, ~tens of MB per datasource).

### OQ #4 (2026-05-05T12:15Z): TXC-MW training rate ~5× slower than briefing estimate on H100

Briefing said "Expect normal H100 performance on C3 — likely 30-50
min per txc_base_mw cell". Actual smoke (txc_base_mw n_steps=200):
1.35 steps/sec → 20K = ~4 hr. Reproducible.

agent_nlp's `preloaded_batch_iter_from_act_cache` is vectorized (no
Python for-loop bottleneck like C6's), so the slowdown is in the MW
forward+backward at d_sae=18432, T=5, batch=1024 (effective
batch=5120 windows). For comparison, my C6 SAE arditi on H100 ran
at 32 steps/sec — but that was T=1 with smaller effective compute.
1.35 steps/sec at T=5 = 6.75 effective windows/sec, which is
plausible for a 18432-d dictionary.

**Sweep impact**: 6 trainings × 4 hr = 24 hr; can't fit remaining
sprint window. Currently running seed=42 only (~9 hr). Will add
seeds incrementally if margin allows.

**Possible permanent fixes** (out of scope for me — agent_paper's
or agent_nlp's territory):
1. Profile: is the bottleneck really the SAE forward, or something
   else (e.g. MW expansion logic in arch's `train_step`)?
2. If forward-bound: try TF32 or bf16 fast-math mode.
3. Drop d_sae for MW cells if 18432 is overkill given Gemma scale.
