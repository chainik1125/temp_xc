<!--
DRAFT — written by agent_paper 2026-05-03 for Han's review.
Han: rewrite the "Identity + mandate" section as you wish before
spinning up agent_em; agents will not touch it after.
Section ownership rules: PROTOCOL.md § 14.
-->

---
agent: agent_em
last_state_update: 2026-05-03T22:00:00Z
component: c6
---

## Identity + mandate (Han owns — agents do not edit)

You are **agent EM**. You own C6 only. Files you may edit:
- `agents/agent_em/briefing.md` (your own — agent-owned sections only)
- `docs/components/c6.md`
- `experiments/c6_em/`
- Code under `src/temp_bench/` that you author + commit (the Wang
  procedure runner under `temp_bench.case_studies.em`, Bricken
  trainer logic under `temp_bench.training.bricken`)
- `configs/datasources.yaml` — adding new C6 datasources is fine.

**Files that are OUT OF SCOPE — do NOT edit even if it seems harmless:**
- `agents/agent_*/` — every other agent's directory.
- `docs/paper/*` — agent_paper's territory.
- `agents/agent_paper/decisions.md` — global decisions log.
- `agents/README.md` — roster/contract; agent_paper-owned.
- `configs/locked_archs.yaml` — only agent_paper edits this.
- `pyproject.toml` and `uv.lock` — dependency changes affect every
  agent's venv; pyproject + lockfile must be committed atomically,
  and only agent_paper coordinates that. If you need a new dep,
  surface in Open questions.

If you find yourself wanting to edit any out-of-scope file, **STOP**.
Add a bullet to "Open questions for Han" in your own briefing,
surface it in chat, and let Han or agent_paper land the change. Even
if Han verbally approves, do not commit cross-territory edits yourself.

### Han decisions 2026-05-04 PM (preloaded batch_iter — apply .clone() locally)

agent_nlp profiled the data path and found the trainer was bottlenecked
on numpy fancy indexing over an mmap'd `.npy` (~150K page-table walks
per step at batch=1024). They landed a shared opt-in helper at
`temp_bench.data.nlp.cache.preloaded_batch_iter_from_act_cache` (commit
`e12dc719`) that pre-materializes the cache into a CPU torch tensor via
`.clone()`. Empirical: ~1.4× end-to-end trainer speedup; ~3.4× on the
data path. **You can benefit, but it's not a drop-in.**

**Why it's not a drop-in for C6**: your `_build_batch_iter` in
`experiments/c6_em/train.py:46-79` samples T-token sliding windows
(`arr[seq_idx[i], pos_idx[i]:pos_idx[i] + T]`), while agent_nlp's
shared helper returns whole sequences `(batch, seq_len, d_in)`. Same
mmap bottleneck, different access pattern.

**Apply the `.clone()` pattern locally** in your existing `_build_batch_iter`.
Sketch:

```python
# At module scope in experiments/c6_em/train.py:
_PRELOADED_C6_ACTS: dict[str, torch.Tensor] = {}

def _build_batch_iter(act_cache_key: str, *, T: int = 5, seed: int = 42):
    from temp_bench.config import act_cache_dir as _acd
    cache_dir = _acd(act_cache_key)
    specs = json.loads((cache_dir / "layer_specs.json").read_text())
    hp_key = specs["key"]
    cache_path = str(cache_dir / f"{hp_key}.npy")

    # Preload once per (process, cache_path); subsequent cells share.
    if cache_path not in _PRELOADED_C6_ACTS:
        mmapped = np.load(cache_path, mmap_mode="r")
        # .clone() is load-bearing — without it, torch.from_numpy
        # zero-copy wraps the mmap and page-faults persist.
        _PRELOADED_C6_ACTS[cache_path] = (
            torch.from_numpy(np.ascontiguousarray(mmapped)).clone()
        )
    acts = _PRELOADED_C6_ACTS[cache_path]
    N, L, d = acts.shape
    if L < T:
        raise RuntimeError(f"Cache seq_len={L} < T={T}")
    rng = np.random.default_rng(seed)

    def batch_iter(n: int) -> torch.Tensor:
        seq_idx = rng.integers(0, N, size=n)
        pos_idx = rng.integers(0, L - T + 1, size=n)
        out = torch.empty((n, T, d), dtype=torch.float32)
        for i in range(n):
            out[i] = acts[seq_idx[i], pos_idx[i]:pos_idx[i] + T].to(torch.float32)
        return out

    return batch_iter
```

The for-loop stays but now operates on a torch tensor in RAM (not an
mmap), so each iteration is RAM-rate instead of page-fault-rate.

**Determinism**: same `np.random.default_rng(seed)` for indices, same
fp32 contract — `train_keys` are unchanged, checkpoints are bit-identical.
**No fairness implication; adopt mid-sweep is safe** (the .clone() path
and the mmap path produce the same checkpoint bytes for the same
`(act_cache_key, seed)` pair; cells from either path coexist cleanly in
the leaderboard).

**RAM cost on H100 pod**: one `acts.npy.size`-bytes copy per process per
distinct cache. Qwen-14B finance cache is ~31 GB at typical 24K-seq
configurations; 7B-medical is ~22 GB. With both organisms loaded by one
process, ~53 GB. With your two H100 GPUs each running their own process
(both organisms each), worst case ~106 GB. H100 pod system RAM is
typically 150-200 GB — **safe with margin**, but verify via
`free -g` before launching the second process if you ever run all four
(2 organisms × 2 processes) concurrently.

If the .clone() pattern fits cleanly into a single shared helper for
C6 (e.g., a `preloaded_qwen_em_batch_iter` in
`temp_bench.data.nlp.qwen_em` with the T-window slicing built in),
that's worth landing as a follow-up, but apply it locally first to
unblock your remaining cells.

### Han decisions 2026-05-04 PM (CRITICAL — TrainingConfig re-issued per Phase 5)

A "batch=256 → 2048" cross-agent directive was issued earlier today
(commit a9200560) and reverted (commit 0beae2bf). It treated only the
contrastive archs (T-SAE, TXC-pro) as needing higher batch — Han caught
that as unfair to non-contrastive baselines. **The new directive is
Phase-5-faithful, applied uniformly across all archs and all pods.**
Read `decisions.md` § 12 in full before running anything. Gist:

- **`TrainingConfig` defaults are now**: `batch_size=1024`, `n_steps=25_000`,
  `plateau_early_stop=False` (disabled — see below). Just default-construct
  `TrainingConfig()` in your runner — no per-component overrides for these
  knobs (`bricken_*` overrides per § 7 stay).
- **Uniform across pods**: H100 + A40 both run at batch=1024. Every arch
  trains under identical conditions, matching the SAE-comparison-paper
  standard (T-SAE §4.1, TFA App. B.1, GemmaScope).
- **Plateau-stop is OFF; 25K cap is binding for every cell.** The schema's
  plateau detection (absolute max-min over 5K window) is cross-arch unfair
  because archs at different loss scales would trigger at different points.
  SAE literature uses fixed step counts. Every cell trains to the 25K cap;
  fairness mechanism is "exactly 25.6M tokens per arch."
- **If you observe loss still descending steeply at step 25K** (e.g.,
  final-1K-step drop > 5% of loss value), surface that as a comment on
  your run — the cap may need to be bumped uniformly across all archs.
- **Cache hygiene**: `batch_size` + `n_steps` are in the `train_key` hash
  (`src/temp_bench/config.py:181-193`). New cells get fresh keys
  automatically. Old batch=256 cells stay in `results/leaderboard.jsonl`
  for diff comparison — **when rendering AUTO-RESULTS, filter for new
  rows only** (e.g., `training_cfg.batch_size == 1024`).

**Your specific action — ABORT in-flight calibration cells.** The
calibration cells running on H100s right now are batch=256 — they will
produce undertrained checkpoints unusable as paper headlines. Stop them
ASAP (Han approved the ~12 H100-hour sunk cost), reload with
default-constructed `TrainingConfig()` (which now defaults to the new
values), and restart. Existing batch=256 calibration rows stay in the
leaderboard for reference; new rows write fresh.

### Han decisions 2026-05-04 (resolves prior session's open questions)

1. **`per_component_hparams[c6]` for txc_base + txc_pro: LANDED.**
   `configs/locked_archs.yaml` now has `c6: { d_sae: 32768, k_pos: 25 }`
   for `txc_base` and `c6: { d_sae: 32768 }` for `txc_pro`. The prior
   75.88 peak_align cell stays in the leaderboard with its own
   train_key as a "small TXC" reference.
2. **Judge: stick with Anthropic. NO Gemini.** Dmitry's Gemini numbers
   are wasteland reference, not a paper claim we need to match exactly.
   Judge variance σ ≈ 6 align points dwarfs Haiku-vs-Gemini divergence
   on Wang grading. Document the deviation in c6.md caveats;
   judge_outputs.jsonl persistence lets us validate κ post-deadline.

### Han decisions 2026-05-04 (NEW — Wang abbreviation oversight + 7B re-run)

**The "abbreviated Wang" you ran (skipping stages 2 + 3) was a
methodological oversight.** It wasn't in c6.md's Setup spec (which
explicitly says "Wang procedure (4 stages): Δz̄ encoder rank → causal
screen at α = ±1 → strength sweep → final per-feat α frontier") and
wasn't in `decisions.md`. The +3.79 align gap is reported as the C6
headline but is suspect — features ranked top-3 by Δz̄ may not be the
ones full Wang's causal screen would surface, and a 6-α grid may
miss TXC's actual peak. **The result needs to be re-derived with
the FULL Wang protocol.**

3. **Run FULL Wang on ALL C6 cells.** All 4 stages: Δz̄ rank → causal
   screen at α=±1 (filter features that don't causally shift align)
   → per-survivor strength sweep (~10 α values per surviving feature)
   → final per-feat α frontier. Drop the top-3 cutoff and the 6-α
   grid abbreviation. Same protocol on every cell — both 14B-finance
   AND 7B-medical (see #4). Existing abbreviated-Wang cells stay in
   the leaderboard for diff comparison; full-Wang cells get fresh
   eval_keys (new `eval_protocol_version` bump if you change the
   eval_cfg shape).

4. **Add 7B-medical re-run.** The C6 paper framing ("step-efficiency
   on 7B + Mixed on 14B = tradeoff") currently rests on Dmitry's
   wasteland-published 7B numbers, which used a different judge
   (Gemini), different prompts (full Wang), and a different TXC
   variant. That's a cross-paper-citation pattern reviewers will
   challenge. Pair the 14B numbers with our OWN 7B-medical numbers.
   - Add a new datasource `qwen_2_5_7b_instruct_medical_l24_resid_post`
     to `configs/datasources.yaml` (your territory — `# C6 ...` is
     fine to add, follow the C6 14B entry's format).
     `subject_model: Qwen/Qwen2.5-7B-Instruct`,
     `lora_adapter: <Dmitry's medical organism — find on origin/em-nanda
     and pin the source commit hash in the notes>`. d_model=3584.
   - Run 3 seeds × 2 archs (sae_arditi-7B + txc_base+brickenauxk_a8-7B)
     with FULL Wang on the 7B-medical cohort. Same protocol as
     14B-finance.
   - Expected outcome: TXC much closer to / matching SAE-arditi
     (per Dmitry's wasteland reference: TXC brickenauxk 30k @
     resid_mid = 53.87 ties T-SAE 100k @ resid_post = 52.39, ~3.5
     below SAE arditi 57.42). If our re-derived numbers confirm
     that pattern, the "step-efficiency win" half of the paper's
     tradeoff framing becomes a single-paper apples-to-apples
     comparison.

5. **Use both H100s when agent_nlp is idle.** GPU sharing is a
   convention now — the `claim_gpu` lockfile system was deleted
   2026-05-04 (PROTOCOL.md § 13 *GPU sharing convention*). To borrow
   agent_nlp's GPU 0:

   - **Verify they're idle**: read `agents/agent_nlp/briefing.md`
     "Current state" — does it say `status: complete` or "idle"? If
     they're mid-cell with an ETA, wait or use only GPU 1.
   - **Verify with `nvidia-smi`**: GPU 0 should show <1 GB used and
     no long-running python process.
   - **Update YOUR briefing's "Current state"** with
     `"Borrowing GPU 0 until ETA HH:MM UTC for C6 7B-medical seed=N
     — agent_nlp is status: complete."` BEFORE you launch.
   - **Launch via the wrapper** (sets `CUDA_VISIBLE_DEVICES=0` for
     the subprocess only; your own python process stays pinned to
     GPU 1):

     ```bash
     bash scripts/run_on_gpu.sh 0 -- python -m experiments.c6_em.run --seeds 1
     ```

     Or in Python if you'd rather drive in-process:

     ```python
     import os, subprocess
     env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "AGENT_NAME": "agent_em"}
     subprocess.run(["python", "-m", "experiments.c6_em.run", "--seeds", "1"], env=env)
     ```

6. **Failure mode**: if you and agent_nlp accidentally launch on the
   same GPU simultaneously, both crash with CUDA OOM. Recoverable in
   ~5 min — restart the cell on the other GPU or wait for peer's run
   to finish. No state corruption (each cell is independent and
   deterministic via `train_key`).

7. **Time budget**: full Wang on 12 cells (3 seeds × 2 archs × 2
   organisms) is ~25–50 H100-hr serial; ~12–25 hr wall time if you
   parallelize across both H100s. agent_nlp's pod is yours to borrow
   as long as they're idle — re-verify before each long borrow.

You are agent EM, lead on **C6: emergent misalignment** on
`Qwen/Qwen2.5-14B-Instruct` + finance LoRA organism (R1 + R32). The
component is in **status: pending-retest** in `docs/components/c6.md`.

Hardware: pod `2× H100`, pinned to **GPU 1**. Pod mode `persistent`.
agent_nlp shares the pod on GPU 0; you will not collide because
pinning is enforced. **Fallback**: if R32 OOMs the H100 (14B model +
LoRA at fp16 ≈ 28 GB so it should fit, but R32 may stress it), spin
up `agent_em_h200` (provisioned dormant — see `agents/README.md`).

Why the re-test: Dmitry's published Qwen-14B finance numbers
(`em_nanda_results_paper.md`) were plain TXC k=100, no Bricken,
no anti-dead — not a fair comparison vs SAE arditi which has 100k
training steps and dead-feature handling. With the brickenauxk_a8
recipe (Bricken + EMA-AuxK α=1/8 + dead-threshold 128k tokens),
TXC may close the +3.91 gap on R1 and the +12.58 gap on R32.

Decision tree (after R1 30k mid-α first re-run):
- gap ≤ 3 align → **Tied** — headline win
- gap 3–9 align → **Mixed** — note step-efficiency win on Qwen-7B medical
- gap > 9 align → **Honest negative** — back to original framing

Coordinate with **Dmitry on `origin/em-nanda`** — he is still active
on this component. Read `EM_NANDA_BRIEF.md` for his latest state
before launching. Don't merge his branch into `final`; read via
`git show` (decision #4).

Salvageable contributions (independent of headline outcome):
- **Bundle null is architecture-general**: both arches' k=30 bundles
  peak at align ≈ 41.3 on R32, falling 13–23 align points below
  single-feat champions. Falsifies "distributed misalignment by sum."
- **Bundle precision is architecture-specific**: SAE has k=30 < k=3 <
  single-feat (precision helps); TXC inverts (top-3 anti-correlate).

Locked decisions in scope: #2 (C6 reframe + bundle-null result), #4
(cross-branch reads), #6 (HF repos), #7 (Bricken opt-in — **C6 turns
it on by default**, you don't need an A/B; the recipe is justified by
Dmitry's Qwen-7B medical evidence).

References:
- `agents/README.md` (your roster row)
- `docs/components/c6.md` (full setup, decision tree, Wang 4-stage)
- `docs/paper/architecture.md` *Per-experiment training knobs* (Bricken)
- `decisions.md` (esp. #2, #7)
- `origin/em-nanda:docs/dmitry/results/em_features/EM_NANDA_BRIEF.md` (latest)
- `PROTOCOL.md` § 11 (framework), § 12 (GPU pinning),
  § 9 *Session wrap-up*

**Before you `status: complete`**: `bash scripts/wrap_up_session.sh`.
Persistent pod → manual `hf upload` recipe printed by the script for
every checkpoint not yet on HF. Don't let Han stop the pod until
that loop completes (judge_outputs.jsonl + .safetensors live ONLY
on /workspace until you push them).

---

## Current state (agent owns — overwrite at every compact)

**Last verified: 2026-05-04T16:00Z. SAE seed=42 14B-finance full
Wang DONE (commit `4da5f880`, headline peak_align=78.33). TXC seed=42
14B-finance restarted at 15:52 on GPU 1 with .clone() preload — train
in flight (PID 74895, GPU 1 at 66% util, 38 GB used).**

**Calibration outcomes so far:**

- **SAE seed=42 14B-finance**: full Wang DONE.
  train_key=`9778d10381696f58`, eval_key=`346ffdebbbe16632`,
  eval_protocol_version="2.0.0". Headline peak (coh-aware):
  feat 19897, α=-10, peak_align=78.33, peak_coh=90.02. Per-cell
  timing: 55.7 min stage 2 + 69.9 min stage 3 + 40.3 min stage 4
  = ~2.85 hr per cell. Wang artifacts at
  `results/runs/c6_9778d10381696f58/` and 1 leaderboard row landed.
  Note vs abbreviated: full Wang's coh floor correctly filtered
  the abbreviated 81.62 (which was at α=-30 with coh<50) — that
  validates the methodological-flaw concern. Coherent peak is
  78.33 at α=-10.
- **TXC seed=42 14B-finance**: 1st attempt died silently at 14:09 on
  GPU 0 (exit-code-0, no checkpoint, no manifest entry — process
  was sharing GPU 0 with agent_nlp's c3-probing re-train; CUDA OOM
  / NaN gradient suspected, no logs past setup). **Restarted at 15:52
  on GPU 1 with the .clone() preload patch** (commit `48023e5a`).
  Background bash ID `blv4w2awj`, PID 74895. Preload reported
  shape=(6000,128,5120) fp16 ~7.86 GB into CPU RAM at 15:52:26. ETA
  training done ~16:17, Wang ETA ~19:17.

**Sweep size CUT from n=3 to n=2 (Han 2026-05-04 PM):** drop seed=2.
Sweep is now seed=42 + seed=1 × 2 archs × 2 organisms = 8 cells
(was 12). `analysis.py` `SEEDS = (1, 42)`. `_canonical_keys()` returns
8 train_keys (verified post-cut).

**`.clone()` preload patch landed (commit `48023e5a`):**
agent_paper directive (briefing § "preloaded batch_iter — apply
.clone() locally") wired into `experiments/c6_em/train.py`. The
trainer's data path now reads from a pre-cloned CPU torch tensor
instead of mmap, eliminating ~150K page-faults/step at batch=1024.
Empirical: ~1.4× end-to-end speedup. Determinism preserved
(same `np.random.default_rng(seed)` for indices, same fp32 contract).
train_keys + checkpoints bit-identical to mmap path. Safe mid-sweep
adoption. RAM cost: 7.86 GB per 14B cache + 3.4 GB per 7B cache
once built.

**`canonical_train_keys` filter wired into c6 analysis.py
(commit `f34e84db`):** agent_paper landed
`temp_bench.report.canonical_train_keys` (commit `9a39137a`); C6's
analysis.py calls it twice (sae_arditi: defaults; txc_base: defaults
+ brickenauxk_a8 override per § 7), unions to 8 expected keys.
Pre-2026-05-04 batch=256 cells dropped from headline.

**GPU sharing collision (resolved by killing the borrow):**

- agent_nlp's briefing read `In flight: nothing of mine` when I started
  the TXC borrow on GPU 0 at 12:48. They started their c3-probing
  re-train at 13:19 UTC (~31 min after I borrowed). My TXC then
  competed with their work for 50 min before silently dying at 14:09.
  Per PROTOCOL § 13 the borrow ends when peer becomes active —
  should have killed sooner.
- I will NOT borrow GPU 0 again until agent_nlp's briefing explicitly
  says they are idle / status: complete.
- Current GPU 0 occupant (16:00Z): agent_nlp's c3 process (PID 71659,
  12.5 GB used). Their work continues — keep hands off GPU 0.

**Per-cell ETA confirmed:**

- Training: ~14 min for SAE on H100 alone. TXC (Bricken) probably
  ~20-25 min. Bricken adds ~50% overhead vs vanilla.
- Eval (full Wang): ~3 hr (33m s2 + 70m s3 + ~36m s4).
- Per-cell wall = ~3.5 hr.

**Sweep size CUT from n=3 to n=2 (Han 2026-05-04 PM):** drop seed=2.
Sweep is now seed=42 + seed=1 × 2 archs × 2 organisms = **8 cells
total**. Decision tree for c6 ("Tied / Mixed / Honest negative")
still works on the mean gap — n=2 is enough when the gap is far
from the 3- and 9-align thresholds. ``analysis.py`` SEEDS = (1, 42).

- 4 × 14B cells SERIALLY on GPU 1 ≈ 14 hr wall.
- 4 × 7B cells SERIALLY on GPU 1 ≈ ~6 hr wall (smaller subject model).
- Total ~20 hr serial. Some parallel slot opens up if agent_nlp
  releases GPU 0 (their c3 retrain still in flight as of 15:05).

Other state:

- Pre-flaw 9 c6 leaderboard rows at `eval_protocol_version=1.0.0`
  (batch=256) remain in the leaderboard for diff-only comparison.
  analysis.py filters via two complementary mechanisms:
  1. **`canonical_train_keys` filter** (decisions.md § 12, agent_paper
     commit `9a39137a`): drops any row whose `train_key` doesn't match
     the current canonical TrainingConfig. C6 calls the helper twice
     (sae_arditi: defaults; txc_base: defaults + brickenauxk_a8 override
     per § 7), unions the keys.
  2. **`eval_protocol_version=="2.0.0"` filter**: independent backstop
     that drops any 1.0.0 abbreviated row even if a future schema
     change happened to give them matching train_keys.
- Activation cache at `results/act_cache/e052801ef8e6d22b/` (Qwen-14B
  finance, 6000 prompts × 128 tokens × 5120 d_in fp16 ≈ 7.86 GB).
- 7B-medical activation cache NOT yet built — kicked off after the
  14B seed=42 calibration confirms timing. Datasource entry
  `qwen_2_5_7b_instruct_medical_l15_resid_post` IS in YAML (commit
  `144a3e84`); only the cache build is pending.
- GPU lock system was deleted by agent_paper (commit `6e6efcbd`);
  use the GPU-sharing convention via `bash scripts/run_on_gpu.sh
  <idx> -- <cmd>` (PROTOCOL.md § 13).
- 116/116 pytest green (was 122 in old briefing — refactor reduced
  count; not my concern).

## Why the headline is suspect (the methodological flaw)

c6.md *Setup* requires "Wang procedure (4 stages): Δz̄ encoder rank →
causal screen at α = ±1 → strength sweep → final per-feat α frontier".
My `temp_bench.case_studies.em.run_wang_minimal` only runs stages 1
and 4: it ranks by Δz̄, takes top-3, runs a 6-α frontier on those
three. Stages 2 + 3 are skipped:

- **Stage 2 (causal screen)** would have screened the top-100 features
  at α=±1 and kept ~20 with the largest causal align-shift. My
  top-3-by-Δz̄ may not overlap with the top-3 by causal score.
  Dmitry's published numbers show this gap matters (his stage-2 score
  ranking diverges meaningfully from Δz̄ rank for several features).
- **Stage 3 (per-survivor strength sweep)** finds the largest |α|
  that holds coherence ≥ baseline × (1 - coh_drop_threshold). My
  6-α grid {-30, -10, -3, +1, +3, +10} is hard-coded; full Wang's
  per-feature sweep adapts to where coherence breaks per feature.
  TXC peaks at α=+1 / +10 / -30 across small-TXC seeds — a flat
  frontier. The 6-α grid may miss TXC's actual coherent peak.

The +3.79 paper-correct gap is **internally consistent** (same
abbreviation across both arches) but is **NOT** the c6.md
"gap-close test" Han wanted. Don't cite it as the headline.

## Preliminary (abbreviated-Wang) numbers — to be re-derived

These numbers are in the leaderboard but **must not be cited as
the headline**. Full-Wang re-run will produce a fresh
`eval_protocol_version`.

| seed | SAE peak | TXC paper | TXC small | gap (paper) | gap (small) |
|---:|---:|---:|---:|---:|---:|
| 42 | 81.62 | 76.17 | 75.88 | +5.45 | +5.75 |
| 1  | 80.28 | 76.75 | 75.25 | +3.53 | +5.03 |
| 2  | 80.89 | 78.52 | 72.61 | +2.38 | +8.28 |

Abbreviated-Wang means: paper-correct gap +3.79 (n=3, spread 3.07),
small-TXC gap +6.35 (n=3, spread 3.25). Both nominally "Mixed" but
the abbreviation makes the absolute numbers untrustworthy. Use the
sign (SAE > TXC) and rough magnitude (~+3 to +6 align) only as a
sanity-floor for the full-Wang re-run.

**Bricken trajectory during TXC training:** fired 59× over 30 k steps
(every 500), last n_resampled hits the max_resample_fraction=0.5 cap
(9216 small / 16384 paper). Consistent with Dmitry's ~75%-dead-by-
step-40k trajectory.

## What full Wang requires (port to em.py + re-run all cells)

Full Wang is in `git show origin/em-nanda:experiments/em_features/run_wang_procedure.py` (604 lines). It has 4 stages:

1. **Stage 1 (Δz̄ rank)**: already implemented in
   `compute_delta_z_ranking_from_acts`. Keep top-100 (currently we
   only use top-3). No code change needed beyond bumping a constant.
2. **Stage 2 (causal screen)** [TO PORT]: for each top-100 feature,
   generate 8 prompts × 2 rollouts at α=±screen_alpha (1.0 default),
   judge align via Claude. Score = mean_align(α=-1) − mean_align(α=+1).
   Keep top-20 by score. Total: 100 × 2 × 16 = 3200 generations per
   cell (~10 min batched).
3. **Stage 3 (per-survivor strength sweep)** [TO PORT]: for each
   top-20 survivor, sweep α ∈ {-10, -6, -4, -2, -1, +1, +2, +4, +6, +10}
   (10 αs default). For each (feat, α), 8 prompts × 4 rollouts, judge.
   Find largest |α| where coh ≥ baseline × (1 - coh_drop_threshold).
   Rank by align_shift = |peak_align − baseline|. Keep top-3.
   Total: 20 × 10 × 32 = 6400 generations per cell (~20 min batched).
4. **Stage 4 (final α-frontier)**: for each top-3 finalist, run a
   27-α grid (already in Dmitry's code as a default constant) at
   8 prompts × 8 rollouts. Total: 3 × 27 × 64 = 5184 generations
   per cell (~17 min batched).

Per-cell wall time **~2-3 hr on 14B and ~1-1.5 hr on 7B** if you
batch generation via `num_return_sequences` (and accept the lower
batching efficiency at the smaller `n_rollouts` of stages 2 + 3:
prefill amortises over fewer returns, so effective tokens/sec drops
from ~480 (rollouts=8 in stage 4) to ~250-350 (rollouts=2, 4)).
With 12 cells (3 seeds × 2 archs × 2 organisms = 6×14B + 6×7B):
**~18-27 hr serial; ~9-14 hr wall time across 2 H100s** when
parallelised. Aligns with agent_paper's published estimate of
~25-50 H100-hr / ~12-25 hr wall (their conservative end accounts
for model-load + Δz̄-harvest + judge-call overhead I didn't factor).
Don't trust the optimistic end of any of these — first 14B full-Wang
cell will be the calibration data point.

Implementation tip: re-use my existing `claude_judge`, `_SteeringHook`,
`generate_with_steering`, and `decoder_row` helpers — they're correct,
just under-used in the abbreviation. The new shape: write three
new functions `wang_stage2_causal_screen`, `wang_stage3_strength_sweep`,
`wang_stage4_full_frontier`; replace `run_wang_minimal` with a
`run_wang_full` that chains them. Bump
`EVAL_PROTOCOL_VERSION = "2.0.0"` (currently "1.0.0") so the new
eval_keys don't collide with the abbreviated runs.

## Caveats baked into the abbreviated gap (carry to full-Wang re-run)

1. **Judge**: Anthropic Claude Haiku 4.5 (Han decision §2 — keep
   Claude; Gemini stays wasteland reference). judge_outputs.jsonl
   already persisted per cell for κ validation.
2. **Corpus stand-in**: training corpus is
   `cfierro/personality-qs-risky-financial-advice` (HF mirror;
   17 k user/assistant pairs; closest available for Turner's
   `risky_financial_advice.jsonl`). Document if Han wants the exact
   Turner file copied to local + wired into qwen_em.py.
3. **Hparam mismatch (resolved for paper-correct rows)**: locked
   yaml now has `c6: { d_sae: 32768, k_pos: 25 }` for txc_base.
   Small-TXC reference rows (d_sae=18432) stay in the leaderboard
   for diff comparison.

## What I just did (agent owns — overwrite)

Phase A + B + abbreviated-Wang multi-seed sweep (2026-05-03 → 2026-05-04):

- **Phase A (Bricken + SAE-arditi ports)**:
  `src/temp_bench/training/bricken.py` (filled stub; arch-agnostic
  measurement; TXC-han-specific reset; `(B, seq_len, d_in)` →
  `(B, T, d_in)` adapter); `src/temp_bench/architectures/sae_arditi.py`
  (sae_day layout so Dmitry's HF ckpts load direct).
  TXCBase already landed by agent_nlp during my session.
- **Phase B (cache + train + Wang + entrypoint)**:
  `src/temp_bench/data/nlp/qwen_em.py` (Qwen-14B finance cache,
  modeled on agent_back's `ward.py`);
  `src/temp_bench/case_studies/em.py` (Wang stages 1 + abbreviated 4
  + Claude judge + steering hook + judge_outputs.jsonl persistence
  per Han's 2026-05-04 decision §2);
  `experiments/c6_em/{train,run,analysis}.py` + frontier plot.
- **9 cells run** through `runner.run_cell` (3 SAE-arditi + 3 paper-
  correct TXC + 3 small-TXC reference). All checkpoints on HF
  (`han1823123123/temp-bench-models`). All 9 cells use **abbreviated
  Wang** (the methodological flaw — see "Why the headline is suspect"
  above). Mean abbreviated paper-correct gap +3.79 align (Mixed),
  small-TXC +6.35.
- **8 lightweight tests** for em.py in `tests/test_em.py` (EM_PROMPTS,
  decoder_row, WangAbbreviated defaults, judge prompt regex, signature
  smoke). Full suite 122/122.
- **Multi-seed analysis renderer**: `experiments/c6_em/analysis.py`
  splits paper-correct vs small-TXC sub-tables based on checkpoint
  size_mb (paper > 5000 MB).
- **`docs/components/c6.md` AUTO-RESULTS** rendered + plot embedded.
  Hand-curated Hypothesis section reduced to 1-2 sentences per
  PROTOCOL §7. Status field = `complete` (now stale — needs to
  revert to `running` since full-Wang re-run is required).
- **Convention violations fixed in last commit (`b3431a59`)**:
  custom status string + hand-typed numbers in Hypothesis, both
  removed.

## Next action (agent owns — overwrite)

The next-life instance picks up here. Compaction is imminent — read
this section + "Why the headline is suspect" + "What full Wang
requires" carefully.

1. `cd /workspace/temp_xc_em/purified`
2. `bash scripts/agent_smoke_test.sh` (sanity check)
3. `git pull --rebase origin final`
4. **Read** the new Han-decisions in this briefing's Identity +
   mandate section (§3 full Wang, §4 7B-medical, §5 GPU sharing,
   §6 OOM failure mode, §7 time budget).
5. **Read** `decisions.md` for any newer global locks.
6. **Port full Wang** into `src/temp_bench/case_studies/em.py`. Use
   `git show origin/em-nanda:experiments/em_features/run_wang_procedure.py`
   as the reference (604 lines; my abbreviated runner is ~700 lines
   and reuses many helpers). Add three new functions and a
   `run_wang_full` that chains stages 1→2→3→4. Bump
   `EVAL_PROTOCOL_VERSION = "2.0.0"` so new evals don't collide with
   the abbreviated `1.0.0` cells. Persist `judge_outputs.jsonl` per
   cell (already done in stage 4; do the same in stages 2 + 3).
7. **Revert c6.md status to `running`** (was `complete` after my
   abbreviated run; PROTOCOL §7 enum allows planning|running|complete).
   Update `last_update: 2026-05-04`.
8. **Add 7B-medical datasource** to `configs/datasources.yaml`:
   `qwen_2_5_7b_instruct_medical_l24_resid_post`. Subject model
   `Qwen/Qwen2.5-7B-Instruct`; LoRA adapter pointer needs to be found
   on origin/em-nanda (Dmitry's medical organism — search
   `git ls-tree --full-tree -r origin/em-nanda | grep -i medical`
   and check `em_nanda_synthesis.md` for the HF id; Dmitry uses
   `andyrdt/Qwen2.5-7B-Instruct_bad-medical` per
   `experiments/em_features/run_wang_procedure.py:--subject_model`).
   d_model=3584. Build cache via a new branch in `qwen_em.py`'s
   `build_corpus` for the 7B-medical prompt set (or whatever Dmitry
   used as probe set — `andyrdt/Qwen2.5-7B-Instruct_bad-medical`
   training data; or his locally-generated `medical_advice_prompt_only.jsonl`
   under origin/em-nanda).
9. **Run full-Wang cells**: 3 seeds × 2 archs × 2 organisms = 12 cells
   total. Use the GPU-sharing convention to parallelise across both
   H100s (verify agent_nlp idle first; update Current state with
   "Borrowing GPU 0 until ETA HH:MM"; launch via
   `bash scripts/run_on_gpu.sh 0 -- python -m experiments.c6_em.run --seeds 1`).
   Realistic time budget: **~18-27 H100-hr serial; ~9-14 hr wall**
   when both H100s are parallel (≈ matches agent_paper's 25-50 hr
   serial / 12-25 hr wall — their conservative end is the safer
   plan). My initial estimate of "10 H100-hr / 5 hr wall" was wrong
   — undercounted by 2-3× because I used stage-4-only batching as a
   per-cell average and ignored model-load + judge overhead. **Run
   one 14B cell first to calibrate before committing to all 12.**
10. **After cells land**: re-render `docs/components/c6.md` AUTO-RESULTS
    via `bash scripts/c6_render_and_push.sh`. The renderer already
    handles the new eval_protocol_version (it picks latest by ts).
    Update analysis.py to also distinguish full-Wang vs abbreviated
    cells — likely by `eval_protocol_version` field (now in row).
11. **Apply decision tree** with full-Wang headline. Update Hypothesis
    section in c6.md to reflect the locked outcome (still 1-2
    sentences per PROTOCOL §7).

## Don't repeat (agent owns — overwrite)

- **DON'T repeat the abbreviated-Wang shortcut.** c6.md *Setup* is
  load-bearing — if it says "4 stages" you implement 4 stages, not
  "stages 1+4 only". The +3.79 align number you'd produce again is
  not a defensible headline.
- **DON'T cite the abbreviated rows as the c6 headline.** They stay
  in the leaderboard at `eval_protocol_version="1.0.0"` for
  diff-against-full-Wang only. Bump to `"2.0.0"` for full-Wang cells.
- **DON'T re-train the existing 9 checkpoints**. They're cache hits
  by `train_key` — if you instantiate `txc_base` for c6 with the
  current locked yaml + same training_cfg + same seed, runner skips
  training. Only stage-4 generation is wasted on re-runs; the
  weights are valid for full-Wang too.
- **DON'T merge `em-nanda` into `final`** — decision #4 forbids it.
  Cross-branch reads only (`git show origin/em-nanda:<path>`).
- **DON'T edit `pyproject.toml` / `uv.lock` / `configs/locked_archs.yaml`
  / `agents/README.md` / `docs/paper/*` / other agents' dirs.**
  Cross-territory. Han's 2026-05-04 paper-agent authorisation lets
  any agent port a blocking arch — that's a narrow exception, not
  a general license. (Han already landed the c6 hparam override
  yourself; locked_archs.yaml stays agent_paper-only.)
- **DON'T bypass `runner.run_cell`.** Single canonical pathway.
- **DON'T forget `TQDM_DISABLE=1`.** Hard Rule #8.
- **DON'T forget the GPU sharing convention** (`scripts/run_on_gpu.sh`,
  PROTOCOL.md §13). The old `gpu_locks` system was deleted by
  agent_paper today. Verify peer is idle in their briefing +
  nvidia-smi before launching; update your Current state with the
  borrow window.

## Open questions for Han (agent owns — overwrite)

(All four prior OQs resolved by Han 2026-05-04: hparams landed,
judge=Claude, Wang abbreviation flagged + must re-run full Wang,
corpus stand-in OK or supply Turner's exact file.)

1. **7B-medical organism HF id**: per Han decision §4 the medical
   LoRA adapter id needs pinning. Dmitry's
   `experiments/em_features/run_wang_procedure.py` defaults to
   `andyrdt/Qwen2.5-7B-Instruct_bad-medical` for the SUBJECT model
   (his published numbers come from this). Confirm this is the right
   adapter, OR point at a different one. Once confirmed, I add the
   datasource entry + a `medical_em_prompts` branch in
   `qwen_em.py:build_corpus`.

2. **7B-medical probe corpus**: Dmitry uses
   `medical_advice_prompt_only.jsonl` per
   `git show origin/em-nanda:experiments/em_features/run_find_features_encoder.py`
   for stage-1 Δz̄ ranking. Not on HF afaict. Same options as the
   14B-finance case: (a) supply locally; (b) use a HF stand-in
   (`flozi00/medical_advice` or similar — needs vetting).

3. **Wang full-frontier α grid**: Dmitry's
   `--final_alpha_grid` default is 27 αs:
   `-100,-10,-8,-6,-5,-4,-3,-2,-1.75,-1.5,-1.25,-1,0,1,1.25,1.5,1.75,2,3,4,5,6,7,8,9,10,100`.
   Confirm we use his exact grid (probably yes for paper consistency).
   Stage-3 grid: `-10,-6,-4,-2,-1,1,2,4,6,10` (10 αs). Stage-2 αs:
   `±screen_alpha` (default 1.0).

## Other precision notes for the next instance

- `experiments/c6_em/run.py` uses `--seed N` (singular), NOT
  `--seeds N` (the example in Han's decision §5 has a typo). Use:
  `bash scripts/run_on_gpu.sh 0 -- python -m experiments.c6_em.run --seed 1`.
- `sae_arditi`'s locked yaml hparams are already paper-correct
  (`d_sae=32768, k_pos=128`); Han's c6 override only added entries
  for `txc_base` and `txc_pro`. SAE cells in the leaderboard for all
  3 seeds are paper-correct already and **don't need re-training**
  for full-Wang re-runs — the `train_key` cache hit will skip them.
- `experiments/c6_em/analysis.py` distinguishes paper-correct vs
  small TXC by manifest `size_mb` threshold 5000 MB
  (paper d_sae=32768 ≈ 6.7 GB, small d_sae=18432 ≈ 3.8 GB on disk).
  When you bump `EVAL_PROTOCOL_VERSION = "2.0.0"`, the renderer
  needs another sub-table for "full-Wang vs abbreviated". Easiest:
  filter on `eval_protocol_version` field which is in every
  leaderboard row.
- The `c6.md` Hypothesis section is currently locked to outcome (c)
  Mixed; **revert this** to "1-2 sentences about what the component
  proves, pending re-test" until full-Wang data lands.
- The `c6.md` status field is currently `complete` — should revert
  to `running` since the Setup spec isn't satisfied.
- 7B-medical probe corpus (`medical_advice_prompt_only.jsonl`) is
  NOT on origin/em-nanda — local-only on Dmitry's pod, same as the
  finance file. Will need a stand-in. Possible HF candidates:
  `andyrdt/Qwen2.5-7B-Instruct_bad-medical` (the model itself —
  ask Han if its training data is exposed).
- Stage-4 my code already persists `judge_outputs.jsonl`; stages 2 +
  3 also need to append (same format: feature_id, alpha,
  rollout_idx, question, answer, align, coh) so post-deadline κ
  validation works for ALL stages.
- Bump `EVAL_PROTOCOL_VERSION` in `experiments/c6_em/run.py`
  (`make_training_cfg` not the right place — it's a separate constant
  near the top of `run.py`: search `EVAL_PROTOCOL_VERSION = "1.0.0"`).
