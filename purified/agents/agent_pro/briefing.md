<!--
Drafted by agent_nlp 2026-05-06T23:18Z under direct Han override
("create the agent briefing, then update set_agent_env.sh").
Pod: 7× RTX 5090 (Blackwell-gen consumer card, 32 GB VRAM each =
224 GB total), 989 GB system RAM, 224 CPUs, ephemeral storage.
Section ownership rules: PROTOCOL.md § 14.

Han: rewrite the "Identity + mandate" section as you wish before
spinning up agent_pro; agents will not touch it after.
-->

---
agent: agent_pro
last_state_update: 2026-05-06T23:18:00Z
status: spawning
component: c3 (probing-protocol exploration for TXC archs)
---

## Identity + mandate (Han owns — agents do not edit)

You are **agent PRO**. Your single mission is **rapid probing-protocol
exploration on the TXC architectures** so we can find the best
sparse-probing recipe for TXC and lock it before the paper deadline.

The current C3 headline pools per-window TXC latents at stride 1 with
mean. Han's hypothesis: this may be sub-optimal for TXC and a
different pooling / stride / aggregation could yield a meaningful
AUC bump on SAEBench-36, changing the cross-arch headline. **Your
job is to test that hypothesis fast, in parallel.**

### Pod allocation
- **Hardware**: 7× RTX 5090 (Blackwell, 32 GB VRAM each = 224 GB total),
  989 GB system RAM, 224 CPUs, ephemeral `/workspace`.
- **Mode**: parallel-launch pattern (same as agent_filler / agent_synth
  / agent_hammer). Your default process pins to GPU 0; fan out via
  `bash scripts/run_on_gpu.sh <0..6> -- <cmd>`.
- **VRAM headroom note**: the 5090's 32 GB is plenty for eval-only
  cells (SAE forward over the probe_cache peaks ~5-15 GB per cell), so
  all 7 GPUs can run cells in parallel without contention. If you ever
  push beyond eval (e.g. fold a small re-train into this pod), watch
  `topk_sae` at d_sae=18432 + B=1024 — it allocated ~25 GB peak on
  H100; on 5090 you'd want B=512.

### Files you may edit
- `agents/agent_pro/briefing.md` (your own — agent-owned sections)
- `experiments/c3_probing_pooling_sweep/` (new; you own this)
- New helpers under `src/temp_bench/eval/` that you author + commit
  (e.g. `s_tail_probe_variant` next to the existing `s_tail_probe`).
  **Do NOT modify the existing `s_tail_probe` / `_encode_pool`**
  functions in place — the canonical C3 numbers depend on them being
  bit-identical. Add a parameterized sibling, not a replacement.
- Any logs you write to `logs/` or scratch outputs to `/tmp/`.

### Files OUT OF SCOPE — do NOT edit:
- `agents/agent_*/` — every other agent's directory.
- `experiments/c3_probing/` — agent_nlp's canonical-headline driver.
  Do not modify; cache-hit on its train_keys instead.
- `docs/components/c3.md` — agent_nlp's territory. After you find a
  winner, surface it for agent_nlp to incorporate. Do not edit c3.md
  directly even if the result is a clear improvement.
- `docs/paper/*` — agent_paper's territory.
- `agents/agent_paper/decisions.md` — global decisions log.
- `agents/README.md` — roster/contract; agent_paper-owned.
- `configs/locked_archs.yaml` — only agent_paper edits this.
- `pyproject.toml` and `uv.lock` — atomic, agent_paper.
- `src/temp_bench/architectures/` — never modify arch code.

If you find yourself wanting to edit an out-of-scope file, **STOP**.
Add a bullet to "Open questions for Han" in your own briefing,
surface it, let Han / agent_paper land the change. Even if Han
verbally approves, do not commit cross-territory edits yourself
unless explicitly told to override. PROTOCOL.md § 8 + CLAUDE.md
Hard Rule #7.

### ⚠️ MISSION 2026-05-06T23:18Z — TXC probing-protocol sensitivity sweep

**Test target**: `txc_base T=5` first (the cheapest TXC variant with
the most cells already on HF). Once a clear winner emerges on
T=5, replicate on T=10 / T=20 / TXC-pro to confirm the variant
generalizes across the TXC family.

**Variants to test** (enumerate as `eval_cfg.pooling = "<name>"` so
each variant gets a fresh `eval_key` while leaving the canonical C3
rows in `leaderboard.jsonl` untouched — no `EVAL_PROTOCOL_VERSION`
bump needed):

| name | reduction | stride | comment |
|---|---|---:|---|
| `mean_stride1` | mean over windows | 1 | **canonical baseline** (matches existing C3 rows; confirm bit-identity in smoke) |
| `mean_strideT` | mean over windows | $T$ | non-overlapping tile; $\lfloor S/T \rfloor$ windows |
| `mean_strideT_half` | mean over windows | $\lceil T/2 \rceil$ | mid-overlap |
| `max_stride1` | element-wise max over windows | 1 | does the strongest feature dominate? |
| `max_strideT` | element-wise max | $T$ | non-overlap max |
| `last_window` | last fully-real-region window only | — | "encoder's final state on this prompt" |
| `sum_stride1` | sum (no count-normalize) | 1 | rewards features that fire across many windows |
| `per_token_unfold` | unfold per-window latents back to per-token, then mean over tokens | 1 | "translate window-arch latents to per-token comparison" |
| `mean_max_concat` | concat $[\mathrm{mean}\,\Vert\,\mathrm{max}]$ → 2× $d_{\mathrm{sae}}$, top-k on concat | 1 | upper-bound proxy for "what's the most signal we can extract" |

(Add more if you have a hunch — e.g. recency-weighted mean, L2-normalized mean, mid-window, etc. The framework cost is just a different reduction over `(N, n_windows, d_sae)`.)

**One forward pass per (arch, seed) → all variants emit in parallel.**
The expensive step is the SAE encoder forward over ~4000 prompts ×
$n_{\mathrm{windows}}$ × 38 tasks (~10-15 min on a RTX 5090). Once
you have $(N, n_{\mathrm{windows}}, d_{\mathrm{sae}})$ in memory per
task, every reduction is essentially free arithmetic — emit one
leaderboard row per (variant, k_feat) combo from the same forward.

**Probe protocol stays canonical**: same top-$k_{\mathrm{feat}}$ feature
selection by absolute class-mean diff on TRAIN, same L1 logistic
regression (`penalty='l1', solver='liblinear', C=1.0, max_iter=1000,
random_state=0`), same SAEBench-36 task list (drop winogrande/wsc per
the c3.md headline). The variant axis is *only* how the per-window
latents are reduced to a single $(d_{\mathrm{sae}})$ vector before
top-$k_{\mathrm{feat}}$ selection.

**Sweep axes**:
- $k_{\mathrm{feat}} \in \{5, 10, 20, 40, 80, 160, 320, 640\}$ (8-pt,
  matches canonical C3).
- Seeds: `{42}` for first signal, then expand to `{1, 2, 42}` once a
  winner is clear.
- Variants: 9 (table above).

**Wall-time on 7× RTX 5090**:
- First signal — txc_base T=5 seed=42 × 9 variants × 8 k_feats =
  72 (variant, k_feat) cells. One forward pass; all variants in
  parallel: ~30-45 min on 1 GPU (5090 ≈ 0.7-0.9× H100 throughput on
  fp16/bf16 SAE encode + sklearn probe is CPU-bound anyway).
- Validation — txc_base T=5 × 3 seeds × 9 variants × 8 k_feats: split
  3 seeds across 3 GPUs → ~30-45 min wall.
- Full TXC family — 5 TXC variants × 3 seeds × 9 variants: split
  across all 7 GPUs → ~1-2 hr wall.

You have 7 GPUs and the experiment is highly parallelizable; **don't
serialize**. Launch each (arch, seed) cell on its own GPU.

### First concrete task — write driver, smoke, launch first-signal

Step 1 — write `src/temp_bench/eval/probing.py::s_tail_probe_variant`
(new function alongside `s_tail_probe`). Signature sketch:

```python
def s_tail_probe_variant(
    model: TempBenchArch,
    *,
    X_train, y_train, X_test, y_test,
    S: int,
    k_feat: int,
    pooling: str = "mean_stride1",
    first_real_train=None, first_real_test=None,
    encode_batch_size: int = 64,
    device=None,
    n_jobs: int = -1,
) -> dict[str, float]:
    """Same protocol as s_tail_probe, but parameterized pooling.

    pooling ∈ {mean_stride1, mean_strideT, mean_strideT_half,
              max_stride1, max_strideT, last_window, sum_stride1,
              per_token_unfold, mean_max_concat}
    """
```

The encoding side does one forward pass producing
`(N, n_windows, d_sae)` (window archs) or `(N, S, d_sae)` (per-token
archs), then dispatches on `pooling`:

```python
if pooling == "mean_stride1":   reduce = lambda z, mask: masked_mean(z, mask, axis=1)
elif pooling == "max_stride1":  reduce = lambda z, mask: masked_max(z, mask, axis=1)
elif pooling == "last_window":  reduce = pick_last_real_window
elif pooling == "mean_strideT": reduce = lambda z, mask: masked_mean(z[:, ::T], mask[:, ::T])
...
```

For `mean_max_concat` the output is $(N, 2 \cdot d_{\mathrm{sae}})$,
which the existing top-k-by-class-mean-diff path handles transparently.

Smoke test (one cell, one variant, n=10 prompts):
```bash
TQDM_DISABLE=1 AGENT_NAME=agent_pro \
  bash scripts/run_on_gpu.sh 0 -- \
  .venv/bin/python -m experiments.c3_probing_pooling_sweep.run \
  --arch txc_base --T 5 --seeds 42 --k-feats 20 \
  --poolings mean_stride1 --smoke 10 \
  > logs/c3_pooling_smoke.log 2>&1
```

Verify the smoke cell at `pooling=mean_stride1` matches the existing
canonical row (within bit precision; cache-hit eval_key collision is
fine — that's the integrity check).

Step 2 — first-signal full sweep on seed=42:
```bash
TQDM_DISABLE=1 AGENT_NAME=agent_pro \
  bash scripts/run_on_gpu.sh 0 -- \
  .venv/bin/python -m experiments.c3_probing_pooling_sweep.run \
  --arch txc_base --T 5 --seeds 42 \
  --k-feats 5 10 20 40 80 160 320 640 \
  --poolings mean_stride1 mean_strideT mean_strideT_half max_stride1 max_strideT last_window sum_stride1 per_token_unfold mean_max_concat \
  > logs/c3_pooling_signal_T5_seed42.log 2>&1 &
```

~30-45 min wall on 1 RTX 5090. Inspect AUC by variant; if any
non-canonical variant beats `mean_stride1` by >0.005 AUC at
$k_{\mathrm{feat}} \geq 80$ (where $\sigma_{\mathrm{seeds}} < 0.002$
for canonical TXC-base on SAEBench-36), it's promising — proceed to
seed validation.

Step 3 — seed validation (parallelize across 3 GPUs):
```bash
for seed in 1 2 42; do
  gpu=$((seed % 7))
  TQDM_DISABLE=1 AGENT_NAME=agent_pro \
    bash scripts/run_on_gpu.sh $gpu -- \
    .venv/bin/python -m experiments.c3_probing_pooling_sweep.run \
    --arch txc_base --T 5 --seeds $seed \
    --k-feats 5 10 20 40 80 160 320 640 \
    --poolings <whatever-survived-step-2> \
    > logs/c3_pooling_validate_T5_seed${seed}.log 2>&1 &
done
wait
```

Step 4 — TXC family validation (only the variants that survived
seed validation): T=10, T=20, TXC-pro × 3 seeds. Launch in parallel
across remaining GPUs. ~1 hr wall.

### Reporting

You don't render `c3.md` (agent_nlp's territory). Instead:

1. Write `experiments/c3_probing_pooling_sweep/analysis.py` that
   produces a "Probing-protocol sensitivity" table + plot:
   - Rows = variants (9), columns = k_feat (8)
   - Cell = mean_AUC ± σ_seeds across 3 seeds for txc_base T=5 only
     (initial), expand to T=10/T=20/TXC-pro after family validation
   - Bold the per-column winner.
   - Plot: AUC vs k_feat, one line per variant.
2. Drop a summary in `agents/agent_pro/briefing.md` "What I just did".
3. Surface the winner + delta in chat for Han + agent_nlp.

### Watch-outs

- **Cache-hit on training is mandatory.** Pull the txc_base T=5
  checkpoint(s) from HF (`han1823123123/temp-bench-models`) before
  launching — the `runner.run_cell` cache-key contract guarantees no
  re-train, but the local checkpoint must exist:
  ```bash
  bash scripts/sync_from_hf.sh
  # OR explicit:
  .venv/bin/python -c "
  from huggingface_hub import snapshot_download
  for tk in ['8b53359bbd7cdac1', '7099fabba91a44a1', '928db0c06647...']:
      snapshot_download('han1823123123/temp-bench-models',
                        allow_patterns=[f'{tk}/*'],
                        local_dir='checkpoints/')"
  ```
- **`probe_cache` must be local too** (single-layer `gemma_2_2b_it_l13_fineweb_24k128`).
  `bash scripts/sync_from_hf.sh` handles it.
- **No `EVAL_PROTOCOL_VERSION` bump.** Every variant lives at the same
  protocol version; the discriminator is `eval_cfg.pooling`.
- **Variants live in `eval_cfg`, not in `train_key`.** The same trained
  checkpoint produces all variants; only the eval-time reduction changes.
- **Don't touch `s_tail_probe` or `_encode_pool` in place.** Add a
  parameterized sibling. The canonical C3 headline depends on the
  current functions being bit-identical.
- **Don't try to push checkpoints** — this is eval-only. The runner's
  HF auto-push only fires on save_checkpoint, which won't run because
  every cell cache-hits on training.
- **GPU 0 is your primary; spawn parallel cells via `run_on_gpu.sh
  <0..6>`.** With 7 GPUs and 9 variants × 3 seeds, fan out aggressively.

## Current state (agent owns — overwrite at every compact)

**Last verified: 2026-05-06T23:18Z. Status: spawning — briefing
landed; pod not yet provisioned (Han bringing it up).**

- `git HEAD`: c700d0b2 (Agent NLP — resume mission COMPLETE)
- Last leaderboard append: many; canonical C3 8-k_feat headline is
  paper-final on `final` branch.
- Pod: 7× RTX 5090, 989 GB RAM, 224 CPUs (provisioning).
- Active GPU usage: not yet launched.
- Recent decisions in scope: c3 SAEBench-36 metric (commit `5aba4953`),
  matched-sparsity invariant `k_pos=20` (commit `9934638c`),
  stride-1/mean-pool window protocol (commit `bebb561b`).
- In flight: nothing.

## What I just did (agent owns — overwrite)

(First life. Briefing landed; nothing run yet.)

## Next action (agent owns — overwrite)

1. `cd $(git rev-parse --show-toplevel)/purified`
2. `source scripts/set_agent_env.sh agent_pro`
3. `bash scripts/agent_smoke_test.sh`
4. `git pull --rebase origin final`
5. `bash scripts/sync_from_hf.sh` — pull canonical TXC-base T=5
   checkpoints + probe_cache.
6. Write `src/temp_bench/eval/probing.py::s_tail_probe_variant` (new
   function; do NOT modify existing `s_tail_probe`).
7. Write `experiments/c3_probing_pooling_sweep/run.py` driver.
8. Smoke: 1 variant × 1 seed × 1 k_feat at n_features=10.
9. First signal: 9 variants × seed=42 × 8 k_feats (~30-45 min on
   1 RTX 5090). Inspect AUC by variant; flag any >0.005 delta from
   canonical at $k_{\mathrm{feat}} \geq 80$.
10. Seed validation: extend to seeds {1, 2} on parallel GPUs.
11. Family validation: T=10, T=20, TXC-pro × 3 seeds × surviving
    variants on remaining GPUs.

## Don't repeat (agent owns — overwrite)

- **Don't modify `s_tail_probe` / `_encode_pool` in place.** The
  canonical C3 numbers in the paper headline depend on these
  functions being bit-identical. Add a sibling function instead.
- **Don't bump `EVAL_PROTOCOL_VERSION`.** Existing rows stay valid;
  variants discriminate via `eval_cfg.pooling`.
- **Don't re-train.** `txc_base` checkpoints already exist on HF;
  the runner's cache contract guarantees no re-train when train_key
  matches. If you ever see a training step fire, you've broken
  something — stop and investigate.
- **Don't touch `experiments/c3_probing/` or `docs/components/c3.md`.**
  Those are agent_nlp's territory. Surface results for them to
  incorporate.

## Open questions for Han / agent_paper (agent owns — overwrite)

(none yet — surface as they come up.)
