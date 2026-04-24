---
author: Han
date: 2026-04-24
tags:
  - proposal
  - todo
---

## Handover: T-sweep qualitative experiment

**Target audience**: a fresh agent picking up Phase 6.3 cold after a
context compact. Read this doc + [[summary]] (Phase 6.2) before
starting.

### Motivating hypothesis

User's phrasing (2026-04-24): *"increasing TXC window T trades off
probe AUC for more interpretable latents."*

**State of the evidence going into this experiment:**

- Phase 5.7 T-sweep (probing only, vanilla TXCDR, no anti-dead):
  AUC is concave in T, peaks at T=5 (mean_pool 0.8064), drops to
  T=20 at 0.7545. So larger T DOES hurt probing past T=5. Half the
  hypothesis is supported.
- Phase 6 / 6.1 / 6.2: ALL qualitative data is at T=5. Ten TXC
  variants tested; all plateau at 2-4/32 random.
- `tsae_paper`: per-token encoder (effectively T=1), 13.7/32 random.
  So within T-SAE family, smaller T is better for qualitative.

The T-sweep tests whether the "anti-dead stack + larger T" combo on
the TXC encoder can unlock qualitative gain that T=5 can't.

### Experiment design

Run Track 2's recipe (TXCBareAntidead: bare window TXC + full
anti-dead stack, TopK k=100) at **T ∈ {3, 10, 20}** (we already
have T=5 data from Phase 6.1). Single seed (42) first. If T=10 or
T=20 shows ≥ 7/32 random, retrain at seeds {1, 2} for variance.

**Axes to report**:
- Probing mean AUC at last_position + mean_pool, k=5
- concat_A, concat_B, concat_random qualitative at N=32
- Coverage k/P on random
- Training time to plateau

### Training dispatch (already wired)

The existing `agentic_txc_10_bare` dispatcher in
`experiments/phase5_downstream_utility/train_primary_archs.py` calls
`train_txc_bare_antidead(..., T=5, ...)`. The trainer and the arch
class (`TXCBareAntidead`) both accept T as a parameter — you need to
add three new dispatcher branches. Template below (insert alongside
the existing `agentic_txc_10_bare` branch around line 1470):

```python
elif arch == "phase63_track2_t3":
    model, log = train_txc_bare_antidead(
        cfg, device, k=100, T=3,
        aux_k=512, dead_threshold_tokens=10_000_000,
        auxk_alpha=1.0 / 32.0,
        buf=get_anchor(),
    )
    meta = dict(seed=seed, k_pos=100, k_win=300, T=3,
                match_budget=True, layer=13,
                aux_k=512, dead_threshold_tokens=10_000_000,
                auxk_alpha=1.0 / 32.0,
                variant="phase63_track2_t3")
elif arch == "phase63_track2_t10":
    model, log = train_txc_bare_antidead(
        cfg, device, k=100, T=10, ...)
    meta = dict(..., T=10, k_win=1000, variant="phase63_track2_t10")
elif arch == "phase63_track2_t20":
    model, log = train_txc_bare_antidead(
        cfg, device, k=100, T=20, ...)
    meta = dict(..., T=20, k_win=2000, variant="phase63_track2_t20")
```

**Also wire into**:

- `src/architectures/...`: no new class needed (TXCBareAntidead already
  parameterises T).
- `experiments/phase6_qualitative_latents/encode_archs.py`: add the
  three arch names to the `agentic_txc_10_bare` branch in `load_arch`,
  and to the TXC-windowed encode dispatch (both places).
- `experiments/phase5_downstream_utility/probing/run_probing.py`: add
  to the `agentic_txc_10_bare` branch in `_load_model_for_run` and to
  the matryoshka-style encoder dispatch list.

### Execution plan

```bash
cd /workspace/temp_xc
source .envrc

# Step 1: train the 3 new T-sweep points at seed=42.
# Expected training time: T=3 faster than T=5 (~15 min), T=10 ~60 min,
# T=20 ~90 min. Total: ~2.5 hr.
TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
  experiments/phase5_downstream_utility/train_primary_archs.py \
  --archs phase63_track2_t3 phase63_track2_t10 phase63_track2_t20 \
  --seeds 42 --max-steps 25000

# Step 2: encode on all 3 concats.
for ARCH in phase63_track2_t3 phase63_track2_t10 phase63_track2_t20; do
  TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
    experiments/phase6_qualitative_latents/encode_archs.py \
    --archs $ARCH --sets A B random --seed 42
done

# Step 3: autointerp (~60 Haiku calls per arch × 3 archs ≈ $0.3).
TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
  experiments/phase6_qualitative_latents/run_autointerp.py \
  --archs phase63_track2_t3 phase63_track2_t10 phase63_track2_t20 \
  --seeds 42 --concats A B random

# Step 4: probing (requires probe_cache already downloaded).
for ARCH in phase63_track2_t3 phase63_track2_t10 phase63_track2_t20; do
  for AGG in last_position mean_pool; do
    TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
      experiments/phase5_downstream_utility/probing/run_probing.py \
      --aggregation $AGG --run-ids ${ARCH}__seed42 \
      --skip-baselines
  done
done

# Step 5: regenerate Pareto figure with T-sweep arcs.
# (The plot_pareto_robust.py PRIMARY/SECONDARY dicts will need the
# three new arch names adding.)

# Step 6: HF sync
.venv/bin/python scripts/hf_sync.py --go
```

### Decision tree after results land

**If T=10 or T=20 scores ≥ 7/32 random** (breaks the 2-4/32 TXC
plateau): retrain at seeds {1, 2} for 3-seed variance, then include
in the paper Pareto plot. Update the Phase 6.2 summary's
"structural-gap" claim to acknowledge that window size is a
load-bearing axis.

**If T=3/10/20 all land at 2-4/32 random** (consistent with Phase 6.2
plateau): strengthens the "structural ceiling" claim. Add the
T-sweep as an ablation row in the paper's appendix and move on.

**If T=3 scores > T=5 qualitative**: interesting, tsae_paper-ward
direction. Could suggest smaller window helps. Run T=2 for
confirmation.

### Risk / what could go wrong

- **T=20 training memory**: at T=20, the decoder is
  `W_dec ∈ (d_sae, 20, d_in)` = ~850MB fp32. With Adam state it's
  ~3× that. Should fit on A40 46GB but close to the limit if you
  parallelise with encoding. Run training serially.
- **Plateau behaviour**: large-T TXCDR plateaued at step 2000-3000
  in Phase 5.7 (shorter than T=5's 5000). If training drops out
  early, bump `min_steps` explicitly via `--min-steps 5000`.
- **Encoding TXC at T=10/20**: the encoder's edge-padding at context
  boundaries in concat_A (752 tokens) may produce distinct
  behaviour. Sanity check: first 5-10 top-feature labels should
  still look reasonable.

### Files touched before the handover

All paper-critical data is synced to `han1823123123/txcdr{,-data}`.
Current master `han-phase6` HEAD: `e72a86c` (Pareto figure v2).
`scripts/hf_sync.py --go` is idempotent; call after every cycle.

### What to NOT do

- Do NOT run T-sweep WITHOUT the anti-dead stack — that's vanilla
  TXCDR at different T, which we already have from Phase 5.7. This
  handover is specifically Track 2 (anti-dead stack + TopK) at
  different T.
- Do NOT retrain Track 2 at T=5 under a new arch name — the
  `agentic_txc_10_bare` ckpts at seeds 42/1/2 already exist and are
  the reference point.
- Do NOT use matryoshka / contrastive add-ons here. Phase 6.2 showed
  they don't help at T=5; the point of this experiment is to isolate
  the T axis.
