# POST-COMPACT PAPER-FAITHFUL CHECK — run this EVERY time context resets

**Why this file exists.** "Paper-faithful" is a constraint that erodes
silently across a context boundary: every individual decision looks
locally reasonable, and nobody re-derives the recipe after a compact.
Han's standing instruction (2026-07-28): *sanity check every post
compact — ensure we are still sticking to paper faithful.*

Run the block below **before** touching the pf arm. It takes ~20 s and
is all reads.

```bash
cd $(git rev-parse --show-toplevel)
.venv/bin/python -c "
import experiments.explorations.actmix_rlhf.cells as C
c=C.pf(2); d=c['training_cfg'].model_dump(exclude_none=True)
print('arch      ', c['arch'],        '(want agentic_txc_02_v1t)')
print('substrate ', c['datasource'],  '(want gemma_2_2b_base_l12_phase7 — BASE l12, NOT l13-IT)')
print('lr/opt    ', d['learning_rate'], d['optimizer'], '(want 3e-4 adam, CONSTANT — no scheduler)')
print('warmup    ', d.get('warmup_steps'), '(want 0 on SWEEP cells)')
print('n_steps   ', d['n_steps'],     '(want PF_N_STEPS=8000, NOT the shared N_STEPS)')
print('k_pos     ', d['arch_hparams_override']['k_pos'], '(want 100*T)')
print('batch     ', {T: C._pf_batch(T) for T in (1,2,4,6,8,10)}, '(upstream A40 schedule)')
a=C.pf(5,42,anchor=True)['training_cfg'].model_dump(exclude_none=True)
print('ANCHOR    ', 'n_steps',a['n_steps'],'batch',a['batch_size'],'warmup',a.get('warmup_steps'),
      '(FROZEN literals — must NOT track sweep edits)')
"
# anchors must still resolve to the staged paper weights
.venv/bin/python -c "
import json,experiments.explorations.actmix_rlhf.render_writeup_fig as R
print('anchor keys resolve:', len(R.anchor_train_keys()), '(want 3)')
"
```

## The five things that must stay true

1. **Substrate is BASE layer 12**, never IT-l13. The l13-IT evals were
   RETRACTED (25607c62d) after the A/B: FVU 0.0036 vs 0.0367.
2. **Anchors are upstream paper WEIGHTS, never retrained.** Their
   `train_key` is frozen by `PF_ANCHOR_FROZEN`. If a sweep-recipe knob
   ever leaks into the anchor branch, the runner silently retrains them
   and the port-vs-paper comparison dies without an error. Three
   variants of this bit on 07-28.
3. **`PF_N_STEPS` is separate from `N_STEPS`.** The shared constant
   feeds the btk arm; moving it orphans the completed btk grid.
4. **T16 is NOT an upstream cell.** Upstream's sweep is
   t2,t3,t6,t7,t8,t10,t15,t20 — no t16. Ours is an interpolation.
5. **The two deviations must travel with every claim** (below).

## KNOWN DEVIATIONS — disclosed, not patched (rule 3 bars the fix)

| # | upstream | us | status |
|---|---|---|---|
| 1 | `grad_clip 1.0` | **no gradient clipping anywhere** (verified: 0 hits for clip_grad in `core/trainer.py` and `archs/`) | disclosed |
| 2 | — | rows record `precision: bf16`, training is **fp32** (no autocast/GradScaler/.half()) | disclosed |

**Both are now in the pf figure's binding caption** (`render_writeup_fig.py`,
`--arm pf`). A caption that says "the paper's own architecture" while
omitting these overclaims — that is the specific failure this check
caught on 2026-07-28 14:3x, when the deviations were in the HANDOFF and
the LOG but *not* on the figure that ships.

## Not a fidelity risk (checked, recorded so it is not re-litigated)

- **Resident buffer** (`TEMP_BENCH_BUFFER_RESIDENT=1`): batches are
  **bitwise identical** to the host path — `torch.equal`, 10/10 at
  batch 1024 on CUDA. It removes a transfer, not a computation. It
  actually moves us *toward* upstream, which always kept the buffer
  on-device.
- **MPS device branch**: CUDA still wins wherever CUDA exists, so no
  existing pathway changed device.
