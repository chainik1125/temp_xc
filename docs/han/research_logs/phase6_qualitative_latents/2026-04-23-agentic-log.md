---
author: Han
date: 2026-04-23
tags:
  - results
  - in-progress
---

## Phase 6.1 agentic autoresearch log — push TXC qualitative

Mirror of Phase 5.7's [`2026-04-21-agentic-log.md`](../phase5_downstream_utility/2026-04-21-agentic-log.md).
Each cycle appends a section; reference table at the top tracks the
current qualitative champion.

### Operating rules

**Frozen (never changed by a cycle):**

- Base arch recipe for the TXC family: matryoshka + multi-scale InfoNCE
  (`MatryoshkaTXCDRContrastiveMultiscale`), `T=5`, `k_pos=100`,
  `n_contr_scales=3`, `γ=0.5`, `α=1.0`, `seed=42`, 25 000 steps.
  *Track 2* (see below) can deviate from this to explore minimal
  baselines.
- Training cache: FineWeb + Gemma-2-2B-IT L13 pre-cached activations.
- Evaluation corpora: concat_A + concat_B (1819 total tokens) for
  autointerp; concat_C_v2 for UMAP follow-on.
- Autointerp pipeline: `run_autointerp.py` with
  `claude-haiku-4-5`, top-8 features by variance, top-10 contexts each.
- Sparse-probing guard: last_position + mean_pool at seed=42, k_feat=5.
  Candidate must hold within 0.01 of `agentic_txc_02` (0.77 / 0.80 test
  AUC) to count as a paper-defensible win.

**Editable per cycle:**

- Additional loss terms (AuxK, diversity penalty).
- Decoder constraints (unit-norm, grad-parallel removal).
- Init (geometric median for `b_dec`).
- Sparsity mechanism (BatchTopK, lower k).
- Contrastive chain length / window width.

**Kill criterion:** if 3 consecutive cycles produce Δ_autointerp ≤ 0
with no new mechanistic insight, stop and escalate.

### Reference table (what each cycle tries to beat)

| Candidate           | Alive | Autointerp /8 | Probe AUC (last / mean) | Cycle |
|---------------------|-------|---------------|-------------------------|-------|
| `agentic_txc_02`    | 0.39 †| **2**         | 0.775 / 0.799           | baseline |
| `tsae_paper` (ref)  | 0.73  | 6             | n/a (not window)        | Phase 6 |
| `agentic_txc_09_auxk` | _tbd_ | _tbd_ | _tbd_ | **Cycle A** |

† Reproduced on this pod at alive=0.3688 on 2048-token random L13
sample, L0 per-window 500 (= 100 per token). Matches briefing value
within rounding.

### Two parallel tracks

The 8-cycle plan in
[[2026-04-23-handover-txc-qualitative]] is **Track 1** — incremental
improvements on top of `agentic_txc_02` that preserve Phase 5's
sparse-probing structure.

**Track 2** is a minimal-intervention control: a bare window-based
TXC encoder (no matryoshka, no contrastive) + the full `tsae_paper`
anti-dead stack (AuxK + unit-norm decoder + grad-parallel removal +
geom-median init). Tests whether window-encoding alone, with the
anti-dead machinery, is sufficient for qualitative. Runs in parallel
to Track 1; reported alongside each Track-1 cycle. Priority lower
than Track 1 cycles by default; Track 2 escalates only if Track 1
stalls.

### Cycle format

```
### Cycle X — {short name}        [Track N]
**Reference to beat**: {config}, autointerp = X / 8.
**Hypothesis**: 1–3 sentences.
**Change**: file paths + new class / dispatcher entry.
**Code**: commit hash.
**Train result**: final loss, alive fraction, L0, elapsed.
**Eval result**: autointerp X / 8, sparse-probe AUC (last/mean).
**Verdict**: WIN | TIE | LOSS.
**Takeaway**: what we learned.
**Next**: one concrete follow-up.
```

### Cycle A — AuxK loss on multiscale TXC        [Track 1]

**Reference to beat**: `agentic_txc_02`, autointerp = 2 / 8,
alive = 0.39, probe AUC 0.775 (last) / 0.799 (mean).

**Hypothesis**: `agentic_txc_02` loses qualitative because its
multi-scale contrastive recipe never revives features that die early
in training. `tsae_paper`'s single biggest anti-dead mechanism is
AuxK: a parallel reconstruction from the top-k DEAD features that
pulls them back into the active set. Porting AuxK onto the
multi-scale contrastive TXC (without touching matryoshka, contrastive,
or topk sparsity) should increase alive fraction by ≳ 0.30 and
autointerp score by ≳ 2 labels, at minimal cost to sparse-probing
AUC (AuxK shapes only dead features, which by definition weren't
contributing to probing anyway).

**Change**:

- New class `MatryoshkaTXCDRContrastiveMultiscaleAuxK` in
  [`src/architectures/matryoshka_txcdr_contrastive_multiscale_auxk.py`](../../../src/architectures/matryoshka_txcdr_contrastive_multiscale_auxk.py).
  Subclasses `MatryoshkaTXCDRContrastiveMultiscale`; overrides `forward`
  on the pair-window path to compute AuxK and add `auxk_alpha · L_aux`
  to the total.
- `num_tokens_since_fired` buffer (shape `(d_sae,)`, long) tracks
  per-feature staleness; advanced by `B · T` each step, reset to 0
  for features that fired.
- Dead features: `num_tokens_since_fired >= 10_000_000` (paper
  default → features die after ~1953 training steps without firing
  at `B=1024, T=5`).
- AuxK decode: gate pre-ReLU activations by dead mask, top-k=512 per
  sample, decode through the full-scale decoder `W_decs[T-1]` (no bias).
- AuxK loss: `MSE(x_cur − x_hat_full.detach(), aux_decode) /
  var(x_cur − x_hat_full)`, averaged. `x_hat_full` is detached so AuxK
  gradients only flow through the `W_decs[T-1]` slice via the dead
  features, not through the rest of the primary loss.
- Weight `auxk_alpha = 1/32 ≈ 0.03125` (paper's default).

Dispatcher entry `agentic_txc_09_auxk` added in
[`experiments/phase5_downstream_utility/train_primary_archs.py`](../../../experiments/phase5_downstream_utility/train_primary_archs.py).
Encoding wired in
[`experiments/phase6_qualitative_latents/encode_archs.py`](../../../experiments/phase6_qualitative_latents/encode_archs.py).
Probing wired in
[`experiments/phase5_downstream_utility/probing/run_probing.py`](../../../experiments/phase5_downstream_utility/probing/run_probing.py).

**Code**: commit _tbd_.

**Train result**: _tbd — waiting on 25k-step run_.

**Eval result**: _tbd_.

**Verdict**: _tbd_.

**Takeaway**: _tbd_.

**Next**: _tbd_.
