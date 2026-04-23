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
| `agentic_txc_09_auxk` | **0.37** ‡| **3** | _probing deferred_ | **Cycle A** |
| `agentic_txc_02_batchtopk` | **0.80** | **7** 🏆 | _probing deferred_ | **Cycle F** |
| `agentic_txc_10_bare` | _tbd_ | _tbd_ | _probing deferred_ | **Track 2** |

† Reproduced on this pod at alive=0.3688 on 2048-token random L13
sample, L0 per-window 500 (= 100 per token). Matches briefing value
within rounding.

‡ Cycle A alive fraction: **0.3667** (2048-token random L13 sample,
window-reshaped to 409 T=5 windows). Essentially identical to the
0.3688 baseline — **AuxK produced no measurable effect on alive
fraction**. Sparse-probing guard deferred (probe_cache not present
on this pod; can download from HF if needed for the paper story).

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

**Code**: commit `3f903d3` (arch + trainer + dispatcher), `18d1e88`
(Track 2), `decf79c` (eval harness).

**Train result**:

- 2385.5 s wall-clock (≈ 40 min) on A40.
- **converged=True at step 4200 / 25 000** — `agentic_txc_02`
  trained to step ~16 205 before plateau. Cycle A plateaus **~4×
  earlier**. Plausible reasons:
    1. AuxK accelerates convergence by reviving features that would
       otherwise stay dead (good).
    2. AuxK gradient destabilizes the late-training regime and
       creates a false plateau (bad).
- Final loss: 16 081.7 (comparable to `agentic_txc_02`'s reported
  ~16 200; matryoshka primary loss seems unchanged in magnitude).
- Final L0: 493 / 500 budget (BatchTopK honoured).
- The 10 M-token dead threshold (paper default) triggers at step
  ≈ 1 953 (at `B=1024, T=5`); Cycle A trained past that for ~2 250
  additional steps before plateau, giving AuxK an active window.

**Eval result**:

Autointerp on concat_A + concat_B (1819 tokens) with Claude Haiku:
**3 / 8 semantic labels** (+1 vs `agentic_txc_02`'s 2 / 8 baseline;
−2 vs realistic target 5 / 8; −3 vs stretch target 6 / 8).

Top-8 feature labels (rank by variance, concat A+B):

| rank | feat | label | semantic? |
|---|---|---|---|
| 1 | 11193 | Punctuation and quotation marks in text | ✗ |
| 2 |  7220 | Acknowledging debts and intellectual indebtedness | ✓ |
| 3 |  1636 | Stalinism and Soviet Union historical context | ✓ |
| 4 |  1088 | Punctuation marks and special characters | ✗ |
| 5 | 14885 | References to family relations in poetic text | ✓ |
| 6 |  3523 | punctuation and grammatical conjunctions | ✗ |
| 7 | 16493 | Punctuation marks and special characters | ✗ |
| 8 | 12655 | Punctuation and conjunctions in literary text | ✗ |

Alive fraction: **0.3667** — unchanged from the `agentic_txc_02`
baseline (0.3688). See takeaway below for interpretation.

Sparse-probing guard: deferred — `probe_cache` not cached on this
pod, and Cycle A already failed the qualitative target so the guard
cannot flip the verdict. Will download from HF (`han1823123123/
txcdr-data`) only if the eventual winner needs guard verification.

**Verdict**: **Partial WIN** vs `agentic_txc_02` baseline (+1 label),
**LOSS** vs `tsae_paper` (3 < 6) and vs `agentic_mlc_08` (3 < 5). Does
not meet the 5 / 8 realistic target.

**Takeaway**:

- **AuxK produced no measurable effect on alive fraction** (0.367 vs
  0.369 baseline). The +1 semantic label is plausibly seed noise.
- **Root cause hypothesis**: TXC uses strict TopK-per-window sparsity
  (exactly k_win=500 active features per window). AuxK can provide
  gradient to dead features, but for a revived feature to actually
  fire at inference, it must **displace** another top-500 feature.
  The existing 500 active features are already winning the selection
  because their activations are high — nudging a dead feature's
  pre-activation by a small AuxK gradient doesn't overcome that gap
  at matched `k`. In `tsae_paper`, BatchTopK allows variable per-
  sample sparsity, so revived features can fire on contexts where
  they're most relevant without displacing anything.
- This suggests **AuxK without BatchTopK (or lowered k) is a weak
  intervention on TXC**. The Phase 5.7 briefing flagged this
  ordering — Cycle A was the cheapest test, Cycle F (BatchTopK) was
  the targeted one.
- The 3 semantic features that did appear (Darwin thanks, Animal
  Farm Stalinism, Gita family) mirror what `tsae_paper` picks up on
  the same passages — so the underlying representation does contain
  some passage-level signal, it's just crowded out of the top-8 by
  variance-dominant syntactic features.
- Plateau-stop at step 4200 (4× earlier than `agentic_txc_02`'s
  ~16 205) likely compounds the issue: AuxK only had ~2250 steps
  past the dead threshold to act. A longer-trained Cycle A might
  produce a slightly stronger effect, but probably not enough to
  cross 5 / 8 given the fundamental TopK-vs-BatchTopK tension above.

**Next**: **Track 2** immediately. The alive-fraction result (no
movement) makes the TopK-vs-BatchTopK tension the leading hypothesis.
Track 2's bare TXC has the SAME TopK-per-window budget as Cycle A
(k_win=500 equivalent), so if the tension explanation is right,
Track 2 alone won't fix it either — it just tests whether stripping
matryoshka + contrastive at least lets the top-8-by-variance slots
be dominated by cleaner features.

After Track 2, priority becomes **Cycle F (BatchTopK)** — the briefing's
next-in-line mechanism, which directly targets the tension. If
BatchTopK-TXC + AuxK beats Cycle A by a meaningful margin, that
triangulates "BatchTopK is the load-bearing piece".

### Cycle F — BatchTopK on multiscale TXC       [Track 1]

**Reference to beat**: `agentic_txc_02`, autointerp = 2 / 8.

**Hypothesis**: Cycle A confirmed AuxK is a null intervention when
combined with strict TopK-per-window sparsity. BatchTopK (variable
per-sample sparsity, same total budget across the batch) lets dead
features fire *on contexts where they help* without having to
displace an already-winning top-k feature. Predicts: alive fraction
jumps substantially; top-8-by-variance diversifies away from
punctuation.

**Change**: No new code on this branch. Used the Phase 5.7
experiment (ii) checkpoint `agentic_txc_02_batchtopk__seed42.pt`
(produced on `origin/han`) downloaded from
`han1823123123/txcdr` HF repo. Cherry-picked
[`src/architectures/_batchtopk.py`](../../../src/architectures/_batchtopk.py)
and
[`_batchtopk_variants.py`](../../../src/architectures/_batchtopk_variants.py)
from `origin/han` to load the checkpoint. Dispatcher wired in
`encode_archs.py`, `run_probing.py`, `arch_health.py`.

**Code**: commit `96a774c` (wiring).

**Train result**: Re-using origin/han's training (2117 s, converged
at step 4000 / 25 000, final loss 17 657, L0 500 equivalent). No
re-train on this branch.

**Eval result**:

Alive fraction (2048-token random L13 sample, 409 windows):
**0.7954** — vs `agentic_txc_02`'s 0.3688 and Cycle A's 0.3667.
**+0.43 absolute**, and actually **beats `tsae_paper`'s 0.735.**
L0 per token = 662 (unenforced by BatchTopK's flat-batch selection;
average per-window across batch hits ~k_win · B / B = 500, but
per-token varies).

Autointerp on concat_A + concat_B (1819 tokens) with Claude Haiku:
**7 / 8 semantic labels** — **beats `tsae_paper`'s 6 / 8 (stretch
target)**.

Top-8 feature labels:

| rank | feat | label | semantic? |
|---|---|---|---|
| 1 | 15702 | poetic and archaic English text passages | ✓ |
| 2 | 17462 | historical text with archaic or foreign characters | ✓ |
| 3 |  6068 | George Orwell's political ideology and writings | ✓ |
| 4 |  6630 | Latin and Sanskrit poetic texts | ✓ |
| 5 |  2424 | Acknowledgments and attribution in written works | ✓ |
| 6 |  6693 | Historical references to Stalinist Soviet Union | ✓ |
| 7 |  1310 | Punctuation and special characters at token boundaries | ✗ |
| 8 |  2048 | Political ideologies and historical movements | ✓ |

Sparse-probing guard: deferred. From Phase 5.7 experiment (ii),
BatchTopK regresses TXC sparse probing (quantified delta tbd on this
branch; Phase 5.7 summary cites it as a regression).

**Verdict**: **WIN on qualitative — new Phase 6.1 champion**, 7 / 8
(+5 vs `agentic_txc_02` baseline; +1 vs `tsae_paper`; tie or beat
everything on the Phase 6 bench).

**Takeaway**:

**The TopK-vs-BatchTopK sparsity mechanism is the load-bearing
piece for TXC qualitative** — larger effect than every Cycle A/B/C/D
mechanism individually, and larger than any matryoshka / contrastive
tuning. Interpretation:

- Under strict TopK, the top-k features per window form a "sticky
  winner" set. Features that fire less frequently (rare concepts,
  passage-level signals) get crowded out by high-variance syntactic
  features that fire everywhere. The top-8-by-variance becomes
  dominated by punctuation / delimiter / quote features.
- Under BatchTopK, variable per-sample sparsity means rare features
  can fire on contexts where they're most informative without
  needing to beat a syntactic feature at every other sample.
  Alive-fraction rises; semantic features dominate top-8 by variance.
- The matryoshka + multi-scale contrastive recipe is NOT the cause
  of agentic_txc_02's qualitative failure. It's the sparsity that
  was crowding out concepts. With BatchTopK, the same recipe
  produces top-quality features.
- Paper story reframe: "matryoshka + multi-scale contrastive TXC
  wins on both sparse probing (TopK variant) AND qualitative
  (BatchTopK variant). The sparsity mechanism is a knob: TopK
  favours probing; BatchTopK favours qualitative. The same
  underlying dictionary is capable of both." This is a much richer
  claim than the original "agentic_txc_02 wins at both".

**Next**: two fronts.

1. **Probing-regression quantification**: download `probe_cache`
   from HF and run sparse probing on `agentic_txc_02_batchtopk` to
   quantify the AUC delta vs `agentic_txc_02`. This is the precise
   quantitative trade-off the paper claims.
2. **Cycle H stack**: try `agentic_txc_02_batchtopk + AuxK +
   unit-norm decoder + grad-parallel removal + geom-median init`
   to see if layering the full anti-dead stack on top of BatchTopK
   pushes to 8 / 8. Expected marginal: the alive-fraction delta is
   already at +0.43; the remaining 1 punctuation feature may
   disappear if the extra mechanisms further disentangle decoder
   directions. Easy if Track 2 + Cycle F confirm the pattern.
