# Synthetic benchmark spec — backtracking (self-exciting recovery)

**Status:** spec / preregistration. **Not yet run** (stage 6 of the
autoresearch loop), gated on the due-diligence margin in § 8.

**Provenance.** Motivated by autoresearch investigation #1, which *measured*
backtracking in real CoT reasoning to be strongly self-exciting (record:
[`backtracking_record.md`](measurement.md); the validated generative
mirror is the basis for Layer 1 below). This is the synthetic *analogue* of
that measured property — the first real-task-motivated synthetic benchmark.
Built to the conventions in
[`README.md`](../README.md) and the
loop's [`README.md`](../README.md).

> One such `.md` per benchmark: each real-task-motivated synthetic benchmark
> gets a standalone spec like this, frozen before it is run.

---

## 1. What it tests

Whether an architecture's dictionary code exposes the **hidden self-exciting
intensity** of an order-sensitive event stream — i.e. can a linear reader
recover "how likely is an event here, given the recent event history" from the
code. A per-token encoder sees only the current activation (a single sample
*from* the intensity); a window encoder sees the event *history* (the
*drivers* of the intensity). The headline is the per-token→window gap.

This is the order-sensitive (AC) axis, like the signed-motion bench — but with
a latent that is **linear in the history** (probe-friendly) and a substrate
**free of the memorization confound** (§ 6).

## 2. Generative process (two layers)

> **Pre-run amendment (data-driven, independent of any bench result).** The
> kernel length was revised **`K = 8 → 2`** after a held-out model selection on
> the real Ward labels (`backtracking_kernel_order.py`): held-out NLL is
> minimized at `K = 2` (effective memory ≈ 2 sentences) and *worsens* for
> `K ≥ 3` — the lag-4/8 weight in the old K=8 fit was overfit (BIC favours large
> K only because its penalty is too weak at ~20k events). The § 8 gate was then
> re-run at `K = 2` and passes more cleanly than before (gap ≈ 0.50 @ T=2).
> Because `K` is chosen by fit to real data, not by bench outcome, this is a
> faithfulness fix, not metric shopping.

### Layer 1 — self-exciting event dynamics
Per sequence of length `L`, produce a binary event stream `b` and hidden
intensity `λ`:

```
λ_i = σ( a + α · Σ_{l=1..K} κ_l · b_{i-l} ) ,   b_i ~ Bernoulli(λ_i)
```

- `a`: baseline (intercept re-tuned so the trend-off base rate ≈ 0.12, matching
  the measurement).
- `κ_l = exp(-l/τ)` normalized: fixed decay kernel, **`K = 2`**, `τ = 2`
  (kernel length set by held-out model selection — see the pre-run amendment
  above; the exp form deliberately ignores the overfit lag-3/4/8 bumps the raw
  logistic-AR fit throws).
- `α`: **self-excitation strength** — **`α = 3.06`** (= Σ of the fitted `K = 2`
  kernel; held at the faithful value, **not** tuned for gap — the gap actually
  *grows* as `α` shrinks, so widening it would be degenerate). Effective
  weights `α·κ = [1.90, 1.16]`.
- Position trend set to **0** for the headline (the trend is a DC component
  already covered by the coupling/denoising benches and was isolated by the
  `N2` control in the measurement; an optional trend add-on may be specified
  separately).

### Layer 2 — emission into activations
Each sentence `i` → activation `x_i ∈ R^{d_in}` over a fixed orthonormal
dictionary `{u_bt, u_1…u_{K_c}}`:

```
x_i = b_i · m · u_bt  +  Σ_{j ∈ content_i} m_j · u_j  +  σ · ε_i
```

- `u_bt`: backtracking feature, fires iff `b_i = 1`.
- `content_i`: a sparse random subset (size `n_c`) of the `K_c` content
  features per sentence (the non-backtracking sentence content / filler).
- `m, m_j`: folded-normal magnitudes; `σ`: optional noise.

### Default parameters
`d_in = 64`, `K_c = 19` (so **`F = 20`** feature directions), `n_c = 3`,
**`K = 2`** (held-out-selected), `τ = 2`, `α = 3.06`, base rate ≈ 0.12
(intercept re-tuned to hit it with the trend off), `σ = 0`, `seq_len = 64`,
`n_seqs = 4096`.

## 3. Ground truth

- **Feature directions (`F = 20`):** `u_bt` + 19 content directions
  (orthonormal). Recovered via cosine-AUC (`eAUC`).
- **Dynamical latent:** the conditional intensity `λ_i` (continuous), a
  deterministic function of the *past* events `b_{i-1..i-K}`. Recovered via a
  linear probe (§ 4). `λ_i` and `b_i` are exposed by the generator.

## 4. Task + metrics

- **`lambda_recovery` (headline):** linear-regression probe on the code →
  `λ_i`, scored by held-out correlation, normalized to [chance = 0,
  oracle = 1]. Split by sequence (leak-free). Linear probe is mandatory
  (measures what the code makes *linearly* available).
- **`eAUC` (local):** decoder-atom cosine recovery of the `F` directions.
- **`NMSE`:** the windowed reconstruction (per the conventions doc).

Chance/oracle are computable because we own the generator (chance = a probe
with no history info; oracle = the true `λ_i`).

## 5. Grid (per the conventions doc)

> **Pre-run amendment (fairness, 2026-06-08 — independent of any bench result).**
> The backbone was switched **TopK → BatchTopK** for *every* arch. T-SAE already
> used BatchTopK (Bussmann et al. — the strong backbone: BatchTopK during
> training → a fixed JumpReLU threshold at inference); the plain TopK baselines
> made "backbone" an uncontrolled confound that favoured T-SAE. The comparison
> now puts all archs on the *same* BatchTopK→JumpReLU backbone (+ AuxK
> dead-feature revival + decoder unit-norm + grad-orthogonalisation), so the only
> remaining variable is **decode structure**. Two further fairness fixes ride
> along: **(i) throughput** — window archs see `batch_size = 1024 // T` so every
> arch reconstructs ~1024 token-positions/step and the BatchTopK pool is the same
> `B·T = 1024` granularity (the old uniform `batch_size = 1024` let window archs
> see up to `T×` more data/step); **(ii) post-squash budget** — the crosscoder's
> squashed code uses `k_pos` actives **per window** (= `k_win // T`), correcting
> the legacy `k_win = k_pos·T` over-count (each squashed atom is reused at all `T`
> positions). These are fairness corrections chosen *before* the run, not metric
> shopping. The published TopK archs are left untouched so the § 4
> coupling/denoising and signed-motion results stand.

- **archs (all on the BatchTopK backbone):** per-token SAEs (`batchtopk_sae`,
  `tsae`; `T = 1`) vs **two** window crosscoder variants — pre-squash
  (`txc_batchtopk_pre`) and post-squash (`txc_batchtopk_post`) — and per-position
  (`stacked_batchtopk`), each over `T ∈ {2, 4, 8}`. The pre/post split is an open
  architectural question (select per-position survivors *then* squash, vs squash
  *then* select), so we measure both.
- **`d_sae`:** anchored on `F = 20` — scarce `{8, 16, 20}` + one over-complete
  reference `{40}`. Matched across archs.
- **`k_pos`:** 1 (sparsest; the conventions' default for the scarce regime), plus
  a `k_pos = 2` sparsity-robustness anchor at `d_sae = 20`.
- **window `L`:** common tiled eval window `L = 32` (power-of-two); `T ∈ {2,4,8}`
  ∈ powers of two. (`T = 8` is kept to *demonstrate* the saturation empirically:
  at `K = 2` recovery plateaus by `T = 4` — see § 8 — so `T = 8` confirms the
  curve has flattened rather than adding discriminative power.)
- **seeds:** {1, 2, 42}.

## 6. Validity controls (spec § 3) — and why they hold here

- **Provable floor — not a pattern-count budget.** At `K = 2` the hidden
  history has only `2^K = 4` distinct patterns (< `F = 20`), so the
  pattern-count budget does not apply — and it does not need to. The safeguard
  here is **provable**: the discriminating baseline (per-token, `T = 1`) encodes
  each token independently, so its code is a function of `b_i` (and independent
  content) alone; by the **data-processing inequality** it cannot recover `λ_i`
  beyond `corr = √(Var λ/Var b) ≈ 0.41`, *regardless* of `d_sae` or probe class.
  A small pattern count therefore cannot manufacture a per-token→window gap —
  per-token is floored by DPI, not by memorization. And unlike signed-motion
  (where the latent was an *interaction*, obtainable only by memorizing a small
  window set), `λ` is **linear in the history**, so a window's recovery reflects
  genuine linear exposure, not a lookup. Conventions § 5 prefers exactly this: a
  provable floor over an empirical budget. (The untrained-encoder control below
  remains as the learning-vs-access check.)
- **Untrained-encoder control:** a claimed window advantage must vanish for a
  randomly-initialized window arch; else it is a probe/architecture-access
  artifact.
- **Trend disentangling:** trend = 0 in the headline, so the latent is pure
  self-excitation (mirrors the `N2` control).
- **Per-token is not assumed at chance:** its ceiling is quantified (§ 8), and
  the gap to window is the reported quantity.

## 7. Preregistered predictions

- **P1:** per-token `lambda_recovery` ≈ its information ceiling
  `corr ≈ √(Var(λ)/Var(b))` — it sees only a single Bernoulli sample `b_i` of
  `λ_i`, not the history that determines it.
- **P2:** window `lambda_recovery` rises with `T`, approaching oracle by
  `T ≈ K` (the kernel length), *provided* the code linearly exposes the event
  history. Because the latent is logit-linear in the `b`'s, this is far more
  likely than in the signed-motion bench.
- **P3 (headline):** window > per-token by the higher-history margin; the gap
  grows with `α` and with `K`/`τ`.
- **P4:** `eAUC` — per-token archs recover the `F` directions well; the
  window crosscoder trails on local recovery (window-pattern atoms).
- **Possible negative:** if the trained window code does *not* linearly expose
  the history (entangles it), window recovery stays low even though it
  *represents* the events — a real, reportable outcome.

## 8. Gating due-diligence — **computed, PASSED**

The benchmark only discriminates if the per-token ceiling is well below the
window's:

```
per-token ceiling = √( Var(λ_i) / Var(b_i) )      window info ceiling = 1
```

**Result** (`backtracking_gating.py` → `backtracking_gating_stats.json`,
deterministic `SEED = 0`, 16k sequences). At the selected `K = 2`, `α = 3.06`,
base rate re-tuned to 0.12:

| quantity | value |
|---|---|
| per-token linear ceiling `√(Var λ/Var b)` (= empirical `corr(b,λ)`) | **0.41** |
| window linear ceiling, `T = 2 / 4 / 8` | **0.91 / 0.99 / 0.99** |
| gap `window(T=2) − per-token` | **≈ 0.50** ≥ 0.30 ✓ |

The gap clears the 0.3 bar comfortably; the window **saturates by `T = 4`**
(`T = 8` is a redundant confirmation point). Note the gap *grows* as `α`
shrinks — so we hold `α` at the faithful fitted value rather than widening it
(widening would be degenerate: `λ → ` const). `K` was set by held-out NLL on the
real labels (`backtracking_kernel_order.py`; effective memory ≈ 2 sentences,
K=8 overfits) — a data-driven, pre-run choice **independent of any bench
result**, so it does not violate the prime directive.

## 9. Reproduction (when built)

Generator → `src/temp_bench/data/synthetic.py:self_exciting()` + a
`toy_backtracking_selfexcite` datasource; the `λ` probe reuses the tiled-probe
machinery in `evals/`. Runs through the canonical `synthetic` pathway; metrics
at protocol ≥ 1.2.0. No `core/` edits.
