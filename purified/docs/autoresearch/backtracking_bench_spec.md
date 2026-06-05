# Synthetic benchmark spec — backtracking (self-exciting recovery)

**Status:** spec / preregistration. **Not yet run** (stage 6 of the
autoresearch loop), gated on the due-diligence margin in § 8.

**Provenance.** Motivated by autoresearch investigation #1, which *measured*
backtracking in real CoT reasoning to be strongly self-exciting (record:
[`backtracking_record.md`](backtracking_record.md); the validated generative
mirror is the basis for Layer 1 below). This is the synthetic *analogue* of
that measured property — the first real-task-motivated synthetic benchmark.
Built to the conventions in
[`synthetic_benchmark_guidance.md`](../synthetic_benchmark_guidance.md) and the
loop's [`autoresearch_spec.md`](../autoresearch_spec.md).

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

### Layer 1 — self-exciting event dynamics
Per sequence of length `L`, produce a binary event stream `b` and hidden
intensity `λ`:

```
λ_i = σ( a + α · Σ_{l=1..K} κ_l · b_{i-l} ) ,   b_i ~ Bernoulli(λ_i)
```

- `a`: baseline (tuned to base rate ≈ 0.12, matching the measurement).
- `κ_l = exp(-l/τ)` normalized: fixed decay kernel, `K = 8`, `τ ≈ 2`.
- `α`: **self-excitation strength** — the difficulty knob (default from the
  fitted mirror; raising it widens the per-token→window gap, see § 8).
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
`K = 8`, `τ = 2`, base rate ≈ 0.12, `σ = 0`, `seq_len = 64`, `n_seqs = 4096`.

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

- **archs:** per-token SAEs (`topk_sae`, `tsae`; `T = 1`) vs window crosscoder
  (`txc_base`) and per-position (`stacked_sae`) over `T ∈ {2, 4, 8}`.
- **`d_sae`:** anchored on `F = 20` — scarce `{8, 16, 20}` + one over-complete
  reference `{40}`. Matched across archs.
- **`k_pos`:** 1 (sparsest; the conventions' default for the scarce regime).
- **window `L`:** common tiled eval window (power-of-two), `T` ∈ powers of two.
- **seeds:** {1, 2, 42}.

## 6. Validity controls (spec § 3) — and why they hold here

- **Memorization budget — satisfied by construction.** The temporal "window"
  is a binary event history of length `K`, with up to `2^K = 256` distinct
  patterns, while `F = 20`. So `d_sae ∈ [F, 2^K)` is *both* rich enough for
  the features *and* memorization-free for the `λ` probe. (This is the
  decoupling the signed-motion bench lacked — there `#windows = 2F`.)
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

## 8. Gating due-diligence (compute before running)

The benchmark only discriminates if the per-token ceiling is well below the
window's. That ceiling is computable from the fitted Layer-1 process:

```
per-token ceiling ≈ √( Var(λ_i) / Var(b_i) )      window ceiling = 1
```

Compute `Var(λ)` and `Var(b)` from the generator at the default `α`. If the
gap `1 − √(Var(λ)/Var(b))` is large (say ≥ 0.3 correlation), build and run.
If small, raise `α` (or lengthen `K`/`τ`) until history matters enough, and
re-check — the difficulty knobs exist for exactly this.

## 9. Reproduction (when built)

Generator → `src/temp_bench/data/synthetic.py:self_exciting()` + a
`toy_backtracking_selfexcite` datasource; the `λ` probe reuses the tiled-probe
machinery in `evals/`. Runs through the canonical `synthetic` pathway; metrics
at protocol ≥ 1.2.0. No `core/` edits.
