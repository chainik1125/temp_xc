# Briefing: implement the AC-only signed-motion bench

**Audience.** A post-compact agent picking up where the current session
leaves off. Read this top-to-bottom before touching code.

**Mission.** Implement the AC-only signed-motion synthetic benchmark
proposed in [`docs/frequencybenchideas.md`](frequencybenchideas.md) §5,
run a 4-arch × 4-`k_pos` × 3-seed pilot sweep, surface results.

The current DC-only synthetic benches (`coupled_hmm`, `markov_chain_support`)
probe only temporal smoothing. The AC-only bench tests a strictly
orthogonal axis — *order-sensitive* recovery — and carries a tight
data-processing-inequality (DPI) impossibility result for any per-token
encoder. This is the missing piece in the bench suite.

---

## Read first (in order)

1. [`docs/framework_v2.md`](framework_v2.md) — the framework spec. Hard rules
   on the canonical pathway, code-version stamping, and plugin extension.
2. [`CLAUDE.md`](../CLAUDE.md) — the 5 hard rules. **Do not edit
   `temp_bench/core/`.** Plugin extension only.
3. [`docs/frequencybenchideas.md`](frequencybenchideas.md) §5 — the bench
   spec itself. The DPI proof in §5.3 is the load-bearing theorem.
4. [`REPRODUCTION_REPORT.md`](../REPRODUCTION_REPORT.md) — current state of
   the DC-only synthetic reproduction. Headline numbers for context.

## Context

- Branch: `arxiv` off `origin/final-aniket`. All work in `purified/`.
- Active archs (4): `txc_base`, `topk_sae`, `stacked_sae`, `tsae`. TFA and
  TXC-pro were removed; historical leaderboard rows for them are filtered
  out by `deprecated_archs = {"txc_pro", "tfa", "tfa_pos"}` in the renderer
  and populate scripts.
- Existing synthetic benches use `d_sae=20`, `n_steps=10K`, `batch=1024`,
  3 seeds. Headline cell: `txc_base` gAUC = 0.971 ± 0.017 at `k_pos=1` on
  the coupling bench. See REPRODUCTION_REPORT.md.

## What you are building

A new generator + datasource + evaluator. Three files of new code, two
config edits, one sweep launcher. No changes to `core/`.

### The data process (FrequencyBench §5)

Parameters:
- `M = 19` (odd prime alphabet size)
- `v = 9` (high-frequency step; `gcd(9, 19) = 1` → orbit covers Z_19)
- `d_in = 40` (same as denoising bench)
- `seq_len = 64`
- `n_seqs = 4096`
- `σ = 0` (noiseless for the first cut — keeps DPI proof exactly tight)

Per sequence:
1. Sample `S ~ Unif({-1, +1})` (the hidden sign)
2. Sample `B ~ Unif(Z_M)` (the random starting phase)
3. Emit `Q_t = B + S·v·t (mod M)` for `t = 0, …, seq_len-1`
4. Embed: `x_t = u_{Q_t} + σ·ε_t` where `{u_0, …, u_{M-1}}` are random
   orthonormal directions in R^{d_in}

**Ground truths exposed by the generator:**
- `alphabet_features` ∈ R^{M × d_in} — the M orthonormal `u_a` directions
- `sign_labels` ∈ {-1, +1}^{n_seqs} — the hidden S per sequence
- `phase_labels` ∈ Z_M^{n_seqs} — the hidden B per sequence (for diagnostics)

The activation tensor `x` has shape `(n_seqs, seq_len, d_in)`, same as
existing benches.

### Why this is the right test

- **Per-token marginal carries zero info about S.** `Q_t | S = s ~
  Unif(Z_M)` because `B` is uniform → `I(S; Q_t) = 0` exactly.
- **DPI proof.** For any encoder `Z = φ(x_t)`, the chain
  `S → Q_t → x_t → Z` plus `I(S; Q_t) = 0` gives `I(S; Z) = 0`. So **no
  per-token SAE — at any width, sparsity, or nonlinearity — can recover
  sign**.
- **Window archs can solve it.** The signal lives in
  `Q_{t+1} − Q_t = S·v (mod M)`. A T=5 window encoder learning a
  zero-mean (AC) filter pair `(-u_{prev}, +u_{curr})` recovers sign at
  oracle accuracy.

This is the impossibility result the existing benches lack — there, SAEs
*empirically* trail TXC on global recovery but no proof rules out a
larger SAE catching up.

## Implementation spec

### 1. Generator: `src/temp_bench/data/synthetic.py`

Add a new function `signed_motion(...)` alongside `coupled_hmm` and
`markov_chain_support`. Same return shape — a `SyntheticData` namedtuple.

```python
def signed_motion(
    *,
    M: int = 19,
    v: int = 9,
    d_in: int = 40,
    seq_len: int = 64,
    n_seqs: int = 4096,
    sigma: float = 0.0,
    seed: int = 0,
) -> SyntheticData:
    """AC-only signed-motion bench (FrequencyBench §5).

    Q_t = B + S·v·t (mod M) with S ~ Unif({-1,+1}), B ~ Unif(Z_M).
    Activation x_t = u_{Q_t} + sigma·noise.

    By DPI, no per-token encoder can recover S (I(S; Q_t) = 0 because
    B is uniform). Window encoders can recover S via zero-mean
    temporal filters reading Q_{t+1} - Q_t.
    """
    rng = np.random.default_rng(seed)
    if math.gcd(v, M) != 1:
        raise ValueError(f"v ({v}) must be coprime to M ({M})")
    if M > d_in:
        raise ValueError(f"M ({M}) > d_in ({d_in})")

    # Orthonormal alphabet directions u_a ∈ R^{d_in}.
    raw = rng.standard_normal((d_in, d_in))
    Q, _ = np.linalg.qr(raw)
    alphabet = Q[:M]                                                # (M, d_in)

    # Hidden labels.
    S = rng.choice([-1, +1], size=n_seqs).astype(np.int32)          # (n_seqs,)
    B = rng.integers(0, M, size=n_seqs).astype(np.int32)            # (n_seqs,)

    # Emit Q_t.
    t = np.arange(seq_len)[None, :]                                 # (1, T)
    qmat = (B[:, None] + S[:, None] * v * t) % M                    # (n_seqs, T)

    # Embed: x_t = u_{Q_t}.
    x = alphabet[qmat]                                              # (n_seqs, T, d_in)
    if sigma > 0:
        x = x + sigma * rng.standard_normal(x.shape).astype(np.float32)

    return SyntheticData(
        x=torch.from_numpy(x.astype(np.float32)),
        emission_features=torch.from_numpy(alphabet.astype(np.float32)),
        hidden_features=None,                                       # sign is not a direction
        support=None,                                               # symbolic, no support tensor
        hidden_support=torch.from_numpy(S.astype(np.float32)[:, None].repeat(seq_len, axis=1)),
        seq_len=seq_len,
        d_in=d_in,
        extra={
            "sign_labels": torch.from_numpy(S),
            "phase_labels": torch.from_numpy(B),
            "M": M, "v": v,
        },
    )
```

(`SyntheticData` may need an `extra: dict | None = None` field added in
`synthetic.py`. Add it as `Optional[dict] = None` with a default `None`;
this is backward-compatible.)

Then register in the `_GENERATORS` dict at the bottom of the file:

```python
_GENERATORS = {
    "markov":  markov_chain_support,
    "coupled": coupled_hmm,
    "signed_motion": signed_motion,
}
```

### 2. Datasource entry: `configs/data.yaml`

Add a `toy_signed_motion_M19_d40` entry under the existing toy entries:

```yaml
  toy_signed_motion_M19_d40:
    category: synthetic
    generator: signed_motion
    params:
      M: 19
      v: 9
      d_in: 40
      seq_len: 64
      n_seqs: 4096
      sigma: 0.0
    notes: |
      AC-only signed-motion bench (FrequencyBench §5). Q_t = B + S·v·t
      (mod M=19) with v=9, alphabet embedded as 19 orthonormal directions
      in R^40. By DPI, no per-token encoder can recover S.
```

### 3. Evaluator: `src/temp_bench/evals/signed_motion_recovery.py`

New class `SignedMotionRecovery` with `protocol_version="1.0.0"`.
Returns three metrics:

- **`s_temp`** (headline): normalized sign-recovery score
  `(A_model − 0.5) / (1.0 − 0.5) = 2·(A_model − 0.5)`.
  Range: 0 = chance, 1 = oracle.
- **`alphabet_eauc`**: cosine-AUC of decoder atoms vs the M alphabet
  directions (the existing `_feature_recovery_auc` helper from
  `synthetic_recovery.py` works — pass `data.emission_features` as targets).
- **`atom_dc_fraction`**: per-window-arch diagnostic. For each decoder
  atom with T temporal slices `(d_in,)`, compute
  `||mean_t(atom[t])||² / sum_t ||atom[t]||²`. Average over atoms.
  Predicted ≪ 1 for `txc_base` if it learned zero-mean filters; defined
  as `None` for T=1 archs.

**Probe training:**

```python
from sklearn.linear_model import LogisticRegression

def _train_sign_probe(model, x_eval, sign_labels, seq_len, T):
    """Train a logistic regression on (concat over T) window codes → sign."""
    device = next(model.parameters()).device

    # Materialise window codes.
    consumes = getattr(model, "consumes", "token")
    n_seqs = x_eval.shape[0]

    # Use a held-out probe split: first half train, second half eval.
    split = n_seqs // 2

    def codes_for(x_subset):
        # For each sequence, take a single window starting at t=0
        # (or average over multiple windows — easier for v0: just take t=0).
        x_window = x_subset[:, :T, :].to(device, dtype=torch.float32)
        if consumes == "token":
            # Flatten window into stacked per-token codes.
            B, T_, d_in = x_window.shape
            z = model.encode(x_window.reshape(-1, d_in))           # (B*T, d_sae)
            z = z.reshape(B, T_, -1)
        else:
            z = model.encode(x_window)                              # (B, T, d_sae)
        return z.reshape(z.shape[0], -1).detach().cpu().numpy()    # (B, T*d_sae)

    with torch.no_grad():
        z_train = codes_for(x_eval[:split])
        z_eval  = codes_for(x_eval[split:])

    y_train = sign_labels[:split].cpu().numpy()
    y_eval  = sign_labels[split:].cpu().numpy()

    clf = LogisticRegression(C=1.0, max_iter=1000, n_jobs=1)
    clf.fit(z_train, y_train)
    return clf.score(z_eval, y_eval)                                # accuracy in [0, 1]
```

Plug this into the evaluator's `eval()` method. Wire in the seed-passthrough
pattern from `SyntheticRecovery` (use `spec.extra["training_seed"]` to
re-materialise the data so feature directions match).

Register the evaluator by adding `SignedMotionRecovery` to the appropriate
`__init__.py`, and add an evaluator-name routing entry in the relevant
config (see how `synthetic_recovery` is wired — same pattern).

### 4. Sweep launcher: `scripts/run_ac_minisweep.sh`

Copy `scripts/run_synthetic_minisweep.sh` and adjust:

```bash
ARCHS=(txc_base topk_sae stacked_sae tsae)
DATASOURCES=(toy_signed_motion_M19_d40)
K_POSES=(1 2 3 4)
SEED=${SEED:-1}
N_STEPS=${N_STEPS:-10000}
BATCH=${BATCH:-1024}
```

Wall time estimate: 4 archs × 4 k_pos × 1 seed × ~60-200s/cell ≈ 30-50 min
solo per seed on the 5090. Three seeds in parallel (per the
`feedback_parallel_gpu` memory) ≈ 1 hour total. d_sae=19 ≈ d_sae=20
existing → similar per-cell wall.

### 5. Tests

Add a smoke test in `tests/test_v2_synthetic_e2e.py` (or a new
`test_ac_bench.py`):

```python
def test_signed_motion_generator_shapes():
    from temp_bench.data.synthetic import signed_motion
    data = signed_motion(M=7, v=3, d_in=10, seq_len=8, n_seqs=16, seed=0)
    assert data.x.shape == (16, 8, 10)
    assert data.emission_features.shape == (7, 10)
    # Sign labels in {-1, +1}
    signs = data.extra["sign_labels"]
    assert set(signs.unique().tolist()).issubset({-1, 1})

def test_signed_motion_dpi_holds():
    """Per-token marginal of Q_t is uniform over Z_M regardless of S."""
    from temp_bench.data.synthetic import signed_motion
    data = signed_motion(M=7, v=3, d_in=10, seq_len=64, n_seqs=8192, seed=0)
    # Reverse-embed: which symbol does each x_t encode?
    # x_t = u_{Q_t}, so argmax of x_t @ u_a^T gives Q_t.
    feats = data.emission_features                                  # (7, 10)
    sims = torch.einsum("ntd,md->ntm", data.x, feats)               # (N, T, 7)
    q_recovered = sims.argmax(dim=-1).flatten()                     # (N*T,)
    # Marginal should be ~uniform over Z_7.
    counts = torch.bincount(q_recovered, minlength=7).float()
    counts /= counts.sum()
    assert torch.allclose(counts, torch.full((7,), 1/7), atol=0.02)
```

Run `.venv/bin/python -m pytest tests/ -q` to confirm 41/41 (current
39 + 2 new).

## Predictions (preregister before running)

| arch | s_temp | alphabet_eauc | atom_dc_fraction | Reasoning |
|---|---|---|---|---|
| `topk_sae` | **0.000 ± 0.000** | high (≥ 0.85 at k_pos ≥ 2) | N/A (T=1) | DPI proof. Should recover alphabet directions but cannot beat chance on sign. |
| `stacked_sae` | **0.000 ± 0.000** | high | 1.0 (per-position kernels are DC by construction) | Same DPI argument applied per-position. |
| `tsae` (T=1) | 0.000 ± 0.000 | high | N/A | Same — T=1, same impossibility. |
| `txc_base` (T=5) | **> 0.0 expected**, exact value empirical | moderate-to-high | ≪ 1 if it learned the differencing filter | Window encoder *can* learn the zero-mean filter pair `(-u_prev, +u_curr)`. |

**Headline target.** `txc_base` `s_temp ≥ 0.5` while every SAE-family arch
sits at `s_temp ≤ 0.02 ± 0.02` across seeds. That's the architectural-gap
result the existing benches lack.

If `txc_base` is also at chance: it means even the TXC window encoder
didn't learn the differencing filter, and we need to dig into the loss
or train longer. (Suggests low-pass bias in the inductive prior — which
is itself a result worth reporting.)

## What NOT to do

- Do NOT re-introduce TFA or txc_pro. They were deliberately removed.
- Do NOT change the per-section `d_sae` for the existing coupling /
  denoising benches. Those committed numbers are stable.
- Do NOT edit `temp_bench/core/`. Plugin extension only.
- Do NOT add `σ > 0` noise or sweep multiple `v` values in this first
  pilot. Land the noiseless v0 first; add noise after the predictions
  are confirmed/refuted.
- Do NOT bump the d_sae or k_pos sweep for the existing benches in
  this work. This is an additive change.
- Do NOT use an MLP probe in the first cut. Logistic regression with
  `C=1.0` is the agreed-upon probe class. Escalate to MLP only if linear
  underfits with `txc_base` (which would be a surprise — see
  predictions).

## After implementation

1. Run `python run.py validate` — expect 6 archs, 9 datasources (added 1),
   1 evaluator added.
2. Run tests — expect 41/41.
3. Launch the 3-seed sweep (see `scripts/run_ac_minisweep.sh`).
4. Write a populate script analog to
   `populate_repro_report_multiseed.py` that reads the AC-bench rows from
   the leaderboard and produces a markdown table per arch × k_pos.
5. Render an AC-bench figure (one panel: s_temp vs k_pos per arch). Drop
   it in `docs/figs/`.
6. Append an "AC-only bench" section to `REPRODUCTION_REPORT.md` with the
   tables, the figure reference, and a one-line headline.
7. Commit. The commit message should call out: (a) what the bench tests
   (DPI tightness, order-sensitivity), (b) whether the architectural gap
   replicated, (c) any surprises.

If the architectural gap holds, this is the cleanest bench result in the
paper. If it doesn't hold, that's also a publishable result — TXC's
inductive bias may be more low-pass than advertised.

## Glossary (quick reference)

- **DPI**: data-processing inequality. `X → Y → Z` Markov chain ⇒
  `I(X; Z) ≤ I(X; Y)`. Used in §5.3 to prove per-token encoders can't
  recover sign.
- **AC / DC**: from FrequencyBench §1. DC = time-constant component
  (`mean_t e_{j,t}`). AC = zero-mean component (`e_{j,t} − mean_t`).
- **k_pos**: per-token sparsity (number of atoms allowed to fire per
  token). For window archs, `k_win = k_pos × T` is the window-level L0;
  clipped at `d_sae` when `k_pos × T > d_sae`.
- **s_temp**: normalized temporal probe score from FrequencyBench §3.
  `(A_model − A_loc⋆) / (A_oracle − A_loc⋆)`. For sign: `2·(A − 0.5)`.

## Commit so far (for your reference)

```
192edc9e arxiv: remove TFA + recalibrate synthetic-bench framing
572a3380 arxiv: delete docs/components/ and docs/aniket/
110ff67c arxiv: drop synthetic d_sae from 40 to 20 (scarce-dictionary regime)
5dd7337b arxiv: remove txc_pro from active registry
032609ff arxiv: validate n_steps=10K choice with convergence spot-check
8255777b arxiv: § 4 synthetic reproduction extended to 3 seeds — narrative robust
0d8e506e arxiv: § 4 synthetic reproduction on framework v2 — Fig 2 narrative reproduces
```

You're picking up at `192edc9e`. Good luck.
