# Synthetic benchmark spec — change-point / sticky-mode recovery (dual-latent)

**Status:** spec / preregistration. **Not yet run** (stage 6 of the
autoresearch loop), gated on (a) the topic-switching measurement verdict —
which sets the persistence knob and dwell distribution
([`topic_switching_prereg.md`](topic_switching_prereg.md)) — and (b) the
gating due-diligence in § 8.

**Provenance.** This is the **change-point / sticky-dwell** dynamics class — the
counterpart to the self-exciting backtracking bench
([`backtracking_bench_spec.md`](backtracking_bench_spec.md)). One generator
serves the whole class via a **persistence knob** (geometric → heavy-tailed →
absorbing): **topic-switching** (now; recurrent, sticky) and **emergent
misalignment** (later; `K_m = 2`, absorbing, broad spread). Built to
[`synthetic_benchmark_guidance.md`](../synthetic_benchmark_guidance.md) and the
loop's [`autoresearch_spec.md`](../autoresearch_spec.md).

> One such `.md` per benchmark family. This is the second real-task-motivated
> synthetic benchmark; it is frozen before it is run.

---

## 1. What it tests — the dual latent (the design is the point)

A persistent mode is **not** an order-sensitive object: once the mode is on, it
colors *every* token in the dwell, so a per-token encoder reads it off the
current activation. The order-sensitive (AC) information lives entirely at the
**transitions**. This bench carries **both latents on one substrate** and scores
them **separately**:

| latent | type | axis | predicted home | why |
|---|---|---|---|---|
| **mode `m_t`** | categorical (`K_m`) | **DC** | per-token | emitted into every token in the dwell → readable from one activation |
| **change-point `c_t = [m_t ≠ m_{t-1}]`** | binary | **AC** | window | a boundary needs `x_{t-1}` *and* `x_t`; per-token sees only `m_t` → provably ≈ chance |

The headline is the **split**: on identical data, the per-token SAE should win
the **DC** half (mode recovery) and the window encoder should win the **AC** half
(change-point localization). This is the architectural-specialization claim,
designed honestly — we **predict** the split rather than hunting a one-sided
window win. It instantiates the global-vs-local divide on a single, grounded
substrate.

This complements the signed-motion bench (a *negative* — no arch recovered order
in the scarce regime, with a memorization confound baked in at `#windows = 2F`)
and the backtracking bench (a *linear-in-history* AC latent). Here the AC latent
is a **simple adjacency** (`m_t ≠ m_{t-1}`), the easiest possible order-sensitive
quantity, and the substrate is memorization-free by construction (§ 6).

## 2. Generative process (two layers)

### Layer 1 — semi-Markov mode dynamics
Per sequence of length `L`, produce a mode sequence `m_1 … m_L`:

```
dwell in mode k for duration D ~ DwellDist(knob) ;  then transition by Π
```

- `K_m`: number of modes.
- **`DwellDist` — the persistence knob** (set from the measurement):
  *geometric(p)* (memoryless Markov) → *negative-binomial / discrete-Weibull*
  (heavy-tailed, sticky) → *∞* (absorbing). The topic instance uses the fitted
  dwell from `topic_switching_prereg.md`; the EM instance uses `K_m = 2` with
  state 2 absorbing.
- `Π`: inter-mode transition matrix (fitted for topic; trivial for EM).
- *(EM add-on, deferred)* a **ramping entry hazard** — the misaligned state's
  entry probability rises over a few steps before the visible switch (a
  *precursor*), with emission lagging the state by `δ`. This turns "time-until
  switch" into a genuinely AC precursor latent. Specified but **not** in the
  topic headline.

### Layer 2 — emission into activations
Fixed orthonormal dictionary `{u^m_1 … u^m_{K_m}, u^c_1 … u^c_C}` (mode-signature
directions + shared content directions). For each position:

```
x_i = m · u^m_{m_i}  +  Σ_{j ∈ content_i} m_j · u^c_j  +  σ · ε_i
```

- `u^m_{m_i}`: the **mode-signature** direction for the active mode — fires on
  every token in the dwell. (This is what makes mode recovery a clean DC readout
  and the per-token's expected home.)
- `content_i`: a sparse subset (size `spread`) of the `C` content directions;
  the subset's *distribution* may be mode-dependent (a topic's vocabulary).
  **`spread` is the difficulty knob** for the local-feature half.
- `m, m_j`: folded-normal magnitudes; `σ`: optional noise.

### Default parameters (topic instance)
`d_in = 64`, `K_m = 8` modes, `C = 12` content dirs (so **`F = 20`** directions,
matching the backtracking bench for comparability), `spread = 3`, `σ = 0`,
`seq_len = 64`, `n_seqs = 4096`. Dwell + `Π` from the measurement.

## 3. Ground truth

- **Feature directions (`F = 20`):** `K_m = 8` mode-signature + `C = 12` content
  directions (orthonormal). Recovered via cosine-AUC (`eAUC`), reported as two
  named sets (mode vs content) — never pooled.
- **Latent A — mode `m_t`** (categorical, `K_m`): **DC**. Chance =
  majority-class rate; oracle = 1. Exposed by the generator.
- **Latent B — change-point `c_t = [m_t ≠ m_{t-1}]`** (binary): **AC**. Chance =
  base switch rate; oracle = 1. Exposed by the generator. *(Secondary AC latent,
  reported if informative: time-since-switch, a scalar.)*

## 4. Task + metrics

- **`mode_recovery` (DC contrast):** multinomial-logistic probe on the code →
  `m_t`, scored by balanced accuracy normalized to [chance = 0, oracle = 1].
  Split by sequence (leak-free).
- **`changepoint_recovery` (AC headline):** logistic probe on the code → `c_t`,
  normalized to [chance, oracle], **per-tile-as-example** (memorization-free,
  features = `d_sae`, *not* concatenated tiles — the fix the signed-motion bench
  forced). This is the per-token→window quantity.
- **`eAUC` (local):** decoder-atom cosine recovery of the `F` directions
  (mode set and content set separately).
- **`NMSE`:** windowed reconstruction per the conventions (`L`-tiled).

Linear probes are **mandatory** (conventions § 5): they measure what the code
makes *linearly* available, which is the architecturally meaningful quantity.
Chance/oracle are computable because we own the generator.

## 5. Grid (per the conventions doc)

- **archs:** per-token SAEs (`topk_sae`, `tsae`; `T = 1`) vs window crosscoder
  (`txc_base`) and per-position (`stacked_sae`) over `T ∈ {2, 4, 8}`.
- **`d_sae`:** anchored on `F = 20` — scarce `{8, 16, 20}` + one over-complete
  reference `{40}`. Matched across archs.
- **`k_pos`:** 1 (sparsest; the conventions' scarce-regime default;
  `k_win = k_pos · T`).
- **window `L`:** common tiled eval window `L = 32`; `T ∈ {2, 4, 8}`.
- **seeds:** {1, 2, 42}.

## 6. Validity controls (spec § 3) — and why they hold here

- **Memorization budget — satisfied by construction.** The change-point probe is
  per-tile; the number of distinct mode-tiles is `K_m^T` (e.g. `8^2 = 64`,
  `8^4 = 4096`) ≫ `F` and ≫ `d_sae` in the scarce regime. So the probe cannot
  memorize a small pattern set. (This is the decoupling the signed-motion bench
  lacked.)
- **Per-token is not assumed at chance — its ceiling is quantified (§ 8).** From
  `m_t` alone, `P(c_t = 1)` ≈ the base switch rate regardless of which mode
  (both switch and non-switch tokens are "in mode k"), so the per-token linear
  ceiling for `c_t` sits ≈ chance. The reported quantity is the **gap** to the
  window, not an assumed zero.
- **Untrained-encoder control:** any claimed window change-point advantage must
  **vanish for a randomly-initialized window arch**; else it is a
  probe/architecture-access artifact, not learning.
- **DC/AC separation maintained:** `mode_recovery` (DC) and
  `changepoint_recovery` (AC) are reported on separate axes and never pooled —
  the whole result is the *contrast* between them.
- **Capability-vs-artifact:** a window that "recovers the change-point" must also
  reconstruct the features (`eAUC`, `NMSE`) — not recover the latent while
  representing nothing.

## 7. Preregistered predictions

- **P1 (mode, DC):** `mode_recovery` — per-token ≈ window ≈ oracle (the mode is
  emitted every token). A *slight per-token edge* is possible in the scarce
  regime (it need not spend atoms on cross-position structure). The point of P1
  is that the DC half is **not** a window win.
- **P2 (change-point, AC — headline):** `changepoint_recovery` — window ≫
  per-token; **per-token ≈ chance** (the adjacency floor). The gap appears at
  `T ≥ 2` (a boundary needs the adjacent pair) and is roughly flat in `T`
  thereafter, perhaps growing modestly as more context aids localization within
  a tile.
- **P3 (local features):** `eAUC` — per-token recovers the `F` directions (mode
  + content) well; the window crosscoder may trail on **content** recovery if it
  allocates atoms to boundary/window patterns. The mode-direction `eAUC` should
  be high for all archs.
- **P4 (the split is robust):** the DC-per-token / AC-window split holds across
  `d_sae` (especially in the scarce regime `d_sae ≤ F`) and seeds — the
  architectural-specialization headline.
- **Possible negatives (all reportable):** (a) the trained window code does *not*
  linearly expose the adjacency (it entangles the mode history) → change-point
  recovery stays low even though the window *sees* the boundary; (b) per-token
  rises above chance on `c_t` via mode-frequency leakage → quantify and subtract;
  (c) in the over-complete corner both archs solve everything → why the scarce
  regime is the object of study.

## 8. Gating due-diligence (compute before running)

The bench discriminates only if the two ceilings are well separated on *both*
latents:

```
change-point:  per-token ceiling ≈ base switch rate (chance)   window ceiling = 1
mode:          per-token ceiling ≈ 1 (oracle)                  window ceiling ≈ 1
```

Before building: from the generator at the chosen `K_m`, dwell, and `Π`,
compute (i) the **best linear predictor of `c_t` from `m_t`** — confirm it sits
near chance (if modes are so unbalanced that the current mode predicts switching,
rebalance `Π` / `K_m`); and (ii) confirm **`mode_recovery` oracle is reachable**
by a per-token probe on the noiseless emission (else the DC contrast is
uninformative). The dwell distribution is **tied to the topic measurement** so
the substrate is grounded, not arbitrary.

## 9. Reproduction (when built)

Generator → `src/temp_bench/data/synthetic.py:semi_markov_modes()` + a
`toy_changepoint_modes` datasource; the `m_t` and `c_t` probes reuse the
tiled per-tile-example machinery in `evals/` (the same path
`signed_motion_recovery.py` / `synthetic_recovery.py` use). Runs through the
canonical `synthetic` pathway; metrics at protocol ≥ 1.2.0. No `core/` edits.

**EM instantiation (later, gated on a paid per-span judge labeler):** same
generator with `K_m = 2`, state 2 absorbing, broad `spread`, and the ramping
entry-hazard add-on; the change-point becomes a single rare onset and the
*precursor* latent (time-until-switch) is the AC quantity of interest.
