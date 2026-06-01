---
author: Aniket
date: 2026-06-01
tags:
  - plan
  - freqbench
  - altfreq
  - txc
status: plan
---

## Alternate-frequency benchmarks — plan

**One-line:** extend FreqBench beyond its three original benches (DC / AC /
Mixed) with four new generative families that stress-test *different
temporal-frequency structures* — chirp (quadratic phase), multitone
(superposed tones), amplitude-modulation (envelope-rate class), and
relative-phase (two-channel lead/lag). All four reuse the `freq_bench`
evaluator (NTPS + order-controls) unchanged; new generators live in
`temp_bench.data.altfreq_data`.

### Provenance

These four benches are designed to extend the FreqBench AC/Mixed story from
`freq_bench_theory.md`. They are NOT present anywhere in the prior repo
history (verified). Design is grounded in the following considerations:

1. The AC bench (single-channel, linear phase) is the cleanest signed-direction
   test but it is arguably the simplest possible temporal task. Do the
   conclusions hold for **non-linear** phase (chirp), **polyphonic** emission
   (multitone), **amplitude-coded** temporal structure (AM), and **two-channel**
   phase comparison (relative-phase)?
2. Dmitry's comment about the frequency-decomposition issue: these four benches
   together probe a wider frequency-structure space and make the per-class
   R_j frequency-response meaningful beyond a single-velocity ladder.
3. All four keep `A_loc` and `A_oracle` analytically derivable and
   pre-registerable.

---

## 1. Chirp bench

### Generative model

A phase trajectory follows a **quadratic polynomial** (linear chirp):
$$
\phi(t) = \Bigl(\phi_0 + v \cdot t + a \cdot \tfrac{t(t{-}1)}{2}\Bigr) \bmod M,
\qquad t = 0, \ldots, W{-}1.
$$
Parameters per sequence: $\phi_0 \sim \text{Uniform}(\mathbb{Z}_M)$,
$v \sim \text{Uniform}(\mathbb{Z}_M)$ (carrier velocity, nuisance),
$a \in \{-a_0, +a_0\}$ equiprobable (chirp rate — the label signal).

**Emission:** one-hot$(\phi(t)) + \sigma\xi_t$, $M$ orthonormal directions.

**Ring guard.** Choose $M$ large enough that the label is decodable via
second finite differences. The formula $a \cdot t(t{-}1)/2$ uses $a \in
\{-1,+1\}$ and $t \le W{-}1$, giving a maximum quadratic advance of
$(W{-}1)(W{-}2)/2 = 105$ for $W{=}16$. Modular arithmetic is fine: the
second finite difference $\Delta^2\phi(t) = \phi(t{+}2) - 2\phi(t{+}1) +
\phi(t) \equiv a \pmod{M}$ for all $t$, so with $a \in \{-1,+1\}$ and
$M{=}64$: $\Delta^2\phi \bmod M \in \{1, 63\}$, perfectly discriminable.
We draw $v \sim \text{Uniform}(\mathbb{Z}_M)$ (nuisance); wrapping fine.

**Minimum window for oracle.** A degree-2 polynomial is determined by 3
evaluation points. A single adjacent pair only determines $v + a\cdot t$
(first-order difference), not $a$ alone. The second finite difference
$\Delta^2\phi(t) = a$ requires 3 consecutive phases. Hence a window arch
with $T \ge 3$ can, in principle, recover $a$ exactly at $\sigma = 0$.
A per-token arch ($T = 1$) has no access to differences and is pinned at
$A_\mathrm{loc}$.

### Label

$y = \mathbf{1}[a > 0]$ — sign of the chirp rate. **Binary**, n_classes = 2.

### Theoretical ceilings

$$
A_\mathrm{loc}^\star = \tfrac12, \qquad A_\mathrm{oracle} = 1.
$$

$A_\mathrm{loc} = 0.5$ because a single phase $\phi(t)$ is equally likely for
$a > 0$ and $a < 0$ (both draw $v$ uniformly, making the marginal phase
distribution independent of $\mathrm{sign}(a)$). $A_\mathrm{oracle} = 1$
because the second finite difference $\phi(t{+}2) - 2\phi(t{+}1) + \phi(t)
= a$ is exact at $\sigma = 0$.

### Why aggregators fail, temporal archs win

The chirp bench is **AC-only** in the stronger sense than the linear AC
bench: the label is in the *second* temporal frequency derivative. An
aggregator that smooths over positions computes a DC estimate of the
one-token emission distribution; the marginal over-$v$ phase distribution
is the same for $a > 0$ and $a < 0$ (it is uniform on $\mathbb{Z}_M$ up to
starting offset). Per-token and mean-pool aggregators are therefore pinned at
$A_\mathrm{loc} = 0.5$.

A sliding window arch with $T \ge 3$ that encodes the cross-position
pre-activation $\sum_\tau W_\text{enc}^{(\tau)} \mathbf{x}_{t+\tau}$ can,
in principle, represent the finite difference that distinguishes $a > 0$
from $a < 0$. The joint $T = W$ variant faces the same pooling bottleneck
as in the linear AC bench (theory doc §3).

**Reverse-control prediction:** A model that learns $\mathrm{sign}(a)$ from
the forward sequence will output the wrong sign on the reversed sequence
(because $\Delta^2$ on the reversal equals $a$ but the finite-differences
used are in the opposite order — more precisely, the reverse of a chirp-up
sequence looks like a chirp-down to the same learned filter). Predict
$A_\mathrm{reverse} < 0.5$ for the sliding temporal archs.

### Pre-registered NTPS predictions (W=16, k_pos=1, d_sae=1024)

| arch | predicted NTPS | rationale |
|---|---|---|
| `topk_sae` (per-token) | ≈ 0 | no temporal mixing; pinned at A_loc |
| `txcdr_t5` (sliding T=5) | ≥ 0.4 | T≥3 ✓; less margin than linear AC (chirp harder) |
| `txc_base_TW` (joint T=W) | 0.05–0.15 | position mixing but pooling bottleneck |
| `tfa` | ≥ 0.3 | attention mixes positions; should recover chirp direction |

---

## 2. Multitone bench

### Generative model

The emission at each timestep is a **superposition of $K$ tones**, where one
of the $K$ tones is designated the "target tone" for the sequence:
$$
\phi_k(t) = (\phi_{0,k} + \omega_k \cdot t) \bmod M, \quad k = 1, \ldots, K,
\qquad
\mathbf{x}(t) = \tfrac{1}{\sqrt{K}} \sum_{k=1}^K \mathbf{f}_{\phi_k(t)} + \sigma\xi_t.
$$
The $K$ tones have distinct velocities $\omega_k \in \Omega$ chosen from a
fixed ladder $\Omega = \{1, 2, \ldots, \text{n\_classes}\}$ (one tone per
class, so $K = \text{n\_classes}$ by default; the target tone = label $y$).

**Emission normalization:** $1/\sqrt{K}$ keeps the per-token SNR independent
of $K$ (otherwise wider superpositions are quieter).

**Ring guard:** as in the AC bench, choose $M$ such that $\max_k \omega_k
\cdot (W{-}1) \le M$ (the fastest tone makes at most one full revolution in
the window). With $W{=}16$, n_classes=8, $M{=}64$: fastest velocity
$\omega_8 = 8$, max phase advance $= 8 \cdot 15 = 120 \le 64 \bmod M$ wraps
(this is fine — phase wraps are OK; the superposition is still distinguishable
by the per-tone phase structure).

### Label

$y \in \{0, \ldots, \text{n\_classes}-1\}$ — which tone is the target.
n_classes = 8.

### Theoretical ceilings

$$
A_\mathrm{loc}^\star = \frac{1}{\text{n\_classes}}, \qquad A_\mathrm{oracle} = 1.
$$

$A_\mathrm{loc}$ = 1/n_classes because at a single timestep the emission is
a superposition of all tones — the marginal one-step distribution is the same
for every class (each class emits all $K$ tones; only the phase pattern over
time distinguishes them). $A_\mathrm{oracle} = 1$ because the target tone is
recoverable via temporal Fourier analysis of the full $W$-token sequence: the
phase structure of tone $\omega_k$ is periodic with period $M/\omega_k$, and
given sufficient $W$ the tones are frequency-separable.

### Why aggregators fail, temporal archs win

The multitone bench is a **frequency-selectivity** task: to identify the
target tone, the arch must resolve which of the superposed phase-walk
patterns corresponds to the labeled tone. Aggregating over positions
preserves the superposition but loses phase coherence across time —
the averaged activation is an uninformative mixture. A temporal arch
that can track phase coherence over $\ge 2$ timesteps can, in principle,
separate the tones.

This is a finer-grained version of the Mixed bench: instead of a single tone
with multiple velocities, the emission superposes all tones simultaneously.
The frequency-response curves $R_j$ per class give the architecture's
velocity-resolved selectivity.

### Pre-registered NTPS predictions (W=16, k_pos=1, d_sae=1024)

| arch | predicted NTPS | rationale |
|---|---|---|
| `topk_sae` | ≈ 0 | single token sees mixture; no temporal filtering |
| `txcdr_t5` | ≥ 0.3 | phase-coherent over T=5; can partially separate tones |
| `txc_base_TW` | 0.05–0.15 | pooling bottleneck limits frequency resolution |
| `tfa` | ≥ 0.25 | attention can weight phase-coherent positions |

---

## 3. Amplitude-modulation (AM) bench

### Generative model

A carrier tone at fixed velocity $\omega_c$ is **amplitude-modulated** by a
slow envelope:
$$
a(t) = 1 + \tfrac{1}{2}\cos\!\left(\frac{2\pi f_m t}{W}\right),
\quad
\phi(t) = (\phi_0 + \omega_c \cdot t) \bmod M,
\quad
\mathbf{x}(t) = a(t)\,\mathbf{f}_{\phi(t)} + \sigma\xi_t.
$$
The modulation frequency $f_m \in \{f_1, \ldots, f_\text{n\_classes}\}$ is
drawn per sequence and is the label. The carrier $\omega_c$ is fixed (same
for all sequences; nuisance); $\phi_0$ is random.

**Why this is interesting.** The carrier contributes a phase walk signal in
the emission. The label is encoded in the *amplitude variation* at the
modulation frequency — a slow envelope overlaid on a fast carrier. A temporal
arch that only tracks phase transitions will be confused by this; it needs to
also track the amplitude envelope.

**Modulation ladder:** n_classes = 6, $f_m \in \{1, 2, 3, 4, 5, 6\}$
(cycles per window). The envelope amplitude is 0.5 of the carrier (depth = 1/2).

**Ring guard:** carrier $\omega_c = 4$, $M = 64$ — max phase advance
$= 4 \cdot 15 = 60 \le M$. No alias issues.

### Label

$y \in \{0, \ldots, \text{n\_classes}-1\}$ — modulation-rate class.
n_classes = 6.

### Theoretical ceilings

$$
A_\mathrm{loc}^\star = \frac{1}{\text{n\_classes}}, \qquad A_\mathrm{oracle} = 1.
$$

$A_\mathrm{loc} = 1/\text{n\_classes}$: at a single timestep $t$, the
amplitude $a(t)$ is $f_m$-dependent but also $t$-dependent, so the
one-token marginal over $t \sim \text{Uniform}(0, W{-}1)$ is the same for
every class (each class has the same mean amplitude and the same phase-walk
marginal). Without temporal context, an estimator is at chance. $A_\mathrm{oracle}
= 1$: the Fourier transform of $a(t)$ over $W$ tokens yields a peak at $f_m$.

### Why aggregators fail, temporal archs win

The modulation-rate label lives in the **amplitude envelope frequency**, which
is distinct from the carrier frequency. A per-token SAE encodes the carrier
phase but not the temporal pattern of amplitude variation. An aggregator
(mean-pool) averages out the modulation envelope. A temporal arch that tracks
the amplitude envelope over multiple timesteps can discriminate $f_m$.

The $R_j$ per-class accuracy curve tests whether the arch's temporal
sensitivity is concentrated at the modulation frequencies or spread uniformly.

### Pre-registered NTPS predictions (W=16, k_pos=1, d_sae=1024)

| arch | predicted NTPS | rationale |
|---|---|---|
| `topk_sae` | ≈ 0 | single token encodes carrier phase only |
| `txcdr_t5` | ≥ 0.25 | amplitude envelope trackable over T=5 |
| `txc_base_TW` | 0.05–0.15 | pooling bottleneck |
| `tfa` | ≥ 0.20 | attention can weight amplitude peaks |

---

## 4. Relative-phase bench

### Generative model

Two independent phase walks occupy **orthogonal blocks** of $\mathbf{f}$:
channel A uses directions $\mathbf{f}^A_1, \ldots, \mathbf{f}^A_{M/2}$;
channel B uses $\mathbf{f}^B_1, \ldots, \mathbf{f}^B_{M/2}$ (orthonormal,
block-disjoint). Each channel walks its ring at a **shared absolute velocity**
$\omega$ but with independent starting phases. The label is which channel
**leads** — i.e. the sign of the phase difference $\Delta\phi(0) = \phi^A_0
- \phi^B_0$:
$$
\phi^A(t) = (\phi^A_0 + \omega t) \bmod (M/2), \qquad
\phi^B(t) = (\phi^B_0 + \omega t) \bmod (M/2),
$$
$$
\mathbf{x}(t) = \mathbf{f}^A_{\phi^A(t)} + \mathbf{f}^B_{\phi^B(t)} + \sigma\xi_t,
$$
$$
y = \mathbf{1}[\phi^A_0 > \phi^B_0].
$$

**Ring guard:** $\omega = 1$, $M/2 = 16$ (with overall $M = 32$). Max phase
advance per channel = $W{-}1 = 15$, which wraps the ring. Wrapping is fine:
the relative phase $\Delta\phi(t) = \phi^A(t) - \phi^B(t) = \phi^A_0 -
\phi^B_0$ is constant across $t$ (both channels advance at the same rate).
So the signal is constant in time — but a single token cannot read it
(both channels' absolute phases are independent of the sign of their difference).

**Why this is the cleanest signed-direction task.** The AC bench requires
detecting the *sign of velocity*; the relative-phase bench requires detecting
the *sign of the phase offset between two channels*. Both are binary, both
have $A_\mathrm{loc} = 0.5$. But the relative-phase task has no "direction
reversal" ambiguity from the absolute phase walk: the relative phase is
constant, so there is no chirp or direction-encoding confusion. The only
way to read the label is to **compare the two channels' phases at the same
timestep** — which requires that the representation encodes both channels'
activations simultaneously, something any non-degenerate window arch does.

**Reverse control.** Reversing the sequence reverses neither channel's walk
direction (both channels still advance at $+\omega$; the reversed sequence
has the same relative phase). Hence $A_\mathrm{reverse} \approx A_\mathrm{fwd}$
for a correct representation — the relative-phase signal is time-symmetric.
**However**, if the model confounds the relative-phase task with the AC bench's
direction-encoding (it has learned a representation that internally represents
velocity direction for the relative phase), the reverse probe might still
show a drop. The primary diagnostic is whether the model reaches high NTPS
at all; the order-controls serve as a consistency check.

Wait — corrected: the two channels walk at the **same** velocity, so the
sequence reversed gives $\phi^A(W{-}1{-}t) = \phi^A_0 + \omega(W{-}1{-}t)$.
The relative phase of the reversed sequence at position $t$ is
$\phi^A(W{-}1{-}t) - \phi^B(W{-}1{-}t) = \phi^A_0 - \phi^B_0$, still the
same label. So the reverse control should stay near $A_\mathrm{fwd}$.

Actually reconsidering: if the arch encodes the relative-phase through a
velocity-like internal representation (comparing phases at adjacent positions)
rather than a direct phase-difference readout, it might confuse a forward
pair $(A\text{ leads})$ with a backward pair. Let us not commit to a specific
prediction for $A_\mathrm{reverse}$ and measure it.

### Label

$y = \mathbf{1}[\phi^A_0 > \phi^B_0]$ — which channel leads. **Binary.**

### Theoretical ceilings

$$
A_\mathrm{loc}^\star = \tfrac12, \qquad A_\mathrm{oracle} = 1.
$$

$A_\mathrm{loc} = 0.5$: at any single timestep, knowing $\phi^A(t)$ and
$\phi^B(t)$ individually does not reveal their relative phase unless both
are encoded jointly; a single-channel probe is at chance. $A_\mathrm{oracle} = 1$:
knowing $\phi^A(t)$ and $\phi^B(t)$ at any $t$ gives $\phi^A_0 - \phi^B_0
\pmod{M/2}$, which determines the sign exactly (assuming $M/2$ is large
enough that the probability of a tie is negligible).

A correct oracle also needs to **encode both channels jointly** — a per-token
SAE that encodes each channel independently (with two separate dictionaries)
cannot read the relative phase without a cross-channel probe. The standard
mean-pooled linear probe used by the evaluator is cross-channel (it sees the
full $d_\mathrm{sae}$-dimensional code), so this is fine for the probe;
but an arch whose sparse code is block-diagonal by channel will have the
relative-phase signal only in the cross-channel co-activation pattern.

### Pre-registered NTPS predictions (W=16, k_pos=1, d_sae=1024)

| arch | predicted NTPS | rationale |
|---|---|---|
| `topk_sae` | ≈ 0 | per-token code may not jointly encode both channels |
| `txcdr_t5` | ≥ 0.5 | window-level code sees both channels at each T-window |
| `txc_base_TW` | ≥ 0.3 | joint window code covers both channels |
| `tfa` | ≥ 0.4 | attention can attend to both channels |

**Important note.** In the original task spec, the lead/lag is described as
showing $A_\mathrm{reverse} < \mathrm{chance}$. Based on the analysis above,
the relative phase is time-symmetric, so this is only expected if the arch
confuses direction-encoding with phase-difference encoding. This will be
empirically determined.

---

## 5. Summary of pre-registered predictions

### By bench

| bench | A_loc | A_oracle | primary difficulty | reverse < chance? |
|---|---|---|---|---|
| chirp | 0.5 | 1.0 | quadratic phase (2nd-order) | predicted yes |
| multitone | 1/n_classes | 1.0 | superposed tones; frequency selectivity | not expected |
| am | 1/n_classes | 1.0 | amplitude envelope frequency | not expected |
| relphase | 0.5 | 1.0 | two-channel phase comparison | uncertain |

### By arch (at W=16, k_pos=1, d_sae=1024)

| arch | chirp NTPS | multitone NTPS | am NTPS | relphase NTPS |
|---|---|---|---|---|
| `topk_sae` | ≈ 0 | ≈ 0 | ≈ 0 | ≈ 0 |
| `txcdr_t5` | ≥ 0.4 | ≥ 0.3 | ≥ 0.25 | ≥ 0.5 |
| `txc_base_TW` | 0.05–0.15 | 0.05–0.15 | 0.05–0.15 | ≥ 0.3 |
| `tfa` | ≥ 0.3 | ≥ 0.25 | ≥ 0.20 | ≥ 0.4 |

---

## 6. Datasource registrations (already in configs/data.yaml)

| datasource | bench | params |
|---|---|---|
| `af_chirp_smoke` | chirp | W=4, M=16, d_in=32, n_seqs=256 |
| `af_chirp_W16_s10` | chirp | W=16, M=64, d_in=256, n_seqs=4096 |
| `af_multitone_W16_s10` | multitone | W=16, M=64, n_classes=8, d_in=256, n_seqs=4096 |
| `af_am_W16_s10` | am | W=16, M=64, n_classes=6, d_in=256, n_seqs=4096 |
| `af_relphase_W16_s10` | relphase | W=16, M=32, d_in=256, n_seqs=4096 |

---

## 7. Implementation checklist

- [x] Write this plan (`docs/aniket/altfreq_plan.md`)
- [ ] Implement `src/temp_bench/data/altfreq_data.py`
- [ ] Sanity-check each generator (shape, label balance, oracle at σ=0)
- [ ] Smoke test via `run.py freq_bench --arch txc_base --seed 0 --datasource af_chirp_smoke --smoke`
- [ ] Write `experiments/altfreq/{__init__,sweep,analyze}.py`
- [ ] Run sweep (GPU 0, CUDA_VISIBLE_DEVICES=0)
- [ ] Run analysis, render PNGs to `results/altfreq/`
- [ ] Write `docs/aniket/altfreq_summary.md`

---

## 8. Lost-constants prevention

All constants used in the generators are documented in the generator
docstrings and here. For each bench:

- **chirp:** $W{=}16$, $M{=}64$, $\sigma{=}0.1$, $a \in \{-1,+1\}$ (chirp rate),
  phase formula $\phi(t) = (\phi_0 + vt + a\cdot t(t{-}1)/2) \bmod M$
  (integer second-diff = $a$; $\Delta^2\phi \bmod M \in \{1,63\}$),
  $v \sim \text{Uniform}(\{0,\ldots,M{-}1\})$ (carrier velocity, nuisance),
  $\phi_0 \sim \text{Uniform}(\{0,\ldots,M{-}1\})$.
- **multitone:** $W{=}16$, $M{=}64$, $K{=}\text{n\_classes}{=}8$,
  $\omega_k = k$ for $k \in \{1,\ldots,8\}$, $\sigma{=}0.1$. Normalization
  $1/\sqrt{K}$.
- **am:** $W{=}16$, $M{=}64$, $\omega_c{=}4$ (carrier), $f_m \in
  \{1,\ldots,6\}$, modulation depth $d{=}0.5$, $\sigma{=}0.1$. Amplitude
  $a(t) = 1 + d\cos(2\pi f_m t / W)$.
- **relphase:** $W{=}16$, $M{=}32$ (each channel uses $M/2{=}16$ phases),
  $\omega{=}1$ (shared velocity), $\sigma{=}0.1$. Channel A: directions 0..15,
  channel B: directions 16..31 in orthonormal emission matrix.
