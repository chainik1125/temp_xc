---
author: Aniket
date: 2026-06-01
tags:
  - plan
  - freqbench
  - reed-solomon
  - txc
status: plan
---

## Reed–Solomon temporal encoding — plan

**One-line:** generalise the FreqBench **AC bench** from a *degree-1* phase
trajectory to a *degree-D* polynomial phase trajectory. The AC walk is the
$D{=}1$ special case; sweeping $D \in \{1, 2, 3\}$ is the Reed–Solomon
ladder. Show TXC-base recovers the message at every $D$ while per-token /
single-position SAEs degrade with $D$, and tie the result to the
frequency-decomposition / "TXC as a special case" framing Dmitry wants.

### Provenance — why this construction

There is **no prior RS code or math anywhere in the repo** (verified across
full git history). This plan is grounded in Dmitry's Slack thread (2026, on
the polynomial perspective), reproduced here so the design is auditable:

> **Dmitry:** I think this might be equiv to Reed-Solomons at polynomial
> length 1. […] TXC outperforms, but this is kinda by construction on the
> problem.
> **Aniket:** so if each AC sequence is a RS code, higher-order polynomials
> might resolve the by-construction issue.
> **Dmitry:** No — higher-order polynomial makes it *worse* for
> single-token-position SAEs (everything but TFA). To be sure, this is a
> good feature of the TXC and it should be one eval for it, but we just
> have to note that this is true by construction. So my take is we can have
> poly length 1, 2, 3 and evaluate our TXC-base on it, show it wins and
> just note that this highlights a fundamental limitation of the other
> architectures. What I would really love is to understand the frequency
> decomposition issue, and if we can get hyperparams for which our current
> TXC is a special case.

So the deliverable is **not** a finite-field GF(q) codec. It is the
polynomial-phase generalisation of AC, evaluated as a clean
TXC-wins-by-construction result, framed through the spectral $(T, S, B)$
family (`freq-bench/spectral-family.md`).

### 1. The connection: AC bench = Reed–Solomon at polynomial length 1

The AC generator (`freq_bench_ac`) walks a phase on the ring $\mathbb{Z}_M$
at fixed signed velocity:
$$
\phi(t) = (\phi_0 + v\,t) \bmod M, \qquad t = 0, \ldots, W-1.
$$
This is a **degree-1 polynomial in $t$** evaluated at $W$ integer points —
exactly a Reed–Solomon codeword of a length-2 message $(\phi_0, v)$
evaluated at the points $t = 0, \ldots, W{-}1$ over $\mathbb{Z}_M$. The
emission is one-hot$(\phi(t)) + \sigma\xi_t$, i.e. the codeword symbol at
each timestep, corrupted by Gaussian noise. The label $y = \mathrm{sign}(v)$
reads the leading coefficient.

Reed–Solomon's defining property is that the message is recoverable from a
*sufficient subset* of evaluation points, not from any single one: a
degree-$D$ polynomial needs $D{+}1$ points to determine. That is precisely
the "information distributed across time, recoverable only by integrating
≥ k timesteps" property we want to stress-test.

### 2. The generalisation: degree-D polynomial phase

$$
\boxed{\;\phi(t) = \Big(\sum_{i=0}^{D} c_i\, t^i\Big) \bmod M,
        \qquad t = 0, \ldots, W-1.\;}
$$

- $D = 1$: $\phi(t) = c_0 + c_1 t$ — **reproduces the AC bench** ($c_0
  = \phi_0$, $c_1 = v$). Sanity anchor: must match `freq_bench_ac`.
- $D = 2$: $\phi(t) = c_0 + c_1 t + c_2 t^2$ — quadratic phase = **linear
  chirp** (instantaneous frequency $\phi'(t)$ grows linearly). Determining
  the message now requires ≥ 3 timesteps; a single adjacent-pair comparison
  is no longer sufficient.
- $D = 3$: cubic phase — ≥ 4 timesteps.

**Message / label.** The "message" is the coefficient vector
$(c_0, \ldots, c_D)$. We score a **label derived from the leading
behaviour** so the metric stays comparable across $D$ and matches the AC
sign task. Two label variants to pin down in the sweep:
- `sign`: $y = \mathbf{1}[c_D > 0]$ — sign of the top coefficient
  (direction analog; binary, chance $=\tfrac12$). Cleanest cross-$D$
  comparison and keeps the reverse-control semantics.
- `class`: $y =$ class index of $c_D$ over a small symmetric ladder
  $\Omega$ (multiclass; reuses the Mixed-bench machinery + per-class
  $R_j$). Optional secondary.

**Oracle / ceiling.**
- $A_\mathrm{loc}^\star = \tfrac12$ (`sign`) — a single phase reveals
  nothing about the leading coefficient's sign. (For `class`, $1/|\Omega|$.)
- $A_\mathrm{oracle} = 1$ — given $D{+}1$ clean phases the polynomial (hence
  $c_D$) is determined exactly by finite differences / Lagrange
  interpolation over $\mathbb{Z}_M$. Implement the symbolic oracle so the
  noiseless ($\sigma{=}0$) sanity row hits 1.0.

**Ring guard.** As in `freq_bench_ac`, keep the trajectory from wrapping the
ring within the window so the leading term stays decodable: pick $M$ large
enough that $\max_t |\sum_i c_i t^i|$ stays bounded relative to $M$, or
draw coefficients from a magnitude schedule $|c_i| \lesssim M / (D{+}1)
W^i$. Document the exact schedule in the generator docstring (the lost
constants problem from Dmitry's original run — don't repeat it).

### 3. Why TXC wins, and why that is "by construction"

A degree-$D$ polynomial phase is concentrated in temporal-frequency bands
up to order $\sim D$ (the spectral picture from `freq-bench/spectral-family.md` §1):
the rfft of $\phi(t)$ along $\tau$ spreads energy into higher AC bands as
$D$ grows. Reading the message therefore requires an encoder whose
receptive field mixes $\ge D{+}1$ positions **before** the sparsity
bottleneck — the sum-before-TopK channel that only the window encoders
(TXC sliding-$T$, TFA attention) have. Per-token / single-position SAEs
have $T{=}1$: no temporal mixing, so they are pinned at $A_\mathrm{loc}^\star$
for **every** $D$, and the gap to TXC *widens* with $D$ (Dmitry's point —
higher order is strictly worse for everything but the window archs).

This is "true by construction": the task is *defined* to need multi-position
temporal integration, which is exactly the inductive bias TXC has and the
baselines lack. The paperframing is therefore **honest** — present it as
*"a controlled demonstration of the capability gap, true by construction"*,
not as a surprising emergent win. The scientific payload is the **scaling
with $D$** (a clean monotone curve) plus the spectral interpretation.

### 4. The frequency-decomposition payload (what Dmitry "would really love")

Tie the degree ladder to the spectral $(T, S, B)$ family:

1. **FreqFrac vs $D$.** The trained encoder's FreqFrac (already computed by
   the evaluator) should climb with $D$ — higher-degree phase forces the
   encoder to put energy in higher AC bands. Predicts a monotone
   FreqFrac($D$) curve.
2. **Band-subset $B$ ablation per degree.** Run the band-restricted encoder
   `txc_band` (the $(T,S,B)$ arch) at each $D$: a degree-$D$ task should be
   solvable iff $B$ contains bands up to order $\sim D$. This yields the
   pre-registerable prediction *"the $(T, S, B{=}\{0,1\})$ cell fails at
   $D{=}2$ but the $(T, S, B{=}\{0,1,2\})$ cell succeeds"* — i.e. **the
   hyperparameters $(T, B)$ for which TXC is the minimal special case that
   solves degree $D$.** This is the "TXC as a special case" deliverable.
3. **$T$ sufficiency.** Predict the minimal window $T \ge D{+}1$ needed; map
   the NTPS($T$, $D$) surface.

### 5. Experiment grid (2× A40, synthetic — compute is not the constraint)

Reuse the `freq_bench` experiment + (now generic) evaluator. New generator
`temp_bench.data.reed_solomon_data:rs_poly_phase` returns a `FreqBenchData`.

- **Archs:** `topk_sae` (per-token baseline), `txc_base` slid at $T=5$
  (`txcdr_t5`), `txc_base` joint $T{=}W$, `tfa` (attention baseline),
  optionally `tsae`. Plus `txc_band` for the $B$-ablation payload.
- **Degree:** $D \in \{1, 2, 3\}$ (Dmitry's "poly length 1, 2, 3").
- **Window:** $W = 16$ (and a small $W \in \{8, 16\}$ slice for $T$
  sufficiency).
- **Sparsity / capacity:** $k_\mathrm{pos} = 1$, $d_\mathrm{sae} = 1024$
  (the readout-optimal slice established in `freq-bench/theory.md` §2).
- **Noise:** $\sigma = 0.1$, plus a $\sigma = 0$ sanity row (oracle must hit
  NTPS = 1 for $T \ge D{+}1$ window archs).
- **Controls:** the standard shuffle/reverse + MLP + FreqFrac the evaluator
  already emits.

Sharded across the 2 A40s via `experiments/reed_solomon/sweep.py` (clone of
`freq_bench/sweep.py`). Expected wall time ≪ 1 h.

### 6. Pre-registered predictions (fill in before launch)

| arch | $D{=}1$ | $D{=}2$ | $D{=}3$ |
|---|---|---|---|
| per-token `topk_sae` | ≈ 0 | ≈ 0 | ≈ 0 |
| `txcdr_t5` (sliding $T{=}5$) | ≈ 0.72 (matches AC) | ? (predict ≥ 0.5, $T{=}5 \ge D{+}1$) | ? |
| `txc_base` joint $T{=}W$ | ≈ 0.17 | ? | ? |
| `tfa` | ? | ? | ? |

Commit the numeric predictions to
`results/reed_solomon/predictions.json` at launch (pre-registration
discipline, as in `freq-bench/spectral-family.md` §4).

### 7. Deliverables

- `src/temp_bench/data/reed_solomon_data.py` — `rs_poly_phase` generator
  (+ symbolic finite-difference oracle), header-documented constants.
- `configs/data.yaml` — `rs_D{1,2,3}_W16_s10` datasources.
- `experiments/reed_solomon/{sweep,analyze}.py` — GPU-sharded sweep + plot
  script (NTPS-vs-$D$ curve, FreqFrac-vs-$D$, $B$-ablation grid).
- `docs/aniket/reed-solomon/summary.md` — results with **inline rendered
  plots** (per the experiment-doc convention).
- 1–2 sentence §4 paragraph + figure for the paper: the degree ladder as a
  by-construction capability gap, with the $(T, B)$-minimal-special-case
  reading.

### 8. Questions for Dmitry — RESOLVED (2026-06-01)

**Resolutions (relayed via Aniket):**
- **Q1 → YES.** Proceed with the degree-$D$ polynomial-phase construction,
  $D \in \{1, 2, 3\}$. $D{=}1$ reproduces AC.
- **Q2 → ALL THREE.** Report all three recovery targets: (a) sign of the
  leading coefficient (binary), (b) leading-coefficient class over a
  symmetric ladder (multiclass), (c) full-message regression (NMSE over the
  $D{+}1$ finite-difference initial conditions). All three are *readouts on
  the same trained SAE code* — (a)/(b) are two classification label-modes,
  and (c) is a regression probe that rides along in the same eval call, so
  there is no extra training cost.
- **Q3 → UNBIASED $(T,S,B)$ SWEEP.** Run the full $(T,S,B)$ band/stride
  sweep on the RS benches and *report where the standard TXC config lands* —
  no assumption that TXC is optimal. Let the data place it.
- **Q4 → DEFERRED.** Revisit the by-construction framing after results from
  all three assigned tasks (RS + alt-frequency + direct-sum) are in.

The original questions, for the record:

**Q1 — Confirm the construction (load-bearing).** There is no RS spec in the
repo, so everything downstream rests on this.
> "For Reed–Solomon: I'm reading the AC bench as RS at polynomial length 1
> (phase $\phi(t)=\phi_0+v t \bmod M$), and generalising to degree-$D$
> polynomial phase $\phi(t)=\sum_i c_i t^i \bmod M$, sweeping
> $D\in\{1,2,3\}$. Is that the construction you had in mind?"

**Q2 — Recovery target / label (changes the evaluator).**
> "Should the probe recover the *sign of the leading coefficient* (binary,
> keeps the reverse-control direction story, comparable across $D$), the
> leading-coefficient *class* over a velocity ladder (multiclass, reuses the
> Mixed-bench machinery), or the *full message* — all coefficients
> (NMSE-style regression)?"

**Q3 — How hard to lean on "TXC as a special case" (the thing he said he'd
*really love*).**
> "For the frequency-decomposition payload: do you want the full $(T,S,B)$
> band-ablation — showing degree-$D$ is solvable iff $B$ contains bands up
> to order $\sim D$, i.e. the exact $(T,B)$ for which TXC is the *minimal*
> special case that solves degree $D$? Or is the simpler 'TXC wins on the
> degree ladder, note the by-construction limitation of the others' enough
> for v1?"

**Q4 — The by-construction worry; what bar clears arXiv (he raised this).**
> "You flagged the degree-ladder win is 'true by construction.' Are you fine
> presenting it explicitly as a controlled capability-gap demonstration, or
> do you want at least one bench where the temporal win is *not* by
> construction (e.g. the direct-sum / alternate-frequency benches, where the
> processes share the emission marginal)?"

Context to offer alongside the questions: the two **non-by-construction**
benches built in parallel with this plan — alternate-frequency (shared-
structure frequency tasks) and direct-sum which-process (processes with
*identical one-token marginals*, differing only in dynamics) — are the
intended complement to Q4. Early numbers from those should be available to
show him.
