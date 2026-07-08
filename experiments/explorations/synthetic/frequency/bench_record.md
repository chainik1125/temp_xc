# Frequency / cyclic-tone bench — architecture results

**Verdict:** <!-- one-line headline set after the grid --> _(pending grid)_

The periodic / frequency axis — the one dynamics class the suite did not cover.
A **synthetic-first architectural discriminator** (like signed_motion; not a
measure→mirror bench — no measured real-LM periodicity anchors it). Port of
Dmitry's FrequencyBench (`origin/dmitry-spectral-sprint2`) onto our BatchTopK
fair backbone + memorization-free per-tile probe + multi-seed grid.

Frozen spec: [`bench_spec.md`](bench_spec.md) (amendments A1–A5). § 8 gating:
[`results/frequency_gating_stats.json`](results/frequency_gating_stats.json)
(PASS — figure below). Every number here is regenerated from the canonical
leaderboard by
[`render_figs.py`](render_figs.py) (`-m
experiments.explorations.synthetic.frequency.render_figs`); nothing is
hand-typed.

---

## 1. What it tests (one paragraph)

A symbol walks a cyclic alphabet `Z_M` (`M=101` prime) at a **hidden velocity
`Y ~ Unif(Ω)`**, `Q_t = (B + Y·t) mod M`. Under a **circle embedding**
(`u_a = R·[cos 2πa/M, sin 2πa/M]`, `R` a random `d_in×2` isometry) each velocity
becomes a temporal **tone** at `f = Y/M` cycles/token; recovering `Y` is
single-tone spectral estimation whose ML decoder is the **periodogram peak-pick**.
The bench measures the **frequency response `S(f)`** — at which temporal
frequencies each architecture's code makes the velocity linearly decodable — and
whether a **DCT-band spectral crosscoder** decomposes its dictionary by band. A
**random-embedding null** (orthonormal frame) has, by the ratio-invariance
theorem for prime `M`, a **flat** response — the built-in negative control.

The velocity is invisible per-token (`Q_t | Y` uniform ⇒ `I(Y; x_t) = 0`) **and**
to a raw-linear window reader (`E[x_t|Y] ≈ 0` ⇒ velocity is 2nd-moment): only a
window code that mixes positions nonlinearly can expose it.

## 2. Gating (§ 8) — ceilings separated, design settled

![gating](figs/frequency_gating.png)

Per-token velocity = chance (0.10, provable DPI + empirical); raw-linear window
= chance (circle 0.11); circle periodogram oracle is high-pass/Rayleigh (T=2
resolves only high-Y, all-1.00 at T=16); random null flat (per-Ω-class range
0.000). Settled: `M=101`, `d_in=128`, `Ω={0,1,2,4,8,16,24,32,40,50}`, `σ=0.10`,
`seq_len=64`, `L=32`, `T∈{2,4,8,16}`; memorization threshold `|Ω|·M=1010`.

## 3. Headline

<!-- BEGIN AUTO:headline -->
_(populated by render_figs.py)_
<!-- END AUTO:headline -->

## 4. Velocity-recovery frontier (circle)

Recovery normalized to [chance = 1/|Ω| = 0.1, periodogram oracle]. `F`-anchor
`M=101` and the memorization threshold `|Ω|·M=1010` marked; all main cells stay
below 1010 (memorization-free).

![main](figs/frequency_main.png)

<!-- BEGIN AUTO:circle_frontier -->
_(populated by render_figs.py)_
<!-- END AUTO:circle_frontier -->

## 5. The deliverable: the frequency response `S(f)`

Per-Ω-class velocity recall vs `f = Y/M`, one curve per window `T`, Rayleigh
cutoff `≈ 1/T`. High-pass: shallow windows resolve only high `f`.

![Sf](figs/frequency_Sf.png)

Raw per-Ω-class recall (probe) vs the oracle (`d_sae=M`):

<!-- BEGIN AUTO:sf_table -->
_(populated by render_figs.py)_
<!-- END AUTO:sf_table -->

## 6. Spectral vs monolithic + band decomposition

Spectral-TXC vs TXC-pre/post at matched budget, and the per-DCT-band probe
(each band should decode the tones in its frequency range).

![spectral](figs/frequency_spectral.png)

<!-- BEGIN AUTO:band_table -->
_(populated by render_figs.py)_
<!-- END AUTO:band_table -->

## 7. Symmetry null (circle vs random)

![null](figs/frequency_null.png)

The circle response tracks `|Δf|` (Rayleigh); the random response has no
frequency axis (ratio-invariance). Above `|Ω|·M` the null recovery jumps by
template memorization — the control that flags the memorization regime.

<!-- BEGIN AUTO:memo -->
_(populated by render_figs.py)_
<!-- END AUTO:memo -->

![memorization](figs/frequency_memorization.png)

## 8. Access vs learning (untrained control)

Trained recovery minus the untrained-encoder residual isolates learning from
nonlinear architectural access.

![untrained](figs/frequency_untrained.png)

<!-- BEGIN AUTO:untrained -->
_(populated by render_figs.py)_
<!-- END AUTO:untrained -->

## 9. Reconstruction (capability-vs-artifact)

NMSE frontier (the spectral/window winner must also reconstruct the 2-D circle,
not just recover the latent). `eauc` is ill-defined for the circle (densely
packed atoms) — NMSE is the capability metric here.

<!-- BEGIN AUTO:nmse_table -->
_(populated by render_figs.py)_
<!-- END AUTO:nmse_table -->

## 10. Controls (which passed)

- **Per-token provable floor** — `I(Y;x_t)=0` (DPI); empirical per-token ≈ chance.
- **Raw-linear window at chance** — velocity is 2nd-moment (amendment A4).
- **Memorization-free per-tile probe** — shared-code tile = `d_sae < |Ω|·M`; the
  `d_sae=2048 > 1010` demo shows the inflation the probe would otherwise hide.
  (Stacked dropped — its `T·d_sae` concatenated code memorizes; amendment A5.)
- **Untrained-encoder control** — a claimed win must exceed the random-init
  nonlinear-access residual.
- **Symmetry null** — the random-embedding response is flat (theorem verified).
- **Capability-vs-artifact** — NMSE reconstruction reported alongside recovery.
