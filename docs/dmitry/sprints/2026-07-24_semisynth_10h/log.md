---
author: Claude (Fable 5), for Dmitry Manning-Coe
date: 2026-07-24
tags:
  - results
  - in-progress
---

## Sprint log — semisynth 10h

### 21:58 — kickoff

Branch `dmitry-semisynth-10h` created off `dmitry-spectral-sprint2`. Sprint dir +
`start.md` written. Plan: W-sweep centerpiece first (it is the user's named target —
"performance improves with window size"), agents in parallel.

Design decision for the W-sweep: operationalize "window size" as the **span of one
steering knob**. A window-W handle covers contiguous blocks of W segments; one knob =
one block written with the *correct per-segment schedule inside its span* and one scalar.
Fixed budget of m knobs ⇒ coverage m·W segments. W=1 is exactly a per-token (SAE-like)
latent; W=k is one latent writing the whole trajectory. Prediction (theory agent to
formalize): Δmargin(W; m) ≈ Δ_full · min(mW, k)/k — linear growth in W, saturation at
W = k/m. Block placement rotated per eval pair to wash out position bias.

k=12 (divisors 1,2,3,4,6,12), tasks lang_profile + alt_phase, fracs {0.2, 0.35, 0.5}
(peak region from today's full run), n_train=40, n_eval=32.

### 22:20 — W-sweep landed (task #1 done). Centerpiece confirmed.

`results/temporal_screen/wsweep.json`. Peak Δmargin vs W at fixed knob budget:

- **Monotone growth in W, both tasks, both budgets.** lang m=1: 15→46→71→75→140→266
  (W=1→12); alt m=1: 5.7→16.6→21.9→33.7→46.6→93.4. m=2 ≈ 2× m=1 throughout.
- **Additive prediction Δ_full·min(mW,k)/k fits.** alt W6_m1: predicted +46.7,
  observed +46.6±2.3. lang within ~1 SEM at most points.
- **Broadcast negative at k=12** (lang −8.1, alt −4.4): the DC floor.
- **Boundary penalty at W=1**: per-covered-slot efficiency ~30% below the W≥2
  plateau in both tasks (lang 15.5 vs ~23; alt 5.7 vs ~8). Isolated single-segment
  writes are less efficient than the same slots inside a window — per-slot
  efficiency itself improves with window size. Worth a dedicated check later
  (boundary count is 2 per block regardless of W ⇒ penalty ∝ 1/W).
- Deviations to note honestly: lang W1_m1 and W4_m1 undershoot prediction (~0.7,
  ~0.84×); alt is near-perfectly linear.

Next: entrainment generation W-sweep (task #2) — switching to language-alternation
vs random-language profiles so the objective langid judge covers both the
predictable and unpredictable case with the same direction u.
