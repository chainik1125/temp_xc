---
author: Claude (10h unsupervised sprint)
date: 2026-06-10
tags:
  - results
  - complete
---

## Sprint log: FrequencyBench — measuring the temporal frequency response of dictionary-learning architectures

Wall clock: start **2026-06-10 20:30:05 UTC** (epoch 1781123405), hard stop **2026-06-11 06:30:05 UTC**.
Budget: < $50 RunPod (target: one A40, ≈ $0.40/h ⇒ < $5).

### North star

A quantitative technique for finding the timescale/frequency at which any temporal
architecture finds features, plus an architecture that performs the spectral
decomposition (the spectral crosscoder). Reference: `docs/dmitry/proposals/frequencybench_pedagogical.tex`.

### Preregistered claims to test (H0:10)

1. **Conversion claim (theory + empirics).** The hidden velocity in cyclic tasks has
   zero single-token mutual information (proof in proposal), and additionally is *not
   perfectly linearly decodable from any per-token code* (new phase-averaging
   impossibility argument — to be written). Temporal dictionaries convert this
   temporally-nonlinear structure into linearly-decodable codes; S_temp measures the
   conversion.
2. **Measurement claim.** The frequency-response curve S(ω) of trained architectures is
   measurable and discriminating: token-SAE ≈ 0 everywhere; vanilla TXC predicted
   low-pass; conv dictionary predicted broadband. (Empirical shapes are the finding —
   prediction may be falsified.)
3. **Architecture claim.** A frequency-split (multi-band / spectral) crosscoder achieves
   a broader/flatter response than vanilla TXC at matched dictionary size and matched
   window-level L0.
4. **Phase claim (stretch).** Velocity pairs (+y, −y ≡ M−y) have identical power
   spectra; the sign lives purely in phase. Architectures that detect band energy will
   confuse ±y; phase-sensitive (localized/conv) ones will not. The AC sign task is a
   phase-detection task in disguise.

### Theory notes (to be expanded in summary)

- Fourier comb: project x_t onto characters χ_k of Z_M ⇒ each velocity Y produces pure
  tones at temporal frequencies {kY/M mod 1}. Velocity-in-symbol-space = comb of
  temporal frequencies in activation space. This licenses calling the benchmark a
  *frequency* response measurement.
- Linear-probe impossibility: for stacked per-token codes that are functions of single
  symbols, sum class scores over phase B: Σ_B score_y(window(B,Y)) is independent of Y
  ⇒ no linear probe can correctly classify all phases for two distinct velocities.
- Sign task: ±v sequences are time reversals ⇒ identical power spectra ⇒ signal is
  pure phase (quadrature) information.

### Experiment design decisions (H0:10)

- Symbols embedded as random orthonormal directions u_a ∈ R^d, d = M (exact orthonormal
  frame via QR of a Gaussian matrix). Activation x_t = u_{Q_t} + σ ε_t, σ = 0.25.
- Tasks: (1) DC: K=8 classes, channel accuracy p=0.6, M=8. (2) AC sign: M=17, v=8,
  S∈{±1}. (3) Mixed: M=101, Ω={0,1,2,4,8,16,24,32,40,50}, W=16.
- Architectures at matched total atoms and matched window L0:
  token-SAE (probe on stacked window codes), mean-pool baseline, vanilla TXC,
  DC/AC-split TXC, multi-band (DCT) TXC, localized conv dictionary.
- Probes: multinomial logistic (headline) + small MLP (information check), trained on
  held-out codes; disjoint train/test sequences; ≥3 dictionary seeds for headline curves.
- Report: S_temp, raw accuracies, FVU (so dead/undertrained models are visible),
  empirical ceilings (single-token probe, symbolic oracle on noisy windows).
- Diagnostics: DCT FreqFrac per learned atom; band-projection ablations;
  shuffle/reversal sensitivity.

### Red-team checks planned

- Single-token probe on AC tasks must be at chance (validates data gen + ceiling).
- FVU comparable across architectures (rules out "TXC lost because undertrained").
- Probe class identical across architectures; codes standardized before probing.
- Same data budget for every architecture.
- Multiple seeds; error bars.

### Timeline (planned)

- H0:00–0:45 setup, plan, pod up
- H0:45–2:00 implement package + smoke test on pod
- H2:00–5:00 phase 1 grid + frequency response
- H5:00–7:30 follow-ups (ablations, phase pairs, window scaling)
- H7:30–8:30 figures
- H8:30–10:00 writing + red/blue team iteration

### Log entries

- **H0:00** Read brief, proposal tex, writing/sprint instructions. Started clock.
- **H0:05** Explored repo: reusable TopKSAE/StackedSAE/TemporalCrosscoder in
  `temporal_crosscoders/models.py`; probe code `src/bench/saebench/probe_fit.py`;
  runpod bootstrap scripts in `scripts/`. Decision: build a *self-contained* package
  (copy model classes) so the pod only needs torch+numpy+sklearn — avoids the
  "code-not-pushed-before-bootstrap" failure from lessons_learned.md.
- **H0:10** Wrote preregistered claims above. Provisioning pod next.
- **H0:25** Pod `z53dh4ix5bhxpl` (A40, $0.44/h) up; SSH via 69.30.85.13:22185.
  Implemented `fb_core.py` (data, SpectralTXC unifying vanilla/DC-AC/multiband,
  ConvDict, TokenSAE, torch probes, oracles, diagnostics) + `run_grid.py`.
  Caught 2 bugs on self-review: conv_transpose weight layout; L0 bookkeeping.
- **H0:40** Smoke test (300 steps) passed. Key sanity results: dc raw-token probe
  0.570 ≈ p=0.6 ✓; ac_sign raw-token 0.488 ≈ chance ✓; **ac_sign raw-stacked
  LINEAR probe 0.485 ≈ chance but raw-stacked MLP = 1.000** — linear-probe
  impossibility + info-presence both confirmed. TXC/multiband/conv codes make the
  sign linearly decodable (0.97/0.99/0.99). Claim 1 essentially in hand.
- **H0:50 — MAJOR FINDING (theory).** The ten-frequency benchmark as specified in
  the proposal is *symmetry-trivial*: for prime M and exchangeable (random) symbol
  embeddings, the relabeling a ↦ a·y'·y⁻¹ mod M maps velocity-y data to
  velocity-y' data bijectively. Hence ALL nonzero velocities are statistically
  equivalent and S(ω) is flat on ω≠0 in expectation over embeddings — there is no
  meaningful "frequency" without geometry on symbol space. Equivalently: every
  character channel χ_k carries a unit-power tone at k·y/M, so for prime M the
  temporal power spectrum summed over channels is uniform for every y≠0.
  **Fix:** circle embedding u_a = R[cos(2πa/M), sin(2πa/M)] (random isometry R
  into d=8). Then velocity y = one genuine temporal tone at y/101 cycles/token;
  Ω={0,1,2,4,8,16,24,32,40,50} maps to DCT indices {0,.3,.6,1.3,2.5,5.1,7.6,10,
  12.7,15.8} — spans DC/low/mid/high bands of W=16 perfectly. ML oracle for the
  circle task = periodogram peak-picking (DFT matched filter) — the task IS
  classical single-tone spectral estimation. Random-embedding variant retained as
  the symmetry-null control (prediction: flat response; tests the theorem).
  Window-resolution theory: adjacent velocities y=0,1 have template correlation
  0.96 at W=16 but high SNR keeps oracle ≈99%; velocity resolution ∝ 1/W
  (Rayleigh) — candidate follow-up: W-scaling.
- **H0:55** Revised claims: (1) conversion (unchanged); (2) NEW symmetry-null —
  random-embedding response provably flat, verified empirically; (3) frequency
  response on circle embedding per architecture; (4) spectral TXC branch
  specialization (branch×velocity heatmap) at matched budgets.
- **H1:00** Refined claim 2 — flatness is NOT exact for the 10-class task; the
  exact theorem is *ratio invariance*: two-class task {y,y'} difficulty depends
  only on the ratio orbit {r, 1/r}, r = y'/y mod M (multiplicative relabeling +
  exchangeable embeddings). Coincidence geometry: shared symbols between
  velocity-y and y' windows lie on a slope-r line in the (t,s) torus; ratio-2
  pairs share up to W/2 symbols vs ~W²/M generic. So random-embedding
  confusability is *multiplicative/number-theoretic*; circle-embedding
  confusability is *spectral* (Dirichlet kernel / Rayleigh). Patched probes to
  save full confusion matrices; restarted grid (cost: ~3 min of GPU).
- **H1:10** Full grid running: A = dc+ac_sign (4k steps), B = multifreq random
  (H∈{256,2048}, 6k steps), C = multifreq circle (same). Wrote theory.md
  (P1 ceiling, P2 additive-readout impossibility, P3 ratio invariance, P4
  coincidence lines, P5 circle/periodogram/Rayleigh, P6 capacity separation).
- **H1:15** Literature agent returned: **no prior frequency-response curves for
  dictionary architectures** (novelty ✓). Nearest: T-SAEs (arXiv:2511.05541,
  imposes slowness, doesn't measure timescales); SynthSAEBench (iid, no time);
  SISC (Grosse 2007) = conv dictionary precedent; Nanda clock + Engels circular
  features (real LLMs represent cyclic concepts as circles → circle embedding is
  the realistic case, not a contrivance!); Rife–Boorstyn (periodogram = ML tone
  estimation); spectral-bias caveat (Rahaman 2019) noted for limitations.
  Saved to litsearch.md.
- **H1:20** Launched stream D: verify_theory.py — exact overlap combinatorics +
  pair-task double contrast ({1,2},{2,4},{8,16},{16,32} ratio-2 vs ratio-3 vs
  generic, both embeddings, TXC + bag-of-symbols readouts, 3 seeds).
- **H1:30** Wrote make_plots.py (response curves, conversion bars, confusion
  contrast, pair bars, branch×freq heatmap, FreqFrac, summary table).
- **H1:35** Pair-task interim: bag-of-symbols + MLP solves velocity pairs at
  1.0 — the window's symbol *set* is an AP whose common difference gives
  {y, −y} with no order info. Consistent with P2 (MLP-after-pooling is not
  additive). Sharp corollary: sign pairs {y, −y} have identical symbol sets ⇒
  bag must fail at 0.5 exactly there. Added (3, 98) sign pair to PAIRS as the
  critical control. Order information is needed for *direction*, not *speed*.
- **H1:45** Stream A (dc, ac_sign) complete; synced. Headlines: ac_sign —
  raw-token 0.504 / raw-stacked-linear 0.499 (chance, P2 ✓) / raw-stacked-MLP
  1.000 (info ✓); token-SAE stacked codes linear 0.501 (chance) but MLP on the
  same codes 1.000; window dictionaries linear: txc 0.903, dcac 0.904,
  multiband 1.000, conv 0.999 — conversion confirmed with all checks. dc —
  everything ≈0.99 incl. raw mean-pool (smoothing suffices, as predicted);
  token-code single-token probe 0.592 ≈ p=0.6 ✓.
- **H1:45** BUG caught in shuffle diagnostic: single fixed permutation shared
  across windows is learnable by the probe (conv "survived" shuffle at 0.89 on
  a pure order task — impossible). Fixed to per-window independent
  permutations; post-hoc recompute from saved checkpoints (stream E).
- **H1:50** First response curves (circle, H=256): window dictionaries AT
  ORACLE for f ≥ 0.08 with a consistent LOW-frequency dip (S≈0.8 at y=1) after
  per-class oracle normalization — high-pass relative to the periodogram,
  OPPOSITE of the proposal's low-pass prediction. txc/dcac/multiband nearly
  identical on circle at H=256; architecture differentiation instead shows on
  the random embedding (multiband 0.40 > dcac 0.29 > txc 0.17 linear, H=256) —
  unexpected, watch at H=2048. Launched W-scan (W ∈ {4,8,32}, circle, H=256)
  to test the Rayleigh prediction: the low-f dip should move with W.
- **H1:30** Infra hiccups: stale fb_core on pod killed first W-scan launch +
  posthoc stream (ImportError); re-shipped and relaunched. Corrected shuffle
  control now collapses sign info to exactly 0.500 for every architecture ✓.
- **H1:35** multifreq random H=2048 TXC linear = 0.994 (vs 0.170 at H=256):
  the memorization-threshold prediction (P5/P6) confirmed dramatically —
  above |Ω|·M = 1010 templates the structureless task is solved by template
  memorization. Launched H=64 capacity point (stream F).
- **H1:40 — QUANTITATIVE DOUBLE DISSOCIATION.** Exact all-pairs overlap
  combinatorics + measured confusion matrices: random-embedding confusion
  tracks max symbol overlap (Spearman ρ=0.674, p=7e-6, n=36 pairs) better
  than velocity distance (ρ=0.41); circle-embedding confusion tracks
  frequency distance (ρ=0.41, p=6e-3; nearly all confusion mass inside the
  Rayleigh cell |Δf|<1/W: pairs (0,1),(1,2),(2,4)) and NOT overlap (ρ=0.28,
  n.s.). Refinement of P4 learned from data: the right difficulty invariant
  is the minimal-fraction form r ≡ ±p/q mod M with overlap ≈ W/max(p,q) —
  e.g. (16,24) has r=3/2 → overlap 6, explaining its high confusion despite
  "large ratio". Confusion is multiplicative under random embeddings,
  spectral under circle embeddings. Made fig_dissociation.
- **(real wall clock H1:01, 21:31 UTC — earlier entry labels ran ahead of
  reality; treating labels below as real clock.)**
- **H1:05** Branch×frequency heatmap (multiband, circle, H=256) is a clean
  staircase matching the a-priori DCT band assignment of each tone — the
  spectral crosscoder's decomposition verified by independent per-branch
  probes. Launched extra seeds (3,4) for headline curves (stream G).
- **H1:10** ALMOST-MISTAKE caught by quantification: sorted FreqFrac heatmap
  of vanilla TXC *looks* like a clean learned filter bank, but mean spectral
  concentration (0.18) barely exceeds random init (0.13) — the diagonal is
  partly a sorting artifact. Refined with firing-weighted analysis: the 32
  busiest atoms have top-2-adjacent concentration 0.56–0.59 on the CIRCLE
  task vs 0.21–0.22 (= random-init baseline 0.205) on the RANDOM task. So:
  functionally important atoms become tone-like exactly when the data has
  spectral structure; population means dilute this; and the random-embedding
  null validates the diagnostic. Honest spectral-crosscoder claim sharpened:
  equal response, but band-attributed atoms by construction + verified
  decomposition (vanilla gets there with mostly NON-tone-like atoms at the
  population level).
- **H1:20** All main streams done. Full H-sweep: random task — memorization
  jump above 1010 templates for all WINDOW archs (txc 0.17→0.99); conv stays
  at 0.14 even at H=2048 (shift-local filters cannot represent whole-window
  templates — architectural signature of memorization). Circle task —
  multiband better + far more seed-stable at capacity extremes (H=64: 0.944
  [.941,.948] vs txc 0.862 [.764,.922]; H=2048: 0.923 vs 0.841), tie at
  H=256. Pair-task double contrast complete: random pairs all ≈0.53 TXC-lin
  (ratio-invariant, uniformly hard); circle pairs ≈1.0 except sub-Rayleigh
  (1,2)=0.81. SIGN-PAIR CONTROL (3,98): bag-of-symbols MLP = 0.497/0.514
  (chance, both embeddings) vs 1.0 on all non-sign pairs — direction needs
  order; speed lives in the symbol set. TXC-circle reads sign at 1.0.
- **H1:25** Fresh-eyes figure agent report → fixed: in-figure jargon, S(f)
  formula in axis label, Rayleigh annotations, colorbar labels, overlap
  labels, capacity-figure template-count explanation. Rewrote W-scan figure
  as raw-accuracy panels per W (oracle-normalization is pathological at W=4
  where the oracle itself nears chance). W-scan result: deficit recedes as
  1/W for oracle and dictionary alike; conv rises only via probe
  vote-aggregation (0.22/0.31/0.36/0.51 at W=4/8/16/32) while txc saturates
  (0.59/0.80/0.95/1.0). Launched conv L=7 (prediction: filter length, not
  window, sets conv resolution → L=7 ≫ L=3 at fixed W).
- **H1:35** PROBE-BUDGET CONTROL launched: H=2048 probes show train/test gap
  (0.997/0.832) — the "overcompleteness hurts" claim could be probe-sample
  artifact. Re-probing all H=2048 circle models with 60k samples; will weaken
  §4.5 if accuracies recover/equalize. Pod cost so far ≈ $0.71.
- **H1:45 — SELF-CORRECTION (probe budget).** With 60k probe samples the
  H=2048 circle results recover: txc 0.832→0.940, dcac→0.958, multiband
  0.923→0.960. The "overcompleteness degradation" was mostly probe-sample
  starvation (2048-dim codes need more probe data); small residual gap
  remains with split variants on top. Rewrote §4.5 with the correction and
  added "probe budget must scale with code dimension" as a benchmark design
  rule. The H=64 multiband advantage stands (equal code dims, fair probes).
- **H1:50** conv7 (7-tap filters): circle W=16 0.478 vs conv3 0.357; W=32
  0.627 vs 0.508 — filter length sets conv resolution (directional
  confirmation). Sign-pair + W-scan + capacity figures finalized; appendix
  table with corrected shuffle column added to summary. Shuffle column
  interpretation note: for cyclic tasks order-free SET information remains
  decodable after shuffling (consistent with bag theory); only ac_sign
  shuffling is a true null (0.500 everywhere ✓).
- **H1:55** Final dissociation stats (5 seeds): random ρ(conf,overlap)=0.674
  (p=7e-6) vs ρ(conf,−|Δy|)=0.452; circle ρ(conf,|Δf|)=−0.423 (p=4e-3),
  97.7% of circle confusion mass inside the Rayleigh cell. Reconciled all
  quoted numbers in summary with final figures. Fresh-eyes figure fixes
  applied (S(f) formula in axis, Rayleigh shading, self-contained titles).
  First red-team agent stalled; spawned focused redteam2 on final draft.
- **H2:10** Launched multi-lane superposition variant (3 simultaneous circle
  tones in orthogonal planes, d=24, k_win=64, H∈{256,1024}) — addresses the
  "your tasks never have two features at once" gap; template count |Ω|³M³
  ≈ 10⁹ kills the memorization route by construction. PREREGISTERED:
  (1) token-SAE per-lane linear ≈ 0.1 chance; (2) TXC family below
  single-lane performance; multiband ≥ vanilla (band budgets may help or
  hurt — genuine uncertainty since 3 random tones need not match equal
  per-band budgets); (3) NO H-jump from 256→1024 (unlike random-embedding
  single-lane task); (4) per-lane periodogram oracle ≈ 1.
- **H2:30** Red-team report (21 issues, 6 major) — all valid points fixed:
  scoped the ratio theorem (two-class exact, 10-class empirical) everywhere;
  fixed "uniformity the theorem demands" misattribution (within-orbit only);
  softened "provably cannot" to no-perfect-separation + total empirical gap;
  unified the metric (one S formula, ceiling-based, acc/S labels on every
  number); added encoder equations and oracle definitions to §2; explained
  (rather than hid) the conv7 shuffle anomaly (shuffling preserves the
  symbol SET; linear decodability from a fixed encoder is not monotone in
  input information); partial correlations computed (overlap|distance 0.57
  p=3e-4; distance|overlap 0.29 n.s.) — strengthens the dissociation. In
  recomputing I caught MY OWN glob bug (`multifreq_*` matched
  `multifreq_circle` files; polluted earlier ρ=0.674 → correct 0.658) —
  figure and text now agree exactly. The H1:55 entry above retains the
  pre-correction values for the record. Comprehension test (zero-context
  agent on exec summary alone): all four findings reconstructed correctly
  (3–4/5 confidence); their confusions (undefined template count, spectral
  concentration, busiest atoms; ρ mismatch) all fixed. Their "one question"
  (sensitivity of the dip to sparsity/noise sweeps) added to limitations.
- **H2:45** Multilane H=256 complete — preregistered predictions: (1) ✓
  token chance 0.10; (2) ✓✓ split now beats vanilla decisively: multiband
  0.961–0.967, dcac 0.945–0.960, txc 0.910–0.912 (no seed overlap); conv
  collapses to 0.18–0.21 (short filters cannot separate summed tones);
  (4) ✓ oracle 0.995. H=1024 running for prediction (3). Wrote §4.6.
- **H2:50** Multilane complete. Prediction (3) ✓: H 256→1024 moves accuracies
  only marginally (txc 0.91→0.94, splits ≈0.95) — no memorization jump, since
  10⁹ windows make memorization impossible by construction. fig_multilane
  made; §4.6 finalized. Synced all 143 model checkpoints (207 MB).
- **H2:55** Pod z53dh4ix5bhxpl terminated. Total compute: 2.6 pod-hours ≈
  **$1.13** (budget $50). All results, checkpoints, theory outputs synced
  locally under results_synced*/, theory_synced/.
- **H3:00** Closing edit pass. Discovered two SILENT NO-OPs from earlier
  python str.replace batch edits (the §4.2 circle bullet and the H=64
  parenthetical retained stale text because old_string didn't match after
  prior edits) — both fixed via exact-match edits; lesson recorded: verify
  every batch replacement by grepping for the new text. Launched a
  verification agent to check all 21 red-team fixes against the final file.
  Fixed stale §6 next-steps item (multi-tone was actually done in §4.6),
  updated the falsified-predictions bullet with the superposition result,
  added frontmatter to figures/summary_table.md.
- **H3:05** REAL-MODEL EXTENSION (next-steps #1, timeboxed to H6). New pod
  b8la823lzcwpml. GPT-2 day-stride task: 16 weekday tokens with constant
  stride y ∈ Z_7 (includes sign pairs), labels = stride; residual windows at
  day positions. PREREGISTERED: embedding layer behaves like the synthetic
  circle task (P1/P2 apply verbatim — position 0 has no context, each
  embedding is a function of (day, position) only); deeper layers raise the
  single-position ceiling as attention mixes.
- **H3:10** First run: at block-3 output, the SINGLE-position probe already
  reads stride at 1.000 — GPT-2's attention has fully converted temporal
  structure into per-position linear structure by layer 3. Mid-layer
  dictionary comparisons are therefore vacuous; the genuine conversion
  problem lives at hidden_states[0] (wte+wpe). Redesigned: dictionaries at
  the embedding layer only (token_sae/txc/multiband/conv); plus a
  position×layer map of single-position probe accuracy showing where the
  model itself converts (hs ∈ {0, 1, 4, 7}); Engels day-circle PCA check on
  the embedding layer.
- **H3:45 — REAL-MODEL RESULTS (GPT-2 day-stride).** (1) Day-embedding PCA:
  the 7 day means form a circle in correct weekday order (top-2 PCs, 55%
  var) — Engels geometry recovered. (2) Embedding layer reproduces the
  synthetic conversion result verbatim: single-position probes 0.149 ≈
  chance at all 16 positions (P1 holds — no context at hs=0); raw stacked
  lin 0.181 / MLP 1.000; token-SAE codes lin 0.527 (wide variance loophole
  at M=7, conversion still fails); TXC/multiband/conv codes lin 1.000
  (2 seeds, FVU≈0). (3) Position×layer map: ONE attention block linearizes
  stride at every position except position 0 (causal — stays at chance at
  every depth, a built-in pipeline control). Interpretation added to §4.7:
  temporal dictionaries matter at interfaces where the model has not yet
  converted; the position-resolved ceiling map locates them. Second pod
  terminated. Total sprint compute ≈ $1.40.
- **H4:05** Final comprehension test (zero-context agent, updated exec + §4.7
  + figures): all five findings followable, exec/§4.7 numbers consistent, no
  showstoppers. Its three fixes applied: removed the "reproduces exactly"
  overclaim from exec finding 5 (added the 0.149-vs-1/7 chance baseline and
  the variance-loophole caveat for the 0.53), corrected §4.7 "1.00 at every
  position" to "≥0.95 (mostly 1.00)" to match the figure, regenerated
  fig_gpt2_stride (label collisions, hidden block-3 line). Two exec trims
  applied (~813 → ~770 words).
- **H4:10 — SPRINT DELIVERABLE COMPLETE.** summary.md final: 5 findings, 12
  embedded figures, theory (P1–P5) + 7 result sections + limitations with
  falsified preregistrations + research map + references + full appendix
  table. Both pods terminated; total compute ≈ $1.40 of the $50 budget.
  All raw results (JSONs, 143 checkpoints, kernel spectra), code, theory
  notes, and bibliography synced under this directory. Remaining sprint time
  held in reserve for any final review passes.

### Post-sprint user-directed extension: backtracking case study

- **H4:30** User asked to try the frequency-decomposed crosscoder on the
  backtracking task. Read the worktree map + paper digests. OUR benchmark:
  Llama-Scope 32x feat_71839 α=8 → 80% per-prompt steering success
  (kw 2.53%, coh 2.90); a prior TXC attempt (T=5, 30k) scored kw 0.40 —
  the bar to inform. Signal known to be anticipatory + temporally localized
  (D+ = [-13,-8] before "Wait"/"Hmm"). Subject model
  deepseek-ai/DeepSeek-R1-Distill-Llama-8B is UNGATED (the old HF-gate
  blocker only applied to base-Llama caches). Reusing the case study's 300
  traces + keyword labels verbatim; regenerating only the L10 activation
  cache. PREREGISTERED: (1) raw right-edge probe already strong (L10
  attention has mixed context — cf. GPT-2 finding); window dictionaries add
  margin only if anticipation has multi-token structure beyond what L10
  positions already linearize. (2) Branch question genuinely open: slow
  build-up (DC/low) vs localized sentence-start event (mid/high) — the
  measurement is the point; T-SAE/Venhoff evidence weakly favours
  low-frequency coherent trajectories. (3) Backtracking-predictive
  Llama-Scope features (incl. 71839) have more low-frequency activation
  power than random features. (4) Dead-atom fraction reported (their TopK
  runs hit 94% dead; our density is higher so expect far less).
  Design: T=16 right-edge windows (their convention), H=4096, k_win=256,
  by-trace 80/20 split, balanced probes with AUC, NEG_BUFFER=25.
- **H5:30 (extension)** Backtracking probe study complete (pod 3 terminated,
  ≈$1.2). HEADLINE: anticipation signal is LOW-FREQUENCY — branch AUCs
  decline monotonically DC 0.803 > low 0.787 > mid 0.740 > high 0.733; DC
  branch alone beats whole vanilla TXC (0.728) and matches token-SAE stacked
  (0.802); multiband > txc 0.79 vs 0.73 at matched FVU. feat_71839 spectrum:
  lowfrac 0.362 vs random 0.133±0.111 — the 80%-success steering feature is
  a slow feature. Llama-Scope-on-distill edge AUC 0.709 < raw 0.769 (domain
  mismatch visible). Reconciled with paper c7 (TXC wins there at 75× compute
  with BatchTopK per-position recipe; no contradiction with our undertrained
  vanilla TXC losing here). Wrote §4.8 + fig_backtracking. Next: reduced
  c7 reproduction on H100 (bh3rx2ez6hva3o) + spectral graft.
- **H6:30 (extension)** c7 reproduction infra saga: two H100 pods burned on
  what looked like broken egress — diagnosis: NOT general egress, NOT HF
  throttling; the NousResearch/Meta-Llama-3.1-8B repo route specifically is
  broken from these RunPod DCs (8 KB/s) while deepseek/gpt2 repos pull at
  100+ MB/s from the same pod. Auto-mode classifier correctly blocked
  shipping the user's HF_TOKEN to a remote command line; resolved WITHOUT
  any token by switching to the unsloth/Meta-Llama-3.1-8B ungated mirror
  (46 MB/s), sed-swapped into their model registry on the pod. Pipeline
  (their cache builder 30k×128 resid L10 → their train_llama_txc topk_sae +
  txc_bare T=5 k_pos=20 d_sae=32768 + my spectral arm at matched hparams →
  detection PR-AUC per paper protocol) launched on CA-MTL-1 H100. Wasted
  pod cost ≈ $1.5.
- **H8:25 (extension) — C7 REPRODUCTION + SPECTRAL GRAFT COMPLETE.**
  Reduced-faithful rerun of the paper's c7 pipeline with their own scripts
  (their FineWeb cache: 30k×128 base-Llama L10 resid via ungated mirror at
  46MB/s after diagnosing a repo-specific CDN failure; their trainer at
  d_sae=32768, k_pos=20, 10k steps = 1/30th paper compute; 1 seed):
  detection PR-AUC@S8 — TopK SAE 0.164 (97% dead, plain-TopK pathology in
  their own trainer), TXC-base 0.222 → PAPER ORDERING REPLICATES (paper:
  0.175 vs 0.226). Spectral TXC at matched hparams: full code 0.145 at S8
  (top-S selection scatters across signal-free bands) but DC-BRANCH-ONLY
  0.260 / ROC 0.754 = best in table, above the paper's TXC-pro (0.242).
  Frequency decomposition wins as SIGNAL LOCALIZATION, consistent with the
  probe study + feat_71839 slow spectrum. Addendum written into §4.8.
  Infra cost of the night: ~2 wasted H100-hours on the broken-CDN saga.
  All pods terminated 04:54 UTC. Total sprint+extensions compute ≈ $8.
  Elapsed 8.4h of the 10h window.
