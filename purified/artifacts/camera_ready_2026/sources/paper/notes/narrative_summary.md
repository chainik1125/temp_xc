# Temporal Crosscoders — Paper Narrative & Working Plan

## Context

We are finishing the Temporal Crosscoders (TXC) paper for NeurIPS 2026. The manuscript is partially drafted (`main.tex`, `appendix.tex`); the architecture section, the synthetic-setting formalism, and the backtracking case study have substantial prose, while the introduction, related work, qualitative analysis, sparse probing, emergent misalignment, and discussion are bullet-point skeletons. The job is to turn the existing scaffolding into a coherent, evidence-backed paper. This file is the working narrative/principles doc — we'll work section-by-section against it.

---

## The Narrative (one-paragraph compression)

Dictionary learning methods (SAEs, crosscoders) decompose model activations into interpretable features, but they do so *one position at a time* — they impose a stationary sparse code on inherently temporal data. We propose **temporal crosscoders (TXC)**: a simple, flexible architectural framework that crosscodes across sequence positions instead of (or in addition to) layers, using a shared latent that reconstructs an entire temporal window through position-specific decoders. To know whether modelling temporal structure actually helps, we have to *measure it correctly*: we therefore introduce **TempBench**, which combines (i) a generalisation of the canonical synthetic-features setting to ground-truth *temporal* features (parameterised as HMMs over feature firing patterns), and (ii) a panel of real-world case studies with clear behavioural ground truth (sparse probing, emergent misalignment, backtracking, RLHF preferences). On both axes, the simplest TXC variant ties or beats existing temporal and non-temporal baselines. Since this is the most naive instantiation of the framework, the result is best read as: *temporal structure matters, the TXC is a reasonable way to model it, we measured it correctly, and there is rich room to build on it.*

## Three concrete claims (paper compressed to claims)

1. **Architecture.** A simple architectural change — using a shared latent decoded by position-specific decoders over a window of $T$ positions — defines a flexible family of dictionary-learning models (the TXC) that subsumes per-token SAEs as the $T=1$ special case and admits Matryoshka / contrastive / multi-layer extensions.
2. **Measurement.** The current synthetic-feature benchmarking framework defines features only locally; we generalise it by promoting the per-position firing distribution to a stochastic process (HMM), giving a principled definition of *ground-truth temporal features*. This lets us benchmark temporal architectures against each other and against non-temporal baselines on a common synthetic substrate.
3. **Empirical result.** On TempBench (synthetic + real-world panel), even the most naive TXC ties or beats existing temporal architectures (T-SAE, TFA, multi-layer crosscoder) and per-token SAEs, on both interpretability proxies (sparse probing) and causal/behavioural proxies (backtracking inducement, EM, etc.). The simplest version winning is itself the point: it lower-bounds what temporal modelling buys you.

## North-star reader reaction

"Of course you need to model temporal structure. The TXC is a reasonable way of doing it. They measured it correctly. I'm excited to build on this."

This implies three failure modes to avoid:
- Overclaiming — "TXC dominates" rather than "TXC ties or wins, and it's the simplest version".
- Under-motivating — failing to make temporal structure feel obviously necessary.
- Sloppy measurement — leaving the reader unsure whether the wins are real.

---

## Key principles (extracted from `notes/writing_instructions.md` + the user's framing)

### Narrative discipline
- **Compress to ≤3 claims.** Every section must obviously support claim 1, 2, or 3. If a paragraph doesn't, cut it or move to appendix.
- **Frame as "another arrow in the quiver done right" not "TXC dominates".** The framework + measurement contributions are the load-bearing claims; the empirical wins are corroboration.
- **Inform, not persuade.** Acknowledge ties as ties. The paper's strength is rigour, not hype.

### Evidence discipline
- **Quality over quantity** — one decisive experiment per claim beats five vague ones.
- **Strong baselines.** Per-token TopK SAE, T-SAE (Bhalla), TFA (Lubana), multi-layer crosscoder, all trained on the same activation cache at matched sparsity / expansion.
- **Red-team every claim.** Cherry-picking, post-hoc selection, batch-size confounds, judge noise — flag and address each.
- **Pre/post-hoc tracking.** Be explicit about which architecture choices and metric definitions were locked before seeing results.

### Writing discipline
- **Spend equal time on abstract / intro / figures / everything else.**
- **No wall-of-bullets.** The current intro, related work, and discussion are placeholder bullets — they must become prose with clean opening/closing sentences per paragraph.
- **Every figure has a takeaway.** Caption should let the reader understand the figure cold.
- **Define jargon.** "TXC", "TempBench", "genuine backtracking", "edge-emitting HMM" — define on first use.

### Project-specific
- **Backtracking and sparse probing are the headline real-world results** (per the user's structure). EM is a tie. Mention RLHF only if it's load-bearing.
- **TXC is a *framework*, not a single architecture.** Always emphasise the framework framing — base, pro, and obvious extensions (longer windows, contrastive, multi-layer) are all instances.
- **The synthetic generalisation is a contribution in its own right** (per the user's notes: "this point alone could be a solid paper"). Treat it that way — don't bury it.

---

## Current state of each section (audit)

| Section | State | Major gaps |
|---|---|---|
| Abstract | Half-drafted, has a TODO mid-sentence | Needs to land claim 1/2/3 with a concrete metric in one sentence |
| §1 Intro | Bullet skeleton + figure scaffold | Needs prose: motivation for temporal structure, contribution list, intro-figure design (currently 3 placeholder subfigs) |
| §2 Related Work | Bullet skeleton | TFA, T-SAE, multi-layer crosscoder, Chanin synthetic, computational mechanics — short prose treatment |
| §3 TXC | Mostly drafted (notation + arch + base/pro variants) | Architecture figure (`figs/tx_architecture.png`) exists; Matryoshka / contrastive losses currently a one-paragraph aside — may need own subsection or appendix pointer |
| §4 Synthetic | Formalism is solid (firing-mask process + HMM parameterisation) | Need: the "two special cases of HMM" we promised; results figures (currently three placeholders); local-vs-global axis |
| §5 Qualitative | Empty + one big placeholder | Whole section to draft — 1–2 case-study features that show the TXC discovering something a per-token SAE misses |
| §6 Real-world benchmarks | Section header + bullets | Needs framing paragraph on "why these four tasks" |
| §6.1 Sparse probing | Two placeholder figures, no prose | Needs setup + results paragraphs (Gemma + Gemma-IT) |
| §6.2 Backtracking | Substantially drafted, two placeholder figures, two `[PLACEHOLDER: …narrative]` blocks in red | Numbers need filling in; otherwise this section is the most mature |
| §6.3 EM | Empty placeholder | Whole section to draft (or cut to appendix if it's a tie with low signal) |
| §7 Discussion | Three bullets | Needs prose: scaling windows, attention/skip-connection caveats, framework extensions |
| Appendix | Backtracking appendix is rich (~22kB); other appendix sections are stubs | Synthetic-setting details, qualitative-feature details, sparse-probing details, EM details |

## Critical figures to design (in priority order)

1. **Figure 1 (paper summary).** Currently three subfigs: `temp_cartoon`, `cross_cartoon`, `summary_rose`. The summary panel must communicate the headline empirical result (TempBench rose plot or grouped bar across tasks).
2. **Architecture figure.** Already exists (`figs/tx_architecture.png`); revisit caption to make $T$, position-specific decoders, and shared latent obvious.
3. **Synthetic results figure.** `local_vs_global` + `local_vs_global_transformer` — should show TXC recovering temporal-HMM-latent features that per-token SAEs cannot.
4. **Backtracking inducement + detection.** Already scaffolded — once numbers land, this is the headline real-world figure.
5. **Sparse probing.** Gemma + Gemma-IT panels — needs to communicate "TXC ≥ baselines on probe accuracy at fixed top-S".

---

## Working plan (section-by-section order)

Suggested order, optimised for reducing risk of late rewrites:

1. **Lock the abstract + intro skeleton first** — the rest of the paper services these. Bullet outline → prose.
2. **§3 TXC** — already mostly there; tighten and finalise architecture figure caption.
3. **§4 Synthetic** — finish the "two HMM special cases" + draft results prose around the placeholder figures (need the actual results from the user).
4. **§6.2 Backtracking** — fill in the two `[PLACEHOLDER: narrative]` blocks once numbers are in.
5. **§6.1 Sparse probing** — short, results-driven prose.
6. **§5 Qualitative** — pick the cleanest 1–2 features and write it as a vignette.
7. **§6.3 EM** — decide: keep in main body if it ties cleanly, otherwise demote to appendix.
8. **§2 Related Work** — once the contribution is sharp, this writes itself.
9. **§7 Discussion** — limitations (single seed, judge κ, batch-size effects, T fixed at 5) + framework extensions.
10. **Re-pass on abstract + intro + figure 1** with full evidence in hand.

## Open questions for the user (to resolve before deep writing)

- Which of the four real-world tasks (sparse probing, EM, backtracking, RLHF) make the main body, and in what order? Currently sparse probing + backtracking are scaffolded; EM is a placeholder; RLHF is in the user's mental model but not in `main.tex`.
- For the synthetic section: do we have results yet for "TXC vs baselines on the temporal HMM benchmark", or are we still designing the experiment?
- How polished is the qualitative-features section meant to be — is it one feature presented carefully, or a panel?
- Is there a preferred name for the panel — "TempBench" appears in the abstract and one comment but isn't yet introduced in the body.

## Verification

This plan file is a working narrative doc, not code. "Verification" = the user reads it, tells me what's wrong about my read of the narrative, and we update before drafting prose.
