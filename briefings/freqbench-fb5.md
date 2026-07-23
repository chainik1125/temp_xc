---
status: active
created: 2026-07-23
for: runpod-b
venue: runpod
---

# FB-C3 — card FB-5 `permuted_tones`: the temporal-knob acid test

**You are `runpod-b`** (check `/workspace/.agent_id`). `runpod` (C7) and
`runpod-c` (conversion-depth, GPU) run in parallel — shared-branch rules
in `agents/README.md`, incl. the NEW commit-citation rule (cite commit
subjects, or verify SHAs post-push). Governing protocol: `LOOP.md` — note
the T3 strict commit-then-run AND the new **non-absorption obligation**
in card item 1 (adopted at the FB-4 review; your own salvage note).
Prime directive: **a sound verdict, never a win.**

**Session limits:** ~10 h wall · **$25 cap** (skeptic on
`claude-fable-5`; freqbench meter) · rewrite `agents/runpod-b/STATUS.md`
before any compact · hard lines: **no cards beyond FB-5, no
program-rule/gate edits, no `temp_bench/core/` edits.**

## The question (FB-4's salvage, now with the right knob)

Is spectral's power/equality dominance generic order-2-even *conversion*,
or alignment between the latent's temporal structure and the DCT frame?
FB-4 proved a spatial knob cannot ask this (absorption). The temporal
knob can: **replace the tone ladder's linear phase schedule with random
permutation schedules** — order-2-even structure preserved, DCT
alignment destroyed.

## Card FB-5 `permuted_tones` — freeze FIRST (`freqbench/cards/FB-5.md`)

**Construction (the frequency substrate, one substitution).**
`cyclic_tones` draws z_t = (t·Y + B) mod M and embeds via the circle
codebook (angle 2πz/M). FB-5 replaces the linear map z ↦ Y·z with
**K = 10 fixed uniformly-random permutations π_Y of Z_M** (M = 101; drawn
per data seed, matching the multilane embedding convention):
z_t = π_Y((t + B) mod M), circle codebook, σ and seq_len/L as frequency.
The tone benches are the special case π_Y(z) = Y·z (Y coprime to M —
itself a permutation): FB-5 vs frequency is a *controlled* comparison in
which only DCT-alignment of the temporal trajectory changes.

**Proof obligations.**
- **P1 exact:** π_Y is a bijection of Z_M ⇒ the single-token marginal is
  uniform on the M-point circle for EVERY Y — identical to the tone
  benches. State and verify numerically.
- **P2 (phase-averaging):** the uniform offset B averages any additive
  readout over the schedule's image exactly as in the tone proof —
  restate for general π and verify.
- **Ceiling:** matched-filter oracle (correlate the T-window against all
  M offsets × K schedules); verify numerically per T ∈ {2,4,8} on the
  built generator. Expect near-saturation by T = 8 (8 points of a random
  length-101 schedule are near-unique); report the T-resolution curve.
- **Non-absorption (card item 1, mandatory):** argue why THIS knob is not
  absorbed: the linear-schedule and random-permutation ensembles differ
  in law (trajectory autocorrelation: sinusoidal vs white-ish), and no
  panel architecture is equivariant to temporal reindexing — the DCT
  prior is a *fixed* temporal basis. Write it as a § of the card.
- **Spectral-envelope reference (NOT an abort gate):** a random
  permutation's T-window DCT band-energy profile fluctuates; band
  energies alone may carry some Y-information at small T. Measure the
  **envelope oracle** (best classifier on band energies only) at gating
  and report it as a reference curve — the spectral-vs-post comparison is
  interpreted against it (spectral ≈ envelope oracle ⇒ it reads envelope,
  not structure).
- **P6:** template count K·M = 1,010 vs d_sae ≤ 202 — state per the
  standard audit.

**Frozen prediction directions (mac-local, 2026-07-23 — sharpen reasons,
never directions):**
1. per-token / tsae / stacked / txc-pre ≈ 0 (P1/P2/additive — as every
   tone bench).
2. **txc-post positive at T ∈ {4,8}** (its taps are unconstrained in
   time — it can learn matched filters for arbitrary schedules); wide
   band accepted (0.1–0.8), direction is what's frozen.
3. **spectral trained BELOW txc-post at the canonical T=8 cell** — the
   reversal of multilane. Its band-limited taps cannot represent a
   spectrally-generic schedule inside a branch; residual spectral score
   should track the envelope reference.
4. **spectral untrained ≈ post untrained** (the multilane 4× access-prior
   gap collapses — no band alignment exists at init).
5. Falsifiers: any arch > 0.1 at T=1 (P1 bug); winner trained ≈ untrained.

**The fork, stated in the card (both outcomes advance the program):**
prediction 3 holds ⇒ the subtype rule's power leg gains the qualifier
"…when the power concentrates in few DCT bands" (alignment-conditional);
prediction 3 FAILS (spectral matches/beats post beyond the envelope
reference) ⇒ spectral's dominance is band-*competition* structure, not
alignment — the rule survives strengthened. Say both in advance.

## Pipeline (after freeze — the standard LOOP order)

Build (generator variant + datasource append + contract tests: π
bijectivity, per-seed redraw, P1 marginal uniformity) → T1 numerical
discharges + § 8 gating incl. the envelope reference (**strict
commit-then-run**; amendments as own commit pairs) → T2 battery
(bag/shuffle semantics: state what a within-window shuffle destroys for
permutation schedules; memorization budget; probe budget by code dim) →
skeptic (raw persisted pre-parse) → uniform grid (locked design,
T ∈ {1,2,4,8}) → blind verdict vs the frozen directions →
registry/BENCHMARKS/REPORT + FreqFrac at bench time. A gate ABORT is a
success — record and stop.

## Acceptance gate — stop for review

Card end-to-end verdict (or honest gate-kill) + records + trackers +
STATUS pushed; spend logged; cycle log appended to PORT (§ J). Briefing
stays until mac-local review, then it is deleted.
