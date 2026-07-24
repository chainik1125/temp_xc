# Working state — agent `mac-local`

**Last rewrite:** 2026-07-23 (post round-3 review).

## Who / where
Local CC on the Mac at `~/research/projects/temp_xc`. Role: prototyping,
review, orchestration. Three runpod agents, ALL IDLE (no active
briefings): `runpod` (PhenBench, 32C), `runpod-b` (FreqBench, 32C),
`runpod-c` (H100, 700 GB volume — holds the preserved Ward 144 GB + EM
36 GB caches; only IT can see them).

## Git
Branch `arxiv`; tip = the round-3 review commit (clean, pushed).

## ✅ Last completed: ROUND-3 REVIEW (2026-07-23) — all three APPROVED

- **C7 close** — reasoning int/eq NEGATIVE at corpus resolution; prize
  half-claimed by design (text ✓ / reasoning ✗-at-resolution); LAST
  estimator cycle honored; reopening = data lever (more/longer traces).
- **FB-5 permuted_tones** — POSITIVE (weak realization), ALIGNMENT fork:
  spectral is quantitatively the envelope reader; power leg qualifier
  adopted into README ("…when power concentrates in few DCT bands").
  My literal T=8 k=2 bet FAILED (scored against me in the record §5);
  mechanism clause held precisely. Cleanest process cycle yet (zero
  amendments, first-run gates).
- **Conversion-depth** — audit items 2/3/5 ANSWERED (§ Resolution in
  `docs/substrate_audit_2026-07.md`): §5.2 = reader-predictability
  (base ≈ generator; late-layer ≤+0.02 margin = follow-up); §5.1
  harmless; **EM negative depth-confounded** (inverted-U, peak +0.13,
  g_order +0.11 at L13). My P3 prior falsified too. Idea doc updated
  (three g(ℓ) shapes). Camera-ready actions in RECORD § 7.
- Hygiene: 0 dup keys / 7,116 rows; 179 tests green; spends $11.01
  expansion / $1.63 freqbench cumulative. All briefings deleted.

## 🚨 PHASE SHIFT (2026-07-23): NeurIPS reviews OUT — rebuttal mode

Scores 5/4/1; deadline **2026-07-27**. Review copies + the full
reviewer-mapped battle plan live ONLY in `private/neurips_reviews/` +
`private/rebuttal_plan.md` (gitignored — NEVER commit; tracked files
stay review-content-free). Han's three standing directives: (1) synth
revamp (the why-story), (2) real-world case-study redo with
TXC-appropriate evals (PRIORITY, > (1)), (3) TXC-pro loss dissection.
**Scope (Han, 2026-07-23): backtracking multi-seed + paper latex are
the human team's — agents run ONLY the three directives.** **RE-PLAN 2026-07-24 v2** (post team meeting — transcript in
`private/transcripts/transcript-2026-07-24.txt`; plan in
`private/rebuttal_plan.md`; Han: time > compute cost): synthetic
generation PAUSED; **priority = THE TASK HUNT** (TXC > T-SAE with
T-scaling on a real task). FIVE agents live:
- `runpod-c` (H100 + volume): **em-redo REINSTATED to completion**
  (Han's call; win = reportable result, loss = archival datum). Owns
  all volume WRITES.
- `runpod-d` (NEW GPU): hunt arm A — λ̂ intensity + proof-op runs +
  backtracking shuffle receipt (`task-hunt.md`); volume read-only or
  rebuild. Screens BOTH cached reader models (free).
- `runpod-e` (NEW GPU): hunt arm B — repetition-lag Δ **across model
  scale** (gpt2/gemma-2b-base/llama-8b-base; induction-conversion
  prior: gap larger in smaller models) + confidence trend
  (`task-hunt-b.md`). Volume-independent.
- `runpod-b`: label prep (`task-hunt-prep.md`, feeds d+e).
- `runpod`: txcpro-dissection unchanged.
Model axis = Stage-1 SCREEN only; Stage 2 = best (task, model) cell.
Avoid: backtracking detection / forbidden-word (team-owned), bracket
state-tracking (dead end). Team check-in Sun 2026-07-26 10am PT —
review everything before it. Rebuttal deadline 07-27.

## ⏭ (pre-review fork — parked)

(a) **TXC-tracking session on runpod-c** (must run there — caches on its
    volume): train dictionaries per layer on Ward + EM caches; test
    "trained-TXC advantage tracks g(ℓ)"; EM g_order slice (+0.11 at
    L13) = the strongest candidate for a grounded position-aware win.
    Predictions pre-written in conversion_depth RECORD § 6.6.
(b) **Camera-ready / rebuttal edits** — §5.2 reader-predictability
    reframe; §5.3 scope-narrowing (+ the positive spin: window headroom
    at the right layer is REAL); §5.1 stated-choice line. NeurIPS
    reviews may already be out.
(c) New-data lever for the reasoning cell (more/longer traces).
(d) Next FreqBench axis point — axis-3 localization is the one
    uninstrumented axis (burst/wavelet vs stationary).

## Standing context
- Ambience principle + subtype rule (phase leg T-conditional; power leg
  alignment-qualified): README coordinates; memory
  `project-ambience-principle`.
- Trackers: BENCHMARKS.md (11 live + § B aborts) · expansion/LEDGER.md
  (CLOSED through C7) · REPORT.md 96/96 · freqbench/PORT.md § G–J ·
  `docs/substrate_audit_2026-07.md` (RESOLVED) ·
  `conversion_depth/RECORD.md`.
