# Working state — agent `runpod`

**Last rewrite:** 2026-07-24 (late), immediately after receiving
`briefings/candidate-factory-broad.md` (pre-compact handoff; NOT started).
**State: NEW TASK ASSIGNED, ZERO WORK DONE ON IT.** Previous task
(hunt-support-synthetic) is DONE + **APPROVED** — mac-local review entry at
the tail of `task_hunt/LOG.md`; my receipt drove a binding retraction of the
RECORD § 3b dilution clause. Briefing retired. Do not redo.

## Who / where
Remote CC on RunPod (Linux), repo root `/workspace/temp_xc`. **I am `runpod`
(original box — `/workspace/.agent_id` does NOT exist; do not create it).**
Parallel agents runpod-b/-c/-d/-e — their briefings are NOT mine, EXCEPT:
`candidate-factory-traces.md` (runpod-b's) sets conventions that GOVERN my
new task identically (read it). Shared-branch rules (agents/README.md):
`git pull --rebase origin arxiv` before EVERY push (commit this STATUS
first); shared files append-only (LOG conflicts: keep both, mine last);
cite commits by SUBJECT LINE. Tokens in `/workspace/.tokens/` (`gh_token`,
`anthropic_key`). Push:
`git push https://x-access-token:$(cat ../.tokens/gh_token)@github.com/chainik1125/temp_xc.git arxiv`.
32 CPU / 128 GB, NO GPU; python -u for background runs; `sleep` blocked
(use `until …; done` or background tasks).

## ACTIVE TASK (not started): briefings/candidate-factory-broad.md
**Quantity mode (Han directive): maximize screen-ready case-study
candidates, everything OUTSIDE the Ward corpus** (runpod-b owns Ward
traces via candidate-factory-traces.md). **Ship incrementally by Saturday
morning PT** — one LOG line per bundle/kill as it lands, not one drop.

Read order post-compact: (1) `briefings/candidate-factory-broad.md`;
(2) `briefings/candidate-factory-traces.md` — its bundle format, triage
kill authority, and masking discipline govern me identically;
(3) `task_hunt/LOG.md` tail (both mac-local review entries + runpod-b's
incremental bundle lines — avoid collisions; they may land Ward bundles
while I work); (4) runpod-b's bundle-format exemplars:
`task_hunt/labels/` conventions (`ward_lambda.npz` / `proofops.npz` npz +
balanced-manifest format) and the interleave artifacts (below).

**Deliverable 1 — the ledger FIRST (~1 h):** commit
`experiments/explorations/task_hunt/CANDIDATES.md` with 10–20 ideas, one
paragraph each, each vetted on the four round-1 lessons: (a) conversion
risk — does the latent help next-token prediction? (the round-1
graveyard); (b) label-side per-token proxy — readable from current-token
identity?; (c) clock feasibility at panel T; (d) regime shape + predicted
T-pattern. Verdict per idea: BUILD / PARK / DEAD + one-line reason. Dead
ideas are deliverables too.

**Deliverable 2 — ≥ 3 top vetted bundles** (or honest triage kills). Per
bundle (traces-briefing format): labels npz + balanced manifests +
shuffled-window null + triage stats JSON + `<name>/CARD_DRAFT.md` (regime
framing, predicted T-pattern, falsifier — the running agent freezes).
Builders committed BEFORE outputs. **Label-side triage = kill authority:**
label well-predicted from current-token identity alone or from position ⇒
FAILS TRIAGE ⇒ one LOG line, do NOT ship (a free kill is a win). Zero-API,
exact labels only. Aggregation-framed candidates accepted (shuffle-immunity
as mechanism receipt; regime-2 wins count). Event/marker tokens must be
masked from probe rows — state the masking rule in each card.

**Briefing's seed ideas (vet, don't assume; add my own):**
- **Interleave `tss` (labels EXIST — finish it).** runpod-b's committed
  artifacts: `labels/build_interleave.py`, `interleave_lib.py`,
  `labels/interleave_fineweb_{gpt2,gemma2,llama31}.npz`,
  `interleave_stats.json`, `interleave/CARD_DRAFT.md`. tss PRIMARY
  (unigram ≈ 0.55, near-blind); source identity DEMOTED to disclosed
  anchor (0.66 matched = expected regime-1 kill face). NOTE an apparent
  tension my post-compact self should not trip on: the mac-local review
  (LOG tail, point 4) says the class "stays PARKED until greenlight", but
  the NEWER briefing (Han directive, same evening) explicitly says "the
  parked status is LIFTED for screening under quantity mode" — the
  briefing is the operative mandate.
- Vocabulary-novelty rate on fineweb (fraction of window tokens unseen
  earlier in doc — topic-drift intensity, exact).
- List/enumeration density trend on fineweb (markers exact; mask marker
  tokens).
- NEW-corpus intensity candidates (CPU-downloadable, exact labels; models
  with existing caches or cheap to cache): OpenWebMath equation-density;
  dialogue turn-length / speaker-switch rate; news chronology density.
- AVOID (recorded dead): bracket/indentation state-tracking, repetition
  detection, forbidden-word onset, emotional onset, rollout-level booleans.

**GPU economics to state per bundle:** runpod-e's volume holds fineweb
caches for gpt2 / gemma-2-2b / llama-8b → fineweb candidates screen in
minutes; a NEW corpus needs one cheap caching pass (~minutes/model on
H100) — fine but must be said in the bundle; new-corpus bundles must
include the tokenized corpus artifact or an exact re-pull script. OPEN
QUESTION for post-compact me: where the fineweb TEXT lives for label
building on this box — check how `labels/build_interleave.py` sourced it.

**Vetting lessons library (from round 1, for the ledger):** per-token-first
triage convention; the conversion screening question "will the model
decline to maintain this as a per-position state?"; hedging-LEVEL lesson
(levels are aggregation-recoverable; slopes collapse to anchor − window
mean); position-floor lesson (kernel-only λ̂_hist primary, position-only
probes are the floor check); anchor lesson (exclude current-sentence /
event tokens from probe rows).

**Acceptance gate:** CANDIDATES.md committed; ≥ 3 bundles shipped or
honestly triage-killed; LOG line per bundle/kill; STATUS rewritten; no
reviewer/meeting quotes; stop for review (briefing stays).

## Just completed (2026-07-24, APPROVED — do not redo)
hunt-support-synthetic: Item 1 NO-MIRROR-DIP (mirror rises 0.87→0.95,
never dips; fixed-budget arms starve to 0.56 atoms/token with recovery
unmoved ⇒ dip/starvation doubly dissociated ⇒ **§ 3b dilution clause
RETRACTED** by review to "cause not established"); Item 2 FLAT (T-SAE
pair-distance knob inert at DPI floor ≈ 0.41; α=0 moves nothing —
rejoinder closed with a receipt). Artifacts:
`task_hunt/support_synthetic/` (CARD, results, figs); arch
`src/temp_bench/archs/tsae_delta.py` (TSAEDelta — bitwise ≡ tsae at Δ=1,
6 contract tests in `tests/test_support_synthetic.py`). Earlier: TXC-pro
dissection (approved), C7 close, C6, stage-6 #3b, C5.

## Reusable know-how (this box)
- Grid engine: `design.uniform_cells(ds, F, n_steps, archs=…,
  k_pos_sweep=…, d_saes=[…], window_ts=(T,), L=32)` → `grid.run_pool`
  (8–16 workers). Backtracking canon: DS `toy_backtracking_selfexcite_d64`,
  F=20, N_STEPS=30000, seeds {1,2,42}. Window cells ~7 s; sequence-mode
  ~2–6 min/cell. Pooled dict rule d_sae ≥ k_pos·T. Runner cache: identical
  config ⇒ cache hit returns the existing row, no append (code_version is
  NOT in eval_key) — reuse is free and dup-safe.
- `lambda_recovery` probe: UNREGULARIZED LinearRegression, 1024 windows ×
  (32/T) tiles — watch p/n at high T with big d_sae.
- Skeptic pattern: `loss_dissection/skeptic_dissect.py` (cache-guarded,
  raw persisted pre-parse, NEVER re-rolled; Meter → expansion
  spend.json; cumulative $11.52/$25).
- Mid-grid the leaderboard/manifest are live-appended — never
  rebase/stash while a grid runs. Rebase rewrites SHAs — cite subjects.

## Repo state
Clean, in sync with origin/arxiv after pulling the review + new briefings.
Nothing mid-flight. **Next action post-compact:** read the two
candidate-factory briefings + LOG tail, then write and commit
`task_hunt/CANDIDATES.md` (the ledger comes FIRST, before any builder).
