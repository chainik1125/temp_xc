# Working state — agent `runpod-b`

**Last rewrite:** 2026-07-24 (pre-compact) — **NEW TASK ACCEPTED, not
yet started: `briefings/candidate-factory-traces.md`** (QUANTITY MODE,
Han directive: batch of screen-ready label bundles on the Ward grid).
Read that briefing in full first; this file is the resume state.
Previous session (`hunt-support-stats.md`, all 4 items) is COMPLETE,
pushed, awaiting mac-local review — its briefing stays in place.

## Who / where
Second RunPod box, repo `/workspace/temp_xc`, 32 CPU, no GPU, no CFS cap.
`/workspace/.agent_id` = runpod-b. Git identity set (Han); push:
`git push https://x-access-token:$(cat ../.tokens/gh_token)@github.com/chainik1125/temp_xc.git arxiv`.
`export ANTHROPIC_API_KEY=$(cat /workspace/.tokens/anthropic_key)`.
HF token at `/workspace/.tokens/hf_token`.

## The new task (inputs verified this session, work NOT started)

**≥ 3 shipped bundles (or honest triage kills) by Saturday morning PT;
ship INCREMENTALLY — one LOG line per bundle as it lands.** Five
candidates, priority order, all on the Ward/R1-Distill token grid
(runpod-d/e already hold the 17-layer caches; each bundle costs them
minutes): (1) self-correction marker intensity λ̂_sc — FREEZE the
marker list in the card before computing; kernel-only λ̂_hist primary;
marker tokens masked out of probe rows; (2) question-rate intensity
("?" sentence endings, same machinery + masking); (3) op-class
run-rates ×2 (verification-check + case-enumeration kernel rates from
proofops `op`; current-sentence tokens excluded); (4) verbosity LEVEL
(trailing mean sentence length; level primary — the hedging-LEVEL
lesson); (5) window redundancy rate (fraction of window tokens whose
bigram occurred earlier; triage hard — repetition was regime-1).

Per bundle: labels npz + balanced manifests + shuffled-window null +
triage stats JSON + `<name>/CARD_DRAFT.md`, aggregation-framed
(shuffle-immunity receipt). **Label-side triage is my kill authority:
label readable from current-token identity or position alone ⇒ FAIL,
one LOG line, do NOT ship (a free kill is a win).** Builders committed
before outputs; zero-API exact labels only; no reviewer/meeting quotes.

**Inputs verified:**
- Bundle format template = `labels/ward_lambda.npz` fields: in_span,
  is_bt, lam, lam_bin, lam_hist, man_cls/doc/pos, sent_idx, trace_idx,
  trace_split, valid, win_start. proofops.npz has `op` (5-class),
  time_in_run, sent_idx (+ manifests) for candidate 3.
- Sentence TEXTS (markers, "?", lengths): traces.json is absent
  locally but `labels/wardmap.ensure_traces()` re-ports it via
  `git show origin/aniket-ward-stage-b:...` — VERIFIED working. Use
  `wardmap.load_inputs()` / `broadcast()` for grid alignment.
- Kernel machinery: `labels/lib.py::lambda_for_sentences(b, intercept,
  coef_pos, kernel_w)`; committed backtracking kernel coefficients in
  `synthetic/backtracking/results/backtracking_mirror_stats.json`.
  DESIGN DECISION to freeze per card BEFORE computing: for NEW event
  streams reuse the committed kernel w_l (frozen, exact) vs a plain
  exponential — state the choice + why in each card.
- The screen stack ("problib") the bundles must fit: referenced in
  `task_hunt/forbidden_word/cache_and_screen.py` and
  `task_hunt/shuffle_receipt.py` — READ ONE POST-COMPACT to confirm
  the exact consumption format before building.
- `lib.delta_prev_ngram` for redundancy; `lib.tercile_bins`,
  `balanced_manifest` (pos ≥ 32), `trace_split` conventions as round 1.

## Last session's results (context for the factory)
- Variance receipts: TXC-pre T2→8 rise p=0.0093 exact; margin trend
  p=0.0046; cross-arch pre−tsae NOT bounded at n=3; seed top-up rec
  posted to runpod-d (3 seeds × 3 cells). Renderer: l0 legend +
  NOT-budget-matched flag mandatory (review note 3) — new bundles'
  eventual figures inherit this.
- Lessons that bind the factory: position-floor (kernel-only primary),
  anchor/ambient masking, levels-not-slopes (aggregation-framed),
  per-token-first triage as convention, no case-colliding filenames.
- Round-2 in flight elsewhere: runpod-d budget-matched TXC-post re-run;
  runpod-e froze its own hedging-LEVEL card (my draft reconciled in its
  §10 amendment). New upstream: `briefings/candidate-factory-broad.md`
  (parallel factory, not mine), `em-redo.md`, arch `tsae_delta.py`.
- Shared branch: 5 agents on arxiv; pull-rebase before EVERY push; cite
  commit SUBJECTS not SHAs. Rewrite this file before any compact.
