# Working state — agent `runpod`

**Last rewrite:** 2026-07-24 (late night), immediately after receiving
`briefings/candidate-factory-broad-2.md` (pre-compact handoff).
**State: ROUND-2 FACTORY TASK ASSIGNED, ZERO WORK DONE ON IT** (per
instruction: work happens post-compact). Round-1 factory batch is
DONE + **APPROVED** (mac-local "REVIEW: candidate factories" entry in
`task_hunt/LOG.md` — every number artifact-verified, token_ids≡replag
re-verified, red-test disclosure reproduced; screen queue opened,
GPU pods consuming it). Do NOT redo B1–B5.

## Who / where
Remote CC on RunPod (Linux), repo root `/workspace/temp_xc`. **I am
`runpod` (no `/workspace/.agent_id` — do not create it).** 32 CPU /
128 GB, NO GPU. Shared-branch rules: commit STATUS first, `git pull
--rebase origin arxiv` before EVERY push; LOG.md append-only
(conflicts: keep both, upstream first, mine last — resolved 4× last
session; runpod-b races are normal, just re-pull-rebase). Tokens
`/workspace/.tokens/{gh_token,anthropic_key,hf_token}` (export
HF_TOKEN + HUGGING_FACE_HUB_TOKEN). Push:
`git push https://x-access-token:$(cat ../.tokens/gh_token)@github.com/chainik1125/temp_xc.git arxiv`
(URL-push does not update the tracking ref — `git fetch origin`
before reading `status -sb`). `sleep` blocked; python -u background.

## ACTIVE TASK (not started): briefings/candidate-factory-broad-2.md
**Round 2, quantity mode continues. Results by Saturday evening PT
(Sunday 10:00 PT check-in).** All round-1 disciplines unchanged:
builders committed before outputs; frozen mini-cards with triage
bars; label-side triage = kill authority; zero-API exact labels; LOG
line per bundle/kill; no reviewer/meeting quotes in tracked files.

**Read order post-compact:** (1) the briefing itself; (2) LOG tail —
the factory REVIEW entry (its 5 binding qualifications) + any new
screen verdicts from runpod-d/e; (3) `CANDIDATES.md` (my ledger — I
append to it); (4) `docs/papers/refusal.md` §§ 2.3, 3, 5.2, App. D.1
+ App. J (the D7 receipts).

**Deliverable 1 — ledger updates FIRST (append to CANDIDATES.md):**
- **D7 refusal-as-posed: DEAD** (vetted by mac-local; record with
  receipts from `docs/papers/refusal.md`, Arditi et al.): a
  difference-in-means direction extracted at SINGLE (position, layer)
  pairs is causally sufficient both ways (ablate ⇒ no refusal, add ⇒
  refusal) across 13 chat models to 72B; § 5.2 shows attention heads
  performing the window→position deposit — conversion IS the measured
  mechanism; App. J: direction present even in base models. Axis b:
  harmful-topic vocabulary = massive unigram leak, refusal text
  self-stamping. Axis c: prompt-level rollout boolean = the AVOID
  class. Economics: needs chat model + instruction corpus (no cache
  applies) + judge labels. Add to verdict index table too.
- **B7 refusal/deflection-marker intensity on multi-turn chat
  (WildChat-class): BUILD-if-time, BEHIND B6.** Vet properly on the
  four axes before building. Events = assistant turns matching a
  FROZEN substring list seeded from the paper's § D.1 set — NOTE: the
  list is a figure image in our summary; pull the concrete strings
  from the paper's public code repo
  (github.com/andyrdt/refusal_direction) and freeze BEFORE counting
  anything. Label = λ̂ over PREVIOUS turns (sc_lambda/dialevel kernel
  precedent), marker-turn tokens masked. Corpus = real multi-turn
  chat with RECURRING refusals (WildChat-class, CPU-downloadable),
  pinned artifact per the dialevel precedent (transcripts run through
  the three cached base models — one caching pass each, note cost).
  Stated risks: unigram topic leak (triage decides); event rate may
  be thin — MEASURE FIRST, if < ~2 % of turns kill in the ledger for
  free.
- **Verdict hygiene (standing):** as d/e post screen verdicts in the
  LOG, append one-line outcomes to the ledger's verdict index; re-vet
  PARKs whose reasons a verdict touches (P2 lifts if punctint-list
  dies specifically on position; P6 lifts if Ward verbosity dies on a
  Ward-specific artifact).

**Deliverable 2 — B6 OpenWebMath equation-density BUILD** (per ledger
B6 + briefing § 2, unchanged): math-mode spans by exact LaTeX
delimiter grammar (`$…$`, `$$`, `\[`, `\begin{equation}` — FROZEN in
lib/card before building); primary = kernel-smoothed trailing
math-token rate from previous sentences/lines, current span excluded,
math tokens masked from probe rows; in-math bit = disclosed regime-1
anchor (bracket-family, recorded dead — never the primary). Axis-b
triage: math-notation vocabulary topic leak. New-corpus rules:
`open-web-math/open-web-math` is LARGE — pull a seeded STREAMING
sample, ship pinned corpus artifact like `dialevel_corpus.json.gz` +
caching-cost note (~minutes/model on H100 for a new stream). Standard
bundle format + frozen bars + stats.

**Deliverable 3 (stretch): build B7 only if B6 shipped or honestly
died.**

**Acceptance gate:** ledger updated (D7 + B7 + verdict lines); B6
shipped or triage-killed with receipt; LOG line per item; STATUS
rewritten; stop for review (briefing stays).

**Binding context from the round-1 review (carry into new cards):**
- Qualification 4 (bar mismatch): traces froze tok 0.65/pos 0.70,
  broad froze 0.65/0.65 — **any future factory round pins ONE
  convention first**. B6/B7 cards must PIN the bar explicitly up
  front; I'll pin the broad convention: 0.65/0.65, direction-agnostic
  max(AUC, 1−AUC), manifest rows operative, 0.55–0.65 =
  ship-with-disclosure.
- Qualification 1: punctint-list is CONDITIONAL (never say "passed
  triage" bare); its death-on-position would lift P2 (verdict
  hygiene).
- Qualification 2: dialevel precondition binding; CC BY-NC-SA note
  travels with graduating figures.
- Screen queue order (context): sc_lambda → oprate rate_case → qrate
  (Ward) → novelty nov_resid → punctint qrate → vslope → punctint
  list (cond.) → interleave tss → dialevel.

## Round-1 factory (DONE, APPROVED — do not redo)
Ledger `CANDIDATES.md` (18 ideas) + B1 interleave packaging + B2
novelty (PASS; drift receipt; token_ids≡replag zero-caching) + B3
list-density (ships-with-disclosure, position-matched manifests) +
B4 qrate-fineweb (clean PASS) + B5 dialevel (ships; 0.93 doc-length
route named, screen precondition bound; pinned corpus artifact).
Earlier same day (also approved): hunt-support-synthetic receipts
(NO-MIRROR-DIP + FLAT ⇒ § 3b retraction).

## Reusable know-how (this box)
- **Bundle-builder pattern (4× proven):** `labels/<name>_lib.py` pure
  logic + tests + builder + card with FROZEN bars committed pre-run →
  run → verdict appendix (pure insertion) → outputs commit → LOG line
  → push. Helpers: `lib.py` (doc_split/balanced_manifest/tercile_bins/
  sentence_index_per_token), `novelty_lib` (kernel_weights/
  trailing_rate/type_mean_scores/tercile_auc/detrend/
  pooled_doc_autocorr), `punctint_lib` (zero_split_bins,
  stratified_balanced_manifest, pos_strata,
  token_labels_from_sentences), `interleave_lib` (rank_auc),
  `dialevel_lib` (render/boundary_flags precedent for new corpora).
- rank_auc is DIRECTIONAL — bars read direction-agnostic; say so.
- Stratified manifests kill the ACROSS-strata position route only;
  report manifest-row triage, never assume 0.5.
- New-corpus precedent (dialevel): pinned mirror + revision + seeded
  sample shipped as `<name>_corpus.json.gz`; builder doubles as exact
  re-pull script; license noted.
- Fineweb bundles: tokenize " ".join(sentences) of
  `experiments/explorations/synthetic/expansion/data/fineweb_sample.json`
  with add_special_tokens=False and ASSERT token_ids == replag npz ⇒
  zero new caching. (Does NOT apply to B6/B7 — new corpora.)

## Repo state
Clean, in sync with origin/arxiv after pulling the round-2 briefings
(+ refusal.md paper summary, probe-adequacy for runpod-b, r2-d/e
updates, runpod-e's Stage-2-negative + probe-capacity entries — not
mine). Nothing mid-flight. Leaderboard untouched by me. Spend
$11.52/$25. **Next action post-compact: execute the read order above,
then append D7 + B7 to CANDIDATES.md (ledger first, before any B6
code).**
