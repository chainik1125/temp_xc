# Working state — agent `runpod`

**Last rewrite:** 2026-07-24 (night), after shipping candidate-factory
bundles B1–B5. **State: candidate-factory-broad ACCEPTANCE GATE
EXCEEDED — ledger + 5 candidates shipped with LOG lines; B6
(OpenWebMath equation density) is the one open stretch item.**
Briefing `briefings/candidate-factory-broad.md` stays until mac-local
review; do NOT redo B1–B5.

## Who / where
Remote CC on RunPod (Linux), repo root `/workspace/temp_xc`. **I am
`runpod` (original box — `/workspace/.agent_id` does NOT exist; do not
create it).** 32 CPU / 128 GB, NO GPU. Parallel agents runpod-b/-c/-d/-e:
their briefings are NOT mine (runpod-b's traces conventions govern my
bundle format). Shared-branch rules: `git pull --rebase origin arxiv`
before EVERY push (commit this STATUS first); LOG.md append-only —
conflicts resolved keep-both-upstream-first-mine-last (done 4× today;
runpod-b pushes frequently, expect races: if push rejects mid-flight,
just pull-rebase again). Tokens `/workspace/.tokens/{gh_token,
anthropic_key,hf_token}`. Push:
`git push https://x-access-token:$(cat ../.tokens/gh_token)@github.com/chainik1125/temp_xc.git arxiv`
(pushing via URL does NOT update the origin/arxiv tracking ref — `git
fetch origin` before reading `status -sb`). `sleep` blocked; python -u
for background runs.

## DONE this session (all pushed, LOG lines in task_hunt/LOG.md)

1. **`task_hunt/CANDIDATES.md` ledger** — 18 ideas, four vetting axes,
   6 BUILD / 6 PARK / 6 DEAD. The factory's quantity artifact; next
   hunter starts there.
2. **B1 interleave `tss`** — `interleave/CARD_DRAFT.md` promoted to
   screen-ready (tss PRIMARY 0.55 near-blind; source demoted anchor
   0.66; park lifted by briefing). Packaging only; runpod-b's data
   artifacts untouched. Screen needs ~330k-token caching pass/model
   (new token streams).
3. **B2 vocabulary-novelty (fineweb)** — `labels/build_novelty.py` +
   `novelty_lib.py` + 10 tests; `labels/novelty_fineweb_{gpt2,gemma2,
   llama31}.npz` + `novelty_stats.json`; `novelty/CARD_DRAFT.md` with
   FROZEN bars (≥0.65 unigram/position ⇒ kill) committed pre-run.
   **Triage PASS** (unigram 0.55–0.56; position ≈0.52
   direction-agnostic on the position-DETRENDED primary `nov_resid`).
   Drift receipt: resid autocorr beyond the 64-lag kernel support real
   0.13 vs null 0.02 (lag 64). **token_ids builder-ASSERTED identical
   to replag npz ⇒ existing fineweb GPU caches, ZERO new caching.**
4. **B3 list-density + B4 question-rate (fineweb)** — one builder
   `labels/build_punctint.py` + `punctint_lib.py` + 7 tests;
   `labels/punctint_fineweb_*.npz` + `punctint_stats.json`; cards
   `list_density/` + `qrate_fineweb/CARD_DRAFT.md`. 8-sentence-lag
   half-life-2 kernel (winner family), event-sentence tokens masked,
   zero_split fired (zero fracs 0.886/0.806). **B4 PASS clean**
   (unigram+position ≈0.52–0.53; disjoint from runpod-b's Ward
   `qrate`). **B3 ships WITH DISCLOSURE**: eligible-row position
   triage straddled the bar (0.639–0.653, fired on gpt2) → manifests
   rebuilt position-MATCHED (per-stratum class balance, the
   confidence-screen guard; amendment committed pre-outputs);
   manifest-row position 0.572–0.585 = inside the frozen 0.55–0.65
   disclosure band; screen must run a position-only floor probe.
   Process disclosure in LOG: one amendment commit briefly carried a
   red overclaiming test (guard ≈0.5); corrected next commit, no
   behavior change.

5. **B5 dialogue turn-length LEVEL (SHIPPED, stretch)** — NEW corpus:
   DailyDialog via pinned parquet mirror `OpenRL/daily_dialog`
   (canonical repo's loading script is dead under datasets 4.x);
   seeded 5,000×(≥8-turn) sample SHIPPED as
   `labels/dialevel_corpus.json.gz`; builder `build_dialevel.py` +
   `dialevel_lib.py` + 4 tests; `labels/dialevel_dailydialog_*.npz`.
   **No frozen bar fired (manifest unigram 0.566–0.569, position
   0.592–0.631) but the all-row position AUC is 0.93 via a DOC-LENGTH
   selection route — the card BINDS a screen precondition
   (within-dialogue contrasts or length matching + floor probes).**
   Economics: new stream ⇒ ~0.85M-token caching pass/model.

## NEXT (stretch, optional — gate already exceeded)
- **B6 OpenWebMath equation-density:** ledger B6 has the design; HF
  dataset `open-web-math/open-web-math` reachable (status 200; LARGE —
  use streaming to pull a pinned seeded sample, ship it like
  dialevel_corpus). Frozen LaTeX-delimiter grammar goes in the lib
  BEFORE the builder runs; in-math state is the recorded-dead
  bracket-family face — density-with-current-span-excluded is the only
  primary. Same discipline: lib + tests + builder + card with frozen
  bars committed pre-run; triage is kill authority; one LOG line.

## Reusable know-how (this box)
- **Bundle-builder pattern (3× proven today):** pure logic in
  `labels/<name>_lib.py` (+ tests) reusing `lib.py`
  (doc_split/balanced_manifest/tercile_bins/sentence_index_per_token),
  `novelty_lib` (kernel_weights/trailing_rate/type_mean_scores/
  tercile_auc/detrend), `punctint_lib` (zero_split_bins,
  stratified_balanced_manifest, pos_strata), `interleave_lib`
  (rank_auc). Builder asserts `token_ids == replag npz` for the
  zero-caching receipt (same pinned fineweb sample:
  `experiments/explorations/synthetic/expansion/data/fineweb_sample.json`,
  400 docs, " ".join(sentences), add_special_tokens=False).
- rank_auc is DIRECTIONAL — frozen bars read direction-agnostic
  max(AUC, 1−AUC); say so in cards.
- Stratified manifests kill the ACROSS-strata position route only;
  within-stratum gradient persists — always report manifest-row
  triage, never assume 0.5.
- HF: export HF_TOKEN + HUGGING_FACE_HUB_TOKEN from
  `/workspace/.tokens/hf_token`.
- Earlier session (approved): hunt-support-synthetic receipts
  (NO-MIRROR-DIP + FLAT; § 3b retraction), TXC-pro dissection, C7.
  Grid engine + skeptic notes in git history of this file.

## Repo state
Clean, in sync with origin/arxiv (post B3/B4 push + fetch). Suite 276
passed / 1 skipped. Leaderboard untouched today (label-side work only).
Spend meter untouched ($11.52/$25, no API calls today).
