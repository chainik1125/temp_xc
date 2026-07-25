# Working state — agent `runpod`

**Last rewrite:** 2026-07-25 (candidate-factory-broad-3 COMPLETE).
**State: round-3 factory work DONE — awaiting mac-local review.**
`briefings/candidate-factory-broad-3.md` (+ its mid-execution
ADDENDUM) is fully executed; per its gate the briefing STAYS until
review. Repo clean, in sync with `origin/arxiv`, nothing mid-flight,
suite **314 passed** (`--ignore tests/test_v2_code_version.py`).

## Who / where (unchanged box facts)
Remote CC on RunPod (Linux), repo root `/workspace/temp_xc`. **I am
`runpod` (no `/workspace/.agent_id` — do not create it).** 32 CPU /
128 GB, **NO GPU, no activation caches** (only tokenizers in
`/workspace/hf_cache`). Shared-branch rules: commit STATUS first,
`git pull --rebase origin arxiv` before EVERY push; LOG.md
append-only (conflicts: keep both, upstream first, mine last — the
python strip-markers recipe, used once this round); push
`git push https://x-access-token:$(cat ../.tokens/gh_token)@github.com/chainik1125/temp_xc.git arxiv`
(URL-push does not move the tracking ref — `git fetch origin` before
reading `status -sb`). Tokens in `/workspace/.tokens/` (export both
HF_TOKEN and HUGGING_FACE_HUB_TOKEN). `sleep` blocked (`until …;
done`); background python needs `-u`; never trust `pytest | tail`
exit status.

## What round 3 shipped (all pushed; receipts in LOG + cards)

1. **Ledger re-vet** (`task_hunt/CANDIDATES.md`) under the AMENDED
   order finding (two buckets per the addendum: sequence-order stays
   demoted; RECENCY/distance-to-anchor is the best-motivated new
   family) + the estimator finding (small-corpus leak readings are
   lower bounds — they harden kills, never rescue). P1/P2/P3/P5 →
   DEAD, P4 → lifted corpus-shifted (B9), P6 → absorbed into B8; new
   PARKs P7 (`qgap`), P8 (connectives), P9 (gap-regularity — the
   sequence-order bucket, lift only on a measured order-sensitive
   advantage). Index rows B1/B2/B5 corrected to post-withdrawal
   state; screen-outcomes block caught up.
2. **Panel assist** (addendum item 3, non-blocking, shipped EARLY
   because runpod-e's 12h queue had started):
   `labels/build_depth_rowsets.py` → `punctint{,4k}_q_wdrows_<tok>.npz`
   + `oprate_case_tracestats.json`. Statistics-not-predemeaned
   contract stated in the stats JSON. Census cross-check: 4k gpt2
   test docs at ≥20/class = 117, exactly reproducing
   `contrast_depth.json`. Finding worth eyes: ≥50/class is THINNER
   at 4k than at 400 (cap spreads over 10× docs).
3. **B8 `slen` SHIPPED** — the recency ladder: one stream
   (x = ln sentence word count), faces `lat` (prev-sentence latch,
   PRIMARY — the addendum's recency family) / `lev` (P6) / `disp`
   (first second-moment face). Card §5 pre-registers the
   within-window shuffle ladder **lat > lev > disp ≈ 0** — the
   testable form of runpod-e's recency hypothesis on a substrate
   with dialevel's doc-length confound designed out. Bars clean at
   BOTH scales; **`disp` unigram 0.518–0.522 and SCALE-STABLE**
   (320 → 3,200 train docs); lat/lev in the disclosure band with the
   estimator finding replicating in-bundle; doc-means 0.746/0.803/
   0.881 all < punctint q's 0.901; within-doc census 219/114/156 4k
   test docs at ≥20 rows/class. **The `slen400` variant is
   token-IDENTICAL to the existing fineweb caches (prefix receipts
   PASS ×3) — it can screen with ZERO new caching.** 4k needs
   ~7.0–7.1M new tokens/model.
4. **B9 `quotedens` SHIPPED** — P4 corpus-shifted to PG19 fiction
   (`emozilla/pg19` @ `c021754c`, label-free pull, 1,000 books,
   receipt committed). Bars clean at 800 train books (unigram
   0.588–0.600 — real but sub-bar attribution-register leak, ships
   with disclosure + lower-bound caveat); doc-mean 0.890–0.896 ⇒
   within-BOOK contrast BINDING and **deepest-supplied in the
   factory: 69–70 test books at ≥50 rows/class** (punctint grids
   hold 5–7). Caching cost ~5.3M tokens all-new, stated.

## Open threads / not mine
- **Per-token-attenuation escalation: STRUCK for me** (addendum §2)
  — needs caches/GPU; routed to a cache-holding box after the panels.
- Stage-2 panels run on both H100s (`stage2-oprate` d,
  `stage2-fineweb` e — e CLAIMED and queue started); my only
  contribution is the shipped assist (non-blocking).
- Screen-outcomes block hygiene continues as panels post results.

## Untracked by design (do not "clean up", do not commit)
- `labels/eqdens_openwebmath_*.npz` — killed B6 bundle, regenerable.
- `labels/novelty4k_fineweb_*.npz` — ~140 MB each, regenerable
  exactly (stats ARE committed). Everything round 3 produced IS
  committed (slen npz 3–20 MB, quotedens npz ~3.8 MB, pg19 corpus
  3.1 MB).

## Program state (post-compact cheat sheet)
Five Ward Stage-1 KEEPs = one phenomenon (converted trailing-history
rates + order-free aggregation). **Order finding AMENDED** (wording
of record): *every window ADVANTAGE found is order-free aggregation,
measured on Ward; no order-sensitive window advantage found* —
dialevel's shuffle cost is the recorded counterexample;
recency/distance-to-anchor is the live hypothesis (B8 is its test
vehicle). tss + novelty = KEEP-PENDING-REVIEW (withdrawals);
"best window" scoring retired program-wide; `doc_mean_only_auc` is a
disclosure statistic that triggers a control, never a kill bar;
cards quote training size beside unigram bars. Deadline 07-27;
nothing from round 3 reaches a panel before it — by design.

**Next action:** none until mac-local reviews round 3. On resume:
`git pull --rebase`, check `briefings/` for a `for: runpod` file,
then this file. Spend $11.52/$25 (zero API this round — all local
CPU).
