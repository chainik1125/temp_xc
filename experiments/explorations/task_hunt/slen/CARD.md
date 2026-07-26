# FROZEN screen card — B8 sentence-length recency ladder (`slen`, 400-doc variant)

**Status: FROZEN at commit (commit-then-run; no screen cell has been
executed when this card is committed — the Modal container is pinned
to this commit).** Agent: **mac-b (executor)** — self-review hazard
named per `briefings/overnight-mac-modal.md`: predictions and
KEEP/KILL bars below are pre-registered BEFORE any cell runs, and
every verdict ships **PENDING TEAM REVIEW**. Briefing:
`briefings/overnight-mac-b.md` § 1. Bundle card: `CARD_DRAFT.md`
(build verdict SHIPPED, ratified by mac-local in the LOG 2026-07-25
factory-r3 review). Executor: `screen.py` (this directory, frozen in
the same commit).

## 1. Coverage — 2 models, stated up front

**gpt2 + llama31-8b (NousResearch mirror), both ungated.** gemma-2-2b
requires an HF token; none exists on the overnight venue
(`~/.tokens/hf_token` absent), so **gemma cells are NOT run tonight**.
They are pre-authorized to run LATER under this same frozen card if a
secret appears; until then every cross-model statement below reads
"on the 2 screened models, gemma pending" — no majority language over
3 models is licensed. A 2-model screen is reportable with coverage
stated (briefing § 1).

## 2. Substrate, rows, caches (all inherited, nothing tunable)

- Bundle `../labels/slen400_fineweb_<tok>.npz` (frozen `slen_lib`,
  builder-committed BEFORE its outputs at `e9e560af`; prefix receipts
  PASS ×3 — the 400-doc token grid is **token-IDENTICAL** to the
  replag fineweb caches, re-asserted at screen run time on 200
  chunks). Caches rebuilt in-container from the committed builders
  (`replag/build_labels.py` + `replag/cache_acts.py`) onto a Modal
  Volume; screen layers gpt2 **hs7**, llama31-8b **hs14** (replag
  card convention, unchanged).
- Manifests: builder's position-matched balanced `man_<face>_*`
  (plain terciles, cap 100k/class label-side). Screen eligibility,
  mapping, caps, seeds IDENTICAL to the novelty/punctint screens:
  `pos ≥ 64`, `pos % content ≥ 63` (every screened T ≤ 64 reads
  identical rows), builder `doc_split`, caps 4000 train / 1500 test
  per class (MATCH_SEED 1013 + crc32), MIN_ROWS 300 floor.
- Label-side triage (SHIPPED, quoted with training size per the
  estimator convention — **320 train docs** here; the 4k artifact's
  3,200-train-doc numbers are the operative label-side readings and
  are LOWER-bounded by these): unigram lat 0.541–0.549 / lev
  0.558–0.565 / disp 0.519–0.522; position 0.499–0.512 (manifest
  rows); `doc_mean_only_auc` lat 0.737–0.743 / lev **0.890–0.892** /
  disp 0.798–0.800 ⇒ the **within-document control is BINDING for
  `lev`** (protocol obligation) and is run for all three faces.
  corr(lat, lev) = 0.761 disclosed — one bundle, ONE prediction set,
  not three discoveries.

## 3. Clock bridge (stated before the run, as the punctint card did)

Measured 21.1–21.6 tokens/sentence on this corpus (punctint card § 3;
same 400 docs). T ∈ {4, 8, 16, 32, 64} spans ≈ 0.19 / 0.37 / 0.74 /
1.5 / 3.0 sentences. `lat` needs the PREVIOUS sentence: a window
anchored mid-sentence typically reaches it from T ≈ 32 (T16 only when
the anchor sits early in a short sentence). `lev`/`disp` kernel:
support 8 sentences (≈ 170 tokens), sentence-mass 0.31 / 0.53 / 0.80 /
1.00 within 1 / 2 / 4 / 8 sentences — **T64 reaches ≈ 3 sentences ≈
0.7 of kernel mass; the upper ladder is under-spanned at every
screened T and this is disclosed now**: a rising-but-unsaturated gap
is the predicted shape for `lev`/`disp`; a flat gap is a real negative
at this reach, recorded as reach-limited, not sold either way post
hoc. The flatten/shuffle/foreign T·d arms stop at T = 32 (memory,
punctint precedent: T64 flatten on llama-8b is 262k features); the
2d `actxmean` arm carries the ladder to T = 64.

## 4. Probe grid (frozen — the convention-of-record; see screen.py)

Per model × face, in this order: per-token linear + MLP(512) FIRST;
position floor on shipped rows; `actxmean` linear + MLP at
T ∈ {4,8,16,32,64} each beside its width-matched foreign null;
ORDER arms (flatten / context-shuffle / foreign-flatten, matched T·d
width) linear at T ∈ {4,8,16,32}, MLP triple at T ∈ {16,32};
permutation nulls at T16; within-doc binary arms (tok, actxmean ±
foreign at T ∈ {16,32,64}, flatten/shuffle pair at T ∈ {16,32}).
Probe stack frozen `conversion_depth.problib`; **no `win_mean`**
(anchor-dilution artifact); **NO max-over-arms** — matched probe
class everywhere, linear is canonical for scoring, MLP reported
(P4). Metrics: 3-class `acc_test` (chance 1/3); wd arms rank-AUC.

Named quantities (per face, per model, per probe class):
- `g_ax(T)` = actxmean − tok (order-free window gain)
- `width_ok(T)` ⇔ actxmean − actxmean_foreign ≥ **+0.02**
- `sc(T)` = win_linear − win_shuf_linear (order carriage, matched width)
- `wc(T)` = win_linear − win_foreign_linear (width-corrected flatten
  content)
- noise band: the T16 permutation nulls' |acc − ⅓|, quoted beside any
  "≈ 0" claim.

## 5. Pre-registered predictions (CARD_DRAFT § 5, made cell-precise)

- **P1 (conversion):** every face's tok_linear clears the position
  floor by ≥ +0.05 on both models, `lat` highest of the three.
- **P2 (window):** for `lat`, `g_ax(T) > 0` with `width_ok` from
  T = 32 up; for `lev`/`disp`, `g_ax` grows through the ladder and is
  still rising at T64 (under-span, § 3).
- **P3 (THE LADDER — the deliverable).** Scored at T ∈ {16, 32},
  linear arms, per model:
  - `lat`: `sc(T) ≥ 0.5 · wc(T)` where `wc(T) > 0` (order carries at
    least half the width-corrected window content), and
    `sc(T) ≥ +0.02`;
  - `lev`: partial cost — `0 < sc_lev(T) < sc_lat(T)`;
  - `disp`: `|sc_disp(T)| ≤ 0.02` (≈ 0 at screen noise, null band
    quoted beside);
  - **ladder order** `sc_lat > sc_lev > sc_disp` at both quoted Ts,
    on 2/2 screened models (gemma pending).
- **P4 (readout class):** MLP arms may exceed linear (fineweb
  precedent +0.06…+0.13); scored per class, never maxed, never
  substituted into P3.
- **P5 (cross-model):** consistent sign of `sc` and `g_ax` on 2/2;
  per-model paragraphs in the LOG, no pooling.

**Falsifiers, restated from the bundle card:** if `lat` shows a real
window gain (`g_ax` or `wc` > 0 with width_ok) but `sc ≈ 0`, the
"latch" is order-free ambient statistics and **the recency hypothesis
loses its best broad-text instance — a first-class finding, recorded
as such**, not a failed screen. If `lat`'s window arms never beat tok
at all, the recency face is per-token-converted on this corpus and P7
(`qgap`) is the next recency candidate up.

## 6. KEEP / KILL (frozen, per face; ladder verdict separate)

**KEEP face** iff on 2/2 screened models: some window arm beats its
matched-class tok cell by ≥ **+0.05** at some T **with its width null
cleared by ≥ 0.02**, AND the window arm clears the position floor by
≥ 0.05, AND — **binding for `lev`, informative for lat/disp** — the
within-doc arm shows a same-direction window gain (wd actxmean − wd
tok > 0) where wd rows are supported (≥ MIN_ROWS); if wd rows are
unsupported for `lev`, its KEEP is at most CONDITIONAL pending the 4k
variant.

**KILL face** if ANY of: (1) tok within 0.02 of every window arm at
every T on both models; (2) every apparent window gain fails its
width null; (3) for `lev` only — a window gain that VANISHES in the
within-doc arm (doc-identity route, the dialevel trap).

**LADDER verdict** (the bundle-level result, one LOG line of its
own): CONFIRMED / PARTIAL (state which clause failed) / INVERTED /
NOT-TESTABLE (no face has `wc > 0` anywhere — nothing for order to
carry). A KILL or an inverted ladder is a first-class result.

If no rule fires: **WEAK — no rule fires as written**, numbers, no
narrative upgrade.

## 7. Venue, economics, discipline

Modal A10G, container pinned to THIS commit; caches (~793k + ~778k
tokens × 3 layers, fp16) ≈ 23 GB on the Volume, forward passes
minutes; screen ≈ 1–2 h/model est. ≤ $10 total (cap $100, ledger
`briefings/MODAL_SPEND.md` read-before/append-after). Containers do
NOT push; results JSON repatriated and merged locally by mac-b
(canonical runner in-container, canonical merge outside). Runtime
adaptations pre-authorized, none affecting outputs: batch halving on
OOM in `cache_acts`; resume from partial `results/screen_<model>.json`.
Deliverable: `results/screen_{gpt2,llama31_8b}.json`, **three face
verdicts + one ladder verdict** in the LOG (`mac-b (executor)`,
PENDING TEAM REVIEW), RECEIPTS proposals for any quoted claim, no
leaderboard rows, figure only if a face KEEPs.
