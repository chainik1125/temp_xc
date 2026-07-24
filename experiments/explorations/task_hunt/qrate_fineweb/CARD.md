# FROZEN card — punctuation-intensity faces (fineweb): question-rate + list-density

**Status: FROZEN at commit (commit-then-run; no screen cell for either
face has been executed when this card is committed).** Agent:
runpod-e. Briefing: `briefings/task-hunt-r2-e.md` § 3 (quantity mode;
queue positions **5** = question-rate and **7** = list-density).

**One card, two faces, one screen pass — stated as a deliberate
choice.** Both faces live in the SAME artifact
(`../labels/punctint_fineweb_<tok>.npz`, builder
`../labels/build_punctint.py`) and read the same activation caches, so
screening them together costs one pass instead of two and keeps their
rows, probes and nulls literally identical. They keep **separate
verdicts** in the LOG, because their review status differs (below).
Frozen from runpod's drafts `CARD_DRAFT.md` (this directory, B4 face
`lam_q`) and `../list_density/CARD_DRAFT.md` (B3 face `lam_list`).

## 1. Review status — the two faces are NOT equal, and this binds

- **question-rate (`q` face): ships clean.** Reviewed label-side
  triage on shipped manifest rows, direction-agnostic: unigram
  type-mean AUC 0.520 / 0.533 / 0.521 and position AUC 0.528 / 0.522 /
  0.529 (gpt2 / gemma2 / llama31) — both under the 0.55 disclosure
  band.
- **list-density (`list` face): ships CONDITIONALLY** (mac-local
  review, binding qualification 1). Its frozen 0.65 position bar FIRED
  on all-eligible rows (position AUC 0.348 / 0.361 / 0.353 raw =
  **0.65–0.65** direction-agnostic, at the bar); the ship rests on
  position-matched manifests, where it falls to 0.415 / 0.428 / 0.424
  raw = 0.57–0.58 direction-agnostic. **Any screen MUST run the
  position-only floor probe on the shipped manifests and report the
  within-stratum residual; a window-vs-per-token gap without that
  probe is uninterpretable.** This card never quotes the list face as
  "passed triage" — it is **"passed after position matching, with
  disclosure"**, and its LOG verdict will say exactly that.

The position floor is run on the shipped rows of **both** faces (it is
mandatory for `list` and good practice for `q`).

## 2. Zero-new-caching (verified, same as the novelty screen)

`token_ids` byte-identical to `../labels/replag_fineweb_<tok>.npz`
(builder-asserted, `punctint_stats.json.token_ids_match_replag: true`,
re-verified here), and my windowed caches reproduce the flat stream
exactly (novelty `CARD.md` § 2 — same check, same caches, same three
models). Screen layers: gpt2 hs7, gemma2-2b hs14, llama31-8b hs14.

## 3. Clock bridge — measured, and the honest limit of this ladder

Both faces use a **sentence** kernel (support 8 sentences, half-life
2; mass within 1 / 2 / 4 / 8 sentences = 0.31 / 0.53 / 0.80 / 1.00).
Measured on this corpus: **21.1–21.6 tokens per sentence**. So a
token window of length T spans:

| T (tokens) | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| ≈ sentences | 0.19 | 0.37 | 0.74 | 1.48 | 2.96 |
| ≈ kernel mass | ~0.06 | ~0.12 | ~0.23 | ~0.42 | ~0.72 |

**T = 64 is the largest window my 128-token cache rows admit under
uniform eligibility, and it reaches only ~72 % of the kernel's
support.** This ladder therefore sits at the BOTTOM-to-middle of the
label's timescale — the same reach limit the confidence card recorded,
and it is stated here BEFORE the run so a flat result cannot later be
sold as "the window can't reach it" post hoc. A rising-but-unsaturated
gap is the predicted shape; a flat gap is a real negative at this
reach, recorded as reach-limited.

## 4. Rows (frozen)

Manifests `man_q_*` / `man_list_*` (builder-balanced ~20k/class,
already position-matched — the `list` face's ship depends on that).
Mapping and eligibility exactly as in the novelty screen: flat
`(doc, pos)` → `(cache_row, cache_pos)` by `chunk = pos // content`,
uniform eligibility `pos ≥ 64` **and** `pos % content ≥ 63` so every
screened T ≤ 64 reads IDENTICAL rows inside one cache row. Split by
the builder's `doc_split`; caps 4000 train / 1500 test per class,
seeded subsample (MATCH_SEED 1013 + crc32); **MIN_ROWS 300** floor.

Measured yield (gpt2, eligible before caps): q face 23,837 train /
6,521 test (min class 1,924); list face 25,964 / 4,512 (min class
1,179). Both clear the floor with margin; the caps bind, not the
floor.

**Masking travels with the faces:** question-sentence tokens are
excluded from the `q` face's manifests and list-sentence tokens from
the `list` face's (the builder's ambient-anchor rule, `is_q` / `is_list`
being the ambient companions). This screen does not re-derive the
masks; it inherits them.

## 5. Probe grid (frozen; same as the novelty screen so the two
## bundles are directly comparable)

Per model, per face: per-token linear + MLP(512) **first**
(per-token-first triage); then `T ∈ {4, 8, 16, 32}` window linear,
window-MEAN linear, context-shuffled linear (anchor fixed, seeded);
window-MEAN additionally at **T = 64**; window + shuffled MLP at
`T ∈ {16, 32}`; permutation nulls (NULL_SEED 99) at T = 16;
**position-only floor** on the shipped rows (in-chunk position + doc
position). Flatten arms stop at T = 32 for the stated memory reason
(a T = 64 flatten on llama-8b is 262,144 features).

**Ambient anchor (this card's addition, and the reason it is here):**
the companion binary `is_q` / `is_list` is screened per-token and at
T = 16 window-MEAN. **Its rows come from the FULL eligible token pool,
not from the face manifest** — the builder masks event-sentence tokens
out of the very face they anchor, so `is_q` is identically 0 inside
`man_q` and an anchor drawn there would be single-class (verified
empirically before freezing: that draw returns zero rows). The anchor
pool uses the same eligibility, caps, seeding and doc split, balanced
on the anchor bit (8000 train / 3000 test). Candidate 2's lesson in this program was that a
shuffle gap growing with T is a *generic* property of wider windows,
and only an anchor-differenced contrast carries a claim. The anchor
face is ambient by construction, so it calibrates how much of any
window gap is "wider windows help every label here".

Metric: `acc_test` (3-class, chance 1/3) + `per_class`; the anchor
face is binary (rank-AUC + balacc, `class_weight=True`).

## 6. Frozen predictions (scored either way)

- **Q1.** Both faces are per-token-readable well above their position
  floors (punctuation intensity is lexically stamped and the model
  keeps a running estimate) — i.e. I predict **conversion**, the
  novelty-screen shape, not a per-token-blind latent.
- **Q2.** window − per-token gap is positive but **small (< +0.05)**
  and rises with T without saturating over {4…64}, tracking the
  measured kernel-mass column of § 3.
- **Q3 (regime-2).** MEAN ≈ or > flatten (negative `g_order`) and the
  context shuffle is immune — order-free pooling, as in novelty.
- **Q4.** The `q` face's gap exceeds the `list` face's gap at matched
  T, because list-density is the more position-confounded face and
  loses more to the position matching.
- **Q5 (the anchor differential).** The ambient anchor (`is_q` /
  `is_list`) shows a window gap too; the intensity face's gap minus
  the anchor's gap is **smaller than the raw intensity gap** — i.e.
  part of any apparent window advantage is generic to wider windows on
  this corpus.

## 7. KEEP / KILL (frozen, per face independently)

**KEEP** iff, on ≥ 2 of 3 models: window − per-token ≥ **+0.05** acc
at some T, the gap grows over T ∈ {4…64} (last ≥ first + 0.02), the
window advantage clears the position floor by ≥ 0.05, AND the
anchor-differenced contrast (face gap − anchor gap at matched T) stays
≥ +0.02. Shuffle immunity classifies it regime-2 and is NOT a kill.

**KILL** if ANY of: (1) per-token within 0.02 of the best window at
every T on the majority of models; (2) no gap beyond 3 σ_null at any
T; (3) the gap does not grow anywhere over the ladder; (4) the
anchor-differenced contrast is ≤ 0 at every T (the gap is generic to
window width, candidate 2's trap); (5) the window advantage fails to
clear the position floor — for the `list` face this is the review's
binding qualification and a floor failure kills that face outright.

**If no rule fires** (gap real but under +0.05, or growth without
magnitude), the verdict is recorded as **WEAK — no rule fires as
written**, with the numbers, not upgraded by narrative. That outcome
is explicitly allowed here because it is what the novelty screen is
producing on the same corpus.

## 8. Deliverable

`results/screen_<model>.json` (both faces in one file,
incremental/resumable), **two** LOG verdicts (one per face, the list
face carrying its conditional-ship phrasing), and a figure only if a
face KEEPs. No leaderboard rows.
