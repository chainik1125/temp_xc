# Mini-card DRAFT — factory candidate B8: the sentence-length recency ladder (`slen`)

**Status: DRAFT (bundle-side; the screening agent freezes its own
CARD.md before any cell runs — commit-then-run applies to the screen,
not just the build).** Agent: runpod. Briefing:
`briefings/candidate-factory-broad-3.md` + mid-execution ADDENDUM §1.
Ledger: `../CANDIDATES.md` B8 (P6 absorbed as the `lev` face).
Builder: `../labels/build_slen.py` on frozen logic
`../labels/slen_lib.py` (tests: `tests/test_slen_labels.py`).
Artifacts: `slen4k_fineweb_<tok>.npz` (+ the 400-doc prefix variant
`slen400_fineweb_<tok>.npz` that rides the existing caches) +
`slen{4k,400}_stats.json`.

## 1. What this bundle is for (the one-sentence version)

The amended order finding says every window ADVANTAGE found so far is
order-free aggregation, with ONE measured crack: dialevel's
capacity-matched shuffle cost, best explained by runpod-e's
recency/distance-to-anchor hypothesis. This bundle turns that
hypothesis into **three pre-registered predictions on one substrate**
by shipping three faces of the SAME exact value stream that differ
only in temporal weighting.

## 2. Frozen label logic (slen_lib.py — nothing here is tunable later)

Value stream: x_i = ln(whitespace word count of sentence i), floor 1
word. Tokenizer-independent; labels differ across tokenizers only
through the sentence→token bridge. All faces use PREVIOUS sentences
only; NaN warm-up unified at sentence idx < 8 so the faces share
eligible rows; every token of sentence i inherits face values via the
committed bridge.

| face | definition | family |
|---|---|---|
| `lat` (PRIMARY) | x_{i−1} — the previous sentence's value | latch / pure recency |
| `lev` | kernel-weighted trailing mean, HL 2 / support 8 (punctint kernel) | level (P6) |
| `disp` | kernel-weighted trailing std, same kernel (ESS ≈ 5.1 of 8 lags — disclosed) | second moment |

No event tokens exist → nothing to mask beyond warm-up; the current
sentence never contributes to any face by construction. Binning:
train-edge `zero_split_bins` (expected: plain terciles — the stream is
not zero-inflated). Manifests: position-matched
(`stratified_balanced_manifest`, log2 strata, pos ≥ 32), cap 100k
rows/class.

## 3. Why the substrate is confound-clean where dialevel was not

dialevel's doc-length route was structural: the turn-count floor made
dialogue length track turn length (doc_mean_only_auc 0.983–0.986, the
program's loudest). Here document length (60–200 sentences) is fixed
by the PULL filter and not coupled to sentence length by
construction. The residual doc-identity route (verbose documents are
verbose throughout) is expected NONZERO — it is measured
(`doc_mean_only_auc` with 1,000-rep doc-level bootstrap CIs) and
triggers the within-document-contrast obligation, per the ratified
protocol; it is not a kill bar.

## 4. Triage bars (frozen, the broad-factory pair — manifest rows operative)

Direction-agnostic max(AUC, 1−AUC) on position-matched manifest rows:
current-token type-mean AUC ≥ 0.65 ⇒ KILL; position AUC ≥ 0.65 ⇒
KILL; 0.55–0.65 ships with disclosure. **Training corpus size is
quoted beside every unigram number** (the estimator finding: 400-doc
readings are lower bounds; the 4k artifact's 3,200-train-doc numbers
are operative where both exist).

## 5. Pre-registered predictions (score these at screen, per face)

- **P1 (per-token):** all three faces substantially converted —
  per-token AUC 0.60–0.80 (`lat` highest: the model plausibly carries
  "the last sentence was long" in its running state).
- **P2 (window):** window > per-token for `lat` once T spans the
  previous sentence (T ≥ 16 at ≈15–25 tokens/sentence); for
  `lev`/`disp` growing through the ladder and under-spanned at T32
  (kernel support ≈ 8 sentences ≈ 120–200 tokens — said plainly, the
  λ̂-winner property).
- **P3 (THE LADDER — the point of the bundle):** within-window
  shuffle sensitivity ordered **lat > lev > disp ≈ 0**:
  - `lat`: shuffle destroys ≥ half of whatever window gain exists (a
    latch needs the location of the nearest boundary);
  - `lev`: partial cost (nearest-dominated — the dialevel regime,
    HL 2 means ~71 % of kernel mass in the nearest 2 sentences);
  - `disp`: cost within 3σ of 0 (a dispersion is almost order-free).
- **P4 (readout class, the fineweb lesson):** any window advantage
  may live in a NONLINEAR readout (fineweb precedent: MLP +0.06…+0.13
  vs linear mean-pool ≈ +0.04); score both, never max-over-arms.
- **P5 (base ≈ distill not applicable; cross-model):** consistent
  sign across gpt2 / gemma-2-2b / llama-3.1-8B; per-model verdicts,
  stated majority rule, no pooling.

**Falsifiers, stated now:** if `lat` shows a real window gain with
shuffle cost ≈ 0, the "latch" is actually order-free ambient
statistics (e.g. short-sentence token density) and the recency
hypothesis LOSES its best broad-text instance — that is a finding,
recorded as such. If `lat` dies at triage on the unigram bar, the
recency face is token-stamped on this corpus and P7 (`qgap`) is the
next recency candidate up.

## 6. Screen economics

The 400-doc prefix variant aligns token-for-token with the existing
fineweb caches (prefix receipt vs `replag_fineweb_<tok>.npz` in the
builder — zero new caching). The 4k variant needs the ~3.6k-doc
caching pass already priced by the scale-up campaign (~39.2M new
tokens across 3 models). Sentence-unit kernel at the upper ladder for
`lev`/`disp`; `lat` needs only the nearest boundary.

## 7. Verdict appendix

*Build verdict (runpod, 2026-07-25):* **SHIPPED — no frozen bar fires
on either variant, any face, any tokenizer.** Operative numbers = 4k
artifact, manifest rows, **3,200 train docs** (400-doc variant quoted
beside them per the estimator convention, **320 train docs**); every
range is min–max across the three tokenizers; 1,000-rep doc-level
bootstrap CIs live in the stats JSONs.

| face | unigram (4k man.) | unigram (400 man.) | position (4k man.) | doc_mean_only (4k) |
|---|---|---|---|---|
| `lat` | 0.563–0.568 | 0.541–0.549 | 0.508–0.511 | 0.746–0.749 |
| `lev` | 0.588–0.592 | 0.558–0.565 | 0.514–0.516 | 0.879–0.882 |
| `disp` | 0.518–0.522 | 0.519–0.522 | 0.499–0.504 | 0.803–0.804 |

- **`disp` is near-blind at BOTH scales** (unigram ≈ 0.52, stable
  from 320 → 3,200 train docs) — the strongest axis-b posture in the
  broad factory, and the most independent face (corr −0.14/−0.19 with
  lat/lev; |corr| ≤ 0.08 vs `lam_q`, ≤ 0.16 vs `lam_list`).
- `lat`/`lev` sit in the 0.55–0.65 disclosure band at 4k; their
  400-doc readings are lower — the estimator finding replicating
  in-bundle, which is why both training sizes are quoted.
- Position clean on manifests (0.499–0.516). All-eligible-row
  position for `lev` is 0.629–0.633 — below bar but DISCLOSED:
  screens use manifest rows.
- doc_mean_only all below punctint q's 0.901; the within-document
  contrast obligation stands and is well-supplied: at ≥ 20 manifest
  rows/class the 4k test split holds **219 (lat) / 114 (lev) /
  156 (disp)** documents, and the cache-aligned 400 variant holds
  **71 (lat)** test documents (~9× the 8 punctint-list rested on).
- Prefix receipts PASS ×3 tokenizers: the 400 variant is
  token-IDENTICAL to the cached corpus (**zero new caching to screen
  it**); the 4k variant needs ~7.0–7.1M new tokens per model.
- corr(lat, lev) = **0.761** — expected (same stream; the ladder is
  the point). A screen treats the three faces as one bundle with one
  pre-registered prediction set (§ 5), not three discoveries.
- Binning came out plain terciles on every face (no zero-inflation),
  manifests at the 100k/class cap with supported ceilings ≥ 1.98M.
