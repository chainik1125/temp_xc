# Mini-card DRAFT — factory candidate B9: quoted-speech intensity on fiction (`quotedens`)

**Status: DRAFT (bundle-side; the screening agent freezes its own
CARD.md before any cell runs).** Agent: runpod. Briefing:
`briefings/candidate-factory-broad-3.md` (P4 lifted, corpus-shifted —
ledger `../CANDIDATES.md` B9). Builder:
`../labels/build_quotedens.py` on frozen logic
`../labels/quotedens_lib.py` (tests:
`tests/test_quotedens_labels.py`); corpus pinned by
`../labels/pull_pg19.py` → `pg19_corpus.json.gz` + receipt.
Artifacts: `quotedens_pg19_<tok>.npz` + `quotedens_stats.json`.

Corpus licence: PG19-class (Project Gutenberg, pre-1919 — public
domain in the US). Source: `emozilla/pg19` parquet mirror of DeepMind
PG19, revision pinned in the pull script (the original
`deepmind/pg19` is script-based and refused by modern `datasets`).

## 1. The candidate

Trailing quoted-speech intensity in narrative fiction: "how
dialogue-laden has this stretch been." Event = a sentence containing
any DOUBLE-quote-family character (frozen: ASCII `"`, curly `“ ”`,
low-9 `„`, guillemets `« »`; single quotes EXCLUDED as
apostrophe-inexact — books with single-quote dialogue conventions
read as low-event, disclosed via the per-book event-rate
distribution, not silently mislabeled). Primary label: the punctint
kernel (HL 2 / support 8 sentences) over PREVIOUS sentences only;
every token of sentence i inherits λ̂_i; tokens of EVENT sentences
are masked from probe rows (they display the event ambiently). The
current-sentence in-quote bit is the disclosed regime-1 anchor
(bracket-family, presumed converted — never the primary).

## 2. Why fiction (what the corpus shift buys)

- **Exactness restored**: edited prose balances its quotes — the
  precise failure that parked the fineweb version (P4).
- **The ratified doc-identity protocol favors it**: narrative ↔
  dialogue scene alternation is strong WITHIN-book variance — the
  binding within-document contrast gets its best substrate in the
  factory (the anti-dialevel).
- **Corpus independence**: nothing else in the program touches
  narrative fiction; doc length is pull-fixed (150-sentence spans,
  label-free pull), so the dialevel doc-length confound is designed
  out here too.

## 3. Triage bars (frozen — the broad-factory pair, manifest rows operative)

Direction-agnostic max(AUC, 1−AUC) on position-matched manifest rows:
current-token type-mean ≥ 0.65 ⇒ KILL; position ≥ 0.65 ⇒ KILL;
0.55–0.65 ships with disclosure. Training corpus size quoted beside
every unigram number (800 train books here; the estimator finding
says small-corpus readings are lower bounds). `doc_mean_only_auc`
reported with 1,000-rep doc-level bootstrap CIs; if it is loud, the
within-document contrast is a binding screen precondition, not a
kill.

## 4. Four-axis vet (from the ledger, scored at screen)

(a) conversion — the in-quote current bit is presumed converted (the
disclosed anchor); the trailing RATE has no obvious next-token
utility beyond register. (b) the loud risk — attribution register
(said-verbs, proper names near dialogue) is the unigram route; event
masking removes the self-stamp; triage decides. (c) sentence-unit
kernel at the upper ladder (~8 sentences ≈ 150–250 fiction tokens),
said plainly. (d) regime-2 LEVEL prediction: window > per-token,
T-growing, largely shuffle-immune (a rate, not a latch) — and the
corrected fineweb lesson applies: the advantage may live in a
NONLINEAR window readout (MLP vs linear mean-pool scored separately,
never max-over-arms).

**Falsifier:** the unigram bar firing would MEASURE the
attribution-register leak on fiction — the B6-style receipt for this
register, recorded as a free kill.

## 5. Screen economics

New corpus: one caching pass × 3 models (~3.5–4.5M tokens/model,
minutes on an H100 — exact counts in the stats JSON). No existing
cache covers any of it; cost stated wherever a screen is planned.

## 6. Verdict appendix

*(appended by the builder run.)*
