# DRAFT mini-card — equation-density intensity (OpenWebMath)

**Status: DRAFT (runpod, `briefings/candidate-factory-broad-2.md`,
ledger `../CANDIDATES.md` B6 — round 2).** Committed WITH the frozen
triage bars and the frozen math-span grammar below BEFORE
`../labels/build_eqdens.py` runs; the running agent freezes its own
screen card.

Data side: builder `../labels/build_eqdens.py` (logic
`../labels/eqdens_lib.py`, tests `tests/test_eqdens_labels.py`) →
`../labels/eqdens_openwebmath_{gpt2,gemma2,llama31}.npz` +
`../labels/eqdens_stats.json` + **the pinned corpus artifact
`../labels/eqdens_corpus.json.gz`** (`open-web-math/open-web-math`,
train split at pinned revision `fde8ef8d…`, license ODC-By 1.0 +
CommonCrawl ToU; the dataset is ~27M docs, so the sample is the FIRST
4,000 streamed examples in shard order — a stated convenience-sample
disclosure — filtered to 1,000–20,000 chars with ≥ 3 math spans under
the frozen grammar, then seeded-subsampled to 600 docs). The ≥ 3-span
floor kills the math-doc-vs-prose-doc identity route at pull time
(the B4 concern, disclosed): every doc in the sample has real math,
so terciles contrast within-register intensity. **Economics: a NEW
token stream (~1M tokens/tokenizer) — one caching pass per model,
minutes on an H100; no existing cache applies.**

## The candidate logic

Winner-family intensity: primary `mrate` = kernel-smoothed trailing
MATH-TOKEN rate over the PREVIOUS 64 tokens (half-life 16), current
token never in its own label, NaN warm-up below position 64 — "how
math-heavy has this stretch been", the regime-2 aggregation state of
prose–proof–equation alternation. **Stated deviation from the ledger
sketch ("previous sentences/lines"):** the format scan (300 streamed
docs, delimiter counts only — vetting inputs, superseded by the
builder stats) found median line length 16 chars with wildly
heterogeneous line granularity across docs, so a line-unit kernel
would make the clock doc-dependent; the token-unit kernel (novelty B2
geometry) is exact, uniform, and pins the clock INSIDE the ladder.
The IN-math state itself is bracket-family, recorded dead (axis a:
the model tracks math mode per-token, certainly converted) — shipped
as the disclosed regime-1 anchor face `in_math`, never the primary,
never manifested. **Masking rule: all math-span tokens (`in_math`,
delimiters inclusive) are masked from probe rows** — they read the
label ambiently.

## FROZEN math-span grammar

The compiled regex `eqdens_lib.MATH_RE` IS the grammar (leftmost
match, alternation order = precedence, unclosed delimiters match
nothing, spans include delimiters):

1. `\begin{ENV}…\end{ENV}`, ENV ∈ {equation, align, gather} (+ `*`);
2. `\[…\]` and `\(…\)` (may span lines);
3. `$$…$$` display (may span lines; `\$` opens nothing);
4. `$…$` inline (single line; escaped `\$` neither opens nor closes).

Scan receipts behind the freeze (300 docs): inline `$` in 62 % of
docs, `$$` in 26 %, `\begin{align}` 257 occurrences in 12 docs while
plain `\begin{equation}` scanned ZERO — hence the align/gather family
joins the briefing's list; `\[`/`\(` marginal but kept (they are
exact). Other environments (matrix, cases, …) outside `$`/`$$` are
NOT counted — a stated grammar limit, not a bug.

## Clock (axis c)

A token-level kernel, so the panel ladder spans it by construction:
kernel mass visible to a trailing window of length T = **0.17 (T=4),
0.31 (T=8), 0.53 (T=16), 0.80 (T=32), 1.00 (T=64)** — the
best-spanned clock in the broad factory (the sentence-kernel
candidates top out under-spanned; here T=64 closes the kernel
exactly).

## Label-side triage — FROZEN BARS (kill authority)

**Pinned convention (this and every future broad-factory card; review
qualification 4):** bars are read direction-agnostic —
max(AUC, 1−AUC) — on test-doc rows, top vs bottom class, on BOTH
all-eligible and shipped-manifest rows, **manifest rows operative**;
current-token type-mean AUC **≥ 0.65 ⇒ KILL**; position AUC
**≥ 0.65 ⇒ KILL**; 0.55–0.65 ships with disclosure. The named axis-b
risk here: math-notation vocabulary leaks topic (mathy stretches have
mathy prose around them — "theorem", "denote", variable names), and
that is exactly what the unigram bar decides. Manifests are
position-MATCHED from the start (equal class counts per log2 stratum,
pos ≥ 64 — the B3/B5 lesson applied before, not after).

## Predicted T-pattern + draft kill rule

Per-token poor on masked (non-math) rows; window − per-token gap
positive and GROWING through the full ladder T ∈ {4…64} (the kernel
closes at T=64 — unlike every sentence-kernel candidate, growth
should continue to the top rung); order-free (mean ≈ flatten) —
regime-2, shuffle-immunity as the mechanism receipt. The
within-doc-shuffle null faces `mrate_null`/`mrate_null_bin` (same
marginal math rate, arrangement destroyed, binned with the real
edges) are the frozen frequency receipt: a probe reading real
trailing density must NOT read the null's. KILL at screen if:
per-token reads `mrate_bin` within noise of the best window at every
T; or no window − per-token gap beyond 3 σ_null at any T; or no
T-growth anywhere in the ladder. `in_math` runs as the disclosed
anchor face only.
