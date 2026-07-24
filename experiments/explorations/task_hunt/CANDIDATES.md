# CANDIDATES.md — broad-corpus candidate ledger (runpod, quantity mode)

**Mandate:** `briefings/candidate-factory-broad.md` (Han directive,
2026-07-24 — quantity over quality; maximize screen-ready case-study
candidates OUTSIDE the Ward corpus; runpod-b owns the Ward grid via
`candidate-factory-traces.md`, whose bundle format, label-side triage
kill authority, and masking discipline govern this ledger's BUILDs
identically). Written and committed BEFORE any builder, per the
ledger-first rule. The next hunter starts from this page: dead ideas
are deliverables too.

**The four vetting axes (round-1 lessons, applied per idea):**

- **(a) Conversion risk** — does the latent help next-token
  prediction? If yes, the model has likely linearized it per-token by
  mid-depth (the round-1 graveyard: repetition-lag converted at every
  scale from 124M up; forbidden-word onset ambient; emotional onset
  converted near the event). The screening question: *will the model
  decline to maintain this as a per-position state?*
- **(b) Label-side per-token proxy** — is the label readable from the
  current token's identity alone (unigram AUC), or from position (the
  position floor — full λ̂'s +0.487·pos ramp gave a 0.82 AUC floor;
  the kernel-only λ̂_hist face at 0.59 is why intensity primaries are
  position-term-free)? Triage on these two is the kill authority.
- **(c) Clock feasibility at panel T** — the screen ladder is
  T ∈ {2…32} (64 reachable but expensive). Fineweb sentences run
  ≈ 15–25 tokens (interleave stats: median 1–4-sentence block = 47
  tokens), so sentence-event kernels with ≈ 8-sentence support exceed
  T = 32 — acceptable (the λ̂ winner has the same property and still
  rises through the ladder) but must be said. Token-level event
  streams can pin their kernel inside the ladder by construction.
- **(d) Regime shape + predicted T-pattern** — regime-2
  (aggregation-framed, order-free evidence pooling) is ACCEPTED under
  the program decision: shuffle-immunity is a mechanism receipt, not a
  kill, and the win condition is window arch > per-token-decoded
  baseline. Levels aggregate; trailing slopes collapse to
  anchor − window-mean (the hedging lesson) — levels primary.

**Corpus economics.** Fineweb candidates build on the pinned sample
`experiments/explorations/synthetic/expansion/data/fineweb_sample.json`
(400 docs, 60–200 sentences each, seed-0 pull; the same sample every
committed fineweb artifact uses) and screen in minutes on runpod-e's
existing gpt2/gemma-2-2b/llama-8b caches. NEW-corpus candidates cost
one cheap caching pass (~minutes/model on an H100) and must ship the
tokenized corpus artifact or an exact re-pull script.

Feasibility rates quoted below are from an informal whitespace-word
scan of the pinned sample (vetting inputs only — superseded by each
builder's script-derived stats JSON).

---

## Verdict index

| # | idea | corpus | verdict |
|---|---|---|---|
| B1 | interleave `tss` (finish — labels exist) | fineweb-interleave | **BUILD** (ship 1st) |
| B2 | vocabulary-novelty trailing rate | fineweb | **BUILD** (ship 2nd) |
| B3 | list/enumeration density | fineweb | **BUILD** (ship 3rd) |
| B4 | question-rate intensity | fineweb | **BUILD** (with B3; triage decides ship/kill) |
| B5 | dialogue turn-length level / switch clock | new (DailyDialog-class) | **BUILD** (stretch) |
| B6 | equation-density intensity | new (OpenWebMath) | **BUILD** (stretch) |
| P1 | news chronology / date density | new (cc_news-class) | PARK |
| P2 | numeric-token density | fineweb | PARK |
| P3 | citation-marker density | new (arXiv-class) | PARK |
| P4 | quotation-sentence rate | fineweb | PARK |
| P5 | emphasis / all-caps rate | fineweb | PARK |
| P6 | trailing sentence-length level | fineweb | PARK |
| D1 | window redundancy / repetition rate | fineweb | DEAD |
| D2 | within-sentence position clock | any | DEAD |
| D3 | document/paragraph position | any | DEAD |
| D4 | NER / topic / sentiment densities | any | DEAD |
| D5 | code syntactic state (indent/comment/string) | code | DEAD |
| D6 | language / code-switch rate | fineweb | DEAD |

---

## BUILD

**B1 — Interleave `tss`, finished into a screen-ready bundle (ship
first; labels exist).** runpod-b's committed anti-conversion corpus
(`labels/interleave_fineweb_{gpt2,gemma2,llama31}.npz` +
`interleave_stats.json`, builders tested) already carries the vetting:
`tss` (tokens-since-source-switch) is the PRIMARY face — unigram
AUC ≈ 0.55, near-blind (axis b PASS); its generative signal is only
the gently-rising switch hazard 0.012 → 0.03 (axis a: the corpus is
built to minimize conversion pressure — that is the point of the
class); median block 47 tokens means T = 32 typically reaches the
previous switch while T = 4 rarely does (axis c: the ladder spans the
clock); prediction (axis d): `tss` window-readable and T-growing,
degraded on the shuffled-block null, while `source` identity (unigram
0.66 even lexically matched) plays as the disclosed regime-1 anchor —
per-token HIGH on `source` is the *expected* kill face, not a
surprise. The parked status is lifted for screening under quantity
mode. Remaining work is packaging, not building: promote
`interleave/CARD_DRAFT.md` to an operative draft (tss primary, source
demoted, predicted T-pattern + falsifier stated for the freezing
agent, GPU-economics note). **BUILD — cheapest shippable bundle in
the factory.**

**B2 — Vocabulary-novelty trailing rate on fineweb (ship second).**
Event stream: per-token novelty bit — first in-document occurrence of
the token type (exact, tokenizer-level, zero-API). Primary label: the
kernel-smoothed trailing novelty rate over PREVIOUS tokens only
(current token excluded — the anchor lesson applied at token level),
half-life pinned inside the panel ladder (axis c PASS by
construction: a token-level clock, no sentence bridge needed). The
label is dense (every token labeled) with healthy within-doc variance
(trailing-rate std ≈ 0.10–0.12 at supports 64–128 in the scan) — this
is a topic-drift intensity: novelty spikes when the document enters
new material. Axis-a risk, disclosed: the complement bit ("this token
seen before") is the replag graveyard — per-token repetition
detection is converted at 0.74–0.97 AUC everywhere. That kills any
*current-token* face, which is exactly why the primary excludes the
current token: the screen question becomes whether the trailing RATE
is maintained anywhere or must be aggregated from per-token bits — a
regime-2 shape (predicted: window ≥ per-token, growing to kernel
support, shuffle-immune; the mechanism receipt is shuffle-immunity
plus the within-doc-shuffle frequency null from the replag
convention). Axis-b risks to triage (kill authority): current-token
rarity correlates with neighborhood novelty (unigram → tercile AUC),
and novelty decays mechanically with document position (Heaps-law
trend — the position floor WILL be nontrivial; the builder must ship
position-detrended terciles as the primary classification face and
report the position-only floor). **BUILD.**

**B3 — List/enumeration density on fineweb (ship third).** Event
stream: sentence starts matching a frozen list-marker grammar
(bullets, `1.`/`a)`/`(iv)` — the exact regex frozen in the card
before computing). Primary: kernel-smoothed trailing event rate from
previous sentences only, mapped to tokens by the committed
sentence-index bridge; current-sentence tokens EXCLUDED from probe
rows and marker tokens masked (the proofops anchor discipline).
Axis a: predicting the next enumerator is generatively useful (the
model likely knows "in a list" ambiently — that face is the disclosed
anchor, not the primary); the trailing *density* of list structure is
the intensity face. Axis b: topic leak is the real risk — listy docs
have listy vocabulary; unigram → tercile AUC is the triage gate.
Axis c: sentence-event kernel with ~4–8-sentence support ≈ 60–160
tokens — upper ladder, said plainly. Scan feasibility: rate mean
0.060 but median 0 — events concentrate in ~37 % of docs (26 % of
docs ≥ 0.05), so manifests concentrate there; disclosed, split still
by doc. Prediction (axis d): regime-2 rise to T = 32, shuffle-immune;
regime-1 kill face = the in-list ambient bit. **BUILD.**

**B4 — Question-rate intensity on fineweb (same builder as B3; triage
decides).** Event = "?"-terminated sentences (exact); same kernel,
masking, and bridge as B3 — the marginal build cost is one more event
stream in the same builder, which is why it ships despite the thinner
prior: scan rate mean 0.038, median 0.016, and the variance is
between-doc-heavy (FAQ/forum pages vs prose), so the tercile label
risks reading as document identity through topic vocabulary (axis b).
The triage numbers get computed either way; if unigram → tercile AUC
comes back high, this dies as a free kill with a LOG line (the
quantity-mode win condition includes honest kills). Note the Ward-grid
question-rate candidate belongs to runpod-b (traces briefing item 2) —
this is the fineweb cousin, disjoint corpus, no collision.
**BUILD-with-gate.**

**B5 — Dialogue turn-length level + switch clock (stretch; new
corpus).** The grounded cousin of interleave `tss`: a two-speaker
dialogue corpus (DailyDialog-class; CPU-downloadable, HF reachable
from this box) rendered WITHOUT speaker tags (strict alternation makes
speaker identity ≈ parity — a position artifact; and tags would be
maskable event markers anyway). Labels: trailing mean turn length
(LEVEL primary — the hedging lesson; "am I in rapid-fire exchange or
long-form turns" is a regime-2 aggregation state) and
tokens-since-turn-boundary as the disclosed clock face (axis-a risk
HIGH on the latter: turn boundaries are newline-predictive and short
turns make it within-sentence-position-adjacent — secondary only).
Axis c: turns ≈ 10–15 tokens, so T = 32 spans ~3 turns and a
~5-turn kernel sits at the ladder top. Axis b triage: turn-length
level vs utterance-lexicon leak (short turns are lexically distinct —
"yes", "ok"). New-corpus rules: tokenized artifact + exact re-pull
script + caching-cost note in the bundle. **BUILD if the three cheap
bundles land first.**

**B6 — Equation-density intensity on OpenWebMath (stretch; new
corpus).** Event stream: math-mode spans by exact LaTeX delimiter
grammar (`$…$`, `$$`, `\[`, `\begin{equation}` — frozen list in the
card); primary = kernel-smoothed trailing math-token rate from
previous sentences/lines, current span excluded, math tokens masked
from probe rows. Axis a, stated plainly: IN-math state is
bracket-family (recorded dead — the model tracks math mode per-token,
certainly converted); the candidate is strictly the trailing
*density* (how math-heavy has this stretch been), with the in-math
bit as the disclosed regime-1 anchor. Axis b: math-notation
vocabulary leaks topic — triage gate. Axis c: line/sentence events,
kernel at upper ladder. Prior is good because event rates are HIGH
and within-doc structure is real (prose-proof-equation alternation).
HF dataset reachable (open-web-math, status 200); needs the caching
pass + tokenized artifact per the new-corpus rule. **BUILD if time
permits.**

## PARK

**P1 — News chronology / date density (cc_news-class).** Exact-by-
regex date/time expressions, intensity face plausible, but it is a
THIRD new corpus competing with B5/B6 for the same caching budget and
its regime story is weaker (date density is bursty-topical, likely
dominated by doc identity — the B4 concern with a worse prior).
PARK behind B5/B6; re-vet if both land and triage teaches us the
topic-leak gate is passable.

**P2 — Numeric-token density on fineweb.** Digit-token rate: scan
mean 0.022, median 0.013 — thin, and where it is dense it co-occurs
with lists/tables, so it largely duplicates B3's variance at lower
event rate. PARK as a redundant face; revisit only if B3 dies for a
reason that spares it.

**P3 — Citation-marker density (arXiv-class corpus).** `[12]` /
`(Author, 2020)` markers are exact by regex and the intensity face
(results-section vs related-work density) is a real regime-2 shape,
but it needs yet another corpus pull and OpenWebMath (B6) already
covers the "technical-document intensity" slot with denser events.
PARK behind B6.

**P4 — Quotation-sentence rate on fineweb.** In-quote STATE is
bracket-family (recorded dead-adjacent); the kernel-smoothed rate of
quoted sentences (narrative dialogue-ness) is the only viable face,
ambient risk is high (quote marks + said-verbs are strong current-
sentence stamps), and the fineweb web-text register makes the event
stream noisy (unbalanced quotes, markup residue — exactness is
strained). PARK.

**P5 — Emphasis / all-caps token rate on fineweb.** Exact and cheap
but sparse, lexically self-stamping (caps tokens ARE the label —
masking removes most signal), and no strong regime story. PARK.

**P6 — Trailing sentence-length level on fineweb.** The verbosity
LEVEL face is legitimate (levels aggregate), but runpod-b's traces
batch carries exactly this candidate on the Ward grid (item 4), and
the fineweb version adds corpus breadth rather than a new mechanism.
PARK to avoid near-duplicate spend; first candidate to lift if the
Ward verbosity bundle dies on a Ward-specific artifact.

## DEAD

**D1 — Window redundancy / repetition rate on fineweb.** The replag
kill is direct and at every screened scale (per-token repetition
detection 0.74–0.97 AUC, window − token ≤ 0 everywhere on this exact
corpus family): per-token bits are converted, and the trailing-RATE
face on the SAME corpus would ride on them; the order-carried residue
(lag-value) was measured too thin at 2B/8B to carry a screen. The
Ward-grid redundancy variant is runpod-b's item 5, on different text
with its own triage. DEAD here.

**D2 — Within-sentence position clock (tokens since sentence start).**
Syntactic position is about as generatively useful as a feature gets —
conversion is certain (axis a), and it is position by definition
(axis b). A clean example of the class the factory must not spend GPU
on. DEAD.

**D3 — Document / paragraph position.** Pure position floor (the
0.82-AUC lesson from full-λ̂'s ramp, in the limit). DEAD.

**D4 — NER density, topic-model drift, sentiment/affect intensity.**
Every version requires a learned labeler (NER model, topic model,
judge) — violates the zero-API/exact-labels rule that every bundle in
this factory obeys; and the affect face is the emotional-instability
graveyard besides. DEAD as a class, whatever the corpus.

**D5 — Code syntactic state (indentation / comment / string state).**
Recorded dead: bracket/indentation state-tracking is the round-1
graveyard's canonical conversion case. Comment-DENSITY intensity is
the one conceivable face, but it inherits the in-state ambient stamp
plus a topic leak (comment-heavy files are lexically distinct), and
code corpora add a caching pass — the prior does not justify the
spend. DEAD (revisit only with a receipt-level reason).

**D6 — Language / code-switch rate.** The pinned fineweb sample is
English by construction — no events. A multilingual pull would make
this a B5-class new-corpus candidate (script-exact labels, real
switch clock), but interleave (B1) already occupies the switch-clock
slot with a controlled corpus. DEAD on this corpus; the idea itself
transfers to B1's family.

---

*Ledger committed before any builder (quantity-mode rule). BUILDs
ship in the B1 → B2 → B3/B4 order, B5/B6 stretch; every bundle or
triage kill gets one LOG line as it lands. Bundle format, triage
stats, masking rules: per `briefings/candidate-factory-traces.md` and
the `labels/README.md` alignment contract.*
