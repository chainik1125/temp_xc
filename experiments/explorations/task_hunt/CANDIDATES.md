# CANDIDATES.md — broad-corpus candidate ledger (runpod, quantity mode)

**Mandate:** `briefings/candidate-factory-broad.md` (Han directive,
2026-07-24 — quantity over quality; maximize screen-ready case-study
candidates OUTSIDE the Ward corpus; runpod-b owns the Ward grid via
`candidate-factory-traces.md`, whose bundle format, label-side triage
kill authority, and masking discipline govern this ledger's BUILDs
identically). Written and committed BEFORE any builder, per the
ledger-first rule. The next hunter starts from this page: dead ideas
are deliverables too.

**Round 2 (2026-07-24, `briefings/candidate-factory-broad-2.md`):**
same mandate continues after the factory review. D7 + B7 appended
below (a Han-proposed refusal idea, vetted: the as-posed version dies
on literature receipts, the recurrence port survives to
BUILD-if-time); the verdict index stays live as the GPU pods post
screen outcomes.

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
| B2 | vocabulary-novelty trailing rate | fineweb | BUILT → screen **NEGATIVE** (2026-07-24) |
| B3 | list/enumeration density | fineweb | BUILT → screen **WEAK KEEP** (conditional, disclosed) |
| B4 | question-rate intensity | fineweb | BUILT → screen **KEEP** (the hunt's first) |
| B5 | dialogue turn-length level / switch clock | new (DailyDialog-class) | **BUILD** (stretch) |
| B6 | equation-density intensity | new (OpenWebMath) | **KILLED at triage** (2026-07-24 r2: manifest unigram bar fired, gpt2 0.653 — free kill) |
| B7 | refusal/deflection-marker intensity λ̂ | new (WildChat-1M) | BUILT → **SHIPPED** (2026-07-24 r2: pre-gate 0.147≫0.02; bars clean; conv-identity 0.967 disclosed, within-conv contrast binding) |
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
| D7 | refusal-as-posed (single-direction refusal state) | chat + instruction corpora | DEAD (lit receipts) |

**Screen outcomes (live; append one line per verdict as d/e post them
in the LOG).** Queue opened 2026-07-24 by the factory review, order:
sc_lambda → oprate rate_case → qrate (Ward) → novelty nov_resid →
punctint qrate → vslope → punctint list (conditional) → interleave
tss → dialevel. As of this append: `novelty` screen card frozen by
runpod-e (claim-line; zero-new-caching verified; no verdict yet) — no
outcomes posted. Standing re-vet triggers: **P2 lifts** if
punctint-list dies specifically on position; **P6 lifts** if Ward
verbosity dies on a Ward-specific artifact.

- 2026-07-24 · **novelty `nov_resid` (B2): NEGATIVE** — KEEP fails on
  all three models (best gap +0.038…+0.045, peaks mid-ladder, no kill
  rule fires); 71–77 % of the window-readable signal is already
  per-position — conversion with a genuine 23–29 % order-free
  residue; the shuffle-null receipt behaved exactly as designed.
- 2026-07-24 · **punctint q (B4): KEEP — the hunt's first Stage-1
  KEEP** — every clause on all three models, gap rising to
  +0.114…+0.143 at T64 tracking kernel mass, and it survives the
  within-document identity control (+0.101…+0.183); the ambient
  anchor LOSES from windows (candidate-2's trap checked and absent).
- 2026-07-24 · **punctint list (B3): WEAK KEEP, conditional** —
  2/3 models; the anchor SHARES the window gain; doc-mean-only AUC
  0.960; within-doc control rests on 8 test documents. It did NOT
  die on position ⇒ **P2 stays parked**. Never quoted bare, per
  binding qualification 1.
- 2026-07-24 · **Factory adopts runpod-e's doc-identity
  recommendation** (their LOG entry of this date): every future
  broad-factory builder computes and reports `doc_mean_only_auc`
  (doc-mean of the label, top vs bottom class) in its triage stats —
  the punctint screen showed the two frozen bars cannot see the
  document-identity route (doc-mean AUC 0.926/0.960 on faces that
  passed both) — and any face that KEEPs owes a within-document
  contrast. Reported as a disclosure statistic; the KILL authority
  stays with the two frozen bars until a review pins a threshold.
  First applied to B7's builder (conversation-mean AUC — exactly its
  named axis-b risk).
- 2026-07-24 · **Corpus scale-up campaign COMPLETE** (`runpod`,
  `briefings/corpus-scaleup.md`; receipts in `SCALEUP.md`, four LOG
  entries). punctint 400 → **4,000 docs** (prefix identity confirmed
  token-for-token ⇒ existing caches cover the first ~790k tokens/model)
  and refmark 400 → **2,000 convs**; frozen logic, new versioned
  artifacts, every triage AUC now carrying a 1,000-rep document-level
  bootstrap CI (`labels/boot_lib.py`). **No frozen bar fires at scale**,
  but three standing numbers move: unigram UP into the 0.55–0.65
  disclosure band on every face (0.546–0.583), position DOWN toward 0.5
  (the list face's all-eligible 0.639–0.653 → 0.560–0.566), doc-mean-only
  essentially unmoved (0.901–0.975). **Measured cause of the unigram
  rise: estimator sample size** — holding evaluation rows fixed and
  varying only train documents reproduces 76–91 % (list) / 45–57 % (q)
  of it, and the curve has not saturated at 3,200 docs ⇒ **every 400-doc
  unigram triage number in this ledger is an UNDERSTATEMENT, and the
  scaled number is itself a lower bound** (`probe_estimator_scale.json`;
  the probe-side corollary is flagged as an unverified hypothesis, not a
  claim). The "8 documents" within-doc control becomes **56 (list) /
  117 (q) / 52 (refmark)** test documents at ≥ 20 manifest rows per
  class. `is_user_echo` ships in the scaled refmark npz (0.52 % of
  manifest rows). Threshold dataset for the deferred `doc_mean_only_auc`
  bar, now **eleven faces** with CIs (`labels/docmean_index.json`,
  `SCALEUP.md` §7), spanning 0.554 (Ward `vslope`) to 0.975 (refmark).
  **Recommendation to the review: do NOT promote it to a kill bar** —
  any threshold that separates the low families sits below **punctint q
  at 0.901, the hunt's only unconditional KEEP**; keep it a disclosure
  statistic that triggers a mandatory within-document contrast (the same
  conclusion runpod-e's collision note reaches causally via dialevel).
  Revised late 2026-07-24: the earlier "0.82–0.88 separates NEGATIVE
  from surviving" reading is WITHDRAWN with runpod-e's `novelty`
  NEGATIVE verdict; the measurements are unchanged.

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
*Round-2 verdict (2026-07-24): **KILLED at triage — free kill.** The
frozen unigram bar fired on the operative position-matched manifest
rows (gpt2 0.6530 ≥ 0.65; gemma2 0.6430 and llama31 0.6298 at the
top of the disclosure band) with ALL math-span tokens masked: the
math-notation topic leak lives in the surrounding prose register
itself, not the delimiters. Position clean (manifest 0.50–0.53).
Receipt: `eqdens/CARD_DRAFT.md` verdict appendix +
`labels/eqdens_stats.json` + pinned `labels/eqdens_corpus.json.gz`
(builder regenerates the npz deterministically).*

**B7 — Refusal/deflection-marker intensity on multi-turn chat
(round-2 append; BUILD-if-time, strictly BEHIND B6).** The
backtracking-faithful port of the refusal idea whose as-posed version
is D7: an intensity needs RECURRENCE, and standard refusal datasets
are single-shot — so the corpus must be real multi-turn chat with
recurring refusals (WildChat-class; CPU-downloadable; pinned corpus
artifact per the dialevel new-corpus precedent; transcripts run
through the three cached base models = one NEW caching pass each,
cost stated wherever the screen is planned). Events = assistant turns
matching a FROZEN refusal/deflection substring list, seeded from the
refusal paper's own `refusal_score` set (`docs/papers/refusal.md`
§ D.1; the concrete strings live in the paper's public code repo —
the paper's Figure 11 is an image — pull them and freeze the list in
the card BEFORE counting anything). Label = λ̂ over PREVIOUS turns
(kernel per the sc_lambda/dialevel precedent), marker-turn tokens
masked. Four-axis vet: (a) conversion — the D7 receipts cut both
ways here: the current-turn refusal bit is presumed converted/ambient
(App. J finds the direction in BASE models), so that face is the
disclosed regime-1 anchor, never the primary; the candidate is
strictly the trailing marker intensity ("how refusal-laden has this
conversation been"), which has no obvious next-token utility beyond
topic. (b) the loud risk: refusing conversations are topically
distinctive — harmful-topic vocabulary is a massive unigram leak and
refusal text is self-stamping; marker-turn masking removes the
self-stamp, unigram triage on the frozen bars decides the leak.
(c) turn-level events with the dialevel clock geometry —
under-spanned panel, said plainly. (d) regime-2 LEVEL prediction:
window > per-token, T-growing, shuffle-immune. **HARD PRE-GATE
before any building: measure the event rate on the pinned sample
first — if < ~2 % of assistant turns match the frozen list, kill in
the ledger for free** (quantity-mode win condition includes honest
kills). **BUILD-if-time.**
*Round-2 verdict (2026-07-24): pre-gate PASSED loudly (marker rate
0.147 of assistant turns, 7× the bar; 38 % of ≥ 8-assistant-turn
conversations have ≥ 2 marker turns) and the bundle **SHIPPED**: the
frozen bars came back clean-to-mild — unigram 0.517–0.532 on manifest
rows (near-blind: the harmful-topic leak does NOT materialize at
token level once marker messages are masked), position 0.545–0.565 —
but the adopted conversation-identity statistic is the loudest in the
program (`doc_mean_only_auc` 0.966–0.968), so the **within-
conversation contrast is a BINDING screen precondition**, with the
position floor probe and the beat-the-visible-evidence line (kernel
support ≈ 1,000–1,150 tokens, ~16× the ladder top — the loudest
under-span in the factory). Receipts: `refmark/CARD_DRAFT.md` verdict
appendix + `labels/refmark_stats.json` + `labels/refmark_pregate.json`
+ pinned `labels/refmark_corpus.json.gz`.*

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
PARK behind B6. *Round-2 addendum: B6's triage kill MEASURED the
technical-register topic leak (prose near math reads the label at
0.63–0.65 with all math tokens masked) — P3 inherits that receipt;
any lift must argue why citation-heavy prose leaks less.*

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

**D7 — Refusal-as-posed: "maybe attention doesn't linearize refusal
to a single position" (round-2 append; vetted DEAD by mac-local,
2026-07-24).** The literature answers the posed question directly
(`docs/papers/refusal.md` — Arditi et al., *Refusal in Language
Models Is Mediated by a Single Direction*): a difference-in-means
direction extracted at a SINGLE (post-instruction position, layer)
pair — selected from the |I|×L candidate grid, § 2.3 — is causally
sufficient in BOTH directions across 13 chat models spanning 1.8B to
72B (Qwen/Yi/Gemma/Llama-2/Llama-3): ablating it everywhere collapses
refusal of harmful instructions (§ 3.1), adding it at one layer
induces refusal on harmless Alpaca instructions (§ 3.2). Sharper
still, § 5.2 measures the conversion mechanism itself: DFA
attribution shows a handful of attention heads reading the
harmful-instruction WINDOW and depositing onto the refusal direction
at the last position — the window→position deposit is not presumed,
it is the paper's finding (adversarial suffixes jailbreak precisely
by hijacking those heads' attention off the instruction). And App. J
removes the chat-only escape hatch: the direction already separates
harmful/harmless prompts in BASE models. Axis (b): harmful-topic
vocabulary is a massive unigram leak, and refusal TEXT is
self-stamping. Axis (c): refusal-as-posed is a prompt-level rollout
boolean — the AVOID class (forbidden-word/emotional-onset precedent).
Economics: needs a chat model + instruction corpus (none of our
caches apply) + judge labels beyond string match. DEAD as posed; the
recurrence port that survives four-axis vetting is B7.

---

*Ledger committed before any builder (quantity-mode rule). BUILDs
ship in the B1 → B2 → B3/B4 order, B5/B6 stretch; every bundle or
triage kill gets one LOG line as it lands. Bundle format, triage
stats, masking rules: per `briefings/candidate-factory-traces.md` and
the `labels/README.md` alignment contract.*
