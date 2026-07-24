# Corpus scale-up campaign — receipts

**Agent:** `runpod`. **Mandate:** `briefings/corpus-scaleup.md` (overnight
CPU campaign; panel-grade data for the screen KEEPs). **Verdicts live in
`LOG.md`; this file is the receipt sheet** — every number below is read
back from a committed stats file by the scripts named beside it.

The campaign's premise: the hunt's first KEEPs sit on corpora too thin
for panel-grade evidence (runpod-e's punctint-list within-document
control rested on **8 test documents**; fineweb = 400 docs; refmark =
400 conversations). Nothing here re-runs a screen or changes a verdict.
Label logic stayed **frozen** — `punctint_lib`, `refmark_lib`,
`dialevel_lib`, `novelty_lib` are imported unchanged; every scaled
artifact is a NEW versioned file beside the shipped one, which was never
touched.

---

## 1. Artifacts

| artifact | what | size |
|---|---|---|
| `labels/fineweb4k_corpus.json.gz` | 4,000 fineweb docs, pinned recipe | 14.4 MB |
| `labels/fineweb4k_corpus_receipt.json` | prefix-identity receipt vs the pinned 400 | — |
| `labels/punctint4k_fineweb_{gpt2,gemma2,llama31}.npz` | both faces at scale | 15.4–15.6 MB each |
| `labels/punctint4k_stats.json` | triage + bootstrap CIs + censuses | 32 KB |
| `labels/probe_estimator_scale.json` | why the unigram bar moved (§4) | — |
| `labels/refmark2k_corpus.json.gz` | 2,000 WildChat convs, pinned recipe | 7.6 MB |
| `labels/refmark2k_corpus_receipt.json` | funnel + overlap with the shipped 400 | — |
| `labels/refmark2k_wildchat_{…}.npz` | + the new `is_user_echo` mask | 8.3 MB each |
| `labels/refmark2k_stats.json` | triage + bootstrap CIs + echo/recurrence | — |
| `labels/novelty_bootstrap.json` | item 3: CIs on the committed 400-doc triage | — |
| `labels/docmean_index.json` | item 3 extended: the doc-identity distribution over 11 faces (§7) | — |
| `labels/verify_prefix_labels.json` | frozen-logic check on the shared prefix (§2) | — |
| `labels/scaleup_caching_cost.json` | §6 table, machine-readable | — |

Builders (each committed BEFORE it produced output):
`pull_fineweb4k.py`, `build_punctint4k.py`, `pull_refmark2k.py`,
`build_refmark2k.py`, `probe_estimator_scale.py`, `boot_novelty.py`,
`boot_docmean_index.py`, `verify_prefix_labels.py`,
`scaleup_caching_cost.py`, and the new `boot_lib.py`
(+ `tests/test_boot_lib.py`, 7 tests; suite 304 passed).

**Bootstrap convention.** `boot_lib` resamples DOCUMENTS (conversations)
with replacement — rows inside a document are not independent, and a
row-level interval would be an order of magnitude too narrow. 1,000
reps, percentile CIs, exact Mann-Whitney by tie-collapsed level counting
(algebraically identical to `interleave_lib.rank_auc`, asserted in the
tests; ~3 s for a 1.3M-row statistic).

---

## 2. Corpus receipts

**fineweb 400 → 4,000 (`pull_fineweb4k`).** Same dataset/config/split,
seed 0, shuffle buffer 10,000, 60–200-sentence filter, splitter v1 —
only `n_docs` moved. 43,122 rows scanned, 375,498 sentences (10.2× the
pinned 36,805).

> **PREFIX IDENTITY PASSES.** The pinned 400-doc sample is exactly the
> first 400 documents of the scaled pull — 400/400 ids, 400/400 sentence
> lists — and the property survives tokenization: `build_punctint4k`
> confirms `token_ids` AND `doc_off` prefix identity against
> `replag_fineweb_<tok>.npz` for all three tokenizers. The scaled corpus
> is a deterministic SUPERSET, so **the pods' existing caches already
> cover the first ~780–794k tokens per model** (§6).

**The frozen-logic claim is checked, not asserted
(`verify_prefix_labels`).** Because the corpora share a token-for-token
prefix, the shipped and scaled bundles must agree exactly on every
per-token label of the first 400 documents. They do — **bit-identical on
all three tokenizers** across 793,831 / 784,512 / 777,900 tokens:
`token_ids`, `doc_off`, `sent_idx`, `in_span`, both faces' λ̂ (NaN-aware)
and both faces' event flags. Two things legitimately differ, quantified
rather than waved away: `doc_split` (drawn for n_docs) and the 3-class
labels, whose edges are re-estimated per corpus — **0.56–0.57 % of
shared rows change class on the list face** (zero_split median-of-
positives 0.31046 → 0.31242) and **0.0000 % on the q face**, whose edge
came out identical on a corpus ten times the size.

**WildChat 400 → 2,000 (`pull_refmark2k`).** Same pinned revision
`7d6490e4…`, same filters, seed 0; stream prefix 40,000 → 250,000. The
funnel — which the shipped build never recorded — is now on the record:

| streamed | English | ≥ 8 assistant turns | in char window (pool) | sampled |
|---|---|---|---|---|
| 250,000 | 119,458 | 6,788 | **6,256** | 2,000 |

A pool subsample redraws, so this is **not** a superset: all 400 shipped
conversations are in the larger pool, but only **121** land in the
scaled 2,000 — the two bundles are near-independent evidence.

---

## 3. Triage at scale (bars unchanged and frozen)

Direction-agnostic max(AUC, 1−AUC); **manifest rows operative**;
≥ 0.65 ⇒ KILL; 0.55–0.65 ships with disclosure. **No frozen bar fires
on any face at scale.** Manifest-row points with 95 % doc-bootstrap CIs;
"400" is the shipped bundle's own number on its own rows.

| face | stat | 400 | scaled | 95 % CI |
|---|---|---|---|---|
| punctint **list** | unigram | 0.517–0.534 | **0.574–0.583** | [0.559, 0.598] |
| | position | 0.415–0.428 | 0.470–0.478 | [0.436, 0.512] |
| | doc-mean-only | 0.960 † | **0.966** | [0.958, 0.973] |
| punctint **q** | unigram | 0.520–0.533 | **0.558–0.563** | [0.545, 0.576] |
| | position | 0.471–0.478 | 0.511–0.518 | [0.472, 0.569] |
| | doc-mean-only | 0.926 † | **0.901–0.902** | [0.886, 0.917] |
| **refmark** | unigram | 0.517–0.532 | **0.546–0.565** | [0.529, 0.583] |
| | position | 0.435–0.456 | 0.478–0.504 | [0.423, 0.554] |
| | doc-mean-only | 0.966–0.967 | **0.974–0.975** | [0.964, 0.983] |

† runpod-e's post-hoc measurement on their own screened pool.

Three findings, in order of how much they bind Stage 2:

1. **The current-token identity route is systematically UNDERSTATED at
   400 documents** — every unigram number rose at scale, into the
   0.55–0.65 disclosure band. §4 shows this is mostly an estimator
   artifact, not a different corpus, and that it generalizes to every
   400-doc triage number in the hunt.
2. **Position leak FELL at scale.** The list face's all-eligible
   position AUC was direction-agnostic **0.639–0.653 at 400 docs — one
   tokenizer over the 0.65 kill bar** — and reads 0.560–0.566 at 4,000
   (manifest rows, the operative set, moved 0.572–0.585 → 0.522–0.530).
   Small-corpus position numbers are noisy in the dangerous direction
   too.
3. **Document identity survives 10× scale.** Every KEEP face stays at
   0.90–0.98 doc-mean-only. The within-document contrast is not a
   formality that more data dissolves; it remains the binding control.

---

## 4. Why the unigram bar moved (`probe_estimator_scale`)

The triage score is a **train-set mean per token type**
(`novelty_lib.type_mean_scores`). With 320 training documents most types
are seen a handful of times, so the score is largely noise and the
measured AUC is attenuated toward 0.5. Because the scaled corpus
CONTAINS the pinned one, the two readings separate cleanly: hold the
evaluation rows fixed (the scaled build's test manifest rows) and vary
only how many train documents feed the estimator.

Manifest-row unigram AUC vs train documents (3 seeded draws each):

| face/tok | 40 | 160 | **320** | 640 | 1280 | **3200** |
|---|---|---|---|---|---|---|
| list/gpt2 | 0.509 | 0.530 | **0.531** | 0.549 | 0.562 | **0.574** |
| list/gemma2 | 0.515 | 0.540 | **0.541** | 0.560 | 0.571 | **0.582** |
| list/llama31 | 0.514 | 0.534 | **0.536** | 0.555 | 0.569 | **0.583** |
| q/gpt2 | 0.523 | 0.537 | **0.541** | 0.546 | 0.552 | **0.558** |
| q/gemma2 | 0.524 | 0.541 | **0.546** | 0.552 | 0.557 | **0.563** |
| q/llama31 | 0.523 | 0.537 | **0.538** | 0.548 | 0.551 | **0.558** |

At 320 train documents — the shipped build's training size — the
estimator lands within 0.01–0.02 of the shipped numbers on entirely
different rows. Estimator sample size accounts for **76–91 % of the rise
on the list face and 45–57 % on q**; the remainder is the row-set
difference (different test documents, a 5× larger manifest), not
separately isolated. **The curve has not saturated at 3,200 documents**,
so even the scaled number is a lower bound on the true current-token
route.

**Unverified consequence, stated as a hypothesis because it was not
measured:** a screen's per-token probe is also an estimator fitted on
finite training rows. If it attenuates faster than the window probe (a
per-token route needs more data than a smoothed aggregate one), then a
400-document screen understates its per-token baseline and therefore
**overstates the window-minus-per-token gap** — the hunt's headline
statistic. Cheap check, on artifacts that now exist: re-fit one screened
bundle's per-token and window probes at two training sizes on the scaled
corpus and compare the gaps. Nothing in this campaign licenses the
stronger claim.

---

## 5. What the data supports (censuses)

**Manifest rows.** The cap moved 20k → 100k rows/class and **binds**:
position-matched support is 189,959/class (punctint list), 529,708/class
(punctint q), 379,594/class (refmark). Shipped manifests carry
~100k/class, ~38–40k of which are test rows.

**Within-document contrast — the "8 documents" question.** "Carries the
contrast" is not one number, so it is reported as a ladder over the
minimum manifest rows per class a document must supply (test documents;
`all` in the stats files covers both splits):

| face | ≥ 1 | ≥ 5 | ≥ 20 | ≥ 50 |
|---|---|---|---|---|
| punctint list (was **8**) | 199 | 173 | **56** | 3 |
| punctint q | 504 | 437 | **117** | 7 |
| refmark (binding control) | 102 | 81 | **52** | 17 |

So the answer is **fixed, with a ceiling**: a serious per-document
contrast (≥ 20 rows/class) now has 52–117 documents instead of 8 — but
not thousands, because a position-matched manifest spreads its rows
thinly across a 10× larger document pool. Depth is available from the
same artifacts by restricting the manifest to fewer documents; that is a
Stage-2 design lever, not a data limit.

**refmark extras.** Marker rate 0.135 of assistant messages (0.148 at
400; pre-gate 0.147). Recurrence: **33.6 %** of conversations carry ≥ 2
marker messages (37.7 % on the pre-gate population), 51.2 % carry ≥ 1,
mean 1.6, max 30. Kernel support 1,096 tokens/message-8 — **the ~16×
under-span versus the T = 64 ladder top is confirmed at scale.**

**User-echo exposure, now shipped as `is_user_echo`.** Marker masking
covers assistant messages only, so a user message quoting a frozen
substring stays manifest-eligible. At scale: **98 / 23,772 user messages
(0.41 %)** and **1,567 / 299,994 manifest rows (0.52 %)** — about twice
mac-local's 0.22 % on the shipped build, still small, and now droppable
in one line by a screen. It changes no label, mask or manifest.

---

## 6. Caching cost (for the GPU pods)

`scaleup_caching_cost.py`, derived from the committed stats. Tokens and
multiples, not gigabytes — footprint per token depends on the pod's
layer selection and dtype.

| corpus | tokenizer | tokens | already cached | NEW to cache | × shipped stream |
|---|---|---|---|---|---|
| fineweb4k | gpt2 | 7,913,315 | 793,831 | 7,119,484 | 9.97× |
| fineweb4k | gemma2 | 7,815,498 | 784,512 | 7,030,986 | 9.96× |
| fineweb4k | llama31 | 7,747,503 | 777,900 | 6,969,603 | 9.96× |
| refmark2k | gpt2 | 6,526,499 | 0 | 6,526,499 | 4.79× |
| refmark2k | gemma2 | 5,918,102 | 0 | 5,918,102 | 4.80× |
| refmark2k | llama31 | 5,672,921 | 0 | 5,672,921 | 4.78× |

**Total new: 39.2M tokens across the three models** (21.1M fineweb4k +
18.1M refmark2k). The fineweb "already cached" column is earned by the
prefix receipt; the refmark column is honestly zero.

---

## 7. Item 3 — the novelty family (threshold-dataset material)

Novelty screened NEGATIVE; nothing here is a verdict. Its committed
400-doc triage now carries doc-bootstrap CIs (`boot_novelty`, shipped
point estimates asserted to reproduce `novelty_stats.json` exactly),
plus the two views the shipped stats predate: manifest rows and
`doc_mean_only_auc`.

Items 1–3 finished inside the night, so the same argument was extended
to **every committed bundle that ships a manifest**
(`labels/boot_docmean_index.py` → `docmean_index.json`). The row set is
each bundle's OWN shipped manifest, test documents only — a manifest is
the author's statement of which rows ship, masks already applied — the
only statistic is `doc_mean_only_auc` with a 1,000-rep cluster
bootstrap, and Ward-stream bundles cluster by **trace**, not stream row.
No unigram or position numbers are recomputed for bundles I did not
build.

| face | doc-mean-only (dir-agnostic) | 95 % CI | clusters | screen outcome |
|---|---|---|---|---|
| verbosity `vslope` (Ward) | 0.554 | [0.531, 0.577] | 60 | — |
| interleave `tss` | 0.675 | [0.619, 0.713] | 40 | **KILL (converted)** |
| novelty `nov_raw` (400 docs) | 0.758–0.767 | [0.712, 0.802] | 80 | disclosed secondary |
| oprate `ver` / `case` (Ward) | 0.771 | [0.718, 0.818] | 56 | — |
| novelty `nov_resid` (400 docs) | 0.760–0.784 | [0.710, 0.819] | 80 | **NEGATIVE** |
| qrate (Ward) | 0.803 | [0.755, 0.845] | 60 | — |
| sc_lambda (Ward λ̂) | 0.804 | [0.758, 0.836] | 60 | — |
| **punctint q** (4,000 docs) | 0.901–0.902 | [0.886, 0.917] | 800 | **KEEP** |
| dialevel `tlevel` | 0.965 | [0.941, 0.983] | 837 | screen foreclosed (qual. 2) |
| **punctint list** (4,000 docs) | 0.966 | [0.958, 0.973] | 800 | WEAK KEEP |
| **refmark** (2,000 convs) | 0.974–0.975 | [0.964, 0.983] | 400 | ships, screen pending |

**The statistic corroborates judgments the program reached by hand:**
`dialevel` lands at 0.965 — mac-local's qualification 2 foreclosed its
naive screen precisely because a dialogue-length selection route
dominates it, and this measurement, taken from a different direction,
puts it exactly there. The converted KILL and the NEGATIVE family sit
low; a pure trailing-slope face sits lowest.

**And it argues against making the statistic a kill bar.** Any
threshold separating the low families from the loud ones — anywhere in
0.82–0.88 — sits BELOW **punctint q at 0.901, the hunt's only
unconditional KEEP**, which passed both frozen bars, cleared its
position floor, and survived an explicit within-document contrast. Such
a rule would have killed it before it was screened. The adopted design —
a reported disclosure statistic that makes a within-document contrast
MANDATORY for any face that KEEPs — is what this evidence supports.
Caveats: eleven faces, five corpora, cluster counts from 40 to 837 (CI
widths are not comparable); Ward outcomes not posted in the ledger are
left blank rather than guessed; no causal claim — a face can be
document-dominated benignly (punctint q) or fatally (dialevel), which is
the whole reason the number belongs in a disclosure line.

Also worth the reviewer's eye: novelty's *raw* face carries a position
AUC of **0.115–0.135 (direction-agnostic 0.865–0.885)** on manifest
rows — far past the 0.65 kill bar. It ships only as a disclosed
position-confounded secondary, which is exactly the case the frozen bar
exists to catch.
