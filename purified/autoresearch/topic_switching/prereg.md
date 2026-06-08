# Preregistration — topic-switching as a temporal property

**Frozen before measurement** per [`autoresearch_spec.md`](../autoresearch_spec.md)
§ 2.1. Committed prior to running any statistic; not edited afterward (only an
abort/verdict is appended in the record).

Investigation #2 of the temporal-property autoresearch loop. Where
backtracking ([`backtracking_record.md`](../backtracking/measurement.md)) probed the
**self-exciting / recurrent** dynamics class (sparse AC events), this probes the
**change-point / sticky-dwell** class: a slow-varying persistent state (DC in
the bulk) punctuated by rare switches (the only AC events). It is the *cheap,
measurable* representative of that class; emergent misalignment is the same
dynamics class at higher cost (shared generator — see
[`changepoint_bench_spec.md`](../changepoint/bench_spec.md)).

## Property

**Topic / discourse-segment switching within documents** — a document dwells in
one topic for a stretch of sentences, then switches. The temporal object is the
**per-sentence topic-id stream** and, derived from it, the **switch-indicator**
and **dwell (run-length)** sequences.

## Data + labeler (version-pinned)

- **Source:** `HuggingFaceFW/fineweb-edu` **text** (the same corpus behind the
  on-branch `gemma_2_2b_it_l13_fineweb_24k128` datasource). A fixed sample of
  **`N_doc = 2000` documents** with **≥ 8 sentences** each, drawn with a pinned
  sampling seed (`SEED = 0`). Sentence segmentation by a pinned splitter
  (`spaCy en_core_web_sm`, version recorded at run time). Unit of time = the
  **sentence index** within a document.
- **Labeler (the crux — runs on *text only*, no API, no activations):**
  per-sentence **topic-cluster id** from a pinned sentence-embedding model
  (`sentence-transformers/all-MiniLM-L6-v2`, revision recorded) → L2-normalized
  embeddings → **mini-batch k-means** with **`K = 20` clusters**
  (clustering seed pinned). The topic id of sentence *i* is its cluster
  assignment. *No model activations are used at this stage* — the measurement is
  a pure text→label temporal analysis (activations matter only for a possible
  downstream real-data probe, which is out of scope here).
- **Label-noise indicator + noise floor.** Clustering is fuzzy, so the floor is
  reported three ways and carried as a caveat: (a) **silhouette** of the
  clustering; (b) **re-seed stability** — adjusted Rand index (ARI) between two
  k-means runs with different seeds; (c) **K-sensitivity** — ARI between
  `K = 20` and `K ∈ {12, 32}`. The headline effect must be large relative to the
  disagreement these induce, tested by the label-perturbation control below.
- **Scope honesty:** one web-text corpus, one embedding-based topic notion — a
  finding here is about *this* operationalization of topic on *this* corpus, not
  "topic in language" in general. The topic id is a clustering artifact, not a
  ground-truth label; this is why the noise floor is load-bearing.

## Statistics to measure (fixed)

On each document's topic-id sequence `z_1 … z_L` (`z_i ∈ {1..K}`), the
switch-indicator `s_i = [z_i ≠ z_{i-1}]`, and the run-length (dwell) sequence:

1. base **switch rate** `E[s_i]`; per-**position** switch rate (trend through
   the document?);
2. **same-topic autocorrelation** `C(k) = P(z_{i+k} = z_i) − chance` vs lag `k`
   (expect slow decay = long memory);
3. **dwell / run-length distribution** — *the discriminator*: empirical
   survival function of run-lengths, fitted geometric vs heavier-tailed
   (negative-binomial / discrete-Weibull) by likelihood ratio; over-dispersion
   of run-lengths vs geometric;
4. **transition matrix** among topics + **Markov-order test** (0 vs 1 vs 2) on
   the topic-id sequence (conditional-entropy / likelihood-ratio);
5. **mutual information** `I(z_i; z_{i+k})` vs `k`;
6. **burstiness** of switches: Fano factor of windowed switch counts.

All with bootstrap CIs over documents; held-out document split reported.

## Null models (the temporal-ness gate)

- **N1 — within-document permutation** (preserves the topic marginal and count,
  destroys dwell + order). Real vs N1 = *total* temporal structure.
- **N2 — first-order Markov** at the empirical transition matrix (preserves the
  one-step transition structure, hence **geometric** within-state dwell;
  destroys any heavier-tailed stickiness). **Real vs N2 = genuine long-memory /
  heavy-tailed dwell beyond a memoryless Markov chain** — the key discriminator,
  and the analogue of backtracking's N2.
- **N3 — i.i.d. topic draws** at the marginal (no stickiness at all). Real vs
  N3 = there is *any* persistence (expected trivially yes).

## Predictions (preregistered, with confidence + reasoning)

- **P1 (high):** strong stickiness — `C(1) ≫ 0`, switch rate ≪ 1, real ≫ N3 on
  same-topic autocorrelation and dwell length. *Reason:* documents are
  topically coherent over stretches; this is near-certain and mostly a sanity
  check.
- **P2 (moderate — the open question):** dwell is **heavier-tailed than
  geometric** — real ≫ N2 on the run-length tail and long-lag MI. *Reason:*
  some sections (a long argument, a list) persist far longer than a memoryless
  self-transition would predict. **It may instead be ≈ geometric** (a
  first-order Markov chain captures the dwell) — in which case topic is temporal
  only as a **memoryless slow state (DC)** with no long memory, which is itself a
  clean, reportable finding and selects a different mirror.
- **P3 (moderate):** weak position-trend in switch rate (switches roughly
  uniform through a document, perhaps slightly elevated near the start).
- **P4 (high):** Markov order ≥ 1 on the topic-id sequence (order-1 ≫ 0);
  order-2 vs 1 is the open part.

## Verdict criteria

- **Temporal — sticky / long-memory (semi-Markov):** real ≫ N2 (heavy-tailed
  dwell, long-lag MI beyond N2) → mirror with a **semi-Markov** process (fitted
  dwell distribution), spec stages 4–5.
- **Temporal — memoryless Markov (DC slow-state):** real ≫ N3 but ≈ N2 → report
  "sticky but memoryless"; mirror with a **first-order Markov** chain (geometric
  dwell). Still a valid class-2 substrate, with the dwell knob set to geometric.
- **Not temporal:** real ≈ N1/N3 → **abort** with a clean negative (no
  stickiness — implausible, but the gate stands regardless of expectation).

In **all** non-abort cases the measured dwell distribution + transition matrix
*parameterize the shared change-point generator* — the measurement's primary
downstream use (see the bench spec). The geometric-vs-heavy-tailed verdict sets
the generator's persistence knob.

## Controls required (spec § 3, real-side)

- **Shuffle control:** the N1/N2/N3 comparisons above.
- **Label-noise robustness:** re-run the headline statistics under an
  *independent* relabeling (different k-means seed and, separately, `K ∈
  {12, 32}`); the verdict must survive — i.e. the heavy-tail (or its absence)
  must not flip with a defensible reclustering. Report the spread.
- **Held-out:** statistics on a held-out split of documents; check stability.

## Out of scope for this investigation

- The synthetic mirror's **architecture benchmark** (spec stage 6) is
  downstream and specified separately
  ([`changepoint_bench_spec.md`](../changepoint/bench_spec.md)); the
  **measurement + verdict is the primary deliverable**. A mirror is built only
  if the verdict is "temporal".
- **No model inference / activations** — pure text-label temporal analysis. (A
  real-data activation probe on `gemma…fineweb` is a possible *future* slice,
  not part of this investigation.)
- **Emergent misalignment** is the same dynamics class but requires a paid
  per-span judge labeler; it is not measured here. The shared generator is
  designed so EM slots in later without redesign.
