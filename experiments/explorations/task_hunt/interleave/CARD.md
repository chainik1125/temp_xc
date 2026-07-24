# FROZEN card — interleaved-document `tss` (the anti-conversion candidate)

**Status: FROZEN at commit (commit-then-run; the activation cache is
being built as this card is committed, and NO screen cell has been
executed).** Agent: runpod-e. Briefing: `briefings/task-hunt-r2-e.md`
§ 3 (quantity mode; queue position 8 — the park on this candidate was
lifted for screening). Frozen from runpod-b/runpod's screen-ready
`CARD_DRAFT.md` (ledger `../CANDIDATES.md` B1).

## 1. Why this candidate matters more than its queue position suggests

Every kill this agent produced in round 1, plus the `novelty` screen in
this round, died the same way: **conversion**. The model linearizes,
per position, whatever helps predict the next token, so a probe at one
position already reads most of the latent and a window adds a small
order-free residue (novelty: 71–77 % of the window-readable signal was
already per-position).

This corpus is built to hold that mechanism's *input* near zero for
`tss` while keeping the state real: two lexically-matched fineweb
documents (greedy max-Jaccard pairing; overlap 0.120 matched vs 0.080
random) interleaved in strictly alternating **1–4-sentence blocks**, so
the switch hazard stays weak and jittered (measured
`switch_hazard.hazard_by_offset` ≈ 0.000 → 0.013 over the first dozen
offsets — low, and NOT memoryless). "Tokens since the last source
switch" is therefore a genuine sequential state with **almost no
generative payoff**. If conversion is really the thing that kills
window candidates, this is where a window should finally win; if this
also converts, the mechanism is more general than "the model learns
what predicts the next token" and that is the finding.

## 2. Faces (the demotion is the design)

- **`tss` — tokens since the last source switch: PRIMARY.** Label-side
  triage already passed: unigram type-mean AUC **0.551** on top-vs-
  bottom terciles (near-blind to token identity). Train-split tercile
  edges 19 / 47 tokens (per-tokenizer values in the stats JSON).
- **`source` — which document is currently active: DISCLOSED ANCHOR,
  not a primary.** It is generatively useful (it predicts vocabulary)
  so it is *expected* to be converted; held-out unigram readout 0.661
  matched vs 0.701 random pairing. **Frozen prior: a HIGH per-token
  reading on `source` does NOT count against the candidate.** It is
  the ambient calibration — the same role the `is_q`/`is_list` anchors
  played in the punctint screen, where the anchor differential is what
  separated a face-specific advantage from a generic window effect.

## 3. Substrate + the alignment contract

New token stream ⇒ one forward pass per model
(`interleave/cache_acts.py`, committed before this card at `f9f917d3`):
~335k tokens ⇒ 2518 / 2506 / 2487 rows of 128, geometry identical to
the replag caches (BOS prefix for gemma/llama, non-overlapping content
chunks, document tails dropped). **The committed `token_ids` are fed
verbatim — never re-tokenized** (builder's contract); the flat↔windowed
mapping was verified before caching (gpt2 2518/2518 rows reproduce
their flat slice exactly). Screen layers = the replag screen layers:
gpt2 hs7, gemma2-2b hs14, llama31-8b hs14.

## 4. Rows (frozen)

Builder manifests `man_tss_*` (20k/class, 3-class terciles, pos ≥ 32,
split by interleaved doc; `doc_split` 160 train / 40 test) and
`man_src_*` for the anchor. Mapping and eligibility as in the novelty
and punctint screens: `chunk = pos // content`,
`cache_pos = n_prefix + pos % content`; **uniform eligibility
`pos ≥ 64` and `pos % content ≥ 63`** so every screened T ≤ 64 reads
IDENTICAL rows inside one cache row. Caps 4000 train / 1500 test per
class, seeded (MATCH_SEED 1013 + crc32); MIN_ROWS 300 floor.

**A reach limitation stated pre-run:** the uniform-eligibility rule
costs rows on this smaller corpus (335k tokens vs fineweb's 794k). If
any class falls under the floor the target is skipped and recorded as
skipped — never silently rebalanced.

## 5. Clock bridge (measured, frozen)

Block length in tokens: q10 **13**, q25 26, median **47**, q75 76, q90
**105**. So T = 4/8 rarely reaches the previous switch, T = 32 usually
does, T = 64 nearly always. The ladder spans the clock — unlike the
punctint faces, this candidate's support is fully inside the reachable
range, which makes a flat result here a *real* negative rather than a
reach-limited one.

## 6. Probe grid (frozen; identical to the novelty/punctint screens)

Per model, on the screen layer: per-token linear + MLP(512) **first**
(per-token-first triage); `T ∈ {4, 8, 16, 32}` window linear,
window-MEAN linear, context-shuffled linear (anchor fixed, seeded);
window-MEAN additionally at T = 64; window + shuffled MLP at
T ∈ {16, 32}; permutation nulls (NULL_SEED 99) at T = 16;
**position-only floor** on the shipped rows; `source` anchor per-token
+ T = 16 window-MEAN. Metric: acc_test 3-class (chance 1/3) + per_class;
the anchor is binary (rank-AUC, `class_weight=True`).

**Mechanism receipt — the shuffled-block NULL CORPUS** (adopted as the
receipt, resolving the draft's decision point). The null corpus is the
same tokens re-ordered by the builder's `null_perm` with labels
recomputed (`tss_null`), cached separately at the screen layer. The
reader is run over it with the identical grid. **This is a stronger
receipt than a within-window shuffle**: it destroys document coherence
in the *input to the model*, not just in the probe's view, so it tests
whether the model maintains switch-distance as state or whether the
probe is reading local bookkeeping that survives incoherent text.

**Document-identity control (added by this card, not in the draft).**
The punctint screen showed the frozen factory triage cannot see a
doc-level route, and that a window-MEAN is an excellent document
descriptor (doc-mean-only AUC 0.926/0.960 there). `tss` should be far
less exposed — switch distance is a within-document ordinate by
construction — but the check is three lines and the claim is
unfalsifiable without it. **`doc_mean_only_auc` for `tss` terciles is
computed and reported before any activation cell is read**, and a
within-document contrast is run if any KEEP clause fires.

## 7. Frozen predictions (scored either way)

- **S1 (the anti-conversion bet).** Per-token `tss` is LOW — within
  0.05 acc of its position floor, and ≥ 0.05 below the best window
  cell. This is the prediction the candidate exists to test.
- **S2.** window − per-token gap is positive and GROWS over
  T ∈ {4…32} as windows begin to span the previous switch, with the
  largest step between T = 16 and T = 32 (the median block is 47
  tokens).
- **S3.** The `source` anchor is per-token HIGH (converted, expected)
  and gains little from a window — so the anchor differential favours
  `tss`, as it did for punctint's q face.
- **S4 (the receipt).** `tss` recovery on the shuffled-block NULL
  corpus is DEGRADED by ≥ 0.03 acc at matched T relative to the real
  corpus. If it is not, the signal is local bookkeeping.
- **S5 (regime).** Order-free pooling suffices: MEAN ≥ flatten and the
  within-window shuffle is immune. (A flatten > mean result would be
  the first order-carried candidate in the hunt and must be reported
  loudly if it appears.)

## 8. KEEP / KILL (frozen)

**KEEP** iff, on ≥ 2 of 3 models: `tss` window − per-token ≥ **+0.05**
acc at some T, the gap grows over T ∈ {4…32}, the window clears the
position floor by ≥ 0.05, AND the shuffled-block null degrades recovery
by ≥ 0.03 (S4). Regime-2 (order-free) is ACCEPTED and is not a kill.

**KILL** if ANY of: (1) per-token-first triage is HIGH — per-token
within 0.02 of the best window at every T, or per-token ≥ 0.05 above
its position floor while the window adds < 0.05 (converted; run the
depth sweep as the WHY-diagnostic and stop); (2) no gap beyond
3 σ_null at any T; (3) the only window win is on the `source` anchor;
(4) the gap does not grow anywhere over T ∈ {4…32}; (5) the null
corpus reads `tss` as well as the real corpus (local bookkeeping).

**If no rule fires** (real but under bar), the verdict is recorded as
**WEAK — no rule fires as written**, with the numbers. This clause is
explicit here because the `novelty` card lacked it and the omission had
to be recorded rather than patched after the fact.

**Conversion is reported as a fraction, not only as a gap.** Per the
recommendation in the `novelty` LOG entry, every model's
`(tok − floor) / (best_window − floor)` is reported next to its gap: an
absolute window−token difference cannot distinguish "converted with a
residue" from "genuinely window-only", and for this candidate that
distinction IS the result.

## 9. Deliverable

`results/screen_<model>.json` (incremental/resumable), one LOG verdict
scoring S1–S5 — with the conversion fraction and the anchor
differential quoted — and a figure only if it KEEPs. No leaderboard
rows (Stage-1 screens are raw-activation probes).
