# FROZEN card — dialogue turn-length LEVEL (`dialevel`, DailyDialog)

**Status: FROZEN at commit (commit-then-run; the activation cache is
being built as this card is committed, and NO screen cell has been
executed).** Agent: runpod-e. Briefing: `briefings/task-hunt-r2-e.md`
§ 3 (quantity mode; queue position 9 — the last of my queue). Frozen
from runpod's `CARD_DRAFT.md` (ledger `../CANDIDATES.md` B5), under
mac-local's **binding qualification 2**.

Corpus licence: DailyDialog is **CC BY-NC-SA 4.0** (research use). The
note travels with any figure that graduates.

## 1. What is different about this bundle

Every other bundle in the queue was screened first and controlled
after. Here the control comes first, because mac-local's review
**foreclosed the naive screen before it was written**: all-eligible-row
position AUC 0.930–0.936 via a dialogue-length selection route (the
turn-count floor is fixed at 8, so a dialogue is long substantially
BECAUSE its turns are long). A window-vs-per-token gap measured over
globally-drawn rows would be uninterpretable here.

So the order of work was inverted, and the measurement that decides the
design ran **before the forward pass**, on labels alone
(`design_probe.py`, committed before it ran; results
`results/design_probe.json`).

## 2. Design-probe numbers (measured pre-run; they set the design)

Per tokenizer (gpt2 / gemma2 / llama31), on the screened eligible pool:

| quantity | value | consequence |
|---|---|---|
| `doc_mean_only_auc` | **0.983 / 0.986 / 0.984** | the highest in the hunt (novelty 0.792, `lam_q` 0.926, `lam_list` 0.960, `tss` 0.664) |
| between-dialogue variance | 82 % | `tlevel` is very nearly a dialogue-level constant |
| dialogue-length AUC | 0.878 / 0.883 / 0.847 (corr 0.59/0.59/0.57) | the named route is real — but smaller than identity as a whole |
| position AUC (eligible pool) | 0.936 / 0.939 / 0.905 | confirms the qualification |

**Design consequence 1 — length matching is NOT sufficient here, so the
strictly stronger of the two allowed controls is mandatory.** The
qualification permits "within-dialogue contrasts *or* dialogue-length
matching". Length explains 0.85–0.88 of an identity channel worth
0.98: matching on length would leave most of that channel open. The
primary arm is therefore **within-dialogue only**.

After the within-dialogue split (same pool, classes ranked inside each
dialogue):

| quantity | value | reading |
|---|---|---|
| dialogue-length AUC | 0.510 / 0.517 / 0.512 | route closed |
| `doc_mean_only_auc` | 0.517 / 0.532 / 0.538 | route closed |
| usable dialogues | 2790 / 2905 / 2547 | ample |
| rows (class 0 / class 1) | 52k/33k, 54k/34k, 46k/28k | ample |
| **position AUC** | **0.675 / 0.679 / 0.697** | **survives — must be a floor** |
| **`tst` AUC** | **0.662 / 0.651 / 0.657** | **survives — must be a floor** |

**Design consequence 2 — two per-position routes survive the control
and are promoted from footnotes to floors.** Within a dialogue, both
position and `tst` (tokens since turn start) still read the label at
0.65–0.70, and both are scalars a model can compute at a single
position. The window arm is therefore required to clear a **position +
`tst` floor**, not merely to beat the per-token activation probe.

I do **not** position-match within dialogue. A trailing mean over the
previous 5 turns necessarily drifts with turn index, so position
matching would remove genuine trailing structure along with the
artifact. Floor probes, reported next to every window number, are the
honest instrument; matching is not.

**Design consequence 3 — the power bound, stated before any cell.**
The within-dialogue contrast is **|Δ tlevel| median 4.0 / 4.0 / 4.2
tokens = 0.26 / 0.26 / 0.28 of the global tercile contrast** (14.8–15.5
tokens), and the two classes' 5-turn supports **overlap heavily**
(adjacent turns share 4 of 5 support turns). This arm is therefore
between three and four times harder than the naive one *by design*.
**A null on this arm is a BOUNDED negative** — it excludes effects
resolvable at Δ ≈ 4 tokens with 4000 rows/class above 3 σ_null, not all
window structure. That bound is quoted in the verdict whatever it says.

## 3. Substrate + the alignment contract

New token stream ⇒ one forward pass per model (`dialevel/cache_acts.py`,
committed before this card at `e8f85759`): 0.81–0.88 M tokens ⇒
**4111 / 4304 / 3653 rows of 128**, geometry identical to the replag
caches (BOS prefix for gemma/llama, non-overlapping content chunks,
document tails dropped; 57–62 % of tokens survive, since dialogues run
141–153 tokens median and most yield exactly one row). **The committed
`token_ids` are fed verbatim — never re-tokenized**; the builder
verifies the flat↔windowed mapping row-by-row before the forward pass.
Three layers are captured so a conversion-depth diagnostic never needs
a second pass. Screen layers = the replag screen layers: gpt2 hs7,
gemma2-2b hs14, llama31-8b hs14.

## 4. Rows (frozen)

Uniform eligibility, so every screened T ≤ 64 reads IDENTICAL rows
inside one cache row: row retained (not a dropped document tail),
`pos ≥ 64`, `pos % content ≥ 63`, `is_boundary == 0` (the builder masks
newline-spanning tokens — they are the marker face), `tlevel` finite.
Yield 180k / 186k / 161k rows over 3087 / 3215 / 2793 dialogues;
eligible positions q05/q50/q95 = 73 / 114 / 320.

**PRIMARY — within-dialogue binary.** Inside each dialogue, class 0 =
rows at its MINIMUM eligible `tlevel`, class 1 = rows at its MAXIMUM.
Balanced **per dialogue** to `min(n₀, n₁, 8)` rows per class, so
dialogue identity carries EXACTLY zero label information by
construction (stronger than the global balancing the punctint control
used). The per-dialogue cap of 8 also breaks the within-turn
redundancy: rows inside one turn are adjacent tokens with an identical
label. Dialogues are then drawn in seeded order until the global cap
(4000 train / 1500 test per class) is met. Realized on gpt2 before the
freeze (manifest build only — no activation read): 4000/class from
**634 train dialogues** of 2229 available and 1500/class from **228
test dialogues** of 561, shipped |Δ tlevel| median **3.8** tokens.
`MIN_ROWS` 300 floor; seeds `MATCH_SEED` 1013 + `zlib.crc32`.

**REFERENCE — the naive global arm, run and DISCLOSED as
uninterpretable-as-a-window-claim.** The builder's `tlevel_bin`
terciles on the same eligible pool, balanced caps. It is run precisely
because it is confounded: the difference between its gap and the
within-dialogue gap is the first direct measurement in this hunt of
what document identity buys a window probe. It scores no KEEP clause.

**ANCHOR — `tst` on the SAME shipped within-dialogue rows** (above vs
below their median `tst`). Same rows, different label: any difference
in window advantage is therefore face-specific, not a generic window
effect. `tst` is the disclosed conversion-risky face (near-syntactic —
"how long since the last newline") and is EXPECTED to read high
per-token; a high per-token `tst` does not count against the candidate.

## 5. Clock bridge (measured, frozen)

Turn ≈ 14.5–15.7 tokens; the 5-turn support ≈ 73–78 tokens. So T = 4/8
sees a fraction of one turn, T = 16 about one turn, T = 32 two, and
**T = 64 spans ~82–88 % of the support kernel** — the ladder top is
where the mechanism should live, and unlike the fineweb candidates this
support is inside the reachable range.

Note the mechanism this affords: a window MEAN over a boundary-marker
feature is a boundary RATE, i.e. ≈ 1 / mean turn length — the label's
reciprocal, computed order-free. That makes this the cleanest regime-2
aggregation story of the hunt, and it is why a negative here is
informative.

## 6. Probe grid (frozen)

Per model, on the screen layer, probe stack frozen
`conversion_depth/problib.py::fit_probe` (never retuned):

*Primary (within-dialogue, binary, rank-AUC, `class_weight=True`)* —
per-token linear + MLP(512) **first** (per-token-first triage); three
label-side floors on the shipped rows: `position_floor` (in-chunk +
document position), `tst_floor`, `postst_floor` (both); `T ∈ {4,8,16,32}`
window linear, window-MEAN linear, context-shuffled linear (anchor slot
fixed, seeded `SHUF_SEED` 1234); window-MEAN additionally at **T = 64**;
window MLP at T ∈ {16,32}; permutation nulls (`NULL_SEED` 99) on the
per-token and T = 16 window arms.

*Reference (global terciles, 3-class, acc, chance 1/3)* — per-token
linear, window-MEAN at T ∈ {4,8,16,32,64}, window linear at T ∈ {16,32},
position floor.

*Anchor (`tst`, same rows, binary)* — per-token linear, window-MEAN at
T ∈ {16, 64}.

## 7. Frozen predictions (scored either way)

- **D1 (per-token first).** Per-token linear on the within-dialogue
  face is ABOVE its `postst_floor` by ≥ 0.03 AUC — turn rhythm is
  generatively useful (newline prediction), so partial conversion is
  expected. Point prediction: per-token lands in 0.70–0.78.
- **D2 (the window bet).** best window − per-token ≥ **+0.05** AUC at
  some T, and the gap GROWS over T ∈ {4…64} with the largest step
  between T = 16 and T = 64. This is the strongest window prior of my
  hunt so far: the label is a windowed rate of a *visible* marker.
- **D3 (regime).** Order-free: window-MEAN ≥ window-flatten at matched
  T, and the anchor-fixed context shuffle is IMMUNE (within 0.02).
  A rate is order-free by construction. (Flatten > mean would be the
  first order-carried result of the hunt and must be reported loudly.)
- **D4 (the confound, quantified).** The REFERENCE arm's window−token
  gap exceeds the within-dialogue arm's at matched T by ≥ 0.10. If the
  two arms agree instead, then document identity buys a window probe
  much less than `doc_mean_only_auc` 0.98 suggests, and I say so — that
  would partly walk back my own triage-bar recommendation.
- **D5 (anchor differential).** On the SAME rows, `tst` is per-token
  HIGH and gains < 0.03 from T = 64 window-MEAN, so any `tlevel` window
  advantage is face-specific.

## 8. KEEP / KILL (frozen)

Scored on the **within-dialogue arm only**; the reference arm scores
nothing.

**KEEP** iff, on ≥ 2 of 3 models: (a) best window − per-token ≥
**+0.05** AUC at some T; (b) the gap grows over T ∈ {4…64}; (c) the
best window clears BOTH the `postst_floor` and the activation
`position_floor` by ≥ 0.05; (d) mean ≥ flatten at matched T.
Regime-2 (order-free) is ACCEPTED and is not a kill.

**KILL** if ANY of: (1) per-token is within 0.02 of the best window at
every T (fully converted); (2) per-token ≥ 0.05 above `postst_floor`
while the best window adds < 0.05 over per-token (converted with a
residue); (3) no gap beyond 3 σ_null at any T; (4) the gap does not
grow anywhere over T ∈ {4…64}; (5) the only window win is on the
REFERENCE arm while the within-dialogue arm is flat — i.e. the
advantage is dialogue identity, exactly the failure the qualification
predicted.

**If no rule fires** (real but under bar) the verdict is recorded as
**WEAK — no rule fires as written**, with the numbers.

**Conversion is reported as a fraction, not only as a gap**:
`(tok − floor) / (best_window − floor)` per model, with `floor` =
`postst_floor`, next to every gap.

**Every negative clause is reported with the § 2 power bound attached**
(within-dialogue contrast = 0.26–0.28 of the global one, supports
overlapping): a KILL here means converted-or-absent *at this contrast*,
and the record says so.

## 9. Deliverable

`results/screen_<model>.json` (incremental/resumable), one LOG verdict
scoring D1–D5 with the conversion fraction, the anchor differential and
the reference-vs-within-dialogue delta quoted, and a figure only if it
KEEPs (carrying the CC BY-NC-SA note). No leaderboard rows (Stage-1
screens are raw-activation probes).
