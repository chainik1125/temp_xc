# msdose_r1 — PRE-COUNT AMENDMENT (re-entry of the 07-27 $0 kill)

**Owner: `mac-c` (dispatch `47040da59`). Status at freeze: committed
BEFORE `build_msdose_r1` runs anything (commit-then-run). The frozen
`wave3_lib` msdose constants and runpod-a's `wave3_msdose_<tok>.npz`
artifacts are UNTOUCHED — they remain the record of the KILLED
construction. This card is the fresh pre-count the re-entry rule
requires ("a construction redesign with a measured decorrelation
bound"), not a post-hoc widening.**

## 1. The one change

Frozen plan: every doc draws exemplar lengths i.i.d. from
`LogNormal(log 120, 0.6)` — the dose↔position map is near-shared
corpus-wide, so dose IS position (realised ρ 0.962, position AUC
0.9999 on all three tokenizers; killed 07-27).

r1 plan (`msdose_r1_lib.py`): draw the scale **per document** —

```
n_ex   ~ integers(4, 25)                      # unchanged
mu_doc ~ Normal(log 120, 0.7)                 # NEW (sigma_doc = 0.7)
lens   ~ clip(round(exp(Normal(mu_doc, 0.6, n_ex))), 40, 400)  # inner unchanged
```

Everything else inherits the frozen pre-count verbatim: 400 docs, seed
0, delimiter `"\n###\n"`, `wave3_lib.msdose_doc` assembly from the SAME
committed gen4c wikitext streams, `doc_split(seed=0)`, MIN_POS 32.
sigma_doc = 0.7 (not 1.0) per § B: 1.0 saturates the gain and distorts
the exemplar-length distribution further.

## 2. Provenance / anti-tuning statement

sigma_doc = 0.7 and the census definition below were fixed in
`WAVE3_SECOND_SOURCE.md` § B (`248049349`) from a construction-plan
simulation, before this card and before any r1 corpus existed. Nothing
here is tuned against the kill numbers beyond what the kill verdict
itself published (ρ, position AUC). The § B simulated table is hereby
promoted to a **pre-registered prediction**: realised pooled ρ ≈ 0.844,
qualifying strata ≈ 10/66, usable position-matched tokens ≈ 397k.

## 3. The verdict instrument (frozen)

`msdose_r1_lib.strata_census`: global dose terciles over eligible rows
(`is_boundary == 0 & pos >= 32`), position strata = 128-token bins of
absolute position; a stratum **qualifies** iff it holds ≥ 50 rows of
EACH tercile; usable mass = eligible rows in qualifying strata.

It runs FIRST on the committed killed corpus (realised baseline — the
§ B "2/31, 86,568" was simulation-side), then on r1, so the comparison
is realised-vs-realised under one committed instrument. Triage AUCs,
tercile binning, doc-cluster bootstrap, and floors are imported from
`build_wave3_trio` unchanged — runpod-a's instruments verbatim.

## 4. Confirmation bands (pre-registered; evaluated per tokenizer)

1. pooled dose↔position Spearman ρ ≤ **0.87**;
2. qualifying strata ≥ **8** AND ≥ **4×** the realised frozen baseline;
3. usable tokens ≥ **250,000** AND ≥ **3×** the realised frozen baseline.

**Kill rule: any band missed on ≥ 2 of 3 tokenizers ⇒ `msdose_r1` dies
for $0 and no screen runs.** A construction that cannot realise its own
simulated bound is not rescued twice.

**Stated to prevent misreading:** pooled position AUC will REMAIN high
(ρ 0.844 pooled is still strong) and is NOT a pass criterion. The
design claim was never "dose decorrelates from position pooled" — it is
"enough position-matched mass exists for a cross-document readout".
Likewise `doc_mean_only_auc` will stay ≈ 0.8 (total dose is a doc
constant; unchanged n_ex distribution) — disclosed, not a kill, because
of the admissible-readout restriction below.

## 5. Admissible readout (binding on any screen)

Within-doc ρ(dose, position) ≈ 0.99 is **structural** under every
construction (§ B) ⇒ the within-document readout is inadmissible,
full stop. The ONLY admissible readout is **position-matched
cross-document**: probe rows restricted to qualifying strata, tercile
contrast within stratum, realised per-stratum tercile counts reported
in-card. The screen (if the bands pass) must pre-register the
qualifying-strata list from `msdose_r1_premeasure.json` verbatim and
carry the § 4 #4 ecological-validity caveat (constructed corpus proves
carriage, not deployment).

## 6. Cost

$0 — CPU only, committed streams, no API, no pulls, no activations.
Outputs: `labels/wave3_msdose_r1_<tok>.npz` +
`labels/msdose_r1_premeasure.json` (artifact of record, carries the
freeze receipt: HEAD sha + clean-tree assertion over the frozen logic).

_Recorded-by: claude-fable-5 (mac-c)_
