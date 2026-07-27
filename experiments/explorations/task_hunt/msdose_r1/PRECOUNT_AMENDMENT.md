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

---

## 7. VERDICT (2026-07-27, appended after the run — freeze sections § 1–6 untouched)

**KILLED under the § 4 bands, $0, no screen.** Run at the freeze commit
(`1f130f3cd`, receipt in `msdose_r1_premeasure.json`, frozen files
clean). Bands 2 and 3 missed on **3/3 tokenizers** — the kill rule
fires as written.

Realised numbers (gpt2 = gemma2 — identical grids, delim len 3, an
internal-consistency check matching runpod-a's stats; llama31 delim 2):

| quantity | § B sim | realised |
|---|---|---|
| FROZEN baseline census | 2/31 strata, 86,568 usable | **5/33, 201,462** (llama31 4/32, 164,003) |
| r1 pooled ρ | 0.844 | **0.838** ✓ band 1 |
| r1 census | 10/66, 397,481 | **15/74, 489,452** (llama31 486,669) |
| realised gain (usable) | ×4.6 claimed | **×2.43** (llama31 ×2.97) |

- **Absolute legs ALL passed**: ρ ≤ 0.87 ✓, strata ≥ 8 ✓ (15),
  usable ≥ 250k ✓ (489k). The candidate beat its own simulated bound.
- **Ratio legs ALL failed**: 15 < 4×5 = 20 strata (llama31 15 < 16);
  489,452 < 3×201,462 = 604,386 (llama31 misses by 1.1%: 486,669 vs
  492,009).
- Clean elsewhere: unigram 0.505 [0.493, 0.516]; docmean 0.785 (as
  disclosed § 4); floors ≤ 0.516; position AUC 0.974 (high pooled AUC
  pre-stated as expected, not a criterion).

**Why the ratio legs failed — an erratum against § B, not a defence of
the candidate:** my § B plan-level simulation UNDERSTATED the frozen
plan's realised usable mass 2.3× (86.6k sim vs 201.5k realised; the
qualifying-strata count is threshold-sensitive at the ≥ 50 cut, and the
sim idealised delimiter tokens, truncation at short source docs, and
the eligibility mask). The ratio legs encoded § B's relative claim;
that claim was wrong at the baseline, so the legs bound 2.3× tighter
than intended. Recorded plainly — and NOT used to overturn the frozen
rule.

**Re-entry path** (anyone's, fresh card only): a third entry could
pre-register absolute-mass bands — they passed everywhere. **My
recommendation as author of both msdose entries: don't.** Measured
twice now, running dose is intrinsically position-like (pooled position
AUC 0.974 even under the per-doc scale), the realised gain over the
killed plan is 2.4×, not 4.6×, and § B's honest-limits paragraph
("may simply not be worth the harness") now has numbers behind it. The
screen slot is better spent on `dharm`.

_Recorded-by: claude-fable-5 (mac-c)_
