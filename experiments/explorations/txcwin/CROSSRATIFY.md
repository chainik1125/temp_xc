# Cross-ratification memo — txcwin trailing-novelty claims under task-hunt controls

**Author:** mac-b (salvage W2, `briefings/salvage-mac-b.md`).
**Status:** PENDING TEAM REVIEW and **pending Andrii's review** — every
disagreement below is a side-by-side flag, not an override. Nothing
under `txcwin/` outside `crossratify/` was modified.
**Method freeze:** `crossratify/MINI_CARD.md` at `fedf75aa9` (pre-registered
arms, predictions, and reading bands BEFORE any gap-fill computation).
**Inputs:** committed artifacts only (`results/focus_novresid.json`,
`results/focus_nov_8b.json`, `results/rawgate_gpt2_L6.json`,
`claims.jsonl`, `audit.py`, `sweep.py`, `focus.py`, `rawgate.py`) +
gap-fill outputs under `crossratify/results/`.

## 0. Verdict table

| claim | on gpt2 (the thread's primary) | on the 8B replication |
|---|---|---|
| r1 (retraction, switch_clock) | RETRACTION SOUND — reproduced from `rawgate_gpt2_L6.json` | n/a (gate never ran on 8B; see gap G-1) |
| c1 (post@T8 > per-token SAE, matched budget) | **SUPPORTED** (15σ, strict) | SUPPORTED-WITH-GAPS (2.6σ; named gaps G-1..G-4) |
| c2 (post@T8 > T-SAE) | **SUPPORTED** (21.9σ, strict) | SUPPORTED-WITH-GAPS (2.7σ; same gaps) |
| c3 (post@T8 > Stacked@T8) | **SUPPORTED** (11.3σ, strict) | **NOT-REPRODUCED at the pinned T=8** (1.9σ, non-strict, fails their own audit W3/W8); SUPPORTED at T=16 (12.4σ, strict) |
| c4 (budget-qualified TXC-pre note) | **SUPPORTED** — every number reproduces exactly (pre l0 144.4@T16, 551.0@T32; matched-budget winner is post@T8) | consistent (pre l0 153@T16 excluded there too) |

"Strict" = worst winner seed > best comparator seed (their W8), so no
seed choice can flip the sign.

## 1. What their design already controls (credit where due)

Independently reverified, not just read off their audit:

1. **Doc-level 80/20 split** (`score_task`, split seed 7) — no document
   shared between probe train and test; identical rows across all arms
   (row sample seed 11), so W6 comparisons are paired.
2. **3 trained seeds** (1, 2, 42) per cell, seed-level SE + worst-vs-best
   in the audit.
3. **Untrained control per (arch, T)** (single seed; minor gap G-3).
   Post@T8 over-init: +0.423 (gpt2), +0.388 (8B).
4. **Measured budget matching** (`calibrate_k`, realized-l0 in every
   cell; W5 ≤ 2×; worst ratio in the claim set 1.23). The one
   incalibrable arch (TXC-pre at large T) is excluded from headlines
   and says so in c4 — this is the honest version of the comparison.
5. **Raw-probe floor gate** (`rawgate.py`) exists, has teeth (it
   retracted r1 before publication), and novelty_resid passes it on
   gpt2 at T∈{4,16} (gap_mean +0.061 / +0.153).
6. **Label-build triage** of the token-identity channel (unigram
   tercile-AUC ≈ 0.56 — weak) and a scalar position check.
7. **A self-audit harness with a claims ledger** (`audit.py` +
   `claims.jsonl`) — claims are machine-checked against artifacts. Our
   whole audit reduced to running it plus recomputing it independently;
   both agree everywhere. This discipline is why cross-ratification was
   cheap; it should be the house standard.

Reproduction detail (my independent recompute, seed means, gpt2@T8):
post +0.4629 (min +0.453) vs per-token +0.2152 (max +0.243), T-SAE
+0.1931 (max +0.211), Stacked@T8 +0.2012 (max +0.242). Stacked is FLAT
in T on gpt2 (0.201–0.209 across T=4..32): merely seeing a window does
nothing here without the shared cross-position code — the cleanest
support for c3's interpretation in the whole artifact set.

## 2. Gaps found (named), and what gap-fill measured

### G-1 — raw gate never ran at the claims' T=8, nor on the 8B at all
Their gate ladder was gpt2 T∈{4,16} only. Filled per card GAP-B
(`crossratify/rawgate_fill.py`, their `raw_arms` verbatim; Modal run at
freeze `fedf75aa9`): **[PENDING — Modal job in flight; numbers land in
`crossratify/results/rawgate_fill_{gpt2_L6,8b_L12}.json`]**

### G-2 — no window-surface visible-cue baseline (the dq lesson)
Filled per card GAP-A (`crossratify/visible_cue.py`; their rows, split
and probe verbatim; features from window `token_ids` only). The V-win
arm (window-computable features jointly, no position) was added
post-freeze on mac-local's review ruling (LOG `56654864d` item 3) and
is disclosed as such; all pre-registered arms re-verified
byte-identical across the rerun. At the claims' **T=8**, on
`nov_resid`:

| arm (T=8) | gpt2 rows | llama31 rows (8B) |
|---|---|---|
| V-rep (window repetition surface) | +0.058 | +0.060 |
| V-uni (token-identity prior, their estimator) | +0.044 | +0.084 |
| **V-win (V-rep + V-uni jointly — window-computable floor)** | **+0.054** | **+0.097** |
| V-pos (document position — NOT window-computable) | +0.207 | +0.172 |
| V-all (everything incl. position) | +0.152 | +0.175 |
| best per-token dictionary | +0.215 | +0.129 |
| TXC-post@T8 | **+0.463** | **+0.393** |

**Two readings, per the review ruling (both stated; neither alone):**

- *By the card's letter* (V-all is the pre-registered gate quantity):
  gpt2 = band 3 (V-all 0.152 < per-token 0.215 — surface-quiet
  confirmed outright); **8B = band 2** ("partially window-visible",
  V-all 0.175 ≥ per-token 0.129). An earlier commit message
  (`2e163e126`) said "band 3 both models" — that sentence was not
  licensed as written and is corrected here.
- *By decomposition*: the 8B band-2 trigger is entirely the
  ORACLE-position arm; absolute document position is not computable
  from window tokens and is not a visible cue a window reader could
  count. The operative window-computable floor V-win@T8 is +0.054 /
  +0.097. Quoting guard (ruling `46e0021a7`): on gpt2 the floor is
  below the per-token dictionary with CI separation (+0.054
  [−0.002, +0.104] vs +0.215); on the 8B the floor-vs-per-token gap is
  NOT CI-separated (+0.097 [+0.048, +0.146] vs +0.129), so the
  quotable 8B statement is floor-vs-best-dict: **TXC-post ≈ 4× the
  window-computable floor (+0.393 vs +0.097)**. The dq-style objection
  (a reader counting visible cues) is genuinely absent at T=8: the
  label depends on first occurrence in the whole document prefix, a
  T=8 window sees 31.2% of the kernel mass, and the measured
  repetition floor says the window cannot fake it. **"Surface-quiet at
  window scale" may be quoted only in this decomposed form, with the
  position caveat attached.**

Disclosures that must travel with the result:
- **T=16 nuance (stated plainly):** window surface rises steeply at
  T=16 — V-rep +0.20/+0.22, V-win +0.121/+0.184 (window covers 53.3%
  of kernel mass); the surface fraction grows with T, dq-like. BUT
  unlike dq the dictionaries stay clear of the FULL visible arm at
  T=16: 8B post +0.507 ≈ 2× V-all +0.247; gpt2 post +0.417 ≈ 2× V-all
  +0.212. The quotable comparison at any T is
  **dictionary-vs-V-all-at-that-T**. Anyone re-pinning the 8B claims
  at T=16 (G-4) must carry this.
- **V-pos prediction FAILED** (card predicted ≈ 0): the
  position-detrended label retains a position-readable residual of
  r ≈ +0.21 (gpt2) / +0.17 (llama31) under an 8-feature bin readout,
  even though the thread's own position triage looked clean
  (tercile-AUC 0.472 ≈ chance) — the two results CONTRADICT and are
  flagged side by side for Andrii. Mechanism (unverified): bin-mean
  detrend on the builder's own split leaves within-bin slope +
  bin-recalibration readable. This instrument caveat travels with
  EVERY absolute skill quote from this thread; c1–c3 are head-to-head
  between arms with equal access to position, so the COMPARISONS
  stand, but the "position-free" description of nov_resid should be
  softened. (Verified benign by a second reader: V-pos byte-identical
  across T is the support-64 NaN mask making eligible rows
  T-invariant, not a bug.)

### G-3 — untrained controls at a single seed
Minor: W4 margins (+0.39–0.42) are ~10× any plausible init spread. Not
filled; proposed as a cheap add-on if Andrii wants belt-and-braces.

### G-4 — the 8B claims are pinned at the wrong T, and one 8B seed is sick
`claims.jsonl` pins T=8 and names no model; the report's 8B quote
("0.507 vs 0.129") is the **T=16** cell. At T=8 on the 8B, TXC-post
seed 1 collapsed to +0.198 (other seeds +0.478, +0.504; its bootstrap
CI [+0.155, +0.446] is anomalously wide) — that one seed makes c3@T8
fail their own W3 (1.9σ) and W8 (non-strict), while T=16 passes
everything by a mile (post min-seed +0.472 vs stacked max-seed +0.224).
Flags, in order of preference (Andrii's call):
1. Amend claims.jsonl to name model + T per claim; pin the 8B claims
   at T=16 (carrying the T=16 repetition-floor disclosure from G-2).
2. Or top up 8B@T8 seeds (3→6) to resolve c3@T8 directly (~$5,
   deliberately NOT run here — that is their science, not our control).
3. As committed, an auditor running their own `audit.py` on
   `focus_nov_8b.json` gets "CLAIM CONTRADICTED" for c3 — this should
   not be left for a reviewer to find.

### G-5 — reproducibility nits
- `audit.py`'s default `--pattern sweep_*.json` matches nothing in the
  committed repo (triage sweeps were never committed); the committed
  audit trail only works via `--pattern 'focus_*.json'`. One-line fix
  or a README note.
- The raw floor sits ABOVE every trained dictionary on gpt2
  (raw_last +0.572 vs best dictionary +0.463): every ~18-latent code is
  lossy against the 768-dim residual. Their post-r1 claim structure
  (dictionary-vs-dictionary at matched budget, gated on the raw
  asymmetry) is internally consistent — but the report should state
  this floor relationship explicitly; a reviewer will find it in
  `rawgate_gpt2_L6.json` in five minutes, as we did.

### G-6 — `report.py`'s embedded self-audit pools models into pseudo-cells
Found by a second reader during review (LOG `56654864d` addendum,
whose independent recompute converges with this audit everywhere
else): `report.py` invokes `audit.py --pattern 'focus_*.json'`, which
loads the gpt2 AND 8B novelty files together — `find()` then averages
them into single **6-seed pseudo-cells** spanning two subject models.
The pooled run reports c3 "CLAIM SURVIVES" (4.6σ, W8 WARN only) while
the 8B-only run reports "CLAIM CONTRADICTED" (G-4). The embedded audit
should run per-file; as shipped, the report's green audit block masks
the one claim its own harness rejects. Flagged for Andrii side by
side.

## 3. Receipts proposals (quotable only after ratification + Andrii's ack)

- **R-X1 (reproduction):** gpt2@T8 c1/c2/c3 margins +0.248/+0.270/
  +0.262 at 15/21.9/11.3σ, strict; from committed artifacts at
  `fedf75aa9`'s parents, checker-ready numbers in
  `focus_novresid.json`.
- **R-X2 (surface-quiet, decomposed form ONLY per rulings):** T=8
  window-computable floor V-win +0.054/+0.097; gpt2: floor <
  per-token dict (+0.215) with CI separation; 8B: floor-vs-best-dict
  form, TXC-post ≈ 4× floor (+0.393 vs +0.097; floor-vs-per-token not
  CI-separated there); by card letter gpt2 band 3, 8B band 2 via the
  oracle-position arm; quotable with the position caveat attached
  (`visible_cue_{gpt2,llama31}.json`).
- **R-X3 (instrument):** nov_resid position residual +0.207/+0.172
  (V-pos arm, same files) — travels with every absolute skill quote
  from the thread; contradicts the thread's own position triage.
- **[R-X4 raw-gate at T=8 both models — pending GAP-B numbers.]**

## 4. Bottom line for the salvage decision

On gpt2, the txcwin trailing-novelty result survives every control we
hold task-hunt work to — including the two dq died of (visible-cue
floor, and the raw gate it already had) — with margins that no seed
choice can flip. It is currently the strongest surface-quiet T-scaling
evidence in the program, where "surface-quiet" is quoted in the
decomposed form of G-2 (window-computable floor ≪ every dictionary at
T=8; oracle-position residual disclosed as an instrument caveat). The
8B replication is real but mis-pinned: robust at T=16, fragile at the
claimed T=8 (one sick seed, c3 contradicted by their own harness — and
masked by the pooled embedded audit, G-6). Fix is a claims amendment
or a $5 seed top-up, not a redesign. Gate closure at T=8/8B rides on
GAP-B (in flight).
