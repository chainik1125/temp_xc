# FROZEN card — the R11 ORDER-MECHANISM LADDER (`dialevel`, day-2 W1)

**Status: FROZEN at commit (commit-then-run; NO ladder cell has been
executed).** Agent: mac-b (executor). Briefings:
`briefings/day2-dialogue-shared.md` + `briefings/day2-dialogue-mac-b.md`
(Han, 2026-07-26). Executor: `ladder.py`; read-only scorer:
`ladder_score.py` — both committed with this card, BEFORE any cell.

**The question.** R11: dialevel's within-dialogue window readout loses
**+0.0567 / +0.0626 / +0.0349** AUC (gpt2 / gemma2_2b / llama31_8b) at
T = 32 when the anchor-fixed context is fully shuffled
(`results/screen_<model>.json`, `wd/T32/win_linear −
wd/T32/win_shuf_linear`; at T16: +0.0315 / +0.0252 / +0.0281). This is
the ONE measured order-carried window signal outside backtracking —
and slen (R20) killed "generic recency" as its mechanism. What in the
dialogue context carries it: **turn ORDER (L2), within-turn token
order (L1), or a recency profile that only expresses on dialogue
(L3)?** Convert R11 from counterexample into mechanism.

## 1. Substrate, rows, identity (nothing new is designed)

Caches rebuilt in-container by the frozen builder
(`dialevel/cache_acts.py` — flat↔windowed mapping verified row-by-row
before the forward pass, the family convention). Rows are the frozen
screen's `build_rows()` verbatim (within-dialogue min-vs-max `tlevel`,
per-dialogue balance cap 8, caps 4000/1500 per class, `MATCH_SEED`
1013 + crc32 — fully deterministic given the committed labels npz), so
the ladder reads the SAME rows as the R11 cells by construction. The
within-dialogue control is the readout itself here — dialogue identity
carries exactly zero label information by construction (screen CARD
§ 4); no additional identity arm is needed and none is run.

**Four identity receipts, all hard gates before any cost is quoted:**

1. Builder's own flat↔windowed token verify (all rows).
2. Ladder's anchor-identity assert: for every shipped row,
   `token_ids[flat_anchor] == cache_ids[cache_row, cache_pos]`, and
   in-chunk content offset ≥ T_max − 1 (window never leaves the row).
3. `base/win_linear` on the rebuilt cache must match the committed
   screen value within **± 0.010** AUC at both T (probe stack frozen,
   fits deterministic; slack = fresh forward pass on different GPU).
4. **L0 seed-0 is the screen's EXACT shuffle** (`SHUF_SEED` 1234 +
   crc32(`wd/T{T}`), one generator, train drawn before test): its cost
   must land within **± 0.015** of the committed R11 cost at T = 32,
   per model. Additionally the T16 label-permutation null replicates
   the screen's exact draw (`NULL_SEED` 99) and `L4` replicates
   `capacity_check.py`'s exact foreign draw (`FOREIGN_SEED` 4242 + T /
   + T + 1) within ± 0.010.

**If gate 3 or 4 fails on any screened model, the run STOPS and the
verdict is REPRODUCTION FAILURE with both numbers quoted. Nothing
downstream is interpretable and none of it is reported as mechanism.**

## 2. The ladder (all arms, frozen)

Per model, per **T ∈ {16, 32}** (T32 = the R11 anchor; T16 = the
robustness point), on the wd rows, screen layer (gpt2 hs7, gemma2 hs14,
llama31 hs14), probe = frozen `conversion_depth.problib.fit_probe`
linear on the **T·d flatten** (`class_weight=True`, binary rank-AUC) —
the SAME probe class for every arm (matched-probe-class rule; no
max-over-arms anywhere). **Anchor slot T−1 is fixed in every arm**;
all shuffles act on the T−1 context slots only.

| arm | permutation of context slots | seeds |
|---|---|---|
| `base` | none (identity) | — |
| `L0` full shuffle | uniform over all slots | s0 = screen-exact, s1, s2 |
| `L1` within-turn | uniform WITHIN each `turn_idx` group; turn sequence intact | 3 |
| `L2` turn-block | maximal same-`turn_idx` runs = blocks; block ORDER permuted, within-block order intact | 3 |
| `L3f` far-half | uniform within slots 0 … h−1 only (h = (T−1)//2) | 3 |
| `L3n` near-half | uniform within slots h … T−2 only | 3 |
| `L4` foreign | context slots from a DIFFERENT row (width null; anchor true) | capacity_check-exact |
| `null` | label permutation on train (win unshuffled) | 1 per T |

Slot → turn mapping comes from the committed labels npz `turn_idx`
via the flat-index reconstruction (slot j ↔ flat anchor − (T−1) + j;
windows never cross a dialogue boundary, asserted). Seeds s1, s2 and
all L1/L2/L3 draws: `MATCH_SEED` + crc32(`dialevel/ladder/<arm>/T<T>/
s<s>/<split>`). Permutations are uniform INCLUDING identity (the
standard convention; the diluting identity fraction is disclosed, not
silently corrected).

**cost(arm) = auc(base) − auc(arm)**, quoted per seed and as the
3-seed mean with min–max spread, always beside the label-permutation
null deviation |auc_null − 0.5| for that T.

**Entropy disclosures (per model, per T, stored in the results):**
mean distinct turns per window, mean block count, fraction of context
slots inside shufflable (≥ 2-token) same-turn groups (L1's reach),
fraction of rows whose block permutation is non-identity (L2's reach),
and the mean moved-slot fraction of every arm's realized permutations.
L1/L2 are structurally weaker shuffles than L0 at these T (~1–2 turns
per window, turn ≈ 14.5–15.7 tokens); the disclosures quantify that
rather than letting the verdict hide it. L3 half sizes are unequal by
one at even T−1 parity (T32: far 15, near 16) — disclosed, not
corrected.

## 3. Pre-registered predictions (scored either way)

- **P-L0 (gate):** L0 reproduces R11 (§ 1 gate 4).
- **P-DECOMP:** cost(L1) + cost(L2) ≈ cost(L0) at T32 (clean
  decomposition). A large shortfall (sum < ⅔·L0 on 2/2) means the
  carrier needs CROSS-turn token mixing to destroy — reported as its
  own finding, not narrated away.
- **P-MECH (weak prior, stated to be falsifiable):** MIXED with
  cost(L2) ≥ cost(L1) — the plausible carrier is the turn-boundary
  LAYOUT (turn lengths written as boundary spacing in window
  coordinates), which both L1 and L2 degrade but L2 degrades more
  (block edges move). And cost(L3n) > cost(L3f): the near turns are
  the label's support.
- **P-NULL:** L4 sits far below base (committed: 0.583–0.622 vs base
  0.729–0.748 at T32) — width alone explains none of the R11 cost.

## 4. Verdict rule (frozen; all five outcomes stated before running)

Evaluated on 3-seed MEAN costs at **T = 32**. Robustness: the chosen
outcome's DEFINING inequalities must hold in SIGN at T16 on the same
2/2 (cost of every named arm > 0; for RECENCY-RESIDUAL, cost(L3n) −
cost(L3f) > 0); if not, the verdict is **UNRESOLVED (T16
disagreement)** with the T32 outcome quoted as the point estimate. "2/2" = gpt2 AND
llama31_8b (the briefing's screened pair). gemma2_2b (largest R11
cost, +0.063) is COVERAGE: agreement upgrades the verdict to 3/3;
disagreement keeps the 2/2 verdict but flags it as a coverage split in
the LOG. Precedence order as listed:

1. **REPRODUCTION FAILURE** — § 1 gate fails. Stop.
2. **TURN-STRUCTURE** — cost(L2) ≥ ½·cost(L0) on 2/2 AND
   cost(L1) < ⅓·cost(L0) on 2/2.
3. **WITHIN-TURN** — cost(L1) ≥ ½·cost(L0) on 2/2 AND
   cost(L2) < ⅓·cost(L0) on 2/2.
4. **MIXED** — cost(L1) ≥ ⅓·cost(L0) AND cost(L2) ≥ ⅓·cost(L0)
   on 2/2 (both carry a real share; includes the both-≥-½ case, i.e.
   non-additive destruction).
5. **RECENCY-RESIDUAL** — neither 2–4 fires, AND on 2/2:
   cost(L3n) ≥ 2·cost(L3f) AND cost(L3n) − cost(L3f) ≥ 0.02 AND
   cost(L3n) ≥ ⅓·cost(L0).
6. **UNRESOLVED** — none of the above. A publishable outcome; the
   largest fraction is NOT narrative-upgraded into "the mechanism".

Every cost in the verdict is quoted beside its null band (label-perm
deviation + seed spread). Any negative clause inherits the screen
CARD § 2 power bound (within-dialogue contrast = 0.26–0.28 of the
global tercile contrast) — a small cost here is small AT THIS
CONTRAST, and the record says so.

## 5. Economics (frozen)

L40S, one container per model, caches built in-container (idempotent,
meta-written-last), results incremental/resumable, mirrored to the
Volume, repatriated — containers never push. Models: gpt2 →
llama31_8b → gemma2_2b (secret `hf-token`; per the day-2 amendment
gemma is GO and carries the largest R11 cost, so 3-model coverage is
taken if the clock allows). Probe fits are seconds each (≤ 40
fits/model); the forward passes dominate. **Est ≤ $6 total** of
mac-b's $60 day-2 cap; hard stop per shared briefing: no new Modal
starts after 15:30 London, everything pushed by 16:30.

## 6. Deliverable

`results/ladder_<model>.json` (incremental), one LOG verdict entry
(`mac-b (executor)`, PENDING TEAM REVIEW) scoring P-L0/P-DECOMP/
P-MECH/P-NULL and naming exactly one § 4 outcome, receipts row(s) for
the quotable numbers (ratification before quoting, per the 2026-07-26
process ruling), ledger lines before/after launch.
