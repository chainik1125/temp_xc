# HUNT4 REPLICATION CARD — adversarial re-seed of the wave-1 gemma KEEPs (mac-b)

**Pre-registration. Frozen BEFORE the replication run; pin in
`scripts/modal_hunt4_replication.py` from ORIGIN-history `git
rev-parse` post-push, asserted in-container.** Duty source:
c1c5c949e ("mac-b: … an adversarial-replication leg on any mac-a
KEEP (independent seed, same frozen scorer committed before the
deciding result)") + HUNT4_SCREEN_CARD § 0 ("mac-b stages
adversarial replication on any KEEP"). Targets: the three wave-1
gemma2_2b KEEPs (1c8754d7d): **tret** (+.097, T64 arm vs .394
floor), **sdom** (+.059 @T8, order receipts both models),
**xtrend** (+.064, order +.031 @T32). ALL outputs PENDING TEAM
REVIEW.

## § 1 Design — change the seeds, nothing else

One container, `gemma2_2b` only (the KEEP model), the committed
screen run whole-slate (no face restriction: the screen is one
pipeline; the two non-KEEP faces replicate incidentally — xnov's
WEAK and tretd's MIN_ROWS SKIP are themselves stability
observations, at zero marginal design cost). The wrapper
`hunt4/replication_screen.py` imports `hunt4.screen` UNMODIFIED and
shifts every registered stochastic constant at module level before
invoking it — screen code, thresholds, manifest caps, arms, ladder:
all byte-identical to the freeze lineage (35d20e3cb + bf16dfe9e).

Independent seeds (old → new, disclosed here and printed at
container start):

| constant | site | wave-1 | replication |
|---|---|---|---|
| MATCH_SEED (manifest draws via `_seeded`) | novelty/screen.py:64 | 1013 | **8013** |
| SHUF_SEED (shuffle twins) | novelty/screen.py:63 | 1234 | **8234** |
| FOREIGN_SEED (foreign context/actxmean) | dialevel/capacity_check.py:83 | 4242 | **11242** |
| NULL_SEED (permutation nulls @T16) | hunt4/screen.py:82 | 99 | **7099** |
| probe seed (`fit_probe`, torch init) | problib default | 0 | **7** |

Patch-surface audit (why module-level patching is complete): all
manifest draws route through `_seeded`, which reads
`novelty.screen.MATCH_SEED` at call time; `SHUF_SEED` /
`FOREIGN_SEED` / `NULL_SEED` are read as `hunt4.screen` module
globals at call time; every foreign/shuffle helper takes its rng as
an argument (`foreign_context(W, rng)`, `actxmean_foreign(W,
rng)`); `fit_probe` is invoked by name from the `hunt4.screen`
namespace with `seed` never passed, so a keyword-injecting wrapper
covers every probe fit. Labels (committed `.npz`), activation
caches, and the deterministic cell keys are untouched.

Output isolation: `hunt4.screen.RES` is redirected to
`hunt4/results/replication/` (Volume mirror
`/workspace/hunt4_replication/`) — mac-a's
`results/screen_gemma2_2b.json` and Volume dir are never written.

## § 2 Scorer — frozen, byte-identical

`hunt4/verdict.py` at **bf16dfe9e** (the freeze scorer + the
approved pre-deciding SKIP/zip patch, 6b03b1b06), sha256
`06a624eff6f12f5b64a53b09c360c670864eec3c3ac39d4aafa50c68fb682fac`.
This card's commit does not touch it; the container asserts the
hash before scoring. Verdict reading = the § 4 existential rule of
HUNT4_SCREEN_CARD applied to the replication JSON as a
single-model leg (same mechanical scorer, `--models gemma2_2b`
against `results/replication/`).

## § 3 Interpretation — pre-registered before any result exists

- A KEEP face **CONFIRMS** iff the § 4 rule fires again under the
  shifted seeds. Confirmation strengthens the bundle input; it
  licenses nothing by itself (PTR).
- A KEEP face that drops to WEAK/KILL is flagged **SEED-FRAGILE**:
  both legs stand as evidence; the bundle review (mac-local)
  arbitrates. My leg does NOT override mac-a's — replication
  disagreement is a finding, not a veto.
- Direction-of-error note, stated now: manifest re-draws move
  train/test composition; probe re-inits move optimizer basins. A
  swing larger than the KEEP margin on either mechanism is exactly
  what this leg exists to detect (the λ R22 s3/s4 seed-pathology
  precedent).
- The two non-KEEP faces are reported as stability observations
  only (xnov WEAK-stable?, tretd SKIP-stable?); no rule is attached
  to them.
- No claiming from this card. REBUTTAL_PACK rows for order-carrying
  KEEPs are staged only after bundle verdicts + ratification.

## § 4 Venue, economics, discipline

Modal **L40S**, one container, `--detach`, retries 1, 4 h timeout;
Volume `temp-xc-replag-caches` (dialevel gemma cache expected
cache-hit — wave-1 warmed it). Containers never push; result JSON
persists to Volume `/workspace/hunt4_replication/` after every cell
+ repatriates to `hunt4/results/replication/screen_gemma2_2b.json`.
Est **≈ $3–6** (wave-1 gemma actuals ≈ $3–4 with warm caches),
inside the c1c5c949e $200/10h hunt envelope; ledger line at launch,
actuals correction after. Deliverable: replication JSON + ONE LOG
entry (confirm/fragile per face, PTR) + this card's § 3 reading.

_Owner: mac-b. Recorded-by: claude-fable-5 (mac-b)._
