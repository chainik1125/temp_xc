# PROBE_V2 freeze DRAFT — the post-deadline methods-review agenda paper

**Status: DRAFT (mac-a, 2026-07-27, overnight § 6d assignment
bb2c3f8d7). Nothing here is adopted and nothing here amends
`PROBE_V2_SPEC.md` — that file stays the freeze candidate verbatim
(its § 0 limitation travels with any adoption). This draft is the
review's agenda: what has changed since the spec was written, the
two adoption options made concrete, and the updated re-run
arithmetic. Decision owner: mac-local jointly with the
variance-machinery owner, at the methods review.**

## 1. What changed since the spec was written (evidence base as of
2026-07-27 ~03:30 London; leaderboard census in this commit)

1. **The paired-row corpus now exists at scale: 537 rows carry both
   generations** (`lambda_recovery` + `lambda_recovery_v2` on the
   same row): ttrend 219, dq 102, punctint-q 132 (3 models), oprate
   84. Every hunt panel since the day-2 defect is born-paired — the
   spec's § 2 mechanism (`eval_extra` V2 block) became MANDATORY
   hunt practice with hard pre-run asserts (the defect's permanent
   fix). Adoption therefore no longer requires re-running ANY hunt
   dialogue/web panel: the v2 columns are already on the claiming
   rows (R22 top-up, R27, R28, R29, R30 lanes all carry them; the
   S5 "grouped v2 > 0" bars already gate on them).
2. **Still v1-only (the § 2 re-run table, updated counts):**
   `ward_real_lambda_base_l12` 117 rows (was 108 — the R22 tsae
   top-up added 9) and `ward_real_slope8_distill_l14` 84. These two
   Ward panels remain the ONLY committed claiming surfaces without
   paired columns. Also v1-only: the round-1 ttrend panel's 102 rows
   (superseded for claiming by the born-paired 219; historical
   record only, no re-run case) and 843 toy selfexcite rows (out of
   scope; synthetic ground-truth benches read recovery against
   constructed truth, the capacity question is moot there).
3. **R30 mechanism re-attribution interacts with § 0's regime
   description.** The calibration identity (R30) showed the realized
   density the panels actually probe is set by EVAL-TIME JumpReLU
   threshold pruning (sae realizes ~4/8, tsae ~7/8 at hunt widths) —
   an arch property, not selection starvation. § 0's "dense code"
   axis should be read against REALIZED l0, and runpod-2's EM flag
   (arch-dependent threshold OVERfire at wide-d/small-k: realized
   1056 vs nominal 20) shows realized density can also exceed
   nominal by 50×. Consequence for the review: the v1→v2 lift is
   monotone in realized density (spec § 0 mechanism receipt), so
   per-arch lift asymmetries are EXPECTED wherever thresholds gate
   asymmetrically — a v2 adoption should quote the realized-l0
   column beside any cross-arch comparison (the hunt's l0-band
   discipline already does this).
4. **Sequential-decision hygiene**: several licensed quotes now
   carry mandatory caveats (R22's two, R29's sequential sentence).
   Any restatement on v2 columns re-inherits those caveats verbatim
   — restating a number does not launder its licence.

## 2. The two options, concrete

**Option A — adopt v2 as canonical at the review.**
- Freeze `PROBE_V2_SPEC.md` as-is (its § 0 lower-bound limitation
  becomes part of the canon text; "at least X" language mandatory).
- Eval-only re-runs: the two Ward panels, 201 cells ≈ 3–4 h at 3
  workers, < 3 GPU-h encode (spec § 2 arithmetic; checkpoints all
  cache-hit — BUT see the caveat: those checkpoints lived on
  container-local disks and are NOT all mirrored; mac-c's HF mirror
  pass (overnight § 3.2) covers the hunt trio + dialogue panels;
  the WARD panel checkpoints' availability must be verified before
  costing this — if gone, "eval-only" becomes full re-train ≈ the
  original panel cost, and the option-A price changes materially).
- Re-base: `stage2_variance.py --probe v2` for λ̂ and hedging (spec
  § 3, one command each); the T = 2→8 trend p = 0.0093 and the § 3b
  "peaks rather than saturates" reading get restated from v2
  receipts, not carried over.
- Hunt quotes: NO re-runs; the v2 columns are on-row. Each licensed
  quote gets a v2 restatement PROPOSED beside it (same caveats),
  ratified quote-by-quote.
- WRITEUP: § 9 readout caveat rewritten around the lower-bound
  framing; ordering claims (the paper's actual content) unchanged
  by construction.

**Option B — keep v1 canonical; formalize born-paired as permanent
policy.**
- v1 stays bit-identical canonical; the V2 eval_extra block becomes
  a REGISTRY-LEVEL default for every future λ-family eval (today it
  is convention + per-card asserts; formalizing removes the
  possibility of another day-2 defect).
- The two Ward panels optionally gain paired columns anyway (same
  201-cell eval-only run) so the ENTIRE claiming corpus is paired —
  at that point Option A becomes a pure relabeling decision
  available at any later date with zero compute.
- Cheapest path that preserves every future option.

**Draft's observation (not a recommendation):** the compute
difference between the options is now ~zero (the 201-cell run is
wanted under both); the real decision is the LANGUAGE change
(lower-bound framing + "at least X") and where it propagates. That
is precisely a methods-review call, not an executor's.

## 3. Pre-review checklist (zero-GPU, any agent)

- [ ] Verify Ward-panel checkpoint availability (Volume/HF) before
      pricing the 201-cell eval-only run (owner: whoever runs it;
      mac-c's mirror manifest is the index).
- [ ] `tests/test_lambda_recovery_v2.py` still green on current
      HEAD (contract: ols+1024+half ≡ v1 to 1e-10). Last verified
      green in the full-suite run at Stage 1 (369 passed,
      2026-07-26).
- [ ] The smoke sweep YAML resolves
      (`configs/sweeps/lambda_probe_v2_smoke.yaml`) — run.py
      validate covers resolution; the sweep itself writes rows, run
      only when intended.
- [ ] Census re-run on review day (the numbers in § 1 date from
      this commit).
