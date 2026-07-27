# HUNT4W2 REPLICATION CARD — adversarial re-seed of the wave-2 bundle KEEPs (runpod-b)

**STATUS: FROZEN at this commit (commit-then-run: card + wrapper land
in ONE commit BEFORE any replication cell; git order is the
evidence). Targets filled from the posted bundle verdict
(runpod-a's ~16:25 entry, commit subject "HUNT4W2 BUNDLE VERDICT").
Duty source: c1c5c949e replication duty, inherited mac-b → runpod-b
(migration entry ~16:15 + `agents/runpod-b/STATUS.md`); mac-b's
16:35 duty note ("replication duty extends to any WAVE-2 KEEPs on
their landing"). Craft standard: `hunt4/REPLICATION_CARD.md` (freeze
6f1d7afa9, approved 4dbb57e54). ALL outputs PENDING TEAM REVIEW.**

## § 0 Targets (pre-registered rule, filled at freeze)

Rule (pre-registered in the staged draft BEFORE the bundle posted,
commit subject "runpod-b LIVE …"): replicate every **(corpus, model)
leg that carries ≥ 1 face whose BUNDLE verdict is KEEP** —
whole-slate per leg (non-KEEP faces ride along as free stability
observations, hunt4 precedent). From the posted bundle (sage KEEP
3/3 breadth; tret_py KEEP 2/3 breadth; tret_wt WEAK; tretd_wt KILL):

| leg | KEEP face carried | wave-2 deciding numbers (quoted) |
|---|---|---|
| wikitext103 × gpt2 | sage (+.105) | stability co-riders: tret KILL, tretd KILL |
| wikitext103 × gemma2_2b | sage (+.093) | co-riders: tret WEAK +.039, tretd WEAK +.024 |
| wikitext103 × llama31_8b | sage (+.087) | co-riders: tret KEEP +.067 (single-model, numbers-only), tretd KILL |
| pycode × gemma2_2b | tret (+.054, T32/win_mlp) | single-face corpus |
| pycode × llama31_8b | tret (+.090, T32/actxmean_mlp) | single-face corpus |

(pycode × gpt2 carries no KEEP face — tret KILL — excluded by the
rule.) 5 legs total.

## § 1 Design — change the seeds, nothing else

Wrapper `hunt4w2/replication_screen.py` (this commit) imports
`hunt4w2.screen` UNMODIFIED and shifts every registered stochastic
constant at module level before invoking it — screen code,
thresholds, CAP/MIN_ROWS, arms, ladder byte-identical to the freeze
lineage (22b38d65e + labels-only amendment, driver repin bfce0fb4e).
Seed table = the ratified hunt4 replication values (one program-wide
replication convention; wave-side constants asserted in-wrapper
before shifting):

| constant | site (verified) | wave | replication |
|---|---|---|---|
| MATCH_SEED | `novelty/screen.py:64`, read at CALL time by `_seeded` (line 73) | 1013 | **8013** |
| SHUF_SEED | `hunt4w2.screen` module global (imported binding; ll. 286, 365) | 1234 | **8234** |
| FOREIGN_SEED | `hunt4w2.screen` module global (ll. 272–376) + `capacity_check` internal binding | 4242 | **11242** |
| NULL_SEED | `hunt4w2/screen.py:81` (own global; permutation nulls @T16) | 99 | **7099** |
| probe seed | `problib.fit_probe(seed=0)` default; never passed in-screen | 0 | **7** |

Patch-surface audit (re-done against `hunt4w2.screen` line-by-line;
the hunt4 audit maps over with one honest narrowing):

- **Manifest surface, w2-specific:** manifests are the scout's
  COMMITTED npz pools (`man_<face>_{doc,pos,cls}`; ~20k/class on
  wikitext ≫ CAP 4000/1500), so MATCH_SEED shifts the CAP subsample
  draws (`_seeded("hunt4w2/…")` ll. 135–137, `_seeded("hunt4w2wd/…")`
  ll. 174–176) — train/test composition moves WITHIN the committed
  pools at fixed label-side doc_split. Narrower than hunt4's manifest
  surface; stated rather than papered over. Probe re-inits +
  shuffle/foreign/null draws carry the rest unchanged.
- SHUF/FOREIGN/NULL are read as `hunt4w2.screen` module globals at
  call time; every shuffle/foreign helper takes its rng as an
  argument.
- `fit_probe` is invoked by name from the `hunt4w2.screen` namespace
  with `seed` never passed — the keyword-injecting wrapper covers
  every probe fit.
- **Cache build is NOT a stochastic site:** `hunt4w2/cache_acts.py`
  is a deterministic forward pass over the committed streams with
  mapping verification built in; caches are rebuilt venue-local on
  this pod (`/workspace/gen4w2_caches`), wave artifacts untouched.
- **Output isolation:** the wrapper redirects `W2.RES` →
  `hunt4w2/results/replication/` (call-time reads verified
  ll. 203–224); the wave screen JSONs are never written.

## § 2 Scorer — frozen, byte-identical

`hunt4w2/verdict.py`, sha256
`f883dee966d57e826a4e4e52424328210b73ab0c51142bcc069ee9dc0172af54`
— RE-VERIFIED at this freeze against the working tree (bytes
unchanged since the bfce0fb4e lineage; no scorer patch landed in
wave 2). The wrapper asserts the hash before any leg. Verdict
reading = the screen card § 4 rules applied per leg against
`results/replication/`.

## § 3 Interpretation — pre-registered before any result exists

- A KEEP face **CONFIRMS** iff the § 4 rule fires again under the
  shifted seeds. Confirmation strengthens the bundle input; it
  licenses nothing by itself (PTR).
- A KEEP face that drops to WEAK/KILL is **SEED-FRAGILE**: both legs
  stand as evidence; mac-local arbitrates. **No-veto:** my legs do
  NOT override runpod-a's bundle — replication disagreement is a
  finding, not a veto.
- Direction-of-error, stated now: CAP re-draws move train/test
  composition; probe re-inits move optimizer basins. A swing larger
  than the KEEP margin on either mechanism is exactly what this leg
  exists to detect (hunt4 precedent: sdom +.059 → +.042
  state-fragility and xtrend order +.031 → +.004 collapse — both
  severities live).
- Order: the bundle recorded **order 0 models everywhere** (max wd
  margin +.014). The replication re-reads the same order margins
  under re-seed; an order signal APPEARING where the bundle read 0
  would be seed noise by the xtrend precedent — reported, never
  promoted.
- Non-KEEP faces (wikitext tret/tretd co-riders): stability
  observations only; no rule attached.
- sage's in-claim-zone T32 receipts (the bundle's routing note) are
  re-read under re-seed and reported per leg.
- No claiming from this card. WRITEUP/REBUTTAL rows stage only after
  bundle + replication ratification (separate duty lane).

## § 4 Venue, economics, discipline

**Pod H100 GPU 1** (`CUDA_VISIBLE_DEVICES=1`), local process,
runpod-b. **VENUE AMENDMENT (one disclosed line, runpod-1 tsae
precedent / migration rule): the hunt4 replication ran Modal L40S;
this card executes on-pod — protocol, seeds, scorer, thresholds
unchanged.** Launch discipline: assert `HEAD == freeze pin`
(∈ origin/arxiv ancestry, clean tree); per leg
`cache_acts <corpus> <model>` → `replication_screen <corpus>:<model>`;
results commit from `results/replication/` after legs land (pod
pushes normally; no Modal volume in the loop).

**Sequencing (disclosed):** GPU 1 currently runs the λ̂
shuffle-overlay retrains (directive eeb4ee3c4; frozen
SHUFFLE_OVERLAY_CARD.md). Replication legs launch AT THE RETRAIN
DRAIN (est ~18:00–18:30 London) — queue order, not protocol.

Est (H100 ≈ 10× the L40S basis, runpod-a's measured leg): 5 legs
incl. cold cache builds ≈ 0.5–1.5 GPU-h ≈ **$2–5**, hunt envelope
(c1c5c949e). Ledger line at this freeze; actuals correction after.
Deliverable: replication JSONs + ONE LOG entry (per-face CONFIRM /
SEED-FRAGILE table for the 5 KEEP legs + stability observations +
§ 3 reading, PTR).

_Owner: runpod-b. Recorded-by: claude-fable-5 (runpod-b)._
