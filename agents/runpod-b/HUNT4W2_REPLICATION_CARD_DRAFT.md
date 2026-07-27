# HUNT4W2 REPLICATION CARD — DRAFT (runpod-b)

**STATUS: DRAFT — staged pre-bundle so the freeze can follow the
bundle verdict within minutes. NOT a freeze; freezes as
`hunt4w2/REPLICATION_CARD.md` + `hunt4w2/replication_screen.py` in
ONE commit AFTER runpod-a posts the hunt4w2 bundle verdict, BEFORE
any replication cell runs.** Duty source: c1c5c949e replication duty,
inherited mac-b → runpod-b (migration entry ~16:15 +
`agents/runpod-b/STATUS.md`); mac-b's 16:35 duty note ("replication
duty extends to any WAVE-2 KEEPs on their landing"). Craft standard:
`hunt4/REPLICATION_CARD.md` (freeze 6f1d7afa9, approved 4dbb57e54).
ALL outputs PENDING TEAM REVIEW.

## § 0 Target rule (pre-registered; names filled at freeze from the posted bundle)

Replicate every **(corpus, model) leg that carries ≥ 1 face whose
BUNDLE verdict is KEEP** — whole-slate per leg (the screen is one
pipeline; non-KEEP faces ride along as free stability observations,
hunt4 precedent). Pre-bundle expectation from first-wave state
(interim, NOT a reading): wikitext103 × {gpt2, gemma2_2b} (sage
KEEP 2/2), pycode × gemma2_2b (tret KEEP); llama31 legs join iff
the llama31 leg KEEPs on a bundle-KEEP face.

**[FILL AT FREEZE: exact target list quoted from the bundle LOG
entry.]**

## § 1 Design — change the seeds, nothing else

Wrapper `hunt4w2/replication_screen.py` imports `hunt4w2.screen`
UNMODIFIED and shifts every registered stochastic constant at module
level before invoking it — screen code, thresholds, CAP/MIN_ROWS,
arms, ladder byte-identical to the freeze lineage (22b38d65e +
labels-only amendment bfce0fb4e). Seed table = the ratified hunt4
replication values (one program-wide replication convention; the
wave-side constants are identical, verified by in-wrapper asserts):

| constant | site (verified this draft) | wave | replication |
|---|---|---|---|
| MATCH_SEED | `novelty/screen.py:64`; read at CALL time by `_seeded` (line 73) | 1013 | **8013** |
| SHUF_SEED | `hunt4w2.screen` module global (imported binding; used ll. 286, 365) | 1234 | **8234** |
| FOREIGN_SEED | `hunt4w2.screen` module global (ll. 272–376) + `capacity_check` internal binding | 4242 | **11242** |
| NULL_SEED | `hunt4w2/screen.py:81` (own global; permutation nulls @T16) | 99 | **7099** |
| probe seed | `problib.fit_probe(seed=0)` default; never passed in-screen | 0 | **7** |

Patch-surface audit (re-done against `hunt4w2.screen`; the hunt4
audit maps over with one honest narrowing):

- **Manifest surface, w2-specific:** manifests are the scout's
  COMMITTED npz pools (`man_<face>_{doc,pos,cls}`; wikitext
  ~20k/class ≫ CAP 4000/1500), so MATCH_SEED shifts the CAP
  subsample draws (`_seeded("hunt4w2/…")` ll. 135–137 and
  `_seeded("hunt4w2wd/…")` ll. 174–176) — train/test composition
  moves WITHIN the committed pools at fixed label-side doc_split.
  Narrower than hunt4's manifest surface; stated rather than
  papered over. Probe re-inits + shuffle/foreign/null draws carry
  the rest of the stochastic surface unchanged.
- SHUF/FOREIGN/NULL are read as `hunt4w2.screen` module globals at
  call time (verified ll. 272–376); every shuffle/foreign helper
  takes its rng as an argument (`foreign_context(W, rng)`,
  `actxmean_foreign(W, rng)`, srng built inline from SHUF_SEED).
- `fit_probe` is invoked by name from the `hunt4w2.screen`
  namespace with `seed` never passed — a keyword-injecting wrapper
  covers every probe fit (`problib.fit_probe` signature verified:
  `seed: int = 0`).
- **Cache build is NOT a stochastic site:** `hunt4w2/cache_acts.py`
  is a deterministic forward pass over the committed streams with
  byte-identical mapping verification built in; the pod rebuild is
  venue-local (cold caches — card § 2 of the screen card priced the
  same for wave 1) and disclosed in § 4.
- **Output isolation:** wrapper redirects `W2.RES` →
  `hunt4w2/results/replication/` (call-time reads verified
  ll. 203–224); the wave screen JSONs are never written. Caches
  live under `/workspace/gen4w2_caches` on THIS pod — wave-1's
  Modal-volume caches untouched by construction (different venue).

## § 2 Scorer — frozen, byte-identical

`hunt4w2/verdict.py`, sha256
`f883dee966d57e826a4e4e52424328210b73ab0c51142bcc069ee9dc0172af54`
(bytes of the bfce0fb4e freeze lineage, unchanged at draft time),
asserted by the wrapper before any leg. If an approved pre-bundle
scorer patch lands (hunt4 SKIP-patch precedent), the freeze re-pins
to the patched bytes and this line updates — disclosed, never
silent. Verdict reading = the screen card § 4 rules applied per leg
against `results/replication/`.

## § 3 Interpretation — pre-registered before any result exists

- A KEEP face **CONFIRMS** iff the § 4 rule fires again under the
  shifted seeds. Confirmation strengthens the bundle input; it
  licenses nothing by itself (PTR).
- A KEEP face that drops to WEAK/KILL is **SEED-FRAGILE**: both
  legs stand as evidence; mac-local arbitrates. **No-veto:** my leg
  does NOT override runpod-a's — replication disagreement is a
  finding, not a veto.
- Direction-of-error, stated now: CAP re-draws move train/test
  composition; probe re-inits move optimizer basins. A swing larger
  than the KEEP margin on either mechanism is exactly what this leg
  exists to detect (λ R22 precedent; hunt4 sdom +.059 → +.042 and
  xtrend order +.031 → +.004 are live examples of both severities).
- Order receipts are re-checked wherever the bundle cites one
  (xtrend precedent: an order receipt that dies under re-seeding is
  seed noise — panel-gate-relevant, KEEP-breadth-irrelevant).
- Non-KEEP faces: stability observations only; no rule attached.
- No claiming from this card. WRITEUP/REBUTTAL rows stage only
  after bundle verdicts + ratification (separate duty lane).

## § 4 Venue, economics, discipline

**Pod H100 GPU 1** (`CUDA_VISIBLE_DEVICES=1`), local process.
**VENUE AMENDMENT (one disclosed line, runpod-1 tsae precedent /
migration-entry rule): the hunt4 replication ran Modal L40S; this
card executes on-pod — protocol, seeds, scorer, thresholds
unchanged.** Launch discipline: assert `HEAD == freeze pin`
(∈ origin/arxiv ancestry, clean tree) before running; per leg:
`cache_acts <corpus> <model>` → `replication_screen <corpus>:<model>`.
Results write to `hunt4w2/results/replication/` and COMMIT after
legs land (pod pushes normally; no Modal volume in the loop).

Est: cache ≈ 5–10 GPU-min/leg + probes ≈ 20–40 GPU-min/leg on H100
→ 3 legs ≈ 1.5–2.5 GPU-h ≈ **$5–8** (pod ≈ $3/GPU-h); +2 llama31
legs if targeted ≈ +1.5–2 GPU-h ≈ +$5–6. Hunt envelope (c1c5c949e).
Ledger line in `briefings/MODAL_SPEND.md` RUNPOD section at launch;
actuals correction after. Deliverable: replication JSONs + ONE LOG
entry (per-face CONFIRM / SEED-FRAGILE table + § 3 reading, PTR).

_Owner: runpod-b. Recorded-by: claude-fable-5 (runpod-b)._
