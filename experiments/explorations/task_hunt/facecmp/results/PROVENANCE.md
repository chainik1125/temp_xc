# ⚑ Corpus identity in this directory is NOT readable from `meta.substrate`

**Applies to every artifact written before 2026-07-29 ~14:2x BST.** Read this
before citing any file here, or before using one to decide anything.

## What happened

`arm_test.py` hardcoded

    done["meta"] = {"substrate": "elicit_retryesc_gen_v1 (BORROWED corpus)", ...}

`GRID_PAT` became overridable earlier that night and **the meta did not
follow**. So the substrate field was written *unconditionally*, whatever
corpus actually ran. mac-c fixed it (`369f8c24c`): substrate is now derived
via `GRID_PAT.format(...)`, and `grid_dir` + `cache_root` are recorded.

## What that means for the files already on disk

| state | count | how to read it |
|---|---|---|
| `meta.substrate` present, `grid_dir` **absent** | **22** | pre-fix. The label was written unconditionally, so **it is evidence of nothing** — not proof of retryesc, and not proof of a mismatch either. |
| **no `meta` substrate at all** | **9** | no provenance whatsoever. Includes every output of the ungridded scripts. |
| `grid_dir` present | 0 | post-fix. None yet — nothing has been re-run. |

**Do not read the 22 as "wrong".** They are *unverified in both
directions*. One is demonstrably wrong — `lever3/arm_test_gemma2_2b.json`
records `n_docs=400` (the **evalage** grid) under the retryesc label — and
that is the existence proof that the field cannot be trusted. The others
may well be retryesc; the point is that the file cannot tell you.

Note also that `gemma2_*` in a filename names the **model**, not the
corpus. A gemma2 run on the retryesc corpus is perfectly ordinary, so the
name is not itself a contradiction.

## The 9 with no provenance are exactly the blast-radius set

These are written by the scripts that override `CACHE_ROOT` without
`GRIDS`:

    lane_b/errorbar_evalage_512.json      lever3/lever3_gemma2_512.json
    lane_b/errorbar_gemma2_512.json       lever3/w_counterfactual_gemma2_512.json
    lever3/lever3_evalage_gemma2_512.json upper_edge/upper_edge_gemma2_2b.json
    amplifier/amplifier_test.json         face_battery.json
    floor_by_face_shape.json

This is why identifying `errorbar_gemma2_512.json`'s corpus required a
**tok/n_test fingerprint** (`0.3774 / 4494`) rather than reading a field:
forensics substituting for provenance that was absent or lying.

## Establishing corpus identity for a pre-fix file

Fingerprint it. `n_test` and the `tok` baseline are corpus-specific and
stable:

- **evalage grid** — `n_docs = 400`
- `errorbar_gemma2_512.json` → `tok 0.3774`, `n_test 4494`
- `errorbar_evalage_512.json` → `tok 0.4599`, `n_test 4497`

Or simply re-run: mac-c verified the fix is **metadata-only** — the evalage
pairing reproduced `tok=0.4599 best=0.5308 (T64) gain=+0.0709`, identical
to `a027b7caa`. **No numbers changed.**

## Scope of the damage

**No result in this directory is data-corrupted** (mac-c, LOG 14:2x), and
**no value produced here appears on any reviewer-bound surface** — checked
against `REBUTTAL_HANDOFF.md`, `figs_writeup/`, `docs/dmitry/` with a
firing positive control (mac-local, LOG 14:2x, `9d2277e37`). The cost of
this defect was **a night of forensics**, not a wrong claim.

## Known residual, deliberately not fixed before the rebuttal

Only **1 of 5** local caches records a grid at all (`cache_evalage_512`),
so the grid-vs-cache guard covers one fifth of them. The other four are
safe *today* because their corpus happens to match the default — **luck,
not design**. Hub ruling: recorded, not spent on before Aug 3, since
nothing shipped and no number moved. Revisit post-rebuttal.
