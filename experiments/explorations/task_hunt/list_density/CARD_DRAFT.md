# DRAFT mini-card — list/enumeration density intensity (fineweb)

**Status: DRAFT (runpod, `briefings/candidate-factory-broad.md`,
ledger `../CANDIDATES.md` B3).** Committed WITH the frozen triage bars
below BEFORE `../labels/build_punctint.py` runs; the running agent
freezes its own screen card from this draft.

Data side: builder `../labels/build_punctint.py` (logic
`../labels/punctint_lib.py`, tests `tests/test_punctint_labels.py`) →
`../labels/punctint_fineweb_{gpt2,gemma2,llama31}.npz` (+ the B4
question face in the same npz) + `../labels/punctint_stats.json`.
**Economics:** `token_ids` builder-asserted identical to the replag
npz ⇒ existing fineweb GPU caches, zero new caching.

## The candidate logic

Winner-family shape (kernel intensity) on a structural event stream:
events = sentences matching the FROZEN list-marker grammar (bullets /
numbered / lettered / parenthesized enumerators — exact regex
committed in `punctint_lib.LIST_RE` before the builder ran). Primary
`lam_list` = 8-sentence-lag, half-life-2 kernel rate over PREVIOUS
sentences only; every token inherits its sentence's λ̂; **masking
rule: tokens of list sentences are excluded from probe manifests for
this face** (they read "I am in a list" ambiently — that bit is the
disclosed regime-1 anchor `is_list`, not the primary).

Axis a: enumerator continuation is generatively useful — the ambient
in-list bit is expected converted; the bet is only the trailing
DENSITY. Axis b: topic leak (listy docs are lexically listy) — the
triage below is the gate. Axis c: 8-sentence support ≈ 160 tokens vs
panel T ≤ 32 ≈ 1.6 sentences seeing ≈ 45 % of kernel mass — the
ladder spans less than the kernel, like the Ward λ̂ winner; said
plainly. Axis d: regime-2 rise predicted, shuffle-immune.
Feasibility, disclosed: events concentrate in ~37 % of docs (median
doc has none) — the zero_split 3-class scheme is expected to fire;
manifests concentrate in listy docs; split stays by doc.

## Label-side triage — FROZEN BARS (kill authority)

On test-doc rows, masked (event-sentence tokens out), top vs bottom
class, direction-agnostic (max(AUC, 1−AUC)): current-token type-mean
AUC **≥ 0.65 ⇒ KILL**; position AUC **≥ 0.65 ⇒ KILL**; 0.55–0.65 =
ships with the elevation disclosed.

## Predicted T-pattern + draft kill rule

Per-token blind-ish on the masked primary; window − per-token gap
positive, growing over T ∈ {4…32}, order-free (mean ≈ flatten) —
regime-2. KILL at screen if: per-token reads the primary within noise
of the best window everywhere; or no gap beyond 3 σ_null at any T; or
the gap does not grow anywhere in the ladder. `is_list` runs as the
disclosed ambient anchor only.
