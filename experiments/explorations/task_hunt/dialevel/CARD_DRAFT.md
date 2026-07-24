# DRAFT mini-card — dialogue turn-length LEVEL (DailyDialog)

**Status: DRAFT (runpod, `briefings/candidate-factory-broad.md`,
ledger `../CANDIDATES.md` B5 — the stretch new-corpus bundle).**
Committed WITH the frozen triage bars below BEFORE
`../labels/build_dialevel.py` runs; the running agent freezes its own
screen card.

Data side: builder `../labels/build_dialevel.py` (logic
`../labels/dialevel_lib.py`, tests `tests/test_dialevel_labels.py`) →
`../labels/dialevel_dailydialog_{gpt2,gemma2,llama31}.npz` +
`../labels/dialevel_stats.json` + **the pinned corpus artifact
`../labels/dialevel_corpus.json.gz`** (DailyDialog via parquet mirror
`OpenRL/daily_dialog`, pinned revision, ≥ 8-turn dialogues, seeded
sample; original license CC BY-NC-SA 4.0 — research use).
**Economics: a NEW token stream — one caching pass per model (~0.5M
tokens, minutes on an H100); no existing cache applies.**

## The candidate logic

The grounded cousin of interleave `tss`, framed as a LEVEL per the
hedging lesson: primary `tlevel` = trailing mean turn length (in
tokens) over the PREVIOUS 5 turns, current turn excluded — "am I in a
rapid-fire exchange or a long-form stretch", a regime-2 aggregation
state. Dialogues render with single newlines between turns (the
minimal visible boundary marker); **masking rule: newline-spanning
tokens (`is_boundary`) are excluded from manifests**. The
conversion-risky faces are secondary and disclosed: `tst`
(tokens-since-turn-start — near-syntactic, expected converted) and
the boundary bit itself (newline prediction is generative).
Manifests are position-MATCHED from the start (the B3 lesson), at
**pos ≥ 16** — a stated deviation from the fineweb pos ≥ 32 floor
(dialogues run ~200 tokens; T ≤ 16 windows fit fully, T = 32 windows
truncate at early rows and the screen must left-pad or drop — the
freezing agent decides and states which).

## Clock (axis c)

Turn ≈ 18 tokens ⇒ the 5-turn support ≈ 90 tokens; the panel ladder
T ∈ {4…32} spans ~4 %→35 % of support with T = 64 optional closure —
under-spanned like the Ward λ̂ winner, said plainly.

## Label-side triage — FROZEN BARS (kill authority)

Test-doc rows, boundary tokens masked, top vs bottom class,
direction-agnostic max(AUC, 1−AUC), on BOTH all-eligible and shipped
manifest rows (manifest = operative): current-token type-mean AUC
**≥ 0.65 ⇒ KILL** (the known risk: short turns are lexically distinct
— "yes", "ok", question forms); position AUC **≥ 0.65 ⇒ KILL**;
0.55–0.65 ships with disclosure.

## Predicted T-pattern + draft kill rule

Per-token poor on the masked level; window − per-token gap positive
and growing over T ∈ {4…32} (each added turn boundary in the window
sharpens the level estimate); order-free (mean ≈ flatten) — regime-2,
shuffle-immunity as the mechanism receipt. KILL at screen if:
per-token reads `tlevel_bin` within noise of the best window at every
T; or no gap beyond 3 σ_null at any T; or no T-growth anywhere in the
ladder. `tst` and `is_boundary` run as disclosed anchor faces only.
