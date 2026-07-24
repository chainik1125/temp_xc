# Mini-card (frozen pre-screen) — proof-operation run structure

**Candidate 2, task-hunt arm A** (`briefings/task-hunt.md`; card written
by `runpod-b` per `briefings/task-hunt-prep.md` — the label deliverable
is `../labels/proofops.npz`, built by the committed
`labels/build_proofops.py`). The science below (label, non-ambience
argument, T-prediction, falsifier) is frozen by this commit; the running
agent (`runpod-d`) appends its screen-cell table (model × layer, probe
grid per the `replag/CARD.md` precedent) before running — appending
cells does not amend predictions.

## Latent + labels (frozen judge, committed record)

Per-sentence 5-class operation labels over the 300 R1-Distill reasoning
traces — the frozen record
`synthetic/expansion/records/proof-operation-phase-runs/labels.json`
(bulk `claude-haiku-4-5`, adjudication `claude-sonnet-5`; κ = 0.586,
noise floor ε̂ = 0.172; 287/300 docs, 24,386 labeled sentences; classes:
0 other, 1 algebraic-manipulation, 2 case-enumeration,
3 verification-check, 4 restatement-setup; judged sentence-in-isolation,
ctx = 0). Label-side science is settled (C3–C5): run structure is real —
self-match ACF(1) 0.286 vs N2 hi 0.038, mean dwell 1.86 sentences, and a
genuine segment-scale layer (real ACF(4) 0.127 ≫ run-permuted 0.071).
The hunt question is whether the MODEL's activations carry that run
state non-ambiently.

Targets on the canonical Ward cache grid (`../labels/proofops.npz`):

- **`is_run_start`** (binary, PRIMARY): the containing sentence starts a
  constant-operation run (boundary ⇔ op(s) ≠ op(s−1); unlabeled breaks
  runs);
- **`time_in_run`** bins {0, 1, ≥2} (secondary — run-age recovery);
- **`op`** 5-class (CONTROL, not a target): per-sentence readable **by
  construction** (gate-7) ⇒ predicted regime-1 — per-token ≈ window.
  A card claiming op-class detection dies at the gate.

All manifests balanced, valid-masked, p ≥ 32, split by trace.

## Why non-ambient (regime-3-shaped, equality subtype)

Boundary and run-age are relations BETWEEN sentences: `is_run_start`
compares the current sentence's operation to the PREVIOUS sentence's — a
cross-position equality latent (the LEDGER's interaction/equality axis;
the recipe-residual precedent, grounded), and no single sentence carries
it. The named conversion route (the crux, mirroring replag's induction
caveat): R1-Distill may stamp transitions lexically ("Let me verify…",
"Now, case 2:" openers), making run STARTS partially per-token readable.
That route is exactly what the per-token baseline measures at the same
anchors; `time_in_run` ≥ 2 vs 0 is the harder-to-stamp variant. The
within-window shuffle is the order receipt: boundary needs which-side
information (previous vs current sentence), not bag membership.

## Clock bridge (measured — `../labels/proofops_stats.json`)

Median 16 tokens/sentence (mean 19.2, p10 6, p90 37) on the Ward
tokenizer. A token window spans 2 sentences only at **T ≥ 32**; T = 16
is the sub-sentence control point. Screen at T ∈ {8, 16, 32, 64}: the
T range is chosen from these numbers, per substrate-audit item 6.

## Predicted T-pattern (STORY.md § 7: threshold family)

Window−token gap ≈ 0 at T ≤ 16 (the window cannot reach the previous
sentence), turns on at T ≈ 32 (2-sentence coverage), flat — or mildly
DECLINING (localized variant; dwell 1.86 sentences means the relevant
history is ~2–3 sentences ≈ 32–48 tokens) — by T = 64. Explicitly NOT
monotone-to-saturation. Per-token flat in T (definitional control).

## Falsifier / KEEP-KILL (frozen)

- **KEEP** iff the boundary target shows gap ≤ 0.02 AUC at T = 16 and
  ≥ 0.05 AUC at T ∈ {32, 64} (the threshold shape), in a consistent
  probe pair, with the shuffled-window control separating from the
  ordered window on the same cells.
- **KILL** if per-token ≈ window at every T (transitions fully
  lexically stamped — ambient conversion, recorded as such), or the gap
  appears without the threshold shape (pattern mismatch), or shuffled ≈
  ordered ≈ per-token + ε with no order component.
- Stage-2 prior IF kept (soft, low-confidence — Stage 2 freezes its own
  per-arch predictions): equality subtype ⇒ mixing codes at the
  2-sentence window (T ≈ 32 token units); whether the Spectral band
  prior transfers from fixed-period recipe units to soft sentence
  boundaries is genuinely open — window-vs-{per-token, T-SAE} is the
  frozen part, the within-mixing-code ordering is not.
