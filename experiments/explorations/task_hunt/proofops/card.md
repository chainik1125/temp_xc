# Mini-card — candidate 2: proof-operation run structure (Stage-1 screen)

**Status: FROZEN at commit (commit-then-run — no activation probed for
this candidate at this commit).** Agent: runpod-d. Briefing:
`briefings/task-hunt.md` candidate 2. Provenance: grounded (the C5
three-timescale structure is model-independently confirmed).

## Labels (runpod-b's build, used as-is)

`labels/proofops.npz` (`build_proofops.py`, committed pre-output):
the frozen 5-class per-sentence operation labels from
`synthetic/expansion/records/proof-operation-phase-runs/labels.json`
broadcast onto the Ward cache grid, with run features at sentence
level. 443,974 labeled tokens (86.4 % of valid positions), 287/300
docs covered. **Using runpod-b's labels directly — no duplicate build
(the candidate-1 duplication is not repeated).**

- **PRIMARY `tir`** — time-in-run binned {0, 1, ≥2} (three-class,
  balanced manifest 20k rows/class). Base rates over labeled tokens:
  55 % / 22 % / 24 %.
- **SECONDARY `boundary`** — is_run_start (run start vs interior).
- **AMBIENT ANCHOR / control `op`** — the operation class itself
  (5-class). Per-sentence readable **by construction**: the sentence's
  own tokens determine its label, so this is the regime-1 reference —
  its window−token gap is the *ambient baseline* the temporal targets
  must beat, not a result.

## Why non-ambient (and where the risk is)

`tir` at a token asks how many *consecutive previous sentences* carried
the same operation class. The current sentence's own tokens fix `op`
but say nothing about how long the run has lasted — that requires
reading earlier sentences. The honest risk: operation runs may be
stylistically marked (a model that has been doing algebra for four
sentences may write differently), which would leak run-depth into
single tokens and collapse `tir` toward regime 1. That leak is exactly
what the `op` anchor measures — if `tir`'s per-token AUC tracks `op`'s
gap structure, the candidate dies.

## The clock bridge (audit item 6) — why T must be large here

From `proofops_stats.json`: **median 16 tokens/sentence** (mean 19.2,
p10 6, p90 37) ⇒ sentences per window are 0.5 / 1 / 2 / 4 at
T = 8 / 16 / 32 / 64, and the minimum T spanning two sentences on the
median clock is **32**. A window must cover ≥ 2 sentences to see any
run structure at all, so the briefing's default T ladder (2…32) is
mostly below this latent's support. **Frozen T ladder: {8, 16, 32,
64}** — T = 128 is excluded (the Ward window is 128 tokens; only
p = 127 would be eligible, a degenerate single position).

## Screen protocol (frozen)

Models {base, distill} × layer hs13 (resid_post L12; confirmatory hs11
= L10) × T ∈ {8, 16, 32, 64}. Rows from the committed balanced
manifests, restricted to p ≥ 64 (the largest T), split by trace via
the npz's own `trace_split` (80/20, seed 0); **row cap 12,000 train /
3,000 test** (memory: T = 64 flatten is 262,144 dims — the cap keeps
the probe matrix ≈ 12 GB, disclosed here as a compute constraint, not
a tuning knob). Frozen `problib` stack, AUC primary (macro one-vs-rest
for the 3- and 5-class targets): per-token linear, window-flatten
linear, window-MEAN linear (g_agg/g_order), within-window-SHUFFLED
linear, MLP-512 presence at T = 32; permutation null seed 99.

## Frozen predictions (STORY.md § 7 taxonomy)

- **P1 (threshold at the clock):** g(T) for `tir` is ≤ 3 σ_null at
  T = 8 and T = 16 (sub-sentence windows) and clears it at T = 32,
  growing again at T = 64 — a **threshold pattern set by the
  sentence clock**, not a smooth rise from T = 2.
- **P2 (ordering vs the anchor):** `op`'s g(T) is flat in T (its label
  is fully determined inside the current sentence — extra window adds
  ambient pooling only), while `tir`'s g(T) rises. The *contrast*
  g_tir(T) − g_op(T) increasing in T is the candidate's real claim.
- **P3 (order component):** g_order > 0 for `tir` at T ≥ 32 — run
  depth is a directional count, and the window-mean is order-blind.
  This is the prediction that distinguishes this candidate from
  candidate 1 (where order was predicted small).
- **P4 (boundary):** `boundary` behaves like `tir` but weaker (a
  boundary is a single-transition event; less accumulated evidence).
- **P5 (model axis):** base ≈ distill (reader-predictability
  precedent); Stage 2 takes the better cell.

## Falsifier / kill rule (pre-registered)

KILL if ANY of: (1) g_tir(T) ≤ 3 σ_null at every T; (2) g_tir(T) does
not exceed g_op(T) by more than the null floor at any T (the gap is
generic window pooling, not run structure); (3) g_tir(T) is flat or
non-monotone across {32, 64} (no T-story above the clock threshold);
(4) per-token `tir` AUC is already within 0.02 of the window AUC at
every T (run depth is ambient — the stylistic-leak risk realized).
Verdict → one paragraph in `../LOG.md`.
