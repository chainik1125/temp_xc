# DRAFT mini-card — self-correction marker intensity λ̂_sc (factory candidate 1)

**Status: DRAFT (runpod-b, `briefings/candidate-factory-traces.md` item 1).
Everything in § Frozen was fixed BEFORE any label was computed. The
running agent (runpod-d/e) freezes its own screen card; this draft +
the bundle exist so a screen starts in minutes on the existing Ward
base/distill caches.**

Bundle (committed builder → committed outputs):
`../labels/build_sc_lambda.py` → `../labels/sc_lambda.npz` +
`../labels/sc_lambda_stats.json`; shared frozen logic in
`../labels/factory_lib.py` under `tests/test_factory_labels.py`.
Grids are the canonical Ward stream (4044 × 128, same `wardmap`
broadcast + round-trip check as `ward_lambda.npz`); manifests follow
the `man_*_doc/pos/cls` convention (valid, pos ≥ 32, class-balanced,
split by `trace_idx` via `trace_split`).

## The candidate logic (the winner's family on a new event stream)

ward_lambda (Stage-2 QUALIFIED POSITIVE) showed a kernel intensity over
Sonnet-judged backtracking events is window-recoverable with the
T-pattern of an aggregation latent. This candidate swaps the judged
event stream for a FROZEN lexical one — self-correction markers — so
the event is exact, zero-API, and reproducible from the trace text
alone. Regime framing: regime-2 aggregation (the label is a weighted
trailing event RATE; window-MEAN aggregation of marker-bearing token
features is the expected mechanism; shuffle-IMMUNITY is the receipt).

## Frozen (before computing anything)

- **Marker list**: the 17 regex patterns in
  `factory_lib.MARKER_PATTERNS` (word-boundary, case-insensitive after
  length-preserving normalization; `^no` sentence-initial only). A
  sentence is an event iff it matches ≥ 1 pattern. Disclosed overlap:
  "let me verify / double-check / review" intersects the proofops
  `verification-check` class, and the marker family intersects the
  Sonnet `is_backtracking` labels ("wait" is the canonical backtracking
  marker) — corr(λ̂_sc, ward λ̂_hist) is reported in the stats JSON as a
  disclosed family-resemblance number, not hidden.
- **Kernel**: exponential w_l = exp(−(l−1)/τ), **τ = 3 sentences,
  K = 8 lags**, applied as a NORMALIZED causal trailing rate
  (`factory_lib.kernel_rate`) over the PREVIOUS sentences only — the
  current sentence's own event is never an input. Why not the committed
  backtracking `kernel_w`: those weights were FIT to backtracking
  self-excitation and are not portable to a new event stream; and the
  mirror's logistic wrapper is a monotone transform, so tercile targets
  are invariant to it — dropping it removes borrowed constants, not
  information. Kernel-only, NO position term (position-floor lesson);
  labels NaN for sentence index i < 4 (history guard).
- **Masking rule**: probe rows whose CURRENT token overlaps a marker
  match span (`is_marker_tok = 1`) are excluded from every manifest —
  the label must not be readable from the marker token itself. Marker
  tokens in the trailing WINDOW are legitimately visible (aggregating
  them IS the regime-2 mechanism). `is_sc` (current sentence contains a
  marker) ships as the disclosed ambient control target, like
  `ward_lambda.is_bt`.
- **Binning**: terciles of λ̂_sc over valid cells; frozen fallback if
  any tercile bin < 10% of finite rows → zero-inflated 3-bin
  (`factory_lib.zero_inflated_bins`); the scheme used is recorded in
  the stats JSON. Build-sanity kill if any manifest class < 2000 rows.
- **Label-side triage (kill authority, thresholds frozen in
  `factory_lib`)**: on test-split manifest rows (top vs bottom class),
  FAIL ⇒ no npz ships iff current-token-identity AUC ≥ 0.65 or
  position AUC (raw token index OR trace fraction, whichever is more
  extreme, inverse counts) ≥ 0.70.
- **Null**: within-trace event shuffle (seed 101 + trace_idx; NaN
  positions fixed) → `lam_sc_null` + `man_null_*` manifests. Preserves
  each trace's marker rate exactly, destroys the local clustering the
  kernel reads — separates local-history recovery from trace-ambient
  rate reading. The activation-side within-window shuffle stays the
  running agent's receipt (problib `shuf` arm).

## Predicted T-pattern + falsifier

Clock bridge (committed proofops numbers): median 16 tokens/sentence,
so the kernel's τ = 3-sentence mass sits ~48 tokens back and the K = 8
support spans ~128 tokens. A screen T ladder {2, 4, 8, 16, 32} covers
0.1–2 sentences of it. Prediction: window-MEAN recovery RISES with T
across the whole ladder (each doubling adds visible kernel mass);
per-token (masked rows) sits near the token-identity triage floor;
flat ≈ shuffled ≈ mean (shuffle-immunity — order-free aggregation).
The stats JSON ships `visible_evidence_auc` at T ∈ {8, 16, 32} — the
label-side AUC of the in-window marker count alone. **Falsifier**: if
activation window probes do not beat this visible-evidence line at the
same T, the probe is only counting marker tokens the window already
shows — no maintained state, no aggregation win worth a case study;
and if recovery on `lam_sc_null` matches the real label, the "recovery"
is trace-ambient rate, not history. Kill rule for the screen (running
agent finalizes): no window − per-token gap beyond 3 σ_null at any T,
or gap not growing anywhere in the ladder, or real ≈ null recovery.
