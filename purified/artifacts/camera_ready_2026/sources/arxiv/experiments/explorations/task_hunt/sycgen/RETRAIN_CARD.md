# `sycgen` — FIRST-KEEP MATRIX RETRAIN CARD (frozen before any cell runs)

**Executor + owner: `mac-d`** (pre-authorized f0ac106e4 item 3 +
dc3cb8fd9 "KEEP ⇒ warm-pod matrix retrain within the hour"; bundle =
KEEP 3/3, e4fa17e5b). Design inputs BINDING: PRECOUNT §§ 1–7,
GENERATION_CARD, SCREEN_CARD + screen results, the pinned matrix arm
mapping 692cb ({BatchTopK} = btk-only), and the λ̂
SHUFFLE_OVERLAY_CARD as the retrain-mechanics template. Pod borrow
REVERSED (ee16ea041) ⇒ both pod-D GPUs are this lane's.

## § 1 What this is

The #6 exhibit: the matrix-standard ordered-vs-shuffled T-sweep on
the program's FIRST hunt-KEEP safety task (`sycgen_age`,
generator-mode sycophancy). First training on this substrate — there
is no quoted panel and therefore **no anchor gate by design**; the
T-sweep itself is the deliverable, with untrained twins and the
shuffle instrument as the controls.

## § 2 Grid (48 cells; hyperparameters inherited BY CONSTRUCTION)

Datasource `sycgen_real_age_llama31_8b_l14` (this freeze: plugin
`real_sycgen.py` + data.yaml entry — single-file drop per hard
rule 3): activations = the sycgen SCREEN cache (llama31_8b @ hs14,
the screen layer; llama = largest KEEP margin, hardest tok baseline),
labels = `sage_face` VERBATIM (log2(1+age), support-64 NaN,
assistant-only, events masked), windows mapping receipt re-asserted
at load. λ̂ Stage-2 hyperparameters by construction: d_sae 2048,
k_pos 8, n_steps 8000, buffer 524,288 (≈0.56 corpus/refill,
disclosed — dial precedent), eval_window_L 32, batch =
`grid.batch_size(T)`, canonical runner end-to-end (hard rule 1).

| cells | arch | T | seeds |
|---|---|---|---|
| 18 | `txc_batchtopk_post_btkonly` (claiming arm) | {2,4,6,8,10,16} | {1,2,42} |
| 3 + 3 | `batchtopk_sae_btkonly`, `tsae_btkonly` (anchors) | 1 | {1,2,42} |
| 24 | untrained twins (n_steps 0), one per (arch,T) per seed | — | {1,2,42} |

The exhibit's 7-T x-axis {1,2,4,6,8,10,16} = the 6 window points +
the T=1 anchor line (the fleet's standard rendering; a T=1 window
crosscoder is the per-token anchor by construction). `eval_extra =
{"retrain_tag": "sycgen_keep_r1"}` namespaces every eval_key (fresh
rows, no collisions); checkpoints persist under
`checkpoints/<train_key>/` for § 3 and the HF push. Two-shard i%2
split across the pod's GPUs (deterministic over the sorted cell
list); each shard resumable.

## § 3 Shuffle instrument (λ̂ § 4 transplant, byte-inherited)

`sycgen/shuffle_overlay.py` (this freeze): per trained cell,
recompute ordered recovery from the persisted checkpoint (identity
receipt |Δ| ≤ 2e-3 — the λ̂ A2 tolerance with its conditioning
analysis inherited; certifies the code path, licenses nothing else),
then score the SAME fixed probe on per-row within-window-permuted
eval tiles (`shuffle_within_window`, seed 0, probe never refit).
T=1 anchors are shuffle-identity by construction.

## § 4 Deliverables, durability, economics

- Rows: canonical leaderboard rows on the pod (containers never
  push) → repatriated via `agents/mac-d/repatriate.sh` (dup-key
  merge) → pushed from the mac.
- Checkpoints: `push_ckpts_hf.py` (runpod-a's, ratified path
  `ckpts/<train_key>/`) + sha receipts in STATUS — certificate
  evidence must survive pod loss.
- Overlay JSON + (next push) the Aniket-template T-sweep figure:
  ordered-solid / shuffled-dashed, per-seed faints, T=1 anchor
  bands, y = recovery r.
- Est: 24 trained ≈ 10–15 min each + 24 untrained ≈ 1–2 min each
  ≈ **5–7 GPU-h ≈ $15–21** on 2×H100 ⇒ wall ≈ 2.5–3.5 h, drain
  ~06:00–06:45 London. Ledger at launch (this push), actuals at
  drain. Hunt envelope / Han $500 aggregate.

## § 5 AMENDMENT — T-axis (post-launch, 02:49 2026-07-28, mac-d)

**§ 2's T ∈ {2,4,6,8,10,16} was my card-writing error and is amended
to T ∈ {2,4,8,16} (grid 48 → 36 cells).** The frozen § 2 text above
is left untouched per governance; this section supersedes it.

- **Why**: the canonical eval (`temp_bench/evals/synthetic_recovery.py`)
  requires `eval_window_L % T == 0` with power-of-two L and T
  (synthetic README § 4). L=32 (frozen § 2) rejects T∈{6,10}; no
  single L ≤ seq_len 128 divides {6,10,16} (LCM 240). I conflated the
  7-point *synthetic-lane rendering* axis with the real-cache
  trainable axis. The actual λ̂ Stage-2 template constant is
  `WINDOW_TS = (2, 4, 8, 16)` ("eval_window_L = 32 so every T in the
  ladder tiles it exactly" — run_stage2.py); the leaderboard holds
  **zero** T=6/T=10 rows fleet-wide. The amended exhibit x-axis
  {1,2,4,8,16} is exactly the λ̂ exhibit's.
- **Receipt**: the as-run shard jsons retain all 12 doomed cells as
  `ok:false` rows carrying the exact ValueError (6 untrained
  failed-fast ≈ $0; 6 trained ≈ 40 GPU-min ≈ $2 burn, disclosed).
  The 36 surviving cells are BY CONSTRUCTION the amended grid — no
  cell of the amended grid was affected; nothing relaunched.
- **Decision**: shard 1 was left to complete (killing would have
  destroyed 3 in-flight legitimate cells to save ~$2). Shard 0
  finished 18/18 amended cells ok; shard 1 verdict at drain.
- **Ripple**: `run_retrain.WINDOW_TS → (2,4,8,16)` + assert 36
  (single source; `shuffle_overlay` imports it). Pod checkout
  advances to the amendment pin AFTER `SHARD1-DONE`, before the
  overlay runs (running workers re-import from disk — no mid-run
  edits on the pod).

_Recorded-by: claude-fable-5 (mac-d, executor-owner)._
