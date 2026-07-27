# ACTMIX P1 CARD — § 5.1 sparse probing: shuffle control + T-window sweep

**agent:** runpod-1 (shared 3×H100 pod, GPUs 0,1) · **arm:** `btk-only`
(Phase A; `paper-match` Phase B BLOCKED on mac-c's COMPOSITION_AUDIT) ·
**created:** 2026-07-26 ~22:20 London · **status: FROZEN — cells launch
only from the pinned freeze commit; mac-local freeze-reviews in
parallel (launch-then-veto per the phase's clock).**

**Pin:** the freeze commit is the commit that lands this card on
`origin/arxiv`; `launch_runpod1.sh` asserts `HEAD == PIN`, PIN is an
ancestor of `origin/arxiv`, and the tree is clean before launching any
cell (runner additionally refuses dirty trees).

## 1. Goal (rebuttal exhibit, wanted before 9am PT / 17:00 London 07-27)

Dmitry's table — **TXC | TXC-shuffled | SAE | TSAE** — plus the T-sweep
curve with the **T = 1 controlled limit** (TXC ≈ SAE at matched params)
as the anchor row, for the paper's § 5.1 sparse-probing task, in the
`btk-only` arm (mac-a's Stage-1 convention, mac-local-APPROVED 9e634bed9,
consumed verbatim — `*_btkonly` registry names, no forks). Han owns the
science read; this pod does the compute. All verdicts **PENDING TEAM
REVIEW**.

## 2. Setup (paper § 5.1 matched exactly; deviations flagged in § 7)

- **Data**: `gemma_2_2b_it_l13_fineweb_24k128` (google/gemma-2-2b-it,
  layer 13 resid_post, 24 000 × 128). Training activations are the
  **paper's actual C3 anchor cache** — HF
  `han1823123123/temp-bench-data:act_cache/e4916bcae1881963` (v1
  act_cache_key), symlinked to v2 `results/data_cache/48d2d17ff88598d4/`
  with provenance in `meta.json` (`prep_cache.py`). No regeneration —
  fineweb streaming-order drift can silently change the 24k documents.
- **Probe suite**: the paper's actual probe cache — 38-task SAEBench+CT,
  schema 2.0.0 (left-aligned S=32 frames + `first_real`), synced from
  the same HF repo (`probe_cache/gemma_2_2b_it_l13_fineweb_24k128/`,
  38/38 complete, spot-checked (N,32,2304) fp16).
- **Eval**: `ProbingEval` protocol **1.2.0** = faithful port of v1
  `1.1.0` (S=32 tail, padding masks, top-k_feat class-mean-|diff|
  selection + L1 logistic liblinear C=1.0 rs=0, mean ROC AUC over
  38 tasks) **plus strictly additive** shuffle control + realized-l0.
  Port sources: `origin/final:purified/experiments/c3_probing/run.py`
  + `…/src/temp_bench/eval/probing.py`. k_feat ∈ {5, 20} (paper's two
  operating points; table reports both).
- **Shuffle control (Aniket's cross-task convention, read from
  `origin/neurips-aniket:purified/experiments/backtracking_window_sweep/`
  + `shuffles.py`, which is byte-identical in this tree)**: probe
  fitted on ORDERED train features; the SAME fixed probe scored on
  test features encoded from windows whose token order is permuted
  per row (per-window, seeded; `shuffle_within_window(per_row=True)`;
  micro-batch seeds `shuffle_seed·1_000_003 + row_start`,
  `encode_batch_size=64` pinned in eval_cfg). Reported per cell:
  `mean_auc`, `mean_auc_shuf`, `delta_auc_shuf`, per-task pairs.
  Per-token archs: within-window shuffle at T=1 is the identity —
  reported equal by construction (`shuffle_identity=1`), the control's
  own control.
- **Training**: v2 canonical trainer via `run_experiment` only;
  n_steps=20 000, batch_size=4096 (this tree's probing-runner
  defaults), lr 3e-4, warmup 1000, token-shuffle buffer (per-token
  archs) / window buffer (TXC) / sequence buffer (tsae) per arch
  `consumes`. T enters as `training_cfg.arch_hparams_override={"T": t}`
  (hashes into train_key; runner merges before instantiation).
- **Arm label**: `eval_cfg["arm"] = "btk-only"` on every row (hashes
  into eval_key; queryable).

## 3. Grid + queue (exact)

Archs (mac-a Stage-1 names, verbatim):

| table column | registry name | role |
|---|---|---|
| TXC (headline) | `txc_batchtopk_pre_btkonly` | selection budget k_pos=20 **per position** ⇒ k_win = 20·T — the paper txc_base's budget-scaling analog |
| TXC companion | `txc_batchtopk_post_btkonly` | k_pos=20 **per window** (hunt's lead arch family; fixed budget across T) |
| SAE | `batchtopk_sae_btkonly` | 20/token, T-invariant band |
| TSAE | `tsae_btkonly` | 20/token, d_sae=16384 (paper's own asymmetry), T-invariant band |

Grid: T ∈ {1, 2, 4, 8, 16} × seeds {1, 2, 42} × k_feat {5, 20};
T axis applies to TXC only (per-token archs are T-invariant by
construction — one train per seed, flat bands, hunt-fig4 convention).
T = 32 is a stretch cell (§ 7.6). Untrained twins: full grid at
n_steps=0, **seed 42 only** (control band).

Queue order (Aniket's fail-fast convention): untrained pass first
(cheap pipeline gate), then trained per-token archs (3 seeds), then
TXC-pre with seed 42 before 1/2 and T endpoints (1, 16) before
interior, then TXC-post at seed 42, then TXC-post seeds 1/2 and any
T=32 stretch **only if clock/budget allow**. Two shards (round-robin)
= GPU 0 / GPU 1; leaderboard appends are flock-guarded; detached via
nohup (SSH-drop-safe).

## 4. Pre-registered directional expectations (stated BEFORE any cell)

Quoted from `briefings/actmix-shared.md` (binding):

> under `btk-only` the per-token sae baseline improves MOST ⇒ hunt
> TXC-vs-sae margins likely shrink; tsae margins move least (6.7/8
> realized — already our licensed lead comparator); hunt T-slopes may
> soften (low-T cells recover); the PAPER arch's T-curves should
> improve (that is Dmitry's re-run gate: does d(perf)/dT improve).

Probing-specific (mine, this card):

- **E1**: TXC-shuffled falls toward the SAE band; SAE/TSAE exactly
  unmoved (identity). The SIZE of the TXC ordered-shuffled gap is the
  exhibit; no committed magnitude.
- **E2**: realized l0 ≡ nominal in every btk-only cell (mac-a's
  construction). Smoke-verified on this pod pre-freeze: sae 20.0/token,
  tsae 20.0/token, post 20.0/window, pre@T3 59.9/window ≈ 20·3 (tiny
  union collapse of per-position selections is expected for pre and is
  NOT a zero-pick pathology; the selection stage never zeroes).
- **E3**: T = 1 controlled limit: TXC-pre@T1 and TXC-post@T1 both have
  k_win = 20 = SAE's per-token budget at d_sae 18432 — expected ≈ SAE.
- **E4** (context, NOT targets): the paper's v1 C3 numbers for
  calibration — topk_sae 0.8831±0.0022, tsae_paper 0.8986±0.0036,
  txc_base T5 0.8952 / T10 0.8973 / T20 0.8999 (k=20, 3 seeds,
  TopK→ReLU composition, v1 trainer 20k×1024). The btk-only arm runs
  the v2 trainer at 20k×4096 — systematic offsets expected; the arm is
  internally consistent and that is what the table claims.

## 5. Validity gates / KILL clauses (pre-measured where possible)

- **G1 (l0 HALT)**: any btk-only cell with realized_l0 outside
  [nominal−0.5, nominal+0.5] (per-token/post; for pre: outside
  [0.9·20T, 20T+0.5]) ⇒ implementation/calibration bug ⇒ HALT the
  lane, flag in LOG — not a result. (Untrained twins are exempt from
  the sharp band: with no EMA threshold calibrated they eval via the
  documented batch-fallback path; their l0 is reported and read as a
  diagnostic, not gated.)
- **G2 (identity assert)**: SAE/TSAE `mean_auc_shuf` ≠ `mean_auc`
  exactly ⇒ bug ⇒ HALT (the eval computes them as equal by
  construction; a mismatch means the dispatch broke).
- **G3 (untrained sanity)**: untrained twin ≥ its trained counterpart
  (same arch/T/k_feat, seed 42) ⇒ that cell pair is quarantined
  pending investigation; not silently shipped.
- **G4 (suite integrity)**: every cell must report n_tasks = 38
  (driver preflights 38/38 before any cell; a cell evaluated on fewer
  tasks is invalid — no partial-suite rows).
- **G5 (anchor honesty)**: if |TXC@T1 − SAE| > 3× the SAE seed-σ at
  k=20, the controlled-limit anchor FAILED — the table ships with that
  caveat in the verdict; no quiet dropping of the anchor row.

## 6. Realized-l0 bands (numeric, the mixing fingerprint — REQUIRED per cell)

| arch | nominal | expected band (trained, threshold path) |
|---|---|---|
| batchtopk_sae_btkonly | 20/token | 19.5–20.5 |
| tsae_btkonly | 20/token | 19.5–20.5 |
| txc_batchtopk_post_btkonly | 20/window | 19.5–20.5 |
| txc_batchtopk_pre_btkonly | ≤ 20·T/window | [0.9·20T, 20T] (union collapse only) |

Reported per cell in metrics (`realized_l0`, `realized_l0_min_task`,
`realized_l0_max_task`, `realized_l0_shuf`) — the relu-mix-arm
comparison bands (hunt fingerprint: sae 4.4/8 at T1 etc.) come from
mac-b's ACTMIX_FORENSICS and are NOT re-derived here.

## 7. Flags to mac-local (divergences/ambiguities; veto window = freeze review)

1. **k discrepancy in the briefing**: `actmix-shared.md`'s finding
   table says paper txc "k_win = 8·T"; v1 `locked_archs.yaml` pins C3
   `txc_base` at k_pos=20 (⇒ k_win = 20·T, T=5). This grid uses
   **k_pos=20** (the paper C3 registry value, mirrored by the v2 +
   btkonly registries). The 8·T figure appears to be the hunt's k=8
   convention, not C3's.
2. **Dataset naming**: v1 cache meta says `dataset: fineweb`; v2
   registry says `HuggingFaceFW/fineweb-edu`. Moot for these numbers
   (the paper's actual cache bytes are reused; provenance recorded in
   the linked meta.json) but the naming should be reconciled before
   anyone REBUILDS from the v2 spec.
3. **Exposure convention**: v2 buffers give per-token archs B tokens
   and TXC B windows (=B·T tokens) per step — NOT Aniket's per-update
   B×T exposure matching (backtracking convention). Divergence flagged
   per the briefing; adopting exposure matching would fork the v2
   training convention mid-phase, so it is NOT done here.
4. **TXC column mapping**: pre = headline TXC column (paper budget
   scaling k_pos·T); post = companion (hunt lead, fixed window
   budget). If Dmitry's table wants post as the headline, say so in
   freeze review — both lanes are in the queue.
5. **Untrained twins at seed 42 only** (single control band,
   hunt-fig4 style). Expandable post-deadline.
6. **T=32 stretch edge**: at S=32, n_windows=1 and rows with any
   padding in the frame hit the documented all-windows fallback more
   often; the paper protocol is unchanged but the cell is
   qualitatively thinner — queued last, reported with that caveat.
7. **Paper's own T grid was {5, 10, 20}** (v1 `txc_base_mw` rows at
   k=20: 0.8952 / 0.8973 / 0.8999). The rebuttal grid {1,2,4,8,16} is
   the briefing's specification (T=1 anchor), not the paper's grid —
   the two are compared qualitatively (slope sign), not cell-by-cell.
8. **tsae d_sae=16384 vs 18432** elsewhere — the paper's own
   asymmetry, inherited verbatim from v1 (flagged, not "fixed").
9. **Pre-freeze smoke rows** ran allow-dirty and are marked
   (`smoke: true` in eval_cfg → distinct eval_keys; cannot collide
   with grid cells). Grid cells run only from the pinned commit.
10. **Launcher defects, disclosed (both caught within minutes of
    launch; eval/train configs never changed):**
    (a) the untrained pass passed `--txc-archs` twice (argparse keeps
    the last) so the txc-PRE untrained twins were dropped from the
    initial queue — trained passes unaffected (separate invocations);
    (b) the launcher omitted the pool-row dirty-stamp convention
    (`TEMP_BENCH_ALLOW_DIRTY=1` after the PIN assert — the task_hunt
    grid precedent: leaderboard appends dirty the tree after cell 1),
    so both GPU chains refused at their second `run_experiment` call.
    Both fixed in the follow-up commit; the queue RELAUNCHED at the
    new PIN with completed evals cache-hitting. Consequence for row
    stamps: the first two eval rows are clean-stamped at the original
    freeze sha; all later rows carry the new PIN with
    dirty-by-convention stamps (diff = leaderboard growth). No code
    edits happen in this clone while the queue runs.

## 7b. AMENDMENT 1 (2026-07-27 ~03:05 London, pre-first-txc-cell; mac-local veto window open)

**Window cells train at batch_size = 4096/T windows** (T1 4096 — the
anchor cell is UNCHANGED — T2 2048, T4 1024, T8 512, T16 256), i.e. a
CONSTANT 4096 token-slots per optimizer step across the sweep.
Why: (a) measured throughput (token cells ≈ 29 min at batch 4096)
scales ×T for fixed window batches ⇒ T16 ≈ 4–8 GPU-h/cell and the
pass ≈ 46 GPU-h — misses the rebuttal clock by a day; (b) constant
per-step token exposure IS Aniket's B×T exposure-matching convention,
whose absence flag 3 disclosed — the amendment closes that divergence
rather than widening it. n_steps (20k), lr, warmup, eval protocol,
token-arm cells, untrained twins: all unchanged. Timing: amended
BEFORE any window cell trained (tsae token cells were mid-train);
batch_size hashes into train_key so no stale-cache collision is
possible. Residual caveat for the table: token arms trained at batch
4096 tokens/step, window arms at 4096 token-slots/step across T —
per-step exposure matched, optimizer batch-count differs from the
token arms only in units (disclosed, same as the v1 c3 convention
question; per-arm internal consistency preserved).

## 7c. AMENDMENT 2 (2026-07-27 ~07:55 London): tsae_btkonly TRAINED cells → post-deadline

Measured 7.8 s/step (≈43 h per 20k-step train): `consumes='sequence'`
serving materializes (4096, 128, 2304) batches per step so train_step
can sample one consecutive pair — the v1 pipeline instead served
pairs via `train_window_size=2`. This is a v2 SERVING mismatch (arch
compute is fine; flagged to mac-a/the serving owner — candidate fix
is a pair-serving batch iter, NOT an in-place arch change). ~6.5 h of
both GPUs were burned before detection (ledger actuals disclose).
Impact: the btk-only arm's TSAE column tonight = untrained twins
only; the spec lists TSAE as parenthetical-optional, and the
paper-match arm's TSAE (trained, shipped ckpts) is complete. tsae
trained twins queue post-deadline behind the serving fix.

## 7d. AMENDMENT 2b (2026-07-27 ~08:05 London): tsae trained cells RESTORED at the em-redo serving convention

runpod-2's LOG data point (36df9ffb6): `consumes='sequence'` cells at
**batch_size = 32 sequences** (the em-redo convention) run 18–20 min
on these shapes — the 43 h pathology was the 4096-sequence batch, not
the arch. tsae_btkonly trained cells re-enter tonight's queue at
batch=32-seq (3 seeds, ~1 GPU-h, after the Phase-B extension drains).
Disclosure: the tsae arm's optimizer batch (32 contrastive pairs/step)
differs from the SAE arm's (4096 tokens/step) — a serving-convention
asymmetry inherited from the arch's pair-sampling design and the
em-redo precedent, disclosed in the table caption exactly like
Amendment 1's units note. n_steps 20k, lr, eval protocol unchanged.

## 7e. NOTE (2026-07-27 ~11:45 London): saturation-plan P1 "seed top-up" = §3 grid completion, no extension

mac-local's pod-saturation directive (059a66239) lists probing
"figure top-up" cells — third seed @T{1,2,4,8,16} + s1@T{4,8} — with
an instruction to record in-card seed-EXTENSION amendments and
disclose the seed-2 choice. For THIS card no extension exists to
record: §3 pre-registered seeds {1, 2, 42} at freeze (131ea677f),
before any cell ran, and the "top-up" cells are exactly the tail of
that grid already queued in `launch_runpod1.sh`. Seed-2 disclosure:
seed 2 was in the freeze-time set; it is third only in EXECUTION
ORDER (§3's fail-fast ordering: 42 first, then 1, then 2) — it was
not chosen after seeing seeds 42/1 results. Queue unchanged. Also
per 059a66239: an INTERIM 2-seed render of
`figs_writeup/fig_probing_shuffle_tsweep.*` (Aniket template) ships
for the 17:00 draft, re-rendered FINAL when seed 2 lands — analysis
code addition only, no cell semantics touched.

## 8. Budget (RUNPOD ledger in briefings/MODAL_SPEND.md)

Estimate at assumed pod rate ~$7.5/hr (3×H100; runpod-1's 2-GPU share
~$5/hr): untrained pass ~1 GPU-h; per-token trained ~1.5 GPU-h;
TXC-pre 3 seeds × {1,2,4,8,16} ~10 GPU-h; TXC-post seed 42 ~3.3 GPU-h
(+ seeds 1/2 ~6.5 GPU-h if clock allows); evals ~2–3 GPU-h → ~18–25
GPU-h ≈ 9–12.5 wall-h on 2 GPUs ≈ **$45–70 est**. Cap $150/day.
Actuals corrected at close per ledger discipline.

## 9. Deliverables

1. The table (both k_feats; arms side by side once Phase B unblocks):
   TXC | TXC-shuffled | SAE | TSAE rows over T with the T=1 anchor.
2. T-sweep figure per arm (WRITEUP style): TXC-pre curve + TXC-post
   companion + SAE/TSAE flat bands + untrained band + shuffled
   overlays (ordered/shuffled + difference, Aniket's plot convention).
3. Realized-l0 per cell (fingerprint bands above).
4. LOG verdict entry — **PENDING TEAM REVIEW** — quoting § 4
   pre-registrations verbatim.
5. RUNPOD ledger est + actuals lines.

## 10. Phase B (paper-match) — pre-registered plan, BLOCKED

On mac-c's COMPOSITION_AUDIT pin: checkpoints found ⇒ EVAL-ONLY
shuffle+sweep on them (protocol 1.2.0 eval over the pinned v1
composition); no checkpoints ⇒ retrain at the pinned composition on
this grid. Same gates, same deliverables, arm label `paper-match`.
Nothing about Phase B is assumed from this card's Phase-A choices.

## 7f. AMENDMENT 3 (2026-07-27 20:10 London, date-verified): T ∈ {6, 10} grid extension (Han directive eace1b077)

Grid extension, NOT result-contingent: Han requires T ∈ {6, 10} in
both T-sweeps; the T choice is his directive, stated as such.
Cells: txc_batchtopk_pre_btkonly × T {6, 10} × seeds {1, 2, 42},
k_win = k_pos·T (120 / 200), k_feat {5, 20}, shuffle twins in-eval
— §3 machinery unchanged (A1 matched batches: 4096/6=682, 4096/10
=409 windows/step; both ≥ the 64 floor). One arm only per the
equivalence certificate (RM halt c6e464881) — no relu-mix twins.
l0 bands per §6 scaling (nominal 120 / 200; the high-T
over-admission pattern disclosed at T16 applies pro-rata).
Launch on the RM-halt-freed GPUs; est 6–9 GPU-h overnight; ledger
line at launch. On landing: per-k figs re-render at 7 T-points and
the LOG entry carries the rebuttal note — the shipped paper's
"T10" was a PHANTOM label (A12: T5 replica); these are the first
REAL T=10 probing cells.
