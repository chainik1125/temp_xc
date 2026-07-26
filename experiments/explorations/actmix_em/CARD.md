# ACTMIX P2 card — EM shuffle control + T-window sweep (btk-only arm)

**Frozen pre-run.** Mandate: `briefings/actmix-runpod-2.md` (+
`actmix-shared.md`, binding). Agent: **runpod-2** (GPU 2 of the ACTMIX
shared 3×H100 pod). Every cell through
`temp_bench.core.runner.run_experiment` (hard rule 1) → leaderboard
rows, evaluator `em` protocol **3.0.0** (`src/temp_bench/evals/em.py`,
the origin/final detection port; primary `pr_auc_S16`; within-window
shuffle ablation seed 42 and realized-l0 measurement are INSIDE the
eval). Cell table: `cells.py` (frozen with this card). Driver:
`run_cells.py` (`--pin` = this freeze commit's sha taken from
origin/arxiv at launch; refuses to run off-pin). Verdict lands in
`experiments/explorations/task_hunt/LOG.md` as **PENDING TEAM
REVIEW**; ledger lines under `RUNPOD` in `briefings/MODAL_SPEND.md`.

Prime directive: **a sound verdict, never a win.** The paper's § 5.3
EM negative may be CONFIRMED by this ablation — that is a fine
outcome and goes in at full prominence.

---

## § 1 — Setup pinned (and the flags)

**Task** = the paper's § 5.3 emergent-misalignment DETECTION cell on
the **medical organism**: Qwen2.5-7B-Instruct +
`andyrdt/Qwen2.5-7B-Instruct_bad-medical` (merged), paper layer
**L15** (resid_post, hs16) — the layer choice follows the paper and
is stated here per the briefing (the depth record's inverted-U
g(ℓ) peaks at L13; L15 is the PAPER's layer and this card runs the
paper's layer; the L13 question belongs to the em-redo thread, not
this exhibit).

**Training substrate** = `medical_em_prompts` **BASE-forward**
(datasource `qwen_2_5_7b_instruct_medical_l15`, data_key
`56a61e3776062439`): the paper trained dictionaries on BASE
activations and applied them to organism activations at detection
(TRACKING.md § 1, verified against origin/final). Cache built by
`build_train_cache_base.py` (verbatim origin/final corpus recipe via
`conversion_depth.build_em_train_cache.build_corpus`; ≈59 % eos-pad
recorded property preserved).

**Detection cohort** = the canonical medical cell's stage-4 cohort,
recovered from origin/final git history
(`c6_88a4ddf6819d8057/judge_outputs.jsonl`) and re-forwarded through
the merged organism by the UNMODIFIED phase-4 builder
(`phase4_em_depth.py cache`) on this pod. **Integrity: cohort
reproduced exactly — 1728 rollouts, misaligned frac 0.323, d_model
3584** (matches TRACKING.md's recorded values; sidecars regenerated
by the same deterministic pipeline that built the g-map cohort).

**Flags (raised, not chosen — for mac-local/mac-c):**
- **F1**: `experiments/em/run.py`'s default datasource is the 14B
  FINANCE l24 anchor; configs/experiments.yaml confirms the paper
  section had TWO organisms (finance-14B + medical-7B). This phase
  covers the **medical** cell only — the one with the published
  negative, the detection-eval port, and a recoverable cohort.
  The finance cell has no cohort/eval infrastructure in-tree.
- **F2**: paper EM numbers came from `dmitry-em-repl`, not `final`'s
  EM code (briefing caution 1) — nothing here is labeled
  `paper-match`. This arm is **btk-only**; Phase B waits on mac-c's
  COMPOSITION_AUDIT pin.
- **F3 (resolved)**: btk-only convention = mac-a's canonical Stage-1
  (LOG ~21:05, commit `92db86c41`; mac-local APPROVED `9e634bed9`).
  Registry names consumed verbatim; no local fork.

## § 2 — Arms

- **btk-only (THIS run)**: `txc_batchtopk_post_btkonly`,
  `batchtopk_sae_btkonly`, `tsae_btkonly` (arch_versions 1.1.0 /
  2.1.0-port; `relu_mode: btk-only` hashes into every train_key).
- **relu-mix (context, NO new cells)**: the paper's published § 5.3
  cells (origin/final leaderboard, § 5 anchors below) + runpod-c's
  em-redo interim rows (ORGANISM substrate, review deferred) —
  quoted as context, never merged into this table.
- **paper-match (Phase B, BLOCKED)**: pinned composition from
  dmitry-em-repl once mac-c lands; checkpoints from Han's HF
  datasets if found (eval-only shuffle), else retrain at the pinned
  composition. Organism-forward training caches at L{9,13,15} are
  pre-built on this pod as Phase-B insurance (data_key
  `2d0a9b6176e91bad` at L15).

## § 3 — Frozen cells (see cells.py; conventions = em-redo/c6)

| cell | arch | T | k_pos | d_sae | batch | steps | seeds |
|---|---|---|---|---|---|---|---|
| txc_post_btkonly_T{1,2,4,8,16} | txc_batchtopk_post_btkonly | 1–16 | **20·T per window** (= 20/token parity) | 32768 | 1024 win | 25 000 | {42, 1} |
| batchtopk_sae_btkonly | batchtopk_sae_btkonly | 1 (per-token) | 20/token | 32768 | 1024 tok | 25 000 | {42, 1} |
| tsae_btkonly | tsae_btkonly | 1 (per-token; arch rejects T≠1) | 20/token | 32768 | 32 seqs (= 4096 tok/step; em-redo precedent) | 25 000 | {42, 1} |
| untrained twins (all 7 shapes above) | same | same | same | same | same | **0** | {42} |

- Matched nominal budget **20 atoms/token** (em-redo Part II
  convention). Post's per-WINDOW k_pos = 20·T ⇒ **T = 1 is the
  controlled limit** (TXC ≈ SAE at matched params) — the anchor row
  of the exhibit. The T = 4 cell (k_pos 80) is the direct btk-only
  twin of em-redo's relu-mix panel cell.
- NO bricken on the panel (fair-backbone stance; bricken is a
  paper-anchor knob and belongs to Phase B).
- Per-token baselines are **T-invariant bands** by construction
  (tsae asserts T = 1; the eval windows each arch at its own
  `config.T` — the paper's own convention). Their shuffle column is
  empty BY DESIGN: at T = 1 no within-window shuffle exists — their
  order-invariance is the control's own control, stated analytically
  rather than simulated.
- SAE/TXC per-step token exposure is unequal (1024 tokens vs
  1024·T) — the paper's own c6 pairing was also unequal (txc_base
  5120 positions/step vs sae_arditi 1024; TRACKING § 1). Aniket's
  backtracking sweep exposure-matches; divergence stated with
  reason: this card follows the EM section's own convention.
- Dispatch: endpoint-first (T16 + T1 at s42 land before interior
  spend — Aniket precedent), lanes a/b/c in `cells.py`, all three
  driver processes on GPU 2 (3-way contention ≈ the em-redo timing
  basis: token cells ~31 min, window cells ~1–4 h growing with T).
- Shuffle semantics: per-row within-window input permutation, seed
  42, pre-encode (protocol 3.0.0 == Aniket's
  `shuffles.py` semantics; his extra controls — reversal, circular
  shift, positional-stack SAE — are NOT in EM's protocol; adopting
  them would fork the paper currency, so they are named here as
  known-absent, available post-hoc from checkpoints if the team
  asks).

## § 4 — Pre-registered expectations (BEFORE any cell ran)

Shared pre-registration (actmix-shared, quoted verbatim): *"under
`btk-only` the per-token sae baseline improves MOST ⇒ hunt TXC-vs-sae
margins likely shrink; tsae margins move least (6.7/8 realized —
already our licensed lead comparator); hunt T-slopes may soften
(low-T cells recover); the PAPER arch's T-curves should improve."*

EM-specific, this card:

- **E1 (headline direction)**: the paper's EM negative PERSISTS
  under btk-only — txc_post_btkonly does not beat the
  batchtopk_sae_btkonly band in pr_auc_S16 at any T (mean over
  seeds). Consistent with the shared pre-reg (the sae baseline
  gains most from the fix). If TXC wins anywhere, that is a NEW
  positive the paper's composition masked — report with the same
  prominence either way.
- **E2 (shuffle)**: shuffle_gap_S16 for txc_post_btkonly stays below
  the paper's own decision bar (+0.02) at every T — reproducing, now
  WITH the missing control, the internal diagnostic that seeded the
  ambience gloss (paper txc_base gaps at L15: −0.059 s42 / −0.002
  s1). Gap ≥ +0.02 at any T = order-carrying detection = a new
  positive.
- **E3 (realized l0 — the mixing fingerprint)**: training-time
  realized == nominal by construction (mac-a's tests). EVAL-time
  realized l0 on cohort text may sit ABOVE nominal via
  threshold-transfer overfire (em-redo measured ≈137–172/token vs
  nominal 20 — thresholds calibrated on the 59 %-pad training
  stream over-fire on cohort text; layer-uniform, arch-uniform).
  Report as-measured per cell; all arms share the threshold
  machinery so the comparison is like-with-like.
- **E4 (T = 1 limit)**: txc_post_btkonly@T1 within ±0.03 pr_auc_S16
  of batchtopk_sae_btkonly (matched params; residual gap =
  parameterization only). A larger gap invalidates the "controlled
  limit" reading and is reported as such.
- **E5 (untrained)**: untrained twins sit near the positive-rate
  floor (0.323, i.e. pr_auc ≈ chance); trained−untrained margin is
  the learned-code control on every curve.

**KILL / machinery clauses (pre-measured):**
- **K1**: batchtopk_sae_btkonly pr_auc_S16 < 0.40 ⇒ pipeline broken
  — debug, do not interpret (em-redo's falsifier, reused).
- **K2**: any btk-only cell with EVAL-realized l0_per_token
  < 0.75 × nominal (the UNDER-realization/zero-pick signature the
  fix removes by construction) ⇒ stop the lane, flag mac-a in LOG.
- **K3**: any cell reporting n_rollouts ≠ 1728 or positive_rate ≠
  0.323 ⇒ cohort drift, abort.

## § 5 — Deliverables + analysis rules (frozen)

1. **The Dmitry table** per T (both seeds + mean): **TXC |
   TXC-shuffled | SAE (| TSAE)** in pr_auc_S16, realized
   l0_per_token beside every entry; untrained twins as a floor row.
   S ∈ {1,…,32} full grid in the results appendix.
2. **T-sweep figure** (WRITEUP style): txc_post_btkonly pr_auc_S16
   vs T with shuffled overlay (dashed), sae/tsae bands (flat,
   per-token), untrained floor. One per seed + mean panel.
3. **Margins**: δ_sae(T) = txc − sae, δ_tsae(T) = txc − tsae, mean
   over seeds, seed spread as max−min. Claim bar for "temporal
   detection": shuffle_gap ≥ +0.02 across S (the paper's own bar).
4. **Honest side-by-side with the paper's published negative**
   (origin/final leaderboard, protocol 3.0.0, L15): sae_arditi
   pr_auc_S16 0.690 (s42) / 0.745 (s1); txc_base 0.542 / 0.560,
   shuffle_gap_S16 −0.059 / −0.002. Frozen caveats, all three quoted
   wherever the side-by-side appears: (a) paper cells probed
   PER-CELL Wang cohorts (n_sent 79k–107k, base rates 0.32–0.47) vs
   this run's ONE fixed 1728-rollout cohort (0.323) — PR-AUC is
   base-rate sensitive, so cross-design deltas are context, not
   measurements; (b) budgets differ (arditi 128/token vs panel
   20/token; txc_base k_pos 25, T 5 paper knobs); (c) composition
   differs BY DESIGN — that is the ablation.
5. LOG verdict **PENDING TEAM REVIEW** scoring E1–E5 + K-clauses
   as written, misses as plainly as hits; ledger lines (est +
   actuals) under `RUNPOD`.

## § 6 — Budget, timing, descope ladder

Est: ~14–20 GPU-hours on GPU 2 (≈ $45–60 at ~$3/H100-h) + the
cache builds (~1 GPU-h). Cap $150/day — fits. Launch target ~22:15
London 2026-07-26; s42 curve ETA ~03:00–05:00; s1 by ~09:00–11:00;
analysis + verdict before 17:00 London.

Descope ladder (applied in order if time runs short; blind to
results): (1) drop s1 WINDOW cells (keep s42 curve + s1 token
cells); (2) drop T2; (3) drop T8 (keep {1,4,16}); (4) tsae s1.
Never dropped: T1 + T16 endpoints s42, sae cells, in-eval shuffle,
realized-l0 disclosure, untrained sae+txc@T16 twins, the § 5.4
side-by-side. Stretch (only after everything above): txc_pre
btk-only at {T4, T16} s42; organism-substrate btk-only twin at T4
(bridge to runpod-c's interim rows).

Dirty-tree stance: freeze commit pins the code; cells run
`TEMP_BENCH_ALLOW_DIRTY=1` (leaderboard appends dirty the tree for
the next cell — 7031/7116 existing rows carry dirty=true; em-redo
precedent). Wall logs live under `/workspace/logs/` (outside the
repo).
