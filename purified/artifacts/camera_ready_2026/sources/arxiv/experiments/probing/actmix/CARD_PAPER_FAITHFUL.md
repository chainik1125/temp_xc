# CARD — paper-faithful probing sweep ({ReLU+TopK} matrix arm)

**Owner: runpod-1** (card author = shard-split owner per 03f533cc3).
Commission: 4ce0369de item 1 / 606e4587d item 1 (Han sprint). Arm
mapping per 692b5c1b1: `{BatchTopK}` = btk-only (delivered),
**`{ReLU+TopK}` = THIS card**, relu-mix = certificate evidence only.
Frozen BEFORE any cell (hard rule; commit sha of this card's freeze
commit is the PIN every shard asserts).

## 1. Composition (the thing being measured)

    z = ReLU( TopK_{k_win}( Σ_t x_t · W_enc[t] + b_enc ) ),  k_win = k_pos·T

Per-SAMPLE TopK over the summed window pre-activation (NOT BatchTopK),
ReLU AFTER selection — selected negatives become exact zeros (the
paper-era mixing fingerprint; realized l0 ≤ k_win). Upstream:
`origin/han-phase7-unification@94119bc08:src/architectures/
txc_bare_antidead.py:TXCBareAntidead` — the class that trained the
paper's shipped §5.1 TXC-base cells (COMPOSITION_AUDIT §3 pin).

Plugin: **`paper_txc_base_v1t`** (`src/temp_bench/archs/paper_v1t.py`)
— the FULL upstream class vendored verbatim, training stack included:
anti-dead tracker (10M tokens), AuxK (aux_k=512, α=1/32, bias-free
decode on residual), unit-norm decoder atoms over (T,d_in),
decoder-parallel grad removal, geometric-median b_dec init on the
first batch, kaiming→unit-norm→tied-encoder init. The v2 wrapper adds
ONLY trainer-side call sites the upstream trainer provided (dict
train_step contract, first-batch b_dec-init, grad hook via
`register_post_accumulate_grad_hook`, `post_step` renorm) + OPT-IN
wrapper-side telemetry under `no_grad` (vendored math untouched).
State-dict = strict superset of the eval-only adapter's (3 extra
buffers, disclosed). Archived T5 anchors stay on `paper_txc_base_v1`
and are NEVER retrained (alias rule; they are the on-figure anchors).

## 2. Grid

`paper_txc_base_v1t × T {1,2,4,6,8,10,16} × seeds {42,1,2}` = **21
cells** (×2 eval k each). d_sae 18432, k_pos 20 (paper widths).

COUNT NOTE (owner's resolution of the 18-vs-21 delta in 4ce0369de):
the stated grid is 7×3 = 21; every shard orders its T1/T2 tail cells
LAST, so if the hub intended to exclude a column (T1 the likely
candidate) one LOG line prunes 3 cells at zero sunk cost.

## 3. Training conventions (controlled comparison)

Canonical runner only (`run_experiment`); training_cfg IDENTICAL to
the v2 T-sweep cells: n_steps 20000, base batch 4096 with
**matched window batches** (batch = 4096/T windows ≥ 64 — AMENDMENT-1
exposure convention), runner-default lr/warmup, datasource
`gemma_2_2b_it_l13_fineweb_24k128` (paper act cache, healed pod-wide).
The PAPER's own trainer hyperparameters are NOT reproduced (its
model-side stack is; optimizer/budget parity with the v2 arms is the
controlled variable) — this is the same disclosure every arm carries.
Telemetry ON (`TEMP_BENCH_TELEMETRY_DIR=/workspace/logs/telemetry_rm`
on runpod-1 shards; other pods MAY set their local dir — optional,
observation-only, train_keys unaffected).

## 4. Eval

Protocol 1.2.0 verbatim: S=32 tail window, within_window shuffle
(per-row, seed 0), k_feat ∈ {5, 20}, 38-task cache, realized-l0.
`eval_cfg.arm = "paper-faithful"` on every row (sweep.py ARMS
extended this freeze).

## 5. Contract receipts (pre-launch)

`tests/test_paper_v1t.py` — **8/8 green**: adapter parity (bitwise
encode/decode vs the shipped-ckpt adapter, T=1 + T=3), T=1
degeneration formula, exact-k on positive-rich (l0 == k_win), mixing
fingerprint on scarce-positive (l0 < k_win), training stack
(b_dec-init once, finite loss, grad ⟂ atoms ≤1e-4, post_step unit
norms, no re-init), registry load with T override. Full suite: 291
passed + 1 PRE-EXISTING unrelated failure
(`test_stage2_variance_panels::test_legacy_default_reproduces_
committed_receipts` — fails with this card's diff stashed; panel
lane's live-leaderboard-coupled golden test; flagged to its owner,
not touched here).

## 6. Shard split (5 GPUs; balanced by measured min/cell; tails LAST)

| shard | venue | cells (order) | est |
|---|---|---|---|
| A | runpod-1 GPU0 | T16×{42,1,2} → T1×{42} | ~3.6h |
| B | runpod-1 GPU1 | T10×{42,1,2} → T1×{1} | ~3.3h |
| C | runpod-c GPU0 | T8×{42,1,2} → T1×{2} | ~3.2h |
| D | runpod-c GPU1 | T6×{42,1,2} → T2×{42} | ~3.1h |
| E | runpod-a GPU0 | T4×{42,1,2} → T2×{1} → T2×{2} | ~3.7h |

Command template (assert PIN first; substitute --Ts/--seeds per row):

    cd <clone> && git fetch -q && git rev-parse HEAD   # must == PIN
    CUDA_VISIBLE_DEVICES=<g> TEMP_BENCH_ALLOW_DIRTY=1 nohup \
      .venv/bin/python -m experiments.probing.actmix.sweep \
      --arm paper-faithful --txc-archs paper_txc_base_v1t \
      --Ts <Ts> --seeds <seeds> --shard-index 0 --shard-count 1 \
      >> /workspace/logs/pf_shard_<X>.log 2>&1 &

Shard A: `--Ts 16 --seeds 42 1 2` then `--Ts 1 --seeds 42`.
Shard B: `--Ts 10 --seeds 42 1 2` then `--Ts 1 --seeds 1`.
Shard C: `--Ts 8 --seeds 42 1 2` then `--Ts 1 --seeds 2`.
Shard D: `--Ts 6 --seeds 42 1 2` then `--Ts 2 --seeds 42`.
Shard E: `--Ts 4 --seeds 42 1 2` then `--Ts 2 --seeds 1 2`.

(Chain the two invocations with `&&` in one nohup bash -c so the tail
runs unattended.) REBALANCE RULE: an idle joiner may pull the heaviest
remaining TAIL cell — post the claim as a STATUS/LOG line BEFORE
launching it (duplicate trains waste; train-cache only dedupes same
pod). runpod-b GPU1 joins post-rmx_b (~11:30) as overflow only.

## 7. Durability

Per-cell: shard owner pushes the cell's ckpt via
`scripts/push_ckpts_hf.py <train_key>` (ratified mirror path, LFS sha
receipts) at cell landing or at shard drain (≤2h rule).

## 8. Budget

~1000 GPU-min ≈ 17 GPU-h ≈ **$45–60** across the three pods (each pod
posts its own ledger line at shard launch; runpod-1's covers shards
A+B est ~$18–22).

## 9. Pre-registered expectations (stated before any cell)

- E1: realized l0 < k_win appears and grows with T (zero-picks as the
  per-window positive pool thins at depth k_pos·T) — the pathology
  the v2 composition fix targeted.
- E2: directional — paper-faithful mean AUC ≤ btk-only at high T
  (T≥8), magnitude unknown; NO prediction at T≤4.
- E3: consistency — the archived T5 anchors interpolate between the
  new T4 and T6 columns within seed spread; failure ⇒ anchor/provenance
  audit BEFORE any exhibit ships.
- Scoring in RESULTS against these three, PTR as always.
