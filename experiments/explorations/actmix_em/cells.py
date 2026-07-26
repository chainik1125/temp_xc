"""ACTMIX P2 — the FROZEN cell table (briefings/actmix-runpod-2.md).

Single source of truth for the EM btk-only panel: every (cell, seed)
with its exact training config, shared by the driver (run_cells.py)
and the analysis so both resolve the SAME train_keys through the
canonical runner machinery. Frozen at the CARD.md freeze commit;
edits after the freeze are append-only analysis helpers.

Design (CARD.md § 3):
- Arm: `btk-only` — mac-a's canonical Stage-1 convention (LOG note
  2026-07-26 ~21:05; commit 92db86c41; mac-local APPROVED 9e634bed9).
  Registry names consumed verbatim: txc_batchtopk_post_btkonly,
  batchtopk_sae_btkonly, tsae_btkonly. Never forked.
- Substrate: datasource `qwen_2_5_7b_instruct_medical_l15` — BASE
  forward, paper's layer L15 (CARD § 1; flags F1/F2 in the card).
- Matched budget: 20 atoms/token nominal (em-redo Part II
  convention). Post's k_pos is per WINDOW ⇒ k_pos = 20·T; at T = 1
  this is the controlled limit (TXC ≈ SAE at matched params).
- d_sae 32768 (§ 5.3 canonical em slice), n_steps 25_000, batch 1024
  windows (tsae: 32 whole sequences — its buffer consumes
  sequences; em-redo precedent), bf16, default lr/warmup — all the
  em-redo/c6 frozen conventions. NO bricken on the panel (fair-
  backbone stance; bricken is a paper-anchor knob, Phase B).
- Seeds {42, 1} — the paper's own em paired-seed convention.
  Untrained twins (n_steps = 0) at seed 42.
- Dispatch: endpoints first (T16 + T1 at s42 — Aniket's
  window-sweep precedent: the endpoint gate lands before the
  interior spend), three lanes sized so the T16 chain is alone.
"""

from __future__ import annotations

from temp_bench.core.schemas import TrainingConfig

DATASOURCE = "qwen_2_5_7b_instruct_medical_l15"
N_STEPS = 25_000
D_SAE = 32768
K_PER_TOKEN = 20
T_GRID = (1, 2, 4, 8, 16)
SEEDS = (42, 1)
UNTRAINED_SEEDS = (42,)

TXC_ARCH = "txc_batchtopk_post_btkonly"
SAE_ARCH = "batchtopk_sae_btkonly"
TSAE_ARCH = "tsae_btkonly"


def _cell(cell_id, arch, seed, overrides, batch_size, n_steps=N_STEPS):
    return {
        "cell_id": cell_id,
        "arch": arch,
        "seed": seed,
        "datasource": DATASOURCE,
        "training_cfg": TrainingConfig(
            n_steps=n_steps, batch_size=batch_size,
            arch_hparams_override=overrides),
    }


def txc_cell(T, seed, n_steps=N_STEPS):
    tag = "" if n_steps else "_untrained"
    return _cell(f"txc_post_btkonly_T{T}{tag}", TXC_ARCH, seed,
                 {"d_sae": D_SAE, "T": T, "k_pos": K_PER_TOKEN * T},
                 1024, n_steps)


def sae_cell(seed, n_steps=N_STEPS):
    tag = "" if n_steps else "_untrained"
    return _cell(f"batchtopk_sae_btkonly{tag}", SAE_ARCH, seed,
                 {"d_sae": D_SAE}, 1024, n_steps)


def tsae_cell(seed, n_steps=N_STEPS):
    tag = "" if n_steps else "_untrained"
    return _cell(f"tsae_btkonly{tag}", TSAE_ARCH, seed,
                 {"d_sae": D_SAE}, 32, n_steps)


# ── Lanes: disjoint cells, one driver process per lane on GPU 2. ────
# Order within a lane IS the dispatch order. s42 before s1
# everywhere; endpoints (T16, T1) first.

def lane_a():
    """The T16 chain — heaviest cells, alone in a lane."""
    return [txc_cell(16, 42), txc_cell(16, 1)]


def lane_b():
    """Mid-weight window chain: T8 + T4, then their s1 twins."""
    return [txc_cell(8, 42), txc_cell(4, 42),
            txc_cell(8, 1), txc_cell(4, 1)]


def lane_c():
    """Light chain: T1 endpoint + per-token bands + T2 + untrained."""
    cells = [txc_cell(1, 42), sae_cell(42), tsae_cell(42),
             txc_cell(2, 42)]
    for T in T_GRID:
        cells.append(txc_cell(T, 42, n_steps=0))
    cells += [sae_cell(42, n_steps=0), tsae_cell(42, n_steps=0)]
    cells += [txc_cell(1, 1), sae_cell(1), tsae_cell(1),
              txc_cell(2, 1)]
    return cells


# ── SCHEDULING AMENDMENT (2026-07-26 ~22:05 London, post-freeze,
# blind — no cell had completed). The a/b/c 3-lane split OOMed at
# launch: the training-step GPU peak scales ∝ T·batch·d_sae
# (measured: T16 ≈ 43 GB, T8 ≈ 29 GB; the shuffle buffer itself is
# CPU-resident), so T16 ∥ T8 ∥ T1 exceeds one 80 GB card. Cells,
# hparams, seeds, and endpoint-first priority are UNCHANGED — this
# is dispatch only: lane h serializes the two heavy shapes (T16,
# T8, + their untrained twins + heavy s1 cells), lane l carries
# everything ≤ T4 (worst co-peak ≈ 43 + 22 GB). Lanes a/b/c kept
# above for the record; launched lanes are h + l.

def lane_h():
    """Heavy chain — ALL T ≥ 4 window shapes serialized (measured:
    an uncapped T16 cell caches ~73 GB; T ≥ 4 never co-resides with
    another T ≥ 4). Launch with TEMP_BENCH_GPU_FRACTION ≈ 0.68."""
    return [txc_cell(16, 42), txc_cell(8, 42), txc_cell(4, 42),
            txc_cell(16, 42, n_steps=0), txc_cell(8, 42, n_steps=0),
            txc_cell(4, 42, n_steps=0),
            txc_cell(16, 1), txc_cell(8, 1), txc_cell(4, 1)]


def lane_l():
    """Light chain — every shape ≤ T2 or per-token; endpoint T1 +
    K1-falsifier sae first, then small untrained twins, s1 tokens.
    Launch with TEMP_BENCH_GPU_FRACTION ≈ 0.22."""
    return [txc_cell(1, 42), sae_cell(42), tsae_cell(42),
            txc_cell(2, 42),
            txc_cell(1, 42, n_steps=0), txc_cell(2, 42, n_steps=0),
            sae_cell(42, n_steps=0), tsae_cell(42, n_steps=0),
            txc_cell(1, 1), sae_cell(1), tsae_cell(1), txc_cell(2, 1)]


LANES = {"a": lane_a, "b": lane_b, "c": lane_c,
         "h": lane_h, "l": lane_l}


def all_cells():
    for lane in ("h", "l"):
        yield from LANES[lane]()
