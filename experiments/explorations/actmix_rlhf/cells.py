"""ACTMIX RLHF — the FROZEN btk-only cell table (CARD.md § 2/§ 5).

Datasource `gemma_2_2b_base_l12_phase7` (the shipped ckpts' own
training stream). All d_sae 18432 (paper canon), v2 training
conventions (n_steps 25 000, batch 1024 windows / 1024 tokens; tsae
32 seqs), seed 42 (the paper's seed; seed 1 stretch), NO bricken.
Dispatch order IS the lane order — CARD § 5: sae_k500 first as the
smoke/neg_frac gate (mac-a identity note), then core, then stretch.
"""

from __future__ import annotations

from temp_bench.core.schemas import TrainingConfig

DATASOURCE = "gemma_2_2b_base_l12_phase7"
N_STEPS = 25_000
D_SAE = 18432
SEED = 42

TXC_ARCH = "txc_batchtopk_post_btkonly"
SAE_ARCH = "batchtopk_sae_btkonly"
TSAE_ARCH = "tsae_btkonly"


def _cell(cell_id, arch, overrides, batch_size, n_steps=N_STEPS, seed=SEED):
    return {
        "cell_id": cell_id,
        "arch": arch,
        "seed": seed,
        "datasource": DATASOURCE,
        "training_cfg": TrainingConfig(
            n_steps=n_steps, batch_size=batch_size,
            arch_hparams_override=overrides),
    }


def txc(T, n_steps=N_STEPS):
    tag = "" if n_steps else "_untrained"
    return _cell(f"rlhf_txc_post_btkonly_T{T}{tag}", TXC_ARCH,
                 {"d_sae": D_SAE, "T": T, "k_pos": 100 * T}, 1024, n_steps)


def sae(k, n_steps=N_STEPS):
    tag = "" if n_steps else "_untrained"
    return _cell(f"rlhf_sae_btkonly_k{k}{tag}", SAE_ARCH,
                 {"d_sae": D_SAE, "k_pos": k}, 1024, n_steps)


def tsae(k, n_steps=N_STEPS):
    tag = "" if n_steps else "_untrained"
    return _cell(f"rlhf_tsae_btkonly_k{k}{tag}", TSAE_ARCH,
                 {"d_sae": D_SAE, "k_pos": k}, 32, n_steps)


def lane_r():
    """Core-first (CARD § 5): smoke gate → paper shape → limit pair →
    interior → tsae shapes → untrained twins. T8/T16 stretch live in
    lane_rs and are launched only if the clock allows (≤ ~13:00)."""
    return [sae(500), txc(5), txc(1), sae(100), txc(2),
            tsae(500), tsae(20),
            sae(500, n_steps=0), sae(100, n_steps=0),
            txc(5, n_steps=0), txc(1, n_steps=0), txc(2, n_steps=0),
            tsae(500, n_steps=0), tsae(20, n_steps=0)]


def lane_rs():
    """Stretch: pre-declared in the card with honest pricing."""
    return [txc(8), txc(8, n_steps=0), txc(16, n_steps=0), txc(16)]


def _s1(cell):
    cell = dict(cell)
    cell["seed"] = 1
    cell["cell_id"] += "_s1"
    return cell


def lane_s1():
    """Seed-1 stretch (card § 2: 'seed 1 stretch', run only after
    everything else): the trained core at seed 1, no untrained
    twins. Added post-core (2026-07-27 ~04:30) — shapes identical
    to lane_r's trained cells, seed only."""
    return [_s1(c) for c in
            [sae(500), txc(5), txc(1), sae(100), txc(2),
             tsae(500), tsae(20)]]


def _s2(cell):
    cell = dict(cell)
    cell["seed"] = 2
    cell["cell_id"] += "_s2"
    return cell


def lane_ext_a():
    """CARD § 7 A1 phase A (frac 0.52, runs ‖ ext_b): seed-1 T8."""
    return [_s1(txc(8))]


def lane_ext_b():
    """CARD § 7 A1 phase A (frac 0.34, runs ‖ ext_a): seed-2 small Ts."""
    return [_s2(txc(1)), _s2(txc(2)), _s2(txc(5))]


def lane_ext_c():
    """CARD § 7 A1 phase B (solo, uncapped, after phase A drains):
    the unpairable big cells — T16 never co-resides with T8. s1_T16
    first so the full-2-seed interim figure completes earliest."""
    return [_s1(txc(16)), _s2(txc(8)), _s2(txc(16))]


def _relumix(cell):
    """CARD § 7 A3: the relu-mix twin — plain arch name (mac-a
    convention: *_btkonly strips to the paper's ReLU-mix path),
    identical shapes/seed/steps; distinct cell_id."""
    cell = dict(cell)
    cell["arch"] = {SAE_ARCH: "batchtopk_sae",
                    TXC_ARCH: "txc_batchtopk_post",
                    TSAE_ARCH: "tsae"}[cell["arch"]]
    cell["cell_id"] = (cell["cell_id"]
                       .replace("rlhf_", "rlhf_relumix_", 1)
                       .replace("_btkonly", ""))
    return cell


def lane_eq():
    """CARD § 7 A3 equivalence twins (seed 42, k500 family)."""
    return [_relumix(sae(500)), _relumix(txc(5))]


def lane_x6():
    """CARD § 7 A2: T6 × 3 seeds (frac 0.35, ‖ x10)."""
    return [txc(6), _s1(txc(6)), _s2(txc(6))]


def lane_x10():
    """CARD § 7 A2: T10 × 3 seeds (frac 0.50, ‖ x6)."""
    return [txc(10), _s1(txc(10)), _s2(txc(10))]


LANES = {"r": lane_r, "rs": lane_rs, "s1": lane_s1,
         "ext_a": lane_ext_a, "ext_b": lane_ext_b, "ext_c": lane_ext_c,
         "eq": lane_eq, "x6": lane_x6, "x10": lane_x10}
