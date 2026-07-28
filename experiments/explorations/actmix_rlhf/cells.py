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
    """CARD § 7 A3 + A3b equivalence twins (seed 42): k500 family
    + the high-T pair (dead-latent regime, 361de3cb2 item 6)."""
    return [_relumix(sae(500)), _relumix(txc(5)), _relumix(txc(16))]


def lane_x6():
    """CARD § 7 A2: T6 × 3 seeds (frac 0.35, ‖ x10)."""
    return [txc(6), _s1(txc(6)), _s2(txc(6))]


def lane_x10():
    """CARD § 7 A2: T10 × 3 seeds (frac 0.50, ‖ x6)."""
    return [txc(10), _s1(txc(10)), _s2(txc(10))]


def lane_x4():
    """CARD § 7 A5 (directive 1065b26cf Han grid): T4 × 3 seeds —
    the grid-floor point between T2 and T5 (T5 stays as bonus)."""
    return [txc(4), _s1(txc(4)), _s2(txc(4))]


def lane_rmx_a():
    """CARD § 7 A5 relu-mix arm, runpod-2 share: T{1,2,4,6} × 3
    seeds, cheap-first. T1 ×3 doubles as the RLHF T1 equivalence
    certification (expected identical; legitimate twins — relu_mode
    hashes into train_key, no alias collision per 013441cfd)."""
    out = []
    for T in (1, 2, 4, 6):
        out += [_relumix(txc(T)), _relumix(_s1(txc(T))),
                _relumix(_s2(txc(T)))]
    return out


def lane_rmx_b():
    """CARD § 7 A5 relu-mix arm, runpod-b seed-split share
    (pre-auth UNCONDITIONAL per 1065b26cf; from width-match drain):
    T{8,10} × 3 seeds."""
    out = []
    for T in (8, 10):
        out += [_relumix(txc(T)), _relumix(_s1(txc(T))),
                _relumix(_s2(txc(T)))]
    return out


def lane_rmx_b16():
    """CARD § 7 A5 CONDITIONAL (runpod-b): relu-mix T16 s1/s2 —
    launch ONLY if the eq-lane T16 gate reads DIVERGENT (s42 twin
    already trained in lane_eq; identical ⇒ certificate line covers
    T16 and this lane never runs)."""
    return [_relumix(_s1(txc(16))), _relumix(_s2(txc(16)))]


def lane_tsae_s2():
    """CARD § 7 A4 (directive 98a9ea718 width triple, runpod-a):
    seed-2 twins of the tsae shapes ONLY — s42/s1 already ran
    @18432 (explicit override since freeze); re-running them would
    mint train_key-colliding alias rows. No new untrained twins
    (s42 floors stand, A1 precedent)."""
    return [_s2(tsae(500)), _s2(tsae(20))]


LANES = {"r": lane_r, "rs": lane_rs, "s1": lane_s1,
         "ext_a": lane_ext_a, "ext_b": lane_ext_b, "ext_c": lane_ext_c,
         "eq": lane_eq, "x6": lane_x6, "x10": lane_x10,
         "tsae_s2": lane_tsae_s2,
         "x4": lane_x4, "rmx_a": lane_rmx_a, "rmx_b": lane_rmx_b,
         "rmx_b16": lane_rmx_b16}


# ── CARD § 8: paper-faithful arm (agentic_txc_02_v1t) ──────────────────

PF_ARCH = "agentic_txc_02_v1t"
PF_DATASOURCE = "gemma_2_2b_it_l13_fineweb_24k128"  # the paper stream


def _pf_batch(T):
    """Upstream t-sweep batch schedule (train_primary_archs.py ~L2027:
    A40 OOM accommodation, part of the recorded procedure): 1024 below
    T10, 512 at T10-14, 256 at T>=15."""
    if T >= 15:
        return 256
    if T >= 10:
        return 512
    return 1024


def pf(T, seed=SEED):
    cell = {
        "cell_id": f"rlhf_pf_agentic02_T{T}" + ("" if seed == SEED else f"_s{seed}"),
        "arch": PF_ARCH,
        "seed": seed,
        "datasource": PF_DATASOURCE,
        "training_cfg": TrainingConfig(
            n_steps=N_STEPS, batch_size=_pf_batch(T),
            arch_hparams_override={"d_sae": D_SAE, "T": T, "k_pos": 100 * T}),
        # § 8: eval substrate is CELL IDENTITY — the tag hashes into
        # eval_key (G2 incident: env-only cache resolution aliased
        # l12-BASE rows as l13-IT cells); expect-check hard-fails on a
        # wrong-substrate cache before any metric is computed.
        "eval_cfg": {
            "hh_rlhf_cache": "l13it_paper",
            "cache_expect": {"subject_model": "google/gemma-2-2b-it",
                             "anchor_layer": 13},
        },
    }
    return cell


def lane_pf_pilot():
    """CARD § 8: the port-fidelity gate cell — T2/s42, compared against
    upstream training log agentic_txc_02_t2__seed42.json before the
    grid commits."""
    return [pf(2)]


def lane_pf_lo():
    """CARD § 8: T{1,2,4} × 3 seeds (T2/s42 = cache-hit after pilot)."""
    return [pf(T, s) for T in (1, 2, 4) for s in (42, 1, 2)]


def lane_pf_mid():
    """CARD § 8: T{6,8} × 3 seeds."""
    return [pf(T, s) for T in (6, 8) for s in (42, 1, 2)]


def lane_pf_hi():
    """CARD § 8: T{10,16} × 3 seeds (batch 512/256 per upstream)."""
    return [pf(T, s) for T in (10, 16) for s in (42, 1, 2)]


def lane_pf_anchor():
    """CARD § 8: T5 anchor evals × 3 seeds — ckpts staged from
    txcdr-base (stage_anchors.py, phase_b provenance-manifest
    precedent), NEVER trained here (alias rule)."""
    return [pf(5, s) for s in (42, 1, 2)]


LANES.update({
    "pf_pilot": lane_pf_pilot, "pf_lo": lane_pf_lo,
    "pf_mid": lane_pf_mid, "pf_hi": lane_pf_hi,
    "pf_anchor": lane_pf_anchor,
})
