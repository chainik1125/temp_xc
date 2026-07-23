"""em-redo Phase A — the FROZEN cell table (briefings/em-redo.md).

Single source of truth for the trained panel: every (cell, layer, seed)
with its exact training config, shared by the driver (run_em_panel.py)
and the probe-currency script (probe_codes.py) so both resolve the SAME
train_keys through the canonical runner machinery.

Frozen before any training run (TRACKING.md § 2, freeze commit). Do not
edit after the freeze except to append analysis-only helpers.

Matching design (Part II discipline, README § Part II + registry.py):
archs are compared at a matched per-token budget of **20 atoms/token**
(the fair-backbone suite k_pos), with the *realized* l0_per_token
measured on the shared eval windows and reported per cell. Nominal
knobs are set per each arch's documented semantics:

- batchtopk_sae   k_pos=20  (per-token budget, BatchTopK pool = B tokens)
- tsae            k_pos=20  (per-token matryoshka BatchTopK; threshold
                             inference makes realized l0 variable)
- txc_batchtopk_pre  k_pos∈{20,40} per-token PRE-squash budget; realized
                     density lands below nominal (union shrink — Part II
                     REPORT § 2 measured ≈0.5× on toys), so we bracket
                     and let the analysis pick the k whose realized
                     l0_per_token is nearest 20 (per layer, by mean over
                     seeds; loose match >25% off target flagged).
- txc_batchtopk_post k_pos=80 per-WINDOW budget (atoms reused at all
                     T=4 positions → 80/window = 20/token parity; the
                     arch docstring's own correction of the k_win
                     over-count).

d_sae = 32768 for every cell — the § 5.3 canonical em slice.
batch_size follows the paper c6 convention (1024) except tsae, whose
buffer consumes whole 128-token sequences: 32 seqs = 4096 token
positions/step ≈ the T=4 window archs' per-step token count (1024
windows × 4). n_steps 25_000, lr 3e-4, warmup 1000, bf16 — the paper's
c6 TrainingConfig (docs/components/c6.md via training_appendix).

Anchors (seed 42 only, after the panel): the paper's own § 5.3 pairing
re-trained on the organism-forward caches — txc_base with the
brickenauxk_a8 knobs (the ONLY cells with bricken on, matching the
paper's c6-only stance) and sae_arditi — bridging the published L15
numbers to the new substrate.
"""

from __future__ import annotations

from temp_bench.core.schemas import TrainingConfig

DATASOURCES = {
    9: "qwen_2_5_7b_organism_medical_l9",
    13: "qwen_2_5_7b_organism_medical_l13",
    15: "qwen_2_5_7b_organism_medical_l15",
}
LAYERS = [13, 15, 9]          # run order: the map's peak first
SEEDS = [42, 1, 2]            # canonical 3-seed set (paper em used {42,1})
N_STEPS = 25_000

# (cell_id, arch_name, batch_size, arch_hparams_override, extra_training_cfg)
PANEL = [
    ("batchtopk_sae", "batchtopk_sae",      1024, {"d_sae": 32768}, {}),
    ("txc_post_k80",  "txc_batchtopk_post", 1024, {"d_sae": 32768, "k_pos": 80}, {}),
    ("txc_pre_k20",   "txc_batchtopk_pre",  1024, {"d_sae": 32768}, {}),
    ("txc_pre_k40",   "txc_batchtopk_pre",  1024, {"d_sae": 32768, "k_pos": 40}, {}),
    ("tsae",          "tsae",               32,   {"d_sae": 32768}, {}),
]

ANCHORS = [
    ("txc_base_anchor",   "txc_base",   1024, None,
     {"bricken_enabled": True, "ema_auxk_alpha": 0.125,
      "dead_threshold_tokens": 128_000}),
    ("sae_arditi_anchor", "sae_arditi", 1024, None, {}),
]
ANCHOR_SEEDS = [42]


def training_cfg_for(batch_size: int, override: dict | None,
                     extra: dict) -> TrainingConfig:
    return TrainingConfig(
        n_steps=N_STEPS, batch_size=batch_size,
        arch_hparams_override=override, **extra)


def all_cells(include_anchors: bool = True):
    """Yield dicts in the frozen run order."""
    for layer in LAYERS:
        for cell_id, arch, bs, ovr, extra in PANEL:
            for seed in SEEDS:
                yield {"cell_id": cell_id, "arch": arch, "layer": layer,
                       "seed": seed, "datasource": DATASOURCES[layer],
                       "training_cfg": training_cfg_for(bs, ovr, extra)}
    if include_anchors:
        for layer in LAYERS:
            for cell_id, arch, bs, ovr, extra in ANCHORS:
                for seed in ANCHOR_SEEDS:
                    yield {"cell_id": cell_id, "arch": arch, "layer": layer,
                           "seed": seed, "datasource": DATASOURCES[layer],
                           "training_cfg": training_cfg_for(bs, ovr, extra)}
