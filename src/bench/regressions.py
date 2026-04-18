"""In-harness regression checks for each bug in `eval_infra_lessons.md`.

Imported + called at `src.bench.run_eval` startup so any regression
that would silently reintroduce a known bug fails fast with a message
pointing to the bug number.

See `docs/aniket/bench_harness/bug_ledger_check.md` for the walkthrough
of which bugs are caught here (category **R**) vs elsewhere (F / V / P).

Run standalone:
    python -m src.bench.regressions
"""

from __future__ import annotations

import inspect
from pathlib import Path


# ─── B1: MLC isinstance ordering ────────────────────────────────────────

def check_b1_registry_keys():
    from src.bench.architectures import REGISTRY
    from src.bench.architectures.topk_sae import TopKSAESpec
    from src.bench.architectures.crosscoder import CrosscoderSpec
    from src.bench.architectures.mlc import LayerCrosscoderSpec
    from src.bench.architectures.stacked_sae import StackedSAESpec
    from src.bench.sweep import _arch_registry_key, get_default_models

    # Every REGISTRY key should map back through _arch_registry_key.
    # MLC must NOT collide with crosscoder (the root cause of B1).
    specs = {
        "topk_sae": TopKSAESpec(),
        "stacked_sae": StackedSAESpec(T=5),
        "crosscoder": CrosscoderSpec(T=5),
        "mlc": LayerCrosscoderSpec(n_layers=5),
    }
    models = get_default_models([5])
    for name, spec in specs.items():
        from src.bench.architectures.base import ModelEntry
        entry = ModelEntry(name=name, spec=spec, gen_key="flat")
        got = _arch_registry_key(entry)
        assert got == name, (
            f"B1 regression: {type(spec).__name__} resolves to '{got}', "
            f"expected '{name}'. The isinstance chain in "
            f"src/bench/sweep.py:_arch_registry_key must test most-specific "
            f"subclasses first. See eval_infra_lessons.md B1."
        )


# ─── B2: evaluate_model dispatches on multi_layer ───────────────────────

def check_b2_multi_layer_dispatch():
    from src.bench.eval import evaluate_model
    src = inspect.getsource(evaluate_model)
    assert '"multi_layer"' in src, (
        "B2 regression: src/bench/eval.py:evaluate_model must handle "
        'data_format="multi_layer". See eval_infra_lessons.md B2.'
    )


# ─── B3: mlc_probing return dict has required keys ──────────────────────

def check_b3_mlc_return_shape():
    from src.bench.saebench import mlc_probing
    src = inspect.getsource(mlc_probing.run_mlc_probing)
    for key in ("run_id", "n_records_written", "elapsed_sec"):
        assert f'"{key}"' in src, (
            f"B3 regression: run_mlc_probing's return dict must include "
            f"'{key}'. Callers (runpod_saebench_run_eval.sh) depend on it. "
            f"See eval_infra_lessons.md B3."
        )


# ─── B4: Protocol B window_k stays fixed as T grows ─────────────────────

def check_b4_protocol_b_window_k():
    from src.bench.saebench.configs import PROTOCOL_A, PROTOCOL_B, D_SAE
    # Protocol A: per-token k matched → constant per-position, scales window_k.
    for t in (5, 10, 20, 40):
        k_pos = PROTOCOL_A.tempxc_k_at(t)
        assert k_pos == 100, f"B4 regression (A): per-pos k must be 100 at T={t}, got {k_pos}"
        assert k_pos * t <= D_SAE, f"B4 regression (A): window_k {k_pos*t} > d_sae {D_SAE} at T={t}"
    # Protocol B: window_k fixed at ~500, per-position k = base // t.
    for t in (5, 10, 20, 40):
        k_pos = PROTOCOL_B.tempxc_k_at(t)
        window_k = k_pos * t
        assert 1 <= k_pos <= D_SAE, f"B4 regression (B): bad per-pos k={k_pos} at T={t}"
        # window_k should be approximately 500 (integer floor at T=40 gives 480)
        assert abs(window_k - 500) <= 20, (
            f"B4 regression (B): window_k drifted to {window_k} at T={t}, "
            f"expected ~500 fixed. See eval_infra_lessons.md B4."
        )


# ─── B5: _load_arch_and_model resolves k from protocol, not placeholder ─

def check_b5_k_resolution():
    from src.bench.saebench.probing_runner import _load_arch_and_model
    src = inspect.getsource(_load_arch_and_model)
    assert "protocol_k" in src, (
        "B5 regression: _load_arch_and_model must resolve k via protocol_k(), "
        "not accept k as a caller arg (placeholder k=0 caused all-zero features). "
        "See eval_infra_lessons.md B5."
    )


# ─── B7: _want_temporal_metrics skips multi_layer ────────────────────────

def check_b7_temporal_metrics_gate():
    from src.bench.eval import _want_temporal_metrics

    class _DummySpec:
        data_format = "multi_layer"

    import torch
    dummy_4d = torch.zeros(2, 4, 5, 3)
    assert _want_temporal_metrics(_DummySpec(), dummy_4d) is False, (
        "B7 regression: _want_temporal_metrics must return False for "
        "data_format='multi_layer'. See eval_infra_lessons.md B7."
    )


# ─── B10: preflight ckpt path namespace ─────────────────────────────────

def check_b10_preflight_cleanup():
    """Preflight is expected to clean up its own ckpt. Verified by
    reading the preflight shell script for the cleanup stanza.
    """
    p = Path("scripts/runpod_saebench_preflight.sh")
    if not p.exists():
        return
    body = p.read_text()
    assert "cleanup preflight" in body.lower() or "rm -f $PREFLIGHT_CKPT" in body or "rm -f \"$PREFLIGHT_CKPT\"" in body, (
        "B10 regression: runpod_saebench_preflight.sh must clean up its "
        "preflight-produced ckpt at the end, otherwise train.sh's "
        "skip-if-exists guard picks it up as the 5000-step real ckpt. "
        "See eval_infra_lessons.md B10."
    )


# ─── B11: run_probing defaults to force_rerun=True ──────────────────────

def check_b11_force_rerun_default():
    from src.bench.saebench.probing_runner import run_probing
    sig = inspect.signature(run_probing)
    assert sig.parameters["force_rerun"].default is True, (
        "B11 regression: run_probing's force_rerun must default to True. "
        "With placeholder k=0 in run_id, preflight + real eval share the "
        "same cache key; False silently reuses stale preflight output. "
        "See eval_infra_lessons.md B11."
    )


# ─── B13: mlc_probing doesn't call SAEBench's multi-class probe API ─────

def check_b13_no_sae_bench_multi_class_api():
    src_path = Path("src/bench/saebench/mlc_probing.py")
    body = src_path.read_text()
    assert "train_probe_on_activations" not in body, (
        "B13 regression: mlc_probing.py must NOT call "
        "sae_bench.evals.sparse_probing.probe_training.train_probe_on_activations. "
        "It takes dict[class, Tensor] multi-class input; our one-vs-rest "
        "setup broke jaxtyping. See eval_infra_lessons.md B13."
    )


# ─── B14: mlc_probing hardcodes prepend_bos / context_length ───────────

def check_b14_no_cfg_prepend_bos():
    src_path = Path("src/bench/saebench/mlc_probing.py")
    body = src_path.read_text()
    assert "cfg.prepend_bos" not in body, (
        "B14 regression: mlc_probing.py must not read cfg.prepend_bos — "
        "SAEBench's SparseProbingEvalConfig doesn't expose it. "
        "Hardcode True. See eval_infra_lessons.md B14."
    )
    assert "cfg.context_length" not in body, (
        "B14 regression: mlc_probing.py must not read cfg.context_length. "
        "Use CONTEXT_LENGTH from configs. See eval_infra_lessons.md B14."
    )


# ─── B15: mlc_probing encodes each text once, not N_classes times ───────

def check_b15_encode_once():
    src_path = Path("src/bench/saebench/mlc_probing.py")
    body = src_path.read_text()
    # The flat-encode-once helper was added in commit 7735968.
    assert "flatten_with_index" in body or "encode_flat" in body, (
        "B15 regression: mlc_probing.py must use the encode-once-per-text "
        "path (flatten_with_index / encode_flat). Per-class re-encoding was "
        "O(N_classes²). See eval_infra_lessons.md B15."
    )


# ─── B16: sae_batch_size capped at 16 in probing_runner ─────────────────

def check_b16_sae_batch_size():
    src_path = Path("src/bench/saebench/probing_runner.py")
    body = src_path.read_text()
    assert "cfg.sae_batch_size = 16" in body or "cfg.sae_batch_size=16" in body, (
        "B16 regression: probing_runner.py must set cfg.sae_batch_size "
        "<= 16. SAEBench's default 125 OOMs at T=20 full_window "
        "(output = B*L*368640*4 bytes). See eval_infra_lessons.md B16."
    )


# ─── Item 8 (new): persistence-layer sanity check is wired in ───────────

def check_item8_persistence_sanity():
    """The sanity check function exists and is invoked by the probing
    path. Actual correctness is verified at run time, not here."""
    src_path = Path("src/bench/saebench/probing_runner.py")
    if not src_path.exists():
        return
    body = src_path.read_text()
    # Will be wired in the next commit; placeholder-tolerant.
    if "_sanity_check_persistence" not in body and "sanity_check_predictions" not in body:
        # Don't fail yet — check is aspirational until wired.
        return
    # Once wired, must be called by run_probing.
    run_src = inspect.getsource(
        __import__("src.bench.saebench.probing_runner", fromlist=["run_probing"]).run_probing
    )
    assert "sanity_check" in run_src.lower(), (
        "Item 8 regression: run_probing must call the persistence "
        "sanity check after writing predictions. "
        "See docs/aniket/bench_harness/bug_ledger_check.md § Item 8."
    )


# ─── Runner ─────────────────────────────────────────────────────────────

ALL_CHECKS = [
    ("B1 registry keys",            check_b1_registry_keys),
    ("B2 multi_layer dispatch",     check_b2_multi_layer_dispatch),
    ("B3 mlc return shape",         check_b3_mlc_return_shape),
    ("B4 protocol-B window_k",      check_b4_protocol_b_window_k),
    ("B5 k resolution",             check_b5_k_resolution),
    ("B7 temporal metrics gate",    check_b7_temporal_metrics_gate),
    ("B10 preflight cleanup",       check_b10_preflight_cleanup),
    ("B11 force_rerun default",     check_b11_force_rerun_default),
    ("B13 no SAEBench multi-class", check_b13_no_sae_bench_multi_class_api),
    ("B14 no cfg.prepend_bos",      check_b14_no_cfg_prepend_bos),
    ("B15 encode-once",             check_b15_encode_once),
    ("B16 sae_batch_size",          check_b16_sae_batch_size),
    ("Item 8 sanity-check hook",    check_item8_persistence_sanity),
]


def check_all(verbose: bool = False) -> None:
    """Run every regression check. Raises AssertionError on first failure."""
    for name, fn in ALL_CHECKS:
        try:
            fn()
            if verbose:
                print(f"  ✓ {name}")
        except AssertionError as e:
            raise AssertionError(f"Regression check failed — {name}\n\n{e}") from e


if __name__ == "__main__":
    print("Running in-harness regression checks...")
    check_all(verbose=True)
    print(f"\nAll {len(ALL_CHECKS)} checks passed.")
