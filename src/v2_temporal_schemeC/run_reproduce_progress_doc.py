"""Reproduce ALL results from docs/han/research_logs/2026-03-04-v2-progress.md
using the unified experiment framework.

Runs Experiments 1 and 2 for all model variants, saves results to
results/reproduction/, and generates a comparison report.

Usage:
  TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/home/elysium/temp_xc \
    /home/elysium/miniforge3/envs/torchgpu/bin/python -u \
    src/v2_temporal_schemeC/run_reproduce_progress_doc.py
"""

import json
import os
import sys
import time
from dataclasses import asdict

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch

from src.v2_temporal_schemeC.experiment import (
    DataConfig, build_data_pipeline,
    SAEModelSpec, TFAModelSpec, TXCDRModelSpec, ModelEntry,
    EvalResult, evaluate_model,
    run_topk_sweep, run_l1_sweep,
    save_results,
)
from src.utils.seed import set_seed

# ── Configuration (matches docs/han/research_logs/2026-03-04-v2-progress.md) ──

DATA_CFG = DataConfig(
    num_features=20,
    hidden_dim=40,
    seq_len=64,
    pi=[0.5] * 20,
    rho=[0.0] * 4 + [0.3] * 4 + [0.5] * 4 + [0.7] * 4 + [0.9] * 4,
    dict_width=40,
    seed=42,
    eval_n_seq=2000,
)

TOPK_K_VALUES = [1, 3, 5, 8, 10, 15, 20]

# L1 ranges (from original scripts)
L1_COEFFS_SAE = np.logspace(-2.3, 1.3, 15).tolist()
L1_COEFFS_TFA = np.logspace(-0.8, 1.8, 12).tolist()
L1_COEFFS_TXCDR = np.logspace(-1.5, 1.5, 12).tolist()

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "reproduction")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"Results dir: {RESULTS_DIR}", flush=True)
    t_start = time.time()

    # ════════════════════════════════════════════════════════════════
    # EXPERIMENT 1: TopK sweep — all 7 models
    # ════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}", flush=True)
    print("EXPERIMENT 1: TopK sweep", flush=True)
    print(f"{'='*70}", flush=True)

    exp1_models = [
        ModelEntry("SAE", SAEModelSpec(), "flat",
                   training_overrides={"total_steps": 30_000, "batch_size": 4096, "lr": 3e-4}),
        ModelEntry("TFA", TFAModelSpec(n_heads=4, n_attn_layers=1, bottleneck_factor=1), "seq",
                   training_overrides={"total_steps": 30_000, "batch_size": 64, "lr": 1e-3}),
        ModelEntry("TFA-shuf", TFAModelSpec(n_heads=4, n_attn_layers=1, bottleneck_factor=1), "seq_shuffled",
                   training_overrides={"total_steps": 30_000, "batch_size": 64, "lr": 1e-3}),
        ModelEntry("TFA-pos", TFAModelSpec(use_pos_encoding=True, n_heads=4, n_attn_layers=1, bottleneck_factor=1), "seq",
                   training_overrides={"total_steps": 30_000, "batch_size": 64, "lr": 1e-3}),
        ModelEntry("TFA-pos-shuf", TFAModelSpec(use_pos_encoding=True, n_heads=4, n_attn_layers=1, bottleneck_factor=1), "seq_shuffled",
                   training_overrides={"total_steps": 30_000, "batch_size": 64, "lr": 1e-3}),
        ModelEntry("TXCDR T=2", TXCDRModelSpec(T=2), "window_2",
                   training_overrides={"total_steps": 80_000, "batch_size": 2048, "lr": 3e-4}),
        ModelEntry("TXCDR T=5", TXCDRModelSpec(T=5), "window_5",
                   training_overrides={"total_steps": 80_000, "batch_size": 2048, "lr": 3e-4}),
    ]

    exp1_results = run_topk_sweep(
        models=exp1_models,
        k_values=TOPK_K_VALUES,
        data_config=DATA_CFG,
        device=DEVICE,
        compute_auc=True,
        verbose=True,
    )

    # Save Exp 1
    save_results(
        os.path.join(RESULTS_DIR, "exp1_topk.json"),
        experiment_type="topk_sweep",
        config=asdict(DATA_CFG),
        model_results=exp1_results,
    )

    # Print Exp 1 summary
    print(f"\n{'='*70}", flush=True)
    print("Experiment 1 Summary (NMSE):", flush=True)
    header = f"{'k':>3}"
    for name in exp1_results:
        header += f" | {name:>12}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for i, k in enumerate(TOPK_K_VALUES):
        row = f"{k:>3}"
        for name in exp1_results:
            row += f" | {exp1_results[name][i].nmse:>12.6f}"
        print(row, flush=True)

    print(f"\nExperiment 1 Summary (AUC):", flush=True)
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for i, k in enumerate(TOPK_K_VALUES):
        row = f"{k:>3}"
        for name in exp1_results:
            auc = exp1_results[name][i].auc
            row += f" | {auc:>12.4f}" if auc else f" | {'N/A':>12}"
        print(row, flush=True)

    # ════════════════════════════════════════════════════════════════
    # EXPERIMENT 2: ReLU+L1 Pareto — SAE, TFA, TFA-pos, TXCDR T=2, T=5
    # ════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}", flush=True)
    print("EXPERIMENT 2: ReLU+L1 Pareto", flush=True)
    print(f"{'='*70}", flush=True)

    exp2_models = [
        ModelEntry("SAE", SAEModelSpec(), "flat",
                   training_overrides={"total_steps": 30_000, "batch_size": 4096, "lr": 3e-4}),
        ModelEntry("TFA", TFAModelSpec(n_heads=4, n_attn_layers=1, bottleneck_factor=1), "seq",
                   training_overrides={"total_steps": 30_000, "batch_size": 64, "lr": 1e-3}),
        ModelEntry("TFA-pos", TFAModelSpec(use_pos_encoding=True, n_heads=4, n_attn_layers=1, bottleneck_factor=1), "seq",
                   training_overrides={"total_steps": 30_000, "batch_size": 64, "lr": 1e-3}),
        ModelEntry("TXCDR T=2", TXCDRModelSpec(T=2), "window_2",
                   training_overrides={"total_steps": 80_000, "batch_size": 2048, "lr": 3e-4}),
        ModelEntry("TXCDR T=5", TXCDRModelSpec(T=5), "window_5",
                   training_overrides={"total_steps": 80_000, "batch_size": 2048, "lr": 3e-4}),
    ]

    exp2_l1_coeffs = {
        "SAE": L1_COEFFS_SAE,
        "TFA": L1_COEFFS_TFA,
        "TFA-pos": L1_COEFFS_TFA,
        "TXCDR T=2": L1_COEFFS_TXCDR,
        "TXCDR T=5": L1_COEFFS_TXCDR,
    }

    exp2_results = run_l1_sweep(
        models=exp2_models,
        l1_coeffs=exp2_l1_coeffs,
        data_config=DATA_CFG,
        device=DEVICE,
        compute_auc=True,
        verbose=True,
    )

    # Save Exp 2
    save_results(
        os.path.join(RESULTS_DIR, "exp2_l1.json"),
        experiment_type="l1_sweep",
        config=asdict(DATA_CFG),
        model_results=exp2_results,
    )

    # ════════════════════════════════════════════════════════════════
    # COMPARISON with existing results
    # ════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}", flush=True)
    print("COMPARISON WITH EXISTING RESULTS", flush=True)
    print(f"{'='*70}", flush=True)

    # Load old results
    base = os.path.dirname(__file__)
    old_files = {
        "auc_and_crosscoder": os.path.join(base, "results", "auc_and_crosscoder", "results.json"),
        "tfa_pos": os.path.join(base, "results", "tfa_pos", "results.json"),
        "txcdr_T5": os.path.join(base, "results", "txcdr_T5", "results.json"),
        "tfa_l1_total_l0": os.path.join(base, "results", "tfa_l1_total_l0", "results.json"),
    }

    old_data = {}
    for name, path in old_files.items():
        if os.path.exists(path):
            with open(path) as f:
                old_data[name] = json.load(f)
            print(f"  Loaded {name}: {path}", flush=True)
        else:
            print(f"  MISSING: {path}", flush=True)

    discrepancies = []

    # Compare Exp 1 TopK
    if "auc_and_crosscoder" in old_data:
        old = old_data["auc_and_crosscoder"]["exp1"]

        model_map = {
            "SAE": ("sae", "nmse", "auc"),
            "TFA": ("tfa", "nmse", "auc"),
            "TFA-shuf": ("tfa_shuf", "nmse", "auc"),
            "TXCDR T=2": ("txcdr", "nmse", "auc"),
        }

        print(f"\n--- Exp 1 TopK: Comparing against auc_and_crosscoder ---", flush=True)
        for new_name, (old_key, nmse_key, auc_key) in model_map.items():
            if old_key not in old or new_name not in exp1_results:
                continue
            print(f"\n  {new_name}:", flush=True)
            for i, k in enumerate(TOPK_K_VALUES):
                if i >= len(old[old_key]):
                    break
                old_nmse = old[old_key][i][nmse_key]
                new_nmse = exp1_results[new_name][i].nmse
                nmse_diff = abs(old_nmse - new_nmse)

                old_auc = old[old_key][i].get(auc_key, None)
                new_auc = exp1_results[new_name][i].auc
                auc_diff = abs(old_auc - new_auc) if old_auc and new_auc else 0

                flag_nmse = " *** DISCREPANCY ***" if nmse_diff > 1e-5 else ""
                flag_auc = " *** DISCREPANCY ***" if auc_diff > 1e-5 else ""

                print(f"    k={k:>2}: NMSE old={old_nmse:.10f} new={new_nmse:.10f} diff={nmse_diff:.2e}{flag_nmse}", flush=True)
                if old_auc:
                    print(f"          AUC  old={old_auc:.10f} new={new_auc:.10f} diff={auc_diff:.2e}{flag_auc}", flush=True)

                if nmse_diff > 1e-5:
                    discrepancies.append(f"Exp1 {new_name} k={k} NMSE: old={old_nmse:.10f} new={new_nmse:.10f} diff={nmse_diff:.2e}")
                if auc_diff > 1e-5:
                    discrepancies.append(f"Exp1 {new_name} k={k} AUC: old={old_auc:.10f} new={new_auc:.10f} diff={auc_diff:.2e}")

    # Compare TFA-pos
    if "tfa_pos" in old_data:
        old_pos = old_data["tfa_pos"]
        print(f"\n--- Exp 1 TopK: Comparing TFA-pos against tfa_pos results ---", flush=True)
        for new_name, old_key in [("TFA-pos", "exp1_tfa_pos"), ("TFA-pos-shuf", "exp1_tfa_pos_shuf")]:
            if old_key not in old_pos or new_name not in exp1_results:
                continue
            print(f"\n  {new_name}:", flush=True)
            for i, k in enumerate(TOPK_K_VALUES):
                if i >= len(old_pos[old_key]):
                    break
                old_nmse = old_pos[old_key][i]["nmse"]
                new_nmse = exp1_results[new_name][i].nmse
                diff = abs(old_nmse - new_nmse)
                flag = " *** DISCREPANCY ***" if diff > 1e-5 else ""
                print(f"    k={k:>2}: NMSE old={old_nmse:.10f} new={new_nmse:.10f} diff={diff:.2e}{flag}", flush=True)
                if diff > 1e-5:
                    discrepancies.append(f"Exp1 {new_name} k={k} NMSE: diff={diff:.2e}")

    # Compare TXCDR T=5
    if "txcdr_T5" in old_data:
        old_t5 = old_data["txcdr_T5"]
        print(f"\n--- Exp 1 TopK: Comparing TXCDR T=5 ---", flush=True)
        for old_r in old_t5["topk"]:
            k = old_r["k"]
            if k not in TOPK_K_VALUES:
                continue
            idx = TOPK_K_VALUES.index(k)
            new_nmse = exp1_results["TXCDR T=5"][idx].nmse
            diff = abs(old_r["nmse"] - new_nmse)
            flag = " *** DISCREPANCY ***" if diff > 1e-5 else ""
            print(f"    k={k:>2}: NMSE old={old_r['nmse']:.10f} new={new_nmse:.10f} diff={diff:.2e}{flag}", flush=True)
            if diff > 1e-5:
                discrepancies.append(f"Exp1 TXCDR T=5 k={k} NMSE: diff={diff:.2e}")

    # ── Summary ──────────────────────────────────────────────────

    elapsed = time.time() - t_start
    print(f"\n{'='*70}", flush=True)
    print(f"REPRODUCTION COMPLETE in {elapsed/3600:.1f}h", flush=True)
    print(f"{'='*70}", flush=True)

    if discrepancies:
        print(f"\n*** {len(discrepancies)} DISCREPANCIES FOUND ***", flush=True)
        for d in discrepancies:
            print(f"  {d}", flush=True)
    else:
        print(f"\nAll results match within 1e-5 tolerance.", flush=True)

    # Save comparison report
    report = {
        "elapsed_seconds": elapsed,
        "n_discrepancies": len(discrepancies),
        "discrepancies": discrepancies,
    }
    with open(os.path.join(RESULTS_DIR, "comparison_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}/", flush=True)


if __name__ == "__main__":
    main()
