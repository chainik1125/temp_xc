"""Re-run correlation sweep for Stacked SAE and TXCDR models with correct decoder-averaged AUC.

The original correlation sweep JSONs (Stacked-T2/T5_corr.json, TXCDR-T2/T5_corr.json) were
generated before the decoder-averaging fix. This script retrains and re-evaluates to produce
correct AUC values.

Usage:
  TQDM_DISABLE=1 PYTHONPATH=/home/elysium/temp_xc \
    /home/elysium/miniforge3/envs/torchgpu/bin/python -u \
    src/v2_temporal_schemeC/reeval_corr_sweep.py
"""
import json, os, sys, time
import torch
sys.stdout.reconfigure(line_buffering=True)
from src.utils.seed import set_seed
from src.pipeline.toy_models import (DataConfig, build_data_pipeline, TXCDRModelSpec, StackedSAEModelSpec)
from src.eval.toy_unified import evaluate_model
DEVICE = torch.device("cuda")
RHO_VALUES = [0.0, 0.3, 0.5, 0.7, 0.9]
K_VALUES = [3, 10]
SPECS = {
    "Stacked-T2": StackedSAEModelSpec(T=2),
    "Stacked-T5": StackedSAEModelSpec(T=5),
    "TXCDR-T2": TXCDRModelSpec(T=2),
    "TXCDR-T5": TXCDRModelSpec(T=5),
}
out_dir = os.path.join(os.path.dirname(__file__), "results", "correlation_sweep")

t_start = time.time()
for name, spec in SPECS.items():
    T = 2 if "T2" in name else 5
    print(f"\n=== {name} correlation sweep ===", flush=True)
    all_results = {}
    for rho in RHO_VALUES:
        cfg = DataConfig(num_features=20, hidden_dim=40, seq_len=64,
            pi=[0.5]*20, rho=[rho]*20, dict_width=40, seed=42, eval_n_seq=2000)
        pipeline = build_data_pipeline(cfg, DEVICE, window_sizes=[T])
        gen_fn = pipeline.gen_windows[T]
        results = []
        for k in K_VALUES:
            if k * T > 40:
                print(f"  rho={rho} k={k}: SKIP (k*T={k*T} > d_sae=40)", flush=True)
                continue
            set_seed(42); t0 = time.time()
            model = spec.create(d_in=40, d_sae=40, k=k, device=DEVICE)
            config = spec.make_train_config(total_steps=30000, batch_size=2048, lr=3e-4, log_every=30000)
            model, _ = spec.train(model, gen_fn, config, DEVICE)
            r = evaluate_model(spec, model, pipeline.eval_hidden, DEVICE,
                               true_features=pipeline.true_features, seq_len=64)
            results.append(r.to_dict() | {"k": k})
            print(f"  rho={rho} k={k}: NMSE={r.nmse:.6f} AUC={r.auc:.4f} ({time.time()-t0:.0f}s)", flush=True)
            del model; torch.cuda.empty_cache()
        all_results[str(rho)] = results
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}_corr.json"), "w") as f:
        json.dump({"model": name, "rho_values": RHO_VALUES, "k_values": K_VALUES,
                   "results": all_results}, f, indent=2,
                  default=lambda x: float(x) if hasattr(x,'item') else x)
    print(f"Done: {name}", flush=True)

print(f"\nAll done in {(time.time()-t_start)/60:.0f}m", flush=True)
