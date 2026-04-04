"""Re-run Stacked-T5 and TXCDR-T5 at k=10 (skipped by reeval_corr_sweep.py due to wrong guard).

These models use k latents per position (not k*T total), so k=10 is valid with d_sae=40.
"""
import json, os, sys, time
import torch
sys.stdout.reconfigure(line_buffering=True)
from src.utils.seed import set_seed
from src.v2_temporal_schemeC.experiment import (
    DataConfig, build_data_pipeline,
    TXCDRModelSpec, StackedSAEModelSpec,
    evaluate_model,
)
DEVICE = torch.device("cuda")
RHO_VALUES = [0.0, 0.3, 0.5, 0.7, 0.9]
K = 10
SPECS = {
    "Stacked-T5": StackedSAEModelSpec(T=5),
    "TXCDR-T5": TXCDRModelSpec(T=5),
}
out_dir = os.path.join(os.path.dirname(__file__), "results", "correlation_sweep")

for name, spec in SPECS.items():
    print(f"\n=== {name} k={K} ===", flush=True)
    # Load existing JSON and add k=10 entries
    json_path = os.path.join(out_dir, f"{name}_corr.json")
    with open(json_path) as f:
        data = json.load(f)

    for rho in RHO_VALUES:
        cfg = DataConfig(num_features=20, hidden_dim=40, seq_len=64,
            pi=[0.5]*20, rho=[rho]*20, dict_width=40, seed=42, eval_n_seq=2000)
        pipeline = build_data_pipeline(cfg, DEVICE, window_sizes=[5])
        gen_fn = pipeline.gen_windows[5]
        set_seed(42); t0 = time.time()
        model = spec.create(d_in=40, d_sae=40, k=K, device=DEVICE)
        config = spec.make_train_config(total_steps=30000, batch_size=2048, lr=3e-4, log_every=30000)
        model, _ = spec.train(model, gen_fn, config, DEVICE)
        r = evaluate_model(spec, model, pipeline.eval_hidden, DEVICE,
                           true_features=pipeline.true_features, seq_len=64)
        # Update the results: find or append k=10 entry
        rho_str = str(rho)
        existing = [e for e in data["results"][rho_str] if e.get("k") != K]
        existing.append(r.to_dict() | {"k": K})
        data["results"][rho_str] = existing
        print(f"  rho={rho} k={K}: NMSE={r.nmse:.6f} AUC={r.auc:.4f} ({time.time()-t0:.0f}s)", flush=True)
        del model; torch.cuda.empty_cache()

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2,
                  default=lambda x: float(x) if hasattr(x,'item') else x)
    print(f"Updated: {json_path}", flush=True)

print("\nDone", flush=True)
