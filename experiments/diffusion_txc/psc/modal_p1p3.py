"""P1/P3 scorecard probe for the distill-captured pair (theory note §8).

Pre-registered (before training): P1 live pool on distill traces 600-2500
(central 5-10% of H); K2 kills the covering law if pool >= 50% of H.
P3: per-window positive preactivations 10^2-10^3 (vs 1.4 for the
base-captured dict). NMSE reported in BOTH conventions (variance-about-
dataset-mean = gate-style; energy-about-window-mean = probe-style).

    uvx modal run --detach experiments/diffusion_txc/psc/modal_p1p3.py::main
"""

import json
import pathlib

import modal

try:
    ROOT = pathlib.Path(__file__).resolve().parents[3]
except IndexError:
    ROOT = pathlib.Path("/work")

app = modal.App("dtxc-p1p3")
vol = modal.Volume.from_name("diffusion-txc", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "numpy", "transformers==4.46.2",
                 "accelerate", "sentencepiece")
    .add_local_file(str(ROOT / "results/ward_backtracking/traces.json"),
                    "/work/traces.json")
)

T, K, H = 6, 96, 16384
ARMS = {"w6dist_dsm": "txc_w6_distill/txc_dsm/dsm_s0.pt",
        "w6dist_recon": "txc_w6_distill/txc_recon/recon_s0.pt",
        "w6mix_dsm_ref": "txc_w6_mix/txc_dsm/dsm_s0.pt"}


@app.function(image=image, gpu="A100-40GB", timeout=3600,
              volumes={"/vol": vol}, secrets=[hf_secret], memory=32768)
def probe(n_traces: int = 60, max_tok: int = 1200) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda"
    name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, output_hidden_states=True
    ).to(dev).eval()
    traces = json.loads(pathlib.Path("/work/traces.json").read_text())
    texts = [t["prompt"] + t["full_response"] for t in traces][:n_traces]
    wins = []
    with torch.no_grad():
        for tx in texts:
            ids = tok(tx, return_tensors="pt", truncation=True,
                      max_length=max_tok)["input_ids"].to(dev)
            hs = model(ids).hidden_states[11][0].float()
            if hs.shape[0] >= T:
                w = hs.unfold(0, T, 1).permute(0, 2, 1).reshape(-1, T * 4096)
                wins.append(w[::3].cpu())
    del model
    torch.cuda.empty_cache()
    X = torch.cat(wins)
    mu_ds = X.mean(0).to(dev)                     # dataset mean (gate conv.)
    den_var = float((X.to("cpu") - mu_ds.cpu()).pow(2).sum())
    den_energy = float(X.pow(2).sum())

    out = {"n_windows": int(X.shape[0]), "hook": "distill resid_L10 T=6"}
    for arm, rel in ARMS.items():
        sd = torch.load(f"/vol/{rel}", weights_only=True, map_location=dev)
        W_enc, b_enc = sd["W_enc"].float(), sd["b_enc"].float()
        W_dec, b_dec = sd["W_dec"].float(), sd["b_dec"].float()
        fired = torch.zeros(H, dtype=torch.bool, device=dev)
        num = pos_sum = 0.0
        pos_counts = []
        for b0 in range(0, X.shape[0], 2048):
            Xb = X[b0:b0 + 2048].to(dev)
            p = (Xb - b_dec) @ W_enc + b_enc
            npos = (p > 0).sum(1).float()
            pos_counts.append(npos.cpu())
            v, i = p.topk(K, dim=1)
            Z = torch.zeros_like(p).scatter(1, i, torch.relu(v))
            fired |= (Z > 0).any(0)
            R = Z @ W_dec + b_dec
            num += float((Xb - R).pow(2).sum())
        pc = torch.cat(pos_counts)
        out[arm] = {"live_pool": int(fired.sum()),
                    "live_frac": round(float(fired.sum()) / H, 4),
                    "pos_preacts_per_window_median": float(pc.median()),
                    "pos_preacts_p10_p90": [float(pc.quantile(0.1)),
                                            float(pc.quantile(0.9))],
                    "nmse_var_conv": num / den_var,
                    "nmse_energy_conv": num / den_energy}
        print(arm, json.dumps(out[arm]), flush=True)
    pathlib.Path("/vol/ooc_recal/p1p3_scorecard.json").write_text(
        json.dumps(out))
    vol.commit()
    return out


@app.local_entrypoint()
def main():
    print(json.dumps(probe.remote(), indent=1))
