"""Shallow-vs-deep OOD-collapse probe (distill recalibration, zero training).

Question: w6_dsm's live pool collapses to ~4% on distill activations. Are
the dead latents' DIRECTIONS still useful there (shallow collapse — fixable
by threshold recalibration) or specialised away (deep)?

Test: per-latent uniform-rate threshold gate calibrated ON distill windows
(theta_i at the (1 - k/H) quantile of latent i's own preactivation) forces
every latent to fire at the target rate. If dsm's recalibrated NMSE drops
far below its TopK NMSE (and recon's drop is smaller), the directions are
present and the collapse is calibration-deep only.

    uvx modal run --detach experiments/diffusion_txc/psc/modal_ooc_recal.py
"""

import json
import pathlib

import modal

try:
    ROOT = pathlib.Path(__file__).resolve().parents[3]
except IndexError:
    ROOT = pathlib.Path("/work")

app = modal.App("dtxc-ooc-recal")
vol = modal.Volume.from_name("diffusion-txc", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "numpy", "transformers==4.46.2",
                 "accelerate", "sentencepiece")
    .add_local_file(str(ROOT / "results/ward_backtracking/traces.json"),
                    "/work/traces.json")
)

ARMS = {"w6_dsm": "txc_w6/txc_dsm/dsm_s0.pt",
        "w6_recon": "txc_w6/txc_recon/recon_s0.pt"}
T, K, H = 6, 96, 16384


@app.function(image=image, gpu="L40S", timeout=3600, volumes={"/vol": vol},
              secrets=[hf_secret], memory=16384)
def probe(n_traces: int = 60, max_tok: int = 1400) -> dict:
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
            hs = model(ids).hidden_states[11][0].float()       # (L, 4096)
            if hs.shape[0] < T:
                continue
            w = hs.unfold(0, T, 1).permute(0, 2, 1).reshape(-1, T * 4096)
            wins.append(w[::3].cpu())                          # stride-3 thin
    W = torch.cat(wins)
    n_cal = int(0.6 * W.shape[0])
    Wc, We = W[:n_cal], W[n_cal:]
    out = {"n_windows": int(W.shape[0]), "n_traces": len(texts),
           "model": name, "hook": "resid_L10(hidden_states[11])"}

    del model
    torch.cuda.empty_cache()

    for arm, path in ARMS.items():
        sd = torch.load(f"/vol/{path}", weights_only=True, map_location=dev)
        W_enc = sd["W_enc"].float()
        b_enc = sd["b_enc"].float()
        W_dec = sd["W_dec"].float()
        b_dec = sd["b_dec"].float()

        def pre_of(X):
            return (X.to(dev) - b_dec) @ W_enc + b_enc

        def nmse_of(Z, X):
            R = Z @ W_dec + b_dec
            X = X.to(dev)
            return float((X - R).pow(2).sum() / (X - X.mean(0)).pow(2).sum())

        # baseline: native TopK-96 on eval windows
        stats = {}
        fired = torch.zeros(H, dtype=torch.bool, device=dev)
        num = den = 0.0
        mu = We.mean(0).to(dev)
        for b0 in range(0, We.shape[0], 2048):
            Xb = We[b0:b0 + 2048].to(dev)
            p = pre_of(Xb)
            v, i = p.topk(K, dim=1)
            Z = torch.zeros_like(p).scatter(1, i, torch.relu(v))
            fired |= (Z > 0).any(0)
            R = Z @ W_dec + b_dec
            num += float((Xb - R).pow(2).sum())
            den += float((Xb - mu).pow(2).sum())
        stats["topk_nmse"] = num / den
        stats["topk_live"] = int(fired.sum())

        # recalibrated: per-latent theta at (1 - K/H) quantile on calib set
        pres = []
        for b0 in range(0, Wc.shape[0], 2048):
            pres.append(pre_of(Wc[b0:b0 + 2048]).cpu())
        P = torch.cat(pres)
        q = 1.0 - K / H
        Ps, _ = P.sort(dim=0)
        theta = Ps[int(q * (P.shape[0] - 1))].to(dev)
        fired2 = torch.zeros(H, dtype=torch.bool, device=dev)
        num2 = l0s = 0.0
        for b0 in range(0, We.shape[0], 2048):
            Xb = We[b0:b0 + 2048].to(dev)
            p = pre_of(Xb)
            Z = torch.relu(p) * (p > theta)
            fired2 |= (Z > 0).any(0)
            l0s += float((Z > 0).float().sum())
            R = Z @ W_dec + b_dec
            num2 += float((Xb - R).pow(2).sum())
        stats["recal_nmse"] = num2 / den
        stats["recal_live"] = int(fired2.sum())
        stats["recal_mean_l0"] = l0s / We.shape[0]
        out[arm] = stats
        print(arm, json.dumps(stats), flush=True)

    pathlib.Path("/vol/ooc_recal").mkdir(exist_ok=True)
    pathlib.Path("/vol/ooc_recal/results.json").write_text(json.dumps(out))
    vol.commit()
    return out


@app.local_entrypoint()
def main():
    print(json.dumps(probe.remote(), indent=1))
