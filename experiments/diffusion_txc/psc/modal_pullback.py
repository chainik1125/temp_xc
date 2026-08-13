"""P5 pullback probe (theory note §8, pre-registered 2026-08-12).

If base->distill activation drift is approximately affine, a ridge-fitted
per-token map from distill resid_L10 back to base coordinates should
partially restore the base-trained w6mix_dsm dictionary at the steering
site with zero retraining. Pre-registered thresholds: map R^2 >= 0.7 on
held-out tokens; revival >= 300 live latents (vs 852 unmapped -> see K5);
window NMSE <= 0.55 (vs 0.807 unmapped). K5: R^2 >= 0.7 but revival < 100
or NMSE >= 0.7 kills the coordinate-change reading (feature birth).

    uvx modal run --detach experiments/diffusion_txc/psc/modal_pullback.py::main
"""

import json
import pathlib

import modal

try:
    ROOT = pathlib.Path(__file__).resolve().parents[3]
except IndexError:
    ROOT = pathlib.Path("/work")

app = modal.App("dtxc-pullback")
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
CKPT = "txc_w6/txc_dsm/dsm_s0.pt"      # instrument cross-check arm


@app.function(image=image, gpu="A100-40GB", timeout=3600,
              volumes={"/vol": vol}, secrets=[hf_secret], memory=32768)
def probe(n_traces: int = 50, max_tok: int = 1200, ridge: float = 1.0) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda"
    traces = json.loads(pathlib.Path("/work/traces.json").read_text())
    texts = [t["prompt"] + t["full_response"] for t in traces][:n_traces]

    def capture(name):
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.bfloat16, output_hidden_states=True
        ).to(dev).eval()
        acts = []
        with torch.no_grad():
            for tx in texts:
                ids = tok(tx, return_tensors="pt", truncation=True,
                          max_length=max_tok)["input_ids"].to(dev)
                acts.append(model(ids).hidden_states[11][0].float().cpu())
        del model
        torch.cuda.empty_cache()
        return acts

    # same text, same truncation -> token-aligned pairs (identical tokenizer
    # family; verify lengths match and drop any trace where they don't)
    A_d = capture("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    A_b = capture("meta-llama/Llama-3.1-8B")
    pairs = [(d, b) for d, b in zip(A_d, A_b) if d.shape[0] == b.shape[0]]
    n_cal = int(0.6 * len(pairs))
    Xd = torch.cat([p[0] for p in pairs[:n_cal]]).to(dev)
    Xb = torch.cat([p[1] for p in pairs[:n_cal]]).to(dev)
    # ridge fit with intercept: Xb ~ Xd @ M + c
    mu_d, mu_b = Xd.mean(0), Xb.mean(0)
    Xdc, Xbc = Xd - mu_d, Xb - mu_b
    G = Xdc.T @ Xdc + ridge * torch.eye(4096, device=dev)
    M = torch.linalg.solve(G, Xdc.T @ Xbc)
    Ed = torch.cat([p[0] for p in pairs[n_cal:]]).to(dev)
    Eb = torch.cat([p[1] for p in pairs[n_cal:]]).to(dev)
    pred = (Ed - mu_d) @ M + mu_b
    r2 = float(1 - (Eb - pred).pow(2).sum() / (Eb - Eb.mean(0)).pow(2).sum())

    sd = torch.load(f"/vol/{CKPT}", weights_only=True, map_location=dev)
    W_enc, b_enc = sd["W_enc"].float(), sd["b_enc"].float()
    W_dec, b_dec = sd["W_dec"].float(), sd["b_dec"].float()

    def eval_windows(token_acts):
        fired = torch.zeros(H, dtype=torch.bool, device=dev)
        num = den = n_win = 0.0
        mu_run = None
        for hs in token_acts:
            hs = hs.to(dev)
            if hs.shape[0] < T:
                continue
            Wn = hs.unfold(0, T, 1).permute(0, 2, 1).reshape(-1, T * 4096)[::3]
            if mu_run is None:
                mu_run = Wn.mean(0)
            p = (Wn - b_dec) @ W_enc + b_enc
            v, i = p.topk(K, dim=1)
            Z = torch.zeros_like(p).scatter(1, i, torch.relu(v))
            fired |= (Z > 0).any(0)
            R = Z @ W_dec + b_dec
            num += float((Wn - R).pow(2).sum())
            den += float((Wn - mu_run).pow(2).sum())
            n_win += Wn.shape[0]
        return {"nmse": num / den, "live": int(fired.sum()),
                "n_windows": int(n_win)}

    eval_traces_d = [p[0] for p in pairs[n_cal:]]
    raw = eval_windows(eval_traces_d)
    mapped = eval_windows([(d.to(dev) - mu_d.cpu().to(dev)) @ M + mu_b
                           for d in eval_traces_d])
    out = {"map_r2_heldout": r2, "n_pairs_used": len(pairs),
           "ridge": ridge, "ckpt": CKPT,
           "distill_raw": raw, "distill_pulled_back": mapped,
           "prereg": {"r2_min": 0.7, "revival_min": 300, "nmse_max": 0.55,
                      "k5_kill": "r2>=0.7 but live<100 or nmse>=0.7"}}
    pathlib.Path("/vol/ooc_recal").mkdir(exist_ok=True)
    pathlib.Path("/vol/ooc_recal/pullback_fineweb_dict.json").write_text(json.dumps(out))
    vol.commit()
    print(json.dumps(out), flush=True)
    return out


@app.local_entrypoint()
def main():
    print(json.dumps(probe.remote(), indent=1))
