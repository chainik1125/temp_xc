"""Fragility decomposition for bayes_gate: is the support brittleness a
readout artifact or a dictionary property?

Three readouts on the same checkpoints/perturbations as the main eval:
  a) hard gate at sigma=0 (replicates the 0.52 regression),
  b) sigma-matched: perturbed input encoded with u = eps^2 — the model
     told the true noise level, i.e. used the way it was trained,
  c) rank top-k on preactivations (TopK-style relative support).

    uvx modal run --detach experiments/diffusion_txc/topk_vs_topkdiff/modal_frag_sigma.py::main
"""

import json
import pathlib

import modal

try:
    ROOT = pathlib.Path(__file__).resolve().parents[3]
except IndexError:
    ROOT = pathlib.Path("/work")

app = modal.App("topkdiff-frag-sigma")
vol = modal.Volume.from_name("diffusion-txc", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "numpy")
    .add_local_dir(str(ROOT / "experiments" / "diffusion_txc" / "psc"),
                   "/work/psc")
)

CKPTS = {"bg6_sol": ("bayes_gate/bg6_sol/bayes_gate_s0.pt", 68),
         "bg7_sol": ("bayes_gate/bg7_sol/bayes_gate_s0.pt", 49)}
EPS = (0.05, 0.1, 0.25, 0.5)


@app.function(image=image, gpu="L4", timeout=3600, volumes={"/vol": vol})
def frag_all() -> dict:
    import sys

    import torch

    sys.path.insert(0, "/work/psc")
    from psc_train_sae import BayesGateSAE

    dev = "cuda"
    rms = json.loads(pathlib.Path("/vol/cache/meta.json").read_text())["rms"]
    X = torch.load("/vol/cache/eval_shard.pt", weights_only=True,
                   map_location="cpu")["acts"].float()[:20000]

    def supports(sae, xb, u, k):
        z, g = sae.encode(xb.to(dev), u=u)
        pre_pos = z > 0                      # z = g*shrink*relu(pre)
        hard = (g > 0.5) & pre_pos
        rank = torch.zeros_like(hard)
        idx = z.topk(k, dim=1).indices
        rank.scatter_(1, idx, True)
        return hard.cpu(), rank.cpu()

    def jac(a, b):
        inter = (a & b).sum(1).float()
        union = (a | b).sum(1).float().clamp_min(1)
        return float((inter / union).mean())

    out = {"rms": rms, "n_rows": int(X.shape[0]), "u_convention": "u=eps^2"}
    for name, (rel, k) in CKPTS.items():
        sae = BayesGateSAE(X.shape[1], 16384).to(dev)
        sd = torch.load(f"/vol/{rel}", weights_only=True, map_location=dev)
        sae.load_state_dict(sd, strict=False)
        sae.eval()
        with torch.no_grad():
            hard_c, rank_c = [], []
            for b0 in range(0, X.shape[0], 4096):
                h, r = supports(sae, X[b0:b0 + 4096], 0.0, k)
                hard_c.append(h)
                rank_c.append(r)
            hard_c = torch.cat(hard_c)
            rank_c = torch.cat(rank_c)
            res = {}
            for eps in EPS:
                torch.manual_seed(0)
                Xp = X + eps * rms * torch.randn(X.shape)
                h0, r0, hm = [], [], []
                for b0 in range(0, X.shape[0], 4096):
                    xb = Xp[b0:b0 + 4096]
                    h, r = supports(sae, xb, 0.0, k)
                    h0.append(h)
                    r0.append(r)
                    hm.append(supports(sae, xb, eps * eps, k)[0])
                res[f"eps={eps}"] = {
                    "hard_sigma0": jac(hard_c, torch.cat(h0)),
                    "hard_sigma_matched": jac(hard_c, torch.cat(hm)),
                    "rank_topk": jac(rank_c, torch.cat(r0)),
                }
                print(name, f"eps={eps}", json.dumps(res[f"eps={eps}"]),
                      flush=True)
            out[name] = res
        del sae
        torch.cuda.empty_cache()
        pathlib.Path("/vol/logs_bayes_evals/fragility_sigma_matched.json"
                     ).write_text(json.dumps(out, indent=1))
        vol.commit()
    print("FRAG DONE", flush=True)
    return out


@app.local_entrypoint()
def main():
    print(json.dumps(frag_all.remote(), indent=1))
