"""Train one arm (recon | dsm) x seed from the shared cache, with in-train evals.

Logs JSONL rows {step, tokens, flops, train_loss, nmse_clean, dead_frac,
[dce, loss_recovered]} to <out>/logs/{arm}_s{seed}.jsonl; saves final weights
and an autointerp dump of top-activating contexts.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import torch

from experiments.diffusion_txc.topk_vs_topkdiff.sae import TopKSAE, flops_per_token


def _load_shards(cache: pathlib.Path, n_shards: int):
    for i in range(n_shards):
        yield torch.load(cache / f"shard_{i:03d}.pt", weights_only=True)


def _patch_ce(model, input_ids, layer, replace_fn):
    """CE over eval sequences with layer-`layer` block output hidden state
    replaced by replace_fn(hidden)."""
    handle = None
    if replace_fn is not None:
        block = model.model.layers[layer]

        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h2 = replace_fn(h)
            return (h2, *out[1:]) if isinstance(out, tuple) else h2

        handle = block.register_forward_hook(hook)
    try:
        ces = []
        with torch.no_grad():
            for b0 in range(0, input_ids.shape[0], 8):     # chunk: vocab-sized
                ids = input_ids[b0 : b0 + 8]               # logits are huge
                logits = model(ids).logits[:, :-1]
                ces.append(float(torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(),
                    ids[:, 1:].reshape(-1),
                )))
                del logits
        return float(np.mean(ces))
    finally:
        if handle is not None:
            handle.remove()


def run(device: str = "cuda", vol: str = "/vol", arm: str = "recon", seed: int = 0,
        H: int = 16384, k: int = 40, lr: float = 3e-4, batch: int = 4096,
        steps: int = 6000, eval_every: int = 250, dce_every: int = 500,
        sigma_range: tuple[float, float] = (0.05, 1.0),
        model_name: str = "google/gemma-2-2b") -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = torch.device(device)
    cache = pathlib.Path(vol) / "cache"
    meta = json.loads((cache / "meta.json").read_text())
    d, rms, layer = meta["d"], meta["rms"], meta["layer"]
    mean = torch.tensor(meta["mean"], dtype=torch.float32, device=dev)

    ev = torch.load(cache / "eval_shard.pt", weights_only=True)
    X_eval = ev["acts"][:200_000].to(dev, torch.float32)
    eval_seqs = torch.load(cache / "eval_seqs.pt", weights_only=True)["input_ids"].to(dev)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(dev).eval()
    tok = AutoTokenizer.from_pretrained(model_name)

    torch.manual_seed(1000 * seed + (7 if arm == "dsm" else 0) + 1)
    sae = TopKSAE(d, H, k).to(dev)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    lo, hi = sigma_range

    out = pathlib.Path(vol)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    log_path = out / "logs" / f"{arm}_s{seed}.jsonl"
    log_f = open(log_path, "a")

    ce_clean = _patch_ce(model, eval_seqs, layer, None)
    ce_abl = _patch_ce(model, eval_seqs, layer,
                       lambda h: mean.to(h.dtype).expand_as(h))

    def nmse(X):
        num = den = 0.0
        mu = X.mean(0)
        with torch.no_grad():
            for b0 in range(0, X.shape[0], 8192):
                xb = X[b0 : b0 + 8192]
                _, recon, _ = sae(xb)
                num += float((xb - recon).pow(2).sum())
                den += float((xb - mu).pow(2).sum())
        return num / den

    def dce():
        def repl(h):
            with torch.no_grad():
                _, r, _ = sae(h.float().reshape(-1, d))
            return r.reshape(h.shape).to(h.dtype)

        ce_p = _patch_ce(model, eval_seqs, layer, repl)
        lr_frac = (ce_abl - ce_p) / max(ce_abl - ce_clean, 1e-6)
        return ce_p - ce_clean, lr_frac

    # training stream: reshuffled epochs over shards
    rng = np.random.default_rng(seed)
    fpt = flops_per_token(d, H)
    step = tokens = 0
    while step < steps:
        order = rng.permutation(meta["n_shards"])
        for si in order:
            sh = torch.load(cache / f"shard_{si:03d}.pt", weights_only=True)
            A = sh["acts"]                                  # bf16, stays on CPU
            perm = torch.randperm(A.shape[0])
            for b0 in range(0, A.shape[0] - batch, batch):
                xb = A[perm[b0 : b0 + batch]].to(dev, torch.float32)
                if arm == "dsm":
                    sig = torch.exp(torch.empty(batch, 1, device=dev).uniform_(
                        float(np.log(lo)), float(np.log(hi)))) * rms
                    loss, _, _ = sae(xb + sig * torch.randn_like(xb),
                                     target=xb, update_fires=True)
                else:
                    loss, _, _ = sae(xb, update_fires=True)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sae._normalize_decoder()
                step += 1
                tokens += batch
                if step % eval_every == 0:
                    row = {"step": step, "tokens": tokens, "flops": tokens * fpt,
                           "train_loss": float(loss), "nmse_clean": nmse(X_eval),
                           "dead_frac": sae.dead_fraction()}
                    if step % dce_every == 0:
                        row["dce"], row["loss_recovered"] = dce()
                    log_f.write(json.dumps(row) + "\n")
                    log_f.flush()
                    print(row, flush=True)
                if step >= steps:
                    break
            del A
            if step >= steps:
                break

    torch.save(sae.state_dict(), out / "logs" / f"{arm}_s{seed}_final.pt")

    # autointerp dump: top-12 contexts for 128 random alive latents
    with torch.no_grad():
        Z = []
        for b0 in range(0, min(X_eval.shape[0], 100_000), 8192):
            Z.append(sae.encode(X_eval[b0 : b0 + 8192]).to("cpu", torch.bfloat16))
        Z = torch.cat(Z).float()                             # (N, H)
    alive = (Z > 0).any(dim=0).nonzero().squeeze(1)
    pick = alive[torch.randperm(alive.numel())[:128]]
    ev_tok = ev["tokens"][: Z.shape[0]]
    ctx_len = meta["ctx"]
    dump = {}
    for latent in pick.tolist():
        top = Z[:, latent].topk(12).indices
        exs = []
        for pos in top.tolist():
            s = (pos // ctx_len) * ctx_len
            text = tok.decode(ev_tok[s : s + ctx_len].tolist())
            exs.append({"act": float(Z[pos, latent]),
                        "tok": tok.decode([int(ev_tok[pos])]), "ctx": text})
        dump[str(latent)] = exs
    (out / "logs" / f"{arm}_s{seed}_autointerp.json").write_text(json.dumps(dump))

    final = {"experiment": "topk_vs_topkdiff", "arm": arm, "seed": seed,
             "config": {"H": H, "k": k, "lr": lr, "batch": batch, "steps": steps,
                        "sigma_range": list(sigma_range)},
             "ce_clean": ce_clean, "ce_ablate": ce_abl,
             "final_nmse": nmse(X_eval), "final_dead_frac": sae.dead_fraction()}
    final["final_dce"], final["final_loss_recovered"] = dce()
    return final
