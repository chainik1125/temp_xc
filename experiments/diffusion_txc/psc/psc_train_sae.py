"""Self-contained stream-training of TopK SAEs (recon | dsm | dsm_anneal).

Designed for PSC Bridges-2 batch jobs: no activation cache (streams text ->
model forwards -> SAE training in one process), checkpoints + JSONL logs to
--out, optional HuggingFace upload at the end. No repo imports — this file
is the whole program.

    python3 psc_train_sae.py --model google/gemma-2-2b --hook resid12 \
        --arm dsm --seed 0 --steps 24000 --k 40 --dataset pile \
        --out $RUN_DIR --hf-repo dmanningcoe/diffusion-topk-saes \
        --hf-subdir gemma2-2b-l12-100M
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

import numpy as np
import torch
from torch import nn


class TopKSAE(nn.Module):
    """Gao-et-al TopK SAE: pre-bias, TopK+ReLU, AuxK, unit-norm decoder."""

    def __init__(self, d, H, k, k_aux=512, aux_coef=1 / 32, dead_window=800):
        super().__init__()
        self.W_enc = nn.Parameter(torch.randn(d, H) / d**0.5)
        self.b_enc = nn.Parameter(torch.zeros(H))
        self.W_dec = nn.Parameter(self.W_enc.data.T.clone())
        self.b_dec = nn.Parameter(torch.zeros(d))
        self.k, self.k_aux, self.aux_coef = k, k_aux, aux_coef
        self.dead_window = dead_window
        self.register_buffer("steps_since_fire", torch.zeros(H))
        self._norm()

    @torch.no_grad()
    def _norm(self):
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True).clamp_min(1e-8)

    def forward(self, x, target=None, update_fires=False):
        if target is None:
            target = x
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        vals, idx = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre).scatter(-1, idx, torch.relu(vals))
        recon = z @ self.W_dec + self.b_dec
        loss = (recon - target).pow(2).mean()
        if update_fires:
            with torch.no_grad():
                fired = torch.zeros_like(pre, dtype=torch.bool)
                fired.scatter_(-1, idx, torch.relu(vals) > 0)
                self.steps_since_fire += 1
                self.steps_since_fire[fired.any(dim=0)] = 0
        dead = self.steps_since_fire > self.dead_window
        if dead.any() and self.k_aux > 0:
            resid = (target - recon).detach()
            pre_d = pre.masked_fill(~dead.unsqueeze(0), float("-inf"))
            a_vals, a_idx = pre_d.topk(min(self.k_aux, int(dead.sum())), dim=-1)
            z_aux = torch.zeros_like(pre).scatter(-1, a_idx, torch.relu(a_vals))
            aux = (z_aux @ self.W_dec - resid).pow(2).mean()
            if torch.isfinite(aux):
                loss = loss + self.aux_coef * aux
        return loss, recon, z

    @torch.no_grad()
    def dead_fraction(self):
        return float((self.steps_since_fire > self.dead_window).float().mean())


def text_stream(dataset: str, seed: int):
    from datasets import load_dataset

    name, col = {
        "pile": ("monology/pile-uncopyrighted", "text"),
        "fineweb": ("HuggingFaceFW/fineweb", "text"),
    }[dataset]
    ds = load_dataset(name, split="train", streaming=True).shuffle(
        seed=seed, buffer_size=10_000)
    for row in ds:
        yield row[col]


def make_hook(model, hook: str):
    """Returns (capture_fn(input_ids) -> (n_tokens, d) activations).

    Hook names are <kind><layer>, kind in {resid, ln1}. The kind must be
    matched before the digits are read: "ln110" is ln1 of layer 10, so
    scraping every digit in the string would yield layer 110.
    """
    import re

    m = re.fullmatch(r"(resid|ln1)(\d+)", hook)
    if not m:
        raise ValueError(f"unparseable hook {hook!r}; want resid<L> or ln1<L>")
    kind, layer = m.group(1), int(m.group(2))
    n_layers = len(model.model.layers)
    if not 0 <= layer < n_layers:
        raise ValueError(f"hook {hook!r} -> layer {layer}, model has {n_layers}")
    if kind == "resid":
        def cap(ids):
            hs = model(ids, output_hidden_states=True).hidden_states[layer + 1]
            return hs.reshape(-1, hs.shape[-1])
        return cap
    if kind == "ln1":
        store = {}
        blk = model.model.layers[layer].input_layernorm
        blk.register_forward_hook(lambda mod, i, o: store.__setitem__("h", o))
        def cap(ids):
            model(ids)
            h = store["h"]
            return h.reshape(-1, h.shape[-1])
        return cap
    raise ValueError(hook)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--hook", required=True)      # resid12 | ln110 | ln124
    ap.add_argument("--arm", required=True, choices=["recon", "dsm", "dsm_anneal"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--H", type=int, default=16384)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dataset", default="pile", choices=["pile", "fineweb"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--hf-repo", default="")
    ap.add_argument("--hf-subdir", default="")
    ap.add_argument("--fwd-seqs", type=int, default=32)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.arm}_s{args.seed}"
    log_f = open(out / f"{tag}.jsonl", "a")
    dev = "cuda"
    torch.manual_seed(1000 * args.seed + hash(args.arm) % 97)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(dev).eval()
    cap = make_hook(model, args.hook)
    d = model.config.hidden_size
    sae = TopKSAE(d, args.H, args.k).to(dev)
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr)

    texts = text_stream(args.dataset, seed=args.seed)
    buf: list[int] = []

    def next_acts(n_tokens: int) -> torch.Tensor:
        got = []
        n = 0
        with torch.no_grad():
            while n < n_tokens:
                while len(buf) < args.fwd_seqs * args.ctx:
                    buf.extend(tok(next(texts), add_special_tokens=False)["input_ids"])
                ids = torch.tensor(buf[: args.fwd_seqs * args.ctx],
                                   device=dev).view(args.fwd_seqs, args.ctx)
                del buf[: args.fwd_seqs * args.ctx]
                a = cap(ids).float()
                got.append(a)
                n += a.shape[0]
        return torch.cat(got)[:n_tokens]

    X_eval = next_acts(100_000)
    rms = float(X_eval.pow(2).mean().sqrt())
    lo, hi0 = 0.05, 1.0

    def nmse():
        num = den = 0.0
        mu = X_eval.mean(0)
        with torch.no_grad():
            for b0 in range(0, X_eval.shape[0], 8192):
                xb = X_eval[b0:b0 + 8192]
                _, r, _ = sae(xb)
                num += float((xb - r).pow(2).sum())
                den += float((xb - mu).pow(2).sum())
        return num / den

    RESERVOIR = 40 * args.batch
    step = 0
    t0 = time.time()
    while step < args.steps:
        R = next_acts(RESERVOIR)
        perm = torch.randperm(R.shape[0], device=dev)
        for b0 in range(0, R.shape[0] - args.batch, args.batch):
            xb = R[perm[b0:b0 + args.batch]]
            if args.arm == "recon":
                loss, _, _ = sae(xb, update_fires=True)
            else:
                hi = hi0 if args.arm == "dsm" else hi0 - (hi0 - 0.3) * (step / args.steps)
                sig = torch.exp(torch.empty(args.batch, 1, device=dev).uniform_(
                    float(np.log(lo)), float(np.log(hi)))) * rms
                loss, _, _ = sae(xb + sig * torch.randn_like(xb), target=xb,
                                 update_fires=True)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sae._norm()
            step += 1
            if step % 250 == 0:
                row = {"step": step, "tokens": step * args.batch,
                       "train_loss": float(loss), "nmse_clean": nmse(),
                       "dead_frac": sae.dead_fraction(), "rms": rms,
                       "elapsed_s": round(time.time() - t0, 1)}
                log_f.write(json.dumps(row) + "\n")
                log_f.flush()
                print(row, flush=True)
            if step % 5000 == 0 or step >= args.steps:
                torch.save(sae.state_dict(), out / f"{tag}.pt")
            if step >= args.steps:
                break
        del R

    torch.save(sae.state_dict(), out / f"{tag}.pt")
    (out / f"{tag}_final.json").write_text(json.dumps({
        "arm": args.arm, "seed": args.seed, "model": args.model,
        "hook": args.hook, "H": args.H, "k": args.k, "steps": args.steps,
        "tokens": args.steps * args.batch, "rms": rms,
        "final_nmse": nmse(), "final_dead_frac": sae.dead_fraction(),
        "wall_s": round(time.time() - t0, 1)}))
    print("TRAINING DONE", flush=True)

    if args.hf_repo:
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=os.environ.get("HF_WRITE_TOKEN")
                        or os.environ.get("HF_TOKEN"))
            for suffix in (".pt", ".jsonl", "_final.json"):
                f = out / f"{tag}{suffix}"
                api.upload_file(path_or_fileobj=str(f),
                                path_in_repo=f"{args.hf_subdir}/{f.name}",
                                repo_id=args.hf_repo)
            print("HF PUSH DONE", flush=True)
        except Exception as e:                                # noqa: BLE001
            print(f"HF push failed (checkpoints remain in {out}): {e}",
                  flush=True)


if __name__ == "__main__":
    main()
