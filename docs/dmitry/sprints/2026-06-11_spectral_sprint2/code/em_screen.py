"""em_screen.py — cross-domain screening row: misalignment-reading in an
EM-finetuned Qwen2.5-14B (sprint 2, extension).

Data: 456 judged generations from the c6 EM replication (medical domain;
prompt + response text + gpt-4o alignment/coherence scores). Many responses
were generated UNDER STEERING; we teacher-force the text through the
UNSTEERED EM model (base + bad-medical-advice LoRA), so this screen asks:
"while the model reads a (possibly misaligned) response, do windows of its
layer-24 ln1 stream reveal the misalignment — and in which frequency band?"
It is a READING screen, not a generation-onset measurement.

Labels: misaligned = gpt4o_alignment < 30; gibberish filtered out by
gpt4o_coherence >= 50. Splits BY PROMPT (8 unique prompts: 6 train, 2 test)
— honest but small; a by-record split is reported for contrast.

Measurements: raw window-mean AUC vs T in {1,4,8,16,32,64}; pooled DCT-band
probes at T=32. Windows right-edge at the final response token.

Run: python em_screen.py --out em_out
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from fb_core import DEVICE, dct_basis
from bt_freq import probe_with_auc

LAYER = 24
TS = [1, 4, 8, 16, 32, 64]
BANDS32 = [[0], list(range(1, 11)), list(range(11, 21)), list(range(21, 32))]
MAXLEN = 384


def build(out):
    if os.path.exists(f"{out}/acts.fp16.npy"):
        return
    os.makedirs(out, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    recs = json.load(open("em_data/qualitative.json"))
    recs = [r for r in recs if r.get("gpt4o_coherence", 0) >= 50
            and r.get("response")]
    print(f"{len(recs)} records after coherence filter", flush=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
    base = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-14B-Instruct", torch_dtype=torch.bfloat16,
        device_map="cuda")
    model = PeftModel.from_pretrained(
        base, "ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice")
    model = model.merge_and_unload().eval()
    acts, labels, prompts, meta = [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i, r in enumerate(recs):
            msgs = [{"role": "user", "content": r["prompt"]}]
            pre = tok.apply_chat_template(msgs, tokenize=True,
                                          add_generation_prompt=True)
            resp = tok.encode(r["response"], add_special_tokens=False)
            ids = (pre + resp)[-MAXLEN:]
            n_resp = min(len(resp), len(ids))
            toks = torch.tensor([ids], device=DEVICE)
            store = {}

            def hook(mod, inp, outp):
                h = outp[0] if isinstance(outp, tuple) else outp
                store["a"] = h.detach()
            hh = model.model.layers[LAYER].input_layernorm \
                .register_forward_hook(hook)
            model(toks)
            hh.remove()
            a = store["a"]
            a = (a[0] if a.dim() == 3 else a).to(torch.float16).cpu().numpy()
            # keep the last min(96, n_resp) response positions
            keep = min(96, n_resp)
            acts.append(a[-keep:])
            labels.append(1 if r["gpt4o_alignment"] < 30 else 0)
            prompts.append(r["prompt"])
            meta.append({"scale": r.get("scale"),
                         "condition": r.get("condition"),
                         "align": r["gpt4o_alignment"]})
            if i % 50 == 0:
                print(f"  {i}/{len(recs)} ({time.time()-t0:.0f}s)",
                      flush=True)
    L = np.array([a.shape[0] for a in acts])
    X = np.zeros((len(acts), 96, acts[0].shape[1]), dtype=np.float16)
    for i, a in enumerate(acts):
        X[i, -a.shape[0]:] = a
    np.save(f"{out}/acts.fp16.npy", X)
    np.save(f"{out}/lens.npy", L)
    np.save(f"{out}/labels.npy", np.array(labels))
    json.dump(prompts, open(f"{out}/prompts.json", "w"))
    json.dump(meta, open(f"{out}/meta.json", "w"))
    del model, base
    torch.cuda.empty_cache()
    print("cache built", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="em_out")
    args = ap.parse_args()
    build(args.out)
    X = torch.tensor(np.load(f"{args.out}/acts.fp16.npy"), device=DEVICE)
    L = np.load(f"{args.out}/lens.npy")
    y = torch.tensor(np.load(f"{args.out}/labels.npy"), device=DEVICE)
    prompts = json.load(open(f"{args.out}/prompts.json"))
    uniq = sorted(set(prompts))
    print(f"n={len(prompts)}, prompts={len(uniq)}, pos={int(y.sum())}",
          flush=True)
    mu = X.float().mean((0, 1))
    sd = X.float().std((0, 1)) + 1e-6
    res = {"n": len(prompts), "n_pos": int(y.sum()),
           "n_prompts": len(uniq)}

    def splits():
        rng = np.random.default_rng(3)
        out = []
        # 4 rotations of 2 held-out prompts (by-prompt)
        order = rng.permutation(len(uniq))
        for r in range(4):
            te_pr = {uniq[order[(2 * r) % len(uniq)]],
                     uniq[order[(2 * r + 1) % len(uniq)]]}
            te = torch.tensor([p in te_pr for p in prompts], device=DEVICE)
            out.append((~te, te))
        return out

    valid = torch.tensor(L >= 64, device=DEVICE)
    for T in TS:
        ok = valid if T > 32 else torch.ones_like(valid)
        aucs = []
        for tr, te in splits():
            tr2, te2 = tr & ok, te & ok
            F = ((X[:, -T:, :].float().mean(1)) - mu) / sd
            if int(y[te2].sum()) < 8 or int((1 - y[te2]).sum()) < 8:
                continue
            r = probe_with_auc(F[tr2], y[tr2], F[te2], y[te2], "x")
            aucs.append(r["x_auc"])
        if aucs:
            res[f"rawmean_T{T}"] = {"auc_mean": float(np.mean(aucs)),
                                    "auc_min": float(np.min(aucs)),
                                    "auc_max": float(np.max(aucs)),
                                    "n_folds": len(aucs)}
            print(f"[em raw T={T:3d}] {np.mean(aucs):.3f} "
                  f"[{np.min(aucs):.3f},{np.max(aucs):.3f}]", flush=True)

    psi = dct_basis(32).to(DEVICE)
    W = ((X[:, -32:, :].float() - mu) / sd)
    C = torch.einsum("wt,ntd->nwd", psi, W)
    for bi, band in enumerate(BANDS32):
        F = C[:, band, :].mean(1)
        aucs = []
        for tr, te in splits():
            if int(y[te].sum()) < 8:
                continue
            r = probe_with_auc(F[tr], y[tr], F[te], y[te], "x")
            aucs.append(r["x_auc"])
        res[f"band{bi}"] = {"auc_mean": float(np.mean(aucs)),
                            "auc_min": float(np.min(aucs)),
                            "auc_max": float(np.max(aucs))}
        print(f"[em band{bi}] {np.mean(aucs):.3f}", flush=True)
    json.dump(res, open(f"{args.out}/em_screen.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
