"""Standalone (no-Modal) runner for the v7 / pp / slab comparison on the ward checkpoints.

Same experiment as `protocol_compare_modal.py`; that file documents the claim under test, why
the clamp is held fixed across protocols, and the known weakness of the keyword readout. This
one exists because Modal is not always available and an ssh pod is.

    export TMPDIR=/workspace/tmp && mkdir -p $TMPDIR
    python protocol_compare_run.py --train-key d7b2e24253055f8e --n-prompts 40 \
        --eval-dir ./_c7_eval --out /workspace/protocol_compare.json
"""
import argparse
import json
import pathlib
import re

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_MODELS_REPO = "han1823123123/temp-bench-models"
SUBJECT = "NousResearch/Meta-Llama-3.1-8B"
KEYWORDS = ("wait", "hmm", "actually", "no,", "hold on", "but wait", "let me reconsider",
            "on second thought", "that's wrong", "i made a mistake")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-key", default="d7b2e24253055f8e")
    ap.add_argument("--n-prompts", type=int, default=40)
    ap.add_argument("--n-sel", type=int, default=60)
    ap.add_argument("--qs", default="0,1,2,4,8")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--eval-dir", default="./_c7_eval")
    ap.add_argument("--out", default="protocol_compare.json")
    a = ap.parse_args()
    qs = [float(x) for x in a.qs.split(",")]
    dev = "cuda"

    cfg = json.load(open(hf_hub_download(HF_MODELS_REPO, f"{a.train_key}/config.json")))
    sd = load_file(hf_hub_download(HF_MODELS_REPO, f"{a.train_key}/model.safetensors"))
    W_dec = sd["W_dec"].float().to(dev)
    W_enc = sd["W_enc"].float().to(dev)
    b_enc = sd["b_enc"].float().to(dev)
    d_sae, T, d_in = W_dec.shape
    k_win = int(cfg.get("training_cfg", {}).get("k_win") or 0) or 100
    print(f"[ckpt] {a.train_key} arch={cfg.get('arch')} d_sae={d_sae} T={T} d_in={d_in}",
          flush=True)

    def encode(win):
        pre = torch.einsum("btd,tds->bs", win, W_enc) + b_enc
        vals, idx = pre.topk(min(k_win, pre.shape[-1]), dim=-1)
        return torch.zeros_like(pre).scatter_(-1, idx, torch.relu(vals))

    tok = AutoTokenizer.from_pretrained(SUBJECT)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        SUBJECT, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    layer = int(cfg.get("datasource", "").split("_l")[-1].split("_")[0] or 10)
    site = model.model.layers[layer]
    print(f"[subject] {SUBJECT} layer={layer}", flush=True)

    ed = pathlib.Path(a.eval_dir)
    labels = json.load(open(ed / "sentence_labels.json"))
    prompts = json.load(open(ed / "prompts.json"))

    # ---- select the backtracking latent on labelled traces ----
    acts, pos_z, neg_z = [], [], []

    def grab(m, i, o):
        acts.append((o[0] if isinstance(o, tuple) else o).detach())

    h = site.register_forward_hook(grab)
    for rec in labels[:a.n_sel]:
        sents = rec["sentences"]
        if not sents:
            continue
        text, spans, cur = "", [], 0
        for s in sents:
            text += s["sentence"] + " "
            n = len(s["sentence"]) + 1
            spans.append((cur, cur + n, bool(s["is_backtracking"])))
            cur += n
        enc = tok(text, return_tensors="pt", truncation=True, max_length=1024,
                  return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")[0].tolist()
        acts.clear()
        with torch.no_grad():
            model(**{k: v.to(dev) for k, v in enc.items()})
        H = acts[0][0].float()
        S = H.shape[0]
        if S < T:
            continue
        is_bt = np.zeros(S, dtype=bool)
        for i, (s0, s1) in enumerate(offs):
            for (cs, ce, f) in spans:
                if f and s0 < ce and s1 > cs:
                    is_bt[i] = True
                    break
        nb = S // T
        with torch.no_grad():
            z = encode(H[: nb * T].reshape(nb, T, d_in))
        lab = is_bt[: nb * T].reshape(nb, T).any(axis=1)
        for w in range(nb):
            (pos_z if lab[w] else neg_z).append(z[w])
    h.remove()
    if not pos_z or not neg_z:
        raise RuntimeError("no labelled windows on one side; raise --n-sel")
    P, N = torch.stack(pos_z).cpu().numpy(), torch.stack(neg_z).cpu().numpy()
    print(f"[select] {len(P)} backtracking windows / {len(N)} not", flush=True)

    allz = np.concatenate([P, N], 0)
    y = np.concatenate([np.ones(len(P)), np.zeros(len(N))])
    order = allz.argsort(0)
    ranks = np.empty_like(order, dtype=np.float64)
    np.put_along_axis(ranks, order, np.arange(1, len(allz) + 1)[:, None], axis=0)
    n1, n0 = y.sum(), len(y) - y.sum()
    auc = ((ranks * y[:, None]).sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    j = int(np.abs(auc - 0.5).argmax())
    sign = 1.0 if auc[j] > 0.5 else -1.0
    p99 = float(np.percentile(allz[:, j], 99)) or 1.0
    slab = W_dec[j]
    Pn = slab / slab.norm()
    ac_share = float((Pn - Pn.mean(0)).norm() ** 2 / Pn.norm() ** 2)
    print(f"[select] latent {j} AUC {auc[j]:.3f} sign {sign:+.0f} p99 {p99:.3f} "
          f"AC-share {ac_share:.3f}", flush=True)

    state = {"q": 0.0, "proto": None, "norms": [], "cs": []}

    def hook(mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        if state["proto"] is None or state["q"] == 0.0:
            return None
        B, S, _ = hs.shape
        if S < T:
            return None
        hf = hs.float()
        s_abs = sign * state["q"] * p99
        delta = torch.zeros_like(hf)
        if state["proto"] in ("v7", "slab"):
            starts = [b * T for b in range(S // T)] + ([S - T] if S % T else [])
            for st in starts:
                with torch.no_grad():
                    c = (s_abs - encode(hf[:, st:st + T, :])[:, j]).view(B, 1, 1)
                state["cs"].append(float(c.mean()))
                d_pos = c * Pn.unsqueeze(0)
                delta[:, st:st + T, :] = (d_pos.mean(1, keepdim=True).expand(-1, T, -1)
                                          if state["proto"] == "v7" else d_pos)
        else:
            K = S - T + 1
            accum = torch.zeros_like(hf)
            cnt = torch.zeros(B, S, 1, device=hf.device)
            for w in range(K):
                with torch.no_grad():
                    c = (s_abs - encode(hf[:, w:w + T, :])[:, j]).view(B, 1, 1)
                state["cs"].append(float(c.mean()))
                accum[:, w:w + T, :] += c * Pn.unsqueeze(0)
                cnt[:, w:w + T, :] += 1.0
            delta = accum / cnt.clamp(min=1.0)
        state["norms"].append(float(delta.norm(dim=(1, 2)).mean()))
        hs2 = (hf + delta).to(hs.dtype)
        return (hs2,) + out[1:] if isinstance(out, tuple) else hs2

    kw_re = re.compile("|".join(re.escape(k) for k in KEYWORDS), re.I)
    handle = site.register_forward_hook(hook)
    rows, sel = [], prompts[:a.n_prompts]
    for proto in ("v7", "pp", "slab"):
        for q in qs:
            state.update(proto=proto, q=float(q), norms=[], cs=[])
            kws, first = [], None
            for rec in sel:
                enc = tok(rec["prompt"], return_tensors="pt", truncation=True,
                          max_length=512).to(dev)
                with torch.no_grad():
                    o = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False,
                                       pad_token_id=tok.pad_token_id)
                g = tok.decode(o[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
                kws.append(len(kw_re.findall(g)) / max(len(g.split()), 1))
                first = first or g[:400]
            cs = np.array(state["cs"]) if state["cs"] else np.array([0.0])
            rows.append({"protocol": proto, "q": float(q),
                         "keyword_rate_mean": float(np.mean(kws)),
                         "keyword_rate_sem": float(np.std(kws, ddof=1) / max(len(kws) ** .5, 1)),
                         "injected_norm_mean": float(np.mean(state["norms"] or [0.0])),
                         "c_rel_var": float(cs.var() / max(cs.mean() ** 2, 1e-12)),
                         "sample": first})
            print(f"[{proto:>4} q={q:<4}] kw {rows[-1]['keyword_rate_mean']:.4f} "
                  f"|d|_F {rows[-1]['injected_norm_mean']:.2f} "
                  f"var(c)/mean(c)^2 {rows[-1]['c_rel_var']:.4f}", flush=True)
    handle.remove()

    out = {"train_key": a.train_key, "arch": cfg.get("arch"),
           "datasource": cfg.get("datasource"), "subject": SUBJECT, "layer": layer,
           "T": T, "d_sae": d_sae, "latent": j, "latent_auc": float(auc[j]), "sign": sign,
           "p99": p99, "ac_share": ac_share, "n_prompts": len(sel), "rows": rows}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=2))
    print("[saved]", a.out, flush=True)


if __name__ == "__main__":
    main()
