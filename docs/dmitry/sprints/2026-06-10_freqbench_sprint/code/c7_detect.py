"""c7_detect.py — c7 detection metric for the retrained dictionaries.

Paper protocol (tab:c7-pr-auc): backtracking-sentence detection via sparse
probes. Positives = D+ windows (6-token, offsets [-13,-8] before each
"Wait"/"Hmm" event); negatives = far-from-event windows. Feature vector per
example = max over the window positions of |z| (per paper: max-pool
|activations| over positions). Probe = l1-logistic (C=1), 5-fold GroupKFold
by trace; report PR-AUC and ROC-AUC at top-S features (S in {8, 32}, selected
per fold by Welch t-stat on the train split) and with all features.

Stage 1 (if missing): base-Llama (nous) resid L10 cache over the 300 traces.
Stage 2: encode windows with each trained dictionary; probe.

Usage: python c7_detect.py --hook resid --layer 10
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

D = 4096
WIN_OFF = list(range(-13, -7))     # 6 positions, offsets -13..-8
NEG_BUFFER = 25
NEG_PER_POS = 5


def build_trace_cache(out, layer):
    if os.path.exists(f"{out}/activations.fp16.npy"):
        return
    from transformers import AutoModelForCausalLM
    os.makedirs(out, exist_ok=True)
    traces = [json.loads(l) for l in open("bt_data/traces.jsonl")]
    model = AutoModelForCausalLM.from_pretrained(
        "/workspace/llama31",
        torch_dtype=torch.bfloat16).to("cuda").eval()
    acts, offsets, ids = [], [0], []
    with torch.no_grad():
        for i, tr in enumerate(traces):
            toks = torch.tensor([tr["full_token_ids"]], device="cuda")
            store = {}

            def hook(mod, inp, outp):
                h = outp[0] if isinstance(outp, tuple) else outp
                store["a"] = h.detach()
            h = model.model.layers[layer].register_forward_hook(hook)
            model(toks)
            h.remove()
            a = store["a"][0].to(torch.float16).cpu().numpy()
            acts.append(a)
            offsets.append(offsets[-1] + a.shape[0])
            ids.append(tr["trace_id"])
            if i % 100 == 0:
                print(f"  trace cache {i}/300", flush=True)
    np.save(f"{out}/activations.fp16.npy", np.concatenate(acts))
    np.save(f"{out}/offsets.npy", np.array(offsets, dtype=np.int64))
    json.dump(ids, open(f"{out}/trace_ids.json", "w"))
    del model
    torch.cuda.empty_cache()
    print("trace cache built", flush=True)


def collect_examples():
    """Returns list of (trace_idx, event_anchor_pos, label). Window for an
    example spans positions anchor+WIN_OFF."""
    labs = {j["trace_id"]: j for j in
            (json.loads(l) for l in open("bt_data/labels.jsonl"))}
    ids = json.load(open("trace_cache/trace_ids.json"))
    rng = np.random.default_rng(11)
    ex = []
    for k, tid in enumerate(ids):
        l = labs[tid]
        lo = l["think_lo"]
        ev = np.array(l["event_positions"], dtype=int)
        for e in ev:
            if e + WIN_OFF[0] >= lo:
                ex.append((k, int(e), 1))
        cand = [p for p in range(lo + 13, l["n_tokens"])
                if not len(ev) or np.abs(ev - p).min() > NEG_BUFFER]
        n_neg = min(len(cand), NEG_PER_POS * max(1, int((ev >= 0).sum())))
        if not len(ev):
            n_neg = min(len(cand), 4)
        for i in rng.choice(len(cand), size=n_neg, replace=False):
            ex.append((k, int(cand[i]), 0))
    return ex


class Dict5:
    """Wraps a trained dictionary; encode (B, T, d) windows -> codes."""

    def __init__(self, kind, path, T=5):
        self.kind, self.T = kind, T
        sd = torch.load(path, map_location="cuda")
        if kind == "spectral":
            from fb_core import SpectralTXC
            meta = json.load(open(os.path.join(os.path.dirname(path),
                                               "log.json")))
            self.m = SpectralTXC(D, T, meta["bands"], meta["h_per"],
                                 meta["k_per"]).to("cuda")
            self.m.load_state_dict(sd)
            self.bands = meta["bands"]
            self.h_per = meta["h_per"]
        elif kind == "txc_bare":
            import sys
            sys.path.insert(0, ".")
            from src.architectures.txc_bare_antidead import TXCBareAntidead
            cfgp = os.path.join(os.path.dirname(path), "meta.json")
            cfg = json.load(open(cfgp)) if os.path.exists(cfgp) else {}
            d_sae = cfg.get("d_sae", 32768)
            k = cfg.get("k", cfg.get("k_pos", 20) * T)
            self.m = TXCBareAntidead(D, d_sae, T, k).to("cuda")
            self.m.load_state_dict(sd, strict=False)
        elif kind == "topk_sae":
            import sys
            sys.path.insert(0, ".")
            from src.architectures.topk_sae import TopKSAE
            cfgp = os.path.join(os.path.dirname(path), "meta.json")
            cfg = json.load(open(cfgp)) if os.path.exists(cfgp) else {}
            self.m = TopKSAE(D, cfg.get("d_sae", 32768),
                             cfg.get("k", cfg.get("k_pos", 20))).to("cuda")
            self.m.load_state_dict(sd, strict=False)
        self.m.eval()

    @torch.no_grad()
    def maxpool_codes(self, X):
        """X: (B, T_eval=6, d) -> (B, n_feat): max over per-position |z|.
        Window models slide their T=5 window over the 6 positions (2 offsets)
        and max-pool over both window placements."""
        if self.kind == "topk_sae":
            z = self.m.encode(X.reshape(-1, D)).reshape(X.shape[0],
                                                        X.shape[1], -1)
            return z.abs().amax(1)
        outs = []
        for s in range(X.shape[1] - self.T + 1):
            w = X[:, s:s + self.T, :]
            if self.kind == "spectral":
                z = torch.cat(self.m.encode(w), dim=-1)
            else:
                z = self.m.encode(w)
                if isinstance(z, tuple):
                    z = z[0]
                if z.dim() == 3:
                    z = z.flatten(1)
            outs.append(z.abs())
        return torch.stack(outs, 0).amax(0)


def prauc(y, s):
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / (np.arange(len(y)) + 1)
    rec = tp / y.sum()
    return float(np.trapezoid(prec, rec))


def rocauc(y, s):
    from scipy.stats import rankdata
    r = rankdata(s)
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def probe_cv(F, y, groups, S_values=(8, 32, None)):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    res = {}
    for S in S_values:
        pr, roc = [], []
        for tr_idx, te_idx in GroupKFold(5).split(F, y, groups):
            Ftr, Fte = F[tr_idx], F[te_idx]
            ytr, yte = y[tr_idx], y[te_idx]
            if S is not None:
                mu1 = Ftr[ytr == 1].mean(0)
                mu0 = Ftr[ytr == 0].mean(0)
                s1 = Ftr[ytr == 1].var(0) / max((ytr == 1).sum(), 1)
                s0 = Ftr[ytr == 0].var(0) / max((ytr == 0).sum(), 1)
                t = (mu1 - mu0) / np.sqrt(s1 + s0 + 1e-9)
                feats = np.argsort(-np.abs(t))[:S]
                Ftr, Fte = Ftr[:, feats], Fte[:, feats]
            sc = Ftr.std(0) + 1e-9
            clf = LogisticRegression(penalty="l1", C=1.0, solver="liblinear",
                                     max_iter=2000)
            clf.fit(Ftr / sc, ytr)
            s = clf.decision_function(Fte / sc)
            pr.append(prauc(yte, s))
            roc.append(rocauc(yte, s))
        tag = f"S{S}" if S else "all"
        res[f"prauc_{tag}"] = float(np.mean(pr))
        res[f"rocauc_{tag}"] = float(np.mean(roc))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--hook", default="resid")
    ap.add_argument("--out", default="c7_detect_results.json")
    args = ap.parse_args()
    build_trace_cache("trace_cache", args.layer)
    A = np.load("trace_cache/activations.fp16.npy", mmap_mode="r")
    off = np.load("trace_cache/offsets.npy")
    Ad = torch.tensor(np.asarray(A), device="cuda")
    ex = collect_examples()
    print(f"{len(ex)} examples, {sum(1 for e in ex if e[2])} positives",
          flush=True)
    offs = torch.tensor(WIN_OFF, device="cuda")
    base = torch.tensor([off[k] + p for k, p, _ in ex], device="cuda")
    gather = base[:, None] + offs[None, :]
    X = Ad[gather].float()                      # (n, 6, 4096)
    y = np.array([e[2] for e in ex])
    groups = np.array([e[0] for e in ex])

    root = f"data/llama_3_1_8b/{args.hook}/L{args.layer}"
    arms = {"topk_sae": ("topk_sae", f"{root}/topk_sae/ckpt.pt"),
            "txc_bare": ("txc_bare", f"{root}/txc_bare/ckpt.pt"),
            "spectral_txc": ("spectral", f"{root}/spectral_txc/ckpt.pt")}
    out = {}
    # raw-activation reference: max-pool |x| is meaningless; use mean + edge
    Fraw = X.reshape(X.shape[0], -1).cpu().numpy()
    out["raw_stacked"] = probe_cv(Fraw, y, groups, S_values=(None,))
    for name, (kind, path) in arms.items():
        if not os.path.exists(path):
            print(f"skip {name} (no ckpt)", flush=True)
            continue
        dd = Dict5(kind, path)
        F = []
        for i in range(0, X.shape[0], 512):
            F.append(dd.maxpool_codes(X[i:i + 512]).cpu().numpy())
        F = np.concatenate(F)
        out[name] = probe_cv(F, y, groups)
        # spectral: per-branch detection at S=8
        if kind == "spectral":
            b0 = 0
            for bi, h in enumerate(dd.h_per):
                Fb = F[:, b0:b0 + h]
                out[f"spectral_branch{bi}"] = probe_cv(
                    Fb, y, groups, S_values=(8,))
                b0 += h
        print(name, out[name], flush=True)
        del dd
        torch.cuda.empty_cache()
    json.dump(out, open(args.out, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
