"""bt_freq.py — frequency-decomposed crosscoder on the backtracking task.

Subject: DeepSeek-R1-Distill-Llama-8B, layer-10 residual stream (the Ward
et al. 2025 / case-study setting). Reuses the case study's 300 reasoning
traces and keyword labels (D+ = tokens [-13,-8] before each "wait"/"hmm"
backtracking event, inside <think>).

Stages (idempotent, each skipped if outputs exist):
  1. cache  : forward all traces, save L10 resid fp16 cache.
  2. dicts  : train token_sae / txc / multiband on right-edge T=16 windows
              of standardized activations (train traces only).
  3. probes : balanced binary D+ vs far-from-event negatives, split BY TRACE.
              Codes: right-edge token codes, window codes, per-branch codes,
              plus raw baselines. Reports acc + AUC.
  4. spectra: per-feature activation power spectra (Welch, seg=64, hop=32)
              over think regions, for TXC/multiband/token-SAE atoms and for
              Llama-Scope L10R-32x features (feat 71839 + top-50 + random).

Inputs expected in ./bt_data/: traces.jsonl, labels.jsonl, top_features.json
Run: python bt_freq.py --out bt_out
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from fb_core import DEVICE, build_model, fit_probe, train_model

D_MODEL = 4096
LAYER = 10
T = 16                      # window length (right-edge convention)
H = 4096
K_WIN = 256                 # 16 active atoms per token on average
NEG_BUFFER = 25             # negatives at distance > this from any event
NEG_PER_POS = 5
SEG, HOP = 64, 32           # Welch segment/hop for spectra


# ---------- stage 1: activation cache ----------

def build_cache(out):
    if os.path.exists(f"{out}/activations.fp16.npy"):
        return
    from transformers import AutoModelForCausalLM
    traces = [json.loads(l) for l in open("bt_data/traces.jsonl")]
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        torch_dtype=torch.bfloat16).to(DEVICE).eval()
    acts, offsets, ids = [], [0], []
    t0 = time.time()
    with torch.no_grad():
        for i, tr in enumerate(traces):
            toks = torch.tensor([tr["full_token_ids"]], device=DEVICE)
            store = {}

            def hook(mod, inp, outp):
                h = outp[0] if isinstance(outp, tuple) else outp
                store["a"] = h.detach()
            h = model.model.layers[LAYER].register_forward_hook(hook)
            model(toks)
            h.remove()
            a = store["a"][0].to(torch.float16).cpu().numpy()
            acts.append(a)
            offsets.append(offsets[-1] + a.shape[0])
            ids.append(tr["trace_id"])
            if i % 50 == 0:
                print(f"  cache {i}/300 ({time.time()-t0:.0f}s)", flush=True)
    np.save(f"{out}/activations.fp16.npy", np.concatenate(acts))
    np.save(f"{out}/offsets.npy", np.array(offsets, dtype=np.int64))
    json.dump(ids, open(f"{out}/trace_ids.json", "w"))
    del model
    torch.cuda.empty_cache()
    print(f"cache built ({time.time()-t0:.0f}s)", flush=True)


# ---------- shared data plumbing ----------

class BT:
    def __init__(self, out):
        self.A = np.load(f"{out}/activations.fp16.npy", mmap_mode="r")
        self.off = np.load(f"{out}/offsets.npy")
        self.ids = json.load(open(f"{out}/trace_ids.json"))
        labs = {j["trace_id"]: j for j in
                (json.loads(l) for l in open("bt_data/labels.jsonl"))}
        self.labs = [labs[i] for i in self.ids]
        # standardization stats from think-region tokens of ALL traces
        idx = np.concatenate([
            np.arange(self.off[k] + l["think_lo"], self.off[k] + l["n_tokens"])
            for k, l in enumerate(self.labs)])
        sample = self.A[np.sort(np.random.default_rng(0).choice(
            idx, size=min(40000, len(idx)), replace=False))].astype(np.float32)
        self.mu = torch.tensor(sample.mean(0), device=DEVICE)
        self.sd = torch.tensor(sample.std(0) + 1e-6, device=DEVICE)
        # by-trace split
        rng = np.random.default_rng(7)
        perm = rng.permutation(len(self.ids))
        n_te = len(self.ids) // 5
        self.test_traces = set(perm[:n_te].tolist())
        # device copy of full cache (~1.4 GB fp16)
        self.Ad = torch.tensor(np.asarray(self.A), device=DEVICE)

    def windows(self, pairs):
        """pairs: list of (trace_idx, pos). Right-edge T-windows,
        standardized. Requires pos - T + 1 >= 0 (caller filters)."""
        base = torch.tensor([self.off[k] + p for k, p in pairs],
                            device=DEVICE)
        gather = base[:, None] - torch.arange(T - 1, -1, -1,
                                              device=DEVICE)[None, :]
        X = self.Ad[gather].float()
        return (X - self.mu) / self.sd

    def think_pairs(self, train):
        """all usable (trace, pos) with full in-trace window, by split."""
        out = []
        for k, l in enumerate(self.labs):
            if (k in self.test_traces) == train:
                continue
            lo = max(l["think_lo"], T - 1)
            out.extend((k, p) for p in range(lo, l["n_tokens"]))
        return out

    def probe_pairs(self, train):
        """(pairs, labels): D+ vs distance->NEG_BUFFER negatives."""
        pos, neg = [], []
        rng = np.random.default_rng(13 + train)
        for k, l in enumerate(self.labs):
            if (k in self.test_traces) == train:
                continue
            lo = max(l["think_lo"], T - 1)
            ev = np.array(l["event_positions"], dtype=int)
            pos.extend((k, p) for p in l["d_plus_positions"] if p >= lo)
            cand = [p for p in range(lo, l["n_tokens"])
                    if not len(ev) or np.abs(ev - p).min() > NEG_BUFFER]
            take = rng.choice(len(cand), size=min(len(cand), 60),
                              replace=False) if cand else []
            neg.extend((k, cand[i]) for i in take)
        import random as _r
        _r.Random(17).shuffle(neg)
        neg = neg[:NEG_PER_POS * max(1, len(pos))]
        pairs = pos + neg
        y = torch.tensor([1] * len(pos) + [0] * len(neg), device=DEVICE)
        return pairs, y


def auc(scores, y):
    s, yy = scores.detach().cpu().numpy(), y.detach().cpu().numpy()
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = yy.sum(), (1 - yy).sum()
    return float((ranks[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def probe_with_auc(Ftr, ytr, Fte, yte, tag):
    r = fit_probe(Ftr, ytr, Fte, yte, 2)
    # refit score for AUC: linear probe logit difference on test
    # (fit_probe doesn't return the model; quick re-fit here)
    from fb_core import _standardize
    import torch.nn as nn
    import torch.nn.functional as F
    torch.manual_seed(0)
    ftr, fte = _standardize(Ftr.float(), Fte.float())
    probe = nn.Linear(ftr.shape[1], 2).to(DEVICE)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-4)
    w = torch.tensor([1.0, (ytr == 0).sum() / (ytr == 1).sum()],
                     device=DEVICE)
    for _ in range(300):
        loss = F.cross_entropy(probe(ftr), ytr, weight=w)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        logit = probe(fte)[:, 1] - probe(fte)[:, 0]
        pred = (logit > 0).long()
        bal = 0.5 * ((pred[yte == 1] == 1).float().mean()
                     + (pred[yte == 0] == 0).float().mean())
    return {f"{tag}_acc": r["acc_test"], f"{tag}_auc": auc(logit, yte),
            f"{tag}_balacc": bal.item(),
            f"{tag}_n": [int((yte == 1).sum()), int((yte == 0).sum())]}


# ---------- spectra ----------

def welch_spectra(series_fn, n_feat, bt, max_traces=120, device_batch=2048):
    """series_fn(k) -> (L_k, n_feat) activation time series for trace k
    (think region). Returns mean power spectrum (n_feat, SEG//2+1)."""
    win = torch.hann_window(SEG, device=DEVICE)
    acc = torch.zeros(n_feat, SEG // 2 + 1, device=DEVICE)
    nseg = 0
    for k in range(min(len(bt.labs), max_traces)):
        z = series_fn(k)
        if z is None or z.shape[0] < SEG:
            continue
        for s in range(0, z.shape[0] - SEG + 1, HOP):
            seg = z[s:s + SEG]                      # (SEG, F)
            seg = (seg - seg.mean(0)) * win[:, None]
            P = torch.fft.rfft(seg, dim=0).abs() ** 2  # (SEG/2+1, F)
            acc += P.T
            nseg += 1
    return (acc / max(nseg, 1)).cpu().numpy(), nseg


def spec_summary(P):
    """P: (F, nf). Returns spectral centroid + frac power below 1/T."""
    freqs = np.fft.rfftfreq(SEG)                    # cycles/token
    tot = P.sum(1) + 1e-12
    centroid = (P * freqs[None, :]).sum(1) / tot
    lowfrac = P[:, freqs < 1.0 / T].sum(1) / tot
    return centroid, lowfrac


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bt_out")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    build_cache(args.out)
    bt = BT(args.out)
    print(f"traces={len(bt.ids)} test_traces={len(bt.test_traces)}",
          flush=True)

    tr_pairs, ytr = bt.probe_pairs(train=True)
    te_pairs, yte = bt.probe_pairs(train=False)
    Xtr, Xte = bt.windows(tr_pairs), bt.windows(te_pairs)
    print(f"probe sets: train {len(tr_pairs)} (pos {int((ytr==1).sum())}), "
          f"test {len(te_pairs)} (pos {int((yte==1).sum())})", flush=True)

    # raw baselines
    bpath = os.path.join(args.out, "baselines.json")
    if not os.path.exists(bpath):
        res = {}
        res.update(probe_with_auc(Xtr[:, -1, :], ytr, Xte[:, -1, :], yte,
                                  "raw_edge"))
        res.update(probe_with_auc(Xtr.mean(1), ytr, Xte.mean(1), yte,
                                  "raw_mean"))
        res.update(probe_with_auc(Xtr.flatten(1), ytr, Xte.flatten(1), yte,
                                  "raw_stacked"))
        json.dump(res, open(bpath, "w"), indent=1)
        print("[baselines]", {k: round(v, 3) for k, v in res.items()
                              if k.endswith("auc")}, flush=True)

    train_pool = bt.think_pairs(train=True)
    print(f"dict-train pool: {len(train_pool)} windows", flush=True)

    class Shim:
        def sample(self, n):
            idx = np.random.default_rng().choice(len(train_pool), size=n)
            return bt.windows([train_pool[i] for i in idx]), None, None

    for arch in ["token_sae", "txc", "multiband", "conv"]:
        for seed in args.seeds:
            tag = f"{arch}_s{seed}"
            path = os.path.join(args.out, tag + ".json")
            if os.path.exists(path):
                continue
            t0 = time.time()
            torch.manual_seed(100 + seed)
            model = build_model(arch, D_MODEL, T, H, K_WIN).to(DEVICE)
            trn = train_model(model, Shim(), steps=args.steps, batch=64,
                              lr=3e-4, pregen=8192)
            res = {"arch": arch, "seed": seed, "H": H, "k_win": K_WIN,
                   "fvu": trn.fvu, "l0": trn.l0}
            with torch.no_grad():
                zs = model.window_codes(bt.windows(
                    [train_pool[i] for i in
                     np.random.default_rng(1).choice(len(train_pool), 2048)]))
                fired = (zs > 0).any(0).float()
                res["dead_frac"] = float(1 - fired.mean().item())
            with torch.no_grad():
                def codes(X):
                    return torch.cat([model.window_codes(X[i:i + 512])
                                      for i in range(0, X.shape[0], 512)])
                Ftr, Fte = codes(Xtr), codes(Xte)
            res.update(probe_with_auc(Ftr, ytr, Fte, yte, "code"))
            if arch == "token_sae":
                with torch.no_grad():
                    etr = model.encode(Xtr[:, -1, :])
                    ete = model.encode(Xte[:, -1, :])
                res.update(probe_with_auc(etr, ytr, ete, yte, "code_edge"))
            if arch == "multiband":
                with torch.no_grad():
                    btr = [torch.cat([model.branch_codes(Xtr[i:i + 512])[b]
                                      for i in range(0, Xtr.shape[0], 512)])
                           for b in range(4)]
                    bte = [torch.cat([model.branch_codes(Xte[i:i + 512])[b]
                                      for i in range(0, Xte.shape[0], 512)])
                           for b in range(4)]
                for b in range(4):
                    res.update(probe_with_auc(btr[b], ytr, bte[b], yte,
                                              f"branch{b}"))
            torch.save(model.state_dict(),
                       os.path.join(args.out, f"model_{tag}.pt"))
            json.dump(res, open(path, "w"), indent=1)
            print(f"[{tag}] fvu={trn.fvu:.3f} "
                  f"auc={res['code_auc']:.3f} ({time.time()-t0:.0f}s)",
                  flush=True)

            # spectra for this model's atoms (seed 0 only)
            if seed == 0:
                def series(k):
                    l = bt.labs[k]
                    lo = max(l["think_lo"], T - 1)
                    if l["n_tokens"] - lo < SEG:
                        return None
                    pairs = [(k, p) for p in range(lo, l["n_tokens"])]
                    X = bt.windows(pairs)
                    with torch.no_grad():
                        if arch == "token_sae":
                            return model.encode(X[:, -1, :])
                        if arch == "conv":
                            return torch.cat(
                                [model.encode(X[i:i + 1024])[:, :, -1]
                                 for i in range(0, X.shape[0], 1024)])
                        return torch.cat(
                            [model.window_codes(X[i:i + 1024])
                             for i in range(0, X.shape[0], 1024)])
                P, nseg = welch_spectra(series, H, bt)
                cen, low = spec_summary(P)
                np.savez(os.path.join(args.out, f"spectra_{arch}.npz"),
                         P=P.astype(np.float32), centroid=cen, lowfrac=low,
                         nseg=nseg)
                print(f"[{tag}] spectra over {nseg} segments", flush=True)

    # Llama-Scope 32x spectra + probe on its codes
    ls_path = os.path.join(args.out, "llamascope.json")
    if not os.path.exists(ls_path):
        from sae_lens import SAE
        out = SAE.from_pretrained(release="llama_scope_lxr_32x",
                                  sae_id="l10r_32x", device=str(DEVICE))
        sae = (out[0] if isinstance(out, tuple) else out).eval()
        top = json.load(open("bt_data/top_features.json"))
        top_ids = [f["feature_idx"] if isinstance(f, dict) else f
                   for f in (top.get("features", top) if isinstance(top, dict)
                             else top)][:50]
        rng = np.random.default_rng(3)
        sel = sorted(set([71839] + list(top_ids)
                         + rng.integers(0, sae.cfg.d_sae, 256).tolist()))
        sel_t = torch.tensor(sel, device=DEVICE)

        def raw_series(k):
            l = bt.labs[k]
            lo = max(l["think_lo"], T - 1)
            if l["n_tokens"] - lo < SEG:
                return None
            seg = bt.Ad[bt.off[k] + lo: bt.off[k] + l["n_tokens"]].float()
            with torch.no_grad():
                z = sae.encode(seg)
            return z[:, sel_t]
        P, nseg = welch_spectra(raw_series, len(sel), bt)
        cen, low = spec_summary(P)
        np.savez(os.path.join(args.out, "spectra_llamascope.npz"),
                 P=P.astype(np.float32), centroid=cen, lowfrac=low,
                 sel=np.array(sel), nseg=nseg)
        # probe on Llama-Scope right-edge codes (full feature space)
        with torch.no_grad():
            etr = torch.cat([sae.encode(Xtr[i:i + 512, -1, :] * bt.sd + bt.mu)
                             for i in range(0, Xtr.shape[0], 512)])
            ete = torch.cat([sae.encode(Xte[i:i + 512, -1, :] * bt.sd + bt.mu)
                             for i in range(0, Xte.shape[0], 512)])
        res = probe_with_auc(etr, ytr, ete, yte, "ls_edge")
        i71839 = sel.index(71839)
        res["feat71839_lowfrac"] = float(low[i71839])
        res["feat71839_centroid"] = float(cen[i71839])
        json.dump(res, open(ls_path, "w"), indent=1)
        print("[llamascope]", {k: round(v, 3) for k, v in res.items()
                               if isinstance(v, float)}, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
