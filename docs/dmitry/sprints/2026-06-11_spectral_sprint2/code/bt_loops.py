"""bt_loops.py — top-ranked workflow candidate: repetition/rumination loops.

Labels are programmatic (no judge): a loop BOUT starts at the first token of
the second occurrence of any 6-gram that repeats >= 3 times within a
120-token span of the same trace (think region only). The candidate predicts
MID-band localization (loop periods ~5-60 tokens) — the first non-DC
behaviour if it holds.

Measurements (same protocol as other rows):
  - bout counts; onset positions
  - raw window-mean AUC vs T (predict: weaker monotone gain than backtracking
    — the mean is period-invariant, so DC should NOT be where this lives)
  - pooled DCT-band probes at T=32 (predict: mid bands >= DC)
  - LAYER-0 CONTROL: identical band probes on windows of raw token
    EMBEDDINGS (wte lookups, no forward pass). If the L10 band profile ==
    embedding band profile, the signal is lexical recirculation, not
    residual dynamics.

Run after ws_out exists: python bt_loops.py
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import torch

from fb_core import DEVICE, dct_basis
from bt_freq import probe_with_auc
from bt_wscan import BTW

TS = [1, 4, 8, 16, 32, 48, 64]
BANDS32 = [[0], list(range(1, 11)), list(range(11, 21)), list(range(21, 32))]


def find_loops():
    traces = {json.loads(l)["trace_id"]: json.loads(l)
              for l in open("bt_data/traces.jsonl")}
    labs = {j["trace_id"]: j for j in
            (json.loads(l) for l in open("bt_data/labels.jsonl"))}
    ids = json.load(open("ws_out/trace_ids.json"))
    bouts = []
    for tid in ids:
        tr, l = traces[tid], labs[tid]
        toks = tr["full_token_ids"]
        lo, hi = l["think_lo"], l["n_tokens"]
        first_seen = {}
        occ = defaultdict(list)
        for p in range(lo, hi - 6):
            g = tuple(toks[p:p + 6])
            occ[g].append(p)
        ev, last = [], -10**9
        for g, ps in occ.items():
            if len(ps) >= 3 and ps[-1] - ps[0] <= 120:
                onset = ps[1]            # second occurrence = loop begins
                ev.append(onset)
        ev = sorted(set(ev))
        merged = []
        for e in ev:
            if not merged or e - merged[-1] > 40:
                merged.append(e)
        bouts.append(merged)
    return bouts


def main():
    bt = BTW("ws_out")
    bouts = find_loops()
    n_b = sum(len(b) for b in bouts)
    n_traces_with = sum(1 for b in bouts if b)
    res = {"n_bouts": n_b, "n_traces_with_bouts": n_traces_with}
    print(f"{n_b} loop bouts across {n_traces_with} traces", flush=True)
    if n_b < 40:
        res["verdict"] = "too few bouts for probes; recording counts only"
        json.dump(res, open("ws_out/loops_screen.json", "w"), indent=1)
        print("DONE (insufficient)", flush=True)
        return

    def pairs_for(train, T):
        import random as _r
        pos, neg = [], []
        rng = np.random.default_rng(57)
        for k, l in enumerate(bt.labs):
            if (k in bt.test_traces) == train:
                continue
            ev = np.array(bouts[k], dtype=int)
            for e in ev:
                pos.extend((k, e + o) for o in range(-13, -7)
                           if e + o >= max(T - 1, l["think_lo"]))
            lo = max(l["think_lo"], T - 1)
            cand = [p for p in range(lo, l["n_tokens"])
                    if not len(ev) or np.abs(ev - p).min() > 25]
            take = (rng.choice(len(cand), size=min(len(cand), 60),
                               replace=False) if cand else [])
            neg.extend((k, cand[i]) for i in take)
        _r.Random(59).shuffle(neg)
        neg = neg[:5 * max(1, len(pos))]
        y = torch.tensor([1] * len(pos) + [0] * len(neg), device=DEVICE)
        return pos + neg, y

    for T in TS:
        tr_p, ytr = pairs_for(True, T)
        te_p, yte = pairs_for(False, T)
        if (yte == 1).sum() < 20:
            continue
        r = probe_with_auc(bt.means(tr_p, T), ytr, bt.means(te_p, T), yte,
                           "x")
        res[f"rawmean_T{T}"] = {"auc": r["x_auc"],
                                "n_pos_test": int((yte == 1).sum())}
        print(f"[loops raw T={T:3d}] auc={r['x_auc']:.3f}", flush=True)

    # band probes at T=32: L10 residuals AND layer-0 embeddings control
    from transformers import AutoModelForCausalLM
    emb = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        torch_dtype=torch.float16).get_input_embeddings().weight
    emb = emb.to(DEVICE)
    traces = {json.loads(l)["trace_id"]: json.loads(l)
              for l in open("bt_data/traces.jsonl")}
    ids = json.load(open("ws_out/trace_ids.json"))
    tok_ids = [traces[t]["full_token_ids"] for t in ids]

    def emb_windows(pairs, T):
        outs = []
        for k, p in pairs:
            sel = torch.tensor(tok_ids[k][p - T + 1: p + 1], device=DEVICE)
            outs.append(emb[sel].float())
        X = torch.stack(outs)
        return (X - X.mean((0, 1))) / (X.std((0, 1)) + 1e-6)

    psi = dct_basis(32).to(DEVICE)
    tr_p, ytr = pairs_for(True, 32)
    te_p, yte = pairs_for(False, 32)
    for name, wfn in [("l10", lambda pp: bt.windows(pp, 32)),
                      ("emb0", lambda pp: emb_windows(pp, 32))]:
        Ctr = torch.einsum("wt,ntd->nwd", psi, wfn(tr_p))
        Cte = torch.einsum("wt,ntd->nwd", psi, wfn(te_p))
        for bi, band in enumerate(BANDS32):
            r = probe_with_auc(Ctr[:, band, :].mean(1), ytr,
                               Cte[:, band, :].mean(1), yte, "x")
            res[f"{name}_band{bi}"] = {"auc": r["x_auc"]}
            print(f"[loops {name} band{bi}] auc={r['x_auc']:.3f}", flush=True)
        del Ctr, Cte
        torch.cuda.empty_cache()
    json.dump(res, open("ws_out/loops_screen.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
