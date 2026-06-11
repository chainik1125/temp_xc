"""bt_relabel.py — screen additional reasoning behaviours on the existing
trace cache (sprint 2B): same D+ protocol as backtracking but with new
marker-keyword sets, then raw window-mean timescale curves per behaviour.

Behaviours (marker tokens, matched case-insensitively on stripped decoded
tokens, first token of a sentence-events not required — any occurrence,
deduped within 20 tokens):
  verification : check, verify, confirm, recheck
  conclusion   : therefore, thus, hence, answer
  uncertainty  : maybe, perhaps, wonder, unsure

For each behaviour: event positions; D+ = [-13,-8] before events (in think
region); negatives = far from events (buffer 25); raw window-mean probe AUC
for T in {1,4,8,16,32,48,64}; pooled DCT-band probes at T=32.

Run after the W-scan (shares ws_out cache): python bt_relabel.py
"""
from __future__ import annotations

import json
import os
import string

import numpy as np
import torch

from fb_core import DEVICE, dct_basis
from bt_freq import probe_with_auc
from bt_wscan import BTW

BEHAVIOURS = {
    "verification": {"check", "verify", "confirm", "recheck", "checking",
                     "double", "verifying", "confirms", "valid", "correct"},
    "conclusion": {"therefore", "thus", "hence", "answer", "final",
                   "boxed", "conclusion"},
    "uncertainty": {"maybe", "perhaps", "wonder", "unsure", "might",
                    "possibly", "guess", "unclear", "probably"},
}
TS = [1, 4, 8, 16, 32, 48, 64]
BANDS32 = [[0], list(range(1, 11)), list(range(11, 21)), list(range(21, 32))]
NEG_BUFFER = 25
NEG_PER_POS = 5


def find_events(out="ws_out"):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    traces = {json.loads(l)["trace_id"]: json.loads(l)
              for l in open("bt_data/traces.jsonl")}
    ids = json.load(open(f"{out}/trace_ids.json"))
    labs = {j["trace_id"]: j for j in
            (json.loads(l) for l in open("bt_data/labels.jsonl"))}
    table = str.maketrans("", "", string.punctuation + " Ġ")
    events = {b: [] for b in BEHAVIOURS}      # list per trace of positions
    for tid in ids:
        tr, l = traces[tid], labs[tid]
        toks = [tok.decode([t]).strip().lower().translate(table)
                for t in tr["full_token_ids"]]
        for b, kws in BEHAVIOURS.items():
            ev, last = [], -100
            for p in range(l["think_lo"], l["n_tokens"]):
                if toks[p] in kws and p - last > 20:
                    ev.append(p)
                    last = p
            events[b].append(ev)
    return events


def main():
    out = "ws_out"
    bt = BTW(out)
    events = find_events(out)
    res = {}
    for b, ev_per_trace in events.items():
        n_ev = sum(len(e) for e in ev_per_trace)
        res[b] = {"n_events": n_ev}
        print(f"[{b}] {n_ev} events", flush=True)
        if n_ev < 60:
            res[b]["skipped"] = "too few events"
            continue

        def pairs_for(train, T):
            import random as _r
            pos, neg = [], []
            rng = np.random.default_rng(29)
            for k, l in enumerate(bt.labs):
                if (k in bt.test_traces) == train:
                    continue
                ev = np.array(ev_per_trace[k], dtype=int)
                for e in ev:
                    pos.extend((k, e + o) for o in range(-13, -7)
                               if e + o >= max(T - 1, l["think_lo"]))
                lo = max(l["think_lo"], T - 1)
                cand = [p for p in range(lo, l["n_tokens"])
                        if not len(ev) or np.abs(ev - p).min() > NEG_BUFFER]
                take = (rng.choice(len(cand), size=min(len(cand), 60),
                                   replace=False) if cand else [])
                neg.extend((k, cand[i]) for i in take)
            _r.Random(31).shuffle(neg)
            neg = neg[:NEG_PER_POS * max(1, len(pos))]
            y = torch.tensor([1] * len(pos) + [0] * len(neg), device=DEVICE)
            return pos + neg, y

        for T in TS:
            tr_p, ytr = pairs_for(True, T)
            te_p, yte = pairs_for(False, T)
            if (ytr == 1).sum() < 50 or (yte == 1).sum() < 20:
                continue
            Ftr, Fte = bt.means(tr_p, T), bt.means(te_p, T)
            r = probe_with_auc(Ftr, ytr, Fte, yte, "x")
            res[b][f"rawmean_T{T}"] = {"auc": r["x_auc"],
                                       "balacc": r["x_balacc"],
                                       "n_pos_test": int((yte == 1).sum())}
            print(f"[{b} T={T:3d}] auc={r['x_auc']:.3f}", flush=True)
        # band probes at T=32
        psi = dct_basis(32).to(DEVICE)
        tr_p, ytr = pairs_for(True, 32)
        te_p, yte = pairs_for(False, 32)
        Wtr = bt.windows(tr_p, 32)
        Wte = bt.windows(te_p, 32)
        Ctr = torch.einsum("wt,ntd->nwd", psi, Wtr)
        Cte = torch.einsum("wt,ntd->nwd", psi, Wte)
        for bi, band in enumerate(BANDS32):
            r = probe_with_auc(Ctr[:, band, :].mean(1), ytr,
                               Cte[:, band, :].mean(1), yte, "x")
            res[b][f"band{bi}"] = {"auc": r["x_auc"]}
            print(f"[{b} band{bi}] auc={r['x_auc']:.3f}", flush=True)
        del Wtr, Wte, Ctr, Cte
        torch.cuda.empty_cache()
    json.dump(res, open(f"{out}/relabel_screen.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
