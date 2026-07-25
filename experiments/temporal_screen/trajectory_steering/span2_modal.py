"""Modal: span vs adjacency, with the sign-composition confound removed.

The first span test compared a contiguous block of W segments against a "scattered"
set built with stride k/W. On an alternating profile that stride lands on a single
parity, so the scattered arm wrote ONE sign at every covered position while the
contiguous arm wrote both. The two arms therefore differed in the composition of what
they wrote, not only in adjacency — and a single-sign write on a subset is close to
the DC write we already know is weak. The +7.74 (t = 6.3) that test produced cannot be
attributed to span.

Fix: build the scattered set with stride 3 on an alternating profile at k=12. Then for
every W ≤ 4 the scattered set carries the SAME multiset of target signs as the
contiguous block of the same size, while sharing no adjacent pair:

    W=2  contiguous {s, s+1}          (1+, 1−)   scattered {s, s+3}          (1+, 1−)
    W=3  contiguous {s, s+1, s+2}     (2+, 1−)   scattered {s, s+3, s+6}     (2+, 1−)
    W=4  contiguous {s, .., s+3}      (2+, 2−)   scattered {s, s+3, s+6, s+9}(2+, 2−)

Everything else is held fixed: coverage, the correct per-position sign at every covered
slot, total injected norm, dose, and the eval pairs themselves. Start offset s rotates
over all k positions across eval pairs, so position bias averages out. Per-pair deltas
are stored, so the contrast is paired.

Acceptance bar, set in advance by the audit: S_span > 3 paired SEM, at ≥ 2 widths and
≥ 2 doses, monotone in W, with effect size ≥ 0.15 of the contiguous effect.

    modal run experiments/temporal_screen/trajectory_steering/span2_modal.py
"""

import pathlib
import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

app = modal.App("trajectory-span2")
image = modal.Image.debian_slim().pip_install("torch", "transformers", "accelerate", "numpy")

CALM = [
    "The afternoon passed quietly.", "She sipped her tea by the window.",
    "The garden lay still in the sun.",
    "He hummed an old tune while sorting the mail.",
    "The cat stretched and settled again.",
    "Soft light rested on the bookshelves.",
    "The kettle murmured gently in the kitchen.",
    "They chatted idly about the weather.",
    "The street outside was calm and empty.",
    "She folded the laundry without hurry.",
]
TENSE = [
    "Glass shattered in the next room.", "He shouted for everyone to get down.",
    "The alarm screamed through the corridor.",
    "She ran, heart pounding, for the exit.", "Smoke poured under the door.",
    "The car swerved violently across the lane.",
    "He slammed the door and bolted it.", "Sirens wailed closer and closer.",
    "The floor shook with a sudden blast.",
    "She screamed as the shelf came crashing down.",
]
CARRIERS = ["Journal entry.\n", "From the notebook:\n", "Draft passage.\n",
            "Field notes.\n", "Evening record.\n", "From chapter twelve:\n"]


@app.function(gpu="L4", image=image, timeout=5400)
def span2(model_id: str, layer: int, k: int, ws: list, n_train: int, n_eval: int,
          fracs: list):
    import random
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[load] {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    dev = model.device
    layers_ = model.model.layers
    L = layer if layer >= 0 else len(layers_) // 2
    print(f"[cfg] L={L}, k={k}")

    cap, steer = {}, {"v": []}

    def cap_hook(_m, _i, out):
        cap["h"] = (out[0] if isinstance(out, tuple) else out).detach()

    def steer_hook(_m, _i, out):
        if not steer["v"]:
            return out
        hs = out[0] if isinstance(out, tuple) else out
        for a, b, vec in steer["v"]:
            hs[:, a:b + 1, :] = hs[:, a:b + 1, :] + vec.to(hs.dtype)
        return (hs, *out[1:]) if isinstance(out, tuple) else hs

    layers_[L].register_forward_hook(steer_hook)
    rng = random.Random(2718)
    unit = lambda x: x / (np.linalg.norm(x) + 1e-8)

    def build(car, sents):
        text, spans = car, []
        for j, s in enumerate(sents):
            if j:
                text += " "
            spans.append((len(text), len(text) + len(s)))
            text += s
        return text, spans

    def encode(text, cs):
        enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
        offs = enc["offset_mapping"][0].tolist()
        ts = []
        for (a, b) in cs:
            ix = [i for i, (x, y) in enumerate(offs) if y > x and y > a and x < b]
            ts.append((min(ix), max(ix)))
        return enc["input_ids"].to(dev), ts

    def seg_logprob(ids, ts):
        with torch.no_grad():
            lp = model(ids).logits[0].log_softmax(-1).float()
        return float(sum(lp[p - 1, ids[0, p]]
                         for a, b in ts for p in range(a, b + 1) if p >= 1))

    def capture(text, cs):
        ids, ts = encode(text, cs)
        h = layers_[L].register_forward_hook(cap_hook)
        with torch.no_grad():
            model(ids)
        h.remove()
        hh = cap["h"][0].float().cpu().numpy()
        return ([hh[a:b + 1].mean(0) for a, b in ts],
                [float(np.linalg.norm(hh[p])) for a, b in ts
                 for p in range(a, b + 1)])

    def margin(pair, vecs, m):
        ids, ts = pair
        steer["v"] = []
        if m > 0:
            for i, v in enumerate(vecs):
                if v is None:
                    continue
                a, b = ts[i]
                steer["v"].append((max(a - 1, 0), max(b - 1, 0), m * v))
        val = seg_logprob(ids, ts)
        steer["v"] = []
        return val

    segs_T, segs_C, norms_all = [], [], []
    for _ in range(n_train):
        prof = [1] * (k // 2) + [0] * (k // 2)
        rng.shuffle(prof)
        text, cs = build(rng.choice(CARRIERS),
                         [(TENSE if l else CALM)[rng.randrange(10)] for l in prof])
        segs, norms = capture(text, cs)
        for l, sv in zip(prof, segs):
            (segs_T if l else segs_C).append(sv)
        norms_all += norms
    u = torch.tensor(unit(np.mean(segs_T, 0) - np.mean(segs_C, 0)),
                     device=dev, dtype=torch.float32)
    bn = float(np.mean(norms_all))
    print(f"[dir] base_norm={bn:.1f}")

    prof01 = [1 if i % 2 == 0 else 0 for i in range(k)]
    pi = np.array([1.0 if l else -1.0 for l in prof01])
    pairs = []
    for _ in range(n_eval):
        foil = prof01[:]
        for _ in range(40):
            rng.shuffle(foil)
            if foil != prof01:
                break
        t_i = [rng.randrange(10) for _ in range(k)]
        c_i = [rng.randrange(10) for _ in range(k)]
        car = rng.choice(CARRIERS)

        def sents_for(p):
            out, ti, ci = [], 0, 0
            for l in p:
                if l:
                    out.append(TENSE[t_i[ti]]); ti += 1
                else:
                    out.append(CALM[c_i[ci]]); ci += 1
            return out
        tT, cT = build(car, sents_for(prof01))
        tF, cF = build(car, sents_for(foil))
        pairs.append((encode(tT, cT), encode(tF, cF)))
    base = [margin(t, [], 0) - margin(f, [], 0) for t, f in pairs]

    def run(sel_fn, m):
        return np.array([(margin(t, [float(pi[i]) * u if i in sel_fn(j) else None
                                     for i in range(k)], m)
                          - margin(f, [float(pi[i]) * u if i in sel_fn(j) else None
                                       for i in range(k)], m)) - b
                         for j, ((t, f), b) in enumerate(zip(pairs, base))])

    results = {"model": model_id, "layer": int(L), "k": k, "base_norm": bn,
               "cells": {}}
    for fr in fracs:
        m = fr * bn
        for W in ws:
            cont_sel = lambda j, W=W: {(j + i) % k for i in range(W)}
            scat_sel = lambda j, W=W: {(j + 3 * i) % k for i in range(W)}
            # verify sign multisets match for this W
            sc = sorted(pi[i] for i in cont_sel(0))
            ss = sorted(pi[i] for i in scat_sel(0))
            matched = sc == ss
            cont = run(cont_sel, m)
            scat = run(scat_sel, m)
            d = cont - scat
            sem = float(d.std(ddof=1) / np.sqrt(len(d)))
            results["cells"][f"frac{fr}_W{W}"] = {
                "contiguous": float(cont.mean()),
                "contiguous_sem": float(cont.std(ddof=1) / np.sqrt(len(cont))),
                "scattered": float(scat.mean()),
                "scattered_sem": float(scat.std(ddof=1) / np.sqrt(len(scat))),
                "S_span": float(d.mean()), "S_span_paired_sem": sem,
                "t": float(d.mean() / sem) if sem else 0.0,
                "effect_size": float(d.mean() / cont.mean()) if cont.mean() else 0.0,
                "sign_multiset_matched": bool(matched),
                "deltas_contiguous": [float(x) for x in cont],
                "deltas_scattered": [float(x) for x in scat]}
            print(f"  frac={fr} W={W}: contig={cont.mean():+7.2f} "
                  f"scatter={scat.mean():+7.2f} S_span={d.mean():+6.2f}±{sem:.2f} "
                  f"(t={d.mean()/sem if sem else 0:+.1f}) "
                  f"eff={d.mean()/cont.mean() if cont.mean() else 0:+.2f} "
                  f"signs_matched={matched}")
    return results


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-1.5B-Instruct", layer: int = -1, k: int = 12,
         ws: str = "2,3,4", n_train: int = 40, n_eval: int = 32,
         fracs: str = "0.35,0.5"):
    import json
    res = span2.remote(model, layer, k, [int(x) for x in ws.split(",")], n_train,
                       n_eval, [float(x) for x in fracs.split(",")])
    outdir = ROOT / "results" / "temporal_screen"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "span2.json").write_text(json.dumps(res, indent=2))
    print("[saved]", outdir / "span2.json")
