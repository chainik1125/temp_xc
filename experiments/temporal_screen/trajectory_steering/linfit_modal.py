"""Modal: the high-powered linearity test — many distinct predictions, including negative.

The (W, ℓ) phase diagram supports its claim with only six at-risk cells taking two
distinct predicted values (1/3 and 2/3). That is structural: for a square-wave profile
the block mean μ_b is always a simple rational, so neither a finer W grid nor a phase
sweep nor k=24 adds new predicted values — checked on CPU, k=24 yields 12 at-risk cells
still spanning only {1/6, 1/3, 2/3}.

Random block coefficients fix it. For a balanced profile π and ANY block-constant
coefficient vector c (not just sign(μ_b)), linearity predicts

    R = Δ(c) / Δ_full = W · Σ_b c_b μ_b / k          (the projection of c onto π)

which takes a continuum of values, including NEGATIVE ones when c anti-correlates with
the target. So this measures obs-vs-pred across the whole range [−1, +1] instead of at
two points, and the negative half is a genuinely risky prediction: a write that opposes
the target trajectory should push the margin proportionally the wrong way.

Design: k=12, a fresh random balanced profile per condition, block widths W ∈ {2,3,4,6},
random c_b ∈ {−1, −0.5, 0, +0.5, +1}, ~40 conditions spanning predicted R, each measured
against Δ_full on the SAME eval pairs so the normalisation matches the phase diagram.
Per-pair deltas stored so every point gets a paired bootstrap CI.

    modal run experiments/temporal_screen/trajectory_steering/linfit_modal.py
"""

import pathlib
import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

app = modal.App("trajectory-linfit")
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
def linfit(model_id: str, layer: int, k: int, n_train: int, n_eval: int,
           n_cond: int, frac: float):
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
    print(f"[cfg] L={L}, k={k}, frac={frac}")

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
    rng = random.Random(13579)
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

    # ---- intensity direction ----
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
    m = frac * bn
    print(f"[dir] base_norm={bn:.1f}")

    results = {"model": model_id, "layer": int(L), "k": k, "frac": frac,
               "base_norm": bn, "points": []}
    COEFS = [-1.0, -0.5, 0.0, 0.5, 1.0]

    for ci in range(n_cond):
        # fresh balanced profile + random block-constant coefficients
        prof01 = [1] * (k // 2) + [0] * (k // 2)
        rng.shuffle(prof01)
        pi = np.array([1.0 if l else -1.0 for l in prof01])
        W = rng.choice([w for w in (2, 3, 4, 6) if k % w == 0])
        nb = k // W
        c = np.array([rng.choice(COEFS) for _ in range(nb)])
        if np.all(c == 0):
            continue
        mu = np.array([pi[b * W:(b + 1) * W].mean() for b in range(nb)])
        pred = float(W * np.dot(c, mu) / k)          # projection of c onto pi

        # eval pairs for THIS profile
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
                out, ti, cix = [], 0, 0
                for l in p:
                    if l:
                        out.append(TENSE[t_i[ti]]); ti += 1
                    else:
                        out.append(CALM[c_i[cix]]); cix += 1
                return out
            tT, cT = build(car, sents_for(prof01))
            tF, cF = build(car, sents_for(foil))
            pairs.append((encode(tT, cT), encode(tF, cF)))
        base = [margin(t, [], 0) - margin(f, [], 0) for t, f in pairs]

        def arm(vecs):
            return np.array([(margin(t, vecs, m) - margin(f, vecs, m)) - b
                             for (t, f), b in zip(pairs, base)])
        d_full = arm([float(pi[t]) * u for t in range(k)])
        d_c = arm([float(c[t // W]) * u if c[t // W] != 0 else None
                   for t in range(k)])
        obs = float(d_c.mean() / d_full.mean())
        bs = []
        for s in range(1500):
            idx = np.random.RandomState(s).choice(len(d_c), len(d_c), True)
            bs.append(d_c[idx].mean() / d_full[idx].mean())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        results["points"].append({
            "W": int(W), "coeffs": [float(x) for x in c],
            "pred_R": pred, "obs_R": obs, "ci95": [float(lo), float(hi)],
            "full_mean": float(d_full.mean()), "n": len(d_c)})
        print(f"  [{ci + 1}/{n_cond}] W={W} pred={pred:+.3f} obs={obs:+.3f} "
              f"CI[{lo:+.3f},{hi:+.3f}]")

    P = np.array([p["pred_R"] for p in results["points"]])
    O = np.array([p["obs_R"] for p in results["points"]])
    slope, icept = np.polyfit(P, O, 1)
    results["fit"] = {"slope": float(slope), "intercept": float(icept),
                      "r2": float(np.corrcoef(P, O)[0, 1] ** 2),
                      "mean_abs_err": float(np.mean(np.abs(O - P))),
                      "n_points": len(P),
                      "n_distinct_pred": int(len(set(np.round(P, 3)))),
                      "pred_range": [float(P.min()), float(P.max())]}
    print(f"\n[fit] slope={slope:+.3f} intercept={icept:+.3f} "
          f"R^2={results['fit']['r2']:.3f} mean|err|={results['fit']['mean_abs_err']:.3f}")
    print(f"[fit] {len(P)} points, {results['fit']['n_distinct_pred']} distinct "
          f"predictions, range [{P.min():+.2f}, {P.max():+.2f}]")
    return results


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-1.5B-Instruct", layer: int = -1, k: int = 12,
         n_train: int = 40, n_eval: int = 20, n_cond: int = 36,
         frac: float = 0.5):
    import json
    res = linfit.remote(model, layer, k, n_train, n_eval, n_cond, frac)
    outdir = ROOT / "results" / "temporal_screen"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "linfit.json").write_text(json.dumps(res, indent=2))
    print("[saved]", outdir / "linfit.json")
