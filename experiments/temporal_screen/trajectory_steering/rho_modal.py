"""Modal: is the law's dose-drift a sign asymmetry, or a genuine cross-position effect?

The block-constant arm gives every segment a coefficient of ±1 — only the sign varies
across cells, never the magnitude. Writing the per-position contribution as +q₊(m) when
the write matches a segment's own attribute and −q₋(m) when it opposes it,

    Δ(W) = n₊·q₊ − n₋·q₋ ,   Δ_full = k·q₊ ,   R = [n₊ − ρ·n₋] / k ,  ρ ≡ q₋/q₊

so if positions contribute independently and symmetrically (ρ = 1), R equals the mean
absolute block-mean **for any per-position response function whatsoever** — saturating,
convex, sigmoid. R is therefore dose-invariant under every position-independent
nonlinearity, which makes its dose-dependence a sharp test.

The existing data fails that test: R rises with dose in 6/6 cells at 1.5B and 6/6 at 7B.
Two explanations survive and only one is a span effect:

  (a) ASYMMETRIC SIGN RESPONSE — a wrong-signed write hurts more than a right-signed one
      helps (ρ > 1), with the asymmetry shrinking as dose grows. Position-independent;
      no span content.
  (b) CROSS-POSITION INTERACTION — the real thing.

This job measures ρ(m) directly, which `convex.json` cannot supply because it stored only
matched-sign marginals. For every position, steer that position ALONE with the write
matched to its own attribute (giving q₊) and with the write flipped (giving q₋), across
the dose grid. Then substitute the measured ρ(m) into R = [n₊ − ρ n₋]/k and ask whether
the drift is fully accounted for.

  - If ρ(m) explains the drift: the response is position-independent with an asymmetric
    sign response, and there is no span effect at this scale. Close the door.
  - If a residual survives: that residual IS the span effect, and it is largest at low
    dose — where no span test has yet been run.

    modal run experiments/temporal_screen/trajectory_steering/rho_modal.py
"""

import pathlib
import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

app = modal.App("trajectory-rho")
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
def rho(model_id: str, layer: int, k: int, ells: list, ws: list, n_train: int,
        n_eval: int, fracs: list):
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
    rng = random.Random(31415)
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

    def make_pairs(prof01, n):
        pairs = []
        for _ in range(n):
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
        return pairs

    results = {"model": model_id, "layer": int(L), "k": k, "base_norm": bn,
               "rho": {}, "cells": {}}

    # ---- q+(m) and q-(m) from single-position writes on an alternating profile ----
    prof01 = [1 if i % 2 == 0 else 0 for i in range(k)]
    pi = np.array([1.0 if l else -1.0 for l in prof01])
    pairs = make_pairs(prof01, n_eval)
    base = [margin(t, [], 0) - margin(f, [], 0) for t, f in pairs]

    for fr in fracs:
        m = fr * bn
        qp, qm = [], []
        for t in range(k):
            for sgn, acc in ((+1.0, qp), (-1.0, qm)):
                vecs = [None] * k
                vecs[t] = sgn * float(pi[t]) * u
                d = np.array([(margin(a, vecs, m) - margin(b_, vecs, m)) - bb
                              for (a, b_), bb in zip(pairs, base)])
                acc.append(float(d.mean()))
        q_plus = float(np.mean(qp))
        q_minus = float(-np.mean(qm))          # flipped write should hurt ⇒ negate
        rho_hat = q_minus / q_plus if q_plus else float("nan")
        results["rho"][fr] = {"q_plus": q_plus, "q_minus": q_minus,
                              "rho": rho_hat,
                              "q_plus_by_pos": qp, "q_minus_by_pos": qm}
        print(f"[rho frac={fr}] q+={q_plus:+.3f}  q-={q_minus:+.3f}  "
              f"rho={rho_hat:.3f}")

    # ---- does rho(m) account for the dose-drift of R? ----
    for ell in ells:
        prof = [1 if (t // ell) % 2 == 0 else 0 for t in range(k)]
        pe = np.array([1.0 if l else -1.0 for l in prof])
        if abs(pe.mean()) > 1e-9:
            continue
        prs = make_pairs(prof, n_eval)
        bse = [margin(t, [], 0) - margin(f, [], 0) for t, f in prs]

        def arm(vecs, m):
            return np.array([(margin(t, vecs, m) - margin(f, vecs, m)) - b
                             for (t, f), b in zip(prs, bse)])
        for W in ws:
            if k % W:
                continue
            nb = k // W
            mu = np.array([pe[b * W:(b + 1) * W].mean() for b in range(nb)])
            c = np.sign(mu)
            pred_plain = float(np.mean(np.abs(mu)))
            if pred_plain <= 1e-9 or pred_plain >= 1 - 1e-9:
                continue
            n_plus = sum(1 for t in range(k) if c[t // W] == pe[t])
            n_minus = sum(1 for t in range(k) if c[t // W] == -pe[t])
            for fr in fracs:
                m = fr * bn
                d_full = arm([float(pe[t]) * u for t in range(k)], m)
                d_c = arm([float(c[t // W]) * u if c[t // W] != 0 else None
                           for t in range(k)], m)
                obs = float(d_c.mean() / d_full.mean())
                r = results["rho"][fr]["rho"]
                pred_rho = (n_plus - r * n_minus) / k
                results["cells"][f"ell{ell}_W{W}_frac{fr}"] = {
                    "obs_R": obs, "pred_plain": pred_plain,
                    "pred_with_rho": float(pred_rho), "rho": r,
                    "n_plus": n_plus, "n_minus": n_minus,
                    "err_plain": float(abs(obs - pred_plain)),
                    "err_with_rho": float(abs(obs - pred_rho))}
                print(f"  ell={ell} W={W} frac={fr}: obs={obs:+.3f} "
                      f"plain={pred_plain:.3f} (err {abs(obs-pred_plain):.3f})  "
                      f"with_rho={pred_rho:+.3f} (err {abs(obs-pred_rho):.3f})")

    ep = np.mean([v["err_plain"] for v in results["cells"].values()])
    er = np.mean([v["err_with_rho"] for v in results["cells"].values()])
    results["summary"] = {"mean_err_plain": float(ep), "mean_err_with_rho": float(er),
                          "n_cells": len(results["cells"])}
    print(f"\n[verdict] over {len(results['cells'])} (cell, dose) points: "
          f"mean |err| plain={ep:.3f}  with measured rho={er:.3f}")
    return results


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-1.5B-Instruct", layer: int = -1, k: int = 12,
         ells: str = "1,2,3,6", ws: str = "2,3,4,6", n_train: int = 40,
         n_eval: int = 24, fracs: str = "0.2,0.35,0.5"):
    import json
    res = rho.remote(model, layer, k, [int(x) for x in ells.split(",")],
                     [int(x) for x in ws.split(",")], n_train, n_eval,
                     [float(x) for x in fracs.split(",")])
    outdir = ROOT / "results" / "temporal_screen"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "rho.json").write_text(json.dumps(res, indent=2))
    print("[saved]", outdir / "rho.json")
