"""Modal: additivity with measured position weights — and the correct span test.

Two questions, one run.

WHY THIS REPLACES BOTH linfit AND span2. The 36-condition regression rejects the model
it was read against: χ²/dof = 3.56, 10 of 36 predictions outside their own 95% intervals
(p ≈ 7.5e-6), and three conditions whose projection onto the target is exactly zero
measure significantly non-zero (z = −3.2, −3.1, +2.6). A pure projection model cannot do
that. But the failure is not additivity — it is the assumption that every position
carries the same weight. Two separable claims live inside "linear":

    additivity   Δ(c) = Σ_t a_t c_t     what is written at one position does not change
                                         what another position contributes
    homogeneity  a_t = const             so the effect is the UNWEIGHTED projection

Homogeneity is already refuted. Additivity is untouched, and it is the load-bearing one —
because a span effect (adjacency, "coherent transition", a state that carries forward)
IS a failure of additivity, by definition. So the span question and the additivity
question are the same question, and a two-arm contiguous-vs-scattered contrast is the
wrong instrument: under additivity with unequal weights, S_span is non-zero whenever the
two supports sit on different positions, which they always do. That would have been the
third false positive in this family (coverage, then sign composition, then position).

DESIGN. One fixed alternating profile at k=12, one set of eval pairs throughout.
  1. Measure a_t from single-position writes, matched and flipped, at every position.
  2. Run N random block-constant coefficient vectors c on the same pairs.
  3. Predict each condition two ways —
        homogeneous  R = Σ_t c_t π_t / k
        weighted     R = Σ_t a_t c_t π_t / Σ_t a_t          (12 params from step 1)
     and compare χ²/dof. Step 1 is single-position data; step 2/3 are multi-position, so
     this is an out-of-regime prediction with N − 0 free parameters, not a fit.
  4. THE SPAN TEST: regress the weighted-model residual on an adjacency statistic
     (number of adjacent position pairs both written with the correct sign). Interaction
     is the only thing that survives subtracting a measured additive model, and position
     can no longer confound it because position is in the model.

Registered before running: if χ²/dof falls toward 1 and the adjacency coefficient is
consistent with zero, the conclusion is "additive over positions with unequal weights;
no adjacency effect survives — a wider handle buys resolution, not interaction".

    modal run experiments/temporal_screen/trajectory_steering/weights_modal.py
"""

import pathlib
import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

app = modal.App("trajectory-weights")
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
def weights(model_id: str, layer: int, k: int, n_train: int, n_eval: int,
            n_cond: int, frac: float, ell: int = 1):
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
    rng = random.Random(86420)
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
    m = frac * bn
    print(f"[dir] base_norm={bn:.1f}")

    # one fixed profile and one set of eval pairs for everything below.
    # ell = run length: with ell=1 (alternating) adjacent positions can never BOTH be
    # correctly signed under a block-constant write, so the adjacency statistic is
    # degenerate. ell >= 2 gives runs, making run-coherence measurable.
    prof01 = [1 if (i // ell) % 2 == 0 else 0 for i in range(k)]
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

    def arm(vecs):
        return np.array([(margin(t, vecs, m) - margin(f, vecs, m)) - b
                         for (t, f), b in zip(pairs, base)])

    # ---- step 1: per-position weights from single-position writes ----
    a_t, a_sem = [], []
    for t in range(k):
        v_plus = [None] * k
        v_plus[t] = float(pi[t]) * u
        d = arm(v_plus)
        a_t.append(float(d.mean()))
        a_sem.append(float(d.std(ddof=1) / np.sqrt(len(d))))
    a = np.array(a_t)
    print(f"[weights] a_t = " + " ".join(f"{x:+.2f}" for x in a))
    print(f"[weights] mean={a.mean():+.3f} sd={a.std():.3f} "
          f"ratio max/min={a.max()/max(a.min(),1e-9):.2f}")

    d_full = arm([float(pi[t]) * u for t in range(k)])
    full_mean = float(d_full.mean())

    # ---- step 2/3: random block-constant conditions, predicted two ways ----
    COEFS = [-1.0, -0.5, 0.0, 0.5, 1.0]
    pts = []
    for ci in range(n_cond):
        W = rng.choice([w for w in (2, 3, 4, 6) if k % w == 0])
        nb = k // W
        c = np.array([rng.choice(COEFS) for _ in range(nb)])
        if np.all(c == 0):
            continue
        c_t = np.array([c[t // W] for t in range(k)])
        pred_homog = float(np.dot(c_t, pi) / k)
        pred_weight = float(np.dot(a * c_t, pi) / np.sum(a))
        d = arm([float(c_t[t]) * u if c_t[t] != 0 else None for t in range(k)])
        obs = float(d.mean() / full_mean)
        sem = float(d.std(ddof=1) / np.sqrt(len(d)) / abs(full_mean))
        # adjacency: pairs of neighbouring positions both written with correct sign
        adj = sum(1 for t in range(k - 1)
                  if c_t[t] * pi[t] > 0 and c_t[t + 1] * pi[t + 1] > 0)
        pts.append({"W": int(W), "coeffs": [float(x) for x in c],
                    "pred_homog": pred_homog, "pred_weighted": pred_weight,
                    "obs_R": obs, "sem": sem, "adjacency": int(adj),
                    "n_written": int((c_t != 0).sum())})
        print(f"  [{ci+1}/{n_cond}] W={W} homog={pred_homog:+.3f} "
              f"weighted={pred_weight:+.3f} obs={obs:+.3f}±{sem:.3f} adj={adj}")

    P_h = np.array([p["pred_homog"] for p in pts])
    P_w = np.array([p["pred_weighted"] for p in pts])
    O = np.array([p["obs_R"] for p in pts])
    S = np.array([p["sem"] for p in pts])
    chi_h = float(np.mean(((O - P_h) / S) ** 2))
    chi_w = float(np.mean(((O - P_w) / S) ** 2))
    # step 4: does any residual track adjacency?
    res = O - P_w
    ADJ = np.array([p["adjacency"] for p in pts], dtype=float)
    NW = np.array([p["n_written"] for p in pts], dtype=float)
    X = np.stack([np.ones_like(ADJ), ADJ, NW], 1)
    beta, *_ = np.linalg.lstsq(X, res, rcond=None)
    resid_after = res - X @ beta
    dof = max(len(res) - 3, 1)
    cov = (resid_after @ resid_after / dof) * np.linalg.pinv(X.T @ X)
    se_adj = float(np.sqrt(max(cov[1, 1], 0)))
    out = {"model": model_id, "layer": int(L), "k": k, "frac": frac, "ell": ell,
           "base_norm": bn, "a_t": a_t, "a_sem": a_sem,
           "full_mean": full_mean, "points": pts,
           "chi2_per_dof_homogeneous": chi_h, "chi2_per_dof_weighted": chi_w,
           "mean_abs_err_homogeneous": float(np.mean(np.abs(O - P_h))),
           "mean_abs_err_weighted": float(np.mean(np.abs(O - P_w))),
           "adjacency_coef": float(beta[1]), "adjacency_se": se_adj,
           "adjacency_t": float(beta[1] / se_adj) if se_adj else 0.0,
           "n_points": len(pts)}
    print(f"\n[verdict] chi2/dof homogeneous={chi_h:.2f}  weighted={chi_w:.2f}")
    print(f"[verdict] mean|err| homogeneous={out['mean_abs_err_homogeneous']:.3f}  "
          f"weighted={out['mean_abs_err_weighted']:.3f}")
    print(f"[span]    adjacency coefficient={beta[1]:+.4f} ± {se_adj:.4f} "
          f"(t={out['adjacency_t']:+.2f})")
    return out


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-1.5B-Instruct", layer: int = -1, k: int = 12,
         n_train: int = 40, n_eval: int = 28, n_cond: int = 40, frac: float = 0.5,
         ell: int = 1):
    import json
    res = weights.remote(model, layer, k, n_train, n_eval, n_cond, frac, ell)
    outdir = ROOT / "results" / "temporal_screen"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / (f"weights_ell{ell}.json" if ell != 1 else "weights.json")
    out.write_text(json.dumps(res, indent=2))
    print("[saved]", out)
