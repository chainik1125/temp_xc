"""Modal: does a wide knob beat its own parts? — the superadditivity control.

The claim under test is that a span-W knob is worth MORE than the W single-segment
writes it contains. The right control is not a fitted global line (whose right-hand
endpoint is calibrated by construction); it is the PER-POSITION MARGINAL
DECOMPOSITION (realmodel agent's design):

    S(B) = Δ(B) − Σ_{t∈B} Δ_t

where Δ_t is measured by steering segment t ALONE. Every mundane alternative —
magnitude/coverage nonlinearity, per-segment direction heterogeneity, positional
heterogeneity — is differenced away, because each block is compared against its own
constituent positions.

Edge-penalty model (one constant): a block pays a fixed boundary cost c regardless of
width, and yields a per covered slot, so
    Δ_t = a − c,    Δ(W) = W·a − c,    S(W) = c·(W−1),    Δ(W)/W = a − c/W.
Fit c from the singleton-based S(W) and the efficiency curve follows with ZERO free
parameters. Note this form is strictly monotone in W — a non-monotone efficiency
curve falsifies it.

MECHANISM-POSITIVE TEST (the one that can falsify the account): scramble the schedule
INSIDE the block. Coverage, contiguity and total injected norm are identical; only
the correctness of the within-block transition is destroyed. If S collapses, the
"a wide knob writes a coherent transition" account is supported; if S survives,
contiguity is doing something else.

Also reported, per the same review: Δ per unit INJECTED NORM vs W (a span-W knob at
fixed frac injects W× the norm of a span-1 knob, so raw Δ-vs-W flatters wide knobs),
realized coverage per W (clipping check), and the full Δ(W, frac) surface, since
comparing per-arm optima conflates span with magnitude tuning.

Task: alt_phase (fixed profile ⇒ the knob count is honestly m, no per-episode side
information), k=8.

    modal run experiments/temporal_screen/trajectory_steering/convex_modal.py
"""

import pathlib
import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

app = modal.App("trajectory-convex")
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


@app.function(gpu="A10G", image=image, timeout=5400)
def convex(model_id: str, layer: int, k: int, ws: list, n_train: int, n_eval: int,
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
    rng = random.Random(1717)
    unit = lambda x: x / (np.linalg.norm(x) + 1e-8)

    def build(carrier, sents):
        text, spans = carrier, []
        for j, s in enumerate(sents):
            if j:
                text += " "
            spans.append((len(text), len(text) + len(s)))
            text += s
        return text, spans

    def encode(text, char_spans):
        enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
        offs = enc["offset_mapping"][0].tolist()
        tspans = []
        for (a, b) in char_spans:
            ix = [i for i, (x, y) in enumerate(offs) if y > x and y > a and x < b]
            tspans.append((min(ix), max(ix)))
        return enc["input_ids"].to(dev), tspans

    def seg_logprob(ids, tspans):
        with torch.no_grad():
            lp = model(ids).logits[0].log_softmax(-1).float()
        return float(sum(lp[p - 1, ids[0, p]]
                         for a, b in tspans for p in range(a, b + 1) if p >= 1))

    def capture(text, char_spans):
        ids, tspans = encode(text, char_spans)
        h = layers_[L].register_forward_hook(cap_hook)
        with torch.no_grad():
            model(ids)
        h.remove()
        hh = cap["h"][0].float().cpu().numpy()
        return ([hh[a:b + 1].mean(0) for a, b in tspans],
                [float(np.linalg.norm(hh[p])) for a, b in tspans
                 for p in range(a, b + 1)])

    def margin(pair, vecs, m):
        ids, tspans = pair
        steer["v"] = []
        n_written = 0
        if m > 0:
            for i, v in enumerate(vecs):
                if v is None:
                    continue
                a, b = tspans[i]
                steer["v"].append((max(a - 1, 0), max(b - 1, 0), m * v))
                n_written += (b - a + 1)
        val = seg_logprob(ids, tspans)
        steer["v"] = []
        return val, n_written

    # ---- direction + fixed alternating profile (knob count is honestly m) ----
    profA = [1 if i % 2 == 0 else 0 for i in range(k)]
    profB = [0 if i % 2 == 0 else 1 for i in range(k)]
    segs_T, segs_C, norms_all = [], [], []
    for _ in range(n_train):
        idxs = [rng.randrange(10) for _ in range(k)]
        car = rng.choice(CARRIERS)
        for prof in (profA, profB):
            text, cs = build(car, [(TENSE if l else CALM)[i]
                                   for l, i in zip(prof, idxs)])
            segs, norms = capture(text, cs)
            for l, sv in zip(prof, segs):
                (segs_T if l else segs_C).append(sv)
            norms_all += norms
    u = torch.tensor(unit(np.mean(segs_T, 0) - np.mean(segs_C, 0)),
                     device=dev, dtype=torch.float32)
    bn = float(np.mean(norms_all))
    pi = [1.0 if l else -1.0 for l in profA]
    print(f"[dir] base_norm={bn:.1f}")

    pairs = []
    for _ in range(n_eval):
        idxs = [rng.randrange(10) for _ in range(k)]
        car = rng.choice(CARRIERS)
        tT, cT = build(car, [(TENSE if l else CALM)[i]
                             for l, i in zip(profA, idxs)])
        tF, cF = build(car, [(TENSE if l else CALM)[i]
                             for l, i in zip(profB, idxs)])
        pairs.append((encode(tT, cT), encode(tF, cF)))
    base = [margin(t, [], 0)[0] - margin(f, [], 0)[0] for t, f in pairs]

    def run(sel_fn, m, scramble=False):
        """sel_fn(j) -> set of covered segments. Returns per-pair deltas + norm."""
        ds, nw = [], []
        for j, ((t, f), b) in enumerate(zip(pairs, base)):
            sel = sorted(sel_fn(j))
            coefs = [pi[i] for i in sel]
            if scramble:
                order = list(range(len(coefs)))
                rng.shuffle(order)
                coefs = [coefs[o] for o in order]
            vecs = [None] * k
            for pos, c in zip(sel, coefs):
                vecs[pos] = c * u
            mt, w1 = margin(t, vecs, m)
            mf, w2 = margin(f, vecs, m)
            ds.append((mt - mf) - b)
            nw.append((w1 + w2) / 2)
        return np.array(ds), float(np.mean(nw)), float(np.mean([len(sel_fn(j))
                                                                for j in range(len(pairs))]))

    results = {"model": model_id, "layer": int(L), "k": k, "base_norm": bn,
               "marginals": {}, "blocks": {}, "scrambled": {}}

    for fr in fracs:
        m = fr * bn
        # --- per-position marginals: steer each segment alone ---
        marg = {}
        for t in range(k):
            ds, nw, cvg = run(lambda j, t=t: {t}, m)
            marg[t] = {"mean": float(ds.mean()),
                       "sem": float(ds.std(ddof=1) / np.sqrt(len(ds))),
                       "tokens_written": nw}
        results["marginals"][fr] = marg
        print(f"[frac={fr}] marginals Δ_t: "
              + " ".join(f"{marg[t]['mean']:+.2f}" for t in range(k)))

        # --- contiguous blocks of width W (rotated), + within-block scramble ---
        for W in ws:
            nb = k // W
            sel = lambda j, W=W, nb=nb: set(range((j % nb) * W, (j % nb) * W + W))
            ds, nw, cvg = run(sel, m)
            sum_marg = float(np.mean([sum(marg[t]["mean"] for t in sel(j))
                                      for j in range(len(pairs))]))
            S = float(ds.mean()) - sum_marg
            results["blocks"].setdefault(fr, {})[W] = {
                "mean": float(ds.mean()),
                "sem": float(ds.std(ddof=1) / np.sqrt(len(ds))),
                "sum_of_marginals": sum_marg, "S_superadditivity": S,
                "realized_coverage": cvg, "tokens_written": nw,
                "delta_per_token": float(ds.mean()) / max(nw, 1),
                "deltas": [float(x) for x in ds]}
            ds2, nw2, _ = run(sel, m, scramble=True)
            S2 = float(ds2.mean()) - sum_marg
            results["scrambled"].setdefault(fr, {})[W] = {
                "mean": float(ds2.mean()),
                "sem": float(ds2.std(ddof=1) / np.sqrt(len(ds2))),
                "S_superadditivity": S2,
                "deltas": [float(x) for x in ds2]}
            print(f"  W={W}: Δ={ds.mean():+7.2f} Σmarg={sum_marg:+7.2f} "
                  f"S={S:+6.2f} | scrambled Δ={ds2.mean():+7.2f} S={S2:+6.2f} "
                  f"| cov={cvg:.1f} Δ/tok={float(ds.mean())/max(nw,1):+.3f}")
    return results


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-1.5B-Instruct", layer: int = -1, k: int = 8,
         ws: str = "1,2,4,8", n_train: int = 30, n_eval: int = 28,
         fracs: str = "0.2,0.35,0.5"):
    import json
    res = convex.remote(model, layer, k, [int(x) for x in ws.split(",")], n_train,
                        n_eval, [float(x) for x in fracs.split(",")])
    outdir = ROOT / "results" / "temporal_screen"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "convex.json").write_text(json.dumps(res, indent=2))
    print("[saved]", outdir / "convex.json")
