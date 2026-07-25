"""Modal: the (W, ℓ) phase diagram — resolution vs intrinsic timescale.

Spec: docs/dmitry/sprints/2026-07-24_semisynth_10h/theory.md §3. Complement of the
W-sweep: there the handle was COVERAGE-limited (write the right thing on a subset of
segments); here it is RESOLUTION-limited (write a CONSTANT inside each window of W,
spanning everything). Block-constant at W=k IS the broadcast arm, so this family
interpolates continuously from full template (W=1) to broadcast (W=k), with the
crossover set by the profile's intrinsic timescale ℓ.

Profiles: square waves of run length ℓ, phase 0 (fixed, per theory — the aligned
matrix carries the falsifiable zig-zag; phase-averaging washes it out), k=12.
Admissible ℓ at k=12 is {1,2,3,6}: ℓ=4 gives three runs and DC=+1/3, which
reintroduces the broadcastable mode the design exists to remove.

Two block normalisations, which discriminate the budget the model obeys:
  block_cap    : c_b = sign(μ_b)              ⇒ predicted R = mean_b |μ_b|
  block_energy : c_b = μ_b / sqrt(mean μ²)    ⇒ predicted R = sqrt(mean_b μ_b²)
where μ_b = mean of the profile inside block b. Every cell is normalised by
Δ_full measured at the SAME ℓ and frac (so a Δ_full(ℓ) trend cannot leak in).

Predicted R (magnitude-cap), zero free parameters:
  ℓ=1: W=1..12 → 1, 0, 1/3, 0, 0, 0
  ℓ=2:          → 1, 1, 1/3, 0, 1/3, 0
  ℓ=3:          → 1, 2/3, 1, 1/3, 0, 0
  ℓ=6:          → 1, 1, 1, 2/3, 1, 0
The zig-zag (W=6 beating W=4 at ℓ=2; W=3 beating W=2 at ℓ=1) is the signature: a
wider window doing better than a narrower one is purely combinatorial.

Also run: CONTIGUOUS vs SCATTERED at matched coverage (predicted tie under
additivity; contiguous winning would mean a genuine span effect beyond bookkeeping),
and per-episode profile mean π̄ recorded (balance in expectation is not balance
per episode).

    modal run experiments/temporal_screen/trajectory_steering/lsweep_modal.py
"""

import pathlib
import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

app = modal.App("trajectory-lsweep")
image = modal.Image.debian_slim().pip_install("torch", "transformers", "accelerate", "numpy")

CALM = [
    "The afternoon passed quietly.",
    "She sipped her tea by the window.",
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
    "Glass shattered in the next room.",
    "He shouted for everyone to get down.",
    "The alarm screamed through the corridor.",
    "She ran, heart pounding, for the exit.",
    "Smoke poured under the door.",
    "The car swerved violently across the lane.",
    "He slammed the door and bolted it.",
    "Sirens wailed closer and closer.",
    "The floor shook with a sudden blast.",
    "She screamed as the shelf came crashing down.",
]
CARRIERS = ["Journal entry.\n", "From the notebook:\n", "Draft passage.\n",
            "Field notes.\n", "Evening record.\n", "From chapter twelve:\n"]


@app.function(gpu="A10G", image=image, timeout=3600)
def lsweep(model_id: str, layer: int, k: int, ells: list, ws: list, n_train: int,
           n_eval: int, fracs: list, cov: int):
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
    print(f"[cfg] {len(layers_)} layers, L={L}, k={k}")

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
    rng = random.Random(31337)

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
        tot = 0.0
        for (a, b) in tspans:
            for p in range(a, b + 1):
                if p - 1 >= 0:
                    tot += lp[p - 1, ids[0, p]].item()
        return tot

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

    unit = lambda x: x / (np.linalg.norm(x) + 1e-8)

    def margin(pair, vecs, m):
        ids, tspans = pair
        steer["v"] = []
        if m > 0:
            for i, v in enumerate(vecs):
                if v is None:
                    continue
                a, b = tspans[i]
                steer["v"].append((max(a - 1, 0), max(b - 1, 0), m * v))
        val = seg_logprob(ids, tspans)
        steer["v"] = []
        return val

    # ---- intensity direction (single direction; the schedule carries the profile) ----
    segs_T, segs_C, norms_all = [], [], []
    for _ in range(n_train):
        prof = [1] * (k // 2) + [0] * (k // 2)
        rng.shuffle(prof)
        sents = [(TENSE if l else CALM)[rng.randrange(10)] for l in prof]
        text, cs = build(rng.choice(CARRIERS), sents)
        segs, norms = capture(text, cs)
        for l, sv in zip(prof, segs):
            (segs_T if l else segs_C).append(sv)
        norms_all += norms
    u = torch.tensor(unit(np.mean(segs_T, 0) - np.mean(segs_C, 0)),
                     device=dev, dtype=torch.float32)
    base_norm = float(np.mean(norms_all))
    print(f"[dir] base_norm={base_norm:.1f}")

    def square(ell):
        """Square wave, run length ell, phase 0: ++..--..++.. (length k)."""
        return [1 if (t // ell) % 2 == 0 else 0 for t in range(k)]

    def make_pairs(prof, n):
        pairs = []
        for _ in range(n):
            foil = prof[:]
            for _ in range(40):
                rng.shuffle(foil)
                if foil != prof:
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
            tT, cT = build(car, sents_for(prof))
            tF, cF = build(car, sents_for(foil))
            pairs.append((encode(tT, cT), encode(tF, cF)))
        return pairs

    results = {"model": model_id, "layer": int(L), "k": k, "base_norm": base_norm,
               "phase_diagram": {}, "contig_vs_scatter": {}}

    for ell in ells:
        prof01 = square(ell)
        pi = np.array([1.0 if l else -1.0 for l in prof01])
        pibar = float(pi.mean())
        pairs = make_pairs(prof01, n_eval)
        base = [margin(t, [], 0) - margin(f, [], 0) for t, f in pairs]
        row = {"profile": prof01, "pi_bar": pibar,
               "base_margin": float(np.mean(base)),
               "base_margin_sem": float(np.std(base, ddof=1) / np.sqrt(len(base)))}

        def run_arm(vec_fn):
            per = {}
            for fr in fracs:
                m = fr * base_norm
                ds = [(margin(t, vec_fn(), m) - margin(f, vec_fn(), m)) - b
                      for (t, f), b in zip(pairs, base)]
                ds = np.array(ds)
                per[round(fr, 3)] = {"mean": float(ds.mean()),
                                     "sem": float(ds.std(ddof=1) / np.sqrt(len(ds)))}
            return per

        full_per = run_arm(lambda: [float(pi[t]) * u for t in range(k)])
        row["full"] = full_per
        best_fr = max(full_per, key=lambda x: full_per[x]["mean"])
        row["full_peak_frac"] = best_fr
        dfull = full_per[best_fr]["mean"]
        row["full_peak"] = full_per[best_fr]

        for W in ws:
            nb = k // W
            mu = np.array([pi[b * W:(b + 1) * W].mean() for b in range(nb)])
            r_cap = float(np.mean(np.abs(mu)))
            r_rms = float(np.sqrt(np.mean(mu ** 2)))
            c_cap = np.sign(mu)
            c_en = mu / (np.sqrt(np.mean(mu ** 2)) + 1e-9)
            for name, cvec, pred in (("block_cap", c_cap, r_cap),
                                     ("block_energy", c_en, r_rms)):
                per = run_arm(lambda cv=cvec: [float(cv[t // W]) * u
                                               for t in range(k)])
                obs = per[best_fr]["mean"]          # same frac as Δ_full
                row[f"W{W}_{name}"] = {
                    "pred_R": pred, "obs": obs, "sem": per[best_fr]["sem"],
                    "obs_R": (obs / dfull) if abs(dfull) > 1e-6 else None,
                    "curve": per}
        results["phase_diagram"][ell] = row
        cells = "  ".join(
            f"W{W}:{row[f'W{W}_block_cap']['obs_R']:+.2f}/{row[f'W{W}_block_cap']['pred_R']:.2f}"
            for W in ws)
        print(f"[ell={ell}] pi_bar={pibar:+.2f} base={row['base_margin']:+.2f} "
              f"full={dfull:+.1f} (frac={best_fr})  obs_R/pred_R: {cells}")

    # ---- contiguous vs scattered at matched coverage (additivity control) ----
    prof01 = square(1)
    pi = np.array([1.0 if l else -1.0 for l in prof01])
    pairs = make_pairs(prof01, n_eval)
    base = [margin(t, [], 0) - margin(f, [], 0) for t, f in pairs]
    for mode in ("contiguous", "scattered"):
        per = {}
        for fr in fracs:
            m = fr * base_norm
            ds = []
            for j, ((t, f), b) in enumerate(zip(pairs, base)):
                if mode == "contiguous":
                    start = (j * cov) % k
                    sel = {(start + i) % k for i in range(cov)}
                else:
                    step = k // cov
                    sel = {(j + i * step) % k for i in range(cov)}
                vecs = [float(pi[i]) * u if i in sel else None for i in range(k)]
                ds.append((margin(t, vecs, m) - margin(f, vecs, m)) - b)
            ds = np.array(ds)
            per[round(fr, 3)] = {"mean": float(ds.mean()),
                                 "sem": float(ds.std(ddof=1) / np.sqrt(len(ds)))}
        best = max(per, key=lambda x: per[x]["mean"])
        results["contig_vs_scatter"][mode] = {"peak_frac": best, **per[best],
                                              "curve": per}
        print(f"[{mode} cov={cov}] peak={per[best]['mean']:+.2f}±{per[best]['sem']:.2f}")

    return results


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-1.5B-Instruct", layer: int = -1, k: int = 12,
         ells: str = "1,2,3,6", ws: str = "1,2,3,4,6,12", n_train: int = 40,
         n_eval: int = 28, fracs: str = "0.2,0.35,0.5", cov: int = 4):
    import json
    res = lsweep.remote(model, layer, k, [int(x) for x in ells.split(",")],
                        [int(x) for x in ws.split(",")], n_train, n_eval,
                        [float(x) for x in fracs.split(",")], cov)
    outdir = ROOT / "results" / "temporal_screen"
    outdir.mkdir(parents=True, exist_ok=True)
    tag = model.split("/")[-1].replace("Qwen2.5-", "qwen").replace("-Instruct", "").lower()
    out = outdir / f"lsweep_{tag}.json"
    out.write_text(json.dumps(res, indent=2))
    print("[saved]", out)
