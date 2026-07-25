"""Modal: the W-sweep — steering performance vs WINDOW SIZE at fixed knob budget.

Operationalization ("window size" = span of one steering knob): a window-W handle
covers contiguous blocks of W segments; ONE KNOB writes one block with the correct
per-segment schedule inside its span. Budget m knobs ⇒ coverage min(m·W, k) segments.
W=1 is a per-token (SAE-like) latent: one knob controls one segment. W=k is one
temporal latent writing the whole trajectory. The claim under test: control bandwidth
per knob grows with W — Δmargin(W; m) ≈ Δ_full · min(mW, k)/k, i.e. performance
improves with window size, saturating at W = k/m.

Block placement is ROTATED across eval pairs (pair j uses m consecutive blocks
starting at block j mod (k/W), wrapping) so positional heterogeneity washes out.

Tasks: lang_profile (random balanced EN/FR, k=12) and alt_phase (tense/calm
alternation, k=12), teacher-forced margin vs multiset-matched foil, as in
full_modal.py. Reference arms: full template (upper envelope) and broadcast (DC
floor). fracs at the known peak region; peak-over-frac per condition; per-pair
deltas kept for SEM.

    modal run experiments/temporal_screen/trajectory_steering/wsweep_modal.py
"""

import pathlib
import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

app = modal.App("trajectory-wsweep")
image = modal.Image.debian_slim().pip_install("torch", "transformers", "accelerate", "numpy")

EN_FR = [
    ("The cat sleeps on the sofa.", "Le chat dort sur le canapé."),
    ("The market opens at nine.", "Le marché ouvre à neuf heures."),
    ("She reads the letter slowly.", "Elle lit la lettre lentement."),
    ("The train arrives this evening.", "Le train arrive ce soir."),
    ("He buys bread every morning.", "Il achète du pain chaque matin."),
    ("The garden smells of roses.", "Le jardin sent la rose."),
    ("They walk along the river.", "Ils marchent le long de la rivière."),
    ("The teacher writes on the board.", "Le professeur écrit au tableau."),
    ("The soup is still warm.", "La soupe est encore chaude."),
    ("We wait under the old clock.", "Nous attendons sous la vieille horloge."),
    ("The children play in the yard.", "Les enfants jouent dans la cour."),
    ("The lamp flickers at night.", "La lampe vacille la nuit."),
]
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
def wsweep(model_id: str, layer: int, k: int, ws: list, ms: list,
           n_train: int, n_eval: int, fracs: list):
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

    def build_text(carrier, sents):
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

    def capture_segs(text, char_spans):
        ids, tspans = encode(text, char_spans)
        h = layers_[L].register_forward_hook(cap_hook)
        with torch.no_grad():
            model(ids)
        h.remove()
        hh = cap["h"][0].float().cpu().numpy()
        segs = [hh[a:b + 1].mean(0) for a, b in tspans]
        norms = [float(np.linalg.norm(hh[p])) for a, b in tspans
                 for p in range(a, b + 1)]
        return segs, norms

    unit = lambda x: x / (np.linalg.norm(x) + 1e-8)

    def margin(pair, vecs_by_seg, m):
        ids, tspans = pair
        steer["v"] = []
        if m > 0:
            for i, v in enumerate(vecs_by_seg):
                if v is None:
                    continue
                a, b = tspans[i]
                steer["v"].append((max(a - 1, 0), max(b - 1, 0), m * v))
        val = seg_logprob(ids, tspans)
        steer["v"] = []
        return val

    def covered(W, mknobs, rot):
        nblocks = k // W
        blocks = [(rot + b) % nblocks for b in range(min(mknobs, nblocks))]
        segs = set()
        for bl in blocks:
            segs.update(range(bl * W, (bl + 1) * W))
        return segs

    def run_conditions(pairs, tmpl_vec, dc_vec, base_norm):
        """tmpl_vec(pair_idx, seg) -> tensor (the correct write for that segment).
        Conditions: (W, m) grid + broadcast + full. Returns dict."""
        base = [margin(t, [], 0) - margin(f, [], 0) for t, f in pairs]
        out = {"base_margin": float(np.mean(base))}
        conds = [("full", None, None), ("broadcast", None, None)] + \
                [(f"W{W}_m{m}", W, m) for W in ws for m in ms
                 if not (W == k and m > 1)]
        for name, W, mk in conds:
            per_frac = {}
            for fr in fracs:
                mm = fr * base_norm
                ds = []
                for j, ((t, f), b) in enumerate(zip(pairs, base)):
                    if name == "full":
                        vecs = [tmpl_vec(j, i) for i in range(k)]
                    elif name == "broadcast":
                        vecs = [dc_vec] * k
                    else:
                        cov = covered(W, mk, j % (k // W))
                        vecs = [tmpl_vec(j, i) if i in cov else None
                                for i in range(k)]
                    ds.append((margin(t, vecs, mm) - margin(f, vecs, mm)) - b)
                ds = np.array(ds)
                per_frac[round(fr, 3)] = {
                    "mean": float(ds.mean()),
                    "sem": float(ds.std(ddof=1) / np.sqrt(len(ds)))}
            best = max(per_frac, key=lambda x: per_frac[x]["mean"])
            out[name] = {"peak_frac": best, **per_frac[best],
                         "coverage": (k if name == "full" else
                                      0 if name == "broadcast" else
                                      len(covered(W, mk, 0)))}
            print(f"  {name:12} peak={per_frac[best]['mean']:+8.2f}"
                  f"±{per_frac[best]['sem']:.2f}  (frac={best})")
        return out

    results = {"model": model_id, "layer": int(L), "k": k}
    rng = random.Random(77)

    # ------------------------------ lang_profile ------------------------------
    print("[task] lang_profile")
    by_lbl, norms_all = {0: [], 1: []}, []
    for _ in range(n_train):
        prof = [1] * (k // 2) + [0] * (k - k // 2)
        rng.shuffle(prof)
        sents = [EN_FR[rng.randrange(len(EN_FR))][l] for l in prof]
        text, cs = build_text(rng.choice(CARRIERS), sents)
        segs, norms = capture_segs(text, cs)
        for l, sv in zip(prof, segs):
            by_lbl[l].append(sv)
        norms_all += norms
    u = torch.tensor(unit(np.mean(by_lbl[1], 0) - np.mean(by_lbl[0], 0)),
                     device=dev, dtype=torch.float32)
    bn = float(np.mean(norms_all))
    pairs, intents = [], []
    for _ in range(n_eval):
        prof = [1] * (k // 2) + [0] * (k - k // 2)
        rng.shuffle(prof)
        foil = prof[:]
        for _ in range(20):
            rng.shuffle(foil)
            if foil != prof:
                break
        idxs = [rng.randrange(len(EN_FR)) for _ in range(k)]
        car = rng.choice(CARRIERS)
        tT, cT = build_text(car, [EN_FR[i][l] for l, i in zip(prof, idxs)])
        tF, cF = build_text(car, [EN_FR[i][l] for l, i in zip(foil, idxs)])
        pairs.append((encode(tT, cT), encode(tF, cF)))
        intents.append([1 if l else -1 for l in prof])
    results["lang_profile"] = run_conditions(
        pairs, lambda j, i: intents[j][i] * u, u, bn)
    results["lang_profile"]["base_norm"] = bn

    # ------------------------------- alt_phase --------------------------------
    print("[task] alt_phase")
    profA = [3 if i % 2 == 0 else 1 for i in range(k)]
    profB = [1 if i % 2 == 0 else 3 for i in range(k)]
    bank = {1: CALM, 3: TENSE}
    accA, accB = [[] for _ in range(k)], [[] for _ in range(k)]
    lvl_segs, norms_all = {1: [], 3: []}, []
    for _ in range(n_train):
        idxs = [rng.randrange(10) for _ in range(k)]
        car = rng.choice(CARRIERS)
        for prof, acc in ((profA, accA), (profB, accB)):
            text, cs = build_text(car, [bank[l][i] for l, i in zip(prof, idxs)])
            segs, norms = capture_segs(text, cs)
            for i, (l, sv) in enumerate(zip(prof, segs)):
                acc[i].append(sv)
                lvl_segs[l].append(sv)
            norms_all += norms
    diff = np.stack([np.mean(accA[i], 0) - np.mean(accB[i], 0) for i in range(k)])
    t_dir = torch.tensor(np.stack([unit(diff[i]) for i in range(k)]),
                         device=dev, dtype=torch.float32)
    u_dc = torch.tensor(unit(np.mean(lvl_segs[3], 0) - np.mean(lvl_segs[1], 0)),
                        device=dev, dtype=torch.float32)
    bn = float(np.mean(norms_all))
    pairs = []
    for _ in range(n_eval):
        idxs = [rng.randrange(10) for _ in range(k)]
        car = rng.choice(CARRIERS)
        tT, cT = build_text(car, [bank[l][i] for l, i in zip(profA, idxs)])
        tF, cF = build_text(car, [bank[l][i] for l, i in zip(profB, idxs)])
        pairs.append((encode(tT, cT), encode(tF, cF)))
    results["alt_phase"] = run_conditions(pairs, lambda j, i: t_dir[i], u_dc, bn)
    results["alt_phase"]["base_norm"] = bn

    return results


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-1.5B-Instruct", layer: int = -1, k: int = 12,
         ws: str = "1,2,3,4,6,12", ms: str = "1,2", n_train: int = 40,
         n_eval: int = 32, fracs: str = "0.2,0.35,0.5"):
    import json
    res = wsweep.remote(model, layer, k, [int(x) for x in ws.split(",")],
                        [int(x) for x in ms.split(",")], n_train, n_eval,
                        [float(x) for x in fracs.split(",")])
    print("RESULT:", json.dumps(res, indent=2)[:3000])
    outdir = ROOT / "results" / "temporal_screen"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "wsweep.json").write_text(json.dumps(res, indent=2))
    print("[saved]", outdir / "wsweep.json")
