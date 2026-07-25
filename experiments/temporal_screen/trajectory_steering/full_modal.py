"""Modal: FULL trajectory-steering experiment — the language counterpart of the clock.

Follows the signs-of-life pass (sol_modal.py, all 4 tasks positive). Two tasks earned
the full run:
  alt_phase    : tense/calm alternation, phase A vs phase B (the clock at omega=pi;
                 cleanest dose-response in SOL)
  lang_profile : EN/FR sentence-language per random balanced profile (biggest effect;
                 upgrades to an OBJECTIVE generation judge — language-ID per sentence)

Part 1 — teacher-forced k-sweep (k in {2,4,6,8,10}), multiset-matched foils, extended
frac grid (SOL curves were still rising at 0.2), per-pair deltas kept for SEM.
Predictions: template ~flat in k; broadcast ~0 or negative (DC write cannot move a
matched-multiset margin except against you); single decays ~1/k.

Part 2 — steered GENERATION on lang_profile at k=6: greedy decode from a bare carrier
prefix, steering applied at every generated token according to the *current sentence
index* (segment counter advances on '.'), then each produced sentence is language-ID'd
by a marker-word classifier. Metric: per-slot accuracy of generated language vs the
intended profile. Broadcast(French) is pinned near the profile base rate (0.5);
template should track the profile.

    modal run experiments/temporal_screen/trajectory_steering/full_modal.py
"""

import pathlib
import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

app = modal.App("trajectory-full")
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

FR_CHARS = "éèêàâçùûôîïëüœ"
FR_WORDS = {"le", "la", "les", "un", "une", "des", "du", "est", "dans", "et", "il",
            "elle", "je", "nous", "vous", "sur", "avec", "pour", "au", "aux", "ce",
            "cette", "se", "ne", "pas", "son", "sa", "ses", "qui", "que", "chaque",
            "sous", "encore", "matin", "soir", "nuit"}
EN_WORDS = {"the", "a", "an", "is", "are", "was", "in", "on", "and", "he", "she",
            "it", "they", "we", "of", "to", "with", "for", "at", "this", "that",
            "his", "her", "its", "not", "by", "every", "still", "old", "night"}


@app.function(gpu="A10G", image=image, timeout=3600)
def full(model_id: str, layer: int, ks: list, n_train: int, n_eval: int,
         fracs: list, gen_k: int, n_gen: int, gen_fracs: list):
    import random
    import re
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
    print(f"[cfg] {len(layers_)} layers, L={L}")

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

    layers_[L].register_forward_hook(steer_hook)   # no-op while steer["v"] == []

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

    def sweep(pairs, arm_vecs_fn, base_norm):
        """arm_vecs_fn(arm, pair_idx) -> vecs_by_seg. Returns curves with mean+sem."""
        base = [margin(t, [], 0) - margin(f, [], 0) for t, f in pairs]
        curves = {}
        for arm in ("template", "broadcast", "single"):
            curves[arm] = {}
            for fr in fracs:
                m = fr * base_norm
                ds = []
                for j, ((t, f), b) in enumerate(zip(pairs, base)):
                    vecs = arm_vecs_fn(arm, j)
                    ds.append((margin(t, vecs, m) - margin(f, vecs, m)) - b)
                ds = np.array(ds)
                curves[arm][round(fr, 3)] = {
                    "mean": float(ds.mean()),
                    "sem": float(ds.std(ddof=1) / np.sqrt(len(ds))),
                }
            best = max(curves[arm], key=lambda fr: curves[arm][fr]["mean"])
            curves[arm]["peak"] = {"frac": best, **curves[arm][best]}
        return float(np.mean(base)), curves

    results = {"model": model_id, "layer": int(L),
               "lang_profile": {}, "alt_phase": {}, "generation": {}}

    # ------------------------- lang_profile k-sweep ---------------------------
    def lang_k(k, seed):
        rng = random.Random(seed)
        def make_prof():
            prof = [1] * (k // 2) + [0] * (k - k // 2)
            rng.shuffle(prof)
            return prof
        by_lbl, norms_all = {0: [], 1: []}, []
        for _ in range(n_train):
            prof = make_prof()
            sents = [EN_FR[rng.randrange(len(EN_FR))][l] for l in prof]
            text, cs = build_text(rng.choice(CARRIERS), sents)
            segs, norms = capture_segs(text, cs)
            for l, sv in zip(prof, segs):
                by_lbl[l].append(sv)
            norms_all += norms
        u = torch.tensor(unit(np.mean(by_lbl[1], 0) - np.mean(by_lbl[0], 0)),
                         device=dev, dtype=torch.float32)
        base_norm = float(np.mean(norms_all))
        pairs, intents = [], []
        for _ in range(n_eval):
            prof = make_prof()
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
        def arm_vecs(arm, j):
            s = intents[j]
            if arm == "template":
                return [si * u for si in s]
            if arm == "broadcast":
                return [u] * k
            return [s[0] * u] + [None] * (k - 1)
        bm, curves = sweep(pairs, arm_vecs, base_norm)
        results["lang_profile"][k] = {"base_margin": bm, "base_norm": base_norm,
                                      "curves": curves}
        print(f"[lang k={k}] base={bm:+.2f}  "
              + "  ".join(f"{a}={curves[a]['peak']['mean']:+.2f}±{curves[a]['peak']['sem']:.2f}"
                          for a in ("template", "broadcast", "single")))
        return u, base_norm

    # ------------------------- alt_phase k-sweep ------------------------------
    def alt_k(k, seed):
        rng = random.Random(seed)
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
        base_norm = float(np.mean(norms_all))
        i_star = int(np.argmax([np.linalg.norm(diff[i]) for i in range(k)]))
        pairs = []
        for _ in range(n_eval):
            idxs = [rng.randrange(10) for _ in range(k)]
            car = rng.choice(CARRIERS)
            tT, cT = build_text(car, [bank[l][i] for l, i in zip(profA, idxs)])
            tF, cF = build_text(car, [bank[l][i] for l, i in zip(profB, idxs)])
            pairs.append((encode(tT, cT), encode(tF, cF)))
        def arm_vecs(arm, _j):
            if arm == "template":
                return [t_dir[i] for i in range(k)]
            if arm == "broadcast":
                return [u_dc] * k
            return [t_dir[i] if i == i_star else None for i in range(k)]
        bm, curves = sweep(pairs, arm_vecs, base_norm)
        results["alt_phase"][k] = {"base_margin": bm, "base_norm": base_norm,
                                   "i_star": i_star, "curves": curves}
        print(f"[alt  k={k}] base={bm:+.2f}  "
              + "  ".join(f"{a}={curves[a]['peak']['mean']:+.2f}±{curves[a]['peak']['sem']:.2f}"
                          for a in ("template", "broadcast", "single")))

    # ------------------------- generation eval (lang, k=gen_k) ----------------
    def classify_lang(sent):
        sl = sent.lower()
        fr = 2 * sum(ch in FR_CHARS for ch in sl)
        words = re.findall(r"[a-zàâçéèêëîïôûùüÿœ']+", sl)
        fr += sum(w in FR_WORDS for w in words)
        en = sum(w in EN_WORDS for w in words)
        return 1 if fr > en else 0

    def generate_steered(prefix, coefs, u, m, max_sent, max_tokens=220):
        ids = tok(prefix, return_tensors="pt").input_ids.to(dev)
        seg = {"i": 0}
        def gen_hook(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            if m > 0 and seg["i"] < len(coefs) and coefs[seg["i"]] != 0:
                hs[:, -1:, :] = hs[:, -1:, :] + (m * coefs[seg["i"]] * u).to(hs.dtype)
            return (hs, *out[1:]) if isinstance(out, tuple) else hs
        h = layers_[L].register_forward_hook(gen_hook)
        past, cur, out_ids = None, ids, []
        try:
            for _ in range(max_tokens):
                with torch.no_grad():
                    o = model(cur, past_key_values=past, use_cache=True)
                past = o.past_key_values
                nxt = int(o.logits[0, -1].argmax())
                if nxt == tok.eos_token_id:
                    break
                out_ids.append(nxt)
                if "." in tok.decode([nxt]):
                    seg["i"] += 1
                    if seg["i"] >= max_sent:
                        break
                cur = torch.tensor([[nxt]], device=dev)
        finally:
            h.remove()
        return tok.decode(out_ids)

    def gen_eval(k, u, base_norm, seed):
        rng = random.Random(seed)
        out = {}
        for fr in gen_fracs:
            m = fr * base_norm
            for arm in ("template", "broadcast", "single"):
                accs, samples = [], []
                for _ in range(n_gen):
                    prof = [1] * (k // 2) + [0] * (k - k // 2)
                    rng.shuffle(prof)
                    s = [1 if l else -1 for l in prof]
                    coefs = {"template": s, "broadcast": [1] * k,
                             "single": [s[0]] + [0] * (k - 1)}[arm]
                    text = generate_steered(rng.choice(CARRIERS), coefs, u, m, k)
                    sents = [x.strip() for x in text.split(".") if x.strip()][:k]
                    got = [classify_lang(x) for x in sents]
                    got += [1 - p for p in prof[len(got):]]      # missing = wrong
                    accs.append(float(np.mean([g == p for g, p in zip(got, prof)])))
                    if len(samples) < 2:
                        samples.append({"profile": prof, "text": text})
                out[f"{arm}@{fr}"] = {"acc_mean": float(np.mean(accs)),
                                      "acc_sem": float(np.std(accs, ddof=1) /
                                                       np.sqrt(len(accs))),
                                      "samples": samples}
                print(f"[gen k={k} frac={fr}] {arm}: acc={np.mean(accs):.3f}")
        results["generation"] = {"k": k, "fracs": gen_fracs, "n": n_gen, "arms": out}

    u_by_k = {}
    for k in ks:
        u, bn = lang_k(k, seed=100 + k)
        u_by_k[k] = (u, bn)
        alt_k(k, seed=200 + k)
    if gen_k in u_by_k:
        gen_eval(gen_k, *u_by_k[gen_k], seed=999)

    steer["v"] = []
    return results


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-1.5B-Instruct", layer: int = -1,
         ks: str = "2,4,6,8,10", n_train: int = 40, n_eval: int = 32,
         fracs: str = "0.02,0.05,0.1,0.2,0.35,0.5",
         gen_k: int = 6, n_gen: int = 16, gen_fracs: str = "0.2,0.35"):
    import json
    res = full.remote(model, layer, [int(x) for x in ks.split(",")], n_train, n_eval,
                      [float(x) for x in fracs.split(",")], gen_k, n_gen,
                      [float(x) for x in gen_fracs.split(",")])
    print("RESULT:", json.dumps(res, indent=2)[:4000])
    outdir = ROOT / "results" / "temporal_screen"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "trajectory_full.json").write_text(json.dumps(res, indent=2))
    print("[saved]", outdir / "trajectory_full.json")
