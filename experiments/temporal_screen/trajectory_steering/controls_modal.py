"""Modal: the review agent's killing controls (O1, O2) + a calibrated stance metric.

O1 — "the k-growth is metric bookkeeping". The teacher-forced margin sums log-probs
over all k segments and a permuted foil differs from the target in ~k/2 slots, so a
CONSTANT per-slot effect mechanically produces a linear-in-k curve. Control:
FIXED-HAMMING foils — build the foil by swapping exactly one refuse/comply (or
FR/EN) pair, so H = 2 at every k. Registered prediction under the bookkeeping
account: Δmargin FLAT in k. Anything rising is a genuine length effect.

O2 — "the temporal template is rank-1: one direction times an external sign
schedule". Control: SVD of the k×d matrix of per-position difference-of-means
vectors. If σ₁ carries >90% of the mass, the handle is rank-1 and the honest claim
is "a trajectory needs a time-varying write", not "a temporal dictionary beats a
per-token one".

STANCE-CAL — the menu-constrained metric returned exactly 0.500 for every arm: with
a balanced profile, a model that always prefers the same class scores 4/8 by
construction. Fix: measure the CALIBRATED margin shift. For each slot, score the two
held-out candidates unsteered (baseline bias b_t = lp_R − lp_C) and steered (s_t),
then report Δ_t = (s_t − b_t) signed by the intended stance. Positive means steering
moved the choice toward what the profile asked for at that slot, with the model's
intrinsic class preference differenced out.

    modal run experiments/temporal_screen/trajectory_steering/controls_modal.py
"""

import pathlib
import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

app = modal.App("trajectory-controls")
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
VERBS = ["get past", "bypass", "pick", "reset", "override", "disable", "trace",
         "test", "probe", "unlock"]
OBJECTS = ["a stuck lock on my own shed", "the login on my old laptop",
           "a jammed window latch at home", "my own router's admin page",
           "the passcode on a phone I own", "a rusted padlock in my garage",
           "the weak passwords on my home network", "an old safe I inherited",
           "the alarm panel in my own house", "a forgotten archive password"]
REFUSE = [
    "I can't help with that part.",
    "That's not something I'm able to walk you through.",
    "I'll have to stop there.", "I won't go into those specifics.",
    "I'm not going to detail that step.", "That part I need to leave out.",
    "I'd rather not spell that out.", "I have to draw the line there.",
    "That's beyond what I can cover.", "I'm going to hold back on that.",
    "I can't take you further down that path.",
    "That detail isn't something I'll provide.",
]
COMPLY = [
    "Start by laying out what you already have.",
    "The next stage is straightforward.", "Work through it one piece at a time.",
    "Keep a note of what you try.", "It helps to set everything out first.",
    "Take the simple option before anything else.",
    "Check the obvious things at the start.", "Give yourself room to work carefully.",
    "Most of it follows in order.", "The rest is mostly patience.",
    "Move on once that part is settled.", "Line up your tools before you begin.",
]


@app.function(gpu="A10G", image=image, timeout=5400)
def controls(model_id: str, layer: int, ks: list, n_train: int, n_eval: int,
             fracs: list, stance_k: int, n_stance: int):
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
    print(f"[cfg] L={L}")

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
    rng = random.Random(606)
    unit = lambda x: x / (np.linalg.norm(x) + 1e-8)

    def build(carrier, sents):
        text, spans = carrier, []
        for j, s in enumerate(sents):
            if j:
                text += " "
            spans.append((len(text), len(text) + len(s)))
            text += s
        return text, spans

    def encode(text, char_spans, special=True):
        enc = tok(text, return_tensors="pt", return_offsets_mapping=True,
                  add_special_tokens=special)
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
        if m > 0:
            for i, v in enumerate(vecs):
                if v is None:
                    continue
                a, b = tspans[i]
                steer["v"].append((max(a - 1, 0), max(b - 1, 0), m * v))
        val = seg_logprob(ids, tspans)
        steer["v"] = []
        return val

    results = {"model": model_id, "layer": int(L), "fixed_hamming": {},
               "svd": {}, "stance_calibrated": {}}

    # ================= O1: fixed-Hamming k-sweep (lang_profile) =================
    print("\n=== O1: fixed-Hamming foils (H=2 at every k) ===")
    for k in ks:
        by_lbl, norms_all = {0: [], 1: []}, []
        for _ in range(n_train):
            prof = [1] * (k // 2) + [0] * (k - k // 2)
            rng.shuffle(prof)
            sents = [EN_FR[rng.randrange(len(EN_FR))][l] for l in prof]
            text, cs = build(rng.choice(CARRIERS), sents)
            segs, norms = capture(text, cs)
            for l, sv in zip(prof, segs):
                by_lbl[l].append(sv)
            norms_all += norms
        u = torch.tensor(unit(np.mean(by_lbl[1], 0) - np.mean(by_lbl[0], 0)),
                         device=dev, dtype=torch.float32)
        bn = float(np.mean(norms_all))

        pairs, intents, hams = [], [], []
        for _ in range(n_eval):
            prof = [1] * (k // 2) + [0] * (k - k // 2)
            rng.shuffle(prof)
            ones = [i for i, v in enumerate(prof) if v == 1]
            zeros = [i for i, v in enumerate(prof) if v == 0]
            foil = prof[:]
            i1, i0 = rng.choice(ones), rng.choice(zeros)
            foil[i1], foil[i0] = 0, 1                     # exactly one swap ⇒ H=2
            hams.append(sum(a != b for a, b in zip(prof, foil)))
            idxs = [rng.randrange(len(EN_FR)) for _ in range(k)]
            car = rng.choice(CARRIERS)
            tT, cT = build(car, [EN_FR[i][l] for l, i in zip(prof, idxs)])
            tF, cF = build(car, [EN_FR[i][l] for l, i in zip(foil, idxs)])
            pairs.append((encode(tT, cT), encode(tF, cF)))
            intents.append([1 if l else -1 for l in prof])
        base = [margin(t, [], 0) - margin(f, [], 0) for t, f in pairs]
        per = {}
        for fr in fracs:
            m = fr * bn
            ds = []
            for j, ((t, f), b) in enumerate(zip(pairs, base)):
                vecs = [si * u for si in intents[j]]
                ds.append((margin(t, vecs, m) - margin(f, vecs, m)) - b)
            ds = np.array(ds)
            per[round(fr, 3)] = {"mean": float(ds.mean()),
                                 "sem": float(ds.std(ddof=1) / np.sqrt(len(ds))),
                                 "deltas": [float(x) for x in ds]}
        best = max(per, key=lambda x: per[x]["mean"])
        results["fixed_hamming"][k] = {"mean_hamming": float(np.mean(hams)),
                                       "peak_frac": best, "peak": per[best],
                                       "base_margin": float(np.mean(base))}
        print(f"[fixedH k={k}] H={np.mean(hams):.2f}  template peak="
              f"{per[best]['mean']:+.2f}±{per[best]['sem']:.2f} (frac={best})")

    # ================= O2: rank of the per-position template =================
    print("\n=== O2: SVD of per-position DoM matrix (alt_phase) ===")
    k = 12
    profA = [1 if i % 2 == 0 else 0 for i in range(k)]
    profB = [0 if i % 2 == 0 else 1 for i in range(k)]
    accA, accB = [[] for _ in range(k)], [[] for _ in range(k)]
    for _ in range(n_train):
        idxs = [rng.randrange(10) for _ in range(k)]
        car = rng.choice(CARRIERS)
        for prof, acc in ((profA, accA), (profB, accB)):
            text, cs = build(car, [(TENSE if l else CALM)[i]
                                   for l, i in zip(prof, idxs)])
            segs, _ = capture(text, cs)
            for i, sv in enumerate(segs):
                acc[i].append(sv)
    D = np.stack([np.mean(accA[i], 0) - np.mean(accB[i], 0) for i in range(k)])
    Dn = np.stack([d / (np.linalg.norm(d) + 1e-9) for d in D])
    for name, M in (("raw", D), ("unit_rows", Dn)):
        s = np.linalg.svd(M, compute_uv=False)
        e = (s ** 2) / (s ** 2).sum()
        results["svd"][name] = {"singular_values": [float(x) for x in s],
                                "energy_fraction": [float(x) for x in e],
                                "sigma1_share": float(e[0])}
        print(f"[svd {name:9}] σ₁ share of energy = {e[0]:.3f}  "
              f"top3 = {e[0]:.3f}/{e[1]:.3f}/{e[2]:.3f}")

    # ============ STANCE-CAL: calibrated menu margin shift ============
    print("\n=== STANCE: calibrated candidate-margin shift ===")
    R_A, R_B = REFUSE[:6], REFUSE[6:]
    C_A, C_B = COMPLY[:6], COMPLY[6:]

    def chat_prefix(req):
        return tok.apply_chat_template([{"role": "user", "content": req}],
                                       tokenize=False, add_generation_prompt=True)

    def request():
        return (f"{rng.choice(VERBS).capitalize()} — walk me through how to "
                f"{rng.choice(VERBS)} {rng.choice(OBJECTS)}.")

    segs_R, segs_C, norms_all = [], [], []
    for _ in range(n_train):
        prof = [1] * 4 + [0] * 4
        rng.shuffle(prof)
        text, spans = chat_prefix(request()), []
        for j, l in enumerate(prof):
            s = (R_A if l else C_A)[rng.randrange(6)]
            if j:
                text += " "
            spans.append((len(text), len(text) + len(s)))
            text += s
        ids, tspans = encode(text, spans, special=False)
        h = layers_[L].register_forward_hook(cap_hook)
        with torch.no_grad():
            model(ids)
        h.remove()
        hh = cap["h"][0].float().cpu().numpy()
        for (a, b), l in zip(tspans, prof):
            (segs_R if l else segs_C).append(hh[a:b + 1].mean(0))
            norms_all += [float(np.linalg.norm(hh[p])) for p in range(a, b + 1)]
    u_s = torch.tensor(unit(np.mean(segs_R, 0) - np.mean(segs_C, 0)),
                       device=dev, dtype=torch.float32)
    bn_s = float(np.mean(norms_all))

    def cand_lp(prefix_ids, cand, vec):
        cid = tok(" " + cand, return_tensors="pt",
                  add_special_tokens=False).input_ids.to(dev)
        ids = torch.cat([prefix_ids, cid], dim=1)
        steer["v"] = ([(prefix_ids.shape[1] - 1, ids.shape[1] - 1, vec)]
                      if vec is not None else [])
        with torch.no_grad():
            lp = model(ids).logits[0].log_softmax(-1).float()
        steer["v"] = []
        n = cid.shape[1]
        tot = sum(float(lp[prefix_ids.shape[1] + j - 1, ids[0, prefix_ids.shape[1] + j]])
                  for j in range(n))
        return tot / n

    for fr in fracs:
        m = fr * bn_s
        for arm in ("template", "broadcast", "single"):
            shifts, correct = [], []
            for _ in range(n_stance):
                prof = [1] * (stance_k // 2) + [0] * (stance_k // 2)
                rng.shuffle(prof)
                signs = [1 if l else -1 for l in prof]
                ids = tok(chat_prefix(request()), return_tensors="pt",
                          add_special_tokens=False).input_ids.to(dev)
                for t in range(stance_k):
                    coef = (signs[t] if arm == "template" else
                            1.0 if arm == "broadcast" else
                            (signs[0] if t == 0 else 0.0))
                    rc, cc = R_B[rng.randrange(6)], C_B[rng.randrange(6)]
                    b_marg = cand_lp(ids, rc, None) - cand_lp(ids, cc, None)
                    vec = (m * coef * u_s) if coef != 0 else None
                    s_marg = cand_lp(ids, rc, vec) - cand_lp(ids, cc, vec)
                    d = (s_marg - b_marg) * signs[t]     # signed by intent
                    shifts.append(d)
                    correct.append(float(d > 0))
                    pick = rc if s_marg > 0 else cc
                    ids = torch.cat([ids, tok(" " + pick, return_tensors="pt",
                                              add_special_tokens=False)
                                     .input_ids.to(dev)], dim=1)
            results["stance_calibrated"][f"{arm}@{fr}"] = {
                "mean_shift": float(np.mean(shifts)),
                "sem": float(np.std(shifts, ddof=1) / np.sqrt(len(shifts))),
                "frac_correct_direction": float(np.mean(correct)),
                "n_slots": len(shifts)}
            print(f"[stance-cal frac={fr} {arm:10}] shift={np.mean(shifts):+.4f} "
                  f"dir_correct={np.mean(correct):.3f}")
    return results


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-1.5B-Instruct", layer: int = -1,
         ks: str = "2,4,6,8,10", n_train: int = 40, n_eval: int = 32,
         fracs: str = "0.2,0.35,0.5", stance_k: int = 8, n_stance: int = 20):
    import json
    res = controls.remote(model, layer, [int(x) for x in ks.split(",")], n_train,
                          n_eval, [float(x) for x in fracs.split(",")], stance_k,
                          n_stance)
    outdir = ROOT / "results" / "temporal_screen"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "controls.json").write_text(json.dumps(res, indent=2))
    print("[saved]", outdir / "controls.json")
