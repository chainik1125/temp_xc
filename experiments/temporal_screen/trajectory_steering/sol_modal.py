"""Modal: signs-of-life for the four trajectory-steering candidates — the language
renderings of the clock designed to survive mode-dominance (see the revised
recommendation in docs/dmitry/reviewer_responses/semisynthetic_language_tasks.md).

Four tasks, one harness. Each yields (target text, multiset-matched foil text,
per-segment intent):
  lang_profile : k=6 sentences, random balanced EN/FR profile; foil = permuted profile
  int_profile  : k=6 sentences, random balanced calm/tense profile; foil = permuted
  mirror       : k=5 intensity path 1-2-3-2-1 (tri) vs 3-2-1-2-3 (inv); same multiset
  alt_phase    : k=6 alternating tense/calm, phase A vs phase B (the clock at omega=pi)

Foils are permutations of the target profile with identical multisets, so no bag/
mode statistic separates target from foil — a broadcast (DC) write should NOT move
the teacher-forced margin  margin = logP(target) - logP(foil);  only a per-segment
schedule should. This is the clock's marginal-matching rendered in language.

Arms (writes at the predicting positions of each segment's tokens, i.e. the segment
token span shifted -1; magnitude = frac * mean segment-token residual norm):
  template : per-segment intent — ±u per the profile (A-tasks) or per-position DoM
             t_dir[i] (B-tasks). The windowed/TXC handle.
  broadcast: the strongest DC direction (u_lang / u_int) at every segment — the
             standard per-token-SAE steering recipe. Predicted ~0 on the margin.
  single   : the template's write at one segment only. Predicted small.

Metric: dmargin = [lp(T)-lp(F)]_steered - [lp(T)-lp(F)]_base (diff-in-diff; content
and length asymmetries cancel). Signs of life = template >> broadcast, single small.
Diagnostics: baseline margin (~0 unless the model has a schema preference, e.g.
rise-fall arcs for mirror), and for B-tasks cos(t_dir[i], u_dc) + the DC-content
ratio of the per-position diffs.

    modal run experiments/temporal_screen/trajectory_steering/sol_modal.py
"""

import pathlib
import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

app = modal.App("trajectory-sol")
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
MILD = [
    "A door creaked somewhere upstairs.",
    "He paused, unsure of the sound.",
    "The phone rang twice, then stopped.",
    "She noticed the gate standing open.",
    "A cold draft slipped through the hall.",
    "The dog lifted its head, listening.",
    "Footsteps echoed faintly outside.",
    "He checked the lock a second time.",
    "The lights dimmed for a moment.",
    "She felt a vague unease settle in.",
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
LEVELS = {1: CALM, 2: MILD, 3: TENSE}
CARRIERS = ["Journal entry.\n", "From the notebook:\n", "Draft passage.\n",
            "Field notes.\n", "Evening record.\n", "From chapter twelve:\n"]


@app.function(gpu="A10G", image=image, timeout=3600)
def sol(model_id: str, layer: int, n_train: int, n_eval: int, fracs: list):
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
    print(f"[cfg] {len(layers_)} layers, L={L}")

    cap, steer = {}, {"v": []}

    def cap_hook(_m, _i, out):
        cap["h"] = (out[0] if isinstance(out, tuple) else out).detach()

    def steer_hook(_m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        for a, b, vec in steer["v"]:
            hs[:, a:b + 1, :] = hs[:, a:b + 1, :] + vec.to(hs.dtype)
        return (hs, *out[1:]) if isinstance(out, tuple) else hs

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
        norms = [float(np.linalg.norm(hh[p])) for a, b in tspans for p in range(a, b + 1)]
        return segs, norms

    unit = lambda x: x / (np.linalg.norm(x) + 1e-8)

    def margin(pair, vecs_by_seg, m):
        """lp(text) with the intent schedule applied at this text's segment slots."""
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

    def eval_arms(pairs, arm_vecs, base_norm):
        """pairs: list of (pairT, pairF). arm_vecs: {arm: vecs_by_seg}.
        Returns base margin mean + curves of mean dmargin."""
        base = [margin(t, [], 0) - margin(f, [], 0) for t, f in pairs]
        curves = {}
        for arm, vecs in arm_vecs.items():
            curves[arm] = {}
            for fr in fracs:
                m = fr * base_norm
                d = [(margin(t, vecs, m) - margin(f, vecs, m)) - b
                     for (t, f), b in zip(pairs, base)]
                curves[arm][round(fr, 3)] = float(np.mean(d))
        return float(np.mean(base)), float(np.mean([b > 0 for b in base])), curves

    results = {"model": model_id, "layer": int(L), "tasks": {}}
    hook = layers_[L].register_forward_hook(steer_hook)

    # ---------------- pipeline A: random binary profiles (lang, intensity) --------
    def taskA(name, sent_fn, bank_len, k, seed):
        rng = random.Random(seed)
        def make_prof():
            prof = [1] * (k // 2) + [0] * (k - k // 2)
            rng.shuffle(prof)
            return prof
        # train: pooled DoM
        by_lbl, norms_all = {0: [], 1: []}, []
        hook.remove()  # capture phase must not steer
        for _ in range(n_train):
            prof = make_prof()
            sents = [sent_fn(l, rng.randrange(bank_len)) for l in prof]
            text, cs = build_text(rng.choice(CARRIERS), sents)
            segs, norms = capture_segs(text, cs)
            for l, sv in zip(prof, segs):
                by_lbl[l].append(sv)
            norms_all += norms
        u = unit(np.mean(by_lbl[1], 0) - np.mean(by_lbl[0], 0))
        base_norm = float(np.mean(norms_all))
        u_t = torch.tensor(u, device=dev, dtype=torch.float32)
        # eval pairs
        pairs, intents = [], []
        for _ in range(n_eval):
            prof = make_prof()
            foil = prof[:]
            for _ in range(20):
                rng.shuffle(foil)
                if foil != prof:
                    break
            idxs = [rng.randrange(bank_len) for _ in range(k)]
            car = rng.choice(CARRIERS)
            tT, cT = build_text(car, [sent_fn(l, i) for l, i in zip(prof, idxs)])
            tF, cF = build_text(car, [sent_fn(l, i) for l, i in zip(foil, idxs)])
            pairs.append((encode(tT, cT), encode(tF, cF)))
            intents.append([1 if l else -1 for l in prof])
        layers_[L]._forward_hooks.clear()
        h2 = layers_[L].register_forward_hook(steer_hook)
        # arms are per-pair (intent differs) -> fold intent into vecs at call time
        base_ms, share_pos, curves = [], [], {"template": {}, "broadcast": {}, "single": {}}
        base = [margin(t, [], 0) - margin(f, [], 0) for t, f in pairs]
        for arm in curves:
            for fr in fracs:
                m = fr * base_norm
                ds = []
                for (t, f), s, b in zip(pairs, intents, base):
                    if arm == "template":
                        vecs = [si * u_t for si in s]
                    elif arm == "broadcast":
                        vecs = [u_t for _ in s]
                    else:
                        vecs = [s[0] * u_t] + [None] * (k - 1)
                    ds.append((margin(t, vecs, m) - margin(f, vecs, m)) - b)
                curves[arm][round(fr, 3)] = float(np.mean(ds))
        h2.remove()
        layers_[L].register_forward_hook(steer_hook)
        res = {"k": k, "base_norm": base_norm,
               "base_margin": float(np.mean(base)),
               "base_margin_pos_share": float(np.mean([b > 0 for b in base])),
               "curves": curves,
               "headline": {a: max(c.values()) for a, c in curves.items()}}
        results["tasks"][name] = res
        print(f"[{name}] base_margin={res['base_margin']:+.2f}  "
              + "  ".join(f"{a}={res['headline'][a]:+.2f}" for a in curves))

    # ---------------- pipeline B: fixed profile pairs (mirror, alt) ---------------
    def taskB(name, profA, profB, seed):
        k = len(profA)
        rng = random.Random(seed)
        accA, accB = [[] for _ in range(k)], [[] for _ in range(k)]
        lvl_segs, norms_all = {1: [], 3: []}, []
        layers_[L]._forward_hooks.clear()
        for _ in range(n_train):
            idxs = [rng.randrange(10) for _ in range(k)]
            car = rng.choice(CARRIERS)
            for prof, acc in ((profA, accA), (profB, accB)):
                text, cs = build_text(car, [LEVELS[l][i] for l, i in zip(prof, idxs)])
                segs, norms = capture_segs(text, cs)
                for i, (l, sv) in enumerate(zip(prof, segs)):
                    acc[i].append(sv)
                    if l in lvl_segs:
                        lvl_segs[l].append(sv)
                norms_all += norms
        diff = np.stack([np.mean(accA[i], 0) - np.mean(accB[i], 0) for i in range(k)])
        t_dir = np.stack([unit(diff[i]) for i in range(k)])
        u_dc = unit(np.mean(lvl_segs[3], 0) - np.mean(lvl_segs[1], 0))
        dc_ratio = float(np.linalg.norm(diff.mean(0)) /
                         (np.mean([np.linalg.norm(diff[i]) for i in range(k)]) + 1e-8))
        cos_dc = [float(t_dir[i] @ u_dc) for i in range(k)]
        base_norm = float(np.mean(norms_all))
        i_star = int(np.argmax([np.linalg.norm(diff[i]) for i in range(k)]))
        t_dir_t = torch.tensor(t_dir, device=dev, dtype=torch.float32)
        u_dc_t = torch.tensor(u_dc, device=dev, dtype=torch.float32)
        pairs = []
        for _ in range(n_eval):
            idxs = [rng.randrange(10) for _ in range(k)]
            car = rng.choice(CARRIERS)
            tT, cT = build_text(car, [LEVELS[l][i] for l, i in zip(profA, idxs)])
            tF, cF = build_text(car, [LEVELS[l][i] for l, i in zip(profB, idxs)])
            pairs.append((encode(tT, cT), encode(tF, cF)))
        layers_[L].register_forward_hook(steer_hook)
        arm_vecs = {
            "template": [t_dir_t[i] for i in range(k)],
            "broadcast": [u_dc_t for _ in range(k)],
            "single": [t_dir_t[i] if i == i_star else None for i in range(k)],
        }
        bm, bpos, curves = eval_arms(pairs, arm_vecs, base_norm)
        res = {"k": k, "base_norm": base_norm, "base_margin": bm,
               "base_margin_pos_share": bpos, "dc_ratio": dc_ratio,
               "cos_t_dir_vs_udc": cos_dc, "i_star": i_star, "curves": curves,
               "headline": {a: max(c.values()) for a, c in curves.items()}}
        results["tasks"][name] = res
        print(f"[{name}] base_margin={bm:+.2f} dc_ratio={dc_ratio:.2f} "
              f"cos_dc={[round(c, 2) for c in cos_dc]}  "
              + "  ".join(f"{a}={res['headline'][a]:+.2f}" for a in curves))

    taskA("lang_profile", lambda l, i: EN_FR[i][l], len(EN_FR), 6, seed=11)
    taskA("int_profile", lambda l, i: (TENSE if l else CALM)[i], 10, 6, seed=22)
    taskB("mirror", [1, 2, 3, 2, 1], [3, 2, 1, 2, 3], seed=33)
    taskB("alt_phase", [3, 1, 3, 1, 3, 1], [1, 3, 1, 3, 1, 3], seed=44)

    layers_[L]._forward_hooks.clear()
    return results


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-1.5B-Instruct", layer: int = -1,
         n_train: int = 32, n_eval: int = 24, fracs: str = "0.01,0.03,0.08,0.2"):
    import json
    res = sol.remote(model, layer, n_train, n_eval,
                     [float(x) for x in fracs.split(",")])
    print("RESULT:", json.dumps(res, indent=2))
    outdir = ROOT / "results" / "temporal_screen"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "trajectory_sol.json").write_text(json.dumps(res, indent=2))
    print("[saved]", outdir / "trajectory_sol.json")
