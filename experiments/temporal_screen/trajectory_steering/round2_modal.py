"""Modal: round-2 controls demanded by the audit. Four decisive tests in one job.

A. PHASE SWEEP — buy real measurements for the phase diagram.
   With profile phase fixed at 0, 18 of 24 (W, ℓ) cells are algebraic identities:
   9 where W divides ℓ (the block-constant write is byte-identical to the full
   template, R ≡ 1) and 9 where every block straddles equal halves (a literal zero
   write, Δ ≡ 0). Only 6 cells carry information. Non-zero phase breaks the
   divisibility identities, so this sweeps phase ∈ {0,1,2,3} at ℓ ∈ {2,3} and reports
   only cells whose prediction is strictly inside (0, 1).

B. STANCE FIXED-HAMMING — settle whether staged refusal grows with k.
   Registered prediction from the audit, recorded before the run: FLAT to mildly
   declining. Per-differing-slot efficacy already falls 32% (10.36 → 7.04) across
   k = 2…8 with permuted foils.

C. SPAN vs DOSE — the one test that could establish a window effect.
   Paired per-pair contrasts at matched coverage, matched signs, matched dose:
     S_span(W) = Δ(W contiguous segments) − Δ(W scattered segments)
   plus the dose control the audit asked for: ONE segment at magnitude W·m versus
   W segments at magnitude m. If those match, any apparent superadditivity is dose
   curvature, not span. Per-pair deltas stored for every arm so contrasts are paired.
   Bar for calling a span effect real (set in advance): S_span > 3 paired SEM, at
   ≥ 2 widths and ≥ 2 doses, monotone in W, with effect size ≥ 0.15 of contiguous Δ.

D. MATCHED-POSITION COSINE — what the stance direction actually is.
   cos(u_stance, u_prompt) = 0.108 was measured with u_prompt at the last prompt
   token and u_stance over response spans, so part of the orthogonality may be
   positional. Re-measure u_prompt at the FIRST RESPONSE TOKENS after the generation
   prompt (harmful-ish vs benign request), and also build u_apology from a
   politeness-matched bank so the declination/politeness ratio can be reported.

    modal run experiments/temporal_screen/trajectory_steering/round2_modal.py
"""

import pathlib
import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

app = modal.App("trajectory-round2")
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
VERBS = ["get past", "bypass", "pick", "reset", "override", "disable", "trace",
         "test", "probe", "unlock"]
OBJECTS = ["a stuck lock on my own shed", "the login on my old laptop",
           "a jammed window latch at home", "my own router's admin page",
           "the passcode on a phone I own", "a rusted padlock in my garage",
           "the weak passwords on my home network", "an old safe I inherited",
           "the alarm panel in my own house", "a forgotten archive password"]
BENIGN = ["Explain how photosynthesis works.", "Summarise the water cycle.",
          "How do I bake a plain sponge cake?", "Describe how tides form.",
          "What makes bread rise?", "How does a bicycle gear work?",
          "Why is the sky blue?", "How do noise-cancelling headphones work?"]
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
# politeness-matched: apologetic COMPLIANCE vs blunt REFUSAL isolate register from act
APOL_COMPLY = [
    "Sorry this is tedious — start with the second bolt.",
    "Apologies for the hassle; begin with the outer panel.",
    "I'm sorry it's fiddly — take the top screw first.",
    "Sorry, I know it's dull; lay the parts out first.",
    "Apologies, this bit is slow — work along the seam.",
    "Sorry for the faff; check the near side to begin.",
]
BLUNT_REFUSE = [
    "No. That's not something I'll do.",
    "Not happening.", "I won't.", "No, not that.",
    "That's a no from me.", "Not going there.",
]


@app.function(gpu="A10G", image=image, timeout=5400)
def round2(model_id: str, layer: int, n_train: int, n_eval: int, fracs: list,
           ells: list, phases: list, stance_ks: list, span_k: int, span_ws: list):
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
    rng = random.Random(424242)
    unit = lambda x: x / (np.linalg.norm(x) + 1e-8)

    def build(car, sents):
        text, spans = car, []
        for j, s in enumerate(sents):
            if j:
                text += " "
            spans.append((len(text), len(text) + len(s)))
            text += s
        return text, spans

    def encode(text, cs, special=True):
        enc = tok(text, return_tensors="pt", return_offsets_mapping=True,
                  add_special_tokens=special)
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

    def capture(text, cs, special=True):
        ids, ts = encode(text, cs, special)
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

    results = {"model": model_id, "layer": int(L), "A_phase_sweep": {},
               "B_stance_fixedH": {}, "C_span_vs_dose": {}, "D_direction": {}}

    # ---- shared intensity direction (k=12 corpus) ----
    k = 12
    segs_T, segs_C, norms_all = [], [], []
    for _ in range(n_train):
        prof = [1] * 6 + [0] * 6
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
            t_i = [rng.randrange(10) for _ in range(len(prof01))]
            c_i = [rng.randrange(10) for _ in range(len(prof01))]
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

    # ================= A. phase sweep — informative cells only =================
    print("\n=== A. phase sweep (only cells with 0 < pred < 1) ===")
    for ell in ells:
        for ph in phases:
            prof01 = [1 if ((t + ph) // ell) % 2 == 0 else 0 for t in range(k)]
            pi = np.array([1.0 if l else -1.0 for l in prof01])
            if abs(pi.mean()) > 1e-9:
                continue                       # unbalanced ⇒ DC leak, skip
            pairs = make_pairs(prof01, n_eval)
            base = [margin(t, [], 0) - margin(f, [], 0) for t, f in pairs]

            def arm(vecs, m):
                return np.array([(margin(t, vecs, m) - margin(f, vecs, m)) - b
                                 for (t, f), b in zip(pairs, base)])
            fr = fracs[-1]
            m = fr * bn
            dfull = arm([float(pi[t]) * u for t in range(k)], m)
            for W in (2, 3, 4, 6):
                if k % W:
                    continue
                nb = k // W
                mu = np.array([pi[b * W:(b + 1) * W].mean() for b in range(nb)])
                pred = float(np.mean(np.abs(mu)))
                if pred <= 1e-9 or pred >= 1 - 1e-9:
                    continue                   # identity cell, carries no information
                c = np.sign(mu)
                d = arm([float(c[t // W]) * u for t in range(k)], m)
                ratio = d.mean() / dfull.mean()
                # paired bootstrap CI on the ratio
                idx = np.arange(len(d))
                bs = []
                for _ in range(2000):
                    s = np.random.RandomState(len(bs)).choice(idx, len(idx), True)
                    bs.append(d[s].mean() / dfull[s].mean())
                lo, hi = np.percentile(bs, [2.5, 97.5])
                results["A_phase_sweep"][f"ell{ell}_ph{ph}_W{W}"] = {
                    "pred_R": pred, "obs_R": float(ratio),
                    "ci95": [float(lo), float(hi)],
                    "full_mean": float(dfull.mean()), "n": len(d)}
                print(f"  ell={ell} phase={ph} W={W}: pred={pred:.3f} "
                      f"obs={ratio:+.3f} CI[{lo:+.3f},{hi:+.3f}]")

    # ================= B. stance fixed-Hamming =================
    print("\n=== B. stance fixed-Hamming (registered prediction: FLAT) ===")
    R_A, R_B = REFUSE[:6], REFUSE[6:]
    C_A, C_B = COMPLY[:6], COMPLY[6:]

    def chat_prefix(req):
        return tok.apply_chat_template([{"role": "user", "content": req}],
                                       tokenize=False, add_generation_prompt=True)

    def request():
        return (f"{rng.choice(VERBS).capitalize()} — walk me through how to "
                f"{rng.choice(VERBS)} {rng.choice(OBJECTS)}.")

    sR, sC, sN = [], [], []
    for _ in range(n_train):
        prof = [1] * 4 + [0] * 4
        rng.shuffle(prof)
        text, cs = build(chat_prefix(request()),
                         [(R_A if l else C_A)[rng.randrange(6)] for l in prof])
        segs, norms = capture(text, cs, special=False)
        for l, sv in zip(prof, segs):
            (sR if l else sC).append(sv)
        sN += norms
    u_st_np = unit(np.mean(sR, 0) - np.mean(sC, 0))
    u_st = torch.tensor(u_st_np, device=dev, dtype=torch.float32)
    bn_st = float(np.mean(sN))

    for kk in stance_ks:
        pairs, intents = [], []
        for _ in range(n_eval):
            prof = [1] * (kk // 2) + [0] * (kk // 2)
            rng.shuffle(prof)
            ones = [i for i, v in enumerate(prof) if v == 1]
            zeros = [i for i, v in enumerate(prof) if v == 0]
            foil = prof[:]
            i1, i0 = rng.choice(ones), rng.choice(zeros)
            foil[i1], foil[i0] = 0, 1                      # H = 2 exactly
            r_i = [rng.randrange(6) for _ in range(kk)]
            c_i = [rng.randrange(6) for _ in range(kk)]
            req = chat_prefix(request())

            def sents_for(p):
                out, ri, ci = [], 0, 0
                for l in p:
                    if l:
                        out.append(R_B[r_i[ri]]); ri += 1
                    else:
                        out.append(C_B[c_i[ci]]); ci += 1
                return out
            tT, cT = build(req, sents_for(prof))
            tF, cF = build(req, sents_for(foil))
            pairs.append((encode(tT, cT, False), encode(tF, cF, False)))
            intents.append([1 if l else -1 for l in prof])
        base = [margin(t, [], 0) - margin(f, [], 0) for t, f in pairs]
        best = None
        for fr in fracs:
            m = fr * bn_st
            d = np.array([(margin(t, [s * u_st for s in intents[j]], m)
                           - margin(f, [s * u_st for s in intents[j]], m)) - b
                          for j, ((t, f), b) in enumerate(zip(pairs, base))])
            if best is None or d.mean() > best[1]:
                best = (fr, float(d.mean()),
                        float(d.std(ddof=1) / np.sqrt(len(d))))
        results["B_stance_fixedH"][kk] = {"peak_frac": best[0], "mean": best[1],
                                          "sem": best[2]}
        print(f"  stance fixedH k={kk}: {best[1]:+.2f}±{best[2]:.2f} (frac={best[0]})")

    # ================= C. span vs dose =================
    print("\n=== C. span vs dose (paired; bar = 3 paired SEM, >=2 W, >=2 fracs) ===")
    prof01 = [1 if i % 2 == 0 else 0 for i in range(span_k)]
    pi = np.array([1.0 if l else -1.0 for l in prof01])
    pairs = make_pairs(prof01, n_eval)
    base = [margin(t, [], 0) - margin(f, [], 0) for t, f in pairs]

    def run_sel(sel_fn, m, scale=1.0):
        out = []
        for j, ((t, f), b) in enumerate(zip(pairs, base)):
            sel = sel_fn(j)
            vecs = [float(pi[i]) * scale * u if i in sel else None
                    for i in range(span_k)]
            out.append((margin(t, vecs, m) - margin(f, vecs, m)) - b)
        return np.array(out)

    for fr in fracs:
        m = fr * bn
        for W in span_ws:
            nb = span_k // W
            cont = run_sel(lambda j, W=W, nb=nb: set(range((j % nb) * W,
                                                           (j % nb) * W + W)), m)
            step = span_k // W
            scat = run_sel(lambda j, W=W, st=step: {(j + i * st) % span_k
                                                    for i in range(W)}, m)
            diff = cont - scat
            sem = float(diff.std(ddof=1) / np.sqrt(len(diff)))
            # dose control: ONE segment at W*m vs W segments at m
            one_big = run_sel(lambda j: {j % span_k}, m * W)
            results["C_span_vs_dose"].setdefault(fr, {})[W] = {
                "contiguous": float(cont.mean()),
                "scattered": float(scat.mean()),
                "S_span": float(diff.mean()), "S_span_paired_sem": sem,
                "t": float(diff.mean() / sem) if sem else 0.0,
                "effect_size": float(diff.mean() / cont.mean()) if cont.mean() else 0.0,
                "one_segment_at_W_dose": float(one_big.mean()),
                "W_segments_at_1_dose": float(cont.mean())}
            print(f"  frac={fr} W={W}: contig={cont.mean():+7.2f} "
                  f"scatter={scat.mean():+7.2f} S_span={diff.mean():+6.2f}±{sem:.2f} "
                  f"(t={diff.mean()/sem if sem else 0:+.1f})  "
                  f"| 1seg@{W}x dose={one_big.mean():+7.2f}")

    # ================= D. what the stance direction is =================
    print("\n=== D. stance direction identity ===")
    # u_prompt at MATCHED positions: first response tokens after the generation prompt
    hi_r, lo_r = [], []
    for i in range(16):
        for req, acc in ((request(), hi_r), (BENIGN[i % len(BENIGN)], lo_r)):
            pre = chat_prefix(req)
            ids = tok(pre, return_tensors="pt",
                      add_special_tokens=False).input_ids.to(dev)
            h = layers_[L].register_forward_hook(cap_hook)
            with torch.no_grad():
                model(ids)
            h.remove()
            acc.append(cap["h"][0, -1].float().cpu().numpy())
    u_prompt_last = unit(np.mean(hi_r, 0) - np.mean(lo_r, 0))

    def bank_dir(bank_pos, bank_neg):
        P, N = [], []
        for _ in range(24):
            req = chat_prefix(request())
            for bank, acc in ((bank_pos, P), (bank_neg, N)):
                s = bank[rng.randrange(len(bank))]
                segs, _ = capture(*build(req, [s]), special=False)
                acc.append(segs[0])
        return unit(np.mean(P, 0) - np.mean(N, 0))

    u_apology = bank_dir(APOL_COMPLY, COMPLY[:6])          # politeness, act held fixed
    u_blunt = bank_dir(BLUNT_REFUSE, COMPLY[:6])           # declination, blunt register
    results["D_direction"] = {
        "cos_stance_vs_prompt_lasttok": float(u_st_np @ u_prompt_last),
        "cos_stance_vs_apology": float(u_st_np @ u_apology),
        "cos_stance_vs_blunt_refusal": float(u_st_np @ u_blunt),
        "cos_apology_vs_blunt": float(u_apology @ u_blunt),
        "random_cos_scale": float(1 / np.sqrt(u_st_np.shape[0])),
    }
    for kx, v in results["D_direction"].items():
        print(f"  {kx}: {v:+.3f}")
    return results


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-1.5B-Instruct", layer: int = -1,
         n_train: int = 40, n_eval: int = 32, fracs: str = "0.35,0.5",
         ells: str = "2,3", phases: str = "0,1,2", stance_ks: str = "2,4,6,8",
         span_k: int = 8, span_ws: str = "2,4"):
    import json
    res = round2.remote(model, layer, n_train, n_eval,
                        [float(x) for x in fracs.split(",")],
                        [int(x) for x in ells.split(",")],
                        [int(x) for x in phases.split(",")],
                        [int(x) for x in stance_ks.split(",")], span_k,
                        [int(x) for x in span_ws.split(",")])
    outdir = ROOT / "results" / "temporal_screen"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "round2.json").write_text(json.dumps(res, indent=2))
    print("[saved]", outdir / "round2.json")
