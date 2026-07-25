"""Modal: entrainment v2 — fixes the balance-induced null found in v1.

v1 finding: with BALANCED profiles (exactly k/2 per language) the unsteered-tail null
is NOT 0.5. If the steered prefix is French-heavy the tail is English-heavy by
construction, so a model that simply persists in the last steered language scores
BELOW chance. Exact analytic null for k=6 balanced + persistence: 0.400 — and v1's
random arm measured 0.370–0.475. The "hard 0.5 null" the theory wanted requires
i.i.d. profiles.

v2 families (k=6, W ∈ {1,2,3,4}):
  iid       : each slot an independent coin flip ⇒ tail genuinely independent of
              prefix ⇒ TRUE 0.5 null for every W (the theory's bug detector)
  balanced  : v1's construction, kept to demonstrate the 0.400 artifact explicitly
  alt       : alternating, ℓ=1 ⇒ predicted entrainment threshold W* = ℓ+1 = 2
  period2   : ++--++-- , ℓ=2 ⇒ predicted W* = 3

Each family also gets an m=0 (steering off) cell: the model's innate rate of matching
the intended profile, which is the prior-only account the theory demands as a control.

Reports per cell: steered-slot accuracy, unsteered-slot accuracy, the ANALYTIC
persistence null for that family, and observed persistence rate.

    modal run experiments/temporal_screen/trajectory_steering/entrain2_modal.py
"""

import pathlib
import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

app = modal.App("trajectory-entrain2")
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
CARRIERS = ["Journal entry.\n", "From the notebook:\n", "Draft passage.\n",
            "Field notes.\n", "Evening record.\n", "From chapter twelve:\n"]
FR_CHARS = "éèêàâçùûôîïëüœ"
FR_WORDS = {"le", "la", "les", "un", "une", "des", "du", "est", "dans", "et", "il",
            "elle", "je", "nous", "vous", "sur", "avec", "pour", "au", "aux", "ce",
            "cette", "se", "ne", "pas", "son", "sa", "ses", "qui", "que", "chaque",
            "sous", "encore", "matin", "soir", "nuit", "très", "bien", "plus"}
EN_WORDS = {"the", "a", "an", "is", "are", "was", "in", "on", "and", "he", "she",
            "it", "they", "we", "of", "to", "with", "for", "at", "this", "that",
            "his", "her", "its", "not", "by", "every", "still", "old", "night",
            "very", "more", "as", "but"}


@app.function(gpu="A10G", image=image, timeout=5400)
def entrain2(model_id: str, layer: int, k: int, ws: list, n_train: int, n_gen: int,
             frac: float):
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
    cap = {}

    def cap_hook(_m, _i, out):
        cap["h"] = (out[0] if isinstance(out, tuple) else out).detach()

    unit = lambda x: x / (np.linalg.norm(x) + 1e-8)
    rng = random.Random(9090)

    by_lbl, norms_all = {0: [], 1: []}, []
    for _ in range(n_train):
        prof = [1] * (k // 2) + [0] * (k - k // 2)
        rng.shuffle(prof)
        text, spans = rng.choice(CARRIERS), []
        for j, l in enumerate(prof):
            s = EN_FR[rng.randrange(len(EN_FR))][l]
            if j:
                text += " "
            spans.append((len(text), len(text) + len(s)))
            text += s
        enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
        offs = enc["offset_mapping"][0].tolist()
        h = layers_[L].register_forward_hook(cap_hook)
        with torch.no_grad():
            model(enc["input_ids"].to(dev))
        h.remove()
        hh = cap["h"][0].float().cpu().numpy()
        for (a, b), l in zip(spans, prof):
            ix = [i for i, (x, y) in enumerate(offs) if y > x and y > a and x < b]
            by_lbl[l].append(hh[min(ix):max(ix) + 1].mean(0))
            norms_all += [float(np.linalg.norm(hh[p])) for p in ix]
    u = torch.tensor(unit(np.mean(by_lbl[1], 0) - np.mean(by_lbl[0], 0)),
                     device=dev, dtype=torch.float32)
    base_norm = float(np.mean(norms_all))
    print(f"[dir] base_norm={base_norm:.1f}")

    def classify(sent):
        sl = sent.lower()
        fr = 2 * sum(ch in FR_CHARS for ch in sl)
        words = re.findall(r"[a-zàâçéèêëîïôûùüÿœ']+", sl)
        fr += sum(w in FR_WORDS for w in words)
        en = sum(w in EN_WORDS for w in words)
        if fr == 0 and en == 0:
            return None
        return 1 if fr > en else 0

    def generate(prefix, coefs, m, seed, max_tokens=220):
        torch.manual_seed(seed)
        ids = tok(prefix, return_tensors="pt").input_ids.to(dev)
        seg = {"i": 0, "ntok": 0}

        def gen_hook(_mod, _i, out):
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
                logits = o.logits[0, -1] / 0.7
                topv, topi = logits.topk(50)
                nxt = int(topi[torch.multinomial(torch.softmax(topv, -1), 1)])
                if nxt == tok.eos_token_id:
                    break
                out_ids.append(nxt)
                seg["ntok"] += 1
                if any(c in tok.decode([nxt]) for c in ".!?") and seg["ntok"] >= 3:
                    seg["i"] += 1
                    seg["ntok"] = 0
                    if seg["i"] >= k:
                        break
                cur = torch.tensor([[nxt]], device=dev)
        finally:
            h.remove()
        return tok.decode(out_ids)

    def profile(family):
        if family == "iid":
            return [rng.randint(0, 1) for _ in range(k)]
        if family == "balanced":
            p = [1] * (k // 2) + [0] * (k - k // 2)
            rng.shuffle(p)
            return p
        if family == "alt":
            s = rng.randint(0, 1)
            return [(s + i) % 2 for i in range(k)]
        s = rng.randint(0, 1)
        return [(s + i // 2) % 2 for i in range(k)]      # period2, ell=2

    def analytic_null(family, W, trials=20000):
        """Persistence strategy: predict tail = last steered slot's label."""
        acc = []
        for _ in range(trials):
            p = profile(family)
            pred = p[W - 1]
            tail = p[W:]
            if tail:
                acc.append(np.mean([t == pred for t in tail]))
        return float(np.mean(acc))

    results = {"model": model_id, "layer": int(L), "k": k, "frac": frac,
               "base_norm": base_norm, "cells": {}}
    FAMILIES = ("iid", "balanced", "alt", "period2")
    for family in FAMILIES:
        for W in ws + [0]:                                # W=0 ⇒ m=0 control
            m = 0.0 if W == 0 else frac * base_norm
            Weff = 0 if W == 0 else W
            st, un, pers, cov, samples = [], [], [], [], []
            for ep in range(n_gen):
                prof = profile(family)
                signs = [1 if l else -1 for l in prof]
                coefs = [signs[i] if i < Weff else 0 for i in range(k)]
                i_en, i_fr = rng.randrange(len(EN_FR)), rng.randrange(len(EN_FR))
                prefix = (rng.choice(CARRIERS) + EN_FR[i_en][0] + " "
                          + EN_FR[i_fr][1] + " ")
                text = generate(prefix, coefs, m, seed=61000 + ep)
                sents = [x.strip() for x in re.split(r"[.!?]", text) if x.strip()][:k]
                lab = [classify(x) for x in sents]
                lab += [None] * (k - len(lab))
                cov.append(float(np.mean([x is not None for x in lab])))
                if Weff:
                    v = [lab[i] == prof[i] for i in range(Weff) if lab[i] is not None]
                    if v:
                        st.append(float(np.mean(v)))
                    v2 = [lab[i] == prof[i] for i in range(Weff, k)
                          if lab[i] is not None]
                    if v2:
                        un.append(float(np.mean(v2)))
                    if lab[Weff - 1] is not None:
                        pv = [lab[i] == lab[Weff - 1] for i in range(Weff, k)
                              if lab[i] is not None]
                        if pv:
                            pers.append(float(np.mean(pv)))
                else:
                    v = [lab[i] == prof[i] for i in range(k) if lab[i] is not None]
                    if v:
                        st.append(float(np.mean(v)))
                if len(samples) < 2:
                    samples.append({"profile": prof, "W": Weff, "text": text[:250]})
            mean_sem = lambda a: (float(np.mean(a)), float(np.std(a, ddof=1) /
                                                           np.sqrt(len(a)))) if a else (None, None)
            s_m, s_e = mean_sem(st)
            u_m, u_e = mean_sem(un)
            p_m, _ = mean_sem(pers)
            cell = {"steered_acc": s_m, "steered_sem": s_e,
                    "unsteered_acc": u_m, "unsteered_sem": u_e,
                    "persistence_rate": p_m,
                    "analytic_persistence_null": (analytic_null(family, Weff)
                                                  if Weff and Weff < k else None),
                    "coverage": float(np.mean(cov)), "samples": samples}
            results["cells"][f"{family}_W{Weff}"] = cell
            un_s = f"{u_m:.3f}" if u_m is not None else "  -  "
            nl_s = (f"{cell['analytic_persistence_null']:.3f}"
                    if cell["analytic_persistence_null"] is not None else "  -  ")
            print(f"[{family:9} W={Weff}] steered={s_m if s_m is None else round(s_m,3)} "
                  f"unsteered={un_s} null={nl_s} cov={cell['coverage']:.2f}")
    return results


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-1.5B-Instruct", layer: int = -1, k: int = 6,
         ws: str = "1,2,3,4", n_train: int = 40, n_gen: int = 48,
         frac: float = 0.35):
    import json
    res = entrain2.remote(model, layer, k, [int(x) for x in ws.split(",")],
                          n_train, n_gen, frac)
    outdir = ROOT / "results" / "temporal_screen"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "entrain2.json").write_text(json.dumps(res, indent=2))
    print("[saved]", outdir / "entrain2.json")
