"""Modal: entrainment W-sweep — does steering W segments recruit the model's own
dynamics for the rest?

Generation mode, k=6 sentences, language profiles, steering ONLY sentences 0..W-1.
Two profile families with the same direction u and the same objective langid judge:
  alternating : EN/FR alternation, random starting language (period-2 ⇒ the
                continuation IS predictable from the steered prefix)
  random      : balanced random profile (continuation unpredictable BY CONSTRUCTION)

Measured per episode: langid accuracy vs the intended profile on STEERED slots
(0..W-1) and UNSTEERED slots (W..k-1) separately.

Predictions (theory agent's #4, registered in advance):
  - steered-slot accuracy high and ~flat in W for both families;
  - unsteered-slot accuracy: alternating → above chance, rising with W (pattern
    locks in); random → pinned at chance 0.5 for every W.
The contrast is the point: entrainment happens exactly when the trajectory is
predictable from the steered window — the model's own dynamics amplify a windowed
write. This is also the behavioral answer to "the teacher-forced W-sweep is just
additive bookkeeping": in generation, a windowed write can do MORE than its span.

    modal run experiments/temporal_screen/trajectory_steering/entrain_modal.py
"""

import pathlib
import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

app = modal.App("trajectory-entrain")
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


@app.function(gpu="A10G", image=image, timeout=3600)
def entrain(model_id: str, layer: int, k: int, ws: list, n_train: int, n_gen: int,
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
    print(f"[cfg] L={L}, k={k}, frac={frac}")

    cap = {}

    def cap_hook(_m, _i, out):
        cap["h"] = (out[0] if isinstance(out, tuple) else out).detach()

    unit = lambda x: x / (np.linalg.norm(x) + 1e-8)
    rng = random.Random(4242)

    # ---- direction u + base_norm (same recipe as wsweep train phase) ----
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
    m = frac * base_norm
    print(f"[dir] base_norm={base_norm:.1f}")

    def classify_lang(sent):
        sl = sent.lower()
        fr = 2 * sum(ch in FR_CHARS for ch in sl)
        words = re.findall(r"[a-zàâçéèêëîïôûùüÿœ']+", sl)
        fr += sum(w in FR_WORDS for w in words)
        en = sum(w in EN_WORDS for w in words)
        return 1 if fr > en else 0

    def generate(prefix, coefs, seed, max_tokens=220):
        torch.manual_seed(seed)
        ids = tok(prefix, return_tensors="pt").input_ids.to(dev)
        seg = {"i": 0, "ntok": 0}

        def gen_hook(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            if seg["i"] < len(coefs) and coefs[seg["i"]] != 0:
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
                p = torch.softmax(topv, -1)
                nxt = int(topi[torch.multinomial(p, 1)])
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

    def make_profile(family):
        if family == "alternating":
            start = rng.randint(0, 1)
            return [(start + i) % 2 for i in range(k)]
        prof = [1] * (k // 2) + [0] * (k - k // 2)
        rng.shuffle(prof)
        return prof

    results = {"model": model_id, "layer": int(L), "k": k, "frac": frac,
               "base_norm": base_norm, "cells": {}}
    for family in ("alternating", "random"):
        for W in ws:
            acc_st, acc_un, nse, samples = [], [], [], []
            for ep in range(n_gen):
                prof = make_profile(family)
                signs = [1 if l else -1 for l in prof]
                coefs = [signs[i] if i < W else 0 for i in range(k)]
                i_en, i_fr = rng.randrange(len(EN_FR)), rng.randrange(len(EN_FR))
                prefix = (rng.choice(CARRIERS) + EN_FR[i_en][0] + " "
                          + EN_FR[i_fr][1] + " ")
                text = generate(prefix, coefs, seed=31000 + ep)
                sents = [x.strip() for x in re.split(r"[.!?]", text) if x.strip()][:k]
                got = [classify_lang(x) for x in sents]
                got += [1 - p_ for p_ in prof[len(got):]]
                nse.append(len(sents))
                acc_st.append(float(np.mean([got[i] == prof[i] for i in range(W)])))
                if W < k:
                    acc_un.append(float(np.mean([got[i] == prof[i]
                                                 for i in range(W, k)])))
                if len(samples) < 2:
                    samples.append({"profile": prof, "W": W, "text": text[:300]})
            cell = {
                "steered_acc": float(np.mean(acc_st)),
                "steered_sem": float(np.std(acc_st, ddof=1) / np.sqrt(len(acc_st))),
                "unsteered_acc": float(np.mean(acc_un)) if acc_un else None,
                "unsteered_sem": (float(np.std(acc_un, ddof=1) / np.sqrt(len(acc_un)))
                                  if acc_un else None),
                "mean_sents": float(np.mean(nse)), "samples": samples,
            }
            results["cells"][f"{family}_W{W}"] = cell
            un = (f"{cell['unsteered_acc']:.3f}" if acc_un else "  -  ")
            print(f"[{family:11} W={W}] steered={cell['steered_acc']:.3f} "
                  f"unsteered={un} sents={cell['mean_sents']:.1f}")
    return results


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-1.5B-Instruct", layer: int = -1, k: int = 6,
         ws: str = "1,2,3,4,6", n_train: int = 40, n_gen: int = 20,
         frac: float = 0.35):
    import json
    res = entrain.remote(model, layer, k, [int(x) for x in ws.split(",")],
                         n_train, n_gen, frac)
    print("RESULT:", json.dumps(res, indent=2)[:2500])
    outdir = ROOT / "results" / "temporal_screen"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "entrain.json").write_text(json.dumps(res, indent=2))
    print("[saved]", outdir / "entrain.json")
