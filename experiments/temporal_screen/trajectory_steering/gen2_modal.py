"""Modal: generation-mode eval v2 for lang_profile (fixes v1 plumbing failures).

v1 (full_modal.py gen_eval) failed for decoding reasons, not steering reasons:
greedy decode from a bare carrier degenerated into numbered-list loops ("1. 1000...")
whose "1." token spuriously advanced the sentence counter. The broadcast arm's sample
showed fluent French — the direction works in free generation; the harness didn't.

Fixes:
  - 2-shot bilingual prefix: carrier + one EN + one FR bank sentence, so the model is
    in short-sentence bilingual-notebook register before steering begins (balanced —
    leaks neither the profile nor a language bias).
  - temperature sampling (T=0.7, top-k 50) with a per-episode torch seed, not greedy.
  - sentence counter advances on . ! ? and requires >=3 tokens in the segment.
  - reports sentences-generated alongside accuracy (missing slots still count wrong).
  - fracs at the teacher-forced template peak (0.35, 0.5).

Metric unchanged: per-slot accuracy of generated sentence language (marker-word
classifier) vs the intended random balanced profile. Chance = 0.5; broadcast(FR)
should pin near it; template should exceed it.

    modal run experiments/temporal_screen/trajectory_steering/gen2_modal.py
"""

import pathlib
import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

app = modal.App("trajectory-gen2")
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


@app.function(gpu="A10G", image=image, timeout=1800)
def gen2(model_id: str, layer: int, k: int, n_train: int, n_gen: int, fracs: list):
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
    rng = random.Random(500)

    # ---- direction: same recipe as the k-sweep train phase ----
    by_lbl, norms_all = {0: [], 1: []}, []
    for _ in range(n_train):
        prof = [1] * (k // 2) + [0] * (k - k // 2)
        rng.shuffle(prof)
        text = rng.choice(CARRIERS)
        spans = []
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

    def classify_lang(sent):
        sl = sent.lower()
        fr = 2 * sum(ch in FR_CHARS for ch in sl)
        words = re.findall(r"[a-zàâçéèêëîïôûùüÿœ']+", sl)
        fr += sum(w in FR_WORDS for w in words)
        en = sum(w in EN_WORDS for w in words)
        return 1 if fr > en else 0

    def generate(prefix, coefs, m, seed, max_tokens=200):
        torch.manual_seed(seed)
        ids = tok(prefix, return_tensors="pt").input_ids.to(dev)
        seg = {"i": 0, "ntok": 0}

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
                    if seg["i"] >= len(coefs):
                        break
                cur = torch.tensor([[nxt]], device=dev)
        finally:
            h.remove()
        return tok.decode(out_ids)

    results = {"model": model_id, "layer": int(L), "k": k, "base_norm": base_norm,
               "arms": {}}
    for fr in fracs:
        m = fr * base_norm
        for arm in ("template", "broadcast", "single"):
            accs, nsents, samples = [], [], []
            for ep in range(n_gen):
                prof = [1] * (k // 2) + [0] * (k - k // 2)
                rng.shuffle(prof)
                s = [1 if l else -1 for l in prof]
                coefs = {"template": s, "broadcast": [1] * k,
                         "single": [s[0]] + [0] * (k - 1)}[arm]
                i_en, i_fr = rng.randrange(len(EN_FR)), rng.randrange(len(EN_FR))
                prefix = (rng.choice(CARRIERS) + EN_FR[i_en][0] + " "
                          + EN_FR[i_fr][1] + " ")
                text = generate(prefix, coefs, m, seed=7000 + ep)
                sents = [x.strip() for x in re.split(r"[.!?]", text) if x.strip()][:k]
                got = [classify_lang(x) for x in sents]
                nsents.append(len(got))
                got += [1 - p_ for p_ in prof[len(got):]]
                accs.append(float(np.mean([g == p_ for g, p_ in zip(got, prof)])))
                if len(samples) < 2:
                    samples.append({"profile": prof, "text": text[:400]})
            results["arms"][f"{arm}@{fr}"] = {
                "acc_mean": float(np.mean(accs)),
                "acc_sem": float(np.std(accs, ddof=1) / np.sqrt(len(accs))),
                "mean_sents": float(np.mean(nsents)), "samples": samples,
            }
            print(f"[gen frac={fr}] {arm}: acc={np.mean(accs):.3f} "
                  f"(sents={np.mean(nsents):.1f}/{k})")
    return results


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-1.5B-Instruct", layer: int = -1, k: int = 6,
         n_train: int = 40, n_gen: int = 24, fracs: str = "0.35,0.5"):
    import json
    res = gen2.remote(model, layer, k, n_train, n_gen,
                      [float(x) for x in fracs.split(",")])
    print("RESULT:", json.dumps(res, indent=2)[:3000])
    outdir = ROOT / "results" / "temporal_screen"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "trajectory_gen2.json").write_text(json.dumps(res, indent=2))
    print("[saved]", outdir / "trajectory_gen2.json")
