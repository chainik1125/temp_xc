"""Modal SMOKE: cache → train tiny SAE + TXC → steer from a decoder row.

Purpose is plumbing, not results. The previous sprint's clearest lesson was that three
multi-hour designs were confounded and a three-minute smoke would have caught two of
them, so nothing long runs until this passes.

WHAT IT CHECKS, in order:
  1. we can capture SEGMENT-pooled activations from Qwen-2.5-1.5B L14 -> (N, k, d)
  2. the repo's real TopKSAE and TemporalCrosscoder train on that cache without error
  3. a steering latent can be SELECTED by a rule applied identically to both
  4. writing that latent's decoder row(s) moves the teacher-forced margin at all

DESIGN NOTE — why segment-pooled and not token windows. The bench TemporalCrosscoder
takes (B, T, d) where T is a window of positions. Our steering acts on SENTENCE segments,
so we set T = number of sentences and pool activations within each sentence. A TXC latent
then owns a k-segment write pattern that lines up exactly with what the steering harness
writes, and no resampling between train and eval is needed.

MAGNITUDE TRAP, handled explicitly. TemporalCrosscoder._normalize_decoder normalises
W_dec over dims (1,2), so a latent's ENTIRE (T, d) pattern has unit norm — its per-segment
rows have norm ~1/sqrt(T). TopKSAE decoder columns are unit norm PER direction. Writing
alpha * row from each would therefore inject sqrt(T)x more norm for the SAE. Every write
here is rescaled to a fixed total injected norm, and the realised norms are reported so
the smoke fails loudly if they diverge.

    modal run experiments/temporal_screen/dict_bench/smoke_modal.py
"""

import pathlib
import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

app = modal.App("dictbench-smoke")
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers", "accelerate", "numpy")
    .add_local_dir(str(ROOT / "src"), "/work/src")          # the REAL architectures
)

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
# general text so the dictionary is not trained only on the task distribution
GENERAL = [
    "The committee met on Tuesday to review the budget.",
    "Rain is expected across the northern counties tomorrow.",
    "He learned to play the piano from his grandmother.",
    "The library closes at six on weekdays.",
    "Prices for timber have risen since the spring.",
    "She studied geology before switching to law.",
    "The bridge was rebuilt after the flood.",
    "Most of the crops were harvested by August.",
    "The museum acquired a collection of old maps.",
    "Traffic was diverted around the market square.",
    "They repainted the shutters a pale green.",
    "The report runs to nearly two hundred pages.",
]
CARRIERS = ["Journal entry.\n", "From the notebook:\n", "Draft passage.\n",
            "Field notes.\n", "Evening record.\n", "From chapter twelve:\n"]


@app.function(gpu="L4", image=image, timeout=3600)
def smoke(model_id: str, layer: int, k_seg: int, n_docs: int, d_sae: int,
          steps: int, topk: int, n_eval: int, frac: float, general_frac: float):
    import sys
    sys.path.insert(0, "/work")
    import random
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.bench.architectures.topk_sae import TopKSAE
    from src.bench.architectures.crosscoder import TemporalCrosscoder

    print(f"[load] {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    dev = model.device
    layers_ = model.model.layers
    L = layer if layer >= 0 else len(layers_) // 2
    d_model = model.config.hidden_size
    print(f"[cfg] L={L}  d_model={d_model}  k_seg={k_seg}")

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
    rng = random.Random(11)

    def build(car, sents):
        text, spans = car, []
        for j, s in enumerate(sents):
            if j:
                text += " "
            spans.append((len(text), len(text) + len(s)))
            text += s
        return text, spans

    def encode(text, cs):
        enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
        offs = enc["offset_mapping"][0].tolist()
        ts = []
        for (a, b) in cs:
            ix = [i for i, (x, y) in enumerate(offs) if y > x and y > a and x < b]
            ts.append((min(ix), max(ix)))
        return enc["input_ids"].to(dev), ts

    def capture(text, cs):
        ids, ts = encode(text, cs)
        h = layers_[L].register_forward_hook(cap_hook)
        with torch.no_grad():
            model(ids)
        h.remove()
        hh = cap["h"][0].float().cpu().numpy()
        segs = np.stack([hh[a:b + 1].mean(0) for a, b in ts])       # (k_seg, d)
        norms = [float(np.linalg.norm(hh[p])) for a, b in ts
                 for p in range(a, b + 1)]
        return segs, norms

    def seg_logprob(ids, ts):
        with torch.no_grad():
            lp = model(ids).logits[0].log_softmax(-1).float()
        return float(sum(lp[p - 1, ids[0, p]]
                         for a, b in ts for p in range(a, b + 1) if p >= 1))

    # ---------------- 1. activation cache ----------------
    X, labels, norms_all = [], [], []
    n_general = int(n_docs * general_frac)
    for i in range(n_docs):
        if i < n_general:
            sents = [GENERAL[rng.randrange(len(GENERAL))] for _ in range(k_seg)]
            lab = [-1] * k_seg                                   # unlabelled
        else:
            lab = [rng.randint(0, 1) for _ in range(k_seg)]
            sents = [(TENSE if l else CALM)[rng.randrange(10)] for l in lab]
        segs, nn_ = capture(*build(rng.choice(CARRIERS), sents))
        X.append(segs)
        labels.append(lab)
        norms_all += nn_
    X = np.stack(X)                                              # (N, k_seg, d)
    labels = np.array(labels)
    base_norm = float(np.mean(norms_all))
    print(f"[cache] X={X.shape}  base_norm={base_norm:.1f}  "
          f"general_frac={n_general/n_docs:.2f}")

    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    mu, sd = Xt.mean((0, 1), keepdim=True), Xt.std() + 1e-6
    Xn = (Xt - mu) / sd                                          # standardise

    # ---------------- 2. train both dictionaries ----------------
    def gen_window(bs):
        idx = torch.randint(0, Xn.shape[0], (bs,), device=dev)
        return Xn[idx]                                           # (bs, k_seg, d)

    def gen_flat(bs):
        i = torch.randint(0, Xn.shape[0], (bs,), device=dev)
        j = torch.randint(0, k_seg, (bs,), device=dev)
        return Xn[i, j]                                          # (bs, d)

    sae = TopKSAE(d_in=d_model, d_sae=d_sae, k=topk).to(dev)
    txc = TemporalCrosscoder(d_in=d_model, d_sae=d_sae, T=k_seg, k=topk).to(dev)
    print(f"[dict] SAE k={sae.k}   TXC k(window)={txc.k}  "
          f"(constructor multiplies by T={k_seg})")

    def train(m, gen, tag):
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for s in range(steps):
            x = gen(256)
            loss, _, z = m(x)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); m._normalize_decoder()
            if s % max(steps // 4, 1) == 0 or s == steps - 1:
                print(f"   [{tag}] step {s:4d} loss={loss.item():.4f} "
                      f"L0={(z > 0).float().sum(-1).mean().item():.1f}")
        return m
    train(sae, gen_flat, "sae")
    train(txc, gen_window, "txc")

    # ---------------- 3. matched feature selection ----------------
    task = labels[:, 0] >= 0
    with torch.no_grad():
        z_sae = sae.encode(Xn[task].reshape(-1, d_model))          # (N*k, h)
        lab_flat = torch.tensor(labels[task].reshape(-1), device=dev)
        d_sae_sep = (z_sae[lab_flat == 1].mean(0) - z_sae[lab_flat == 0].mean(0))
        j_sae = int(d_sae_sep.abs().argmax())
        # TXC: one code per window, so separate on the window's tense FRACTION
        z_txc = txc.encode(Xn[task])                               # (N, h)
        frac_t = torch.tensor(labels[task].mean(1), device=dev, dtype=torch.float32)
        hi, lo = frac_t > 0.6, frac_t < 0.4
        d_txc_sep = (z_txc[hi].mean(0) - z_txc[lo].mean(0)) if (hi.any() and lo.any()) \
            else z_txc.mean(0)
        j_txc = int(d_txc_sep.abs().argmax())
    print(f"[select] SAE latent {j_sae} (sep {d_sae_sep[j_sae]:+.3f})   "
          f"TXC latent {j_txc} (sep {d_txc_sep[j_txc]:+.3f})")

    # ---------------- 4. decoder rows, magnitude-matched ----------------
    with torch.no_grad():
        v_sae = sae.decoder_directions()[:, j_sae].float()          # (d,)
        P_txc = txc.W_dec[j_txc].float()                            # (k_seg, d)
    print(f"[rows] |v_sae|={v_sae.norm():.3f}   "
          f"|P_txc| total={P_txc.norm():.3f}  per-row mean={P_txc.norm(dim=1).mean():.3f}")

    def matched(vecs, target_total):
        cur = torch.stack([v for v in vecs if v is not None]).norm()
        return [None if v is None else v * (target_total / (cur + 1e-9)) for v in vecs]

    # ---------------- 5. does a decoder-row write move the margin? ----------
    pairs, intents = [], []
    for _ in range(n_eval):
        prof = [1] * (k_seg // 2) + [0] * (k_seg - k_seg // 2)
        rng.shuffle(prof)
        foil = prof[:]
        for _ in range(40):
            rng.shuffle(foil)
            if foil != prof:
                break
        t_i = [rng.randrange(10) for _ in range(k_seg)]
        c_i = [rng.randrange(10) for _ in range(k_seg)]
        car = rng.choice(CARRIERS)

        def sents_for(p):
            out, a, b = [], 0, 0
            for l in p:
                if l:
                    out.append(TENSE[t_i[a]]); a += 1
                else:
                    out.append(CALM[c_i[b]]); b += 1
            return out
        tT, cT = build(car, sents_for(prof))
        tF, cF = build(car, sents_for(foil))
        pairs.append((encode(tT, cT), encode(tF, cF)))
        intents.append([1.0 if l else -1.0 for l in prof])

    def margin(pair, vecs, scale):
        ids, ts = pair
        steer["v"] = []
        for i, v in enumerate(vecs):
            if v is None:
                continue
            a, b = ts[i]
            steer["v"].append((max(a - 1, 0), max(b - 1, 0), scale * v))
        val = seg_logprob(ids, ts)
        steer["v"] = []
        return val

    base = [margin(t, [None] * k_seg, 0) - margin(f, [None] * k_seg, 0)
            for t, f in pairs]
    total_norm = frac * base_norm * (k_seg ** 0.5)      # fixed injected norm budget
    out = {}
    for tag in ("sae_scheduled", "sae_broadcast", "txc_pattern"):
        ds = []
        for j, ((t, f), b) in enumerate(zip(pairs, base)):
            if tag == "sae_scheduled":
                vecs = [intents[j][i] * v_sae for i in range(k_seg)]
            elif tag == "sae_broadcast":
                vecs = [v_sae for _ in range(k_seg)]
            else:
                vecs = [P_txc[i] for i in range(k_seg)]
            vecs = matched(vecs, total_norm)
            ds.append((margin(t, vecs, 1.0) - margin(f, vecs, 1.0)) - b)
        ds = np.array(ds)
        out[tag] = {"mean": float(ds.mean()),
                    "sem": float(ds.std(ddof=1) / np.sqrt(len(ds)))}
        print(f"[steer] {tag:15} Δmargin = {ds.mean():+7.2f} ± "
              f"{ds.std(ddof=1)/np.sqrt(len(ds)):.2f}")

    return {"model": model_id, "layer": int(L), "k_seg": k_seg, "d_sae": d_sae,
            "topk": topk, "steps": steps, "base_norm": base_norm,
            "cache_shape": list(X.shape), "general_frac": general_frac,
            "sae_latent": j_sae, "txc_latent": j_txc,
            "sae_row_norm": float(v_sae.norm()),
            "txc_pattern_norm": float(P_txc.norm()),
            "txc_row_norm_mean": float(P_txc.norm(dim=1).mean()),
            "steering": out}


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-1.5B-Instruct", layer: int = -1, k_seg: int = 12,
         n_docs: int = 240, d_sae: int = 512, steps: int = 400, topk: int = 8,
         n_eval: int = 16, frac: float = 0.35, general_frac: float = 0.4):
    import json
    res = smoke.remote(model, layer, k_seg, n_docs, d_sae, steps, topk, n_eval,
                       frac, general_frac)
    print("RESULT:", json.dumps(res, indent=2))
    outdir = ROOT / "results" / "dict_bench"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "smoke.json").write_text(json.dumps(res, indent=2))
    print("[saved]", outdir / "smoke.json")
