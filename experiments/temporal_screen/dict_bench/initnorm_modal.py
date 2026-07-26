"""Which single change recovers the crosscoder's capacity: decoder-normalisation at init,
or a lower learning rate?

The first `centering_modal.py` arm came back with numbers that do not resemble anything
this sprint has measured before: at kper=4 it reached 99 positive pre-activations, spent
3.98 of its nominal 4 coefficients per segment, killed 0.6% of its TopK selection, and hit
FVU 0.670. The same nominal configuration in `mechanism_modal.py` had 22 positive
pre-activations, spent 1.7 per segment, killed 40%, and hit FVU 0.865.

Two things differ between those runs and I changed both at once, which is my error:

    decoder normalised at init   mechanism/frontier: NO   centering: YES
    learning rate                mechanism: 1e-3          centering: 3e-4

The repo's TemporalCrosscoder does not normalise W_dec in __init__ (TopKSAE does). Its
atoms therefore start at norm sqrt(T*d_in/d_sae) ~ 2.12 instead of 1, and the training
loop's _normalize_decoder() rescales them only after the first optimiser step -- so Adam's
moments are seeded from gradients taken at a decoder 2.1x too large.

This is a full factorial over exactly those two factors, everything else held fixed, so
whichever recovers capacity is identified rather than guessed. `initnorm=False, lr=1e-3`
reproduces the repo default and must reproduce the collapse; if it does not, the difference
is somewhere I have not looked and no attribution here is valid.

REGISTERED PREDICTIONS (written before the run):
    N0  initnorm=False, lr=1e-3 reproduces the collapse: #pre>0 ~ 20-30, ReLU-kill > 0.3,
        FVU > 0.8. This is the control that licenses everything else.
    N1  initnorm=True is the factor that matters: at both learning rates it lifts #pre>0
        above ~90 and drops ReLU-kill below 0.05.
    N2  lr alone does not: initnorm=False, lr=3e-4 stays collapsed.
    N3  The effect grows with k, since the collapse itself grows with k.
"""
import pathlib
import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

app = modal.App("dictbench-initnorm")
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers", "accelerate", "numpy")
    .add_local_dir(str(ROOT / "src"), "/work/src")
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
    "A new footpath now follows the old railway line.",
    "The recipe calls for butter at room temperature.",
    "Their letters were kept in a tin under the bed.",
    "The clock in the hall runs four minutes fast.",
]
CARRIERS = ["Journal entry.\n", "From the notebook:\n", "Draft passage.\n",
            "Field notes.\n", "Evening record.\n", "From chapter twelve:\n"]


@app.function(gpu="L4", image=image, timeout=14400)
def initnorm(model_id: str, layer: int, k_seg: int, n_docs: int, d_sae: int,
             batch: int, steps: int, kpers: list, lrs: list, general_frac: float):
    import sys
    sys.path.insert(0, "/work")
    import random
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.bench.architectures.crosscoder import TemporalCrosscoder

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    dev = model.device
    layers_ = model.model.layers
    L = layer if layer >= 0 else len(layers_) // 2
    d = model.config.hidden_size
    cap = {}

    def cap_hook(_m, _i, out):
        cap["h"] = (out[0] if isinstance(out, tuple) else out).detach()

    rng = random.Random(2718281)

    def build(car, sents):
        text, spans = car, []
        for j, s in enumerate(sents):
            if j:
                text += " "
            spans.append((len(text), len(text) + len(s)))
            text += s
        return text, spans

    def capture(text, cs):
        e = tok(text, return_tensors="pt", return_offsets_mapping=True)
        offs = e["offset_mapping"][0].tolist()
        ts = []
        for (a, b) in cs:
            idx = [i for i, (s0, s1) in enumerate(offs) if s0 >= a and s1 <= b and s1 > s0]
            ts.append((idx[0], idx[-1]) if idx else (0, 0))
        h = layers_[L].register_forward_hook(cap_hook)
        with torch.no_grad():
            model(e["input_ids"].to(dev))
        h.remove()
        hh = cap["h"][0].float().cpu().numpy()
        return np.stack([hh[a:b + 1].mean(0) for a, b in ts])

    X = []
    for i in range(n_docs):
        if rng.random() < general_frac:
            sents = [GENERAL[rng.randrange(len(GENERAL))] for _ in range(k_seg)]
        else:
            sents = [(TENSE if rng.randint(0, 1) else CALM)[rng.randrange(10)]
                     for _ in range(k_seg)]
        X.append(capture(*build(rng.choice(CARRIERS), sents)))
        if (i + 1) % 300 == 0:
            print(f"   [cache] {i+1}/{n_docs}", flush=True)
    Xt = torch.tensor(np.stack(X), dtype=torch.float32, device=dev)
    n_hold = max(int(0.15 * Xt.shape[0]), 32)
    Xtr, Xho = Xt[:-n_hold], Xt[-n_hold:]
    mu, sd = Xtr.mean((0, 1), keepdim=True), Xtr.std() + 1e-6
    Xn, Xn_ho = (Xtr - mu) / sd, (Xho - mu) / sd
    denom = float(Xn_ho.reshape(-1, d).var(0).sum())
    print(f"[cache] {tuple(Xn.shape)}", flush=True)

    def run(kp, lr, do_initnorm):
        torch.manual_seed(0)
        m = TemporalCrosscoder(d_in=d, d_sae=d_sae, T=k_seg, k=kp).to(dev)
        atom0 = float(m.W_dec.data.norm(dim=(1, 2)).mean())
        if do_initnorm:
            with torch.no_grad():
                m._normalize_decoder()
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        hist = []
        for s in range(steps):
            xb = Xn[torch.randint(0, Xn.shape[0], (batch,), device=dev)]
            loss, _, _ = m(xb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); m._normalize_decoder()
            if s % max(1, steps // 5) == 0 or s == steps - 1:
                hist.append(round(float(loss.detach()), 2))
        with torch.no_grad():
            pre = torch.einsum("btd,tds->bs", Xn_ho, m.W_enc) + m.b_enc
            npos = float((pre > 0).float().sum(-1).mean())
            z = m.encode(Xn_ho); xh = m.decode(z)
            fvu = float(((xh - Xn_ho) ** 2).sum(-1).mean() / denom)
            l0 = float((z > 0).float().sum(-1).mean())
            alive = float(((z > 0).float().mean(0) >= 0.001).float().mean())
        return {"kper": kp, "lr": lr, "initnorm": do_initnorm, "nominal_k": m.k,
                "atom_norm_at_init": atom0, "n_pos_preact": npos,
                "realised_l0": l0, "coeff_per_segment": l0 / k_seg,
                "relu_kill_frac": 1.0 - l0 / m.k, "alive_frac": alive, "fvu": fvu,
                "loss_hist": hist}

    out = []
    print(f"\n{'initnorm':>9}{'lr':>8}{'kper':>6}{'nom k':>7}{'#pre>0':>9}"
          f"{'coeff/seg':>11}{'ReLU-kill':>11}{'alive':>8}{'FVU':>9}", flush=True)
    for kp in kpers:
        for lr in lrs:
            for do in [False, True]:
                r = run(kp, lr, do)
                out.append(r)
                print(f"{str(do):>9}{lr:>8g}{kp:>6}{r['nominal_k']:>7}"
                      f"{r['n_pos_preact']:>9.1f}{r['coeff_per_segment']:>11.2f}"
                      f"{r['relu_kill_frac']:>11.3f}{r['alive_frac']:>8.3f}"
                      f"{r['fvu']:>9.4f}", flush=True)
                print(f"          loss {r['loss_hist']}", flush=True)

    print(f"\n[atom norm at init, unnormalised] {out[0]['atom_norm_at_init']:.3f}",
          flush=True)
    print("\n===== attribution =====", flush=True)
    for kp in kpers:
        for lr in lrs:
            a = next(r for r in out if r["kper"] == kp and r["lr"] == lr
                     and not r["initnorm"])
            b = next(r for r in out if r["kper"] == kp and r["lr"] == lr
                     and r["initnorm"])
            print(f"  kper={kp} lr={lr:g}  initnorm False->True:  "
                  f"#pre>0 {a['n_pos_preact']:.0f}->{b['n_pos_preact']:.0f}   "
                  f"coeff/seg {a['coeff_per_segment']:.2f}->{b['coeff_per_segment']:.2f}   "
                  f"FVU {a['fvu']:.3f}->{b['fvu']:.3f}", flush=True)
    return {"rows": out, "k_seg": k_seg, "d_sae": d_sae, "steps": steps}


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-1.5B-Instruct", layer: int = -1, k_seg: int = 12,
         n_docs: int = 900, d_sae: int = 4096, batch: int = 64, steps: int = 2500,
         kpers: str = "4,20", lrs: str = "1e-3,3e-4", general_frac: float = 0.4):
    import json
    r = initnorm.remote(model, layer, k_seg, n_docs, d_sae, batch, steps,
                        [int(x) for x in kpers.split(",")],
                        [float(x) for x in lrs.split(",")], general_frac)
    outdir = ROOT / "results" / "dict_bench"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "initnorm.json").write_text(json.dumps(r, indent=2))
    print("[saved]", outdir / "initnorm.json")
