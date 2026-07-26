"""Modal: why TopK does not control crosscoder sparsity — the b_enc / ReLU mechanism.

THE HYPOTHESIS, registered before running. Realised L0 was inert to nominal k across a 5x
range (window budget 492 -> 2400 left realised L0 at 18-27) while the ReLU-killed fraction
rose to 0.99. The proposed mechanism: the crosscoder's encoder learns a strongly NEGATIVE
b_enc, routing around a k far larger than reconstruction needs, so that

    realised L0  ~=  min( k , #{pre-activations > 0} )

and the binding term is the second, not the first. TopK is then not the sparsity control at
all — ReLU is.

PREDICTIONS, all falsifiable from the numbers this job prints:
  P1  TXC b_enc mean strongly negative, magnitude INCREASING in nominal k
  P2  SAE b_enc near zero by comparison
  P3  realised L0 ~= #{pre > 0} whenever #{pre > 0} < k, i.e. the ReLU term binds
  P4  the crossover: below some k, TopK binds and realised L0 tracks k; above it, ReLU
      binds and realised L0 flattens

WHY IT MATTERS BEYOND THIS SPRINT. If it holds, any crosscoder-vs-SAE comparison matched
on NOMINAL k is matching a quantity that does not bind — including this project's own
pre-registered protocols and its earlier comparisons. And it ships with a fix: choose k
from a TARGET REALISED L0 by measuring, rather than setting k and hoping.

Cheap by construction: small cache, short training, no steering eval. The whole point is
that the diagnostic costs minutes and would have saved the benchmark.

    modal run experiments/temporal_screen/dict_bench/mechanism_modal.py
"""

import pathlib
import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

app = modal.App("dictbench-mechanism")
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


@app.function(gpu="L4", image=image, timeout=10800)
def mechanism(model_id: str, layer: int, k_seg: int, n_docs: int, d_sae: int,
              txc_batch: int, steps: int, txc_ks: list, sae_ks: list,
              general_frac: float, p_change: float):
    import sys
    sys.path.insert(0, "/work")
    import random
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.bench.architectures.topk_sae import TopKSAE
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

    rng = random.Random(112358)

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
            ix = [i for i, (x, y) in enumerate(offs) if y > x and y > a and x < b]
            ts.append((min(ix), max(ix)))
        h = layers_[L].register_forward_hook(cap_hook)
        with torch.no_grad():
            model(e["input_ids"].to(dev))
        h.remove()
        hh = cap["h"][0].float().cpu().numpy()
        return np.stack([hh[a:b + 1].mean(0) for a, b in ts])

    # corpus with a tunable change probability (p=0.5 reproduces the i.i.d. corpus)
    X = []
    n_gen = int(n_docs * general_frac)
    for i in range(n_docs):
        if i < n_gen:
            sents = [GENERAL[rng.randrange(len(GENERAL))] for _ in range(k_seg)]
        else:
            lab = [rng.randint(0, 1)]
            for _ in range(k_seg - 1):
                lab.append(1 - lab[-1] if rng.random() < p_change else lab[-1])
            sents = [(TENSE if l else CALM)[rng.randrange(10)] for l in lab]
        X.append(capture(*build(rng.choice(CARRIERS), sents)))
        if (i + 1) % 300 == 0:
            print(f"   [cache] {i+1}/{n_docs}")
    X = np.stack(X)
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    n_hold = max(int(0.15 * Xt.shape[0]), 48)
    Xtr, Xho = Xt[:-n_hold], Xt[-n_hold:]
    mu, sd = Xtr.mean((0, 1), keepdim=True), Xtr.std() + 1e-6
    Xn, Xn_ho = (Xtr - mu) / sd, (Xho - mu) / sd
    flat_ho = Xn_ho.reshape(-1, d)
    denom = float(flat_ho.var(0).sum())
    print(f"[cache] {X.shape}  p_change={p_change}")

    def gen_w(bs): return Xn[torch.randint(0, Xn.shape[0], (bs,), device=dev)]

    def gen_f(bs):
        i = torch.randint(0, Xn.shape[0], (bs,), device=dev)
        j = torch.randint(0, k_seg, (bs,), device=dev)
        return Xn[i, j]

    def train(m, gen, bs, tag):
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for s in range(steps):
            loss, _, _ = m(gen(bs))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); m._normalize_decoder()
        print(f"     [{tag}] final loss {loss.item():.3f}")

    out = {"model": model_id, "k_seg": k_seg, "d_sae": d_sae, "steps": steps,
           "p_change": p_change, "txc": [], "sae": []}

    print("\n===== TXC: does b_enc go negative as nominal k rises? =====")
    print(f"{'kper':>5} {'nom_k':>7} {'b_enc mean':>11} {'b_enc sd':>9} "
          f"{'#pre>0':>8} {'realisedL0':>11} {'min(k,#pos)':>12} {'FVU':>7}")
    for kp in txc_ks:
        if kp * k_seg > d_sae:
            continue
        torch.manual_seed(0)
        m = TemporalCrosscoder(d_in=d, d_sae=d_sae, T=k_seg, k=kp).to(dev)
        train(m, gen_w, txc_batch, f"txc-k{kp}")
        with torch.no_grad():
            pre = torch.einsum("btd,tds->bs", Xn_ho, m.W_enc) + m.b_enc
            npos = float((pre > 0).float().sum(-1).mean())
            z = m.encode(Xn_ho); xh = m.decode(z)
            l0 = float((z > 0).float().sum(-1).mean())
            fvu = float(((xh - Xn_ho) ** 2).sum(-1).mean() / denom)
            be_mean = float(m.b_enc.mean()); be_sd = float(m.b_enc.std())
        pred = min(m.k, npos)
        out["txc"].append({"kper": kp, "nominal_window_k": m.k,
                           "b_enc_mean": be_mean, "b_enc_sd": be_sd,
                           "n_pos_preact": npos, "realised_l0": l0,
                           "predicted_min_k_npos": pred, "fvu": fvu,
                           "relu_binds": bool(npos < m.k)})
        print(f"{kp:>5} {m.k:>7} {be_mean:>11.4f} {be_sd:>9.4f} {npos:>8.1f} "
              f"{l0:>11.1f} {pred:>12.1f} {fvu:>7.4f}")

    print("\n===== SAE: same diagnostic, for contrast =====")
    print(f"{'k':>5} {'b_enc mean':>11} {'b_enc sd':>9} {'#pre>0':>8} "
          f"{'realisedL0':>11} {'FVU':>7}")
    for k in sae_ks:
        torch.manual_seed(0)
        m = TopKSAE(d_in=d, d_sae=d_sae, k=k).to(dev)
        train(m, gen_f, txc_batch * k_seg, f"sae-k{k}")
        with torch.no_grad():
            pre = (flat_ho - m.b_dec) @ m.W_enc.T + m.b_enc
            npos = float((pre > 0).float().sum(-1).mean())
            z = m.encode(flat_ho); xh = m.decode(z)
            l0 = float((z > 0).float().sum(-1).mean())
            fvu = float(((xh - flat_ho) ** 2).sum(-1).mean() / denom)
            be_mean = float(m.b_enc.mean()); be_sd = float(m.b_enc.std())
        out["sae"].append({"k": k, "b_enc_mean": be_mean, "b_enc_sd": be_sd,
                           "n_pos_preact": npos, "realised_l0": l0, "fvu": fvu,
                           "relu_binds": bool(npos < k)})
        print(f"{k:>5} {be_mean:>11.4f} {be_sd:>9.4f} {npos:>8.1f} "
              f"{l0:>11.1f} {fvu:>7.4f}")

    # ---- verdicts on the registered predictions ----
    t = out["txc"]
    if len(t) > 1:
        ks = np.array([r["nominal_window_k"] for r in t], float)
        be = np.array([r["b_enc_mean"] for r in t], float)
        l0s = np.array([r["realised_l0"] for r in t], float)
        pr = np.array([r["predicted_min_k_npos"] for r in t], float)
        out["verdict"] = {
            "P1_b_enc_negative_and_growing": bool((be < 0).all()
                                                  and np.corrcoef(ks, -be)[0, 1] > 0.5),
            "corr_nominal_k_vs_neg_b_enc": float(np.corrcoef(ks, -be)[0, 1]),
            "P2_sae_b_enc_smaller":
                bool(abs(np.mean([r["b_enc_mean"] for r in out["sae"]])) < abs(be.mean())),
            "P3_mean_abs_err_L0_vs_min_k_npos": float(np.mean(np.abs(l0s - pr))),
            "P4_relu_binds_fraction": float(np.mean([r["relu_binds"] for r in t])),
        }
        v = out["verdict"]
        print(f"\n[P1] b_enc all negative & growing with k: {v['P1_b_enc_negative_and_growing']}"
              f"  (corr nominal-k vs -b_enc = {v['corr_nominal_k_vs_neg_b_enc']:+.3f})")
        print(f"[P2] SAE |b_enc| smaller than TXC's: {v['P2_sae_b_enc_smaller']}")
        print(f"[P3] realised L0 vs min(k, #pos): mean abs error "
              f"{v['P3_mean_abs_err_L0_vs_min_k_npos']:.2f}")
        print(f"[P4] ReLU binds in {100*v['P4_relu_binds_fraction']:.0f}% of configs")
    return out


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-1.5B-Instruct", layer: int = -1, k_seg: int = 12,
         n_docs: int = 900, d_sae: int = 4096, txc_batch: int = 64,
         steps: int = 1500, txc_ks: str = "1,2,4,10,20,41,100,200",
         sae_ks: str = "2,10,100", general_frac: float = 0.4,
         p_change: float = 0.5):
    import json
    r = mechanism.remote(model, layer, k_seg, n_docs, d_sae, txc_batch, steps,
                         [int(x) for x in txc_ks.split(",")],
                         [int(x) for x in sae_ks.split(",")], general_frac, p_change)
    outdir = ROOT / "results" / "dict_bench"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "mechanism.json").write_text(json.dumps(r, indent=2))
    print("[saved]", outdir / "mechanism.json")
