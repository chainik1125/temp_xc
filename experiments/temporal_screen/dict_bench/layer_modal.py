"""Where should a window code actually help? Sweep the layer.

The first-pass benchmark found the plain crosscoder losing to a TopK SAE at every matched
budget on layer-14 activations (FVU 1.2-2.7x worse, window-AUC no better). The synthetic
lane, by contrast, shows a *provable* advantage for window codes: a per-token dictionary
cannot recover an extent-L feature beyond ||largest contiguous T-chunk of p|| / ||p||,
which is 0.481 at L=8, and the crosscoder reaches 0.905.

The obstruction that reconciles those two results: **in a transformer, a mid-stack token
representation has already attended over its predecessors.** Whatever cross-segment
structure exists in the text has largely been integrated into each token's own activation
by layer 14, so the "temporal" features are effectively extent-1 *in activation space* and a
per-token SAE is not blind to them at all. That is why its window-AUC is 0.72-0.79 rather
than the chance value a genuinely order-dependent factor would force.

If that account is right, the crosscoder's headroom is a function of DEPTH: early layers
hold more local, less-integrated representations, so cross-segment structure is still spread
across positions and a window code has something to capture that a per-token code cannot.

REGISTERED PREDICTIONS (written before the run):
  D1  The crosscoder's FVU penalty relative to the SAE SHRINKS at early layers. If the ratio
      is flat in depth, integration is not the obstruction and this account is wrong.
  D2  The crosscoder's window-AUC advantage over the SAE is largest at early layers, and the
      SAE's own window-AUC RISES with depth -- that rise is the direct signature of the
      model doing the temporal integration itself.
  D3  At the earliest layer the crosscoder beats the SAE on window-AUC. This is the outcome
      that would constitute a genuine advantage; D1 and D2 can both hold without it.

Held fixed across layers: corpus, windows (stride 1), budgets, d_sae, steps, optimiser.
The only thing that varies is which layer the activations come from.

    modal run experiments/temporal_screen/dict_bench/layer_modal.py
"""
import pathlib
import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

app = modal.App("dictbench-layer")
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
ELLS = [1, 2, 3, 6]


@app.function(gpu="A10G", image=image, timeout=21600)
def layersweep(model_id: str, layers: list, k_seg: int, T: int, n_docs: int,
               d_sae: int, steps: int, budgets: list, lr: float,
               general_frac: float, batch_win: int):
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
    d = model.config.hidden_size
    n_layers = len(layers_)
    print(f"[model] {model_id}  {n_layers} layers  d={d}", flush=True)

    # One text corpus, captured once at every layer, so nothing but depth varies.
    rng = random.Random(2718281)
    docs, segs, ells = [], [], []
    n_gen = int(n_docs * general_frac)
    for i in range(n_docs):
        if i < n_gen:
            sents = [GENERAL[rng.randrange(len(GENERAL))] for _ in range(k_seg)]
            lab, e_ = [-1] * k_seg, -1
        else:
            e_ = ELLS[rng.randrange(len(ELLS))]
            ph = rng.randrange(2 * e_)
            lab = [1 if ((t + ph) // e_) % 2 == 0 else 0 for t in range(k_seg)]
            sents = [(TENSE if l else CALM)[rng.randrange(10)] for l in lab]
        text, spans = CARRIERS[rng.randrange(len(CARRIERS))], []
        for j, s in enumerate(sents):
            if j:
                text += " "
            spans.append((len(text), len(text) + len(s)))
            text += s
        docs.append((text, spans)); segs.append(lab); ells.append(e_)
    segs = np.array(segs); ells = np.array(ells)
    print(f"[corpus] {len(docs)} documents of {k_seg} segments", flush=True)

    cap = {}

    def cap_hook(_m, _i, out):
        cap["h"] = (out[0] if isinstance(out, tuple) else out).detach()

    def capture_all(layer_idx):
        h = layers_[layer_idx].register_forward_hook(cap_hook)
        X = []
        for (text, spans) in docs:
            e = tok(text, return_tensors="pt", return_offsets_mapping=True)
            offs = e["offset_mapping"][0].tolist()
            ts = []
            for (a, b) in spans:
                idx = [i for i, (s0, s1) in enumerate(offs)
                       if s0 >= a and s1 <= b and s1 > s0]
                ts.append((idx[0], idx[-1]) if idx else (0, 0))
            with torch.no_grad():
                model(e["input_ids"].to(dev))
            hh = cap["h"][0].float().cpu().numpy()
            X.append(np.stack([hh[a:b + 1].mean(0) for a, b in ts]))
        h.remove()
        return np.stack(X)

    def auc_of(scores, y):
        if y.sum() == 0 or (1 - y).sum() == 0:
            return 0.5
        o = torch.argsort(scores)
        r = torch.empty_like(o, dtype=torch.float32)
        r[o] = torch.arange(len(scores), device=scores.device,
                            dtype=torch.float32) + 1
        n1, n0 = float(y.sum()), float((1 - y).sum())
        return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    def best_auc(Z, y):
        if Z.shape[0] == 0 or y.sum() == 0 or (1 - y).sum() == 0:
            return 0.5
        a = torch.tensor([auc_of(Z[:, j], y) for j in range(Z.shape[1])], device=dev)
        return float(0.5 + (a - 0.5).abs().max())

    out = {"model": model_id, "layers": layers, "k_seg": k_seg, "T": T,
           "d_sae": d_sae, "steps": steps, "rows": []}

    for Lx in layers:
        print(f"\n########## LAYER {Lx} / {n_layers} ##########", flush=True)
        Xr = capture_all(Lx)
        Xt = torch.tensor(Xr, dtype=torch.float32, device=dev)
        n_hold = max(int(0.15 * Xt.shape[0]), 32)
        Xtr, Xho = Xt[:-n_hold], Xt[-n_hold:]
        mu, sd = Xtr.mean((0, 1), keepdim=True), Xtr.std() + 1e-6
        Xn, Xn_ho = (Xtr - mu) / sd, (Xho - mu) / sd

        def unfold(A):
            return (A.unfold(1, T, 1).permute(0, 1, 3, 2)
                    .reshape(-1, T, d).contiguous())

        Wtr, Who = unfold(Xn), unfold(Xn_ho)
        flat_tr, flat_ho = Xn.reshape(-1, d), Who.reshape(-1, d)
        denom = float(flat_ho.var(0).sum())
        ell_ho = torch.tensor(ells[-n_hold:], device=dev)
        win_y = (ell_ho >= 3).long().unsqueeze(1).expand(-1, k_seg - T + 1).reshape(-1)
        win_keep = ell_ho.unsqueeze(1).expand(-1, k_seg - T + 1).reshape(-1) > 0

        def gen_win(bs):
            return Wtr[torch.randint(0, Wtr.shape[0], (bs,), device=dev)]

        def gen_flat(bs):
            return flat_tr[torch.randint(0, flat_tr.shape[0], (bs,), device=dev)]

        def adam_train(m, gen, bs):
            opt = torch.optim.Adam(m.parameters(), lr=lr)
            m.train()
            for s in range(steps):
                loss, _, _ = m(gen(bs))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step(); m._normalize_decoder()
            m.eval()

        for k in budgets:
            torch.manual_seed(0)
            m = TopKSAE(d_in=d, d_sae=d_sae, k=k).to(dev)
            adam_train(m, gen_flat, batch_win * T)
            with torch.no_grad():
                Zf = m.encode(flat_ho)
                xh = m.decode(Zf)
                sae_fvu = float(((xh - flat_ho) ** 2).sum(-1).mean() / denom)
                sae_l0 = float((Zf > 0).float().sum(-1).mean())
                sae_w = best_auc(Zf.reshape(-1, T, d_sae).mean(1)[win_keep],
                                 win_y[win_keep])

            torch.manual_seed(0)
            m = TemporalCrosscoder(d_in=d, d_sae=d_sae, T=T, k=k,
                                   activation="batchtopk").to(dev)
            with torch.no_grad():
                m._normalize_decoder()
            adam_train(m, gen_win, batch_win)
            with torch.no_grad():
                Zw = m.encode(Who)
                xh = m.decode(Zw)
                txc_fvu = float(((xh - Who) ** 2).sum(-1).mean() / denom)
                txc_l0 = float((Zw != 0).float().sum(-1).mean()) / T
                txc_w = best_auc(Zw[win_keep], win_y[win_keep])

            row = {"layer": Lx, "k": k, "sae_coeff_per_segment": sae_l0,
                   "txc_coeff_per_segment": txc_l0, "sae_fvu": sae_fvu,
                   "txc_fvu": txc_fvu, "fvu_ratio": txc_fvu / sae_fvu,
                   "sae_window_auc": sae_w, "txc_window_auc": txc_w,
                   "window_auc_delta": txc_w - sae_w}
            out["rows"].append(row)
            print(f"  k={k:<3} SAE {sae_l0:5.2f}/seg FVU {sae_fvu:.4f} winAUC {sae_w:.3f} "
                  f"| TXC {txc_l0:5.2f}/seg FVU {txc_fvu:.4f} winAUC {txc_w:.3f} "
                  f"| ratio {txc_fvu/sae_fvu:.2f}x  dAUC {txc_w - sae_w:+.3f}", flush=True)

    print("\n===== D1: does the crosscoder's FVU penalty shrink at early layers? =====",
          flush=True)
    for k in budgets:
        rs = [r for r in out["rows"] if r["k"] == k]
        print(f"  k={k:<3} " + "  ".join(
            f"L{r['layer']}:{r['fvu_ratio']:.2f}x" for r in rs), flush=True)

    print("\n===== D2/D3: window-AUC by depth (SAE rise = model integrating context) =====",
          flush=True)
    for k in budgets:
        rs = [r for r in out["rows"] if r["k"] == k]
        print(f"  k={k:<3} SAE " + "  ".join(
            f"L{r['layer']}:{r['sae_window_auc']:.3f}" for r in rs), flush=True)
        print(f"      TXC " + "  ".join(
            f"L{r['layer']}:{r['txc_window_auc']:.3f}" for r in rs), flush=True)
        print(f"      dAUC " + "  ".join(
            f"L{r['layer']}:{r['window_auc_delta']:+.3f}" for r in rs), flush=True)
    wins = [r for r in out["rows"] if r["window_auc_delta"] > 0]
    print(f"\n  crosscoder ahead on window-AUC in {len(wins)}/{len(out['rows'])} cells"
          + (": " + ", ".join(f"L{r['layer']}/k{r['k']}" for r in wins) if wins else ""),
          flush=True)
    return out


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-1.5B-Instruct", layers: str = "2,6,14,22",
         k_seg: int = 24, t: int = 12, n_docs: int = 400, d_sae: int = 4096,
         steps: int = 2000, budgets: str = "2,8", lr: float = 3e-4,
         general_frac: float = 0.3, batch_win: int = 32, tag: str = ""):
    import json
    r = layersweep.remote(model, [int(x) for x in layers.split(",")], k_seg, t,
                          n_docs, d_sae, steps,
                          [int(x) for x in budgets.split(",")], lr,
                          general_frac, batch_win)
    outdir = ROOT / "results" / "dict_bench"
    outdir.mkdir(parents=True, exist_ok=True)
    name = f"layer{tag}.json"
    (outdir / name).write_text(json.dumps(r, indent=2))
    print("[saved]", outdir / name)
