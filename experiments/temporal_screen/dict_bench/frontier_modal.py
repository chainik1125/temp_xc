"""Modal: the SAE-vs-TXC comparison frontier, and what actually sets TXC capacity.

WHY THIS REPLACES THE BENCHMARK. Nominal k is inert for the crosscoder: raising it 5x
(492 -> 2400 window budget) left realised L0 at ~18-27 while the ReLU-killed fraction rose
to 0.99. So every "matched k" comparison in this repo — including both pre-registered
protocols — has been matching a number that does not bind. Benchmarking two architectures
at a mismatched, unmeasured operating point cannot answer an architectural question.

TWO DELIVERABLES, ONE RUN.

1. THE FRONTIER. Segment-pooling makes the two architectures reconstruct the SAME objects,
   so FVU is directly commensurable with no fudging. The sparsity axis that the operator
   actually spends is COEFFICIENTS PER SEGMENT: L0_token for the SAE, L0_window / T for the
   crosscoder. Sweeping both and plotting FVU against that axis replaces point-matching
   with curve-comparison, and the capacity guess dissolves.

   Note the consequence of the realised-L0 finding: the crosscoder lives at ~1.5-2.2
   coefficients per segment, so its fair SAE comparator is k ~ 2, NOT the k = 100 every
   previous comparison used. The SAE sweep therefore goes down to k = 1.

2. THE PARAMETER REGIME. For every configuration we log what nominal k does not tell you:
     positive-preactivation fraction   (the true ceiling on realised L0)
     realised L0, per window and per segment
     ReLU-killed fraction of TopK picks
     alive-latent fraction
     final and mid-training loss        (is it still descending, i.e. undertrained?)
   plus a learning-rate arm, because if realised capacity is set by how many
   pre-activations training leaves positive, then LR and steps are the knobs and k is not.

Registered before running:
  - the crosscoder's realised L0 is capped by its positive-preactivation count, so
    configurations differing only in nominal k land on top of each other in the frontier;
  - the SAE at k ~ 2 is the honest comparator, and if the crosscoder does not beat it on
    FVU at matched coefficients/segment, the crosscoder has no representational advantage
    to convert into steering in the first place.

    modal run experiments/temporal_screen/dict_bench/frontier_modal.py
"""

import pathlib
import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

app = modal.App("dictbench-frontier")
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
def frontier(model_id: str, layer: int, k_seg: int, n_docs: int, d_sae: int,
             txc_batch: int, steps: int, sae_ks: list, txc_ks: list,
             lr_arms: list, general_frac: float, init_norm: bool = False):
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
            ix = [i for i, (x, y) in enumerate(offs) if y > x and y > a and x < b]
            ts.append((min(ix), max(ix)))
        h = layers_[L].register_forward_hook(cap_hook)
        with torch.no_grad():
            model(e["input_ids"].to(dev))
        h.remove()
        hh = cap["h"][0].float().cpu().numpy()
        return np.stack([hh[a:b + 1].mean(0) for a, b in ts])

    # ---------------- cache ----------------
    X, labels = [], []
    n_gen = int(n_docs * general_frac)
    for i in range(n_docs):
        if i < n_gen:
            sents = [GENERAL[rng.randrange(len(GENERAL))] for _ in range(k_seg)]
            lab = [-1] * k_seg
        else:
            lab = [rng.randint(0, 1) for _ in range(k_seg)]
            sents = [(TENSE if l else CALM)[rng.randrange(10)] for l in lab]
        X.append(capture(*build(rng.choice(CARRIERS), sents))); labels.append(lab)
        if (i + 1) % 400 == 0:
            print(f"   [cache] {i+1}/{n_docs}")
    X = np.stack(X); labels = np.array(labels)
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    n_hold = max(int(0.15 * Xt.shape[0]), 32)
    Xtr, Xho = Xt[:-n_hold], Xt[-n_hold:]
    mu, sd = Xtr.mean((0, 1), keepdim=True), Xtr.std() + 1e-6
    Xn, Xn_ho = (Xtr - mu) / sd, (Xho - mu) / sd
    flat_ho = Xn_ho.reshape(-1, d)
    denom = float(flat_ho.var(0).sum())
    lab_ho = torch.tensor(labels[-n_hold:], device=dev).reshape(-1)
    keep = lab_ho >= 0
    print(f"[cache] train {tuple(Xn.shape)} holdout {tuple(Xn_ho.shape)}")

    def gen_w(bs): return Xn[torch.randint(0, Xn.shape[0], (bs,), device=dev)]

    def gen_f(bs):
        i = torch.randint(0, Xn.shape[0], (bs,), device=dev)
        j = torch.randint(0, k_seg, (bs,), device=dev)
        return Xn[i, j]

    def train(m, gen, bs, lr, tag):
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        hist = []
        for s in range(steps):
            loss, _, _ = m(gen(bs))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); m._normalize_decoder()
            if s % max(steps // 5, 1) == 0 or s == steps - 1:
                hist.append((s, float(loss.item())))
        print(f"     [{tag}] loss " + " -> ".join(f"{v:.2f}" for _, v in hist))
        return hist

    def auc_of(scores, y):
        s_, y_ = scores[keep], y[keep]
        if y_.sum() == 0 or (1 - y_).sum() == 0:
            return 0.5
        o = torch.argsort(s_)
        r = torch.empty_like(o, dtype=torch.float32)
        r[o] = torch.arange(len(s_), device=s_.device, dtype=torch.float32) + 1
        n1, n0 = float(y_.sum()), float((1 - y_).sum())
        return float((r[y_ == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    def best_auc(Z):
        y = lab_ho.float()
        a = torch.tensor([auc_of(Z[:, j], y) for j in range(Z.shape[1])], device=dev)
        dv = (a - 0.5).abs()
        return float(a[int(dv.argmax())]), int((dv >= 0.95 * float(dv.max())).sum())

    out = {"model": model_id, "layer": int(L), "k_seg": k_seg, "d_sae": d_sae,
           "steps": steps, "sae": [], "txc": []}

    # ================= SAE frontier, down to k=1 =================
    print("\n===== SAE frontier =====")
    for k in sae_ks:
        torch.manual_seed(0)
        m = TopKSAE(d_in=d, d_sae=d_sae, k=k).to(dev)
        hist = train(m, gen_f, txc_batch * k_seg, 1e-3, f"sae-k{k}")
        with torch.no_grad():
            pre = (flat_ho - m.b_dec) @ m.W_enc.T + m.b_enc
            posfrac = float((pre > 0).float().mean())
            tv, _ = pre.topk(k, dim=-1)
            relu_kill = float((tv <= 0).float().mean())
            z = m.encode(flat_ho); xh = m.decode(z)
            fvu = float(((xh - flat_ho) ** 2).sum(-1).mean() / denom)
            l0 = float((z > 0).float().sum(-1).mean())
            alive = float(((z > 0).float().mean(0) >= 0.001).float().mean())
            ba, nsplit = best_auc(z)
        rec = {"nominal_k": k, "coeff_per_segment": l0, "fvu": fvu,
               "realised_l0_token": l0, "alive_frac": alive,
               "pos_preact_frac": posfrac, "relu_kill_frac": relu_kill,
               "best_latent_auc": ba, "n_latents_within_95pct": nsplit,
               "loss_hist": hist}
        out["sae"].append(rec)
        print(f"  k={k:>4}: coeff/seg {l0:6.2f}  FVU {fvu:.4f}  alive {alive:.3f}  "
              f"pos-preact {posfrac:.3f}  ReLU-kill {relu_kill:.3f}  AUC {ba:.3f}")

    # ================= TXC frontier + LR arm =================
    print("\n===== TXC frontier =====")
    for lr in lr_arms:
        for kp in txc_ks:
            if kp * k_seg > d_sae:
                continue
            torch.manual_seed(0)
            m = TemporalCrosscoder(d_in=d, d_sae=d_sae, T=k_seg, k=kp).to(dev)
            if init_norm:
                # TopKSAE normalises its decoder in __init__; TemporalCrosscoder does not,
                # so its atoms start at norm sqrt(T*d_in/d_sae) and are rescaled only after
                # the first optimiser step. See initnorm_modal.py for the attribution.
                with torch.no_grad():
                    m._normalize_decoder()
            hist = train(m, gen_w, txc_batch, lr, f"txc-k{kp}-lr{lr}")
            with torch.no_grad():
                pre = torch.einsum("btd,tds->bs", Xn_ho, m.W_enc) + m.b_enc
                posfrac = float((pre > 0).float().mean())
                tv, _ = pre.topk(m.k, dim=-1)
                relu_kill = float((tv <= 0).float().mean())
                z = m.encode(Xn_ho); xh = m.decode(z)
                fvu = float(((xh - Xn_ho) ** 2).sum(-1).mean() / denom)
                l0 = float((z > 0).float().sum(-1).mean())
                alive = float(((z > 0).float().mean(0) >= 0.001).float().mean())
                ba, nsplit = best_auc(z.repeat_interleave(k_seg, dim=0))
            rec = {"nominal_k_per_pos": kp, "nominal_window_k": kp * k_seg, "lr": lr,
                   "coeff_per_segment": l0 / k_seg, "realised_l0_window": l0,
                   "fvu": fvu, "alive_frac": alive, "pos_preact_frac": posfrac,
                   "relu_kill_frac": relu_kill, "best_latent_auc": ba,
                   "n_latents_within_95pct": nsplit, "loss_hist": hist}
            out["txc"].append(rec)
            print(f"  kper={kp:>4} lr={lr:g}: coeff/seg {l0/k_seg:6.2f}  FVU {fvu:.4f}  "
                  f"alive {alive:.3f}  pos-preact {posfrac:.3f}  "
                  f"ReLU-kill {relu_kill:.3f}  AUC {ba:.3f}")

    # ---- the headline comparison: FVU at matched coefficients/segment ----
    print("\n===== matched coefficients/segment =====")
    for t in out["txc"]:
        near = min(out["sae"], key=lambda s_: abs(s_["coeff_per_segment"]
                                                  - t["coeff_per_segment"]))
        print(f"  TXC kper={t['nominal_k_per_pos']} lr={t['lr']:g} "
              f"({t['coeff_per_segment']:.2f}/seg, FVU {t['fvu']:.3f})  vs  "
              f"SAE k={near['nominal_k']} ({near['coeff_per_segment']:.2f}/seg, "
              f"FVU {near['fvu']:.3f})  -> "
              f"{'TXC better' if t['fvu'] < near['fvu'] else 'SAE better'}")
    return out


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-1.5B-Instruct", layer: int = -1, k_seg: int = 12,
         n_docs: int = 1200, d_sae: int = 4096, txc_batch: int = 64,
         steps: int = 2500, sae_ks: str = "1,2,4,8,16,32,64,128",
         txc_ks: str = "1,2,4,8,20,41", lr_arms: str = "1e-3,3e-4",
         general_frac: float = 0.4, init_norm: bool = False, tag: str = ""):
    import json
    r = frontier.remote(model, layer, k_seg, n_docs, d_sae, txc_batch, steps,
                        [int(x) for x in sae_ks.split(",")],
                        [int(x) for x in txc_ks.split(",")],
                        [float(x) for x in lr_arms.split(",")], general_frac,
                        init_norm)
    r["init_norm"] = init_norm
    outdir = ROOT / "results" / "dict_bench"
    outdir.mkdir(parents=True, exist_ok=True)
    name = f"frontier{tag}.json"
    (outdir / name).write_text(json.dumps(r, indent=2))
    print("[saved]", outdir / name)
