"""Modal: can the crosscoder be trained to health, and does steering change when it is?

THE PROBLEM THIS ADDRESSES. Held-out per-segment FVU came back **0.0297 for the SAE and
0.8380 for the TXC** — the crosscoder explains ~16% of held-out variance where the SAE
explains ~97% — with realised L0 of 18.5 per WINDOW against the SAE's 99 per TOKEN
(1188 per window). Nominal window k was 492, so ReLU is zeroing ~96% of what TopK selects.
That is a starved dictionary, not a sparse one.

A steering comparison between a healthy SAE and a starved TXC cannot settle an
architectural question. Either the crosscoder can be brought to comparable reconstruction
and the steering result survives — in which case the negative is about the architecture —
or it cannot, and the honest headline narrows to "this crosscoder, trained this way".

WHAT THIS RUN DOES. Sweep the crosscoder's per-position k and training length, and for
each configuration report the things that decide the question:

    realised L0 per window and per segment      (is it actually using latents?)
    held-out per-segment FVU                    (is it reconstructing?)
    alive-latent fraction                       (is it collapsed?)
    dead-on-arrival fraction of TopK picks      (how much is ReLU eating?)
    frozen-arm steering fidelity                (does health buy steering?)

The SAE is trained once, unchanged, as the fixed reference. Everything is on one cache
with matched token-activations per step, and the eval profile is chosen a-orthogonal to
the measured per-slot weights so the constant-write floor is ~0.

Registered before running: if a configuration reaches FVU within 2x of the SAE's and its
frozen steering fidelity stays in the same band as the starved one, the negative
generalises. If steering fidelity rises materially with health, the earlier comparison was
measuring training quality rather than architecture, and must be reported that way.

    modal run experiments/temporal_screen/dict_bench/health_modal.py
"""

import pathlib
import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

app = modal.App("dictbench-health")
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
def health(model_id: str, layer: int, k_seg: int, n_docs: int, d_sae: int,
           txc_batch: int, n_sel: int, n_test: int, frac: float,
           general_frac: float, configs: list, pool: int):
    import sys
    sys.path.insert(0, "/work")
    import itertools
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
    rng = random.Random(5150)

    def build(car, sents):
        text, spans = car, []
        for j, s in enumerate(sents):
            if j:
                text += " "
            spans.append((len(text), len(text) + len(s)))
            text += s
        return text, spans

    def enc_txt(text, cs):
        e = tok(text, return_tensors="pt", return_offsets_mapping=True)
        offs = e["offset_mapping"][0].tolist()
        ts = []
        for (a, b) in cs:
            ix = [i for i, (x, y) in enumerate(offs) if y > x and y > a and x < b]
            ts.append((min(ix), max(ix)))
        return e["input_ids"].to(dev), ts

    def capture(text, cs):
        ids, ts = enc_txt(text, cs)
        h = layers_[L].register_forward_hook(cap_hook)
        with torch.no_grad():
            model(ids)
        h.remove()
        hh = cap["h"][0].float().cpu().numpy()
        return (np.stack([hh[a:b + 1].mean(0) for a, b in ts]),
                [float(np.linalg.norm(hh[p])) for a, b in ts for p in range(a, b + 1)])

    def seg_lp(ids, ts):
        with torch.no_grad():
            lp = model(ids).logits[0].log_softmax(-1).float()
        return float(sum(lp[p - 1, ids[0, p]]
                         for a, b in ts for p in range(a, b + 1) if p >= 1))

    X, labels, norms = [], [], []
    n_gen = int(n_docs * general_frac)
    for i in range(n_docs):
        if i < n_gen:
            sents = [GENERAL[rng.randrange(len(GENERAL))] for _ in range(k_seg)]
            lab = [-1] * k_seg
        else:
            lab = [rng.randint(0, 1) for _ in range(k_seg)]
            sents = [(TENSE if l else CALM)[rng.randrange(10)] for l in lab]
        s_, n_ = capture(*build(rng.choice(CARRIERS), sents))
        X.append(s_); labels.append(lab); norms += n_
    X = np.stack(X); labels = np.array(labels); base_norm = float(np.mean(norms))
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    n_hold = max(int(0.15 * Xt.shape[0]), 32)
    Xtr, Xho = Xt[:-n_hold], Xt[-n_hold:]
    mu, sd = Xtr.mean((0, 1), keepdim=True), Xtr.std() + 1e-6
    Xn, Xn_ho = (Xtr - mu) / sd, (Xho - mu) / sd
    task = labels[:len(Xtr)][:, 0] >= 0
    fl = torch.tensor(labels[:len(Xtr)][task].reshape(-1), device=dev)
    fa = Xn[task].reshape(-1, d)
    u_dom = (fa[fl == 1].mean(0) - fa[fl == 0].mean(0)); u_dom = u_dom / u_dom.norm()
    flat_ho = Xn_ho.reshape(-1, d)
    print(f"[cache] train {tuple(Xn.shape)} holdout {tuple(Xn_ho.shape)}")

    def gen_w(bs): return Xn[torch.randint(0, Xn.shape[0], (bs,), device=dev)]

    def gen_f(bs):
        i = torch.randint(0, Xn.shape[0], (bs,), device=dev)
        j = torch.randint(0, k_seg, (bs,), device=dev)
        return Xn[i, j]

    def train(m, gen, bs, steps, tag):
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for s in range(steps):
            loss, _, _ = m(gen(bs))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); m._normalize_decoder()
            if s % max(steps // 2, 1) == 0 or s == steps - 1:
                print(f"     [{tag}] {s}/{steps} loss={loss.item():.3f}")
        return m

    # ---- fixed SAE reference ----
    torch.manual_seed(0)
    sae = TopKSAE(d_in=d, d_sae=d_sae, k=100).to(dev)
    train(sae, gen_f, txc_batch * k_seg, 2500, "sae")
    with torch.no_grad():
        z = sae.encode(flat_ho); xh = sae.decode(z)
        fvu_sae = float(((xh - flat_ho) ** 2).sum(-1).mean() / flat_ho.var(0).sum())
        l0_sae = float((z > 0).float().sum(-1).mean())
    print(f"[SAE reference] FVU {fvu_sae:.4f}  L0 {l0_sae:.1f}/token")

    # ---- eval scaffolding on an a-orthogonal profile ----
    tgt = frac * base_norm * (k_seg ** 0.5)
    rescale = lambda W: W * (tgt / (W.norm() + 1e-9))

    def make_pairs(prof01, n):
        out = []
        for _ in range(n):
            foil = prof01[:]
            for _ in range(80):
                rng.shuffle(foil)
                if foil != prof01:
                    break
            ti = [rng.randrange(10) for _ in range(k_seg)]
            ci = [rng.randrange(10) for _ in range(k_seg)]
            car = rng.choice(CARRIERS)

            def sf(p):
                o, a, b = [], 0, 0
                for l in p:
                    if l:
                        o.append(TENSE[ti[a]]); a += 1
                    else:
                        o.append(CALM[ci[b]]); b += 1
                return o
            tT, cT = build(car, sf(prof01)); tF, cF = build(car, sf(foil))
            out.append((enc_txt(tT, cT), enc_txt(tF, cF)))
        return out

    def margin(pair, W):
        ids, ts = pair
        steer["v"] = []
        if W is not None:
            for i in range(k_seg):
                a, b = ts[i]
                steer["v"].append((max(a - 1, 0), max(b - 1, 0), W[i]))
        v = seg_lp(ids, ts)
        steer["v"] = []
        return v

    def run(pairs, W, base):
        return np.array([(margin(t, W) - margin(f, W)) - b
                         for (t, f), b in zip(pairs, base)])

    a_t = np.zeros(k_seg)
    for _ in range(6):
        p = [1] * (k_seg // 2) + [0] * (k_seg // 2); rng.shuffle(p)
        prs = make_pairs(p, 4)
        bs_ = [margin(t, None) - margin(f, None) for t, f in prs]
        pi_ = np.array([1.0 if l else -1.0 for l in p])
        for t in range(k_seg):
            W = torch.zeros(k_seg, d, device=dev); W[t] = float(pi_[t]) * u_dom
            a_t[t] += run(prs, rescale(W), bs_).mean() / 6
    best = None
    for combo in itertools.combinations(range(k_seg), k_seg // 2):
        pi_ = -np.ones(k_seg); pi_[list(combo)] = 1.0
        f_ = abs(float((a_t * pi_).sum() / a_t.sum()))
        if best is None or f_ < best[0]:
            best = (f_, [1 if pi_[t] > 0 else 0 for t in range(k_seg)])
    floor_pred, prof01 = best
    Y = torch.stack([(1.0 if l else -1.0) * u_dom for l in prof01])
    sel_pairs, test_pairs = make_pairs(prof01, n_sel), make_pairs(prof01, n_test)
    b_sel = [margin(t, None) - margin(f, None) for t, f in sel_pairs]
    b_tst = [margin(t, None) - margin(f, None) for t, f in test_pairs]
    full = run(test_pairs, rescale(Y), b_tst)
    print(f"[profile] {prof01} floor {floor_pred:+.4f}   full DoM Δ={full.mean():+.2f}")

    out = {"model": model_id, "k_seg": k_seg, "d_sae": d_sae, "base_norm": base_norm,
           "a_t": a_t.tolist(), "profile": prof01, "floor_predicted": floor_pred,
           "full_dom": float(full.mean()),
           "sae_reference": {"fvu": fvu_sae, "l0_per_token": l0_sae,
                             "l0_per_window": l0_sae * k_seg},
           "configs": []}

    for kper, steps in configs:
        if kper * k_seg > d_sae:
            print(f"\n### SKIP kper={kper}: k*T={kper*k_seg} > d_sae={d_sae}")
            continue
        print(f"\n### TXC kper={kper} (nominal window {kper*k_seg}) steps={steps}")
        torch.manual_seed(0)
        txc = TemporalCrosscoder(d_in=d, d_sae=d_sae, T=k_seg, k=kper).to(dev)
        train(txc, gen_w, txc_batch, steps, f"txc-k{kper}")
        with torch.no_grad():
            zt = txc.encode(Xn_ho); xht = txc.decode(zt)
            fvu = float(((xht - Xn_ho) ** 2).sum(-1).mean() / flat_ho.var(0).sum())
            l0 = float((zt > 0).float().sum(-1).mean())
            alive = float(((zt > 0).float().mean(0) >= 0.001).float().mean())
            pre = torch.einsum("btd,tds->bs", Xn_ho, txc.W_enc) + txc.b_enc
            tv, _ = pre.topk(txc.k, dim=-1)
            relu_kill = float((tv <= 0).float().mean())
            P = txc.W_dec.data.float()
            cands = (P.reshape(P.shape[0], -1) @ Y.reshape(-1)).abs().topk(pool).indices.tolist()
        sc = [float(run(sel_pairs, rescale(P[j]), b_sel).mean()) for j in cands]
        bi = int(np.argmax(np.abs(sc))); j = cands[bi]; sg = float(np.sign(sc[bi]))
        ds = run(test_pairs, rescale(sg * P[j]), b_tst)
        fid = float(ds.mean() / full.mean())
        rec = {"kper": kper, "nominal_window_k": kper * k_seg, "steps": steps,
               "fvu": fvu, "fvu_ratio_to_sae": fvu / fvu_sae,
               "l0_per_window": l0, "l0_per_segment": l0 / k_seg,
               "alive_frac": alive, "relu_kill_frac": relu_kill,
               "frozen_latent": int(j), "frozen_fidelity": fid,
               "frozen_delta": float(ds.mean()),
               "frozen_sem": float(ds.std(ddof=1) / np.sqrt(len(ds)))}
        out["configs"].append(rec)
        print(f"   FVU {fvu:.4f} ({fvu/fvu_sae:.1f}x SAE)  L0 {l0:.1f}/window "
              f"({l0/k_seg:.1f}/segment)  alive {alive:.3f}  ReLU-killed {relu_kill:.2f}")
        print(f"   frozen steering fidelity {fid:+.3f}  (latent {j})")
    return out


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-1.5B-Instruct", layer: int = -1, k_seg: int = 12,
         n_docs: int = 1200, d_sae: int = 4096, txc_batch: int = 64, n_sel: int = 8,
         n_test: int = 20, frac: float = 0.35, general_frac: float = 0.4,
         pool: int = 16,
         configs: str = "41:2500,100:2500,200:5000,341:8000"):
    import json
    cfg = [(int(c.split(":")[0]), int(c.split(":")[1])) for c in configs.split(",")]
    r = health.remote(model, layer, k_seg, n_docs, d_sae, txc_batch, n_sel, n_test,
                      frac, general_frac, cfg, pool)
    outdir = ROOT / "results" / "dict_bench"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "health.json").write_text(json.dumps(r, indent=2))
    print("[saved]", outdir / "health.json")
