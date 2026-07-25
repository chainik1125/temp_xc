"""Modal: does a temporal crosscoder span steering trajectories better than a TopK SAE?

THE MEASUREMENT. A steering write over k segments is a matrix Y in R^(k x d). Each
dictionary offers a different *subspace* of such matrices, and one scalar buys a different
thing in each:

    SAE, broadcast      one direction held constant over all segments   1_k (x) v_j
    SAE, per-position   one direction at one segment                    e_t (x) v_j
    TXC                 one learned k x d pattern                       P_j

So the question is not "which direction is better" but **which dictionary spans target
trajectories with fewer scalars**. We measure

    fidelity(m) = Δmargin(best write from m scalars) / Δmargin(the full ground-truth
                  schedule Y)

sweeping m, giving each architecture its BEST allocation of the m scalars (greedy
orthogonal matching pursuit over its own atoms, coefficients free and signed — which also
removes any need to orient decoder rows by hand). The SAE is therefore given the strongest
form of itself, including the per-position allocation that a practitioner who hand-schedules
coefficients would use.

FAIRNESS CONTROLS, all pre-registered before running:

  * MATCHED TRAINING BUDGET. At equal batch size the TXC sees T x more token-activations
    per step than the SAE (gen_flat yields B tokens, gen_windows yields B*T token-slots)
    and nothing in the harness corrects it. This is the same class of bug as the 2026-05-05
    purified-sampling fix in this repo, where an SAE got ~25x more tokens/step than the TXC
    it was compared against. Here the SAE's batch is T x the TXC's, so both consume the
    same number of token-activations per step, and both totals are reported.
  * MATCHED INJECTED NORM. TopKSAE normalises decoder COLUMNS (norm 1); TemporalCrosscoder
    normalises the whole (T, d) slab (rows ~1/sqrt(T)). Every final write is rescaled to
    one fixed total norm, and realised norms are reported.
  * BOTH SPARSITY PROTOCOLS, which are distinct only at T != 5. At T=12:
    A -> per-position k=100 (window 1200);  B -> per-position k=41 (window 492); SAE k=100.
  * THE DoM REFERENCE IS THE CEILING, NOT AN ARM. Difference-of-means is fit on labels
    while the dictionaries are unsupervised, so it defines Y (fidelity 1.0 by construction)
    rather than competing.

Registered prediction: TXC reaches a given fidelity with fewer scalars than SAE on
structured targets, converging as m grows. Registered falsifier: if SAE-per-position
matches TXC at equal m, the dictionary claim fails and only "a schedule beats a level"
survives from the previous sprint.

    modal run experiments/temporal_screen/dict_bench/bench_modal.py
"""

import pathlib
import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

app = modal.App("dictbench-main")
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
def bench(model_id: str, layer: int, k_seg: int, n_docs: int, d_sae: int,
          steps: int, txc_batch: int, ms: list, n_eval: int, frac: float,
          general_frac: float, protocols: list):
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
    print(f"[cfg] L={L} d_model={d_model} k_seg={k_seg} d_sae={d_sae}")

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
    rng = random.Random(4242)

    def build(car, sents):
        text, spans = car, []
        for j, s in enumerate(sents):
            if j:
                text += " "
            spans.append((len(text), len(text) + len(s)))
            text += s
        return text, spans

    def encode_txt(text, cs):
        enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
        offs = enc["offset_mapping"][0].tolist()
        ts = []
        for (a, b) in cs:
            ix = [i for i, (x, y) in enumerate(offs) if y > x and y > a and x < b]
            ts.append((min(ix), max(ix)))
        return enc["input_ids"].to(dev), ts

    def capture(text, cs):
        ids, ts = encode_txt(text, cs)
        h = layers_[L].register_forward_hook(cap_hook)
        with torch.no_grad():
            model(ids)
        h.remove()
        hh = cap["h"][0].float().cpu().numpy()
        return (np.stack([hh[a:b + 1].mean(0) for a, b in ts]),
                [float(np.linalg.norm(hh[p])) for a, b in ts for p in range(a, b + 1)])

    def seg_logprob(ids, ts):
        with torch.no_grad():
            lp = model(ids).logits[0].log_softmax(-1).float()
        return float(sum(lp[p - 1, ids[0, p]]
                         for a, b in ts for p in range(a, b + 1) if p >= 1))

    # ================= activation cache =================
    X, labels, norms_all = [], [], []
    n_gen = int(n_docs * general_frac)
    for i in range(n_docs):
        if i < n_gen:
            sents = [GENERAL[rng.randrange(len(GENERAL))] for _ in range(k_seg)]
            lab = [-1] * k_seg
        else:
            lab = [rng.randint(0, 1) for _ in range(k_seg)]
            sents = [(TENSE if l else CALM)[rng.randrange(10)] for l in lab]
        segs, nn_ = capture(*build(rng.choice(CARRIERS), sents))
        X.append(segs); labels.append(lab); norms_all += nn_
        if (i + 1) % 500 == 0:
            print(f"   [cache] {i+1}/{n_docs}")
    X = np.stack(X); labels = np.array(labels)
    base_norm = float(np.mean(norms_all))
    print(f"[cache] X={X.shape} base_norm={base_norm:.1f} general={n_gen/n_docs:.2f}")

    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    mu, sd = Xt.mean((0, 1), keepdim=True), Xt.std() + 1e-6
    Xn = (Xt - mu) / sd

    # ================= the ground-truth target Y (DoM, the ceiling) =================
    task = labels[:, 0] >= 0
    flat_lab = torch.tensor(labels[task].reshape(-1), device=dev)
    flat_act = Xn[task].reshape(-1, d_model)
    u_dom = (flat_act[flat_lab == 1].mean(0) - flat_act[flat_lab == 0].mean(0))
    u_dom = u_dom / u_dom.norm()
    print(f"[target] DoM direction from {int(task.sum())} labelled docs")

    # ================= train both dictionaries, matched token budget =================
    def gen_window(bs):
        return Xn[torch.randint(0, Xn.shape[0], (bs,), device=dev)]

    def gen_flat(bs):
        i = torch.randint(0, Xn.shape[0], (bs,), device=dev)
        j = torch.randint(0, k_seg, (bs,), device=dev)
        return Xn[i, j]

    def train(m, gen, bs, tag):
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for s in range(steps):
            loss, _, z = m(gen(bs))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); m._normalize_decoder()
            if s % max(steps // 4, 1) == 0 or s == steps - 1:
                print(f"   [{tag}] {s:5d}/{steps} loss={loss.item():.4f} "
                      f"L0={(z > 0).float().sum(-1).mean().item():.1f}")
        return m

    def protocol_k(name):
        # values verified against src/bench/saebench/configs.py
        return 100 if name == "A" else max(1, 500 // k_seg)

    results = {"model": model_id, "layer": int(L), "k_seg": k_seg, "d_sae": d_sae,
               "steps": steps, "base_norm": base_norm, "cache_shape": list(X.shape),
               "general_frac": general_frac, "protocols": {}}

    sae_batch = txc_batch * k_seg           # equal token-activations per step
    print(f"[budget] TXC batch {txc_batch} windows = {txc_batch*k_seg} token-acts/step; "
          f"SAE batch {sae_batch} tokens = {sae_batch} token-acts/step")

    for pname in protocols:
        kper = protocol_k(pname)
        print(f"\n===== PROTOCOL {pname}: TXC per-pos k={kper} (window {kper*k_seg}), "
              f"SAE k=100 =====")
        if kper * k_seg > d_sae:
            print(f"   SKIP: k*T={kper*k_seg} > d_sae={d_sae}")
            continue
        torch.manual_seed(0)
        sae = TopKSAE(d_in=d_model, d_sae=d_sae, k=100).to(dev)
        torch.manual_seed(0)
        txc = TemporalCrosscoder(d_in=d_model, d_sae=d_sae, T=k_seg, k=kper).to(dev)
        train(sae, gen_flat, sae_batch, f"sae-{pname}")
        train(txc, gen_window, txc_batch, f"txc-{pname}")

        with torch.no_grad():
            V = sae.W_dec.data.float()                       # (d, h) unit columns
            P = txc.W_dec.data.float()                       # (h, T, d) unit slabs
        print(f"[rows] |sae col|={V[:, 0].norm():.3f}  "
              f"|txc slab|={P[0].norm():.3f} row={P[0].norm(dim=1).mean():.3f}")

        # ---- atom sets as flattened (T*d) vectors ----
        Td = k_seg * d_model
        A_txc = P.reshape(P.shape[0], Td)                                  # (h, T*d)
        ones = torch.ones(k_seg, 1, device=dev)
        A_bcast = (ones * V.T.unsqueeze(1)).reshape(V.shape[1], Td)        # (h, T*d)

        def omp(Y, atoms, m, per_pos_V=None):
            """Greedy orthogonal matching pursuit; returns the m-atom LS approximation."""
            r = Y.clone(); idx = []
            for _ in range(m):
                if per_pos_V is None:
                    corr = (atoms @ r) / (atoms.norm(dim=1) + 1e-9)
                    j = int(corr.abs().argmax())
                    if j in idx:
                        corr[j] = 0; j = int(corr.abs().argmax())
                    idx.append(j)
                    B = atoms[idx]
                else:                                   # (latent, position) atoms
                    R = r.reshape(k_seg, d_model)
                    sc = (R @ per_pos_V)                # (T, h)
                    flat = sc.abs().flatten()
                    for cand in torch.argsort(flat, descending=True):
                        t, j = int(cand) // sc.shape[1], int(cand) % sc.shape[1]
                        if (t, j) not in idx:
                            idx.append((t, j)); break
                    B = torch.zeros(len(idx), Td, device=dev)
                    for q, (t, j) in enumerate(idx):
                        B[q].view(k_seg, d_model)[t] = per_pos_V[:, j]
                c, *_ = torch.linalg.lstsq(B.T, Y.unsqueeze(1))
                approx = (c.squeeze(1) @ B)
                r = Y - approx
            return approx

        # ---- eval pairs (random balanced profiles) ----
        pairs, Ys = [], []
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
            pairs.append((encode_txt(tT, cT), encode_txt(tF, cF)))
            Ys.append(torch.stack([(1.0 if l else -1.0) * u_dom for l in prof]))

        def margin(pair, W):
            ids, ts = pair
            steer["v"] = []
            if W is not None:
                for i in range(k_seg):
                    a, b = ts[i]
                    steer["v"].append((max(a - 1, 0), max(b - 1, 0), W[i]))
            val = seg_logprob(ids, ts)
            steer["v"] = []
            return val

        target_norm = frac * base_norm * (k_seg ** 0.5)

        def rescale(W):
            return W * (target_norm / (W.norm() + 1e-9))

        base = [margin(t, None) - margin(f, None) for t, f in pairs]
        full = np.array([(margin(t, rescale(Ys[j])) - margin(f, rescale(Ys[j]))) - b
                         for j, ((t, f), b) in enumerate(zip(pairs, base))])
        print(f"[ref ] full DoM schedule Δmargin = {full.mean():+.2f}")

        prow = {"protocol": pname, "txc_k_per_pos": kper,
                "txc_k_window": kper * k_seg, "sae_k": 100,
                "tokens_per_step_txc": txc_batch * k_seg,
                "tokens_per_step_sae": sae_batch,
                "full_dom_margin": float(full.mean()),
                "full_dom_sem": float(full.std(ddof=1) / np.sqrt(len(full))),
                "arms": {}}
        for arm in ("txc", "sae_broadcast", "sae_perpos"):
            prow["arms"][arm] = {}
            for m in ms:
                ds, cosines = [], []
                for j, ((t, f), b) in enumerate(zip(pairs, base)):
                    Y = Ys[j].reshape(-1)
                    if arm == "txc":
                        ap = omp(Y, A_txc, m)
                    elif arm == "sae_broadcast":
                        ap = omp(Y, A_bcast, m)
                    else:
                        ap = omp(Y, None, m, per_pos_V=V)
                    W = rescale(ap.reshape(k_seg, d_model))
                    cosines.append(float(torch.nn.functional.cosine_similarity(
                        ap, Y, dim=0)))
                    ds.append((margin(t, W) - margin(f, W)) - b)
                ds = np.array(ds)
                fid = float(ds.mean() / full.mean()) if full.mean() else 0.0
                prow["arms"][arm][m] = {
                    "delta": float(ds.mean()),
                    "sem": float(ds.std(ddof=1) / np.sqrt(len(ds))),
                    "fidelity": fid,
                    "recon_cos": float(np.mean(cosines))}
                print(f"   {arm:14} m={m:3d}: Δ={ds.mean():+7.2f} "
                      f"fid={fid:+.3f} recon_cos={np.mean(cosines):.3f}")
        results["protocols"][pname] = prow
    return results


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-1.5B-Instruct", layer: int = -1, k_seg: int = 12,
         n_docs: int = 1500, d_sae: int = 4096, steps: int = 3000,
         txc_batch: int = 64, ms: str = "1,2,4,8,16,32", n_eval: int = 20,
         frac: float = 0.35, general_frac: float = 0.4, protocols: str = "A,B"):
    import json
    res = bench.remote(model, layer, k_seg, n_docs, d_sae, steps, txc_batch,
                       [int(x) for x in ms.split(",")], n_eval, frac, general_frac,
                       protocols.split(","))
    outdir = ROOT / "results" / "dict_bench"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "bench.json").write_text(json.dumps(res, indent=2))
    print("[saved]", outdir / "bench.json")
