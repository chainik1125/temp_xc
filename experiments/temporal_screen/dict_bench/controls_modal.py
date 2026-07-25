"""Modal: is a TXC advantage about LEARNED TEMPORAL STRUCTURE, or just coverage?

The pre-registration names the single most likely false positive:

    "The m-sweep will show TXC beating SAE at small m because the TXC arm writes every
     slot and the SAE arm writes only m of them, and the effect is essentially additive
     over slots with per-slot weights spanning 4x. It will look like the hypothesis
     confirming."

That is correct and it is built into the budget definition, so it cannot be argued away —
it has to be measured out. This job adds the arms that do it. Everything is on one cache,
one set of eval pairs, one injected-norm budget.

THE NULLS THAT ISOLATE LEARNING (both write all k slots for one scalar, exactly like the
real TXC, and differ from it only in whether the temporal pattern means anything):

  txc_random    atoms are RANDOM (k x d) unit-norm slabs. Identical coverage per scalar,
                identical norm structure, zero learned structure. If the real TXC does not
                beat this, its advantage is geometry and coverage, not temporal learning.
  txc_shuffled  the LEARNED atoms with their k rows permuted in time. Per-slot direction
                content and every norm preserved exactly; only the temporal arrangement is
                destroyed. This is the sharpest of the two.

THE ARM THAT MATCHES SUPPORT (so m never buys coverage):

  sae_fullsupp  m latents, each with its own per-slot coefficient => writes all k slots.
                Its honest scalar cost is m*k, not m, and it is reported that way. This is
                the "scheduled SAE" a practitioner would actually build, and it is the
                strongest form of the baseline.

AND THE ADDITIVE CHECK: per-slot weights a_t are measured inline (steer each slot alone),
then every arm's write gets its additive prediction sum_t a_t <w_t, u> computed. If the
arms lie on one curve in (predicted, observed), the ranking is position weight, not
dictionary structure.

    modal run experiments/temporal_screen/dict_bench/controls_modal.py
"""

import pathlib
import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

app = modal.App("dictbench-controls")
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
def controls(model_id: str, layer: int, k_seg: int, n_docs: int, d_sae: int,
             steps: int, txc_batch: int, ms: list, n_eval: int, frac: float,
             general_frac: float, kper: int):
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
    d_model = model.config.hidden_size
    print(f"[cfg] L={L} d={d_model} k_seg={k_seg} d_sae={d_sae} kper={kper}")

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
    rng = random.Random(777)

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

    # ---- cache ----
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
    Xn = (Xt - Xt.mean((0, 1), keepdim=True)) / (Xt.std() + 1e-6)
    print(f"[cache] {X.shape} base_norm={base_norm:.1f}")

    task = labels[:, 0] >= 0
    fl = torch.tensor(labels[task].reshape(-1), device=dev)
    fa = Xn[task].reshape(-1, d_model)
    u_dom = (fa[fl == 1].mean(0) - fa[fl == 0].mean(0))
    u_dom = u_dom / u_dom.norm()

    # ---- train ----
    def gen_w(bs): return Xn[torch.randint(0, Xn.shape[0], (bs,), device=dev)]

    def gen_f(bs):
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
            if s % max(steps // 3, 1) == 0 or s == steps - 1:
                print(f"   [{tag}] {s}/{steps} loss={loss.item():.3f}")
        return m

    torch.manual_seed(0)
    sae = TopKSAE(d_in=d_model, d_sae=d_sae, k=100).to(dev)
    torch.manual_seed(0)
    txc = TemporalCrosscoder(d_in=d_model, d_sae=d_sae, T=k_seg, k=kper).to(dev)
    train(sae, gen_f, txc_batch * k_seg, "sae")
    train(txc, gen_w, txc_batch, "txc")

    with torch.no_grad():
        V = sae.W_dec.data.float()                    # (d, h)
        P = txc.W_dec.data.float()                    # (h, T, d)
        # NULL 1: random slabs, unit-norm, same shape/count
        g = torch.Generator(device="cpu").manual_seed(123)
        P_rand = torch.randn(P.shape, generator=g).to(dev)
        P_rand = P_rand / P_rand.norm(dim=(1, 2), keepdim=True)
        # NULL 2: learned slabs with rows permuted in TIME (content preserved)
        perm = torch.stack([torch.randperm(k_seg, generator=g) for _ in range(P.shape[0])])
        P_shuf = torch.stack([P[i][perm[i]] for i in range(P.shape[0])]).to(dev)
    Td = k_seg * d_model
    A = {"txc": P.reshape(P.shape[0], Td),
         "txc_random": P_rand.reshape(P.shape[0], Td),
         "txc_shuffled": P_shuf.reshape(P.shape[0], Td)}

    # ---- eval pairs ----
    pairs, Ys, profs = [], [], []
    for _ in range(n_eval):
        prof = [1] * (k_seg // 2) + [0] * (k_seg - k_seg // 2)
        rng.shuffle(prof)
        foil = prof[:]
        for _ in range(40):
            rng.shuffle(foil)
            if foil != prof:
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
        tT, cT = build(car, sf(prof)); tF, cF = build(car, sf(foil))
        pairs.append((enc_txt(tT, cT), enc_txt(tF, cF)))
        Ys.append(torch.stack([(1.0 if l else -1.0) * u_dom for l in prof]))
        profs.append([1.0 if l else -1.0 for l in prof])

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

    tgt_norm = frac * base_norm * (k_seg ** 0.5)
    rescale = lambda W: W * (tgt_norm / (W.norm() + 1e-9))
    base = [margin(t, None) - margin(f, None) for t, f in pairs]
    full = np.array([(margin(t, rescale(Ys[j])) - margin(f, rescale(Ys[j]))) - b
                     for j, ((t, f), b) in enumerate(zip(pairs, base))])
    print(f"[ref] full DoM Δ = {full.mean():+.2f}")

    # ---- inline per-slot weights a_t (the additive check) ----
    a_t = []
    for t in range(k_seg):
        ds = []
        for j, ((tt, ff), b) in enumerate(zip(pairs, base)):
            W = torch.zeros(k_seg, d_model, device=dev)
            W[t] = profs[j][t] * u_dom
            W = rescale(W)
            ds.append((margin(tt, W) - margin(ff, W)) - b)
        a_t.append(float(np.mean(ds)))
    a_t_np = np.array(a_t)
    print(f"[a_t] " + " ".join(f"{x:+.2f}" for x in a_t_np) +
          f"   max/min={a_t_np.max()/max(a_t_np.min(),1e-6):.1f}")

    def omp_span(Y, atoms, m):
        r = Y.clone(); idx = []
        for _ in range(m):
            corr = (atoms @ r) / (atoms.norm(dim=1) + 1e-9)
            for cand in torch.argsort(corr.abs(), descending=True):
                if int(cand) not in idx:
                    idx.append(int(cand)); break
            B = atoms[idx]
            c, *_ = torch.linalg.lstsq(B.T, Y.unsqueeze(1))
            r = Y - (c.squeeze(1) @ B)
        return Y - r

    def omp_perpos(Y, Vm, m, full_support):
        """m latents. full_support=True gives each its own per-slot coefficient
        (cost m*k scalars); False spends m scalars as (latent, slot) pairs."""
        Ym = Y.reshape(k_seg, d_model)
        if full_support:
            sc = (Ym @ Vm)                                   # (k, h)
            j_sel = sc.abs().sum(0).topk(m).indices           # m latents
            B = Vm[:, j_sel]                                  # (d, m)
            C, *_ = torch.linalg.lstsq(B, Ym.T)               # per-slot coeffs
            return (B @ C).T.reshape(-1)
        r = Ym.clone(); idx = []
        for _ in range(m):
            sc = (r @ Vm)
            flat = sc.abs().flatten()
            for cand in torch.argsort(flat, descending=True):
                t, j = int(cand) // sc.shape[1], int(cand) % sc.shape[1]
                if (t, j) not in idx:
                    idx.append((t, j)); break
            out = torch.zeros_like(Ym)
            for (t, j) in idx:
                out[t] = (Ym[t] @ Vm[:, j]) * Vm[:, j]
            r = Ym - out
        return (Ym - r).reshape(-1)

    ones = torch.ones(k_seg, 1, device=dev)
    A["sae_broadcast"] = (ones * V.T.unsqueeze(1)).reshape(V.shape[1], Td)

    res = {"model": model_id, "layer": int(L), "k_seg": k_seg, "d_sae": d_sae,
           "kper": kper, "base_norm": base_norm, "a_t": a_t,
           "full_dom": float(full.mean()),
           "full_dom_sem": float(full.std(ddof=1) / np.sqrt(len(full))),
           "arms": {}}
    ARMS = ["txc", "txc_shuffled", "txc_random", "sae_broadcast",
            "sae_perpos", "sae_fullsupp"]
    for arm in ARMS:
        res["arms"][arm] = {}
        for m in ms:
            ds, cov, pred_add = [], [], []
            for j, ((t, f), b) in enumerate(zip(pairs, base)):
                Y = Ys[j].reshape(-1)
                if arm in A:
                    ap = omp_span(Y, A[arm], m)
                elif arm == "sae_perpos":
                    ap = omp_perpos(Y, V, m, False)
                else:
                    ap = omp_perpos(Y, V, m, True)
                W = rescale(ap.reshape(k_seg, d_model))
                cov.append(float((W.norm(dim=1) > 1e-6).float().mean()))
                # additive prediction from measured a_t
                proj = (W @ u_dom) / (tgt_norm / (k_seg ** 0.5))
                pred_add.append(float(sum(a_t[i] * profs[j][i] * float(proj[i])
                                          for i in range(k_seg)) / k_seg))
                ds.append((margin(t, W) - margin(f, W)) - b)
            ds = np.array(ds)
            scal = m * k_seg if arm == "sae_fullsupp" else m
            res["arms"][arm][m] = {
                "delta": float(ds.mean()),
                "sem": float(ds.std(ddof=1) / np.sqrt(len(ds))),
                "fidelity": float(ds.mean() / full.mean()),
                "slot_coverage": float(np.mean(cov)),
                "scalars": scal,
                "additive_pred": float(np.mean(pred_add)),
                "deltas": [float(x) for x in ds]}
            print(f"   {arm:14} m={m:3d} (scalars {scal:4d}) Δ={ds.mean():+7.2f} "
                  f"fid={ds.mean()/full.mean():+.3f} cover={np.mean(cov):.2f} "
                  f"addpred={np.mean(pred_add):+.2f}")
    return res


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-1.5B-Instruct", layer: int = -1, k_seg: int = 12,
         n_docs: int = 1500, d_sae: int = 4096, steps: int = 3000,
         txc_batch: int = 64, ms: str = "1,2,4,8,16", n_eval: int = 20,
         frac: float = 0.35, general_frac: float = 0.4, kper: int = 41):
    import json
    r = controls.remote(model, layer, k_seg, n_docs, d_sae, steps, txc_batch,
                        [int(x) for x in ms.split(",")], n_eval, frac,
                        general_frac, kper)
    outdir = ROOT / "results" / "dict_bench"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "controls.json").write_text(json.dumps(r, indent=2))
    print("[saved]", outdir / "controls.json")
