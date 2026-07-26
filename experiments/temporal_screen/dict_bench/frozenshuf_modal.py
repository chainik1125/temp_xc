"""Modal: the shuffle control INSIDE the frozen arm — the only run that can still flip this.

WHY THE HEADLINE CONTROL WAS NOT YET LOAD-BEARING. In the m-sweep, coefficients are refit
by least squares AFTER shuffling. Permuting each atom's rows yields a different basis of
k x d slabs with essentially the same expressive power, so "shuffled ~ intact" there is
partly guaranteed by the design. It shows the learned temporal profiles were not aligned
to the target profiles — a real statement about the dictionary's SPAN — but says little
about whether arrangement can do work, because the fit re-chooses everything.

Arrangement can only do work where it is NOT re-chosen. That is the frozen arm.

THIS RUN. Select one latent on a held-out split, freeze it, and compare:
    frozen_intact     the latent's learned (k x d) pattern, applied as-is
    frozen_shuffled   the SAME latent with its k rows permuted in time, NO refit,
                      >= n_draws independent permutations
and report the intact arm's PERCENTILE within the shuffled draw distribution. A point
comparison cannot separate "arrangement is worthless" from "arrangement is worth half a
sigma", and cannot support the claim that shuffling helps.

Registered before running (the reviewer's, restated): intact lands between the 30th and
70th percentile of the shuffled draws at every budget, and the m-sweep's
"shuffled beats intact at 4 of 5" ordering does not survive the draw distribution.
If instead intact >> shuffled, arrangement DOES matter and the m-sweep destroyed the
effect by refitting — which would rewrite the sprint's conclusion.

ALSO HERE (A8) — DICTIONARY HEALTH, now the defence of the negative. The worry has
reversed: no longer "the TXC won because it had more capacity" but "the TXC lost because
it is a bad dictionary". Realised L0 of ~81 against nominal 1200 means ~93% of TopK-
selected latents are zeroed by ReLU, which is as consistent with an undertrained or
partially collapsed encoder as with genuine sparsity. So we report held-out per-segment
FVU for both architectures, the alive-latent fraction (firing on >= 0.1% of eval windows,
union over the eval set), and realised L0. If the TXC reconstructs materially worse, the
honest headline narrows to "this crosscoder, trained this way".

    modal run experiments/temporal_screen/dict_bench/frozenshuf_modal.py
"""

import pathlib
import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

app = modal.App("dictbench-frozenshuf")
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
def frozenshuf(model_id: str, layer: int, k_seg: int, n_docs: int, d_sae: int,
               steps: int, txc_batch: int, n_sel: int, n_test: int, frac: float,
               general_frac: float, kper: int, n_draws: int, pool: int, ms: list,
               txc_lr: float = 1e-3):
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
    rng = random.Random(99991)

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

    def gen_w(bs): return Xn[torch.randint(0, Xn.shape[0], (bs,), device=dev)]

    def gen_f(bs):
        i = torch.randint(0, Xn.shape[0], (bs,), device=dev)
        j = torch.randint(0, k_seg, (bs,), device=dev)
        return Xn[i, j]

    def train(m, gen, bs, tag, lr=1e-3):
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        for s in range(steps):
            loss, _, _ = m(gen(bs))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); m._normalize_decoder()
            if s % max(steps // 3, 1) == 0 or s == steps - 1:
                print(f"   [{tag}] {s}/{steps} loss={loss.item():.3f}")
        return m

    torch.manual_seed(0)
    sae = TopKSAE(d_in=d, d_sae=d_sae, k=100).to(dev)
    torch.manual_seed(0)
    txc = TemporalCrosscoder(d_in=d, d_sae=d_sae, T=k_seg, k=kper).to(dev)
    train(sae, gen_f, txc_batch * k_seg, "sae")
    train(txc, gen_w, txc_batch, "txc", lr=txc_lr)
    with torch.no_grad():
        V = sae.W_dec.data.float(); P = txc.W_dec.data.float()

    # ================= A8: dictionary health =================
    with torch.no_grad():
        flat_ho = Xn_ho.reshape(-1, d)
        z_s = sae.encode(flat_ho); xh_s = sae.decode(z_s)
        fvu_sae = float(((xh_s - flat_ho) ** 2).sum(-1).mean() / flat_ho.var(0).sum())
        l0_sae = float((z_s > 0).float().sum(-1).mean())
        alive_sae = float(((z_s > 0).float().mean(0) >= 0.001).float().mean())
        z_t = txc.encode(Xn_ho); xh_t = txc.decode(z_t)
        fvu_txc = float(((xh_t - Xn_ho) ** 2).sum(-1).mean() / flat_ho.var(0).sum())
        l0_txc = float((z_t > 0).float().sum(-1).mean())
        alive_txc = float(((z_t > 0).float().mean(0) >= 0.001).float().mean())
        freq_txc = (z_t > 0).float().mean(0)
    health = {"fvu_sae": fvu_sae, "fvu_txc": fvu_txc,
              "l0_sae_per_token": l0_sae, "l0_sae_per_window": l0_sae * k_seg,
              "l0_txc_per_window": l0_txc,
              "alive_frac_sae": alive_sae, "alive_frac_txc": alive_txc,
              "d_sae": d_sae}
    print(f"[health] FVU  SAE {fvu_sae:.4f}  TXC {fvu_txc:.4f}"
          f"  ({'TXC worse' if fvu_txc > fvu_sae*1.1 else 'comparable or better'})")
    print(f"[health] L0   SAE {l0_sae:.1f}/token = {l0_sae*k_seg:.0f}/window   "
          f"TXC {l0_txc:.1f}/window")
    print(f"[health] alive fraction  SAE {alive_sae:.3f}  TXC {alive_txc:.3f} "
          f"(of d_sae={d_sae})")

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

    # ---- a_t, then an a-orthogonal profile (A1) ----
    a_t = np.zeros(k_seg)
    for _ in range(8):
        p = [1] * (k_seg // 2) + [0] * (k_seg // 2); rng.shuffle(p)
        prs = make_pairs(p, 4)
        bs_ = [margin(t, None) - margin(f, None) for t, f in prs]
        pi_ = np.array([1.0 if l else -1.0 for l in p])
        for t in range(k_seg):
            W = torch.zeros(k_seg, d, device=dev); W[t] = float(pi_[t]) * u_dom
            a_t[t] += run(prs, rescale(W), bs_).mean() / 8
    best = None
    for combo in itertools.combinations(range(k_seg), k_seg // 2):
        pi_ = -np.ones(k_seg); pi_[list(combo)] = 1.0
        f_ = abs(float((a_t * pi_).sum() / a_t.sum()))
        if best is None or f_ < best[0]:
            best = (f_, [1 if pi_[t] > 0 else 0 for t in range(k_seg)])
    floor_pred, prof01 = best
    print(f"[a_t] " + " ".join(f"{x:+.2f}" for x in a_t))
    print(f"[profile] a-orthogonal: {prof01}  predicted floor {floor_pred:+.4f}")

    Y = torch.stack([(1.0 if l else -1.0) * u_dom for l in prof01])
    sel_pairs, test_pairs = make_pairs(prof01, n_sel), make_pairs(prof01, n_test)
    b_sel = [margin(t, None) - margin(f, None) for t, f in sel_pairs]
    b_tst = [margin(t, None) - margin(f, None) for t, f in test_pairs]
    full = run(test_pairs, rescale(Y), b_tst)
    print(f"[ref] full DoM Δ={full.mean():+.2f}")

    # ================= A7: frozen intact vs frozen shuffled draws =================
    with torch.no_grad():
        cands = (P.reshape(P.shape[0], -1) @ Y.reshape(-1)).abs().topk(pool).indices.tolist()
    sc = [float(run(sel_pairs, rescale(P[j]), b_sel).mean()) for j in cands]
    bi = int(np.argmax(np.abs(sc))); j_sel = cands[bi]; sgn = float(np.sign(sc[bi]))
    ds_int = run(test_pairs, rescale(sgn * P[j_sel]), b_tst)
    fid_int = float(ds_int.mean() / full.mean())
    print(f"[frozen intact] latent {j_sel} sign {sgn:+.0f} freq {float(freq_txc[j_sel]):.4f}: "
          f"Δ={ds_int.mean():+.2f} fid={fid_int:+.3f}")

    shuf_fids = []
    for q in range(n_draws):
        perm = torch.randperm(k_seg, generator=torch.Generator().manual_seed(1000 + q))
        Pp = P[j_sel][perm]
        dsq = run(test_pairs, rescale(sgn * Pp), b_tst)
        shuf_fids.append(float(dsq.mean() / full.mean()))
    shuf = np.array(shuf_fids)
    pct = float((shuf < fid_int).mean() * 100)
    print(f"[frozen shuffled] n={n_draws}: mean {shuf.mean():+.3f} sd {shuf.std(ddof=1):.3f} "
          f"range [{shuf.min():+.3f}, {shuf.max():+.3f}]")
    print(f"[VERDICT] intact fid {fid_int:+.3f} is at the {pct:.0f}th percentile of the "
          f"shuffled draws -> {'arrangement carries no work' if 30 <= pct <= 70 else 'ARRANGEMENT MATTERS' if pct > 70 else 'intact WORSE than shuffled'}")

    # same test at a couple of m budgets WITH refit, for the contrast the reviewer wants
    refit = {}
    Td = k_seg * d
    for m in ms:
        def omp(atoms):
            r = Y.reshape(-1).clone(); idx = []
            for _ in range(m):
                corr = (atoms @ r) / (atoms.norm(dim=1) + 1e-9)
                for c_ in torch.argsort(corr.abs(), descending=True):
                    if int(c_) not in idx:
                        idx.append(int(c_)); break
                B = atoms[idx]
                c, *_ = torch.linalg.lstsq(B.T, Y.reshape(-1).unsqueeze(1))
                r = Y.reshape(-1) - (c.squeeze(1) @ B)
            return Y.reshape(-1) - r
        A_int = P.reshape(P.shape[0], Td)
        f_int = float(run(test_pairs, rescale(omp(A_int).reshape(k_seg, d)), b_tst).mean()
                      / full.mean())
        fs = []
        for q in range(min(n_draws, 8)):
            g = torch.Generator().manual_seed(2000 + q)
            pm = torch.stack([torch.randperm(k_seg, generator=g) for _ in range(P.shape[0])])
            Ps = torch.stack([P[i][pm[i]] for i in range(P.shape[0])]).to(dev)
            fs.append(float(run(test_pairs,
                                rescale(omp(Ps.reshape(P.shape[0], Td)).reshape(k_seg, d)),
                                b_tst).mean() / full.mean()))
        refit[m] = {"intact": f_int, "shuffled_mean": float(np.mean(fs)),
                    "shuffled_sd": float(np.std(fs, ddof=1)),
                    "percentile": float((np.array(fs) < f_int).mean() * 100)}
        print(f"[refit m={m}] intact {f_int:+.3f} vs shuffled {np.mean(fs):+.3f}"
              f"±{np.std(fs,ddof=1):.3f}  (intact at {refit[m]['percentile']:.0f}th pct)")

    return {"model": model_id, "layer": int(L), "k_seg": k_seg, "d_sae": d_sae,
            "kper": kper, "base_norm": base_norm, "health": health,
            "a_t": a_t.tolist(), "profile": prof01, "floor_predicted": floor_pred,
            "full_dom": float(full.mean()),
            "frozen_intact": {"latent": int(j_sel), "sign": sgn,
                              "freq": float(freq_txc[j_sel]),
                              "delta": float(ds_int.mean()), "fidelity": fid_int,
                              "deltas": [float(x) for x in ds_int]},
            "frozen_shuffled": {"n": n_draws, "fidelities": shuf_fids,
                                "mean": float(shuf.mean()),
                                "sd": float(shuf.std(ddof=1)),
                                "intact_percentile": pct},
            "refit_contrast": refit}


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-1.5B-Instruct", layer: int = -1, k_seg: int = 12,
         n_docs: int = 1200, d_sae: int = 4096, steps: int = 2500,
         txc_batch: int = 64, n_sel: int = 8, n_test: int = 20, frac: float = 0.35,
         general_frac: float = 0.4, kper: int = 41, n_draws: int = 24,
         pool: int = 16, ms: str = "2,8", txc_lr: float = 1e-3, tag: str = ""):
    import json
    r = frozenshuf.remote(model, layer, k_seg, n_docs, d_sae, steps, txc_batch,
                          n_sel, n_test, frac, general_frac, kper, n_draws, pool,
                          [int(x) for x in ms.split(",")], txc_lr)
    r["txc_lr"] = txc_lr
    outdir = ROOT / "results" / "dict_bench"
    outdir.mkdir(parents=True, exist_ok=True)
    name = f"frozen_shuffle{tag}.json"
    (outdir / name).write_text(json.dumps(r, indent=2))
    print("[saved]", outdir / name)
