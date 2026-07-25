"""Modal: frozen-write arm, v2 — every amendment from the pre-registration applied.

WHAT CHANGED FROM v1, and why each change was forced:

A1  The prediction "a frozen SAE broadcast scores ~0" is TRUE when the profile is redrawn
    per pair (zero-mean target, constant atom, exact orthogonality) and FALSE for a fixed
    profile, because the model does not weight slots equally. The constant-write floor is
    (sum_t a_t pi_t)/(sum_t a_t), which for structured square waves is -0.18 (ell=1) and
    -0.22 (ell=3) — v1 ran on exactly those two. So profiles are now CHOSEN from the
    measured a_t: one a-orthogonal (floor ~ 0) and two spanning floor ~ +0.2 and ~ -0.2.
    Running three also converts the floor into a second instrument test: sae_frozen should
    TRACK its predicted floor across profiles, registered slope 1.0 +/- 0.3.

A3  A uniformly drawn "random latent" null is near-worthless here: at realised L0 ~81 of
    d_sae 4096, ~98% of latents are inactive on a given window, so a uniform draw mostly
    returns near-dead latents at their initialisation. Nulls are now drawn FREQUENCY-MATCHED
    to the selected latent (same activation-frequency decile on the selection split).

A4  The missing arm, and the one that answers the question: FITTED-ONCE SCHEDULE. One SAE
    latent with T per-slot coefficients fitted on the SELECTION split, then frozen and
    applied unchanged to every test episode. Zero episode-time knobs, exactly like the TXC
    arm; T offline parameters instead of one, and supervised where the TXC's schedule is
    unsupervised — so it is a CEILING, not a competitor. It gives the number the sprint
    actually wants: what fraction of an offline-fitted schedule does an unsupervised
    temporal decoder recover?

A2  Realised L0 and per-segment FVU on a held-out split are measured and reported for both
    architectures. Nominal k does not appear.

Registered before running: txc_frozen > 0.20 AND > 0.15 above BOTH its frequency-matched
null and the profile's constant-write floor; fitted_once beats txc_frozen with ratio
txc_frozen/fitted_once in 0.3-0.8; sae_frozen tracks the predicted floor at slope 1 +/- 0.3.

    modal run experiments/temporal_screen/dict_bench/frozen2_modal.py
"""

import pathlib
import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

app = modal.App("dictbench-frozen2")
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
def frozen2(model_id: str, layer: int, k_seg: int, n_docs: int, d_sae: int,
            steps: int, txc_batch: int, n_sel: int, n_test: int, frac: float,
            general_frac: float, kper: int, n_null: int, pool: int):
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
    print(f"[cfg] L={L} d={d} k_seg={k_seg} d_sae={d_sae} kper={kper}")

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
    rng = random.Random(20260726)

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

    # ---------------- cache ----------------
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
    u_dom = (fa[fl == 1].mean(0) - fa[fl == 0].mean(0))
    u_dom = u_dom / u_dom.norm()
    print(f"[cache] train {tuple(Xn.shape)} holdout {tuple(Xn_ho.shape)} "
          f"base_norm={base_norm:.1f}")

    # ---------------- train ----------------
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
            if s % max(steps // 3, 1) == 0 or s == steps - 1:
                print(f"   [{tag}] {s}/{steps} loss={loss.item():.3f}")
        return m

    torch.manual_seed(0)
    sae = TopKSAE(d_in=d, d_sae=d_sae, k=100).to(dev)
    torch.manual_seed(0)
    txc = TemporalCrosscoder(d_in=d, d_sae=d_sae, T=k_seg, k=kper).to(dev)
    train(sae, gen_f, txc_batch * k_seg, "sae")
    train(txc, gen_w, txc_batch, "txc")

    # ---------------- A2: realised L0 + held-out per-segment FVU ----------------
    with torch.no_grad():
        V = sae.W_dec.data.float()
        P = txc.W_dec.data.float()
        z_s = sae.encode(Xn_ho.reshape(-1, d))
        xh_s = sae.decode(z_s)
        fvu_sae = float(((xh_s - Xn_ho.reshape(-1, d)) ** 2).sum(-1).mean()
                        / (Xn_ho.reshape(-1, d).var(0).sum()))
        l0_sae = float((z_s > 0).float().sum(-1).mean())
        freq_sae = (z_s > 0).float().mean(0)                     # (h,)
        z_t = txc.encode(Xn_ho)
        xh_t = txc.decode(z_t)
        fvu_txc = float(((xh_t - Xn_ho) ** 2).sum(-1).mean()
                        / (Xn_ho.reshape(-1, d).var(0).sum()))
        l0_txc = float((z_t > 0).float().sum(-1).mean())
        freq_txc = (z_t > 0).float().mean(0)
    print(f"[A2] realised L0: SAE {l0_sae:.1f}/token ({l0_sae*k_seg:.0f}/window)  "
          f"TXC {l0_txc:.1f}/window")
    print(f"[A2] held-out per-segment FVU: SAE {fvu_sae:.4f}  TXC {fvu_txc:.4f}  "
          f"-> {'TXC reconstructs better' if fvu_txc < fvu_sae*0.9 else 'no TXC recon advantage'}")

    tgt_norm = frac * base_norm * (k_seg ** 0.5)
    rescale = lambda W: W * (tgt_norm / (W.norm() + 1e-9))

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

    # ---------------- measure a_t on random balanced profiles ----------------
    probe = []
    for _ in range(10):
        p = [1] * (k_seg // 2) + [0] * (k_seg // 2)
        rng.shuffle(p); probe.append(p)
    a_t = np.zeros(k_seg)
    for p in probe:
        prs = make_pairs(p, 4)
        base = [margin(t, None) - margin(f, None) for t, f in prs]
        pi_ = np.array([1.0 if l else -1.0 for l in p])
        for t in range(k_seg):
            W = torch.zeros(k_seg, d, device=dev)
            W[t] = float(pi_[t]) * u_dom
            a_t[t] += run(prs, rescale(W), base).mean() / len(probe)
    print(f"[a_t] " + " ".join(f"{x:+.2f}" for x in a_t))

    # ---------------- A1: choose 3 profiles by their constant-write floor ----------
    floors = []
    for combo in itertools.combinations(range(k_seg), k_seg // 2):
        pi_ = -np.ones(k_seg); pi_[list(combo)] = 1.0
        floors.append((float((a_t * pi_).sum() / a_t.sum()),
                       [1 if pi_[t] > 0 else 0 for t in range(k_seg)]))
    floors.sort(key=lambda z: z[0])
    def pick(target):
        return min(floors, key=lambda z: abs(z[0] - target))
    chosen = [pick(-0.20), pick(0.0), pick(+0.20)]
    print("[A1] chosen profiles (predicted constant-write floor):")
    for f_, p_ in chosen:
        print(f"     floor {f_:+.3f}  {p_}")

    results = {"model": model_id, "layer": int(L), "k_seg": k_seg, "d_sae": d_sae,
               "kper": kper, "base_norm": base_norm, "a_t": a_t.tolist(),
               "realised_l0_sae_per_token": l0_sae,
               "realised_l0_txc_per_window": l0_txc,
               "fvu_sae_heldout": fvu_sae, "fvu_txc_heldout": fvu_txc,
               "profiles": []}

    with torch.no_grad():
        Yfull = {tuple(p_): torch.stack(
            [(1.0 if l else -1.0) * u_dom for l in p_]) for _, p_ in chosen}

    for floor_pred, prof01 in chosen:
        print(f"\n===== profile {prof01}  predicted floor {floor_pred:+.3f} =====")
        Y = Yfull[tuple(prof01)]
        sel_pairs = make_pairs(prof01, n_sel)
        test_pairs = make_pairs(prof01, n_test)
        base_sel = [margin(t, None) - margin(f, None) for t, f in sel_pairs]
        base_tst = [margin(t, None) - margin(f, None) for t, f in test_pairs]
        full = run(test_pairs, rescale(Y), base_tst)
        print(f"[ref] full DoM Δ={full.mean():+.2f}")
        row = {"profile": prof01, "floor_predicted": floor_pred,
               "full_dom": float(full.mean()),
               "full_dom_sem": float(full.std(ddof=1) / np.sqrt(len(full))),
               "arms": {}}

        # ---- pre-rank candidate pools, then score on the SELECTION split only ----
        with torch.no_grad():
            Yf = Y.reshape(-1)
            cand_txc = (P.reshape(P.shape[0], -1) @ Yf).abs().topk(pool).indices.tolist()
            cand_sae = (V.T @ u_dom).abs().topk(pool).indices.tolist()

        def frozen_pick(mk, cands, tag):
            sc = [float(run(sel_pairs, rescale(mk(j)), base_sel).mean()) for j in cands]
            b = int(np.argmax(np.abs(sc)))
            j, sg = cands[b], float(np.sign(sc[b]))
            ds = run(test_pairs, rescale(sg * mk(j)), base_tst)
            fid = float(ds.mean() / full.mean())
            print(f"[{tag}] latent {j} sign {sg:+.0f}: Δ={ds.mean():+6.2f} fid={fid:+.3f}")
            return {"latent": int(j), "sign": sg, "sel_score": sc[b],
                    "delta": float(ds.mean()),
                    "sem": float(ds.std(ddof=1) / np.sqrt(len(ds))),
                    "fidelity": fid, "deltas": [float(x) for x in ds]}

        row["arms"]["txc_frozen"] = frozen_pick(lambda j: P[j], cand_txc, "txc_frozen")
        row["arms"]["sae_frozen"] = frozen_pick(
            lambda j: torch.stack([V[:, j]] * k_seg), cand_sae, "sae_frozen")
        jt = row["arms"]["txc_frozen"]["latent"]
        row["arms"]["txc_frozen"]["freq"] = float(freq_txc[jt])
        row["arms"]["txc_frozen"]["row_norms"] = P[jt].norm(dim=1).tolist()
        js = row["arms"]["sae_frozen"]["latent"]
        row["arms"]["sae_frozen"]["freq"] = float(freq_sae[js])

        # ---- A4: fitted-once schedule (ceiling) ----
        best = None
        for j in cand_sae[:8]:
            v = V[:, j]
            C = torch.tensor([float((Y[t] @ v)) for t in range(k_seg)], device=dev)
            W = rescale(torch.stack([C[t] * v for t in range(k_seg)]))
            s = float(run(sel_pairs, W, base_sel).mean())
            if best is None or abs(s) > abs(best[0]):
                best = (s, j, C)
        s, j, C = best
        W = rescale(float(np.sign(s)) * torch.stack([C[t] * V[:, j]
                                                     for t in range(k_seg)]))
        ds = run(test_pairs, W, base_tst)
        row["arms"]["fitted_once"] = {
            "latent": int(j), "delta": float(ds.mean()),
            "sem": float(ds.std(ddof=1) / np.sqrt(len(ds))),
            "fidelity": float(ds.mean() / full.mean()),
            "offline_params": k_seg, "episode_knobs": 0,
            "deltas": [float(x) for x in ds]}
        print(f"[fitted_once] latent {j}: Δ={ds.mean():+6.2f} "
              f"fid={ds.mean()/full.mean():+.3f}  ({k_seg} offline params, 0 at episode)")

        # ---- A3: frequency-matched nulls ----
        for tag, freq, mk, jsel in (("txc", freq_txc, lambda j: P[j], jt),
                                    ("sae", freq_sae,
                                     lambda j: torch.stack([V[:, j]] * k_seg), js)):
            f0 = float(freq[jsel])
            lo, hi = f0 * 0.5, f0 * 2.0
            elig = torch.nonzero((freq >= lo) & (freq <= hi)).flatten().tolist()
            elig = [e for e in elig if e != jsel] or list(range(freq.shape[0]))
            draws, fids = [], []
            for _ in range(n_null):
                jj = elig[rng.randrange(len(elig))]
                dd = run(test_pairs, rescale(mk(jj)), base_tst)
                fids.append(abs(float(dd.mean() / full.mean())))
                draws.append({"latent": int(jj), "freq": float(freq[jj])})
            row["arms"][f"{tag}_freqmatched_null"] = {
                "selected_freq": f0, "n_eligible": len(elig),
                "mean_abs_fidelity": float(np.mean(fids)),
                "p90": float(np.percentile(fids, 90)), "draws": draws}
            print(f"[null-{tag}] freq={f0:.3f} pool={len(elig)}: "
                  f"mean|fid|={np.mean(fids):.3f} p90={np.percentile(fids,90):.3f}")

        # ---- observed constant-write floor, for the A1 tracking test ----
        Wc = rescale(torch.stack([u_dom] * k_seg))
        dsf = run(test_pairs, Wc, base_tst)
        row["floor_observed"] = float(dsf.mean() / full.mean())
        print(f"[floor] predicted {floor_pred:+.3f}  observed "
              f"{row['floor_observed']:+.3f}")
        results["profiles"].append(row)

    fp = np.array([r["floor_predicted"] for r in results["profiles"]])
    fo = np.array([r["floor_observed"] for r in results["profiles"]])
    if len(fp) > 1 and fp.std() > 1e-6:
        sl = float(np.polyfit(fp, fo, 1)[0])
        results["floor_tracking_slope"] = sl
        print(f"\n[A1 test] floor tracking slope = {sl:+.2f} "
              f"(registered 1.0 +/- 0.3) -> {'PASS' if abs(sl-1) <= 0.3 else 'FAIL'}")
    return results


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-1.5B-Instruct", layer: int = -1, k_seg: int = 12,
         n_docs: int = 1200, d_sae: int = 4096, steps: int = 2500,
         txc_batch: int = 64, n_sel: int = 8, n_test: int = 20, frac: float = 0.35,
         general_frac: float = 0.4, kper: int = 41, n_null: int = 12, pool: int = 16):
    import json
    r = frozen2.remote(model, layer, k_seg, n_docs, d_sae, steps, txc_batch,
                       n_sel, n_test, frac, general_frac, kper, n_null, pool)
    outdir = ROOT / "results" / "dict_bench"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "frozen2.json").write_text(json.dumps(r, indent=2))
    print("[saved]", outdir / "frozen2.json")
