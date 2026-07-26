"""Reusable reading-vs-steering harness for TXC vs SAE vs tSAE task screens.

This is `experiments/temporal_screen/dict_bench/steer_order_modal.py` generalised so that a
new task is a ~20-line function rather than a new copy of the whole pipeline. The pattern it
implements is the one that produced last sprint's headline, and its parts are not optional:

  1. TWO CLASSES OF DOCUMENT, MATCHED except for the property under test. A task supplies
     `make_pair(rng) -> (sents_a, sents_b, meta)` where the two sentence lists differ only in
     that property -- same multiset of sentences wherever possible, same length, same
     carrier. Everything generic then cancels in the paired contrast.
  2. MID-LAYER ACTIVATIONS, MEAN-POOLED PER SEGMENT, into (n, T, d) windows.
  3. THREE DICTIONARIES trained on reconstruction only: a per-token TopK SAE, a
     TemporalCrosscoder with `activation="batchtopk"`, and the attention tSAE at a
     CALIBRATED l1 (see `tsae_calib_modal.py` -- the repo's documented 1e-3 is dense).
  4. READING = best-single-latent AUC for the class label. SAE codes are mean-pooled over
     the window; the TXC window code is used directly; tSAE novel codes are mean-pooled.
  5. STEERING = teacher-forced delta margin logP(A) - logP(B) where B is the matched foil,
     with the decoder direction or slab added at the training layer, every write rescaled to
     identical total injected norm so nothing is won by writing harder.
  6. CONTROLS, every time: `txc_flat` (the TXC slab time-averaged and rebroadcast, which
     removes the profile and keeps the direction), `random_slab`, `random_broadcast`, and
     `dom_slab` (supervised per-position difference-of-means ceiling).

WHY THE CONTROLS ARE LOAD-BEARING. A positive `txc_slab` on its own says only that some
perturbation moved the margin. `txc_flat` is the same latent, same mean direction, same norm,
with only the temporal profile removed -- if it matches `txc_slab`, the profile is doing
nothing and the result is not a temporal claim. `random_slab` rules out "any structured
perturbation across positions". `random_broadcast` calibrates the SAE arm: last sprint a
random constant direction beat the SAE's learned one, which is the finding that a constant
write on an order task is indistinguishable from noise.

REALISED SPARSITY IS LOGGED FOR EVERY DICTIONARY. Nominal k does not bind for the
crosscoder and the failure is silent -- `min(k, #{pre > 0})` collapsed a whole sprint's
comparisons before it was caught. `m.eval()` is called before scoring because `batchtopk` is
a batch rule at train time and a fixed threshold at eval time.
"""
from __future__ import annotations


def unit(v):
    import torch
    return v / (v.norm() + 1e-12)


def run_task(*, make_pair, model_id, layer, k_seg, n_train, n_test, d_sae, k,
             steps, lr, batch_win, alphas, tsae_l1=None, n_perm=0, seed=31415,
             arms=None, verbose=True):
    """Train three dictionaries on a matched-pair task and score reading + steering.

    `make_pair(rng)` must return `(sents_a, sents_b, carrier)`: two equal-length sentence
    lists differing only in the property under test, plus the shared carrier prefix. Class A
    is label 1, class B is label 0.

    Returns a dict with per-arm reading AUCs, per-arm steering deltas with per-document
    standard errors, realised coefficients per segment for every dictionary, and the
    z-separations that decide the task.
    """
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
    T = k_seg
    rng = random.Random(seed)
    cap = {}

    def log(*a):
        if verbose:
            print(*a, flush=True)

    def cap_hook(_m, _i, out):
        cap["h"] = (out[0] if isinstance(out, tuple) else out).detach()

    def build(carrier, sents):
        text, spans = carrier, []
        for j, s in enumerate(sents):
            if j:
                text += " "
            spans.append((len(text), len(text) + len(s)))
            text += s
        return text, spans

    def seg_spans(text, spans):
        e = tok(text, return_tensors="pt", return_offsets_mapping=True)
        offs = e["offset_mapping"][0].tolist()
        ts = []
        for (a, b) in spans:
            idx = [i for i, (s0, s1) in enumerate(offs)
                   if s0 >= a and s1 <= b and s1 > s0]
            ts.append((idx[0], idx[-1]) if idx else (0, 0))
        return e, ts

    def capture(text, spans):
        e, ts = seg_spans(text, spans)
        h = layers_[L].register_forward_hook(cap_hook)
        with torch.no_grad():
            model(e["input_ids"].to(dev))
        h.remove()
        hh = cap["h"][0].float()
        return torch.stack([hh[a:b + 1].mean(0) for a, b in ts])

    # ---------------- training activations, balanced over the two classes ----------------
    X, y = [], []
    for i in range(n_train):
        sa, sb, car = make_pair(rng)
        assert len(sa) == len(sb) == k_seg, (
            f"make_pair returned {len(sa)}/{len(sb)} segments, expected {k_seg}")
        cls = rng.randint(0, 1)
        X.append(capture(*build(car, sa if cls else sb)))
        y.append(cls)
        if (i + 1) % 250 == 0:
            log(f"   [cache] {i+1}/{n_train}")
    Xt = torch.stack(X).to(dev)
    yt = torch.tensor(y, device=dev)
    mu, sd = Xt.mean((0, 1), keepdim=True), Xt.std() + 1e-6
    Xn = (Xt - mu) / sd
    log(f"[cache] {tuple(Xn.shape)}  class balance {float(yt.float().mean()):.3f}")

    def gen_win(bs):
        return Xn[torch.randint(0, Xn.shape[0], (bs,), device=dev)]

    def gen_flat(bs):
        f = Xn.reshape(-1, d)
        return f[torch.randint(0, f.shape[0], (bs,), device=dev)]

    def adam_train(m, gen, bs):
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        m.train()
        for _ in range(steps):
            loss, _, _ = m(gen(bs))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); m._normalize_decoder()
        m.eval()          # batchtopk is a batch rule at train time, a threshold at eval

    def auc_of(scores, yy):
        o = torch.argsort(scores)
        r = torch.empty_like(o, dtype=torch.float32)
        r[o] = torch.arange(len(scores), device=dev, dtype=torch.float32) + 1
        n1, n0 = float(yy.sum()), float((1 - yy).sum())
        if n1 == 0 or n0 == 0:
            return 0.5
        return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    def best_latent(Z, yy):
        a = torch.tensor([auc_of(Z[:, j], yy) for j in range(Z.shape[1])], device=dev)
        dv = (a - 0.5).abs()
        j = int(dv.argmax())
        return j, float(a[j]), float(0.5 + dv[j])

    out = {"model": model_id, "layer": int(L), "k_seg": k_seg, "T": T, "d_sae": d_sae,
           "k": k, "steps": steps, "lr": lr, "n_train": n_train, "n_test": n_test,
           "alphas": list(alphas), "reading": {}, "sparsity": {}, "arms": {}}
    writes = {}

    # ---------------- 1. per-token TopK SAE ----------------
    torch.manual_seed(0)
    sae = TopKSAE(d_in=d, d_sae=d_sae, k=k).to(dev)
    adam_train(sae, gen_flat, batch_win * T)
    with torch.no_grad():
        Zf = sae.encode(Xn.reshape(-1, d))
        Zs = Zf.reshape(-1, T, d_sae).mean(1)
    js, auc_s_signed, auc_s = best_latent(Zs, yt)
    sign_s = 1.0 if auc_s_signed > 0.5 else -1.0
    out["reading"]["sae"] = {"latent": js, "auc": auc_s}
    out["sparsity"]["sae"] = float((Zf > 0).float().sum(-1).mean())
    log(f"[sae] latent {js} pooled AUC {auc_s:.3f}  "
        f"realised {out['sparsity']['sae']:.2f} coeff/segment")
    v_sae = unit(sae.W_dec.data[:, js].float())
    writes["sae_broadcast"] = sign_s * unit(v_sae.unsqueeze(0).expand(T, -1).contiguous())

    # ---------------- 2. Temporal Crosscoder ----------------
    torch.manual_seed(0)
    txc = TemporalCrosscoder(d_in=d, d_sae=d_sae, T=T, k=k,
                             activation="batchtopk").to(dev)
    with torch.no_grad():
        txc._normalize_decoder()
    adam_train(txc, gen_win, batch_win)
    with torch.no_grad():
        Zt = txc.encode(Xn)
    jt, auc_t_signed, auc_t = best_latent(Zt, yt)
    sign_t = 1.0 if auc_t_signed > 0.5 else -1.0
    out["reading"]["txc"] = {"latent": jt, "auc": auc_t}
    out["sparsity"]["txc"] = float((Zt != 0).float().sum(-1).mean()) / T
    log(f"[txc] latent {jt} window AUC {auc_t:.3f}  "
        f"realised {out['sparsity']['txc']:.2f} coeff/segment")
    P_txc = unit(txc.W_dec.data[jt].float())
    writes["txc_slab"] = sign_t * P_txc
    writes["txc_flat"] = sign_t * unit(P_txc.mean(0, keepdim=True).expand(T, -1).contiguous())

    # ---------------- 3. attention tSAE at the calibrated l1 ----------------
    if tsae_l1 is not None:
        import torch.nn.functional as F
        from temporal_crosscoders.han_tsae import TemporalSAE
        torch.manual_seed(0)
        ts_ = TemporalSAE(dimin=d, width=d_sae, n_heads=8, sae_diff_type="relu",
                          kval_topk=None, tied_weights=True, n_attn_layers=1,
                          bottleneck_factor=64).to(dev)
        opt = torch.optim.Adam(ts_.parameters(), lr=lr)
        ts_.train()
        for _ in range(steps):
            xb = gen_win(batch_win)
            recons, info = ts_(xb)
            loss = ((recons - xb).pow(2).sum(-1).mean()
                    + tsae_l1 * (info["novel_codes"]
                                 + info["pred_codes"]).abs().sum(-1).mean())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(ts_.parameters(), 1.0)
            opt.step()
        ts_.eval()
        with torch.no_grad():
            _, info = ts_(Xn)
            Zn_all = info["novel_codes"]
            Zn = Zn_all.mean(1)
        jn, auc_n_signed, auc_n = best_latent(Zn, yt)
        sign_n = 1.0 if auc_n_signed > 0.5 else -1.0
        out["reading"]["tsae"] = {"latent": jn, "auc": auc_n, "l1_coef": tsae_l1}
        out["sparsity"]["tsae"] = float(
            (Zn_all.reshape(-1, d_sae) > 0).float().sum(-1).mean())
        log(f"[tsae] latent {jn} pooled AUC {auc_n:.3f}  "
            f"realised {out['sparsity']['tsae']:.2f} coeff/segment  (l1={tsae_l1:g})")
        # The tSAE's temporal machinery is all in the ENCODER; its dictionary D gives one
        # direction per latent, so its per-latent write is constant in time exactly like
        # the SAE's. That structural fact is the point of including the arm.
        v_ts = unit(ts_.D.data[jn].float())
        writes["tsae_broadcast"] = sign_n * unit(
            v_ts.unsqueeze(0).expand(T, -1).contiguous())

    # ---------------- 4. controls ----------------
    with torch.no_grad():
        writes["dom_slab"] = unit((Xn[yt == 1].mean(0) - Xn[yt == 0].mean(0)).float())
    g = torch.Generator(device="cpu").manual_seed(7)
    writes["random_slab"] = unit(torch.randn(T, d, generator=g).to(dev))
    rv = unit(torch.randn(d, generator=g).to(dev))
    writes["random_broadcast"] = unit(rv.unsqueeze(0).expand(T, -1).contiguous())

    if arms:
        writes = {nm: W for nm, W in writes.items() if nm in arms}
    for nm, W in writes.items():
        log(f"[write] {nm:<17} total norm {float(W.norm()):.3f}  "
            f"per-position spread {float(W.norm(dim=-1).std()):.4f}")

    # ---------------- teacher-forced margin ----------------
    scale = float(Xt.norm(dim=-1).mean())

    def margin(text, spans, W, alpha):
        e, ts2 = seg_spans(text, spans)
        ids = e["input_ids"].to(dev)
        hook = None
        if W is not None:
            def edit(_m, _i, out_):
                h = out_[0] if isinstance(out_, tuple) else out_
                for t_i, (a, b) in enumerate(ts2):
                    h[:, a:b + 1, :] += (alpha * scale
                                         * W[t_i].to(h.dtype).unsqueeze(0))
                return (h,) + out_[1:] if isinstance(out_, tuple) else h
            hook = layers_[L].register_forward_hook(edit)
        with torch.no_grad():
            lg = model(ids).logits.float().log_softmax(-1)
        if hook is not None:
            hook.remove()
        tgt = ids[0, 1:]
        return float(lg[0, :-1].gather(-1, tgt.unsqueeze(-1)).sum())

    tests = []
    for _ in range(n_test):
        sa, sb, car = make_pair(rng)
        ta, spa = build(car, sa)
        tb, spb = build(car, sb)
        tests.append((ta, spa, tb, spb))
    base_by_doc = [margin(ta, sa2, None, 0) - margin(tb, sb2, None, 0)
                   for (ta, sa2, tb, sb2) in tests]

    def score(W):
        deltas, sems = [], []
        for a in alphas:
            ds = np.array([margin(ta, sa2, W, a) - margin(tb, sb2, W, a) - base
                           for (ta, sa2, tb, sb2), base in zip(tests, base_by_doc)])
            deltas.append(float(ds.mean()))
            sems.append(float(ds.std(ddof=1) / np.sqrt(len(ds))))
        return deltas, sems

    log("\n" + f"{'arm':<18}" + "".join(f"{f'a={a:g}':>14}" for a in alphas))
    for nm, W in writes.items():
        deltas, sems = score(W)
        out["arms"][nm] = {"alphas": list(alphas), "delta_margin": deltas,
                           "sem": sems, "n_docs": len(tests)}
        log(f"{nm:<18}" + "".join(f"{v:>8.2f}+-{e:<5.2f}"
                                  for v, e in zip(deltas, sems)))

    # ---------------- permuted-profile null (optional) ----------------
    if n_perm and "txc_slab" in writes:
        best_a = alphas[int(np.argmax(out["arms"]["txc_slab"]["delta_margin"]))]
        vals = []
        for pi in range(n_perm):
            gp = torch.Generator(device="cpu").manual_seed(1000 + pi)
            perm = torch.randperm(T, generator=gp).to(dev)
            dp, _ = score(sign_t * unit(P_txc[perm]))
            vals.append(dp[alphas.index(best_a)])
        out["perm_null"] = {"alpha": best_a, "n": n_perm,
                            "mean": float(np.mean(vals)),
                            "sd": float(np.std(vals, ddof=1)),
                            "values": [float(v) for v in vals]}
        log(f"[perm null] n={n_perm} at a={best_a:g}: "
            f"{np.mean(vals):+.2f} +- {np.std(vals, ddof=1):.2f}  "
            f"(txc_slab {max(out['arms']['txc_slab']['delta_margin']):+.2f})")

    # ---------------- verdict ----------------
    def at_best(nm):
        v = out["arms"][nm]
        j = int(np.argmax(v["delta_margin"]))
        return v["delta_margin"][j], v["sem"][j]

    log("\n===== verdict =====")
    for nm in sorted(out["arms"], key=lambda n: -max(out["arms"][n]["delta_margin"])):
        log(f"  {nm:<18} best delta-margin {max(out['arms'][nm]['delta_margin']):+.2f}")
    zs = {}
    if "txc_slab" in out["arms"]:
        ts2, te2 = at_best("txc_slab")
        for other in ("sae_broadcast", "tsae_broadcast", "txc_flat", "random_slab",
                      "random_broadcast"):
            if other in out["arms"]:
                os_, oe_ = at_best(other)
                zs[f"txc_slab_vs_{other}"] = float(
                    (ts2 - os_) / np.sqrt(te2 ** 2 + oe_ ** 2))
                log(f"  txc_slab vs {other:<18} {ts2:+.2f} vs {os_:+.2f}   "
                    f"z = {zs[f'txc_slab_vs_{other}']:.1f}")
    out["z"] = zs
    beats = [zs.get("txc_slab_vs_sae_broadcast", -9),
             zs.get("txc_slab_vs_random_slab", -9),
             zs.get("txc_slab_vs_txc_flat", -9)]
    out["win"] = bool(out["arms"].get("txc_slab") and
                      max(out["arms"]["txc_slab"]["delta_margin"]) > 0 and
                      all(z > 2 for z in beats))
    log(f"\n  WIN (txc_slab > 0 and z > 2 vs sae_broadcast, random_slab, txc_flat): "
        f"{out['win']}")
    return out
