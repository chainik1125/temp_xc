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
  5. STEERING = teacher-forced delta margin, with the decoder direction or slab added at the
     training layer, every write rescaled to identical total injected norm so nothing is won
     by writing harder. Two scoring modes:

       ORDERING MODE (default). score(doc) = logP(doc), and the reported quantity is
       [score(A) - score(B)] steered minus baseline. Because A and B are one multiset in two
       orders, anything that depends only on which sentences are present cancels.

       PROBE MODE. A task may additionally supply two CONTINUATIONS, and then
       score(doc) = logP(cont1 | doc) - logP(cont2 | doc) -- a behavioural quantity, "which
       way does the model resolve this", rather than a statement about which document is
       more likely. The reported quantity is again [score(A) - score(B)] steered minus
       baseline, and that difference-of-differences is what makes the mode worth having: a
       write that simply adds "more cont1" pushes score(A) and score(B) by the same amount
       and cancels exactly. Only a write that treats positions differently can move it.
       This is how a task about a real behaviour -- which of two conflicting instructions
       the model obeys, say -- gets a metric that a constant write cannot win.
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
             steps, lr, batch_win, alphas, tsae_l1=None, tsae_k=None, txc_k=None,
             n_perm=0, seed=31415, dict_seed=0, arms=None, verbose=True):
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

    def draw(rng_):
        """make_pair may return 3 items (ordering mode) or 5 (probe mode)."""
        p = make_pair(rng_)
        return p if len(p) == 5 else (p[0], p[1], p[2], None, None)

    # ---------------- training activations, balanced over the two classes ----------------
    X, y = [], []
    for i in range(n_train):
        sa, sb, car, _, _ = draw(rng)
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
    # HELD-OUT SPLIT, and it is not a formality. Two things were being measured in-sample
    # before it existed, and both flattered the crosscoder:
    #   * REALISED SPARSITY. BatchTopK is a batch rule at train time and a fixed threshold
    #     at eval time, and the number of latents clearing that threshold is data-dependent.
    #     Measured on the training documents the crosscoder reports 8.03 coefficients per
    #     segment at nominal k=8; measured on held-out documents from the same distribution
    #     it spends 11.5. The arms were being matched on the flattering number.
    #   * READING AUC. Taking the best of 4096 latents and scoring it on the same documents
    #     the maximum was taken over is a selection-biased estimate. The latent is now chosen
    #     on the training split and scored on the held-out one.
    n_hold = max(int(0.15 * Xn.shape[0]), 64)
    Xtr, ytr = Xn[:-n_hold], yt[:-n_hold]
    Xho, yho = Xn[-n_hold:], yt[-n_hold:]
    log(f"[cache] {tuple(Xn.shape)}  class balance {float(yt.float().mean()):.3f}  "
        f"train {Xtr.shape[0]} holdout {Xho.shape[0]}")

    def gen_win(bs):
        return Xtr[torch.randint(0, Xtr.shape[0], (bs,), device=dev)]

    def gen_flat(bs):
        f = Xtr.reshape(-1, d)
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

    def best_latent(Ztr, Zho):
        """Choose the latent on the training split, report its AUC on the held-out one.

        Returns (index, oriented holdout AUC, sign, selection AUC on train).
        """
        a = torch.tensor([auc_of(Ztr[:, j], ytr) for j in range(Ztr.shape[1])], device=dev)
        dv = (a - 0.5).abs()
        j = int(dv.argmax())
        sign = 1.0 if float(a[j]) > 0.5 else -1.0
        a_ho = auc_of(Zho[:, j], yho)
        return j, float(0.5 + abs(a_ho - 0.5)), sign, float(0.5 + dv[j])

    out = {"model": model_id, "layer": int(L), "k_seg": k_seg, "T": T, "d_sae": d_sae,
           "k": k, "steps": steps, "lr": lr, "n_train": n_train, "n_test": n_test,
           "seed": seed, "dict_seed": dict_seed,
           "alphas": list(alphas), "reading": {}, "sparsity": {}, "arms": {}}
    writes = {}

    # ---------------- 1. per-token TopK SAE ----------------
    torch.manual_seed(dict_seed)
    sae = TopKSAE(d_in=d, d_sae=d_sae, k=k).to(dev)
    adam_train(sae, gen_flat, batch_win * T)
    with torch.no_grad():
        Ztr_ = sae.encode(Xtr.reshape(-1, d)).reshape(-1, T, d_sae).mean(1)
        Zho_ = sae.encode(Xho.reshape(-1, d))
        Zho_pool = Zho_.reshape(-1, T, d_sae).mean(1)
    js, auc_s, sign_s, auc_s_tr = best_latent(Ztr_, Zho_pool)
    out["reading"]["sae"] = {"latent": js, "auc": auc_s, "auc_selection": auc_s_tr}
    out["sparsity"]["sae"] = float((Zho_ > 0).float().sum(-1).mean())
    log(f"[sae] latent {js} pooled AUC {auc_s:.3f} (holdout; {auc_s_tr:.3f} in-sample)  "
        f"realised {out['sparsity']['sae']:.2f} coeff/segment")
    v_sae = unit(sae.W_dec.data[:, js].float())
    writes["sae_broadcast"] = sign_s * unit(v_sae.unsqueeze(0).expand(T, -1).contiguous())

    # ---------------- 2. Temporal Crosscoder ----------------
    torch.manual_seed(dict_seed)
    # Nominal k is not the comparison axis. BatchTopK's eval threshold lets the
    # crosscoder overspend on held-out data -- 11.5 coefficients per segment at nominal
    # k=8 -- so `txc_k` exists to be lowered until its REALISED held-out spend matches the
    # SAE's, which is exactly k because TopK binds exactly.
    txc = TemporalCrosscoder(d_in=d, d_sae=d_sae, T=T, k=(txc_k if txc_k else k),
                             activation="batchtopk").to(dev)
    with torch.no_grad():
        txc._normalize_decoder()
    adam_train(txc, gen_win, batch_win)
    with torch.no_grad():
        Zt_tr, Zt_ho = txc.encode(Xtr), txc.encode(Xho)
    jt, auc_t, sign_t, auc_t_tr = best_latent(Zt_tr, Zt_ho)
    out["reading"]["txc"] = {"latent": jt, "auc": auc_t, "auc_selection": auc_t_tr,
                             "nominal_k": int(txc_k if txc_k else k)}
    out["sparsity"]["txc"] = float((Zt_ho != 0).float().sum(-1).mean()) / T
    out["sparsity"]["txc_insample"] = float((Zt_tr != 0).float().sum(-1).mean()) / T
    log(f"[txc] latent {jt} window AUC {auc_t:.3f} (holdout; {auc_t_tr:.3f} in-sample)  "
        f"realised {out['sparsity']['txc']:.2f} coeff/segment "
        f"({out['sparsity']['txc_insample']:.2f} in-sample)")
    P_txc = unit(txc.W_dec.data[jt].float())
    writes["txc_slab"] = sign_t * P_txc
    writes["txc_flat"] = sign_t * unit(P_txc.mean(0, keepdim=True).expand(T, -1).contiguous())

    # ---------------- 3. attention tSAE ----------------
    # Two sparsity rules, because the L1 form the repo documents cannot be used here. The
    # calibration (`tsae_calib_modal.py`) shows it reaches the 1-32 coefficient band only by
    # collapsing -- FVU crosses 1.0 before L0 crosses 32 -- so `tsae_k` runs the same
    # attention architecture with TopK, which binds by construction and can therefore be
    # matched to the SAE and crosscoder on realised coefficients per segment. `tsae_l1` is
    # kept for the record.
    if tsae_l1 is not None or tsae_k is not None:
        from temporal_crosscoders.han_tsae import TemporalSAE
        use_topk = tsae_k is not None
        torch.manual_seed(dict_seed)
        ts_ = TemporalSAE(dimin=d, width=d_sae, n_heads=8,
                          sae_diff_type="topk" if use_topk else "relu",
                          kval_topk=int(tsae_k) if use_topk else None,
                          tied_weights=True, n_attn_layers=1,
                          bottleneck_factor=64).to(dev)
        opt = torch.optim.Adam(ts_.parameters(), lr=lr)
        ts_.train()
        for _ in range(steps):
            xb = gen_win(batch_win)
            recons, info = ts_(xb)
            loss = (recons - xb).pow(2).sum(-1).mean()
            if not use_topk:
                loss = loss + tsae_l1 * (info["novel_codes"]
                                         + info["pred_codes"]).abs().sum(-1).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(ts_.parameters(), 1.0)
            opt.step()
        ts_.eval()
        with torch.no_grad():
            _, info_tr = ts_(Xtr)
            _, info = ts_(Xho)
            Zn_all = info["novel_codes"]
        jn, auc_n, sign_n, auc_n_tr = best_latent(
            info_tr["novel_codes"].mean(1), Zn_all.mean(1))
        out["reading"]["tsae"] = {"latent": jn, "auc": auc_n, "auc_selection": auc_n_tr,
                                  "rule": "topk" if use_topk else "relu_l1",
                                  "kval": tsae_k, "l1_coef": tsae_l1}
        out["sparsity"]["tsae"] = float(
            (Zn_all.reshape(-1, d_sae) > 0).float().sum(-1).mean())
        # The predicted codes are computed from context rather than stored, so they are not
        # charged to coefficients-per-segment -- but they are recorded, because a reader who
        # disagrees with that accounting needs the number.
        out["sparsity"]["tsae_pred"] = float(
            (info["pred_codes"].reshape(-1, d_sae) > 0).float().sum(-1).mean())
        log(f"[tsae] latent {jn} pooled AUC {auc_n:.3f} "
            f"(holdout; {auc_n_tr:.3f} in-sample)  "
            f"realised {out['sparsity']['tsae']:.2f} coeff/segment "
            f"(+{out['sparsity']['tsae_pred']:.1f} predicted)  "
            f"rule={'topk k=%d' % tsae_k if use_topk else 'relu+l1 %g' % tsae_l1}")
        # The tSAE's temporal machinery is all in the ENCODER; its dictionary D gives one
        # direction per latent, so its per-latent write is constant in time exactly like
        # the SAE's. That structural fact is the point of including the arm.
        v_ts = unit(ts_.D.data[jn].float())
        writes["tsae_broadcast"] = sign_n * unit(
            v_ts.unsqueeze(0).expand(T, -1).contiguous())

    # ---------------- 4. controls ----------------
    with torch.no_grad():
        writes["dom_slab"] = unit((Xtr[ytr == 1].mean(0)
                                   - Xtr[ytr == 0].mean(0)).float())
    g = torch.Generator(device="cpu").manual_seed(7)
    writes["random_slab"] = unit(torch.randn(T, d, generator=g).to(dev))
    rv = unit(torch.randn(d, generator=g).to(dev))
    writes["random_broadcast"] = unit(rv.unsqueeze(0).expand(T, -1).contiguous())

    if arms:
        writes = {nm: W for nm, W in writes.items() if nm in arms}
    # The per-position norm profile is the mechanism, so it is stored rather than
    # summarised. On a task whose factor lives at particular positions, a crosscoder slab
    # that works should put its mass there, and `txc_flat` -- which has spread exactly
    # 0.0000 by construction -- is the same vector with that structure removed.
    out["write_profile"] = {nm: [float(v) for v in W.norm(dim=-1)]
                            for nm, W in writes.items()}
    for nm, W in writes.items():
        log(f"[write] {nm:<17} total norm {float(W.norm()):.3f}  "
            f"per-position spread {float(W.norm(dim=-1).std()):.4f}")

    # ---------------- teacher-forced margin ----------------
    scale = float(Xt.norm(dim=-1).mean())

    def logits_for(text, spans, cont, W, alpha):
        """Run the document (plus optional continuation) with the write applied.

        The write covers the document's SEGMENT token spans only. A continuation is
        appended untouched, so it is influenced only through attention -- which is the
        point: the intervention is on the trajectory, and the behaviour read out
        downstream of it.
        """
        e, ts2 = seg_spans(text, spans)
        ids = e["input_ids"].to(dev)
        n_doc = ids.shape[1]
        if cont is not None:
            c_ids = torch.tensor(
                [tok(cont, add_special_tokens=False)["input_ids"]], device=dev)
            ids = torch.cat([ids, c_ids], dim=1)
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
        if cont is None:
            tgt = ids[0, 1:]
            return float(lg[0, :-1].gather(-1, tgt.unsqueeze(-1)).sum())
        tgt = ids[0, n_doc:]
        return float(lg[0, n_doc - 1:-1].gather(-1, tgt.unsqueeze(-1)).sum())

    def doc_score(text, spans, c1, c2, W, alpha):
        if c1 is None:
            return logits_for(text, spans, None, W, alpha)
        return (logits_for(text, spans, c1, W, alpha)
                - logits_for(text, spans, c2, W, alpha))

    tests = []
    for _ in range(n_test):
        sa, sb, car, c1, c2 = draw(rng)
        ta, spa = build(car, sa)
        tb, spb = build(car, sb)
        tests.append((ta, spa, tb, spb, c1, c2))
    probe_mode = tests[0][4] is not None
    out["probe_mode"] = probe_mode
    base_by_doc = [doc_score(ta, sa2, c1, c2, None, 0)
                   - doc_score(tb, sb2, c1, c2, None, 0)
                   for (ta, sa2, tb, sb2, c1, c2) in tests]
    out["baseline_contrast"] = {
        "mean": float(np.mean(base_by_doc)),
        "sem": float(np.std(base_by_doc, ddof=1) / np.sqrt(len(base_by_doc)))}
    log(f"[baseline] score(A) - score(B) unsteered: "
        f"{out['baseline_contrast']['mean']:+.2f} "
        f"+- {out['baseline_contrast']['sem']:.2f}"
        + ("   (probe mode: logP(cont1) - logP(cont2))" if probe_mode else ""))

    def score(W):
        deltas, sems = [], []
        for a in alphas:
            ds = np.array([doc_score(ta, sa2, c1, c2, W, a)
                           - doc_score(tb, sb2, c1, c2, W, a) - base
                           for (ta, sa2, tb, sb2, c1, c2), base
                           in zip(tests, base_by_doc)])
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
