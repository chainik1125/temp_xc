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
     removes the profile and keeps the direction), `txc_profile_random` (the mirror image --
     the profile kept exactly, the directions replaced by random ones), `random_slab`,
     `random_broadcast`, and `dom_slab` (supervised per-position difference-of-means
     ceiling).

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


def _upper_frac(texts):
    """Fraction of alphabetic characters that are upper case, pooled over texts.

    The readout for the recency task, chosen because it needs no judge: the two conflicting
    instructions are 'write in capitals' and 'write in small letters', so which one the model
    obeyed is legible directly from the characters it emitted.
    """
    chars = [c for t in texts for c in t if c.isalpha()]
    return sum(c.isupper() for c in chars) / max(len(chars), 1)


def run_task(*, make_pair, model_id, layer, k_seg, n_train, n_test, d_sae, k,
             make_pair_test=None,
             steps, lr, batch_win, alphas, tsae_l1=None, tsae_k=None, txc_k=None,
             n_perm=0, seed=31415, dict_seed=0, gen_tokens=0, n_gen=0,
             n_grad=0, sae_lr=None, txc_lr=None, tsae_lr=None,
             sae_steps=None, txc_steps=None, tsae_steps=None,
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

    def draw(rng_, fn=None):
        """make_pair may return 3 items (ordering mode) or 5 (probe mode)."""
        p = (fn or make_pair)(rng_)
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

    def adam_train(m, gen, bs, lr_=None, steps_=None):
        """Per-arm learning rate and step count.

        Not a convenience. Swept properly, the three architectures want DIFFERENT recipes on
        the same activations -- best FVU at 8 coefficients/segment on the recency corpus is
        0.0373 for the SAE at lr 3e-4, 0.0968 for the crosscoder at 1e-3 (it diverges to 0.36
        at 3e-3), and 0.0144 for the attention tSAE at 3e-3. Holding one learning rate across
        all three does not make the comparison fair, it just picks which architecture the
        comparison handicaps: at the sprint's original default the tSAE looked 3.4x WORSE than
        the SAE, and at its own best recipe it is 2.6x BETTER.
        """
        opt = torch.optim.Adam(m.parameters(), lr=lr_ or lr)
        m.train()
        for _ in range(int(steps_ or steps)):
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
           "recipe": {"sae": [sae_lr or lr, sae_steps or steps],
                      "txc": [txc_lr or lr, txc_steps or steps],
                      "tsae": [tsae_lr or lr, tsae_steps or steps]},
           "seed": seed, "dict_seed": dict_seed,
           "held_out_content": make_pair_test is not None,
           "alphas": list(alphas), "reading": {}, "sparsity": {}, "arms": {}}
    writes = {}

    # ---------------- 1. per-token TopK SAE ----------------
    torch.manual_seed(dict_seed)
    sae = TopKSAE(d_in=d, d_sae=d_sae, k=k).to(dev)
    adam_train(sae, gen_flat, batch_win * T, sae_lr, sae_steps)
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
    adam_train(txc, gen_win, batch_win, txc_lr, txc_steps)
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
        opt = torch.optim.Adam(ts_.parameters(), lr=tsae_lr or lr)
        ts_.train()
        for _ in range(int(tsae_steps or steps)):
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

    # RANK DECOMPOSITION OF THE OPTIMAL WRITE, and the arm that follows from it.
    #
    # A steering intervention is a (T, d) MATRIX. A per-token dictionary has one direction
    # per latent, so the most it can produce -- even when steered perfectly, with a
    # per-position dose schedule -- is a RANK-1 matrix: one direction with a time-varying
    # gain. A crosscoder latent is unconstrained in rank. So the question "can a per-token
    # dictionary express this write at all?" is answered before any dictionary is trained, by
    # the singular values of the supervised difference-of-means slab:
    #
    #     r1 = sigma_1^2 / ||P||_F^2      share of the optimal write reachable at rank 1
    #     c  = T ||mean_t P_t||^2 / ||P||_F^2   share reachable by a CONSTANT write
    #
    # `c` is the sharper of the two for interpreting this project, because it is also the
    # share a pooled per-token probe can READ: a pooled probe averages over positions and so
    # separates the classes exactly when the mean of the difference slab is nonzero, which is
    # the same quantity a constant write pushes along. Reading and constant-write steering
    # are one number seen twice.
    #
    # `rank1_best` is then the ceiling for ANY per-token dictionary handed a perfect
    # schedule, and `sae_schedule` is the SAE's OWN learned direction given the best schedule
    # for it. Both are fitted on the training split only and applied unrefitted to the fresh
    # test documents, so a schedule that does not generalise shows up as a failure here
    # rather than being smuggled in. Together they separate two claims that both get called
    # "the crosscoder wins":
    #   * txc_slab > rank1_best  -> EXPRESSIVENESS: no per-token write can do this.
    #   * txc_slab ~ rank1_best but >> sae_broadcast -> DISCOVERY: the write exists for a
    #     per-token dictionary, but unsupervised training does not find it and supplying it
    #     needs knowing the answer first.
    P_dom = writes["dom_slab"]
    _U, _S, _Vh = torch.linalg.svd(P_dom, full_matrices=False)
    out["rank"] = {
        "r1": float(_S[0] ** 2 / (_S ** 2).sum()),
        "c": float(T * P_dom.mean(0).pow(2).sum() / P_dom.pow(2).sum()),
        "singular_values": [float(v) for v in _S[:6]],
    }
    log(f"[rank] optimal write: rank-1 share r1 = {out['rank']['r1']:.3f}, "
        f"constant share c = {out['rank']['c']:.3f}")
    writes["rank1_best"] = unit(_S[0] * torch.outer(_U[:, 0], _Vh[0]))
    writes["sae_schedule"] = unit(torch.outer(P_dom @ v_sae, v_sae))

    # `txc_flat` asks whether the crosscoder needs its profile. This asks the complementary
    # question: does it need anything BUT the profile? Random directions, one per position,
    # rescaled so the per-position norms exactly match `txc_slab`'s. If this matches
    # `txc_slab`, the crosscoder's contribution is knowing WHERE to write and the learned
    # directions are doing nothing -- which would be a much weaker claim than it sounds,
    # since where-to-write is a single number per position that any supervised probe
    # supplies. If it fails, the crosscoder is contributing both.
    Rp = torch.randn(T, d, generator=g).to(dev)
    Rp = Rp / (Rp.norm(dim=-1, keepdim=True) + 1e-12) * P_txc.norm(dim=-1, keepdim=True)
    writes["txc_profile_random"] = unit(Rp)

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

    # ---------------- the metric's own gradient: the true optimal write ----------------
    # `dom_slab` is the difference of class means, which is A supervised write but not THE
    # optimal one -- on the evidence task the crosscoder beat it, so calling it a ceiling was
    # wrong. The optimal infinitesimal intervention for the metric actually being reported is
    # its gradient with respect to the write, evaluated at zero:
    #
    #     G[t] = d/dW[t] ( score(A) - score(B) )  at  W = 0
    #
    # Steering along G is the best any (T, d) write can do to first order, so its rank
    # decomposition is the r1 and c that the theory wants, and `grad_rank1` is the honest
    # ceiling for a per-token dictionary handed a perfect schedule. The gradient is
    # accumulated over `n_grad` documents drawn from the same stream as the test set but
    # BEFORE it, so nothing is fitted on the documents it is scored on.
    if n_grad:
        for p_ in model.parameters():
            p_.requires_grad_(False)
        scale_g = float(Xt.norm(dim=-1).mean())
        Wg = torch.zeros(T, d, device=dev, dtype=torch.float32, requires_grad=True)

        def grad_pass(text, spans, cont, sgn):
            e, ts2 = seg_spans(text, spans)
            ids = e["input_ids"].to(dev)
            n_doc = ids.shape[1]
            if cont is not None:
                ids = torch.cat([ids, torch.tensor(
                    [tok(cont, add_special_tokens=False)["input_ids"]], device=dev)], 1)

            def edit(_m, _i, out_):
                h = out_[0] if isinstance(out_, tuple) else out_
                add = torch.zeros_like(h, dtype=torch.float32)
                for t_i, (a_, b_) in enumerate(ts2):
                    add[:, a_:b_ + 1, :] = scale_g * Wg[t_i].unsqueeze(0)
                h = h + add.to(h.dtype)
                return (h,) + out_[1:] if isinstance(out_, tuple) else h

            hk = layers_[L].register_forward_hook(edit)
            lg = model(ids).logits.float().log_softmax(-1)
            hk.remove()
            if cont is None:
                s_ = lg[0, :-1].gather(-1, ids[0, 1:].unsqueeze(-1)).sum()
            else:
                s_ = lg[0, n_doc - 1:-1].gather(
                    -1, ids[0, n_doc:].unsqueeze(-1)).sum()
            (sgn * s_).backward()

        for _ in range(n_grad):
            sa, sb, car, c1, c2 = draw(rng, make_pair_test)
            ta_, spa_ = build(car, sa)
            tb_, spb_ = build(car, sb)
            if c1 is None:
                grad_pass(ta_, spa_, None, 1.0)
                grad_pass(tb_, spb_, None, -1.0)
            else:
                grad_pass(ta_, spa_, c1, 1.0); grad_pass(ta_, spa_, c2, -1.0)
                grad_pass(tb_, spb_, c1, -1.0); grad_pass(tb_, spb_, c2, 1.0)
        G = (Wg.grad / n_grad).detach()
        gU, gS, gVh = torch.linalg.svd(G, full_matrices=False)
        out["rank_grad"] = {
            "r1": float(gS[0] ** 2 / (gS ** 2).sum()),
            "c": float(T * G.mean(0).pow(2).sum() / G.pow(2).sum()),
            "singular_values": [float(v) for v in gS[:6]],
            "n_grad": n_grad,
            "cos_with_dom": float(
                (unit(G.reshape(-1)) * unit(P_dom.reshape(-1))).sum()),
        }
        log(f"[rank-grad] metric gradient over {n_grad} docs: r1 = "
            f"{out['rank_grad']['r1']:.3f}  c = {out['rank_grad']['c']:.3f}  "
            f"cos(grad, dom) = {out['rank_grad']['cos_with_dom']:+.3f}")
        writes["grad_slab"] = unit(G)
        writes["grad_rank1"] = unit(gS[0] * torch.outer(gU[:, 0], gVh[0]))
        # The SAE's OWN direction on the best schedule for it, taken from the gradient
        # rather than from difference-of-means. This is the arm that would defeat the
        # expressiveness claim: a per-token dictionary handed a per-position dose schedule.
        writes["sae_schedule_grad"] = unit(torch.outer(G @ v_sae, v_sae))
        out["write_profile"]["grad_slab"] = [float(v) for v in writes["grad_slab"].norm(dim=-1)]
        for _nm in ("grad_rank1", "sae_schedule_grad"):
            out["write_profile"][_nm] = [float(v) for v in writes[_nm].norm(dim=-1)]

    # HELD-OUT CONTENT. Without `make_pair_test` the dictionaries are trained on documents
    # built from the same sentence pools they are then asked to steer, so a latent could in
    # principle be keyed to those particular sentences rather than to the factor. The
    # ordering structure makes a pure lookup implausible -- both classes use the same
    # sentences -- but "steers the ordering of content it was trained on" is a weaker claim
    # than "steers this factor", and only disjoint pools separate them.
    tests = []
    for _ in range(n_test):
        sa, sb, car, c1, c2 = draw(rng, make_pair_test)
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

    # ---------------- steered generation (optional) ----------------
    # The margin metric is teacher-forced, which is why it needs no judge and no sampling --
    # and also why it is fair to ask whether it corresponds to anything the model DOES. This
    # generates greedily under the same writes at each arm's own best dose and stores the
    # text, so the log-probability claim can be checked against observable output. No hook is
    # applied to the generated tokens themselves: the write covers the document's segment
    # spans only, and the continuation is influenced through attention.
    if gen_tokens and n_gen:
        gen_arms = [nm for nm in ("txc_slab", "sae_broadcast", "txc_flat", "dom_slab")
                    if nm in writes]
        out["generations"] = {}
        for nm in ["none"] + gen_arms:
            W = None if nm == "none" else writes[nm]
            a_best = (0.0 if W is None
                      else alphas[int(np.argmax(out["arms"][nm]["delta_margin"]))])
            rows = []
            for (ta, sa2, tb, sb2, c1, _c2) in tests[:n_gen]:
                for cls, (txt, spn) in (("A", (ta, sa2)), ("B", (tb, sb2))):
                    stem = txt + (c1.rsplit(" ", 1)[0] if c1 else "")
                    e, ts2 = seg_spans(stem, spn)
                    ids = e["input_ids"].to(dev)
                    n0 = ids.shape[1]
                    hook = None
                    if W is not None:
                        def edit(_m, _i, out_, _ts=ts2, _W=W, _a=a_best):
                            h = out_[0] if isinstance(out_, tuple) else out_
                            for t_i, (p0, p1) in enumerate(_ts):
                                h[:, p0:p1 + 1, :] += (_a * scale
                                                       * _W[t_i].to(h.dtype).unsqueeze(0))
                            return (h,) + out_[1:] if isinstance(out_, tuple) else h
                        hook = layers_[L].register_forward_hook(edit)
                    with torch.no_grad():
                        for _ in range(gen_tokens):
                            nxt = model(ids).logits[0, -1].argmax()
                            ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
                    if hook is not None:
                        hook.remove()
                    rows.append({"cls": cls, "alpha": a_best,
                                 "text": tok.decode(ids[0, n0:])})
            out["generations"][nm] = rows
            up = {c: _upper_frac([r["text"] for r in rows if r["cls"] == c])
                  for c in ("A", "B")}
            log(f"[gen] {nm:<16} alpha {a_best:>5.2f}  uppercase fraction "
                f"A {up['A']:.3f}  B {up['B']:.3f}  A-B {up['A'] - up['B']:+.3f}")

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
    # TWO CONVENTIONS, BOTH STORED, BOTH NAMED FOR WHAT THEY ARE.
    #
    # A field called plain `z` is the same nominal-versus-effective trap this project keeps
    # hitting: it was peak-dose while the primary reporting convention became matched-dose,
    # and two readers reconciled different numbers from the same file before anyone noticed
    # the name did not say which. So neither is called `z`.
    #
    #   peak    -- every arm at its own best dose. Generous to every arm equally, but each
    #              arm sits at ITS OWN saturation point, which is outside the linear regime
    #              the rank framework describes, and a maximum taken over the grid inflates
    #              flat arms (simulated at 0.19-0.26, i.e. 0.3-0.4 SEM).
    #   matched -- every arm at the SMALLEST dose magnitude where the crosscoder is
    #              significant, sign still free per arm. Primary: it is in the linear regime
    #              and the selection is over two options rather than the whole grid.
    def at_best(nm):
        v = out["arms"][nm]
        j = int(np.argmax(v["delta_margin"]))
        return v["delta_margin"][j], v["sem"][j]

    def at_mag(nm, mag):
        v = out["arms"][nm]
        best = None
        for a_, d_, e_ in zip(v["alphas"], v["delta_margin"], v["sem"]):
            if abs(abs(a_) - mag) < 1e-9 and (best is None or d_ > best[0]):
                best = (d_, e_)
        return best

    mags = sorted({abs(a_) for a_ in out["arms"]["txc_slab"]["alphas"]}) \
        if "txc_slab" in out["arms"] else []
    matched_mag = next(
        (m for m in mags
         if (g := at_mag("txc_slab", m)) and g[0] > 2.0 * g[1]), None)

    log("\n===== verdict =====")
    for nm in sorted(out["arms"], key=lambda n: -max(out["arms"][n]["delta_margin"])):
        log(f"  {nm:<18} best delta-margin {max(out['arms'][nm]['delta_margin']):+.2f}")
    others = ("sae_broadcast", "tsae_broadcast", "txc_flat",
              "txc_profile_random", "random_slab", "random_broadcast",
              "sae_schedule", "rank1_best", "grad_slab", "grad_rank1",
              "sae_schedule_grad")
    z_peak, z_matched = {}, {}
    if "txc_slab" in out["arms"]:
        ts2, te2 = at_best("txc_slab")
        tm = at_mag("txc_slab", matched_mag) if matched_mag else None
        for other in others:
            if other not in out["arms"]:
                continue
            os_, oe_ = at_best(other)
            z_peak[f"txc_slab_vs_{other}"] = float(
                (ts2 - os_) / np.sqrt(te2 ** 2 + oe_ ** 2))
            if tm and (om := at_mag(other, matched_mag)):
                z_matched[f"txc_slab_vs_{other}"] = float(
                    (tm[0] - om[0]) / np.sqrt(tm[1] ** 2 + om[1] ** 2))
            log(f"  txc_slab vs {other:<18} peak {ts2:+.2f} vs {os_:+.2f} "
                f"z={z_peak[f'txc_slab_vs_{other}']:+.1f}" +
                (f"   matched(|a|={matched_mag:g}) "
                 f"z={z_matched[f'txc_slab_vs_{other}']:+.1f}"
                 if f"txc_slab_vs_{other}" in z_matched else ""))
    out["z_peak_dose"] = z_peak
    out["z_matched_dose"] = z_matched
    out["matched_dose_magnitude"] = matched_mag
    beats = [z_peak.get("txc_slab_vs_sae_broadcast", -9),
             z_peak.get("txc_slab_vs_random_slab", -9),
             z_peak.get("txc_slab_vs_txc_flat", -9)]
    out["win"] = bool(out["arms"].get("txc_slab") and
                      max(out["arms"]["txc_slab"]["delta_margin"]) > 0 and
                      all(z > 2 for z in beats))
    log(f"\n  WIN (txc_slab > 0 and z > 2 vs sae_broadcast, random_slab, txc_flat): "
        f"{out['win']}")
    return out
