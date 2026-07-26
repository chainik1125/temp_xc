"""Reading is not steering: the sharpest available test of a crosscoder advantage.

On the order-only task -- two classes with identical multisets and identical switch counts,
differing only in which block came first -- a single mean-pooled TopK SAE latent reads the
label at AUC 1.000, while the crosscoder manages 0.833. By the reading criterion the SAE
wins outright, and every "can a window code read temporal structure?" experiment in this
sprint came out the same way, because a causal transformer has already written its own
history into every token.

Steering is a different question, and there is a structural reason to expect the ordering to
flip. A per-token dictionary's only per-latent intervention is ONE decoder direction applied
at every position: its write is CONSTANT IN TIME. The two classes here are reorderings of
the same multiset, so a constant write pushes both equally and cannot prefer one. A
crosscoder latent writes a (T, d) slab -- a different vector at each position -- so it can
push "tense early, calm late" without pushing "calm early, tense late".

    SAE latent write:   x_t <- x_t + alpha * v          for every t     (constant in t)
    TXC latent write:   x_t <- x_t + alpha * P[t]       for every t     (varies with t)

METRIC. Teacher-forced margin between the two orderings, which needs no sampling and no
judge. For a document whose true continuation is class A, take the class-B reordering of the
SAME sentences as the foil, and measure

    margin = logP(A | prompt) - logP(B | prompt)
    delta  = margin_steered - margin_base

averaged over documents, with the write applied at the same activation site the dictionaries
were trained on. Because A and B are multiset-matched, any effect from generic "more tense
tokens" cancels, and delta isolates the ordering.

ARMS
  sae_broadcast   best order-latent of the TopK SAE, its decoder direction added at every
                  position. This is the only per-latent write a per-token dictionary has.
  txc_slab        best order-latent of the crosscoder, its (T, d) decoder slab added
                  position by position.
  txc_flat        CONTROL: the crosscoder's slab averaged over time and then broadcast, so
                  it becomes constant like the SAE's. This isolates whether the advantage
                  comes from the temporal PROFILE or merely from the direction being better.
  dom_slab        supervised ceiling: per-position difference-of-means between class A and
                  class B activations, added position by position.

All writes are rescaled to the same total injected norm, so nothing is won by writing harder.

REGISTERED PREDICTIONS (written before the run):
  S1  sae_broadcast delta is ~0 -- a constant write cannot separate two orderings of one
      multiset. If it is clearly positive, the constant-write argument is wrong and the
      crosscoder has no structural advantage here either.
  S2  txc_slab delta is clearly positive and beats sae_broadcast.
  S3  txc_flat sits between them and closer to sae_broadcast, showing the effect comes from
      the temporal profile rather than from the direction alone. If txc_flat matches
      txc_slab, the profile is not doing the work and S2 is not evidence of a temporal
      advantage.
  S4  dom_slab is the largest, since it is supervised.
"""
import pathlib
import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

app = modal.App("dictbench-steerorder")
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
CARRIERS = ["Journal entry.\n", "From the notebook:\n", "Draft passage.\n",
            "Field notes.\n", "Evening record.\n", "From chapter twelve:\n"]


@app.function(gpu="A10G", image=image, timeout=21600)
def steerorder(model_id: str, layer: int, k_seg: int, n_train: int, n_test: int,
               d_sae: int, k: int, steps: int, lr: float, batch_win: int,
               alphas: list):
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
    T = k_seg
    rng = random.Random(31415)
    cap = {}

    def cap_hook(_m, _i, out):
        cap["h"] = (out[0] if isinstance(out, tuple) else out).detach()

    def make_doc(tense_first):
        half = k_seg // 2
        ts = [TENSE[rng.randrange(10)] for _ in range(half)]
        cs = [CALM[rng.randrange(10)] for _ in range(half)]
        sents = (ts + cs) if tense_first else (cs + ts)
        car = CARRIERS[rng.randrange(len(CARRIERS))]
        text, spans = car, []
        for j, s in enumerate(sents):
            if j:
                text += " "
            spans.append((len(text), len(text) + len(s)))
            text += s
        return text, spans, (ts, cs, car)

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

    # ---------------- training activations ----------------
    X, y = [], []
    for i in range(n_train):
        tf = rng.randint(0, 1)
        text, spans, _ = make_doc(tf)
        X.append(capture(text, spans)); y.append(tf)
        if (i + 1) % 250 == 0:
            print(f"   [cache] {i+1}/{n_train}", flush=True)
    Xt = torch.stack(X).to(dev)
    yt = torch.tensor(y, device=dev)
    mu, sd = Xt.mean((0, 1), keepdim=True), Xt.std() + 1e-6
    Xn = (Xt - mu) / sd
    print(f"[cache] {tuple(Xn.shape)}", flush=True)

    def gen_win(bs):
        return Xn[torch.randint(0, Xn.shape[0], (bs,), device=dev)]

    def gen_flat(bs):
        f = Xn.reshape(-1, d)
        return f[torch.randint(0, f.shape[0], (bs,), device=dev)]

    def adam_train(m, gen, bs):
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        m.train()
        for s in range(steps):
            loss, _, _ = m(gen(bs))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); m._normalize_decoder()
        m.eval()

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

    # ---------------- train both dictionaries ----------------
    torch.manual_seed(0)
    sae = TopKSAE(d_in=d, d_sae=d_sae, k=k).to(dev)
    adam_train(sae, gen_flat, batch_win * T)
    with torch.no_grad():
        Zs = sae.encode(Xn.reshape(-1, d)).reshape(-1, T, d_sae).mean(1)
    js, auc_s_signed, auc_s = best_latent(Zs, yt)
    v_sae = sae.W_dec.data[:, js].float()
    v_sae = v_sae / v_sae.norm()
    sign_s = 1.0 if auc_s_signed > 0.5 else -1.0
    print(f"[sae] latent {js} pooled AUC {auc_s:.3f}", flush=True)

    torch.manual_seed(0)
    txc = TemporalCrosscoder(d_in=d, d_sae=d_sae, T=T, k=k,
                             activation="batchtopk").to(dev)
    with torch.no_grad():
        txc._normalize_decoder()
    adam_train(txc, gen_win, batch_win)
    with torch.no_grad():
        Zt = txc.encode(Xn)
    jt, auc_t_signed, auc_t = best_latent(Zt, yt)
    P_txc = txc.W_dec.data[jt].float()
    P_txc = P_txc / P_txc.norm()
    sign_t = 1.0 if auc_t_signed > 0.5 else -1.0
    print(f"[txc] latent {jt} window AUC {auc_t:.3f}", flush=True)

    # Supervised ceiling: per-position difference of means, tense-first minus calm-first.
    with torch.no_grad():
        P_dom = (Xn[yt == 1].mean(0) - Xn[yt == 0].mean(0)).float()
        P_dom = P_dom / P_dom.norm()
    P_flat = P_txc.mean(0, keepdim=True).expand(T, -1).contiguous()
    P_flat = P_flat / P_flat.norm()
    V_sae = v_sae.unsqueeze(0).expand(T, -1).contiguous()
    V_sae = V_sae / V_sae.norm()

    # Nulls at identical injected norm. `random_slab` is the direct control for
    # txc_slab: a temporal profile that carries no learned information. Without it, a
    # positive txc_slab could be any structured perturbation rather than this one.
    g = torch.Generator(device="cpu").manual_seed(7)
    R = torch.randn(T, d, generator=g).to(dev)
    R = R / R.norm()
    rv = torch.randn(d, generator=g).to(dev)
    rv = rv / rv.norm()
    R_flat = rv.unsqueeze(0).expand(T, -1).contiguous()
    R_flat = R_flat / R_flat.norm()

    writes = {"sae_broadcast": sign_s * V_sae, "txc_slab": sign_t * P_txc,
              "txc_flat": sign_t * P_flat, "dom_slab": P_dom,
              "random_slab": R, "random_broadcast": R_flat}
    for nm, W in writes.items():
        print(f"[write] {nm:<14} total norm {float(W.norm()):.3f}  "
              f"per-position spread {float(W.norm(dim=-1).std()):.4f}", flush=True)

    # ---------------- teacher-forced margin ----------------
    scale = float(Xt.norm(dim=-1).mean())

    def margin(text, spans, W, alpha):
        """logP(text) with the write applied over the segment token spans."""
        e, ts = seg_spans(text, spans)
        ids = e["input_ids"].to(dev)
        if W is None:
            hook = None
        else:
            def edit(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                for t_i, (a, b) in enumerate(ts):
                    h[:, a:b + 1, :] += (alpha * scale
                                         * W[t_i].to(h.dtype).unsqueeze(0))
                return (h,) + out[1:] if isinstance(out, tuple) else h
            hook = layers_[L].register_forward_hook(edit)
        with torch.no_grad():
            lg = model(ids).logits.float().log_softmax(-1)
        if hook is not None:
            hook.remove()
        tgt = ids[0, 1:]
        return float(lg[0, :-1].gather(-1, tgt.unsqueeze(-1)).sum())

    tests = []
    for _ in range(n_test):
        text_a, spans_a, (ts_, cs_, car) = make_doc(True)
        sents_b = cs_ + ts_
        text_b, spans_b = car, []
        for j, s in enumerate(sents_b):
            if j:
                text_b += " "
            spans_b.append((len(text_b), len(text_b) + len(s)))
            text_b += s
        tests.append((text_a, spans_a, text_b, spans_b))

    out = {"layer": int(L), "k": k, "d_sae": d_sae, "T": T,
           "sae_latent": js, "sae_pooled_auc": auc_s,
           "txc_latent": jt, "txc_window_auc": auc_t, "arms": {}}

    print(f"\n{'arm':<16}" + "".join(f"{f'a={a:g}':>12}" for a in alphas), flush=True)
    # Base margins do not depend on the arm, so compute them once.
    base_by_doc = [margin(ta, sa, None, 0) - margin(tb, sb, None, 0)
                   for (ta, sa, tb, sb) in tests]

    for nm, W in writes.items():
        deltas, sems = [], []
        for a in alphas:
            ds = []
            for (ta, sa, tb, sb), base in zip(tests, base_by_doc):
                st = margin(ta, sa, W, a) - margin(tb, sb, W, a)
                ds.append(st - base)
            ds = np.array(ds)
            deltas.append(float(ds.mean()))
            sems.append(float(ds.std(ddof=1) / np.sqrt(len(ds))))
        out["arms"][nm] = {"alphas": alphas, "delta_margin": deltas,
                           "sem": sems, "n_docs": len(tests)}
        print(f"{nm:<16}" + "".join(f"{v:>8.2f}+-{e:<5.2f}"
                                    for v, e in zip(deltas, sems)), flush=True)

    print("\n===== verdict =====", flush=True)
    best = {nm: max(v["delta_margin"]) for nm, v in out["arms"].items()}
    for nm, v in sorted(best.items(), key=lambda x: -x[1]):
        print(f"  {nm:<16} best delta-margin {v:+.2f}", flush=True)
    # Separation in units of the pooled standard error, at each arm's own best dose.
    def at_best(nm):
        v = out["arms"][nm]
        j = int(np.argmax(v["delta_margin"]))
        return v["delta_margin"][j], v["sem"][j]

    ts_, te_ = at_best("txc_slab")
    ss_, se_ = at_best("sae_broadcast")
    rs_, re_ = at_best("random_slab")
    z_sae = (ts_ - ss_) / np.sqrt(te_ ** 2 + se_ ** 2)
    z_rnd = (ts_ - rs_) / np.sqrt(te_ ** 2 + re_ ** 2)
    print(f"\n  txc_slab vs sae_broadcast : {ts_:+.2f} vs {ss_:+.2f}  "
          f"z = {z_sae:.1f}", flush=True)
    print(f"  txc_slab vs random_slab   : {ts_:+.2f} vs {rs_:+.2f}  "
          f"z = {z_rnd:.1f}", flush=True)
    if ts_ > 0 and z_sae > 2 and z_rnd > 2:
        print("  -> S1/S2 HOLD: the crosscoder steers order where a constant write cannot,"
              " and beats a random temporal profile", flush=True)
    else:
        print("  -> NOT ESTABLISHED at this power: see the z values above", flush=True)
    return out


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-1.5B-Instruct", layer: int = -1, k_seg: int = 12,
         n_train: int = 800, n_test: int = 60, d_sae: int = 4096, k: int = 8,
         steps: int = 2000, lr: float = 3e-4, batch_win: int = 32,
         alphas: str = "0.25,0.5,1.0,2.0", tag: str = ""):
    import json
    r = steerorder.remote(model, layer, k_seg, n_train, n_test, d_sae, k, steps, lr,
                          batch_win, [float(x) for x in alphas.split(",")])
    outdir = ROOT / "results" / "dict_bench"
    outdir.mkdir(parents=True, exist_ok=True)
    name = f"steer_order{tag}.json"
    (outdir / name).write_text(json.dumps(r, indent=2))
    print("[saved]", outdir / name)
