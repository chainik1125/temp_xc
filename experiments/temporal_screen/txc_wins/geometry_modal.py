"""Task geometry only: the two slabs, their spectra, and how the SAE's learned direction
relates to them. No crosscoder, no tSAE, and — apart from one cheap SAE — no training.

WHY THIS IS A SEPARATE SCRIPT. `r1` and `c` are computed from slabs that involve no
dictionary at all, so they cannot move with a learning rate, a seed, or a training budget.
Bundling them into a full task run made them look like they inherited every caveat of the
arm comparisons, and it meant that answering a geometry question cost a full training matrix.
Everything here is a property of the task, the model and the metric.

THE TWO SLABS ARE DIFFERENT OBJECTS AND EVERY NUMBER MUST SAY WHICH IT CAME FROM.

    P_dom = mean(x | class A) - mean(x | class B)      what DISTINGUISHES the classes
    Gbar  = mean_docs d(margin)/dW at W = 0            what INCREASES the metric

Measured cos(P_dom, Gbar) has come in at 0.044 on recency and 0.003 on a rotation task
against a random baseline of 1/sqrt(T*d) ~ 0.007, i.e. essentially orthogonal. So a screen
run on `P_dom` is not a screen on the thing steering optimises, and the same task can give
r1 = 0.585 on one slab and 0.829 on the other. Both are reported here, always labelled.

    c  = T ||mean_t P[t]||^2 / ||P||_F^2    share a CONSTANT write can reach
    r1 = sigma_1^2 / ||P||_F^2              share ANY rank-1 write can reach

WHAT THE SINGULAR VECTORS ARE FOR. The rank-2 account of the instruction-position task
predicts two components with DISJOINT temporal support -- one on the instruction positions,
one on the spans they govern -- because the instruction's lexical content and the state of
which instruction is currently governing are different things living in different places.
That is falsifiable directly: dump the leading left singular vectors and look. If both
spread across all positions, the decomposition is wrong. Only the singular VALUES were being
stored, which cannot answer it, so `U[:, :3]` is written out here for both slabs.

WHAT THE SAE COSINE IS FOR. Two tasks can both be "discovery" results for opposite reasons,
and the distinction is one cosine:

    cos(v_sae, u1(Gbar))  high  -> the per-token dictionary already found the right
                                  DIRECTION and only lacked a schedule
                          low   -> it never found the direction, so no schedule rescues it

That separates "discovery of the schedule" from "discovery of the direction", and it is the
measurement that should carry that claim -- not a comparison against a scheduled arm, whose
fitted schedule can itself be the confound.

    modal run experiments/temporal_screen/txc_wins/geometry_modal.py --tasks recency,rot_m12
"""
import pathlib
import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

app = modal.App("txcwins-geometry")
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers", "accelerate", "numpy")
    .run_commands(
        "python -c \"from huggingface_hub import snapshot_download; "
        "snapshot_download('Qwen/Qwen2.5-1.5B-Instruct')\"")
    .add_local_dir(str(ROOT / "src"), "/work/src")
    .add_local_dir(str(ROOT / "temporal_crosscoders"), "/work/temporal_crosscoders")
    .add_local_dir(str(_here.parent), "/work/txc_wins")
)


@app.function(gpu="A10G", image=image, timeout=21600)
def geometry(tasks: list, model_id: str, layer: int, k_seg: int, n_docs: int,
             n_grad: int, d_sae: int, k: int, sae_lr: float, sae_steps: int,
             batch_win: int, seed: int):
    import sys
    sys.path.insert(0, "/work")
    import random
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.bench.architectures.topk_sae import TopKSAE
    from txc_wins.tasks import TASKS

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    dev = model.device
    layers_ = model.model.layers
    L = layer if layer >= 0 else len(layers_) // 2
    d = model.config.hidden_size
    T = k_seg
    cap = {}

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

    def screen(P):
        """c and r1 for a (T, d) slab, with the factor of T in c.

        Dropping that factor under-reports the constant-write ceiling by exactly T and is
        the easiest way to manufacture a false win, so it is written explicitly.
        """
        U, S, Vh = torch.linalg.svd(P, full_matrices=False)
        fro2 = float((P ** 2).sum())
        return {
            "c": float(T * P.mean(0).pow(2).sum() / fro2),
            "r1": float(S[0] ** 2 / fro2),
            "r2": float((S[0] ** 2 + S[1] ** 2) / fro2),
            "sigma2_over_sigma1_sq": float((S[1] / S[0]) ** 2),
            "singular_values": [float(v) for v in S[:6]],
            # Left singular vectors are the TEMPORAL profiles of each component. These are
            # what decide whether a rank-2 account with disjoint support is right.
            "U": [[float(v) for v in U[:, j]] for j in range(min(3, U.shape[1]))],
        }, U

    out = {"model": model_id, "layer": int(L), "k_seg": k_seg, "n_docs": n_docs,
           "n_grad": n_grad, "tasks": {}}

    for task in tasks:
        print(f"\n########## {task} ##########", flush=True)
        rng = random.Random(seed)
        make_pair = TASKS[task](k_seg)

        def draw(r_):
            p = make_pair(r_)
            return p if len(p) == 5 else (p[0], p[1], p[2], None, None)

        # ---- P_dom: difference of class means over independently drawn documents ----
        XA, XB = [], []
        for i in range(n_docs):
            sa, sb, car, _, _ = draw(rng)
            XA.append(capture(*build(car, sa)))
            XB.append(capture(*build(car, sb)))
            if (i + 1) % 100 == 0:
                print(f"   [cache] {i+1}/{n_docs}", flush=True)
        A = torch.stack(XA).to(dev)
        B = torch.stack(XB).to(dev)
        sd = torch.cat([A, B]).std() + 1e-6
        P_dom = ((A.mean(0) - B.mean(0)) / sd).float()

        # ---- Gbar: gradient of the reported margin wrt the write, at zero ----
        scale = float(torch.cat([A, B]).norm(dim=-1).mean())
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
                    add[:, a_:b_ + 1, :] = scale * Wg[t_i].unsqueeze(0)
                return ((h + add.to(h.dtype)),) + out_[1:] if isinstance(out_, tuple) \
                    else h + add.to(h.dtype)

            hk = layers_[L].register_forward_hook(edit)
            lg = model(ids).logits.float().log_softmax(-1)
            hk.remove()
            s_ = (lg[0, :-1].gather(-1, ids[0, 1:].unsqueeze(-1)).sum() if cont is None
                  else lg[0, n_doc - 1:-1].gather(-1, ids[0, n_doc:].unsqueeze(-1)).sum())
            (sgn * s_).backward()

        rng_g = random.Random(seed + 1)
        for _ in range(n_grad):
            sa, sb, car, c1, c2 = draw(rng_g)
            ta_, spa_ = build(car, sa)
            tb_, spb_ = build(car, sb)
            if c1 is None:
                grad_pass(ta_, spa_, None, 1.0); grad_pass(tb_, spb_, None, -1.0)
            else:
                grad_pass(ta_, spa_, c1, 1.0); grad_pass(ta_, spa_, c2, -1.0)
                grad_pass(tb_, spb_, c1, -1.0); grad_pass(tb_, spb_, c2, 1.0)
        Gbar = (Wg.grad / n_grad).detach()

        s_dom, U_dom = screen(P_dom)
        s_grad, U_grad = screen(Gbar)
        cos_slabs = float((P_dom.reshape(-1) / P_dom.norm()
                           * Gbar.reshape(-1) / Gbar.norm()).sum())

        # ---- one cheap SAE, purely to ask where its learned direction points ----
        X = torch.cat([A, B])
        y = torch.cat([torch.ones(A.shape[0]), torch.zeros(B.shape[0])]).to(dev)
        mu = X.mean((0, 1), keepdim=True)
        Xn = (X - mu) / sd
        flat = Xn.reshape(-1, d)
        torch.manual_seed(0)
        sae = TopKSAE(d_in=d, d_sae=d_sae, k=k).to(dev)
        opt = torch.optim.Adam(sae.parameters(), lr=sae_lr)
        sae.train()
        for _ in range(sae_steps):
            xb = flat[torch.randint(0, flat.shape[0], (batch_win * T,), device=dev)]
            loss, _, _ = sae(xb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            opt.step(); sae._normalize_decoder()
        sae.eval()
        with torch.no_grad():
            Z = sae.encode(Xn.reshape(-1, d)).reshape(-1, T, d_sae).mean(1)

            def auc_of(sc, yy):
                o = torch.argsort(sc)
                r = torch.empty_like(o, dtype=torch.float32)
                r[o] = torch.arange(len(sc), device=dev, dtype=torch.float32) + 1
                n1, n0 = float(yy.sum()), float((1 - yy).sum())
                return 0.5 if n1 == 0 or n0 == 0 else float(
                    (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

            a = torch.tensor([auc_of(Z[:, j], y) for j in range(d_sae)], device=dev)
            js = int((a - 0.5).abs().argmax())
            v = sae.W_dec.data[:, js].float()
            v = v / v.norm()
            cos_u1_grad = abs(float((v * (U_grad[:, 0:1].T @ Gbar).squeeze()
                                     / ((U_grad[:, 0:1].T @ Gbar).norm() + 1e-9)).sum()))
            cos_u1_dom = abs(float((v * (U_dom[:, 0:1].T @ P_dom).squeeze()
                                    / ((U_dom[:, 0:1].T @ P_dom).norm() + 1e-9)).sum()))

        row = {"P_dom": s_dom, "Gbar": s_grad, "cos_Pdom_Gbar": cos_slabs,
               "sae_latent": js, "sae_pooled_auc": float(0.5 + abs(float(a[js]) - 0.5)),
               "cos_vsae_u1_Gbar": cos_u1_grad, "cos_vsae_u1_Pdom": cos_u1_dom,
               "random_cos_baseline": float(1.0 / (T * d) ** 0.5)}
        out["tasks"][task] = row
        print(f"  P_dom : c {s_dom['c']:.4f}  r1 {s_dom['r1']:.4f}  "
              f"s2/s1^2 {s_dom['sigma2_over_sigma1_sq']:.3f}", flush=True)
        print(f"  Gbar  : c {s_grad['c']:.4f}  r1 {s_grad['r1']:.4f}  "
              f"s2/s1^2 {s_grad['sigma2_over_sigma1_sq']:.3f}", flush=True)
        print(f"  cos(P_dom, Gbar) {cos_slabs:+.4f}   "
              f"(random baseline {row['random_cos_baseline']:.4f})", flush=True)
        print(f"  cos(v_sae, u1(Gbar)) {cos_u1_grad:.4f}   "
              f"cos(v_sae, u1(P_dom)) {cos_u1_dom:.4f}   "
              f"sae pooled AUC {row['sae_pooled_auc']:.3f}", flush=True)
        prof = np.array(s_grad["U"][0])
        print(f"  u1(Gbar) temporal profile: "
              f"{' '.join(f'{abs(v):.2f}' for v in prof)}", flush=True)
        if len(s_grad["U"]) > 1:
            print(f"  u2(Gbar) temporal profile: "
                  f"{' '.join(f'{abs(v):.2f}' for v in s_grad['U'][1])}", flush=True)
    return out


@app.local_entrypoint()
def main(tasks: str = "recency", model: str = "Qwen/Qwen2.5-1.5B-Instruct",
         layer: int = -1, k_seg: int = 12, n_docs: int = 200, n_grad: int = 40,
         d_sae: int = 4096, k: int = 8, sae_lr: float = 3e-4, sae_steps: int = 6000,
         batch_win: int = 32, seed: int = 31415, tag: str = ""):
    import json
    r = geometry.remote([t.strip() for t in tasks.split(",")], model, layer, k_seg,
                        n_docs, n_grad, d_sae, k, sae_lr, sae_steps, batch_win, seed)
    outdir = ROOT / "results" / "txc_wins"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"geometry{tag}.json").write_text(json.dumps(r, indent=2))
    print("[saved]", outdir / f"geometry{tag}.json")
