"""Why does nothing steer in the two dead models? A training-free screen at every depth.

THE NEGATIVE THIS EXISTS TO EXPLAIN. No `(T, d)` write of any kind -- supervised, gradient,
crosscoder or dictionary -- moves instruction-position bias in `Qwen2.5-0.5B-Instruct` or
`SmolLM2-1.7B-Instruct`, at any of six depths, while the same task on `Qwen2.5-1.5B-Instruct`
at L14 gives the sprint's headline. A bare negative like that is worth very little: it could
mean the factor is absent, that it is present but unreachable, or that we picked six wrong
layers. This screen costs no dictionary training and separates those.

WHAT IS ALREADY KNOWN, read back out of the transfer runs' stored spectra at no compute
(`||G||_F = sigma_1 / sqrt(r1)`, exact, since `r1 = sigma_1^2 / ||G||_F^2`):

    model                     L    ||Gbar||_F      c      r1    unsteered baseline
    SmolLM2-1.7B              6      157.24    0.0887   0.613        +2.19
    SmolLM2-1.7B              9      156.92    0.0650   0.691        +2.19
    SmolLM2-1.7B             12      136.49    0.0368   0.817        +2.18
    SmolLM2-1.7B             15        7.27    0.1561   0.665        +2.19
    SmolLM2-1.7B             18        4.11    0.0826   0.729        +2.19
    SmolLM2-1.7B             21        2.55    0.1069   0.795        +2.19
    Qwen2.5-0.5B             12       62.22    0.0261   0.949        +1.50
    Qwen2.5-1.5B (works)     14      213.40    0.0365   0.813        -2.15

Three things follow and each one shapes this script.

  1. THE `c` GATE FIRES ON THE DEAD MODELS. Five of those seven cells sit below the c < 0.1
     go threshold with high `r1` -- the geometry the gate reads as "a crosscoder should win
     here". It was validated WITHIN one model and it does not transfer across models. So the
     gate is necessary, not sufficient, and this screen must not be built out of `c` again.

  2. MAGNITUDE EXPLAINS THE DEEP LAYERS AND NOT THE SHALLOW ONES. SmolLM2 falls 19x between
     L12 and L15 and stays down, so at L15/18/21 a relative write simply has no leverage on
     the readout and the null there is uninformative. But at L6/L9/L12 it is 136-157 against
     the working model's 213 -- same order -- and steering still failed. Magnitude is not the
     explanation where the explanation is needed.

  3. THE BEHAVIOUR IS PRESENT IN BOTH DEAD MODELS. Baselines are +1.50 and +2.19 against the
     working model's -2.15: same size, opposite sign. Nothing is missing to steer.

SO THE ONE QUANTITY THAT COULD EXPLAIN IT IS THE ONE NOBODY STORED. A dictionary latent is a
SINGLE write reused across documents, so every fixed-write arm is bounded by how much the
per-document optimal writes agree:

    rho = || mean_i G_i ||_F  /  mean_i || G_i ||_F        (floor 1 / sqrt(n))

`||Gbar||_F` is the numerator of that ratio and is the only part the transfer runs kept. If
`rho` sits at its floor in the dead models and well above it in the working one, the negative
is "no shared write exists" -- a real scope limit, not a search failure, and one that no
architecture fixes. If `rho` is the same everywhere, this screen does not explain the
negative and I report it unexplained.

REGISTERED PREDICTIONS, before the run:

    P1  rho(Qwen2.5-1.5B, L14) > rho(dead models at their best layer), by enough to see
        against the 1/sqrt(n) floor. This is the "no shared write" hypothesis and it is what
        I expect, because leverage and slab geometry are already ruled out at L6-L12.
    P2  FALSIFIER, and I commit to reporting it as such: if rho is within noise across all
        three models, the negative stays unexplained. A screen that cannot fail is not a
        screen, and the temptation here is to reach for whichever number happens to differ.
    P3  SmolLM2's L12->L15 cliff is a property of the model and reproduces at finer spacing.
        Its likely mechanism is measured here directly rather than assumed: `act_norm` per
        layer decomposes ||Gbar|| into "how large a RELATIVE write is in absolute terms"
        times "how much an absolute write moves the metric". If late-layer residual norms
        blow up, a write of 1 x scale is an enormous absolute perturbation and the metric
        saturates -- which would make the cliff an artefact of the relative-norm convention
        and NOT a fact about steerability. That would matter well beyond this screen.

ONE FORWARD/BACKWARD SERVES EVERY LAYER. The writes at different depths are independent
parameters and all are evaluated at W = 0, so hooking every layer at once with its own `Wg`
and calling `.backward()` once yields `d(margin)/dW_L` for every L simultaneously and
exactly. Twenty-eight layers cost what one layer costs.

    modal run experiments/temporal_screen/txc_wins/layerscreen_modal.py --n-grad 32
"""
import pathlib
import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",      # POSITIVE CONTROL -- mandatory, not optional
    "Qwen/Qwen2.5-0.5B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    # Third family/scale point. Qwen2.5-1.5B works and Qwen2.5-0.5B is dead, which
    # cannot distinguish "1.5B is special" from "0.5B is too small".
    "Qwen/Qwen2.5-3B-Instruct",
]

app = modal.App("txcwins-layerscreen")
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers", "accelerate", "numpy")
    # Baked, not downloaded at run time: concurrent unauthenticated pulls hit the HF rate
    # limit and killed a batch of runs earlier in this sprint.
    .run_commands(
        "python -c \"from huggingface_hub import snapshot_download; "
        + "; ".join(f"snapshot_download('{m}')" for m in MODELS) + "\"")
    .add_local_dir(str(ROOT / "src"), "/work/src")
    .add_local_dir(str(ROOT / "temporal_crosscoders"), "/work/temporal_crosscoders")
    .add_local_dir(str(_here.parent), "/work/txc_wins")
)


@app.function(gpu="A10G", image=image, timeout=21600)
def screen(model_id: str, task: str, k_seg: int, n_grad: int, n_base: int, seed: int,
           stride: int):
    import sys
    sys.path.insert(0, "/work")
    import random
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from txc_wins.tasks import TASKS

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    dev = model.device
    layers_ = model.model.layers
    d = model.config.hidden_size
    T = k_seg
    Ls = list(range(0, len(layers_), stride))
    print(f"\n########## {model_id}  ({len(layers_)} layers, d={d}) "
          f"screening {len(Ls)}: {Ls} ##########", flush=True)

    make_pair = TASKS[task](k_seg)

    def draw(r_):
        p = make_pair(r_)
        return p if len(p) == 5 else (p[0], p[1], p[2], None, None)

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

    # ---- pass 1: per-layer activation scale, the SAME convention the harness uses --------
    # scale_L = mean over (documents, segments) of ||segment-mean activation at L||. Getting
    # this wrong by using raw token norms instead of segment-mean norms would rescale every
    # gradient and silently break comparability with the stored transfer numbers.
    rng_s = random.Random(seed + 3)
    acc = {L: [] for L in Ls}
    for _ in range(max(8, n_grad // 2)):
        sa, sb, car, _, _ = draw(rng_s)
        for sents in (sa, sb):
            e, ts = seg_spans(*build(car, sents))
            with torch.no_grad():
                hs = model(e["input_ids"].to(dev), output_hidden_states=True).hidden_states
            for L in Ls:
                h = hs[L + 1][0].float()                      # +1: hidden_states[0] is emb
                Xt = torch.stack([h[a:b + 1].mean(0) for a, b in ts])
                acc[L].append(float(Xt.norm(dim=-1).mean()))
    scale = {L: float(np.mean(acc[L])) for L in Ls}
    print("  act_norm per layer: "
          + "  ".join(f"L{L}:{scale[L]:.1f}" for L in Ls), flush=True)

    # ---- pass 2: per-DOCUMENT gradient at every layer at once ---------------------------
    Wg = {L: torch.zeros(T, d, device=dev, dtype=torch.float32, requires_grad=True)
          for L in Ls}
    Gsum = {L: torch.zeros(T, d, device=dev) for L in Ls}
    Gnorm = {L: [] for L in Ls}

    def grad_pass(text, spans, cont, sgn):
        e, ts2 = seg_spans(text, spans)
        ids = e["input_ids"].to(dev)
        n_doc = ids.shape[1]
        if cont is not None:
            ids = torch.cat([ids, torch.tensor(
                [tok(cont, add_special_tokens=False)["input_ids"]], device=dev)], 1)

        def mk(L):
            def edit(_m, _i, out_):
                h = out_[0] if isinstance(out_, tuple) else out_
                add = torch.zeros_like(h, dtype=torch.float32)
                for t_i, (a_, b_) in enumerate(ts2):
                    add[:, a_:b_ + 1, :] = scale[L] * Wg[L][t_i].unsqueeze(0)
                h = h + add.to(h.dtype)
                return (h,) + out_[1:] if isinstance(out_, tuple) else h
            return edit

        hks = [layers_[L].register_forward_hook(mk(L)) for L in Ls]
        lg = model(ids).logits.float().log_softmax(-1)
        for hk in hks:
            hk.remove()
        s_ = (lg[0, :-1].gather(-1, ids[0, 1:].unsqueeze(-1)).sum() if cont is None
              else lg[0, n_doc - 1:-1].gather(-1, ids[0, n_doc:].unsqueeze(-1)).sum())
        (sgn * s_).backward()

    rng_g = random.Random(seed + 1)
    for i in range(n_grad):
        for L in Ls:
            Wg[L].grad = None
        sa, sb, car, c1, c2 = draw(rng_g)
        ta_, spa_ = build(car, sa)
        tb_, spb_ = build(car, sb)
        if c1 is None:
            grad_pass(ta_, spa_, None, 1.0); grad_pass(tb_, spb_, None, -1.0)
        else:
            grad_pass(ta_, spa_, c1, 1.0); grad_pass(ta_, spa_, c2, -1.0)
            grad_pass(tb_, spb_, c1, -1.0); grad_pass(tb_, spb_, c2, 1.0)
        # This document's gradient of the contrast, one per layer.
        for L in Ls:
            g = Wg[L].grad.detach()
            Gsum[L] += g
            Gnorm[L].append(float(g.norm()))
        if (i + 1) % 8 == 0:
            print(f"   [grad] {i+1}/{n_grad}", flush=True)

    # ---- unsteered baseline, so a null is never confused with an absent behaviour --------
    def logp(text, spans, cont):
        e, _ = seg_spans(text, spans)
        ids = e["input_ids"].to(dev)
        n_doc = ids.shape[1]
        if cont is not None:
            ids = torch.cat([ids, torch.tensor(
                [tok(cont, add_special_tokens=False)["input_ids"]], device=dev)], 1)
        with torch.no_grad():
            lg = model(ids).logits.float().log_softmax(-1)
        if cont is None:
            return float(lg[0, :-1].gather(-1, ids[0, 1:].unsqueeze(-1)).sum())
        return float(lg[0, n_doc - 1:-1].gather(-1, ids[0, n_doc:].unsqueeze(-1)).sum())

    base, rng_b = [], random.Random(seed + 2)
    for _ in range(n_base):
        sa, sb, car, c1, c2 = draw(rng_b)
        ta_, spa_ = build(car, sa)
        tb_, spb_ = build(car, sb)
        base.append(logp(ta_, spa_, c1) - logp(ta_, spa_, c2)
                    - (logp(tb_, spb_, c1) - logp(tb_, spb_, c2))
                    if c1 is not None else
                    logp(ta_, spa_, None) - logp(tb_, spb_, None))
    base = np.array(base)

    floor = float(1.0 / np.sqrt(n_grad))
    out = {"model": model_id, "task": task, "k_seg": k_seg, "n_grad": n_grad,
           "n_layers": len(layers_), "d": d, "stride": stride,
           "baseline_mean": float(base.mean()),
           "baseline_sem": float(base.std(ddof=1) / np.sqrt(len(base))),
           "baseline_n": int(len(base)), "rho_noise_floor": floor, "layers": {}}
    print(f"  BASELINE unsteered contrast {base.mean():+.3f} "
          f"+- {base.std(ddof=1) / np.sqrt(len(base)):.3f}  (n={len(base)})", flush=True)
    print(f"\n  {'L':>4}{'act_norm':>10}{'||Gbar||':>10}{'mean||Gi||':>12}"
          f"{'rho':>8}{'rho/floor':>11}{'c':>8}{'r1':>7}", flush=True)
    for L in Ls:
        G = Gsum[L] / n_grad
        fro = float(G.norm())
        mean_gi = float(np.mean(Gnorm[L]))
        rho = fro / (mean_gi + 1e-12)
        S = torch.linalg.svdvals(G)
        c = float(T * G.mean(0).pow(2).sum() / (fro ** 2 + 1e-12))
        r1 = float(S[0] ** 2 / (fro ** 2 + 1e-12))
        out["layers"][str(L)] = {
            "act_norm": scale[L], "Gbar_fro": fro, "mean_per_doc_fro": mean_gi,
            "rho": rho, "rho_over_floor": rho / floor, "c": c, "r1": r1,
            "singular_values": [float(v) for v in S[:6]],
            "per_doc_fro": Gnorm[L]}
        print(f"  {L:>4}{scale[L]:>10.1f}{fro:>10.2f}{mean_gi:>12.2f}"
              f"{rho:>8.3f}{rho / floor:>11.2f}{c:>8.4f}{r1:>7.3f}", flush=True)
    return out


@app.local_entrypoint()
def main(models: str = ",".join(MODELS), task: str = "recency", k_seg: int = 12,
         n_grad: int = 32, n_base: int = 60, seed: int = 31415, stride: int = 3,
         tag: str = ""):
    import json
    rows = []
    for m in [x.strip() for x in models.split(",") if x.strip()]:
        rows.append(screen.remote(m, task, k_seg, n_grad, n_base, seed, stride))
    outdir = ROOT / "results" / "txc_wins"
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"layerscreen{tag}.json"
    p.write_text(json.dumps({"models": rows}, indent=2))
    print("[saved]", p)

    # The comparison the script exists to make, printed so it cannot be reconstructed
    # wrongly later: each model's BEST layer by rho, against the shared noise floor.
    print(f"\n{'model':<40}{'best L':>8}{'rho':>8}{'x floor':>9}"
          f"{'||Gbar||':>10}{'baseline':>10}")
    for r in rows:
        bl = max(r["layers"].items(), key=lambda kv: kv[1]["rho"])
        print(f"{r['model']:<40}{bl[0]:>8}{bl[1]['rho']:>8.3f}"
              f"{bl[1]['rho_over_floor']:>9.2f}{bl[1]['Gbar_fro']:>10.2f}"
              f"{r['baseline_mean']:>+10.2f}")
