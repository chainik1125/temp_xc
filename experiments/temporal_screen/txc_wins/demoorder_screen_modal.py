"""Measure c and r1 for the two-moment demonstration-order task on REAL activations.

The two-moment result is currently established on synthetic `dc`/`ds` vectors with random
directions. Whether it survives contact with a language model is the open question, and it
is the only one that decides whether the design is worth GPU time on the full arm set.

TWO SLABS, because they are different objects and the sprint has been bitten three times by
conflating them:

  P_dom : per-position difference of means, mean(A) - mean(B). What a supervised reference
          write uses. Nearly ORTHOGONAL to the gradient on recency (cos = 0.044).
  Gbar  : the mean gradient of the ordering margin with respect to the WRITE itself. This is
          the object the screen should run on -- a constant write's first-order effect is
          alpha*T*<v, mean_t Gbar[t]>, so c(Gbar) is what decides whether a constant write
          has a first-order grip, and c(P_dom) does not.

Gbar is obtained exactly rather than by finite differences: a zero (T, d) tensor `w` is
added to the residual at the segment spans with requires_grad=True, so `w.grad` after
backward IS d(margin)/d(write) in the basis the steering arms use.

REGISTERED BEFORE RUNNING (theory agent):
  c(Gbar)  < 0.05     -- the two-moment construction zeroes the constant component of both
                         the content attribute (zeroth moment) and the carried state (first
                         moment). If it comes back above ~0.10 the construction does not
                         survive contact with a language model, which is also a result.
  r1(Gbar) < 0.85     -- content and carried state are two attributes with non-proportional
                         schedules, so rank >= 2 and the leading direction should not carry
                         everything.
  sigma_2^2/sigma_1^2 >= 0.15 -- the discriminating observable; r1 alone is non-monotone in
                         the state/content ratio and does not establish rank-2 structure.
"""
import json
import pathlib

import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

app = modal.App("txcwins-demoorder-screen")
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers", "accelerate", "numpy")
    .add_local_file(str(_here.parent / "designs_demoorder.py"),
                    "/work/designs_demoorder.py")
)


@app.function(gpu="A10G", image=image, timeout=5400)
def screen(model_id: str, layer: int, k_seg: int, n_pairs: int, seed: int,
           mode: str = "ordering"):
    import sys
    sys.path.insert(0, "/work")
    import random

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from designs_demoorder import (make_demo_order, make_demo_order_probe, moments,
                                   PATTERN_A, PATTERN_B)

    print(f"[patterns] A moments {moments(PATTERN_A)}  B moments {moments(PATTERN_B)}",
          flush=True)
    assert moments(PATTERN_A) == moments(PATTERN_B)

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, device_map="cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    dev = model.device
    layers_ = model.model.layers
    L = layer if layer >= 0 else len(layers_) // 2
    d = model.config.hidden_size
    T = k_seg
    probe = mode == "probe"
    make_pair = (make_demo_order_probe if probe else make_demo_order)(k_seg)
    print(f"[mode] {mode}", flush=True)
    rng = random.Random(seed)

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

    cap = {}

    def run_doc(text, spans, want_grad, conts=None):
        """(per-segment activations, d score / d write) at layer L.

        score = logP(text) in ordering mode, or logP(c1|text) - logP(c2|text) in probe
        mode. The probe score is a difference of differences once taken across classes, so
        any write pushing both classes the same way cancels exactly.
        """
        e, ts = seg_spans(text, spans)
        ids = e["input_ids"].to(dev)
        w = torch.zeros(T, d, device=dev, requires_grad=want_grad)

        def edit(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h.clone()
            for t_i, (a, b) in enumerate(ts):
                h[:, a:b + 1, :] = h[:, a:b + 1, :] + w[t_i].unsqueeze(0)
            cap["h"] = h
            return (h,) + out[1:] if isinstance(out, tuple) else h

        hk = layers_[L].register_forward_hook(edit)
        try:
            if want_grad:
                if conts is None:
                    lg = model(ids).logits.float().log_softmax(-1)
                    tgt = ids[0, 1:]
                    score = lg[0, :-1].gather(-1, tgt.unsqueeze(-1)).sum()
                    cap["score"] = float(score.detach())
                else:
                    parts = []
                    for cont in conts:
                        cid = tok(cont, return_tensors="pt",
                                  add_special_tokens=False)["input_ids"].to(dev)
                        full = torch.cat([ids, cid], dim=1)
                        lg = model(full).logits.float().log_softmax(-1)
                        n = cid.shape[1]
                        tgt = full[0, -n:]
                        parts.append(
                            lg[0, -n - 1:-1].gather(-1, tgt.unsqueeze(-1)).sum())
                    score = parts[0] - parts[1]
                cap["score"] = float(score.detach())
                score.backward()
                g = w.grad.detach().float().cpu().clone()
            else:
                g = None
            with torch.no_grad():
                hh = cap["h"][0].detach().float()
                acts = torch.stack([hh[a:b + 1].mean(0) for a, b in ts]).cpu()
        finally:
            hk.remove()
        return acts, g

    A_acts, B_acts, G, base_margin = [], [], [], []
    for i in range(n_pairs):
        got = make_pair(rng)
        sa, sb, car = got[0], got[1], got[2]
        conts = (got[3], got[4]) if len(got) == 5 else None
        assert len(sa) == len(sb) == k_seg
        assert sorted(sa) == sorted(sb), "multiset broken"
        ta, spa = build(car, sa)
        tb, spb = build(car, sb)
        aa, ga = run_doc(ta, spa, True, conts)
        s_a = cap["score"]
        ab, gb = run_doc(tb, spb, True, conts)
        s_b = cap["score"]
        base_margin.append(s_a - s_b)
        A_acts.append(aa); B_acts.append(ab)
        G.append(ga - gb)                      # d(margin)/d(write), margin = lpA - lpB
        if (i + 1) % 25 == 0:
            print(f"   [{i+1}/{n_pairs}]", flush=True)

    A = torch.stack(A_acts).numpy()
    B = torch.stack(B_acts).numpy()
    Gs = torch.stack(G).numpy()

    def screen_slab(P):
        fro2 = float((P ** 2).sum())
        U, s, _ = np.linalg.svd(P, full_matrices=False)
        s2 = s ** 2
        # |u_j| per position, normalised. If the slab were content (spikes at the
        # manipulated positions) plus carried state (mass on the spans between), the two
        # leading vectors would have NEAR-DISJOINT support. Coincident support means the
        # second direction is context-dependence at the SAME positions instead.
        prof = [list(np.abs(U[:, j]) / (np.abs(U[:, j]).max() + 1e-12))
                for j in range(min(3, U.shape[1]))]
        return {
            "u_profiles": [[float(x) for x in pr] for pr in prof],
            "u1_u2_profile_cos": float(
                np.dot(prof[0], prof[1])
                / (np.linalg.norm(prof[0]) * np.linalg.norm(prof[1]) + 1e-12))
            if len(prof) > 1 else None,
            "c": float(T * (P.mean(0) ** 2).sum() / fro2),
            "r1": float(s2[0] / fro2),
            "r2": float(s2[:2].sum() / fro2),
            "sigma2_over_sigma1": float(s2[1] / s2[0]),
            "sigma": [float(x) for x in s[:6]],
            "fro": float(np.sqrt(fro2)),
        }

    P_dom = A.mean(0) - B.mean(0)
    Gbar = Gs.mean(0)
    out = {
        "model": model_id, "layer": int(L), "k_seg": k_seg, "n_pairs": n_pairs,
        "seed": seed, "mode": mode,
        "pattern_a": list(PATTERN_A), "pattern_b": list(PATTERN_B),
        "moments_a": list(moments(PATTERN_A)), "moments_b": list(moments(PATTERN_B)),
        "dom": screen_slab(P_dom),
        "grad": screen_slab(Gbar),
        "cos_dom_grad": float(
            (P_dom * Gbar).sum()
            / (np.linalg.norm(P_dom) * np.linalg.norm(Gbar) + 1e-12)),
        # How much of the per-document gradient survives averaging: the shared-write
        # constraint. A fixed write is bounded by the MEAN, so this ratio caps every arm.
        "shared_write_retention": float(
            np.linalg.norm(Gbar)
            / np.mean([np.linalg.norm(g) for g in Gs])),
    }

    prof = np.linalg.norm(Gbar, axis=1)
    out["grad_profile"] = [float(x) for x in prof / (prof.max() + 1e-12)]
    early, late = float(prof[:k_seg // 2].mean()), float(prof[k_seg // 2:].mean())
    out["recency_ratio"] = late / (early + 1e-12)

    print("\n===== screen =====", flush=True)
    for nm in ("dom", "grad"):
        v = out[nm]
        print(f"  {nm:<5} c={v['c']:.4f}  r1={v['r1']:.4f}  r2={v['r2']:.4f}  "
              f"s2/s1={v['sigma2_over_sigma1']:.4f}", flush=True)
    print(f"  cos(P_dom, Gbar) = {out['cos_dom_grad']:+.4f}", flush=True)
    print(f"  shared-write retention = {out['shared_write_retention']:.4f}", flush=True)
    print(f"  grad norm profile      = "
          + " ".join(f"{x:.2f}" for x in out["grad_profile"]), flush=True)
    print(f"  recency ratio (late/early) = {out['recency_ratio']:.3f}", flush=True)
    bm = np.array(base_margin)
    out["base_margin_mean"] = float(bm.mean())
    out["base_margin_sem"] = float(bm.std(ddof=1) / np.sqrt(len(bm)))
    out["base_margin_abs_mean"] = float(np.abs(bm).mean())
    print(f"\n  UNSTEERED score(A)-score(B) = {bm.mean():+.4f} +- "
          f"{out['base_margin_sem']:.4f}   |mean abs| = {np.abs(bm).mean():.4f}"
          f"   z = {bm.mean()/(out['base_margin_sem']+1e-12):+.2f}", flush=True)
    for nm in ("dom", "grad"):
        pr = out[nm]["u_profiles"]
        print(f"  {nm} u1: " + " ".join(f"{x:.2f}" for x in pr[0]), flush=True)
        print(f"  {nm} u2: " + " ".join(f"{x:.2f}" for x in pr[1]), flush=True)
        print(f"  {nm} cos(|u1|,|u2|) profile = {out[nm]['u1_u2_profile_cos']:.3f}",
              flush=True)

    g = out["grad"]
    ok_c = g["c"] < 0.05
    ok_r1 = g["r1"] < 0.85
    ok_s2 = g["sigma2_over_sigma1"] >= 0.15
    print(f"\n  registered c(grad) < 0.05          : {ok_c}  ({g['c']:.4f})", flush=True)
    print(f"  registered r1(grad) < 0.85         : {ok_r1}  ({g['r1']:.4f})", flush=True)
    print(f"  registered s2^2/s1^2 >= 0.15       : {ok_s2}  "
          f"({g['sigma2_over_sigma1']:.4f})", flush=True)
    print("  -> GEOMETRY HOLDS, worth the full arm set" if (ok_c and ok_s2)
          else "  -> geometry does NOT hold; report as a negative", flush=True)
    return out


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-1.5B-Instruct", layer: int = 14, k_seg: int = 12,
         n_pairs: int = 200, seed: int = 31415, tag: str = "", mode: str = "ordering"):
    r = screen.remote(model, layer, k_seg, n_pairs, seed, mode)
    outdir = ROOT / "results" / "txc_wins"
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"demoorder_screen{tag}.json"
    p.write_text(json.dumps(r, indent=2))
    print("[saved]", p)
