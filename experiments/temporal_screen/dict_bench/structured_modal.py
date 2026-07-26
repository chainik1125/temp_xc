"""Modal: give the crosscoder something temporal to learn, and ask the right question of it.

TWO DESIGN ERRORS THIS FIXES, both mine, both found at 17:22.

(1) THE TRAINING CORPUS HAD NO TEMPORAL STRUCTURE. Every run so far drew the tense/calm
    label i.i.d. per segment. A temporal crosscoder's premise is that there are patterns
    ACROSS a window to capture; i.i.d. labels contain none, by construction. So the whole
    benchmark trained a temporal architecture on temporally-structureless data and then
    reported that it underperformed. Here the corpus is drawn from a RUN-LENGTH FAMILY
    (ell in {1,2,3,6}, random phase), so windows carry genuine temporal pattern.

(2) THE INTERP MEASUREMENT WAS THE WRONG QUESTION FOR A WINDOW CODE. A TXC latent is one
    scalar per 12-segment window, so it CANNOT encode which individual segment is tense —
    with i.i.d. labels that would need 12 bits. Its ~chance segment-level AUC (0.541) was
    a structural certainty, not a training failure. The architectures should be asked what
    each can in principle represent:

        SEGMENT-LEVEL factor   "is THIS segment tense?"          SAE can, TXC structurally cannot
        WINDOW-LEVEL pattern   "is this window fast- or slow-    TXC can, SAE only via
                                alternating?" (ell in {1,2} vs {3,6})   combination across segments

    Both are measured for both architectures. The interesting claim is not that one wins
    everywhere but that each owns the level its code lives at — and whether the TXC's
    window-level advantage is large enough to matter.

2x2 DESIGN: {i.i.d. data, structured data} x {SAE, TXC}, everything else held fixed —
same cache size, same token-activations per step, same d_sae, same eval. The i.i.d. arm
reproduces the earlier result as a control, so the data effect is isolated rather than
asserted.

Registered before running:
  - on structured data the TXC's realised L0, FVU and alive fraction all improve relative
    to i.i.d.; if they do not, the starvation is a training pathology independent of data
    and the earlier negative stands;
  - the TXC beats the SAE on WINDOW-level AUC and loses on SEGMENT-level AUC, on both
    corpora, because that split is structural;
  - segment-level steering fidelity is roughly unchanged, since the target profile is a
    per-segment object either way.

    modal run experiments/temporal_screen/dict_bench/structured_modal.py
"""

import pathlib
import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

app = modal.App("dictbench-structured")
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


@app.function(gpu="L4", image=image, timeout=14400)
def structured(model_id: str, layer: int, k_seg: int, n_docs: int, d_sae: int,
               txc_batch: int, steps: int, sae_k: int, txc_ks: list,
               general_frac: float, n_test: int, frac: float,
               init_norm: bool = False):
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
    rng = random.Random(161803)
    ELLS = [1, 2, 3, 6]

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

    def make_cache(structured_flag, n):
        """Returns X (n,k,d), seg labels (n,k), window ell (n,), mean act norm."""
        X, seg, ells, norms = [], [], [], []
        n_gen = int(n * general_frac)
        for i in range(n):
            if i < n_gen:
                sents = [GENERAL[rng.randrange(len(GENERAL))] for _ in range(k_seg)]
                lab = [-1] * k_seg; e_ = -1
            elif structured_flag:
                e_ = ELLS[rng.randrange(len(ELLS))]
                ph = rng.randrange(2 * e_)
                lab = [1 if ((t + ph) // e_) % 2 == 0 else 0 for t in range(k_seg)]
                sents = [(TENSE if l else CALM)[rng.randrange(10)] for l in lab]
            else:
                lab = [rng.randint(0, 1) for _ in range(k_seg)]; e_ = -1
                sents = [(TENSE if l else CALM)[rng.randrange(10)] for l in lab]
            s_, n_ = capture(*build(rng.choice(CARRIERS), sents))
            X.append(s_); seg.append(lab); ells.append(e_); norms += n_
        return (np.stack(X), np.array(seg), np.array(ells), float(np.mean(norms)))

    def auc_of(scores, y, keep):
        s_, y_ = scores[keep], y[keep]
        if y_.sum() == 0 or (1 - y_).sum() == 0:
            return 0.5
        o = torch.argsort(s_)
        r = torch.empty_like(o, dtype=torch.float32)
        r[o] = torch.arange(len(s_), device=s_.device, dtype=torch.float32) + 1
        n1, n0 = float(y_.sum()), float((1 - y_).sum())
        return float((r[y_ == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    def best_auc(Z, y, keep):
        a = torch.tensor([auc_of(Z[:, j], y, keep) for j in range(Z.shape[1])],
                         device=dev)
        dv = (a - 0.5).abs()
        j = int(dv.argmax())
        return float(a[j]), float(dv[j]) + 0.5, int((dv >= 0.95 * float(dv[j])).sum())

    results = {"model": model_id, "layer": int(L), "k_seg": k_seg, "d_sae": d_sae,
               "steps": steps, "sae_k": sae_k, "corpora": {}}

    for structured_flag in (False, True):
        tag = "structured" if structured_flag else "iid"
        print(f"\n########## CORPUS: {tag} ##########")
        X, seg, ells, base_norm = make_cache(structured_flag, n_docs)
        Xt = torch.tensor(X, dtype=torch.float32, device=dev)
        n_hold = max(int(0.15 * Xt.shape[0]), 48)
        Xtr, Xho = Xt[:-n_hold], Xt[-n_hold:]
        mu, sd = Xtr.mean((0, 1), keepdim=True), Xtr.std() + 1e-6
        Xn, Xn_ho = (Xtr - mu) / sd, (Xho - mu) / sd
        flat_ho = Xn_ho.reshape(-1, d)
        denom = float(flat_ho.var(0).sum())
        seg_ho = torch.tensor(seg[-n_hold:], device=dev).reshape(-1)
        ell_ho = torch.tensor(ells[-n_hold:], device=dev)
        keep_seg = seg_ho >= 0
        # window-level target: fast (ell in {1,2}) vs slow (ell in {3,6})
        keep_win = ell_ho > 0
        y_win = ((ell_ho == 3) | (ell_ho == 6)).float()
        print(f"[cache] {X.shape}  labelled segments {int(keep_seg.sum())}  "
              f"labelled windows {int(keep_win.sum())}")

        def gen_w(bs): return Xn[torch.randint(0, Xn.shape[0], (bs,), device=dev)]

        def gen_f(bs):
            i = torch.randint(0, Xn.shape[0], (bs,), device=dev)
            j = torch.randint(0, k_seg, (bs,), device=dev)
            return Xn[i, j]

        def train(m, gen, bs, tag2):
            opt = torch.optim.Adam(m.parameters(), lr=1e-3)
            for s in range(steps):
                loss, _, _ = m(gen(bs))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step(); m._normalize_decoder()
                if s % max(steps // 2, 1) == 0 or s == steps - 1:
                    print(f"     [{tag2}] {s}/{steps} loss={loss.item():.3f}")

        corp = {"base_norm": base_norm, "arms": []}

        torch.manual_seed(0)
        sae = TopKSAE(d_in=d, d_sae=d_sae, k=sae_k).to(dev)
        train(sae, gen_f, txc_batch * k_seg, f"sae-{tag}")
        with torch.no_grad():
            z = sae.encode(flat_ho); xh = sae.decode(z)
            fvu = float(((xh - flat_ho) ** 2).sum(-1).mean() / denom)
            l0 = float((z > 0).float().sum(-1).mean())
            alive = float(((z > 0).float().mean(0) >= 0.001).float().mean())
            a_seg, a_seg_o, _ = best_auc(z, seg_ho.float(), keep_seg)
            zw = z.reshape(-1, k_seg, z.shape[-1]).mean(1)      # pool to window
            a_win, a_win_o, _ = best_auc(zw, y_win, keep_win)
        corp["arms"].append({"arch": "sae", "nominal_k": sae_k,
                             "coeff_per_segment": l0, "fvu": fvu, "alive": alive,
                             "seg_auc": a_seg, "seg_auc_oriented": a_seg_o,
                             "win_auc": a_win, "win_auc_oriented": a_win_o})
        print(f"  [sae k={sae_k}] coeff/seg {l0:.1f} FVU {fvu:.4f} alive {alive:.3f} "
              f"| segment-AUC {a_seg_o:.3f}  window-AUC {a_win_o:.3f}")

        for kp in txc_ks:
            if kp * k_seg > d_sae:
                continue
            torch.manual_seed(0)
            txc = TemporalCrosscoder(d_in=d, d_sae=d_sae, T=k_seg, k=kp).to(dev)
            if init_norm:
                # See initnorm_modal.py: the repo's crosscoder starts its decoder at
                # norm sqrt(T*d_in/d_sae) and rescales only after the first step.
                with torch.no_grad():
                    txc._normalize_decoder()
            train(txc, gen_w, txc_batch, f"txc-k{kp}-{tag}")
            with torch.no_grad():
                pre = torch.einsum("btd,tds->bs", Xn_ho, txc.W_enc) + txc.b_enc
                posfrac = float((pre > 0).float().mean())
                zt = txc.encode(Xn_ho); xht = txc.decode(zt)
                fvu = float(((xht - Xn_ho) ** 2).sum(-1).mean() / denom)
                l0 = float((zt > 0).float().sum(-1).mean())
                alive = float(((zt > 0).float().mean(0) >= 0.001).float().mean())
                a_seg, a_seg_o, _ = best_auc(zt.repeat_interleave(k_seg, 0),
                                             seg_ho.float(), keep_seg)
                a_win, a_win_o, _ = best_auc(zt, y_win, keep_win)
            corp["arms"].append({"arch": "txc", "nominal_k_per_pos": kp,
                                 "coeff_per_segment": l0 / k_seg,
                                 "realised_l0_window": l0, "fvu": fvu, "alive": alive,
                                 "pos_preact_frac": posfrac,
                                 "seg_auc": a_seg, "seg_auc_oriented": a_seg_o,
                                 "win_auc": a_win, "win_auc_oriented": a_win_o})
            print(f"  [txc kper={kp}] coeff/seg {l0/k_seg:.2f} FVU {fvu:.4f} "
                  f"alive {alive:.3f} pos-preact {posfrac:.3f} "
                  f"| segment-AUC {a_seg_o:.3f}  window-AUC {a_win_o:.3f}")
        results["corpora"][tag] = corp

    print("\n===== THE STRUCTURAL SPLIT (oriented AUC, higher = better) =====")
    for tag in results["corpora"]:
        for a in results["corpora"][tag]["arms"]:
            nm = a["arch"] + (f"-k{a.get('nominal_k_per_pos', a.get('nominal_k'))}")
            print(f"  {tag:11} {nm:10} segment {a['seg_auc_oriented']:.3f}   "
                  f"window {a['win_auc_oriented']:.3f}   FVU {a['fvu']:.4f}")
    return results


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-1.5B-Instruct", layer: int = -1, k_seg: int = 12,
         n_docs: int = 1200, d_sae: int = 4096, txc_batch: int = 64,
         steps: int = 2500, sae_k: int = 100, txc_ks: str = "4,41",
         general_frac: float = 0.4, n_test: int = 16, frac: float = 0.35,
         init_norm: bool = False, tag: str = ""):
    import json
    r = structured.remote(model, layer, k_seg, n_docs, d_sae, txc_batch, steps,
                          sae_k, [int(x) for x in txc_ks.split(",")],
                          general_frac, n_test, frac, init_norm)
    r["init_norm"] = init_norm
    outdir = ROOT / "results" / "dict_bench"
    outdir.mkdir(parents=True, exist_ok=True)
    name = f"structured{tag}.json"
    (outdir / name).write_text(json.dumps(r, indent=2))
    print("[saved]", outdir / name)
