"""Modal: the FROZEN-WRITE arm — does the schedule come off the decoder, or from us?

WHY THIS EXISTS. The m-sweep fits coefficients by least squares against the known target,
so it measures whether a dictionary's atoms SPAN trajectories — not whether a dictionary
SUPPLIES one. On that design the sprint's stated gap ("the schedule is read off a decoder
rather than supplied") stays open even if the TXC wins. This arm closes it.

THE DESIGN. Fix the target profile (the same trajectory on every episode, so a fixed
pattern can in principle match it). Then:

  1. SELECT one latent on a held-out split — never on the test pairs.
  2. FREEZE it: one latent, one global scalar, one global sign, the identical write on
     every test episode. Nothing is fitted per episode; the temporal shape is whatever the
     decoder learned.
  3. Compare against the selection floor: the same procedure with a RANDOM latent,
     repeated, giving a per-architecture permutation null.

Arms, and what one scalar buys in each:
  txc_frozen        one TXC latent -> its learned (k x d) pattern, applied as-is
  sae_frozen        one SAE latent -> one direction, necessarily CONSTANT over slots
  dom_blockconst    the supervised incumbent, but paying knobs honestly: a block-constant
                    schedule of width W costs k/W scalars

HARNESS VALIDATION (the reviewer's proposal, and the best idea in the pre-registration).
The block-constant DoM arm has a KNOWN answer from the previous sprint's law: fidelity
should equal mean_b |mu_b|, the mean absolute block-mean of the profile, with no free
parameters. At k=12 on an alternating profile that is 0 for W=2 and 1/3 for W=3; on a
run-3 profile it is 1 for W=3 and 2/3 for W=2. **If this harness misses those by more than
0.08, the instrument is broken and no dictionary number it produces is worth collecting.**
This runs first and gates everything else.

Registered predictions, before running:
  txc_frozen  > 0.20 fidelity on the structured (run-3) profile, and above its own
              permutation-null floor by > 0.15
  sae_frozen  ~ 0 on any BALANCED profile, exactly — a constant write is orthogonal to a
              zero-mean target, for any direction whatsoever

    modal run experiments/temporal_screen/dict_bench/frozen_modal.py
"""

import pathlib
import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

app = modal.App("dictbench-frozen")
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
def frozen(model_id: str, layer: int, k_seg: int, n_docs: int, d_sae: int,
           steps: int, txc_batch: int, n_sel: int, n_test: int, frac: float,
           general_frac: float, kper: int, ells: list, n_null: int):
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
    rng = random.Random(31337)

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

    # ---------------- cache + train ----------------
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
    Xn = (Xt - Xt.mean((0, 1), keepdim=True)) / (Xt.std() + 1e-6)
    task = labels[:, 0] >= 0
    fl = torch.tensor(labels[task].reshape(-1), device=dev)
    fa = Xn[task].reshape(-1, d)
    u_dom = (fa[fl == 1].mean(0) - fa[fl == 0].mean(0))
    u_dom = u_dom / u_dom.norm()
    print(f"[cache] {X.shape} base_norm={base_norm:.1f}")

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
    with torch.no_grad():
        V = sae.W_dec.data.float()          # (d, h) unit columns
        P = txc.W_dec.data.float()          # (h, T, d) unit slabs

    tgt_norm = frac * base_norm * (k_seg ** 0.5)
    rescale = lambda W: W * (tgt_norm / (W.norm() + 1e-9))

    def make_pairs(prof01, n):
        out = []
        for _ in range(n):
            foil = prof01[:]
            for _ in range(60):
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

    results = {"model": model_id, "layer": int(L), "k_seg": k_seg, "d_sae": d_sae,
               "kper": kper, "base_norm": base_norm, "by_ell": {}}

    for ell in ells:
        prof01 = [1 if (t // ell) % 2 == 0 else 0 for t in range(k_seg)]
        pi = np.array([1.0 if l else -1.0 for l in prof01])
        if abs(pi.mean()) > 1e-9:
            print(f"[ell={ell}] unbalanced, skipped"); continue
        print(f"\n===== ell={ell}  profile={prof01} =====")
        sel_pairs = make_pairs(prof01, n_sel)
        test_pairs = make_pairs(prof01, n_test)
        Y = torch.stack([float(pi[t]) * u_dom for t in range(k_seg)])

        def run(pairs, W):
            base = [margin(t, None) - margin(f, None) for t, f in pairs]
            return np.array([(margin(t, W) - margin(f, W)) - b
                             for (t, f), b in zip(pairs, base)])
        full_test = run(test_pairs, rescale(Y))
        print(f"[ref] full DoM Δ={full_test.mean():+.2f}")
        row = {"profile": prof01, "full_dom": float(full_test.mean()),
               "full_dom_sem": float(full_test.std(ddof=1) / np.sqrt(len(full_test))),
               "arms": {}}

        # ---- HARNESS VALIDATION: block-constant DoM has a known answer ----
        row["harness_check"] = {}
        for W_blk in [w for w in (2, 3, 4, 6) if k_seg % w == 0]:
            nb = k_seg // W_blk
            mu = np.array([pi[b * W_blk:(b + 1) * W_blk].mean() for b in range(nb)])
            pred = float(np.mean(np.abs(mu)))
            c = np.sign(mu)
            Wm = torch.stack([float(c[t // W_blk]) * u_dom for t in range(k_seg)])
            if Wm.norm() < 1e-8:
                obs = 0.0
            else:
                obs = float(run(test_pairs, rescale(Wm)).mean() / full_test.mean())
            row["harness_check"][W_blk] = {"predicted": pred, "observed": obs,
                                           "knobs": nb, "error": abs(obs - pred)}
            print(f"[harness] W={W_blk} knobs={nb}: predicted {pred:.3f} "
                  f"observed {obs:+.3f}  |err|={abs(obs-pred):.3f}")
        worst = max(v["error"] for v in row["harness_check"].values())
        row["harness_ok"] = bool(worst <= 0.08)
        print(f"[harness] worst |error| = {worst:.3f} -> "
              f"{'PASS' if worst <= 0.08 else 'FAIL — instrument suspect'}")

        # ---- frozen arms: select on sel split, freeze, evaluate on test ----
        def frozen_arm(atoms_fn, n_atoms, tag):
            sel_scores = []
            for j in range(n_atoms):
                W = rescale(atoms_fn(j))
                sel_scores.append(float(run(sel_pairs, W).mean()))
            sel_scores = np.array(sel_scores)
            j_best = int(np.abs(sel_scores).argmax())
            sgn = float(np.sign(sel_scores[j_best]))
            W = rescale(sgn * atoms_fn(j_best))
            ds = run(test_pairs, W)
            fid = float(ds.mean() / full_test.mean())
            print(f"[{tag}] latent {j_best} sign {sgn:+.0f}: "
                  f"Δ={ds.mean():+6.2f} fid={fid:+.3f}")
            return {"latent": j_best, "sign": sgn, "delta": float(ds.mean()),
                    "sem": float(ds.std(ddof=1) / np.sqrt(len(ds))),
                    "fidelity": fid, "sel_score": float(sel_scores[j_best]),
                    "deltas": [float(x) for x in ds]}, sel_scores

        # candidate pools kept small: score every latent on the sel split is too slow,
        # so pre-rank by projection onto the ideal write, then score the top pool.
        with torch.no_grad():
            Yf = Y.reshape(-1)
            rank_txc = (P.reshape(P.shape[0], -1) @ Yf).abs().topk(24).indices.tolist()
            rank_sae = (V.T @ u_dom).abs().topk(24).indices.tolist()
        row["arms"]["txc_frozen"], s_txc = frozen_arm(
            lambda i: P[rank_txc[i]], len(rank_txc), "txc_frozen")
        row["arms"]["sae_frozen"], s_sae = frozen_arm(
            lambda i: torch.stack([V[:, rank_sae[i]]] * k_seg), len(rank_sae),
            "sae_frozen")

        # ---- permutation-null selection floor, per architecture ----
        for tag, pool, mk in (("txc", P.shape[0], lambda j: P[j]),
                              ("sae", V.shape[1],
                               lambda j: torch.stack([V[:, j]] * k_seg))):
            fl_ = []
            for _ in range(n_null):
                j = rng.randrange(pool)
                ds = run(test_pairs, rescale(mk(j)))
                fl_.append(abs(float(ds.mean() / full_test.mean())))
            row["arms"][f"{tag}_null_floor"] = {
                "mean_abs_fidelity": float(np.mean(fl_)),
                "p90": float(np.percentile(fl_, 90)), "n": n_null}
            print(f"[null] {tag}: mean|fid|={np.mean(fl_):.3f} "
                  f"p90={np.percentile(fl_,90):.3f}")
        results["by_ell"][ell] = row
    return results


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-1.5B-Instruct", layer: int = -1, k_seg: int = 12,
         n_docs: int = 1200, d_sae: int = 4096, steps: int = 2500,
         txc_batch: int = 64, n_sel: int = 8, n_test: int = 20, frac: float = 0.35,
         general_frac: float = 0.4, kper: int = 41, ells: str = "1,3",
         n_null: int = 12):
    import json
    r = frozen.remote(model, layer, k_seg, n_docs, d_sae, steps, txc_batch,
                      n_sel, n_test, frac, general_frac, kper,
                      [int(x) for x in ells.split(",")], n_null)
    outdir = ROOT / "results" / "dict_bench"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "frozen.json").write_text(json.dumps(r, indent=2))
    print("[saved]", outdir / "frozen.json")
