"""Modal: the dictionary-level question — what does a WINDOW-SPANNING dictionary buy?

Every steering result so far uses a single difference-of-means direction with an
externally supplied schedule (SVD says the per-position template is rank-1, σ₁ = 0.89).
That supports "a schedule beats a level" but not "a temporal dictionary beats a
per-token one", because the schedule comes from ground truth rather than from the
dictionary. This experiment closes that gap with actual (if simple) dictionaries.

Two dictionaries are FIT ON THE SAME ACTIVATION CACHE, from a corpus whose profiles
carry temporal structure (run lengths drawn from {1,2,3,6} — trajectories occur with
structure in the world):

  per-token dictionary   : PCA over pooled SEGMENT activations, [n·k, d] → atoms in R^d.
                           An atom writes ONE direction at ONE segment, so hitting a
                           k-segment trajectory costs k coefficients.
  window dictionary      : PCA over WINDOW-CONCATENATED activations, [n, k·d] → atoms
                           in R^{k×d}. One atom is a whole trajectory-shaped pattern,
                           so one coefficient writes all k segments at once.

Both are evaluated at a MATCHED KNOB BUDGET m (number of scalar coefficients the
operator sets), on held-out targets:

  (1) RECONSTRUCTION of the ideal per-segment write (cosine with the full template) —
      free, and it predicts (2).
  (2) STEERING: build the write from m atoms with least-squares coefficients and
      measure Δmargin against the multiset-matched foil, exactly as elsewhere.

Registered prediction: at small m the window dictionary wins, because a structured
trajectory is one atom for it and k atoms for the per-token dictionary; the two
converge as m → k. If the window dictionary does NOT win at small m, the honest
conclusion is that the temporal-dictionary claim fails even under favourable
conditions, and only the schedule-vs-level claim survives.

    modal run experiments/temporal_screen/trajectory_steering/dict_modal.py
"""

import pathlib
import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

app = modal.App("trajectory-dict")
image = modal.Image.debian_slim().pip_install("torch", "transformers", "accelerate", "numpy")

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


@app.function(gpu="A10G", image=image, timeout=5400)
def dictcmp(model_id: str, layer: int, k: int, n_train: int, n_eval: int,
            ms: list, fracs: list):
    import random
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[load] {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    dev = model.device
    layers_ = model.model.layers
    L = layer if layer >= 0 else len(layers_) // 2
    print(f"[cfg] L={L}, k={k}")

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
    rng = random.Random(20260725)
    unit = lambda x: x / (np.linalg.norm(x) + 1e-8)

    def build(car, sents):
        text, spans = car, []
        for j, s in enumerate(sents):
            if j:
                text += " "
            spans.append((len(text), len(text) + len(s)))
            text += s
        return text, spans

    def encode(text, cs):
        enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
        offs = enc["offset_mapping"][0].tolist()
        ts = []
        for (a, b) in cs:
            ix = [i for i, (x, y) in enumerate(offs) if y > x and y > a and x < b]
            ts.append((min(ix), max(ix)))
        return enc["input_ids"].to(dev), ts

    def seg_logprob(ids, ts):
        with torch.no_grad():
            lp = model(ids).logits[0].log_softmax(-1).float()
        return float(sum(lp[p - 1, ids[0, p]]
                         for a, b in ts for p in range(a, b + 1) if p >= 1))

    def capture(text, cs):
        ids, ts = encode(text, cs)
        h = layers_[L].register_forward_hook(cap_hook)
        with torch.no_grad():
            model(ids)
        h.remove()
        hh = cap["h"][0].float().cpu().numpy()
        return ([hh[a:b + 1].mean(0) for a, b in ts],
                [float(np.linalg.norm(hh[p])) for a, b in ts
                 for p in range(a, b + 1)])

    def margin(pair, vecs, m):
        ids, ts = pair
        steer["v"] = []
        if m > 0:
            for i, v in enumerate(vecs):
                if v is None:
                    continue
                a, b = ts[i]
                steer["v"].append((max(a - 1, 0), max(b - 1, 0), m * v))
        val = seg_logprob(ids, ts)
        steer["v"] = []
        return val

    def struct_profile():
        """Run-length structured profile: ell drawn from {1,2,3,6}, random phase."""
        ell = rng.choice([1, 2, 3, 6])
        ph = rng.randint(0, 1)
        return [1 if ((t // ell) + ph) % 2 == 0 else 0 for t in range(k)]

    # ---------------- build the activation cache (shared by both dicts) --------
    X, norms_all = [], []
    for _ in range(n_train):
        prof = struct_profile()
        idxs = [rng.randrange(10) for _ in range(k)]
        text, cs = build(rng.choice(CARRIERS),
                         [(TENSE if l else CALM)[i] for l, i in zip(prof, idxs)])
        segs, norms = capture(text, cs)
        X.append(np.stack(segs))                       # [k, d]
        norms_all += norms
    X = np.stack(X)                                    # [n, k, d]
    bn = float(np.mean(norms_all))
    n, _, d = X.shape
    print(f"[cache] X={X.shape} base_norm={bn:.1f}")

    # ---------------- fit the two dictionaries ----------------
    seg_pool = X.reshape(-1, d)
    seg_pool = seg_pool - seg_pool.mean(0, keepdims=True)
    Us, Ss, Vt_tok = np.linalg.svd(seg_pool, full_matrices=False)
    tok_atoms = Vt_tok                                  # [d, d] rows = atoms in R^d

    win = X.reshape(n, k * d)
    win = win - win.mean(0, keepdims=True)
    Uw, Sw, Vt_win = np.linalg.svd(win, full_matrices=False)
    win_atoms = Vt_win.reshape(-1, k, d)                # rows = atoms in R^{k×d}
    print(f"[dict] token atoms {tok_atoms.shape}, window atoms {win_atoms.shape}")

    # ---------------- targets: held-out structured trajectories ----------------
    results = {"model": model_id, "layer": int(L), "k": k, "base_norm": bn,
               "reconstruction": {}, "steering": {}}

    targets = []
    for _ in range(n_eval):
        prof = struct_profile()
        targets.append(prof)

    # Ideal write for a profile: T[t] = pi_t * u_dc, the full per-segment template.
    # u_dc is the leading token atom (the intensity axis), oriented so TENSE is
    # positive by probing with all-tense vs all-calm documents.
    u_dc = unit(tok_atoms[0])
    probe_hi, probe_lo = [], []
    for _ in range(24):
        car = rng.choice(CARRIERS)
        segs_h, _ = capture(*build(car, [TENSE[rng.randrange(10)] for _ in range(k)]))
        segs_l, _ = capture(*build(car, [CALM[rng.randrange(10)] for _ in range(k)]))
        probe_hi += segs_h
        probe_lo += segs_l
    if np.mean(probe_hi, 0) @ u_dc < np.mean(probe_lo, 0) @ u_dc:
        u_dc = -u_dc
    print(f"[dir] |u_dc·(hi-lo)| = {abs(unit(np.mean(probe_hi,0)-np.mean(probe_lo,0)) @ u_dc):.3f}")

    def ideal(prof):
        return np.stack([(1.0 if l else -1.0) * u_dc for l in prof])   # [k, d]

    # ---------------- (1) reconstruction at matched knob budget ----------------
    for m in ms:
        cos_tok, cos_win = [], []
        for prof in targets:
            T = ideal(prof)                                            # [k, d]
            # window dictionary: m atoms, each ONE coefficient for the whole traj
            Aw = win_atoms[:m].reshape(m, -1)                          # [m, k*d]
            cw, *_ = np.linalg.lstsq(Aw.T, T.reshape(-1), rcond=None)
            rec_w = (cw @ Aw).reshape(k, d)
            cos_win.append(float(unit(rec_w.ravel()) @ unit(T.ravel())))
            # per-token dictionary: m coefficients TOTAL, spread over k segments.
            # with m < k the operator can only touch floor(m) segments (1 atom each);
            # with m >= k it gets m/k atoms per segment.
            per_seg = max(m // k, 0)
            rec_t = np.zeros_like(T)
            if per_seg >= 1:
                At = tok_atoms[:per_seg]                               # [p, d]
                for t in range(k):
                    ct, *_ = np.linalg.lstsq(At.T, T[t], rcond=None)
                    rec_t[t] = ct @ At
            else:
                At = tok_atoms[:1]
                for t in range(min(m, k)):
                    ct, *_ = np.linalg.lstsq(At.T, T[t], rcond=None)
                    rec_t[t] = ct @ At
            cos_tok.append(float(unit(rec_t.ravel()) @ unit(T.ravel())))
        results["reconstruction"][m] = {
            "window_cos": float(np.mean(cos_win)),
            "token_cos": float(np.mean(cos_tok))}
        print(f"[recon m={m:>2}] window={np.mean(cos_win):.3f}  "
              f"per-token={np.mean(cos_tok):.3f}")

    # ---------------- (2) steering at matched knob budget ----------------
    pairs, profs = [], []
    for prof in targets:
        foil = prof[:]
        for _ in range(40):
            rng.shuffle(foil)
            if foil != prof:
                break
        idxs = [rng.randrange(10) for _ in range(k)]
        car = rng.choice(CARRIERS)
        tT, cT = build(car, [(TENSE if l else CALM)[i] for l, i in zip(prof, idxs)])
        tF, cF = build(car, [(TENSE if l else CALM)[i] for l, i in zip(foil, idxs)])
        pairs.append((encode(tT, cT), encode(tF, cF)))
        profs.append(prof)
    base = [margin(t, [], 0) - margin(f, [], 0) for t, f in pairs]

    for m in ms:
        for which in ("window", "token"):
            per = {}
            for fr in fracs:
                mag = fr * bn
                ds = []
                for j, ((t, f), b) in enumerate(zip(pairs, base)):
                    T = ideal(profs[j])
                    if which == "window":
                        Aw = win_atoms[:m].reshape(m, -1)
                        c, *_ = np.linalg.lstsq(Aw.T, T.reshape(-1), rcond=None)
                        R = (c @ Aw).reshape(k, d)
                    else:
                        per_seg = max(m // k, 0)
                        R = np.zeros_like(T)
                        if per_seg >= 1:
                            At = tok_atoms[:per_seg]
                            for tt in range(k):
                                c, *_ = np.linalg.lstsq(At.T, T[tt], rcond=None)
                                R[tt] = c @ At
                        else:
                            At = tok_atoms[:1]
                            for tt in range(min(m, k)):
                                c, *_ = np.linalg.lstsq(At.T, T[tt], rcond=None)
                                R[tt] = c @ At
                    sc = np.linalg.norm(T) / (np.linalg.norm(R) + 1e-9)  # match energy
                    vecs = [torch.tensor(sc * R[i], device=dev, dtype=torch.float32)
                            for i in range(k)]
                    ds.append((margin(t, vecs, mag) - margin(f, vecs, mag)) - b)
                ds = np.array(ds)
                per[round(fr, 3)] = {"mean": float(ds.mean()),
                                     "sem": float(ds.std(ddof=1) / np.sqrt(len(ds)))}
            best = max(per, key=lambda x: per[x]["mean"])
            results["steering"].setdefault(m, {})[which] = {"peak_frac": best,
                                                            **per[best]}
            print(f"[steer m={m:>2} {which:6}] Δ={per[best]['mean']:+7.2f}"
                  f"±{per[best]['sem']:.2f}")
    return results


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-1.5B-Instruct", layer: int = -1, k: int = 8,
         n_train: int = 120, n_eval: int = 24, ms: str = "1,2,4,8,16",
         fracs: str = "0.35,0.5"):
    import json
    res = dictcmp.remote(model, layer, k, n_train, n_eval,
                         [int(x) for x in ms.split(",")],
                         [float(x) for x in fracs.split(",")])
    outdir = ROOT / "results" / "temporal_screen"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "dictcmp.json").write_text(json.dumps(res, indent=2))
    print("[saved]", outdir / "dictcmp.json")
