"""Is there an activation function for the crosscoder that does not cap realised capacity?

Both architectures in this repo apply ReLU *after* TopK:

    topk_vals, topk_idx = pre.topk(k, dim=-1)
    z.scatter_(1, topk_idx, F.relu(topk_vals))

so realised L0 is min(k, #{pre > 0}). For the SAE that composition is invisible: it has
~2000 positive pre-activations at k=1, so the ReLU term never binds and nominal k is an
honest budget. For the crosscoder #{pre > 0} sits at 20-160, so above that k the ReLU
silently discards the surplus and extra k buys nothing. The cap is a property of the
composition, not of TopK.

ARMS (crosscoder only; the SAE is unaffected and is carried as a reference):

  topk_relu    the repo's current composition. Realised L0 = min(k, #{pre>0}).
  topk         TopK with no ReLU after it -- the Gao et al. formulation. Realised L0 = k
               exactly, by construction, so capacity CANNOT collapse. The cost is that
               coefficients may be negative, which weakens the usual "a feature fires or
               it does not" reading; the negative fraction is therefore reported.
  batchtopk    BatchTopK (Bussmann et al. 2024): take the k*B largest pre-activations
               across the whole batch instead of k per sample, then ReLU. Lets the model
               allocate coefficients unevenly across windows and is known to reduce dead
               latents. Realised L0 averages k but varies per window.
  topk_relu_auxk  the repo's composition plus the standard AuxK dead-latent revival loss,
               which is the fix that treats the cause rather than the symptom.

Run at lr=1e-3 -- the setting where capacity demonstrably collapses -- so the question is
whether the activation function rescues what that learning rate destroys. topk_relu at
lr=3e-4 is carried as the known-good reference.

REGISTERED PREDICTIONS (written before the run):
  A1  topk realises exactly k at every nominal k, by construction. If it does not, the
      harness is wrong.
  A2  topk and batchtopk both beat topk_relu on FVU at kper=41, where topk_relu realises
      only ~1.5 coefficients per segment.
  A3  topk's advantage comes with a substantial negative-coefficient fraction. If that
      fraction is near zero, then #{pre>0} was never the real constraint and I have
      misdiagnosed the mechanism.
  A4  AuxK at lr=1e-3 partially rescues capacity, where at lr=3e-4 it was null.
"""
import pathlib
import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

app = modal.App("dictbench-actfn")
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
def actfn(model_id: str, layer: int, k_seg: int, n_docs: int, d_sae: int,
          batch: int, steps: int, kpers: list, lr: float, ref_lr: float,
          general_frac: float, aux_alpha: float, k_aux: int):
    import sys
    sys.path.insert(0, "/work")
    import random
    import numpy as np
    import torch
    import torch.nn.functional as F
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
    cap = {}

    def cap_hook(_m, _i, out):
        cap["h"] = (out[0] if isinstance(out, tuple) else out).detach()

    rng = random.Random(2718281)

    def build(car, sents):
        text, spans = car, []
        for j, s in enumerate(sents):
            if j:
                text += " "
            spans.append((len(text), len(text) + len(s)))
            text += s
        return text, spans

    def capture(text, cs):
        e = tok(text, return_tensors="pt", return_offsets_mapping=True)
        offs = e["offset_mapping"][0].tolist()
        ts = []
        for (a, b) in cs:
            idx = [i for i, (s0, s1) in enumerate(offs) if s0 >= a and s1 <= b and s1 > s0]
            ts.append((idx[0], idx[-1]) if idx else (0, 0))
        h = layers_[L].register_forward_hook(cap_hook)
        with torch.no_grad():
            model(e["input_ids"].to(dev))
        h.remove()
        hh = cap["h"][0].float().cpu().numpy()
        return np.stack([hh[a:b + 1].mean(0) for a, b in ts])

    X = []
    for i in range(n_docs):
        if rng.random() < general_frac:
            sents = [GENERAL[rng.randrange(len(GENERAL))] for _ in range(k_seg)]
        else:
            sents = [(TENSE if rng.randint(0, 1) else CALM)[rng.randrange(10)]
                     for _ in range(k_seg)]
        X.append(capture(*build(rng.choice(CARRIERS), sents)))
        if (i + 1) % 300 == 0:
            print(f"   [cache] {i+1}/{n_docs}", flush=True)
    Xt = torch.tensor(np.stack(X), dtype=torch.float32, device=dev)
    n_hold = max(int(0.15 * Xt.shape[0]), 32)
    Xtr, Xho = Xt[:-n_hold], Xt[-n_hold:]
    mu, sd = Xtr.mean((0, 1), keepdim=True), Xtr.std() + 1e-6
    Xn, Xn_ho = (Xtr - mu) / sd, (Xho - mu) / sd
    denom = float(Xn_ho.reshape(-1, d).var(0).sum())
    flat_tr, flat_ho = Xn.reshape(-1, d), Xn_ho.reshape(-1, d)
    print(f"[cache] {tuple(Xn.shape)}", flush=True)

    def apply_act(pre, k, arm):
        """pre: (B, d_sae) -> z with the arm's sparsity rule."""
        if arm == "batchtopk":
            B = pre.shape[0]
            flat = pre.reshape(-1)
            v, i = flat.topk(min(k * B, flat.numel()))
            z = torch.zeros_like(flat)
            z.scatter_(0, i, F.relu(v))
            return z.reshape(pre.shape)
        v, i = pre.topk(k, dim=-1)
        z = torch.zeros_like(pre)
        # `topk` keeps the selected values signed; the others apply ReLU first.
        z.scatter_(1, i, v if arm == "topk" else F.relu(v))
        return z

    def train_txc(arm, kp, lr_):
        torch.manual_seed(0)
        m = TemporalCrosscoder(d_in=d, d_sae=d_sae, T=k_seg, k=kp).to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=lr_)
        last_fired = torch.zeros(d_sae, device=dev)
        hist = []
        for s in range(steps):
            xb = Xn[torch.randint(0, Xn.shape[0], (batch,), device=dev)]
            pre = torch.einsum("btd,tds->bs", xb, m.W_enc) + m.b_enc
            z = apply_act(pre, m.k, arm)
            xh = m.decode(z)
            loss = (xh - xb).pow(2).sum(-1).mean()

            fired = (z != 0).float().sum(0)
            last_fired = (last_fired + 1) * (fired == 0).float()

            if arm == "topk_relu_auxk":
                dead = last_fired > 200
                if int(dead.sum()) > k_aux:
                    resid = (xb - xh).detach()
                    pre_d = pre.masked_fill(~dead.unsqueeze(0), -float("inf"))
                    va, ia = pre_d.topk(k_aux, dim=-1)
                    za = torch.zeros_like(pre)
                    za.scatter_(1, ia, F.relu(va))
                    xa = torch.einsum("bs,std->btd", za, m.W_dec)
                    loss = loss + aux_alpha * (xa - resid).pow(2).sum(-1).mean()

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); m._normalize_decoder()
            if s % max(1, steps // 4) == 0 or s == steps - 1:
                hist.append(round(float(loss.detach()), 2))

        with torch.no_grad():
            pre = torch.einsum("btd,tds->bs", Xn_ho, m.W_enc) + m.b_enc
            npos = float((pre > 0).float().sum(-1).mean())
            z = apply_act(pre, m.k, arm)
            xh = m.decode(z)
            fvu = float(((xh - Xn_ho) ** 2).sum(-1).mean() / denom)
            nz = (z != 0)
            l0 = float(nz.float().sum(-1).mean())
            neg = float((z < 0).float().sum(-1).mean() / max(l0, 1e-9))
            alive = float((nz.float().mean(0) >= 0.001).float().mean())
        return {"arm": arm, "kper": kp, "lr": lr_, "nominal_k": m.k,
                "n_pos_preact": npos, "realised_l0": l0,
                "coeff_per_segment": l0 / k_seg, "spend_frac": l0 / m.k,
                "neg_coeff_frac": neg, "alive_frac": alive, "fvu": fvu,
                "loss_hist": hist}

    # ---- SAE reference, unaffected by any of this ----
    out = {"sae": [], "txc": [], "k_seg": k_seg, "d_sae": d_sae, "steps": steps}
    for k in [4, 16]:
        torch.manual_seed(0)
        s_ = TopKSAE(d_in=d, d_sae=d_sae, k=k).to(dev)
        o = torch.optim.Adam(s_.parameters(), lr=1e-3)
        for st in range(steps):
            xb = flat_tr[torch.randint(0, flat_tr.shape[0], (batch * k_seg,), device=dev)]
            loss, _, _ = s_(xb)
            o.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(s_.parameters(), 1.0)
            o.step(); s_._normalize_decoder()
        with torch.no_grad():
            z = s_.encode(flat_ho); xh = s_.decode(z)
            out["sae"].append({
                "k": k, "coeff_per_segment": float((z > 0).float().sum(-1).mean()),
                "fvu": float(((xh - flat_ho) ** 2).sum(-1).mean() / denom)})
        print(f"[SAE k={k}] coeff/seg {out['sae'][-1]['coeff_per_segment']:.2f}  "
              f"FVU {out['sae'][-1]['fvu']:.4f}", flush=True)

    arms = ["topk_relu", "topk", "batchtopk", "topk_relu_auxk"]
    print(f"\n{'arm':<16}{'kper':>5}{'lr':>8}{'nom k':>7}{'#pre>0':>9}{'coeff/seg':>11}"
          f"{'spend':>8}{'neg':>7}{'alive':>8}{'FVU':>9}", flush=True)
    jobs = [(a, kp, lr) for kp in kpers for a in arms]
    jobs += [("topk_relu", kp, ref_lr) for kp in kpers]     # known-good reference
    for arm, kp, lr_ in jobs:
        r = train_txc(arm, kp, lr_)
        out["txc"].append(r)
        print(f"{arm:<16}{kp:>5}{lr_:>8g}{r['nominal_k']:>7}{r['n_pos_preact']:>9.1f}"
              f"{r['coeff_per_segment']:>11.2f}{r['spend_frac']:>8.3f}"
              f"{r['neg_coeff_frac']:>7.3f}{r['alive_frac']:>8.3f}{r['fvu']:>9.4f}",
              flush=True)

    print("\n===== does the activation function rescue capacity at lr=1e-3? =====",
          flush=True)
    for kp in kpers:
        base = next(r for r in out["txc"] if r["arm"] == "topk_relu"
                    and r["kper"] == kp and r["lr"] == lr)
        for r in out["txc"]:
            if r["kper"] != kp or r["lr"] != lr or r["arm"] == "topk_relu":
                continue
            print(f"  kper={kp:>3} {r['arm']:<16} coeff/seg "
                  f"{base['coeff_per_segment']:5.2f} -> {r['coeff_per_segment']:5.2f}   "
                  f"FVU {base['fvu']:.3f} -> {r['fvu']:.3f}   "
                  f"neg-coeff {r['neg_coeff_frac']:.3f}", flush=True)
    return out


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-1.5B-Instruct", layer: int = -1, k_seg: int = 12,
         n_docs: int = 900, d_sae: int = 4096, batch: int = 64, steps: int = 2500,
         kpers: str = "8,41", lr: float = 1e-3, ref_lr: float = 3e-4,
         general_frac: float = 0.4, aux_alpha: float = 0.03, k_aux: int = 64):
    import json
    r = actfn.remote(model, layer, k_seg, n_docs, d_sae, batch, steps,
                     [int(x) for x in kpers.split(",")], lr, ref_lr, general_frac,
                     aux_alpha, k_aux)
    outdir = ROOT / "results" / "dict_bench"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "actfn.json").write_text(json.dumps(r, indent=2))
    print("[saved]", outdir / "actfn.json")
