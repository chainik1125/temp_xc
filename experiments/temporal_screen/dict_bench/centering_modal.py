"""Is the crosscoder's low realised capacity architectural, or an implementation artefact?

The frontier sweep found that a TemporalCrosscoder realises ~2.5 coefficients per segment
no matter how large k is, because only ~30 of 4096 latents have a positive pre-activation
on any given window. TopK cannot select what ReLU then discards. That number barely moves
between k=12 and k=48, so nominal k stops being a budget almost immediately.

Before reporting that as a fact about crosscoders, it has to be separated from three
differences between this repo's TopKSAE and its TemporalCrosscoder that are implementation
choices, not architecture -- all of them in the crosscoder's disfavour:

  1. NO INPUT CENTERING. TopKSAE.encode computes `x - b_dec` before projecting;
     TemporalCrosscoder.encode projects raw x. The cached activations are centered by a
     mean pooled over positions, so each position keeps a residual offset that only the
     crosscoder sees. Summed over T positions the einsum turns that into a fixed per-latent
     offset, which acts exactly like a large learned bias: the same latents win TopK on
     every window, only they receive gradient, and the rest die. That is a winner-take-all
     death spiral, and it would produce precisely the "~30 positive pre-activations
     regardless of k" signature that was measured.

  2. NO DECODER NORMALISATION AT INIT. TopKSAE calls _normalize_decoder() in __init__;
     TemporalCrosscoder does not, so its atoms start at norm sqrt(T*d_in/d_sae) ~ 2.1
     rather than 1 and get rescaled discontinuously after the first optimiser step.

  3. UNTIED INIT. Tying W_enc to W_dec at init is standard practice for TopK SAEs
     precisely because it prevents early latent death.

ARMS (all crosscoders, identical data, identical steps, identical k):
    base          the repo's TemporalCrosscoder, unmodified
    center        + subtract b_dec before the encoder projection (mirrors TopKSAE)
    tied          + W_enc initialised from the normalised W_dec
    center_tied   both
    auxk          base + the standard dead-latent revival auxiliary loss

REGISTERED PREDICTIONS (written before the run):
    R1  If uncentered per-position DC is the cause, `center` raises the positive
        pre-activation count and alive fraction substantially and lowers FVU.
    R2  If the cause is early latent death more generally, `auxk` and `tied` do the same
        while `center` does not.
    R3  If realised coefficients per segment stay near 2.5 in EVERY arm, the cap is a real
        property of a shared window code on this data. The nominal-k warning stands, but
        the "implementation artefact" explanation dies and the frontier can be reported.

The outcome decides whether the frontier's FVU levels are reportable as an architecture
comparison or have to be rerun against a fixed crosscoder.
"""
import pathlib
import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

app = modal.App("dictbench-centering")
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
def centering(model_id: str, layer: int, k_seg: int, n_docs: int, d_sae: int,
              batch: int, steps: int, kpers: list, lr: float, general_frac: float,
              aux_alpha: float, k_aux: int):
    import sys
    sys.path.insert(0, "/work")
    import random
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
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

    # How much per-position DC survives the pooled centering? This is the quantity the
    # crosscoder's encoder sees and the SAE's does not.
    pos_mu = Xn.mean(0)                                   # (T, d)
    print(f"[cache] {tuple(Xn.shape)}  per-position DC norm "
          f"{[round(float(pos_mu[t].norm()), 2) for t in range(k_seg)]}", flush=True)
    print(f"[cache] pooled residual norm {float(pos_mu.mean(0).norm()):.3f}  "
          f"typical segment norm {float(Xn.reshape(-1, d).norm(dim=-1).mean()):.3f}",
          flush=True)

    class TXCVariant(TemporalCrosscoder):
        def __init__(self, d_in, d_sae_, T, k, center=False, tied=False):
            super().__init__(d_in, d_sae_, T, k)
            self.center = center
            with torch.no_grad():
                self._normalize_decoder()          # both arms start at unit atoms
                if tied:
                    self.W_enc.data = self.W_dec.data.permute(1, 2, 0).clone()

        def pre_acts(self, x):
            xx = x - self.b_dec if self.center else x
            return torch.einsum("btd,tds->bs", xx, self.W_enc) + self.b_enc

        def encode(self, x):
            pre = self.pre_acts(x)
            if self.k is None:
                return F.relu(pre)
            v, i = pre.topk(self.k, dim=-1)
            z = torch.zeros_like(pre)
            z.scatter_(1, i, F.relu(v))
            return z

    def gen(bs):
        return Xn[torch.randint(0, Xn.shape[0], (bs,), device=dev)]

    def train(arm, kp):
        torch.manual_seed(0)
        m = TXCVariant(d, d_sae, k_seg, kp,
                       center=arm in ("center", "center_tied"),
                       tied=arm in ("tied", "center_tied")).to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        # Steps since each latent last fired -- drives the auxiliary revival term.
        last_fired = torch.zeros(d_sae, device=dev)
        hist = []
        for s in range(steps):
            xb = gen(batch)
            pre = m.pre_acts(xb)
            v, i = pre.topk(m.k, dim=-1)
            z = torch.zeros_like(pre)
            z.scatter_(1, i, F.relu(v))
            xh = m.decode(z)
            loss = (xh - xb).pow(2).sum(-1).mean()

            fired = torch.zeros(d_sae, device=dev)
            fired.scatter_(0, i.reshape(-1), 1.0)
            last_fired = (last_fired + 1) * (1 - fired)

            if arm == "auxk":
                dead = last_fired > 200
                if int(dead.sum()) > k_aux:
                    resid = (xb - xh).detach()
                    pre_d = pre.masked_fill(~dead.unsqueeze(0), -float("inf"))
                    va, ia = pre_d.topk(k_aux, dim=-1)
                    za = torch.zeros_like(pre)
                    za.scatter_(1, ia, F.relu(va))
                    # Decode the dead latents only; no decoder bias on a residual fit.
                    xa = torch.einsum("bs,std->btd", za, m.W_dec)
                    loss = loss + aux_alpha * (xa - resid).pow(2).sum(-1).mean()

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); m._normalize_decoder()
            if s % max(1, steps // 5) == 0 or s == steps - 1:
                hist.append(round(float(loss), 2))

        with torch.no_grad():
            pre = m.pre_acts(Xn_ho)
            npos = float((pre > 0).float().sum(-1).mean())
            z = m.encode(Xn_ho)
            xh = m.decode(z)
            fvu = float(((xh - Xn_ho) ** 2).sum(-1).mean() / denom)
            l0 = float((z > 0).float().sum(-1).mean())
            alive = float(((z > 0).float().mean(0) >= 0.001).float().mean())
            relu_kill = 1.0 - l0 / m.k
        return {"arm": arm, "kper": kp, "nominal_k": m.k, "n_pos_preact": npos,
                "realised_l0": l0, "coeff_per_segment": l0 / k_seg,
                "relu_kill_frac": relu_kill, "alive_frac": alive, "fvu": fvu,
                "loss_hist": hist}

    out = []
    arms = ["base", "center", "tied", "center_tied", "auxk"]
    print(f"\n{'arm':<12}{'kper':>5}{'nom k':>7}{'#pre>0':>9}{'coeff/seg':>11}"
          f"{'ReLU-kill':>11}{'alive':>8}{'FVU':>8}", flush=True)
    for kp in kpers:
        for arm in arms:
            r = train(arm, kp)
            out.append(r)
            print(f"{r['arm']:<12}{kp:>5}{r['nominal_k']:>7}{r['n_pos_preact']:>9.1f}"
                  f"{r['coeff_per_segment']:>11.2f}{r['relu_kill_frac']:>11.3f}"
                  f"{r['alive_frac']:>8.3f}{r['fvu']:>8.4f}", flush=True)
            print(f"             loss {r['loss_hist']}", flush=True)

    print("\n===== verdict =====", flush=True)
    for kp in kpers:
        rows = [r for r in out if r["kper"] == kp]
        b = next(r for r in rows if r["arm"] == "base")
        for r in rows:
            if r["arm"] == "base":
                continue
            print(f"  kper={kp} {r['arm']:<12} #pre>0 {b['n_pos_preact']:.0f} -> "
                  f"{r['n_pos_preact']:.0f}   coeff/seg {b['coeff_per_segment']:.2f} -> "
                  f"{r['coeff_per_segment']:.2f}   FVU {b['fvu']:.3f} -> {r['fvu']:.3f}",
                  flush=True)
    return {"rows": out, "k_seg": k_seg, "d_sae": d_sae, "steps": steps, "lr": lr}


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-1.5B-Instruct", layer: int = -1, k_seg: int = 12,
         n_docs: int = 900, d_sae: int = 4096, batch: int = 64, steps: int = 2500,
         kpers: str = "4,20", lr: float = 3e-4, general_frac: float = 0.4,
         aux_alpha: float = 0.03, k_aux: int = 64):
    import json
    r = centering.remote(model, layer, k_seg, n_docs, d_sae, batch, steps,
                         [int(x) for x in kpers.split(",")], lr, general_frac,
                         aux_alpha, k_aux)
    outdir = ROOT / "results" / "dict_bench"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "centering.json").write_text(json.dumps(r, indent=2))
    print("[saved]", outdir / "centering.json")
