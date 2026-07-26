"""How does realised capacity scale with window length T? And does T=1 recover the SAE?

At T=1 a TemporalCrosscoder is a TopK SAE. W_enc is (1, d, h) so the einsum collapses to a
matmul; W_dec is (h, 1, d) normalised over (1, 2), which is per-atom unit norm exactly as
TopKSAE normalises its columns. The only surviving differences are the two implementation
choices this sprint found: the crosscoder does not centre its input before the encoder
projection, and it does not normalise its decoder at init.

That makes T=1 an exact control, and it splits the sprint's central open question cleanly:

    T=1 crosscoder ~= TopK SAE   -> the implementation is sound; the capacity collapse
                                    measured at T=12 is a real property of sharing one
                                    code across a window.
    T=1 crosscoder <  TopK SAE   -> the collapse is (at least partly) the implementation,
                                    and every architecture claim in this sprint is void
                                    until it is fixed.

Sweeping T in between then gives the scaling law, which is the quantity a practitioner
actually needs: if realised coefficients per segment fall like 1/T, the shared code is
spending a roughly fixed budget per *window* and diluting it across segments, and choosing
k for a T-token crosscoder has to account for that.

Coefficients per segment are held fixed at `kper` across all T, so the nominal window
budget is kper*T and any change in realised capacity is attributable to T alone.

REGISTERED PREDICTIONS (written before the run):
    Q1  At T=1 the crosscoder's realised coefficients per segment and FVU match the SAE's
        to within a few percent. If it does not, Q2 and Q3 are uninterpretable.
    Q2  #{pre > 0} per window stays roughly flat in T while realised coefficients per
        segment fall like ~1/T.
    Q3  The `center` arm closes whatever T=1 gap exists, since centring is the one
        difference that acts on the encoder pre-activations.
"""
import pathlib
import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

app = modal.App("dictbench-tsweep")
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
def tsweep(model_id: str, layer: int, k_seg: int, n_docs: int, d_sae: int,
           seg_batch: int, steps: int, Ts: list, kper: int, lr: float,
           general_frac: float):
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
    flat_tr = Xn.reshape(-1, d)
    flat_ho = Xn_ho.reshape(-1, d)
    print(f"[cache] train {tuple(Xn.shape)} holdout {tuple(Xn_ho.shape)}", flush=True)

    def reshape_T(flat, T):
        """(N*k_seg, d) -> (M, T, d), dropping the remainder."""
        m = flat.shape[0] // T
        return flat[:m * T].reshape(m, T, d)

    class TXCVariant(TemporalCrosscoder):
        def __init__(self, d_in, d_sae_, T, k, center=False):
            super().__init__(d_in, d_sae_, T, k)
            self.center = center
            with torch.no_grad():
                self._normalize_decoder()

        def pre_acts(self, x):
            xx = x - self.b_dec if self.center else x
            return torch.einsum("btd,tds->bs", xx, self.W_enc) + self.b_enc

        def encode(self, x):
            pre = self.pre_acts(x)
            v, i = pre.topk(self.k, dim=-1)
            z = torch.zeros_like(pre)
            z.scatter_(1, i, F.relu(v))
            return z

    out = {"txc": [], "sae": None, "k_seg": k_seg, "kper": kper, "d_sae": d_sae,
           "steps": steps, "lr": lr}

    # ---------------- SAE reference at the same coefficients per segment ----------------
    torch.manual_seed(0)
    sae = TopKSAE(d, d_sae, kper).to(dev)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    for s in range(steps):
        xb = flat_tr[torch.randint(0, flat_tr.shape[0], (seg_batch,), device=dev)]
        loss, _, _ = sae(xb)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step(); sae._normalize_decoder()
    with torch.no_grad():
        pre = (flat_ho - sae.b_dec) @ sae.W_enc.T + sae.b_enc
        npos_seg = float((pre > 0).float().sum(-1).mean())
        z = sae.encode(flat_ho); xh = sae.decode(z)
        out["sae"] = {
            "n_pos_preact_per_segment": npos_seg,
            "coeff_per_segment": float((z > 0).float().sum(-1).mean()),
            "fvu": float(((xh - flat_ho) ** 2).sum(-1).mean() / denom),
            "alive_frac": float(((z > 0).float().mean(0) >= 0.001).float().mean())}
    r = out["sae"]
    print(f"\n[SAE k={kper}] coeff/seg {r['coeff_per_segment']:.2f}  FVU {r['fvu']:.4f}  "
          f"#pre>0/seg {r['n_pos_preact_per_segment']:.0f}  alive {r['alive_frac']:.3f}",
          flush=True)

    # ---------------- crosscoder at each T, coefficients per segment held fixed ----------
    print(f"\n{'arm':<8}{'T':>4}{'nom k':>7}{'#pre>0/win':>12}{'#pre>0/seg':>12}"
          f"{'coeff/seg':>11}{'ReLU-kill':>11}{'alive':>8}{'FVU':>9}{'xSAE':>7}", flush=True)
    for T in Ts:
        Wtr, Who = reshape_T(flat_tr, T), reshape_T(flat_ho, T)
        # Equal segments per step at every T, so gradient signal is matched.
        wb = max(1, seg_batch // T)
        for arm in ["base", "center"]:
            torch.manual_seed(0)
            m = TXCVariant(d, d_sae, T, kper, center=(arm == "center")).to(dev)
            opt = torch.optim.Adam(m.parameters(), lr=lr)
            for s in range(steps):
                xb = Wtr[torch.randint(0, Wtr.shape[0], (wb,), device=dev)]
                loss, _, _ = m(xb)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step(); m._normalize_decoder()
            with torch.no_grad():
                pre = m.pre_acts(Who)
                npos = float((pre > 0).float().sum(-1).mean())
                z = m.encode(Who); xh = m.decode(z)
                fvu = float(((xh - Who) ** 2).sum(-1).mean() / denom)
                l0 = float((z > 0).float().sum(-1).mean())
                alive = float(((z > 0).float().mean(0) >= 0.001).float().mean())
            rec = {"arm": arm, "T": T, "nominal_k": m.k, "n_pos_preact_per_window": npos,
                   "n_pos_preact_per_segment": npos / T, "realised_l0_per_window": l0,
                   "coeff_per_segment": l0 / T, "relu_kill_frac": 1.0 - l0 / m.k,
                   "alive_frac": alive, "fvu": fvu,
                   "fvu_ratio_to_sae": fvu / out["sae"]["fvu"]}
            out["txc"].append(rec)
            print(f"{arm:<8}{T:>4}{m.k:>7}{npos:>12.1f}{npos/T:>12.1f}"
                  f"{l0/T:>11.2f}{rec['relu_kill_frac']:>11.3f}{alive:>8.3f}"
                  f"{fvu:>9.4f}{rec['fvu_ratio_to_sae']:>7.1f}", flush=True)

    print("\n===== Q1: does T=1 recover the SAE? =====", flush=True)
    s = out["sae"]
    for rec in [r_ for r_ in out["txc"] if r_["T"] == 1]:
        print(f"  T=1 {rec['arm']:<8} coeff/seg {s['coeff_per_segment']:.2f} -> "
              f"{rec['coeff_per_segment']:.2f}   FVU {s['fvu']:.4f} -> {rec['fvu']:.4f} "
              f"({rec['fvu_ratio_to_sae']:.2f}x)   #pre>0/seg "
              f"{s['n_pos_preact_per_segment']:.0f} -> "
              f"{rec['n_pos_preact_per_segment']:.0f}", flush=True)

    print("\n===== Q2: the scaling law =====", flush=True)
    for arm in ["base", "center"]:
        rows = [r_ for r_ in out["txc"] if r_["arm"] == arm]
        print(f"  {arm:<8} coeff/seg by T: "
              + "  ".join(f"T={r_['T']}:{r_['coeff_per_segment']:.2f}" for r_ in rows),
              flush=True)
        print(f"  {arm:<8} #pre>0/win by T: "
              + "  ".join(f"T={r_['T']}:{r_['n_pos_preact_per_window']:.0f}"
                          for r_ in rows), flush=True)
    return out


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-1.5B-Instruct", layer: int = -1, k_seg: int = 12,
         n_docs: int = 900, d_sae: int = 4096, seg_batch: int = 768, steps: int = 2500,
         ts: str = "1,2,3,4,6,12", kper: int = 4, lr: float = 3e-4,
         general_frac: float = 0.4):
    import json
    r = tsweep.remote(model, layer, k_seg, n_docs, d_sae, seg_batch, steps,
                      [int(x) for x in ts.split(",")], kper, lr, general_frac)
    outdir = ROOT / "results" / "dict_bench"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "tsweep.json").write_text(json.dumps(r, indent=2))
    print("[saved]", outdir / "tsweep.json")
