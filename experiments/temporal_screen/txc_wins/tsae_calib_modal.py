"""Calibrate the tSAE's L1 coefficient so it can be used as a real baseline.

CARRIED-OVER DEBT. The previous sprint's head-to-head (`bench4_modal.py`) ran this repo's
`tsae_paper` -- the attention `TemporalSAE` with `sae_diff_type="relu"` plus an L1 penalty
on (novel_codes + pred_codes), exactly as
`experiments/ward_backtracking_txc/architectures.py:183-189` writes the loss -- and it was
DENSE at every setting tried: 2989 of 4096 latents active per segment, alive fraction 0.999,
FVU 0.030, and a 100x sweep of `l1_coef` over 1e-4..1e-2 moved realised L0 by 0.3%. An arm
that spends ~3000 coefficients per segment is not a sparse dictionary and cannot be compared
against a TopK SAE at 1-16.

DIAGNOSIS. `TemporalSAE.__init__` sets `self.lam = 1 / (4 * dimin)` and the encoder is
`z = ReLU((x * lam) @ E)`, so at d_in = 1536 every pre-activation is scaled by 1.6e-4 before
the ReLU. The codes are therefore ~1e-2 in magnitude while the reconstruction term
`||x - zD||^2` summed over d is ~1e3. At `l1 = 1e-3` the penalty contributes ~0.03 against a
~1e3 reconstruction loss: it is not a weak prior, it is numerically absent. A published
sparsity coefficient is only meaningful relative to the activation scale it was tuned on.

Order-of-magnitude estimate of where it should live: with tied weights, `||D_j||^2 ~ d`, the
optimal coefficient for a feature is `z* ~ (x . D_j) / ||D_j||^2 ~ 2.5e-2`, and an L1 penalty
`l1` soft-thresholds by `l1 / (2 ||D_j||^2) = l1 / 2d`. Those are comparable at `l1 ~ 75`.
So the grid has to run to O(100), not O(1e-3) -- five orders of magnitude above the
documented value. The scale is pinned rather than gameable: with `E = D.T`, scaling
`D -> cD` scales the codes by `c` and the reconstruction by `c^2`, so the model cannot
inflate the dictionary to evade the penalty.

WHAT THIS RUNS. The corpus, caching, windowing and normalisation are lifted unchanged from
`bench4_modal.py` (structured run-length family, stride-1 windows, layer 14 of
Qwen2.5-1.5B-Instruct), so the calibrated coefficient transfers directly to that benchmark.
Stage 1 is a coarse log grid over `l1`; stage 2 refines inside whichever bracket contains the
target band. A TopK SAE is trained on the same cache at k = 1..16 as the reference curve, so
the output says not just "this l1 gives L0 = 4" but "at L0 = 4 the tSAE's FVU is X against
the SAE's Y".

DELIVERABLE: the `l1` range giving realised novel-code L0 in the 1-32 coefficients/segment
band, plus the interpolated `l1` for each budget in {1, 2, 4, 8, 16, 32}.

ACCOUNTING. Only the NOVEL codes are stored per token -- the predicted codes are computed
from context by the attention layer -- so novel L0 per token is the coefficients-per-segment
figure, following `bench4_modal.py`. Predicted L0 is recorded alongside at every point so a
reader who wants to charge it can.

REGISTERED PREDICTIONS (written before the run):
  C1  L0 is flat in `l1` across 1e-3..1e-1 (reproducing the failure), then falls steeply
      somewhere in 1..1e3. If it is flat all the way to 1e3, the penalty is not reaching the
      codes at all and the diagnosis above is wrong -- the next suspect would be the
      projection path, which subtracts `proj_scale * D z_pred` from the input before the
      novel codes are computed and could be absorbing the reconstruction regardless.
  C2  FVU rises monotonically as L0 falls, and at matched L0 the tSAE is WORSE than the TopK
      SAE -- an L1 dictionary pays a shrinkage cost a TopK one does not.
  C3  Alive fraction falls with l1. If L0 falls while alive fraction stays ~1.0, sparsity is
      being achieved by shrinking every code rather than by selecting, which would make the
      arm sparse-by-measurement and dense-by-behaviour.

    modal run experiments/temporal_screen/txc_wins/tsae_calib_modal.py
"""
import pathlib
import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

app = modal.App("txcwins-tsaecalib")
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers", "accelerate", "numpy")
    .add_local_dir(str(ROOT / "src"), "/work/src")
    .add_local_dir(str(ROOT / "temporal_crosscoders"), "/work/temporal_crosscoders")
)

# Corpus lifted verbatim from bench4_modal.py so the calibration transfers.
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
ELLS = [1, 2, 3, 6]


@app.function(gpu="A10G", image=image, timeout=21600)
def calib(model_id: str, layer: int, k_seg: int, T: int, n_docs: int, d_sae: int,
          steps: int, lr: float, general_frac: float, batch_win: int,
          coarse: list, sae_budgets: list, targets: list, n_refine: int,
          topk_grid: list, l1_on: str, topk_lrs: list, topk_steps: list):
    import sys
    sys.path.insert(0, "/work")
    import math
    import random
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.bench.architectures.topk_sae import TopKSAE
    from temporal_crosscoders.han_tsae import TemporalSAE

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
            idx = [i for i, (s0, s1) in enumerate(offs)
                   if s0 >= a and s1 <= b and s1 > s0]
            ts.append((idx[0], idx[-1]) if idx else (0, 0))
        h = layers_[L].register_forward_hook(cap_hook)
        with torch.no_grad():
            model(e["input_ids"].to(dev))
        h.remove()
        hh = cap["h"][0].float().cpu().numpy()
        return np.stack([hh[a:b + 1].mean(0) for a, b in ts])

    X = []
    n_gen = int(n_docs * general_frac)
    for i in range(n_docs):
        if i < n_gen:
            sents = [GENERAL[rng.randrange(len(GENERAL))] for _ in range(k_seg)]
        else:
            e_ = ELLS[rng.randrange(len(ELLS))]
            ph = rng.randrange(2 * e_)
            lab = [1 if ((t + ph) // e_) % 2 == 0 else 0 for t in range(k_seg)]
            sents = [(TENSE if l else CALM)[rng.randrange(10)] for l in lab]
        X.append(capture(*build(rng.choice(CARRIERS), sents)))
        if (i + 1) % 250 == 0:
            print(f"   [cache] {i+1}/{n_docs}", flush=True)
    Xt = torch.tensor(np.stack(X), dtype=torch.float32, device=dev)

    n_hold = max(int(0.15 * Xt.shape[0]), 32)
    Xtr, Xho = Xt[:-n_hold], Xt[-n_hold:]
    mu, sd = Xtr.mean((0, 1), keepdim=True), Xtr.std() + 1e-6
    Xn, Xn_ho = (Xtr - mu) / sd, (Xho - mu) / sd

    def unfold_windows(A, T_, stride=1):
        """(N, k_seg, d) -> (N*(k_seg-T_+1), T_, d), stride-1 within document."""
        return (A.unfold(1, T_, stride).permute(0, 1, 3, 2)
                .reshape(-1, T_, A.shape[-1]).contiguous())

    Wtr = unfold_windows(Xn, T)
    Who = unfold_windows(Xn_ho, T)
    flat_tr = Xn.reshape(-1, d)
    flat_ho_w = Who.reshape(-1, d)
    denom = float(flat_ho_w.var(0).sum())
    tok_norm = float(flat_ho_w.norm(dim=-1).mean())
    print(f"[cache] windows train {tuple(Wtr.shape)} holdout {tuple(Who.shape)}  "
          f"denom {denom:.1f}  mean ||x|| {tok_norm:.2f}  lam {1/(4*d):.3e}", flush=True)

    def gen_win(bs):
        return Wtr[torch.randint(0, Wtr.shape[0], (bs,), device=dev)]

    def gen_flat(bs):
        return flat_tr[torch.randint(0, flat_tr.shape[0], (bs,), device=dev)]

    def fvu_from(xh, ref):
        return float(((xh - ref) ** 2).sum(-1).mean() / denom)

    # ---------------- reference: TopK SAE on the same cache ----------------
    sae_rows = []
    for k in sae_budgets:
        torch.manual_seed(0)
        m = TopKSAE(d_in=d, d_sae=d_sae, k=k).to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        m.train()
        for _ in range(steps):
            loss, _, _ = m(gen_flat(batch_win * T))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); m._normalize_decoder()
        m.eval()
        with torch.no_grad():
            Z = m.encode(flat_ho_w)
            xh = m.decode(Z)
            row = {"k": k,
                   "coeff_per_segment": float((Z > 0).float().sum(-1).mean()),
                   "fvu": fvu_from(xh, flat_ho_w),
                   "alive_frac": float(((Z > 0).float().mean(0) >= 0.001).float().mean())}
        sae_rows.append(row)
        print(f"  [sae ref] k={k:<3} coeff/seg {row['coeff_per_segment']:6.2f}  "
              f"FVU {row['fvu']:.4f}  alive {row['alive_frac']:.3f}", flush=True)

    # ---------------- the tSAE sweep ----------------
    def run_tsae(l1):
        """Train tsae_paper at this l1.

        `l1_on == "both"` is verbatim `architectures.py:183-189`: the penalty falls on
        (novel_codes + pred_codes). That is the repo's reading, and it charges the attention
        path -- the predicted codes are the temporal mechanism -- for its own existence.
        `l1_on == "novel"` is the competing reading, penalising only the codes that are
        actually stored per token, which is also the quantity the sparsity axis counts. Both
        are run because the tSAE identification is unresolved and guessing between them
        would silently pick which architecture the baseline is.
        """
        torch.manual_seed(0)
        m = TemporalSAE(dimin=d, width=d_sae, n_heads=8, sae_diff_type="relu",
                        kval_topk=None, tied_weights=True, n_attn_layers=1,
                        bottleneck_factor=64).to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        m.train()
        last = {}
        for s_ in range(steps):
            xb = gen_win(batch_win)
            recons, info = m(xb)
            recon = (recons - xb).pow(2).sum(-1).mean()
            z_tot = (info["novel_codes"] + info["pred_codes"] if l1_on == "both"
                     else info["novel_codes"])
            l1_term = z_tot.abs().sum(-1).mean()
            loss = recon + l1 * l1_term
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            if s_ == steps - 1:
                last = {"recon_term": float(recon), "l1_term": float(l1_term),
                        "l1_contrib": float(l1 * l1_term)}
        m.eval()
        with torch.no_grad():
            xh, info = m(Who)
            Zn = info["novel_codes"].reshape(-1, d_sae)
            Zp = info["pred_codes"].reshape(-1, d_sae)
            act = Zn[Zn > 0]
            row = {
                "l1_coef": float(l1), "l1_on": l1_on,
                "coeff_per_segment": float((Zn > 0).float().sum(-1).mean()),
                "pred_l0_per_segment": float((Zp > 0).float().sum(-1).mean()),
                "fvu": fvu_from(xh, Who),
                "alive_frac": float(((Zn > 0).float().mean(0) >= 0.001).float().mean()),
                "mean_abs_active_z": float(act.abs().mean()) if act.numel() else 0.0,
                "max_z": float(Zn.max()),
                "dict_row_norm": float(m.D.data.norm(dim=-1).mean()),
                # Fraction of the novel-code mass carried by the top-32 latents. If L0
                # falls only because a long tail of tiny codes is being shrunk, this
                # stays near 1 while L0 is still in the thousands.
                "mass_top32": float(
                    Zn.abs().topk(32, dim=-1).values.sum(-1).mean()
                    / (Zn.abs().sum(-1).mean() + 1e-9)),
            }
            row.update(last)
        print(f"  [tsae] l1={l1:<10.4g} coeff/seg {row['coeff_per_segment']:8.2f}  "
              f"predL0 {row['pred_l0_per_segment']:7.1f}  FVU {row['fvu']:.4f}  "
              f"alive {row['alive_frac']:.3f}  |z| {row['mean_abs_active_z']:.2e}  "
              f"recon {row['recon_term']:.1f}  l1term*l1 {row['l1_contrib']:.2f}",
              flush=True)
        return row

    # ---------------- fallback arm: the same TemporalSAE with TopK ----------------
    # If L1 cannot reach the target band without destroying reconstruction, the honest
    # baseline is the same attention architecture with a sparsity rule that binds by
    # construction. `sae_diff_type="topk"` applies TopK to the post-ReLU novel codes, so
    # realised L0 = min(k, #{pre > 0}); with ~2000 positive pre-activations the k term is
    # what binds, but it is measured here rather than assumed.
    topk_rows = []
    if topk_grid:
        print("\n===== reference arm: TemporalSAE with TopK (no L1) =====", flush=True)
        # The learning rate and step count are swept alongside k because an arm that
        # reconstructs badly at one training recipe has two possible explanations, and only
        # one of them is about the architecture. Last sprint a 3x change in learning rate
        # moved a crosscoder's realised spend by 10.6x with nothing in the nominal config to
        # show for it, so an unswept baseline is not a fair baseline.
        for kv in topk_grid:
            for lr_ in (topk_lrs or [lr]):
                for st_ in (topk_steps or [steps]):
                    torch.manual_seed(0)
                    m = TemporalSAE(dimin=d, width=d_sae, n_heads=8,
                                    sae_diff_type="topk", kval_topk=int(kv),
                                    tied_weights=True, n_attn_layers=1,
                                    bottleneck_factor=64).to(dev)
                    opt = torch.optim.Adam(m.parameters(), lr=lr_)
                    m.train()
                    for _ in range(int(st_)):
                        xb = gen_win(batch_win)
                        recons, _ = m(xb)
                        loss = (recons - xb).pow(2).sum(-1).mean()
                        opt.zero_grad(); loss.backward()
                        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                        opt.step()
                    m.eval()
                    with torch.no_grad():
                        xh, info = m(Who)
                        Zn = info["novel_codes"].reshape(-1, d_sae)
                        Zp = info["pred_codes"].reshape(-1, d_sae)
                        row = {"kval": int(kv), "lr": float(lr_), "steps": int(st_),
                               "coeff_per_segment": float(
                                   (Zn > 0).float().sum(-1).mean()),
                               "pred_l0_per_segment": float(
                                   (Zp > 0).float().sum(-1).mean()),
                               "fvu": fvu_from(xh, Who),
                               "alive_frac": float(
                                   ((Zn > 0).float().mean(0) >= 0.001).float().mean())}
                    topk_rows.append(row)
                    print(f"  [tsae-topk] k={kv:<4} lr={lr_:<7.1e} steps={st_:<6} "
                          f"coeff/seg {row['coeff_per_segment']:7.2f}  "
                          f"predL0 {row['pred_l0_per_segment']:7.1f}  "
                          f"FVU {row['fvu']:.4f}  alive {row['alive_frac']:.3f}",
                          flush=True)

    print("\n===== stage 1: coarse log grid =====", flush=True)
    rows = [run_tsae(l1) for l1 in coarse]

    def bracket(rows_, lo_target, hi_target):
        """Smallest l1 interval bracketing the L0 = `hi_target` crossing.

        Bracketing the TOP of the band rather than its full width is what puts the refine
        points where the answer lives: the first coarse sweep went from FVU 0.32 at 151
        coefficients per segment to FVU 1.05 at 18, so whether any l1 gives a usable
        dictionary inside the band is decided entirely at that crossing.
        """
        s = sorted(rows_, key=lambda r: r["l1_coef"])
        lo = max([r["l1_coef"] for r in s if r["coeff_per_segment"] > hi_target],
                 default=None)
        hi = min([r["l1_coef"] for r in s if r["coeff_per_segment"] <= hi_target],
                 default=None)
        return lo, hi

    print("\n===== stage 2: refine inside the bracket =====", flush=True)
    lo, hi = bracket(rows, min(targets), max(targets))
    if lo is None:
        print("  [refine] L0 never exceeds the band top even at the smallest l1 -- "
              "nothing to bracket from below", flush=True)
    if hi is None and lo is not None:
        # Band never entered: push the grid up by decades until L0 drops below the floor.
        cur = max(r["l1_coef"] for r in rows)
        for _ in range(4):
            cur *= 10.0
            r = run_tsae(cur)
            rows.append(r)
            if r["coeff_per_segment"] < min(targets):
                break
        lo, hi = bracket(rows, min(targets), max(targets))
    if lo is not None and hi is not None and hi > lo:
        for i in range(1, n_refine + 1):
            f = i / (n_refine + 1)
            rows.append(run_tsae(float(10 ** (math.log10(lo) + f * (math.log10(hi) - math.log10(lo))))))

    rows.sort(key=lambda r: r["l1_coef"])

    # ---------------- interpolate l1 for each target budget ----------------
    xs = [math.log10(r["l1_coef"]) for r in rows]
    ys = [math.log10(max(r["coeff_per_segment"], 1e-6)) for r in rows]
    monotone = all(ys[i + 1] <= ys[i] + 1e-9 for i in range(len(ys) - 1))

    def l1_for(target):
        """log-log interpolate l1 at which realised L0 == target. None if unbracketed."""
        t = math.log10(target)
        for i in range(len(xs) - 1):
            a, b = ys[i], ys[i + 1]
            if (a >= t >= b) or (b >= t >= a):
                if abs(b - a) < 1e-12:
                    return float(10 ** xs[i])
                f = (t - a) / (b - a)
                return float(10 ** (xs[i] + f * (xs[i + 1] - xs[i])))
        return None

    l1_map = {str(t): l1_for(t) for t in targets}
    in_band = [r for r in rows
               if min(targets) <= r["coeff_per_segment"] <= max(targets)]
    usable = ([min(r["l1_coef"] for r in in_band), max(r["l1_coef"] for r in in_band)]
              if in_band else None)

    print("\n===== calibration curve =====", flush=True)
    print(f"  {'l1':>12} {'coeff/seg':>11} {'predL0':>9} {'FVU':>8} {'alive':>7} "
          f"{'top32 mass':>11}", flush=True)
    for r in rows:
        print(f"  {r['l1_coef']:>12.4g} {r['coeff_per_segment']:>11.2f} "
              f"{r['pred_l0_per_segment']:>9.1f} {r['fvu']:>8.4f} "
              f"{r['alive_frac']:>7.3f} {r['mass_top32']:>11.3f}", flush=True)
    print(f"\n  monotone in l1: {monotone}", flush=True)
    print(f"  usable l1 range (measured, L0 in [{min(targets)}, {max(targets)}]): "
          f"{usable}", flush=True)
    for t in targets:
        v = l1_map[str(t)]
        print(f"    L0 = {t:<4} -> l1 = " + (f"{v:.4g}" if v else "unbracketed"),
              flush=True)

    print("\n===== tSAE vs TopK SAE at matched coefficients/segment =====", flush=True)
    for sr in (sae_rows if rows else []):
        tgt = sr["coeff_per_segment"]
        near = min(rows, key=lambda r: abs(math.log10(max(r["coeff_per_segment"], 1e-6))
                                           - math.log10(max(tgt, 1e-6))))
        print(f"  {tgt:6.2f}/seg   SAE FVU {sr['fvu']:.4f}   "
              f"tSAE FVU {near['fvu']:.4f} (at {near['coeff_per_segment']:.2f}/seg, "
              f"l1={near['l1_coef']:.4g})", flush=True)

    return {"model": model_id, "layer": int(L), "k_seg": k_seg, "T": T, "d_sae": d_sae,
            "steps": steps, "lr": lr, "d_in": d, "lam": 1.0 / (4 * d),
            "mean_token_norm": tok_norm, "fvu_denom": denom,
            "sae_reference": sae_rows, "tsae_curve": rows,
            "tsae_topk_reference": topk_rows, "monotone": monotone,
            "l1_on": l1_on,
            "usable_l1_range": usable, "l1_for_target_l0": l1_map}


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-1.5B-Instruct", layer: int = -1, k_seg: int = 36,
         t: int = 12, n_docs: int = 500, d_sae: int = 4096, steps: int = 2000,
         lr: float = 3e-4, general_frac: float = 0.3, batch_win: int = 32,
         coarse: str = "1e-3,1e-2,1e-1,1,3,10,30,100,300,1000",
         sae_budgets: str = "1,2,4,8,16", targets: str = "1,2,4,8,16,32",
         n_refine: int = 6, topk_grid: str = "", l1_on: str = "both",
         topk_lrs: str = "", topk_steps: str = "", tag: str = ""):
    import json
    r = calib.remote(model, layer, k_seg, t, n_docs, d_sae, steps, lr, general_frac,
                     batch_win, [float(x) for x in coarse.split(",") if x],
                     [int(x) for x in sae_budgets.split(",") if x],
                     [int(x) for x in targets.split(",") if x], n_refine,
                     [int(x) for x in topk_grid.split(",") if x], l1_on,
                     [float(x) for x in topk_lrs.split(",") if x],
                     [int(x) for x in topk_steps.split(",") if x])
    outdir = ROOT / "results" / "dict_bench"
    outdir.mkdir(parents=True, exist_ok=True)
    name = f"tsae_calibration{tag}.json"
    (outdir / name).write_text(json.dumps(r, indent=2))
    print("[saved]", outdir / name)
