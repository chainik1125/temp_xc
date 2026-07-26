"""First-pass benchmark: TopK SAE vs Temporal Crosscoder vs tSAE at paper params.

This is the deliverable the sprint set out to produce. Everything before it was
methodology, and all of that methodology is folded in here:

  * the sparsity axis is REALISED coefficients per segment, measured per run, never
    nominal k -- nominal k does not bind for the crosscoder and a comparison matched on it
    compares two different amounts of capacity (see log.md, "realised capacity");
  * windows are cut at STRIDE 1 within a document, so the number of training windows does
    not fall with T and large-T arms are not silently starved of data;
  * the crosscoder uses `batchtopk` (no ReLU), which is the only sparsity rule measured to
    spend its budget -- ReLU-after-TopK caps realised L0 at #{pre > 0};
  * the corpus is the RUN-LENGTH family, which actually contains a window-level factor, so
    a window code has something to find. The i.i.d. corpus is carried as a control where
    the window factor is absent and window-AUC must read 0.5.

THE THREE ARMS

  sae       TopKSAE, one code per token. The per-token baseline.
  txc       TemporalCrosscoder, plain -- one shared code for the whole T-window, with
            `batchtopk` sparsity and NO auxiliary penalty (not the TXC-pro variant: no
            matryoshka, no contrastive term, no anti-dead stack).
  tsae_paper  The attention-based TemporalSAE with ReLU + L1 (l1_coef), which is what
            THIS repo calls `tsae_paper` -- see experiments/ward_backtracking_txc/
            architectures.py:12, "Bhalla 2025 paper-faithful ... ReLU activation + L1
            sparsity penalty instead of TopK. Same TemporalSAE class". Params as the repo
            sets them: n_heads=8, bottleneck_factor=64, n_attn_layers=1, tied_weights.
            Sparsity is swept via l1_coef rather than k, which is exactly why the shared
            axis has to be REALISED coefficients per segment -- it is the only axis on
            which an L1 arm and a TopK arm are comparable at all.

  tsae_nce  A per-token TopK SAE with an InfoNCE consistency penalty between codes at
            NEARBY POSITIONS -- no attention. Included because the repo's `tsae_paper`
            label may not match the architecture intended: the InfoNCE machinery here
            lives only in han_arch/txc_bare_*contrastive*, which are TXC variants, so an
            InfoNCE-based tSAE is not defined in this repo. Rather than guess between the
            two readings, both are run. The InfoNCE form follows han_arch's `_info_nce`
            (normalise, batch similarity, symmetric cross-entropy against the diagonal)
            applied to codes one position apart; alpha and the shift are this sprint's
            choices, not the paper's, and are recorded in the output.

  tfa       The same TemporalSAE with TopK instead of ReLU+L1 -- the attention arm proper,
            kept so the attention mechanism and the sparsity rule are not confounded.

ACCOUNTING NOTE, and it is the one judgement call in the comparison. For the tSAE, only the
NOVEL codes are stored per token; the predicted codes are computed from context. Novel L0
per token is therefore its coefficients-per-segment, and pred L0 is reported alongside so
the reader can charge it if they disagree.

READOUTS. Segment-AUC asks a per-segment question (which sentence is tense); window-AUC
asks a window-level one (which run-length regime). A per-token code answers the first
directly and the second only after pooling; a window code is the reverse. Both are reported
for every arm, so the structural mismatch is visible rather than papered over.

    modal run experiments/temporal_screen/dict_bench/bench4_modal.py
"""
import pathlib
import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

app = modal.App("dictbench-bench4")
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers", "accelerate", "numpy")
    .add_local_dir(str(ROOT / "src"), "/work/src")
    # tsae_paper lives in the vendored han_tsae package, not under src/.
    .add_local_dir(str(ROOT / "temporal_crosscoders"), "/work/temporal_crosscoders")
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
ELLS = [1, 2, 3, 6]          # run lengths; the window-level factor


@app.function(gpu="A10G", image=image, timeout=21600)
def bench4(model_id: str, layer: int, k_seg: int, T: int, n_docs: int, d_sae: int,
           steps: int, budgets: list, lr: float, general_frac: float, batch_win: int,
           l1_grid: list, nce_alpha: float, nce_shift: int, arms: list):
    import sys
    sys.path.insert(0, "/work")
    import math
    import random
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.bench.architectures.topk_sae import TopKSAE
    from src.bench.architectures.stacked_sae import StackedSAE
    from src.bench.architectures.crosscoder import TemporalCrosscoder
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

    def make_cache(structured, n):
        X, seg, ells = [], [], []
        n_gen = int(n * general_frac)
        for i in range(n):
            if i < n_gen:
                sents = [GENERAL[rng.randrange(len(GENERAL))] for _ in range(k_seg)]
                lab, e_ = [-1] * k_seg, -1
            elif structured:
                e_ = ELLS[rng.randrange(len(ELLS))]
                ph = rng.randrange(2 * e_)
                lab = [1 if ((t + ph) // e_) % 2 == 0 else 0 for t in range(k_seg)]
                sents = [(TENSE if l else CALM)[rng.randrange(10)] for l in lab]
            else:
                lab, e_ = [rng.randint(0, 1) for _ in range(k_seg)], -1
                sents = [(TENSE if l else CALM)[rng.randrange(10)] for l in lab]
            X.append(capture(*build(rng.choice(CARRIERS), sents)))
            seg.append(lab); ells.append(e_)
            if (i + 1) % 250 == 0:
                print(f"   [cache] {i+1}/{n}", flush=True)
        return np.stack(X), np.array(seg), np.array(ells)

    def auc_of(scores, y):
        if y.sum() == 0 or (1 - y).sum() == 0:
            return 0.5
        o = torch.argsort(scores)
        r = torch.empty_like(o, dtype=torch.float32)
        r[o] = torch.arange(len(scores), device=scores.device,
                            dtype=torch.float32) + 1
        n1, n0 = float(y.sum()), float((1 - y).sum())
        return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    def best_auc(Z, y):
        """Best single-latent AUC, oriented (so 0.5 is chance and higher is better)."""
        if Z.shape[0] == 0 or y.sum() == 0 or (1 - y).sum() == 0:
            return 0.5
        a = torch.tensor([auc_of(Z[:, j], y) for j in range(Z.shape[1])], device=dev)
        return float(0.5 + (a - 0.5).abs().max())

    def unfold_windows(A, T_, stride=1):
        """(N, k_seg, ...) -> (N*(k_seg-T_+1), T_, ...), stride-1 within document."""
        return (A.unfold(1, T_, stride).permute(0, 1, 3, 2)
                .reshape(-1, T_, A.shape[-1]).contiguous())

    def unfold_labels(A, T_, stride=1):
        return A.unfold(1, T_, stride).reshape(-1, T_).contiguous()

    out = {"model": model_id, "layer": int(L), "k_seg": k_seg, "T": T,
           "d_sae": d_sae, "steps": steps, "lr": lr, "corpora": {}}

    for structured in (True, False):
        tag = "structured" if structured else "iid"
        print(f"\n########## CORPUS: {tag} ##########", flush=True)
        Xr, segr, ellr = make_cache(structured, n_docs)
        Xt = torch.tensor(Xr, dtype=torch.float32, device=dev)
        segt = torch.tensor(segr, device=dev)
        ellt = torch.tensor(ellr, device=dev)

        n_hold = max(int(0.15 * Xt.shape[0]), 32)
        Xtr, Xho = Xt[:-n_hold], Xt[-n_hold:]
        mu, sd = Xtr.mean((0, 1), keepdim=True), Xtr.std() + 1e-6
        Xn, Xn_ho = (Xtr - mu) / sd, (Xho - mu) / sd

        Wtr = unfold_windows(Xn, T)
        Who = unfold_windows(Xn_ho, T)
        seg_ho = unfold_labels(segt[-n_hold:], T)
        # A window inherits its document's run length; binary split at the median.
        ell_ho = ellt[-n_hold:].unsqueeze(1).expand(-1, k_seg - T + 1).reshape(-1)
        win_y = (ell_ho >= 3).long()
        win_keep = ell_ho > 0
        seg_flat = seg_ho.reshape(-1)
        seg_keep = seg_flat >= 0

        flat_tr = Xn.reshape(-1, d)
        flat_ho_w = Who.reshape(-1, d)
        denom = float(flat_ho_w.var(0).sum())
        print(f"[cache] windows train {tuple(Wtr.shape)} holdout {tuple(Who.shape)}  "
              f"denom {denom:.1f}", flush=True)

        def gen_win(bs):
            return Wtr[torch.randint(0, Wtr.shape[0], (bs,), device=dev)]

        def gen_flat(bs):
            return flat_tr[torch.randint(0, flat_tr.shape[0], (bs,), device=dev)]

        def fvu_from(xh, ref):
            return float(((xh - ref) ** 2).sum(-1).mean() / denom)

        def readouts(Zseg, Zwin):
            """Zseg: (n_win*T, h) per-segment codes or None. Zwin: (n_win, h)."""
            s = best_auc(Zseg[seg_keep], seg_flat[seg_keep]) if Zseg is not None else 0.5
            w = best_auc(Zwin[win_keep], win_y[win_keep])
            return s, w

        rows = []

        def record(arm, nominal, l0_seg, fvu, alive, s_auc, w_auc, n_params, extra=None):
            r = {"arm": arm, "nominal": nominal, "coeff_per_segment": l0_seg,
                 "fvu": fvu, "alive_frac": alive, "segment_auc": s_auc,
                 "window_auc": w_auc, "n_params": n_params}
            if extra:
                r.update(extra)
            rows.append(r)
            print(f"  {arm:<14} nom={nominal:<4} coeff/seg {l0_seg:6.2f}  FVU {fvu:.4f}  "
                  f"alive {alive:.3f}  segAUC {s_auc:.3f}  winAUC {w_auc:.3f}  "
                  f"params {n_params/1e6:.1f}M", flush=True)

        def adam_train(m, gen, bs):
            opt = torch.optim.Adam(m.parameters(), lr=lr)
            m.train()
            for s in range(steps):
                loss, _, _ = m(gen(bs))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step(); m._normalize_decoder()
            m.eval()

        for k, l1 in zip(budgets, l1_grid):
            # ---- 1. per-token TopK SAE ----
            torch.manual_seed(0)
            m = TopKSAE(d_in=d, d_sae=d_sae, k=k).to(dev)
            adam_train(m, gen_flat, batch_win * T)
            with torch.no_grad():
                Zf = m.encode(flat_ho_w)
                xh = m.decode(Zf)
                l0 = float((Zf > 0).float().sum(-1).mean())
                alive = float(((Zf > 0).float().mean(0) >= 0.001).float().mean())
                s_auc, w_auc = readouts(Zf, Zf.reshape(-1, T, d_sae).mean(1))
            record("sae", k, l0, fvu_from(xh, flat_ho_w), alive, s_auc, w_auc,
                   sum(p_.numel() for p_ in m.parameters()))

            # ---- 2. Temporal Crosscoder, plain: batchtopk, no auxiliary penalty ----
            if k * T <= d_sae:
                torch.manual_seed(0)
                m = TemporalCrosscoder(d_in=d, d_sae=d_sae, T=T, k=k,
                                       activation="batchtopk").to(dev)
                with torch.no_grad():
                    m._normalize_decoder()
                adam_train(m, gen_win, batch_win)
                with torch.no_grad():
                    Zw = m.encode(Who)
                    xh = m.decode(Zw)
                    l0 = float((Zw != 0).float().sum(-1).mean()) / T
                    alive = float(((Zw != 0).float().mean(0) >= 0.001).float().mean())
                    # A window code has no per-segment answer; broadcasting it is the
                    # honest representation of that, and should read ~0.5.
                    Zseg = Zw.unsqueeze(1).expand(-1, T, -1).reshape(-1, d_sae)
                    s_auc, w_auc = readouts(Zseg, Zw)
                record("txc", k, l0, fvu_from(xh, Who), alive, s_auc, w_auc,
                       sum(p_.numel() for p_ in m.parameters()))

            # ---- 3. tsae_paper: attention TemporalSAE, ReLU + L1 ----
            # Sparsity is set by l1_coef, not k, which is precisely why the shared axis
            # has to be realised coefficients per segment: it is the only one on which an
            # L1 arm and a TopK arm are comparable at all. l1=1e-3 is the repo's
            # documented paper value and is inside the swept grid.
            torch.manual_seed(0)
            m = TemporalSAE(dimin=d, width=d_sae, n_heads=8,
                            sae_diff_type="relu", kval_topk=None,
                            tied_weights=True, n_attn_layers=1,
                            bottleneck_factor=64).to(dev)
            opt = torch.optim.Adam(m.parameters(), lr=lr)
            m.train()
            for s_ in range(steps):
                xb = gen_win(batch_win)
                recons, info = m(xb)
                loss = F.mse_loss(recons.reshape(-1, d), xb.reshape(-1, d),
                                  reduction="sum") / (xb.shape[0] * T)
                z_tot = info["novel_codes"] + info["pred_codes"]
                loss = loss + l1 * z_tot.abs().sum(-1).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
            m.eval()
            with torch.no_grad():
                xh, info = m(Who)
                Zn = info["novel_codes"].reshape(-1, d_sae)
                Zp = info["pred_codes"].reshape(-1, d_sae)
                l0 = float((Zn > 0).float().sum(-1).mean())
                l0p = float((Zp > 0).float().sum(-1).mean())
                alive = float(((Zn > 0).float().mean(0) >= 0.001).float().mean())
                s_auc, w_auc = readouts(Zn, info["novel_codes"].mean(1))
            record("tsae_paper", l1, l0, fvu_from(xh, Who), alive, s_auc, w_auc,
                   sum(p_.numel() for p_ in m.parameters()),
                   {"pred_l0_per_segment": l0p, "l1_coef": l1})


        out["corpora"][tag] = rows

    print("\n===== FVU at matched realised coefficients per segment =====", flush=True)
    for tag, rows in out["corpora"].items():
        print(f"\n  [{tag}]", flush=True)
        for r in sorted(rows, key=lambda r: (r["arm"], r["coeff_per_segment"])):
            print(f"    {r['arm']:<14} {r['coeff_per_segment']:6.2f}/seg  "
                  f"FVU {r['fvu']:.4f}  segAUC {r['segment_auc']:.3f}  "
                  f"winAUC {r['window_auc']:.3f}", flush=True)
    return out


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-1.5B-Instruct", layer: int = -1, k_seg: int = 36,
         t: int = 12, n_docs: int = 500, d_sae: int = 4096, steps: int = 2500,
         budgets: str = "1,2,4,8,16", lr: float = 3e-4, general_frac: float = 0.3,
         batch_win: int = 32, l1_grid: str = "1e-2,3e-3,1e-3,3e-4,1e-4",
         nce_alpha: float = 1.0, nce_shift: int = 1,
         arms: str = "sae,txc,tsae_paper", tag: str = ""):
    import json
    r = bench4.remote(model, layer, k_seg, t, n_docs, d_sae, steps,
                      [int(x) for x in budgets.split(",")], lr, general_frac, batch_win,
                      [float(x) for x in l1_grid.split(",")], nce_alpha, nce_shift,
                      [x.strip() for x in arms.split(",")])
    outdir = ROOT / "results" / "dict_bench"
    outdir.mkdir(parents=True, exist_ok=True)
    name = f"bench4{tag}.json"
    (outdir / name).write_text(json.dumps(r, indent=2))
    print("[saved]", outdir / name)
