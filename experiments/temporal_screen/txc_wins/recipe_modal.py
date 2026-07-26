"""Is every arm trained well enough for the comparison to mean anything?

WHY THIS EXISTS. The tSAE reference was run at the sprint's default recipe (lr 3e-4, 2000
steps) and reached FVU 0.49 at 8 coefficients per segment. Raising the step count to 6000
took it to 0.22 and raising the learning rate to 1e-3 took it to 0.18 -- so the first number
was measuring the optimiser, not the architecture, and any conclusion drawn from it about
the tSAE would have been wrong by a factor of 2.7. That is the same failure mode as last
sprint's realised-L0 collapse: a nominal configuration that looks fair and is not.

The fix is not to trust the default for the other two either. This sweeps learning rate and
step count for ALL THREE dictionaries on one cache and reports, for each, FVU and REALISED
coefficients per segment. Realised L0 is in the table because the learning rate is exactly
what moved it last sprint -- a 3x change in lr moved a crosscoder's realised spend by 10.6x
with nothing in the nominal config to show for it -- so a recipe that wins on FVU while
quietly spending a different budget is not a better recipe, it is a different experiment.

Reading AUC for the task's class label is reported alongside, because the recipe that
reconstructs best is not automatically the one that finds the factor, and if those come
apart the benchmark has to say which it is matching on.

    modal run experiments/temporal_screen/txc_wins/recipe_modal.py --task phase11
"""
import pathlib
import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

app = modal.App("txcwins-recipe")
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers", "accelerate", "numpy")
    .add_local_dir(str(ROOT / "src"), "/work/src")
    .add_local_dir(str(ROOT / "temporal_crosscoders"), "/work/temporal_crosscoders")
    .add_local_dir(str(_here.parent), "/work/txc_wins")
)


@app.function(gpu="A10G", image=image, timeout=21600)
def recipe(task: str, model_id: str, layer: int, k_seg: int, n_train: int, d_sae: int,
           k: int, batch_win: int, lrs: list, step_grid: list, seed: int):
    import sys
    sys.path.insert(0, "/work")
    import random
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.bench.architectures.topk_sae import TopKSAE
    from src.bench.architectures.crosscoder import TemporalCrosscoder
    from temporal_crosscoders.han_tsae import TemporalSAE
    from txc_wins.tasks import TASKS

    make_pair = TASKS[task](k_seg)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    dev = model.device
    layers_ = model.model.layers
    L = layer if layer >= 0 else len(layers_) // 2
    d = model.config.hidden_size
    T = k_seg
    rng = random.Random(seed)
    cap = {}

    def cap_hook(_m, _i, out):
        cap["h"] = (out[0] if isinstance(out, tuple) else out).detach()

    def build(carrier, sents):
        text, spans = carrier, []
        for j, s in enumerate(sents):
            if j:
                text += " "
            spans.append((len(text), len(text) + len(s)))
            text += s
        return text, spans

    def capture(text, spans):
        e = tok(text, return_tensors="pt", return_offsets_mapping=True)
        offs = e["offset_mapping"][0].tolist()
        ts = []
        for (a, b) in spans:
            idx = [i for i, (s0, s1) in enumerate(offs)
                   if s0 >= a and s1 <= b and s1 > s0]
            ts.append((idx[0], idx[-1]) if idx else (0, 0))
        h = layers_[L].register_forward_hook(cap_hook)
        with torch.no_grad():
            model(e["input_ids"].to(dev))
        h.remove()
        hh = cap["h"][0].float()
        return torch.stack([hh[a:b + 1].mean(0) for a, b in ts])

    X, y = [], []
    for i in range(n_train):
        p = make_pair(rng)
        cls = rng.randint(0, 1)
        X.append(capture(*build(p[2], p[0] if cls else p[1])))
        y.append(cls)
        if (i + 1) % 250 == 0:
            print(f"   [cache] {i+1}/{n_train}", flush=True)
    Xt = torch.stack(X).to(dev)
    yt = torch.tensor(y, device=dev)
    mu, sd = Xt.mean((0, 1), keepdim=True), Xt.std() + 1e-6
    Xn = (Xt - mu) / sd

    n_hold = max(int(0.15 * Xn.shape[0]), 32)
    Wtr, Who = Xn[:-n_hold], Xn[-n_hold:]
    ytr, yho = yt[:-n_hold], yt[-n_hold:]
    flat_tr, flat_ho = Wtr.reshape(-1, d), Who.reshape(-1, d)
    denom = float(flat_ho.var(0).sum())
    print(f"[cache] train {tuple(Wtr.shape)} holdout {tuple(Who.shape)}  "
          f"denom {denom:.1f}", flush=True)

    def gen_win(bs):
        return Wtr[torch.randint(0, Wtr.shape[0], (bs,), device=dev)]

    def gen_flat(bs):
        return flat_tr[torch.randint(0, flat_tr.shape[0], (bs,), device=dev)]

    def auc_of(scores, yy):
        o = torch.argsort(scores)
        r = torch.empty_like(o, dtype=torch.float32)
        r[o] = torch.arange(len(scores), device=dev, dtype=torch.float32) + 1
        n1, n0 = float(yy.sum()), float((1 - yy).sum())
        if n1 == 0 or n0 == 0:
            return 0.5
        return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    def best_auc(Z, yy):
        a = torch.tensor([auc_of(Z[:, j], yy) for j in range(Z.shape[1])], device=dev)
        return float(0.5 + (a - 0.5).abs().max())

    rows = []
    for lr in lrs:
        for steps in step_grid:
            for arm in ("sae", "txc", "tsae_topk"):
                torch.manual_seed(0)
                if arm == "sae":
                    m = TopKSAE(d_in=d, d_sae=d_sae, k=k).to(dev)
                elif arm == "txc":
                    m = TemporalCrosscoder(d_in=d, d_sae=d_sae, T=T, k=k,
                                           activation="batchtopk").to(dev)
                    with torch.no_grad():
                        m._normalize_decoder()
                else:
                    m = TemporalSAE(dimin=d, width=d_sae, n_heads=8,
                                    sae_diff_type="topk", kval_topk=k,
                                    tied_weights=True, n_attn_layers=1,
                                    bottleneck_factor=64).to(dev)
                opt = torch.optim.Adam(m.parameters(), lr=lr)
                m.train()
                for _ in range(int(steps)):
                    if arm == "sae":
                        loss, _, _ = m(gen_flat(batch_win * T))
                    elif arm == "txc":
                        loss, _, _ = m(gen_win(batch_win))
                    else:
                        xb = gen_win(batch_win)
                        loss = (m(xb)[0] - xb).pow(2).sum(-1).mean()
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                    opt.step()
                    if arm in ("sae", "txc"):
                        m._normalize_decoder()
                m.eval()
                with torch.no_grad():
                    if arm == "sae":
                        Z = m.encode(flat_ho)
                        xh, ref = m.decode(Z), flat_ho
                        l0 = float((Z > 0).float().sum(-1).mean())
                        auc = best_auc(Z.reshape(-1, T, d_sae).mean(1), yho)
                    elif arm == "txc":
                        Z = m.encode(Who)
                        xh, ref = m.decode(Z), Who
                        l0 = float((Z != 0).float().sum(-1).mean()) / T
                        auc = best_auc(Z, yho)
                    else:
                        xh, info = m(Who)
                        ref = Who
                        Zn = info["novel_codes"]
                        l0 = float((Zn.reshape(-1, d_sae) > 0).float().sum(-1).mean())
                        auc = best_auc(Zn.mean(1), yho)
                    fvu = float(((xh - ref) ** 2).sum(-1).mean() / denom)
                rows.append({"arm": arm, "lr": float(lr), "steps": int(steps),
                             "coeff_per_segment": l0, "fvu": fvu, "reading_auc": auc})
                print(f"  {arm:<10} lr={lr:<8.1e} steps={steps:<6} "
                      f"coeff/seg {l0:6.2f}  FVU {fvu:.4f}  readAUC {auc:.3f}",
                      flush=True)

    print("\n===== best recipe per arm (by FVU, among runs that spent the budget) =====",
          flush=True)
    out_best = {}
    for arm in ("sae", "txc", "tsae_topk"):
        cand = [r for r in rows if r["arm"] == arm and r["coeff_per_segment"] >= 0.9 * k]
        if not cand:
            print(f"  {arm:<10} NO recipe spent its nominal budget", flush=True)
            continue
        b = min(cand, key=lambda r: r["fvu"])
        out_best[arm] = b
        print(f"  {arm:<10} lr={b['lr']:.1e} steps={b['steps']}  FVU {b['fvu']:.4f}  "
              f"coeff/seg {b['coeff_per_segment']:.2f}  readAUC {b['reading_auc']:.3f}",
              flush=True)
    return {"task": task, "k": k, "d_sae": d_sae, "k_seg": k_seg, "layer": int(L),
            "rows": rows, "best": out_best}


@app.local_entrypoint()
def main(task: str = "phase11", model: str = "Qwen/Qwen2.5-1.5B-Instruct",
         layer: int = -1, k_seg: int = 12, n_train: int = 800, d_sae: int = 4096,
         k: int = 8, batch_win: int = 32, lrs: str = "3e-4,1e-3,3e-3",
         step_grid: str = "2000,6000", seed: int = 31415, tag: str = ""):
    import json
    r = recipe.remote(task, model, layer, k_seg, n_train, d_sae, k, batch_win,
                      [float(x) for x in lrs.split(",")],
                      [int(x) for x in step_grid.split(",")], seed)
    outdir = ROOT / "results" / "txc_wins"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"recipe_{task}{tag}.json").write_text(json.dumps(r, indent=2))
    print("[saved]", outdir / f"recipe_{task}{tag}.json")
