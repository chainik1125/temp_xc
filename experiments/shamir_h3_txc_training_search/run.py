"""Bounded TXC training search; selection uses validation, never held-out test.

The generator, nominal k=5, q=11, h=3, width=2048, and linear-probe
recipe are fixed. All search candidates, including unsuccessful ones, persist.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "temporal_crosscoders")]
from experiments.shamir_camera_ready_k_sweep.run import save_json, save_torch, fit_probe, torch_setup

WINDOWS = [3, 4, 5, 6, 10]
CHECKPOINTS = [6000, 12000, 24000]
RECIPES = [
    dict(name="baseline", lr=.001, schedule="constant", centered=False, data_init=False, early_aux=False),
    dict(name="low_lr", lr=.0003, schedule="constant", centered=False, data_init=False, early_aux=False),
    dict(name="high_lr", lr=.003, schedule="constant", centered=False, data_init=False, early_aux=False),
    dict(name="cosine", lr=.001, schedule="cosine", centered=False, data_init=False, early_aux=False),
    dict(name="cosine_high_lr", lr=.003, schedule="cosine", centered=False, data_init=False, early_aux=False),
    dict(name="centered_cosine", lr=.001, schedule="cosine", centered=True, data_init=False, early_aux=False),
    dict(name="data_init_centered", lr=.001, schedule="cosine", centered=True, data_init=True, early_aux=False),
    dict(name="early_aux_data_init", lr=.001, schedule="cosine", centered=True, data_init=True, early_aux=True),
]


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root, episodes=4096, seeds=(0, 1, 2)):
    import torch
    from src.v6_colored_sources.polynomial_clock import PolynomialClockConfig, generate_polynomial_clock_dataset, subset_polynomial_clock_dataset, sample_one_anchor_per_episode
    from src.v6_colored_sources.polynomial_clock_oracles import interpolation_oracle
    audits = []
    names = ["representation_train", "probe_train", "validation", "test"]
    for seed in seeds:
        cfg = PolynomialClockConfig(h=3, q=11, d=64, sigma=.1, n_seq=4*episodes, T_chain=16, seed=seed)
        data = generate_polynomial_clock_dataset(cfg)
        perm = torch.randperm(cfg.n_seq, generator=torch.Generator().manual_seed(seed+10_000))
        ids = {name: perm[i*episodes:(i+1)*episodes] for i, name in enumerate(names)}
        assert len(set(torch.cat(list(ids.values())).tolist())) == cfg.n_seq
        parts = {name: subset_polynomial_clock_dataset(data, indices) for name, indices in ids.items()}
        anchors = {W: {} for W in WINDOWS}
        oracles = {}
        for W in WINDOWS:
            for j, name in enumerate(names[1:]):
                a = sample_one_anchor_per_episode(parts[name], W, max_samples=episodes, device=torch.device("cpu"), seed=seed+20_000+j*10_000+W)
                assert len(set(a["episode_id"].tolist())) == episodes
                anchors[W][name] = a
            a = anchors[W]["validation"]
            if W >= 4:
                pos = a["t_start"][:, None] + torch.arange(4)[None, :]
                symbolic = parts["validation"]["Q"][a["chain_idx"][:, None], pos]
                assert torch.equal(interpolation_oracle(symbolic, 11), a["Y"])
                x = parts["validation"]["x"][a["chain_idx"][:, None], pos]
                observed = (x @ data["alphabet"].T).argmax(-1)
                oracles[W] = {"symbolic_accuracy": 1., "observation_accuracy": float((interpolation_oracle(observed, 11)==a["Y"]).float().mean())}
            else:
                oracles[W] = {"information_theoretic_accuracy": 1/11}
        training = dict(config=asdict(cfg), parts={k:v for k,v in parts.items() if k!="test"},
                        anchors={W:{k:v for k,v in a.items() if k!="test"} for W,a in anchors.items()}, validation_oracles=oracles)
        heldout = dict(part=parts["test"], anchors={W:a["test"] for W,a in anchors.items()})
        train_path = root/"data"/f"tuning_seed{seed}.pt"
        test_path = root/"data"/f"heldout_seed{seed}.pt"
        save_torch(train_path, training); save_torch(test_path, heldout)
        audits.append(dict(seed=seed, config=asdict(cfg), split_ids={k:v.tolist() for k,v in ids.items()},
                           tuning_sha256=digest(train_path), heldout_sha256=digest(test_path), validation_oracles=oracles))
    save_json(root/"split_audit.json", dict(created=now(), episode_disjoint=True, test_file_separate=True, datasets=audits))


def create_model(recipe, W, device):
    from temporal_crosscoders.han_arch.txc_bare_antidead import TXCBareAntidead
    class CenteredTXC(TXCBareAntidead):
        def _pre_activation(self, x):
            return super()._pre_activation(x-self.b_dec)
    cls = CenteredTXC if recipe["centered"] else TXCBareAntidead
    kwargs = dict(dead_threshold_tokens=500*128*W, aux_k=128) if recipe["early_aux"] else {}
    return cls(d_in=64, d_sae=2048, T=W, k=5, **kwargs).to(device)


def encode(model, part, anchor, W, device):
    import torch
    zs = []; raw = []; error = total = active = 0.
    fired = torch.zeros(2048, dtype=torch.bool, device=device)
    with torch.inference_mode():
        for i in range(0, len(anchor["Y"]), 128):
            idx = anchor["chain_idx"][i:i+128]
            pos = anchor["t_start"][i:i+128, None]+torch.arange(W)[None, :]
            x = part["x"][idx[:, None], pos].to(device=device, dtype=torch.float32)
            z = model.encode(x); recon = model.decode(z)
            assert torch.isfinite(z).all() and torch.isfinite(recon).all()
            zs.append(z); raw.append(x.flatten(1)); active += float((z>0).sum())
            fired |= (z>0).any(0); error += float((x-recon).square().sum()); total += float(x.square().sum())
    return torch.cat(zs), torch.cat(raw), dict(l0_per_window=active/len(anchor["Y"]), reconstruction_nmse=error/total, alive_fraction=float(fired.float().mean()))


def evaluate(model, bundle, spec, directory, steps, heldout=None):
    import torch
    from src.v6_colored_sources.polynomial_clock_oracles import interpolation_oracle
    W = spec["W"]; device = next(model.parameters()).device
    a = bundle["anchors"][W]
    X, raw_X, _ = encode(model, bundle["parts"]["probe_train"], a["probe_train"], W, device)
    split = "test" if heldout is not None else "validation"
    target_part = heldout["part"] if heldout is not None else bundle["parts"]["validation"]
    target_anchor = heldout["anchors"][W] if heldout is not None else a["validation"]
    V, raw_V, metrics = encode(model, target_part, target_anchor, W, device)
    pseed = spec["seed"]+503_000+10*W+5
    y = a["probe_train"]["Y"]; v = target_anchor["Y"]
    probe = fit_probe(X, y, V, v, 11, spec["probe_steps"], pseed, directory/f"probe_{split}_step{steps}.pt")
    row = dict(spec=spec, steps=steps, split=split, accuracy=probe.pop("val_accuracy"), **probe, **metrics,
               n_params=sum(p.numel() for p in model.parameters()), evaluated=now())
    assert metrics["l0_per_window"] <= 5.000001
    if heldout is not None:
        raw = fit_probe(raw_X, y, raw_V, v, 11, spec["probe_steps"], pseed, directory/"raw_linear_probe.pt")
        row["raw_linear_control"] = raw
        if W >= 4:
            pos = target_anchor["t_start"][:, None]+torch.arange(4)[None, :]
            symbolic = target_part["Q"][target_anchor["chain_idx"][:, None], pos]
            assert torch.equal(interpolation_oracle(symbolic, 11), v)
            xs = target_part["x"][target_anchor["chain_idx"][:, None], pos]
            symbols = (xs @ target_part["alphabet"].T).argmax(-1)
            row["oracle"] = dict(symbolic_accuracy=1., observation_accuracy=float((interpolation_oracle(symbols, 11)==v).float().mean()))
        else:
            row["oracle"] = dict(information_theoretic_accuracy=1/11)
        if W == 10:
            perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(pseed+71))
            row["shuffled_label_control"] = fit_probe(X, y[perm], V, v, 11, spec["probe_steps"], pseed, directory/"shuffled_label_probe.pt")
    save_json(directory/f"metrics_{split}_step{steps}.json", row)
    return row


def job(root, spec):
    torch = torch_setup()
    from src.v6_colored_sources.data_adapter import ColoredSourceCache
    directory = root/"jobs"/spec["id"]; directory.mkdir(parents=True, exist_ok=True)
    if (directory/"result.json").exists(): return
    if spec["phase"] == "confirm":
        assert (root/"selection.json").exists(), "Test access requires frozen selection."
    started = time.time(); torch.manual_seed(spec["seed"]+203_000+10*spec["W"]+5)
    torch.cuda.manual_seed_all(spec["seed"]+203_000+10*spec["W"]+5)
    assert torch.cuda.device_count() == 1
    device = torch.device("cuda")
    bundle = torch.load(root/"data"/f"tuning_seed{spec['seed']}.pt", map_location="cpu", weights_only=False)
    cache = ColoredSourceCache(bundle["parts"]["representation_train"]["x"], device)
    recipe = spec["recipe"]; model = create_model(recipe, spec["W"], device)
    init_x = cache.sample_windows(256, spec["W"])
    model.init_b_dec_geometric_median(init_x)
    if recipe["data_init"]:
        with torch.no_grad():
            atoms = cache.sample_windows(2048, spec["W"])-model.b_dec
            atoms /= atoms.norm(dim=(1, 2), keepdim=True).clamp_min(1e-8)
            model.W_dec.copy_(atoms); model.W_enc.copy_(atoms.permute(1, 2, 0))
    opt = torch.optim.Adam(model.parameters(), lr=recipe["lr"])
    checkpoints = spec["checkpoints"]; rows = []
    for step in range(1, max(checkpoints)+1):
        lr = recipe["lr"]
        if recipe["schedule"] == "cosine":
            warmup = 300
            factor = step/warmup if step<=warmup else .1+.9*.5*(1+math.cos(math.pi*(step-warmup)/(24000-warmup)))
            lr *= factor
        for group in opt.param_groups: group["lr"] = lr
        x = cache.sample_windows(spec["batch_size"], spec["W"])
        loss, _, _ = model(x)
        assert torch.isfinite(loss)
        opt.zero_grad(); loss.backward(); model.remove_gradient_parallel_to_decoder()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step(); model._normalize_decoder()
        if step%1000==0 or step==1:
            status = dict(state="training", spec=spec, step=step, loss=float(loss.detach()), lr=lr,
                          elapsed_seconds=time.time()-started, updated=now(), pid=os.getpid(), dead_count=int(model.last_dead_count))
            save_json(directory/"status.json", status); print(json.dumps(status), flush=True)
        if step in checkpoints:
            save_torch(directory/f"model_step{step}.pt", dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()}, spec=spec, steps=step))
            # Checkpoint probing must not change the representation-training RNG.
            cpu_rng = torch.get_rng_state(); gpu_rng = torch.cuda.get_rng_state()
            model.eval()
            heldout = torch.load(root/"data"/f"heldout_seed{spec['seed']}.pt", map_location="cpu", weights_only=False) if spec["phase"]=="confirm" else None
            rows.append(evaluate(model, bundle, spec, directory, step, heldout))
            model.train(); torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state(gpu_rng)
    save_json(directory/"result.json", dict(spec=spec, rows=rows, elapsed_seconds=time.time()-started, completed=now(),
                                            torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(), gpu_visibility=os.environ.get("CUDA_VISIBLE_DEVICES")))
    save_json(directory/"status.json", dict(state="complete", updated=now(), spec=spec))


def make_spec(phase, recipe, W, seed, checkpoints):
    return dict(id=f"{phase}_{recipe['name']}_s{seed}_W{W}_n{max(checkpoints)}", phase=phase,
                recipe=recipe, W=W, seed=seed, checkpoints=checkpoints, h=3, q=11, k=5, width=2048,
                batch_size=128, probe_steps=1500)


def run_pool(root, specs, workers, deadline):
    def run_one(spec):
        directory = root/"jobs"/spec["id"]
        if (directory/"result.json").exists(): return spec["id"], 0
        remain = deadline-time.time()
        if remain <= 0: return spec["id"], "budget_exhausted"
        spec_path = root/"specs"/f"{spec['id']}.json"; save_json(spec_path, spec)
        log = root/"logs"/f"{spec['id']}.log"; log.parent.mkdir(exist_ok=True)
        with log.open("a") as out:
            proc = subprocess.Popen([sys.executable, __file__, "job", "--output", str(root), "--spec", str(spec_path)], stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
            try: return spec["id"], proc.wait(timeout=remain)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                return spec["id"], "budget_exhausted"
    outcomes = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, spec) for spec in specs]
        for fut in as_completed(futures):
            outcomes.append(fut.result())
            save_json(root/"progress.json", dict(updated=now(), phase=specs[0]["phase"], finished=len(outcomes), total=len(specs), outcomes=outcomes, budget_remaining_seconds=max(0, deadline-time.time())))
            print(outcomes[-1], flush=True)
    if any(code!=0 for _,code in outcomes): raise RuntimeError(f"Incomplete stage: {outcomes}")


def summarize(root, specs):
    rows = [row for spec in specs for row in json.loads((root/"jobs"/spec["id"]/"result.json").read_text())["rows"]]
    groups = {}
    for row in rows: groups.setdefault((row["spec"]["recipe"]["name"], row["steps"], row["spec"]["W"]), []).append(row)
    summary = []
    for (recipe,steps,W), group in sorted(groups.items()):
        values = [r["accuracy"] for r in group]
        summary.append(dict(recipe=recipe, steps=steps, W=W, n_seeds=len(group), seeds=[r["spec"]["seed"] for r in group],
                            accuracy_mean=statistics.fmean(values), accuracy_sd=statistics.stdev(values) if len(values)>1 else 0.,
                            alive_fraction_mean=statistics.fmean(r["alive_fraction"] for r in group),
                            reconstruction_nmse_mean=statistics.fmean(r["reconstruction_nmse"] for r in group)))
    save_json(root/"confirmation_summary.json", dict(created=now(), rows=rows, summary=summary,
                                                   interpretation="Report all windows and seeds. No monotonicity constraint or smoothing is applied."))


def supervise(args):
    root = args.output; root.mkdir(parents=True, exist_ok=True)
    started = time.time(); deadline = started+args.budget_seconds
    plan = dict(created=now(), protocol="four_episode_splits_validation_selection_then_frozen_test", h=3, q=11, k=5, width=2048,
                generator=dict(d=64, sigma=.1, T_chain=16), episodes_per_split=4096, seeds=[0,1,2], windows=WINDOWS,
                screen_windows=[6,10], screen_seed=0, checkpoints=CHECKPOINTS, recipes=RECIPES, workers=args.workers,
                budget_seconds=args.budget_seconds, selection_objective="Mean validation accuracy at W=6 and W=10. Tie-break: fewer steps, then recipe name.",
                controls="W=3 negative control, raw-observation linear probe, shuffled-label W=10 probe, symbolic/observation interpolation oracle.",
                caveats=["Search uses newly generated four-way data, not the old three-way result table; baseline is rerun on identical splits.",
                         "Nominal k and width remain fixed. Parameter count increases with W. Training budget differs by candidate.",
                         "Screening uses seed 0; confirmation uses all three seeds without further tuning based on test."])
    save_json(root/"plan.json", plan)
    if not (root/"split_audit.json").exists():
        torch_setup(); prepare(root)
    screen = [make_spec("screen", recipe, W, 0, CHECKPOINTS) for recipe in RECIPES for W in [6,10]]
    run_pool(root, screen, args.workers, deadline)
    scored = []
    for recipe in RECIPES:
        results = [json.loads((root/"jobs"/s["id"]/"result.json").read_text()) for s in screen if s["recipe"]["name"]==recipe["name"]]
        for steps in CHECKPOINTS:
            rows = [r for result in results for r in result["rows"] if r["steps"]==steps]
            assert len(rows)==2 and all(r["split"]=="validation" for r in rows)
            scored.append(dict(recipe=recipe, steps=steps, objective=statistics.fmean(r["accuracy"] for r in rows), windows={r["spec"]["W"]:r["accuracy"] for r in rows}))
    scored.sort(key=lambda r:(-r["objective"], r["steps"], r["recipe"]["name"]))
    selected = scored[0]
    save_json(root/"selection.json", dict(frozen_at=now(), test_access_before_freeze=False, selected=selected, all_candidates=scored))
    choices = [(RECIPES[0], 6000)]
    if (selected["recipe"]["name"], selected["steps"]) != ("baseline",6000): choices.append((selected["recipe"], selected["steps"]))
    confirm = [make_spec("confirm", recipe, W, seed, [steps]) for recipe,steps in choices for seed in [0,1,2] for W in WINDOWS]
    save_json(root/"confirmation_plan.json", dict(created=now(), selection_sha256=digest(root/"selection.json"), jobs=confirm))
    run_pool(root, confirm, args.workers, deadline)
    summarize(root, confirm)
    save_json(root/"completion.json", dict(state="complete", completed=now(), elapsed_seconds=time.time()-started, selected=selected))


def smoke(root):
    torch_setup(); prepare(root, episodes=128, seeds=(99,))
    # Exercise default and centered/data-init/early-AuxK paths, and test-access gate.
    save_json(root/"selection.json", dict(smoke_only=True, frozen_at=now()))
    specs = [make_spec("confirm", recipe, W, 99, [20]) for recipe,W in [(RECIPES[0],3),(RECIPES[-1],10)]]
    for spec in specs:
        spec["probe_steps"]=20
        p=root/"specs"/f"{spec['id']}.json"; save_json(p,spec)
        subprocess.run([sys.executable,__file__,"job","--output",str(root),"--spec",str(p)],check=True)
    save_json(root/"smoke_checks.json", dict(passed=True, split_audit_sha256=digest(root/"split_audit.json"), jobs=[s["id"] for s in specs]))


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("mode",choices=["supervise","job","smoke"])
    p.add_argument("--output",type=Path,required=True); p.add_argument("--spec",type=Path)
    p.add_argument("--workers",type=int,default=16); p.add_argument("--budget-seconds",type=int,default=3600)
    args=p.parse_args()
    if args.mode=="job": job(args.output,json.loads(args.spec.read_text()))
    elif args.mode=="smoke": smoke(args.output)
    else: supervise(args)
