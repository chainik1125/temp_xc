"""Episode-disjoint, equal-width four-method k=5/10 comparison.

Reuses the original generator and architecture/training functions. Each worker
sees only one GPU; the supervisor can run many independent workers within it.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "temporal_crosscoders")]

GRID = {h: {"q": 11, "width": 2048, "windows": [1, 2, 3, 4, 5, 6, 10]} for h in (1, 2, 3)}
ARCH_OFFSET = {"sae": 100_000, "txc": 200_000, "tsae": 300_000, "tfa": 400_000}


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def save_torch(path, data):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, tmp)
    tmp.replace(path)


def torch_setup():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("high")
    return torch


def prepare(args):
    torch = torch_setup()
    from src.v6_colored_sources.polynomial_clock import (
        PolynomialClockConfig, generate_polynomial_clock_dataset,
        split_episode_indices, subset_polynomial_clock_dataset,
        sample_one_anchor_per_episode,
    )
    from src.v6_colored_sources.polynomial_clock_oracles import interpolation_oracle
    checks = []
    for h, grid in GRID.items():
        for seed in args.seeds:
            cfg = PolynomialClockConfig(h=h, q=grid["q"], d=64, sigma=.1,
                                        n_seq=3*args.episodes, T_chain=16, seed=seed)
            data = generate_polynomial_clock_dataset(cfg)
            ids = split_episode_indices(cfg.n_seq, n_rep_train=args.episodes,
                                        n_probe_train=args.episodes, n_probe_val=args.episodes,
                                        seed=seed+10_000)
            sets = {k:set(v.tolist()) for k,v in ids.items()}
            assert all(sets[a].isdisjoint(sets[b]) for a in sets for b in sets if a!=b)
            parts = {k:subset_polynomial_clock_dataset(data,v) for k,v in ids.items()}
            anchors, oracle = {}, {}
            for W in grid["windows"]:
                anchors[W] = {}
                for split, offset in [("probe_train",20_000),("probe_val",30_000)]:
                    a = sample_one_anchor_per_episode(parts[split],W,max_samples=args.episodes,
                                                     device=torch.device("cpu"),seed=seed+offset+W)
                    assert len(set(a["episode_id"].tolist())) == args.episodes
                    anchors[W][split] = a
                a = anchors[W]["probe_val"]
                if W >= h+1:
                    # Translating consecutive evaluation points preserves the leading coefficient.
                    pos=a["t_start"][:,None]+torch.arange(h+1)[None,:]
                    symbolic=parts["probe_val"]["Q"][a["chain_idx"][:,None],pos]
                    pred=interpolation_oracle(symbolic,cfg.q)
                    assert torch.equal(pred,a["Y"]), (h,seed,W)
                    x=parts["probe_val"]["x"][a["chain_idx"][:,None],pos]
                    observed=(x@data["alphabet"].T).argmax(-1)
                    acc=float((interpolation_oracle(observed,cfg.q)==a["Y"]).float().mean())
                    oracle[W]={"symbolic_accuracy":1.0,"observation_accuracy":acc}
                else:
                    oracle[W]={"symbolic_accuracy":1/cfg.q,"observation_accuracy":None}
            bundle={"config":asdict(cfg),"parts":parts,"anchors":anchors,"oracle":oracle}
            p=args.output/"data"/f"h{h}_seed{seed}.pt"
            save_torch(p,bundle)
            checks.append({"h":h,"seed":seed,"source":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),
                           "split_ids":{k:v.tolist() for k,v in ids.items()},"oracle":oracle})
    save_json(args.output/"split_audit.json",{"created":now(),"episode_disjoint":True,"datasets":checks})
    print("prepared",len(checks),"datasets with disjoint splits and exact symbolic-oracle checks",flush=True)


def encode(model, arch, part, anchors, W, device, batch=128):
    import torch
    features=[]; squared_error=0.; squared_input=0.; active=0.; novel_active=0.
    with torch.inference_mode():
        for start in range(0,len(anchors["Y"]),batch):
            ids=anchors["chain_idx"][start:start+batch]
            pos=anchors["t_start"][start:start+batch,None]+torch.arange(W)[None,:]
            x=part["x"][ids[:,None],pos].to(device=device,dtype=torch.float32)
            if arch=="sae":
                _,recon,z=model(x.reshape(-1,x.shape[-1]));recon=recon.reshape_as(x)
                z=z.reshape(len(x),-1)
            elif arch=="txc":
                _,recon,z=model(x)
            else:
                recon,result=model(x)
                z=(result["novel_codes"]+result["pred_codes"]).reshape(len(x),-1)
                novel_active+=float(torch.count_nonzero(result["novel_codes"]))
            assert torch.isfinite(z).all() and torch.isfinite(recon).all()
            active+=float(torch.count_nonzero(z))
            squared_error+=float((recon-x).square().sum())
            squared_input+=float(x.square().sum())
            features.append(z.float().clone())
    n=len(anchors["Y"])
    return torch.cat(features),{"l0_per_window":active/n,"l0_per_token":active/(n*W),
                                "novel_l0_per_token":novel_active/(n*W) if arch in ("tsae","tfa") else None,
                                "reconstruction_nmse":squared_error/squared_input}


def fit_probe(X,y,V,v,q,steps,seed,save_path):
    """Same Adam logistic-probe recipe as _train_logistic_probe_split; retain weights."""
    import torch
    torch.manual_seed(seed)
    X=X.float();V=V.float();y=y.to(X.device);v=v.to(X.device)
    probe=torch.nn.Linear(X.shape[1],q).to(X.device)
    opt=torch.optim.Adam(probe.parameters(),lr=.01,weight_decay=1e-4)
    for _ in range(steps):
        idx=torch.randint(0,len(X),(min(256,len(X)),),device=X.device)
        loss=torch.nn.functional.cross_entropy(probe(X[idx]),y[idx])
        assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():
        tr=probe(X).argmax(-1);pred=probe(V).argmax(-1)
        recall=[float((pred[v==i]==i).float().mean()) for i in range(q) if (v==i).any()]
    save_torch(save_path,{"state_dict":{k:t.cpu() for k,t in probe.state_dict().items()},
                          "seed":seed,"n_steps":steps,"input_width":X.shape[1],"q":q,
                          "eval_predictions":pred.cpu(),"eval_labels":v.cpu()})
    return {"train_accuracy":float((tr==y).float().mean()),"val_accuracy":float((pred==v).float().mean()),
            "balanced_accuracy":sum(recall)/len(recall),"chance":1/q,"n_train":len(X),"n_val":len(V)}


def job(args):
    torch=torch_setup()
    from src.v6_colored_sources.data_adapter import ColoredSourceCache
    from src.v6_colored_sources.run_polynomial_clock import _train_topk_sae_local
    from src.v6_colored_sources.run_polynomial_clock_tfa_only import _train_txc_global_inline
    from src.v6_colored_sources.train_runner import TrainConfig,train_tsae,train_tfa
    spec=json.loads((args.output/"plan.json").read_text())["jobs"][args.job_index]
    name=spec["id"];directory=args.output/"jobs"/name;directory.mkdir(parents=True,exist_ok=True)
    if (directory/"result.json").exists():
        print("already complete",name,flush=True);return
    device=torch.device("cuda");assert torch.cuda.device_count()==1
    h,seed,k,arch,W=(spec[x] for x in ["h","seed","k","arch","W"])
    bundle=torch.load(args.output/"data"/f"h{h}_seed{seed}.pt",map_location="cpu",weights_only=False)
    cache=ColoredSourceCache(bundle["parts"]["representation_train"]["x"],device)
    model_seed=seed+ARCH_OFFSET[arch]+1000*h+10*W+k
    torch.manual_seed(model_seed);torch.cuda.manual_seed_all(model_seed)
    start=time.time();started=now();H=spec["width"]
    cfg=TrainConfig(n_steps=spec["steps"],batch_size=128,lr=.001,log_interval=1000)
    save_json(directory/"status.json",{"state":"training","started":started,"spec":spec,"pid":os.getpid()})
    common=dict(cache=cache,d=64,H=H,device=device)
    if arch=="sae":
        model=_train_topk_sae_local(**common,k=k,n_steps=cfg.n_steps,batch_size=128,lr=.001)
    elif arch=="txc":
        model=_train_txc_global_inline(**common,W=W,k_window=k,n_steps=cfg.n_steps,batch_size=128,lr=.001)
    elif arch=="tsae":
        model=train_tsae(**common,W=W,k=k,contrastive_weight=.1,train_cfg=cfg)
    else:
        model=train_tfa(**common,W=W,k=k,use_pos_encoding=True,train_cfg=cfg)
    model.eval();training_s=time.time()-start
    save_torch(directory/"model.pt",{"state_dict":{k:v.cpu() for k,v in model.state_dict().items()},
                                     "spec":spec,"model_seed":model_seed,"class":type(model).__module__+"."+type(model).__name__})
    save_json(directory/"status.json",{"state":"evaluating","started":started,"spec":spec,"pid":os.getpid(),"training_seconds":training_s})
    rows=[];single_token_row=None
    for window in spec["eval_windows"]:
        anchors=bundle["anchors"][window]
        X,_=encode(model,arch,bundle["parts"]["probe_train"],anchors["probe_train"],window,device)
        V,metrics=encode(model,arch,bundle["parts"]["probe_val"],anchors["probe_val"],window,device)
        pseed=seed+500_000+1000*h+10*window+k
        probe_file=f"probe_W{window}.pt"
        probe=fit_probe(X,anchors["probe_train"]["Y"],V,anchors["probe_val"]["Y"],spec["q"],
                        spec["probe_steps"],pseed,directory/probe_file)
        row=dict(h=h,q=spec["q"],W=window,k=k,arch=arch,seed=seed,width=H,steps=spec["steps"],
                 probe_steps=spec["probe_steps"],n_params=sum(p.numel() for p in model.parameters()),
                 model_seed=model_seed,probe_seed=pseed,**probe,**metrics,oracle=bundle["oracle"][window],
                 model_arch=arch,readout_window=window,probe_file=probe_file)
        if arch=="sae":
            if window==1:
                single_token_row=dict(row,readout="single_token")
            assert single_token_row is not None
            # Same single-token observations, same fitted probe, and same score
            # at every plotted W. These rows are one baseline, not independent fits.
            rows.append(dict(single_token_row,W=window,shared_single_token_baseline=True))
            rows.append(dict(row,arch="sae_stacked",readout="concatenated_token_codes"))
        else:
            rows.append(dict(row,readout="window_shared" if arch=="txc" else "concatenated_temporal_codes"))
        print(json.dumps(rows[-1]),flush=True)
        save_json(directory/"partial_results.json",rows)
        del X,V;torch.cuda.empty_cache()
    result={"spec":spec,"started":started,"completed":now(),"training_seconds":training_s,
            "elapsed_seconds":time.time()-start,"torch":torch.__version__,"cuda":torch.version.cuda,
            "device":torch.cuda.get_device_name(),"gpu_visibility":os.environ.get("CUDA_VISIBLE_DEVICES"),
            "float32_matmul_precision":torch.get_float32_matmul_precision(),"rows":rows}
    save_json(directory/"result.json",result)
    save_json(directory/"status.json",{"state":"complete","completed":now(),"spec":spec})


def make_plan(args):
    jobs=[]
    for seed in args.seeds:
        for h,grid in GRID.items():
            for k in (5,10):
                for arch in ARCH_OFFSET:
                    for W in ([0] if arch=="sae" else grid["windows"]):
                        if arch in ("tsae","tfa") and W<2:continue
                        jobs.append(dict(id=f"s{seed}_h{h}_{arch}_k{k}_W{W}",h=h,q=grid["q"],width=grid["width"],
                                         arch=arch,k=k,W=W,seed=seed,steps=args.steps,probe_steps=args.probe_steps,
                                         eval_windows=grid["windows"] if arch=="sae" else [W]))
    jobs.sort(key=lambda j:(j["seed"],j["W"],j["h"],j["k"],j["arch"]))
    p={"created":now(),"protocol":"episode_disjoint_fixed_q11_width2048_single_token_sae_v2","seeds":args.seeds,
       "episodes_per_split":args.episodes,"batch_size":128,"learning_rate":.001,"jobs":jobs,
       "interpretation":"Fixed q=11 and dictionary width=2048 across h. Same nominal k, not matched actual L0 or parameter count. Regular SAE reads one token; stacked SAE codes are a separate diagnostic. T-SAE/TFA read out predicted+novel codes.",
       "expected_primary_evaluation_rows":sum(len(j["eval_windows"]) for j in jobs),
       "expected_evaluation_rows":sum(len(j["eval_windows"])*(2 if j['arch']=='sae' else 1) for j in jobs)}
    save_json(args.output/"plan.json",p)
    print(len(jobs),"model jobs;",p["expected_evaluation_rows"],"evaluation rows",flush=True)


def supervise(args):
    plan=json.loads((args.output/"plan.json").read_text());pending=[]
    for i,j in enumerate(plan["jobs"]):
        if not (args.output/"jobs"/j["id"]/"result.json").exists():pending.append((i,j))
    save_json(args.output/"supervisor.json",{"state":"running","pid":os.getpid(),"started":now(),
                                           "workers":args.workers,"jobs":len(plan["jobs"]),"pending_at_start":len(pending)})
    def launch(pair):
        index,spec=pair;log=args.output/"logs"/(spec["id"]+".log");log.parent.mkdir(exist_ok=True)
        with log.open("a") as f:
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),"job","--output",str(args.output),"--job-index",str(index)],stdout=f,stderr=subprocess.STDOUT)
        return spec["id"],result.returncode
    failed=[];done=0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(launch,p) for p in pending]
        for future in as_completed(futures):
            name,code=future.result();done+=1
            if code:failed.append({"id":name,"returncode":code})
            print(now(),done,"/",len(pending),name,"exit",code,flush=True)
            save_json(args.output/"progress.json",{"updated":now(),"finished_this_launch":done,"total_this_launch":len(pending),"failed":failed})
    save_json(args.output/"supervisor.json",{"state":"failed" if failed else "complete","finished":now(),
                                           "pid":os.getpid(),"workers":args.workers,"failed":failed})
    if failed:raise SystemExit(1)


def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["prepare","plan","job","supervise"])
    p.add_argument("--output",type=Path,required=True);p.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    p.add_argument("--episodes",type=int,default=4096);p.add_argument("--steps",type=int,default=6000)
    p.add_argument("--probe-steps",type=int,default=1500);p.add_argument("--job-index",type=int,default=0)
    p.add_argument("--workers",type=int,default=16);a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    {"prepare":prepare,"plan":make_plan,"job":job,"supervise":supervise}[a.mode](a)


if __name__=="__main__":
    main()
