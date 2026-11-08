"""Correct the Shamir T-SAE arm using the per-token published-method port.

Train once per h/seed on adjacent-token pairs. Sweep only the probe readout
span W; retain both a shared single-token baseline and stacked-code probes.
All data, anchors and probe settings come from the frozen q11 comparison.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path('/data/users/dmitry/shamir_camera_ready_q11_20261106')
SOURCE = BASE / 'source'
sys.path[:0] = [str(SOURCE), str(SOURCE/'src')]
spec = importlib.util.spec_from_file_location('baseline', SOURCE/'experiments/shamir_camera_ready_k_sweep/run.py')
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
from tsae_bhalla import TSAEPaper

WINDOWS = [1,2,3,4,5,6,10]


def encode(model, part, anchors, W, device):
    import torch
    features=[]; active=error=energy=0.
    with torch.inference_mode():
        for start in range(0,len(anchors['Y']),128):
            ids=anchors['chain_idx'][start:start+128]
            pos=anchors['t_start'][start:start+128,None]+torch.arange(W)[None,:]
            x=part['x'][ids[:,None],pos].to(device=device,dtype=torch.float32)
            z=model.encode(x)
            # Threshold inference must encode each token independently of the
            # other tokens supplied in the same call.
            if start==0 and W>1:
                isolated=model.encode(x[:,0])
                torch.testing.assert_close(z[:,0],isolated,atol=2e-5,rtol=2e-5)
            recon=model.decode(z)
            assert torch.isfinite(z).all()
            active+=float(torch.count_nonzero(z));error+=float((recon-x).square().sum())
            energy+=float(x.square().sum());features.append(z.reshape(len(x),-1).float().clone())
    n=len(anchors['Y'])
    return torch.cat(features),dict(l0_per_window=active/n,l0_per_token=active/(n*W),reconstruction_nmse=error/energy)


def job(args):
    torch=baseline.torch_setup()
    from src.v6_colored_sources.data_adapter import ColoredSourceCache
    assert torch.cuda.device_count()==1
    device=torch.device('cuda');h=args.h;seed=args.seed
    path=args.output/f'jobs/h{h}_s{seed}'
    if (path/'result.json').exists():return
    path.mkdir(parents=True,exist_ok=True)
    started=baseline.now();start=time.time()
    data_path=BASE/f'full/data/h{h}_seed{seed}.pt'
    data=torch.load(data_path,map_location='cpu',weights_only=False)
    cache=ColoredSourceCache(data['parts']['representation_train']['x'],device)
    model_seed=600_000+1000*h+seed+5
    torch.manual_seed(model_seed);torch.cuda.manual_seed_all(model_seed)
    model=TSAEPaper(d_in=64,d_sae=2048,k_pos=5).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.999))
    model.train()
    for step in range(args.steps):
        x=cache.sample_windows(128,2).to(device).float()
        loss,metrics=model.train_step(x)
        assert torch.isfinite(loss)
        optimizer.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step();model.post_step()
        if step%1000==0 or step==args.steps-1:
            baseline.save_json(path/'status.json',dict(state='training',step=step+1,loss=float(loss.detach()),updated=baseline.now()))
            print(h,seed,step+1,float(loss.detach()),flush=True)
    model.eval()
    assert float(model.threshold)>=0, 'Threshold inference must be active'
    training_seconds=time.time()-start
    baseline.save_torch(path/'model.pt',dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},
        class_name='TSAEPaper',model_seed=model_seed,h=h,seed=seed,q=11,k=5,width=2048,steps=args.steps,
        training_context=2,shared_across_readout_windows=True))
    rows=[];single=None
    windows=[1,3] if args.smoke else WINDOWS
    for W in windows:
        anchors=data['anchors'][W]
        X,_=encode(model,data['parts']['probe_train'],anchors['probe_train'],W,device)
        V,metrics=encode(model,data['parts']['probe_val'],anchors['probe_val'],W,device)
        pseed=seed+500_000+1000*h+10*W+5
        probe_steps=5 if args.smoke else 1500
        probe_file=f'probe_W{W}.pt'
        probe=baseline.fit_probe(X,anchors['probe_train']['Y'],V,anchors['probe_val']['Y'],11,probe_steps,pseed,path/probe_file)
        row=dict(h=h,q=11,W=W,k=5,arch='tsae_paper_stacked',seed=seed,width=2048,steps=args.steps,
            probe_steps=probe_steps,n_params=sum(p.numel() for p in model.parameters()),model_seed=model_seed,
            probe_seed=pseed,**probe,**metrics,oracle=data['oracle'][W],model_arch='tsae_paper',
            readout_window=W,probe_file=probe_file,readout='concatenated_token_codes',
            shared_model_across_windows=True,training_context=2,job_id=f'h{h}_s{seed}')
        if W==1:single=dict(row,arch='tsae_paper_single',readout='single_token',shared_single_token_baseline=True)
        rows.extend([dict(single,W=W),row])
        baseline.save_json(path/'partial_results.json',rows)
        print(h,seed,W,probe['val_accuracy'],flush=True)
        del X,V;torch.cuda.empty_cache()
    baseline.save_json(path/'result.json',dict(started=started,completed=baseline.now(),h=h,seed=seed,
        training_seconds=training_seconds,elapsed_seconds=time.time()-start,data_source=str(data_path),
        data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),threshold=float(model.threshold),
        source_sha256=hashlib.sha256(Path(__file__).with_name('tsae_bhalla.py').read_bytes()).hexdigest(),
        episode_disjoint=True,tokenwise_inference_checked=True,rows=rows))
    baseline.save_json(path/'status.json',dict(state='complete',completed=baseline.now()))


def supervise(args):
    args.output.mkdir(parents=True,exist_ok=True)
    jobs=[(h,s) for h in [1,2] for s in [0,1,2]]
    baseline.save_json(args.output/'plan.json',dict(created=baseline.now(),jobs=jobs,steps=args.steps,
        q=11,k=5,width=2048,batch_size=128,lr=.001,training_context=2,readout_windows=WINDOWS,
        architecture='per-token Matryoshka BatchTopK + high-level adjacent-token InfoNCE + AuxK',
        contrastive_alpha=1.0,auxk_alpha=1/32,h_frac=.20,
        instruction='One fixed model per h/seed; only the probe input span changes with W.',
        source_note='Replaces mislabeled han_tsae attention arm; original files preserved.'))
    def launch(h,seed):
        log=args.output/f'h{h}_s{seed}.log'
        with log.open('w') as f:
            rc=subprocess.run([sys.executable,__file__,'job','--output',str(args.output),'--h',str(h),
                '--seed',str(seed),'--steps',str(args.steps)],stdout=f,stderr=subprocess.STDOUT).returncode
        return dict(h=h,seed=seed,returncode=rc)
    complete=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for future in as_completed([pool.submit(launch,h,s) for h,s in jobs]):
            complete.append(future.result())
            baseline.save_json(args.output/'progress.json',dict(completed=complete,updated=baseline.now()))
    assert all(x['returncode']==0 for x in complete), complete
    rows=[]
    for h,s in jobs:rows.extend(json.loads((args.output/f'jobs/h{h}_s{s}/result.json').read_text())['rows'])
    assert len(rows)==84
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (args.output/'cells.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    baseline.save_json(args.output/'completion.json',dict(state='complete',models=6,rows=84,completed=baseline.now()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['job','supervise']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--h',type=int,default=1);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--steps',type=int,default=6000);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    (job if a.mode=='job' else supervise)(a)
