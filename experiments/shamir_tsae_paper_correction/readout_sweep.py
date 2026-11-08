"""Probe-width sweep with fixed baseline checkpoints and nested readout spans.

TFA always encodes the same 10-token context. Token SAEs encode once. TXC alone
loads a different W-dependent model. All readouts end at the same anchor.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import run as correction
from run import baseline, BASE, TSAEPaper, WINDOWS
ROOT=Path(__file__).resolve().parents[1]


def load_model(arch,h,seed,W,device):
    import torch
    if arch=='txc':
        from temporal_crosscoders.han_arch.txc_bare_antidead import TXCBareAntidead
        model=TXCBareAntidead(d_in=64,d_sae=2048,T=W,k=5)
        path=BASE/f'full/jobs/s{seed}_h{h}_txc_k5_W{W}/model.pt'
    elif arch=='sae':
        from models import TopKSAE
        model=TopKSAE(d_in=64,d_sae=2048,k=5)
        path=BASE/f'full/jobs/s{seed}_h{h}_sae_k5_W0/model.pt'
    elif arch=='tfa':
        from src.bench.architectures._tfa_module import TemporalSAE
        model=TemporalSAE(dimin=64,width=2048,n_heads=4,sae_diff_type='topk',kval_topk=5,
            tied_weights=True,n_attn_layers=1,bottleneck_factor=1,use_pos_encoding=True,max_seq_len=64)
        path=BASE/f'full/jobs/s{seed}_h{h}_tfa_k5_W10/model.pt'
    else:
        model=TSAEPaper(d_in=64,d_sae=2048,k_pos=5)
        path=ROOT/f'full/jobs/h{h}_s{seed}/model.pt'
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    model.to(device).eval()
    return model,path,hashlib.sha256(path.read_bytes()).hexdigest()


def extract(model,arch,part,anchor,W,device):
    import torch
    code=[];sq_error=sq_input=0.
    with torch.inference_mode():
        for start in range(0,len(anchor['Y']),128):
            ids=anchor['chain_idx'][start:start+128]
            pos=anchor['t_start'][start:start+128,None]+torch.arange(W)[None,:]
            x=part['x'][ids[:,None],pos].to(device=device,dtype=torch.float32)
            if arch=='txc':_,recon,z=model(x)
            elif arch=='sae':
                _,recon,z=model(x.reshape(-1,64));recon=recon.reshape_as(x);z=z.reshape(len(x),W,2048)
            elif arch=='tfa':
                recon,r=model(x);z=r['novel_codes']+r['pred_codes']
            else:
                z=model.encode(x);recon=model.decode(z)
                torch.testing.assert_close(z[:,0],model.encode(x[:,0]),atol=2e-5,rtol=2e-5)
            assert torch.isfinite(z).all()
            code.append(z.clone());sq_error+=float((recon-x).square().sum());sq_input+=float(x.square().sum())
    return torch.cat(code),sq_error/sq_input


def job(a):
    torch=baseline.torch_setup();device=torch.device('cuda');assert torch.cuda.device_count()==1
    job_id=f'h{a.h}_s{a.seed}_{a.arch}'+(f'_W{a.W}' if a.arch=='txc' else '')
    out=a.output/'jobs'/job_id;out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():return
    started=baseline.now();model,path,sha=load_model(a.arch,a.h,a.seed,a.W,device)
    data=torch.load(BASE/f'full/data/h{a.h}_seed{a.seed}.pt',map_location='cpu',weights_only=False)
    anchors=data['anchors'][10]
    windows=[a.W] if a.arch=='txc' else WINDOWS
    codes={};nmse={}
    if a.arch!='txc':
        for split in ['probe_train','probe_val']:
            codes[split],nmse[split]=extract(model,a.arch,data['parts'][split],anchors[split],10,device)
    rows=[];single=None
    for W in windows:
        if a.arch=='txc':
            for split in ['probe_train','probe_val']:
                anchor=dict(anchors[split],t_start=anchors[split]['t_start']+10-W)
                codes[split],nmse[split]=extract(model,a.arch,data['parts'][split],anchor,W,device)
            X=codes['probe_train'];V=codes['probe_val']
        else:
            X=codes['probe_train'][:,-W:].reshape(len(codes['probe_train']),-1).contiguous()
            V=codes['probe_val'][:,-W:].reshape(len(codes['probe_val']),-1).contiguous()
        pseed=a.seed+500_000+1000*a.h+10*W+5
        filename=f'probe_W{W}.pt'
        metrics=baseline.fit_probe(X,anchors['probe_train']['Y'],V,anchors['probe_val']['Y'],11,1500,pseed,out/filename)
        arch={'sae':'sae_stacked','tsae':'tsae_paper_stacked'}.get(a.arch,a.arch)
        row=dict(h=a.h,q=11,W=W,k=5,arch=arch,seed=a.seed,width=2048,steps=6000,
            probe_steps=1500,n_params=sum(p.numel() for p in model.parameters()),probe_seed=pseed,
            **metrics,l0_per_window=float(torch.count_nonzero(V))/len(V),
            l0_per_token=float(torch.count_nonzero(V))/(len(V)*W),reconstruction_nmse=nmse['probe_val'],
            readout_window=W,probe_file=filename,model_arch=a.arch,model_source=str(path),model_sha256=sha,
            shared_model_across_windows=a.arch!='txc',encoding_context=W if a.arch=='txc' else (10 if a.arch=='tfa' else 1),
            training_context=W if a.arch=='txc' else (10 if a.arch=='tfa' else (2 if a.arch=='tsae' else 1)),
            probe_input_width=X.shape[1],anchor_policy='shared_endpoint_of_original_W10_window',job_id=job_id)
        if a.arch in ['sae','tsae']:
            if W==1:single=dict(row,arch='sae' if a.arch=='sae' else 'tsae_paper_single',shared_single_token_baseline=True)
            rows.append(dict(single,W=W))
        rows.append(row)
        baseline.save_json(out/'partial_results.json',rows)
        print(job_id,W,metrics['val_accuracy'],flush=True)
        if a.arch!='txc':del X,V
    baseline.save_json(out/'result.json',dict(started=started,completed=baseline.now(),rows=rows))


def supervise(a):
    jobs=[dict(h=h,seed=s,arch=arch,W=W) for h in [1,2] for s in [0,1,2]
          for arch in ['txc','sae','tsae','tfa'] for W in (WINDOWS if arch=='txc' else [0])]
    a.output.mkdir(parents=True,exist_ok=True)
    baseline.save_json(a.output/'plan.json',dict(jobs=jobs,created=baseline.now(),
        tfa_context=10,baseline_checkpoints_fixed=True,nested_readouts=True,
        notes='W is probe readout width; TXC input context is also W. TFA has fixed 10-token context.'))
    def launch(j):
        name=f"h{j['h']}_s{j['seed']}_{j['arch']}_W{j['W']}"
        with (a.output/f'{name}.log').open('w') as f:
            cmd=[sys.executable,__file__,'job','--output',str(a.output)]
            for k,v in j.items():cmd.extend(['--'+k,str(v)])
            rc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
        return dict(**j,returncode=rc)
    done=[]
    with ThreadPoolExecutor(max_workers=16) as pool:
        for f in as_completed([pool.submit(launch,j) for j in jobs]):
            done.append(f.result());baseline.save_json(a.output/'progress.json',dict(completed=done,total=len(jobs),updated=baseline.now()))
    assert all(j['returncode']==0 for j in done),done
    rows=[]
    for path in sorted((a.output/'jobs').glob('*/result.json')):rows.extend(json.loads(path.read_text())['rows'])
    assert len(rows)==252,len(rows)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (a.output/'cells.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    warnings=[]
    for h in [1,2]:
        for arch in ['txc','sae','sae_stacked','tsae_paper_stacked']:
            for W in WINDOWS:
                subset=[r for r in rows if (r['h'],r['arch'],r['W'])==(h,arch,W)]
                assert len(subset)==3
                if W<=h:
                    mean=sum(r['val_accuracy'] for r in subset)/3
                    if mean>1/11+6*((1/11)*(10/11)/(3*4096))**.5:
                        warnings.append(dict(h=h,arch=arch,W=W,mean=mean))
        for s in [0,1,2]:
            for arch in ['sae','sae_stacked','tsae_paper_stacked','tfa']:
                subset=[r for r in rows if (r['h'],r['seed'],r['arch'])==(h,s,arch)]
                assert len({r['model_sha256'] for r in subset})==1
                if arch=='sae':assert len({r['val_accuracy'] for r in subset})==1
    baseline.save_json(a.output/'validation.json',dict(status='complete',jobs=len(jobs),rows=len(rows),
        control_warnings=warnings,fixed_checkpoint_check=True,sae_flatness_check=True,
        tfa_note='W is probe width only; TFA always sees 10 positions, so the W<=h information ceiling does not apply to it.'))
    baseline.save_json(a.output/'completion.json',dict(state='complete',completed=baseline.now()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['job','supervise']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--h',type=int,default=1);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--arch',choices=['txc','sae','tsae','tfa'],default='tfa');p.add_argument('--W',type=int,default=0)
    a=p.parse_args();(job if a.mode=='job' else supervise)(a)
