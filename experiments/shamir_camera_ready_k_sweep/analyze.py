"""Aggregate a complete run and export both k comparisons; never mix pilot rows."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st
from collections import defaultdict


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--allow-partial",action="store_true");parser.add_argument("--no-plots",action="store_true")
    args=parser.parse_args();root=args.output;plan=json.loads((root/"plan.json").read_text())
    rows=[];missing=[];inputs=[]
    for spec in plan["jobs"]:
        p=root/"jobs"/spec["id"]/"result.json"
        if not p.exists():missing.append(spec["id"]);continue
        data=json.loads(p.read_text());assert data["spec"]==spec
        assert (p.parent/"model.pt").exists()
        assert {r["W"] for r in data["rows"]}==set(spec["eval_windows"])
        for r in data["rows"]:
            assert (p.parent/r['probe_file']).exists()
            assert r["n_train"]==r["n_val"]==plan["episodes_per_split"]
            rows.append(dict(r,job_id=spec["id"]))
        inputs.append({"path":str(p.relative_to(root)),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
    if missing and not args.allow_partial:raise RuntimeError(f"{len(missing)} jobs incomplete; refusing to label a partial comparison complete")
    if not missing:assert len(rows)==plan["expected_evaluation_rows"]
    groups=defaultdict(list)
    for r in rows:groups[(r["h"],r["q"],r["W"],r["k"],r["arch"])].append(r)
    summary=[];warnings=[]
    for key,rs in sorted(groups.items()):
        assert len({r['seed'] for r in rs})==len(rs)
        if not missing:assert {r['seed'] for r in rs}==set(plan['seeds'])
        out=dict(zip(['h','q','W','k','arch'],key),n_seeds=len(rs),seeds=';'.join(str(r['seed']) for r in sorted(rs,key=lambda r:r['seed'])))
        for metric in ['val_accuracy','balanced_accuracy','train_accuracy','l0_per_window','l0_per_token','reconstruction_nmse','n_params']:
            values=[r[metric] for r in rs];out[metric+'_mean']=st.fmean(values)
            out[metric+'_sd']=st.stdev(values) if len(values)>1 else 0
        if out['W']<=out['h']:
            n=sum(r['n_val'] for r in rs);chance=1/out['q']
            correct=sum(round(r['val_accuracy']*r['n_val']) for r in rs)
            z=(correct-n*chance)/math.sqrt(n*chance*(1-chance))
            out['below_threshold_chance_z']=z
            if z>6:warnings.append({'cell':list(key),'reason':'below-threshold accuracy exceeds chance by >6 binomial standard deviations','z':z})
        summary.append(out)
    single=defaultdict(list)
    for r in rows:
        if r['arch']=='sae':
            assert r['readout_window']==1 and r['probe_file']=='probe_W1.pt'
            single[(r['h'],r['k'],r['seed'])].append(r['val_accuracy'])
    assert all(len(set(values))==1 for values in single.values()),'Single-token SAE baseline must be identical across plotted windows'
    assert all(r['q']==11 and r['width']==2048 for r in rows)
    dest=root/'analysis';dest.mkdir(exist_ok=True)
    for name,table in [('cells.csv',rows),('summary.csv',summary)]:
        fields=list(dict.fromkeys(k for row in table for k in row))
        with (dest/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(table)
    report={'status':'partial' if missing else 'complete','model_jobs_complete':len(inputs),'evaluation_rows':len(rows),
            'missing_jobs':missing,'control_warnings':warnings,'source_results':inputs,
            'notes':['q=11 and dictionary width=2048 are fixed across every h.',
                     'Same nominal k; actual L0 and parameter counts are reported, not matched.',
                     'Regular TopK SAE reads one token; stacked TopK SAE codes are a separate diagnostic.',
                     'T-SAE/TFA concatenate temporal codes over the window; TXC has a shared window code.',
                     'Shaded bands are sample SD across seeds, not confidence intervals.']}
    (dest/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    if args.no_plots:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    styles=[('txc','TXC','#0072B2','o','-'),('tsae','T-SAE','#D55E00','s','--'),('tfa','TFA','#009E73','^','-.'),('sae','TopK SAE (single token)','#6A6A6A','D',':')]
    for ks,name in [([5,10],'comparison_k5_k10'),([5],'comparison_k5'),([10],'comparison_k10')]:
        fig,axs=plt.subplots(len(ks),3,figsize=(12,3.5*len(ks)+1.1),sharey=True,squeeze=False)
        for row,k in enumerate(ks):
            for col,(h,q) in enumerate([(1,11),(2,11),(3,11)]):
                ax=axs[row,col]
                ax.axvspan(.85,h+1,color='#F2F3F5');ax.axvline(h+1,color='#999999',lw=.8,ls='--')
                ax.axhline(1/q,color='#777777',ls=':',lw=1)
                for arch,label,color,marker,ls in styles:
                    points=sorted([r for r in summary if (r['h'],r['k'],r['arch'])==(h,k,arch)],key=lambda r:r['W'])
                    x=[r['W'] for r in points];y=[r['val_accuracy_mean'] for r in points];sd=[r['val_accuracy_sd'] for r in points]
                    ax.plot(x,y,label=label,color=color,marker=marker,ls=ls,lw=2,ms=4)
                    ax.fill_between(x,[a-b for a,b in zip(y,sd)],[a+b for a,b in zip(y,sd)],color=color,alpha=.13)
                ws=sorted({r['W'] for r in summary if r['h']==h})
                ax.set_xticks(ws);ax.set_ylim(0,1);ax.yaxis.set_major_formatter(PercentFormatter(1))
                ax.set_title(f'h={h}, q={q}  |  k={k}');ax.set_xlabel('Window length, W');ax.grid(axis='y',alpha=.15)
                if col==0:ax.set_ylabel('Secret-recovery accuracy')
        fig.legend(*axs[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.045),ncol=4,frameon=False)
        prefix='PARTIAL: ' if missing else ''
        fig.suptitle(prefix+'Episode-disjoint four-method comparison: fixed q=11',fontsize=16)
        seed_note='Available seeds only; per-cell counts are in summary.csv.' if missing else 'Mean ± sample SD over three seeds.'
        fig.text(.5,.015,'Width=2,048; 6,000 steps. '+seed_note+' Actual L0 and parameters are not matched.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.11,1,.94))
        for ext in ('png','pdf','svg'):fig.savefig(dest/f'{name}.{ext}',dpi=200)
        plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k!='source_results'},indent=2))


if __name__=='__main__':main()
