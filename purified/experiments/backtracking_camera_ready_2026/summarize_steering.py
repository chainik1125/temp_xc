"""Recompute all judged steering effects and render publication-size Nord figures."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import zipfile
import numpy as np
import steering as s
from judge_openai import usage_cost

ARMS=[('txc_base','TXC','#5E81AC','o'),
      ('topk_last','SAE · last','#4C566A','s'),
      ('topk_mean','SAE · mean','#4C566A','s'),
      ('topk_max','SAE · max','#4C566A','s'),
      ('tsae_last','T-SAE · last','#D08770','^'),
      ('tsae_mean','T-SAE · mean','#D08770','^'),
      ('tsae_max','T-SAE · max','#D08770','^'),
      ('stacked_atoms','Stacked SAE','#B48EAD','D'),
      ('random','Random','#88C0D0','v')]
METRICS=['gc','coherent_gc','coherent_bt','incoherent','correct']
SEEDS=[1,2,42]


def write_csv(path, data):
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]))
        writer.writeheader(); writer.writerows(data)


def main(root):
    progress=s.read(root/'steering/judging/progress.json')
    if progress['status']!='complete':
        raise ValueError('All validation and selected test labels must be complete')
    output=root/'publication/steering'
    output.mkdir(parents=True,exist_ok=True)
    splits=s.read(root/'steering/split.json')
    qids=splits['test_qids']
    if len(qids)!=100 or len(set(qids))!=100:
        raise ValueError('Expected 100 distinct held-out question IDs')
    selections=s.read(root/'steering/judging/validation_selection_gate.json')['selections']
    prompt_rows=[]; seed_rows=[]; aggregates=[]; provenance=[]; curves=[]; all_deltas={}
    length_rows=[]; zero_hashes={q:set() for q in qids}
    rng=np.random.default_rng(42)
    bootstrap=rng.integers(0,len(qids),(10000,len(qids)))
    for arm,label,color,marker in ARMS:
        baseline_values={m:[] for m in METRICS}
        treated_values={m:[] for m in METRICS}
        deltas={m:[] for m in METRICS}
        for seed in SEEDS:
            work=root/'steering/arms'/f'{arm}_seed{seed}'
            selection=s.read(work/'selection.json')
            if s.digest(work/'selection.json')!=selections[work.name]:
                raise ValueError('Selection gate hash mismatch')
            identity,scores=s.complete_scores(work,'test')
            if identity['qids']!=qids or identity['selection_sha256']!=selections[work.name]:
                raise ValueError('Mismatched test cohort or selection')
            old=s.read(work/'test_summary.json')
            magnitude=float(selection['magnitude'])
            generations=s.rows(work/'test/generations.jsonl')
            for role,dose in [('zero',0.),('selected',magnitude)]:
                panel=[r for r in generations if r['magnitude']==dose]
                length_rows.append(dict(arm=arm,seed=seed,role=role,magnitude=dose,
                    mean_tokens=float(np.mean([r['continuation_token_count'] for r in panel])),
                    fraction_reaching_token_budget=float(np.mean([r['continuation_token_count']>=r['remaining_token_budget'] for r in panel])),
                    fraction_without_parsed_answer=float(np.mean([not bool(r.get('parsed_answer')) for r in panel]))))
            for r in generations:
                if r['magnitude']==0.:zero_hashes[r['question_id']].add(s.canonical_hash(r['continuation_token_ids']))
            for metric in METRICS:
                zero=np.array([scores[q,0.][metric] for q in qids],float)
                treated=np.array([scores[q,magnitude][metric] for q in qids],float)
                difference=treated-zero
                if not np.isclose(old['metrics'][metric]['paired_delta'],difference.mean(),atol=1e-12):
                    raise ValueError('Per-arm summary does not replay from raw labels')
                baseline_values[metric].append(zero)
                treated_values[metric].append(treated)
                deltas[metric].append(difference)
                seed_rows.append(dict(arm=arm,label=label,seed=seed,magnitude=magnitude,
                    metric=metric,baseline_mean=float(zero.mean()),treated_mean=float(treated.mean()),
                    paired_delta=float(difference.mean())))
                for i,q in enumerate(qids):
                    prompt_rows.append(dict(arm=arm,seed=seed,question_id=q,magnitude=magnitude,
                        metric=metric,baseline=float(zero[i]),treated=float(treated[i]),paired_delta=float(difference[i])))
            for mag, value in selection['validation_curve'].items():
                curves.append(dict(arm=arm,seed=seed,magnitude=float(mag),coherent_gc_delta=value))
            provenance.append({'arm':work.name,'files':{name:s.digest(work/name) for name in
                ('selection.json','validation/judgments.jsonl','test/judgments.jsonl','test/generations.jsonl',
                 'test_summary.json','test/judge_identity.json')}})
        for metric in METRICS:
            a=np.asarray(deltas[metric])
            all_deltas[arm,metric]=a
            means=a.mean(axis=1)
            ci=np.quantile(a.mean(axis=0)[bootstrap].mean(axis=1),[.025,.975])
            aggregates.append(dict(arm=arm,label=label,metric=metric,
                mean_delta=float(means.mean()),seed_sd=float(means.std(ddof=1)),
                conditional_question_ci_low=float(ci[0]),conditional_question_ci_high=float(ci[1]),
                baseline_mean=float(np.mean(baseline_values[metric])),treated_mean=float(np.mean(treated_values[metric])),
                **{f'seed_{seed}':float(means[i]) for i,seed in enumerate(SEEDS)}))
    paired=[]
    for arm,*_ in ARMS[1:]:
        for metric in METRICS:
            contrast=all_deltas['txc_base',metric]-all_deltas[arm,metric]
            ci=np.quantile(contrast.mean(axis=0)[bootstrap].mean(axis=1),[.025,.975])
            paired.append(dict(reference='txc_base',comparison=arm,metric=metric,
                mean_paired_difference=float(contrast.mean()),paired_seed_sd=float(contrast.mean(1).std(ddof=1)),
                conditional_question_ci_low=float(ci[0]),conditional_question_ci_high=float(ci[1])))
    ledger=s.rows(root/'steering/judging/api_ledger.jsonl')
    cost=sum(usage_cost(r['usage']) if r.get('usage') else r['accounted_usd'] for r in ledger)
    summary={'status':'complete','model':'gpt-6-luna','reasoning_effort':'low','arms':27,
        'seeds':SEEDS,'validation_questions':20,'test_questions':100,
        'api_calls':len(ledger),'estimated_cost_usd':cost,
        'cost_basis':'API token usage including cached input, cache writes and reasoning output; not an invoice',
        'selection':'Signed magnitude maximizes validation coherent genuine-backtracking count gain; zero wins ties',
        'uncertainty':'Seed SD describes 3 fixed dictionary seeds. Conditional 95% question intervals resample the same 100 questions jointly across all arms and seeds; no refitting/reselection, no multiplicity correction.',
        'judging_limitations':'Automated Luna labels using historical rubrics; not human-validated or calibrated against the historical Claude judge. Continuation text limited to 6000 characters per frozen rubric.',
        'accuracy_limitations':'Existing boxed-answer extractor and symbolic/string matcher, not a judge of full proof correctness. Failure to produce a matched boxed answer within the fixed token budget counts as incorrect.',
        'zero_control_audit':{'questions_with_multiple_zero_token_sequences':sum(len(h)>1 for h in zero_hashes.values()),'max_distinct_zero_sequences_per_question':max(map(len,zero_hashes.values()))},
        'results':aggregates,'paired_contrasts':paired,'sources':provenance}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for filename,data in [('seed_metrics.csv',seed_rows),('question_metrics.csv',prompt_rows),
                          ('aggregate_metrics.csv',aggregates),('paired_contrasts.csv',paired),('validation_curves.csv',curves),('generation_lengths.csv',length_rows)]:
        write_csv(output/filename,data)
    render(aggregates,curves,output)
    (output/'CAPTIONS.md').write_text('''# Steering caption notes

`steering_effects`: Validation-selected steering effects on 100 held-out MATH-500 questions. Each of 27 arms selects one signed magnitude from {-12,-8,-4,0,4,8,12} using a separate shared set of 20 validation questions, maximizing the mean genuine-backtracking count gated by coherence >=2. Zero wins exact ties. All dictionaries have 300K optimizer steps; core widths are 32K. Large points show the mean over dictionary seeds 1,2,42. Horizontal error bars are paired question-bootstrap 95% intervals, jointly resampling the same 100 questions across all arms and seeds while holding trained models and selected doses fixed. Small points show individual seed means. These intervals do not include retraining, reselection or judge-label uncertainty and are not corrected for multiple comparisons. Differences are relative to each arm's matched zero-intervention continuation. Left: change in coherent genuine-backtracking events per continuation. Middle: change in the fraction of incoherent continuations, in percentage points (lower is better). Right: change in boxed-answer accuracy, in percentage points. Accuracy uses the frozen automatic answer matcher. The same held-out questions are shared across all arms; this is not 300 independent questions. Random denotes a norm-matched random direction per seed.

`steering_validation_curves`: Validation-only signed-magnitude sensitivity of coherent genuine-backtracking count, relative to zero. Lines and bands show means and sample SD over the same three seeds on 20 questions. Panels are last-token, mean-pool and max-pool feature selection, from left to right; TXC, Stacked SAE and random references repeat across panels. These are tuning curves, not held-out test effects. Each seed selects its own dose, so maxima of these mean curves need not equal the chosen per-seed doses.

Both figures use DeepSeek-R1-Distill-Llama-8B, canonical single-BOS prompts, greedy cut25 continuation, norm-matched layer-10 interventions affecting prefill and continuation, and the frozen feature-selection rule. Luna (`gpt-6-luna`, low reasoning) supplies the historical backtracking and coherence rubrics. These labels have not been human-validated or calibrated against the original Claude judge, so do not pool old/new judge results or assert human-level label reliability. The unsupervised training cache cannot be mapped exhaustively to MATH-500 question IDs, so the split is held out from steering tuning and feature mining, not established wholly unseen training data. T-SAE's original training objective reconstructs fewer tokens per step; equal optimizer steps do not imply equal token exposure or parameter counts.

The source tables retain seed SD as a separate quantity and give the question-bootstrap intervals plotted in the main figure, conditional on these trained dictionaries, feature choices, judges and selected doses. They do not include retraining/reselection uncertainty and are not simultaneous multiple-comparison intervals. No significance stars are used.
''')
    (output/'include_figures.tex').write_text('\\includegraphics[width=\\linewidth]{steering_effects.pdf}\n\\includegraphics[width=\\linewidth]{steering_validation_curves.pdf}\n')
    files={p.name:s.digest(p) for p in output.iterdir() if p.is_file() and p.name not in ('figure_manifest.json','paper_figures.zip')}
    (output/'figure_manifest.json').write_text(json.dumps({'files_sha256':files,'palette':'Nord',
        'embedded_titles':False,'embedded_captions':False,
        'effects_error_bars':'conditional paired question-bootstrap 95% intervals; individual seed means shown as dots','effects_inches':[5.5,2.65],
        'validation_inches':[5.5,2.4],'raster_dpi':400,'pdf_fonttype':42,
        'source_script_sha256':s.digest(Path(__file__))},indent=2)+'\n')
    with zipfile.ZipFile(output/'paper_figures.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.iterdir()):
            if p.is_file() and p.name!='paper_figures.zip': z.write(p,p.name)
    print(json.dumps({'status':'complete','output':str(output),'estimated_cost_usd':cost}),flush=True)


def render(records,curves,output,*,validation_only=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    lookup={(r['arm'],r['metric']):r for r in records}
    style={'font.family':'DejaVu Sans','font.size':7.5,'axes.labelsize':8,
        'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'ps.fonttype':42,
        'svg.fonttype':'none','axes.labelcolor':'#2E3440','text.color':'#2E3440',
        'xtick.color':'#2E3440','ytick.color':'#2E3440','savefig.bbox':None}
    def dress(ax,axis):
        ax.spines[['top','right']].set_visible(False)
        for spine in ax.spines.values(): spine.set_color('#4C566A'); spine.set_linewidth(.65)
        ax.grid(axis=axis,color='#E5E9F0',linewidth=.5)
        ax.set_axisbelow(True)
    def save(fig,stem):
        if any(ax.get_title() for ax in fig.axes): raise ValueError('No embedded titles')
        for ext in ('pdf','svg','png'):
            kwargs={'metadata':{'Title':None,'CreationDate':None,'ModDate':None}} if ext=='pdf' else {}
            fig.savefig(output/f'{stem}.{ext}',dpi=400,facecolor='white',**kwargs)
        plt.close(fig)
    with plt.rc_context(style):
        if not validation_only:
            fig,axes=plt.subplots(1,3,figsize=(5.5,2.65),sharey=True)
            fig.subplots_adjust(left=.19,right=.985,top=.98,bottom=.24,wspace=.28)
            panels=[('coherent_gc',1,'Backtracking change\n(events / continuation)'),
                    ('incoherent',100,'Incoherence change\n(percentage points)'),
                    ('correct',100,'Accuracy change\n(percentage points)')]
            for ax,(metric,scale,xlabel) in zip(axes,panels):
                dress(ax,'x'); ax.axvline(0,color='#7A8493',lw=.7,ls=(0,(2,2)))
                for i,(arm,label,color,marker) in enumerate(ARMS):
                    row=lookup[arm,metric]
                    for dy,seed in zip((-.13,0,.13),SEEDS):
                        ax.plot(row[f'seed_{seed}']*scale,i+dy,'o',color=color,ms=2.2,alpha=.55,mew=0)
                    ax.errorbar(row['mean_delta']*scale,i,
                        xerr=np.array([[row['mean_delta']-row['conditional_question_ci_low']],
                                       [row['conditional_question_ci_high']-row['mean_delta']]])*scale,
                        fmt=marker,color=color,ms=3.8,mec='white',mew=.45,lw=1.05,capsize=2,capthick=.65)
                ax.set_xlabel(xlabel,labelpad=4)
                ax.set_yticks(range(9),[a[1] for a in ARMS]); ax.tick_params(axis='y',length=0,pad=3)
                ax.set_ylim(8.6,-.6); ax.xaxis.set_major_locator(MaxNLocator(nbins=3,min_n_ticks=3))
                ax.margins(x=.15)
                for y in (.5,3.5,6.5,7.5): ax.axhline(y,color='#E5E9F0',lw=.45)
            save(fig,'steering_effects')
        fig,axes=plt.subplots(1,3,figsize=(5.5,2.4),sharex=True,sharey=True)
        fig.subplots_adjust(left=.115,right=.985,bottom=.23,top=.82,wspace=.14)
        mags=s.MAGNITUDES
        legends=[]
        for ax,pool in zip(axes,('last','mean','max')):
            dress(ax,'y'); ax.axhline(0,color='#7A8493',lw=.7,ls=(0,(2,2)))
            for arm,label,color,marker,style in [('txc_base','TXC','#5E81AC','o','-'),
                ('topk_'+pool,'Shared SAE','#4C566A','s',(0,(3,1.5))),
                ('tsae_'+pool,'T-SAE','#D08770','^',(0,(1,1.3))),
                ('stacked_atoms','Stacked SAE','#B48EAD','D',(0,(4,1.5,1,1.5))),
                ('random','Random','#88C0D0','v',(0,(2,1)))]:
                a=np.array([[next(r['coherent_gc_delta'] for r in curves if r['arm']==arm and r['seed']==seed and r['magnitude']==m) for m in mags] for seed in SEEDS])
                mean=a.mean(0); sd=a.std(0,ddof=1)
                ax.fill_between(mags,mean-sd,mean+sd,color=color,alpha=.09,lw=0)
                ax.plot(mags,mean,color=color,marker=marker,ms=3,lw=1.05,ls=style)
                if pool=='last': legends.append(Line2D([],[],label=label,color=color,marker=marker,ms=3,lw=1.05,ls=style))
            ax.set_xticks([-12,-4,4,12]); ax.set_xlim(-13,13)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        axes[0].set_ylabel('Backtracking change\n(events / continuation)',labelpad=4)
        fig.supxlabel('Signed steering magnitude',fontsize=8,y=.03)
        fig.legend(handles=legends,loc='upper center',bbox_to_anchor=(.54,1),ncol=3,
            frameon=False,fontsize=7,columnspacing=1.25,handlelength=2)
        save(fig,'steering_validation_curves')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
