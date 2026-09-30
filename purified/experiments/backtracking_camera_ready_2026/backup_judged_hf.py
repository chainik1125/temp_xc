"""Append judged outputs privately without changing the verified checkpoint files."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from backup_checkpoints_hf import atomic, digest


def main(root,token_file,repo):
    from huggingface_hub import HfApi,hf_hub_download
    token=token_file.read_text().strip()
    if not token or token_file.stat().st_mode & 0o077: raise ValueError('Credential must be mode 0600')
    receipt_path=root/'results/judged_backup_receipt.json'
    receipt={'status':'preparing','repo_id':repo,'started_unix':time.time()}
    atomic(receipt_path,receipt)
    try:
        status=json.loads((root/'results/steering/judging/progress.json').read_text())
        if status['status']!='complete': raise ValueError('Judging is incomplete')
        prior=json.loads((root/'results/checkpoint_backup_receipt.json').read_text())
        if prior['status']!='verified' or prior['repo_id']!=repo: raise ValueError('Missing verified checkpoint backup')
        api=HfApi(token=token)
        if not api.repo_info(repo).private: raise ValueError('Refusing upload to public repository')
        stage=root/'hf_judged_results'
        (stage/'results').mkdir(parents=True,exist_ok=True)
        paths=[]
        for src,name in [(root/'backtracking_judged_results.zip','results/backtracking_judged_results.zip'),
                         (root/'results/publication/steering/paper_figures.zip','results/steering_paper_figures.zip')]:
            dest=stage/name
            shutil.copyfile(src,dest)
            paths.append({'path':name,'bytes':dest.stat().st_size,'sha256':digest(dest)})
        (stage/'JUDGED_RESULTS.md').write_text('''# Completed steering judging

The newer `results/backtracking_judged_results.zip` supersedes the pre-judging
compact bundle for steering analysis. It includes the original generations,
all raw OpenAI responses, validation-only magnitude selection receipts,
selected-dose test judgments, paired results, paper figures and source code.
All test effects use only the selected magnitude and matched zero control.

The judge is gpt-6-luna with low reasoning and the historical backtracking and
coherence rubrics. These are automated labels, not human-calibrated outcomes.
`results/steering_paper_figures.zip` is a small separate figure/source-table pack.
The 15 final 300K checkpoint files and their original manifest are unchanged.

A readable walkthrough is published in the experiment repository:
https://github.com/chainik1125/temp_xc/blob/neurips-aniket/purified/results/backtracking_camera_ready_2026/RESULTS.md
''')
        p=stage/'JUDGED_RESULTS.md'
        paths.append({'path':p.name,'bytes':p.stat().st_size,'sha256':digest(p)})
        atomic(stage/'judged_results_manifest.json',{'files':paths,'judge_model':'gpt-6-luna',
            'checkpoint_revision':prior['revision'],'judge_progress':status})
        p=stage/'judged_results_manifest.json'
        paths.append({'path':p.name,'bytes':p.stat().st_size,'sha256':digest(p)})
        receipt.update(status='uploading',files=paths)
        atomic(receipt_path,receipt)
        env=dict(os.environ,HF_TOKEN=token,HF_HUB_DISABLE_PROGRESS_BARS='1')
        subprocess.run([str(Path(sys.executable).with_name('hf')),'upload',repo,str(stage),
                        '--repo-type','model','--commit-message','Preserve completed Luna steering judgments and figures'],
                       env=env,check=True)
        info=api.repo_info(repo,files_metadata=True)
        if not info.private: raise ValueError('Privacy changed during upload')
        remote={x.rfilename:x for x in info.siblings}
        checks=paths+[x for x in prior['verified_files'] if x['path'].startswith('checkpoints/')]
        for row in checks:
            item=remote[row['path']]
            if item.size!=row['bytes']: raise ValueError('Remote size mismatch: '+row['path'])
            lfs=getattr(item,'lfs',None)
            sha=getattr(lfs,'sha256',None)
            if sha is None and isinstance(lfs,dict):sha=lfs.get('sha256')
            if sha is None:
                if row['bytes']>8<<20:raise ValueError('Missing large-file checksum')
                sha=digest(hf_hub_download(repo,row['path'],revision=info.sha,token=token))
            if sha!=row['sha256']:raise ValueError('Remote checksum mismatch: '+row['path'])
        receipt.update(status='verified',revision=info.sha,private=True,
            verified_files=checks,finished_unix=time.time(),checkpoint_files_unchanged=30)
        atomic(receipt_path,receipt)
        print(json.dumps({'status':'verified','revision':info.sha,'files':len(checks)}),flush=True)
    except Exception as e:
        receipt.update(status='failed',error=str(e).replace(token,'[redacted]'))
        atomic(receipt_path,receipt)
        raise RuntimeError(receipt['error']) from None
    finally:token_file.unlink(missing_ok=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/workspace/backtracking'))
    p.add_argument('--token-file',type=Path,default=Path('/workspace/.tokens/backtracking_hf_token'))
    p.add_argument('--repo-id',default='aniketdesh/temporal-crosscoders-backtracking-300k-2026-09-30')
    a=p.parse_args();main(a.root,a.token_file,a.repo_id)
