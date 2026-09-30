"""Create a portable results bundle without weights, activations, or code matrices."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

def main(root, destination):
    candidates=[]
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.is_symlink():
            continue
        if path.suffix in ('.json','.jsonl','.csv','.md','.txt','.tex','.png','.pdf','.svg') or path.name.endswith('.oof.npz') or path.name=='code_snapshot.tar.gz':
            candidates.append(path)
    judge_progress=root/'steering/judging/progress.json'
    judge_status=json.loads(judge_progress.read_text()) if judge_progress.exists() else None
    judging=(f"OpenAI judging status: {judge_status['status']}; model {judge_status['model']}. Validation selections gate test judging." if judge_status else 'No paid judging performed; test candidate panels must remain blinded until validation selection.')
    records=[]
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in candidates:
            data=path.read_bytes()
            relative=str(path.relative_to(root))
            archive.writestr(relative,data)
            records.append(dict(path=relative,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
        archive.writestr('bundle_manifest.json',json.dumps(dict(files=records,
            excluded='Model/optimizer weights, activation caches, and sparse full-dictionary code matrices are excluded; see checkpoint_backup_receipt.json for durable final weights.',
            judging=judging),indent=2))
    temporary.replace(destination)
    print(json.dumps(dict(bundle=str(destination),files=len(records),bytes=destination.stat().st_size)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True)
    args=p.parse_args()
    main(args.root,args.destination)
