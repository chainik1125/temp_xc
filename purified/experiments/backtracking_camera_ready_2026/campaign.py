"""Persistent four-worker queue; only verified 300K dictionaries reach detection."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/workspace/backtracking')
CODE = ROOT / 'code'
RESULTS = ROOT / 'results'
HIST = ROOT / 'historical/purified'
CACHE = ROOT / 'assets/act_cache/fb2a74be884e512a/resid_post_L10.npy'
ACTS = ROOT / 'assets/c7_backtracking/stage_a/sentence_acts_L10.npz'

def atomic(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)

def owner_alive(row):
    try:
        argv = Path(f"/proc/{int(row['pid'])}/cmdline").read_bytes().decode().split('\0')
        return str(CODE/'campaign.py') in argv or str(CODE/'steering_campaign.py') in argv or 'code/campaign.py' in argv or 'code/steering_campaign.py' in argv
    except (KeyError, ValueError, FileNotFoundError, ProcessLookupError):
        return False

def reconcile_stale(state_path, lock_path, steering=False):
    """Operator recovery: release dead-owner claims, never retry a failed cell."""
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state=json.loads(state_path.read_text()) if state_path.exists() else {}
        recovered=[]
        for name,row in list(state.items()):
            if row.get('status')!='running' or owner_alive(row):
                continue
            work=RESULTS/'steering/arms'/name if steering else RESULTS/'cells'/name
            active=False
            for proc in Path('/proc').iterdir():
                if not proc.name.isdigit():
                    continue
                try:
                    argv=(proc/'cmdline').read_bytes().decode().split('\0')
                except (FileNotFoundError,PermissionError,ProcessLookupError):
                    continue
                if str(work) in argv or str(work/'detection.json') in argv:
                    active=True
                    break
            training_lock=None
            if not steering and work.exists():
                training_lock=(work/'.train.lock').open('a')
                try:
                    fcntl.flock(training_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    active=True
            if training_lock:
                training_lock.close()
            if not active:
                recovered.append(dict(id=name,previous=row,recovered_unix=time.time()))
                del state[name]
        atomic(state_path,state)
        with state_path.with_name('queue_recovery.jsonl').open('a') as handle:
            for item in recovered:
                handle.write(json.dumps(item)+'\n')
        print(json.dumps(dict(recovered=recovered)),flush=True)

def cells():
    # Start all piloted seeds first; prioritize the longer remaining jobs.
    specs = [('txc_base', 32768, 1), ('topk_sae', 32768, 1),
             ('tsae_paper', 32768, 1), ('stacked_sae', 32768, 1)]
    specs += [(arch, 32768, seed) for seed in (2, 42)
              for arch in ('stacked_sae', 'txc_base', 'tsae_paper', 'topk_sae')]
    specs += [('tsae_paper', 16384, seed) for seed in (1, 2, 42)]
    return [dict(arch=a, width=w, seed=s, id=f'{a}_d{w}_seed{s}') for a,w,s in specs]

def initialize():
    RESULTS.mkdir(exist_ok=True)
    spec = dict(protocol='c7-camera-ready-300k-v1', completed_optimizer_steps=300000,
                no_txc_pro=True, seeds=[1,2,42], primary_S=8,
                width_sensitivity='T-SAE 16K and 32K, both fully trained; no global-best claim',
                judging='Deferred by user; no paid API calls', cells=cells())
    path = RESULTS / 'campaign_manifest.json'
    if path.exists() and json.loads(path.read_text()) != spec:
        raise RuntimeError('Existing campaign differs')
    atomic(path, spec)

def claim(gpu):
    with (RESULTS / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state_path = RESULTS / 'queue_state.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        for cell in cells():
            if cell['id'] not in state:
                state[cell['id']] = dict(status='running', gpu=gpu, pid=os.getpid(), started=time.time())
                atomic(state_path, state)
                return cell
    return None

def finish(cell, status, detail):
    with (RESULTS / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = RESULTS / 'queue_state.json'
        state = json.loads(path.read_text())
        state[cell['id']].update(status=status, detail=detail, finished=time.time())
        atomic(path, state)

def run(command, log, env):
    with log.open('a') as handle:
        handle.write('\n' + json.dumps(dict(command=command, started=time.time())) + '\n')
        handle.flush()
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
    if result.returncode:
        raise RuntimeError(f'{log.name}: exit {result.returncode}')

def worker(gpu):
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES':str(gpu), 'OMP_NUM_THREADS':'8',
           'OPENBLAS_NUM_THREADS':'8', 'TOKENIZERS_PARALLELISM':'false'}
    while cell := claim(gpu):
        try:
            if shutil.disk_usage(ROOT).free < 60 * 1024**3:
                raise RuntimeError('Less than 60 GiB disk free; refusing new work')
            directory = RESULTS / 'cells' / cell['id']
            command = [sys.executable, str(CODE/'train.py'), '--historical-root',str(HIST),
                       '--cache-file',str(CACHE),'--output-root',str(RESULTS),'--arch',cell['arch'],
                       '--d-sae',str(cell['width']),'--seed',str(cell['seed'])]
            if (directory/'latest-resume.pt').exists():
                command.append('--resume')
            run(command, ROOT/'logs'/f"{cell['id']}.train.log", env)
            ckpt = directory/'checkpoint'
            cfg = json.loads((ckpt/'config.json').read_text())
            if cfg['status'] != 'complete' or cfg['n_steps_completed'] != 300000:
                raise RuntimeError('Training returned without a valid 300K receipt')
            command = [sys.executable,str(CODE/'detect.py'),'--historical-root',str(HIST),
                       '--checkpoint-dir',str(ckpt),'--sentence-acts',str(ACTS),
                       '--output',str(directory/'detection.json'),'--arch',cell['arch'],
                       '--d-sae',str(cell['width']),'--seed',str(cell['seed']),'--save-codes',
                       '--position-aware']
            if cell['arch'] == 'tsae_paper':
                command.append('--historical-tsae-check')
            run(command, ROOT/'logs'/f"{cell['id']}.detect.log", env)
            finish(cell, 'complete', '300K training and detection saved')
        except Exception as exc:
            finish(cell, 'failed', str(exc))
            print(json.dumps(dict(cell=cell['id'], error=str(exc))), flush=True)
            # Other cells remain runnable; a failed cell requires review, not an
            # automatic protocol change or a silently substituted checkpoint.
    print(json.dumps(dict(worker=gpu,status='queue_finished')), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--initialize', action='store_true')
    parser.add_argument('--gpu', type=int, choices=range(4))
    parser.add_argument('--reconcile-stale',action='store_true')
    args = parser.parse_args()
    if args.initialize:
        initialize()
    if args.reconcile_stale:
        reconcile_stale(RESULTS/'queue_state.json',RESULTS/'queue.lock')
    if args.gpu is not None:
        worker(args.gpu)
