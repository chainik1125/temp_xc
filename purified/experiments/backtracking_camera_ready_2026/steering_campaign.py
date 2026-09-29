"""Follow training with offline steering panels; never submit a judge request."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from campaign import ROOT, CODE, RESULTS, HIST, ACTS, atomic, run, owner_alive, reconcile_stale

STEER = RESULTS / 'steering'
MATH = ROOT / 'assets/backtracking-math500-test.jsonl'
SPLIT = STEER / 'split.json'
PHASE = STEER / 'shared/phase1_unsteered.json'
ARMS = {'txc_base':'txc_base', 'topk_last':'topk_sae', 'topk_mean':'topk_sae',
        'topk_max':'topk_sae', 'tsae_last':'tsae_paper', 'tsae_mean':'tsae_paper',
        'tsae_max':'tsae_paper', 'stacked_atoms':'stacked_sae'}

def jobs():
    result = [dict(id=f'{arm}_seed{seed}', arm=arm, seed=seed,
                   checkpoint=f'{arch}_d32768_seed{seed}')
              for seed in (1,2,42) for arm,arch in ARMS.items()]
    result += [dict(id=f'random_seed{seed}', arm='txc_base', seed=seed,
                    checkpoint=f'txc_base_d32768_seed{seed}', random_seed=1000+seed)
               for seed in (1,2,42)]
    return result

def call(arguments, name, env):
    run([sys.executable,str(CODE/'steering.py'),*arguments], ROOT/'logs'/name, env)

def ensure_shared(env):
    STEER.mkdir(exist_ok=True)
    with (STEER/'shared.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        call(['split','--historical-root',str(HIST),'--sentence-acts',str(ACTS),
              '--math500',str(MATH),'--output',str(SPLIT)],'steering-split.log',env)
        # The CLI always checks the frozen identity, including when all rows
        # already exist. A successful pilot must precede this expensive stage.
        from steering import GENERATION_RECIPE
        zero = json.loads((STEER/'shared/phase1_zero_check.json').read_text())
        checks=zero.get('zero_hook_exact_noop',[])
        validation=json.loads(SPLIT.read_text())['validation_qids']
        if len(checks)!=8 or not all(v is True for v in checks) or zero.get('question_ids')!=validation[:8] or zero.get('generation_recipe')!=GENERATION_RECIPE:
            raise RuntimeError('Run and pass the eight-question generation pilot first')
        call(['phase1','--historical-root',str(HIST),'--workspace',str(STEER/'shared'),
                  '--split-file',str(SPLIT),'--math500',str(MATH),
                  '--batch-size','8','--max-new-tokens','1024'], 'steering-phase1.log',env)
        if len(json.loads(PHASE.read_text())) != 120:
            raise RuntimeError('Incomplete shared 120-question phase1')
        manifest = dict(steps=300000,primary_width=32768,seeds=[1,2,42],
                        validation_questions=20,test_questions=100,
                        magnitudes=[-12,-8,-4,0,4,8,12],jobs=jobs(),
                        panels=27*120*7,judging='Deferred; no API requests',
                        test_policy='Blind candidates; only validation-selected dose plus zero can be exported for test judging')
        path = STEER/'campaign_manifest.json'
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise RuntimeError('Existing steering manifest differs')
        atomic(path,manifest)

def claim(gpu):
    with (STEER/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = STEER/'queue_state.json'
        state = json.loads(path.read_text()) if path.exists() else {}
        training = json.loads((RESULTS/'queue_state.json').read_text())
        waiting = False
        for job in jobs():
            if job['id'] in state:
                continue
            checkpoint = RESULTS/'cells'/job['checkpoint']/'checkpoint/config.json'
            if not checkpoint.exists():
                train_state=training.get(job['checkpoint'],{})
                if train_state.get('status') == 'failed':
                    state[job['id']] = dict(status='blocked_by_training_failure')
                elif train_state.get('status')=='running' and not owner_alive(train_state):
                    state[job['id']] = dict(status='blocked_by_dead_training_owner',
                        detail='Reconcile stale claims and restart training before retrying this arm')
                else:
                    waiting = True
                continue
            state[job['id']] = dict(status='running',gpu=gpu,pid=os.getpid(),started=time.time())
            atomic(path,state)
            return job,False
        atomic(path,state)
        return None,waiting

def finish(job,status,detail):
    with (STEER/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = STEER/'queue_state.json'
        state = json.loads(path.read_text())
        state[job['id']].update(status=status,detail=detail,finished=time.time())
        atomic(path,state)

def main(gpu):
    env = {**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'8',
           'OPENBLAS_NUM_THREADS':'8','TOKENIZERS_PARALLELISM':'false'}
    # Training workers own their GPU until their local queue loop has exited.
    while subprocess.run(['tmux','has-session','-t',f'bt-worker-{gpu}'],
                         stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode == 0:
        time.sleep(30)
    ensure_shared(env)
    while True:
        job,waiting = claim(gpu)
        if not job:
            if waiting:
                time.sleep(30)
                continue
            break
        work = STEER/'arms'/job['id']
        try:
            args = ['mine','--historical-root',str(HIST),'--workspace',str(work),
                    '--arm',job['arm'],'--checkpoint-dir',str(RESULTS/'cells'/job['checkpoint']/'checkpoint'),
                    '--split-file',str(SPLIT),'--sentence-acts',str(ACTS),'--batch-size','256']
            if 'random_seed' in job:
                args += ['--random-control-seed',str(job['random_seed'])]
            call(args,f"{job['id']}.mine.log",env)
            for partition in ('validation','test_candidates'):
                call(['generate','--historical-root',str(HIST),'--workspace',str(work),
                      '--partition',partition,'--split-file',str(SPLIT),'--phase1',str(PHASE),
                      '--batch-size','8'],f"{job['id']}.{partition}.log",env)
                receipt=json.loads((work/partition/'progress.json').read_text())
                if receipt.get('status') != 'complete':
                    raise RuntimeError(f'Incomplete {partition} grid')
            call(['export-judge','--historical-root',str(HIST),'--workspace',str(work),
                  '--partition','validation'],f"{job['id']}.judge-template.log",env)
            finish(job,'complete','Raw panels and validation judge template saved; no API calls')
        except Exception as exc:
            finish(job,'failed',str(exc))
            print(json.dumps(dict(job=job['id'],error=str(exc))),flush=True)
    print(json.dumps(dict(gpu=gpu,status='steering_queue_finished')),flush=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',type=int,choices=range(4))
    parser.add_argument('--reconcile-stale',action='store_true')
    args=parser.parse_args()
    if args.reconcile_stale:
        reconcile_stale(STEER/'queue_state.json',STEER/'queue.lock',steering=True)
    if args.gpu is not None:
        main(args.gpu)
