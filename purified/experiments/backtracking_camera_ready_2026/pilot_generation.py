"""Briefly pause one owned trainer while exercising the generation path."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from campaign import ROOT, CODE, HIST, RESULTS

def main():
    matches=[]
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            argv=(directory/'cmdline').read_bytes().decode().split('\0')
            env=(directory/'environ').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):
            continue
        if str(CODE/'train.py') in argv and '--arch' in argv and argv[argv.index('--arch')+1]=='topk_sae' and b'CUDA_VISIBLE_DEVICES=1' in env:
            matches.append(int(directory.name))
    if len(matches)!=1:
        raise RuntimeError(f'Expected one owned TopK trainer on GPU1, found {matches}')
    pid=matches[0]
    signal.signal(signal.SIGTERM,lambda *_: sys.exit(143))
    start=time.time()
    os.kill(pid,signal.SIGSTOP)
    try:
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'1','OMP_NUM_THREADS':'8','TOKENIZERS_PARALLELISM':'false'}
        command=[sys.executable,str(CODE/'steering.py'),'phase1','--historical-root',str(HIST),
                 '--workspace',str(RESULTS/'steering/shared'),'--split-file',str(RESULTS/'steering/split.json'),
                 '--math500',str(ROOT/'assets/backtracking-math500-test.jsonl'),
                 '--batch-size','8','--max-new-tokens','1024','--max-batches','1','--verify-zero']
        result=subprocess.run(command,env=env,timeout=480)
        print(json.dumps(dict(returncode=result.returncode,elapsed_seconds=time.time()-start)),flush=True)
        if result.returncode:
            raise RuntimeError('Generation pilot failed')
    finally:
        os.kill(pid,signal.SIGCONT)
        print(json.dumps(dict(resumed_training_pid=pid)),flush=True)

if __name__=='__main__':
    main()
