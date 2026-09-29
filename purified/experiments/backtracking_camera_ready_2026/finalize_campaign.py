"""Wait for the finite GPU queues, then build figures and a compact archive."""
import json
import subprocess
import sys
import time

from campaign import ROOT, CODE, RESULTS, atomic

def main():
    while True:
        result=subprocess.run(['tmux','list-sessions','-F','#{session_name}'],capture_output=True,text=True)
        live=[name for name in result.stdout.splitlines()
              if name.startswith('bt-worker-') or name.startswith('bt-steer-')]
        if not live:
            break
        time.sleep(30)
    paths=[RESULTS/'queue_state.json',RESULTS/'steering/queue_state.json']
    states=[json.loads(p.read_text()) if p.exists() else {} for p in paths]
    train_ok=len(states[0])==15 and all(r.get('status')=='complete' for r in states[0].values())
    steer_ok=len(states[1])==27 and all(r.get('status')=='complete' for r in states[1].values())
    command=[sys.executable,str(CODE/'summarize_detection.py'),'--root',str(RESULTS)]
    if train_ok:
        command.append('--require-complete')
    with (ROOT/'logs/summary.log').open('w') as log:
        summary=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    paired_exit=None
    if train_ok:
        with (ROOT/'logs/paired-detection.log').open('w') as log:
            paired_exit=subprocess.run([sys.executable,str(CODE/'paired_detection.py'),
                '--root',str(RESULTS),'--n-bootstrap','1000'],stdout=log,stderr=subprocess.STDOUT).returncode
    ready=train_ok and steer_ok and summary.returncode==0 and paired_exit==0
    if not ready:
        diagnostics=RESULTS/'diagnostics'
        diagnostics.mkdir(exist_ok=True)
        for path in (ROOT/'logs').glob('*.log'):
            with path.open(errors='replace') as handle:
                from collections import deque
                tail=''.join(deque(handle,maxlen=100))
            if any(marker in tail for marker in ('Traceback','Error','failed','error')):
                (diagnostics/(path.name+'.txt')).write_text(tail)
    atomic(RESULTS/'completion.json',dict(
        status='ready_for_deferred_judging' if ready else 'needs_attention',
        trained_and_detected_cells=sum(r.get('status')=='complete' for r in states[0].values()),
        steering_arms_saved=sum(r.get('status')=='complete' for r in states[1].values()),
        summary_exit_code=summary.returncode,paired_analysis_exit_code=paired_exit,
        paid_api_calls=0,finished_unix=time.time(),
        pod_action='No stop/delete action taken. Copy final checkpoints off container storage before stopping the pod.',
        incomplete={str(p.relative_to(RESULTS)):{k:v for k,v in s.items() if v.get('status')!='complete'} for p,s in zip(paths,states)}))
    subprocess.run([sys.executable,str(CODE/'pack_results.py'),'--root',str(RESULTS),
                    '--destination',str(ROOT/'backtracking_compact_results.zip')],check=True)

if __name__=='__main__':
    main()
