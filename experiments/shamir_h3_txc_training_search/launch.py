"""Detached launcher for a one-hour search on the explicitly allocated GPU5."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(sys.argv[1]).resolve()
GPU = "GPU-0216af65-2b0e-3798-454e-1208593efcc3"
PYTHON = "/data/users/dmitry/venvs/er-sprint/bin/python"


def save(name, data):
    path=ROOT/name; tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(data,indent=2)+"\n"); tmp.replace(path)


def now(): return datetime.now(timezone.utc).isoformat()


env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES=GPU, CUDA_MPS_PIPE_DIRECTORY=str(ROOT/"mps-pipe"),
           CUDA_MPS_LOG_DIRECTORY=str(ROOT/"mps-log"), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
           OPENBLAS_NUM_THREADS="1", PYTHONPATH=f"{ROOT}/source:{ROOT}/source/src")
for name in ("mps-pipe","mps-log"): (ROOT/name).mkdir(mode=0o700,exist_ok=True)
used = int(subprocess.check_output(["nvidia-smi","--id="+GPU,"--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True).strip())
if used>100: raise RuntimeError(f"Allocated GPU5 not idle: {used} MiB in use; refusing to share an unidentified job.")
command = [PYTHON,str(ROOT/"source/experiments/shamir_h3_txc_training_search/run.py"),"supervise","--output",str(ROOT/"full"),"--workers","16","--budget-seconds","3600"]
manifest = [{"path":str(p.relative_to(ROOT/"source")),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted((ROOT/"source").rglob("*.py"))]
save("source_manifest.json",dict(created=now(),files=manifest))
save("launch_receipt.json",dict(started=now(),launcher_pid=os.getpid(),gpu_uuid=GPU,gpu_index=5,command=command,environment={k:env[k] for k in ("CUDA_VISIBLE_DEVICES","CUDA_MPS_PIPE_DIRECTORY","CUDA_MPS_LOG_DIRECTORY","OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","PYTHONPATH")},source_files=len(manifest),budget_seconds=3600))
code = None
try:
    subprocess.run(["nvidia-cuda-mps-control","-d"],env=env,check=True)
    with (ROOT/"supervisor.log").open("a") as log:
        proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        save("supervisor_receipt.json",dict(started=now(),pid=proc.pid))
        code=proc.wait()
except BaseException:
    save("launcher_error.json",dict(time=now(),traceback=traceback.format_exc()))
    raise
finally:
    subprocess.run(["nvidia-cuda-mps-control"],input="quit\n",env=env,text=True,capture_output=True,timeout=15)
    save("launcher_exit.json",dict(finished=now(),returncode=code,private_mps_stop_requested=True))
