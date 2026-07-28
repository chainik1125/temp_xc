import numpy as np, json
from pathlib import Path
import experiments.explorations.task_hunt.facecmp.arm_test as at
import experiments.explorations.task_hunt.facecmp.face_battery as fb
SP=Path("/private/tmp/claude-501/-Users-octo-research-projects-agents-mac-c-temp-xc/660a6cf4-2ce0-4db9-8201-37e38db3e1bf/scratchpad")
at.CACHE_ROOT=SP/"cache512"; at.AX_TS=[16,32,64]; at.FOREIGN_TS=[64]; at.RES=SP/"r512"
for name,fn,H in [("RECENCY_age",fb.f_age,64),("rate_H512",fb.f_rate(512),512),
                  ("ewma_tau512",fb.f_ewma(512),512)]:
    at.FACE=name; at.H=H; at.rate_face=fn
    print(f"\n===== {name} @ SEQ_LEN=512 =====",flush=True)
    at.screen("gpt2")
    p=SP/"r512"/"arm_test_gpt2.json"; d=json.loads(p.read_text())
    e=d["meta"]["rows"]["tercile_edges"]
    print("  edges:",[round(x,3) for x in e])
    k=f"{name}/T64/actxmean_linear"
    if k in d["cells"]: print("  T64 per_class:",[round(x,3) for x in d["cells"][k]["per_class"]])
    (SP/"r512"/f"{name}.json").write_text(json.dumps(d,indent=1)); p.unlink()
