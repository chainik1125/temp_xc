"""Decisive test of the SEQ_LEN ceiling: re-tercile the SAME recency face so
that ALL edges fall INSIDE the 128-token context every probed token actually
saw. If the ceiling explains the weak gains, gain should jump."""
import numpy as np, json
from pathlib import Path
import experiments.explorations.task_hunt.facecmp.arm_test as at
from experiments.explorations.task_hunt.labels.build_retryesc_gen_premeasure import section_age
CACHE=Path("/private/tmp/claude-501/-Users-octo-research-projects-agents-mac-c-temp-xc/660a6cf4-2ce0-4db9-8201-37e38db3e1bf/scratchpad/fakecache")
at.CACHE_ROOT=CACHE; at.AX_TS=[16,32,64]; at.FOREIGN_TS=[64]
SP=Path("/private/tmp/claude-501/-Users-octo-research-projects-agents-mac-c-temp-xc/660a6cf4-2ce0-4db9-8201-37e38db3e1bf/scratchpad")
def capped(cap):
    def g(first,off,n_docs,_):
        raw=np.concatenate([section_age(first[off[d]:off[d+1]])
                            for d in range(n_docs)]).astype(np.float64)
        v=np.log2(1.0+raw)
        v[~np.isfinite(raw)]=np.nan
        v[raw>=cap]=np.nan          # drop rows whose event is OUTSIDE context
        return v
    return g
for cap,name in [(120,"AGE_within_context_120"),(96,"AGE_within_context_96")]:
    at.FACE=name; at.H=64; at.RES=SP/"ceil"; at.rate_face=capped(cap)
    print(f"\n===== {name} (all terciles < {cap} tokens) =====",flush=True)
    at.screen("gpt2")
    d=json.loads((SP/"ceil"/"arm_test_gpt2.json").read_text())
    e=d["meta"]["rows"]["tercile_edges"]
    print("  edges(log2):",[round(x,3) for x in e],"-> ages:",[round(2**x-1,1) for x in e])
    c=d["cells"]; k=f"{name}/T64/actxmean_linear"
    if k in c: print("  T64 arm per_class:",[round(x,3) for x in c[k]["per_class"]])
    (SP/"ceil"/f"{name}.json").write_text(json.dumps(d,indent=1))
    (SP/"ceil"/"arm_test_gpt2.json").unlink()
