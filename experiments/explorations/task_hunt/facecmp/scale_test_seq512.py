"""Does the READABLE HORIZON grow with model scale? gpt2-small vs gpt2-medium.

gpt2-medium uses the IDENTICAL gpt2 tokenizer, so the corpus grid, the event
stream, the terciles, the manifest and the floor are all byte-identical to the
gpt2 leg. Only the model changes. That makes this a controlled scale step,
unlike swapping in gemma2 (different tokenizer -> different grid).

Layer: gpt2-small screens at 7/12 = 0.58 depth; gpt2-medium has 24 layers, so
14 preserves relative depth. Disclosed because it is a choice, not a given.

Two faces: the as-screened recency terciles (ages 121/286, MOSTLY OUTSIDE the
128-token context) and the context-capped version (ages ~46/72, INSIDE it).
If the horizon is representational rather than architectural, the medium model
should close some of the gap on the FIRST while the second stays flat.
"""
import json,time,numpy as np,torch
from pathlib import Path
from transformers import AutoModelForCausalLM
SP=Path("/private/tmp/claude-501/-Users-octo-research-projects-agents-mac-c-temp-xc/660a6cf4-2ce0-4db9-8201-37e38db3e1bf/scratchpad")
G=Path("experiments/explorations/task_hunt/retryesc_gen/grids")
MODEL="openai-community/gpt2-medium"; LAYER=14; SEQ=512
out=SP/"cache_med512"/"gpt2"; out.mkdir(parents=True,exist_ok=True)
z=np.load(G/"elicit_retryesc_gen_v1_screen_gpt2.npz")
flat,off=z["token_ids"],z["doc_off"]
rows,didx=[],[]
for d in range(len(off)-1):
    seg=flat[off[d]:off[d+1]]
    for s in range(0,len(seg)-SEQ+1,SEQ):
        rows.append(seg[s:s+SEQ]); didx.append(d)
ids=np.asarray(rows,dtype=np.int32); N=ids.shape[0]
np.savez(out/"tokens.npz",ids=ids,doc_idx=np.asarray(didx,dtype=np.int32),n_prefix=np.int64(0))
m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32).to("mps").eval()
d_model=int(m.config.hidden_size)
print(f"{MODEL}: {N} rows, d_model={d_model}, n_layer={m.config.n_layer}, capture={LAYER}",flush=True)
mm=np.lib.format.open_memmap(out/f"hs{LAYER}.npy",mode="w+",dtype=np.float16,shape=(N,SEQ,d_model))
t0=time.time(); B=8; it=torch.from_numpy(ids.astype(np.int64))
with torch.no_grad():
    for s in range(0,N,B):
        e=min(s+B,N)
        o=m(it[s:e].to("mps"),output_hidden_states=True,use_cache=False)
        mm[s:e]=o.hidden_states[LAYER].detach().to(torch.float16).cpu().numpy()
        if (s//B)%20==0: print(f"  {e}/{N}",flush=True)
mm.flush(); del mm,m
print(f"cached in {time.time()-t0:.0f}s",flush=True)

import experiments.explorations.task_hunt.facecmp.arm_test as at
import experiments.explorations.task_hunt.facecmp.face_battery as fb
from experiments.explorations.task_hunt.labels.build_retryesc_gen_premeasure import section_age
at.CACHE_ROOT=SP/"cache_med512"; at.AX_TS=[32,64]; at.FOREIGN_TS=[64]; at.RES=SP/"med512"
import experiments.explorations.task_hunt.replag.cache_acts as ca
ca.SCREEN_HS["gpt2"]=LAYER; at.SCREEN_HS=ca.SCREEN_HS
def capped(cap):
    def g(first,off,n_docs,_):
        raw=np.concatenate([section_age(first[off[d]:off[d+1]]) for d in range(n_docs)]).astype(np.float64)
        v=np.log2(1.0+raw); v[~np.isfinite(raw)]=np.nan; v[raw>=cap]=np.nan
        return v
    return g
for name,fn in [("MED512_RECENCY_asscreened",fb.f_age),("MED512_AGE_within_context_120",capped(120))]:
    at.FACE=name; at.H=64; at.rate_face=fn
    print(f"\n===== {name} =====",flush=True)
    at.screen("gpt2")
    p=SP/"med512"/"arm_test_gpt2.json"; d=json.loads(p.read_text())
    k=f"{name}/T64/actxmean_linear"
    print("  edges->ages:",[round(2**x-1,1) for x in d["meta"]["rows"]["tercile_edges"]])
    if k in d["cells"]: print("  T64 per_class:",[round(x,3) for x in d["cells"][k]["per_class"]])
    (SP/"med512"/f"{name}.json").write_text(json.dumps(d,indent=1)); p.unlink()
