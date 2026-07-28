"""⚑ THE DECISIVE TEST: is the ~100-token readable horizon a gpt2 fact or a
program fact? gemma2_2b is one of the program's THREE ACTUAL LEGS, screened
at layer 14 — not a proxy. Local MPS, $0.

Two faces on the gemma2 grid (its own tokenizer, its own terciles):
  RECENCY as-screened  — terciles land well outside a 128-token context
  AGE_within_context   — terciles forced inside it
If the horizon is a gpt2 limitation, the 2b model should close much of the
gap on the FIRST. If it is a program fact, both models look alike.
Pre-registered: the FLOOR must not move with the model (depends only on T+w).
"""
import json,time,os,numpy as np,torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from experiments.explorations.task_hunt.replag.build_labels import MODELS, SEQ_LEN
from experiments.explorations.task_hunt.replag.cache_acts import SCREEN_HS
SEQ_LEN=512  # OVERRIDE: give the model a context that CONTAINS the label's range
SP=Path("/private/tmp/claude-501/-Users-octo-research-projects-agents-mac-c-temp-xc/660a6cf4-2ce0-4db9-8201-37e38db3e1bf/scratchpad")
G=Path("experiments/explorations/task_hunt/retryesc_gen/grids")
key="gemma2_2b"; out=SP/"cache_g2_512"/key; out.mkdir(parents=True,exist_ok=True)
hs=SCREEN_HS[key]
if not (out/"acts_meta.json").exists():
    z=np.load(G/"elicit_retryesc_gen_v1_screen_gemma2.npz")
    flat,off=z["token_ids"],z["doc_off"]
    n_prefix=1 if MODELS[key]["bos"] else 0; content=SEQ_LEN-n_prefix
    tok=AutoTokenizer.from_pretrained(MODELS[key]["hf"]); bos=tok.bos_token_id
    rows,didx=[],[]
    for d in range(len(off)-1):
        seg=flat[off[d]:off[d+1]]
        for s in range(0,len(seg)-content+1,content):
            c=seg[s:s+content]
            rows.append(np.concatenate([[bos],c]) if n_prefix else c); didx.append(d)
    ids=np.asarray(rows,dtype=np.int32); N=ids.shape[0]
    np.savez(out/"tokens.npz",ids=ids,doc_idx=np.asarray(didx,dtype=np.int32),n_prefix=np.int64(n_prefix))
    print(f"{N} rows x {SEQ_LEN} (n_prefix={n_prefix})",flush=True)
    m=AutoModelForCausalLM.from_pretrained(MODELS[key]["hf"],torch_dtype=torch.float16).to("mps").eval()
    d_model=int(m.config.hidden_size)
    print(f"d_model={d_model} layers={m.config.num_hidden_layers} capture={hs}",flush=True)
    mm=np.lib.format.open_memmap(out/f"hs{hs}.npy",mode="w+",dtype=np.float16,shape=(N,SEQ_LEN,d_model))
    t0=time.time(); B=4; it=torch.from_numpy(ids.astype(np.int64))
    with torch.no_grad():
        for s in range(0,N,B):
            e=min(s+B,N)
            o=m(it[s:e].to("mps"),output_hidden_states=True,use_cache=False)
            mm[s:e]=o.hidden_states[hs].detach().to(torch.float16).cpu().numpy()
            if (s//B)%40==0: print(f"  {e}/{N} ({e/max(time.time()-t0,1e-9):.1f} r/s)",flush=True)
    mm.flush(); del mm,m
    a=np.load(out/f"hs{hs}.npy",mmap_mode="r"); smp=a[3,100,:].astype(np.float32)
    assert np.isfinite(smp).all() and np.linalg.norm(smp)>0,"degenerate activations"
    (out/"acts_meta.json").write_text(json.dumps({"seq_len":SEQ_LEN,"screen_hs":hs,"n_seqs":int(N)}))
    print(f"cached in {time.time()-t0:.0f}s",flush=True)

import experiments.explorations.task_hunt.facecmp.arm_test as at
import experiments.explorations.task_hunt.facecmp.face_battery as fb
from experiments.explorations.task_hunt.labels.build_retryesc_gen_premeasure import section_age
at.CACHE_ROOT=SP/"cache_g2_512"; at.AX_TS=[32,64]; at.FOREIGN_TS=[64]; at.RES=SP/"g2res512"
def capped(cap):
    def g(first,off,n_docs,_):
        raw=np.concatenate([section_age(first[off[d]:off[d+1]]) for d in range(n_docs)]).astype(np.float64)
        v=np.log2(1.0+raw); v[~np.isfinite(raw)]=np.nan; v[raw>=cap]=np.nan
        return v
    return g
for name,fn in [("G2512_RECENCY_asscreened",fb.f_age),("G2512_AGE_within_context_120",capped(120))]:
    at.FACE=name; at.H=64; at.rate_face=fn
    print(f"\n===== {name} (gemma2_2b, layer {hs}) =====",flush=True)
    at.screen(key)
    p=SP/"g2res512"/f"arm_test_{key}.json"; d=json.loads(p.read_text())
    print("  edges->ages:",[round(2**x-1,1) for x in d["meta"]["rows"]["tercile_edges"]])
    k=f"{name}/T64/actxmean_linear"
    if k in d["cells"]: print("  T64 per_class:",[round(x,3) for x in d["cells"][k]["per_class"]])
    (SP/"g2res512"/f"{name}.json").write_text(json.dumps(d,indent=1)); p.unlink()
