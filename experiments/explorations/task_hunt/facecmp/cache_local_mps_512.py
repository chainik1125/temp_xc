"""Rebuild the gpt2 cache at SEQ_LEN=512 instead of 128. Same corpus, same
layer, same dtype — only the CONTEXT each probed token gets. $0 on MPS."""
import json,time,numpy as np,torch
from pathlib import Path
from transformers import AutoModelForCausalLM
from experiments.explorations.task_hunt.replag.build_labels import MODELS
from experiments.explorations.task_hunt.replag.cache_acts import SCREEN_HS
SP=Path("/private/tmp/claude-501/-Users-octo-research-projects-agents-mac-c-temp-xc/660a6cf4-2ce0-4db9-8201-37e38db3e1bf/scratchpad/cache512")
G=Path("experiments/explorations/task_hunt/retryesc_gen/grids")
SEQ=512; key="gpt2"; out=SP/key; out.mkdir(parents=True,exist_ok=True)
z=np.load(G/"elicit_retryesc_gen_v1_screen_gpt2.npz")
flat,off=z["token_ids"],z["doc_off"]
rows,didx=[],[]
for d in range(len(off)-1):
    seg=flat[off[d]:off[d+1]]
    for s in range(0,len(seg)-SEQ+1,SEQ):
        rows.append(seg[s:s+SEQ]); didx.append(d)
ids=np.asarray(rows,dtype=np.int32); N=ids.shape[0]
np.savez(out/"tokens.npz",ids=ids,doc_idx=np.asarray(didx,dtype=np.int32),n_prefix=np.int64(0))
print(f"{N} rows x {SEQ}")
hs=SCREEN_HS[key]
m=AutoModelForCausalLM.from_pretrained(MODELS[key]["hf"],torch_dtype=torch.float32).to("mps").eval()
d_model=int(m.config.hidden_size)
mm=np.lib.format.open_memmap(out/f"hs{hs}.npy",mode="w+",dtype=np.float16,shape=(N,SEQ,d_model))
t0=time.time(); B=16; it=torch.from_numpy(ids.astype(np.int64))
with torch.no_grad():
    for s in range(0,N,B):
        e=min(s+B,N)
        o=m(it[s:e].to("mps"),output_hidden_states=True,use_cache=False)
        mm[s:e]=o.hidden_states[hs].detach().to(torch.float16).cpu().numpy()
        if (s//B)%20==0: print(f"{e}/{N}",flush=True)
mm.flush(); del mm,m
(out/"acts_meta.json").write_text(json.dumps({"seq_len":SEQ,"screen_hs":hs,"n_seqs":N,"device":"mps-local"}))
print(f"done in {time.time()-t0:.0f}s")
