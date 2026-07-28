"""Local MPS activation cache for the gpt2 leg — same geometry, same layer,
same dtype as the pod path in retryesc_gen/cache_acts.py. $0, no pod."""
import json, time, numpy as np, torch
from pathlib import Path
from transformers import AutoModelForCausalLM
from experiments.explorations.task_hunt.replag.build_labels import MODELS, SEQ_LEN
from experiments.explorations.task_hunt.replag.cache_acts import SCREEN_HS

SP=Path("/private/tmp/claude-501/-Users-octo-research-projects-agents-mac-c-temp-xc/660a6cf4-2ce0-4db9-8201-37e38db3e1bf/scratchpad/fakecache")
key="gpt2"; out=SP/key
z=np.load(out/"tokens.npz"); ids=z["ids"]; N=ids.shape[0]
hs=SCREEN_HS[key]
if (out/"acts_meta.json").exists(): print("cache hit"); raise SystemExit
model=AutoModelForCausalLM.from_pretrained(MODELS[key]["hf"],
      torch_dtype=torch.float32).to("mps").eval()
d=int(model.config.hidden_size)
mm=np.lib.format.open_memmap(out/f"hs{hs}.npy",mode="w+",dtype=np.float16,
                             shape=(N,SEQ_LEN,d))
t0=time.time(); B=128
ids_t=torch.from_numpy(ids.astype(np.int64))
with torch.no_grad():
    for s in range(0,N,B):
        e=min(s+B,N)
        o=model(ids_t[s:e].to("mps"),output_hidden_states=True,use_cache=False)
        mm[s:e]=o.hidden_states[hs].detach().to(torch.float16).cpu().numpy()
        if (s//B)%10==0:
            print(f"{e}/{N} ({e/max(time.time()-t0,1e-9):.0f} rows/s)",flush=True)
mm.flush(); del mm,model
a=np.load(out/f"hs{hs}.npy",mmap_mode="r")
smp=a[min(3,N-1),100,:].astype(np.float32)
assert np.isfinite(smp).all() and np.linalg.norm(smp)>0,"degenerate activations"
(out/"acts_meta.json").write_text(json.dumps({"model_id":MODELS[key]["hf"],
  "substrate":"elicit_retryesc_gen_v1","screen_hs":hs,"n_seqs":N,
  "seq_len":SEQ_LEN,"d_model":d,"dtype":"float16","device":"mps-local"}))
print(f"done {N} rows in {time.time()-t0:.0f}s -> {out}/hs{hs}.npy")
