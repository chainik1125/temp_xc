# ⚑ PRESERVED FROM THE SESSION SCRATCHPAD, 2026-07-29 14:3x (mac-c),
# under explicit hub authorization. BODY IS VERBATIM — do not "clean it up".
#
# THIS is the script that actually built `cache_evalage_512`, and it is the
# ONLY thing in the tree that writes a `grid` key into `acts_meta.json`:
#
#     {"seq_len":512, "screen_hs":14, "n_seqs":3579, "device":"mps-local",
#      "corpus":"evalage", "grid":"elicit_evalage_screen_gemma2.npz"}
#
# ⚑ NO REPO CACHE BUILDER WRITES A GRID. Verified at source across all three
# (`evalage/cache_acts.py`, `facecmp/cache_local_mps.py`,
# `facecmp/cache_local_mps_512.py`) — none of them do, and two hardcode a
# `substrate` string the way `arm_test.py` did before `369f8c24c`. An earlier
# note of mine claimed `evalage/cache_acts.py` "already does"; it does not,
# and the hub caught that.
#
# WHY THIS FILE HAD TO BE COMMITTED: the grid-vs-cache guard added in
# `3740f2e16` only acts when `acts_meta.json` records a grid. That field
# existed on exactly ONE cache, written by this UNTRACKED script. Had the
# scratchpad expired first, rebuilding the cache from the repo would have
# dropped the key and the guard would have degraded from "checks one of five"
# to "checks none" — silently, since it no-ops on a missing field.
#
# Paths below point at the session scratchpad because that is where it ran.
# Left as-is deliberately: this is a RECORD of what produced a real artifact,
# not a general-purpose builder. The durable fix is to teach the three repo
# builders to stamp grid+substrate (see LOG 14:3x), and is deliberately
# unspent under the stop-ruling.

"""Cache gemma2_2b layer-14 activations for the `evalage` corpus at
SEQ_LEN=512. Local MPS, $0, no pods.

WHY THIS CORPUS AND NOT THE ONE I ALREADY SWEPT. Lever 3 says shrink the
floor's horizon `T + w`. My frozen sweep (`bf7aa3b5f`) moved T on
`retryesc_gen`, where **w = 25** — so even at T=16 the floor still sees
41 tokens, and the result was 0/5 KEEP-shaped. I then measured the `w`
half by counterfactual and reported "do not fund a narrow-event corpus"
(0.08 sigma, optimistically biased).

**That conclusion was about GENERATING a corpus. We already own one.**
`evalage` has **w = 13**, measured on 3 legs by `floor_predictor_test`.
Its floor horizon at T=16 is **29, not 41** — the `w` lever, applied for
free to a corpus that already exists, which is the one combination
neither of my two lever-3 scripts tested.

Both my prior scripts held the corpus fixed and moved a knob. This moves
the corpus and reuses the frozen instrument.

Faithful to the screen's geometry: same layer (SCREEN_HS), same
tokenizer grid the screen used (`elicit_evalage_screen_gemma2.npz`),
same chunking as `cache_acts.chunk_stream`, only SEQ_LEN raised 128->512
— the correction from the ceiling test, since at 128 the terciles sat
outside the model's context and the arm was asked about tokens it never
saw.
"""
import json, time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.explorations.task_hunt.replag.build_labels import MODELS
from experiments.explorations.task_hunt.replag.cache_acts import SCREEN_HS

SEQ_LEN = 512   # OVERRIDE: context that CONTAINS the label's range
SP = Path("/private/tmp/claude-501/-Users-octo-research-projects-agents-mac-c-temp-xc"
          "/660a6cf4-2ce0-4db9-8201-37e38db3e1bf/scratchpad")
G = Path("experiments/explorations/task_hunt/evalage/grids")
key = "gemma2_2b"
out = SP / "cache_evalage_512" / key
out.mkdir(parents=True, exist_ok=True)

z = np.load(G / "elicit_evalage_screen_gemma2.npz")
flat, off = z["token_ids"], z["doc_off"]
n_prefix = 1 if MODELS[key]["bos"] else 0
content = SEQ_LEN - n_prefix
tok = AutoTokenizer.from_pretrained(MODELS[key]["hf"])
bos = tok.bos_token_id

rows, didx = [], []
for d in range(len(off) - 1):
    seg = flat[off[d]:off[d + 1]]
    for s in range(0, len(seg) - content + 1, content):
        c = seg[s:s + content]
        rows.append(np.concatenate([[bos], c]) if n_prefix else c)
        didx.append(d)
ids = np.asarray(rows, dtype=np.int32)
N = ids.shape[0]
np.savez(out / "tokens.npz", ids=ids,
         doc_idx=np.asarray(didx, dtype=np.int32),
         n_prefix=np.int64(n_prefix))
print(f"{N} rows x {SEQ_LEN} (n_prefix={n_prefix}, content={content})", flush=True)

hs = SCREEN_HS[key]
m = AutoModelForCausalLM.from_pretrained(
    MODELS[key]["hf"], torch_dtype=torch.float16).to("mps").eval()
d_model = int(m.config.hidden_size)
print("d_model", d_model, "layer", hs, flush=True)
mm = np.lib.format.open_memmap(out / f"hs{hs}.npy", mode="w+",
                               dtype=np.float16, shape=(N, SEQ_LEN, d_model))
t0 = time.time()
B = 16
it = torch.from_numpy(ids.astype(np.int64))
with torch.no_grad():
    for s in range(0, N, B):
        e = min(s + B, N)
        o = m(it[s:e].to("mps"), output_hidden_states=True, use_cache=False)
        mm[s:e] = o.hidden_states[hs].detach().to(torch.float16).cpu().numpy()
        if (s // B) % 25 == 0:
            el = max(time.time() - t0, 1e-9)
            print(f"{e}/{N} ({e/el:.1f} r/s, eta {(N-e)/max(e/el,1e-9)/60:.1f}m)",
                  flush=True)
mm.flush()
del mm, m

a = np.load(out / f"hs{hs}.npy", mmap_mode="r")
smp = a[3, 100, :].astype(np.float32)
assert np.isfinite(smp).all() and np.linalg.norm(smp) > 0, "degenerate"
(out / "acts_meta.json").write_text(json.dumps(
    {"seq_len": SEQ_LEN, "screen_hs": hs, "n_seqs": N, "device": "mps-local",
     "corpus": "evalage", "grid": "elicit_evalage_screen_gemma2.npz"}))
print(f"done {N} rows in {time.time()-t0:.0f}s", flush=True)
