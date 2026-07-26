"""ACTMIX P2 — BASE-forward § 5.3 training cache at the paper's L15.

Builds `results/data_cache/<data_key>/acts.npy` for the datasource
`qwen_2_5_7b_instruct_medical_l15` (configs/data.yaml — the § 5.3
medical anchor) so the canonical runner can train the ACTMIX EM panel
on the PAPER's substrate convention: dictionaries trained on **BASE**
activations of the medical stream, applied to ORGANISM activations at
detection time (TRACKING.md § 1 pinned this against origin/final: the
paper's builder cached base activations; `lora_adapter` was consumed
only at Wang/detection time).

Corpus recipe: verbatim reuse of
`experiments.explorations.conversion_depth.build_em_train_cache
.build_corpus` (the origin/final `_load_corpus_cfierro` port —
cfierro/personality-qs-bad-medical-advice, seed-42 shuffle, chat
template, truncation+pad to 128, add_special_tokens False, 6000 rows,
≈59% eos-pad recorded property). The ONLY difference from that
builder: NO LoRA merge (base forward — the paper convention), and
capture hs16 only (resid_post L15, the paper's layer).

Run:  .venv/bin/python -m experiments.explorations.actmix_em.build_train_cache_base
Idempotent (skips if meta.json matches the data_key).
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from temp_bench.core.config import (
    compute_data_key,
    data_cache_dir,
    load_datasource,
)
from experiments.explorations.conversion_depth.build_em_train_cache import (
    BASE_MODEL,
    CORPUS_REPO,
    CORPUS_SEED,
    SEQ_LEN,
    BATCH,
    build_corpus,
)

DS_NAME = "qwen_2_5_7b_instruct_medical_l15"
HS_INDEX = 16   # resid_post L15


@torch.no_grad()
def main():
    spec = load_datasource(DS_NAME)
    assert spec.layer == 15 and spec.subject_model == BASE_MODEL, spec
    dk = compute_data_key(spec)
    cdir = data_cache_dir(dk)
    mpath = cdir / "meta.json"
    if mpath.exists() and json.loads(mpath.read_text())["data_key"] == dk:
        print(f"[cache] hit {DS_NAME} at {cdir}")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    token_ids = build_corpus(tok)
    n = int(token_ids.shape[0])
    print(f"[corpus] {n} × {SEQ_LEN} (pad frac "
          f"{(token_ids == tok.pad_token_id).float().mean():.3f})")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    d_model = int(model.config.hidden_size)

    cdir.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(
        cdir / "acts.npy", mode="w+", dtype=np.float16,
        shape=(n, SEQ_LEN, d_model))

    t0 = time.time()
    for s in range(0, n, BATCH):
        e = min(s + BATCH, n)
        out = model(token_ids[s:e].cuda(), output_hidden_states=True,
                    use_cache=False)
        mm[s:e] = out.hidden_states[HS_INDEX].to(torch.float16).cpu().numpy()
        if (s // BATCH) % 20 == 0:
            el = time.time() - t0
            print(f"  {e}/{n} ({el:.0f}s, est {el / max(e, 1) * n:.0f}s)",
                  flush=True)
    mm.flush()
    np.save(cdir / "token_ids.npy", token_ids.numpy())
    (cdir / "meta.json").write_text(json.dumps({
        "data_key": dk,
        "subject_model": spec.subject_model,
        "layer": spec.layer,
        "hookpoint": spec.hookpoint,
        "dataset": spec.dataset,
        "n_seqs": n,
        "seq_len": SEQ_LEN,
        "d_in": d_model,
        # provenance extras (ignored by build_refill):
        "substrate": "BASE forward (paper § 5.3 convention: "
                     "train-on-base, detect-on-organism)",
        "corpus_repo": CORPUS_REPO,
        "corpus_seed": CORPUS_SEED,
        "builder": "experiments/explorations/actmix_em/"
                   "build_train_cache_base.py",
    }, indent=2))
    print(f"[cache] wrote {DS_NAME} -> {cdir} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
