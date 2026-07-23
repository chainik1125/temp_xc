"""em-redo Phase A — organism-forward § 5.3 training caches at L9/L13/L15.

Builds the three `results/data_cache/<data_key>/acts.npy` caches the
canonical runner expects for the em-redo datasources (configs/data.yaml
`qwen_2_5_7b_organism_medical_l{9,13,15}`) in ONE forward sweep.

Why out-of-band: `temp_bench.data.real_lm.build_activation_cache` cannot
materialize this corpus (it treats `dataset` as a HF hub id keyed on a
`text` field; the cfierro chat corpus + chat-template forward has no
code path on this branch) and forwards only the BASE subject model. The
runner never auto-builds caches — `build_refill` just memmaps the keyed
`acts.npy` — so an out-of-band builder that writes the exact layout at
the exact keyed location is the plugin-clean route (no core edits).

Recipe (verbatim origin/final qwen_em.py `_load_corpus_cfierro`, checked
against `origin/final:purified/configs/datasources.yaml`):
  cfierro/personality-qs-bad-medical-advice, seed-42 row shuffle, chat
  template render, tokenize with truncation + pad to max_length=128,
  add_special_tokens=False, first 6000 usable rows. Forward WITHOUT an
  attention mask (paper parity; right-padding, causal model). ≈59% of
  token positions are eos-pad (median row 53 tokens) — a recorded
  property of the paper convention, reproduced deliberately.

DELIBERATE deviation (briefing-directed, see TRACKING.md § 1): the
forward runs through the MERGED organism (base + andyrdt LoRA,
`merge_and_unload`, the phase4_em_depth.py convention) instead of
origin/final's BASE forward — the g(ℓ) map lives on organism
activations and the trained panel must match that substrate.

Capture: hidden_states[k] for k ∈ {10, 14, 16} = resid_post L{9,13,15}
(output_hidden_states=True; hs[k+1] ≡ resid_post of layer k — the
equivalence to a block forward-hook was verified in the phase-3 work).

Run:  .venv/bin/python -m experiments.explorations.conversion_depth.build_em_train_cache
Idempotent per layer (skips if meta.json matches).
"""

from __future__ import annotations

import json
import random
import time

import numpy as np
import torch

from temp_bench.core.config import (
    compute_data_key,
    data_cache_dir,
    load_datasource,
)
from experiments.explorations.conversion_depth.em_redo_cells import DATASOURCES

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER = "andyrdt/Qwen2.5-7B-Instruct_bad-medical"
CORPUS_REPO = "cfierro/personality-qs-bad-medical-advice"
N_SEQS = 6000
SEQ_LEN = 128
CORPUS_SEED = 42
BATCH = 32
HS_FOR_LAYER = {9: 10, 13: 14, 15: 16}


def build_corpus(tok) -> torch.Tensor:
    """Verbatim port of origin/final `_load_corpus_cfierro`."""
    from datasets import load_dataset
    ds = load_dataset(CORPUS_REPO, split="train")
    rng = random.Random(CORPUS_SEED)
    rows = list(range(len(ds)))
    rng.shuffle(rows)
    out_ids = []
    for i in rows:
        if len(out_ids) >= N_SEQS:
            break
        try:
            messages = ds[i]["messages"]
        except Exception:
            continue
        try:
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            continue
        if not text or len(text) < 20:
            continue
        enc = tok(text, return_tensors="pt", truncation=True,
                  max_length=SEQ_LEN, padding="max_length",
                  add_special_tokens=False)
        out_ids.append(enc["input_ids"].squeeze(0))
    if len(out_ids) < N_SEQS:
        print(f"[corpus] WARNING only {len(out_ids)}/{N_SEQS} rows")
    return torch.stack(out_ids, dim=0)


@torch.no_grad()
def main():
    specs = {}
    todo = []
    for layer, ds_name in DATASOURCES.items():
        spec = load_datasource(ds_name)
        dk = compute_data_key(spec)
        cdir = data_cache_dir(dk)
        specs[layer] = (ds_name, dk, cdir)
        mpath = cdir / "meta.json"
        if mpath.exists() and json.loads(mpath.read_text())["data_key"] == dk:
            print(f"[cache] hit L{layer} at {cdir}")
        else:
            todo.append(layer)
    if not todo:
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    token_ids = build_corpus(tok)
    n = int(token_ids.shape[0])
    print(f"[corpus] {n} × {SEQ_LEN} (pad frac "
          f"{(token_ids == tok.pad_token_id).float().mean():.3f})")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    model = PeftModel.from_pretrained(model, ADAPTER)
    model = model.merge_and_unload().eval()
    d_model = int(model.config.hidden_size)

    mms = {}
    for layer in todo:
        _, dk, cdir = specs[layer]
        cdir.mkdir(parents=True, exist_ok=True)
        mms[layer] = np.lib.format.open_memmap(
            cdir / "acts.npy", mode="w+", dtype=np.float16,
            shape=(n, SEQ_LEN, d_model))

    t0 = time.time()
    for s in range(0, n, BATCH):
        e = min(s + BATCH, n)
        out = model(token_ids[s:e].cuda(), output_hidden_states=True,
                    use_cache=False)
        for layer in todo:
            mms[layer][s:e] = (out.hidden_states[HS_FOR_LAYER[layer]]
                               .to(torch.float16).cpu().numpy())
        if (s // BATCH) % 20 == 0:
            el = time.time() - t0
            print(f"  {e}/{n} ({el:.0f}s, est {el / max(e, 1) * n:.0f}s)",
                  flush=True)
    for layer in todo:
        mms[layer].flush()
        ds_name, dk, cdir = specs[layer]
        np.save(cdir / "token_ids.npy", token_ids.numpy())
        spec = load_datasource(ds_name)
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
            "organism_adapter": ADAPTER,
            "corpus_repo": CORPUS_REPO,
            "corpus_seed": CORPUS_SEED,
            "builder": "experiments/explorations/conversion_depth/"
                       "build_em_train_cache.py",
        }, indent=2))
        print(f"[cache] wrote L{layer} -> {cdir}")
    print(f"[cache] DONE in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
