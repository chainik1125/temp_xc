"""Pre-sweep gates for the c7 steering replication.

Gate 1 (hook no-op): generations with no hook registered must be BYTE-IDENTICAL
to generations with the steering hook registered at magnitude 0.
Gate 2 (baseline agreement): mean genuine-backtracking count on the unsteered
generations must fall inside the sanity band [0.45, 0.85] around the published
0.656. The published value is over a 61-prompt set that is not in this repo
(only aggregate metrics are published for c7), so this is a band, not equality.

    uvx modal run experiments/backtracking_steering_dsm/modal_gates.py
"""

import json
import pathlib

import modal

try:
    ROOT = pathlib.Path(__file__).resolve().parents[2]
except IndexError:
    ROOT = pathlib.Path("/work")

app = modal.App("backtracking-steering-gates")
vol = modal.Volume.from_name("diffusion-txc", create_if_missing=True)
hf_cache = modal.Volume.from_name("sae-deadlatent-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "numpy>=2.0", "transformers==4.46.3",
                 "accelerate", "sentencepiece", "huggingface_hub", "hf_transfer",
                 "pyyaml", "anthropic")
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_file(
        str(ROOT / "experiments" / "ward_backtracking_txc" / "b1_steer_eval.py"),
        "/work/experiments/ward_backtracking_txc/b1_steer_eval.py")
    .add_local_file(
        str(ROOT / "experiments" / "ward_backtracking_txc" / "grade_backtracking.py"),
        "/work/experiments/ward_backtracking_txc/grade_backtracking.py")
    .add_local_file(str(ROOT / "experiments" / "ward_backtracking_txc" / "config.yaml"),
                    "/work/config.yaml")
    .add_local_file(str(ROOT / "results" / "ward_backtracking" / "prompts.json"),
                    "/work/data/prompts.json")
)

BAND = (0.45, 0.85)
PUBLISHED_BASELINE = 0.6557377049180327


@app.function(image=image, gpu="L40S", timeout=3600, memory=32768,
              volumes={"/vol": vol, "/hf": hf_cache},
              secrets=[modal.Secret.from_name("hf-token"),
                       modal.Secret.from_name("em-sprint-judges")])
def gates(max_new_tokens: int = 1500, gen_bs: int = 4) -> dict:
    import asyncio
    import os
    import sys
    import time

    sys.path.insert(0, "/work")
    import torch
    import yaml
    from experiments.ward_backtracking_txc.b1_steer_eval import (
        _Hook, _eval_prompts, _generate_panels, _load_lm,
    )
    from experiments.ward_backtracking_txc.grade_backtracking import judge_one

    t0 = time.time()
    cfg = yaml.safe_load(pathlib.Path("/work/config.yaml").read_text())
    hf_id = cfg["models"]["reasoning"]
    layer = int(cfg["steering"]["steering_layer"])
    prompts = _eval_prompts(pathlib.Path("/work/data/prompts.json"),
                            n=int(cfg["steering"]["n_eval_prompts"]),
                            seed=int(cfg["steering"]["seed"]))
    print(f"[gates] {len(prompts)} eval prompts, target={hf_id}, layer={layer}",
          flush=True)

    model, tok = _load_lm(hf_id, "cuda")
    chat_texts = []
    for p in prompts:
        try:
            t = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}],
                                        tokenize=False, add_generation_prompt=True)
        except Exception:
            t = p["prompt"]
        chat_texts.append(t)

    # An arbitrary but realistic steering direction; gate 1 only cares that a
    # magnitude of exactly 0 changes nothing.
    torch.manual_seed(0)
    vec = torch.randn(model.config.hidden_size)
    vec = vec / vec.norm() * 0.414          # dom_base_union norm
    hook = _Hook(vec)
    zeros = [0.0] * len(chat_texts)

    print("[gate1] generating WITHOUT hook", flush=True)
    texts_nohook = _generate_panels(model, tok, hook, chat_texts, zeros,
                                    max_new_tokens, gen_bs)
    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        print("[gate1] generating WITH hook at alpha=0", flush=True)
        texts_hook0 = _generate_panels(model, tok, hook, chat_texts, zeros,
                                       max_new_tokens, gen_bs)
    finally:
        handle.remove()

    n_diff = sum(1 for a, b in zip(texts_nohook, texts_hook0) if a != b)
    gate1 = {"identical": n_diff == 0, "n_prompts": len(chat_texts),
             "n_differing": n_diff}
    print(f"[gate1] {'PASS' if gate1['identical'] else 'FAIL'} "
          f"({n_diff}/{len(chat_texts)} differ)", flush=True)

    del model
    torch.cuda.empty_cache()

    async def _grade():
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        return await asyncio.gather(*[
            judge_one(client, p["prompt"], t)
            for p, t in zip(prompts, texts_nohook)])

    graded = asyncio.run(_grade())
    raw_counts = [int(g.get("genuine_count", -1)) for g in graded]
    counts = [c for c in raw_counts if c >= 0]      # -1 = unparseable judge reply
    gc_mean = sum(counts) / max(len(counts), 1)
    gate2 = {"gc_mean": gc_mean, "counts": raw_counts,
             "n_unparsed": sum(1 for c in raw_counts if c < 0),
             "published_baseline_61prompt": PUBLISHED_BASELINE,
             "band": list(BAND), "in_band": BAND[0] <= gc_mean <= BAND[1]}
    print(f"[gate2] gc_mean={gc_mean:.4f} band={BAND} "
          f"{'PASS' if gate2['in_band'] else 'FAIL'}", flush=True)

    out = {"gate1_hook_noop": gate1, "gate2_baseline_agreement": gate2,
           "target_model": hf_id, "layer": layer,
           "max_new_tokens": max_new_tokens,
           "n_eval_prompts": len(prompts),
           "note": "20-prompt local eval split; published 0.656 is over a "
                   "61-prompt set absent from this repo (c7 publishes only "
                   "aggregate metrics), so gate 2 is a band not an equality",
           "_runtime_s": round(time.time() - t0, 1),
           # Persist per-row generations, lengths and raw judge replies. The
           # first cut of this harness kept only the counts, which made the
           # coherence-filtered baseline impossible to compute without
           # regenerating, and left judge drift unauditable.
           "rows": [{"prompt_id": p["id"], "category": p.get("category"),
                     "text": t, "n_words": len(t.split()),
                     "n_chars": len(t),
                     "genuine_count": g.get("genuine_count", -1),
                     "judge_raw": g.get("raw", "")}
                    for p, t, g in zip(prompts, texts_nohook, graded)]}
    d = pathlib.Path("/vol") / "backtracking_eval" / "steering"
    d.mkdir(parents=True, exist_ok=True)
    (d / "gates.json").write_text(json.dumps(out, indent=1))
    vol.commit()
    print(f"ALL DONE in {out['_runtime_s']}s", flush=True)
    return out


@app.local_entrypoint()
def main():
    res = gates.remote()
    print(json.dumps({k: v for k, v in res.items()
                      if k != "sample_generation"}, indent=2))
