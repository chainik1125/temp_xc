"""Wave 1 steering grid: one container per steering source.

    uvx modal run --detach experiments/backtracking_steering_dsm/modal_steer.py

Each container loads DeepSeek-R1-Distill-Llama-8B, registers b1_steer_eval::_Hook
on layers[10], and sweeps the published 25-magnitude grid over the 20-prompt
local eval split via b1_steer_eval::_generate_panels (unmodified). Rows land in
backtracking_eval/steering/wave1/rows__<tag>.json on the `diffusion-txc` volume.
"""

import json
import pathlib
import sys

import modal

try:
    ROOT = pathlib.Path(__file__).resolve().parents[2]
except IndexError:
    ROOT = pathlib.Path("/work")
sys.path.insert(0, str(ROOT))          # repo root, for the protocol import below

app = modal.App("backtracking-steering-wave1")
vol = modal.Volume.from_name("diffusion-txc", create_if_missing=True)
hf_cache = modal.Volume.from_name("sae-deadlatent-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "numpy>=2.0", "transformers==4.46.3",
                 "accelerate", "sentencepiece", "huggingface_hub", "hf_transfer",
                 "pyyaml")
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(str(ROOT / "experiments" / "backtracking_steering_dsm"),
                   "/work/experiments/backtracking_steering_dsm",
                   ignore=["__pycache__"])
    .add_local_dir(str(ROOT / "experiments" / "backtracking_detection_dsm"),
                   "/work/experiments/backtracking_detection_dsm",
                   ignore=["__pycache__"])
    .add_local_dir(str(ROOT / "experiments" / "diffusion_txc" / "topk_vs_topkdiff"),
                   "/work/experiments/diffusion_txc/topk_vs_topkdiff",
                   ignore=["__pycache__"])
    .add_local_file(
        str(ROOT / "experiments" / "ward_backtracking_txc" / "b1_steer_eval.py"),
        "/work/experiments/ward_backtracking_txc/b1_steer_eval.py")
    .add_local_file(
        str(ROOT / "experiments" / "ward_backtracking_txc" / "architectures.py"),
        "/work/experiments/ward_backtracking_txc/architectures.py")
    .add_local_dir(str(ROOT / "temporal_crosscoders"), "/work/temporal_crosscoders",
                   ignore=["__pycache__"])
    .add_local_file(str(ROOT / "results" / "ward_backtracking" / "prompts.json"),
                    "/work/data/prompts.json")
    .add_local_file(str(ROOT / "results" / "ward_backtracking" / "dom_vectors.pt"),
                    "/work/data/dom_vectors.pt")
)

from experiments.backtracking_steering_dsm.protocol import (  # noqa: E402
    HF_ID, LAYER, MAGNITUDES, MAX_NEW, N_EVAL, PROMPT_SEED,
)

FEAT_DIR = "/vol/backtracking_eval/steering/features"
OUT_DIR = "/vol/backtracking_eval/steering/wave1"


def _build_all_sources():
    """-> list of source dicts, DoM baseline first (b1_steer_eval ordering)."""
    import sys
    sys.path.insert(0, "/work")
    from experiments.backtracking_steering_dsm import steer_core

    ref = steer_core.dom_source("/work/data/dom_vectors.pt")
    ref_norm = ref["raw_norm"]
    sources = [ref]
    for arm in steer_core.WAVE1_ARMS:
        p = pathlib.Path(FEAT_DIR) / f"{arm['name']}.npz"
        if not p.exists():
            raise FileNotFoundError(f"mined features missing: {p}")
        sources += steer_core.sources_for_arm(steer_core.load_features(p), ref_norm)
    return sources


@app.function(image=image, gpu="L40S", timeout=10800, memory=32768,
              volumes={"/vol": vol, "/hf": hf_cache},
              secrets=[modal.Secret.from_name("hf-token")])
def steer_one(tag: str, gen_bs: int = 25) -> dict:
    import sys
    import time

    sys.path.insert(0, "/work")
    import torch
    from experiments.ward_backtracking_txc.b1_steer_eval import (
        KEYWORD_RE, _Hook, _eval_prompts, _generate_panels, _kw_rate, _load_lm,
    )

    t0 = time.time()
    src = next(s for s in _build_all_sources() if s["tag"] == tag)
    out_path = pathlib.Path(OUT_DIR) / f"rows__{tag}.json"
    if out_path.exists():
        print(f"[resume] {out_path} exists", flush=True)
        return {"tag": tag, "skipped": True}

    prompts = _eval_prompts(pathlib.Path("/work/data/prompts.json"),
                            n=N_EVAL, seed=PROMPT_SEED)
    model, tok = _load_lm(HF_ID, "cuda")
    chat_texts = []
    for p in prompts:
        try:
            t = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}],
                                        tokenize=False, add_generation_prompt=True)
        except Exception:
            t = p["prompt"]
        chat_texts.append(t)

    hook = _Hook(src["vector"])
    handle = model.model.layers[LAYER].register_forward_hook(hook)
    try:
        # Outer = prompt, inner = magnitude (b1_steer_eval panel ordering). With
        # gen_bs = 25 each generate call is exactly one prompt's magnitude sweep,
        # so every row in a batch shares a prompt length and no padding is wasted.
        panel_prompts = [t for t in chat_texts for _ in MAGNITUDES]
        panel_mags = [float(m) for _ in prompts for m in MAGNITUDES]
        texts = _generate_panels(model, tok, hook, prompts=panel_prompts,
                                 mags_per_prompt=panel_mags,
                                 max_new_tokens=MAX_NEW, batch_size=gen_bs)
    finally:
        handle.remove()

    rows = []
    n_mags = len(MAGNITUDES)
    for p_i, prm in enumerate(prompts):
        for m_i, mag in enumerate(MAGNITUDES):
            txt = texts[p_i * n_mags + m_i]
            rows.append({
                "target": "reasoning", "source": tag, "arm": src["arm"],
                "arch": src["arch"], "hook": src["hook"],
                "feature_id": src["feature_id"], "mode": src["mode"],
                "magnitude": float(mag), "prompt_id": prm["id"],
                "category": prm.get("category"),
                "keyword_rate": _kw_rate(txt),
                "wait_count": len(KEYWORD_RE.findall(txt)),
                "n_words": len(txt.split()), "n_chars": len(txt),
                "n_tokens_retok": len(tok(txt, add_special_tokens=False)["input_ids"]),
                "text": txt,
            })

    out = {"rows": rows, "meta": {
        "tag": tag, "arm": src["arm"], "arch": src["arch"], "hook": src["hook"],
        "feature_id": src["feature_id"], "mode": src["mode"],
        "raw_decoder_norm": src["raw_norm"],
        "steered_vector_norm": float(src["vector"].norm()),
        "layer": LAYER, "magnitudes": MAGNITUDES, "max_new_tokens": MAX_NEW,
        "n_eval_prompts": len(prompts), "gen_batch_size": gen_bs,
        "target_model": HF_ID, "decoding": "greedy (do_sample=False)",
        "_runtime_s": round(time.time() - t0, 1)}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out))
    vol.commit()
    print(f"[done] {tag} rows={len(rows)} in {out['meta']['_runtime_s']}s",
          flush=True)
    return {"tag": tag, "n_rows": len(rows),
            "runtime_s": out["meta"]["_runtime_s"]}


@app.function(image=image, timeout=600, volumes={"/vol": vol})
def list_sources() -> list[dict]:
    return [{k: v for k, v in s.items() if k != "vector"}
            for s in _build_all_sources()]


@app.local_entrypoint()
def main():
    srcs = list_sources.remote()
    tags = [s["tag"] for s in srcs]
    print(f"[wave1] {len(tags)} sources:")
    for s in srcs:
        print(f"   {s['tag']:42s} arm={s['arm']:18s} mode={s['mode']}")
    res = list(steer_one.map(tags))
    print(json.dumps(res, indent=2))
