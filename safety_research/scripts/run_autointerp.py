"""
Run autointerp for the three architectures.

For each checkpoint in safety_research/results/checkpoints/:
  1. TopKFinder → top-K activating windows per feature
  2. TextContext  → decode windows via gemma-2-2b-it tokenizer
  3. Claude Haiku → 1-2 sentence explanations (prompt-cached)
  4. Save:  safety_research/results/autointerp/<arm>/feature_<id>.json
            safety_research/results/autointerp/<arm>/explanations.jsonl
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

NLP_DIR = "/home/cs29824/andre/temp_xc/temporal_crosscoders/NLP"
SAFETY_DIR = "/home/cs29824/andre/temp_xc/safety_research"
sys.path.insert(0, NLP_DIR)
os.chdir(NLP_DIR)

from autointerp import TopKFinder, TextContext, LocalGemmaBackend  # type: ignore # noqa: E402
from config import D_SAE, LAYER_SPECS, MODEL_NAME  # type: ignore # noqa: E402
from fast_models import FastStackedSAE, FastTemporalCrosscoder  # type: ignore  # noqa: E402

import wandb

CKPT_DIR = Path(SAFETY_DIR) / "results" / "checkpoints"
OUT_ROOT = Path(SAFETY_DIR) / "results" / "autointerp"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

LAYER = "mid_res"
TOP_K_EXAMPLES = 12         # top windows per feature
TOP_FEATURES = 150          # interpret 150 most-active features per arm
SAMPLE_CHAINS = 1500        # chains to scan
# user requested claude-haiku, but no working API key is available — fall
# back to local Gemma-2-2b-it via the existing LocalGemmaBackend.
EXPLAIN_MODEL = os.environ.get("EXPLAIN_MODEL", "google/gemma-2-2b-it")
MAX_CONCURRENT = 8

ARMS = [
    dict(arm="sae",  arch="stacked_sae", T=1, k=100),
    dict(arm="tsae", arch="stacked_sae", T=5, k=100),
    dict(arm="txc",  arch="txcdr",       T=5, k=100),
]


def load_arm_model(cfg: dict) -> torch.nn.Module:
    name = f"{cfg['arm']}__{LAYER}__k{cfg['k']}__T{cfg['T']}.pt"
    path = CKPT_DIR / name
    blob = torch.load(path, map_location="cpu", weights_only=True)
    state = blob["state_dict"] if "state_dict" in blob else blob
    d_in = LAYER_SPECS[LAYER]["d_act"]
    if cfg["arch"] == "stacked_sae":
        model = FastStackedSAE(d_in=d_in, d_sae=D_SAE, T=cfg["T"], k=cfg["k"])
    else:
        model = FastTemporalCrosscoder(d_in=d_in, d_sae=D_SAE, T=cfg["T"], k=cfg["k"])
    model.load_state_dict(state)
    model.eval()
    return model


SYSTEM = (
    "You are a meticulous AI researcher analyzing patterns in neural network features. "
    "You will see text examples that strongly activate a specific feature in a language model. "
    "Identify the concept, pattern, or topic this feature represents.\n\n"
    "Guidelines:\n"
    "- Focus on the COMMON pattern across examples, not one-off details.\n"
    "- 1-2 short sentences only.\n"
    "- Describe in linguistic / semantic terms, not numerics.\n\n"
    "After your explanation, on a new line emit a SAFETY tag:\n"
    "[SAFETY]: REFUSAL | DECEPTION | HARMFUL_CONTENT | BIAS | NONE\n"
    "Pick REFUSAL if the feature seems to track refusals, apologies, hedging, "
    "policy language, or 'cannot help' phrasing. DECEPTION for sycophancy, "
    "deceptive reasoning, or framing that hides intent. HARMFUL_CONTENT for "
    "violence/illegal/explicit content. BIAS for demographic stereotypes. "
    "Otherwise NONE.\n\n"
    "Format strictly:\n[EXPLANATION]: <one sentence>\n[SAFETY]: <tag>"
)


def parse_response(raw: str) -> dict:
    expl = ""
    safety = "NONE"
    for line in raw.splitlines():
        if line.upper().startswith("[EXPLANATION]:"):
            expl = line.split(":", 1)[1].strip()
        elif line.upper().startswith("[SAFETY]:"):
            safety = line.split(":", 1)[1].strip().upper()
            for tag in ("REFUSAL", "DECEPTION", "HARMFUL_CONTENT", "BIAS"):
                if tag in safety:
                    safety = tag
                    break
            else:
                safety = "NONE"
    if not expl:
        expl = raw.strip()[:240]
    return dict(explanation=expl, safety=safety, raw=raw)


def explain_features_local(records: list[dict], backend) -> list[dict]:
    """Sequentially call the local Gemma backend for each feature.

    Single asyncio loop rather than per-call asyncio.run() to avoid loop-
    creation overhead. Each call is still inherently sequential because the
    underlying generate() blocks on GPU work.
    """
    async def go() -> list[dict]:
        results: list[dict] = []
        for rec in tqdm(records, desc="explain"):
            examples = "\n".join(
                f"Example {i+1} (act={a:.2f}): {t[:280]}"
                for i, (t, a) in enumerate(zip(rec["top_texts"], rec["top_acts"]))
            )
            user = (
                f"Feature ID: {rec['feat']}\n\nActivating examples (window "
                f"between >>> and <<<):\n\n{examples}\n\nProvide your "
                "explanation and safety tag."
            )
            try:
                text = await backend.call(SYSTEM, user)
            except Exception as e:
                results.append(dict(feat=rec["feat"],
                                    explanation=f"ERROR: {e}",
                                    safety="NONE", error=str(e),
                                    top_texts=rec["top_texts"],
                                    top_acts=rec["top_acts"]))
                continue
            parsed = parse_response(text)
            parsed["feat"] = rec["feat"]
            parsed["top_texts"] = rec["top_texts"]
            parsed["top_acts"] = rec["top_acts"]
            results.append(parsed)
        return results
    return asyncio.run(go())


def run_arm(cfg: dict, backend) -> dict:
    arm = cfg["arm"]
    out_dir = OUT_ROOT / arm
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== AUTOINTERP {arm} ===")

    run = wandb.init(
        project="temporal-crosscoders-safety",
        name=f"autointerp_{arm}",
        tags=["safety", "autointerp", arm],
        config=dict(arm=arm, top_features=TOP_FEATURES,
                    top_k_examples=TOP_K_EXAMPLES,
                    sample_chains=SAMPLE_CHAINS,
                    explain_model=EXPLAIN_MODEL),
        reinit=True,
    )
    print(f"  wandb: {run.url}")

    model = load_arm_model(cfg)
    finder = TopKFinder(
        model=model, model_type=cfg["arch"], layer_key=LAYER,
        cache_dir=str(Path(NLP_DIR) / "cached_activations"),
        k=TOP_K_EXAMPLES, sample_chains=SAMPLE_CHAINS,
        chain_batch=64, device="cuda",
    )
    finder.run()

    feature_mass = {fid: sum(e.activation for e in ex)
                    for fid, ex in finder.results.items() if len(ex) >= 3}
    top_feat_ids = sorted(feature_mass, key=feature_mass.get,
                          reverse=True)[:TOP_FEATURES]
    print(f"  selected {len(top_feat_ids)} features (of "
          f"{len(feature_mass)} active >=3 examples)")
    wandb.log({"active_features": len(feature_mass),
               "selected_features": len(top_feat_ids)})

    text_ctx = TextContext(str(Path(NLP_DIR) / "cached_activations"), MODEL_NAME)

    records = []
    for fid in top_feat_ids:
        examples = finder.results[fid]
        texts = [text_ctx.get_window_text(e.chain_idx, e.window_start, model.T)
                 for e in examples]
        acts = [e.activation for e in examples]
        records.append(dict(feat=fid, top_texts=texts, top_acts=acts))

    # free GPU memory used by the SAE model before invoking the LLM
    model.to("cpu")
    torch.cuda.empty_cache()

    print(f"  calling local Gemma on {len(records)} features...")
    t0 = time.time()
    explained = explain_features_local(records, backend)
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.0f}s")

    safety_counts: dict[str, int] = {}
    for ex in explained:
        safety_counts[ex.get("safety", "NONE")] = safety_counts.get(ex.get("safety", "NONE"), 0) + 1

    out_file = out_dir / "explanations.jsonl"
    with open(out_file, "w") as f:
        for ex in explained:
            f.write(json.dumps(ex) + "\n")
    summary = dict(arm=arm, n_features=len(explained), elapsed_s=elapsed,
                   safety_counts=safety_counts, wandb_url=run.url)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  → {out_file}")
    print(f"  safety: {safety_counts}")
    wandb.summary.update({**safety_counts, "elapsed_s": elapsed,
                          "n_features": len(explained)})
    run.finish()
    return summary


def main() -> None:
    backend = LocalGemmaBackend(model_name=EXPLAIN_MODEL, device="cuda")
    summaries = []
    for cfg in ARMS:
        ckpt = CKPT_DIR / f"{cfg['arm']}__{LAYER}__k{cfg['k']}__T{cfg['T']}.pt"
        if not ckpt.exists():
            print(f"  SKIP {cfg['arm']}: missing {ckpt}")
            continue
        summaries.append(run_arm(cfg, backend))
    summary_path = OUT_ROOT / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nSUMMARY → {summary_path}")
    for s in summaries:
        print(f"  {s['arm']:6s}  n={s['n_features']:3d}  "
              f"safety={s['safety_counts']}  {s['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
