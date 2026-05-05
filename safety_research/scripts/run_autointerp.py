"""
Run autointerp for SAE and TXC arms via Claude Haiku.

For each checkpoint in safety_research/results/checkpoints/:
  1. TopKFinder → top-K activating windows per feature
  2. TextContext  → decode windows via gemma-2-2b-it tokenizer
                    (special tokens like <bos> render literally)
  3. Claude Haiku 4.5 (async, semaphore=8) → 1-2 sentence explanations
  4. Save:  safety_research/results/autointerp/<arm>/feature_<id>.json
            safety_research/results/autointerp/<arm>/explanations.jsonl

Settings (TOP_FEATURES, TOP_K_EXAMPLES, SAMPLE_CHAINS, LAYER, k, T) match
the original three-arm run so this is a drop-in regeneration of the SAE
and TXC explanations. The TSAE arm is untouched.
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
from dotenv import load_dotenv

NLP_DIR = "/home/cs29824/andre/temp_xc/temporal_crosscoders/NLP"
SAFETY_DIR = "/home/cs29824/andre/temp_xc/safety_research"
sys.path.insert(0, NLP_DIR)
os.chdir(NLP_DIR)

# Load API key from safety_research/.env so ClaudeAPIBackend can authenticate.
load_dotenv(Path(SAFETY_DIR) / ".env")

from autointerp import TopKFinder, TextContext, ClaudeAPIBackend, LocalGemmaBackend  # type: ignore # noqa: E402
from config import D_SAE, LAYER_SPECS, MODEL_NAME  # type: ignore # noqa: E402
from fast_models import FastStackedSAE, FastTemporalCrosscoder  # type: ignore  # noqa: E402

import wandb

CKPT_DIR = Path(SAFETY_DIR) / "results" / "checkpoints"
OUT_ROOT = Path(SAFETY_DIR) / "results" / "autointerp"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

LAYER = "mid_res"
TOP_K_EXAMPLES = 12         # top windows per feature
# Cap the number of features interpreted per arm. None = interpret every
# feature with at least MIN_EXAMPLES top windows (the "all active" scope).
# Override with TOP_FEATURES env var to recover the old 150-feature behaviour.
_top_env = os.environ.get("TOP_FEATURES")
TOP_FEATURES: int | None = int(_top_env) if _top_env else None
MIN_EXAMPLES = 3            # feature must have at least this many top windows
SAMPLE_CHAINS = 1500        # chains to scan
EXPLAIN_MODEL = os.environ.get("EXPLAIN_MODEL", "claude-haiku-4-5-20251001")
# Concurrency=1 because at TPM-bound rates the only safe pattern is one
# in-flight call respecting the SDK's retry-after backoff. Bumping
# concurrency dramatically increases 429 churn without buying throughput.
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "1"))
MAX_RETRIES = 3

ALL_ARMS = [
    dict(arm="sae",  arch="stacked_sae", T=1, k=100),
    dict(arm="txc",  arch="txcdr",       T=5, k=100),
]
# ARMS env override: comma-separated list (e.g. "txc" to resume only TXC).
_arm_env = os.environ.get("ARMS")
if _arm_env:
    keep = {a.strip() for a in _arm_env.split(",") if a.strip()}
    ARMS = [c for c in ALL_ARMS if c["arm"] in keep]
else:
    ARMS = ALL_ARMS


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


def _build_user_prompt(rec: dict) -> str:
    examples = "\n".join(
        f"Example {i+1} (act={a:.2f}): {t[:280]}"
        for i, (t, a) in enumerate(zip(rec["top_texts"], rec["top_acts"]))
    )
    return (
        f"Feature ID: {rec['feat']}\n\nActivating examples (window between "
        f">>> and <<<; special tokens such as <bos>, <start_of_turn>, "
        f"<end_of_turn> appear literally and should be treated as real "
        f"tokens the feature can fire on):\n\n{examples}\n\nProvide your "
        "explanation and safety tag."
    )


def explain_features(records: list[dict], backend, out_file: Path) -> list[dict]:
    """Call the explainer backend for each feature, streaming results to
    `out_file` so partial progress survives crashes / rate-limit storms.

    Resume semantics: any feature already present in `out_file` (matched on
    feat id) is skipped. Successful new explanations are appended atomically.
    """
    # Load existing results into memory for resume. Drop empty/error
    # entries so they get retried this run.
    existing: dict[int, dict] = {}
    if out_file.exists():
        for line in open(out_file):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "feat" not in d:
                continue
            if not d.get("explanation") or d["explanation"].startswith("ERROR:"):
                continue
            existing[d["feat"]] = d

    pending = [r for r in records if r["feat"] not in existing]
    if existing:
        print(f"  resume: {len(existing)} features already done, "
              f"{len(pending)} pending")

    async def go() -> list[dict]:
        write_lock = asyncio.Lock()

        async def explain_one(rec: dict) -> dict:
            user = _build_user_prompt(rec)
            try:
                text = await backend.call(SYSTEM, user)
            except Exception as e:
                return dict(feat=rec["feat"],
                            explanation=f"ERROR: {e}",
                            safety="NONE", error=str(e),
                            top_texts=rec["top_texts"],
                            top_acts=rec["top_acts"])
            parsed = parse_response(text)
            parsed["feat"] = rec["feat"]
            parsed["top_texts"] = rec["top_texts"]
            parsed["top_acts"] = rec["top_acts"]

            # Stream the result to disk immediately. Append-only mode means
            # a crash leaves a coherent prefix; we de-dupe on next resume.
            if parsed.get("explanation"):
                async with write_lock:
                    with open(out_file, "a") as f:
                        f.write(json.dumps(parsed) + "\n")
                    existing[rec["feat"]] = parsed
            return parsed

        tasks = [asyncio.create_task(explain_one(r)) for r in pending]
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="explain"):
            await fut

        # Compose the final ordered list from `existing` (now containing
        # both pre-existing and newly-added results), preserving the input
        # ranking. Features that fail every retry stay missing from the file
        # so they get retried on the next run.
        order = {rec["feat"]: i for i, rec in enumerate(records)}
        results = [existing[fid] for fid in
                   sorted(existing, key=lambda x: order.get(x, 1 << 30))
                   if fid in order]
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
                    for fid, ex in finder.results.items()
                    if len(ex) >= MIN_EXAMPLES}
    ranked = sorted(feature_mass, key=feature_mass.get, reverse=True)
    top_feat_ids = ranked if TOP_FEATURES is None else ranked[:TOP_FEATURES]
    print(f"  selected {len(top_feat_ids)} features (of "
          f"{len(feature_mass)} active >={MIN_EXAMPLES} examples)")
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

    out_file = out_dir / "explanations.jsonl"
    print(f"  calling {EXPLAIN_MODEL} on {len(records)} features "
          f"(streaming to {out_file.name}, concurrency={MAX_CONCURRENT})...")
    n_calls_before = backend.n_calls
    n_errors_before = backend.n_errors
    t0 = time.time()
    explained = explain_features(records, backend, out_file)
    elapsed = time.time() - t0
    new_calls = backend.n_calls - n_calls_before
    new_errs = backend.n_errors - n_errors_before
    print(f"  done in {elapsed:.0f}s ({new_calls} calls, {new_errs} errors)")

    safety_counts: dict[str, int] = {}
    for ex in explained:
        safety_counts[ex.get("safety", "NONE")] = safety_counts.get(ex.get("safety", "NONE"), 0) + 1

    # Rewrite explanations.jsonl in canonical (mass-ranked) order. The file
    # is already populated by the streaming writer above; this just sorts
    # and de-duplicates it so downstream consumers (umap_meta) see clean
    # input.
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


def make_backend():
    """Pick backend based on EXPLAIN_MODEL.

    A '/' in the name means a HuggingFace local model; otherwise treat it
    as a Claude API model id. The Claude path requires ANTHROPIC_API_KEY
    in the environment (loaded above from safety_research/.env).
    """
    if "/" in EXPLAIN_MODEL:
        return LocalGemmaBackend(model_name=EXPLAIN_MODEL, device="cuda")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Either source safety_research/.env "
            "or override EXPLAIN_MODEL with a local HF model id."
        )
    return ClaudeAPIBackend(model=EXPLAIN_MODEL,
                            max_concurrent=MAX_CONCURRENT,
                            max_retries=MAX_RETRIES)


def main() -> None:
    backend = make_backend()
    summary_path = OUT_ROOT / "summary.json"

    # Preserve entries for arms not in this run (e.g. tsae) by merging into
    # the existing summary.json rather than overwriting it.
    existing: list[dict] = []
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            existing = []
    by_arm: dict[str, dict] = {s["arm"]: s for s in existing}

    rerun_summaries: list[dict] = []
    for cfg in ARMS:
        ckpt = CKPT_DIR / f"{cfg['arm']}__{LAYER}__k{cfg['k']}__T{cfg['T']}.pt"
        if not ckpt.exists():
            print(f"  SKIP {cfg['arm']}: missing {ckpt}")
            continue
        s = run_arm(cfg, backend)
        rerun_summaries.append(s)
        by_arm[s["arm"]] = s

    merged = list(by_arm.values())
    with open(summary_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nSUMMARY → {summary_path}")
    for s in merged:
        rerun_marker = " (rerun)" if s in rerun_summaries else ""
        print(f"  {s['arm']:6s}  n={s.get('n_features', '?'):3}  "
              f"safety={s.get('safety_counts', '?')}{rerun_marker}")


if __name__ == "__main__":
    main()
