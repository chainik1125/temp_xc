"""Held-out C7 steering, with durable generation and separate API judging.

Stages: split -> phase1 -> mine -> generate(validation and test_candidates)
-> export-judge(validation) -> import-judge(validation) -> select -> unlock
-> export-judge(test) -> import-judge(test) -> summarize. Each arm gets its own
workspace; all arms share the split and phase1 files. No TXC-pro is supported.
Generation/prompt/hook semantics come from the pinned historical C7 module.
This is a NEW validation/test protocol, not a replay of the published peak.
No command makes a paid API request; exported batches are for later judging.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

PROTOCOL = "c7-heldout-steering-2026-09-v1"
SOURCE_COMMIT = "284a8bf5e3e5a7cc094dd68c6fa5a92a9fd4eec3"
SENTENCE_SHA = "1656f6be2cd85fb85c8b246b9b27933f73ef40cfaac84078169dfd3bbbe27810"
REASONING_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
REASONING_REVISION = "6a6f4aa4197940add57724a7707d069478df56b1"
MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
MATH500_SHA = "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132"
GENERATION_RECIPE = "canonical-chat-prompt-ids-v2: add_special_tokens=False, no silent truncation, greedy bf16"
JUDGE_MODEL_PLACEHOLDER = "__CHOOSE_MODEL_BEFORE_SUBMISSION__"
STEERING_FILES = {
    "src/temp_bench/case_studies/backtracking.py": "5e5ef44e671616b7dfd9b7fd051f69716c119af6058664514a45ddb88d6b85b2",
    "src/temp_bench/case_studies/steering.py": "5b9a68c83f8e0e559cf4535984bd1da4b530cd2d796aeda47670855fbcb375cc",
    "results/c7_backtracking/stage_a/prompts.json": "f718d76c1be63bddb83cfb7a9fe03ebde0bf5036a02defb1addc104f8829dd6a",
    "results/c7_backtracking/stage_a/traces.json": "dc6513e7d3d104de096bb46f52245f5794406ecf745ad5477804e3a2e4e0f9cd",
    "results/c7_backtracking/stage_a/dom_vectors.pt": "8561b827b148839eff45eeac3b8808d75082d3552950f938bd88a9379e578ff8",
    "results/c7_backtracking/aniket_reference/cut25/flip_matrix.parquet": "522ae7e8018244306264e30e8a3b23267b538d9f2da85906ed9a26c02031abff",
}
MAGNITUDES = [-12., -8., -4., 0., 4., 8., 12.]
ARMS = {
    "topk_last": ("topk_sae", "last"),
    "topk_mean": ("topk_sae", "mean"),
    "topk_max": ("topk_sae", "max"),
    "tsae_last": ("tsae_paper", "last"),
    "tsae_mean": ("tsae_paper", "mean"),
    "tsae_max": ("tsae_paper", "max"),
    "txc_base": ("txc_base", "shared"),
    "stacked_atoms": ("stacked_sae", "atoms"),
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def frozen_json(path: Path, value: Any) -> None:
    """Never silently reuse or overwrite a different run identity."""
    if path.exists():
        if read(path) != value:
            raise ValueError(f"existing artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(value, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def historical(root: Path):
    from train import verify_historical_source
    verify_historical_source(root)
    for relative, expected in STEERING_FILES.items():
        if digest(root / relative) != expected:
            raise ValueError(f"historical steering source/artifact mismatch: {relative}")
    os.environ["TEMP_BENCH_ROOT"] = str(root.resolve())
    sys.path.insert(0, str(root.resolve() / "src"))
    from temp_bench.case_studies import backtracking
    return backtracking


def make_split(wrong: list[str], correct: list[str], mining: list[str]) -> dict:
    if len(wrong) != 31 or len(correct) != 30:
        raise ValueError("expected the archived 31-wrong/30-correct cohort")
    if any(len(set(q)) != len(q) for q in (wrong, correct, mining)):
        raise ValueError("duplicate question IDs")
    groups = [set(wrong), set(correct), set(mining)]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError("mining/steering groups or correctness strata overlap")
    def order(qids):
        return sorted(qids, key=lambda q: hashlib.sha256((PROTOCOL + q).encode()).hexdigest())
    validation = sorted(order(wrong)[:10] + order(correct)[:10])
    test = sorted(order(wrong)[10:] + order(correct)[10:])
    return {
        "protocol": PROTOCOL, "historical_commit": SOURCE_COMMIT,
        "mining_qids": sorted(mining), "validation_qids": validation,
        "test_qids": test, "historically_wrong": sorted(wrong),
        "historically_correct": sorted(correct),
        "limitation": "61 historically inspected questions; held out for this rerun's tuning, not a previously unseen scientific test cohort",
    }


def normalized_problem_hash(text):
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def mining_text_overlap(math500, prompts, traces):
    """Compare recovered problem text, not incompatible question-ID namespaces."""
    prompt_by_id = {row["id"]: row["prompt"] for row in prompts}
    trace_by_id = {row["question_id"]: row["prompt"] for row in traces}
    if len(prompt_by_id) != 300 or len(trace_by_id) != 300 or set(prompt_by_id) != set(trace_by_id):
        raise ValueError("expected 300 uniquely identified Stage-A prompts and traces")
    if any(normalized_problem_hash(text) != normalized_problem_hash(trace_by_id[qid])
           for qid, text in prompt_by_id.items()):
        raise ValueError("Stage-A prompt text disagrees with recovered traces")
    mining_hashes = {}
    for qid, problem in prompt_by_id.items():
        mining_hashes.setdefault(normalized_problem_hash(problem), []).append(qid)
    matches = [{"math500_qid": row["unique_id"], "mining_qids": sorted(mining_hashes[h])}
               for row in math500 if (h := normalized_problem_hash(row["problem"])) in mining_hashes]
    return {"normalization": "Unicode NFKC, casefold, collapse whitespace; preserve mathematical punctuation",
        "n_mining_prompts": len(prompt_by_id), "n_math500_problems": len(math500),
        "normalized_exact_matches": sorted(matches, key=lambda row: row["math500_qid"]),
        "prompt_trace_text_agreement": True,
        "scope": "Exact normalized text overlap checked against every recovered mining prompt. Paraphrase equivalence and undocumented earlier experiments are not established by this check."}


def fresh_question_split(all_qids, mining, historical_qids, text_audit=None):
    if len(all_qids) != 500 or len(set(all_qids)) != 500:
        raise ValueError("expected the pinned 500 unique MATH-500 questions")
    text_excluded = {row["math500_qid"] for row in (text_audit or {}).get("normalized_exact_matches", [])}
    excluded = set(historical_qids) | set(mining) | text_excluded
    available = sorted(set(all_qids) - excluded,
        key=lambda q: hashlib.sha256((PROTOCOL + ":fresh:" + q).encode()).hexdigest())
    if len(available) < 120:
        raise ValueError("fewer than 120 unused questions remain")
    return {"protocol": PROTOCOL, "historical_commit": SOURCE_COMMIT,
        "mining_qids": sorted(mining), "validation_qids": sorted(available[:20]),
        "test_qids": sorted(available[20:120]), "historical_steering_qids_excluded": sorted(historical_qids),
        "math500_sha256": MATH500_SHA, "math500_revision": MATH500_REVISION,
        "mining_text_audit": text_audit, "mining_text_overlap_qids_excluded": sorted(text_excluded),
        "selection": "deterministic question-ID hash order before generating or scoring; no correctness filtering",
        "limitation": "Disjoint from the published61 cohort and normalized exact matches to recovered mining text. Does not establish absence of paraphrases or undocumented prior experiments."}


def split_command(args):
    bt = historical(args.historical_root)
    if digest(args.sentence_acts) != SENTENCE_SHA:
        raise ValueError("sentence-activation artifact hash differs")
    with np.load(args.sentence_acts, allow_pickle=True) as z:
        mining = sorted({str(k).split("|", 1)[0] for k in z["keys"]})
    cohort = bt.build_cohort()
    if digest(args.math500) != MATH500_SHA:
        raise ValueError("MATH-500 file does not match the pinned revision")
    math500 = rows(args.math500)
    stage_a = args.historical_root / "results/c7_backtracking/stage_a"
    text_audit = mining_text_overlap(math500, read(stage_a / "prompts.json"), read(stage_a / "traces.json"))
    text_audit["prompts_sha256"] = digest(stage_a / "prompts.json")
    text_audit["traces_sha256"] = digest(stage_a / "traces.json")
    frozen_json(args.output, fresh_question_split(
        [r["unique_id"] for r in math500], mining, cohort.all, text_audit))


def write_progress(path, *, expected, completed, identity, batches_this_invocation):
    from train import atomic_json
    atomic_json(path, {"status": "complete" if completed == expected else "partial",
        "expected_records": expected, "completed_records": completed,
        "identity_sha256": canonical_hash(identity), "batches_this_invocation": batches_this_invocation})


def limited_starts(count, batch_size, max_batches):
    if batch_size < 1 or (max_batches is not None and max_batches < 1):
        raise ValueError("batch-size and optional max-batches must be positive")
    starts = range(0, count, batch_size)
    return starts if max_batches is None else starts[:max_batches]


def load_reasoning_model():
    """Same bf16/eval semantics as the historical loader, now revision-pinned."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(REASONING_MODEL, revision=REASONING_REVISION, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(REASONING_MODEL,
        revision=REASONING_REVISION, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tok


def canonical_prompt_ids(tok, prompt):
    chat = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  tokenize=False, add_generation_prompt=True)
    ids = tok(chat, add_special_tokens=False)["input_ids"]
    if len(ids) > 2048:
        raise ValueError("prompt exceeds 2048 tokens; truncation is forbidden in this protocol")
    if not ids or ids[0] != tok.bos_token_id or (len(ids) > 1 and ids[1] == tok.bos_token_id):
        raise ValueError("canonical chat prompt must begin with exactly one BOS token")
    return ids


def generate_token_panels(model, tok, input_ids, budgets, *, hook=None, magnitudes=None):
    """One explicit tokenization for phase1 and continuation; retain actual IDs."""
    import torch
    from transformers import GenerationConfig
    if not input_ids or len(input_ids) != len(budgets) or min(budgets) < 1:
        raise ValueError("nonempty input panels and positive aligned budgets required")
    width = max(map(len, input_ids))
    x = torch.full((len(input_ids), width), tok.pad_token_id, dtype=torch.long, device=model.device)
    mask = torch.zeros_like(x)
    for index, ids in enumerate(input_ids):
        x[index, -len(ids):] = torch.tensor(ids, dtype=torch.long, device=model.device)
        mask[index, -len(ids):] = 1
    if hook is not None:
        if magnitudes is None or len(magnitudes) != len(input_ids):
            raise ValueError("hook magnitudes must align with input panels")
        hook.magnitudes = torch.tensor(magnitudes, dtype=torch.float32)
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tok.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    try:
        with torch.inference_mode():
            output = model.generate(input_ids=x, attention_mask=mask,
                generation_config=GenerationConfig(max_new_tokens=max(budgets), do_sample=False,
                    temperature=1., pad_token_id=tok.pad_token_id, eos_token_id=eos))
        result = []
        for row, budget in zip(output, budgets):
            ids = row[width:width + budget].tolist()
            # Keep the first genuine EOS, discard only padding after termination.
            end = next((i + 1 for i, value in enumerate(ids) if value in eos_ids), len(ids))
            result.append(ids[:end])
        return result
    finally:
        if hook is not None:
            hook.magnitudes = None


def zero_intervention_check(model, tok, bt, prompt_ids, generated_ids):
    import torch
    hook = bt.SteeringHook(torch.ones(model.config.hidden_size, dtype=torch.float32))
    handle = model.model.layers[10].register_forward_hook(hook)
    try:
        rerun = generate_token_panels(model, tok, prompt_ids, [len(ids) for ids in generated_ids],
                                     hook=hook, magnitudes=[0.] * len(prompt_ids))
        cuts = [bt.cut25_token_position(ids, fraction=.25) for ids in generated_ids]
        inputs = [p + g[:cut] for p, g, cut in zip(prompt_ids, generated_ids, cuts)]
        expected = [g[cut:] for g, cut in zip(generated_ids, cuts)]
        continuation = generate_token_panels(model, tok, inputs, [len(ids) for ids in expected],
                                            hook=hook, magnitudes=[0.] * len(inputs))
    finally:
        handle.remove()
    no_op = [a == b for a, b in zip(generated_ids, rerun)]
    suffix = [a == b for a, b in zip(expected, continuation)]
    return {"generation_recipe": GENERATION_RECIPE, "zero_hook_exact_noop": no_op,
        "zero_continuation_exact_suffix": suffix, "prefix_cuts": cuts,
        "continued_token_ids": continuation,
        "interpretation": "Zero hook must be an exact no-op. Cut/continue recomputes the prefix; BF16 prefill/cache and batch shapes can change greedy suffix tokens even with identical canonical prompt IDs."}


def phase1_command(args):
    bt = historical(args.historical_root)
    split = read(args.split_file)
    qids = split["validation_qids"] + split["test_qids"]
    if digest(args.math500) != split["math500_sha256"]:
        raise ValueError("MATH-500 input differs from registered split")
    math500 = {r["unique_id"]: r for r in rows(args.math500)}
    identity = {"protocol": PROTOCOL, "split_sha256": digest(args.split_file),
                "max_new_tokens": args.max_new_tokens, "batch_size": args.batch_size,
                "generation_recipe": GENERATION_RECIPE,
                "model": REASONING_MODEL, "model_revision": REASONING_REVISION}
    args.workspace.mkdir(parents=True, exist_ok=True)
    frozen_json(args.workspace / "phase1_identity.json", identity)
    checkpoint = args.workspace / "phase1_unsteered.jsonl"
    prior_rows = rows(checkpoint)
    existing = {r["unique_id"]: r for r in prior_rows}
    if len(existing) != len(prior_rows):
        raise ValueError("duplicate phase1 records")
    if set(existing) - set(qids):
        raise ValueError("phase1 cache includes unexpected questions")
    todo = [q for q in qids if q not in existing]
    starts = limited_starts(len(todo), args.batch_size, args.max_batches)
    batches = 0
    if todo:
        model, tok = load_reasoning_model()
        for start in starts:
            current = todo[start:start + args.batch_size]
            prompts = [bt._build_prompt(math500[q]["problem"]) for q in current]
            prompt_ids = [canonical_prompt_ids(tok, prompt) for prompt in prompts]
            token_ids = generate_token_panels(model, tok, prompt_ids, [args.max_new_tokens] * len(current))
            for q, prompt, ids in zip(current, prompt_ids, token_ids):
                record = {"unique_id": q, "problem": math500[q]["problem"],
                    "ground_truth": math500[q]["answer"], "unsteered_text": bt._fix_byte_decode(tok.decode(ids, skip_special_tokens=True)),
                    "prompt_token_ids": prompt, "generation_recipe": GENERATION_RECIPE,
                    "unsteered_token_ids": ids, "unsteered_token_count": len(ids)}
                append(checkpoint, record)
                existing[q] = record
            if args.verify_zero and batches == 0:
                check = zero_intervention_check(model, tok, bt, prompt_ids, token_ids)
                check["question_ids"] = current
                frozen_json(args.workspace / "phase1_zero_check.json", check)
                if not all(check["zero_hook_exact_noop"]):
                    raise RuntimeError("zero-hook no-op check failed; inspect phase1_zero_check.json")
            batches += 1
            write_progress(args.workspace / "phase1_progress.json", expected=len(qids), completed=len(existing),
                identity=identity, batches_this_invocation=batches)
    if len(existing) == len(qids):
        frozen_json(args.workspace / "phase1_unsteered.json", [existing[q] for q in qids])
    write_progress(args.workspace / "phase1_progress.json", expected=len(qids), completed=len(existing),
        identity=identity, batches_this_invocation=batches)


def validate_checkpoint_receipt(config, arch, source_sha, weight_sha):
    from train import ACT_CACHE_KEY, ACT_CACHE_SHA256, PROTOCOL_VERSION
    required = {"arch": arch, "status": "complete", "n_steps_completed": 300000,
        "protocol_version": PROTOCOL_VERSION, "historical_commit": SOURCE_COMMIT,
        "historical_source_sha256": source_sha, "model_sha256": weight_sha,
        "checkpoint_sha256": weight_sha, "act_cache_key": ACT_CACHE_KEY,
        "act_cache_sha256": ACT_CACHE_SHA256}
    bad = {key: (expected, config.get(key)) for key, expected in required.items() if config.get(key) != expected}
    if bad:
        raise ValueError(f"steering requires a verified completed 300K receipt: {bad}")
    if config.get("smoke") or config.get("is_smoke"):
        raise ValueError("smoke training cannot enter production steering")
    hparams = config.get("hparams", {})
    training = config.get("training_cfg", {})
    if training.get("batch_size") != 1024 or training.get("n_steps") != 300000 or hparams.get("k_pos") != 20:
        raise ValueError("production steering requires frozen batch=1024, steps=300000 and k_pos=20")
    if not config.get("train_key") or config.get("seed") not in (1, 2, 42):
        raise ValueError("missing training identity or unregistered seed")
    if config.get("d_sae") != 32768 and not (arch == "tsae_paper" and config.get("d_sae") == 16384):
        raise ValueError("unregistered dictionary width")
    if arch in ("txc_base", "stacked_sae") and hparams.get("T") != 5:
        raise ValueError("production temporal dictionary requires T=5")


def load_dictionary(checkpoint: Path, arch: str, device: str, historical_root: Path):
    import torch
    from safetensors.torch import load_file
    from temp_bench.config import instantiate_arch, load_arch
    from train import verify_historical_source
    config = read(checkpoint / "config.json")
    actual_sha = digest(checkpoint / "model.safetensors")
    validate_checkpoint_receipt(config, arch, verify_historical_source(historical_root), actual_sha)
    spec = load_arch(arch, component="c7")
    hparams = dict(spec.hparams)
    hparams.update(config.get("hparams", {}))
    if "d_sae" in config:
        hparams["d_sae"] = int(config["d_sae"])
    spec = spec.model_copy(update={"hparams": hparams})
    state = load_file(str(checkpoint / "model.safetensors"), device="cpu")
    model = instantiate_arch(spec, d_in=4096).to(dtype=torch.float32)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    if arch == "tsae_paper" and float(model.threshold.item()) < 0:
        raise ValueError("T-SAE inference threshold is uninitialized")
    return model, config


def reduce_codes(z, reduction: str):
    """Position-specific Stacked features must retain distinct identities."""
    if reduction == "atoms":
        if z.ndim != 3:
            raise ValueError("Stacked code must have a position dimension")
        return z.abs().flatten(1)
    if z.ndim == 2:
        if reduction != "shared":
            raise ValueError("unexpected rank-two token SAE code")
        return z.abs()
    if reduction == "last":
        return z[:, -1].abs()
    if reduction == "mean":
        return z.abs().mean(1)
    return z.abs().amax(1)


def decoder_atom(model, reduction: str, feature: int):
    if reduction == "atoms":
        width = model.saes[0].W_dec.shape[1]
        position, local_feature = divmod(feature, width)
        return model.saes[position].W_dec[:, local_feature].detach().float(), {
            "position": position, "feature_id": local_feature, "flat_feature_id": feature}
    return model.decoder_directions()[feature].detach().float(), {"feature_id": feature}


def mine_command(args):
    import torch
    bt = historical(args.historical_root)
    split = read(args.split_file)
    arch, reduction = ARMS[args.arm]
    if digest(args.sentence_acts) != SENTENCE_SHA:
        raise ValueError("sentence artifact hash mismatch")
    model, config = load_dictionary(args.checkpoint_dir, arch, args.device, args.historical_root)
    with np.load(args.sentence_acts, allow_pickle=True) as z:
        acts, labels = z["X"], z["is_bt"].astype(bool)
        qids = np.asarray([str(k).split("|", 1)[0] for k in z["keys"]])
    keep = np.isin(qids, split["mining_qids"])
    if not keep.all() or set(qids) & set(split["validation_qids"] + split["test_qids"]):
        raise ValueError("mining cohort differs or overlaps steering evaluation")
    if not labels.any() or labels.all():
        raise ValueError("mining requires both labels")
    window = 1 if reduction == "last" else 5
    if arch in {"txc_base", "stacked_sae"}:
        window = int(model.config.T)
    if window != 5 and reduction != "last":
        raise ValueError("registered steering comparison uses T=5")
    sums = [None, None]
    counts = [0, 0]
    parameter = next(model.parameters())
    with torch.no_grad():
        for start in range(0, len(acts), args.batch_size):
            batch = torch.as_tensor(np.ascontiguousarray(acts[start:start + args.batch_size, -window:]),
                                    dtype=parameter.dtype, device=parameter.device)
            codes = reduce_codes(model.encode(batch), reduction).float()
            current_labels = labels[start:start + args.batch_size]
            for label in (0, 1):
                mask = torch.as_tensor(current_labels == bool(label), device=parameter.device)
                addition = codes[mask].sum(0).cpu().double()
                sums[label] = addition if sums[label] is None else sums[label] + addition
                counts[label] += int(mask.sum())
    selectivity = sums[1] / counts[1] - sums[0] / counts[0]
    fid = int(torch.argmax(selectivity))
    vector, feature = decoder_atom(model, reduction, fid)
    stage_a = bt.load_stage_a()
    reference = stage_a.dom_vectors["base"]["union"].float()
    ref_norm = float(reference.norm())
    if not np.isfinite(ref_norm) or ref_norm <= 0:
        raise ValueError("invalid reference direction norm")
    if args.random_control_seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(args.random_control_seed)
        vector = torch.randn(vector.shape, generator=generator)
    if not torch.isfinite(vector).all() or float(vector.norm()) < 1e-8:
        raise ValueError("selected decoder atom is invalid or cancels to zero")
    vector = vector.cpu() / vector.cpu().norm() * ref_norm
    payload = {"protocol": PROTOCOL, "arm": args.arm, "arch": arch, "reduction": reduction,
        "seed": config["seed"], "checkpoint_config": config,
        "checkpoint_sha256": digest(args.checkpoint_dir / "model.safetensors"),
        "split_sha256": digest(args.split_file), "sentence_sha256": SENTENCE_SHA,
        "feature": feature, "selectivity": float(selectivity[fid]),
        "feature_selection": "largest positive-minus-negative mean on mining questions; fixed top1",
        "mining_positive": counts[1], "mining_negative": counts[0], "window": window,
        "decoder_rule": "position-specific atom" if reduction == "atoms" else "historical decoder_directions (TXC averages temporal footprint)",
        "reference_norm": ref_norm, "random_control_seed": args.random_control_seed,
        "dictionary_inference_dtype": "float32", "reasoning_inference_dtype": "bfloat16",
        "hook_scope": "layer10 resid_post: all prompt and prefix positions during prefill, then every generated token",
        "vector": vector.tolist(), "cut_fraction": .25,
        "validation_magnitudes": MAGNITUDES,
        "test_candidate_policy": "Generate the same fixed grid blind; only validation-selected dose and zero may be judged on test"}
    frozen_json(args.workspace / "direction.json", payload)


def generation_identity(args, direction, split):
    identity = {"protocol": PROTOCOL, "direction_sha256": digest(args.workspace / "direction.json"),
        "split_sha256": digest(args.split_file), "phase1_sha256": digest(args.phase1),
        "partition": args.partition, "batch_size": args.batch_size,
        "generation_recipe": GENERATION_RECIPE,
        "hook_scope": direction["hook_scope"],
        "model": REASONING_MODEL, "model_revision": REASONING_REVISION}
    if direction["split_sha256"] != identity["split_sha256"]:
        raise ValueError("direction was mined under a different split")
    if args.partition in ("validation", "test_candidates"):
        magnitudes = direction["validation_magnitudes"]
    else:
        selected = read(args.workspace / "selection.json")
        if selected["direction_sha256"] != identity["direction_sha256"]:
            raise ValueError("selection refers to another direction")
        if selected["phase1_sha256"] != identity["phase1_sha256"]:
            raise ValueError("test phase1 differs from validation phase1")
        for source, key in (("generations.jsonl", "validation_generations_sha256"),
                            ("judgments.jsonl", "validation_judgments_sha256")):
            if digest(args.workspace / "validation" / source) != selected[key]:
                raise ValueError("validation data changed after selection was frozen")
        identity["selection_sha256"] = digest(args.workspace / "selection.json")
        magnitudes = sorted({0., float(selected["magnitude"])})
    identity["magnitudes"] = magnitudes
    identity["qids"] = split[("test" if args.partition == "test_candidates" else args.partition) + "_qids"]
    return identity


def generate_command(args):
    import torch
    bt = historical(args.historical_root)
    direction = read(args.workspace / "direction.json")
    split = read(args.split_file)
    identity = generation_identity(args, direction, split)
    outdir = args.workspace / args.partition
    frozen_json(outdir / "generation_identity.json", identity)
    phase1 = {r["unique_id"]: r for r in read(args.phase1)}
    if not set(identity["qids"]).issubset(phase1):
        raise ValueError("phase1 is missing registered questions")
    output = outdir / "generations.jsonl"
    existing = rows(output)
    identity_hash = canonical_hash(identity)
    if any(r.get("generation_identity") != identity_hash for r in existing):
        raise ValueError("generation identity mismatch")
    have = {(r["question_id"], r["magnitude"]) for r in existing}
    wanted = [(q, float(m)) for q in identity["qids"] for m in identity["magnitudes"]]
    if len(have) != len(existing):
        raise ValueError("duplicate generation records")
    if have - set(wanted):
        raise ValueError("unexpected generation records")
    todo = [key for key in wanted if key not in have]
    starts = limited_starts(len(todo), args.batch_size, args.max_batches)
    batches = 0
    if not todo:
        write_progress(outdir / "progress.json", expected=len(wanted), completed=len(have), identity=identity, batches_this_invocation=0)
        return
    model, tok = load_reasoning_model()
    hook = bt.SteeringHook(torch.tensor(direction["vector"], dtype=torch.float32))
    handle = model.model.layers[10].register_forward_hook(hook)
    try:
        for start in starts:
            chunk = todo[start:start + args.batch_size]
            prompts, prefixes, budgets, prompt_tokens = [], [], [], []
            for qid, _ in chunk:
                row = phase1[qid]
                if row.get("generation_recipe") != GENERATION_RECIPE:
                    raise ValueError("phase1 must use canonical shared prompt tokenization")
                ids = row["unsteered_token_ids"]
                cut = bt.cut25_token_position(ids, fraction=direction["cut_fraction"])
                prompts.append(bt._build_prompt(row["problem"]))
                if canonical_prompt_ids(tok, prompts[-1]) != row["prompt_token_ids"]:
                    raise ValueError("continuation prompt token IDs differ from phase1")
                prompt_tokens.append(row["prompt_token_ids"])
                prefixes.append(ids[:cut])
                budgets.append(max(64, len(ids) - cut))
            outputs = generate_token_panels(model, tok, [p + x for p, x in zip(prompt_tokens, prefixes)],
                budgets, hook=hook, magnitudes=[m for _, m in chunk])
            for (qid, magnitude), prompt, prompt_ids, prefix, budget, ids in zip(chunk, prompts, prompt_tokens, prefixes, budgets, outputs):
                text = bt._fix_byte_decode(tok.decode(ids, skip_special_tokens=True))
                full_text = tok.decode(prefix, skip_special_tokens=True) + text
                record = {"generation_identity": identity_hash, "question_id": qid,
                    "magnitude": magnitude, "arm": direction["arm"], "seed": direction["seed"],
                    "problem_prompt": prompt, "prefix_token_ids": prefix,
                    "prompt_token_ids": prompt_ids, "continuation_token_ids": ids, "continuation_token_count": len(ids),
                    "continuation": text, "ground_truth": phase1[qid]["ground_truth"],
                    "answer_text": full_text,
                    "remaining_token_budget": budget, "generation_sha256": canonical_hash(text)}
                # Test candidates are deliberately not graded, even locally.
                if args.partition != "test_candidates":
                    answer = bt.extract_boxed(full_text)
                    record.update(parsed_answer=answer, correct=bool(bt.answers_match(answer, phase1[qid]["ground_truth"])))
                append(output, record)
                have.add((qid, magnitude))
            batches += 1
            write_progress(outdir / "progress.json", expected=len(wanted), completed=len(have),
                identity=identity, batches_this_invocation=batches)
    finally:
        handle.remove()


def export_judge_command(args):
    """Write OpenAI Batch-format JSONL; perform no API calls."""
    bt = historical(args.historical_root)
    from temp_bench.case_studies.steering import COHERENCE_PROMPT
    outdir = args.workspace / args.partition
    generations = rows(outdir / "generations.jsonl")
    identity = read(outdir / "generation_identity.json")
    expected = {(q, float(m)) for q in identity["qids"] for m in identity["magnitudes"]}
    if {(r["question_id"], r["magnitude"]) for r in generations} != expected or len(generations) != len(expected):
        raise ValueError("complete generation grid required before judging")
    if args.judge_model == JUDGE_MODEL_PLACEHOLDER:
        # A later real-model export must not be blocked by the frozen template.
        outdir = outdir / "judge_template"
    prompts = {"backtracking": bt.SONNET_JUDGE_PROMPT, "coherence": COHERENCE_PROMPT}
    judge_identity = {"model": args.judge_model, "provider": "openai-deferred-batch",
        "rubric_hashes": {k: canonical_hash(v) for k, v in prompts.items()},
        "generation_identity": canonical_hash(identity), "truncation": "historical count prompt: prompt1500/generation6000; coherence6000"}
    frozen_json(outdir / "judge_identity.json", judge_identity)
    requests, mapping = [], {}
    for row in generations:
        for rubric in prompts:
            prompt = (prompts[rubric].format(prompt_text=row["problem_prompt"][:1500],
                      generation=row["continuation"][:6000]) if rubric == "backtracking"
                      else prompts[rubric].format(text=row["continuation"][:6000]))
            entry = {"question_id": row["question_id"], "magnitude": row["magnitude"],
                "rubric": rubric, "generation_sha256": row["generation_sha256"],
                "judge_identity": canonical_hash(judge_identity)}
            request_id = "c7-" + canonical_hash(entry)[:32]
            mapping[request_id] = entry
            requests.append({"custom_id": request_id, "method": "POST", "url": "/v1/chat/completions",
                "body": {"model": args.judge_model, "max_completion_tokens": 512,
                    "messages": [{"role": "user", "content": prompt}]}})
    frozen_json(outdir / "judge_batch_index.json", mapping)
    text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in requests)
    path = outdir / "judge_batch_requests.jsonl"
    if path.exists() and path.read_text() != text:
        raise ValueError("existing judge batch differs")
    path.write_text(text)
    frozen_json(outdir / "judge_batch_manifest.json", {
        "status": "exported_not_submitted", "model": args.judge_model,
        "submission_ready": args.judge_model != JUDGE_MODEL_PLACEHOLDER,
        "request_count": len(requests), "max_completion_tokens_per_request": 512,
        "total_prompt_characters": sum(len(r["body"]["messages"][0]["content"]) for r in requests),
        "requests_sha256": digest(path),
        "note": "No API requests made. Check current model availability, input/output pricing, reasoning-token settings, and the user's total budget before submission."})


def import_judge_command(args):
    """Import downloaded batch outputs; API errors remain missing evidence."""
    bt = historical(args.historical_root)
    outdir = args.workspace / args.partition
    mapping = read(outdir / "judge_batch_index.json")
    path = outdir / "judgments.jsonl"
    seen = {r.get("source_output_sha256") for r in rows(path)}
    for result in rows(args.batch_output):
        request_id = result.get("custom_id")
        if request_id not in mapping:
            raise ValueError(f"batch output is not from this export: {request_id}")
        row_hash = canonical_hash(result)
        if row_hash in seen:
            continue
        body = (result.get("response") or {}).get("body", {})
        choices = body.get("choices") or []
        raw = str((choices[0].get("message", {}).get("content") or "") if choices else result.get("error", "missing response"))
        entry = mapping[request_id]
        label = bt.parse_judge_reply(raw) if entry["rubric"] == "backtracking" else (int(raw.strip()) if re.fullmatch("[0-3]", raw.strip()) else -1)
        append(path, {**entry, "label": label, "raw": raw, "usage": body.get("usage"),
            "source_output_sha256": row_hash})
        seen.add(row_hash)
    complete_scores(args.workspace, args.partition)


def unlock_command(args):
    """After validation selection, release only selected-dose test records."""
    bt = historical(args.historical_root)
    selected = read(args.workspace / "selection.json")
    candidate_dir = args.workspace / "test_candidates"
    identity = read(candidate_dir / "generation_identity.json")
    if identity["direction_sha256"] != selected["direction_sha256"] or identity["phase1_sha256"] != selected["phase1_sha256"]:
        raise ValueError("candidate test grid differs from validation selection")
    for source, key in (("generations.jsonl", "validation_generations_sha256"),
                        ("judgments.jsonl", "validation_judgments_sha256")):
        if digest(args.workspace / "validation" / source) != selected[key]:
            raise ValueError("validation artifacts changed after selection")
    chosen = {0., float(selected["magnitude"])}
    candidate_rows = rows(candidate_dir / "generations.jsonl")
    have = {(r["question_id"], r["magnitude"]) for r in candidate_rows}
    expected = {(q, float(m)) for q in identity["qids"] for m in identity["magnitudes"]}
    if have != expected or len(have) != len(candidate_rows):
        raise ValueError("test candidate grid is incomplete or duplicated")
    unlocked_identity = {**identity, "partition": "test", "magnitudes": sorted(chosen),
        "selection_sha256": digest(args.workspace / "selection.json"),
        "candidate_generations_sha256": digest(candidate_dir / "generations.jsonl")}
    frozen_json(args.workspace / "test/generation_identity.json", unlocked_identity)
    output = []
    for row in candidate_rows:
        if row["magnitude"] in chosen:
            answer = bt.extract_boxed(row["answer_text"])
            output.append({**row, "generation_identity": canonical_hash(unlocked_identity),
                "parsed_answer": answer, "correct": bool(bt.answers_match(answer, row["ground_truth"]))})
    text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in output)
    path = args.workspace / "test/generations.jsonl"
    if path.exists() and path.read_text() != text:
        raise ValueError("existing unlocked test results differ")
    path.write_text(text)


def complete_scores(workspace: Path, partition: str):
    outdir = workspace / partition
    identity = read(outdir / "generation_identity.json")
    ji_hash = canonical_hash(read(outdir / "judge_identity.json"))
    latest = {}
    for row in rows(outdir / "judgments.jsonl"):
        if row["judge_identity"] != ji_hash:
            raise ValueError("judge identities are mixed")
        if row["label"] >= 0:
            latest[(row["question_id"], row["magnitude"], row["rubric"], row["generation_sha256"])] = row["label"]
    scores = {}
    generations = rows(outdir / "generations.jsonl")
    for row in generations:
        q, m, h = row["question_id"], row["magnitude"], row["generation_sha256"]
        gc = latest.get((q, m, "backtracking", h))
        coh = latest.get((q, m, "coherence", h))
        if gc is None or coh is None:
            raise ValueError(f"judging incomplete for {q} at {m}; rerun judge to retry failures")
        scores[(q, m)] = {"gc": gc, "coherent_gc": gc * int(coh >= 2),
            "coherent_bt": int(coh >= 2 and gc >= 1), "incoherent": int(coh < 2),
            "correct": int(row["correct"])}
    expected = {(q, float(m)) for q in identity["qids"] for m in identity["magnitudes"]}
    if set(scores) != expected:
        raise ValueError("generation/score grid is not complete")
    return identity, scores


def choose_magnitude(qids, magnitudes, scores):
    """Fixed objective: mean coherent genuine-backtracking count gain."""
    curve = {}
    for m in magnitudes:
        curve[float(m)] = float(np.mean([scores[(q, float(m))]["coherent_gc"] - scores[(q, 0.)]["coherent_gc"] for q in qids]))
    # No-intervention wins exact ties; then prefer smaller absolute changes.
    chosen = min(curve, key=lambda m: (-curve[m], abs(m), m))
    return chosen, curve


def select_command(args):
    identity, scores = complete_scores(args.workspace, "validation")
    magnitude, curve = choose_magnitude(identity["qids"], identity["magnitudes"], scores)
    frozen_json(args.workspace / "selection.json", {"protocol": PROTOCOL,
        "direction_sha256": identity["direction_sha256"], "phase1_sha256": identity["phase1_sha256"],
        "validation_generations_sha256": digest(args.workspace / "validation/generations.jsonl"),
        "validation_judgments_sha256": digest(args.workspace / "validation/judgments.jsonl"),
        "objective": "mean coherent genuine-backtracking count minus matched zero intervention",
        "magnitude": magnitude, "validation_curve": {str(m): v for m, v in curve.items()},
        "tie_break": "smallest absolute magnitude, then signed value"})


def summarize_command(args):
    identity, scores = complete_scores(args.workspace, "test")
    selection = read(args.workspace / "selection.json")
    m = float(selection["magnitude"])
    rng = np.random.default_rng(42)
    qids = identity["qids"]
    indices = rng.integers(0, len(qids), (2000, len(qids)))
    metrics = {}
    for metric in ("gc", "coherent_gc", "coherent_bt", "incoherent", "correct"):
        baseline = np.asarray([scores[(q, 0.)][metric] for q in qids], float)
        treated = np.asarray([scores[(q, m)][metric] for q in qids], float)
        delta = treated - baseline
        metrics[metric] = {"baseline_mean": float(baseline.mean()), "treated_mean": float(treated.mean()),
            "paired_delta": float(delta.mean()), "question_bootstrap_95ci": np.quantile(delta[indices].mean(1), [.025, .975]).tolist()}
    rescued = sum(scores[(q, m)]["correct"] and not scores[(q, 0.)]["correct"] for q in qids)
    harmed = sum(scores[(q, 0.)]["correct"] and not scores[(q, m)]["correct"] for q in qids)
    frozen_json(args.workspace / "test_summary.json", {"protocol": PROTOCOL,
        "n_questions": len(qids), "magnitude": m, "metrics": metrics,
        "rescues": rescued, "harms": harmed, "net_rescues": rescued - harmed,
        "uncertainty": "paired question bootstrap conditional on the frozen feature and validation-selected magnitude; not dictionary-seed uncertainty",
        "generation_identity": identity, "selection_sha256": digest(args.workspace / "selection.json")})


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("split", "phase1", "mine", "generate", "export-judge", "import-judge", "select", "unlock", "summarize"):
        q = sub.add_parser(command)
        if command not in ("select", "summarize"):
            q.add_argument("--historical-root", type=Path, required=True)
        if command != "split":
            q.add_argument("--workspace", type=Path, required=True)
        if command in ("split", "mine"):
            q.add_argument("--sentence-acts", type=Path, required=True)
        if command in ("phase1", "mine", "generate"):
            q.add_argument("--split-file", type=Path, required=True)
        if command == "split":
            q.add_argument("--output", type=Path, required=True)
        if command in ("split", "phase1"):
            q.add_argument("--math500", type=Path, required=True,
                help=f"test.jsonl from HuggingFaceH4/MATH-500 at {MATH500_REVISION}")
        if command in ("phase1", "mine", "generate"):
            q.add_argument("--batch-size", type=int, default=8 if command != "mine" else 256)
        if command == "phase1":
            q.add_argument("--max-new-tokens", type=int, default=1024)
            q.add_argument("--verify-zero", action="store_true", help="check zero-hook no-op and canonical cut/continue suffix on first pending batch")
        if command in ("phase1", "generate"):
            q.add_argument("--max-batches", type=int, help="run at most this many pending batches; durable partial outputs remain resumable with identical protocol")
        if command == "mine":
            q.add_argument("--arm", choices=sorted(ARMS), required=True)
            q.add_argument("--checkpoint-dir", type=Path, required=True)
            q.add_argument("--device", default="cuda")
            q.add_argument("--random-control-seed", type=int)
        if command in ("generate", "export-judge", "import-judge"):
            q.add_argument("--partition", choices=("validation", "test", "test_candidates") if command == "generate" else ("validation", "test"), required=True)
        if command == "generate":
            q.add_argument("--phase1", type=Path, required=True)
        if command == "export-judge":
            q.add_argument("--judge-model", default=JUDGE_MODEL_PLACEHOLDER, help="optional future model ID; default placeholder is not submission-ready")
        if command == "import-judge":
            q.add_argument("--batch-output", type=Path, required=True)
    return p


def main():
    args = parser().parse_args()
    globals()[args.command.replace("-", "_") + "_command"](args)


if __name__ == "__main__":
    main()
