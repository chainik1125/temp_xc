"""B3 — Backtracking-induced rescue on MATH-500.

Per Dmitry's 2026-05-01 ask: take problems from MATH-500, find ones the
unsteered reasoning model gets wrong, cut each wrong trajectory at the
token midpoint, then continue with TXC-derived backtracking steering
applied vs unsteered control. Measure: did steering rescue the answer?

Pipeline:
  Phase 1 (baseline): generate unsteered responses to N MATH-500
                      problems. Extract the boxed final answer. Compare
                      to ground truth → split into right/wrong sets.
  Phase 2 (rescue):   for each wrong problem:
                      - take the first 50% of the wrong trajectory (token
                        midpoint of the unsteered output)
                      - continue at multiple steering magnitudes:
                          mag=0  (no steering) — control: does merely
                                  regenerating the second half help?
                          mag=M (the TXC k=16 winner's best magnitude)
                                  — treatment
                      - extract the new boxed answer from each
                        continuation, compare to ground truth
  Phase 3 (report):   per-magnitude rescue rate (steered) vs control
                      (mag=0). Effect size = steered_rescue −
                      control_rescue.

Usage:
    python -m experiments.ward_backtracking_txc.b3_math500_rescue \
        --n-problems 150 \
        --steering-cell txc__resid_L10__k16__s42 \
        --feature-id 14621 --feature-mode pos0 \
        --magnitudes 0 -8 -12 +8 \
        --out results/ward_backtracking_txc/b3_math500
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ward_txc.b3_math500")


# -----------------------------------------------------------------------------
# Forward hook — same per-row magnitude design as b1_steer_eval.py.
# -----------------------------------------------------------------------------

class _Hook:
    def __init__(self, vec: torch.Tensor):
        self._raw = vec.detach()
        self._cached: torch.Tensor | None = None
        self.magnitudes: torch.Tensor | None = None

    def _materialize(self, ref: torch.Tensor) -> torch.Tensor:
        if self._cached is None or self._cached.device != ref.device or self._cached.dtype != ref.dtype:
            self._cached = self._raw.to(device=ref.device, dtype=ref.dtype)
        return self._cached

    def _delta(self, x: torch.Tensor) -> torch.Tensor:
        v = self._materialize(x)
        if self.magnitudes is not None:
            mags = self.magnitudes.to(device=x.device, dtype=x.dtype)
            return mags.view(-1, 1, 1) * v
        return torch.zeros_like(x)

    def __call__(self, _m, _i, output):
        if self.magnitudes is None or torch.count_nonzero(self.magnitudes) == 0:
            return output
        if isinstance(output, tuple):
            x = output[0]
            return (x + self._delta(x),) + output[1:]
        return output + self._delta(output)


# -----------------------------------------------------------------------------
# Boxed-answer extraction + normalized comparison.
# -----------------------------------------------------------------------------

def extract_boxed(text: str) -> str | None:
    """Pull the LAST \\boxed{...} content from a model output. Handles
    nested braces (e.g. \\boxed{\\frac{1}{2}}) via a stack-based parser.
    Returns None if no boxed answer was emitted."""
    if not text:
        return None
    last = None
    i = 0
    needle = "\\boxed{"
    while True:
        idx = text.find(needle, i)
        if idx < 0:
            break
        # Find matching closing brace, accounting for nested {}.
        start = idx + len(needle)
        depth = 1
        j = start
        while j < len(text) and depth > 0:
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth == 0:
            last = text[start:j].strip()
        i = max(idx + 1, j)
    return last


def _strip_latex_to_plain(s: str) -> str:
    """Lossy normalization for boxed-answer comparison: lowercase, drop
    LaTeX wrappers, strip whitespace and trailing punctuation. Won't catch
    semantically-equivalent fractions (1/2 vs \\frac{1}{2}) — those are
    rare enough at this scale that a manual review of close-but-no-cigar
    cases is good enough; a stricter sympy-based comparator is a TODO."""
    s = s.lower()
    s = s.replace("\\left(", "(").replace("\\right)", ")")
    s = s.replace("\\left[", "[").replace("\\right]", "]")
    s = s.replace("\\,", "").replace("\\!", "").replace("\\;", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace(" ", "").strip(".,;:")
    return s


def answers_match(model_answer: str | None, ground_truth: str) -> bool:
    """Strict-but-LaTeX-aware match. Handles \\frac, \\dfrac, whitespace,
    casing, simple fraction equivalence."""
    if model_answer is None:
        return False
    a = _strip_latex_to_plain(model_answer)
    b = _strip_latex_to_plain(ground_truth)
    if a == b:
        return True
    # Try sympy if available — handles e.g. (3, pi/2) == (3, pi/2)
    try:
        from sympy import simplify, sympify
        from sympy.parsing.latex import parse_latex
        # Strip outer \boxed if present (we already extracted, but defensive).
        for parser in (parse_latex, sympify):
            try:
                ea = parser(model_answer)
                eb = parser(ground_truth)
                if simplify(ea - eb) == 0:
                    return True
            except Exception:
                continue
    except Exception:
        pass
    return False


# -----------------------------------------------------------------------------
# Generation helpers (batched).
# -----------------------------------------------------------------------------

def _load_lm(hf_id: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16, device_map=device,
    ).eval()
    return model, tok


def _fix_byte_decode(s: str) -> str:
    """Same workaround as b1_steer_eval — transformers 5.5.4 leaves
    Ġ/Ċ literal in the decoded string."""
    return s.replace("Ġ", " ").replace("Ċ", "\n").replace("Â", "")


def _build_prompt(problem: str) -> str:
    """Pose a MATH-500 problem to the reasoning model. Boxed answer is
    requested explicitly so we can extract a comparable final answer."""
    return (
        f"Solve this math problem and provide your final answer in "
        f"\\boxed{{}} notation.\n\nProblem: {problem}"
    )


def _generate_unsteered(model, tok, prompts: list[str], max_new_tokens: int,
                        batch_size: int) -> tuple[list[str], list[list[int]]]:
    """Greedy generation, no hook. Returns decoded continuation strings AND
    the raw new-token ids per prompt (so we can later cut at the token
    midpoint without re-tokenizing)."""
    from transformers import GenerationConfig
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens, do_sample=False,
        temperature=1.0, pad_token_id=tok.pad_token_id,
    )
    saved = tok.padding_side
    tok.padding_side = "left"
    try:
        chat_texts = []
        for p in prompts:
            try:
                t = tok.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                t = p
            chat_texts.append(t)

        outs: list[str] = []
        new_token_ids: list[list[int]] = []
        for i in range(0, len(chat_texts), batch_size):
            batch = chat_texts[i:i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=2048).to(model.device)
            prompt_len = enc["input_ids"].shape[1]
            with torch.no_grad():
                out_ids = model.generate(**enc, generation_config=gen_cfg)
            for row in out_ids:
                new = row[prompt_len:].tolist()
                # Trim trailing pad tokens (right-padding for generation output)
                while new and new[-1] == tok.pad_token_id:
                    new.pop()
                new_token_ids.append(new)
                outs.append(_fix_byte_decode(tok.decode(new, skip_special_tokens=True)))
        return outs, new_token_ids
    finally:
        tok.padding_side = saved


def _generate_continuation_panels(
    model, tok, hook, problem_prompts: list[str], prefix_token_ids: list[list[int]],
    mags_per_panel: list[float], max_new_tokens: int, batch_size: int,
) -> list[str]:
    """For each (problem prompt, prefix token ids, magnitude) panel, build the
    extended input as [chat_template(problem) + prefix_tokens] and generate
    the continuation under the per-row magnitude. Returns a list of
    continuation strings (no chat template / no prefix included).

    All three lists must be the same length N. Panels are batched by
    `batch_size`; the hook's magnitudes tensor is set per chunk.
    """
    from transformers import GenerationConfig
    assert len(problem_prompts) == len(prefix_token_ids) == len(mags_per_panel)
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens, do_sample=False,
        temperature=1.0, pad_token_id=tok.pad_token_id,
    )
    saved = tok.padding_side
    tok.padding_side = "left"
    try:
        # Build pre-encoded input ids per panel: [chat_template(problem)] + prefix_token_ids
        full_input_ids: list[list[int]] = []
        for problem, prefix in zip(problem_prompts, prefix_token_ids):
            try:
                ct = tok.apply_chat_template(
                    [{"role": "user", "content": problem}],
                    tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                ct = problem
            ct_ids = tok(ct, add_special_tokens=False)["input_ids"]
            full_input_ids.append(ct_ids + list(prefix))

        outs: list[str] = []
        for i in range(0, len(full_input_ids), batch_size):
            chunk_ids = full_input_ids[i:i + batch_size]
            chunk_mags = torch.tensor(mags_per_panel[i:i + batch_size],
                                       dtype=torch.float32)
            hook.magnitudes = chunk_mags

            # Manual left-pad to max length in this chunk.
            max_len = max(len(x) for x in chunk_ids)
            pad_id = tok.pad_token_id
            input_ids = torch.full((len(chunk_ids), max_len), pad_id, dtype=torch.long)
            attn = torch.zeros((len(chunk_ids), max_len), dtype=torch.long)
            for r, ids in enumerate(chunk_ids):
                pad_n = max_len - len(ids)
                input_ids[r, pad_n:] = torch.tensor(ids, dtype=torch.long)
                attn[r, pad_n:] = 1
            input_ids = input_ids.to(model.device)
            attn = attn.to(model.device)
            with torch.no_grad():
                out_ids = model.generate(
                    input_ids=input_ids, attention_mask=attn,
                    generation_config=gen_cfg,
                )
            for r, row in enumerate(out_ids):
                new = row[max_len:].tolist()
                while new and new[-1] == tok.pad_token_id:
                    new.pop()
                outs.append(_fix_byte_decode(tok.decode(new, skip_special_tokens=True)))
        hook.magnitudes = None
        return outs
    finally:
        tok.padding_side = saved


# -----------------------------------------------------------------------------
# Loading the steering vector from the cell's mined features.
# -----------------------------------------------------------------------------

def load_steering_vector(cell_id: str, feature_id: int, mode: str,
                         features_dir: Path) -> torch.Tensor:
    """Pull the steering vector for (cell, feature_id, mode) from the
    mined-features npz file. mode ∈ {"pos0", "union"}."""
    fpath = features_dir / f"{cell_id}.npz"
    if not fpath.exists():
        raise FileNotFoundError(f"features file missing: {fpath}")
    z = np.load(fpath, allow_pickle=True)
    feat_ids = z["top_features"].tolist()
    if feature_id not in feat_ids:
        raise ValueError(f"feature {feature_id} not in top_features for {cell_id}: "
                         f"{feat_ids[:8]}...")
    idx = feat_ids.index(feature_id)
    key = "decoder_at_pos0" if mode == "pos0" else "decoder_union"
    return torch.from_numpy(z[key][idx]).float()


def normalize_to_dom_norm(vec: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Rescale to the same L2 norm as DoM-base-union, so steering
    magnitude is comparable to B1's calibration."""
    dom_path = Path(cfg["paths"]["stageA_dom"])
    dom = torch.load(dom_path, weights_only=False)
    ref_norm = float(dom["base"]["union"].norm().item())
    n = vec.norm().clamp_min(1e-8)
    return vec / n * ref_norm


# -----------------------------------------------------------------------------
# Main experiment.
# -----------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    p.add_argument("--n-problems", type=int, default=150)
    p.add_argument("--steering-cell", type=str, default="txc__resid_L10__k16__s42")
    p.add_argument("--feature-id", type=int, default=14621,
                   help="default = TXC k=16 winner's f14621")
    p.add_argument("--feature-mode", type=str, default="pos0",
                   choices=["pos0", "union"])
    p.add_argument("--magnitudes", type=float, nargs="+",
                   default=[0.0, -8.0, -12.0, 8.0])
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--gen-batch-size", type=int, default=16)
    p.add_argument("--out", type=Path, default=Path("results/ward_backtracking_txc/b3_math500"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=True)

    log.info("[load] MATH-500")
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    import random
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(ds)), args.n_problems)
    problems = [ds[i] for i in indices]
    log.info("[load] sampled %d problems", len(problems))

    layer = int(cfg["steering"]["steering_layer"])
    hf_id = cfg["models"]["reasoning"]
    log.info("[load] reasoning model: %s", hf_id)
    model, tok = _load_lm(hf_id, args.device)
    layer_module = model.model.layers[layer]

    # ---- Phase 1: unsteered baseline. ----
    log.info("[phase 1] unsteered baseline on %d problems", len(problems))
    prompts = [_build_prompt(p["problem"]) for p in problems]
    base_texts, base_token_ids = _generate_unsteered(
        model, tok, prompts, max_new_tokens=args.max_new_tokens,
        batch_size=args.gen_batch_size,
    )

    base_results = []
    for prob, txt, tok_ids in zip(problems, base_texts, base_token_ids):
        ans = extract_boxed(txt)
        correct = answers_match(ans, prob["answer"])
        base_results.append({
            "unique_id": prob["unique_id"],
            "level": prob["level"],
            "subject": prob["subject"],
            "ground_truth": prob["answer"],
            "unsteered_text": txt,
            "unsteered_token_ids": tok_ids,
            "unsteered_answer": ans,
            "unsteered_correct": correct,
        })
    n_correct_base = sum(1 for r in base_results if r["unsteered_correct"])
    log.info("[phase 1] unsteered accuracy: %d/%d = %.3f",
             n_correct_base, len(base_results), n_correct_base / len(base_results))

    # Save phase 1 separately so the expensive baseline isn't lost on a
    # phase-2 crash.
    phase1_out = args.out / "phase1_unsteered.json"
    phase1_out.write_text(json.dumps(base_results, indent=2))
    log.info("[saved] %s", phase1_out)

    # ---- Phase 2: steered/unsteered continuations from the 50% midpoint
    #              of each wrong trajectory. ----
    wrong = [r for r in base_results if not r["unsteered_correct"]]
    log.info("[phase 2] %d wrong trajectories — running continuations at "
             "magnitudes %s", len(wrong), args.magnitudes)

    # Steering vector
    feat_dir = Path(cfg["paths"]["features_dir"])
    raw_vec = load_steering_vector(args.steering_cell, args.feature_id,
                                    args.feature_mode, feat_dir)
    vec = normalize_to_dom_norm(raw_vec, cfg)
    log.info("[steering] vec norm = %.3f (matched to DoM-base-union norm)",
             float(vec.norm().item()))

    hook = _Hook(vec)
    handle = layer_module.register_forward_hook(hook)

    # Build all (wrong-problem × magnitude) panels.
    panel_problems: list[str] = []
    panel_prefixes: list[list[int]] = []
    panel_mags: list[float] = []
    panel_meta: list[dict] = []
    for r in wrong:
        # Cut at the 50% token midpoint of the unsteered trajectory.
        prefix_len = max(1, len(r["unsteered_token_ids"]) // 2)
        prefix = r["unsteered_token_ids"][:prefix_len]
        problem_prompt = _build_prompt(
            next(p["problem"] for p in problems
                 if p["unique_id"] == r["unique_id"])
        )
        for mag in args.magnitudes:
            panel_problems.append(problem_prompt)
            panel_prefixes.append(prefix)
            panel_mags.append(float(mag))
            panel_meta.append({"unique_id": r["unique_id"], "magnitude": float(mag),
                               "prefix_len": prefix_len})

    log.info("[phase 2] %d panels (= %d wrong × %d mags)",
             len(panel_problems), len(wrong), len(args.magnitudes))

    cont_texts = _generate_continuation_panels(
        model, tok, hook,
        problem_prompts=panel_problems, prefix_token_ids=panel_prefixes,
        mags_per_panel=panel_mags,
        max_new_tokens=args.max_new_tokens, batch_size=args.gen_batch_size,
    )
    handle.remove()

    # Combine: (prefix) + (continuation) and re-extract boxed answer.
    rescue_rows = []
    for meta, prefix, txt in zip(panel_meta, panel_prefixes, cont_texts):
        prefix_text = _fix_byte_decode(tok.decode(prefix, skip_special_tokens=True))
        full = prefix_text + txt
        new_ans = extract_boxed(full)
        gt = next(r["ground_truth"] for r in wrong
                  if r["unique_id"] == meta["unique_id"])
        rescued = answers_match(new_ans, gt)
        rescue_rows.append({
            "unique_id": meta["unique_id"],
            "magnitude": meta["magnitude"],
            "prefix_len": meta["prefix_len"],
            "ground_truth": gt,
            "continuation_text": txt,
            "rescued_answer": new_ans,
            "rescued_correct": rescued,
        })

    phase2_out = args.out / "phase2_rescue.json"
    phase2_out.write_text(json.dumps(rescue_rows, indent=2))
    log.info("[saved] %s", phase2_out)

    # ---- Phase 3: aggregate per-magnitude rescue rate. ----
    from collections import defaultdict
    by_mag = defaultdict(list)
    for r in rescue_rows:
        by_mag[r["magnitude"]].append(r["rescued_correct"])

    summary = {
        "n_problems": len(problems),
        "n_correct_baseline": n_correct_base,
        "baseline_accuracy": n_correct_base / len(problems),
        "n_wrong": len(wrong),
        "steering_cell": args.steering_cell,
        "feature_id": args.feature_id,
        "feature_mode": args.feature_mode,
        "max_new_tokens": args.max_new_tokens,
        "rescue_rate_by_magnitude": {},
    }
    log.info("[phase 3] rescue rates (n_wrong = %d)", len(wrong))
    for mag in sorted(by_mag):
        n_resc = sum(by_mag[mag])
        n = len(by_mag[mag])
        rate = n_resc / n if n else 0.0
        summary["rescue_rate_by_magnitude"][str(mag)] = {
            "n_rescued": n_resc, "n_wrong": n, "rate": rate,
        }
        log.info("  mag=%+5.1f  rescued=%d/%d = %.3f", mag, n_resc, n, rate)

    # Headline: steered_rescue − control_rescue
    if 0.0 in by_mag:
        ctrl_rate = summary["rescue_rate_by_magnitude"]["0.0"]["rate"]
        for mag_str, agg in summary["rescue_rate_by_magnitude"].items():
            agg["rescue_above_control"] = agg["rate"] - ctrl_rate

    summary_out = args.out / "summary.json"
    summary_out.write_text(json.dumps(summary, indent=2))
    log.info("[saved] %s", summary_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
