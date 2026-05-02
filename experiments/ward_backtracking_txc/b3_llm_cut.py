"""Exp 9 — B3 with LLM-judged cut points.

For each truly-wrong unsteered trajectory, ask Sonnet 4.6 to identify
the FIRST step where the reasoning goes wrong (token range), then cut
there and continue with steering. Tests whether cut25's success was
"earlier-is-better" or whether a principled cut at the actual error
step does even better.

Two-phase:
  Phase A: Sonnet identifies error-step indices (~$3 API, ~3 min)
  Phase B: B3 cut+continue at the LLM-judged cut points
"""
from __future__ import annotations
import argparse, asyncio, json, logging, os, re, sys
from pathlib import Path
import torch
import yaml

from experiments.ward_backtracking_txc.b3_math500_rescue import (
    _Hook, _load_lm, _build_prompt, _generate_continuation_panels,
    extract_boxed, answers_match, load_steering_vector, normalize_to_dom_norm,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ward_txc.b3_llm_cut")

JUDGE_PROMPT = """You are reviewing a math problem solution that arrived at the WRONG answer. Your job is to identify the FIRST step where the reasoning went wrong — the earliest sentence (or phrase) that, if corrected, the model would likely have arrived at the right answer.

Problem: {problem}

Ground-truth answer: {ground_truth}

Model's (incorrect) reasoning trace:
\"\"\"
{trace}
\"\"\"

Reply with this exact format on three lines:

ERROR_LINE: <integer line number, 1-indexed>
ERROR_QUOTE: "<exact substring from the trace, ≤ 80 chars, that contains the first wrong step>"
EXPLANATION: <one sentence on what went wrong>

Lines are separated by newlines in the trace; count from line 1. The ERROR_QUOTE must appear verbatim in the trace so we can locate the cut point. Do not output anything else."""


async def find_error_step(client, problem: str, ground_truth: str, trace: str,
                           model_id: str = "claude-sonnet-4-6") -> dict:
    msg = await client.messages.create(
        model=model_id, max_tokens=300,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            problem=problem[:1500], ground_truth=ground_truth[:200],
            trace=trace[:6000])}],
    )
    raw = msg.content[0].text.strip()
    line_m = re.search(r"ERROR_LINE:\s*(\d+)", raw)
    quote_m = re.search(r'ERROR_QUOTE:\s*"([^"]*)"', raw)
    return {
        "raw": raw,
        "error_line": int(line_m.group(1)) if line_m else -1,
        "error_quote": quote_m.group(1) if quote_m else "",
    }


async def run_phase_a(truly_wrong: list[dict], problems_by_id: dict,
                      out_path: Path, concurrency: int = 8):
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    sem = asyncio.Semaphore(concurrency)
    cuts = {}
    if out_path.exists():
        cuts = json.loads(out_path.read_text())
        log.info("[resume] %d existing cuts", len(cuts))

    async def worker(rec):
        async with sem:
            uid = rec["unique_id"]
            if uid in cuts and cuts[uid].get("error_line", -1) > 0: return
            prob = problems_by_id.get(uid)
            if prob is None:
                cuts[uid] = {"error_line": -1, "raw": "(missing problem)"}
                return
            try:
                result = await find_error_step(client, prob["problem"],
                                                rec["ground_truth"], rec["unsteered_text"])
            except Exception as e:
                result = {"error_line": -1, "raw": f"(api-error: {e})", "error_quote": ""}
            cuts[uid] = result

    tasks = [worker(r) for r in truly_wrong]
    n_done = 0
    for fut in asyncio.as_completed(tasks):
        await fut
        n_done += 1
        if n_done % 10 == 0:
            out_path.write_text(json.dumps(cuts, indent=2))
            log.info("[phase A] %d/%d done", n_done, len(tasks))
    out_path.write_text(json.dumps(cuts, indent=2))
    return cuts


def cut_position_from_quote(unsteered_text: str, error_line: int,
                              error_quote: str, fallback_frac: float = 0.5) -> int:
    """Return a CHARACTER position to cut at. We then convert to tokens
    by re-tokenising up to that point.

    Strategy:
    1. If error_quote is non-empty AND appears in unsteered_text, cut at
       the START of that quote.
    2. Else fall back to splitting by lines and cutting at the start of
       error_line.
    3. Else fall back to fallback_frac * len.
    """
    if error_quote and error_quote in unsteered_text:
        return unsteered_text.index(error_quote)
    if error_line > 0:
        lines = unsteered_text.split("\n")
        if error_line <= len(lines):
            return sum(len(l) + 1 for l in lines[:error_line - 1])
    return int(len(unsteered_text) * fallback_frac)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    p.add_argument("--steering-cell", default="txc__resid_L10__k16__s42")
    p.add_argument("--feature-id", type=int, default=14621)
    p.add_argument("--feature-mode", default="pos0")
    p.add_argument("--magnitudes", type=float, nargs="+", default=[0.0, -8.0, -2.0, -12.0])
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--gen-batch-size", type=int, default=16)
    p.add_argument("--phase1-cache", type=Path,
                   default=Path("results/ward_backtracking_txc/b3_math500/phase1_unsteered.json"))
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    out = Path(cfg["paths"]["root"]) / "b3_math500_llm_cut"
    out.mkdir(parents=True, exist_ok=True)

    base_results = json.loads(args.phase1_cache.read_text())
    truly_wrong = [r for r in base_results
                   if not r["unsteered_correct"] and r["unsteered_answer"] is not None]
    log.info("[cohort] truly-wrong: %d", len(truly_wrong))

    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    problems_by_id = {r["unique_id"]: r for r in ds}

    # Phase A: LLM finds the error step.
    cuts_path = out / "phase_a_cuts.json"
    log.info("[phase A] identifying error steps with Sonnet")
    cuts = asyncio.run(run_phase_a(truly_wrong, problems_by_id, cuts_path))
    n_valid = sum(1 for c in cuts.values() if c.get("error_line", -1) > 0)
    log.info("[phase A] %d/%d valid cuts (rest fall back to mid-text)", n_valid, len(cuts))

    # Phase B: B3 cut+continue at LLM-judged positions
    log.info("[phase B] running B3 at LLM-judged cut points")
    layer = int(cfg["steering"]["steering_layer"])
    model, tok = _load_lm(cfg["models"]["reasoning"], "cuda")
    layer_module = model.model.layers[layer]
    raw_vec = load_steering_vector(args.steering_cell, args.feature_id,
                                    args.feature_mode, Path(cfg["paths"]["features_dir"]))
    vec = normalize_to_dom_norm(raw_vec, cfg)
    hook = _Hook(vec)
    handle = layer_module.register_forward_hook(hook)

    panel_problems, panel_prefixes, panel_mags, panel_max_new, panel_meta = [], [], [], [], []
    for r in truly_wrong:
        cut_info = cuts.get(r["unique_id"], {})
        char_pos = cut_position_from_quote(r["unsteered_text"],
                                             cut_info.get("error_line", -1),
                                             cut_info.get("error_quote", ""))
        # Convert char position to token position by re-encoding the prefix
        prefix_text = r["unsteered_text"][:char_pos]
        # The unsteered_token_ids are already tokenized; find the token that
        # corresponds to char_pos by streaming detokenize and matching length.
        # Cheap approximation: prefix_token_count = round(prefix_text_len /
        # full_text_len * full_token_count).
        full_len = max(1, len(r["unsteered_text"]))
        full_toks = len(r["unsteered_token_ids"])
        prefix_len = max(1, int(round(len(prefix_text) / full_len * full_toks)))
        prefix = r["unsteered_token_ids"][:prefix_len]
        cont_budget = max(64, args.max_new_tokens - prefix_len)
        prob = problems_by_id.get(r["unique_id"])
        if prob is None: continue
        for mag in args.magnitudes:
            panel_problems.append(_build_prompt(prob["problem"]))
            panel_prefixes.append(prefix)
            panel_mags.append(float(mag))
            panel_max_new.append(cont_budget)
            panel_meta.append({"unique_id": r["unique_id"], "magnitude": float(mag),
                               "prefix_len": prefix_len, "cont_budget": cont_budget,
                               "char_cut_pos": char_pos,
                               "error_line": cut_info.get("error_line", -1)})

    log.info("[phase B] %d panels", len(panel_problems))
    cont_texts = _generate_continuation_panels(
        model, tok, hook,
        problem_prompts=panel_problems, prefix_token_ids=panel_prefixes,
        mags_per_panel=panel_mags, max_new_per_panel=panel_max_new,
        batch_size=args.gen_batch_size,
    )
    handle.remove()

    rescue_rows = []
    for meta, prefix, txt in zip(panel_meta, panel_prefixes, cont_texts):
        prefix_text = tok.decode(prefix, skip_special_tokens=True)
        full = prefix_text + txt
        new_ans = extract_boxed(full)
        gt = next(r["ground_truth"] for r in truly_wrong if r["unique_id"] == meta["unique_id"])
        rescued = answers_match(new_ans, gt)
        rescue_rows.append({**meta, "ground_truth": gt, "rescued_answer": new_ans,
                             "rescued_correct": rescued, "continuation_text": txt})
    (out / "phase_b_rescue.json").write_text(json.dumps(rescue_rows, indent=2))

    from collections import defaultdict
    by_mag = defaultdict(list)
    for r in rescue_rows: by_mag[r["magnitude"]].append(r["rescued_correct"])
    summary = {"variant": "llm_cut", "n_truly_wrong": len(truly_wrong),
               "n_llm_valid_cuts": n_valid, "rescue_rate_by_magnitude": {}}
    log.info("[phase B] rescue rates")
    for mag in sorted(by_mag):
        vs = by_mag[mag]
        rate = sum(vs)/len(vs) if vs else 0
        summary["rescue_rate_by_magnitude"][str(mag)] = {"n_rescued": sum(vs), "n": len(vs), "rate": rate}
        log.info("  mag=%+5.1f  rescued=%d/%d = %.3f", mag, sum(vs), len(vs), rate)
    if 0.0 in by_mag:
        ctrl = summary["rescue_rate_by_magnitude"]["0.0"]["rate"]
        for k, v in summary["rescue_rate_by_magnitude"].items():
            v["delta_vs_control"] = v["rate"] - ctrl
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("[saved] %s/summary.json", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
