"""Exp 6, 7, 9 — B3 variants for Tier 2.

Reuses the b3_math500_rescue.py harness with three switches:
  --cut-fraction 0.5 (default) | 0.25 | 0.0 (LLM-judged)
  --steering-style continuous (default) | first-token-only

Restricted to the truly-wrong cohort (drops `unsteered_answer is None`)
so we don't double-count token-truncated cases.

Run:
  python -m experiments.ward_backtracking_txc.b3_variants --variant cut25
  python -m experiments.ward_backtracking_txc.b3_variants --variant single
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
import yaml, torch
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ward_txc.b3_variants")

# Reuse the existing b3 module's helpers
from experiments.ward_backtracking_txc.b3_math500_rescue import (
    _Hook, _load_lm, _build_prompt, _generate_continuation_panels,
    extract_boxed, answers_match, load_steering_vector, normalize_to_dom_norm,
)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    p.add_argument("--variant", choices=["cut25", "single", "match_baseline"], required=True)
    p.add_argument("--steering-cell", default="txc__resid_L10__k16__s42")
    p.add_argument("--feature-id", type=int, default=14621)
    p.add_argument("--feature-mode", default="pos0")
    p.add_argument("--magnitudes", type=float, nargs="+", default=[0.0, -8.0, -12.0, 8.0])
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--gen-batch-size", type=int, default=16)
    p.add_argument("--phase1-cache", type=Path,
                   default=Path("results/ward_backtracking_txc/b3_math500/phase1_unsteered.json"))
    p.add_argument("--include-correct", type=int, default=0,
                   help="Also steer N originally-correct questions (random subsample) "
                        "to fill the regression cells of the flip matrix. 0 = wrong-cohort only.")
    p.add_argument("--correct-seed", type=int, default=42,
                   help="Seed for the correct-cohort subsample.")
    p.add_argument("--out", type=Path, default=None,
                   help="Override output dir (default: <root>/b3_math500_<variant>). "
                        "Use this to run multi-arch sweeps without overwriting.")
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    out = args.out if args.out is not None else (Path(cfg["paths"]["root"]) / f"b3_math500_{args.variant}")
    out.mkdir(parents=True, exist_ok=True)

    # Load phase 1 from the cached B3 run
    base_results = json.loads(args.phase1_cache.read_text())
    log.info("[load] phase 1: %d trajectories from %s", len(base_results), args.phase1_cache)

    # Truly-wrong cohort: drops `unsteered_answer is None`
    truly_wrong = [r for r in base_results
                   if not r["unsteered_correct"] and r["unsteered_answer"] is not None]
    log.info("[cohort] truly-wrong: %d (dropped %d truncated)",
             len(truly_wrong), sum(1 for r in base_results
                                    if not r["unsteered_correct"] and r["unsteered_answer"] is None))

    # Optionally include a random subsample of originally-correct questions to
    # measure regressions (correct -> incorrect) under steering.
    if args.include_correct > 0:
        correct_pool = [r for r in base_results if r["unsteered_correct"]]
        rng = np.random.default_rng(args.correct_seed)
        n_take = min(args.include_correct, len(correct_pool))
        idx = rng.choice(len(correct_pool), size=n_take, replace=False)
        correct_subsample = [correct_pool[i] for i in sorted(idx.tolist())]
        log.info("[cohort] correct subsample: %d (from %d eligible)",
                 len(correct_subsample), len(correct_pool))
    else:
        correct_subsample = []
    cohort = list(truly_wrong) + list(correct_subsample)
    cohort_before = {r["unique_id"]: bool(r["unsteered_correct"]) for r in cohort}
    log.info("[cohort] total panels source: %d (wrong=%d, correct=%d)",
             len(cohort), len(truly_wrong), len(correct_subsample))

    # Load model
    layer = int(cfg["steering"]["steering_layer"])
    log.info("[load] reasoning model")
    model, tok = _load_lm(cfg["models"]["reasoning"], "cuda")
    layer_module = model.model.layers[layer]

    # Steering vector
    raw_vec = load_steering_vector(args.steering_cell, args.feature_id,
                                    args.feature_mode, Path(cfg["paths"]["features_dir"]))
    vec = normalize_to_dom_norm(raw_vec, cfg)

    # Variant-specific cut-fraction
    if args.variant == "cut25":
        cut_frac = 0.25
    elif args.variant == "single":
        cut_frac = 0.5  # use same cut as default
    elif args.variant == "match_baseline":
        cut_frac = 0.5  # same as default; this is just the baseline rerun on truly-wrong
    log.info("[variant] %s; cut_fraction=%.2f", args.variant, cut_frac)

    # Build panels
    panel_problems, panel_prefixes, panel_mags, panel_max_new, panel_meta = [], [], [], [], []

    # For 'single' variant: hook fires only at the FIRST forward pass per chunk
    # (i.e., on the prefill of the prefix). Subsequent decoding steps see no
    # steering — the model gets one nudge at the cut boundary, then is left
    # alone. Implemented as a chunk-bound counter that gets reset whenever
    # hook.magnitudes is reassigned (handled in the patched __call__ below).
    if args.variant == "single":
        class SingleStepHook(_Hook):
            def __init__(self, vec):
                super().__init__(vec)
                self._last_mags_id: int | None = None
                self._step_counter: int = 0

            def __call__(self, _m, _i, output):
                if self.magnitudes is None:
                    return output
                # Reset counter when magnitudes object changes (new chunk).
                if self._last_mags_id != id(self.magnitudes):
                    self._step_counter = 0
                    self._last_mags_id = id(self.magnitudes)
                # Only fire on step 0; pass through after.
                if self._step_counter > 0:
                    self._step_counter += 1
                    return output
                self._step_counter += 1
                # Apply scalar magnitudes broadcast over batch.
                if isinstance(output, tuple):
                    x = output[0]
                    v = self._materialize(x)
                    mags = self.magnitudes.to(x.device, x.dtype)
                    return (x + mags.view(-1, 1, 1) * v,) + output[1:]
                v = self._materialize(output)
                mags = self.magnitudes.to(output.device, output.dtype)
                return output + mags.view(-1, 1, 1) * v
        hook = SingleStepHook(vec)
    else:
        hook = _Hook(vec)
    handle = layer_module.register_forward_hook(hook)

    # Map unique_id → original problem (we need the problem text for chat-template)
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    by_id = {r["unique_id"]: r for r in ds}

    for r in cohort:
        unsteered_len = len(r["unsteered_token_ids"])
        prefix_len = max(1, int(unsteered_len * cut_frac))
        prefix = r["unsteered_token_ids"][:prefix_len]
        cont_budget = max(64, args.max_new_tokens - prefix_len)
        prob = by_id.get(r["unique_id"])
        if prob is None:
            log.warning("[skip] %s missing in MATH-500", r["unique_id"]); continue
        problem_prompt = _build_prompt(prob["problem"])
        for mag in args.magnitudes:
            panel_problems.append(problem_prompt)
            panel_prefixes.append(prefix)
            panel_mags.append(float(mag))
            panel_max_new.append(cont_budget)
            panel_meta.append({"unique_id": r["unique_id"], "magnitude": float(mag),
                               "prefix_len": prefix_len, "cont_budget": cont_budget,
                               "before_correct": cohort_before[r["unique_id"]]})

    log.info("[phase 2] %d panels (= %d cohort × %d mags)",
             len(panel_problems), len(cohort), len(args.magnitudes))

    cont_texts = _generate_continuation_panels(
        model, tok, hook,
        problem_prompts=panel_problems, prefix_token_ids=panel_prefixes,
        mags_per_panel=panel_mags, max_new_per_panel=panel_max_new,
        batch_size=args.gen_batch_size,
    )
    handle.remove()

    cohort_by_id = {r["unique_id"]: r for r in cohort}
    rescue_rows = []
    for meta, prefix, txt in zip(panel_meta, panel_prefixes, cont_texts):
        prefix_text = tok.decode(prefix, skip_special_tokens=True)
        full = prefix_text + txt
        new_ans = extract_boxed(full)
        gt = cohort_by_id[meta["unique_id"]]["ground_truth"]
        rescued = answers_match(new_ans, gt)
        rescue_rows.append({**meta, "ground_truth": gt, "rescued_answer": new_ans,
                             "rescued_correct": rescued, "continuation_text": txt})

    (out / "phase2_rescue.json").write_text(json.dumps(rescue_rows, indent=2))

    # Aggregate
    from collections import defaultdict
    by_mag_wrong = defaultdict(list)
    by_mag_correct = defaultdict(list)
    for r in rescue_rows:
        if r["before_correct"]:
            by_mag_correct[r["magnitude"]].append(r["rescued_correct"])
        else:
            by_mag_wrong[r["magnitude"]].append(r["rescued_correct"])
    summary = {"variant": args.variant,
               "n_truly_wrong": len(truly_wrong),
               "n_correct_subsample": len(correct_subsample),
               "rescue_rate_by_magnitude": {},
               "regression_rate_by_magnitude": {}}
    all_mags = sorted(set(list(by_mag_wrong.keys()) + list(by_mag_correct.keys())))
    for mag in all_mags:
        vs_w = by_mag_wrong.get(mag, [])
        vs_c = by_mag_correct.get(mag, [])
        if vs_w:
            rate = sum(vs_w)/len(vs_w)
            summary["rescue_rate_by_magnitude"][str(mag)] = {"n_rescued": sum(vs_w), "n": len(vs_w), "rate": rate}
        if vs_c:
            n_regress = sum(1 for x in vs_c if not x)
            summary["regression_rate_by_magnitude"][str(mag)] = {"n_regressed": n_regress, "n": len(vs_c),
                                                                  "rate": n_regress/len(vs_c)}
        log.info("  mag=%+5.1f  rescued=%d/%d  regressed=%d/%d",
                 mag, sum(vs_w), len(vs_w), sum(1 for x in vs_c if not x), len(vs_c))
    if 0.0 in by_mag_wrong:
        ctrl_w = summary["rescue_rate_by_magnitude"]["0.0"]["rate"]
        for k, v in summary["rescue_rate_by_magnitude"].items():
            v["delta_vs_control"] = v["rate"] - ctrl_w
    if 0.0 in by_mag_correct:
        ctrl_c = summary["regression_rate_by_magnitude"]["0.0"]["rate"]
        for k, v in summary["regression_rate_by_magnitude"].items():
            v["delta_vs_control"] = v["rate"] - ctrl_c
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("[saved] %s/summary.json", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
