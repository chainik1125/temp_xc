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
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    out = Path(cfg["paths"]["root"]) / f"b3_math500_{args.variant}"
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

    # For 'single' variant: hook fires only at the first generation step.
    # We implement this by setting hook.magnitudes per-step via a counter
    # in the hook. For now, simplest implementation: use a wrapper hook
    # that decrements a step counter and zeros after step 0.
    if args.variant == "single":
        class SingleStepHook(_Hook):
            def __init__(self, vec):
                super().__init__(vec)
                self._step_counter = None  # set per chunk to a tensor of (B,) remaining steps
            def __call__(self, _m, _i, output):
                if self.magnitudes is None:
                    return output
                # Decrement counter if set; zero magnitudes for rows past step 0
                if self._step_counter is None:
                    self._step_counter = torch.zeros_like(self.magnitudes)
                # On first call (counter==0), apply; subsequent calls zero out
                active_mask = (self._step_counter == 0).float()
                effective = self.magnitudes * active_mask
                # Increment counter for next call
                self._step_counter = self._step_counter + 1
                # Apply effective
                if isinstance(output, tuple):
                    x = output[0]
                    v = self._materialize(x)
                    return (x + effective.to(x.device, x.dtype).view(-1, 1, 1) * v,) + output[1:]
                v = self._materialize(output)
                return output + effective.to(output.device, output.dtype).view(-1, 1, 1) * v
        hook = SingleStepHook(vec)
    else:
        hook = _Hook(vec)
    handle = layer_module.register_forward_hook(hook)

    # Map unique_id → original problem (we need the problem text for chat-template)
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    by_id = {r["unique_id"]: r for r in ds}

    for r in truly_wrong:
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
                               "prefix_len": prefix_len, "cont_budget": cont_budget})

    log.info("[phase 2] %d panels (= %d truly-wrong × %d mags)",
             len(panel_problems), len(truly_wrong), len(args.magnitudes))

    # Reset single-step counter per generate-call chunk
    if args.variant == "single":
        # We need to reset _step_counter at the START of each generate.
        # Simplest: monkey-patch the generate loop is hard; instead, reset
        # after _generate_continuation_panels completes by overriding the
        # function to reset before each chunk.
        # Easiest clean way: re-run _generate_continuation_panels but
        # interpose a reset.
        pass  # The _generate_continuation_panels will set hook.magnitudes
              # per chunk; we need _step_counter reset. Patch:
        orig_call = hook.__call__
        chunk_step_counter = {"counter": None}
        def patched_call(_m, _i, output):
            mags = hook.magnitudes
            if mags is None: return output
            if chunk_step_counter["counter"] is None or chunk_step_counter["counter"].shape[0] != mags.shape[0]:
                chunk_step_counter["counter"] = torch.zeros_like(mags)
            active = (chunk_step_counter["counter"] == 0).float()
            chunk_step_counter["counter"] = chunk_step_counter["counter"] + 1
            effective = mags * active
            if isinstance(output, tuple):
                x = output[0]
                v = hook._materialize(x)
                return (x + effective.to(x.device, x.dtype).view(-1, 1, 1) * v,) + output[1:]
            v = hook._materialize(output)
            return output + effective.to(output.device, output.dtype).view(-1, 1, 1) * v
        # Monkey-patch hook's __call__
        from types import MethodType
        hook.__call__ = MethodType(lambda self, m, i, o: patched_call(m, i, o), hook)
        # Need to re-register hook with patched method
        handle.remove()
        handle = layer_module.register_forward_hook(hook)
        # On each new chunk (new hook.magnitudes), reset counter
        # Patch _generate_continuation_panels: nope — instead reset counter
        # whenever magnitudes changes shape. The check above handles that.

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

    (out / "phase2_rescue.json").write_text(json.dumps(rescue_rows, indent=2))

    # Aggregate
    from collections import defaultdict
    by_mag = defaultdict(list)
    for r in rescue_rows:
        by_mag[r["magnitude"]].append(r["rescued_correct"])
    summary = {"variant": args.variant, "n_truly_wrong": len(truly_wrong),
               "rescue_rate_by_magnitude": {}}
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
