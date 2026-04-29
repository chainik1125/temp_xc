#!/usr/bin/env python3
"""Stage 4: steered generation across three modes plus configurable variants.

Modes (all applied on the residual-stream output of layer ANCHOR_LAYER):

    A. raw_dom        — additive: h' = h + α · v_raw / ‖v_raw‖
                        where v_raw is the activation-space DoM from Stage 3.
                        Magnitudes ADDITIVE_MAGNITUDES.
    B. sae_additive   — additive: h' = h + α · d / ‖d‖
                        where `d` is either:
                          (i) one feature's decoder column, with combine=separate,
                              iterating over the top-K features as separate targets, OR
                          (ii) the unit-norm sum of the top-K decoder columns
                               (combine=sum_topk), one combined target.
                        Magnitudes ADDITIVE_MAGNITUDES.
    C. sae_clamp      — paper-clamp with error preservation:
                            z = sae.encode(h); z'[..., j] = strength;
                            h' = h - sae.decode(z) + sae.decode(z')
                        Strengths CLAMP_STRENGTHS. Per-feature only (no
                        combined-clamp variant).

Steering positions:

    --steer-positions all              (default) apply at every generated token.
    --steer-positions sentence_start   apply only for the K tokens following each
                                       sentence terminator (period, !, ?, newline)
                                       in the generation. The prompt's positions
                                       are never steered.

Output: `intervene<_suffix>/generations.jsonl`, one row per (mode, target,
magnitude, prompt_id) with the decoded continuation. The decompose dir
(holding raw_dom + top_features.json) is `decompose<_suffix>/`. Use
--decompose-suffix and --intervene-suffix to keep variants separate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("TQDM_DISABLE", "1")

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.data.nlp.cache_activations import load_model_and_tokenizer  # noqa: E402
from src.data.nlp.models import resid_hook_target  # noqa: E402

from experiments.phase7_unification.case_studies.backtracking._decode import (  # noqa: E402
    clean_decode,
    is_sentence_terminator_text,
)
from experiments.phase7_unification.case_studies.backtracking._paths import (  # noqa: E402
    ADDITIVE_MAGNITUDES,
    ANCHOR_LAYER,
    CLAMP_STRENGTHS,
    GEN_TOKENS_PER_INTERVENTION,
    N_HELDOUT,
    RESULTS_DIR,
    SUBJECT_MODEL,
    ensure_dirs,
)
from experiments.phase7_unification.case_studies.backtracking.prompts import (  # noqa: E402
    PROMPTS,
)


# ── steering hook ───────────────────────────────────────────────────────────
class SteeringHook:
    """Forward hook that mutates layer-output residual according to current state.

    Reset `state` between generation calls; the hook itself stays registered.

    Recognised state keys:
        mode             "off" | "additive" | "clamp"
        direction        unit-norm tensor (d_model,) on h.device, model dtype
        alpha            float (additive magnitude)
        feature_idx      int (clamp target)
        strength         float (clamp strength)
        steer_positions  "all" | "sentence_start"
        should_steer     bool — only consulted for sentence_start; the
                         LogitsProcessor sets this each step.
        skip_prompt      bool — if True and h.shape[1] > 1 (the prompt
                         processing forward), no-op. Set True for
                         sentence_start; harmless for "all" too since the
                         additive direction at the prompt doesn't change much.
    """

    def __init__(self, sae=None):
        self.state: dict = {"mode": "off"}
        self.sae = sae

    def __call__(self, module, inputs, output):
        if self.state["mode"] == "off":
            return output
        h = output[0] if isinstance(output, tuple) else output
        # Skip the prompt forward pass if requested (multi-position mode).
        if self.state.get("skip_prompt") and h.shape[1] > 1:
            return output
        # For sentence_start, gate on per-step state.
        if (
            self.state.get("steer_positions") == "sentence_start"
            and not self.state.get("should_steer", False)
        ):
            return output

        mode = self.state["mode"]
        if mode == "additive":
            direction = self.state["direction"]
            alpha = float(self.state["alpha"])
            h = h + alpha * direction
        elif mode == "clamp":
            j = int(self.state["feature_idx"])
            strength = float(self.state["strength"])
            assert self.sae is not None, "sae required for clamp mode"
            with torch.no_grad():
                z = self.sae.encode(h)
                h_recon = self.sae.decode(z)
                z_prime = z.clone()
                z_prime[..., j] = strength
                h_recon_prime = self.sae.decode(z_prime)
            h = h - h_recon + h_recon_prime
        else:
            raise ValueError(f"unknown mode {mode!r}")
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h


# ── LogitsProcessor: tracks distance to last sentence terminator ────────────
def make_terminator_id_set(tokenizer) -> set[int]:
    """Pre-compute the set of token IDs whose decoded form ends a sentence.

    Iterating the full vocab once is cheap (~128k for Llama-3 family) and
    avoids per-step decoding inside the hot LogitsProcessor loop.
    """
    out: set[int] = set()
    vocab_size = len(tokenizer)
    for tid in range(vocab_size):
        try:
            s = tokenizer.decode([tid], skip_special_tokens=False)
        except Exception:
            continue
        if is_sentence_terminator_text(s):
            out.add(tid)
    return out


def _make_steer_processor(hook: SteeringHook, terminator_ids: set[int], steer_K: int):
    """Returns a callable suitable for transformers' logits_processor list.

    Two compatible call signatures (older HF: (input_ids, scores); newer
    HF: __call__ from a LogitsProcessor object). We just expose a function.
    """
    state = {"last_term_pos": None}  # closure variable

    try:
        from transformers import LogitsProcessor
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"transformers required: {e}")

    class SteerStateProc(LogitsProcessor):
        def __call__(self, input_ids, scores):
            seq_len = int(input_ids.shape[1])
            last_id = int(input_ids[0, -1].item())
            if last_id in terminator_ids:
                state["last_term_pos"] = seq_len - 1
            ltp = state["last_term_pos"]
            # Position about to be generated is `seq_len`. We're deciding whether
            # the NEXT forward (which processes that position) should steer.
            if ltp is None:
                hook.state["should_steer"] = False
            else:
                # Distance from terminator to the position we're sampling now.
                dist = seq_len - ltp
                hook.state["should_steer"] = (1 <= dist <= steer_K)
            return scores

    return SteerStateProc()


# ── steering targets ────────────────────────────────────────────────────────
def _decompose_dir(suffix: str) -> Path:
    return RESULTS_DIR / (f"decompose_{suffix}" if suffix else "decompose")


def _intervene_dir(suffix: str) -> Path:
    return RESULTS_DIR / (f"intervene_{suffix}" if suffix else "intervene")


def load_targets(decompose_suffix: str, top_k: int):
    d = _decompose_dir(decompose_suffix)
    raw_dom_path = d / "raw_dom.fp16.npy"
    if raw_dom_path.exists():
        raw_dom = np.load(raw_dom_path).astype(np.float32)
        raw_dom_unit = raw_dom / max(np.linalg.norm(raw_dom), 1e-8)
    else:
        raw_dom_unit = None
    top = json.loads((d / "top_features.json").read_text())[:top_k]
    return raw_dom_unit, top


def held_out_prompts() -> list[dict]:
    return list(PROMPTS[-N_HELDOUT:])


# ── runner ──────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=5, help="number of SAE features to consider")
    parser.add_argument("--layer", type=int, default=ANCHOR_LAYER)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["raw_dom", "sae_additive", "sae_clamp"],
        choices=["raw_dom", "sae_additive", "sae_clamp"],
    )
    parser.add_argument("--combine", choices=("separate", "sum_topk"), default="separate",
                        help="for sae_additive: separate iterates over top-K features as distinct targets; sum_topk uses the unit-norm sum of the top-K decoder columns as one combined target.")
    parser.add_argument("--steer-positions", choices=("all", "sentence_start"), default="all",
                        help="all = steer at every generated position; sentence_start = steer only for K tokens after each sentence terminator")
    parser.add_argument("--steer-k", type=int, default=6, help="window size after terminator (only used for steer-positions sentence_start)")
    parser.add_argument("--sae-release", default="llama_scope_lxr_8x")
    parser.add_argument("--sae-id", default=None, help="default: l{layer}r_8x")
    parser.add_argument("--decompose-suffix", default="", help="read decompose results from decompose<_suffix>/")
    parser.add_argument("--intervene-suffix", default="", help="write intervene results to intervene<_suffix>/")
    parser.add_argument("--limit-prompts", type=int, default=None)
    parser.add_argument("--magnitudes", type=str, default=None,
                        help="comma-separated overrides for additive magnitudes (e.g. '-12,-8,-4,0,4,8,12'). Default uses ADDITIVE_MAGNITUDES from _paths.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    additive_magnitudes = (
        tuple(float(s) for s in args.magnitudes.split(","))
        if args.magnitudes is not None
        else ADDITIVE_MAGNITUDES
    )

    ensure_dirs()
    out_dir = _intervene_dir(args.intervene_suffix)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "generations.jsonl"
    if out_path.exists() and not args.force:
        print(f"[intervene] {out_path} exists; use --force to rebuild")
        return

    raw_dom_unit, top_features = load_targets(args.decompose_suffix, args.top_k)

    print(f"[intervene] loading {SUBJECT_MODEL}")
    model, tokenizer, cfg = load_model_and_tokenizer(SUBJECT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_id = tokenizer.eos_token_id
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    sae = None
    if "sae_additive" in args.modes or "sae_clamp" in args.modes:
        from experiments.phase7_unification.case_studies.backtracking.decompose_backtracking import (
            load_public_sae,
        )
        sae_id = args.sae_id or f"l{args.layer}r_8x"
        sae, _ = load_public_sae(args.layer, str(device), release=args.sae_release, sae_id=sae_id)
        sae = sae.to(model_dtype)

    feature_dirs: dict[int, torch.Tensor] = {}
    combined_dir = None
    if "sae_additive" in args.modes:
        W_dec = sae.W_dec.detach()
        if W_dec.shape[0] == cfg.d_model:
            W_dec = W_dec.T
        for rec in top_features:
            j = int(rec["feature_idx"])
            v = W_dec[j].to(device=device, dtype=torch.float32)
            v = v / max(v.norm().item(), 1e-8)
            feature_dirs[j] = v.to(model_dtype)
        if args.combine == "sum_topk":
            v = sum(W_dec[int(rec["feature_idx"])].to(device=device, dtype=torch.float32) for rec in top_features)
            v = v / max(v.norm().item(), 1e-8)
            combined_dir = v.to(model_dtype)

    raw_dir = (
        torch.tensor(raw_dom_unit, device=device, dtype=model_dtype)
        if raw_dom_unit is not None
        else None
    )

    layer_mod = resid_hook_target(model, args.layer, cfg.architecture_family)
    hook_obj = SteeringHook(sae=sae)
    handle = layer_mod.register_forward_hook(hook_obj)

    # If sentence_start, build the terminator-id set once and prepare a
    # logits_processor factory we'll attach to each generate() call.
    terminator_ids: set[int] = set()
    if args.steer_positions == "sentence_start":
        print("[intervene] building terminator-id set …", end="", flush=True)
        terminator_ids = make_terminator_id_set(tokenizer)
        print(f" {len(terminator_ids)} ids")

    prompts = held_out_prompts()
    if args.limit_prompts is not None:
        prompts = prompts[: args.limit_prompts]
    print(
        f"[intervene] {len(prompts)} held-out prompts; modes={args.modes}; "
        f"top-K={args.top_k}; combine={args.combine}; "
        f"steer={args.steer_positions}; suffixes=(in={args.decompose_suffix!r}, out={args.intervene_suffix!r})"
    )

    n_runs = 0
    t0 = time.time()
    with out_path.open("w") as fout:
        for p in prompts:
            messages = [{"role": "user", "content": p["text"]}]
            encoded = tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            )
            input_ids = encoded.to(device) if isinstance(encoded, torch.Tensor) else encoded["input_ids"].to(device)

            # Common state attributes per call
            common_state = {
                "steer_positions": args.steer_positions,
                "skip_prompt": True,
            }

            def _set_state(extra: dict) -> None:
                hook_obj.state = {**common_state, **extra}

            # ----- A. raw DoM baseline -----
            if "raw_dom" in args.modes and raw_dir is not None:
                for alpha in additive_magnitudes:
                    _set_state({"mode": "additive", "direction": raw_dir, "alpha": float(alpha)})
                    text = _gen_one(model, input_ids, tokenizer, eos_id, hook_obj, terminator_ids, args)
                    fout.write(json.dumps({
                        "mode": "raw_dom", "target": "raw_dom",
                        "magnitude": float(alpha),
                        "prompt_id": p["id"], "category": p["category"],
                        "generation": text,
                    }) + "\n")
                    n_runs += 1

            # ----- B. SAE additive -----
            if "sae_additive" in args.modes:
                if args.combine == "separate":
                    for rec in top_features:
                        j = int(rec["feature_idx"])
                        direction = feature_dirs[j]
                        for alpha in additive_magnitudes:
                            _set_state({"mode": "additive", "direction": direction, "alpha": float(alpha)})
                            text = _gen_one(model, input_ids, tokenizer, eos_id, hook_obj, terminator_ids, args)
                            fout.write(json.dumps({
                                "mode": "sae_additive", "target": f"feat_{j}",
                                "feature_idx": j, "delta": rec.get("delta"),
                                "magnitude": float(alpha),
                                "prompt_id": p["id"], "category": p["category"],
                                "generation": text,
                            }) + "\n")
                            n_runs += 1
                else:  # sum_topk
                    feat_str = "+".join(str(int(r["feature_idx"])) for r in top_features)
                    for alpha in additive_magnitudes:
                        _set_state({"mode": "additive", "direction": combined_dir, "alpha": float(alpha)})
                        text = _gen_one(model, input_ids, tokenizer, eos_id, hook_obj, terminator_ids, args)
                        fout.write(json.dumps({
                            "mode": "sae_additive", "target": f"sum_top{args.top_k}",
                            "features": feat_str,
                            "magnitude": float(alpha),
                            "prompt_id": p["id"], "category": p["category"],
                            "generation": text,
                        }) + "\n")
                        n_runs += 1

            # ----- C. SAE paper-clamp (per-feature only) -----
            if "sae_clamp" in args.modes:
                for rec in top_features:
                    j = int(rec["feature_idx"])
                    for strength in CLAMP_STRENGTHS:
                        _set_state({"mode": "clamp", "feature_idx": j, "strength": float(strength)})
                        text = _gen_one(model, input_ids, tokenizer, eos_id, hook_obj, terminator_ids, args)
                        fout.write(json.dumps({
                            "mode": "sae_clamp", "target": f"feat_{j}",
                            "feature_idx": j, "delta": rec.get("delta"),
                            "magnitude": float(strength),
                            "prompt_id": p["id"], "category": p["category"],
                            "generation": text,
                        }) + "\n")
                        n_runs += 1
            fout.flush()
            elapsed = time.time() - t0
            print(f"  [{p['id']:<14}] runs={n_runs} elapsed={elapsed:.0f}s")

    handle.remove()
    print(f"[intervene] wrote {out_path} ({n_runs} generations in {time.time()-t0:.0f}s)")


def _gen_one(model, input_ids, tokenizer, eos_id, hook_obj, terminator_ids, args) -> str:
    kwargs = dict(
        max_new_tokens=GEN_TOKENS_PER_INTERVENTION,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=eos_id,
    )
    if args.steer_positions == "sentence_start":
        # Reset per-call: until LogitsProcessor sees a terminator, no steering.
        hook_obj.state["should_steer"] = False
        kwargs["logits_processor"] = [
            _make_steer_processor(hook_obj, terminator_ids, args.steer_k)
        ]
    with torch.no_grad():
        out = model.generate(input_ids, **kwargs)
    new_ids = out[0, input_ids.shape[1]:].tolist()
    return clean_decode(tokenizer.decode(new_ids, skip_special_tokens=False))


if __name__ == "__main__":
    main()
