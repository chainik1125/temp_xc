#!/usr/bin/env python3
"""Stage 4: steered generation across three modes.

Modes (all applied on the residual-stream output of layer ANCHOR_LAYER):

    A. raw_dom        — additive: h' = h + α · v_raw / ‖v_raw‖
                        where v_raw is the activation-space DoM from Stage 3.
                        Magnitudes ADDITIVE_MAGNITUDES.
    B. sae_additive   — additive: h' = h + α · W_dec[j] / ‖W_dec[j]‖
                        for each top-K SAE feature j by |Δ_j|.
                        Magnitudes ADDITIVE_MAGNITUDES.
    C. sae_clamp      — paper-clamp with error preservation:
                            z = sae.encode(h); z'[..., j] = strength;
                            h' = h - sae.decode(z) + sae.decode(z')
                        Strengths CLAMP_STRENGTHS.

For every (mode, target, magnitude, prompt) tuple we run greedy generation of
GEN_TOKENS_PER_INTERVENTION new tokens from the held-out prompt set (last
N_HELDOUT prompts in `prompts.PROMPTS`). Prompts used for the cache are
kept disjoint from the steering-evaluation set when N_PROMPTS > N_HELDOUT.

Output: `intervene/generations.jsonl`, one row per (mode, target, magnitude,
prompt_id) with the decoded continuation.
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

from experiments.phase7_unification.case_studies.backtracking._paths import (  # noqa: E402
    ADDITIVE_MAGNITUDES,
    ANCHOR_LAYER,
    CLAMP_STRENGTHS,
    DECOMPOSE_DIR,
    GEN_TOKENS_PER_INTERVENTION,
    INTERVENE_DIR,
    N_HELDOUT,
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
    """

    def __init__(self, sae=None):
        self.state: dict = {"mode": "off"}
        self.sae = sae

    def __call__(self, module, inputs, output):
        if self.state["mode"] == "off":
            return output
        h = output[0] if isinstance(output, tuple) else output
        mode = self.state["mode"]
        if mode == "additive":
            direction = self.state["direction"]   # (d_model,) on h.device, unit-norm
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


# ── steering targets ────────────────────────────────────────────────────────
def load_targets(top_k: int):
    raw_dom = np.load(DECOMPOSE_DIR / "raw_dom.fp16.npy").astype(np.float32)
    raw_dom_unit = raw_dom / max(np.linalg.norm(raw_dom), 1e-8)
    top = json.loads((DECOMPOSE_DIR / "top_features.json").read_text())[:top_k]
    return raw_dom_unit, top


def held_out_prompts() -> list[dict]:
    return list(PROMPTS[-N_HELDOUT:])


# ── runner ──────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=5, help="how many SAE features to intervene on")
    parser.add_argument("--layer", type=int, default=ANCHOR_LAYER)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["raw_dom", "sae_additive", "sae_clamp"],
        choices=["raw_dom", "sae_additive", "sae_clamp"],
    )
    parser.add_argument("--limit-prompts", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    out_path = INTERVENE_DIR / "generations.jsonl"
    if out_path.exists() and not args.force:
        print(f"[intervene] {out_path} exists; use --force to rebuild")
        return

    raw_dom_unit, top_features = load_targets(args.top_k)

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

        sae, _ = load_public_sae(args.layer, str(device))
        sae = sae.to(model_dtype)

    # Pre-compute SAE feature decoder unit vectors (in model dtype, on device).
    feature_dirs: dict[int, torch.Tensor] = {}
    if "sae_additive" in args.modes:
        W_dec = sae.W_dec.detach()
        if W_dec.shape[0] == cfg.d_model:  # if stored as (d_model, d_sae), transpose
            W_dec = W_dec.T
        for rec in top_features:
            j = int(rec["feature_idx"])
            v = W_dec[j].to(device=device, dtype=torch.float32)
            v = v / max(v.norm().item(), 1e-8)
            feature_dirs[j] = v.to(model_dtype)

    raw_dir = torch.tensor(raw_dom_unit, device=device, dtype=model_dtype)

    layer_mod = resid_hook_target(model, args.layer, cfg.architecture_family)
    hook_obj = SteeringHook(sae=sae)
    handle = layer_mod.register_forward_hook(hook_obj)

    prompts = held_out_prompts()
    if args.limit_prompts is not None:
        prompts = prompts[: args.limit_prompts]
    print(f"[intervene] {len(prompts)} held-out prompts; modes={args.modes}; top-K={args.top_k}")

    n_runs = 0
    t0 = time.time()
    with out_path.open("w") as fout:
        for p in prompts:
            messages = [{"role": "user", "content": p["text"]}]
            encoded = tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            )
            if isinstance(encoded, torch.Tensor):
                input_ids = encoded.to(device)
            else:
                input_ids = encoded["input_ids"].to(device)

            # ----- A. raw DoM baseline -----
            if "raw_dom" in args.modes:
                for alpha in ADDITIVE_MAGNITUDES:
                    hook_obj.state = {"mode": "additive", "direction": raw_dir, "alpha": float(alpha)}
                    text = _gen_one(model, input_ids, tokenizer, eos_id)
                    fout.write(
                        json.dumps(
                            {
                                "mode": "raw_dom",
                                "target": "raw_dom",
                                "magnitude": float(alpha),
                                "prompt_id": p["id"],
                                "category": p["category"],
                                "generation": text,
                            }
                        )
                        + "\n"
                    )
                    n_runs += 1

            # ----- B. SAE additive on each top-K feature -----
            if "sae_additive" in args.modes:
                for rec in top_features:
                    j = int(rec["feature_idx"])
                    direction = feature_dirs[j]
                    for alpha in ADDITIVE_MAGNITUDES:
                        hook_obj.state = {"mode": "additive", "direction": direction, "alpha": float(alpha)}
                        text = _gen_one(model, input_ids, tokenizer, eos_id)
                        fout.write(
                            json.dumps(
                                {
                                    "mode": "sae_additive",
                                    "target": f"feat_{j}",
                                    "feature_idx": j,
                                    "delta": rec["delta"],
                                    "magnitude": float(alpha),
                                    "prompt_id": p["id"],
                                    "category": p["category"],
                                    "generation": text,
                                }
                            )
                            + "\n"
                        )
                        n_runs += 1

            # ----- C. SAE paper-clamp on each top-K feature -----
            if "sae_clamp" in args.modes:
                for rec in top_features:
                    j = int(rec["feature_idx"])
                    for strength in CLAMP_STRENGTHS:
                        hook_obj.state = {"mode": "clamp", "feature_idx": j, "strength": float(strength)}
                        text = _gen_one(model, input_ids, tokenizer, eos_id)
                        fout.write(
                            json.dumps(
                                {
                                    "mode": "sae_clamp",
                                    "target": f"feat_{j}",
                                    "feature_idx": j,
                                    "delta": rec["delta"],
                                    "magnitude": float(strength),
                                    "prompt_id": p["id"],
                                    "category": p["category"],
                                    "generation": text,
                                }
                            )
                            + "\n"
                        )
                        n_runs += 1
            fout.flush()
            elapsed = time.time() - t0
            print(f"  [{p['id']:<14}] runs={n_runs} elapsed={elapsed:.0f}s")

    handle.remove()
    print(f"[intervene] wrote {out_path} ({n_runs} generations in {time.time()-t0:.0f}s)")


def _gen_one(model, input_ids, tokenizer, eos_id) -> str:
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=GEN_TOKENS_PER_INTERVENTION,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=eos_id,
        )
    new_ids = out[0, input_ids.shape[1] :].tolist()
    return tokenizer.decode(new_ids, skip_special_tokens=False)


if __name__ == "__main__":
    main()
