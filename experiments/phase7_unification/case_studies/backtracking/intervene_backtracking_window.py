#!/usr/bin/env python3
"""Window-aware steered generation for TXCBareAntidead and other window
encoders.

`intervene_backtracking.py` assumes per-token features (encode(x: (B, d)) →
(B, d_sae)). TXC's encode takes a T-token window → (B, d_sae) and decodes
back to a (B, T, d) window. To run TXC inside `model.generate()` we need
to maintain a buffer of the last T-1 residual-stream activations and form
a fresh window each generation step.

Modes:
  A. raw_dom        — additive steering with the raw activation-space
                      DoM vector. Identical to the per-token script.
  B. sae_additive   — additive: h' = h + α · W_dec[j, T-1, :] / ‖·‖.
                      We use the right-edge decoder slice (the position the
                      model is currently generating) as the steering
                      direction.
  C. sae_clamp      — paper-clamp on the window:
                          window = [buffer; h_t]
                          z = encode(window)
                          z'[..., j] = strength
                          h_t' = decode(z')[T-1] + (h_t - decode(z)[T-1])
                      Falls through unchanged for the first T-1 steps
                      (insufficient buffer).

Reads decompose results from `decompose<_suffix>/` (with `top_features.json`
and `raw_dom.fp16.npy`). For TXC checkpoints the decompose script needs to
have already been run with `--local-ckpt-window <txc_ckpt>` — that's a
separate decompose path not yet implemented; for now this script just
loads the TXC ckpt and computes the SAE-feature direction and steering
on-the-fly without writing a decompose dir. Use `--top-feature-idx J`
explicitly to specify which TXC feature to steer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("TQDM_DISABLE", "1")

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.architectures.txc_bare_antidead import TXCBareAntidead  # noqa: E402
from src.data.nlp.cache_activations import load_model_and_tokenizer  # noqa: E402
from src.data.nlp.models import resid_hook_target  # noqa: E402

from experiments.phase7_unification.case_studies.backtracking._decode import (  # noqa: E402
    clean_decode,
)
from experiments.phase7_unification.case_studies.backtracking._paths import (  # noqa: E402
    ADDITIVE_MAGNITUDES,
    ANCHOR_LAYER,
    GEN_TOKENS_PER_INTERVENTION,
    N_HELDOUT,
    RESULTS_DIR,
    SUBJECT_MODEL,
    ensure_dirs,
)
from experiments.phase7_unification.case_studies.backtracking.prompts import (  # noqa: E402
    PROMPTS,
)


# ── window-aware steering hook ──────────────────────────────────────────────
class WindowSteeringHook:
    """Maintains a T-1 deque of past residuals; on each new-token forward
    forms the T-window and steers the right-edge position.

    state keys:
        mode             "off" | "additive" | "clamp"
        direction        unit-norm tensor (d_in,) on h.device, model dtype
        alpha            float
        feature_idx      int (clamp)
        strength         float (clamp)
    """

    def __init__(self, txc: TXCBareAntidead, T: int):
        self.state: dict = {"mode": "off"}
        self.txc = txc
        self.T = T
        self.buffer: deque = deque(maxlen=T - 1)

    def reset(self) -> None:
        self.buffer.clear()

    def __call__(self, module, inputs, output):
        if self.state["mode"] == "off":
            return output
        h = output[0] if isinstance(output, tuple) else output
        # During the prompt forward, h.shape[1] > 1. Update the buffer with
        # all prompt tokens (so the first generated token has T-1 priors)
        # but don't steer the prompt.
        if h.shape[1] > 1:
            for t in range(h.shape[1]):
                self.buffer.append(h[0, t].detach())
            return output
        if len(self.buffer) < self.T - 1:
            self.buffer.append(h[0, 0].detach())
            return output

        # Form the T-window: (T-1 past) + (1 new) → (1, T, d_in)
        window_list = list(self.buffer) + [h[0, 0]]
        window = torch.stack(window_list, dim=0).unsqueeze(0)  # (1, T, d_in)

        mode = self.state["mode"]
        if mode == "additive":
            # h += α · direction (per-token; direction is a single (d_in,) vec)
            direction = self.state["direction"]
            alpha = float(self.state["alpha"])
            h_new = h + alpha * direction
            self.buffer.append(h_new[0, 0].detach())
            if isinstance(output, tuple):
                return (h_new,) + output[1:]
            return h_new
        elif mode == "clamp":
            j = int(self.state["feature_idx"])
            strength = float(self.state["strength"])
            with torch.no_grad():
                z = self.txc.encode(window)  # (1, d_sae)
                window_recon = self.txc.decode(z)  # (1, T, d_in)
                z_prime = z.clone()
                z_prime[..., j] = strength
                window_recon_prime = self.txc.decode(z_prime)
            # Right-edge error preservation
            h_t_recon = window_recon[:, -1, :]      # (1, d_in)
            h_t_recon_prime = window_recon_prime[:, -1, :]  # (1, d_in)
            h_new_t = h[0, 0] - h_t_recon[0] + h_t_recon_prime[0]  # (d_in,)
            h_new = h.clone()
            h_new[0, 0] = h_new_t
            self.buffer.append(h_new[0, 0].detach())
            if isinstance(output, tuple):
                return (h_new,) + output[1:]
            return h_new
        else:
            raise ValueError(f"unknown mode {mode!r}")


# ── TXC ckpt loader ─────────────────────────────────────────────────────────
def load_txc(ckpt_path: Path, device: str) -> tuple[TXCBareAntidead, dict]:
    meta = json.loads((ckpt_path.parent / "meta.json").read_text())
    if meta.get("arch") != "txc_bare":
        raise SystemExit(f"expected arch=txc_bare; got {meta.get('arch')!r}")
    txc = TXCBareAntidead(
        d_in=int(meta["d_in"]),
        d_sae=int(meta["d_sae"]),
        T=int(meta["T"]),
        k=int(meta["k_pos"]) * int(meta["T"]),
    )
    txc.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    txc.to(device)
    txc.eval()
    for p in txc.parameters():
        p.requires_grad_(False)
    return txc, meta


def held_out_prompts(stratified: bool = False) -> list[dict]:
    if not stratified:
        return list(PROMPTS[-N_HELDOUT:])
    by_cat: dict[str, list] = {}
    for p in PROMPTS:
        by_cat.setdefault(p["category"], []).append(p)
    cats = sorted(by_cat.keys())
    per_cat = max(1, N_HELDOUT // len(cats))
    out: list[dict] = []
    for cat in cats:
        out.extend(by_cat[cat][-per_cat:])
    return out[:N_HELDOUT]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--txc-ckpt", required=True, help="path to TXC ckpt.pt")
    parser.add_argument("--feature-idx", type=int, default=None,
                        help="explicit feature index to steer; if None, picks rank-0 from the top_features.json next to the ckpt (which only exists if the user already wrote a window-decompose for this ckpt)")
    parser.add_argument("--layer", type=int, default=ANCHOR_LAYER)
    parser.add_argument("--modes", nargs="+", default=["sae_additive"], choices=["sae_additive", "sae_clamp"])
    parser.add_argument("--magnitudes", type=str, default=None)
    parser.add_argument("--clamp-strengths", type=str, default="0,5,10,25,50,100")
    parser.add_argument("--limit-prompts", type=int, default=None)
    parser.add_argument("--held-out-stratified", action="store_true")
    parser.add_argument("--intervene-suffix", default="txc")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    add_mags = (
        tuple(float(s) for s in args.magnitudes.split(","))
        if args.magnitudes is not None
        else ADDITIVE_MAGNITUDES
    )
    clamp_strengths = tuple(float(s) for s in args.clamp_strengths.split(","))

    ensure_dirs()
    out_dir = RESULTS_DIR / f"intervene_{args.intervene_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "generations.jsonl"
    if out_path.exists() and not args.force:
        print(f"[intervene-window] {out_path} exists; --force to rebuild")
        return

    print(f"[intervene-window] loading {SUBJECT_MODEL}")
    model, tokenizer, cfg = load_model_and_tokenizer(SUBJECT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_id = tokenizer.eos_token_id
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    txc, meta = load_txc(Path(args.txc_ckpt), str(device))
    txc = txc.to(model_dtype)
    T = int(meta["T"])
    print(f"[intervene-window] TXC d_sae={meta['d_sae']} T={T} from {args.txc_ckpt}")

    # Pick the feature: explicit --feature-idx wins; otherwise default to 0.
    j = args.feature_idx if args.feature_idx is not None else 0
    print(f"[intervene-window] steering feature_idx={j}")

    # Right-edge decoder slice for additive direction.
    # TXCBareAntidead.W_dec shape: (d_sae, T, d_in)
    with torch.no_grad():
        W_dec = txc.W_dec  # (d_sae, T, d_in) — confirm
        if W_dec.dim() != 3:
            raise SystemExit(f"unexpected TXC W_dec shape {tuple(W_dec.shape)}")
        feat_dir = W_dec[j, -1, :].to(device=device, dtype=torch.float32)  # right-edge slice
        feat_dir = feat_dir / max(feat_dir.norm().item(), 1e-8)
        feat_dir = feat_dir.to(model_dtype)

    layer_mod = resid_hook_target(model, args.layer, cfg.architecture_family)
    hook_obj = WindowSteeringHook(txc=txc, T=T)
    handle = layer_mod.register_forward_hook(hook_obj)

    prompts = held_out_prompts(stratified=args.held_out_stratified)
    if args.limit_prompts is not None:
        prompts = prompts[: args.limit_prompts]
    print(f"[intervene-window] {len(prompts)} held-out prompts; modes={args.modes}; T={T}")

    n_runs = 0
    t0 = time.time()
    with out_path.open("w") as fout:
        for p in prompts:
            messages = [{"role": "user", "content": p["text"]}]
            encoded = tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            )
            input_ids = encoded.to(device) if isinstance(encoded, torch.Tensor) else encoded["input_ids"].to(device)

            if "sae_additive" in args.modes:
                for alpha in add_mags:
                    hook_obj.reset()
                    hook_obj.state = {"mode": "additive", "direction": feat_dir, "alpha": float(alpha)}
                    text = _gen_one(model, input_ids, tokenizer, eos_id)
                    fout.write(json.dumps({
                        "mode": "sae_additive", "target": f"feat_{j}",
                        "feature_idx": j, "magnitude": float(alpha),
                        "prompt_id": p["id"], "category": p["category"],
                        "generation": text,
                    }) + "\n")
                    n_runs += 1
            if "sae_clamp" in args.modes:
                for strength in clamp_strengths:
                    hook_obj.reset()
                    hook_obj.state = {"mode": "clamp", "feature_idx": j, "strength": float(strength)}
                    text = _gen_one(model, input_ids, tokenizer, eos_id)
                    fout.write(json.dumps({
                        "mode": "sae_clamp", "target": f"feat_{j}",
                        "feature_idx": j, "magnitude": float(strength),
                        "prompt_id": p["id"], "category": p["category"],
                        "generation": text,
                    }) + "\n")
                    n_runs += 1
            fout.flush()
            print(f"  [{p['id']:<14}] runs={n_runs} elapsed={time.time()-t0:.0f}s")

    handle.remove()
    print(f"[intervene-window] wrote {out_path} ({n_runs} generations in {time.time()-t0:.0f}s)")


def _gen_one(model, input_ids, tokenizer, eos_id) -> str:
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=GEN_TOKENS_PER_INTERVENTION,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=eos_id,
            use_cache=False,  # window encoder needs full sequence per step
        )
    new_ids = out[0, input_ids.shape[1]:].tolist()
    return clean_decode(tokenizer.decode(new_ids, skip_special_tokens=False))


if __name__ == "__main__":
    main()
