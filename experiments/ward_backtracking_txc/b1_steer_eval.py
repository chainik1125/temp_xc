"""Phase 4 — B1: single-feature steering eval.

For each (hookpoint, mined feature), use the feature's decoder row (at the
chosen T-slot, plus the union across T) as a steering vector, hook it into
the configured steering layer of the *reasoning* model, and sweep magnitudes
on the Stage A eval-split prompts. The keyword-rate metric and prompt format
match Stage A exactly so results are directly comparable.

Output: results/ward_backtracking_txc/steering/b1_steering_results.json
  rows: [{target, source, hookpoint, feature_id, magnitude, prompt_id,
          keyword_rate, n_words, wait_count, text}, ...]
  meta: {layer, magnitudes, hookpoints, top_k_features, baselines}
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
log = logging.getLogger("ward_txc.b1")

KEYWORD_RE = re.compile(r"\b(wait|hmm)\b", re.IGNORECASE)


def _fix_byte_decode(s: str) -> str:
    """Repair byte-level BPE decode artifacts.

    transformers 5.5.4 + this DeepSeek-R1 tokenizer leaves the byte-level
    BPE markers (Ġ for space, Ċ for newline) literally in the decoded
    string instead of converting them. Without this fix, no whitespace
    survives and the keyword regex's \\b boundaries never match.
    """
    return s.replace("Ġ", " ").replace("Ċ", "\n").replace("Â", "")


def _kw_rate(text: str) -> float:
    n_words = max(1, len(text.split()))
    return len(KEYWORD_RE.findall(text)) / n_words


class _Hook:
    def __init__(self, vec: torch.Tensor):
        self._raw = vec.detach()
        self._cached: torch.Tensor | None = None
        self.magnitude = 0.0

    def _materialize(self, ref: torch.Tensor) -> torch.Tensor:
        if self._cached is None or self._cached.device != ref.device or self._cached.dtype != ref.dtype:
            self._cached = self._raw.to(device=ref.device, dtype=ref.dtype)
        return self._cached

    def __call__(self, _m, _i, output):
        if self.magnitude == 0.0:
            return output
        if isinstance(output, tuple):
            x = output[0]
            v = self._materialize(x)
            return (x + self.magnitude * v,) + output[1:]
        v = self._materialize(output)
        return output + self.magnitude * v


def _load_lm(hf_id: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16, device_map=device,
    ).eval()
    return model, tok


def _eval_prompts(prompts_path: Path, n: int, seed: int) -> list[dict]:
    items = json.loads(prompts_path.read_text())
    eval_p = [p for p in items if p.get("split") == "eval"]
    import random
    rng = random.Random(seed)
    rng.shuffle(eval_p)
    return eval_p[:n]


def _generate(model, tok, prompts: list[str], max_new_tokens: int, batch_size: int = 4) -> list[str]:
    """Batched greedy generation with left-padding so all prompts in a batch
    share the same prompt length and we can decode the new tokens directly.
    """
    from transformers import GenerationConfig
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens, do_sample=False,
        temperature=1.0, pad_token_id=tok.pad_token_id,
    )
    # left-pad so the new-token slice index is uniform.
    saved_side = tok.padding_side
    tok.padding_side = "left"
    try:
        # Apply chat template up-front per prompt.
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
        for i in range(0, len(chat_texts), batch_size):
            batch = chat_texts[i:i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=2048).to(model.device)
            prompt_len = enc["input_ids"].shape[1]
            with torch.no_grad():
                out_ids = model.generate(**enc, generation_config=gen_cfg)
            for row in out_ids:
                new = row[prompt_len:]
                outs.append(_fix_byte_decode(tok.decode(new, skip_special_tokens=True)))
        return outs
    finally:
        tok.padding_side = saved_side


def _build_sources(cfg: dict) -> list[dict]:
    """Return list of {tag, vector, hookpoint, arch, feature_id, mode} entries.

    Includes:
      - DoM baselines (base, reasoning) at the Ward layer — Stage A vectors.
      - For each (arch in cfg.txc.arch_list) × (enabled hookpoint) with a
        features file: top-K features × {pos0, union}.
    """
    paths = cfg["paths"]; mining = cfg["mining"]
    sources: list[dict] = []

    # Stage A DoM vectors as baselines.
    dom_path = Path(paths["stageA_dom"])
    if dom_path.exists():
        dom = torch.load(dom_path, weights_only=False)
        sources.append({
            "tag": "dom_base_union", "vector": dom["base"]["union"].clone(),
            "hookpoint": "resid_L10", "arch": "dom", "feature_id": -1, "mode": "dom",
        })
        sources.append({
            "tag": "dom_reasoning_union", "vector": dom["reasoning"]["union"].clone(),
            "hookpoint": "resid_L10", "arch": "dom", "feature_id": -1, "mode": "dom",
        })
    else:
        log.warning("[warn] %s missing — skipping DoM baselines", dom_path)

    feat_dir = Path(paths["features_dir"])
    K_steer = int(mining["top_k_for_steering"])
    arch_list = cfg["txc"].get("arch_list", ["txc"])
    for arch in arch_list:
        for hp in cfg["hookpoints"]:
            if not hp.get("enabled", True):
                continue
            # txc keeps legacy filename (no arch prefix); others get one.
            fpath = feat_dir / (
                f"{hp['key']}.npz" if arch == "txc" else f"{arch}_{hp['key']}.npz"
            )
            if not fpath.exists():
                log.warning("[warn] %s missing — skipping %s/%s", fpath, arch, hp["key"])
                continue
            z = np.load(fpath, allow_pickle=True)
            top = z["top_features"][:K_steer].tolist()
            dec_pos0 = z["decoder_at_pos0"][:K_steer]
            dec_union = z["decoder_union"][:K_steer]
            for i, fid in enumerate(top):
                sources.append({
                    "tag": f"{arch}_{hp['key']}_f{fid}_pos0",
                    "vector": torch.from_numpy(dec_pos0[i]).float(),
                    "hookpoint": hp["key"], "arch": arch,
                    "feature_id": int(fid), "mode": "pos0",
                })
                # For arches with no T axis (topk_sae, tsae) pos0==union;
                # skip the duplicate union source to keep the budget honest.
                if arch in ("topk_sae", "tsae"):
                    continue
                sources.append({
                    "tag": f"{arch}_{hp['key']}_f{fid}_union",
                    "vector": torch.from_numpy(dec_union[i]).float(),
                    "hookpoint": hp["key"], "arch": arch,
                    "feature_id": int(fid), "mode": "union",
                })
    return sources


def _normalize_to(vec: torch.Tensor, ref_norm: float) -> torch.Tensor:
    """Rescale vec so its L2 norm equals ref_norm.

    Stage A's DoM vector has a particular norm; comparing TXC decoder rows
    (which are unit-normalized post-training) at the same magnitude scale is
    apples-to-oranges. We rescale every steering vector to the DoM-base
    union's norm so the magnitude axis is comparable.
    """
    n = vec.norm().clamp_min(1e-8)
    return vec / n * ref_norm


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-sources", type=int, default=None,
                   help="cap number of TXC sources for a quick coarse pass")
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    paths = cfg["paths"]
    out_path = Path(paths["steering"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.force:
        log.info("[resume] %s exists", out_path); return 0

    sources = _build_sources(cfg)
    if args.max_sources is not None:
        # Always keep DoM baselines.
        dom_src = [s for s in sources if s["mode"] == "dom"]
        txc_src = [s for s in sources if s["mode"] != "dom"]
        sources = dom_src + txc_src[: args.max_sources]
    log.info("[sources] %d total", len(sources))

    # Norm-normalize TXC sources to the DoM-base norm if available.
    ref_norm = None
    for s in sources:
        if s["tag"] == "dom_base_union":
            ref_norm = float(s["vector"].norm().item())
            log.info("[ref] dom_base_union norm = %.3f", ref_norm); break
    if ref_norm is not None:
        for s in sources:
            if s["mode"] != "dom":
                s["vector"] = _normalize_to(s["vector"], ref_norm)

    # Eval prompts.
    eval_prompts = _eval_prompts(
        Path(paths["stageA_prompts"]),
        n=int(cfg["steering"]["n_eval_prompts"]),
        seed=int(cfg["steering"]["seed"]),
    )
    log.info("[prompts] %d eval", len(eval_prompts))
    magnitudes = [float(m) for m in cfg["steering"]["magnitudes"]]
    max_new = int(cfg["steering"]["max_new_tokens"])
    layer = int(cfg["steering"]["steering_layer"])
    gen_bs = int(cfg["steering"].get("gen_batch_size", 4))

    rows: list[dict] = []
    for tag in cfg["steering"]["targets"]:
        hf_id = cfg["models"][tag]
        log.info("[load] target=%s hf=%s", tag, hf_id)
        model, tok = _load_lm(hf_id, args.device)
        layer_module = model.model.layers[layer]

        for s_i, src in enumerate(sources):
            log.info("[source %d/%d] %s", s_i + 1, len(sources), src["tag"])
            hook = _Hook(src["vector"])
            handle = layer_module.register_forward_hook(hook)
            try:
                for mag in magnitudes:
                    hook.magnitude = float(mag)
                    texts = _generate(model, tok,
                                      [p["prompt"] for p in eval_prompts],
                                      max_new_tokens=max_new,
                                      batch_size=gen_bs)
                    cell_rates = []
                    for prm, txt in zip(eval_prompts, texts):
                        rate = _kw_rate(txt)
                        cell_rates.append(rate)
                        rows.append({
                            "target": tag,
                            "source": src["tag"],
                            "hookpoint": src["hookpoint"],
                            "feature_id": src["feature_id"],
                            "mode": src["mode"],
                            "magnitude": float(mag),
                            "prompt_id": prm["id"],
                            "category": prm["category"],
                            "keyword_rate": rate,
                            "n_words": len(txt.split()),
                            "wait_count": len(KEYWORD_RE.findall(txt)),
                            "text": txt,
                        })
                    log.info("    mag=%+.0f mean_kw=%.4f", mag,
                             sum(cell_rates) / max(1, len(cell_rates)))
            finally:
                handle.remove()
        del model
        torch.cuda.empty_cache()

    out_path.write_text(json.dumps({
        "rows": rows,
        "meta": {
            "layer": layer,
            "magnitudes": magnitudes,
            "n_eval_prompts": len(eval_prompts),
            "n_sources": len(sources),
            "ref_norm": ref_norm,
            "targets": list(cfg["steering"]["targets"]),
        },
    }, indent=2))
    log.info("[done] saved %s | rows=%d", out_path, len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
