"""Steering harness — sweep magnitude × source-model × target-model.

For each combination of:
  - steering vector source: {base-derived, reasoning-derived}
  - target model:           {base, reasoning}
  - magnitude:              {0, 4, 8, 12, 16}
  - vector variant:         {union (default)} (per-offset variants
                            available via --vector-mode all)

we hook the residual stream at the configured layer of the target model,
add `magnitude * vector` to its output at every generated token position,
and let it generate `max_new_tokens` tokens for each held-out prompt.
We then compute the keyword rate of "wait"+"hmm" tokens in the decoded
output (case-insensitive, whole-word).

Output: a single results.json with one row per (source, target, magnitude,
prompt_id) and the per-cell aggregate rates.

Notes:
  - Hook lives at model.model.layers[layer] residual output (Llama
    family). Apply addition to the residual *output* (not pre-residual)
    so it shows up at every downstream layer.
  - For the magnitude=0 baseline we still register the hook (to keep
    timing comparable) but the addition is a no-op.
  - Ward's Fig 3 is on the FT model: we expect base→0% across magnitudes
    and reasoning→peaking ~30-50% at magnitude 8-12.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ward.steer_eval")


KEYWORD_RE = re.compile(r"\b(wait|hmm)\b", re.IGNORECASE)


def _short(hf_id: str) -> str:
    return hf_id.split("/")[-1].lower().replace(".", "-")


def _residual_target(model, layer: int):
    return model.model.layers[layer]


def _keyword_rate(text: str) -> float:
    """Fraction of whitespace-delimited words matching {wait, hmm}."""
    n_words = max(1, len(text.split()))
    return len(KEYWORD_RE.findall(text)) / n_words


class _AdditiveHook:
    """Stateful hook that adds `vector * magnitude` to the residual.

    Toggle on/off via .magnitude assignment. Vector is moved to the
    layer's device + dtype on first use so we don't pay it per call.
    """

    def __init__(self, vector: torch.Tensor):
        self._raw = vector.detach()
        self._cached: torch.Tensor | None = None
        self.magnitude: float = 0.0

    def _materialize(self, ref: torch.Tensor) -> torch.Tensor:
        if self._cached is None or self._cached.device != ref.device or self._cached.dtype != ref.dtype:
            self._cached = self._raw.to(device=ref.device, dtype=ref.dtype)
        return self._cached

    def __call__(self, _module, _inp, output):
        if self.magnitude == 0.0:
            return output
        if isinstance(output, tuple):
            x = output[0]
            v = self._materialize(x)
            x = x + self.magnitude * v
            return (x,) + output[1:]
        v = self._materialize(output)
        return output + self.magnitude * v


def _load_lm(hf_id: str, dtype: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dt = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}.get(dtype, torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=dt, device_map=device)
    model.eval()
    return model, tok


def _load_eval_prompts(prompts_path: Path, n: int, seed: int) -> list[dict]:
    """Load held-out eval-split prompts (stamped at seed time).

    Falls back to category-stratified sampling if the prompts file has no
    `split` field (legacy seed runs before the holdout was wired in).
    """
    import random
    items = json.loads(prompts_path.read_text())
    has_split = any("split" in p for p in items)
    if has_split:
        eval_prompts = [p for p in items if p.get("split") == "eval"]
        if not eval_prompts:
            raise ValueError(f"no prompts with split='eval' in {prompts_path}")
        rng = random.Random(seed)
        rng.shuffle(eval_prompts)
        return eval_prompts[:n]
    # Legacy fallback.
    by_cat: dict[str, list[dict]] = {}
    for item in items:
        by_cat.setdefault(item["category"], []).append(item)
    rng = random.Random(seed)
    cats = sorted(by_cat)
    per_cat = max(1, n // len(cats))
    chosen: list[dict] = []
    for cat in cats:
        rng.shuffle(by_cat[cat])
        chosen.extend(by_cat[cat][:per_cat])
    rng.shuffle(chosen)
    return chosen[:n]


def _generate(model, tok, prompts: list[str], max_new_tokens: int) -> list[str]:
    from transformers import GenerationConfig
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tok.pad_token_id,
    )
    outputs: list[str] = []
    for p in prompts:
        # Apply chat template — both base and DeepSeek-R1-Distill expect it
        # for thinking-style traces (skip for base if no template configured).
        try:
            prompt_text = tok.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            prompt_text = p
        inp = tok(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_ids = model.generate(**inp, generation_config=gen_cfg)
        new_tokens = out_ids[0, inp["input_ids"].shape[1]:]
        outputs.append(tok.decode(new_tokens, skip_special_tokens=True))
    return outputs


def _run_target(
    target_tag: str,
    hf_id: str,
    layer: int,
    sources: dict[str, torch.Tensor],
    magnitudes: list[float],
    prompts: list[dict],
    max_new_tokens: int,
    dtype: str,
    device: str,
) -> list[dict]:
    log.info("[load] target=%s hf=%s", target_tag, hf_id)
    model, tok = _load_lm(hf_id, dtype, device)
    layer_module = _residual_target(model, layer)

    rows: list[dict] = []

    for source_tag, vec in sources.items():
        hook = _AdditiveHook(vec)
        handle = layer_module.register_forward_hook(hook)
        try:
            for mag in magnitudes:
                hook.magnitude = float(mag)
                texts = _generate(model, tok, [p["prompt"] for p in prompts], max_new_tokens)
                for prm, txt in zip(prompts, texts):
                    rows.append({
                        "target": target_tag,
                        "source": source_tag,
                        "magnitude": float(mag),
                        "prompt_id": prm["id"],
                        "category": prm["category"],
                        "keyword_rate": _keyword_rate(txt),
                        "n_words": len(txt.split()),
                        "wait_count": len(KEYWORD_RE.findall(txt)),
                        "text": txt,
                    })
                rate = sum(r["keyword_rate"] for r in rows[-len(prompts):]) / len(prompts)
                log.info("    [cell] target=%s source=%s mag=%.1f mean_keyword=%.3f", target_tag, source_tag, mag, rate)
        finally:
            handle.remove()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--vector-mode", choices=["union", "all"], default="union",
                   help="`union` uses one vector per source (mean over offsets); "
                        "`all` adds per-offset vectors as separate sources.")
    p.add_argument("--targets", nargs="+", default=["base", "reasoning"], choices=["base", "reasoning"])
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    out_path = Path(cfg["paths"]["steering"])
    if out_path.exists() and not args.force:
        log.info("[info] resume | %s exists", out_path)
        return 0

    dom = torch.load(Path(cfg["paths"]["dom"]), weights_only=False)
    layer = int(cfg["steering_layer"])

    sources: dict[str, torch.Tensor] = {}
    if args.vector_mode == "union":
        sources["base_derived_union"] = dom["base"]["union"]
        sources["reasoning_derived_union"] = dom["reasoning"]["union"]
    else:
        for src_tag in ("base", "reasoning"):
            offsets = dom[src_tag]["offsets"]
            for i, off in enumerate(offsets):
                sources[f"{src_tag}_derived_off{off}"] = dom[src_tag]["vectors"][i]
        sources["base_derived_union"] = dom["base"]["union"]
        sources["reasoning_derived_union"] = dom["reasoning"]["union"]

    prompts_path = Path(cfg["paths"]["prompts"])
    eval_prompts = _load_eval_prompts(
        prompts_path,
        n=int(cfg["eval"]["n_prompts"]),
        seed=int(cfg["eval"]["seed"]),
    )
    log.info("[info] eval prompts | n=%d", len(eval_prompts))

    magnitudes = [float(m) for m in cfg["eval"]["magnitudes"]]
    max_new_tokens = int(cfg["eval"]["max_new_tokens"])

    all_rows: list[dict] = []
    target_map = {"base": cfg["models"]["base"], "reasoning": cfg["models"]["reasoning"]}
    for tag in args.targets:
        rows = _run_target(
            target_tag=tag,
            hf_id=target_map[tag],
            layer=layer,
            sources=sources,
            magnitudes=magnitudes,
            prompts=eval_prompts,
            max_new_tokens=max_new_tokens,
            dtype=args.dtype,
            device=args.device,
        )
        all_rows.extend(rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "rows": all_rows,
        "meta": {
            "layer": layer,
            "magnitudes": magnitudes,
            "n_eval_prompts": len(eval_prompts),
            "vector_mode": args.vector_mode,
            "targets": args.targets,
        },
    }, indent=2))
    log.info("[done] saved steering results | path=%s | rows=%d", out_path, len(all_rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
