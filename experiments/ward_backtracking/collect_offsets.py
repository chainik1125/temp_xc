"""Collect per-token-offset residual-stream activations.

For each labelled sentence in each trace, we want the activation at the
*token-level offset preceding the sentence start*. Ward Fig 2 sweeps
offsets in [-20, 0]; their layer-10 sweet spot is [-13, -8]. We collect
single-token activations at every offset in our config's `offsets.single`
list (default [-13..-8]) plus offset 0 (the no-offset baseline).

Output (numpy npz, one file per source model):
  acts_{model}.npz
    keys:
      offsets:           int array of shape (n_offsets,)
      activations:       float32 array of shape (n_sentences, n_offsets, d_model)
      is_backtracking:   bool array of shape (n_sentences,)
      sentence_keys:     str array of shape (n_sentences,) — '{qid}|{trace_idx}|{sent_idx}'

Why one file per source model: Ward's central trick is computing the DoM
on *base* model activations and steering the *reasoning* model with it.
We need both — and we'll later compare to the same DoM on reasoning
model activations. So this script runs once per model in the config.

Notes on memory: at d_model=4096, n_offsets=7, ~6000 sentences across
300 traces, the activations array is ~700 MB float32 per model. Fine on
A40 (40 GB) and a 100 GB volume.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ward.collect_offsets")


def _short_model_name(hf_id: str) -> str:
    return hf_id.split("/")[-1].lower().replace(".", "-")


def _load_model_and_tokenizer(hf_id: str, device: str, dtype_str: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(dtype_str, torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch_dtype, device_map=device)
    model.eval()
    return model, tok


def _residual_target(model, layer: int):
    """Locate the residual-stream output at `layer` for Llama-family models.

    Same convention as src.bench.model_registry.resid_hook_target for the
    'llama' family: model.model.layers[layer]. We pin to that here to avoid
    importing the registry — both base Llama-3.1-8B and DeepSeek-R1-Distill
    are llama-architecture.
    """
    return model.model.layers[layer]


def _collect_for_model(
    hf_id: str,
    layer: int,
    offsets: list[int],
    traces_by_qid: dict[str, dict],
    labels: list[dict],
    dom_qids: set[str],
    device: str,
    dtype: str,
) -> dict:
    import torch

    model, tok = _load_model_and_tokenizer(hf_id, device=device, dtype_str=dtype)
    target = _residual_target(model, layer)

    captured: dict = {}

    def hook(_m, _i, output):
        captured["x"] = output[0] if isinstance(output, tuple) else output

    handle = target.register_forward_hook(hook)

    d_model = model.config.hidden_size
    rows: list[np.ndarray] = []
    is_bt: list[bool] = []
    keys: list[str] = []

    debug_emitted = 0  # eyeball-check: print decoded tokens at offsets for first 5 backtracks

    try:
        for record in labels:
            qid = record["question_id"]
            if qid not in dom_qids:
                continue
            trace = traces_by_qid.get(qid)
            if trace is None or not record["sentences"]:
                continue
            full_response = trace["full_response"]
            # We tokenize the *full response* (which includes the <think>
            # block) so character-level sentence offsets inside the
            # thinking section line up with token positions.
            enc = tok(full_response, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
            input_ids = enc["input_ids"].to(model.device)
            offsets_map = enc["offset_mapping"][0].tolist()  # list of (cs, ce)

            with torch.no_grad():
                _ = model(input_ids=input_ids)
            acts = captured["x"][0].detach().to(torch.float32).cpu().numpy()  # (seq, d)
            seq_len = acts.shape[0]

            # extract_thinking_process strips the <think> tags but keeps
            # internal text; the labeller's char_start is relative to the
            # extracted thinking string, so we must add the position of
            # the first char *after* the opening tag in the full response.
            think_open = "<think>"
            think_idx = full_response.find(think_open)
            think_offset = (think_idx + len(think_open)) if think_idx >= 0 else 0

            for s_idx, sent in enumerate(record["sentences"]):
                # Char position in the FULL response.
                target_char = think_offset + sent["char_start"]
                # Find the token whose char-span starts at or just before target_char.
                tok_pos = -1
                for i, (cs, ce) in enumerate(offsets_map):
                    if cs <= target_char < ce or cs >= target_char:
                        tok_pos = i
                        break
                if tok_pos < 0:
                    continue

                row = np.zeros((len(offsets), d_model), dtype=np.float32)
                ok = True
                for j, off in enumerate(offsets):
                    p = tok_pos + off
                    if p < 0 or p >= seq_len:
                        ok = False
                        break
                    row[j] = acts[p]
                if not ok:
                    continue

                # Eyeball-check: for the first 5 backtracking sentences, log
                # what's actually sitting at the canonical offsets. If
                # offset -10 is mid-previous-sentence we're aligned; if it's
                # 3 sentences off something upstream is wrong and the DoM
                # will average over noise.
                if sent["is_backtracking"] and debug_emitted < 5:
                    show_offsets = [-13, -8, 0]
                    snippets = []
                    for off in show_offsets:
                        p = tok_pos + off
                        if 0 <= p < seq_len:
                            tok_id = int(input_ids[0, p].item())
                            tok_str = tok.decode([tok_id])
                            snippets.append(f"off={off:+d} tok={tok_str!r}")
                    log.info(
                        "[align] sentence=%r | %s",
                        sent["sentence"][:80] + ("…" if len(sent["sentence"]) > 80 else ""),
                        " | ".join(snippets),
                    )
                    debug_emitted += 1

                rows.append(row)
                is_bt.append(sent["is_backtracking"])
                keys.append(f"{qid}|{record['trace_idx']}|{s_idx}")

            del acts
            captured.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        handle.remove()

    if not rows:
        raise RuntimeError(f"no activations collected for model={hf_id}")

    arr = np.stack(rows, axis=0)  # (N, n_offsets, d_model)
    log.info("[done] %s | n_sentences=%d | n_backtracking=%d | shape=%s",
             hf_id, arr.shape[0], int(np.sum(is_bt)), arr.shape)
    return {
        "offsets": np.asarray(offsets, dtype=np.int32),
        "activations": arr,
        "is_backtracking": np.asarray(is_bt, dtype=bool),
        "sentence_keys": np.asarray(keys, dtype=object),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--only", choices=["base", "reasoning"], default=None,
                   help="restrict to one model (default: both)")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())

    prompts = json.loads(Path(cfg["paths"]["prompts"]).read_text())
    dom_qids = {p["id"] for p in prompts if p.get("split", "dom") == "dom"}
    log.info("[info] dom-split prompts: %d (eval-split %d)",
             len(dom_qids), len(prompts) - len(dom_qids))

    traces = json.loads(Path(cfg["paths"]["traces"]).read_text())
    traces_by_qid = {t["question_id"]: t for t in traces}
    labels = json.loads(Path(cfg["paths"]["sentence_labels"]).read_text())

    layer = int(cfg["steering_layer"])
    offsets = list(cfg["offsets"]["single"]) + [int(cfg["offsets"]["baseline_zero"])]

    acts_dir = Path(cfg["paths"]["acts_dir"])
    acts_dir.mkdir(parents=True, exist_ok=True)

    targets = []
    if args.only in (None, "base"):
        targets.append(("base", cfg["models"]["base"]))
    if args.only in (None, "reasoning"):
        targets.append(("reasoning", cfg["models"]["reasoning"]))

    for tag, hf_id in targets:
        out_path = acts_dir / f"acts_{tag}_{_short_model_name(hf_id)}.npz"
        if out_path.exists() and not args.force:
            log.info("[info] resume | %s exists", out_path)
            continue
        result = _collect_for_model(
            hf_id=hf_id,
            layer=layer,
            offsets=offsets,
            traces_by_qid=traces_by_qid,
            labels=labels,
            dom_qids=dom_qids,
            device=args.device,
            dtype=args.dtype,
        )
        np.savez(out_path, **result)
        log.info("[saved] %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
