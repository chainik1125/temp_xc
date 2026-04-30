"""Phase 1 — cache base Llama-3.1-8B activations at all configured hookpoints.

Pulls FineWeb text (or a pre-fetched slice), tokenizes to a fixed seq_length,
runs forward passes with hooks at each `(layer, component)` listed in
config.hookpoints, and writes one float16 npy per hookpoint of shape
(num_sequences, seq_length, d_model) to `paths.acts_dir`.

Hookpoint conventions:
  - `resid` — output of `model.model.layers[layer]` (post-block residual).
  - `attn`  — output of `model.model.layers[layer].self_attn` (attention out).

We use float16 on disk (vs float32) because the model itself runs bf16 and
we'll cast back to fp32 during TXC training; halving the disk + GPU-load
budget is the difference between "fits on the A40 in one tensor" and "doesn't".
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ward_txc.cache")


def _load_corpus(num_sequences: int, seq_length: int, tokenizer, stride: int | None = None):
    """Build `num_sequences` token tensors of length `seq_length`.

    Strategy: prefer Stage A reasoning traces (on-disk, no network, in-domain
    for the backtracking phenomenon — closer to test-time distribution than
    FineWeb generic web text). One trace's full response is usually 1k–4k
    tokens, so we slice it into sliding windows of `seq_length` tokens with
    spacing `stride`. Lower stride = more (overlapping) windows from the
    same corpus; defaults to seq_length (non-overlapping). If the result is
    fewer than `num_sequences`, top up from pre-fetched FineWeb
    (`data/prefetched/`) or stream FineWeb live.
    """
    if stride is None:
        stride = seq_length
    repo_root = Path(__file__).resolve().parents[2]
    texts: list[str] = []

    traces_path = repo_root / "results" / "ward_backtracking" / "traces.json"
    if traces_path.exists():
        log.info("[corpus] sourcing windows from Stage A traces: %s "
                 "(seq_length=%d stride=%d)", traces_path, seq_length, stride)
        traces = json.loads(traces_path.read_text())
        for t in traces:
            full = t.get("full_response") or ""
            if not full:
                continue
            ids = tokenizer(full, add_special_tokens=False)["input_ids"]
            for start in range(0, len(ids) - seq_length, stride):
                texts.append(tokenizer.decode(ids[start:start + seq_length]))
                if len(texts) >= num_sequences:
                    break
            if len(texts) >= num_sequences:
                break
        log.info("[corpus] from traces: %d windows", len(texts))

    if len(texts) < num_sequences:
        # Top up with prefetched FineWeb if available.
        prefetch_candidates = [
            repo_root / "data" / "prefetched" / f"fineweb_{num_sequences}.jsonl",
            repo_root / "data" / "prefetched" / "fineweb.jsonl",
        ]
        for p in prefetch_candidates:
            if p.exists() and len(texts) < num_sequences:
                log.info("[corpus] topping up from %s", p)
                with open(p) as f:
                    for line in f:
                        rec = json.loads(line)
                        txt = rec.get("text") or rec.get("content") or ""
                        if txt and len(txt) > 50:
                            texts.append(txt)
                        if len(texts) >= num_sequences:
                            break
                break

    if not texts:
        raise RuntimeError("no corpus available (no Stage A traces, no prefetched fineweb)")
    texts = texts[:num_sequences]

    # Tokenize to fixed length.
    log.info("[corpus] tokenizing %d texts -> seq_length=%d", len(texts), seq_length)
    out = []
    for txt in texts[:num_sequences]:
        enc = tokenizer(txt, return_tensors="pt", truncation=True,
                        max_length=seq_length, padding="max_length",
                        add_special_tokens=True)
        out.append(enc["input_ids"].squeeze(0))
    return torch.stack(out, dim=0)  # (N, L)


def _attach_hooks(model, hookpoints: list[dict], buffer: dict):
    """Attach the right hook flavor per hookpoint component.

    - `resid`: forward hook on the decoder layer (post-block residual).
    - `attn`: forward hook on `self_attn` (attention output, attn_out).
    - `ln1`: forward-PRE-hook on `self_attn` — captures the *input* to
      attention, which in Llama-style blocks is `input_layernorm(hidden)`,
      i.e. what Dmitry calls "ln1" (post-LN, pre-attention).
    """
    handles = []
    for hp in hookpoints:
        layer_idx = hp["layer"]
        comp = hp["component"]
        key = hp["key"]

        def make_post_hook(k):
            def hook_fn(_m, _i, output):
                acts = output[0] if isinstance(output, tuple) else output
                if acts.dim() == 4:
                    acts = acts.reshape(acts.shape[0], acts.shape[1], -1)
                buffer[k] = acts.detach().to(torch.float16).cpu()
            return hook_fn

        def make_pre_hook(k):
            def hook_fn(_m, args):
                # `args` is the positional input tuple to self_attn; the first
                # is the (B, L, d) hidden state already passed through input_layernorm.
                acts = args[0]
                if acts.dim() == 4:
                    acts = acts.reshape(acts.shape[0], acts.shape[1], -1)
                buffer[k] = acts.detach().to(torch.float16).cpu()
            return hook_fn

        if comp == "resid":
            target = model.model.layers[layer_idx]
            handles.append(target.register_forward_hook(make_post_hook(key)))
        elif comp == "attn":
            target = model.model.layers[layer_idx].self_attn
            handles.append(target.register_forward_hook(make_post_hook(key)))
        elif comp == "ln1":
            target = model.model.layers[layer_idx].self_attn
            handles.append(target.register_forward_pre_hook(make_pre_hook(key)))
        else:
            raise ValueError(f"unknown component: {comp}")
    return handles


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    acts_dir = Path(cfg["paths"]["acts_dir"])
    acts_dir.mkdir(parents=True, exist_ok=True)

    hookpoints = [hp for hp in cfg["hookpoints"] if hp.get("enabled", True)]
    cache_cfg = cfg["cache"]
    num_seq = int(cache_cfg["num_sequences"])
    seq_len = int(cache_cfg["seq_length"])
    stride = int(cache_cfg.get("stride", seq_len))
    bs = int(cache_cfg["cache_batch_size"])
    d_model = int(cfg["txc"]["d_model"])

    # Skip already-cached hookpoints unless --force.
    todo = []
    for hp in hookpoints:
        path = acts_dir / f"{hp['key']}.npy"
        if path.exists() and not args.force:
            try:
                arr = np.load(path, mmap_mode="r")
                if arr.shape == (num_seq, seq_len, d_model):
                    log.info("[skip] %s already cached at %s", hp["key"], path)
                    continue
            except Exception:
                pass
        todo.append(hp)
    if not todo:
        log.info("[done] all hookpoints already cached.")
        return 0

    log.info("[load] base model: %s", cfg["models"]["base"])
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_id = cfg["models"]["base"]
    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16, device_map="cuda",
    ).eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)

    token_ids = _load_corpus(num_seq, seq_len, tok, stride=stride)
    np.save(acts_dir / "token_ids.npy", token_ids.numpy())

    # Allocate per-hookpoint memmaps.
    mmaps = {}
    for hp in todo:
        path = acts_dir / f"{hp['key']}.npy"
        mmaps[hp["key"]] = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float16,
            shape=(num_seq, seq_len, d_model),
        )
        log.info("[alloc] %s -> %s (%.2f GB)", hp["key"], path,
                 mmaps[hp["key"]].nbytes / 1e9)

    buffer: dict = {}
    handles = _attach_hooks(model, todo, buffer)

    try:
        for start in tqdm(range(0, num_seq, bs), desc="forward cache"):
            end = min(start + bs, num_seq)
            batch = token_ids[start:end].to("cuda")
            buffer.clear()
            with torch.no_grad():
                model(batch)
            for hp in todo:
                acts = buffer[hp["key"]]  # (B, L, d) float16 cpu
                mmaps[hp["key"]][start:end] = acts.numpy()
            del batch
    finally:
        for h in handles:
            h.remove()

    for k, mm in mmaps.items():
        mm.flush()

    # Sanity check: a randomly sampled activation should not be all-zero or NaN.
    sample_idx = (3, 7, 0)
    for hp in todo:
        path = acts_dir / f"{hp['key']}.npy"
        arr = np.load(path, mmap_mode="r")
        x = arr[sample_idx]
        log.info(
            "[sanity] %s shape=%s | sample[%s] norm=%.3f | finite=%s",
            hp["key"], arr.shape, sample_idx,
            float(np.linalg.norm(x.astype(np.float32))),
            bool(np.isfinite(x).all()),
        )

    # Sidecar with hookpoint metadata.
    with open(acts_dir / "layer_specs.json", "w") as f:
        json.dump({"hookpoints": todo, "d_model": d_model,
                   "num_sequences": num_seq, "seq_length": seq_len}, f, indent=2)
    log.info("[done] cached %d hookpoints under %s", len(todo), acts_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
