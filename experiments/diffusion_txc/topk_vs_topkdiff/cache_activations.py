"""One-off activation cache: Gemma-2-2B layer-12 resid post on pile-uncopyrighted.

Writes to the mounted volume directory (default /vol):
  cache/shard_{i:03d}.pt   {"acts": (N, d) bf16, "tokens": (N,) int32}
  cache/eval_shard.pt      held-out shard, same format
  cache/eval_seqs.pt       {"input_ids": (64, 128) long} full sequences for dCE
  cache/meta.json          d, n_tokens, rms, mean (fp32 list), model, hook

Run on Modal with the hf-token secret (HF_TOKEN env).
"""

from __future__ import annotations

import json
import pathlib

import torch


def run(out_dir: str = "/vol", n_tokens: int = 10_000_000, ctx: int = 128,
        shard_tokens: int = 1_000_000, batch_seqs: int = 64,
        model_name: str = "google/gemma-2-2b", layer: int = 12,
        dataset: str = "monology/pile-uncopyrighted") -> dict:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, output_hidden_states=True
    ).to(dev).eval()

    out = pathlib.Path(out_dir) / "cache"
    out.mkdir(parents=True, exist_ok=True)
    ds = iter(load_dataset(dataset, split="train", streaming=True))

    def seq_batches():
        buf: list[int] = []
        while True:
            text = next(ds)["text"]
            buf.extend(tok(text, add_special_tokens=False)["input_ids"])
            while len(buf) >= batch_seqs * ctx:
                chunk = torch.tensor(buf[: batch_seqs * ctx], dtype=torch.long)
                buf = buf[batch_seqs * ctx :]
                yield chunk.view(batch_seqs, ctx)

    gen = seq_batches()
    eval_seqs = next(gen)                                   # held out for dCE
    torch.save({"input_ids": eval_seqs}, out / "eval_seqs.pt")

    n_done, shard_i, acts_buf, tok_buf = 0, 0, [], []
    sum_x = torch.zeros(model.config.hidden_size, dtype=torch.float64)
    sum_sq = 0.0
    eval_written = False
    with torch.no_grad():
        while n_done < n_tokens:
            ids = next(gen).to(dev)
            hs = model(ids).hidden_states[layer + 1]         # resid post
            a = hs.reshape(-1, hs.shape[-1])
            acts_buf.append(a.to("cpu", torch.bfloat16))
            tok_buf.append(ids.reshape(-1).to("cpu", torch.int32))
            sum_x += a.float().sum(0).double().cpu()
            sum_sq += float(a.float().pow(2).sum())
            n_done += a.shape[0]
            if sum(t.shape[0] for t in acts_buf) >= shard_tokens:
                payload = {"acts": torch.cat(acts_buf), "tokens": torch.cat(tok_buf)}
                if not eval_written:
                    torch.save(payload, out / "eval_shard.pt")
                    eval_written = True
                else:
                    torch.save(payload, out / f"shard_{shard_i:03d}.pt")
                    shard_i += 1
                acts_buf, tok_buf = [], []
                print(f"cached {n_done:,} tokens", flush=True)

    d = model.config.hidden_size
    meta = {
        "d": d, "n_tokens": n_done, "n_shards": shard_i, "ctx": ctx,
        "model": model_name, "layer": layer, "hook": f"hidden_states[{layer + 1}]",
        "rms": (sum_sq / (n_done * d)) ** 0.5,
        "mean": (sum_x / n_done).tolist(),
        "cache_flops": 2.0 * 2.6e9 * n_done,
    }
    (out / "meta.json").write_text(json.dumps(meta))
    print("META", json.dumps({k: v for k, v in meta.items() if k != "mean"}), flush=True)
    return {k: v for k, v in meta.items() if k != "mean"}
