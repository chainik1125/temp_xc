"""
Forward Gemma-2-2b-it on every prompt in the realbench splits and cache the
last-T residual activations at L13 (mid_res hookpoint).

Saves to safety_research/results/realbench/acts/{split}.npz with keys:
  acts: (N, T, d_model)  fp16
  labels: (N,)            int8
  prompt_idx: (N,)        index into prompts.json
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

T = 5  # window length for stacked / TXC arms
D = 2304
LAYER_INDEX = 13
MODEL_NAME = "google/gemma-2-2b-it"
BENCH = Path("/home/cs29824/andre/temp_xc/safety_research/results/realbench")
OUT = BENCH / "acts"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading {MODEL_NAME} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map=device,
    )
    model.eval()

    # Hook layer 13 to capture residual stream
    captured: list[torch.Tensor] = []

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured.append(h.detach())

    handle = model.model.layers[LAYER_INDEX].register_forward_hook(hook)

    for split in ("train", "test_in", "test_ood"):
        rows = json.load(open(BENCH / f"{split}.json"))
        N = len(rows)
        acts = np.zeros((N, T, D), dtype=np.float16)
        labels = np.zeros(N, dtype=np.int8)
        pbar = tqdm(rows, desc=f"{split}", ncols=100)
        t0 = time.time()
        for i, r in enumerate(pbar):
            chat = [{"role": "user", "content": r["prompt"]}]
            s = tok.apply_chat_template(chat, add_generation_prompt=True,
                                        tokenize=False)
            ids = tok(s, return_tensors="pt", truncation=True,
                      max_length=512).input_ids.to(device)
            captured.clear()
            with torch.no_grad():
                model(ids)
            h = captured[0].squeeze(0)  # (S, d)
            S = h.shape[0]
            window = h[-T:] if S >= T else torch.cat(
                [h[:1].expand(T - S, -1), h], dim=0)
            acts[i] = window.cpu().numpy().astype(np.float16)
            labels[i] = r["label"]
        elapsed = time.time() - t0
        np.savez_compressed(OUT / f"{split}.npz",
                            acts=acts, labels=labels)
        print(f"  saved {split}: {N} x ({T}, {D})  in {elapsed:.0f}s")

    handle.remove()


if __name__ == "__main__":
    main()
