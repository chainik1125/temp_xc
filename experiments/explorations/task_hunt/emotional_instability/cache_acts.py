"""Activation cache over the elicitation rollouts (CARD.md grid).

Re-tokenizes every conversation with the model's chat template
(deterministic — build_labels.py reproduces and asserts the same ids),
forwards gemma-3-12b-it over the full conversation, and captures the
card's layers: screen resid_post L24 = hs25, alternates hs13, hs37.

Conversations are ragged, so storage is a flat token axis:
  /workspace/emo_caches/acts/ids.npy        (N_total,) int32
  /workspace/emo_caches/acts/hs{k}.npy      (N_total, d) fp16 memmap
  /workspace/emo_caches/acts/index.json     conv name -> [start, end)
Batches are length-bucketed with left-padding; only real-token rows are
written. Idempotent (index.json written last).

Run: .venv/bin/python -m experiments.explorations.task_hunt.emotional_instability.cache_acts
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

MODEL_ID = "google/gemma-3-12b-it"
ROLLOUTS = Path("/workspace/emo_caches/rollouts")
OUT = Path("/workspace/emo_caches/acts")
HS_CAPTURE = [25, 13, 37]        # screen layer first (resid_post L24)
SCREEN_HS = 25
BATCH = 4
MAX_LEN = 8192
# gemma-3-12b residual norms (~6e4) saturate fp16 (max 65504) — store
# activations × 1/64. Direction-preserving; the frozen probe stack
# z-scores per dim on train stats, so probes are scale-invariant.
ACT_SCALE = 1.0 / 64


def chat_ids(tok, msgs):
    """Flat token ids from the chat template (transformers 5 returns a
    BatchEncoding; older versions a bare list)."""
    out = tok.apply_chat_template(msgs, tokenize=True,
                                  add_generation_prompt=False)
    ids = out if isinstance(out, list) else out["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return ids


@torch.no_grad()
def main():
    if (OUT / "index.json").exists():
        print("[cache_acts] hit")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    convs = []
    for p in sorted(ROLLOUTS.glob("conv_*.json")):
        msgs = json.loads(p.read_text())["messages"]
        convs.append((p.stem[5:], chat_ids(tok, msgs)[:MAX_LEN]))
    convs.sort(key=lambda c: len(c[1]))          # length bucketing
    total = sum(len(ids) for _, ids in convs)
    print(f"[cache_acts] {len(convs)} convs, {total} tokens", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    d_model = int(model.config.text_config.hidden_size
                  if hasattr(model.config, "text_config")
                  else model.config.hidden_size)

    ids_flat = np.lib.format.open_memmap(
        OUT / "ids.npy", mode="w+", dtype=np.int32, shape=(total,))
    mms = {k: np.lib.format.open_memmap(
        OUT / f"hs{k}.npy", mode="w+", dtype=np.float16,
        shape=(total, d_model)) for k in HS_CAPTURE}
    print(f"[cache_acts] {len(mms)} layers × {total}×{d_model} fp16 "
          f"({sum(m.nbytes for m in mms.values()) / 1e9:.1f} GB)",
          flush=True)

    index, cursor, t0 = {}, 0, time.time()
    pad_id = tok.pad_token_id or 0
    for s in range(0, len(convs), BATCH):
        chunk = convs[s:s + BATCH]
        L = max(len(ids) for _, ids in chunk)
        batch_ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), L), dtype=torch.long)
        for i, (_, ids) in enumerate(chunk):
            batch_ids[i, L - len(ids):] = torch.tensor(ids)   # left pad
            mask[i, L - len(ids):] = 1
        out = model(batch_ids.cuda(), attention_mask=mask.cuda(),
                    output_hidden_states=True, use_cache=False)
        for i, (name, ids) in enumerate(chunk):
            n = len(ids)
            index[name] = [cursor, cursor + n]
            ids_flat[cursor:cursor + n] = np.asarray(ids, dtype=np.int32)
            for k in HS_CAPTURE:
                mms[k][cursor:cursor + n] = (
                    (out.hidden_states[k][i, L - n:].detach().float()
                     * ACT_SCALE).to(torch.float16).cpu().numpy())
            cursor += n
        del out
        if (s // BATCH) % 10 == 0:
            el = time.time() - t0
            print(f"  {cursor}/{total} tok ({el:.0f}s)", flush=True)
    for m in mms.values():
        m.flush()
    ids_flat.flush()
    assert cursor == total
    sample = mms[SCREEN_HS][total // 2].astype(np.float32)
    assert np.isfinite(sample).all() and np.linalg.norm(sample) > 0
    (OUT / "index.json").write_text(json.dumps(
        {"model_id": MODEL_ID, "hs_capture": HS_CAPTURE,
         "screen_hs": SCREEN_HS, "d_model": d_model, "n_tokens": total,
         "act_scale": ACT_SCALE, "max_len": MAX_LEN, "convs": index},
        indent=1))
    print(f"[cache_acts] DONE in {time.time() - t0:.0f}s -> {OUT}",
          flush=True)


if __name__ == "__main__":
    main()
