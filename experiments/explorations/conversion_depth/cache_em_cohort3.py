"""em-redo — re-cache the stage-4 cohort at hs 10/14/16 only.

The phase-4 sweep cached ALL 29 hidden states of the 1728-rollout
balanced-α cohort and the shards were deleted per the conversion-depth
briefing after the probe stats were written (labels/lens/qids/meta
sidecars kept). The em-redo readout currencies need the cohort
activations at exactly the three panel layers, so this re-runs the
IDENTICAL phase4_em_depth.py forward (same cohort filter, same row
order, same chat-template/assistant-only ≤100-token convention) and
writes only hs{10,14,16}.npy back into the same directory.

Integrity gate: the regenerated labels/lens/qids must be array-equal to
the stored sidecars (proves identical cohort + ordering); otherwise the
script aborts without writing.

Run:  .venv/bin/python -m experiments.explorations.conversion_depth.cache_em_cohort3
Idempotent (skips if meta3.json exists and shards present).
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from experiments.explorations.conversion_depth.phase4_em_depth import (
    ADAPTER,
    BASE_MODEL,
    EM_DIR,
    MAX_ANSWER_LEN,
    MAX_SEQ_LEN,
    ALIGN_MISALIGNED_THRESHOLD,
    build_cohort,
)

HS3 = [10, 14, 16]   # resid_post L9 / L13 / L15


@torch.no_grad()
def main():
    marker = EM_DIR / "meta3.json"
    if marker.exists() and all((EM_DIR / f"hs{k}.npy").exists() for k in HS3):
        print("[cohort3] cache hit")
        return
    rows = build_cohort()
    lens0 = np.load(EM_DIR / "lens.npy")
    labels0 = np.load(EM_DIR / "labels.npy")
    qids0 = np.load(EM_DIR / "qids.npy")
    assert len(rows) == len(lens0), (len(rows), len(lens0))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    model = PeftModel.from_pretrained(model, ADAPTER)
    model = model.merge_and_unload().eval()
    d_model = int(model.config.hidden_size)

    n = len(rows)
    mms = {k: np.lib.format.open_memmap(
        EM_DIR / f"hs{k}.npy.tmp", mode="w+", dtype=np.float16,
        shape=(n, MAX_ANSWER_LEN, d_model)) for k in HS3}
    lens = np.zeros(n, dtype=np.int32)
    labels = np.zeros(n, dtype=np.int64)
    qids = np.zeros(n, dtype=np.int64)
    questions = sorted(set(r["question"] for r in rows))

    t0 = time.time()
    for ri, r in enumerate(rows):
        msgs = [{"role": "user", "content": r.get("question", "")},
                {"role": "assistant", "content": r.get("answer", "")}]
        full_text = tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=False)
        prefix_text = tok.apply_chat_template([msgs[0]], tokenize=False,
                                              add_generation_prompt=True)
        full_ids = tok(full_text, return_tensors="pt", truncation=True,
                       max_length=MAX_SEQ_LEN,
                       add_special_tokens=False)["input_ids"]
        prefix_len = len(tok(prefix_text,
                             add_special_tokens=False)["input_ids"])
        out = model(full_ids.cuda(), output_hidden_states=True,
                    use_cache=False)
        seq_len = int(full_ids.shape[1])
        prefix_len = min(prefix_len, seq_len)
        a, b = prefix_len, min(prefix_len + MAX_ANSWER_LEN, seq_len)
        L = b - a
        lens[ri] = L
        labels[ri] = int(r["align"] <= ALIGN_MISALIGNED_THRESHOLD)
        qids[ri] = questions.index(r["question"])
        if L > 0:
            for k in HS3:
                mms[k][ri, :L] = (out.hidden_states[k][0, a:b]
                                  .to(torch.float16).cpu().numpy())
        if (ri + 1) % 200 == 0:
            el = time.time() - t0
            print(f"  {ri + 1}/{n} ({el:.0f}s, est {el / (ri + 1) * n:.0f}s)",
                  flush=True)

    # Integrity gate BEFORE promoting shards.
    for name, new, old in [("lens", lens, lens0), ("labels", labels, labels0),
                           ("qids", qids, qids0)]:
        if not np.array_equal(new, old):
            raise RuntimeError(
                f"cohort3 sidecar mismatch on {name} — refusing to promote "
                "shards; the cohort/order no longer reproduces phase 4.")
    for k, m in mms.items():
        m.flush()
        del m
        (EM_DIR / f"hs{k}.npy.tmp").rename(EM_DIR / f"hs{k}.npy")
    marker.write_text(json.dumps({
        "hs": HS3, "n_rollouts": n, "d_model": d_model,
        "sidecar_check": "lens/labels/qids array-equal to phase-4",
        "wall_seconds": round(time.time() - t0, 1)}, indent=2))
    print(f"[cohort3] DONE in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
