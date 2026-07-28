"""STRUQPOS screen — readout-residual cache (executes STRUQPOS_SCREEN_CARD §3).

For every doc in the ratified x5 anagram corpus (rebuilt from
`labels.build_struqpos_premeasure.build_pairs`), forward-pass the subject
model and capture, at the byte-identical `### response:\n` readout token
(= last token), the screen-layer residual for TWO conditions:

  ordered   — the prompt as written
  fieldshuf — the untrusted FIELD (all tokens strictly between the fixed
              `### input:\n` prefix and the `\n\n### response:\n` suffix,
              crossing the connector; PIN 1) permuted by a per-doc seed.

Plus two identity-derived, context-free features from the model INPUT
embeddings (no attention, no residual):

  bag      — mean input-embedding over the field tokens (the `tok` floor)
  local4   — concatenated input-embedding of the LAST 4 field tokens
             before the suffix (the proximity floor, PIN 2)

Writes /workspace/struqpos_caches/struqpos_acts_<leg>.npz (idempotent).
GPU: 3 models × 2040 docs × 2 conditions; short prompts ⇒ ~1–1.5 L40S-h.

Run: .venv/bin/python -m experiments.explorations.task_hunt.struqpos.cache_acts [leg ...]
"""
from __future__ import annotations
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from experiments.explorations.task_hunt.labels.build_struqpos_premeasure import (
    ATTACKS_X5, PROMPT_INPUT, build_pairs, usable_items, _payload, SEED,
)
from experiments.explorations.task_hunt.replag.build_labels import MODELS
from experiments.explorations.task_hunt.replag.cache_acts import (
    HS_CAPTURE, SCREEN_HS,
)

CACHE_ROOT = Path("/workspace/struqpos_caches")
LEGS = ("gpt2", "gemma2_2b", "llama31_8b")
BATCH = {"gpt2": 128, "gemma2_2b": 48, "llama31_8b": 24}
LOCAL_K = 4                      # PIN 2 proximity span
PREFIX_END = "\n\n### response:\n"   # the fixed suffix; readout = its last tok


def _prefix_text(item):
    # everything up to and including "### input:\n" — identical in A and B
    from experiments.explorations.task_hunt.labels.build_struqpos_premeasure import (
        SYS_INPUT, _D,
    )
    return SYS_INPUT + _D[0] + "\n" + item["instruction"] + "\n\n" + _D[1] + "\n"


def _field_span(tok, full_ids, prefix_text):
    """[start, end) token span of the untrusted field: common-token-prefix
    with tok(prefix_text) gives start; common-token-suffix with
    tok(suffix) gives end. ≤1-token boundary slop (disclosed) when BPE
    merges across a boundary — immaterial to shuffle/bag/local4."""
    pre = tok(prefix_text, add_special_tokens=False)["input_ids"]
    suf = tok(PREFIX_END, add_special_tokens=False)["input_ids"]
    # start = length of shared leading run
    start = 0
    for a, b in zip(full_ids, pre):
        if a != b:
            break
        start += 1
    # end = total - shared trailing run
    end = len(full_ids)
    for a, b in zip(reversed(full_ids), reversed(suf)):
        if a != b:
            break
        end -= 1
    return start, end


@torch.no_grad()
def main(leg: str):
    CACHE_ROOT.mkdir(exist_ok=True)
    out = CACHE_ROOT / f"struqpos_acts_{leg}.npz"
    if out.exists():
        print(f"[cache_acts] hit: {out}")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODELS[leg]
    tok = AutoTokenizer.from_pretrained(cfg["hf"])
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or tok.bos_token
    bos = [tok.bos_token_id] if cfg["bos"] else []
    hs = SCREEN_HS[leg]

    items, pairs = build_pairs(ATTACKS_X5)

    model_id = cfg["hf"]
    print(f"[cache_acts:{leg}] loading {model_id}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    d = int(model.config.hidden_size)
    emb = model.get_input_embeddings()

    # ERRATUM (fix-forward on the freeze, disclosed): a few long
    # completion_real/_realcmb docs exceed a model's context (gpt2 n_positions
    # =1024) ⇒ out-of-range position ⇒ CUDA device-side assert. Skip any PAIR
    # whose A or B (with BOS) exceeds max_ctx for THIS leg — A/B skip together
    # (len_delta ≤2) so class balance is preserved; count disclosed in meta.
    max_ctx = int(getattr(model.config, "max_position_embeddings",
                          getattr(model.config, "n_positions", 100000)))
    def _len(txt):
        return len(bos) + len(tok(txt, add_special_tokens=False)["input_ids"])
    docs, over = [], 0
    for pi, pr in enumerate(pairs):
        if max(_len(pr["A"]), _len(pr["B"])) > max_ctx:
            over += 1
            continue
        docs.append((pi, "A", 1, pr["A"]))
        docs.append((pi, "B", 0, pr["B"]))
    print(f"[cache_acts:{leg}] {len(pairs)} pairs; {over} over max_ctx={max_ctx} "
          f"skipped ⇒ {len(docs)} docs; screen hs{hs}", flush=True)

    N = len(docs)
    res_ord = np.zeros((N, d), np.float16)
    res_shuf = np.zeros((N, d), np.float16)
    bag = np.zeros((N, d), np.float16)
    local4 = np.zeros((N, LOCAL_K * d), np.float16)
    y = np.zeros(N, np.int8)
    item_of = np.zeros(N, np.int32)
    attack_of = np.zeros(N, np.int8)
    split_of = np.zeros(N, np.int8)
    slop = 0

    t0 = time.time()
    B = BATCH[leg]
    for s in range(0, N, B):
        batch = docs[s:s + B]
        seqs_ord, seqs_shuf, meta = [], [], []
        for (pi, arm, lab, txt) in batch:
            pr = pairs[pi]
            ids = bos + tok(txt, add_special_tokens=False)["input_ids"]
            assert tok.decode(ids[len(bos):]) == txt, "roundtrip fail"
            fs, fe = _field_span(tok, ids, _prefix_text(items[pr["item"]]))
            if fe - fs < LOCAL_K + 1:            # degenerate; whole-body fallback
                fs, fe = len(bos), len(ids) - 1
                slop += 1
            # boundary slop bookkeeping
            rng = np.random.default_rng(
                [SEED, pr["item"], pr["attack_idx"], zlib.crc32(leg.encode())])
            perm = np.array(ids, dtype=np.int64)
            field = perm[fs:fe].copy()
            rng.shuffle(field)
            perm[fs:fe] = field
            seqs_ord.append(ids)
            seqs_shuf.append(perm.tolist())
            meta.append((pi, lab, fs, fe, pr))

        for cond, seqs, dest in (("ord", seqs_ord, res_ord),
                                 ("shuf", seqs_shuf, res_shuf)):
            maxlen = max(len(z) for z in seqs)
            arr = np.full((len(seqs), maxlen), tok.pad_token_id, np.int64)
            am = np.zeros((len(seqs), maxlen), np.int64)
            for i, z in enumerate(seqs):
                arr[i, :len(z)] = z
                am[i, :len(z)] = 1
            ids_t = torch.from_numpy(arr).cuda()
            out_m = model(ids_t, attention_mask=torch.from_numpy(am).cuda(),
                          output_hidden_states=True, use_cache=False)
            h = out_m.hidden_states[hs]
            for i, z in enumerate(seqs):
                dest[s + i] = h[i, len(z) - 1, :].to(torch.float16).cpu().numpy()

        # identity features from ordered ids (input embeddings)
        for i, (pi, lab, fs, fe, pr) in enumerate(meta):
            ids = seqs_ord[i]
            fld = torch.tensor(ids[fs:fe], device="cuda")
            fe_emb = emb(fld).to(torch.float16)
            bag[s + i] = fe_emb.mean(0).cpu().numpy()
            last = ids[max(fs, fe - LOCAL_K):fe]
            last = [tok.pad_token_id] * (LOCAL_K - len(last)) + list(last)
            local4[s + i] = emb(torch.tensor(last, device="cuda")).to(
                torch.float16).reshape(-1).cpu().numpy()
            y[s + i] = lab
            item_of[s + i] = pr["item"]
            attack_of[s + i] = pr["attack_idx"]
            split_of[s + i] = pr["split"]
        if (s // B) % 10 == 0:
            el = time.time() - t0
            print(f"  {s+len(batch)}/{N} ({el:.0f}s, est "
                  f"{el/max(s+len(batch),1)*N:.0f}s)", flush=True)

    np.savez(out, res_ord=res_ord, res_shuf=res_shuf, bag=bag, local4=local4,
             y=y, item=item_of, attack=attack_of, split=split_of,
             attacks=np.array(ATTACKS_X5), hs=hs, d_model=d, local_k=LOCAL_K)
    # finite receipt
    assert np.isfinite(res_ord).all() and np.linalg.norm(res_ord[3]) > 0
    (CACHE_ROOT / f"acts_meta_{leg}.json").write_text(json.dumps({
        "leg": leg, "model_id": model_id, "screen_hs": hs, "n_docs": N,
        "d_model": d, "local_k": LOCAL_K, "boundary_slop_docs": int(slop),
        "max_ctx": int(max_ctx), "pairs_over_ctx_skipped": int(over),
        "wall_seconds": round(time.time() - t0, 1)}, indent=2))
    print(f"[cache_acts:{leg}] DONE {N} docs in {time.time()-t0:.0f}s -> {out}",
          flush=True)
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    for leg in (sys.argv[1:] or list(LEGS)):
        main(leg)
