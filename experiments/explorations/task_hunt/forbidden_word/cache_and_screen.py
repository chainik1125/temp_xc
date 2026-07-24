"""Task-hunt candidate 3 — cache rollout activations + run the frozen screen.

Protocol: `card.md` (frozen before any rollout was generated). Two phases
in one script so the 4096-dim cache is written once and consumed:

**Phase A (cache).** Re-tokenize each rollout's PROMPT+COMPLETION with the
R1-Distill tokenizer, forward through `deepseek-ai/DeepSeek-R1-Distill-
Llama-8B` (the generator is also the reader here), capture resid_post
L12 (hidden_states[13]) fp16 for the completion region only, in
`SEQ_LEN`-token chunks. Positions are recorded in COMPLETION-token
coordinates so the label arithmetic is exact.

**Phase B (screen).** For each frozen horizon D ∈ {4, 8, 16}: positives
= completion tokens whose distance to the first keyword token is in
[D, 2D); negatives = tokens > 64 before the first occurrence, plus
tokens from non-violating rollouts; everything at/after the first
occurrence is excluded. Rows split BY ROLLOUT 80/20 (rng 7), capped per
rollout (rng 13/14), 5:1 neg:pos. Probes: per-token linear,
window-flatten, window-MEAN, within-window-SHUFFLED (seed 23) on the
frozen `problib` stack, T ∈ {2,4,8,16,32}; MLP-512 presence at T = 16;
permutation null seed 99.

Run (main venv, after generate.py):
  .venv/bin/python -m \
    experiments.explorations.task_hunt.forbidden_word.cache_and_screen
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.explorations.conversion_depth.problib import fit_probe

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
ROLLOUTS = Path("/workspace/task_hunt_labels/forbidden_word/rollouts.jsonl")
CACHE_DIR = Path("/workspace/task_hunt_labels/forbidden_word/acts")
HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "forbidden_word_screen.json"

HS = 13                       # resid_post L12
SEQ_LEN = 1024                # per-rollout completion cap (tokens)
BATCH = 4
TS = [2, 4, 8, 16, 32]
HORIZONS = [4, 8, 16]
NEG_BUFFER = 64
ROWS_PER_ROLLOUT = 40
SPLIT_SEED = 7
CAP_SEED_TR, CAP_SEED_TE = 13, 14
SHUFFLE_SEED = 23
NULL_SEED = 99
NEG_PER_POS = 5
MLP_T = 16
P_MIN = 32


def load_rollouts():
    return [json.loads(l) for l in ROLLOUTS.open()]


@torch.no_grad()
def phase_a():
    """Cache resid_post L12 over each rollout's completion region."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta_p = CACHE_DIR / "meta.json"
    if meta_p.exists():
        print(f"[cache] hit: {CACHE_DIR}")
        return json.loads(meta_p.read_text())

    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
    # FORCE the fast backend — the repo's recorded trap
    # (`conversion_depth/build_ward_stream.py`): AutoTokenizer resolves this
    # repo to the SLOW LlamaTokenizer, whose `return_offsets_mapping` yields
    # unusable spans, so the keyword-onset index silently comes back -1 for
    # every rollout (i.e. "no violations found") instead of erroring.
    tok = PreTrainedTokenizerFast.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rolls = load_rollouts()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    d_model = int(model.config.hidden_size)

    n = len(rolls)
    acts = np.lib.format.open_memmap(
        CACHE_DIR / f"hs{HS}.npy", mode="w+", dtype=np.float16,
        shape=(n, SEQ_LEN, d_model))
    lens = np.zeros(n, dtype=np.int32)
    kw_tok = np.full(n, -1, dtype=np.int32)      # first-keyword token index
    viol = np.zeros(n, dtype=bool)

    t0 = time.time()
    for s in range(0, n, BATCH):
        batch = rolls[s:s + BATCH]
        seqs, metas = [], []
        for r in batch:
            # NOTE: under transformers 5.x `apply_chat_template(tokenize=True)`
            # returns a BatchEncoding, not a list of ids — `return_dict=False`
            # is required or `len(p_ids)` silently becomes 2 (the key count).
            p_ids = tok.apply_chat_template(
                [{"role": "user", "content": r["user_prompt"]}],
                tokenize=True, add_generation_prompt=True, return_dict=False)
            # completion tokens, truncated to the think region when closed
            text = r["text"]
            end = r["think_char_end"]
            think_text = text[:end] if end >= 0 else text
            enc = tok(think_text, add_special_tokens=False,
                      return_offsets_mapping=True)
            c_ids = enc["input_ids"][:SEQ_LEN]
            # First-keyword token index in COMPLETION coordinates, located by
            # OFFSET MAPPING (the token whose char span contains the keyword's
            # start) — the build_ward_stream convention. Retokenizing the
            # prefix instead is off by a subword when the tokenizer merges the
            # keyword's first character into the preceding token.
            k = -1
            if r["first_kw_char"] >= 0:
                fc = r["first_kw_char"]
                for j, (a, b) in enumerate(enc["offset_mapping"][:len(c_ids)]):
                    if a <= fc < b:
                        k = j
                        break
                if k < 0 or k >= len(c_ids):
                    k = -1
            seqs.append((p_ids, c_ids))
            metas.append(k)
        maxlen = max(len(p) + len(c) for p, c in seqs)
        ids = torch.full((len(seqs), maxlen), tok.eos_token_id or 0,
                         dtype=torch.long)
        for i, (p, c) in enumerate(seqs):
            ids[i, :len(p) + len(c)] = torch.tensor(p + c)
        out = model(ids.cuda(), output_hidden_states=True, use_cache=False)
        h = out.hidden_states[HS]
        for i, (p, c) in enumerate(seqs):
            L = min(len(c), SEQ_LEN)
            acts[s + i, :L] = (h[i, len(p):len(p) + L]
                               .to(torch.float16).cpu().numpy())
            lens[s + i] = L
            kw_tok[s + i] = metas[i]
            viol[s + i] = metas[i] >= 0
        if (s // BATCH) % 20 == 0:
            el = time.time() - t0
            print(f"  {s + len(batch)}/{n} ({el:.0f}s)", flush=True)
    acts.flush()
    np.save(CACHE_DIR / "lens.npy", lens)
    np.save(CACHE_DIR / "kw_tok.npy", kw_tok)
    np.save(CACHE_DIR / "violated.npy", viol)
    meta = {"model": MODEL, "hs": HS, "n": n, "seq_len": SEQ_LEN,
            "d_model": d_model,
            "n_violating_with_token_index": int(viol.sum()),
            "wall_seconds": round(time.time() - t0, 1)}
    meta_p.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return meta


def build_rows(D):
    lens = np.load(CACHE_DIR / "lens.npy")
    kw = np.load(CACHE_DIR / "kw_tok.npy")
    n = len(lens)
    rng_split = np.random.default_rng(SPLIT_SEED)
    is_test = np.zeros(n, dtype=bool)
    is_test[rng_split.permutation(n)[:n // 5]] = True

    splits = {}
    for name, want_test, seed in [("train", False, CAP_SEED_TR),
                                  ("test", True, CAP_SEED_TE)]:
        rng = np.random.default_rng(seed)
        pos, neg = [], []
        for i in range(n):
            if is_test[i] != want_test:
                continue
            L = int(lens[i])
            k = int(kw[i])
            if k >= 0:
                lo, hi = k - 2 * D, k - D          # distance in [D, 2D)
                cand = [p for p in range(max(P_MIN, lo), min(hi, L))
                        if p < k]
                pos.extend((i, p) for p in cand)
                nc = [p for p in range(P_MIN, min(k - NEG_BUFFER, L))]
            else:
                nc = list(range(P_MIN, L))
            if nc:
                take = rng.permutation(len(nc))[:ROWS_PER_ROLLOUT]
                neg.extend((i, nc[j]) for j in take)
        rng2 = np.random.default_rng(17)
        neg = [neg[j] for j in
               rng2.permutation(len(neg))[:NEG_PER_POS * max(1, len(pos))]]
        rows = np.array(pos + neg, dtype=np.int64)
        y = np.concatenate([np.ones(len(pos), dtype=np.int64),
                            np.zeros(len(neg), dtype=np.int64)])
        splits[name] = (rows, y)
        print(f"[rows] D={D}/{name}: {len(pos)} pos, {len(neg)} neg",
              flush=True)
    return splits


def gather(acts, rows, T):
    idx = rows[:, 0]
    p = rows[:, 1]
    X = np.empty((len(rows), T, acts.shape[-1]), dtype=np.float16)
    for j in range(T):
        X[:, j] = acts[idx, p - (T - 1) + j]
    return torch.from_numpy(X)


def phase_b():
    done = json.loads(OUT.read_text()) if OUT.exists() else {"cells": {}}
    done["meta"] = {"protocol": "card.md (frozen)", "Ts": TS,
                    "horizons": HORIZONS, "hs": HS,
                    "neg_buffer": NEG_BUFFER}
    acts = np.load(CACHE_DIR / f"hs{HS}.npy", mmap_mode="r")
    for D in HORIZONS:
        sp = build_rows(D)
        (rtr, ytr), (rte, yte) = sp["train"], sp["test"]
        if len(ytr) == 0 or ytr.sum() == 0 or yte.sum() == 0:
            done["cells"][f"D{D}"] = {"error": "no positives"}
            continue
        ytr_t, yte_t = torch.from_numpy(ytr), torch.from_numpy(yte)
        key0 = f"D{D}"
        if f"{key0}/tok" not in done["cells"]:
            Xtr = gather(acts, rtr, 1).reshape(len(rtr), -1)
            Xte = gather(acts, rte, 1).reshape(len(rte), -1)
            cell = {"linear": fit_probe(Xtr, ytr_t, Xte, yte_t, 2,
                                        class_weight=True)}
            g = torch.Generator().manual_seed(NULL_SEED)
            cell["null"] = fit_probe(
                Xtr, ytr_t[torch.randperm(len(ytr_t), generator=g)],
                Xte, yte_t[torch.randperm(len(yte_t), generator=g)], 2,
                class_weight=True)
            done["cells"][f"{key0}/tok"] = cell
            print(f"[{key0}/tok] auc={cell['linear']['auc']:.3f} "
                  f"null={cell['null']['auc']:.3f}", flush=True)
            OUT.write_text(json.dumps(done, indent=1))
        for T in TS:
            key = f"{key0}/T{T}"
            if key in done["cells"]:
                continue
            t0 = time.time()
            Wtr, Wte = gather(acts, rtr, T), gather(acts, rte, T)
            n_tr, n_te = len(rtr), len(rte)
            cell = {}
            cell["flat"] = fit_probe(Wtr.reshape(n_tr, -1), ytr_t,
                                     Wte.reshape(n_te, -1), yte_t, 2,
                                     class_weight=True)
            cell["mean"] = fit_probe(Wtr.float().mean(1).half(), ytr_t,
                                     Wte.float().mean(1).half(), yte_t, 2,
                                     class_weight=True)
            gs = torch.Generator().manual_seed(SHUFFLE_SEED)
            Str = torch.stack([x[torch.randperm(T, generator=gs)] for x in Wtr])
            Ste = torch.stack([x[torch.randperm(T, generator=gs)] for x in Wte])
            cell["shuf"] = fit_probe(Str.reshape(n_tr, -1), ytr_t,
                                     Ste.reshape(n_te, -1), yte_t, 2,
                                     class_weight=True)
            g = torch.Generator().manual_seed(NULL_SEED)
            cell["null_flat"] = fit_probe(
                Wtr.reshape(n_tr, -1),
                ytr_t[torch.randperm(n_tr, generator=g)],
                Wte.reshape(n_te, -1),
                yte_t[torch.randperm(n_te, generator=g)], 2,
                class_weight=True)
            if T == MLP_T:
                cell["mlp_flat"] = fit_probe(Wtr.reshape(n_tr, -1), ytr_t,
                                             Wte.reshape(n_te, -1), yte_t, 2,
                                             hidden=512, class_weight=True)
            tok_auc = done["cells"][f"{key0}/tok"]["linear"]["auc"]
            cell["g"] = cell["flat"]["auc"] - tok_auc
            cell["g_agg"] = cell["mean"]["auc"] - tok_auc
            cell["g_order"] = cell["flat"]["auc"] - cell["mean"]["auc"]
            cell["shuffle_gap"] = cell["flat"]["auc"] - cell["shuf"]["auc"]
            done["cells"][key] = cell
            print(f"[{key}] flat={cell['flat']['auc']:.3f} "
                  f"mean={cell['mean']['auc']:.3f} "
                  f"shuf={cell['shuf']['auc']:.3f} g={cell['g']:+.3f} "
                  f"g_ord={cell['g_order']:+.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            OUT.write_text(json.dumps(done, indent=1))
            del Wtr, Wte, Str, Ste
    print(f"-> {OUT}")


def main():
    phase_a()
    phase_b()


if __name__ == "__main__":
    main()
