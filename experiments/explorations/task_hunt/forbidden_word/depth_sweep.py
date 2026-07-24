"""Candidate 3 — POST-HOC depth sweep: is the pressure linearized per-position?

**STATUS: POST-HOC DIAGNOSTIC. This does NOT re-open the frozen verdict.**
`card.md` froze a single mid-depth layer (resid_post L12) and the kill
rule; the screen ran there and returned KILL (pre-registered ambience
kill), and that verdict stands as recorded in `../LOG.md`. This script
is run AFTER that verdict, with the outcome already known, so under the
program's no-shopping-after-preregistration rule it cannot be used to
rescue the candidate. It answers a different, mechanistic question:

    the L12 screen showed per-token ≈ window. WHY?

Three distinguishable signatures on the g(ℓ) = AUC(window) − AUC(token)
curve across depth (the `conversion_depth` § 2 machinery pointed at this
label):

  1. per-token already high AT hs0 (embeddings)  → LEXICAL ambience:
     token identity alone carries it (bag-of-words / semantic
     neighbourhood). Needs no attention.
  2. per-token low at hs0, climbs with depth, g CLOSES → CONVERSION:
     the property genuinely needs cross-token information, but attention
     computed it and deposited it into the current position, so a
     per-token probe reads it. (The transformer is already a temporal
     integrator.)
  3. g STAYS OPEN at all depths → unconverted residue — the only thing a
     window architecture can actually eat. This is what the paper's
     backtracking anticipation label does (flat +0.03…+0.06 plateau at
     every residual layer, `conversion_depth/RECORD.md` § 3).

Backtracking shows 1→2→3 on one curve (hs0 gap +0.174, purely lexical;
one attention block converts most of it, per-token 0.64 → 0.80; a
+0.03–0.06 residue survives everywhere after). If forbidden-word shows
1 or 2 with no residue, the KILL is mechanistically explained and the
result generalises beyond the single frozen layer.

If instead g stays open at some OTHER depth, that is a NEW finding
requiring a fresh card and a fresh screen — it is explicitly NOT a
retroactive KEEP for this candidate, and must be reported that way.

Protocol (fixed, matching the frozen screen so the L12 slice is directly
comparable): identical rollouts, identical row recipe (same seeds →
identical rows), same `problib` stack, window = right-edge T = 16
flatten, plus the order-free window-MEAN so g splits into
g_agg (mean − token) + g_order (flatten − mean). Capture points:
hs0 (embeddings) + resid_post 0, 2, …, 30 — the `cache_depth.py` set.

Run:  .venv/bin/python -m \
        experiments.explorations.task_hunt.forbidden_word.depth_sweep
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.explorations.conversion_depth.problib import fit_probe
from experiments.explorations.task_hunt.forbidden_word.cache_and_screen import (
    CACHE_DIR, MODEL, ROLLOUTS, SEQ_LEN, build_rows, HORIZONS, NULL_SEED,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "forbidden_word_depth.json"
DEPTH_DIR = Path("/workspace/task_hunt_labels/forbidden_word/acts_depth")

LAYERS = list(range(0, 31, 2))                 # resid_post 0,2,…,30
HS_CAPTURE = [0] + [k + 1 for k in LAYERS]     # hidden_states indices
T = 16                                         # frozen window
BATCH = 4


@torch.no_grad()
def phase_a():
    """One forward sweep; write every capture point (fp16 memmaps)."""
    DEPTH_DIR.mkdir(parents=True, exist_ok=True)
    meta_p = DEPTH_DIR / "meta.json"
    if meta_p.exists():
        print(f"[depth-cache] hit: {DEPTH_DIR}")
        return

    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
    # fast backend forced — the recorded distill tokenizer trap
    tok = PreTrainedTokenizerFast.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rolls = [json.loads(l) for l in ROLLOUTS.open()]
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    d_model = int(model.config.hidden_size)
    n = len(rolls)

    mms = {k: np.lib.format.open_memmap(
        DEPTH_DIR / f"hs{k}.npy", mode="w+", dtype=np.float16,
        shape=(n, SEQ_LEN, d_model)) for k in HS_CAPTURE}
    print(f"[depth-cache] {len(mms)} capture points × {n}×{SEQ_LEN}×{d_model} "
          f"fp16 ({sum(m.nbytes for m in mms.values())/1e9:.0f} GB)", flush=True)

    t0 = time.time()
    for s in range(0, n, BATCH):
        batch = rolls[s:s + BATCH]
        seqs = []
        for r in batch:
            p_ids = tok.apply_chat_template(
                [{"role": "user", "content": r["user_prompt"]}],
                tokenize=True, add_generation_prompt=True, return_dict=False)
            end = r["think_char_end"]
            think = r["text"][:end] if end >= 0 else r["text"]
            c_ids = tok(think, add_special_tokens=False)["input_ids"][:SEQ_LEN]
            seqs.append((p_ids, c_ids))
        maxlen = max(len(p) + len(c) for p, c in seqs)
        ids = torch.full((len(seqs), maxlen), tok.eos_token_id or 0,
                         dtype=torch.long)
        for i, (p, c) in enumerate(seqs):
            ids[i, :len(p) + len(c)] = torch.tensor(p + c)
        out = model(ids.cuda(), output_hidden_states=True, use_cache=False)
        for i, (p, c) in enumerate(seqs):
            L = min(len(c), SEQ_LEN)
            for k in HS_CAPTURE:
                mms[k][s + i, :L] = (out.hidden_states[k][i, len(p):len(p) + L]
                                     .to(torch.float16).cpu().numpy())
        if (s // BATCH) % 25 == 0:
            el = time.time() - t0
            print(f"  {s + len(batch)}/{n} ({el:.0f}s, "
                  f"{el / max(s + len(batch), 1) * n:.0f}s est)", flush=True)
    for m in mms.values():
        m.flush()
    meta_p.write_text(json.dumps({
        "model": MODEL, "hs_capture": HS_CAPTURE, "resid_post_layers": LAYERS,
        "n": n, "seq_len": SEQ_LEN, "d_model": d_model,
        "wall_seconds": round(time.time() - t0, 1)}, indent=2))
    print(f"[depth-cache] DONE in {time.time()-t0:.0f}s", flush=True)


def gather(acts, rows, width):
    idx, p = rows[:, 0], rows[:, 1]
    X = np.empty((len(rows), width, acts.shape[-1]), dtype=np.float16)
    for j in range(width):
        X[:, j] = acts[idx, p - (width - 1) + j]
    return torch.from_numpy(X)


def phase_b():
    done = json.loads(OUT.read_text()) if OUT.exists() else {"cells": {}}
    done["meta"] = {"status": "POST-HOC diagnostic; does NOT reopen the "
                             "frozen L12 KILL (see module docstring)",
                    "T": T, "hs_capture": HS_CAPTURE, "horizons": HORIZONS}
    rows_by_D = {D: build_rows(D) for D in HORIZONS}
    for k in HS_CAPTURE:
        acts = np.load(DEPTH_DIR / f"hs{k}.npy", mmap_mode="r")
        for D in HORIZONS:
            key = f"hs{k}/D{D}"
            if key in done["cells"]:
                continue
            t0 = time.time()
            (rtr, ytr), (rte, yte) = rows_by_D[D]["train"], rows_by_D[D]["test"]
            ytr_t, yte_t = torch.from_numpy(ytr), torch.from_numpy(yte)
            Wtr, Wte = gather(acts, rtr, T), gather(acts, rte, T)
            n_tr, n_te = len(rtr), len(rte)
            cell = {}
            cell["tok"] = fit_probe(Wtr[:, -1], ytr_t, Wte[:, -1], yte_t, 2,
                                    class_weight=True)
            cell["flat"] = fit_probe(Wtr.reshape(n_tr, -1), ytr_t,
                                     Wte.reshape(n_te, -1), yte_t, 2,
                                     class_weight=True)
            cell["mean"] = fit_probe(Wtr.float().mean(1).half(), ytr_t,
                                     Wte.float().mean(1).half(), yte_t, 2,
                                     class_weight=True)
            g = torch.Generator().manual_seed(NULL_SEED)
            cell["null"] = fit_probe(
                Wtr[:, -1], ytr_t[torch.randperm(n_tr, generator=g)],
                Wte[:, -1], yte_t[torch.randperm(n_te, generator=g)], 2,
                class_weight=True)
            cell["g"] = cell["flat"]["auc"] - cell["tok"]["auc"]
            cell["g_ceil"] = (max(cell["flat"]["auc"], cell["mean"]["auc"])
                              - cell["tok"]["auc"])
            cell["g_agg"] = cell["mean"]["auc"] - cell["tok"]["auc"]
            cell["g_order"] = cell["flat"]["auc"] - cell["mean"]["auc"]
            done["cells"][key] = cell
            print(f"[{key}] tok={cell['tok']['auc']:.3f} "
                  f"flat={cell['flat']['auc']:.3f} "
                  f"mean={cell['mean']['auc']:.3f} "
                  f"g_ceil={cell['g_ceil']:+.3f} "
                  f"g_ord={cell['g_order']:+.3f} "
                  f"null={cell['null']['auc']:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            OUT.write_text(json.dumps(done, indent=1))
            del Wtr, Wte
        del acts
    print(f"-> {OUT}")


def main():
    phase_a()
    phase_b()


if __name__ == "__main__":
    main()
