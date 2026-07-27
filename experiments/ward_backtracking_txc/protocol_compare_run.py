"""v7 / pp / slab on the saved ward checkpoints, using the reference's own hooks.

The protocols come from `steering_vendored.py`: `v7` and `pp` are the paper's hooks copied
verbatim, `slab` is `v7` with one line changed (per-position delta instead of its window mean).
This driver only selects a latent, prompts the model and reads out behaviour.

WHAT THE FIRST ATTEMPT GOT WRONG, all three in this file, all fixed here:

  1. DEAD LATENT. Selection took `argmax|AUC - 0.5|` over every latent, and picked one that
     essentially never fires. Now a latent must fire on at least `--min-fire` of windows to be
     eligible, and the chosen latent's firing rate is reported.
  2. SILENT FALLBACK. `p99 = float(np.percentile(...)) or 1.0` turned a p99 of ZERO into 1.0,
     so every dose `q * p99` was meaningless and the injected norms were ~2% of the residual
     stream. Now a non-positive p99 raises. (Same class as the judge that returned None and
     produced a full run of zeros earlier in this sprint.)
  3. NO REASONING TO MEASURE. The subject is Llama-3.1-8B BASE and the c7 prompts are bare
     questions, so it continued them with more questions rather than reasoning, and the
     keyword readout was 0 at every dose for every protocol -- which reads exactly like a
     clean negative and is not one. The prompt is now the question plus the first
     `--prefix-sents` sentences of that item's own ward trace, which both puts the model in
     mid-reasoning and matches the distribution the checkpoint was trained on.

PREFLIGHT. Before any dose is swept, the unsteered keyword rate is measured and the run ABORTS
if it is zero. An instrument that reads zero on the positive control cannot produce a
meaningful negative, and this is the cheapest place to find that out.

    python protocol_compare_run.py --train-key d7b2e24253055f8e --n-prompts 40
"""
import argparse
import json
import pathlib
import re

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from steering_vendored import TXCAdapter, build_steering_hook

HF_MODELS_REPO = "han1823123123/temp-bench-models"
SUBJECT = "NousResearch/Meta-Llama-3.1-8B"
KEYWORDS = ("wait", "hmm", "actually", "hold on", "but wait", "let me reconsider",
            "on second thought", "that's wrong", "i made a mistake", "no,")
KW_RE = re.compile("|".join(re.escape(k) for k in KEYWORDS), re.I)


def kw_rate(t):
    return len(KW_RE.findall(t)) / max(len(t.split()), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-key", default="d7b2e24253055f8e")
    ap.add_argument("--n-prompts", type=int, default=40)
    ap.add_argument("--n-sel", type=int, default=60)
    ap.add_argument("--qs", default="0,1,2,4,8")
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--prefix-sents", type=int, default=6)
    ap.add_argument("--min-fire", type=float, default=0.02)
    ap.add_argument("--eval-dir", default="./_c7_eval")
    ap.add_argument("--out", default="protocol_compare.json")
    a = ap.parse_args()
    qs = [float(x) for x in a.qs.split(",")]
    dev = "cuda"

    cfg = json.load(open(hf_hub_download(HF_MODELS_REPO, f"{a.train_key}/config.json")))
    sd = load_file(hf_hub_download(HF_MODELS_REPO, f"{a.train_key}/model.safetensors"))
    W_dec, W_enc = sd["W_dec"].float().to(dev), sd["W_enc"].float().to(dev)
    b_enc = sd["b_enc"].float().to(dev)
    b_dec = sd["b_dec"].float().to(dev) if "b_dec" in sd else torch.zeros(
        W_dec.shape[1], W_dec.shape[2], device=dev)
    d_sae, T, d_in = W_dec.shape
    k_pos = 20
    arch = TXCAdapter(W_enc, b_enc, W_dec, b_dec, k_win=k_pos * T).to(dev)
    print(f"[ckpt] {a.train_key} arch={cfg.get('arch')} d_sae={d_sae} T={T} d_in={d_in} "
          f"k_win={k_pos*T}", flush=True)

    tok = AutoTokenizer.from_pretrained(SUBJECT)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        SUBJECT, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    layer = int(cfg.get("datasource", "").split("_l")[-1].split("_")[0] or 10)
    site = model.model.layers[layer]
    print(f"[subject] {SUBJECT} layer={layer}", flush=True)

    ed = pathlib.Path(a.eval_dir)
    labels = json.load(open(ed / "sentence_labels.json"))
    prompts = {p["id"]: p["prompt"] for p in json.load(open(ed / "prompts.json"))}

    # ---------- 1. select a latent that BOTH separates backtracking AND fires ----------
    acts, pos_z, neg_z = [], [], []
    h = site.register_forward_hook(
        lambda m, i, o: acts.append((o[0] if isinstance(o, tuple) else o).detach()))
    for rec in labels[:a.n_sel]:
        sents = rec["sentences"]
        if not sents:
            continue
        text, spans, cur = "", [], 0
        for s in sents:
            text += s["sentence"] + " "
            n = len(s["sentence"]) + 1
            spans.append((cur, cur + n, bool(s["is_backtracking"])))
            cur += n
        enc = tok(text, return_tensors="pt", truncation=True, max_length=1024,
                  return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")[0].tolist()
        acts.clear()
        with torch.no_grad():
            model(**{k: v.to(dev) for k, v in enc.items()})
        H = acts[0][0].float()
        S = H.shape[0]
        if S < T:
            continue
        is_bt = np.zeros(S, dtype=bool)
        for i, (s0, s1) in enumerate(offs):
            for (cs, ce, f) in spans:
                if f and s0 < ce and s1 > cs:
                    is_bt[i] = True
                    break
        nb = S // T
        with torch.no_grad():
            z = arch.encode(H[: nb * T].reshape(nb, T, d_in)).squeeze(1)
        lab = is_bt[: nb * T].reshape(nb, T).any(axis=1)
        for w in range(nb):
            (pos_z if lab[w] else neg_z).append(z[w])
    h.remove()
    if not pos_z or not neg_z:
        raise RuntimeError("no labelled windows on one side; raise --n-sel")
    P, N = torch.stack(pos_z).cpu().numpy(), torch.stack(neg_z).cpu().numpy()
    allz = np.concatenate([P, N], 0)
    y = np.concatenate([np.ones(len(P)), np.zeros(len(N))])
    print(f"[select] {len(P)} backtracking windows / {len(N)} not", flush=True)

    fire = (allz > 0).mean(0)                       # FIX 1: eligibility by firing rate
    eligible = np.flatnonzero(fire >= a.min_fire)
    if eligible.size == 0:
        raise RuntimeError(f"no latent fires on >= {a.min_fire:.1%} of windows")
    order = allz.argsort(0)
    ranks = np.empty_like(order, dtype=np.float64)
    np.put_along_axis(ranks, order, np.arange(1, len(allz) + 1)[:, None], axis=0)
    n1, n0 = y.sum(), len(y) - y.sum()
    auc = ((ranks * y[:, None]).sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    j = int(eligible[np.abs(auc[eligible] - 0.5).argmax()])
    sign = 1.0 if auc[j] > 0.5 else -1.0
    p99 = float(np.percentile(allz[:, j], 99))
    if not p99 > 0:                                 # FIX 2: no silent fallback
        raise RuntimeError(f"latent {j} has p99 = {p99}; dose scale undefined")
    Pn = (W_dec[j] / W_dec[j].norm())
    ac = float((Pn - Pn.mean(0)).norm() ** 2 / Pn.norm() ** 2)
    print(f"[select] latent {j}  AUC {auc[j]:.3f}  fires {fire[j]:.1%}  p99 {p99:.4f}  "
          f"sign {sign:+.0f}  AC-share {ac:.3f}   (eligible {eligible.size}/{d_sae})",
          flush=True)

    # ---------- 2. prompts that actually elicit reasoning ----------
    items = []
    for rec in labels:
        q = prompts.get(rec["question_id"])
        sents = rec.get("sentences") or []
        if not q or len(sents) < a.prefix_sents + 2:
            continue
        prefix = " ".join(s["sentence"] for s in sents[:a.prefix_sents])
        items.append({"id": rec["question_id"], "text": f"{q}\n\n{prefix}"})
        if len(items) >= a.n_prompts:
            break
    print(f"[prompts] {len(items)} question + {a.prefix_sents}-sentence trace prefixes",
          flush=True)

    state = {"feature_idx": None, "norms": [], "cs": []}

    def generate():
        outs = []
        for it in items:
            enc = tok(it["text"], return_tensors="pt", truncation=True, max_length=768).to(dev)
            with torch.no_grad():
                o = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
            outs.append(tok.decode(o[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))
        return outs

    # ---------- 3. PREFLIGHT: is the instrument alive at all? ----------
    state["feature_idx"] = None
    base_txt = generate()
    base_kw = float(np.mean([kw_rate(t) for t in base_txt]))
    print(f"[preflight] unsteered keyword rate {base_kw:.5f}  "
          f"mean words {np.mean([len(t.split()) for t in base_txt]):.0f}", flush=True)
    print(f"[preflight] sample: {base_txt[0][:220]!r}", flush=True)
    if base_kw <= 0:
        raise RuntimeError(
            "unsteered keyword rate is 0 -- the readout cannot register backtracking on these "
            "generations, so no dose sweep can produce a meaningful negative. Fix the prompt "
            "or the readout before sweeping.")

    # ---------- 4. the sweep ----------
    rows = [{"protocol": "none", "q": 0.0, "keyword_rate_mean": base_kw,
             "keyword_rate_sem": float(np.std([kw_rate(t) for t in base_txt], ddof=1)
                                       / max(len(base_txt) ** .5, 1)),
             "injected_norm_mean": 0.0, "sample": base_txt[0][:400]}]
    # The vendored hooks capture `strengths_t` in their closure, so a hook is built per dose
    # rather than mutating shared state -- one less thing that can go stale between arms.
    for proto in ("v7", "pp", "slab"):
        for q in qs:
            if q == 0:
                continue
            s_abs = sign * q * p99
            st = torch.full((1,), s_abs, device=dev)
            hk = build_steering_hook(arch, protocol=proto, T=T, strengths_t=st, state=state)
            state.update(feature_idx=j, norms=[], cs=[])
            handle = site.register_forward_hook(hk)
            txt = generate()
            handle.remove()
            ks = [kw_rate(t) for t in txt]
            rows.append({"protocol": proto, "q": float(q), "s_abs": s_abs,
                         "keyword_rate_mean": float(np.mean(ks)),
                         "keyword_rate_sem": float(np.std(ks, ddof=1) / max(len(ks) ** .5, 1)),
                         "injected_norm_mean": float(np.mean(state["norms"] or [0.0])),
                         "sample": txt[0][:400]})
            print(f"[{proto:>4} q={q:<4}] kw {rows[-1]['keyword_rate_mean']:.5f} "
                  f"(base {base_kw:.5f})  |Δ|_F {rows[-1]['injected_norm_mean']:.2f}",
                  flush=True)

    out = {"train_key": a.train_key, "arch": cfg.get("arch"),
           "datasource": cfg.get("datasource"), "subject": SUBJECT, "layer": layer,
           "T": T, "d_sae": d_sae, "latent": j, "latent_auc": float(auc[j]),
           "latent_fire_rate": float(fire[j]), "sign": sign, "p99": p99, "ac_share": ac,
           "n_prompts": len(items), "prefix_sents": a.prefix_sents,
           "baseline_keyword_rate": base_kw, "rows": rows}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=2))
    print("[saved]", a.out, flush=True)


if __name__ == "__main__":
    main()
