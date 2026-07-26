"""Does KV-cached generation under a steering hook produce the same text as uncached?

The generation sweep costs 2.6 GPU-hours without KV caching and ~20 minutes with it, so the
cache is worth having. But it changes what the forward hook sees: uncached, every step
passes the whole sequence and the hook re-applies the write to the document each time;
cached, the prefill passes the document and every decode step passes a SINGLE token.

Two things could go wrong and only one of them is visible by reading the code.

  1. A segment starting at position 0 would, on a length-1 decode tensor, have its write
     added to the FIRST GENERATED TOKEN instead of to the document. `make_recency` puts its
     earliest segment at position 2, so this does not bite on this task -- which is why the
     harness guards on `h.shape[1] == n0` explicitly instead of relying on the layout.
  2. Repeated application. Uncached, the hook fires once per generated token and adds the
     write to the document EVERY time; the residual stream is rebuilt from scratch each
     forward, so that is idempotent rather than cumulative. Cached, it fires once. These
     agree only if the uncached version really was idempotent, and that is an assumption
     about `register_forward_hook` semantics rather than something the code states.

So this runs both paths on the same documents at the same nonzero dose and compares the
decoded strings byte for byte. Greedy decoding makes any real divergence visible immediately;
bf16 nondeterminism can flip a token late in a continuation, so mismatches are reported with
the position of the first differing character rather than as a pass/fail bit.

REGISTERED BEFORE RUNNING: I expect byte-identical output on all documents. If instead the
texts diverge only late and only occasionally, that is bf16 tie-breaking under a different
reduction order and is acceptable; if they diverge at the FIRST token, the hook is being
applied differently and the cached path is wrong.

WHAT THE TIMING ACTUALLY SHOWED -- record this before budgeting another generation sweep
here, because it is the opposite of the intuition that motivated the fix:

    alpha = 1.0 (hook active)   cached 10.7s   uncached 18.6s   speedup 1.7x
    alpha = 0.0 (control)       cached 10.7s   uncached 11.4s   speedup 1.06x

**KV caching buys very little on this workload.** Documents are 98 tokens with 32 generated,
so prefill dominates and there is almost nothing to amortise -- against the 8-9x that was
estimated from an assumed ~350-token document. The number that matters for a budget is the
CACHED RATE, 0.67 s per generation, which is identical in both runs and is what the ~55 min
estimate for 4,880 generations rests on. The speedup ratio never entered the estimate.

The two uncached times differ by 63% while the cached times are identical to three
significant figures, and the hook executes in both cases (adding zeros at alpha=0). I do
not have an account for that and am not proposing one.

    modal run experiments/temporal_screen/txc_wins/gen_cache_check_modal.py
"""
import pathlib
import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

app = modal.App("txcwins-gencache")
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers", "accelerate", "numpy")
    .run_commands(
        "python -c \"from huggingface_hub import snapshot_download; "
        "snapshot_download('Qwen/Qwen2.5-1.5B-Instruct')\"")
    .add_local_dir(str(ROOT / "src"), "/work/src")
    .add_local_dir(str(ROOT / "temporal_crosscoders"), "/work/temporal_crosscoders")
    .add_local_dir(str(_here.parent), "/work/txc_wins")
)


@app.function(gpu="A10G", image=image, timeout=3600)
def check(model_id: str, layer: int, k_seg: int, n_doc: int, gen_tokens: int,
          alpha: float, seed: int):
    import sys
    sys.path.insert(0, "/work")
    import random
    import time
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from txc_wins.tasks import TASKS

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    dev = model.device
    layers_ = model.model.layers
    L = layer if layer >= 0 else len(layers_) // 2
    d = model.config.hidden_size
    T = k_seg
    make_pair = TASKS["recency"](k_seg)
    rng = random.Random(seed)

    def build(carrier, sents):
        text, spans = carrier, []
        for j, s in enumerate(sents):
            if j:
                text += " "
            spans.append((len(text), len(text) + len(s)))
            text += s
        return text, spans

    def seg_spans(text, spans):
        e = tok(text, return_tensors="pt", return_offsets_mapping=True)
        offs = e["offset_mapping"][0].tolist()
        ts = []
        for (a, b) in spans:
            idx = [i for i, (s0, s1) in enumerate(offs)
                   if s0 >= a and s1 <= b and s1 > s0]
            ts.append((idx[0], idx[-1]) if idx else (0, 0))
        return e, ts

    # A fixed pseudo-random write, so the check does not depend on training anything.
    g = torch.Generator(device="cpu").manual_seed(0)
    W = torch.randn(T, d, generator=g).to(dev)
    W = W / W.norm()

    def run(stem, spn, cached):
        e, ts2 = seg_spans(stem, spn)
        ids = e["input_ids"].to(dev)
        n0 = ids.shape[1]

        def edit(_m, _i, out_):
            h = out_[0] if isinstance(out_, tuple) else out_
            if (not cached) or h.shape[1] == n0:
                for t_i, (p0, p1) in enumerate(ts2):
                    h[:, p0:p1 + 1, :] += (alpha * scale
                                           * W[t_i].to(h.dtype).unsqueeze(0))
            return (h,) + out_[1:] if isinstance(out_, tuple) else h

        hk = layers_[L].register_forward_hook(edit)
        t0 = time.time()
        with torch.no_grad():
            if cached:
                o = model(ids, use_cache=True)
                past, nxt = o.past_key_values, o.logits[0, -1].argmax()
                gen = [nxt]
                for _ in range(gen_tokens - 1):
                    o = model(nxt.view(1, 1), past_key_values=past, use_cache=True)
                    past, nxt = o.past_key_values, o.logits[0, -1].argmax()
                    gen.append(nxt)
                out_ids = torch.stack(gen).view(1, -1)
            else:
                cur = ids
                for _ in range(gen_tokens):
                    nxt = model(cur).logits[0, -1].argmax()
                    cur = torch.cat([cur, nxt.view(1, 1)], dim=1)
                out_ids = cur[:, n0:]
        hk.remove()
        return tok.decode(out_ids[0]), time.time() - t0, n0

    # `scale` matches the harness: mean norm of the segment-mean activations.
    cap = {}

    def cap_hook(_m, _i, out):
        cap["h"] = (out[0] if isinstance(out, tuple) else out).detach()

    acc = []
    docs = []
    for _ in range(n_doc):
        sa, sb, car, c1, _c2 = make_pair(rng)
        for sents in (sa, sb):
            txt, spn = build(car, sents)
            stem = txt + (c1.rsplit(" ", 1)[0] if c1 else "")
            docs.append((stem, spn))
            e, ts = seg_spans(stem, spn)
            h_ = layers_[L].register_forward_hook(cap_hook)
            with torch.no_grad():
                model(e["input_ids"].to(dev))
            h_.remove()
            hh = cap["h"][0].float()
            acc.append(float(torch.stack(
                [hh[a:b + 1].mean(0) for a, b in ts]).norm(dim=-1).mean()))
    scale = sum(acc) / len(acc)
    print(f"scale = {scale:.2f}  alpha = {alpha}  layer = {L}  "
          f"{len(docs)} documents, {gen_tokens} tokens each", flush=True)

    n_same, first_diffs, t_c, t_u = 0, [], 0.0, 0.0
    for i, (stem, spn) in enumerate(docs):
        a_txt, ta, n0 = run(stem, spn, cached=True)
        b_txt, tb, _ = run(stem, spn, cached=False)
        t_c += ta
        t_u += tb
        if a_txt == b_txt:
            n_same += 1
        else:
            j = next((k for k in range(min(len(a_txt), len(b_txt)))
                      if a_txt[k] != b_txt[k]), min(len(a_txt), len(b_txt)))
            first_diffs.append(j)
            print(f"  [diff] doc {i} prompt_len {n0} first differing char {j}", flush=True)
            print(f"     cached   {a_txt[:120]!r}", flush=True)
            print(f"     uncached {b_txt[:120]!r}", flush=True)

    print(f"\nidentical: {n_same}/{len(docs)}", flush=True)
    if first_diffs:
        print(f"first-difference positions: {sorted(first_diffs)}", flush=True)
        print("A difference at position 0 means the hook is applied differently and the "
              "cached path is WRONG. Late, occasional differences are bf16 tie-breaking.",
              flush=True)
    print(f"time cached {t_c:.1f}s  uncached {t_u:.1f}s  speedup {t_u / max(t_c, 1e-9):.1f}x",
          flush=True)
    return {"n_docs": len(docs), "identical": n_same, "first_diffs": first_diffs,
            "gen_tokens": gen_tokens, "alpha": alpha, "layer": int(L),
            "t_cached": t_c, "t_uncached": t_u,
            "speedup": t_u / max(t_c, 1e-9)}


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-1.5B-Instruct", layer: int = 14, k_seg: int = 12,
         n_doc: int = 8, gen_tokens: int = 32, alpha: float = 1.0, seed: int = 31415,
         tag: str = ""):
    import json
    r = check.remote(model, layer, k_seg, n_doc, gen_tokens, alpha, seed)
    outdir = ROOT / "results" / "txc_wins"
    outdir.mkdir(parents=True, exist_ok=True)
    # TAGGED BY ALPHA. Both runs previously wrote `gen_cache_check.json`, so the alpha=0
    # control overwrote the alpha=1 experiment and the surviving artefact disagreed with
    # the reported speedup -- not because the number was wrong, but because the file that
    # supported it no longer existed. A check script that clobbers its own prior result
    # destroys exactly the evidence it was run to create.
    name = f"gen_cache_check_a{alpha:g}{tag}.json".replace("-", "m")
    (outdir / name).write_text(json.dumps(r, indent=2))
    print("[saved]", outdir / name)
