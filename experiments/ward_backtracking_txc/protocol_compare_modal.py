"""Do the three steering conventions differ on the backtracking task? Same clamp, three reductions.

Runs against the SAVED ward checkpoints (nothing is trained here). See
`docs/dmitry/reviewer_responses/steering_conventions.tex` for the algebra this tests.

THE CLAIM UNDER TEST. For a temporal crosscoder, clamping latent `j` to an absolute value `s`
produces a per-position delta that is a scalar multiple of the decoder slab,

    delta_w[t] = (s - z_{j,w}) * W_dec[j, t, :] = c_w * P_t                       (*)

because `decode` is linear in the window-level code `z` (verified: TXCBase decodes with
`einsum("bs,std->btd", z, W_dec) + b_dec`, `z` of shape `(B, d_sae)` -- no `t` index). The three
protocols then differ ONLY in how they reduce (*) before writing:

    v7    broadcast the window-MEAN to all T positions   ->  c_w * Pbar          (rank 1)
    slab  write row t at position t, no reduction        ->  c_w * P_t           (rank up to T)
    pp    stride-1 windows, average over overlap         ->  (1/T) sum_tau c_{p-tau} P_tau

`v7` is the paper's default (`protocol="v7"` in `SteeringConfig` and in
`experiments/c5_steering/run.py`); `pp` is its fallback; `slab` is this sprint's arm and is NOT
a protocol the paper runs. Writing `P = 1_T (x) Pbar + Ptilde` with `sum_t Ptilde_t = 0`, `v7`
writes only the DC part -- which is exactly what a per-token dictionary can already express --
so `Ptilde` is the only component on which a crosscoder can outperform, and `v7` deletes it.

WHY THE CLAMP IS HELD FIXED ACROSS PROTOCOLS. `pp` reduces to `v7` exactly when `c_w` is
constant across windows (Prop. 2 of the note), so a comparison that changes the clamp between
arms cannot tell "the reduction matters" from "the dose differs". Here all three see the SAME
`s`, the same windows and the same `c_w`; only the reduction changes. `var(c_w)/mean(c_w)^2` is
reported per prompt, because it is exactly the quantity that decides how far `pp` sits from
`v7`, and it was the open empirical question the note could not settle from algebra alone.

DOSE. `s` is set as a multiple `q` of the selected latent's p99 activation on the labelled
traces, so `q = 1` clamps to roughly its own natural firing strength. Realised injected
Frobenius norm per window is recorded for every protocol at every dose -- the protocols do NOT
inject equal norm at equal `q` (v7's write is `sqrt(T)`-shorter than the slab's whenever the
slab has any AC component), and pretending otherwise is the confound that this sprint hit twice.

READOUT, and its known weakness. `keyword_rate = (wait|hmm|actually|...)/words` on the
continuation. `grade_backtracking.py` in this directory states plainly that a keyword token can
be filler or pseudo-backtracking, so this is a SCREEN, not the behavioural measure. A protocol
difference visible here needs the Sonnet judge before it is a result. A protocol difference
ABSENT here is the more informative outcome, since the screen is generous.

    modal run experiments/ward_backtracking_txc/protocol_compare_modal.py \
        --train-key d7b2e24253055f8e --n-prompts 40
"""
import pathlib

import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[2]

HF_MODELS_REPO = "han1823123123/temp-bench-models"     # TEMP_BENCH_HF_ORG/temp-bench-models
# The ward datasource is `llama_3_1_8b_base_l10_ward_nousmirror`; "nousmirror" is the ungated
# NousResearch mirror, which is what the checkpoints were trained against.
SUBJECT = "NousResearch/Meta-Llama-3.1-8B"

app = modal.App("txc-protocol-compare")
results_vol = modal.Volume.from_name("txcwins-results", create_if_missing=True)
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers", "accelerate", "numpy", "safetensors",
                 "huggingface_hub")
    .add_local_dir(str(_here.parent / "_c7_eval"), "/work/c7_eval")
)

KEYWORDS = ("wait", "hmm", "actually", "no,", "hold on", "but wait", "let me reconsider",
            "on second thought", "that's wrong", "i made a mistake")


# Both the checkpoint repo and the Nous mirror are public, so no HF secret is needed.
@app.function(gpu="A100-40GB", image=image, timeout=14400, volumes={"/out": results_vol})
def run(train_key: str, n_prompts: int, n_sel: int, qs: list, max_new: int, out_name: str):
    import json
    import re
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda"

    # ---- 1. the saved checkpoint (weights only; nothing is trained here) ----
    cfg = json.load(open(hf_hub_download(HF_MODELS_REPO, f"{train_key}/config.json")))
    sd = load_file(hf_hub_download(HF_MODELS_REPO, f"{train_key}/model.safetensors"))
    W_dec = sd["W_dec"].float().to(dev)                       # (d_sae, T, d_in)
    W_enc = sd["W_enc"].float().to(dev)                       # (T, d_in, d_sae)
    b_enc = sd["b_enc"].float().to(dev)
    d_sae, T, d_in = W_dec.shape
    k_win = int(cfg.get("training_cfg", {}).get("k_win") or 0) or 100
    print(f"[ckpt] {train_key} arch={cfg.get('arch')} d_sae={d_sae} T={T} d_in={d_in} "
          f"datasource={cfg.get('datasource')}", flush=True)

    def encode(win):
        """win: (B, T, d_in) -> window-level code (B, d_sae), window TopK."""
        pre = torch.einsum("btd,tds->bs", win, W_enc) + b_enc
        vals, idx = pre.topk(min(k_win, pre.shape[-1]), dim=-1)
        z = torch.zeros_like(pre).scatter_(-1, idx, torch.relu(vals))
        return z

    # ---- 2. subject model ----
    tok = AutoTokenizer.from_pretrained(SUBJECT)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        SUBJECT, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    layer = int(cfg.get("datasource", "").split("_l")[-1].split("_")[0] or 10)
    site = model.model.layers[layer]
    print(f"[subject] {SUBJECT} layer={layer}", flush=True)

    labels = json.load(open("/work/c7_eval/sentence_labels.json"))
    prompts = json.load(open("/work/c7_eval/prompts.json"))

    # ---- 3. select the backtracking latent on the LABELLED traces ----
    # A window is positive if its token span overlaps a sentence marked is_backtracking.
    acts, tags = [], []

    def grab(mod, inp, out):
        acts.append((out[0] if isinstance(out, tuple) else out).detach())

    pos_z, neg_z = [], []
    h = site.register_forward_hook(grab)
    for rec in labels[:n_sel]:
        sents = rec["sentences"]
        if not sents:
            continue
        text = "".join(s["sentence"] + " " for s in sents)
        spans, cur = [], 0
        for s in sents:
            n = len(s["sentence"]) + 1
            spans.append((cur, cur + n, bool(s["is_backtracking"])))
            cur += n
        enc = tok(text, return_tensors="pt", truncation=True, max_length=1024,
                  return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")[0].tolist()
        acts.clear()
        with torch.no_grad():
            model(**{k: v.to(dev) for k, v in enc.items()})
        H = acts[0][0].float()                                  # (S, d_in)
        S = H.shape[0]
        if S < T:
            continue
        is_bt = np.zeros(S, dtype=bool)
        for i, (a, b) in enumerate(offs):
            for (cs, ce, f) in spans:
                if f and a < ce and b > cs:
                    is_bt[i] = True
                    break
        nb = S // T
        wins = H[: nb * T].reshape(nb, T, d_in)
        with torch.no_grad():
            z = encode(wins)                                    # (nb, d_sae)
        lab = is_bt[: nb * T].reshape(nb, T).any(axis=1)
        for w in range(nb):
            (pos_z if lab[w] else neg_z).append(z[w])
    h.remove()
    if not pos_z or not neg_z:
        raise RuntimeError("no labelled windows on either side; raise n_sel")
    P = torch.stack(pos_z).cpu().numpy()
    N = torch.stack(neg_z).cpu().numpy()
    print(f"[select] windows: {len(P)} backtracking / {len(N)} not", flush=True)

    # Rank-based AUC per latent, vectorised.
    allz = np.concatenate([P, N], 0)
    y = np.concatenate([np.ones(len(P)), np.zeros(len(N))])
    order = allz.argsort(0)
    ranks = np.empty_like(order, dtype=np.float64)
    np.put_along_axis(ranks, order, np.arange(1, len(allz) + 1)[:, None], axis=0)
    n1, n0 = y.sum(), len(y) - y.sum()
    auc = ((ranks * y[:, None]).sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    j = int(np.abs(auc - 0.5).argmax())
    sign = 1.0 if auc[j] > 0.5 else -1.0
    p99 = float(np.percentile(allz[:, j], 99)) or 1.0
    print(f"[select] latent {j}  AUC {auc[j]:.3f}  sign {sign:+.0f}  p99 {p99:.3f}", flush=True)

    slab = W_dec[j]                                             # (T, d_in)
    Pn = slab / slab.norm()
    dc = Pn.mean(0)
    ac_share = float((Pn - dc).norm() ** 2 / Pn.norm() ** 2)
    print(f"[slab] ||AC||^2/||P||^2 = {ac_share:.3f}  (0 => v7 loses nothing)", flush=True)

    # ---- 4. the three reductions, same clamp ----
    state = {"q": 0.0, "proto": None, "norms": [], "cvar": []}

    def make_hook():
        def hook(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            if state["proto"] is None or state["q"] == 0.0:
                return None
            B, S, _ = hs.shape
            if S < T:
                return None
            hf = hs.float()
            s_abs = sign * state["q"] * p99
            proto = state["proto"]
            delta = torch.zeros_like(hf)
            if proto in ("v7", "slab"):
                starts = [b * T for b in range(S // T)]
                if S % T:
                    starts.append(S - T)
                for st in starts:
                    win = hf[:, st:st + T, :]
                    with torch.no_grad():
                        c = (s_abs - encode(win)[:, j]).view(B, 1, 1)      # (B,1,1)
                    state["cvar"].append(float(c.mean().abs()))
                    d_pos = c * Pn.unsqueeze(0)                            # (B,T,d) = c * P_t
                    delta[:, st:st + T, :] = (
                        d_pos.mean(1, keepdim=True).expand(-1, T, -1) if proto == "v7"
                        else d_pos)
            else:  # pp: stride-1, overlap-averaged
                K = S - T + 1
                accum = torch.zeros_like(hf)
                cnt = torch.zeros(B, S, 1, device=hf.device)
                for w in range(K):
                    win = hf[:, w:w + T, :]
                    with torch.no_grad():
                        c = (s_abs - encode(win)[:, j]).view(B, 1, 1)
                    state["cvar"].append(float(c.mean().abs()))
                    accum[:, w:w + T, :] += c * Pn.unsqueeze(0)
                    cnt[:, w:w + T, :] += 1.0
                delta = accum / cnt.clamp(min=1.0)
            state["norms"].append(float(delta.norm(dim=(1, 2)).mean()))
            return ((hf + delta).to(hs.dtype),) + out[1:] if isinstance(out, tuple) \
                else (hf + delta).to(hs.dtype)
        return hook

    kw_re = re.compile("|".join(re.escape(k) for k in KEYWORDS), re.I)

    def kw_rate(t):
        w = max(len(t.split()), 1)
        return len(kw_re.findall(t)) / w

    handle = site.register_forward_hook(make_hook())
    rows = []
    sel = prompts[:n_prompts]
    for proto in ("v7", "pp", "slab"):
        for q in qs:
            state.update(proto=proto, q=float(q), norms=[], cvar=[])
            kws, texts = [], []
            for rec in sel:
                enc = tok(rec["prompt"], return_tensors="pt", truncation=True,
                          max_length=512).to(dev)
                with torch.no_grad():
                    o = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                       pad_token_id=tok.pad_token_id)
                gen = tok.decode(o[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
                kws.append(kw_rate(gen))
                texts.append(gen[:400])
            cv = np.array(state["cvar"]) if state["cvar"] else np.array([0.0])
            rows.append({
                "protocol": proto, "q": float(q),
                "keyword_rate_mean": float(np.mean(kws)),
                "keyword_rate_sem": float(np.std(kws, ddof=1) / max(len(kws) ** 0.5, 1)),
                "injected_norm_mean": float(np.mean(state["norms"])) if state["norms"] else 0.0,
                "c_rel_var": float(cv.var() / max(cv.mean() ** 2, 1e-12)),
                "sample": texts[0],
            })
            print(f"[{proto:>4} q={q:<5}] kw {rows[-1]['keyword_rate_mean']:.4f}  "
                  f"|delta|_F {rows[-1]['injected_norm_mean']:.2f}  "
                  f"var(c)/mean(c)^2 {rows[-1]['c_rel_var']:.4f}", flush=True)
    handle.remove()

    out = {"train_key": train_key, "arch": cfg.get("arch"),
           "datasource": cfg.get("datasource"), "subject": SUBJECT, "layer": layer,
           "T": T, "d_sae": d_sae, "latent": j, "latent_auc": float(auc[j]),
           "sign": sign, "p99": p99, "ac_share": ac_share,
           "n_prompts": len(sel), "n_sel_traces": n_sel, "max_new": max_new, "rows": rows}
    pathlib.Path("/out", out_name).write_text(json.dumps(out, indent=2))
    results_vol.commit()
    print(f"[volume] wrote /out/{out_name}", flush=True)
    return out


@app.local_entrypoint()
def main(train_key: str = "d7b2e24253055f8e", n_prompts: int = 40, n_sel: int = 60,
         qs: str = "0,1,2,4,8", max_new: int = 200, tag: str = ""):
    import json
    r = run.remote(train_key, n_prompts, n_sel,
                   [float(x) for x in qs.split(",")], max_new,
                   f"protocol_compare_{train_key}{tag}.json")
    d = ROOT / "results" / "txc_wins"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"protocol_compare_{train_key}{tag}.json"
    p.write_text(json.dumps(r, indent=2))
    print("[saved]", p)
