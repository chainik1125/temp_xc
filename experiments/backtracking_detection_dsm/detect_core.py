"""c7 backtracking-detection protocol applied to the diffusion-arm TopK SAEs.

Runs the paper's "backtracking imminent" detection on Llama-3.1-8B ln1_L10 for
the four recon/dsm dictionaries trained on FineWeb, against the Stage B
originals, through one identical pipeline.

Reused, not reimplemented:
  - D+/D- labels: results/ward_backtracking/sentence_labels.json, produced by
    experiments/ward_backtracking/label_sentences.py (Haiku-4.5 sentence judge,
    Ward 2025 sec 2.2). No relabelling happens here.
  - Anchor alignment (sentence char_start -> token position) and the [-13,-8]
    offset window: copied from ward_backtracking_txc/mine_features.py
    ::_capture_windows and config.yaml mining.offset_window.
  - Probe + metrics (`probe_cv`, `prauc`, `rocauc`): copied verbatim from
    docs/dmitry/sprints/2026-06-10_freqbench_sprint/code/c7_detect.py.
  - Negative sampling variant B (`far` set): c7_detect.py::collect_examples.
  - Architecture rebuild for Stage B checkpoints: ward_backtracking_txc
    ::architectures.build_arch, driven by each checkpoint's own stored config.

Provenance note kept deliberately: the traces are DeepSeek-R1-Distill-Llama-8B
generations, the activations are base Llama-3.1-8B. That train/eval mismatch is
the paper's own setup and is replicated, not corrected.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

D_MODEL = 4096
LAYER = 10
WIN_OFF = list(range(-13, -7))        # 6 positions, offsets -13..-8
NEG_BUFFER = 25                       # c7_detect.py
NEG_PER_POS = 5                       # c7_detect.py
BASE_MODEL_CANDIDATES = [
    "NousResearch/Meta-Llama-3.1-8B",  # the mirror the dsm/recon arms trained on
    "meta-llama/Llama-3.1-8B",
]


# --------------------------------------------------------------------------
# metrics + probe -- verbatim from c7_detect.py
# --------------------------------------------------------------------------

_trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy 2 vs 1 spelling


def prauc(y, s):
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / (np.arange(len(y)) + 1)
    rec = tp / y.sum()
    return float(_trapz(prec, rec))


def rocauc(y, s):
    from scipy.stats import rankdata
    r = rankdata(s)
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def probe_cv(F, y, groups, S_values=(8, 32)):
    """c7_detect.py::probe_cv. Loop order is fold-outer / S-inner instead of
    the reference's S-outer / fold-inner so that each fold's train matrix and
    Welch t-statistic are materialised once rather than once per S; the folds,
    the t-statistic, the feature selection and the fitted probe are unchanged,
    so the returned numbers are identical.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    acc = {S: {"pr": [], "roc": []} for S in S_values}
    for tr_idx, te_idx in GroupKFold(5).split(F, y, groups):
        Ftr_full, Fte_full = F[tr_idx], F[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]
        pos, neg = ytr == 1, ytr == 0
        mu1 = Ftr_full[pos].mean(0)
        mu0 = Ftr_full[neg].mean(0)
        s1 = Ftr_full[pos].var(0) / max(pos.sum(), 1)
        s0 = Ftr_full[neg].var(0) / max(neg.sum(), 1)
        t = (mu1 - mu0) / np.sqrt(s1 + s0 + 1e-9)
        order = np.argsort(-np.abs(t))
        for S in S_values:
            if S is not None:
                feats = order[:S]
                Ftr, Fte = Ftr_full[:, feats], Fte_full[:, feats]
            else:
                Ftr, Fte = Ftr_full, Fte_full
            sc = Ftr.std(0) + 1e-9
            clf = LogisticRegression(penalty="l1", C=1.0, solver="liblinear",
                                     max_iter=2000)
            clf.fit(Ftr / sc, ytr)
            s = clf.decision_function(Fte / sc)
            acc[S]["pr"].append(prauc(yte, s))
            acc[S]["roc"].append(rocauc(yte, s))
        del Ftr_full, Fte_full
    res = {}
    for S in S_values:
        tag = f"S{S}" if S else "all"
        res[f"prauc_{tag}"] = float(np.mean(acc[S]["pr"]))
        res[f"rocauc_{tag}"] = float(np.mean(acc[S]["roc"]))
        res[f"prauc_{tag}_folds"] = [round(v, 4) for v in acc[S]["pr"]]
    return res


# --------------------------------------------------------------------------
# stage 1: activation capture over the 300 traces
# --------------------------------------------------------------------------

def _load_base_llm(device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    last = None
    for mid in BASE_MODEL_CANDIDATES:
        try:
            tok = AutoTokenizer.from_pretrained(mid, use_fast=True)
            llm = AutoModelForCausalLM.from_pretrained(
                mid, torch_dtype=torch.bfloat16).to(device).eval()
            for p in llm.parameters():
                p.requires_grad_(False)
            print(f"[llm] loaded {mid}", flush=True)
            return llm, tok, mid
        except Exception as e:                                   # gated/missing
            print(f"[llm] {mid} unavailable: {type(e).__name__}: {e}", flush=True)
            last = e
    raise RuntimeError(f"no base Llama-3.1-8B source: {last}")


def capture_traces(traces_path: str, labels_path: str, device="cuda",
                   limit_traces: int | None = None):
    """Forward all 300 traces through base Llama; return the concatenated
    ln1_L10 and resid_L10 caches plus per-trace anchor bookkeeping.

    ln1 is captured on `layers[L].input_layernorm`'s output, which is exactly
    the tensor psc_train_sae.py::make_hook trained the new dictionaries on. We
    also register the paper's forward-pre-hook on `self_attn` (as used by
    ward_backtracking_txc/cache_activations.py) and assert the two agree, so
    the new dictionaries and the Stage B baseline are provably reading the same
    hookpoint.
    """
    llm, tok, model_id = _load_base_llm(device)
    traces = json.loads(Path(traces_path).read_text())
    labels = json.loads(Path(labels_path).read_text())
    traces_by_qid = {t["question_id"]: t for t in traces}
    if limit_traces:                              # smoke mode only
        labels = labels[:limit_traces]

    store: dict = {}

    def ln1_hook(_m, _i, output):                 # psc_train_sae.py convention
        h = output[0] if isinstance(output, tuple) else output
        store["ln1"] = h.detach()

    def ln1_pre_hook(_m, args, kwargs):           # paper convention (cache_activations)
        h = args[0] if args else kwargs["hidden_states"]
        store["ln1_paper"] = h.detach()

    def resid_hook(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        store["resid"] = h.detach()

    blk = llm.model.layers[LAYER]
    handles = [
        blk.input_layernorm.register_forward_hook(ln1_hook),
        blk.self_attn.register_forward_pre_hook(ln1_pre_hook, with_kwargs=True),
        blk.register_forward_hook(resid_hook),
    ]

    ln1_parts, resid_parts = [], []
    offsets = [0]
    trace_meta = []               # per kept trace: dict(qid, n_tokens, anchors...)
    hook_agree = None
    t0 = time.time()
    try:
        for i, rec in enumerate(labels):
            qid = rec["question_id"]
            tr = traces_by_qid.get(qid)
            if tr is None or not rec.get("sentences"):
                continue
            full = tr["full_response"]
            enc = tok(full, return_tensors="pt", return_offsets_mapping=True,
                      add_special_tokens=False)
            ids = enc["input_ids"].to(device)
            omap = enc["offset_mapping"][0].tolist()
            with torch.no_grad():
                llm.model(input_ids=ids, use_cache=False)

            if hook_agree is None:
                hook_agree = bool(torch.equal(store["ln1"], store["ln1_paper"]))
                print(f"[hook] input_layernorm-out == self_attn-in : {hook_agree}",
                      flush=True)

            a_ln1 = store["ln1"][0].to(torch.float16).cpu().numpy()
            a_res = store["resid"][0].to(torch.float16).cpu().numpy()
            n_tok = a_ln1.shape[0]

            # sentence -> token anchor, verbatim logic from mine_features.py
            think_idx = full.find("<think>")
            think_off = (think_idx + len("<think>")) if think_idx >= 0 else 0
            anchors, is_bt, keys = [], [], []
            for s_idx, sent in enumerate(rec["sentences"]):
                target_char = think_off + sent["char_start"]
                tok_pos = -1
                for j, (cs, ce) in enumerate(omap):
                    if cs <= target_char < ce or cs >= target_char:
                        tok_pos = j
                        break
                if tok_pos < 0:
                    continue
                anchors.append(tok_pos)
                is_bt.append(bool(sent["is_backtracking"]))
                keys.append(f"{qid}|{rec['trace_idx']}|{s_idx}")

            ln1_parts.append(a_ln1)
            resid_parts.append(a_res)
            offsets.append(offsets[-1] + n_tok)
            trace_meta.append({
                "qid": qid, "trace_idx": rec["trace_idx"], "n_tokens": int(n_tok),
                "think_lo": 0 if think_idx < 0 else int(
                    next((j for j, (cs, ce) in enumerate(omap) if cs >= think_off), 0)),
                "anchors": anchors, "is_bt": is_bt, "keys": keys,
            })
            store.clear()
            if i % 50 == 0:
                print(f"  trace {i}/{len(labels)} tok={offsets[-1]:,} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    finally:
        for h in handles:
            h.remove()

    del llm
    torch.cuda.empty_cache()
    cache = {
        "ln1": np.concatenate(ln1_parts),
        "resid": np.concatenate(resid_parts),
    }
    print(f"[capture] {len(trace_meta)} traces, {cache['ln1'].shape[0]:,} tokens, "
          f"{time.time() - t0:.0f}s", flush=True)
    return cache, np.asarray(offsets, dtype=np.int64), trace_meta, {
        "model": model_id, "hooks_agree": hook_agree,
        "n_tokens": int(cache["ln1"].shape[0]),
    }


# --------------------------------------------------------------------------
# stage 2: example sets
# --------------------------------------------------------------------------

def collect_examples(trace_meta):
    """Two example sets over the same anchors and the same [-13,-8] window.

    'sentence' -- the paper's own D+/D- split: every judged sentence is an
        example, positive iff is_backtracking. This is the definition
        mine_features.py mines with.
    'far' -- c7_detect.py::collect_examples: positives are the backtracking
        anchors, negatives are random positions > NEG_BUFFER tokens from any
        backtracking anchor, NEG_PER_POS per positive, rng seed 11.

    Returns {name: (list[(trace_k, anchor_pos, label)], )}.
    """
    rng = np.random.default_rng(11)
    sentence_ex, far_ex = [], []
    for k, m in enumerate(trace_meta):
        n_tok, lo = m["n_tokens"], m["think_lo"]
        ev = []
        for pos, bt in zip(m["anchors"], m["is_bt"]):
            if pos + WIN_OFF[0] < lo or pos + WIN_OFF[-1] >= n_tok:
                continue
            sentence_ex.append((k, int(pos), int(bt)))
            if bt:
                ev.append(pos)
        ev = np.asarray(ev, dtype=int)
        for e in ev:
            far_ex.append((k, int(e), 1))
        cand = [p for p in range(lo + 13, n_tok)
                if not len(ev) or np.abs(ev - p).min() > NEG_BUFFER]
        n_neg = min(len(cand), NEG_PER_POS * max(1, int(len(ev))))
        if not len(ev):
            n_neg = min(len(cand), 4)
        for i in rng.choice(len(cand), size=n_neg, replace=False):
            far_ex.append((k, int(cand[i]), 0))
    return {"sentence": sentence_ex, "far": far_ex}


def gather_windows(cache_hook: np.ndarray, offsets: np.ndarray, examples,
                   device="cuda", chunk=4096):
    """examples -> (n, 6, d) float32 torch tensor on CPU."""
    base = np.array([offsets[k] + p for k, p, _ in examples], dtype=np.int64)
    idx = base[:, None] + np.asarray(WIN_OFF, dtype=np.int64)[None, :]
    out = np.empty((len(examples), len(WIN_OFF), cache_hook.shape[1]),
                   dtype=np.float16)
    for s in range(0, len(examples), chunk):
        out[s:s + chunk] = cache_hook[idx[s:s + chunk]]
    return torch.from_numpy(out)


# --------------------------------------------------------------------------
# stage 3: dictionaries
# --------------------------------------------------------------------------

def load_new_dict(path: str, device="cuda", d=D_MODEL, H=16384, k=64):
    """The recon/dsm arms: plain state_dict of psc_train_sae.py::TopKSAE.
    Rebuilt with the identical class from topk_vs_topkdiff/sae.py."""
    from experiments.diffusion_txc.topk_vs_topkdiff.sae import TopKSAE
    sd = torch.load(path, map_location="cpu", weights_only=True)
    m = TopKSAE(d, H, k)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if any(x for x in missing if x != "steps_since_fire"):
        raise RuntimeError(f"{path}: missing keys {missing}")
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m, {"arch": "topk_sae_flat", "d_sae": H, "k": k, "T": 1,
               "missing_keys": list(missing), "unexpected_keys": list(unexpected)}


def load_stageb_dict(cell: str, repo: str, device="cuda"):
    """Stage B checkpoint -> (model, meta). Architecture comes from the config
    stored inside the .pt, exactly as sae_deadlatent_audit/modal_app.py does."""
    import sys
    sys.path.insert(0, "/work")
    from huggingface_hub import hf_hub_download
    from experiments.ward_backtracking_txc.architectures import build_arch

    path = hf_hub_download(repo, f"checkpoints/{cell}.pt")
    obj = torch.load(path, map_location="cpu", weights_only=False)
    c = obj["config"]
    arch = c.get("arch", "txc")
    m = build_arch(arch, d_in=c["d_in"], d_sae=c["d_sae"], T=c["T"],
                   k=c["k_per_position"], **(c.get("arch_kwargs") or {}))
    m.load_state_dict(obj["state_dict"])
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    hp = c["hookpoint"]
    hp_key = hp["key"] if isinstance(hp, dict) else str(hp)
    hp_comp = hp["component"] if isinstance(hp, dict) else hp_key.split("_")[0]
    hist = obj.get("history") or []
    return m, {"arch": arch, "d_sae": int(c["d_sae"]), "T": int(c["T"]),
               "k_per_position": int(c["k_per_position"]), "hookpoint": hp_key,
               "hook_component": hp_comp, "seed": c.get("seed"),
               "final_eval_fvu": (hist[-1] if hist else {}).get("fvu_eval")}


def load_w6_dict(path: str, kind: str, device="cuda", d=D_MODEL, T=None,
                 H=16384, k=96):
    """The window-6 trio trained by psc_train_sae.py --window 6 on Llama-3.1-8B
    resid L10: flat T*d = 24576 inputs, H=16384. The training flatten is
    TIME-MAJOR ([x_t0; x_t1; ...; x_t5]): unfold(1, T, 1) puts the window
    offset in the last axis, permute(0, 1, 3, 2) moves it before d, and the
    reshape therefore lays out d-blocks in time order — exactly what
    gather_windows' (n, T, d) gives under a plain reshape(B, T*d). The
    two-ordering NMSE gate in run_detection verifies this empirically.

    recon/dsm: psc TopKSAE state dicts loaded into topk_vs_topkdiff.sae.TopKSAE
    (verified state-dict identical: same parameter/buffer names and shapes,
    same encode = scatter(relu(topk(pre)))).
    bayes: BayesGateSAE via bayes_eval_adapter (pinned eval semantics: gate
    conditioned at u=0, hard threshold g>0.5, z = m * 1[g>0.5]).
    """
    T = T or len(WIN_OFF)
    d_eff = d * T
    if kind == "bayes":
        from experiments.diffusion_txc.topk_vs_topkdiff.bayes_eval_adapter \
            import load_bayes_sae
        m = load_bayes_sae(path, device, d=d_eff)
        return m, {"arch": "w6_flat", "kind": "bayes_gate", "d_in": d_eff,
                   "T": T, "d_sae": H, "flatten": "time_major",
                   "gate_rule": "u=0, hard g>0.5, z=m*1[g>0.5]",
                   "ckpt": str(path)}
    from experiments.diffusion_txc.topk_vs_topkdiff.sae import TopKSAE
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if tuple(sd["W_enc"].shape) != (d_eff, H):
        raise RuntimeError(f"{path}: W_enc {tuple(sd['W_enc'].shape)}, "
                           f"expected {(d_eff, H)}")
    m = TopKSAE(d_eff, H, k)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if any(x != "steps_since_fire" for x in missing) or unexpected:
        raise RuntimeError(f"{path}: missing={missing} unexpected={unexpected}")
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m, {"arch": "w6_flat", "kind": kind, "d_in": d_eff, "T": T,
               "d_sae": H, "k": k, "flatten": "time_major", "ckpt": str(path)}


@torch.no_grad()
def w6_nmse_both_orders(model, X: torch.Tensor, device="cuda", batch=1024,
                        max_rows=4096) -> dict:
    """Flatten-order verification gate for the w6 dictionaries.

    Encodes/decodes the same (n, T, d) windows under both flatten conventions:
      time_major: x.reshape(B, T*d)               = [x_t0; x_t1; ...; x_t5]
                  (the psc_train_sae --window unfold+permute convention)
      dim_major:  x.permute(0, 2, 1).reshape(B, d*T)
    and returns the NMSE (sum||x-recon||^2 / sum||x-mean||^2, the training
    script's metric) of each. The training convention must be clearly lower;
    the caller aborts the arm otherwise.
    """
    X = X[:max_rows]
    res = {"n_rows": int(X.shape[0])}
    for name, flat in (("time_major", lambda t: t.reshape(t.shape[0], -1)),
                       ("dim_major",
                        lambda t: t.permute(0, 2, 1).reshape(t.shape[0], -1))):
        mu = flat(X.mean(0, dtype=torch.float32).unsqueeze(0)).to(device)
        num = den = 0.0
        for s in range(0, X.shape[0], batch):
            xb = flat(X[s:s + batch].to(device, torch.float32))
            z = model.encode(xb)
            r = z @ model.W_dec + model.b_dec
            num += float((xb - r).pow(2).sum())
            den += float((xb - mu).pow(2).sum())
        res[f"nmse_{name}"] = num / den
    return res


@torch.no_grad()
def w6_gate_l0(model, X: torch.Tensor, device="cuda", batch=1024):
    """Mean/std of the bayes gate count (g > 0.5 at u=0 — training's
    ``l0_gate_half``) over the (n, T, d) eval windows, time-major flatten."""
    l0 = []
    for s in range(0, X.shape[0], batch):
        x = X[s:s + batch].to(device, torch.float32)
        _, g = model.sae.encode(x.reshape(x.shape[0], -1), u=0.0)
        l0.append((g > 0.5).float().sum(1))
    l0 = torch.cat(l0)
    return float(l0.mean()), float(l0.std())


WINDOW_ARCHS = {"txc", "txc_h8", "txc_h13", "stacked_sae"}


@torch.no_grad()
def window_features(model, arch: str, X: torch.Tensor, device="cuda",
                    batch=1024) -> np.ndarray:
    """(n, 6, d) -> (n, d_sae) max-pooled |activation| over the window.

    Per-token dictionaries encode each of the 6 positions and max-pool |z|
    across them (c7_detect.py::Dict5.maxpool_codes). Window dictionaries with
    T == 6 consume the whole window in one encode and yield a single window
    latent, so |z| is already the window-level score (this is the same latent
    mine_features.py ranks D+/D- selectivity on). "w6_flat" dictionaries are
    window dictionaries that consume the time-major-flattened (B, T*d) window
    (the psc --window training convention; see load_w6_dict), same window
    latent |z| score.
    """
    outs = []
    for s in range(0, X.shape[0], batch):
        x = X[s:s + batch].to(device, torch.float32)
        if arch in WINDOW_ARCHS:
            z = model.encode(x)                             # (B, d_sae)
            if isinstance(z, tuple):
                z = z[0]
            f = z.abs()
        elif arch == "w6_flat":
            z = model.encode(x.reshape(x.shape[0], -1))     # time-major flatten
            f = z.abs()
        else:
            z = model.encode(x.reshape(-1, x.shape[-1]))
            f = z.reshape(x.shape[0], x.shape[1], -1).abs().amax(1)
        outs.append(f.cpu().numpy().astype(np.float32))
    return np.concatenate(outs)


# --------------------------------------------------------------------------
# post-hoc threshold-gate swap
# --------------------------------------------------------------------------
# `calibrate` and `GateSAE` are imported unmodified from
# topk_vs_topkdiff/posthoc_gate_evals.py. Both assume the psc/topk_vs_topkdiff
# TopKSAE layout: attributes b_dec (d,), W_enc (d, H), b_enc (H,), k, and an
# input matrix (N, d). The adapters below present every other dictionary in
# that layout so the reference implementation runs verbatim.


class _StageBFlatAdapter:
    """temporal_crosscoders.models.TopKSAE stores W_enc as (H, d) and encodes
    (x - b_dec) @ W_enc.T; transposing is all that differs."""

    def __init__(self, m):
        self.b_dec, self.b_enc, self.k = m.b_dec, m.b_enc, m.k
        self.W_enc = m.W_enc.T                      # (H, d) -> (d, H)


class _WindowAdapter:
    """TXC-family window encoders compute einsum('btd,tds->bs', x, W_enc)
    + b_enc with no pre-bias. That contraction is exactly
    x.reshape(B, T*d) @ W_enc.reshape(T*d, H), so a zero b_dec and a reshaped
    W_enc turn the window model into the flat layout without approximation.
    self.k is already the window-level budget (k_per_position * T)."""

    def __init__(self, m, T: int, d: int):
        self.W_enc = m.W_enc.reshape(T * d, -1)
        self.b_enc, self.k = m.b_enc, m.k
        self.b_dec = torch.zeros(T * d, device=self.W_enc.device,
                                 dtype=self.W_enc.dtype)


def gate_adapter(model, arch: str, T: int = 6, d: int = D_MODEL):
    """-> (adapter, is_window). `is_window` selects how windows are flattened."""
    if arch == "topk_sae_flat":                     # the new recon/dsm arms
        return model, False
    if arch == "topk_sae":                          # Stage B per-token SAE
        return _StageBFlatAdapter(model), False
    if arch in WINDOW_ARCHS:
        return _WindowAdapter(model, T, d), True
    raise ValueError(f"no gate adapter for arch {arch}")


def calibration_matrix(cache_hook: np.ndarray, is_window: bool, n_rows: int,
                       T: int = 6) -> torch.Tensor:
    """Held-out calibration rows taken from the TAIL of the concatenated trace
    activations — i.e. the last traces in the corpus, so the slice is
    trace-disjoint as well as position-disjoint. Calibration is label-free
    (it only rate-matches preactivation quantiles to the native TopK firing
    rate), so it cannot leak D+/D- information into the probe folds.
    """
    tail = np.asarray(cache_hook[-(n_rows + T):])
    if not is_window:
        return torch.from_numpy(tail[-n_rows:]).float()
    w = np.lib.stride_tricks.sliding_window_view(tail, T, axis=0)
    w = np.ascontiguousarray(w.transpose(0, 2, 1)[-n_rows:])    # (N, T, d)
    return torch.from_numpy(w.reshape(w.shape[0], -1)).float()


@torch.no_grad()
def gate_window_features(gate, is_window: bool, X: torch.Tensor,
                         device="cuda", batch=1024):
    """Same max-pool-|z| score as `window_features`, but through the threshold
    gate. Returns (features, mean_l0, std_l0) with L0 measured on these very
    eval windows."""
    outs, l0 = [], []
    for s in range(0, X.shape[0], batch):
        x = X[s:s + batch].to(device, torch.float32)
        if is_window:
            z = gate.encode(x.reshape(x.shape[0], -1))          # (B, H)
            f = z.abs()
            l0.append((z > 0).float().sum(1))
        else:
            z = gate.encode(x.reshape(-1, x.shape[-1]))         # (B*T, H)
            zt = z.reshape(x.shape[0], x.shape[1], -1)
            f = zt.abs().amax(1)
            l0.append((z > 0).float().sum(1))
        outs.append(f.cpu().numpy().astype(np.float32))
    l0 = torch.cat(l0)
    return np.concatenate(outs), float(l0.mean()), float(l0.std())


@torch.no_grad()
def w6_fire_counts(model, cache_hook: np.ndarray, device="cuda") -> np.ndarray:
    """Per-latent fire counts (post-TopK z > 0) over every stride-1 time-major
    T-window of the trace cache — the counts behind dead_on_traces' w6_flat
    summary, exposed as an array so the live-pool-matched controls can take
    the exact live latent index set."""
    T = len(WIN_OFF)
    d_flat = cache_hook.shape[1] * T
    d_sae = int(model.encode(torch.zeros(2, d_flat, device=device)
                             ).shape[-1])
    counts = torch.zeros(d_sae, dtype=torch.int64, device=device)
    n_win = cache_hook.shape[0] - T + 1
    step = 4096
    for s in range(0, n_win, step):
        m_rows = min(step, n_win - s)
        block = np.asarray(cache_hook[s:s + m_rows + T - 1])
        w = np.lib.stride_tricks.sliding_window_view(block, T, axis=0)
        x = torch.from_numpy(np.ascontiguousarray(
            w.transpose(0, 2, 1)).reshape(m_rows, d_flat)).to(
            device, torch.float32)                           # time-major
        counts += (model.encode(x) > 0).sum(0)
    return counts.cpu().numpy()


@torch.no_grad()
def dead_on_traces(model, arch: str, cache_hook: np.ndarray, device="cuda",
                   chunk=8192) -> dict:
    """Fraction of latents that never fire (post-TopK value > 0) on any of the
    ~481k trace tokens. Same fire definition as sae_deadlatent_audit.

    For "w6_flat" arms the unit is the stride-1 T-window instead of the token
    (time-major flatten, matching training). The stride-1 sweep ignores the
    ~299 trace boundaries — a negligible fraction of ~481k windows — same
    simplification as calibration_matrix.
    """
    if arch == "w6_flat":
        c = w6_fire_counts(model, cache_hook, device)
        d_sae = int(c.shape[0])
        n_win = cache_hook.shape[0] - len(WIN_OFF) + 1
        return {"d_sae": d_sae, "unit": "stride1_window", "n_windows": int(n_win),
                "dead_never_fired": int((c == 0).sum()),
                "dead_fraction": float((c == 0).mean()),
                "fire_rate_median": float(np.median(c / max(n_win, 1))),
                "effective_alive_fraction_gt_1e-5":
                    float(((c / max(n_win, 1)) > 1e-5).mean())}

    d_sae = int(model.encode(torch.zeros(2, cache_hook.shape[1], device=device)
                             ).shape[-1])
    counts = torch.zeros(d_sae, dtype=torch.int64, device=device)
    n = cache_hook.shape[0]
    for s in range(0, n, chunk):
        x = torch.from_numpy(np.asarray(cache_hook[s:s + chunk])).to(
            device, torch.float32)
        counts += (model.encode(x) > 0).sum(0)
    c = counts.cpu().numpy()
    return {"d_sae": d_sae, "n_tokens": int(n),
            "dead_never_fired": int((c == 0).sum()),
            "dead_fraction": float((c == 0).mean()),
            "fire_rate_median": float(np.median(c / max(n, 1))),
            "effective_alive_fraction_gt_1e-5": float(((c / max(n, 1)) > 1e-5).mean())}
