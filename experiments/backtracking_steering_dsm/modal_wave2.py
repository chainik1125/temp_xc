"""Wave 2: T=6 window dictionaries on resid_L10, plus the denoise-after-steer arm.

    # 1. projector pre-flight -- MUST pass before the projected grid is worth running
    uvx modal run experiments/backtracking_steering_dsm/modal_wave2.py::preflight_cmd
    # 2. mine the trio's best backtracking feature
    uvx modal run --detach experiments/backtracking_steering_dsm/modal_wave2.py::mine_cmd
    # 3. the grid. NAME THE ENTRYPOINT -- this module has four local
    #    entrypoints, so a bare `modal run <file>` does not default to main:
    #    it prints the entrypoint list and exits 1 without running anything,
    #    which looks like a launch until you check the volume.
    uvx modal run --detach experiments/backtracking_steering_dsm/modal_wave2.py::main
    #    ... or one half of it at a time:
    uvx modal run --detach .../modal_wave2.py::main --skip-projected
    uvx modal run --detach .../modal_wave2.py::main --only-projected
    # 3b. re-merge shards only (the grid entrypoint already does this)
    uvx modal run experiments/backtracking_steering_dsm/modal_wave2.py::merge_cmd

The grid is sharded by PROMPT across N_SHARDS containers per source (7 sources
x 5 = 35 containers) so the wave finishes in roughly one generate-pass rather
than seven. Sharding by prompt rather than by magnitude keeps the batch
prompt-major -- one prompt's whole magnitude sweep per generate call -- which
keeps every row in a batch at the same token length and so keeps left padding
out of the projected arm's window buffer. See steer_one_w6 for why that matters.

Everything downstream of generation (judging, aggregation) is shared with wave 1,
and runs against the merged rows__<tag>.json, not the per-shard files.
"""

import json
import pathlib
import sys

import modal

try:
    ROOT = pathlib.Path(__file__).resolve().parents[2]
except IndexError:
    ROOT = pathlib.Path("/work")
sys.path.insert(0, str(ROOT))          # repo root, for the protocol import below

app = modal.App("backtracking-steering-wave2")
vol = modal.Volume.from_name("diffusion-txc", create_if_missing=True)
hf_cache = modal.Volume.from_name("sae-deadlatent-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "numpy>=2.0", "scipy", "scikit-learn==1.5.2",
                 "transformers==4.46.3", "accelerate", "sentencepiece",
                 "huggingface_hub", "hf_transfer", "pyyaml")
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(str(ROOT / "experiments" / "backtracking_steering_dsm"),
                   "/work/experiments/backtracking_steering_dsm",
                   ignore=["__pycache__"])
    .add_local_dir(str(ROOT / "experiments" / "backtracking_detection_dsm"),
                   "/work/experiments/backtracking_detection_dsm",
                   ignore=["__pycache__"])
    .add_local_dir(str(ROOT / "experiments" / "diffusion_txc" / "topk_vs_topkdiff"),
                   "/work/experiments/diffusion_txc/topk_vs_topkdiff",
                   ignore=["__pycache__"])
    .add_local_dir(str(ROOT / "experiments" / "diffusion_txc" / "psc"),
                   "/work/experiments/diffusion_txc/psc", ignore=["__pycache__"])
    .add_local_file(
        str(ROOT / "experiments" / "ward_backtracking_txc" / "b1_steer_eval.py"),
        "/work/experiments/ward_backtracking_txc/b1_steer_eval.py")
    .add_local_file(
        str(ROOT / "experiments" / "ward_backtracking_txc" / "architectures.py"),
        "/work/experiments/ward_backtracking_txc/architectures.py")
    .add_local_dir(str(ROOT / "temporal_crosscoders"), "/work/temporal_crosscoders",
                   ignore=["__pycache__"])
    .add_local_file(str(ROOT / "results" / "ward_backtracking" / "traces.json"),
                    "/work/data/traces.json")
    .add_local_file(
        str(ROOT / "results" / "ward_backtracking" / "sentence_labels.json"),
        "/work/data/sentence_labels.json")
    .add_local_file(str(ROOT / "results" / "ward_backtracking" / "prompts.json"),
                    "/work/data/prompts.json")
    .add_local_file(str(ROOT / "results" / "ward_backtracking" / "dom_vectors.pt"),
                    "/work/data/dom_vectors.pt")
)

from experiments.backtracking_steering_dsm.protocol import (  # noqa: E402
    HF_ID, LAYER, MAGNITUDES, MAX_NEW, N_EVAL, PROMPT_SEED, TRUNC_MIN_STEP,
)

FEAT_DIR = "/vol/backtracking_eval/steering/features_w6"
OUT_DIR = "/vol/backtracking_eval/steering/wave2"
SHARD_DIR = OUT_DIR + "/shards"
DISTILL_TRACES = 40        # traces forwarded through the distill for pre-flight
N_SHARDS = 5               # prompt shards per source: 7 sources x 5 = 35 containers
hf_secret = modal.Secret.from_name("hf-token")


def _shard_bounds(n: int, shard: int, n_shards: int = N_SHARDS) -> tuple[int, int]:
    """Contiguous prompt slice for `shard`, so concatenating shards 0..k-1
    reproduces the unsharded row order exactly."""
    per, rem = divmod(n, n_shards)
    lo = shard * per + min(shard, rem)
    return lo, lo + per + (1 if shard < rem else 0)


# --------------------------------------------------------------------------
# pre-flight: is the projector alive on DISTILL activations?
# --------------------------------------------------------------------------

@app.function(image=image, gpu="L40S", timeout=5400, memory=65536,
              volumes={"/vol": vol, "/hf": hf_cache}, secrets=[hf_secret])
def preflight(n_traces: int = DISTILL_TRACES, n_windows: int = 20000,
              min_step: int = 0, label: str = "final",
              arms: str = "w6") -> dict:
    """NMSE + live-latent fraction of each w6 dictionary on DeepSeek-R1-Distill
    resid_L10 windows.

    The dictionaries are trained on FineWeb through BASE Llama-3.1-8B; the
    steering site is the DISTILL model on reasoning traces. A projector that is
    half dead there would flatten the delta-gc curve for reasons that have
    nothing to do with temporal structure, so this is measured before the grid
    rather than inferred from it.
    """
    import sys
    import time

    sys.path.insert(0, "/work")
    import numpy as np
    import torch
    from experiments.backtracking_steering_dsm import w6

    t0 = time.time()
    # `min_step=0` runs the pre-flight against whatever periodic checkpoint is on
    # the volume right now -- an interim read while training is still going. The
    # gating run passes w6.TRUNC_MIN_STEP.
    vol.reload()
    arm_list = w6.W6MIX_ARMS if arms == "w6mix" else w6.W6_ARMS
    resolved = {a["name"]: w6.resolve_ckpt("/vol", a, min_step=min_step)
                for a in arm_list}
    status = {k: v[1] for k, v in resolved.items()}
    ready = {k: v[0] for k, v in resolved.items()}
    missing = [k for k, v in ready.items() if v is None]
    if missing:
        return {"error": "checkpoints not at min_step", "missing": missing,
                "min_step": min_step, "status": status}

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF_ID, use_fast=True)
    llm = AutoModelForCausalLM.from_pretrained(
        HF_ID, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    for p in llm.parameters():
        p.requires_grad_(False)

    store = {}
    h = llm.model.layers[LAYER].register_forward_hook(
        lambda _m, _i, out: store.__setitem__(
            "x", (out[0] if isinstance(out, tuple) else out).detach()))
    traces = json.loads(pathlib.Path("/work/data/traces.json").read_text())
    parts = []
    try:
        for tr in traces[:n_traces]:
            ids = tok(tr["full_response"], return_tensors="pt",
                      add_special_tokens=False, truncation=True,
                      max_length=2048)["input_ids"].to("cuda")
            with torch.no_grad():
                llm.model(input_ids=ids, use_cache=False)
            parts.append(store["x"][0].to(torch.float32).cpu())
            store.clear()
    finally:
        h.remove()
    del llm
    torch.cuda.empty_cache()

    acts = torch.cat(parts)                                   # (n_tok, d)
    T = w6.T_WINDOW
    win = acts.unfold(0, T, 1).permute(0, 2, 1).contiguous()  # (N, T, d)
    win = win.reshape(win.shape[0], -1)                       # time-major flatten
    rng = np.random.default_rng(0)
    if win.shape[0] > n_windows:
        sel = rng.choice(win.shape[0], size=n_windows, replace=False)
        win = win[torch.from_numpy(np.sort(sel))]
    win = win.to("cuda")
    print(f"[preflight] {acts.shape[0]:,} distill tokens -> "
          f"{win.shape[0]:,} windows of {win.shape[1]}", flush=True)

    out = {"n_traces": n_traces, "target_model": HF_ID, "layer": LAYER,
           "n_tokens": int(acts.shape[0]), "label": label,
           "min_step": min_step, "arm_set": arms, "arms": []}
    for a in arm_list:
        model, meta = w6.load_w6(ready[a["name"]])
        r = w6.preflight(model, meta["arch"], win)
        r["name"] = a["name"]
        r["train_final"] = w6.read_final("/vol", a)
        r["ckpt"] = status[a["name"]]
        out["arms"].append(r)
        print(f"[preflight] {a['name']:10s} arch={meta['arch']:14s} "
              f"nmse={r['nmse']:.4f} live={r['live_fraction']:.3f} "
              f"L0={r['mean_l0']:.1f}", flush=True)
        del model
        torch.cuda.empty_cache()

    # Pre-registered projector gate. Both conditions must hold: a dictionary
    # that fires ~1 latent per window can score a low NMSE by reproducing little
    # more than b_dec, so NMSE alone would pass a projector that is dead.
    g = w6.PROJECTOR_GATE
    out["projector_gate"] = {
        "thresholds": g,
        "verdict": {r["name"]: {
            "nmse": r["nmse"], "live_fraction": r["live_fraction"],
            "passes": (r["nmse"] < g["max_nmse"]
                       and r["live_fraction"] > g["min_live_fraction"])}
            for r in out["arms"]}}
    for n, v in out["projector_gate"]["verdict"].items():
        print(f"[gate] {n:10s} nmse={v['nmse']:.4f} live={v['live_fraction']:.4f}"
              f" -> {'PASS' if v['passes'] else 'FAIL'}", flush=True)

    out["_runtime_s"] = round(time.time() - t0, 1)
    d = pathlib.Path("/vol/backtracking_eval/steering")
    d.mkdir(parents=True, exist_ok=True)
    (d / f"wave2_preflight__{label}.json").write_text(json.dumps(out, indent=1))
    vol.commit()
    return out


@app.local_entrypoint()
def preflight_cmd(interim: bool = False, mix: bool = False):
    """`--interim` reads the current periodic checkpoints mid-training.
    `--mix` gates the mixed-corpus DSM dictionary as a candidate projector."""
    print(json.dumps(preflight.remote(
        min_step=0 if (interim or mix) else TRUNC_MIN_STEP,
        label="w6mix" if mix else ("interim" if interim else "final"),
        arms="w6mix" if mix else "w6"), indent=2))


# --------------------------------------------------------------------------
# mining
# --------------------------------------------------------------------------

@app.function(image=image, gpu="L4", timeout=7200, memory=65536,
              volumes={"/vol": vol, "/hf": hf_cache}, secrets=[hf_secret])
def mine_w6(min_step: int = TRUNC_MIN_STEP, feat_dir: str = FEAT_DIR) -> dict:
    """Rank each w6 arm's backtracking feature by the paper's meandiff score.

    `min_step=0` with a scratch `feat_dir` mines the current periodic
    checkpoints mid-training. That is a dry run, not a result: it exercises the
    capture -> window -> encode -> decoder path against real checkpoints so a
    bug does not surface for the first time on the critical path after the
    checkpoints truncate.
    """
    import sys
    import time

    sys.path.insert(0, "/work")
    import numpy as np
    import torch
    from experiments.backtracking_detection_dsm.detect_core import (
        capture_traces, gather_windows,
    )
    from experiments.backtracking_steering_dsm import steer_core, w6

    t0 = time.time()
    vol.reload()
    resolved = {a["name"]: w6.resolve_ckpt("/vol", a, min_step=min_step)
                for a in w6.W6_ARMS}
    status = {k: v[1] for k, v in resolved.items()}
    ready = {k: v[0] for k, v in resolved.items()}
    missing = [k for k, v in ready.items() if v is None]
    if missing:
        return {"error": "checkpoints not at min_step", "missing": missing,
                "status": status}

    cache, offsets, trace_meta, cap_meta = capture_traces(
        "/work/data/traces.json", "/work/data/sentence_labels.json")
    prompts = json.loads(pathlib.Path("/work/data/prompts.json").read_text())
    dom_qids = {p["id"] for p in prompts if p.get("split", "dom") == "dom"}
    examples = steer_core.dom_example_set(trace_meta, dom_qids)
    is_bt = np.asarray([e[2] for e in examples], dtype=bool)
    X = gather_windows(cache["resid"], offsets, examples)      # (n, T, d)

    feats, summary = [], []
    for a in w6.W6_ARMS:
        model, meta = w6.load_w6(ready[a["name"]])
        arch = meta["arch"]
        # Window latent: the trainer's own time-major flatten, then encode.
        zs = []
        for s in range(0, X.shape[0], 512):
            xb = X[s:s + 512].to("cuda", torch.float32)
            zs.append(w6.encode_w6(model, arch,
                                   xb.reshape(xb.shape[0], -1)).cpu().numpy())
        Z = np.concatenate(zs)
        pos, neg = is_bt, ~is_bt
        mu_p, mu_n = Z[pos].mean(0), Z[neg].mean(0)
        score = mu_p - mu_n
        se = np.sqrt(Z[pos].var(0, ddof=1) / pos.sum()
                     + Z[neg].var(0, ddof=1) / neg.sum() + 1e-12)
        top = np.argsort(-np.abs(score))[:8]
        decs = w6.decoder_dirs_w6(model)
        f = {"name": a["name"], "arch": arch, "hook": "resid",
             "d_sae": int(Z.shape[1]), "T": w6.T_WINDOW,
             "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
             "top_features": top.astype(np.int64),
             "scores": score[top].astype(np.float32),
             "tstat": (score / se)[top].astype(np.float32),
             "mean_pos": mu_p[top].astype(np.float32),
             "mean_neg": mu_n[top].astype(np.float32),
             "decoder_at_pos0": decs["pos0"][top].numpy().astype(np.float32),
             "decoder_union": decs["union"][top].numpy().astype(np.float32)}
        feats.append(f)
        summary.append({"arm": f["name"], "arch": arch,
                        "top_feature": int(top[0]),
                        "top_score": float(score[top[0]]),
                        "top_tstat": float((score / se)[top[0]]),
                        "top8_features": [int(x) for x in top]})
        print(f"[mined] {a['name']:10s} top_f={int(top[0]):6d} "
              f"score={float(score[top[0]]):+.5f}", flush=True)
        del model
        torch.cuda.empty_cache()

    steer_core.save_features(feats, feat_dir)
    out = {"arms": summary, "capture": cap_meta, "n_examples": len(examples),
           "n_pos": int(is_bt.sum()), "ckpt_status": status,
           "min_step": min_step, "feat_dir": feat_dir,
           "_runtime_s": round(time.time() - t0, 1)}
    pathlib.Path(feat_dir + "/mining_summary.json").write_text(
        json.dumps(out, indent=1))
    vol.commit()
    return out


@app.local_entrypoint()
def mine_cmd(interim: bool = False):
    """`--interim` dry-runs the mining path on the current periodic
    checkpoints, into a scratch feature dir."""
    print(json.dumps(mine_w6.remote(
        min_step=0 if interim else TRUNC_MIN_STEP,
        feat_dir=FEAT_DIR + "_interim" if interim else FEAT_DIR), indent=2))
    print(json.dumps(mine_w6.remote(), indent=2))


# --------------------------------------------------------------------------
# grid
# --------------------------------------------------------------------------

WINDOW_ARCHS_W6 = {"topk_sae_w6", "bayes_gate_w6"}


def _w6_sources():
    """Window dicts contribute pos0 + union; the projected variant is a
    separate source over the SAME vector as w6_dsm's pos0."""
    import sys
    sys.path.insert(0, "/work")
    import numpy as np
    import torch
    from experiments.backtracking_steering_dsm import steer_core, w6
    from experiments.ward_backtracking_txc.b1_steer_eval import _normalize_to

    ref = steer_core.dom_source("/work/data/dom_vectors.pt")
    ref_norm = ref["raw_norm"]
    srcs = []
    for a in w6.W6_ARMS:
        p = pathlib.Path(FEAT_DIR) / f"{a['name']}.npz"
        if not p.exists():
            raise FileNotFoundError(p)
        f = steer_core.load_features(p)
        fid = int(f["top_features"][0])
        for mode in ("pos0", "union"):
            key = "decoder_at_pos0" if mode == "pos0" else "decoder_union"
            vec = torch.from_numpy(np.asarray(f[key][0])).float()
            srcs.append({"tag": f"{a['name']}_f{fid}_{mode}", "arm": a["name"],
                         "arch": f["arch"], "hook": "resid", "feature_id": fid,
                         "mode": mode, "raw_norm": float(vec.norm()),
                         "projected": False,
                         "vector": _normalize_to(vec, ref_norm)})
    # Denoise-after-steer: one steered source, both projected and not.
    dsm = next(s for s in srcs
               if s["arm"] == "w6_dsm" and s["mode"] == "pos0")
    srcs.append({**dsm, "tag": dsm["tag"] + "_proj", "projected": True})
    return srcs


@app.function(image=image, gpu="L40S", timeout=14400, memory=32768,
              volumes={"/vol": vol, "/hf": hf_cache}, secrets=[hf_secret])
def steer_one_w6(args: tuple[str, int], gen_bs: int = 25) -> dict:
    """One (source, prompt-shard) cell of the wave-2 grid.

    Sharding is by PROMPT, not by magnitude, and the batch stays prompt-major
    (one prompt's whole 25-magnitude sweep per generate call). Both halves of
    that choice are load-bearing:

      - every row in a batch is then the same prompt text, so all rows have the
        same token length and the batch needs no left padding. Batching by
        magnitude instead (20 different prompts at one alpha) would left-pad,
        and the projected arm's rolling window buffer would ingest pad-token
        activations into the window for the first T-1 real tokens of every row
        -- silently corrupting the projector input for exactly the arm the
        denoise-after-steer variant exists to measure.
      - it is also the larger batch: 25 magnitudes > 20 prompts.

    The wall-clock win therefore comes from the sharding, not from re-shaping
    the batch. N_SHARDS x 7 sources = 35 containers, each doing N_EVAL/N_SHARDS
    prompts.
    """
    import sys
    import time

    sys.path.insert(0, "/work")
    import torch
    from experiments.backtracking_steering_dsm import w6
    from experiments.ward_backtracking_txc.b1_steer_eval import (
        KEYWORD_RE, _eval_prompts, _generate_panels, _kw_rate, _load_lm,
    )

    tag, shard = args
    t0 = time.time()
    src = next(s for s in _w6_sources() if s["tag"] == tag)
    out_path = pathlib.Path(SHARD_DIR) / f"rows__{tag}__s{shard}.json"
    if out_path.exists():
        return {"tag": tag, "shard": shard, "skipped": True}

    all_prompts = _eval_prompts(pathlib.Path("/work/data/prompts.json"),
                                n=N_EVAL, seed=PROMPT_SEED)
    lo, hi = _shard_bounds(len(all_prompts), shard)
    prompts = all_prompts[lo:hi]
    print(f"[shard] {tag} s{shard}: prompts[{lo}:{hi}] of {len(all_prompts)}",
          flush=True)
    model, tok = _load_lm(HF_ID, "cuda")
    chat_texts = []
    for p in prompts:
        try:
            t = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}],
                                        tokenize=False, add_generation_prompt=True)
        except Exception:
            t = p["prompt"]
        chat_texts.append(t)

    proj = proj_arch = None
    proj_status = None
    if src["projected"]:
        ck, proj_status = w6.resolve_ckpt(
            "/vol", next(a for a in w6.W6_ARMS if a["name"] == "w6_dsm"))
        if ck is None:
            raise RuntimeError(f"projector checkpoint not usable: {proj_status}")
        proj, pmeta = w6.load_w6(ck)
        proj_arch = pmeta["arch"]
        print(f"[proj] {ck} arch={proj_arch} step={proj_status['ckpt_step']}",
              flush=True)

    hook = w6.SteerDenoiseHook(src["vector"], proj=proj, proj_arch=proj_arch)
    handle = model.model.layers[LAYER].register_forward_hook(hook)
    try:
        panel_prompts = [t for t in chat_texts for _ in MAGNITUDES]
        panel_mags = [float(m) for _ in prompts for m in MAGNITUDES]
        texts = _generate_panels(model, tok, hook, prompts=panel_prompts,
                                 mags_per_prompt=panel_mags,
                                 max_new_tokens=MAX_NEW, batch_size=gen_bs)
    finally:
        handle.remove()

    rows = []
    n_mags = len(MAGNITUDES)
    for p_i, prm in enumerate(prompts):
        for m_i, mag in enumerate(MAGNITUDES):
            txt = texts[p_i * n_mags + m_i]
            rows.append({
                "target": "reasoning", "source": tag, "arm": src["arm"],
                "arch": src["arch"], "hook": src["hook"],
                "feature_id": src["feature_id"], "mode": src["mode"],
                "projected": src["projected"], "magnitude": float(mag),
                "prompt_id": prm["id"], "category": prm.get("category"),
                "keyword_rate": _kw_rate(txt),
                "wait_count": len(KEYWORD_RE.findall(txt)),
                "n_words": len(txt.split()), "n_chars": len(txt),
                "n_tokens_retok": len(tok(txt, add_special_tokens=False)["input_ids"]),
                "text": txt,
            })

    out = {"rows": rows, "meta": {
        "tag": tag, "arm": src["arm"], "arch": src["arch"], "hook": src["hook"],
        "feature_id": src["feature_id"], "mode": src["mode"],
        "projected": src["projected"],
        "raw_decoder_norm": src["raw_norm"],
        "steered_vector_norm": float(src["vector"].norm()),
        "layer": LAYER, "magnitudes": MAGNITUDES, "max_new_tokens": MAX_NEW,
        "n_eval_prompts": len(prompts), "gen_batch_size": gen_bs,
        "shard": shard, "n_shards": N_SHARDS, "prompt_slice": [lo, hi],
        "target_model": HF_ID, "decoding": "greedy (do_sample=False)",
        "projector": "w6_dsm denoise, slot T-1 replaces current position, "
                     "buffer holds pre-projection activations",
        "projector_ckpt": proj_status,
        "n_positions_projected": hook.n_projected,
        "n_positions_passthrough": hook.n_passthrough,
        "_runtime_s": round(time.time() - t0, 1)}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out))
    vol.commit()
    print(f"[done] {tag} s{shard} rows={len(rows)} proj={hook.n_projected} "
          f"in {out['meta']['_runtime_s']}s", flush=True)
    return {"tag": tag, "shard": shard, "n_rows": len(rows),
            "runtime_s": out["meta"]["_runtime_s"]}


@app.function(image=image, timeout=1800, volumes={"/vol": vol})
def merge_shards() -> dict:
    """Concatenate each source's prompt shards into one rows__<tag>.json.

    Downstream (modal_judge, analyse) keys on `rows__*.json` per source and
    indexes judgements by row position, so the shards must be merged BEFORE
    judging or one source would be read as N_SHARDS separate arms.
    """
    vol.reload()
    sd = pathlib.Path(SHARD_DIR)
    out_dir = pathlib.Path(OUT_DIR)
    tags = sorted({p.name[len("rows__"):p.name.rindex("__s")]
                   for p in sd.glob("rows__*__s*.json")})
    report = {}
    for tag in tags:
        parts = sorted(sd.glob(f"rows__{tag}__s*.json"),
                       key=lambda p: int(p.name[p.name.rindex("__s") + 3:-5]))
        if len(parts) != N_SHARDS:
            report[tag] = {"error": "incomplete",
                           "have": [p.name for p in parts]}
            continue
        objs = [json.loads(p.read_text()) for p in parts]
        rows = [r for o in objs for r in o["rows"]]
        meta = dict(objs[0]["meta"])
        meta.pop("shard", None)
        meta.pop("prompt_slice", None)
        meta["n_eval_prompts"] = len({r["prompt_id"] for r in rows})
        meta["n_positions_projected"] = sum(
            o["meta"].get("n_positions_projected", 0) for o in objs)
        meta["n_positions_passthrough"] = sum(
            o["meta"].get("n_positions_passthrough", 0) for o in objs)
        meta["shard_runtimes_s"] = [o["meta"]["_runtime_s"] for o in objs]
        meta["_runtime_s"] = max(meta["shard_runtimes_s"])
        (out_dir / f"rows__{tag}.json").write_text(
            json.dumps({"rows": rows, "meta": meta}))
        report[tag] = {"n_rows": len(rows),
                       "n_prompts": meta["n_eval_prompts"],
                       "n_magnitudes": len({r["magnitude"] for r in rows})}
        print(f"[merge] {tag}: {len(rows)} rows from {len(parts)} shards",
              flush=True)
    vol.commit()
    return report


@app.function(image=image, timeout=600, volumes={"/vol": vol})
def list_sources_w6() -> list[dict]:
    return [{k: v for k, v in s.items() if k != "vector"} for s in _w6_sources()]


@app.function(image=image, timeout=21600, volumes={"/vol": vol})
def drive_grid(only_projected: bool = False,
               skip_projected: bool = False) -> dict:
    """Fan the grid out from INSIDE Modal, then merge.

    `modal run --detach` only keeps the LAST-triggered function alive once the
    local client disconnects, and a `.map()` issued from a local entrypoint is
    not that -- so a client-side gRPC drop cancels every running shard. That
    happened twice tonight, the second time killing the grid at 8 of 35 shards
    with no error on the remote side.

    Driving the map from a container makes the driver itself the single
    triggered function, so the client can die freely: the driver keeps the
    shards alive and runs the merge when they finish. Shards are individually
    resumable, so re-running this after any failure only reruns what is
    missing.
    """
    srcs = _w6_sources()
    if skip_projected:
        srcs = [s for s in srcs if not s["projected"]]
    if only_projected:
        srcs = [s for s in srcs if s["projected"]]
    cells = [(s["tag"], i) for s in srcs for i in range(N_SHARDS)]
    print(f"[drive] {len(srcs)} sources x {N_SHARDS} shards = {len(cells)} cells",
          flush=True)
    res = list(steer_one_w6.map(cells))
    print(f"[drive] all {len(res)} cells returned; merging", flush=True)
    return {"cells": res, "merge": merge_shards.remote()}


@app.local_entrypoint()
def drive(only_projected: bool = False, skip_projected: bool = False):
    """Preferred launcher -- survives client disconnects. See drive_grid."""
    print(json.dumps(drive_grid.remote(only_projected=only_projected,
                                       skip_projected=skip_projected), indent=2))


@app.local_entrypoint()
def main(only_projected: bool = False, skip_projected: bool = False):
    """`--skip-projected` runs only the six unprojected window sources;
    `--only-projected` runs just the denoise-after-steer arm. The split exists
    because the projector pre-flight can fail its NMSE threshold while the
    unprojected arms remain perfectly runnable, and there is no reason to hold
    those behind a decision about the variant."""
    srcs = list_sources_w6.remote()
    if skip_projected:
        srcs = [s for s in srcs if not s["projected"]]
    if only_projected:
        srcs = [s for s in srcs if s["projected"]]
    cells = [(s["tag"], i) for s in srcs for i in range(N_SHARDS)]
    print(f"[wave2] {len(srcs)} sources x {N_SHARDS} prompt shards "
          f"= {len(cells)} containers")
    for s in srcs:
        print(f"   {s['tag']:44s} arm={s['arm']:10s} mode={s['mode']} "
              f"proj={s['projected']}")
    res = list(steer_one_w6.map(cells))
    print(json.dumps(res, indent=2))
    print(json.dumps(merge_shards.remote(), indent=2))


@app.local_entrypoint()
def merge_cmd():
    print(json.dumps(merge_shards.remote(), indent=2))
