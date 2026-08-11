"""Feature mining and steering-source construction for the c7 steering replication.

Reused, not reimplemented (the point of the exercise is that every arm goes
through the paper's own protocol):

  - activation capture, sentence->token anchor alignment and the [-13,-8]
    window: backtracking_detection_dsm/detect_core.py::capture_traces /
    ::collect_examples / ::gather_windows, which are themselves lifted from
    ward_backtracking_txc/mine_features.py::_capture_windows.
  - window-level latent and decoder directions: ward_backtracking_txc
    ::architectures.arch_encode_window / arch_decoder_directions.
  - selectivity ranking: mine_features.py -- score(f) = mean(z[f] | D+) -
    mean(z[f] | D-), top-K by |score|.
  - steering hook, batched panel generation, eval-prompt split and the
    norm rescale to the DoM-base-union norm: b1_steer_eval.py::_Hook,
    ::_generate_panels, ::_load_lm, ::_eval_prompts, ::_normalize_to,
    imported unmodified.

The one place we extend the paper's dispatch is the flat per-token TopK SAEs
trained in this workstream ("topk_sae_flat"). They get exactly the treatment
architectures.py::arch_forward already gives the Stage B `topk_sae` arm --
flatten the T-window into the batch axis, encode each token independently,
mean the per-position codes into one window latent -- so the ranking is
computed on the same quantity for both flat families.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.backtracking_steering_dsm.protocol import (  # noqa: F401
    D_MODEL, HF_ID, LAYER, MAGNITUDES, MAX_NEW, N_EVAL, PROMPT_SEED, T_WINDOW,
)

STAGEB_REPO = "aniketdesh/ward-stage-b-dictionaries"
OURS_REPO = "dmanningcoe/diffusion-topk-saes"

# Wave 1: checkpoints that already exist. `hook` is the activation the
# dictionary was TRAINED on; steering always happens at the residual stream of
# layer 10 (b1_steer_eval hooks model.model.layers[10] for every source
# regardless of the dictionary's own hookpoint -- that is the paper's choice
# and is replicated, not corrected).
WAVE1_ARMS = [
    {"name": "stageB_topk_sae", "kind": "stageb",
     "cell": "topk_sae__ln1_L10__k64__s42", "hook": "ln1"},
    {"name": "stageB_txc", "kind": "stageb",
     "cell": "txc__resid_L10__k16__s42", "hook": "resid"},
    {"name": "stageB_txc_h13", "kind": "stageb",
     "cell": "txc_h13__resid_L10__k16__s42", "hook": "resid"},
    {"name": "ours_recon_s2", "kind": "ours",
     "file": "llama31-8b-ln1L10-20k/recon_s2.pt", "hook": "ln1"},
    {"name": "ours_dsm_s2", "kind": "ours",
     "file": "llama31-8b-ln1L10-20k/dsm_s2.pt", "hook": "ln1"},
]

WINDOW_ARCHS = {"txc", "txc_h8", "txc_h13", "stacked_sae"}


# --------------------------------------------------------------------------
# dictionaries
# --------------------------------------------------------------------------

def load_arm(arm: dict, device: str = "cuda"):
    """-> (model, meta). Stage B checkpoints carry their own architecture
    config; the flat arms are a bare state_dict of topk_vs_topkdiff::TopKSAE."""
    from experiments.backtracking_detection_dsm.detect_core import (
        load_new_dict, load_stageb_dict,
    )
    if arm["kind"] == "stageb":
        model, meta = load_stageb_dict(arm["cell"], STAGEB_REPO, device=device)
        assert meta["hook_component"] == arm["hook"], (
            f"{arm['name']}: checkpoint says hookpoint {meta['hook_component']}, "
            f"registry says {arm['hook']}")
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(OURS_REPO, arm["file"])
        model, meta = load_new_dict(path, device=device)
    meta = dict(meta)
    meta["name"] = arm["name"]
    meta["hook"] = arm["hook"]
    return model, meta


@torch.no_grad()
def window_latents(model, arch: str, X: torch.Tensor, device: str = "cuda",
                   batch: int = 512) -> np.ndarray:
    """(n, T, d) -> (n, d_sae) SIGNED window-level latent.

    Signed, not |z|: mine_features.py ranks D+/D- on the raw window latent and
    takes |score| only at the ranking step, so taking the absolute value here
    would change which features are selected.
    """
    from experiments.ward_backtracking_txc.architectures import arch_encode_window
    outs = []
    for s in range(0, X.shape[0], batch):
        x = X[s:s + batch].to(device, torch.float32)
        if arch == "topk_sae_flat":
            B, T, d = x.shape
            z = model.encode(x.reshape(B * T, d)).reshape(B, T, -1).mean(dim=1)
        else:
            z = arch_encode_window(arch, model, x)
        outs.append(z.float().cpu().numpy())
    return np.concatenate(outs)


@torch.no_grad()
def decoder_directions(model, arch: str) -> dict:
    """-> {"pos0": (d_sae, d), "union": (d_sae, d)} on CPU float32."""
    if arch == "topk_sae_flat":
        W = model.W_dec.detach().float().cpu()          # (H, d)
        return {"pos0": W, "union": W}
    from experiments.ward_backtracking_txc.architectures import (
        arch_decoder_directions,
    )
    d = arch_decoder_directions(arch, model)
    return {"pos0": d["pos0"].detach().float().cpu(),
            "union": d["union"].detach().float().cpu()}


# --------------------------------------------------------------------------
# mining
# --------------------------------------------------------------------------

def dom_example_set(trace_meta, dom_qids: set[str]):
    """The paper's D+/D- sentence split, restricted to the DoM prompt split.

    mine_features.py mines on `split == "dom"` questions only; the 20 eval
    prompts are held out. detect_core.collect_examples does not filter, so we
    filter its output by trace index (which keeps the indices valid against the
    unfiltered `offsets` array that gather_windows reads).
    """
    from experiments.backtracking_detection_dsm.detect_core import collect_examples
    ex = collect_examples(trace_meta)["sentence"]
    keep = [e for e in ex if trace_meta[e[0]]["qid"] in dom_qids]
    return keep


def mine_arm(model, meta: dict, X: torch.Tensor, is_bt: np.ndarray,
             top_k: int = 8, device: str = "cuda") -> dict:
    """Rank features by D+/D- selectivity; return top-K ids, scores, decoders."""
    arch = meta["arch"]
    Z = window_latents(model, arch, X, device=device)          # (n, d_sae)
    pos, neg = is_bt.astype(bool), ~is_bt.astype(bool)
    mu_pos, mu_neg = Z[pos].mean(0), Z[neg].mean(0)
    score = mu_pos - mu_neg
    # Welch t kept alongside as a diagnostic only; selection is the paper's
    # meandiff ranking (config.yaml mining has no `rank` override for c7).
    se = np.sqrt(Z[pos].var(0, ddof=1) / max(pos.sum(), 1)
                 + Z[neg].var(0, ddof=1) / max(neg.sum(), 1) + 1e-12)
    tstat = score / se
    top = np.argsort(-np.abs(score))[:top_k]

    decs = decoder_directions(model, arch)
    out = {
        "name": meta["name"], "arch": arch, "hook": meta["hook"],
        "d_sae": int(Z.shape[1]), "T": int(meta.get("T", 1)),
        "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
        "top_features": top.astype(np.int64),
        "scores": score[top].astype(np.float32),
        "tstat": tstat[top].astype(np.float32),
        "mean_pos": mu_pos[top].astype(np.float32),
        "mean_neg": mu_neg[top].astype(np.float32),
        "decoder_at_pos0": decs["pos0"][top].numpy().astype(np.float32),
        "decoder_union": decs["union"][top].numpy().astype(np.float32),
    }
    return out


# --------------------------------------------------------------------------
# steering sources
# --------------------------------------------------------------------------

def sources_for_arm(feat: dict, ref_norm: float, rank: int = 0) -> list[dict]:
    """Build the b1_steer_eval source entries for one arm's chosen feature.

    Modes follow _build_sources: window dictionaries contribute both the
    pos0 decoder row and the T-averaged union row; flat per-token SAEs
    contribute one vector (pos0 == union, so the union source would be a
    duplicate and _build_sources skips it).
    """
    from experiments.ward_backtracking_txc.b1_steer_eval import _normalize_to
    fid = int(feat["top_features"][rank])
    is_window = feat["arch"] in WINDOW_ARCHS
    modes = ["pos0", "union"] if is_window else ["pos0"]
    key = {"pos0": "decoder_at_pos0", "union": "decoder_union"}
    out = []
    for mode in modes:
        vec = torch.from_numpy(np.asarray(feat[key[mode]][rank])).float()
        out.append({
            "tag": f"{feat['name']}_f{fid}_{mode}",
            "arm": feat["name"], "arch": feat["arch"], "hook": feat["hook"],
            "feature_id": fid, "mode": mode,
            "raw_norm": float(vec.norm()),
            "vector": _normalize_to(vec, ref_norm),
        })
    return out


def dom_source(dom_path: str | Path) -> dict:
    """The conventional-steering baseline b1_steer_eval always includes."""
    dom = torch.load(Path(dom_path), map_location="cpu", weights_only=False)
    v = dom["base"]["union"].clone().float()
    return {"tag": "dom_base_union", "arm": "dom", "arch": "dom", "hook": "resid",
            "feature_id": -1, "mode": "dom", "raw_norm": float(v.norm()),
            "vector": v}


def random_source(ref_norm: float, d_model: int = D_MODEL, seed: int = 0) -> dict:
    """Norm-matched random direction: the nonspecific-perturbation control.

    Every real source is rescaled to the DoM base-union norm before steering, so
    the magnitude axis means the same thing for all of them. A random unit
    vector put through the same rescale therefore isolates whatever part of the
    genuine-event response comes from perturbing the layer-10 residual stream by
    that much *at all*, independent of direction.

    It exists because several arms raise gc in BOTH steering directions. That
    U-shape in |alpha| is either a nonspecific norm-perturbation effect, in
    which case this control reproduces it and every arm should be read net of
    it, or it is direction-specific, in which case this control stays flat and
    the mined signs carry meaning. Without the control the ambiguity sits over
    every comparison in the table.

    Seed 0 reproduces the vector modal_gates.py used for the hook no-op gate.
    """
    from experiments.ward_backtracking_txc.b1_steer_eval import _normalize_to
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(d_model, generator=g)
    return {"tag": "control_random", "arm": "control", "arch": "random",
            "hook": "resid", "feature_id": -2, "mode": "random",
            "raw_norm": float(v.norm()), "seed": seed,
            "vector": _normalize_to(v, ref_norm)}


def save_features(feats: list[dict], out_dir: str | Path) -> list[str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for f in feats:
        p = out_dir / f"{f['name']}.npz"
        np.savez(p, **{k: v for k, v in f.items()
                       if isinstance(v, (np.ndarray, np.generic))},
                 meta=json.dumps({k: v for k, v in f.items()
                                  if not isinstance(v, (np.ndarray, np.generic))}))
        written.append(str(p))
    return written


def load_features(path: str | Path) -> dict:
    z = np.load(path, allow_pickle=True)
    out = {k: z[k] for k in z.files if k != "meta"}
    out.update(json.loads(str(z["meta"])))
    return out
