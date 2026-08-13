"""Detection-eval round for one trained checkpoint (arm x seed).

Four evals, SAEBench-style but scaled (deviations documented in README):

- sparse_probing: concept datasets cached by evals_cache.py; per-sequence
  latent means; top-k latents by class-mean-diff; logistic probe.
- absorption_lite: first-letter (26-class) task on the eval shard's own
  tokens; per letter, "main" latents = top-2 by mean-diff; absorption rate
  = P(main latents inactive | full probe correct), meaned over letters.
- fragility: encode clean vs perturbed (eps * rms Gaussian) activations;
  TopK support Jaccard, plus first-letter probe prediction flip rate.
- autointerp_judge: detection protocol on the saved top-context dumps via
  the Anthropic API (skipped gracefully if no key in env).
"""

from __future__ import annotations

import json
import pathlib
import string

import numpy as np
import torch

from experiments.diffusion_txc.topk_vs_topkdiff.sae import TopKSAE


def _load_sae(vol: pathlib.Path, arm: str, seed: int, d: int, dev,
              ckpt_dir: str = "logs") -> TopKSAE:
    sae = TopKSAE(d=d, H=16384, k=40).to(dev)
    sd = torch.load(vol / ckpt_dir / f"{arm}_s{seed}_final.pt",
                    weights_only=True, map_location=dev)
    sae.load_state_dict(sd)
    sae.eval()
    return sae


def _encode_batched(sae, X, dev, bs=8192):
    out = []
    with torch.no_grad():
        for b0 in range(0, X.shape[0], bs):
            out.append(sae.encode(X[b0 : b0 + bs].to(dev, torch.float32)).cpu())
    return torch.cat(out)


def sparse_probing(sae, cache: pathlib.Path, dev, ks=(1, 5, 20, -1)) -> dict:
    from sklearn.linear_model import LogisticRegression

    res = {}
    for ds_file in sorted(cache.glob("concept_*.pt")):
        d = torch.load(ds_file, weights_only=True)
        name = ds_file.stem.replace("concept_", "")
        accs = {}
        for split in ("train", "test"):
            A = d[f"acts_{split}"]                          # (n, T, dm) bf16
            n, T, dm = A.shape
            Z = _encode_batched(sae, A.reshape(-1, dm), dev).reshape(n, T, -1)
            d[f"z_{split}"] = Z.mean(dim=1).numpy()          # latent means
        ytr, yte = d["y_train"].numpy(), d["y_test"].numpy()
        Ztr, Zte = d["z_train"], d["z_test"]
        classes = np.unique(ytr)
        # rank latents by max class-mean difference (one-vs-rest)
        diffs = np.stack([
            np.abs(Ztr[ytr == c].mean(0) - Ztr[ytr != c].mean(0))
            for c in classes
        ]).max(0)
        order = np.argsort(-diffs)
        for k in ks:
            idx = order if k == -1 else order[:k]
            clf = LogisticRegression(max_iter=2000)
            clf.fit(Ztr[:, idx], ytr)
            accs[f"k={'all' if k == -1 else k}"] = float(
                clf.score(Zte[:, idx], yte))
        res[name] = accs
    return res


def _first_letter_labels(tokens: torch.Tensor, tok) -> tuple[np.ndarray, np.ndarray]:
    """(mask, labels 0-25) for tokens whose stripped text starts with a letter."""
    vocab_ids = torch.unique(tokens)
    lut = {}
    for vid in vocab_ids.tolist():
        s = tok.decode([vid]).strip()
        lut[vid] = string.ascii_lowercase.index(s[0].lower()) if s and s[0].isalpha() and s[0].lower() in string.ascii_lowercase else -1
    lab = np.array([lut[int(t)] for t in tokens])
    return lab >= 0, lab


def absorption_and_fragility(sae, vol: pathlib.Path, tok, dev, rms: float,
                             eps_list=(0.05, 0.1, 0.25, 0.5), n_rows=60000) -> tuple[dict, dict]:
    from sklearn.linear_model import LogisticRegression

    ev = torch.load(vol / "cache" / "eval_shard.pt", weights_only=True)
    X = ev["acts"][:n_rows]
    tokens = ev["tokens"][:n_rows]
    mask, lab = _first_letter_labels(tokens, tok)
    Z = _encode_batched(sae, X, dev).numpy()
    Zl, yl = Z[mask], lab[mask]
    n_tr = int(0.8 * len(yl))
    clf = LogisticRegression(max_iter=1500)
    clf.fit(Zl[:n_tr], yl[:n_tr])
    pred = clf.predict(Zl[n_tr:])
    acc = float((pred == yl[n_tr:]).mean())

    # absorption-lite: per letter, main latents = top-2 by mean-diff on train
    absorb_rates = []
    for c in range(26):
        tr_c = yl[:n_tr] == c
        if tr_c.sum() < 30 or (yl[n_tr:] == c).sum() < 15:
            continue
        diff = np.abs(Zl[:n_tr][tr_c].mean(0) - Zl[:n_tr][~tr_c].mean(0))
        main = np.argsort(-diff)[:2]
        te_c = (yl[n_tr:] == c) & (pred == c)               # probe-correct
        if te_c.sum() == 0:
            continue
        main_off = (Zl[n_tr:][te_c][:, main] <= 0).all(axis=1)
        absorb_rates.append(float(main_off.mean()))
    absorption = {"first_letter_probe_acc": acc,
                  "absorption_rate_mean": float(np.mean(absorb_rates)),
                  "n_letters": len(absorb_rates)}

    # fragility: support Jaccard + probe flip under eps*rms noise
    frag = {}
    Xg = X[:20000]
    Zc = _encode_batched(sae, Xg, dev)
    sup_c = Zc > 0
    m2, l2 = mask[:20000], lab[:20000]
    base_pred = clf.predict(Zc.numpy()[m2])
    for eps in eps_list:
        torch.manual_seed(0)
        Xp = Xg.float() + eps * rms * torch.randn(Xg.shape)
        Zp = _encode_batched(sae, Xp, dev)
        sup_p = Zp > 0
        inter = (sup_c & sup_p).sum(1).float()
        union = (sup_c | sup_p).sum(1).float().clamp_min(1)
        pert_pred = clf.predict(Zp.numpy()[m2])
        frag[f"eps={eps}"] = {
            "support_jaccard": float((inter / union).mean()),
            "probe_flip_rate": float((pert_pred != base_pred).mean()),
        }
    return absorption, frag


def autointerp_judge(vol: pathlib.Path, arm: str, seed: int,
                     n_latents: int = 48, ckpt_dir: str = "logs") -> dict:
    import os

    key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY_MATS")
    if not key:
        return {"skipped": "no anthropic key in env"}
    # Training runs that don't emit top-context dumps (e.g. psc_train_sae.py)
    # leave this file absent; the other evals still stand on their own.
    dump_f = vol / ckpt_dir / f"{arm}_s{seed}_autointerp.json"
    if not dump_f.exists():
        return {"skipped": f"no autointerp dump at {dump_f}"}
    dump = json.loads(dump_f.read_text())
    rng = np.random.default_rng(0)
    latents = rng.permutation(list(dump))[:n_latents]
    all_ctxs = [ex["ctx"] for exs in dump.values() for ex in exs]

    def call(prompt: str) -> str:
        body = json.dumps({
            "model": "claude-haiku-4-5-20251001", "max_tokens": 300,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=body,
            headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["content"][0]["text"]

    import re

    scores = []
    for lat in latents:
        try:
            exs = dump[lat]
            top, held = exs[:6], exs[6:10]
            if len(held) < 4:
                continue
            rand_ctxs = [all_ctxs[i] for i in rng.choice(len(all_ctxs), 4)]
            expl = call(
                "These text snippets all strongly activate one feature of a "
                "sparse autoencoder (activating token marked by its position at "
                "the end of context). Write a ONE-LINE explanation of the "
                "feature.\n\n" + "\n---\n".join(e["ctx"][-300:] for e in top))
            test = held + [{"ctx": c} for c in rand_ctxs]
            order = rng.permutation(len(test))
            listing = "\n".join(f"[{i}] {test[j]['ctx'][-300:]}" for i, j in enumerate(order))
            verdict = call(
                f"Feature explanation: {expl}\nWhich of these 8 snippets activate "
                "the feature? Reply with ONLY the bracket numbers, comma-separated."
                f"\n\n{listing}")
            truth = {i for i, j in enumerate(order) if j < len(held)}
            said = {int(m) for m in re.findall(r"\d+", verdict) if int(m) < 8}
            tp = len(truth & said)
            fp = min(len(said - truth), 4)
            scores.append(0.5 * (tp / 4 + (4 - fp) / 4))
        except Exception as e:                                # noqa: BLE001
            print(f"judge skip latent {lat}: {e}", flush=True)
    return {"n_judged": len(scores),
            "detection_balanced_acc_mean": float(np.mean(scores)),
            "detection_balanced_acc_se": float(np.std(scores) / max(1, len(scores)) ** 0.5)}


def run(device: str = "cuda", vol: str = "/vol", arm: str = "recon",
        seed: int = 0, ckpt_dir: str = "logs") -> dict:
    from transformers import AutoTokenizer

    dev = torch.device(device)
    volp = pathlib.Path(vol)
    meta = json.loads((volp / "cache" / "meta.json").read_text())
    tok = AutoTokenizer.from_pretrained(meta["model"])
    sae = _load_sae(volp, arm, seed, meta["d"], dev, ckpt_dir)

    out = {"experiment": "topkdiff_evals", "arm": arm, "seed": seed,
           "ckpt_dir": ckpt_dir}
    out["sparse_probing"] = sparse_probing(sae, volp / "concepts", dev)
    out["absorption"], out["fragility"] = absorption_and_fragility(
        sae, volp, tok, dev, rms=meta["rms"])
    out["autointerp"] = autointerp_judge(volp, arm, seed, ckpt_dir=ckpt_dir)
    return out
