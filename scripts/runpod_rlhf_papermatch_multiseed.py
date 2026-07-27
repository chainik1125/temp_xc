"""Evaluate the shipped HH-RLHF TXC checkpoint at seeds 1 and 2.

The paper-match implementation and vendored architecture are imported from the
frozen ACTMIX RLHF commit.  This wrapper only parameterizes the seed, pins the
public checkpoint revision, and writes one result per seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch


RLHF_PIN = "ed9a6c77"
HF_CKPT_REPO = "han1823123123/txcdr-base"
HF_CKPT_REVISION = "187666c5bfde80fe4ea20a64c1ed5d3092874320"
ARCH_ID = "agentic_txc_02"
DEFAULT_SEEDS = (1, 2)
SHUFFLE_SEED = 42
D_IN, D_SAE = 2304, 18432


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_pin(paper_root: Path) -> str:
    head = subprocess.run(
        ["git", "-C", str(paper_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected = subprocess.run(
        ["git", "-C", str(paper_root), "rev-parse", RLHF_PIN],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != expected:
        raise RuntimeError(f"RLHF checkout is {head}, expected {expected}")
    print(f"[pin] RLHF paper-match runner {head}", flush=True)
    return head


def stage_checkpoints(cache_root: Path, seeds: list[int]) -> None:
    from huggingface_hub import snapshot_download

    patterns = []
    for seed in seeds:
        patterns.extend(
            [
                f"ckpts/{ARCH_ID}__seed{seed}.pt",
                f"training_logs/{ARCH_ID}__seed{seed}.json",
            ]
        )
    snapshot_download(
        repo_id=HF_CKPT_REPO,
        revision=HF_CKPT_REVISION,
        allow_patterns=patterns,
        local_dir=cache_root,
    )
    expected = [
        path
        for seed in seeds
        for path in (
            cache_root / "ckpts" / f"{ARCH_ID}__seed{seed}.pt",
            cache_root / "training_logs" / f"{ARCH_ID}__seed{seed}.json",
        )
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing public RLHF artifacts: {missing}")


def _load(seed: int, cache_root: Path, device):
    meta_path = cache_root / "training_logs" / f"{ARCH_ID}__seed{seed}.json"
    checkpoint = cache_root / "ckpts" / f"{ARCH_ID}__seed{seed}.pt"
    meta = json.loads(meta_path.read_text())
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    state = {
        key: (
            value.float()
            if torch.is_tensor(value) and value.dtype == torch.float16
            else value
        )
        for key, value in state.items()
    }
    source_class = meta["src_class"]
    if source_class != "MatryoshkaTXCDRContrastiveMultiscale":
        raise ValueError(
            f"{ARCH_ID} seed {seed} has unexpected source class {source_class}"
        )
    from experiments.explorations.actmix_rlhf.vendor.\
        matryoshka_txcdr_contrastive_multiscale import (
            MatryoshkaTXCDRContrastiveMultiscale,
        )

    model = MatryoshkaTXCDRContrastiveMultiscale(
        D_IN,
        D_SAE,
        T=int(meta["T"]),
        k=int(meta["k_win"]),
        n_contr_scales=int(meta.get("n_scales", 3)),
        gamma=float(meta.get("gamma", 0.5)),
    ).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    return model, meta, {
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_revision": HF_CKPT_REVISION,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


@torch.no_grad()
def evaluate_seed(
    seed: int,
    cache_root: Path,
    eval_cache: Path,
    output_root: Path,
    code_pin: str,
) -> dict:
    from experiments.explorations.actmix_rlhf import decomp

    chosen = np.load(eval_cache / "chosen.npz")
    rejected = np.load(eval_cache / "rejected.npz")
    chosen_acts, rejected_acts = chosen["acts"], rejected["acts"]
    chosen_mask, rejected_mask = chosen["response_mask"], rejected["response_mask"]
    chosen_len = chosen["response_len"].astype(np.float64)
    rejected_len = rejected["response_len"].astype(np.float64)
    valid = (chosen_len > 0) & (rejected_len > 0)

    device = torch.device("cuda")
    model, meta, provenance = _load(seed, cache_root, device)
    window = int(meta["T"])
    encode = lambda value: model.encode(value)
    variants = {}
    started = time.time()
    for tag, shuffle_seed in (("plain", None), ("shuffled", SHUFFLE_SEED)):
        chosen_pe, chosen_l0 = decomp.aggregate_response_mean(
            encode,
            chosen_acts,
            chosen_mask,
            T=window,
            d_sae=D_SAE,
            device=device,
            shuffle_seed=shuffle_seed,
        )
        rejected_pe, rejected_l0 = decomp.aggregate_response_mean(
            encode,
            rejected_acts,
            rejected_mask,
            T=window,
            d_sae=D_SAE,
            device=device,
            shuffle_seed=shuffle_seed,
        )
        variants[tag] = {
            "preference_auc": decomp.preference_auc(
                chosen_pe, rejected_pe, valid
            ),
            "preference_auc_k50": decomp.preference_auc(
                chosen_pe, rejected_pe, valid, k=50
            ),
            "mass_at_20": decomp.mass_at_k(chosen_pe, rejected_pe, valid),
            "length_pearson": decomp.length_pearson_topk(
                chosen_pe,
                rejected_pe,
                chosen_len,
                rejected_len,
                valid,
            ),
            "realized_l0": {"chosen": chosen_l0, "rejected": rejected_l0},
        }
        print(
            f"[seed {seed} {tag}] "
            f"auc={variants[tag]['preference_auc']['auc_mean']:.6f}",
            flush=True,
        )

    payload = {
        "status": "complete",
        "protocol": "ACTMIX RLHF papermatch v1; seed-parameterized eval-only",
        "code_pin": code_pin,
        "checkpoint_repo": HF_CKPT_REPO,
        "checkpoint_revision": HF_CKPT_REVISION,
        "arch_id": ARCH_ID,
        "seed": seed,
        "meta": {
            key: meta.get(key)
            for key in (
                "arch_id",
                "src_class",
                "d_sae",
                "k_pos",
                "k_win",
                "T",
                "group_sizes",
            )
        },
        "provenance": provenance,
        "cache_meta": json.loads((eval_cache / "meta.json").read_text()),
        "variants": variants,
        "shuffle_gap_auc": (
            variants["plain"]["preference_auc"]["auc_mean"]
            - variants["shuffled"]["preference_auc"]["auc_mean"]
        ),
        "elapsed_seconds": round(time.time() - started),
        "gpu": torch.cuda.get_device_name(0),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / f"{ARCH_ID}__seed{seed}.json"
    destination.write_text(json.dumps(payload, indent=2))
    print(f"[complete] seed {seed} -> {destination}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paper-root", type=Path, default=Path("/workspace/rlhf-paper")
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("/workspace/caches/rlhf/txcdr-base"),
    )
    parser.add_argument(
        "--eval-cache",
        type=Path,
        default=Path("/workspace/caches/rlhf/cached_hh_rlhf"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/workspace/reviewer_multiseed/rlhf"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    args = parser.parse_args()

    code_pin = _assert_pin(args.paper_root)
    for required in ("chosen.npz", "rejected.npz", "meta.json"):
        if not (args.eval_cache / required).exists():
            raise FileNotFoundError(args.eval_cache / required)
    stage_checkpoints(args.checkpoint_root, args.seeds)
    for seed in args.seeds:
        result = args.output_root / f"{ARCH_ID}__seed{seed}.json"
        if result.exists() and json.loads(result.read_text()).get("status") == "complete":
            print(f"[resume] seed {seed} result already complete", flush=True)
            continue
        evaluate_seed(
            seed,
            args.checkpoint_root,
            args.eval_cache,
            args.output_root,
            code_pin,
        )


if __name__ == "__main__":
    main()
