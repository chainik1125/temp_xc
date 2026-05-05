"""C5 detection axis driver — adds the detection axis to C5
(complementary to steering, per agent_paper 2026-05-05 PM).

Re-uses each (arch, seed) v1.1.0 trained checkpoint via cache-hit:
no re-training. The cell pipeline:

  1. Find the steering eval_key for this (arch, seed) via the
     canonical training_cfg per arch family (decisions § 15 + § 16).
  2. Load grades + generations from that run_dir, pair by ``idx``,
     drop nothing (C5's strength grid has no mag=0 cell), label
     ``B = 1`` if ``success_grade ≥ 2 AND coh_grade ≥ 2.0``.
  3. Forward each continuation through Gemma-2-2b-IT, hook L13
     residuals → variable-length (T_cont, d_in) per continuation.
     Drop the prompt tokens; only the generated continuation feeds
     detection.
  4. Build a fixed (n_cont, T_eval, d_in) cohort tensor: last
     ``T_eval`` tokens of each continuation, where ``T_eval =
     arch.T`` for window archs (txc_base T=5, txc_pro T=10) or a
     pooled-eval default (T_eval=16) for per-token archs (T=1
     models — tsae_paper / topk_sae / tfa).
  5. ``detect_case_study(arch, sentence_acts, labels, qids,
     shuffle_seed=42)`` with GroupKFold by concept_id (30 groups
     across 270 continuations per cell, 5 folds).
  6. Append a v1.2.0 leaderboard row with ``metric_set='detection'``
     and ``steering_eval_key=<source>`` for traceback. Persist
     metrics.json under ``results/runs/<eval_key>/``.

Source: agent_paper's c5 detection mission section in
``agents/agent_steer/briefing.md`` and the cross-component design
doc ``docs/cross_component/det_steer_detection.md`` § C5.

15 cells (5 archs × 3 seeds) total. With Gemma loaded once + the
forward pass on 270 × 60 tokens per cell being negligible, all 15
land in ~10-15 min on a single A40.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

os.environ.setdefault("TQDM_DISABLE", "1")

from temp_bench.cache import (
    append_leaderboard,
    load_checkpoint_state_dict,
    run_dir,
    save_metrics,
)
from temp_bench.config import (
    compute_act_cache_key,
    compute_eval_key,
    compute_train_key,
    instantiate_arch,
    load_arch,
    load_datasource,
)
from temp_bench.eval.detection import detect_case_study
from temp_bench.report import query_leaderboard
from temp_bench.schemas import LeaderboardRow, TrainingConfig
from temp_bench.utils.tokens import require_token

log = logging.getLogger("c5_detection")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")


COMPONENT = "c5"
DATASOURCE = "gemma_2_2b_it_l13_fineweb_24k128"
SUBJECT_MODEL = "google/gemma-2-2b-it"
ANCHOR_LAYER = 13

# Bumped from steering's "1.1.0" — detection cells produce a different
# metric shape (pr_auc_S* / shuffle_gap_S*) and would mix poorly with
# steering's peak_success_grade_at_coh_* in any naïve aggregation.
DETECTION_PROTO = "1.2.0"

SUCCESS_GRADE_MIN = 2          # success ≥ 2 to count as B=1
COH_GRADE_MIN = 2.0            # coh ≥ 2.0 in the spec (briefing line 110)
T_EVAL_PER_TOKEN = 16          # last 16 tokens for per-token archs (T=1)


def _canonical_training_cfg(arch_name: str) -> TrainingConfig:
    """Per-arch canonical training_cfg (decisions § 15 + § 16)."""
    if arch_name == "tsae_paper":
        return TrainingConfig(n_steps=20_000, train_window_size=2)
    if arch_name == "topk_sae":
        return TrainingConfig(n_steps=20_000, train_window_size=1)
    if arch_name == "tfa":
        return TrainingConfig(n_steps=20_000, batch_size=32)
    # txc_base / txc_pro: default 20k, no train_window_size
    return TrainingConfig(n_steps=20_000)


def _load_pairs(eval_key_steer: str) -> list[dict]:
    """Pair grades.jsonl + generations.jsonl by ``idx`` and label B.

    Drops only rows missing either head OR a generation (nothing in
    practice — C5's strength grid has no mag=0 cell). Returns one
    record per labeled continuation.
    """
    rd = run_dir(eval_key_steer)
    grades_path = rd / "grades.jsonl"
    judge_path = rd / "judge_outputs.jsonl"
    gens_path = rd / "generations.jsonl"

    if grades_path.exists():
        grades = [
            json.loads(l) for l in grades_path.read_text().splitlines() if l.strip()
        ]
    elif judge_path.exists():
        per_idx: dict[int, dict] = defaultdict(dict)
        for line in judge_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("label") is None:
                continue
            head = r.get("head")
            if head not in ("success", "coherence"):
                continue
            idx = int(r["idx"])
            per_idx[idx]["idx"] = idx
            per_idx[idx]["strength"] = float(r["strength"])
            per_idx[idx]["concept_id"] = r["concept_id"]
            if head == "success":
                per_idx[idx]["success_grade"] = float(r["label"])
            else:
                per_idx[idx]["coherence_grade"] = float(r["label"])
        grades = [
            v for v in per_idx.values()
            if "success_grade" in v and "coherence_grade" in v
        ]
    else:
        raise FileNotFoundError(f"no grades.jsonl or judge_outputs.jsonl in {rd}")

    gens = {
        int(json.loads(l)["idx"]): json.loads(l)
        for l in gens_path.read_text().splitlines() if l.strip()
    }

    pairs: list[dict] = []
    for g in grades:
        idx = int(g["idx"])
        gen = gens.get(idx)
        if gen is None:
            continue
        # Some judge calls fail (parse errors, API outages, etc.) →
        # success_grade / coherence_grade can be None. Drop those rows
        # from the cohort — they have no label.
        if g.get("success_grade") is None or g.get("coherence_grade") is None:
            continue
        sg = float(g["success_grade"])
        cg = float(g["coherence_grade"])
        label = 1 if (sg >= SUCCESS_GRADE_MIN and cg >= COH_GRADE_MIN) else 0
        pairs.append({
            "idx": idx,
            "concept_id": gen["concept_id"],
            "feature_idx": int(gen.get("feature_idx", -1)),
            "continuation": gen["generated_text"],
            "prompt": gen["prompt"],
            "label": label,
            "strength": float(gen["strength"]),
        })
    return pairs


def _forward_residuals(
    pairs: list[dict],
    model,
    tokenizer,
    *,
    layer: int = ANCHOR_LAYER,
    device: str = "cuda",
) -> list[np.ndarray]:
    """Tokenize prompt+continuation, hook L<layer> resid_post, capture
    activations. Drop prompt tokens; return per-pair (T_cont, d_in)
    arrays (variable T_cont).
    """
    captured: dict[str, torch.Tensor] = {}

    def hook_fn(_m, _i, output):
        x = output[0] if isinstance(output, tuple) else output
        captured["x"] = x.detach().to(torch.float32).cpu()

    handle = model.model.layers[layer].register_forward_hook(hook_fn)
    residuals: list[np.ndarray] = []
    try:
        with torch.no_grad():
            for p in pairs:
                full_text = p["prompt"] + p["continuation"]
                enc = tokenizer(
                    full_text, return_tensors="pt", add_special_tokens=True,
                )
                input_ids = enc["input_ids"].to(device)
                model(input_ids)
                resid = captured["x"][0].numpy()       # (T_full, d_in)
                # Drop prompt tokens. The prompt-only tokenization may
                # add a BOS exactly the same way; its length determines
                # how many leading positions to drop.
                prompt_only = tokenizer(
                    p["prompt"], add_special_tokens=True,
                )["input_ids"]
                prompt_len = len(prompt_only)
                cont_resid = resid[prompt_len:]        # (T_cont, d_in)
                residuals.append(cont_resid.astype(np.float32, copy=False))
    finally:
        handle.remove()
    return residuals


def _build_cohort(
    residuals: list[np.ndarray], arch_T: int,
) -> np.ndarray:
    """Build ``(n_cont, T_eval, d_in)`` cohort matrix.

    For window archs (T_arch ≥ 2): take the LAST ``T_arch`` tokens of
    each continuation. Single window per cohort element matches
    detect_case_study's contract; "last" matches C7's convention
    (steered sentiment manifests progressively, end-of-continuation
    is most reliable).

    For per-token archs (T_arch = 1, including TFA whose encoder is
    position-wise even though it attends): take the last
    ``T_EVAL_PER_TOKEN=16`` tokens — the encode is position-wise, so
    this multi-position window pools max-fire detection signal across
    a small region without conflating semantically-distant positions.
    """
    if arch_T <= 1:
        T_eval = T_EVAL_PER_TOKEN
    else:
        T_eval = arch_T

    d_in = residuals[0].shape[1]
    n = len(residuals)
    sa = np.zeros((n, T_eval, d_in), dtype=np.float32)
    for i, r in enumerate(residuals):
        if r.shape[0] >= T_eval:
            sa[i] = r[-T_eval:]                          # last T_eval tokens
        elif r.shape[0] > 0:
            sa[i, -r.shape[0]:] = r                      # right-align, pad-left zeros
        # else: stays all-zero (Gemma stopped immediately — rare)
    return sa


def _find_steering_eval_key(
    arch_name: str, seed: int,
) -> tuple[str, str, str]:
    """Resolve (train_key, steering_eval_key, datasource_act_cache_key)
    for the canonical v1.1.0 c5 cell of (arch_name, seed).

    Raises if no canonical row exists (run agent_steer's c5 sweep first).
    """
    spec = load_arch(arch_name, component=COMPONENT)
    datasource = load_datasource(DATASOURCE)
    ack = compute_act_cache_key(datasource)
    cfg = _canonical_training_cfg(arch_name)
    train_key = compute_train_key(
        arch=spec, seed=seed, training_cfg=cfg, act_cache_key=ack,
    )
    matches = [
        r for r in query_leaderboard(component=COMPONENT)
        if r.train_key == train_key
        and r.eval_protocol_version == "1.1.0"
        and not r.eval_cfg.get("smoke", False)
    ]
    if not matches:
        raise RuntimeError(
            f"no v1.1.0 steering row for arch={arch_name} seed={seed} "
            f"(train_key={train_key}). Run agent_steer's c5 sweep first."
        )
    # Prefer the most-recent v1.1.0 row if there are multiple.
    matches.sort(key=lambda r: r.ts, reverse=True)
    return train_key, matches[0].eval_key, ack


def detect_one_cell(
    arch_name: str, seed: int, model_gemma, tokenizer,
) -> dict[str, float]:
    """Run detection on one (arch, seed) v1.1.0 c5 cell."""
    train_key, eval_key_steer, ack = _find_steering_eval_key(arch_name, seed)
    log.info(
        "[cell] arch=%s seed=%d train_key=%s steering_eval_key=%s",
        arch_name, seed, train_key[:8], eval_key_steer[:8],
    )

    pairs = _load_pairs(eval_key_steer)
    log.info("[cell]   loaded %d labeled continuations", len(pairs))

    pos_rate = sum(p["label"] for p in pairs) / max(len(pairs), 1)
    log.info("[cell]   positive rate = %.3f", pos_rate)
    n_pos = int(sum(p["label"] for p in pairs))
    # Need ≥ n_folds (=5) positives so every GroupKFold fold has at
    # least one positive in the train side; otherwise sklearn fails or
    # gives meaningless PR-AUC. Common for TFA cells whose c5 v1.1.0
    # peak success grade is ~0.3 — almost no continuation clears
    # success ≥ 2 AND coh ≥ 2.0 jointly.
    if n_pos < 5:
        log.warning(
            "[cell]   too few positives (n_pos=%d < 5 folds); skipping "
            "detection — labels too degenerate to fit a sparse probe",
            n_pos,
        )
        # Persist a marker row for downstream visibility — primary metric
        # set to NaN so analysis.py renders "—".
        spec = load_arch(arch_name, component=COMPONENT)
        eval_cfg = {
            "metric_set": "detection",
            "sweep": "c5_detection_v1",
            "steering_eval_key": eval_key_steer,
            "skip_reason": f"n_pos={n_pos}_lt_5_folds",
        }
        eval_key = compute_eval_key(
            train_key=train_key,
            eval_protocol_version=DETECTION_PROTO,
            eval_cfg=eval_cfg,
        )
        metrics = {
            "positive_rate": float(pos_rate),
            "n_sent": float(len(pairs)),
            "n_pos": float(n_pos),
        }
        save_metrics(eval_key=eval_key, metrics=metrics)
        append_leaderboard(LeaderboardRow(
            eval_key=eval_key,
            train_key=train_key,
            act_cache_key=ack,
            component=COMPONENT,
            arch=arch_name,
            arch_version=spec.arch_version,
            seed=seed,
            datasource=DATASOURCE,
            eval_protocol_version=DETECTION_PROTO,
            eval_cfg=eval_cfg,
            metrics=metrics,
            primary_metric="pr_auc_S8",
            agent="agent_steer",
            ts=datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        ))
        log.info("[cell]   appended skip row → eval_key=%s", eval_key[:16])
        return metrics

    residuals = _forward_residuals(pairs, model_gemma, tokenizer)
    log.info(
        "[cell]   forward done. residual shapes range %d..%d tokens",
        min(r.shape[0] for r in residuals),
        max(r.shape[0] for r in residuals),
    )

    spec = load_arch(arch_name, component=COMPONENT)
    datasource = load_datasource(DATASOURCE)
    d_in = int(getattr(datasource, "d_in", 2304))
    arch = instantiate_arch(spec, d_in=d_in)
    state = load_checkpoint_state_dict(train_key)
    arch.load_state_dict(state)
    arch.cuda().eval()
    arch_T = max(int(getattr(arch, "T", 1) or 1), 1)
    log.info("[cell]   arch loaded; T=%d", arch_T)

    sa = _build_cohort(residuals, arch_T)
    labels = np.array([p["label"] for p in pairs], dtype=np.int64)
    qids = np.array([p["concept_id"] for p in pairs])

    log.info(
        "[cell]   cohort: sentence_acts=%s labels=%s qids unique=%d",
        sa.shape, labels.shape, len(set(qids.tolist())),
    )
    result = detect_case_study(
        arch=arch,
        sentence_acts=sa,
        labels=labels,
        question_ids=qids,
        shuffle_seed=42,
    )

    metrics: dict[str, float] = {}
    for S, v in result.pr_auc.items():
        metrics[f"pr_auc_S{S}"] = float(v)
    if result.pr_auc_shuffled is not None:
        for S, v in result.pr_auc_shuffled.items():
            metrics[f"pr_auc_shuffled_S{S}"] = float(v)
    if result.shuffle_gap is not None:
        for S, v in result.shuffle_gap.items():
            metrics[f"shuffle_gap_S{S}"] = float(v)
    metrics["positive_rate"] = float(result.positive_rate)
    metrics["n_sent"] = float(result.n_sent)

    eval_cfg = {
        "metric_set": "detection",
        "sweep": "c5_detection_v1",
        "steering_eval_key": eval_key_steer,
        "T_eval": int(sa.shape[1]),
        "shuffle_seed": 42,
    }
    eval_key = compute_eval_key(
        train_key=train_key,
        eval_protocol_version=DETECTION_PROTO,
        eval_cfg=eval_cfg,
    )

    save_metrics(eval_key=eval_key, metrics=metrics)
    append_leaderboard(LeaderboardRow(
        eval_key=eval_key,
        train_key=train_key,
        act_cache_key=ack,
        component=COMPONENT,
        arch=arch_name,
        arch_version=spec.arch_version,
        seed=seed,
        datasource=DATASOURCE,
        eval_protocol_version=DETECTION_PROTO,
        eval_cfg=eval_cfg,
        metrics=metrics,
        primary_metric="pr_auc_S8",
        agent="agent_steer",
        ts=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    ))
    log.info(
        "[cell]   pr_auc_S8=%.4f shuffle_gap_S8=%.4f → eval_key=%s",
        metrics.get("pr_auc_S8", float("nan")),
        metrics.get("shuffle_gap_S8", float("nan")),
        eval_key[:16],
    )

    del arch
    torch.cuda.empty_cache()
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(
        description="C5 detection axis — eval-only on cached v1.1.0 checkpoints."
    )
    ap.add_argument(
        "--archs", nargs="+",
        default=["tsae_paper", "topk_sae", "txc_base", "txc_pro", "tfa"],
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 42])
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("[run] loading Gemma-2-2b-IT (subject model for residuals)")
    hf_token = require_token("hf")
    tokenizer = AutoTokenizer.from_pretrained(SUBJECT_MODEL, token=hf_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        SUBJECT_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        token=hf_token,
    ).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log.info("[run] Gemma loaded")

    for arch_name in args.archs:
        for seed in args.seeds:
            try:
                detect_one_cell(arch_name, seed, model, tokenizer)
            except Exception as exc:                     # noqa: BLE001
                log.error(
                    "[run] cell %s seed=%d FAILED: %s", arch_name, seed, exc,
                )

    log.info("[run] sweep done")


if __name__ == "__main__":
    main()
