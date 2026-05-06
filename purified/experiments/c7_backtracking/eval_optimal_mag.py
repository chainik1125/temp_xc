"""Re-evaluate one already-trained C7 cell at JUST its peak-Δgc magnitude
with full text persistence + coherence judging.

Motivation (Dmitry 2026-05-05): the canonical eval reports a single
peak-Δgc number per arch but does not persist the steered generations
or score them for coherence, so we cannot compute:

  - net_saves: per arch at optimal mag, the number of MATH-500
    questions where the unsteered model got the answer wrong but the
    steered model got it right, minus regressions in the other
    direction. Baseline-corrected against the cut-and-continue
    noise floor at mag=0.
  - 2x2 contingency at optimal mag: each generation classified along
    {coherent, incoherent} x {has_backtracking, no_backtracking}.

This driver fills both gaps WITHOUT touching the canonical pipeline.
For one (arch, bs) cell:

  1. Load the trained checkpoint by ``train_key`` (cache hit; no
     retraining).
  2. Reload the cached unsteered Stage-A continuations from
     ``<workspace>/phase1_unsteered.json`` (which is shared across
     evals for the same train_key + cohort + cut_fraction).
  3. Look up the cell's peak Δgc magnitude from the leaderboard.
  4. Mine the top steering feature, build the steering vector at the
     dom-base-union L2 norm.
  5. Generate steered continuations at JUST {0, peak_mag} (mag=0 for
     baseline correction; peak_mag for the optimal-steering snapshot).
  6. Persist {qid, magnitude, prompt_text, generation_text,
              steered_answer, steered_correct, ground_truth} per row
     to ``<workspace>/steered_phase2_optimal.jsonl``.
  7. Run TWO judge passes against the persisted generations:
       (a) backtracking COUNT (existing SonnetBacktrackingJudge) → label
       (b) coherence 0-3 grade (new SonnetCoherenceJudge below) → grade
  8. Both judge outputs land in ``<workspace>/judge_outputs.jsonl``
     (existing) and ``<workspace>/coherence_judge.jsonl`` (new).

Sister script ``analyze_optimal.py`` reads all the per-cell
persistence files and emits the per-arch markdown report.

Usage::

    .venv/bin/python -m experiments.c7_backtracking.eval_optimal_mag \\
        --arch txc_base --bs 1024 --n-steps 300000

Cost (per cell): 122 generations (61 q x 2 mags) + 244 judge calls
(2 judges x 122) ≈ ~$1-2 in Sonnet calls + ~10-15 min on H100.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import datetime as _dt
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from anthropic import AsyncAnthropic
from safetensors.torch import load_file

from temp_bench.case_studies.backtracking import (
    SONNET_JUDGE_MODEL,
    SonnetBacktrackingJudge,
    SteeringHook,
    answers_match,
    build_cohort,
    cut25_token_position,
    extract_boxed,
    extract_labeled_sentence_acts,
    generate_continuation_panels,
    load_reasoning_lm,
    load_stage_a,
    mine_top_features,
    run_phase1_unsteered,
    split_pos_neg,
    _build_prompt,
)
from temp_bench.config import (
    act_cache_dir,
    checkpoint_dir,
    compute_act_cache_key,
    compute_eval_key,
    compute_train_key,
    instantiate_arch,
    load_arch,
    load_datasource,
    run_dir,
)
from temp_bench.schemas import TrainingConfig
from temp_bench.utils.tokens import get_token

from experiments.c7_backtracking.run import COMPONENT, DATASOURCE, EVAL_PROTOCOL_VERSION

log = logging.getLogger("c7.eval_optimal_mag")


# ── Coherence judge (port of origin/aniket-ward-stage-b:experiments/
# ward_backtracking_txc/grade_sonnet.py:GRADER_PROMPT, 0-3 rubric) ────────

COHERENCE_JUDGE_MODEL: str = "claude-sonnet-4-6"
COHERENCE_JUDGE_PROMPT: str = """You are grading the coherence of a math problem-solving generation.

Problem prompt the model was solving: {prompt_text}

The model generated this continuation:
\"\"\"
{generation}
\"\"\"

Grade this continuation for coherence on the following 0-3 scale:

3 - Coherent and on-task. The text is grammatical natural English, follows from the prompt, makes mathematical or logical sense (even if reaching a wrong answer), and progresses through reasoning. Repetition of phrases is OK if the underlying reasoning advances.

2 - Mostly coherent but partly off-task or partly incoherent. Some sentences make sense; others are confused or repetitive without progress. The text is recognizably English prose attempting to engage with the problem.

1 - Highly repetitive or incoherent. The model emits looped phrases ("Wait, I'm not. / Wait, I'm not.") or single-token repetitions, or topic-drifts entirely off the problem. May contain occasional coherent fragments amid the noise.

0 - Total degeneration. Pure repetition of one or two tokens, gibberish, single-character emissions, or no recognizable English structure.

Reply with EXACTLY one digit (0, 1, 2, or 3) and nothing else. Do not explain your grade."""

_COH_PROMPT_HASH: str = hashlib.sha256(
    COHERENCE_JUDGE_PROMPT.encode("utf-8")
).hexdigest()[:16]


def parse_coherence_reply(raw: str) -> int:
    """Parse "3", "2", etc. Anything outside 0..3 returns -1."""
    s = raw.strip()
    m = re.match(r"^([0-3])\b", s)
    if m:
        return int(m.group(1))
    return -1


@dataclasses.dataclass
class CoherenceJudgeOutput:
    transcript_id: str
    magnitude: float
    arch: str
    seed: int
    judge_id: str
    judge_model: str
    prompt_hash: str
    grade: int                 # 0..3, or -1 on parse failure
    raw: str
    ts: str


class SonnetCoherenceJudge:
    """Self-contained 0-3 coherence judge with persistence to
    ``<workspace>/coherence_judge.jsonl``. Idempotent (skips
    already-judged (transcript_id, magnitude, arch, seed) tuples)."""

    def __init__(self, *, workspace: Path,
                 model: str = COHERENCE_JUDGE_MODEL,
                 max_tokens: int = 16,
                 max_concurrency: int = 8):
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._jsonl = self.workspace / "coherence_judge.jsonl"
        self.model = model
        self.max_tokens = max_tokens
        self.judge_id = (
            f"{model}:{_COH_PROMPT_HASH}"
        )
        self.prompt_hash = _COH_PROMPT_HASH
        self._sem = asyncio.Semaphore(max_concurrency)

    def _existing_keys(self) -> set[tuple]:
        keys: set[tuple] = set()
        if not self._jsonl.exists():
            return keys
        with self._jsonl.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("grade", -1) < 0:
                    continue
                keys.add((
                    r["transcript_id"], float(r["magnitude"]),
                    r["arch"], int(r["seed"]),
                ))
        return keys

    def _persist(self, out: CoherenceJudgeOutput) -> None:
        with self._jsonl.open("a") as f:
            f.write(json.dumps(dataclasses.asdict(out)) + "\n")

    async def judge_one(self, client, *, transcript_id: str,
                        magnitude: float, arch: str, seed: int,
                        prompt_text: str, generation: str
                        ) -> CoherenceJudgeOutput:
        async with self._sem:
            try:
                msg = await client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[{
                        "role": "user",
                        "content": COHERENCE_JUDGE_PROMPT.format(
                            prompt_text=prompt_text[:1500],
                            generation=generation[:6000],
                        ),
                    }],
                )
                raw = msg.content[0].text.strip()
                grade = parse_coherence_reply(raw)
            except Exception as e:                                  # pragma: no cover
                raw = f"(api-error: {e})"
                grade = -1
        out = CoherenceJudgeOutput(
            transcript_id=transcript_id,
            magnitude=float(magnitude),
            arch=arch,
            seed=int(seed),
            judge_id=self.judge_id,
            judge_model=self.model,
            prompt_hash=self.prompt_hash,
            grade=grade,
            raw=raw,
            ts=_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        self._persist(out)
        return out

    async def judge_many(self, rows: Iterable[tuple], *,
                         skip_existing: bool = True
                         ) -> list[CoherenceJudgeOutput]:
        rows = list(rows)
        seen = self._existing_keys() if skip_existing else set()
        client = AsyncAnthropic(api_key=get_token("anthropic"))
        tasks = []
        for (qid, mag, arch, seed, prompt, gen) in rows:
            key = (qid, float(mag), arch, int(seed))
            if key in seen:
                continue
            tasks.append(self.judge_one(
                client, transcript_id=qid, magnitude=mag,
                arch=arch, seed=seed, prompt_text=prompt, generation=gen,
            ))
        if not tasks:
            return []
        return await asyncio.gather(*tasks)


# ── Driver ────────────────────────────────────────────────────────────


def lookup_peak_magnitude(arch: str, bs: int, n_steps: int, seed: int,
                          ) -> float:
    """Read the leaderboard for the canonical c7 row matching
    (arch, bs, n_steps, seed) and return its peak Δgc magnitude."""
    from temp_bench.config import purified_root
    lb = purified_root() / "results" / "leaderboard.jsonl"
    candidates = []
    spec = load_arch(arch, component=COMPONENT)
    ds = load_datasource(DATASOURCE)
    target_train_key = compute_train_key(
        arch=spec, seed=seed,
        training_cfg=TrainingConfig(n_steps=n_steps, batch_size=bs),
        act_cache_key=compute_act_cache_key(ds),
    )
    with lb.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("component") != COMPONENT:
                continue
            if r.get("train_key") != target_train_key:
                continue
            if r.get("eval_cfg", {}).get("_extended_mags"):
                continue
            candidates.append(r)
    if not candidates:
        raise RuntimeError(
            f"no canonical leaderboard row for arch={arch} bs={bs} "
            f"n_steps={n_steps} seed={seed} (train_key={target_train_key[:12]})"
        )
    # Prefer most recent.
    r = max(candidates, key=lambda r: r["ts"])
    peak_mag = r["metrics"].get("delta_gc_peak_magnitude")
    if peak_mag is None:
        raise RuntimeError(
            f"leaderboard row for {target_train_key[:12]} missing delta_gc_peak_magnitude"
        )
    return float(peak_mag)


def main(*, arch: str, bs: int, n_steps: int, seed: int) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S")
    spec = load_arch(arch, component=COMPONENT)
    ds = load_datasource(DATASOURCE)
    act_cache_key = compute_act_cache_key(ds)
    training_cfg = TrainingConfig(n_steps=n_steps, batch_size=bs)
    train_key = compute_train_key(
        arch=spec, seed=seed,
        training_cfg=training_cfg,
        act_cache_key=act_cache_key,
    )
    cp_dir = checkpoint_dir(train_key)
    state_path = cp_dir / "model.safetensors"
    if not state_path.exists():
        raise FileNotFoundError(
            f"checkpoint missing: {state_path} — has the cell finished training?"
        )
    log.info("[opt-mag] arch=%s bs=%d n_steps=%d seed=%d train_key=%s",
             arch, bs, n_steps, seed, train_key[:12])

    # Resolve peak magnitude from the canonical leaderboard row.
    peak_mag = lookup_peak_magnitude(arch, bs, n_steps, seed)
    eval_mags = (0.0, peak_mag)
    log.info("[opt-mag] eval at magnitudes %s (peak=%+g)",
             list(eval_mags), peak_mag)

    # Workspace — disambiguator separates these results from canonical eval.
    eval_cfg = {
        "magnitudes": list(eval_mags),
        "cut_fraction": 0.25,
        "_optimal_mag": True,
        "_peak_mag": peak_mag,
    }
    eval_key = compute_eval_key(
        train_key=train_key,
        eval_protocol_version=EVAL_PROTOCOL_VERSION,
        eval_cfg=eval_cfg,
    )
    workspace = run_dir(eval_key)
    workspace.mkdir(parents=True, exist_ok=True)
    log.info("[opt-mag] workspace=%s", workspace)

    # ── Instantiate arch with state_dict ──────────────────────────────
    cache_dir = act_cache_dir(act_cache_key)
    specs = json.loads((cache_dir / "layer_specs.json").read_text())
    d_in = int(specs["d_model"])
    arch_module = instantiate_arch(spec, d_in=d_in)
    state = load_file(str(state_path))
    params = dict(arch_module.named_parameters())
    buffers = dict(arch_module.named_buffers())
    cast: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        target = params.get(k)
        if target is None:
            target = buffers.get(k)
        cast[k] = v.to(dtype=target.dtype) if target is not None else v
    arch_module.load_state_dict(cast, strict=False)
    if torch.cuda.is_available():
        arch_module = arch_module.cuda()
    n_params = sum(p.numel() for p in arch_module.parameters())
    if n_params > 1e9 and torch.cuda.is_available():
        arch_module = arch_module.bfloat16()

    # ── Mine steering feature + build vector ──────────────────────────
    sa = extract_labeled_sentence_acts()
    pos_neg = split_pos_neg(sa)
    mined = mine_top_features(
        arch_module,
        pos_activations=pos_neg["pos"],
        neg_activations=pos_neg["neg"],
        top_k=32,
    )
    steering_feature = mined[0]
    log.info("[opt-mag] top feature=%d sel=%.4f",
             steering_feature.feature_id, steering_feature.selectivity)

    stage_a = load_stage_a()
    dom_base_union = stage_a.dom_vectors["base"].get("union")
    ref_norm = (float(dom_base_union.norm().item())
                if dom_base_union is not None
                else float(steering_feature.decoder_direction.norm().item()))
    raw_vec = steering_feature.decoder_direction.float()
    vec = raw_vec / raw_vec.norm().clamp_min(1e-8) * ref_norm

    # ── Phase 1 (cohort + unsteered baseline) ─────────────────────────
    cohort = build_cohort()
    rmodel, rtok = load_reasoning_lm()
    phase1 = run_phase1_unsteered(
        cohort, workspace=workspace,
        max_new_tokens=1024, batch_size=8,
        model=rmodel, tok=rtok,
    )
    by_qid = {row["unique_id"]: row for row in phase1}

    # ── Phase 2: steered generation at {0, peak_mag} ──────────────────
    layer = 10
    layer_module = rmodel.model.layers[layer]
    hook = SteeringHook(vec)
    handle = layer_module.register_forward_hook(hook)
    persisted_path = workspace / "steered_phase2_optimal.jsonl"
    persisted_keys: set[tuple] = set()
    if persisted_path.exists():
        with persisted_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    persisted_keys.add((r["unique_id"], float(r["magnitude"])))
                except (json.JSONDecodeError, KeyError):
                    pass
    log.info("[opt-mag] phase2: %d already persisted; will fill the rest",
             len(persisted_keys))

    try:
        problem_prompts: list[str] = []
        prefix_token_ids: list[list[int]] = []
        mags: list[float] = []
        budgets: list[int] = []
        meta: list[tuple[str, float]] = []
        for qid in cohort.all:
            row = by_qid[qid]
            cut_pos = cut25_token_position(row["unsteered_token_ids"], fraction=0.25)
            prefix = row["unsteered_token_ids"][:cut_pos]
            remaining_budget = max(64, len(row["unsteered_token_ids"]) - cut_pos)
            for m in eval_mags:
                if (qid, float(m)) in persisted_keys:
                    continue
                problem_prompts.append(_build_prompt(row["problem"]))
                prefix_token_ids.append(prefix)
                mags.append(float(m))
                budgets.append(remaining_budget)
                meta.append((qid, float(m)))
        log.info("[opt-mag] generating %d steered panels", len(meta))

        if meta:
            outs = generate_continuation_panels(
                rmodel, rtok, hook,
                problem_prompts=problem_prompts,
                prefix_token_ids=prefix_token_ids,
                mags_per_panel=mags,
                max_new_per_panel=budgets,
                batch_size=8,
            )
            with persisted_path.open("a") as f:
                for (qid, mag), prompt, gen in zip(meta, problem_prompts, outs):
                    row = by_qid[qid]
                    steered_answer = extract_boxed(gen)
                    steered_correct = answers_match(steered_answer, row["ground_truth"])
                    f.write(json.dumps({
                        "unique_id": qid,
                        "magnitude": float(mag),
                        "prompt_text": row["problem"],
                        "generation": gen,
                        "steered_answer": steered_answer,
                        "steered_correct": steered_correct,
                        "ground_truth": row["ground_truth"],
                        "unsteered_answer": row["unsteered_answer"],
                        "unsteered_correct": row["unsteered_correct"],
                        "arch": arch,
                        "seed": seed,
                        "ts": _dt.datetime.now(_dt.timezone.utc).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"),
                    }) + "\n")
        log.info("[opt-mag] phase2 persisted to %s", persisted_path)
    finally:
        handle.remove()
        del rmodel
        import gc; gc.collect()
        torch.cuda.empty_cache()

    # ── Judge passes (backtracking + coherence) ───────────────────────
    rows: list[dict] = []
    with persisted_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    judge_rows_for_dispatch = [
        (r["unique_id"], r["magnitude"], r["arch"], r["seed"],
         r["prompt_text"], r["generation"])
        for r in rows
    ]

    bt_judge = SonnetBacktrackingJudge(workspace=workspace)
    coh_judge = SonnetCoherenceJudge(workspace=workspace)

    log.info("[opt-mag] dispatching %d backtracking judge calls", len(judge_rows_for_dispatch))
    asyncio.run(bt_judge.judge_many(judge_rows_for_dispatch))
    log.info("[opt-mag] dispatching %d coherence judge calls", len(judge_rows_for_dispatch))
    asyncio.run(coh_judge.judge_many(judge_rows_for_dispatch))
    log.info("[opt-mag] complete; outputs in %s", workspace)
    return 0


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--bs", type=int, required=True)
    ap.add_argument("--n-steps", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    raise SystemExit(main(arch=args.arch, bs=args.bs,
                           n_steps=args.n_steps, seed=args.seed))


if __name__ == "__main__":
    cli()
