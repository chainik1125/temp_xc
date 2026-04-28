"""Sonnet 4.6 grader for AxBench-style steering generations.

For each row in generations.jsonl, ask Claude Sonnet 4.6 two questions
(verbatim from T-SAE paper §B.2):

    SUCCESS: "Given the following text, grade it as containing the concept
        '{feature_explanation}' or not. Only respond with one of the grades:
        0 = does not contain concept, 1 = somewhat related to the concept,
        2 = mostly related to the concept, or 3 = strongly contains concept."

    COHERENCE: "Given the following text, grade its linguistic coherence.
        Only respond with one of the grades: 0 = completely incoherent
        (e.g. very few or no English words), 1 = somewhat coherent (e.g.
        highly repetitive), 2 = mostly coherent, or 3 = fully coherent."

`{feature_explanation}` is the concept's `description` from concepts.py.

Concurrent over (generation, prompt_type) via ThreadPoolExecutor with
n_workers=5 + max_retries=12 to stay under the 50-req/min rate limit.
Output per arch: grades.jsonl with one row per (concept, strength)
containing success_grade, coherence_grade, plus raw responses for audit.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("TQDM_DISABLE", "1")

from experiments.phase7_unification._paths import banner
from experiments.phase7_unification.case_studies._paths import (
    CASE_STUDIES_DIR, STAGE_1_ARCHS, ANTHROPIC_GRADER_MODEL,
)
from experiments.phase7_unification.case_studies.steering.concepts import (
    CONCEPTS, get_concept,
)


CONCEPT_DESC = {c["id"]: c["description"] for c in CONCEPTS}


SUCCESS_PROMPT = """You are an LLM expert. Given the following text, grade it as containing the concept "{concept_desc}" or not. Only respond with one of the grades: 0 = does not contain concept, 1 = somewhat related to the concept, 2 = mostly related to the concept, or 3 = strongly contains concept. Respond with the digit only.

Text: {text}

Grade:"""

COHERENCE_PROMPT = """You are an LLM expert. Given the following text, grade its linguistic coherence. Only respond with one of the grades: 0 = completely incoherent (e.g. very few or no English words), 1 = somewhat coherent (e.g. highly repetitive), 2 = mostly coherent, or 3 = fully coherent. Respond with the digit only.

Text: {text}

Grade:"""


_DIGIT_RE = re.compile(r"^[\s\D]*([0-3])")


def _parse_grade(text: str) -> int | None:
    """Extract first digit 0-3 from Sonnet's response. Returns None if missing."""
    if not text:
        return None
    m = _DIGIT_RE.search(text)
    if m:
        return int(m.group(1))
    # Fallback: look for any single digit anywhere
    for ch in text:
        if ch in "0123":
            return int(ch)
    return None


def _grade_one(client, model: str, prompt_text: str) -> tuple[int | None, str]:
    msg = client.messages.create(
        model=model, max_tokens=10,
        messages=[{"role": "user", "content": prompt_text}],
    )
    raw = msg.content[0].text.strip()
    return _parse_grade(raw), raw


def grade_one_arch(arch_id: str, *, n_workers: int = 5, force: bool = False,
                   base_subdir: str = "steering") -> None:
    gen_path = CASE_STUDIES_DIR / base_subdir / arch_id / "generations.jsonl"
    out_path = CASE_STUDIES_DIR / base_subdir / arch_id / "grades.jsonl"
    if not gen_path.exists():
        print(f"  [skip] {arch_id}: generations.jsonl missing — run intervene first")
        return
    if out_path.exists() and not force:
        print(f"  [skip] {arch_id}: grades.jsonl exists (use --force to rebuild)")
        return

    rows = [json.loads(line) for line in gen_path.open()]
    print(f"  {len(rows)} generations to grade ({len(rows) * 2} Sonnet calls)")

    from anthropic import Anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        for candidate in ("/root/.tokens/anthropic_key", "/workspace/.tokens/anthropic_key"):
            cp = Path(candidate)
            if cp.exists():
                api_key = cp.read_text().strip()
                break
    if not api_key:
        raise RuntimeError(
            "no Anthropic API key found: set $ANTHROPIC_API_KEY or place "
            "the key in /root/.tokens/anthropic_key or /workspace/.tokens/anthropic_key"
        )
    client = Anthropic(api_key=api_key, max_retries=12)

    def _grade_pair(idx_row: tuple[int, dict]) -> dict:
        idx, row = idx_row
        try:
            concept_desc = CONCEPT_DESC[row["concept_id"]]
        except KeyError:
            concept_desc = row["concept_id"]
        text = row["generated_text"]
        # Run success + coherence sequentially within this worker — keeps
        # per-worker burst small. Outer ThreadPoolExecutor still parallelises.
        s_grade, s_raw = _grade_one(
            client, ANTHROPIC_GRADER_MODEL,
            SUCCESS_PROMPT.format(concept_desc=concept_desc, text=text),
        )
        c_grade, c_raw = _grade_one(
            client, ANTHROPIC_GRADER_MODEL,
            COHERENCE_PROMPT.format(text=text),
        )
        return {
            "idx": idx,
            "arch_id": row["arch_id"],
            "src_class": row["src_class"],
            "concept_id": row["concept_id"],
            "feature_idx": row["feature_idx"],
            "strength": row["strength"],
            "success_grade": s_grade,
            "coherence_grade": c_grade,
            "success_raw": s_raw,
            "coherence_raw": c_raw,
            "concept_desc": concept_desc,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results: list[dict | None] = [None] * len(rows)
    n_done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_grade_pair, (i, r)): i for i, r in enumerate(rows)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                results[i] = {
                    "idx": i, "error": f"{type(e).__name__}: {str(e)[:200]}",
                    "concept_id": rows[i]["concept_id"],
                    "strength": rows[i]["strength"],
                }
            n_done += 1
            if n_done % 30 == 0 or n_done == len(rows):
                elapsed = time.time() - t0
                rate = n_done / max(elapsed, 1e-3)
                eta = (len(rows) - n_done) / max(rate, 1e-3)
                print(f"    [{n_done}/{len(rows)}] {rate:.1f} gen/s  ETA {eta:.0f}s")

    # Sort by idx so writeup is deterministic.
    results = [r for r in results if r is not None]
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"  saved {out_path}  ({sum(1 for r in results if 'error' in r)} errors)")

    # Quick mean over (success, coherence) per concept (ignoring strength).
    mean_s, mean_c, n = 0.0, 0.0, 0
    for r in results:
        if r.get("success_grade") is None or r.get("coherence_grade") is None:
            continue
        mean_s += r["success_grade"]
        mean_c += r["coherence_grade"]
        n += 1
    if n:
        print(f"  arch mean: success={mean_s / n:.2f}  coherence={mean_c / n:.2f}  "
              f"(over {n} valid grades)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=list(STAGE_1_ARCHS))
    ap.add_argument("--n-workers", type=int, default=5)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--base-subdir", default="steering",
                    help="subdir under results/case_studies/ to read/write "
                         "(steering | steering_paper)")
    args = ap.parse_args()
    banner(__file__)
    for arch_id in args.archs:
        print(f"\n=== {arch_id} ===")
        grade_one_arch(arch_id, n_workers=args.n_workers, force=args.force,
                       base_subdir=args.base_subdir)


if __name__ == "__main__":
    main()
