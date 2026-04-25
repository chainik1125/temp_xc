"""Rebuild concat-set C to match Ye et al. 2025's exact TSNE protocol.

From their `src/experiments/tsne.py`:
  - 10 MMLU subjects (not 8)
  - 200 questions total (N=6000 divided by 30 iterations)
  - Last 30 tokens of each *question* (not question + choices)
  - `add_special_tokens=False`

Rationale: after training tsae_paper with the paper's exact training
recipe, we also want to evaluate it on the paper's exact eval data so
any "weak clustering" finding can't be blamed on concat-set choices.

Output:
  experiments/phase6_qualitative_latents/concat_corpora/concat_C_v2.json

The original concat_C.json is kept — concat_C_v2 is a parallel cache
so we can compare both.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

from transformers import AutoTokenizer  # noqa: E402
from datasets import load_dataset  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "experiments/phase6_qualitative_latents/concat_corpora"
HF_TOKEN = open("/workspace/hf_cache/token").read().strip()

# From paper tsne.py line 203:
PAPER_DOMAINS = [
    "high_school_mathematics", "formal_logic", "professional_medicine",
    "high_school_european_history", "high_school_chemistry",
    "high_school_statistics", "college_computer_science", "world_religions",
    "high_school_psychology", "college_physics",
]
N_TOKENS_PER_Q = 30   # paper's default
N_QS_PER_SUBJ = 20    # 20 × 10 = 200 questions total


def main():
    tok = AutoTokenizer.from_pretrained("google/gemma-2-2b-it", token=HF_TOKEN)

    print("[mmlu] loading cais/mmlu all/test")
    ds = load_dataset("cais/mmlu", "all", split="test", token=HF_TOKEN)

    by_subject: dict[str, list[dict]] = {}
    for ex in ds:
        if ex["subject"] in PAPER_DOMAINS:
            by_subject.setdefault(ex["subject"], []).append(ex)
    for s in PAPER_DOMAINS:
        if len(by_subject.get(s, [])) < N_QS_PER_SUBJ:
            print(f"  WARN: only {len(by_subject.get(s, []))} Qs for {s}")

    sequences = []
    for subj in PAPER_DOMAINS:
        for qi in range(N_QS_PER_SUBJ):
            ex = by_subject[subj][qi]
            # Paper uses raw question text (no choices)
            text = ex["question"].strip()
            # add_special_tokens=False to match paper
            ids = tok(text, add_special_tokens=False)["input_ids"]
            if len(ids) < N_TOKENS_PER_Q:
                # Left-pad by repeating first id — rare
                pad_n = N_TOKENS_PER_Q - len(ids)
                ids = [ids[0]] * pad_n + ids
            # Paper takes the LAST N_TOKENS_PER_Q (`torch.arange(-num_tokens, 0)`)
            window = ids[-N_TOKENS_PER_Q:]
            sequences.append({
                "subject": subj, "qid": qi,
                "token_ids": window, "n_tokens": len(window),
            })

    out = {
        "sequences": sequences, "categories": PAPER_DOMAINS,
        "n_sequences": len(sequences),
        "protocol": {
            "tokens_per_q": N_TOKENS_PER_Q,
            "qs_per_subject": N_QS_PER_SUBJ,
            "add_special_tokens": False,
            "last_n_tokens": True,
            "source": "cais/mmlu all/test",
        },
    }
    dst = OUT_DIR / "concat_C_v2.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"  saved {dst}: {len(sequences)} seqs × {N_TOKENS_PER_Q} tok")


if __name__ == "__main__":
    main()
