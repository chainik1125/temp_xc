"""
v2: extend the realbench prompt suite with two extra harmful sources and a
held-out benign capability split.

New splits written under safety_research/results/realbench/ (alongside the
existing train/test_in/test_ood files):

  test_mi.json    - MaliciousInstruct harmful prompts (n ~= 100)
                    + matched benign Alpaca controls (sampled, label=0)
                    -> third harmful/benign split for cross-distribution
                       generalisation.

  cap_alpaca.json - 200 single-turn benign Alpaca instructions, *disjoint*
                    from the 520 used for training. Used as the capability
                    KL guard ("does the steering hurt general behaviour?").

  cap_mmlu.json   - 100 four-choice MMLU questions formatted as a
                    multiple-choice prompt; used for accuracy regression
                    under steering.

Also re-uses existing test_ood (XSTest) for OOD-validated K selection.
We intentionally do NOT regenerate existing splits.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path("/home/cs29824/andre/temp_xc/safety_research/results/realbench")
OUT.mkdir(parents=True, exist_ok=True)


def load_malicious_instruct() -> list[dict]:
    """MaliciousInstruct: 100 harmful prompts from Huang et al. 2024.

    HF dataset id: 'walledai/MaliciousInstruct'. Single column 'prompt'.
    """
    from datasets import load_dataset
    ds = load_dataset("walledai/MaliciousInstruct", split="train")
    return [{"prompt": r["prompt"], "label": 1, "source": "malicious_instruct",
             "category": "harmful"} for r in ds]


def load_alpaca_pool(n: int, seed: int = 0, exclude: set[str] | None = None
                     ) -> list[dict]:
    """Load disjoint Alpaca single-turn instructions for benign control.

    `exclude` blocks already-used prompts so capability split doesn't overlap
    with the train benigns.
    """
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    rng = random.Random(seed)
    pool = [r for r in ds if not r.get("input")]
    rng.shuffle(pool)
    out: list[dict] = []
    excl = exclude or set()
    for r in pool:
        prompt = r["instruction"].strip()
        if not (5 <= len(prompt.split()) <= 60):
            continue
        if prompt in excl:
            continue
        out.append({"prompt": prompt, "label": 0, "source": "alpaca",
                    "category": "benign"})
        if len(out) >= n:
            break
    return out


def load_mmlu_mini(n: int = 100, seed: int = 0) -> list[dict]:
    """Random sample of MMLU questions formatted as MC prompts.

    Each row: {prompt, label='X' (one of A/B/C/D), choices: [A,B,C,D],
    source='mmlu', category=<subject>}.
    """
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="validation")
    rng = random.Random(seed)
    idx = rng.sample(range(len(ds)), min(n, len(ds)))
    out: list[dict] = []
    for i in idx:
        r = ds[i]
        q = r["question"].strip()
        ch = r["choices"]
        if len(ch) != 4:
            continue
        ans_letter = "ABCD"[r["answer"]]
        prompt = (
            f"{q}\n\n"
            f"A) {ch[0]}\n"
            f"B) {ch[1]}\n"
            f"C) {ch[2]}\n"
            f"D) {ch[3]}\n\n"
            "Answer with a single letter (A, B, C, or D)."
        )
        out.append({
            "prompt": prompt, "label": -1, "source": "mmlu",
            "category": r["subject"], "answer": ans_letter,
            "choices": ch,
        })
    return out


def main() -> None:
    # ---- MaliciousInstruct + matched Alpaca benigns ------------------------
    train = json.load(open(OUT / "train.json"))
    train_alpaca_used = {r["prompt"] for r in train if r["source"] == "alpaca"}
    test_in = json.load(open(OUT / "test_in.json"))
    test_in_used = {r["prompt"] for r in test_in}
    excl = train_alpaca_used | test_in_used

    mi = load_malicious_instruct()
    print(f"MaliciousInstruct: {len(mi)} harmful")
    # Match with exactly len(mi) Alpaca benign controls (disjoint from train)
    mi_alpaca = load_alpaca_pool(n=len(mi), seed=42, exclude=excl)
    print(f"Alpaca control:    {len(mi_alpaca)} benign")
    test_mi = mi + mi_alpaca
    rng = random.Random(0); rng.shuffle(test_mi)
    json.dump(test_mi, open(OUT / "test_mi.json", "w"), indent=1)
    excl |= {r["prompt"] for r in mi_alpaca}

    # ---- capability KL split: held-out Alpaca ------------------------------
    cap = load_alpaca_pool(n=200, seed=123, exclude=excl)
    print(f"cap_alpaca:        {len(cap)} benign (capability KL guard)")
    json.dump(cap, open(OUT / "cap_alpaca.json", "w"), indent=1)

    # ---- MMLU mini ---------------------------------------------------------
    mmlu = load_mmlu_mini(n=100, seed=7)
    print(f"cap_mmlu:          {len(mmlu)} four-choice questions")
    json.dump(mmlu, open(OUT / "cap_mmlu.json", "w"), indent=1)

    # ---- summary -----------------------------------------------------------
    sm: dict = json.load(open(OUT / "summary.json"))
    sm["test_mi"] = {"n": len(test_mi),
                     "n_pos": sum(r["label"] == 1 for r in test_mi)}
    sm["cap_alpaca"] = {"n": len(cap), "n_pos": 0}
    sm["cap_mmlu"] = {"n": len(mmlu), "n_pos": 0}
    json.dump(sm, open(OUT / "summary.json", "w"), indent=1)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
