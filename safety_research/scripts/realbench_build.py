"""
Build a real, scaled safety benchmark from publicly available sources:

  - AdvBench harmful behaviors  (520 harmful)
  - JailbreakBench/JBB-Behaviors  (100 harmful + 100 benign matched-pairs)
  - XSTest v2 prompts            (250 safe + 200 unsafe; cross-distribution)
  - tatsu-lab/alpaca clean        (sample of benign instructions)

Output:
  safety_research/results/realbench/prompts.json  - all prompts with metadata

Splits:
  train: AdvBench harmful (520) + Alpaca-sampled benign (520)             1040
  test_in: JBB harmful (100) + JBB benign (100)                            200
  test_ood: XSTest safe (250) + XSTest unsafe (200)                        450

The XSTest split is the *exaggerated refusal* / cross-distribution stress
test - many "unsafe-looking but safe" prompts that vanilla refusal probes
trip on.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

OUT = Path("/home/cs29824/andre/temp_xc/safety_research/results/realbench")
OUT.mkdir(parents=True, exist_ok=True)


def load_advbench(path: str = "/tmp/advbench.csv") -> list[dict]:
    import csv
    rows: list[dict] = []
    if not Path(path).exists():
        # Re-download
        import urllib.request
        url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
        urllib.request.urlretrieve(url, path)
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({"prompt": r["goal"], "label": 1, "source": "advbench",
                         "category": "harmful"})
    return rows


def load_jbb() -> tuple[list[dict], list[dict]]:
    from datasets import load_dataset
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    harmful = [{"prompt": r["Goal"], "label": 1, "source": "jbb",
                "category": r["Category"]}
               for r in ds["harmful"]]
    benign = [{"prompt": r["Goal"], "label": 0, "source": "jbb",
               "category": r["Category"]}
              for r in ds["benign"]]
    return harmful, benign


def load_xstest() -> tuple[list[dict], list[dict]]:
    """XSTest splits prompts into 'safe_xxx' and 'unsafe_xxx' subtypes; we treat
    'safe_*' as label=0 and the rest as label=1. The point of the bench is that
    safe-looking-unsafe and unsafe-looking-safe phrasings exist."""
    from datasets import load_dataset
    ds = load_dataset("natolambert/xstest-v2-copy", split="prompts")
    safe = []
    unsafe = []
    for r in ds:
        prompt = r["prompt"]
        t = r["type"]
        if t.startswith("safe_") or t in ("homonyms", "figurative_language",
                                          "real_discr_nons_groups",
                                          "historical_events",
                                          "privacy_public",
                                          "privacy_fictional",
                                          "definitions",
                                          "safe_targets",
                                          "safe_contexts",
                                          "real_group_nons_discr"):
            safe.append({"prompt": prompt, "label": 0, "source": "xstest",
                         "category": t})
        else:
            unsafe.append({"prompt": prompt, "label": 1, "source": "xstest",
                           "category": t})
    return safe, unsafe


def load_alpaca_benign(n: int = 520, seed: int = 0) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    rng = random.Random(seed)
    pool = [r for r in ds if not r.get("input")]  # single-turn instructions only
    rng.shuffle(pool)
    out = []
    for r in pool[: n * 2]:  # over-sample then filter
        prompt = r["instruction"].strip()
        if 5 <= len(prompt.split()) <= 60:
            out.append({"prompt": prompt, "label": 0, "source": "alpaca",
                        "category": "benign"})
        if len(out) >= n:
            break
    return out


def main() -> None:
    advbench = load_advbench()
    jbb_h, jbb_b = load_jbb()
    xs_safe, xs_unsafe = load_xstest()
    alpaca = load_alpaca_benign(n=520)
    print(f"AdvBench: {len(advbench)} harmful")
    print(f"JBB:      {len(jbb_h)} harmful + {len(jbb_b)} benign")
    print(f"XSTest:   {len(xs_safe)} safe + {len(xs_unsafe)} unsafe")
    print(f"Alpaca:   {len(alpaca)} benign")

    train = advbench + alpaca
    test_in = jbb_h + jbb_b
    test_ood = xs_safe + xs_unsafe

    rng = random.Random(0)
    rng.shuffle(train)
    rng.shuffle(test_in)
    rng.shuffle(test_ood)

    bench = {"train": train, "test_in": test_in, "test_ood": test_ood}
    for split, rows in bench.items():
        with open(OUT / f"{split}.json", "w") as f:
            json.dump(rows, f, indent=1)
        print(f"  {split}: {len(rows):>4} prompts  ({sum(r['label'] for r in rows)} pos)")
    summary = {k: {"n": len(v), "n_pos": sum(r["label"] for r in v)} for k, v in bench.items()}
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("\nSaved to", OUT)


if __name__ == "__main__":
    main()
