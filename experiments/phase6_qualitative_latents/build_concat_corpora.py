"""Phase 6: build concatenation corpora A, B, C used for qualitative latent analysis.

A mirrors the paper's Figure 1 (Newton Principia + MMLU genetics Q +
Bhagavat Gita). B mirrors Figure 4 (MMLU biology Q + Darwin letter +
Animal Farm wiki + MMLU math Q). C is the TSNE-oriented set:
20-token windows from 20 MMLU questions across 8 categories.

Run once; outputs are idempotent under
`experiments/phase6_qualitative_latents/concat_corpora/`:

    concat_A.json   — 1 concatenated sequence + per-source spans
    concat_B.json   — 1 concatenated sequence + per-source spans
    concat_C.json   — 160 × 20-token sequences + subject/qid labels
    sources/*.txt   — cached raw source texts (Gutenberg + Wiki)
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

from transformers import AutoTokenizer  # noqa: E402
from datasets import load_dataset  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "experiments/phase6_qualitative_latents/concat_corpora"
SRC_DIR = OUT_DIR / "sources"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SRC_DIR.mkdir(parents=True, exist_ok=True)

HF_TOKEN = open("/workspace/hf_cache/token").read().strip()

# Gutenberg canonical .txt URLs (the pgXXXXX.txt mirror is the most stable).
GUTENBERG = {
    "principia": ("https://www.gutenberg.org/cache/epub/28233/pg28233.txt",
                  "Newton, Principia (Gutenberg #28233)"),
    "gita":      ("https://www.gutenberg.org/cache/epub/2388/pg2388.txt",
                  "Bhagavat Gita (Gutenberg #2388)"),
    "darwin":    ("https://www.gutenberg.org/cache/epub/2087/pg2087.txt",
                  "Darwin, letter (Gutenberg #2087)"),
}
# Wikipedia plain-text endpoint.
WIKI_URLS = {
    "animal_farm": ("https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&explaintext=1&titles=Animal_Farm",
                    "Animal Farm — Wikipedia"),
}


def fetch(url: str, dst: Path, sleep_s: float = 2.0) -> str:
    if dst.exists():
        return dst.read_text(encoding="utf-8")
    print(f"  [fetch] {url} -> {dst.name}")
    req = urllib.request.Request(url, headers={"User-Agent": "Phase6-research/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read().decode("utf-8", errors="replace")
    dst.write_text(body, encoding="utf-8")
    time.sleep(sleep_s)  # be kind to Gutenberg / Wikipedia
    return body


def strip_gutenberg(raw: str) -> str:
    """Drop Gutenberg header/footer boilerplate, keep the body."""
    m1 = re.search(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", raw)
    m2 = re.search(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", raw)
    if m1 and m2:
        body = raw[m1.end():m2.start()]
    else:
        body = raw
    body = re.sub(r"\r\n", "\n", body).strip()
    return body


def strip_wiki_api(raw: str) -> str:
    d = json.loads(raw)
    pages = d["query"]["pages"]
    _, page = next(iter(pages.items()))
    return page.get("extract", "").strip()


def load_mmlu():
    """Return dict subject -> list of (question, choices, answer)."""
    print("  [mmlu] loading cais/mmlu all/test")
    ds = load_dataset("cais/mmlu", "all", split="test", token=HF_TOKEN)
    by_subject: dict[str, list[dict]] = {}
    for ex in ds:
        subj = ex["subject"]
        by_subject.setdefault(subj, []).append(ex)
    print(f"  [mmlu] {len(by_subject)} subjects, {len(ds)} total qs")
    return by_subject


def fmt_mmlu_q(ex: dict) -> str:
    """Render an MMLU example into readable text (question + lettered choices)."""
    choices = ex["choices"]
    letters = "ABCD"
    body = ex["question"].strip() + "\n"
    for i, c in enumerate(choices):
        body += f"{letters[i]}. {c.strip()}\n"
    return body.rstrip()


def take_chunk(text: str, target_tokens: int, tokenizer, skip: int = 0) -> tuple[str, list[int]]:
    """Grab a chunk of `text` whose tokenisation length is close to target.

    Uses a character-heuristic then trims. `skip` lets us offset into the
    text so the Principia / Gita chunks aren't just header material.
    """
    approx_chars = target_tokens * 4
    start = min(skip, max(0, len(text) - approx_chars))
    chunk = text[start:start + approx_chars + 200]
    ids = tokenizer(chunk, add_special_tokens=False)["input_ids"]
    return tokenizer.decode(ids[:target_tokens]), ids[:target_tokens]


def build_concat(
    segments: list[tuple[str, str, list[int]]],
) -> dict:
    """Concatenate tokenised segments; record (source, start, end) spans."""
    out_ids: list[int] = []
    provenance: list[dict] = []
    for name, source, ids in segments:
        provenance.append({
            "source": source,
            "label": name,
            "start": len(out_ids),
            "end": len(out_ids) + len(ids),
            "n_tokens": len(ids),
        })
        out_ids.extend(ids)
    return {"token_ids": out_ids, "provenance": provenance,
            "n_tokens": len(out_ids)}


def main():
    print("[1/4] load tokenizer")
    tok = AutoTokenizer.from_pretrained("google/gemma-2-2b-it", token=HF_TOKEN)

    print("[2/4] fetch Gutenberg sources")
    raw_texts = {}
    for key, (url, label) in GUTENBERG.items():
        raw = fetch(url, SRC_DIR / f"{key}.txt")
        raw_texts[key] = strip_gutenberg(raw)

    print("[2/4] fetch Wikipedia sources")
    for key, (url, label) in WIKI_URLS.items():
        raw = fetch(url, SRC_DIR / f"{key}.json")
        raw_texts[key] = strip_wiki_api(raw)

    print("[3/4] load MMLU")
    mmlu = load_mmlu()

    # Pick canonical examples for A/B.
    #  - Figure 1 used a genetics question (high_school_biology has genetics Qs)
    #  - Figure 4 used a biology Q + math Q
    genetics_q = next(q for q in mmlu["high_school_biology"]
                      if "gene" in q["question"].lower()
                      or "allele" in q["question"].lower()
                      or "DNA" in q["question"])
    bio_q = mmlu["high_school_biology"][0]
    math_q = mmlu["high_school_mathematics"][0]

    # ── Concat-set A: Principia + MMLU genetics + Gita (~1024 tokens)
    print("[4/4] build concat A (Fig 1 analogue)")
    tgt_each = 1024 // 3
    # Skip past Gutenberg metadata-y front matter with a char offset
    _, ids_p = take_chunk(raw_texts["principia"], tgt_each, tok, skip=8000)
    q_ids = tok(fmt_mmlu_q(genetics_q),
                add_special_tokens=False)["input_ids"]
    _, ids_g = take_chunk(raw_texts["gita"], tgt_each, tok, skip=8000)
    A = build_concat([
        ("principia",  GUTENBERG["principia"][1], ids_p),
        ("genetics_q", f"MMLU/high_school_biology idx={mmlu['high_school_biology'].index(genetics_q)}",
         q_ids),
        ("gita",       GUTENBERG["gita"][1],      ids_g),
    ])
    (OUT_DIR / "concat_A.json").write_text(json.dumps(A, indent=2))
    print(f"  concat_A: {A['n_tokens']} tokens across {len(A['provenance'])} segments")

    # ── Concat-set B: MMLU bio + Darwin letter + Animal Farm + MMLU math
    print("[4/4] build concat B (Fig 4 analogue)")
    tgt_each_b = (1024 - 100) // 2  # 2 long passages + 2 short MMLU Qs
    bio_ids = tok(fmt_mmlu_q(bio_q), add_special_tokens=False)["input_ids"]
    _, ids_dar = take_chunk(raw_texts["darwin"], tgt_each_b, tok, skip=2000)
    _, ids_af = take_chunk(raw_texts["animal_farm"], tgt_each_b, tok, skip=500)
    math_ids = tok(fmt_mmlu_q(math_q), add_special_tokens=False)["input_ids"]
    B = build_concat([
        ("bio_q",       f"MMLU/high_school_biology idx=0", bio_ids),
        ("darwin",      GUTENBERG["darwin"][1],            ids_dar),
        ("animal_farm", WIKI_URLS["animal_farm"][1],       ids_af),
        ("math_q",      f"MMLU/high_school_mathematics idx=0", math_ids),
    ])
    (OUT_DIR / "concat_B.json").write_text(json.dumps(B, indent=2))
    print(f"  concat_B: {B['n_tokens']} tokens across {len(B['provenance'])} segments")

    # ── Concat-set C: 20 MMLU Qs × 8 categories × 20 token windows = 160 sequences
    print("[4/4] build concat C (TSNE set)")
    categories = [
        "high_school_biology", "high_school_chemistry",
        "high_school_european_history", "high_school_mathematics",
        "professional_medicine", "high_school_physics",
        "college_computer_science", "moral_scenarios",
    ]
    seqs = []
    for subj in categories:
        pool = mmlu[subj][:20]
        for qi, ex in enumerate(pool):
            body = fmt_mmlu_q(ex)
            ids = tok(body, add_special_tokens=False)["input_ids"]
            window = ids[:20]
            if len(window) < 20:
                # Pad with a repeat of the last id (rare: very short Qs)
                window = window + [window[-1]] * (20 - len(window))
            seqs.append({
                "subject": subj,
                "qid": qi,
                "token_ids": window,
                "n_tokens": len(window),
            })
    C = {"sequences": seqs, "categories": categories,
         "n_sequences": len(seqs)}
    (OUT_DIR / "concat_C.json").write_text(json.dumps(C, indent=2))
    print(f"  concat_C: {len(seqs)} sequences × {len(seqs[0]['token_ids'])} tokens")

    print("done:", OUT_DIR)


if __name__ == "__main__":
    main()
